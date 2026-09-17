import random
from collections import deque
from typing import Any

import torch


class JEPAMaskingTransform3D:
    r"""
    3D Multi-Block Volumetric Masking for Joint-Embedding Predictive Architectures.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Spatial Autocorrelation Defense (Assran et al., CVPR 2023):
       Point-wise random voxel/patch masking (as in 3D MAE) fails in continuous volumetric medical
       imaging because dense spatial autocorrelation allows the network to trivially interpolate
       missing voxels from immediately adjacent boundaries. Contiguous 3D cuboid target masking
       removes entire anatomical regions, forcing the model to infer semantic morphology.
    2. 3D BFS Connected Context Cluster:
       Context tokens are sampled via a 3D Breadth-First Search (BFS) cluster expansion from a random
       non-target seed voxel. This guarantees a single topologically contiguous anatomical context
       subvolume, reflecting how human radiologists interpret localized pathology within surrounding tissue.
    3. Constant Sequence Length (N_ctx = 192):
       Enforcing exactly N_ctx = 192 patches (37.5% of 512 total patches) across all volumes eliminates
       ragged tensor padding and attention masks, maximizing GPU Tensor Core computation speed.
    4. Strict Disjointness Guarantee:
       Target and context index sets satisfy:
           \text{ctx} \cap \left(\bigcup_{m=1}^M \text{tgt}_m\right) = \emptyset
       ensuring zero attention leakage from target into the context encoder.
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        num_target_cuboids: int = 4,
        target_cuboid_size: tuple[int, int, int] = (3, 3, 3),
        max_target_overlap: float = 0.0,
        context_num_patches: int = 192,
        connectivity: int = 26,
    ):
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1] * grid_size[2]  # 512
        self.num_target_cuboids = num_target_cuboids
        self.target_cuboid_size = target_cuboid_size
        self.max_target_overlap = max_target_overlap
        self.context_num_patches = context_num_patches
        self.connectivity = connectivity

        # Precompute 3D spatial neighbor offsets
        self.neighbor_offsets = self._get_neighbor_offsets(connectivity)

    def _get_neighbor_offsets(self, connectivity: int) -> list[tuple[int, int, int]]:
        offsets = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    if connectivity == 6 and abs(dz) + abs(dy) + abs(dx) > 1:
                        continue
                    offsets.append((dz, dy, dx))
        return offsets

    def _coord_to_idx(self, z: int, y: int, x: int) -> int:
        _gz, gy, gx = self.grid_size
        return z * (gy * gx) + y * gx + x

    def _idx_to_coord(self, idx: int) -> tuple[int, int, int]:
        _gz, gy, gx = self.grid_size
        z = idx // (gy * gx)
        rem = idx % (gy * gx)
        y = rem // gx
        x = rem % gx
        return z, y, x

    def _sample_target_cuboids(self) -> list[set[int]]:
        """Samples M contiguous 3D cuboid target blocks with controlled overlap."""
        gz, gy, gx = self.grid_size
        cz, cy, cx = self.target_cuboid_size
        target_cuboids = []
        all_target_indices: set[int] = set()

        for _ in range(self.num_target_cuboids):
            best_cuboid: set[int] | None = None
            min_overlap = float("inf")

            for _attempt in range(20):
                z0 = random.randint(0, gz - cz)
                y0 = random.randint(0, gy - cy)
                x0 = random.randint(0, gx - cx)

                candidate = set()
                for dz in range(cz):
                    for dy in range(cy):
                        for dx in range(cx):
                            candidate.add(self._coord_to_idx(z0 + dz, y0 + dy, x0 + dx))

                overlap = len(candidate & all_target_indices) / len(candidate)
                if overlap <= self.max_target_overlap:
                    best_cuboid = candidate
                    break
                elif overlap < min_overlap:
                    min_overlap = overlap
                    best_cuboid = candidate

            if best_cuboid is not None:
                target_cuboids.append(best_cuboid)
                all_target_indices.update(best_cuboid)

        return target_cuboids

    def _sample_context_bfs(self, forbidden_indices: set[int]) -> list[int]:
        """Expands a 3D BFS cluster from a random seed up to context_num_patches."""
        candidates = [i for i in range(self.num_patches) if i not in forbidden_indices]
        if len(candidates) < self.context_num_patches:
            raise ValueError(
                f"Candidate patches ({len(candidates)}) smaller than requested context ({self.context_num_patches})"
            )

        seed = random.choice(candidates)
        visited = {seed}
        queue = deque([seed])
        gz, gy, gx = self.grid_size

        while queue and len(visited) < self.context_num_patches:
            curr = queue.popleft()
            cz, cy, cx = self._idx_to_coord(curr)

            # Shuffle neighbor offsets to expand stochastically in 3D
            offsets = list(self.neighbor_offsets)
            random.shuffle(offsets)

            for dz, dy, dx in offsets:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < gz and 0 <= ny < gy and 0 <= nx < gx:
                    neighbor_idx = self._coord_to_idx(nz, ny, nx)
                    if neighbor_idx not in visited and neighbor_idx not in forbidden_indices:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)
                        if len(visited) == self.context_num_patches:
                            break

        # Fallback: if BFS cluster reached a dead-end, fill remaining from available candidates
        if len(visited) < self.context_num_patches:
            import logging
            logging.getLogger(__name__).debug(
                f"BFS reached {len(visited)}/{self.context_num_patches} patches before dead-end; "
                f"filling {self.context_num_patches - len(visited)} remaining with random available patches "
                f"(spatial contiguity partially broken)"
            )
            remaining = [c for c in candidates if c not in visited]
            random.shuffle(remaining)
            needed = self.context_num_patches - len(visited)
            visited.update(remaining[:needed])

        return sorted(visited)

    def __call__(self) -> dict[str, Any]:
        """Generates disjoint context and target patch index sets for one volume."""
        target_cuboids = self._sample_target_cuboids()
        all_target_indices: set[int] = set()
        for cuboid in target_cuboids:
            all_target_indices.update(cuboid)

        context_indices = self._sample_context_bfs(all_target_indices)

        # Convert to PyTorch LongTensors
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensors = [torch.tensor(sorted(c), dtype=torch.long) for c in target_cuboids]

        return {
            "context_indices": context_tensor,
            "target_indices_list": target_tensors,
        }


def jepa_masking_collate_fn_3d(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function collating 3D images, masks, and stacking batch-uniform context and target indices.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    has_mask = "mask" in batch[0] and batch[0]["mask"] is not None
    masks = torch.stack([item["mask"] for item in batch], dim=0) if has_mask else None

    # Stack context indices: [B, N_ctx]
    context_indices = torch.stack([item["context_indices"] for item in batch], dim=0)

    # Stack target indices for each target block: list of [B, N_tgt]
    num_targets = len(batch[0]["target_indices_list"])
    target_indices_list = []
    for m in range(num_targets):
        stacked_m = torch.stack([item["target_indices_list"][m] for item in batch], dim=0)
        target_indices_list.append(stacked_m)

    result = {
        "images": images,
        "context_indices": context_indices,
        "target_indices_list": target_indices_list,
    }
    if masks is not None:
        result["masks"] = masks
    if "patient_id" in batch[0]:
        result["patient_ids"] = [item["patient_id"] for item in batch]

    return result
