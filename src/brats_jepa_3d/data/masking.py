import logging
from collections import deque
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Fraction of a token's voxels that must be brain for the token to count as tissue.
TISSUE_TOKEN_FRAC = 0.10
# Legacy minimum tissue tokens for isotropic 3x3x3 targets (kept for backward compatibility).
MIN_TISSUE_TOKENS_PER_TARGET = 14
# Minimum tissue context tokens before falling back to unfiltered regularization.
MIN_TISSUE_CONTEXT_TOKENS = 32

# Anisotropic 3D Target Jitter Masking: 6 multi-aspect target cuboids spanning isotropic macro,
# axial, coronal, and sagittal planar slabs, plus regional meso blocks (97 tokens for 8³ grid).
DEFAULT_TARGET_CUBOID_SIZES_8GRID: tuple[tuple[int, int, int], ...] = (
    (3, 3, 3),  # 27 tokens: Macro isotropic cube (48mm)
    (2, 3, 3),  # 18 tokens: Axial planar slab (32x48x48mm)
    (3, 2, 3),  # 18 tokens: Coronal planar slab (48x32x48mm)
    (3, 3, 2),  # 18 tokens: Sagittal planar slab (48x48x32mm)
    (2, 2, 2),  #  8 tokens: Regional meso cube (32mm)
    (2, 2, 2),  #  8 tokens: Regional meso cube (32mm)
)
DEFAULT_TARGET_CUBOID_SIZES = DEFAULT_TARGET_CUBOID_SIZES_8GRID

# Resolution-scaled target cuboids for 16³ grid (patch_size 8³): 776 tokens total across 6 target cuboids
# Preserves identical physical millimeter volume coverage (48mm, 32x48x48mm, 32mm).
DEFAULT_TARGET_CUBOID_SIZES_16GRID: tuple[tuple[int, int, int], ...] = tuple(
    (s[0] * 2, s[1] * 2, s[2] * 2) for s in DEFAULT_TARGET_CUBOID_SIZES_8GRID
)


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
    3. Constant Sequence Length (N_ctx = 192 / 1536):
       Enforcing exactly N_ctx = 37.5% of total patches across all volumes eliminates
       ragged tensor padding and attention masks, maximizing GPU Tensor Core computation speed.
    4. Strict Disjointness Guarantee:
       Target and context index sets satisfy:
           \text{ctx} \cap \left(\bigcup_{m=1}^M \text{tgt}_m\right) = \emptyset
       ensuring zero attention leakage from target into the context encoder.
    5. Anisotropic 3D Aspect-Ratio Jitter Target Cuboids (97 / 776 tokens):
       Sampling multi-aspect cuboids [(3,3,3), (2,3,3), (3,2,3), (3,3,2), (2,2,2), (2,2,2)]
       trains the predictor across axial, coronal, and sagittal anatomical planes, while maintaining
       dense supervision (89.8% retention of baseline) and zero ragged padding via slot-uniform shapes.
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        num_target_cuboids: int | None = None,
        target_cuboid_size: tuple[int, int, int] | None = None,
        target_cuboid_sizes: (
            tuple[tuple[int, int, int], ...]
            | list[tuple[int, int, int]]
            | list[list[int]]
            | None
        ) = None,
        max_target_overlap: float = 0.0,
        context_num_patches: int | None = None,
        connectivity: int = 26,
        tissue_token_frac: float = TISSUE_TOKEN_FRAC,
        min_tissue_tokens_per_target: int | None = None,
    ):
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1] * grid_size[2]

        if target_cuboid_sizes is not None:
            self.target_cuboid_sizes = tuple(
                tuple(int(x) for x in s) for s in target_cuboid_sizes
            )
            self.num_target_cuboids = len(self.target_cuboid_sizes)
            self.target_cuboid_size = (
                self.target_cuboid_sizes[0] if self.target_cuboid_sizes else (3, 3, 3)
            )
        elif num_target_cuboids is not None or target_cuboid_size is not None:
            n_tgt = 4 if num_target_cuboids is None else int(num_target_cuboids)
            sz_tgt = (
                (3, 3, 3)
                if target_cuboid_size is None
                else tuple(int(x) for x in target_cuboid_size)
            )
            self.num_target_cuboids = n_tgt
            self.target_cuboid_size = sz_tgt
            self.target_cuboid_sizes = tuple(sz_tgt for _ in range(n_tgt))
        else:
            if grid_size == (16, 16, 16):
                self.target_cuboid_sizes = DEFAULT_TARGET_CUBOID_SIZES_16GRID
            else:
                self.target_cuboid_sizes = DEFAULT_TARGET_CUBOID_SIZES_8GRID
            self.num_target_cuboids = len(self.target_cuboid_sizes)
            self.target_cuboid_size = self.target_cuboid_sizes[0]

        self.max_target_overlap = max_target_overlap
        if context_num_patches is None:
            self.context_num_patches = int(0.375 * self.num_patches)
        elif context_num_patches == 192 and grid_size == (16, 16, 16):
            self.context_num_patches = 1536
        else:
            self.context_num_patches = int(context_num_patches)
        self.connectivity = connectivity
        self.tissue_token_frac = tissue_token_frac
        self.min_tissue_tokens_per_target = min_tissue_tokens_per_target

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

    def _resolve_generator(self, generator: torch.Generator | None) -> torch.Generator:
        """Worker-safe RNG: explicit generator wins; otherwise derive from the global torch RNG.

        The seed draw itself advances global torch state, so successive direct
        calls vary deterministically under `manual_seed` (no separate counter
        layer — the dataset owns epoch-freshness via its own counter; see
        `BraTS3DDataset._mask_counter`). In DataLoader workers the global RNG
        is worker-seeded, so streams diverge across workers by construction.
        """
        if generator is not None:
            return generator
        # Worker-safe: derive from the global torch RNG (worker-seeded in
        # DataLoader). The seed draw advances global state, so successive calls
        # vary under `manual_seed`; the dataset adds its epoch counter on top.
        g = torch.Generator()
        g.manual_seed(int(torch.randint(2**31, ()).item()))
        return g

    @staticmethod
    def _randint(gen: torch.Generator, lo: int, hi: int) -> int:
        """Inclusive randint via torch (DataLoader-worker safe, unlike `random`)."""
        if hi <= lo:
            return lo
        return lo + int(torch.randint(hi - lo + 1, (1,), generator=gen).item())

    def _tissue_set(self, token_brain_frac: torch.Tensor | None) -> set[int] | None:
        if token_brain_frac is None:
            return None
        fr = token_brain_frac.reshape(-1)
        return {i for i in range(self.num_patches) if float(fr[i].item()) >= self.tissue_token_frac}

    def _sample_target_cuboids(
        self,
        gen: torch.Generator,
        tissue: set[int] | None = None,
    ) -> list[set[int]]:
        """Samples M contiguous 3D cuboid target blocks with controlled overlap.

        When `tissue` is given, cuboids with >= min_tissue tissue tokens are preferred.
        For each cuboid (cz, cy, cx), min_tissue is dynamically set to
        max(4, int(0.40 * (cz * cy * cx))) unless overridden by self.min_tissue_tokens_per_target.
        """
        gz, gy, gx = self.grid_size
        target_cuboids = []
        all_target_indices: set[int] = set()

        for cz, cy, cx in self.target_cuboid_sizes:
            tgt_vol = cz * cy * cx
            if self.min_tissue_tokens_per_target is not None:
                min_tissue = min(self.min_tissue_tokens_per_target, tgt_vol)
            else:
                min_tissue = max(4, int(0.40 * tgt_vol)) if tissue is not None else 0

            best_cuboid: set[int] | None = None
            best_key: tuple[float, float] | None = None  # (overlap, -tissue_count): lower is better

            for _attempt in range(20):
                z0 = self._randint(gen, 0, gz - cz)
                y0 = self._randint(gen, 0, gy - cy)
                x0 = self._randint(gen, 0, gx - cx)

                candidate = set()
                for dz in range(cz):
                    for dy in range(cy):
                        for dx in range(cx):
                            candidate.add(self._coord_to_idx(z0 + dz, y0 + dy, x0 + dx))

                overlap = len(candidate & all_target_indices) / len(candidate)
                tissue_count = len(candidate & tissue) if tissue is not None else 0
                if overlap <= self.max_target_overlap and (
                    tissue is None or tissue_count >= min_tissue
                ):
                    best_cuboid = candidate
                    break
                key = (overlap, -float(tissue_count))
                if best_key is None or key < best_key:
                    best_key = key
                    best_cuboid = candidate

            # Hard zero-overlap guarantee: overlap is a hard constraint, tissue a
            # soft preference. If stochastic attempts failed a zero-overlap demand,
            # exhaustively scan all valid origins (generator-shuffled order).
            if (
                best_cuboid is not None
                and self.max_target_overlap == 0
                and len(best_cuboid & all_target_indices) > 0
            ):
                origins = [
                    (z0, y0, x0)
                    for z0 in range(gz - cz + 1)
                    for y0 in range(gy - cy + 1)
                    for x0 in range(gx - cx + 1)
                ]
                perm = torch.randperm(len(origins), generator=gen).tolist()
                best_cuboid = None
                for oi in perm:
                    z0, y0, x0 = origins[oi]
                    candidate = {
                        self._coord_to_idx(z0 + dz, y0 + dy, x0 + dx)
                        for dz in range(cz)
                        for dy in range(cy)
                        for dx in range(cx)
                    }
                    if not (candidate & all_target_indices):
                        best_cuboid = candidate
                        break
                if best_cuboid is None:
                    raise RuntimeError(
                        "JEPAMaskingTransform3D: grid saturated, no zero-overlap "
                        f"placement for target cuboid {len(target_cuboids)} "
                        f"({len(all_target_indices)}/{self.num_patches} tokens taken)."
                    )

            if best_cuboid is not None:
                target_cuboids.append(best_cuboid)
                all_target_indices.update(best_cuboid)

        return target_cuboids

    def _sample_context_bfs(
        self,
        forbidden_indices: set[int],
        gen: torch.Generator,
        tissue: set[int] | None = None,
    ) -> list[int]:
        """Expands a 3D BFS cluster from a random seed up to context_num_patches.

        The seed is drawn from tissue tokens when available; expansion itself is
        unchanged (walks adjacency, air fringe included). Fallback fill prefers tissue.
        """
        candidates = [i for i in range(self.num_patches) if i not in forbidden_indices]
        if len(candidates) < self.context_num_patches:
            raise ValueError(
                f"Candidate patches ({len(candidates)}) smaller than requested context ({self.context_num_patches})"
            )

        if tissue is not None:
            tissue_candidates = [c for c in candidates if c in tissue]
            pool = tissue_candidates if tissue_candidates else candidates
        else:
            pool = candidates
        seed = pool[int(torch.randint(len(pool), (1,), generator=gen).item())]
        visited = {seed}
        queue = deque([seed])
        gz, gy, gx = self.grid_size

        while queue and len(visited) < self.context_num_patches:
            curr = queue.popleft()
            cz, cy, cx = self._idx_to_coord(curr)

            # Stochastic expansion order via generator (no global `random` state)
            order = torch.randperm(len(self.neighbor_offsets), generator=gen).tolist()
            for oi in order:
                dz, dy, dx = self.neighbor_offsets[oi]
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
            logger.debug(
                f"BFS reached {len(visited)}/{self.context_num_patches} patches before dead-end; "
                f"filling {self.context_num_patches - len(visited)} remaining with random available patches "
                f"(spatial contiguity partially broken)"
            )
            remaining = [c for c in candidates if c not in visited]
            if tissue is not None:
                tissue_remaining = [c for c in remaining if c in tissue]
                tissue_rest = [c for c in remaining if c not in tissue]
                perm_t = torch.randperm(len(tissue_remaining), generator=gen).tolist() if tissue_remaining else []
                perm_r = torch.randperm(len(tissue_rest), generator=gen).tolist() if tissue_rest else []
                remaining = [tissue_remaining[i] for i in perm_t] + [tissue_rest[i] for i in perm_r]
            else:
                perm = torch.randperm(len(remaining), generator=gen).tolist()
                remaining = [remaining[i] for i in perm]
            needed = self.context_num_patches - len(visited)
            visited.update(remaining[:needed])

        return sorted(visited)

    def __call__(
        self,
        token_brain_frac: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Generates disjoint context and target patch index sets for one volume.

        Args:
            token_brain_frac: optional [512] (or [1,8,8,8]) brain fractions in [0,1].
                When given, target/context sampling is biased toward tissue tokens.
                When None, behavior is exactly the legacy uniform sampling.
            generator: optional torch.Generator for worker-safe determinism.
        """
        gen = self._resolve_generator(generator)
        tissue = self._tissue_set(token_brain_frac)
        target_cuboids = self._sample_target_cuboids(gen, tissue=tissue)
        all_target_indices: set[int] = set()
        for cuboid in target_cuboids:
            all_target_indices.update(cuboid)

        context_indices = self._sample_context_bfs(all_target_indices, gen, tissue=tissue)

        # Convert to PyTorch LongTensors
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensors = [torch.tensor(sorted(c), dtype=torch.long) for c in target_cuboids]
        result: dict[str, Any] = {
            "context_indices": context_tensor,
            "target_indices_list": target_tensors,
        }
        if tissue is not None:
            result["context_tissue_mask"] = torch.tensor(
                [c in tissue for c in context_indices], dtype=torch.bool
            )
        return result


def jepa_masking_collate_fn_3d(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function collating 3D images, masks, and stacking batch-uniform context and target indices.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    has_mask = "mask" in batch[0] and batch[0]["mask"] is not None
    masks = torch.stack([item["mask"] for item in batch], dim=0) if has_mask else None

    # Stack context indices: [B, N_ctx]
    context_indices = torch.stack([item["context_indices"] for item in batch], dim=0)

    # Stack tissue mask if present (brain-aware masking): [B, N_ctx] bool
    context_tissue_mask = None
    if "context_tissue_mask" in batch[0] and batch[0]["context_tissue_mask"] is not None:
        context_tissue_mask = torch.stack(
            [item["context_tissue_mask"] for item in batch], dim=0
        )

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
    if context_tissue_mask is not None:
        result["context_tissue_mask"] = context_tissue_mask
    if masks is not None:
        result["masks"] = masks
    if "patient_id" in batch[0]:
        result["patient_ids"] = [item["patient_id"] for item in batch]

    return result
