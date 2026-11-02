#!/usr/bin/env python3
"""
scripts/figures/common.py
-------------------------
Common TikZ and PlotNeuralNet helper definitions for 3D volumetric model diagrams.
Provides standardized color palettes, layer blocks, and build utilities.
"""

from pathlib import Path
import subprocess
import shutil
import sys
import re

# Add PlotNeuralNet to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOT_NEURAL_NET_DIR = PROJECT_ROOT / "tools" / "PlotNeuralNet"
if str(PLOT_NEURAL_NET_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_NEURAL_NET_DIR))

try:
    from pycore.tikzeng import *
    from pycore.blocks import *
except ImportError as e:
    raise ImportError(
        f"Failed to import PlotNeuralNet from {PLOT_NEURAL_NET_DIR}. "
        "Ensure tools/PlotNeuralNet is properly cloned."
    ) from e


def safe_caption(s: str) -> str:
    """Ensures caption strings do not break TikZ key-value parsing and contain NO quotes."""
    if not s:
        return "{}"
    cleaned = str(s).strip()
    # Strip any existing quotes or braces repeatedly
    while cleaned and (cleaned[0] in ('"', "'", "{") or cleaned[-1] in ('"', "'", "}")):
        cleaned = cleaned.strip('"').strip("'").strip("{}").strip()
    cleaned = (
        cleaned.replace('"', "")
        .replace("'", "")
        .replace("\\\\", " ")
        .replace("^3", "3")
        .replace("\\times", "x")
        .replace("_", " ")
        .replace("%", " pct")
        .replace("&", " and ")
    )
    return f"{{{cleaned}}}"


def to_custom_cor():
    """Defines publication-grade custom colors for medical 3D architectures (authentic PlotNeuralNet palette)."""
    return r"""
% Authentic PlotNeuralNet Palette - Simple, Cohesive, Publication-Standard
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}           % Warm Cream/Ochre for Convolutions
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}         % Deeper Ochre for Conv+Norm+GELU band
\def\PoolColor{rgb:red,1;black,0.3}                    % Dark Wine/Brick Red for Downsampling/Stem
\def\UnpoolColor{rgb:blue,5;green,2.5;white,5}         % Soft Sky Blue for Transposed Convolutions
\def\DcnvColor{rgb:blue,5;green,2.5;white,5}           % Soft Sky Blue alias
\def\SoftmaxColor{rgb:magenta,5;black,7}               % Deep Plum/Purple for Output Heads
\def\SumColor{rgb:blue,5;green,15}                     % Soft Sage/Teal for Fusion & Skips
\def\ViTColor{rgb:yellow,6;red,2;white,3}              % Soft Amber for Vision Transformer Blocks
\def\PredColor{rgb:red,4;yellow,4;white,3}             % Harmonized Coral/Sand for JEPA Predictor
\def\ProjColor{rgb:green,4;blue,3;white,2}             % Harmonized Sage for Auxiliary Projector
\def\DsColor{rgb:magenta,5;black,7}                    % Matching Plum for Deep Supervision Heads
\def\StemColor{rgb:red,1;black,0.3}                    % Matching Wine Red for Native Stem
\def\InputColor{rgb:cyan,1;white,9}                    % Very Pale Subtle Ice Blue for Input Volume
\def\LossColor{rgb:red,4;black,2}                      % Subtle Crimson Red for Loss Terms
\def\MaskColor{rgb:gray,4;black,4}                     % Dark Gray for Mask Tokens
"""


def to_custom_head(projectpath=None):
    if projectpath is None:
        projectpath = str(PLOT_NEURAL_NET_DIR)
    pathlayers = (Path(projectpath) / "layers").as_posix() + "/"
    return (
        r"""
\documentclass[border=15pt, multi, tikz]{standalone} 
\usepackage{import}
\subimport{"""
        + pathlayers
        + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d}
\usetikzlibrary{arrows.meta}
\usetikzlibrary{decorations.pathreplacing}
\usetikzlibrary{calc}
"""
    )


def to_custom_begin():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}
\newcommand{\dsmidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:magenta,5;black,7}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.75]
\tikzstyle{dsconnection}=[ultra thick,dashed,every node/.style={sloped,allow upside down},draw={rgb:magenta,5;black,7},opacity=0.85]
"""


def to_custom_Conv(
    name,
    s_filer="128",
    n_filer="4",
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2.5,
    height=36,
    depth=36,
    caption="",
    fill=r"\ConvColor",
    opacity=0.85,
):
    """Creates a single-block 3D convolution box with safe caption."""
    clean_cap = safe_caption(caption)
    return f"""
\\pic[shift={{{offset}}}] at {to} 
    {{Box={{
        name={name},
        caption={clean_cap},
        xlabel={{{{{n_filer}, }}}},
        zlabel={s_filer},
        fill={fill},
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};
"""


def to_custom_ConvConvRelu(
    name,
    s_filer="128",
    n_filer=(32, 32),
    offset="(0,0,0)",
    to="(0,0,0)",
    width=(1.5, 1.5),
    height=36,
    depth=36,
    caption="",
    fill=r"\ConvColor",
    bandfill=r"\ConvReluColor",
    opacity=0.85,
):
    """Creates a dual-banded 3D convolution block representing a two-conv residual stage."""
    clean_cap = safe_caption(caption)
    if isinstance(n_filer, (list, tuple)):
        xlabel_str = ", ".join(str(x) for x in n_filer)
    else:
        xlabel_str = f"{n_filer}, {n_filer}"
    if isinstance(width, (list, tuple)):
        width_str = f"{{ {width[0]} , {width[1]} }}"
    else:
        half_w = float(width) / 2.0
        width_str = f"{{ {half_w:.2f} , {half_w:.2f} }}"

    return f"""
\\pic[shift={{{offset}}}] at {to} 
    {{RightBandedBox={{
        name={name},
        caption={clean_cap},
        xlabel={{{{{xlabel_str}}}}},
        zlabel={s_filer},
        fill={fill},
        bandfill={bandfill},
        opacity={opacity},
        height={height},
        width={width_str},
        depth={depth}
        }}
    }};
"""


def to_custom_UnPool(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=1.5,
    height=32,
    depth=32,
    opacity=0.85,
    caption="",
):
    """Creates a 3D transposed convolution upsampling slice without cluttering captions."""
    clean_cap = safe_caption(caption)
    return f"""
\\pic[shift={{{offset}}}] at {to} 
    {{Box={{
        name={name},
        caption={clean_cap},
        fill=\\UnpoolColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};
"""


def to_skip(of, to, pos=1.25):
    """
    Creates an arched skip connection where the horizontal bridge is guaranteed to be
    100% straight and level, by raising the first vertical segment to match the destination height.
    Uses unique coordinate names based on 'of' and 'to' to prevent collisions.
    """
    top_name = f"{of}_{to}_top"
    of_top = f"{of}_top_{to}"
    return f"""
\\path ({to}-south)  -- ({to}-north)  coordinate[pos={pos}] ({top_name}) ;
\\coordinate ({of_top}) at ({of}-northeast |- {top_name}) ;
\\draw [copyconnection]  ({of}-northeast)  
-- node {{\\copymidarrow}} ({of_top})
-- node {{\\copymidarrow}} ({top_name})
-- node {{\\copymidarrow}} ({to}-north);
"""


def to_custom_Softmax(
    name,
    s_filer="128",
    n_filer="5",
    offset="(0,0,0)",
    to="(0,0,0)",
    width=1.5,
    height=36,
    depth=36,
    caption="Output (5ch)",
):
    """Creates a final multi-class segmentation output head."""
    clean_cap = safe_caption(caption)
    return f"""
\\pic[shift={{{offset}}}] at {to} 
    {{Box={{
        name={name},
        caption={clean_cap},
        xlabel={{{{{n_filer}, }}}},
        zlabel={s_filer},
        fill=\\SoftmaxColor,
        opacity=0.85,
        height={height},
        width={width},
        depth={depth}
        }}
    }};
"""


def to_ds_head(name, of, offset="(0, -2.5, 0)", s_filer=16, n_filer=5, width=1, height=10, depth=10, caption="DS Head"):
    """Creates a deep supervision 1x1x1 conv output head branching off a decoder stage."""
    clean_cap = safe_caption(caption)
    return f"""
\\pic[shift={{{offset}}}] at ({of}-south) 
    {{Box={{
        name={name},
        caption={clean_cap},
        xlabel={{{{{n_filer}, }}}},
        zlabel={s_filer},
        fill=\\DsColor,
        opacity=0.85,
        height={height},
        width={width},
        depth={depth}
        }}
    }};
\\draw [dsconnection] ({of}-south) -- node {{\\dsmidarrow}} ({name}-north);
"""


def to_ViT_block(name, s_filer="16", n_filer="384", offset="(1.5,0,0)", to="(0,0,0)", width=4.0, height=20, depth=20, caption="ViT Layer"):
    """Creates a Vision Transformer block (amber/gold)."""
    clean_cap = safe_caption(caption)
    return f"""
\\pic[shift={{{offset}}}] at {to} 
    {{RightBandedBox={{
        name={name},
        caption={clean_cap},
        xlabel={{{{{n_filer}, }}}},
        zlabel={s_filer},
        fill=\\ViTColor,
        bandfill=\\ConvReluColor,
        opacity=0.85,
        height={height},
        width={width},
        depth={depth}
        }}
    }};
"""


def to_Predictor_block(name, s_filer="16", n_filer="192", offset="(1.5,0,0)", to="(0,0,0)", width=3.5, height=15, depth=15, caption="Predictor"):
    """Creates a JEPA Predictor block (coral/orange)."""
    clean_cap = safe_caption(caption)
    return f"""
\\pic[shift={{{offset}}}] at {to} 
    {{Box={{
        name={name},
        caption={clean_cap},
        xlabel={{{{{n_filer}, }}}},
        zlabel={s_filer},
        fill=\\PredColor,
        opacity=0.85,
        height={height},
        width={width},
        depth={depth}
        }}
    }};
"""


def to_Projector_block(name, s_filer="1536", n_filer="128", offset="(0,-2.5,0)", to="(0,0,0)", width=2.5, height=10, depth=10, caption="Projector"):
    """Creates an Auxiliary Projector MLP block (emerald green)."""
    clean_cap = safe_caption(caption)
    return f"""
\\pic[shift={{{offset}}}] at {to} 
    {{Box={{
        name={name},
        caption={clean_cap},
        xlabel={{{{{n_filer}, }}}},
        zlabel={s_filer},
        fill=\\ProjColor,
        opacity=0.85,
        height={height},
        width={width},
        depth={depth}
        }}
    }};
"""


def compile_tex_to_pdf_and_png(tex_path: Path, output_dir: Path, base_name: str, width_px: int = 2800):
    """Compiles a TikZ LaTeX file to vector PDF and high-res PNG."""
    tex_path = tex_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = tex_path.parent

    cmd = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
    proc = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    pdf_produced = build_dir / f"{tex_path.stem}.pdf"

    if not pdf_produced.exists():
        print(f"Error compiling {tex_path.name}:", file=sys.stderr)
        print(proc.stdout[-2000:], file=sys.stderr)
        raise RuntimeError(f"LaTeX compilation failed for {tex_path}")

    target_pdf = output_dir / f"{base_name}.pdf"
    target_png = output_dir / f"{base_name}.png"

    shutil.copy2(pdf_produced, target_pdf)

    # Convert to PNG using macOS sips
    cmd_sips = [
        "sips",
        "-s", "format", "png",
        str(target_pdf),
        "--resampleWidth", str(width_px),
        "--out", str(target_png),
    ]
    subprocess.run(cmd_sips, capture_output=True, check=True)

    # Clean intermediate auxiliary files
    for ext in (".aux", ".log", ".out"):
        aux_file = build_dir / f"{tex_path.stem}{ext}"
        if aux_file.exists():
            aux_file.unlink()

    # Clean temporary compiled PDF in build_dir if build_dir != output_dir
    if build_dir != output_dir and pdf_produced.exists():
        pdf_produced.unlink()

    print(f"✓ Generated: {target_pdf.name} and {target_png.name}")
    return target_pdf, target_png


def export_architecture_figure(arch: list, base_name: str, width_px: int = 3000):
    """Exports TikZ architecture to docs/figures/tex/{base_name}.tex,
    compiles vector PDF and 300-DPI PNG, and distributes to docs/figures/ and paper/latex/figures/.
    Guarantees scripts/figures/ remains completely clean.
    """
    tex_dir = PROJECT_ROOT / "docs" / "figures" / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = tex_dir / f"{base_name}.tex"

    to_generate(arch, str(tex_path))
    print(f"Generated LaTeX TikZ code: {tex_path}")

    out_dirs = [
        PROJECT_ROOT / "docs" / "figures",
        PROJECT_ROOT / "paper" / "latex" / "figures",
    ]
    for out_dir in out_dirs:
        compile_tex_to_pdf_and_png(tex_path, out_dir, base_name, width_px=width_px)

