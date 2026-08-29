"""Register and resolve fonts for publication plots."""

from __future__ import annotations

import os
from pathlib import Path

from matplotlib import font_manager

PREFERRED_FONT = "Aptos Narrow"
FALLBACK_FONT = "DejaVu Sans"

_PLOTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PLOTS_DIR.parent
_APTOS_FONTS_DIR = _PROJECT_ROOT / "fonts" / "Microsoft Aptos Fonts"
_APTOS_NARROW_FILES = (
    "Aptos-Narrow.ttf",
    "Aptos-Narrow-Bold.ttf",
    "Aptos-Narrow-Italic.ttf",
    "Aptos-Narrow-Bold-Italic.ttf",
)


def _register_fonts_from(path: Path) -> None:
    if not path.is_dir():
        return
    for font_path in font_manager.findSystemFonts(fontpaths=[str(path)]):
        font_manager.fontManager.addfont(font_path)


def register_plot_fonts() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        _register_fonts_from(Path(conda_prefix) / "fonts")
    _register_fonts_from(_APTOS_FONTS_DIR)
    for filename in _APTOS_NARROW_FILES:
        font_path = _APTOS_FONTS_DIR / filename
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))


def get_plot_font_family() -> str:
    register_plot_fonts()
    available = {f.name for f in font_manager.fontManager.ttflist}
    if PREFERRED_FONT in available:
        return PREFERRED_FONT
    for name in available:
        if name.lower() == PREFERRED_FONT.lower():
            return name
    return FALLBACK_FONT


register_plot_fonts()
PLOT_FONT_FAMILY = get_plot_font_family()

if PLOT_FONT_FAMILY == FALLBACK_FONT:
    print(f"Plot font: {FALLBACK_FONT} ({PREFERRED_FONT} not found)")
else:
    print(f"Plot font: {PLOT_FONT_FAMILY}")
