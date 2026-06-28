"""
Shared matplotlib style for Bass LV experiments.

Usage
-----
    from plot_style import set_style, C
    set_style()
    ax.plot(x, y, color=C['market'], label='Heston')
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Semantic colour palette (Tableau-derived, print-safe)
C = {
    'market':  '#E15759',   # coral red   — market / Heston target
    'model':   '#4E79A7',   # steel blue  — Bass LV / model output
    'ref':     '#59A14F',   # green       — reference SABR
    'aux':     '#F28E2B',   # orange      — auxiliary third curve
    'neutral': '#76B7B2',   # teal        — bands / median
    'dark':    '#2D2D2D',   # near-black  — annotation lines
}

_RC = {
    # figure
    'figure.dpi':           100,
    'figure.facecolor':     'white',
    'savefig.dpi':          150,
    'savefig.bbox':         'tight',
    # axes
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'grid.alpha':           0.3,
    'grid.linestyle':       '--',
    'grid.linewidth':       0.6,
    'axes.labelsize':       11,
    'axes.titlesize':       12,
    'axes.titleweight':     'normal',
    # lines & markers
    'lines.linewidth':      1.8,
    'lines.markersize':     5,
    # legend
    'legend.frameon':       False,
    'legend.fontsize':      10,
    'legend.handlelength':  1.6,
    # fonts
    'font.size':            11,
    'font.family':          'sans-serif',
    # math
    'mathtext.fontset':     'stix',
}


def set_style():
    """Apply the project-wide matplotlib style. Call once per notebook."""
    mpl.rcParams.update(_RC)


def vline(ax, x, **kwargs):
    """Thin dotted vertical line marking a maturity or glueing point."""
    kw = dict(color=C['dark'], lw=0.9, linestyle=':', alpha=0.55)
    kw.update(kwargs)
    ax.axvline(x, **kw)


def hline(ax, y, **kwargs):
    kw = dict(color=C['dark'], lw=0.8, linestyle=':', alpha=0.35)
    kw.update(kwargs)
    ax.axhline(y, **kw)
