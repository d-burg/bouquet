"""
Plotting utilities for perturbed equilibria.

Provides:
  - Core drawing functions (``draw_kinetic_profiles``,
    ``draw_pressure_profiles``, ``draw_jphi_profiles``) that operate
    on pre-loaded data arrays and matplotlib axes.
  - ``plot_bouquet`` -- self-contained notebook-friendly API that loads
    everything from the ``.h5`` file and returns ``(fig, axes)``.
  - ``plot_tokamaker_comparison`` -- overview / single-shot comparison
    of TokaMaker reconstructions against source geqdsk files.
    Requires ``mygs`` (TokaMaker solver) and ``all_results`` (dict) to
    be available as module-level names in the calling namespace.
  - Legacy wrappers (``plot_kinetic_profiles``, ``plot_jphi_profiles``)
    for backward compatibility.
"""

import os
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt

from .schema import find_bytes_dataset
from .utils import (
    load_equilibrium,
    load_equilibrium_by_path,
    load_baseline_profiles,
    count_equilibria,
    list_equilibrium_indices,
    discover_scan_keys,
    select_closed_lcfs,
)
from .io import read_geqdsk

# =====================================================================
# Publication plot style (Wong palette, serif, dotted grids, lw=2)
# Adapted from ~/Desktop/plasma/CTM-processing/publication_plot_rules.md.
# Uses mathtext (not usetex) so plots render without a LaTeX install.
# Auto-applied on import; call set_plot_style() to re-apply or tweak.
# =====================================================================
# Wong colorblind-safe palette (rules order)
WONG = ['#000000', '#E69F00', '#56B4E9', '#009E73',
        '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
#         0 black   1 orange  2 skyblue 3 bgreen
#         4 yellow  5 blue    6 vermil  7 rpurple

# Warm profile scheme: black baseline reference + gold/orange/red data, used by
# the bouquet profile panels (matches plot_geqdsk_bouquet's black + C1=gold).
_GOLD = '#E69F00'    # primary perturbed-draw color
_ORANGE = '#D55E00'  # secondary component (vermilion)
_RED = '#B22222'     # accent (e.g. edge bootstrap)


def set_plot_style(usetex=False):
    """Apply the bouquet publication plot style to matplotlib's rcParams.

    Wong colorblind palette as the color cycle, serif fonts (mathtext Computer
    Modern -- no LaTeX needed), dotted grids, thick lines, compact label sizes.
    Auto-applied when ``bouquet.plotting`` is imported. ``usetex=True`` switches
    to a real LaTeX backend (needs ``pdflatex`` + ``mathptmx`` on PATH).
    """
    import matplotlib as _mpl
    from cycler import cycler
    rc = {
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.prop_cycle': cycler(color=WONG),
        'axes.grid': True, 'grid.linestyle': ':', 'grid.alpha': 0.5,
        'lines.linewidth': 2, 'lines.markersize': 4,
        'axes.labelsize': 11, 'axes.titlesize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'legend.frameon': False,
        'figure.dpi': 90,
    }
    if usetex:
        rc.update({'text.usetex': True,
                   'text.latex.preamble': r'\usepackage{mathptmx}'})
    _mpl.rcParams.update(rc)


set_plot_style()   # auto-apply on import


def _display_figs_row(figs, dpi=110):
    """Display separate matplotlib figures side by side in a notebook.

    Each figure is rendered to its own PNG and laid out in a wrapping flex row,
    so the panels sit horizontally (less vertical scroll) while staying
    individually copy-paste-able images. The source figures are closed so the
    inline backend doesn't also stack them. No-op (returns False) outside a
    notebook, leaving the figures for the caller to ``show()``.
    """
    try:
        from IPython.display import HTML, display
    except Exception:
        return False
    import io
    import base64
    parts = ['<div style="display:flex;flex-wrap:wrap;gap:8px;'
             'align-items:flex-start;">']
    for f in figs:
        buf = io.BytesIO()
        f.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        parts.append(f'<img src="data:image/png;base64,{b64}"/>')
        plt.close(f)
    parts.append('</div>')
    display(HTML(''.join(parts)))
    return True


# =====================================================================
# Comparison plots: TokaMaker vs geqdsk / overview of all results
# =====================================================================
import matplotlib.cm as _cm
from scipy.spatial import cKDTree as _cKDTree
from matplotlib.collections import LineCollection as _LC
import matplotlib.colors as _mcolors_dev

_LW = 1.5   # universal line width

def _lcfs_from_psi(mygs, psi_arr, isoflux_fallback, psi_lcfs_val=None):
    r"""Extract the TokaMaker LCFS contour for a stored :math:`\psi` array.

    Uses ``tricontour`` on the TokaMaker mesh at the value
    ``psi_lcfs_val`` (defaults to ``mygs.psi_bounds[0]``) and returns
    the longest closed path as an ``(N, 2)`` array of ``[R, Z]`` points.
    Falls back to *isoflux_fallback* if no contour is found.

    .. note::
        Requires ``mygs`` to be set as a module-level (or notebook-level)
        name before calling.

    Parameters
    ----------
    psi_arr : ndarray
        Raw poloidal flux on the TokaMaker mesh.
    isoflux_fallback : ndarray, shape (N, 2)
        Isoflux target points used as a fallback when no contour path
        is found.
    psi_lcfs_val : float or None
        The :math:`\psi` value of the LCFS.  ``None`` uses
        ``mygs.psi_bounds[0]``.
    """
    if psi_lcfs_val is None:
        psi_lcfs_val = float(mygs.psi_bounds[0])
    _fig_tmp, _ax_tmp = plt.subplots(1, 1)
    try:
        _cs = _ax_tmp.tricontour(
            mygs.r[:, 0], mygs.r[:, 1], mygs.lc, psi_arr,
            levels=[psi_lcfs_val])
        _segs = [v for seg in _cs.allsegs for v in seg if len(v) > 4]
    finally:
        plt.close(_fig_tmp)
    # Longest CLOSED segment: on a diverted equilibrium the open separatrix
    # branch to the divertor can be longer than the LCFS itself (issue #33).
    _lcfs = select_closed_lcfs(_segs, context="_lcfs_from_psi")
    if _lcfs is not None:
        return _lcfs
    return isoflux_fallback


def _core_contours(mygs, ax, psi_raw, nlevels=9):
    r"""Overplot core flux surface contours on *ax*.

    Normalises *psi_raw* by its own min/max and draws *nlevels* contours
    between :math:`\hat{\psi} = 0.1` and :math:`0.9`.

    .. note::
        Requires ``mygs`` to be set as a module-level (or notebook-level)
        name before calling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    psi_raw : ndarray
        Raw poloidal flux on the TokaMaker mesh.
    nlevels : int
        Number of contour levels.
    """
    _p_lo = psi_raw.min()
    _p_hi = psi_raw.max()
    if abs(_p_hi - _p_lo) < 1e-10:
        return
    _psi_n = (psi_raw - _p_lo) / (_p_hi - _p_lo)
    ax.tricontour(mygs.r[:, 0], mygs.r[:, 1], mygs.lc, _psi_n,
                  levels=np.linspace(0.1, 0.9, nlevels),
                  colors='steelblue', linewidths=0.5, alpha=0.4)


def _isoflux_deviation_plot(ax, fig, iso_pts, lcfs_pts, R_bnd, Z_bnd,
                             max_dev_mm=10.0, max_seg_len=0.1):
    """Colour-coded boundary deviation. Returns (devs, max_mm, rms_mm)."""
    tree = _cKDTree(lcfs_pts)
    devs, _ = tree.query(iso_pts)

    dev_cmap = _mcolors_dev.LinearSegmentedColormap.from_list(
        'dev_cmap', ['limegreen', 'yellow', 'red'])
    dev_norm = _mcolors_dev.Normalize(vmin=0.0, vmax=max_dev_mm * 1e-3)

    seg_list, col_list = [], []
    for _si in range(len(iso_pts) - 1):
        p0, p1 = iso_pts[_si], iso_pts[_si + 1]
        if np.linalg.norm(p1 - p0) <= max_seg_len:
            seg_list.append([p0, p1])
            col_list.append(devs[_si])

    if seg_list:
        lc_coll = _LC(np.array(seg_list), cmap=dev_cmap, norm=dev_norm,
                      linewidth=3, zorder=5)
        lc_coll.set_array(np.array(col_list))
        ax.add_collection(lc_coll)
        cax = ax.inset_axes([1.02, 0, 0.05, 1])
        cbar = fig.colorbar(lc_coll, cax=cax, label='Boundary deviation [mm]')
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x * 1e3:.1f}'))

    lcfs_R, lcfs_Z = [lcfs_pts[0, 0]], [lcfs_pts[0, 1]]
    for _bi in range(1, len(lcfs_pts)):
        if np.linalg.norm(lcfs_pts[_bi] - lcfs_pts[_bi - 1]) > max_seg_len:
            lcfs_R.append(np.nan); lcfs_Z.append(np.nan)
        lcfs_R.append(lcfs_pts[_bi, 0]); lcfs_Z.append(lcfs_pts[_bi, 1])
    ax.plot(lcfs_R, lcfs_Z, 'k-', lw=_LW, label='TokaMaker LCFS')

    ax.set_xlabel('$R$ [m]'); ax.set_ylabel('$Z$ [m]')
    ax.set_aspect('equal')
    ax.set_xlim(R_bnd.min() - 0.12, R_bnd.max() + 0.12)
    ax.set_ylim(Z_bnd.min() - 0.20, Z_bnd.max() + 0.20)
    ax.legend(fontsize=8, loc='lower right')

    max_mm = devs.max() * 1e3
    rms_mm = np.sqrt((devs**2).mean()) * 1e3
    return devs, max_mm, rms_mm


def plot_tokamaker_comparison(mygs, all_results, plot_idx=None):
    """Compare TokaMaker reconstructions against source geqdsk files.

    .. note::
        Requires ``mygs`` (TokaMaker solver object) and ``all_results``
        (dict mapping geqdsk file names to result dicts) to be available
        as module-level or notebook-level names.

    Parameters
    ----------
    plot_idx : int or None
        ``None`` → overview: overplot ALL results on each subplot.
        ``int``  → single-shot: compare that result against its geqdsk.

    Boundary panels (row 2):
      axes[2,0] — TokaMaker LCFS + geqdsk boundary + core flux surface
                  contours.
      axes[2,1] — Quantified boundary deviation: colour-coded segments
                  (green→red) showing nearest-neighbour distance from
                  each isoflux target point to the TokaMaker LCFS;
                  overview mode shows a max/RMS bar chart.
      axes[2,2] — :math:`l_i(1)` and :math:`I_p` % error bars
                  (TokaMaker − geqdsk) / |geqdsk|.
    """
    keys = list(all_results.keys())

    # colourblind-safe palette (Wong / Okabe-Ito)
    _C1 = '#0072B2'   # deep blue
    _C2 = '#D55E00'   # deep orange
    _C3 = '#009E73'   # green

    # -----------------------------------------------------------------------
    # OVERVIEW MODE
    # -----------------------------------------------------------------------
    if plot_idx is None:
        N = max(len(keys), 1)
        colors = _cm.tab10(np.linspace(0, 0.9, N))

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle('TokaMaker overview — all reconstructions', fontsize=13, y=0.98)

        _li_efit_vals, _li_tkmkr_vals, _short_labels = [], [], []
        _li_pct_vals, _Ip_pct_vals = [], []
        _ov_max_dev_mm, _ov_rms_dev_mm = [], []

        ax_te = axes[1, 2].twinx()

        for color, gkey in zip(colors, keys):
            r = all_results[gkey]
            psi_N = r['psi_N_grid']
            lbl = gkey.replace('.geqdsk', '')
            _short_labels.append(lbl)

            # (0,0) total j_phi — dashed=TokaMaker, dotted=geqdsk
            axes[0, 0].plot(psi_N, r['j_phi_fit'] / 1e6, color=color, lw=_LW, ls='--', label=lbl)
            axes[0, 0].plot(psi_N, r['eqdsk_jtor'] / 1e6, color=color, lw=_LW, ls=':', alpha=0.75)

            # (0,1) j_phi components (both TokaMaker — dashed=j_ind, dash-dot=j_BS)
            axes[0, 1].plot(psi_N, r['j_inductive_fit'] / 1e6, color=color, lw=_LW, ls='--',
                            label=f'{lbl} $j_{{\\rm ind}}$')
            axes[0, 1].plot(psi_N, r['j_BS_used'] / 1e6, color=color, lw=_LW, ls='-.',
                            label=f'{lbl} $j_{{BS}}$')

            # (0,2) residuals
            res = (r['j_phi_fit'] - r['eqdsk_jtor']) / 1e6
            rms = np.sqrt(np.mean(res**2))
            axes[0, 2].plot(psi_N, res, color=color, lw=_LW,
                            label=f'{lbl}  RMS={rms:.4f}')

            # (1,0) pressure — dashed=TokaMaker, dotted=geqdsk
            axes[1, 0].plot(psi_N, r['pres_tokamaker'] / 1e3, color=color, lw=_LW, ls='--', label=lbl)
            axes[1, 0].plot(psi_N, r['eqdsk_pres'] / 1e3, color=color, lw=_LW, ls=':', alpha=0.75)

            # (1,1) pprime (TokaMaker only)
            axes[1, 1].plot(psi_N, np.abs(r['pprime']), color=color, lw=_LW, ls='--', label=lbl)

            # (1,2) ne dashed, te dash-dot
            axes[1, 2].plot(psi_N, r['ne'] / 1e19, color=color, lw=_LW, ls='--')
            ax_te.plot(psi_N, r['te'] / 1e3, color=color, lw=_LW, ls='-.')

            # (2,0) FF' — dashed=TokaMaker, dotted=geqdsk
            axes[2, 0].plot(psi_N, r['ffprime'], color=color, lw=_LW, ls='--', label=lbl)
            axes[2, 0].plot(psi_N, r['eqdsk_ffprim'], color=color, lw=_LW, ls=':', alpha=0.75)

            # Boundary deviation stats (LCFS still needed for deviation panel)
            _lcfs = _lcfs_from_psi(mygs, r['psi'], r['isoflux_pts'], r.get('psi_lcfs_val'))
            _tree_ov = _cKDTree(_lcfs)
            _devs_ov, _ = _tree_ov.query(r['isoflux_pts'])
            _ov_max_dev_mm.append(_devs_ov.max() * 1e3)
            _ov_rms_dev_mm.append(np.sqrt((_devs_ov**2).mean()) * 1e3)

            # li and Ip % errors
            li_eqdsk = r['eqdsk_li']
            _li_efit = li_eqdsk.get('li(1)_EFIT', li_eqdsk.get('li(1)', 0.0))
            _li_efit_vals.append(_li_efit)
            _li_tkmkr_vals.append(r['li_final'])
            _li_pct_vals.append((r['li_final'] - _li_efit) / abs(_li_efit) * 100.0)
            _Ip_ref = abs(r['eqdsk_Ip'])
            _Ip_tkmkr_ov = r.get('Ip_tokamaker', float('nan'))
            _Ip_pct_vals.append((_Ip_tkmkr_ov - _Ip_ref) / _Ip_ref * 100.0)

        # ----- Axes labels / titles -----
        axes[0, 0].set_xlabel(r'$\psi_N$')
        axes[0, 0].set_ylabel(r'$j_\phi$ [MA m$^{-2}$]')
        axes[0, 0].set_title(r'Total $j_\phi$ (dashed=TokaMaker, dotted=geqdsk)')
        axes[0, 0].legend(fontsize=7)
        axes[0, 0].grid(ls=':')

        axes[0, 1].set_xlabel(r'$\psi_N$')
        axes[0, 1].set_ylabel(r'$j$ [MA m$^{-2}$]')
        axes[0, 1].set_title(r'$j_\phi$ components (dashed=$j_{\rm ind}$, dash-dot=$j_{BS}$)')
        axes[0, 1].legend(fontsize=6, ncol=2)
        axes[0, 1].grid(ls=':')

        axes[0, 2].axhline(0, color='k', ls=':', lw=_LW)
        axes[0, 2].set_xlabel(r'$\psi_N$')
        axes[0, 2].set_ylabel(r'$\Delta j_\phi$ [MA m$^{-2}$]')
        axes[0, 2].set_title(r'$j_\phi$ residuals (TokaMaker \u2212 geqdsk)')
        axes[0, 2].legend(fontsize=7)
        axes[0, 2].grid(ls=':')

        axes[1, 0].set_xlabel(r'$\psi_N$')
        axes[1, 0].set_ylabel(r'$p$ [kPa]')
        axes[1, 0].set_title('Pressure (dashed=TokaMaker, dotted=geqdsk)')
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(ls=':')

        axes[1, 1].set_xlabel(r'$\psi_N$')
        axes[1, 1].set_ylabel(r"$P'$ [Pa Wb$^{-1}$]")
        axes[1, 1].set_title(r"$P'(\psi_N)$")
        axes[1, 1].legend(fontsize=7)
        axes[1, 1].grid(ls=':')

        axes[1, 2].set_xlabel(r'$\psi_N$')
        axes[1, 2].set_ylabel(r'$n_e$ [$10^{19}$ m$^{-3}$] (dashed)')
        ax_te.set_ylabel(r'$T_e$ [keV] (dash-dot)')
        axes[1, 2].set_title('Kinetic profiles')
        axes[1, 2].grid(ls=':')
        from matplotlib.lines import Line2D as _L2D
        axes[1, 2].legend(
            [_L2D([0], [0], color='k', lw=_LW, ls='--'),
             _L2D([0], [0], color='k', lw=_LW, ls='-.')],
            [r'$n_e$', r'$T_e$'], fontsize=8, loc='upper right')

        # (2,0) FF' panel
        axes[2, 0].set_xlabel(r'$\psi_N$')
        axes[2, 0].set_ylabel(r"$FF'$ [T$^2$ m$^2$ Wb$^{-1}$]")
        axes[2, 0].set_title(r"$FF'(\psi_N)$ (dashed=TokaMaker, dotted=geqdsk)")
        axes[2, 0].legend(fontsize=7)
        axes[2, 0].grid(ls=':')

        # (2,1) boundary deviation bar chart
        _x_bars = np.arange(len(keys))
        _bar_w = 0.35
        axes[2, 1].bar(_x_bars - _bar_w/2, _ov_max_dev_mm, _bar_w,
                       label='Max dev.', color=_C2, edgecolor='k')
        axes[2, 1].bar(_x_bars + _bar_w/2, _ov_rms_dev_mm, _bar_w,
                       label='RMS dev.', color=_C1, edgecolor='k')
        axes[2, 1].set_xticks(_x_bars)
        axes[2, 1].set_xticklabels(_short_labels, rotation=30, ha='right', fontsize=8)
        axes[2, 1].set_ylabel('Boundary deviation [mm]')
        axes[2, 1].set_title('TokaMaker LCFS vs geqdsk boundary deviation')
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(axis='y', ls=':')

        # (2,2) % error bars for li(1) and Ip per shot
        _x_err = np.arange(len(keys))
        _w_err = 0.35
        axes[2, 2].bar(_x_err - _w_err/2, _li_pct_vals, _w_err,
                       label=r'$l_i(1)$', color=_C1, edgecolor='k')
        axes[2, 2].bar(_x_err + _w_err/2, _Ip_pct_vals, _w_err,
                       label=r'$I_p$', color=_C2, edgecolor='k')
        axes[2, 2].axhline(0, color='k', ls='--', lw=0.8)
        axes[2, 2].set_xticks(_x_err)
        axes[2, 2].set_xticklabels(_short_labels, rotation=30, ha='right', fontsize=8)
        axes[2, 2].set_ylabel('% error  (TokaMaker \u2212 geqdsk) / |geqdsk|')
        axes[2, 2].set_title(r'$l_i(1)$ and $I_p$ % error')
        axes[2, 2].legend(fontsize=8)
        axes[2, 2].grid(axis='y', ls=':')

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()

    # -----------------------------------------------------------------------
    # SINGLE-SHOT COMPARISON MODE
    # -----------------------------------------------------------------------
    else:
        geqdsk_key = keys[plot_idx]
        r = all_results[geqdsk_key]
        eqdsk_ref = read_geqdsk(geqdsk_key)

        psi_N = r['psi_N_grid']
        R_bnd = r['eqdsk_boundary_R']
        Z_bnd = r['eqdsk_boundary_Z']
        residual_jphi = r['j_phi_fit'] - r['eqdsk_jtor']
        rms_jphi = np.sqrt(np.mean(residual_jphi**2))

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(f'TokaMaker vs geqdsk comparison:  {geqdsk_key}', fontsize=13, y=0.98)

        # (0,0) Total j_phi
        ax = axes[0, 0]
        ax.plot(psi_N, r['eqdsk_jtor'] / 1e6, 'k-', lw=_LW, label=r'geqdsk $j_\phi$')
        ax.plot(psi_N, r['j_phi_fit'] / 1e6, color=_C2, ls='--', lw=_LW,
                label=r'TokaMaker $j_\phi$')
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$j_\phi$ [MA m$^{-2}$]')
        ax.set_title(r'Total $j_\phi$'); ax.legend(fontsize=8); ax.grid(ls=':')

        # (0,1) j_phi components
        ax = axes[0, 1]
        ax.plot(psi_N, r['j_inductive_fit'] / 1e6, color=_C1, lw=_LW,
                label=r'$j_\mathrm{inductive}$ (fit)')
        ax.plot(psi_N, r['j_BS_used'] / 1e6, color=_C3, lw=_LW, ls='-.',
                label=f'$j_{{BS}}$ (\u00d7{r["bs_factor_final"]:.3f})')
        ax.plot(psi_N, r['j_phi_fit'] / 1e6, color=_C2, ls='--', lw=_LW,
                label=r'$j_\mathrm{ind} + j_{BS}$')
        ax.plot(psi_N, r['eqdsk_jtor'] / 1e6, 'k-', lw=_LW,
                label=r'geqdsk $j_\phi$')
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$j$ [MA m$^{-2}$]')
        ax.set_title(r'$j_\phi$ components'); ax.legend(fontsize=7); ax.grid(ls=':')

        # (0,2) Residual
        ax = axes[0, 2]
        ax.plot(psi_N, residual_jphi / 1e6, color=_C2, lw=_LW)
        ax.axhline(0, color='k', ls=':', lw=_LW)
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$\Delta j_\phi$ [MA m$^{-2}$]')
        ax.set_title(rf'$j_\phi$ residual  (RMS = {rms_jphi/1e6:.4f} MA m$^{{-2}}$)')
        ax.grid(ls=':')

        # (1,0) Pressure
        ax = axes[1, 0]
        ax.plot(psi_N, eqdsk_ref.pres / 1e3, 'k-', lw=_LW, label='geqdsk $p$')
        ax.plot(psi_N, r['pres_tokamaker'] / 1e3, color=_C2, ls='--', lw=_LW,
                label='TokaMaker $p$ (kinetic)')
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$p$ [kPa]')
        ax.set_title('Pressure profile'); ax.legend(fontsize=8); ax.grid(ls=':')

        # (1,1) pprime
        ax = axes[1, 1]
        ax.plot(psi_N, np.abs(eqdsk_ref.pprime), 'k-', lw=_LW, label="geqdsk $P'$")
        ax.plot(psi_N, np.abs(r['pprime']), color=_C2, ls='--', lw=_LW, label="TokaMaker $P'$")
        ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r"$P'$ [Pa Wb$^{-1}$]")
        ax.set_title(r"$P'(\psi)$ comparison"); ax.legend(fontsize=8); ax.grid(ls=':')

        # (1,2) Kinetics
        ax = axes[1, 2]; ax2 = ax.twinx()
        ax.plot(psi_N, r['ne'] / 1e19, color=_C1, lw=_LW,
                label=r'$n_e$ [$10^{19}$ m$^{-3}$]')
        ax2.plot(psi_N, r['te'] / 1e3, color=_C2, lw=_LW, ls='--',
                 label=r'$T_e$ [keV]')
        ax.set_xlabel(r'$\psi_N$')
        ax.set_ylabel(r'$n_e$ [$10^{19}$ m$^{-3}$]', color=_C1)
        ax2.set_ylabel(r'$T_e$ [keV]', color=_C2)
        ax.tick_params(axis='y', labelcolor=_C1)
        ax2.tick_params(axis='y', labelcolor=_C2)
        h1_, l1_ = ax.get_legend_handles_labels()
        h2_, l2_ = ax2.get_legend_handles_labels()
        ax.legend(h1_ + h2_, l1_ + l2_, fontsize=7, loc='upper right')
        ax.set_title('Kinetic profiles'); ax.grid(ls=':')

        # --- Shared LCFS extraction ---
        _tk_lcfs = _lcfs_from_psi(mygs,r['psi'], r['isoflux_pts'], r.get('psi_lcfs_val'))

        # --- (2,0) FF' comparison ---
        ax_ffp = axes[2, 0]
        ax_ffp.plot(psi_N, eqdsk_ref.ffprim, 'k-', lw=_LW, label=r"geqdsk $FF'$")
        ax_ffp.plot(psi_N, r['ffprime'], color=_C2, ls='--', lw=_LW, label=r"TokaMaker $FF'$")
        ax_ffp.set_xlabel(r'$\psi_N$'); ax_ffp.set_ylabel(r"$FF'$ [T$^2$ m$^2$ Wb$^{-1}$]")
        ax_ffp.set_title(r"$FF'(\psi)$ comparison"); ax_ffp.legend(fontsize=8); ax_ffp.grid(ls=':')

        # --- (2,1) Quantified boundary deviation ---
        ax_dev = axes[2, 1]
        mygs.plot_machine(fig, ax_dev)
        _devs_ss, _dev_max_mm, _dev_rms_mm = _isoflux_deviation_plot(
            ax_dev, fig,
            iso_pts=r['isoflux_pts'],
            lcfs_pts=_tk_lcfs,
            R_bnd=R_bnd, Z_bnd=Z_bnd,
            max_dev_mm=10.0, max_seg_len=0.1)
        ax_dev.set_title(
            f'Boundary deviation  max={_dev_max_mm:.2f} mm  RMS={_dev_rms_mm:.2f} mm')

        # (2,2) % error bars for li(1) and Ip
        ax = axes[2, 2]
        li_eqdsk  = r['eqdsk_li']
        li_efit   = li_eqdsk.get('li(1)_EFIT', li_eqdsk.get('li(1)', 0.0))
        li_tkmkr  = r['li_final']
        li_pct    = (li_tkmkr - li_efit) / abs(li_efit) * 100.0
        Ip_eqdsk  = abs(r['eqdsk_Ip'])
        Ip_tkmkr  = r.get('Ip_tokamaker', float('nan'))
        Ip_pct    = (Ip_tkmkr - Ip_eqdsk) / Ip_eqdsk * 100.0
        _pct_vals = [li_pct, Ip_pct]
        _bar_cols = [_C1 if v >= 0 else _C2 for v in _pct_vals]
        bars = ax.bar([0, 1], _pct_vals, color=_bar_cols, edgecolor='k', width=0.5)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([r'$l_i(1)$', r'$I_p$'], fontsize=11)
        ax.set_ylabel('% error  (TokaMaker \u2212 geqdsk) / |geqdsk|')
        ax.set_title(r'TokaMaker vs geqdsk: $l_i(1)$ and $I_p$ % error')
        ax.grid(axis='y', ls=':')
        for bar_, val_ in zip(bars, _pct_vals):
            _yoff = abs(val_) * 0.05 + 0.02
            _va   = 'bottom' if val_ >= 0 else 'top'
            _ytxt = val_ + (_yoff if val_ >= 0 else -_yoff)
            ax.text(bar_.get_x() + bar_.get_width() / 2, _ytxt,
                    f'{val_:+.3f}%', ha='center', va=_va, fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()

        print(f"\n--- Summary for {geqdsk_key} ---")
        print(f"  bs_factor_final : {r['bs_factor_final']:.6f}")
        print(f"  li(1) achieved  : {r['li_final']:.6f}")
        print(f"  j_phi RMS error : {rms_jphi:.2f} A/m^2  ({rms_jphi/1e6:.4f} MA/m^2)")
        print(f"  eqdsk Ip        : {r['eqdsk_Ip']:.0f} A")
        print(f"  TokaMaker Ip    : {Ip_tkmkr:.0f} A  ({Ip_pct:+.3f}%)")
        print(f"  Boundary dev.   : max={_dev_max_mm:.2f} mm  RMS={_dev_rms_mm:.2f} mm")


# ====================================================================
#  Core drawing functions (no I/O -- operate on axes + arrays)
# ====================================================================
def draw_kinetic_profiles(axes, psi_N, ne, ni, te, ti,
                          sigma_ne, sigma_ni, sigma_te, sigma_ti,
                          perturbed_data_list=None):
    r"""Draw kinetic profiles (ne, ni, Te, Ti) on a 2x2 axes array.

    Parameters
    ----------
    axes : ndarray of Axes, shape (2, 2)
        ``[0,0]`` = :math:`n_e`, ``[0,1]`` = :math:`n_i`,
        ``[1,0]`` = :math:`T_e`, ``[1,1]`` = :math:`T_i`.
    psi_N : 1-D array
        Normalised poloidal flux grid (baseline).
    ne, ni, te, ti : 1-D arrays
        Baseline kinetic profiles.
    sigma_ne, sigma_ni, sigma_te, sigma_ti : 1-D arrays
        1-:math:`\sigma` uncertainty envelopes.
    perturbed_data_list : list[dict] or None
        Each dict must have keys ``'n_e'``, ``'n_i'``,
        ``'T_e'``, ``'T_i'``.
    """
    _pairs = [
        #  axis       orig  scale  sigma      color        label      ylabel
        (axes[0, 0], ne, 1.0,  sigma_ne, "0.55", r"$n_e$", r"$n_e$ [m$^{-3}$]"),
        (axes[0, 1], ni, 1.0,  sigma_ni, "0.55", r"$n_i$", r"$n_i$ [m$^{-3}$]"),
        (axes[1, 0], te, 1e-3, sigma_te, "0.55", r"$T_e$", r"$T_e$ [keV]"),
        (axes[1, 1], ti, 1e-3, sigma_ti, "0.55", r"$T_i$", r"$T_i$ [keV]"),
    ]
    _keys = ["n_e", "n_i", "T_e", "T_i"]

    # ---- baseline + sigma bands ------------------------------------------
    for a, orig, scale, sig, clr, lbl, ylabel in _pairs:
        a.cla()
        a.plot(psi_N, orig * scale, c="k", lw=2,
               label=f"input {lbl}", zorder=1)
        a.fill_between(
            psi_N,
            (orig - sig) * scale,
            (orig + sig) * scale,
            alpha=0.25, color=clr,
            label=r"$\pm\,1\sigma_{\rm exp}$", zorder=1,
        )
        a.plot(psi_N, (orig + 2 * sig) * scale, c="k", ls=":",
               lw=1.5, alpha=0.5, label=r"$\pm\,2\sigma_{\rm exp}$",
               zorder=1)
        a.plot(psi_N, (orig - 2 * sig) * scale, c="k", ls=":",
               lw=1.5, alpha=0.5, zorder=1)
        a.grid(ls=":")
        if ylabel:
            a.set_ylabel(ylabel)

    # ---- overlay perturbed profiles --------------------------------------
    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            _psi_pert = data.get("psi_N_kinetic", psi_N)
            for (a, orig, scale, sig, clr, lbl, ylabel), key in zip(
                _pairs, _keys
            ):
                a.plot(
                    _psi_pert, data[key] * scale,
                    c=_GOLD, alpha=0.55, lw=1.0,
                    label=f"perturbed ({n_equils})" if i == 0 else None,
                    zorder=3,
                )

    # ---- legends and axis labels -----------------------------------------
    for a, *_ in _pairs:
        a.legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel(r"$\hat{\psi}$")
    axes[1, 1].set_xlabel(r"$\hat{\psi}$")


def draw_pressure_profiles(ax, psi_N, pressure, perturbed_data_list=None,
                           pressure_thermal=None, bl=None):
    """Draw the TOTAL pressure overlay (baseline + perturbed draws), with the
    baseline thermal / impurity / fast components as gray reference lines.

    Only the total is drawn as a draw cloud -- a single, easy-to-follow gold
    cloud. When ``bl`` (the baseline dict) is given, the fixed decomposition is
    overlaid in gray with distinct line styles -- thermal dashed, fast dotted,
    impurity dash-dot -- so thermal+impurity+fast = the total is visible without
    a second panel. Fast is the fixed beam pressure and impurity perturbs only
    negligibly, so these baseline curves represent every draw. Falls back to just
    the dashed thermal line when only ``pressure_thermal`` is available.

    Parameters
    ----------
    ax : Axes
    psi_N : 1-D array
    pressure : 1-D array
        Baseline total pressure.
    perturbed_data_list : list[dict] or None
        Each dict provides ``'pressure'`` (total).
    pressure_thermal : 1-D array or None
        Baseline thermal pressure (dashed reference fallback).
    bl : dict or None
        Baseline dict; when present, thermal/impurity/fast are overlaid.
    """
    _kPa = 1e-3
    ax.cla()

    # perturbed TOTAL cloud only (single colour -> easy to read)
    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            if "pressure" in data:
                ax.plot(psi_N, data["pressure"] * _kPa,
                        c=_GOLD, alpha=0.55, lw=1.0,
                        label=f"perturbed total ({n_equils})" if i == 0 else None,
                        zorder=3)

    ax.plot(psi_N, pressure * _kPa, c="k", lw=2,
            label="baseline total", zorder=5)

    comp = _pressure_components(bl, psi_eq=psi_N) if bl is not None else None
    if comp is not None:
        thermal, imp, fast = comp
        ax.plot(psi_N, thermal * _kPa, c="0.45", lw=1.1, ls="--",
                label="thermal", zorder=4)
        ax.plot(psi_N, fast * _kPa, c="0.45", lw=1.1, ls=":",
                label="fast", zorder=4)
        ax.plot(psi_N, imp * _kPa, c="0.45", lw=1.1, ls="-.",
                label="impurity (C)", zorder=4)
    elif pressure_thermal is not None:
        ax.plot(psi_N, pressure_thermal * _kPa, c="0.45", lw=1.2, ls="--",
                label="baseline thermal", zorder=4)

    ax.grid(ls=":")
    ax.set_xlabel(r"$\hat{\psi}$")
    ax.set_ylabel("Pressure [kPa]")
    ax.legend(loc="best", fontsize=7)


def _pressure_components(bl, psi_eq=None):
    r"""Split a baseline dict into (thermal, impurity, fast) pressure.

    Thermal is stored (``pressure_thermal``); impurity is recomputed from the
    stored kinetics via the same ``impurity_pressure`` the solve used; fast is
    the remainder ``total - thermal - impurity`` (the fixed beam/fast-ion
    pressure, which does not perturb). Returns ``None`` if the inputs needed
    are missing (e.g. an older archive without ``pressure_thermal``).

    Dual-grid archives (e.g. the geqdsk p-file path) store kinetics on
    ``psi_N_kinetic`` while the total/thermal pressure live on the equilibrium
    grid ``psi_eq``. The impurity term -- recomputed from the kinetics -- is
    regridded onto the total's grid before differencing. Single-grid archives
    (OMAS) skip this since the shapes already match.

    Fast ions.  The solve derives ``Z_imp`` and ``p_imp`` on the THERMAL
    electron density ``ne - z_fast``, and the archived ``n_i`` is itself
    thermal, so this path must subtract ``z_fast`` too -- pairing the archive's
    full ``ne`` with a thermal ``n_i`` makes the inversion charge the whole
    beam to the impurity (a ~30x p_imp on a 20 % fast fraction).  Archives
    written before ``z_fast`` was stored carry no such key; those are
    pre-subtraction archives whose ``n_i`` is a total, and the full ``ne`` is
    the right partner for them.  Either way the pair is self-consistent.
    """
    if "pressure" not in bl or "pressure_thermal" not in bl:
        return None
    total = np.asarray(bl["pressure"], float)
    thermal = np.asarray(bl["pressure_thermal"], float)
    imp = np.zeros_like(total)
    try:
        ne = np.asarray(bl["n_e"], float)
        ni = np.asarray(bl["n_i"], float)
        ti = np.asarray(bl["T_i"], float)
        zeff = np.asarray(bl["aux_zeff"] if "aux_zeff" in bl else bl["Zeff"], float)
        from .physics import effective_impurity_charge, impurity_pressure
        # ne and n_i must describe the same population: a thermal n_i (any
        # archive carrying z_fast) pairs with ne - z_fast, a total one with
        # the full ne.  Either way the pair is self-consistent.
        if "z_fast" in bl:
            zf = np.asarray(bl["z_fast"], float)   # same grid as n_e / n_i
            if zf.shape == ne.shape:
                ne = np.maximum(ne - zf, 0.0)
        # Prefer the archived Z_imp: inverting (ne, ni, Zeff) for it needs to
        # know whether Zeff's numerator counts the fast ions, and the archive
        # does not record that.  The inversion stays as the fallback for
        # archives written before Z_imp was stored (no fast ions there, where
        # the two conventions coincide and it is exact).
        Z_imp = bl.get("Z_imp") if hasattr(bl, "get") else None
        if not Z_imp:
            Z_imp = effective_impurity_charge(ne, ni, zeff)
        imp = impurity_pressure(ne, ni, ti, Z_imp)
    except Exception:
        imp = np.zeros_like(total)
    # Align kinetics-derived terms (imp; thermal defensively) onto the total's
    # grid when an archive is dual-grid.
    psi_kin = np.asarray(bl["psi_N_kinetic"], float) if "psi_N_kinetic" in bl else None
    def _to_total(arr):
        if arr.shape == total.shape:
            return arr
        if psi_eq is not None and psi_kin is not None and arr.shape == psi_kin.shape:
            return np.interp(np.asarray(psi_eq, float), psi_kin, arr)
        return None
    imp = _to_total(imp)
    if imp is None:                     # can't regrid -> drop impurity, keep thermal/fast
        imp = np.zeros_like(total)
    thermal = _to_total(thermal)
    if thermal is None:                 # can't align thermal -> no component overlay
        return None
    fast = total - thermal - imp
    return thermal, imp, fast


def draw_jphi_total(ax, psi_N, j_phi, sigma_jphi,
                    perturbed_data_list=None):
    r"""Draw total :math:`j_\phi` with uncertainty bands on a single axes.

    Parameters
    ----------
    ax : Axes
    psi_N : 1-D array
    j_phi : 1-D array
        Baseline total :math:`j_\phi` [A m^-2].
    sigma_jphi : 1-D array
    perturbed_data_list : list[dict] or None
        Each dict must have ``'j_phi'``.
    """
    _MA = 1e-6  # A → MA

    ax.cla()
    ax.plot(psi_N, j_phi * _MA, c="k", lw=2,
            label=r"input $j_\phi$", zorder=1)
    ax.fill_between(
        psi_N,
        (j_phi - sigma_jphi) * _MA,
        (j_phi + sigma_jphi) * _MA,
        alpha=0.22, color="0.55",
        label=r"$\pm\,1\sigma_{\rm exp}$", zorder=1,
    )
    ax.plot(psi_N, (j_phi + 2 * sigma_jphi) * _MA, c="k", ls=":", lw=1.5,
            alpha=0.5, label=r"$\pm\,2\sigma_{\rm exp}$", zorder=1)
    ax.plot(psi_N, (j_phi - 2 * sigma_jphi) * _MA, c="k", ls=":", lw=1.5,
            alpha=0.5, zorder=1)
    ax.set_ylabel(r"$j_\phi$ [MA/m$^2$]")
    ax.set_xlabel(r"$\hat{\psi}$")
    ax.grid(ls=":")

    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            ax.plot(psi_N, data["j_phi"] * _MA, c=_GOLD,
                    lw=1.0, alpha=0.55,
                    label=f"perturbed ({n_equils})" if i == 0 else None,
                    zorder=3)

    ax.legend(loc="best", fontsize=8)


def draw_jphi_components(axes, psi_N, perturbed_data_list=None):
    r"""Draw the :math:`j_\phi` component decomposition on a (2,) axes array.

    ``axes[0]`` = the bootstrap variant ACTUALLY summed into :math:`j_\phi` --
    the isolated edge spike when the run used ``isolate_edge_jBS=True`` (archives
    that carry ``j_BS,edge``), otherwise the full Sauter ``j_BS``. Only that one
    variant is drawn -- never both the isolated edge and the full Sauter
    bootstrap. ``axes[1]`` = :math:`j_{\rm inductive}`, computed the FAITHFUL way
    :func:`plot_jphi` uses -- ``j_phi - that_j_BS`` -- so it closes to the total
    and never exceeds it, rather than the post-hoc stored ``j_inductive`` (which
    can over-state the core). Total :math:`j_\phi` is the dashed reference.

    Parameters
    ----------
    axes : array-like of 2 Axes
    psi_N : 1-D array
    perturbed_data_list : list[dict] or None
        Each dict must have ``'j_phi'`` and the applied bootstrap
        (``'j_BS,edge'`` if isolated, else ``'j_BS'``).
    """
    _MA = 1e-6  # A → MA
    for ax in axes:
        ax.cla()
        ax.grid(ls=":")

    if not perturbed_data_list:
        axes[0].set_ylabel(r"$j_{\rm BS}$ [MA/m$^2$]")
        axes[1].set_ylabel(r"$j_{\rm inductive}$ [MA/m$^2$]")
        axes[-1].set_xlabel(r"$\hat{\psi}$")
        return

    n_equils = len(perturbed_data_list)
    # Which bootstrap variant was applied to j_phi? The isolated edge spike if
    # the run stored one (isolate_edge_jBS=True), else the full Sauter j_BS.
    has_spike = "j_BS,edge" in perturbed_data_list[0]
    bs_key = "j_BS,edge" if has_spike else "j_BS"
    bs_label = r"$j_{\rm BS,edge}$" if has_spike else r"$j_{\rm BS}$"

    for i, data in enumerate(perturbed_data_list):
        if "j_phi" not in data or bs_key not in data:
            continue
        lbl = f"perturbed ({n_equils})" if i == 0 else None
        total = np.asarray(data["j_phi"], float)
        bs = np.asarray(data[bs_key], float)
        jind = total - bs   # faithful inductive: closes to total, <= total

        # reference: total j_phi on both panels
        for ax in axes:
            ax.plot(psi_N, total * _MA, c="k", ls="--", lw=1.2, alpha=0.4,
                    label=r"$j_\phi$ (total)" if i == 0 else None, zorder=1)
        axes[0].plot(psi_N, bs * _MA, c=_GOLD, ls="-", lw=1.5, alpha=0.7,
                     label=lbl, zorder=3)
        axes[1].plot(psi_N, jind * _MA, c=_GOLD, ls="--", lw=1.5, alpha=0.7,
                     label=lbl, zorder=3)

    axes[0].set_ylabel(bs_label + r" [MA/m$^2$]")
    axes[1].set_ylabel(r"$j_{\rm inductive}$ [MA/m$^2$]")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(r"$\hat{\psi}$")


def draw_jphi_profiles(axes, psi_N, j_phi, sigma_jphi,
                       perturbed_data_list=None):
    r"""**Deprecated** -- use :func:`draw_jphi_total` and
    :func:`draw_jphi_components` instead.

    Draws on 3 vertically stacked axes for backward compatibility.
    """
    warnings.warn(
        "draw_jphi_profiles() is deprecated.  Use draw_jphi_total() and "
        "draw_jphi_components() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    draw_jphi_total(axes[0], psi_N, j_phi, sigma_jphi,
                    perturbed_data_list=perturbed_data_list)
    draw_jphi_components(axes[1:], psi_N,
                         perturbed_data_list=perturbed_data_list)


def _draw_boundary_panels(ax_lcfs, ax_zoom, boundaries):
    """LCFS-shape overlay + zoom on the maximum-deviation region.

    ``ax_lcfs`` gets the full boundary curves (black dashed baseline, gold
    draws); ``ax_zoom`` magnifies the neighbourhood of the baseline vertex
    with the largest mean deviation across draws -- at full scale the curves
    are indistinguishable (mm deviations on a metre-scale plot). Per-draw
    RMS/max deviation numbers live in ``plot_traces`` / ``plot_spec_summary``
    (the bar panel that used to be here duplicated them, and silently
    dropped draws with no stored boundary).
    """
    if not boundaries:
        return
    bR0, bZ0 = boundaries[0]
    ax_lcfs.plot(bR0, bZ0, "k--", lw=1.0, label="baseline", zorder=2)
    for i, (bR, bZ) in enumerate(boundaries[1:], 1):
        ax_lcfs.plot(bR, bZ, "-", color=_GOLD, lw=0.7, alpha=0.7,
                     label="perturbed" if i == 1 else None, zorder=1)
    ax_lcfs.set_aspect("equal")
    ax_lcfs.set_xlabel("R [m]"); ax_lcfs.set_ylabel("Z [m]")
    ax_lcfs.set_title("LCFS shape variation")
    _framed_legend(ax_lcfs, fontsize=8)
    ax_lcfs.grid(ls=":")

    # mean deviation of each baseline vertex across draws -> zoom center
    ref_pts = np.column_stack([bR0, bZ0])
    dev_sum = np.zeros(len(ref_pts))
    n_dev = 0
    for bR, bZ in boundaries[1:]:
        tree = _cKDTree(np.column_stack([bR, bZ]))
        d, _ = tree.query(ref_pts)
        dev_sum += d
        n_dev += 1
    if n_dev == 0:
        ax_zoom.axis("off")
        return
    dev_mean = dev_sum / n_dev
    ic = int(np.argmax(dev_mean))
    Rc, Zc = ref_pts[ic]
    half = max(4.0 * float(np.max(dev_mean)), 0.01)   # window >= ±1 cm
    for bR, bZ in boundaries[1:]:
        ax_zoom.plot(bR, bZ, "-", color=_GOLD, lw=1.0, alpha=0.8, zorder=1)
    ax_zoom.plot(bR0, bZ0, "k--", lw=1.2, zorder=2)
    ax_zoom.set_xlim(Rc - half, Rc + half)
    ax_zoom.set_ylim(Zc - half, Zc + half)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel("R [m]"); ax_zoom.set_ylabel("Z [m]")
    ax_zoom.set_title(f"zoom: max-deviation region  "
                      f"(mean {dev_mean[ic] * 1e3:.1f} mm)", fontsize=10)
    ax_zoom.grid(ls=":")
    # mark the zoomed window on the full view
    from matplotlib.patches import Rectangle
    ax_lcfs.add_patch(Rectangle((Rc - half, Zc - half), 2 * half, 2 * half,
                                fill=False, edgecolor="0.4", lw=0.8))


def draw_zeff(ax, psi_N_kin, bl, perturbed_data_list=None):
    r"""Draw the effective charge :math:`Z_{\rm eff}` profile: the baseline
    (with its :math:`\pm1\sigma_{\rm exp}` band) and the perturbed draws.

    Read from the stored ``aux_zeff`` datasets (one :math:`Z_{\rm eff}` per
    draw, the Z_eff-primary perturbation scheme), on the kinetic grid.
    """
    ax.cla()
    ax.grid(ls=":")
    ax.set_ylabel(r"$Z_{\rm eff}$")
    ax.set_xlabel(r"$\hat{\psi}$")

    if perturbed_data_list:
        n_equils = len(perturbed_data_list)
        for i, data in enumerate(perturbed_data_list):
            z = data.get("aux_zeff", data.get("Zeff"))
            if z is None:
                continue
            x = data.get("psi_N_kinetic", psi_N_kin)
            ax.plot(x, z, c=_GOLD, alpha=0.55, lw=1.0,
                    label=f"perturbed ({n_equils})" if i == 0 else None,
                    zorder=3)

    zb = bl.get("aux_zeff", bl.get("Zeff"))
    if zb is not None:
        zb = np.asarray(zb, float)
        sb = bl.get("sigma_aux_zeff")
        if sb is not None and np.shape(sb) == np.shape(zb):
            sb = np.asarray(sb, float)
            ax.fill_between(psi_N_kin, zb - sb, zb + sb, color="0.7",
                            alpha=0.3, zorder=2, label=r"$\pm\,1\sigma_{\rm exp}$")
            ax.plot(psi_N_kin, zb + 2 * sb, c="k", ls=":", lw=1.5, alpha=0.5,
                    label=r"$\pm\,2\sigma_{\rm exp}$", zorder=2)
            ax.plot(psi_N_kin, zb - 2 * sb, c="k", ls=":", lw=1.5, alpha=0.5,
                    zorder=2)
        ax.plot(psi_N_kin, zb, c="k", lw=2, label="baseline", zorder=5)
    ax.legend(loc="best", fontsize=7)


def _plot_bouquet_dashboard(psi_N, psi_N_kin, bl, perturbed,
                            baseline_ff=None, perturbed_ff=None):
    r"""Single combined figure of the profile panels (minimal scrolling).

    Layout (4x2): kinetic 2x2 on top (:math:`n_e, n_i, T_e, T_i`); then
    pressure / :math:`j_\phi` total; then :math:`Z_{\rm eff}` / q. The boundary
    panels and the roomier :math:`j_\phi` decomposition (total / inductive /
    bootstrap) get their own figures. ``baseline_ff`` / ``perturbed_ff`` are the
    flux-function payloads from :func:`_load_flux_functions`; when absent the q
    panel is left blank. Returns the figure.
    """
    fig, axes = plt.subplots(4, 2, figsize=(9.5, 11.0))

    draw_kinetic_profiles(
        axes[0:2, :], psi_N_kin,
        bl["n_e"], bl["n_i"], bl["T_e"], bl["T_i"],
        bl["sigma_ne"], bl["sigma_ni"],
        bl["sigma_te"], bl["sigma_ti"],
        perturbed_data_list=perturbed,
    )

    draw_pressure_profiles(axes[2, 0], psi_N, bl["pressure"],
                           perturbed_data_list=perturbed,
                           pressure_thermal=bl.get("pressure_thermal"),
                           bl=bl)
    draw_jphi_total(axes[2, 1], psi_N, bl["j_phi"],
                    bl["sigma_jphi"], perturbed_data_list=perturbed)
    draw_zeff(axes[3, 0], psi_N_kin, bl, perturbed_data_list=perturbed)
    draw_flux_function(axes[3, 1], "q", r"$q$", baseline_ff, perturbed_ff,
                       q_marker=True)
    fig.tight_layout()
    return fig


# ====================================================================
#  Data loading helper
# ====================================================================
def _load_all_perturbations(h5path, scan_key=None, indices=None):
    """Load all perturbed equilibria for a scan value as a list of dicts.

    Handles non-contiguous indices (from skipped equilibria) by
    discovering actual stored group names rather than assuming
    sequential 0..N-1.  When ``indices`` is given (an iterable of stored
    draw indices), only those draws are loaded -- used to honour a
    filter selection.
    """
    from .utils import _scan_key
    bkey = _scan_key(scan_key)
    with h5py.File(h5path, "r") as hf:
        if bkey is not None:
            parent = hf[f"scan/{bkey}"]
        else:
            parent = hf
        # Find all integer-keyed groups (skip _baseline, scan, etc.)
        stored_counts = sorted(
            int(k) for k in parent.keys()
            if k not in ("_baseline", "scan") and k.isdigit()
        )
    if indices is not None:
        keep = set(indices)
        stored_counts = [i for i in stored_counts if i in keep]
    return [
        load_equilibrium_by_path(h5path, count=i, scan_key=scan_key)
        for i in stored_counts
    ]


def _has_aux(h5path, scan_key=None):
    r"""True when the archive carries plottable "switchboard" auxiliary profiles
    -- the rotation / transport ``aux_<name>`` datasets (omega_tor, chi_e, chi_i,
    e_r) that :func:`plot_aux_profiles` renders.

    This drives whether :func:`plot_bouquet` appends the auxiliary-profile
    figure. It is deliberately keyed on the *presence of aux profiles*, NOT on
    the source path: the switchboard is source-decoupled (it lives in
    ``UncertaintyConfig.aux_baselines``), so a geqdsk run that supplies and
    perturbs ``omega_tor`` / ``chi`` gets the figure too -- which is what you'd
    want. In practice IMAS runs always have these and plain reconstruction runs
    do not, so it also happens to separate the two paths. ``aux_zeff`` is
    excluded -- both paths store it (the Z_eff-primary scheme that derives ni) --
    so a run with only ``aux_zeff`` reads as no switchboard. ``False`` on any
    read error.
    """
    try:
        from .utils import _scan_key
        bkey = _scan_key(scan_key)
        with h5py.File(h5path, "r") as hf:
            if bkey is not None and f"scan/{bkey}" in hf:
                grp = hf[f"scan/{bkey}"]
            elif "scan" in hf and len(hf["scan"].keys()):
                grp = hf["scan"][next(iter(hf["scan"].keys()))]
            else:
                grp = hf
            cand = []
            if "_baseline" in grp:
                cand.append(grp["_baseline"])
            cand += [grp[k] for k in grp.keys() if str(k).isdigit()][:1]
            for g in cand:
                if hasattr(g, "keys") and any(
                        str(k).startswith(("aux_", "extra_"))
                        and not str(k).endswith("zeff")
                        for k in g.keys()):
                    return True
        return False
    except Exception:
        return False


def _source_kind(h5path, scan_key=None):
    r"""Return the stored provenance marker (``'imas'`` / ``'geqdsk'``) written
    on the baseline group, or ``None`` for archives generated before it existed.

    This is the robust path discriminator -- independent of the source-decoupled
    aux switchboard, so it is not fooled by a geqdsk run that supplies
    ``omega_tor`` / ``chi``. Plotting falls back to :func:`_has_aux` when the
    marker is absent (older archives).
    """
    try:
        from .utils import _scan_key
        bkey = _scan_key(scan_key)
        bl_path = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"
        with h5py.File(h5path, "r") as hf:
            if bl_path in hf and "source_kind" in hf[bl_path].attrs:
                v = hf[bl_path].attrs["source_kind"]
                return v.decode() if isinstance(v, bytes) else str(v)
        return None
    except Exception:
        return None


def _lcfs_from_psigrid(eq):
    r"""Contour the LCFS from an eqdsk's 2-D :math:`\psi` grid (``psi_RZ``).

    Uses the same marching-squares + main-contour-selection + X-point
    closure path as bouquet's geqdsk flux-surface tracer. The stored
    ``boundary_R/Z`` (RBBBS) are TokaMaker's ``trace_surf`` points, which
    pile unevenly near the X-point (hundreds of extra vertices at the lower
    null) -- that uneven sampling skews vertex-wise comparisons and makes the
    overlay look offset where it is densest. Contouring the grid instead
    yields a uniformly-sampled curve produced the SAME way for the baseline
    and every draw, so the overlay is directly comparable. Returns ``(R, Z)``
    or ``None`` (caller falls back to the stored boundary).
    """
    from .io.geqdsk import (_trace_contours, _select_main_contour,
                            _crop_at_xpoint)
    try:
        R = np.asarray(eq.R_grid, float)
        Z = np.asarray(eq.Z_grid, float)
        PSI = np.asarray(eq.psi_RZ, float)
        R0 = float(eq._raw["RMAXIS"]); Z0 = float(eq._raw["ZMAXIS"])
        segs = _trace_contours(R, Z, PSI, [float(eq.psi_boundary)])[0]
        seg = _select_main_contour(segs, R0, Z0, 1, 1)
        if seg is None or len(seg) < 10:
            return None
        seg = _crop_at_xpoint(seg, R0, Z0)
        if seg is None or len(seg) < 4:
            return None
        seg = np.asarray(seg)
        return seg[:, 0], seg[:, 1]
    except Exception:
        return None


def _load_all_boundaries(h5path, scan_key=None, indices=None):
    """Load LCFS boundaries from stored geqdsk bytes for all equilibria.

    Returns a list of (R, Z) tuples with the **baseline first** (so callers
    overlaying against ``boundaries[0]`` reference the true baseline, not the
    first draw), followed by the draws. Each boundary is re-contoured from the
    eqdsk's 2-D psi grid via :func:`_lcfs_from_psigrid` (uniform, identical
    sampling for clean overlay), falling back to the stored ``boundary_R/Z``.
    Returns an empty list when the file contains no geqdsk bytes. When
    ``indices`` is given, only those stored draws are loaded (honours a filter
    selection).
    """
    from .io import GEQDSKEquilibrium
    from .utils import _scan_key, _group_path

    if indices is None:
        indices = list_equilibrium_indices(h5path, scan_key=scan_key)

    def _bnd(eq):
        lc = _lcfs_from_psigrid(eq)
        return lc if lc is not None else (eq.boundary_R, eq.boundary_Z)

    boundaries = []
    with h5py.File(h5path, "r") as hf:
        # Baseline first -- contoured the same way as the draws so the overlay
        # references the TRUE baseline rather than draw 0.
        _bk = _scan_key(scan_key)
        bl_path = f"scan/{_bk}/_baseline" if _bk is not None else "_baseline"
        if bl_path in hf and "eqdsk" in hf[bl_path]:
            try:
                beq = GEQDSKEquilibrium.from_bytes(
                    bytes(hf[bl_path]["eqdsk"][()]))
                boundaries.append(_bnd(beq))
            except Exception:
                pass

        for i in indices:
            grp_path = _group_path(scan_key, i)
            if grp_path not in hf:
                continue
            grp = hf[grp_path]

            eqdsk_ds = find_bytes_dataset(grp)
            if eqdsk_ds is None:
                continue

            raw = bytes(grp[eqdsk_ds][()])
            try:
                eq = GEQDSKEquilibrium.from_bytes(raw)
                boundaries.append(_bnd(eq))
            except Exception:
                continue

    return boundaries


def _load_flux_functions(h5path, scan_key=None, indices=None):
    r"""Per-draw (and baseline) safety factor q and :math:`FF'`.

    Both live in the stored eqdsk bytes (``GEQDSKEquilibrium.qpsi`` /
    ``.ffprim``) on the geqdsk's own uniform :math:`\hat\psi` grid (0..1,
    length ``nw``), which is independent of the kinetic ``psi_N``. Returns
    ``(baseline, draws)`` where each entry is a dict
    ``{"psi_N", "q", "ffprime"}`` (``baseline`` is ``None`` when the
    baseline group carries no eqdsk; draws without eqdsk are skipped).
    """
    from .io import GEQDSKEquilibrium
    from .utils import _scan_key, _group_path

    def _ff(raw):
        eq = GEQDSKEquilibrium.from_bytes(raw)
        q = np.asarray(eq.qpsi, dtype=float)
        return {"psi_N": np.linspace(0.0, 1.0, len(q)),
                "q": q, "ffprime": np.asarray(eq.ffprim, dtype=float)}

    def _eqdsk_in(grp):
        name = find_bytes_dataset(grp)
        if name is None:
            return None
        try:
            return _ff(bytes(grp[name][()]))
        except Exception:
            return None

    bkey = _scan_key(scan_key)
    bl_grp = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"
    baseline = None
    with h5py.File(h5path, "r") as hf:
        if bl_grp in hf:
            baseline = _eqdsk_in(hf[bl_grp])

    if indices is None:
        indices = list_equilibrium_indices(h5path, scan_key=scan_key)
    draws = []
    with h5py.File(h5path, "r") as hf:
        for i in indices:
            grp_path = _group_path(scan_key, i)
            if grp_path not in hf:
                continue
            ff = _eqdsk_in(hf[grp_path])
            if ff is not None:
                draws.append(ff)
    return baseline, draws


def draw_flux_function(ax, key, ylabel, baseline_ff, perturbed_ff,
                       q_marker=False):
    r"""Draw a flux-function profile (``key`` = ``"q"`` or ``"ffprime"``).

    Baseline in black, perturbed draws as thin gold curves -- mirroring
    :func:`draw_jphi_total`. Each draw carries its own ``psi_N`` (the
    geqdsk grid). ``q_marker`` adds the q=1 sawtooth reference line.
    """
    ax.cla()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$\hat{\psi}$")
    ax.grid(ls=":")
    if perturbed_ff:
        n = len(perturbed_ff)
        for i, d in enumerate(perturbed_ff):
            if d is None:
                continue
            ax.plot(d["psi_N"], d[key], c=_GOLD, lw=1.0, alpha=0.55,
                    label=f"perturbed ({n})" if i == 0 else None, zorder=3)
    if baseline_ff is not None:
        ax.plot(baseline_ff["psi_N"], baseline_ff[key], c="k", lw=2,
                label="baseline", zorder=1)
    if q_marker:
        ax.axhline(1.0, color="0.6", ls="--", lw=0.8, zorder=0)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)


# ====================================================================
#  Notebook-friendly API
# ====================================================================
def plot_bouquet(h5path_or_header, scan_key=None, mode="kinetic",
                 selection="all", layout="stack", pub_style=False):
    """Plot a family of perturbed equilibria from an HDF5 file.

    Parameters
    ----------
    h5path_or_header : str
        Path to the ``.h5`` file, or the header string (without
        extension).
    scan_key : str, float, or None
        Baseline scan-value label.  ``None`` for flat-layout files.
    mode : str
        ``'kinetic'``, ``'pressure'``, ``'j-phi'``, ``'boundary'``,
        or ``'all'``.
    selection : str
        Which draws to draw, honouring filter flags written by
        :func:`bouquet.filter_coil_currents` /
        :func:`bouquet.filter_boundaries`:

          - ``'all'``      : every stored draw (default; unchanged)
          - ``'selected'`` : only draws passing all applied filters
            (all draws if no filter has been run)
          - ``'excluded'`` : only draws cut by a filter

    Returns
    -------
    fig : Figure  or  list[Figure]   (when *mode* = ``'all'``)
    axes : Axes   or  list[Axes]     (when *mode* = ``'all'``)
    """
    # ---- resolve path ----------------------------------------------------
    if not h5path_or_header.endswith(".h5"):
        h5path = os.path.abspath(f"{h5path_or_header}.h5")
    else:
        h5path = os.path.abspath(h5path_or_header)

    # ---- auto-resolve scan value (so plot_bouquet(h5) "just works") ------
    if scan_key is None:
        _svs = discover_scan_keys(h5path)
        if _svs:
            scan_key = _svs[0]

    # ---- resolve which draws to show (filter selection) ------------------
    sel_indices = None
    if selection != "all":
        from .filtering import select_indices as _select_indices
        sel_indices = _select_indices(h5path, scan_key=scan_key,
                                      selection=selection)

    # ---- load data -------------------------------------------------------
    try:
        bl = load_baseline_profiles(h5path, scan_key=scan_key)
    except KeyError:
        avail = discover_scan_keys(h5path)
        msg = (
            f"No data for scan_key={scan_key!r} in {h5path}.\n"
            f"Available scan values: {avail}"
        )
        raise KeyError(msg) from None
    psi_N = bl["psi_N"]
    perturbed = _load_all_perturbations(h5path, scan_key=scan_key,
                                        indices=sel_indices)

    # Use psi_N_kinetic for kinetic profiles if available
    psi_N_kin = bl.get("psi_N_kinetic", psi_N)

    # Flux functions (q, FF') from the stored eqdsk -- only the j_phi-bearing
    # views need them; absent eqdsk leaves the panels blank rather than failing.
    baseline_ff = perturbed_ff = None
    if mode in ("all", "j-phi"):
        try:
            baseline_ff, perturbed_ff = _load_flux_functions(
                h5path, scan_key=scan_key, indices=sel_indices)
        except Exception:
            baseline_ff = perturbed_ff = None

    # Default for mode='all': a combined dashboard of the profile panels plus a
    # separate (roomier) boundary figure -- minimal scroll, but the LCFS panels
    # aren't cramped. pub_style=True instead gives the separate publication
    # figures.
    if mode == "all" and not pub_style:
        figs = [_plot_bouquet_dashboard(psi_N, psi_N_kin, bl, perturbed,
                                        baseline_ff, perturbed_ff)]
        boundaries = _load_all_boundaries(h5path, scan_key=scan_key,
                                          indices=sel_indices)
        if boundaries:
            fig_bd, ax_bd = plt.subplots(1, 2, figsize=(8.5, 4.0))
            _draw_boundary_panels(ax_bd[0], ax_bd[1], boundaries)
            fig_bd.tight_layout()
            figs.append(fig_bd)

        # j_phi current decomposition (total / inductive / bootstrap) for a
        # roomier zoom than the dashboard's single j_phi cell.
        try:
            _jr = plot_jphi(h5path, scan_key=scan_key, selection=selection)
            figs.append(_jr[0] if isinstance(_jr, tuple) else _jr)
        except Exception:
            pass

        # Append the auxiliary-profile figure only when the run carries a
        # rotation/transport switchboard (omega_tor / chi / e_r) -- zeff alone
        # does NOT trigger it, since zeff has its own dashboard panel. Covers
        # both IMAS and any geqdsk run that supplied those channels.
        if _has_aux(h5path, scan_key):
            try:
                ex_fig, _ = plot_aux_profiles(h5path, scan_key=scan_key,
                                              selection=selection)
                if ex_fig is not None:
                    figs.append(ex_fig)
            except Exception:
                pass

        return figs, [f.axes for f in figs]

    figs = []
    axes_list = []

    if mode in ("kinetic", "all"):
        fig, ax = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True)
        draw_kinetic_profiles(
            ax, psi_N_kin,
            bl["n_e"], bl["n_i"],
            bl["T_e"],   bl["T_i"],
            bl["sigma_ne"], bl["sigma_ni"],
            bl["sigma_te"],   bl["sigma_ti"],
            perturbed_data_list=perturbed,
        )
        fig.tight_layout()
        figs.append(fig)
        axes_list.append(ax)

    if mode in ("pressure", "all"):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        draw_pressure_profiles(
            ax, psi_N, bl["pressure"],
            perturbed_data_list=perturbed,
            pressure_thermal=bl.get("pressure_thermal"),
            bl=bl,
        )
        fig.tight_layout()
        figs.append(fig)
        axes_list.append(ax)

    if mode in ("j-phi", "all"):
        fig_jt, ax_jt = plt.subplots(figsize=(5.5, 4))
        draw_jphi_total(
            ax_jt, psi_N,
            bl["j_phi"], bl["sigma_jphi"],
            perturbed_data_list=perturbed,
        )
        fig_jt.tight_layout()
        figs.append(fig_jt)
        axes_list.append(ax_jt)

        fig_jc, ax_jc = plt.subplots(2, 1, figsize=(5.5, 5), sharex=True)
        draw_jphi_components(
            ax_jc, psi_N,
            perturbed_data_list=perturbed,
        )
        fig_jc.tight_layout()
        figs.append(fig_jc)
        axes_list.append(ax_jc)

        # Flux-function pair (FF', q) -- the equilibrium analogue of the j_phi
        # decomposition, on the geqdsk's own psi grid.
        fig_ff, ax_ff = plt.subplots(1, 2, figsize=(8.5, 3.6))
        draw_flux_function(ax_ff[0], "ffprime", r"$FF'$ [T$^2$ m$^2$/Wb]",
                           baseline_ff, perturbed_ff)
        draw_flux_function(ax_ff[1], "q", r"$q$", baseline_ff, perturbed_ff,
                           q_marker=True)
        fig_ff.tight_layout()
        figs.append(fig_ff)
        axes_list.append(ax_ff)

    if mode in ("boundary", "all"):
        boundaries = _load_all_boundaries(h5path, scan_key=scan_key,
                                          indices=sel_indices)
        if boundaries:
            fig_bd, ax_bd = plt.subplots(1, 2, figsize=(8.5, 4.0))
            _draw_boundary_panels(ax_bd[0], ax_bd[1], boundaries)
            fig_bd.tight_layout()
            figs.append(fig_bd)
            axes_list.append(ax_bd)

    # Auxiliary-profile figure: appended only when a rotation/transport
    # switchboard is present (zeff alone has its own dashboard panel).
    if mode == "all" and _has_aux(h5path, scan_key):
        try:
            ex_fig, _ = plot_aux_profiles(h5path, scan_key=scan_key,
                                          selection=selection)
            if ex_fig is not None:
                figs.append(ex_fig)
                axes_list.append(ex_fig.axes)
        except Exception:
            pass

    # Optional side-by-side layout: render the separate figures in a wrapping
    # flex row (less vertical scroll) while keeping each an individual image.
    if layout == "row" and len(figs) > 1:
        _display_figs_row(figs)

    if mode == "all":
        return figs, axes_list
    if len(figs) == 1:
        return figs[0], axes_list[0]
    return figs, axes_list


# ====================================================================
#  Legacy wrappers (deprecated)
# ====================================================================
def plot_kinetic_profiles(header, n_equils, psi_N, ne, ni, te, ti,
                          sigma_ne, sigma_ni, sigma_te, sigma_ti):
    """**Deprecated** -- use :func:`plot_bouquet` instead."""
    warnings.warn(
        "plot_kinetic_profiles() is deprecated.  Use plot_bouquet() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), sharex=True)
    perturbed = [load_equilibrium(header, count=i) for i in range(n_equils)]
    draw_kinetic_profiles(
        axes, psi_N, ne, ni, te, ti,
        sigma_ne, sigma_ni, sigma_te, sigma_ti,
        perturbed_data_list=perturbed,
    )
    plt.tight_layout()
    plt.show()


def plot_jphi_profiles(psi_N, input_j_phi, sigma_jphi, header, n_equils):
    """**Deprecated** -- use :func:`plot_bouquet` instead."""
    warnings.warn(
        "plot_jphi_profiles() is deprecated.  Use plot_bouquet() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 7), sharex=True)
    perturbed = [load_equilibrium(header, count=i) for i in range(n_equils)]
    draw_jphi_profiles(
        axes, psi_N, input_j_phi, sigma_jphi,
        perturbed_data_list=perturbed,
    )
    plt.tight_layout()
    plt.show()


# =====================================================================
# Standalone GEQDSK and p-file plotting for DIII-D / OMFIT workflows
# =====================================================================

def _resolve_x_coord(psi_N, x_coord, eq=None, psi_pf=None):
    """Return (x_values, x_label) for the chosen radial coordinate.

    For ``"rho"`` an equilibrium object with ``rhovn`` is required.
    When *psi_pf* is provided (p-file grid different from g-file),
    the rho mapping is interpolated onto it.
    """
    if x_coord == "psi_N":
        x = psi_pf if psi_pf is not None else psi_N
        return x, r"$\psi_N$"
    elif x_coord == "rho":
        if eq is None:
            raise ValueError(
                "x_coord='rho' requires an eq (GEQDSKEquilibrium) "
                "for the rhovn mapping"
            )
        from scipy.interpolate import interp1d
        psi_eq = np.linspace(0, 1, len(eq.rhovn))
        grid = psi_pf if psi_pf is not None else psi_N
        rho = interp1d(psi_eq, eq.rhovn, fill_value="extrapolate")(grid)
        return rho, r"$\rho$"
    else:
        raise ValueError(f"x_coord must be 'psi_N' or 'rho', got {x_coord!r}")


def _imas_input_profiles(source):
    r"""Raw input ``(psi_N, pressure[Pa], q)`` from the IDS ``equilibrium``
    profiles_1d at ``source.time`` -- the values the IMAS forward solve starts
    from (for the input-vs-solved comparison). ``q`` is ``None`` when the IDS
    does not store it (some FUSE/OMAS exports omit equilibrium q)."""
    import json
    d = json.load(open(source.ids_path))
    eq = d["equilibrium"]
    t = np.asarray(eq["time"], float)
    tt = getattr(source, "time", None)
    ie = 0 if tt is None else int(np.argmin(np.abs(t - float(tt))))
    p1 = eq["time_slice"][ie]["profiles_1d"]
    psi = np.asarray(p1["psi"], float)
    psiN = (psi - psi[0]) / (psi[-1] - psi[0]) if psi[-1] != psi[0] else psi
    q = np.asarray(p1["q"], float) if "q" in p1 else None
    jt = np.asarray(p1["j_tor"], float) if "j_tor" in p1 else None
    return psiN, np.asarray(p1["pressure"], float), q, jt


def plot_input_vs_recon(run, npsi=80, max_dev_mm=10.0):
    r"""Compare the reconstructed / forward-solved baseline against the RAW
    INPUT -- pressure, :math:`j_\phi`, q, and the separatrix (with the green→red
    colour-coded boundary deviation reused from the reconstruction diagnostic).

    Operates on a live :class:`~bouquet.Bouquet` ``run`` AFTER
    ``reconstruct()`` / ``prepare_baseline()`` -- it reads ``run.baseline`` and
    the solved ``run.mygs``. The raw input is the g-file on the reconstruction
    path and the IDS ``equilibrium`` on the IMAS path (auto-detected via
    ``baseline.provenance``). On the reconstruction path all four panels carry
    signal (the fit vs the g-file); on the IMAS path the pressure / current are
    prescribed, so those panels mostly confirm faithful input while q and the
    separatrix show the forward solve's geometric result. Returns the Figure.

    Separatrix panel caveat: deviations concentrated at the X-point corner are
    dominated by flux expansion (grad-psi -> 0 there, so any traced
    near-separatrix surface pulls away from the corner), not solve error --
    judge the solve by the rest of the boundary and the RMS.
    """
    import matplotlib.pyplot as plt
    from .TokaMaker_interface import safe_trace_surf
    bl = getattr(run, "baseline", None)
    mygs = getattr(run, "mygs", None)
    if bl is None or mygs is None:
        raise RuntimeError(
            "plot_input_vs_recon(run) needs a prepared run -- call "
            "setup_solver() then reconstruct()/prepare_baseline() first.")
    psi_pad = float(getattr(run.config.source, "psi_pad", 1e-3))
    is_imas = (str(getattr(bl, "provenance", "")) == "imas"
               or type(run.config.source).__name__ == "ImasSource")

    # ---- reconstructed / solved side (live TokaMaker solve) ----------------
    psiN_p, _f, _fp, p_sol, _pp = mygs.get_profiles(npsi=npsi, psi_pad=psi_pad)
    psiN_q, q_sol = mygs.get_q(npsi=npsi, psi_pad=psi_pad)[:2]
    psiN_q = np.asarray(psiN_q, float)
    if psiN_q.size and psiN_q.max() > 1.5:          # in case get_q returns psi, not psi_N
        psiN_q = (psiN_q - psiN_q.min()) / (psiN_q.max() - psiN_q.min())
    # Trace the panel LCFS as close to the separatrix as the tracer allows
    # (1 - 1e-4; same level as the boundary-diag fallback) rather than at
    # 1 - psi_pad (typically 0.999): near the X-point grad-psi -> 0, so the
    # 0.999 surface legitimately sits CENTIMETERS inside the separatrix and
    # the deviation panel paints a large red corner that is flux expansion,
    # not solve error. The finer level tightens that corner; deviations that
    # REMAIN there are genuine X-point mismatch. Falls back to 1 - psi_pad
    # if the near-separatrix trace fails.
    _lcfs_raw = safe_trace_surf(mygs, 1.0 - min(psi_pad, 1e-4))
    if _lcfs_raw is None:
        _lcfs_raw = safe_trace_surf(mygs, 1.0 - psi_pad)
    lcfs = np.asarray(_lcfs_raw, float)
    # "Reconstructed" j_phi = the ACHIEVED flux-surface-averaged toroidal
    # current of the CONVERGED solve (get_jphi_from_GS on the live equilibrium,
    # the same formula the recon corrective loop matches) -- NOT the prescribed
    # target profile. On the geqdsk path the per-solve corrective iteration
    # drives achieved ~= target, so this matches the archived baseline; on the
    # IMAS path the single-pass solve lands a few % off its anchor and this
    # panel now shows that honestly (consistent with the q panel, which is also
    # computed from the converged state). Falls back to the baseline arrays if
    # the live extraction fails.
    j_sol_x = np.asarray(bl.psi_N, float)
    try:
        from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS
        _psj, _f, _fp, _, _pp = mygs.get_profiles(npsi=len(j_sol_x), psi_pad=psi_pad)
        _, _, _ravgs, _, _, _ = mygs.get_q(npsi=len(j_sol_x), psi_pad=psi_pad)
        _psj = np.asarray(_psj, float)
        if _psj.size and (_psj.max() > 1.5 or _psj.min() < -0.5):
            _psj = (_psj - _psj.min()) / (_psj.max() - _psj.min())
        from .physics import q_ravg
        _j_ach = np.asarray(get_jphi_from_GS(
            np.asarray(_f, float) * np.asarray(_fp, float), np.asarray(_pp, float),
            np.asarray(q_ravg(_ravgs, "<R>"), float),
            np.asarray(q_ravg(_ravgs, "<1/R>"), float)), float)
        # match the baseline current's sign convention for a direct overlay
        if np.nanmedian(_j_ach) * np.nanmedian(np.asarray(bl.j_phi, float)) < 0:
            _j_ach = -_j_ach
        j_sol_x, j_sol = _psj, _j_ach
    except Exception:
        j_sol = np.asarray(bl.j_phi, float)

    # Anchors (IMAS diff workflow): jphi_diff / jBS_diff are FIXED offsets on
    # the kinetic grid (bl.j_BS + jBS_diff is the effective/FUSE bootstrap in
    # the solve; bl.j_BS itself is the raw SWB Sauter). Used below so the
    # component overlay reflects the bootstrap actually in the solve. Both are
    # None on the geqdsk path.
    _kin_x = np.asarray(getattr(bl, "psi_N_kinetic", bl.psi_N), float)

    # ---- raw input side ----------------------------------------------------
    if not is_imas:
        eq = read_geqdsk(run.config.source.geqdsk_path)
        in_x = np.asarray(eq.psi_N, float)
        in_p = np.asarray(eq.pres, float)
        in_q = np.asarray(eq.q_profile, float)
        in_jx = in_x; in_j = np.asarray(eq.j_tor_averaged_direct, float)
        in_bR = np.asarray(eq.boundary_R, float)
        in_bZ = np.asarray(eq.boundary_Z, float)
        in_lbl = "input g-file"
    else:
        from .io.imas import read_imas_geometry
        in_x, in_p, in_q, in_jt = _imas_input_profiles(run.config.source)
        if in_jt is not None:
            in_jx, in_j = in_x, in_jt         # true IDS equilibrium j_tor input
        else:
            in_jx, in_j = j_sol_x, j_sol      # IDS without equilibrium j_tor
        _F0, bRZ = read_imas_geometry(run.config.source)
        bRZ = np.asarray(bRZ, float)
        in_bR, in_bZ = bRZ[:, 0], bRZ[:, 1]
        in_lbl = "input IDS"

    # q is plotted as |q| so the input and the solve overlay regardless of the
    # COCOS sign convention (the g-file / IDS q can carry the opposite sign to
    # TokaMaker's), making the shape comparison direct.
    in_q_abs = np.abs(in_q) if in_q is not None else None
    q_sol_abs = np.abs(q_sol)

    C_IN, C_SOL = "tab:blue", "k"
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.5))
    _panels = [
        (ax[0, 0], in_x, in_p / 1e3, psiN_p, p_sol / 1e3, "pressure [kPa]"),
        (ax[0, 1], in_jx, in_j / 1e6, j_sol_x, j_sol / 1e6, r"$j_\phi$ [MA/m$^2$]"),
        (ax[1, 0], in_x, in_q_abs, psiN_q, q_sol_abs, r"$|q|$"),
    ]
    for a, ix, iy, sx, sy, ylab in _panels:
        a.cla(); a.grid(ls=":")
        if iy is not None:                       # input may be absent (e.g. IDS q)
            a.plot(ix, iy, ls=":", c=C_IN, lw=2.0, label=in_lbl, zorder=3)
        else:
            a.text(0.5, 0.92, "(input not in IDS)", fontsize=7, color=C_IN,
                   ha="center", va="top", transform=a.transAxes)
        a.plot(sx, sy, "-", c=C_SOL, lw=2.0, label="reconstructed", zorder=4)
        a.set_xlabel(r"$\hat{\psi}$"); a.set_ylabel(ylab)
        a.set_xlim(0, 1); a.legend(fontsize=8, loc="best")

    # j_phi panel: overlay the EFFECTIVE inductive + bootstrap components of the
    # solve -- reduced-opacity dashed (j_ind) and dash-dot (j_BS). "Effective" =
    # including the fixed anchors: on the IMAS diff path the bootstrap shown is
    # bl.j_BS + jBS_diff (the FUSE bootstrap actually in the solve, pedestal at
    # the right psi_N); the inductive is the residual against the ACHIEVED
    # total, so the two curves sum to the plotted (solved) total on every path.
    # All baseline arrays are interpolated from their native grids onto the
    # solver's uniform psi_N grid (same length does NOT imply same grid).
    if getattr(bl, "j_BS", None) is not None:
        _blx = np.asarray(bl.psi_N, float)

        def _to_jx(a, native_x):
            a = np.asarray(a, float)
            return np.interp(j_sol_x, np.asarray(native_x, float), a)

        _jBS_eff = _to_jx(bl.j_BS, _blx)
        if getattr(bl, "jBS_diff", None) is not None:
            _d = np.asarray(bl.jBS_diff, float)
            _jBS_eff = _jBS_eff + _to_jx(_d, _kin_x if _d.shape == _kin_x.shape else _blx)
        _j_ind_eff = j_sol - _jBS_eff
        for _fx in (getattr(bl, "j_NBI", None), getattr(bl, "j_RF", None)):
            if _fx is not None:
                _fx = np.asarray(_fx, float)
                _j_ind_eff = _j_ind_eff - _to_jx(
                    _fx, _kin_x if _fx.shape == _kin_x.shape else _blx)
        ax[0, 1].plot(j_sol_x, _j_ind_eff / 1e6, ls="--", c=C_SOL, lw=1.5,
                      alpha=0.45, label=r"recon $j_{ind}$", zorder=2)
        ax[0, 1].plot(j_sol_x, _jBS_eff / 1e6, ls="-.", c=C_SOL, lw=1.5,
                      alpha=0.45, label=r"recon $j_{BS}$", zorder=2)
        ax[0, 1].legend(fontsize=8, loc="best")

    # q panel: tame the edge divergence. q climbs steeply toward a diverted
    # separatrix, and if one profile (the input g-file q or the solve) shoots
    # far above the other at the edge it compresses the whole panel and buries
    # the core/mid comparison. Cap the y-axis 1.0 above the SMALLER of the two
    # edge q(psi_N=1) values -- the taller profile's divergent tail is clipped
    # while both core shapes stay legible.
    _q_edges, _q_finite = [], []
    for _qv in (in_q_abs, q_sol_abs):
        if _qv is None:
            continue
        _f = np.asarray(_qv, float)
        _f = _f[np.isfinite(_f)]
        if len(_f):
            _q_edges.append(_f[-1])       # q at the edge (last valid psi_N)
            _q_finite.append(_f)
    if _q_edges:
        _q_top = min(_q_edges) + 1.0
        _q_bot = max(0.0, float(np.min(np.concatenate(_q_finite))) - 0.2)
        if _q_top > _q_bot:
            ax[1, 0].set_ylim(_q_bot, _q_top)

    # separatrix panel with the colour-coded boundary deviation
    iso_pts = np.column_stack([in_bR, in_bZ])
    try:
        _, mx, rms = _isoflux_deviation_plot(
            ax[1, 1], fig, iso_pts, lcfs, in_bR, in_bZ, max_dev_mm=max_dev_mm)
        ax[1, 1].plot(in_bR, in_bZ, ls=":", c=C_IN, lw=1.5, label=in_lbl)
        ax[1, 1].set_title(f"separatrix  (max {mx:.1f} mm · RMS {rms:.1f} mm)",
                           fontsize=10)
        ax[1, 1].legend(fontsize=8, loc="lower right")
    except Exception as exc:                          # never let one panel kill the figure
        ax[1, 1].text(0.5, 0.5, f"separatrix unavailable\n({exc})",
                      ha="center", va="center", transform=ax[1, 1].transAxes)

    fig.suptitle(f"Input vs reconstruction  —  {getattr(bl, 'provenance', '?')}",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    return fig


def plot_geqdsk_bouquet(geqdsk_path_or_eq=None, x_coord="psi_N",
                        h5path=None, scan_key=None, count=None):
    """Plot one or more geqdsk equilibria: LCFS contours + profile panels.

    Layout: narrow flux-surface panel on the left, 2x2 grid of profiles
    on the right (pressure, q, |j_phi|, normalized P' and FF').

    Usage modes:

    1. **Single file:**
       ``plot_geqdsk_bouquet("shot.geqdsk")``

    2. **All perturbed from HDF5 (all scan values overplotted):**
       ``plot_geqdsk_bouquet(h5path="header.h5")``

    3. **All perturbed from HDF5 for one scan value:**
       ``plot_geqdsk_bouquet(h5path="header.h5", scan_key=0)``

    4. **Single perturbed case from HDF5:**
       ``plot_geqdsk_bouquet(h5path="header.h5", scan_key=0, count=2)``

    5. **Multiple files overplotted:**
       ``plot_geqdsk_bouquet(["a.geqdsk", "b.geqdsk"])``

    Parameters
    ----------
    geqdsk_path_or_eq : str, GEQDSKEquilibrium, list, or None
        Path(s) to g-file(s), or already-loaded equilibrium object(s).
        When a list is provided, all equilibria are overplotted.
        ``None`` when loading from *h5path*.
    x_coord : ``"psi_N"`` or ``"rho"``
        Radial coordinate for the profile panels.
    h5path : str or None
        Path to a bouquet HDF5 database.  When provided, loads and
        overplots all stored geqdsk equilibria (or a single one if
        *count* is specified).  When *scan_key* is ``None``, loads
        all scan values.
    scan_key : str, float, or None
        Scan-value label for HDF5 mode.  ``None`` loads all.
    count : int or None
        If given with *h5path*, load only this equilibrium index.

    Returns
    -------
    fig, axes
    """
    from .io import GEQDSKEquilibrium

    # --- resolve inputs to a list of equilibrium objects ---
    if h5path is not None:
        if not h5path.endswith(".h5"):
            h5path = os.path.abspath(f"{h5path}.h5")

        # Build list of (scan_key, count) pairs to load
        load_pairs = []
        if count is not None:
            load_pairs.append((scan_key, count))
        elif scan_key is not None:
            load_pairs.extend(
                (scan_key, i)
                for i in list_equilibrium_indices(h5path, scan_key=scan_key))
        else:
            # No scan_key specified: load ALL scan values
            svs = discover_scan_keys(h5path)
            if svs is not None:
                for sv in svs:
                    load_pairs.extend(
                        (sv, i)
                        for i in list_equilibrium_indices(h5path, scan_key=sv))
            else:
                load_pairs.extend(
                    (None, i)
                    for i in list_equilibrium_indices(h5path, scan_key=None))

        eqs = []
        from .utils import _group_path, _scan_key

        # Load baseline geqdsk from _baseline group if available
        baseline_eq = None
        bl_scan = scan_key if scan_key is not None else (
            load_pairs[0][0] if load_pairs else None)
        if bl_scan is not None:
            bl_key = _scan_key(bl_scan)
            bl_grp = f"scan/{bl_key}/_baseline" if bl_key else "_baseline"
        else:
            bl_grp = "_baseline"
        with h5py.File(h5path, "r") as hf:
            if bl_grp in hf and "eqdsk" in hf[bl_grp]:
                raw = bytes(hf[bl_grp]["eqdsk"][()])
                baseline_eq = GEQDSKEquilibrium.from_bytes(raw)

        # Load perturbed equilibria
        for sv, idx in load_pairs:
            grp_path = _group_path(sv, idx)
            with h5py.File(h5path, "r") as hf:
                if grp_path not in hf:
                    continue
                grp = hf[grp_path]
                eqdsk_ds = find_bytes_dataset(grp)
                if eqdsk_ds is not None:
                    raw = bytes(grp[eqdsk_ds][()])
                    eqs.append(GEQDSKEquilibrium.from_bytes(raw))

        # Prepend baseline if found; otherwise first perturbed is "baseline"
        if baseline_eq is not None:
            eqs.insert(0, baseline_eq)

        if not eqs:
            print("No geqdsk data found in HDF5.")
            return None, None
    elif geqdsk_path_or_eq is not None:
        inputs = geqdsk_path_or_eq
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        eqs = []
        for inp in inputs:
            if isinstance(inp, str):
                eqs.append(read_geqdsk(inp))
            else:
                eqs.append(inp)
    else:
        raise ValueError("Provide geqdsk_path_or_eq or h5path")

    n_eq = len(eqs)
    # h5 mode has a real baseline; file-list mode treats all entries equally
    has_baseline = (h5path is not None and n_eq > 1)

    fig = plt.figure(figsize=(9, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.6, 1, 1],
                          wspace=0.35, hspace=0.35)

    ax_lcfs = fig.add_subplot(gs[:, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[0, 2])
    ax_j = fig.add_subplot(gs[1, 1])
    ax_ff = fig.add_subplot(gs[1, 2])

    # When loading from HDF5: baseline first (behind), then perturbed on top.
    # When given a list of files: all plotted with the same style.
    colors_list = plt.cm.tab10(np.linspace(0, 1, max(n_eq, 10)))
    for idx in [0] + list(range(1, n_eq)):
        eq = eqs[idx]
        is_baseline = has_baseline and (idx == 0)
        if has_baseline:
            if is_baseline:
                c = "k"
                lw = 1.5
                alpha = 1.0
                lbl = "Baseline"
            else:
                c = "C1"
                lw = 1.5
                alpha = 0.7
                lbl = "Perturbed" if idx == 1 else None
        else:
            # File-list mode: uniform styling
            c = colors_list[idx] if n_eq > 1 else "k"
            lw = 1.5
            alpha = 1.0
            lbl = None

        psi_N = np.linspace(0, 1, len(eq.pres))
        x, xlabel = _resolve_x_coord(psi_N, x_coord, eq=eq)

        # LCFS
        if is_baseline:
            ax_lcfs.contour(eq.R_grid, eq.Z_grid, eq.psi_RZ,
                            levels=30, colors="0.6", linewidths=0.4)
            if eq.limiter_R is not None and len(eq.limiter_R) > 0:
                ax_lcfs.plot(eq.limiter_R, eq.limiter_Z, "k-", lw=1.0,
                             label="Limiter")
        ax_lcfs.plot(eq.boundary_R, eq.boundary_Z, "-", color=c,
                     lw=lw, alpha=alpha, label=lbl if lbl and "LCFS" not in str(lbl) else lbl)

        # Pressure
        ax_p.plot(x, eq.pres / 1e3, "-", color=c, lw=lw, alpha=alpha, label=lbl)

        # q
        ax_q.plot(x, eq.qpsi, "-", color=c, lw=lw, alpha=alpha, label=lbl)

        # |j_phi|
        jt = eq.j_tor_averaged
        ax_j.plot(x, np.abs(jt) / 1e6, "-", color=c, lw=lw, alpha=alpha, label=lbl)

        # Normalized P' and FF'
        pp = eq.pprime
        ff = eq.ffprim
        pp_max = np.max(np.abs(pp)) if np.max(np.abs(pp)) > 0 else 1.0
        ff_max = np.max(np.abs(ff)) if np.max(np.abs(ff)) > 0 else 1.0
        if is_baseline:
            ax_ff.plot(x, pp / pp_max, "-", color=_GOLD, lw=lw,
                       label=r"$p' / |p'|_{\max}$")
            ax_ff.plot(x, ff / ff_max, "--", color=_ORANGE, lw=lw,
                       label=r"$FF' / |FF'|_{\max}$")
        else:
            ax_ff.plot(x, pp / pp_max, "-", color=_GOLD, lw=lw, alpha=alpha)
            ax_ff.plot(x, ff / ff_max, "--", color=_ORANGE, lw=lw, alpha=alpha)

    # Labels and formatting (use first eq for sign labels)
    eq0 = eqs[0]
    Bt_sign = "+" if eq0.B_center >= 0 else "-"
    Ip_sign = "+" if eq0.Ip >= 0 else "-"

    ax_lcfs.set_aspect("equal")
    ax_lcfs.set_xlabel("R [m]")
    ax_lcfs.set_ylabel("Z [m]")
    ax_lcfs.set_title("Flux surfaces")
    if has_baseline:
        ax_lcfs.legend(fontsize=6)
    ax_lcfs.grid(ls=":")

    ax_p.set_ylabel("Pressure [kPa]")
    ax_p.set_title("Pressure")
    ax_p.grid(ls=":")

    ax_q.set_ylabel("q")
    ax_q.set_title("Safety factor")
    ax_q.grid(ls=":")
    # always show q=1 (the sawtooth surface) on the y-axis, with a tick at 1
    _qlo, _qhi = ax_q.get_ylim()
    ax_q.set_ylim(min(0.9, _qlo), _qhi)
    ax_q.set_yticks(sorted(set(list(ax_q.get_yticks()) + [1.0])))
    ax_q.set_ylim(min(0.9, _qlo), _qhi)
    ax_q.axhline(1.0, color="0.6", ls="--", lw=0.8, zorder=0)

    ax_j.set_xlabel(xlabel)
    ax_j.set_ylabel(r"$|\langle J_\phi \rangle|$ [MA/m$^2$]")
    ax_j.set_title(
        rf"$|J_\phi|$ (std)  [$B_t$:{Bt_sign}, $I_p$:{Ip_sign}]"
    )
    ax_j.grid(ls=":")

    ax_ff.set_xlabel(xlabel)
    ax_ff.set_ylabel("Normalized")
    ax_ff.set_title(r"$p'$ and $FF'$ (normalized)")
    if ax_ff.get_legend_handles_labels()[0]:   # single-eq view has no labels
        ax_ff.legend(fontsize=7)
    ax_ff.grid(ls=":")

    if has_baseline:
        ax_p.legend(fontsize=6)

    # the flux-surface panel's fixed aspect is formally incompatible with
    # tight_layout; the result is fine, so silence the cosmetic warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    return fig, fig.axes


def plot_pfile_bouquet(pfile_path_or_pf=None, x_coord="psi_N", eq=None,
                       h5path=None, scan_key=None, count=None):
    """Plot one or more p-file kinetic profiles in a multi-panel grid.

    Automatically includes all available profiles, skipping any that
    are absent.  Zeff is computed on the fly if ion species data is
    available.

    Usage modes:

    1. **Single file:**
       ``plot_pfile_bouquet("shot.peqdsk")``

    2. **All perturbed from HDF5 (requires pfile_bytes stored):**
       ``plot_pfile_bouquet(h5path="header.h5", scan_key=0)``

    3. **Single perturbed case from HDF5:**
       ``plot_pfile_bouquet(h5path="header.h5", scan_key=0, count=2)``

    4. **Multiple files overplotted:**
       ``plot_pfile_bouquet(["a.peqdsk", "b.peqdsk"])``

    .. note::
       HDF5 mode requires that ``pfile_bytes`` was passed to
       ``generate_bouquet()`` or ``store_equilibrium()`` when the
       data was generated.  If no p-file data is stored, use the
       file-path mode instead.

    Parameters
    ----------
    pfile_path_or_pf : str, PFile, list, or None
        Path(s) to p-file(s), or already-loaded PFile object(s).
        When a list is provided, all p-files are overplotted.
        ``None`` when loading from *h5path*.
    x_coord : ``"psi_N"`` or ``"rho"``
        Radial coordinate.  ``"rho"`` requires *eq*.
    eq : GEQDSKEquilibrium or None
        Required when ``x_coord="rho"`` to provide the rhovn mapping.
    h5path : str or None
        Path to a bouquet HDF5 database.  When provided, loads and
        overplots all stored p-file equilibria.
    scan_key : str, float, or None
        Scan-value label for HDF5 mode.
    count : int or None
        If given with *h5path*, load only this p-file index.

    Returns
    -------
    fig, axes
    """
    from .io.pfile import PFile as _PFile, read_pfile as _read_pf

    # --- resolve inputs to a list of PFile objects ---
    if h5path is not None:
        if not h5path.endswith(".h5"):
            h5path = os.path.abspath(f"{h5path}.h5")

        # Build list of (scan_key, count) pairs to load
        load_pairs = []
        if count is not None:
            load_pairs.append((scan_key, count))
        elif scan_key is not None:
            load_pairs.extend(
                (scan_key, i)
                for i in list_equilibrium_indices(h5path, scan_key=scan_key))
        else:
            svs = discover_scan_keys(h5path)
            if svs is not None:
                for sv in svs:
                    load_pairs.extend(
                        (sv, i)
                        for i in list_equilibrium_indices(h5path, scan_key=sv))
            else:
                load_pairs.extend(
                    (None, i)
                    for i in list_equilibrium_indices(h5path, scan_key=None))

        pfiles = []
        from .utils import _group_path, _scan_key

        # Load baseline pfile from _baseline group if available
        baseline_pf = None
        bl_scan = scan_key if scan_key is not None else (
            load_pairs[0][0] if load_pairs else None)
        if bl_scan is not None:
            bl_key = _scan_key(bl_scan)
            bl_grp = f"scan/{bl_key}/_baseline" if bl_key else "_baseline"
        else:
            bl_grp = "_baseline"
        with h5py.File(h5path, "r") as hf:
            if bl_grp in hf and "pfile" in hf[bl_grp]:
                raw = bytes(hf[bl_grp]["pfile"][()])
                baseline_pf = _PFile.from_bytes(raw)

        # Load perturbed pfiles (these now contain actual perturbed
        # kinetic profiles, not copies of the baseline)
        for sv, idx in load_pairs:
            grp_path = _group_path(sv, idx)
            with h5py.File(h5path, "r") as hf:
                if grp_path not in hf:
                    continue
                grp = hf[grp_path]
                pf_ds = (["pfile"] if "pfile" in grp else [])
                if pf_ds:
                    raw = bytes(grp[pf_ds[0]][()])
                    pfiles.append(_PFile.from_bytes(raw))

        # Prepend baseline if found
        if baseline_pf is not None:
            pfiles.insert(0, baseline_pf)

        if not pfiles:
            print("No p-file data found in HDF5. "
                  "Pass pfile_bytes to generate_bouquet() to store p-files, "
                  "and eqdsk_bytes/pfile_bytes to store_baseline_profiles().")
            return None, None
    elif pfile_path_or_pf is not None:
        inputs = pfile_path_or_pf
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        pfiles = []
        for inp in inputs:
            if isinstance(inp, str):
                pfiles.append(_read_pf(inp))
            else:
                pfiles.append(inp)
    else:
        raise ValueError("Provide pfile_path_or_pf or h5path")

    n_pf = len(pfiles)
    has_baseline = (h5path is not None and n_pf > 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_pf, 10)))

    # Define the panel catalogue: (raw_key, label, units)
    _PANEL_KEYS = [
        ("ne",    r"$n_e$",                  r"$10^{20}$/m$^3$"),
        ("te",    r"$T_e$",                  "keV"),
        ("ni",    r"$n_i$",                  r"$10^{20}$/m$^3$"),
        ("ti",    r"$T_i$",                  "keV"),
        ("ptot",  r"$p_{\rm tot}$",          "kPa"),
        ("pb",    r"$p_b$ (fast)",           "kPa"),
        ("nz1",   r"$n_{z1}$",              r"$10^{20}$/m$^3$"),
        ("nb",    r"$n_b$ (beam)",           r"$10^{20}$/m$^3$"),
        ("zeff",  r"$Z_{\rm eff}$",          ""),          # computed
        ("omeg",  r"$\omega_\phi$ (tor)",    "kRad/s"),
        ("omegp", r"$\omega_\theta$ (pol)",  "kRad/s"),
        ("omgeb", r"$\omega_{E \times B}$",  "kRad/s"),
        ("omgpp", r"$\omega_{\rm dia}$",     "kRad/s"),
        ("er",    r"$E_r$",                  "kV/m"),
        ("omghb", r"$\omega_{\rm HB}$",      "kRad/s"),
        ("kpol",  r"$K_{\rm pol}$",          ""),
    ]

    # Determine which panels have data in at least one p-file
    active_keys = []
    for key, label, units in _PANEL_KEYS:
        for pf in pfiles:
            if key == "zeff":
                if (pf.ion_species is not None
                        and pf.ne is not None and pf.ni is not None):
                    active_keys.append((key, label, units))
                    break
            elif pf._get_data(key) is not None:
                active_keys.append((key, label, units))
                break

    n = len(active_keys)
    if n == 0:
        print("No profiles to plot.")
        return None, None

    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 2.8 * nrows),
                             squeeze=False)

    for panel_idx, (key, label, units) in enumerate(active_keys):
        r, c = divmod(panel_idx, ncols)
        ax = axes[r][c]

        draw_order = [0] + list(range(1, n_pf))
        for pf_idx in draw_order:
            pf = pfiles[pf_idx]
            is_baseline = has_baseline and (pf_idx == 0)
            if has_baseline:
                if is_baseline:
                    col, lw, alpha = "k", 1.5, 1.0
                    lbl = ("Baseline" if panel_idx == 0 else None)
                else:
                    col, lw, alpha = "C1", 1.5, 0.7
                    lbl = ("Perturbed" if pf_idx == 1 and panel_idx == 0
                           else None)
            else:
                col = colors[pf_idx] if n_pf > 1 else "k"
                lw, alpha = 1.5, 1.0
                lbl = None

            if key == "zeff":
                if (pf.ion_species is not None
                        and pf.ne is not None and pf.ni is not None):
                    try:
                        psi_z, zeff = pf.compute_zeff()
                        x, xlabel = _resolve_x_coord(
                            None, x_coord, eq=eq, psi_pf=psi_z)
                        ax.plot(x, zeff, "-", color=col, lw=lw,
                                alpha=alpha, label=lbl)
                    except Exception:
                        pass
            else:
                d = pf._get_data(key)
                if d is not None:
                    psi_pf = pf.psinorm_for(key)
                    x, xlabel = _resolve_x_coord(
                        None, x_coord, eq=eq, psi_pf=psi_pf)
                    ax.plot(x, d, "-", color=col, lw=lw,
                            alpha=alpha, label=lbl)

        ax.set_title(label, fontsize=10)
        if units:
            ax.set_ylabel(units, fontsize=8)
        if r == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(ls=":")

    # Legend on the first panel when overplotting from HDF5
    if has_baseline and n > 0:
        axes[0][0].legend(fontsize=6)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    return fig, axes




def _framed_legend(ax, handles=None, **kw):
    """Legend with a readable semi-opaque frame.

    The per-draw diagnostics scatter markers across the whole axis, so an
    unframed legend ends up with data markers directly adjacent to (and
    indistinguishable from) the legend keys.
    """
    # fully opaque white: with translucent frames, bouquet markers behind the
    # legend remain visible and read as extra legend entries
    # frameon=True is REQUIRED: the bouquet publication style sets
    # legend.frameon=False in rcParams, which silently discards the frame
    # patch no matter what framealpha/facecolor say
    opts = dict(frameon=True, framealpha=1.0, facecolor='white',
                fancybox=True, borderpad=0.5)
    opts.update(kw)
    leg = (ax.legend(handles=handles, **opts) if handles is not None
           else ax.legend(**opts))
    leg.get_frame().set_edgecolor('0.8')
    leg.set_zorder(10)
    return leg


def plot_coil_currents(h5path_or_header, scan_key=None, vsc_coils=('F9A', 'F9B'),
                       exclude_coils=('ECOILA', 'ECOILB'), annotate=None):
    """Per-coil drift heatmap: coils x draws, % drift from the recon baseline.

    One diverging-colormap panel where each cell is
    ``100 * (I_draw - I_baseline) / |I_baseline|``, saturating at the in-spec
    limit -- a cell at full color is at or beyond spec, and cells exceeding
    their class spec are edged in black. The VSC pair (``vsc_coils``) is split
    off at the bottom with its own spec. ``exclude_coils`` (default: the
    DIII-D E-coils) are omitted -- they are floor-limited in the QP, not
    relative-drift-limited, and excluded from ``in_spec``; coils with a tiny
    baseline current (< 5% of the median |I_baseline|) are likewise dropped,
    since relative drift on a ~zero baseline is not engineering-meaningful.
    Draw tick labels are red for out-of-spec draws.

    Parameters
    ----------
    h5path_or_header : str
        Path to the ``.h5`` file or header string.
    scan_key : float or int, optional
        Scan value; defaults to the first one in the file.
    vsc_coils : tuple of str
        Names of the vertical-stability pair (separate spec class).
    annotate : bool, optional
        Write the % value in each cell. Default: only when the grid is
        small enough to stay legible (<= 12 draws and <= 24 coils).

    Returns
    -------
    (fig, ax)
    """
    import json as _json
    import matplotlib.colors as _mcolors
    from .utils import _scan_key

    h5path = (h5path_or_header if h5path_or_header.endswith(".h5")
              else os.path.abspath(f"{h5path_or_header}.h5"))

    with h5py.File(h5path, 'r') as hf:
        scan_keys = sorted(hf['scan'].keys()) if 'scan' in hf else []
        if scan_key is not None:
            sk = _scan_key(scan_key)
        elif scan_keys:
            sk = scan_keys[0]
            if len(scan_keys) > 1:
                print(f"plot_coil_currents: multiple scan values "
                      f"{scan_keys}; showing scan {sk}")
        else:
            print("No scan groups found in the HDF5 file.")
            return None, None
        parent = hf.get(f'scan/{sk}')
        if parent is None or '_baseline' not in parent:
            print("No baseline coil data found in the HDF5 file.")
            return None, None
        bl = parent['_baseline']
        if 'coil_currents' not in bl or 'coil_names' not in bl:
            print("No baseline coil data found in the HDF5 file.")
            return None, None
        ref_names = [s.decode() if isinstance(s, bytes) else s
                     for s in np.array(bl['coil_names'])]
        ref = dict(zip(ref_names, np.asarray(bl['coil_currents'], dtype=float)))
        draws = sorted(int(k) for k in parent.keys() if k.isdigit())
        cols, in_spec, spec_F, spec_VSC = [], [], 0.02, 0.02
        for c in draws:
            g = parent[str(c)]
            if 'coil_currents' not in g or 'coil_names' not in g:
                continue
            # schema v2 (F15): per-draw coil names are a string DATASET (matching
            # the baseline), not a JSON attr -- read them the same way as ref_names.
            d_names = [s.decode() if isinstance(s, bytes) else s
                       for s in np.array(g['coil_names'])]
            cols.append((c, dict(zip(d_names,
                                     np.asarray(g['coil_currents'], dtype=float)))))
            in_spec.append(bool(g.attrs.get('in_spec', True)))
            spec_F = float(g.attrs.get('inspec_F_max', spec_F))
            spec_VSC = float(g.attrs.get('inspec_VSC_max', spec_VSC))
    if not cols:
        print("No coil current data found in the HDF5 file.")
        return None, None

    # row selection: drop spec-exempt coils (E-coils by default -- they are
    # floor-limited in the QP, not relative-drift-limited) and any coil with
    # a tiny baseline (relative drift on ~zero is meaningless); order
    # non-VSC first, then the VSC pair under a separator
    base_mag = {n: abs(v) for n, v in ref.items()}
    floor = 0.05 * np.median([m for m in base_mag.values()]) if base_mag else 0.0
    rows = [n for n in ref_names
            if base_mag.get(n, 0.0) >= floor
            and n not in vsc_coils and n not in exclude_coils]
    rows += [n for n in vsc_coils if n in ref_names]
    dropped = [n for n in ref_names if n not in rows]
    n_vsc = sum(1 for n in vsc_coils if n in ref_names)

    M = np.full((len(rows), len(cols)), np.nan)
    for j, (_, d) in enumerate(cols):
        for i, n in enumerate(rows):
            r, v = ref.get(n, np.nan), d.get(n, np.nan)
            if np.isfinite(r) and np.isfinite(v) and abs(r) > 0:
                M[i, j] = 100.0 * (v - r) / abs(r)

    specF_pct, specV_pct = 100.0 * spec_F, 100.0 * spec_VSC
    vmax = max(specF_pct, specV_pct)
    fig, ax = plt.subplots(
        figsize=(min(2.8 + 0.5 * len(cols), 11.0),
                 min(1.5 + 0.28 * len(rows), 6.5)))
    im = ax.imshow(M, aspect='auto', cmap='RdBu_r',
                   norm=_mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    # thin white grid + spec-violation edges
    ax.set_xticks(np.arange(len(cols)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', lw=0.6)
    ax.tick_params(which='minor', length=0)
    from matplotlib.patches import Rectangle
    for i in range(len(rows)):
        spec = specV_pct if rows[i] in vsc_coils else specF_pct
        for j in range(len(cols)):
            if np.isfinite(M[i, j]) and abs(M[i, j]) > spec:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor='black', lw=1.4))
    if n_vsc and len(rows) > n_vsc:
        ax.axhline(len(rows) - n_vsc - 0.5, color='black', lw=1.0)
    if annotate is None:
        annotate = (len(cols) <= 12 and len(rows) <= 24)
    if annotate:
        for i in range(len(rows)):
            for j in range(len(cols)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.1f}", ha='center', va='center',
                            fontsize=6.5,
                            color=('white' if abs(M[i, j]) > 0.7 * vmax else 'black'))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c, _ in cols])
    for tick, ok in zip(ax.get_xticklabels(), in_spec):
        tick.set_color('black' if ok else '#D55E00')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel('draw  (red = out-of-spec)')
    note = f"; {'/'.join(dropped)} excluded from spec" if dropped else ""
    ax.set_title(f"Coil drift from baseline  "
                 f"(spec ±{specF_pct:.0f}% F, ±{specV_pct:.0f}% VSC{note})",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label('drift [%]')
    if specF_pct < vmax:   # mark the tighter (F) spec when the classes differ
        for s in (-specF_pct, specF_pct):
            cb.ax.axhline(s, color='black', lw=0.8, ls=':')
    fig.tight_layout()
    return fig, ax


def plot_spec_summary(h5path_or_header, scan_key=None, rms_max_mm=5.0):
    """The in-spec filter story in one figure.

    Left: per-draw **fraction of spec** for the two filter criteria --
    coil drift (the worse of max-F/spec_F and max-VSC/spec_VSC) and LCFS
    RMS deviation (RMS / ``rms_max_mm``). 1.0 = exactly at spec; a draw
    above the dashed line fails that criterion.
    Right: the same two fractions as one point per draw, with the pass
    region shaded -- where each draw sits relative to BOTH specs at once.
    Fill = selected, open red = excluded (the stored ``selected`` flag).

    Returns
    -------
    (fig, axes)
    """
    from .utils import _scan_key
    from .filtering import _baseline_boundary, _boundary_devs

    h5path = (h5path_or_header if h5path_or_header.endswith(".h5")
              else os.path.abspath(f"{h5path_or_header}.h5"))

    idxs, coil_f, bnd_f, selected = [], [], [], []
    with h5py.File(h5path, 'r') as hf:
        scan_keys = sorted(hf['scan'].keys()) if 'scan' in hf else []
        sk = (_scan_key(scan_key) if scan_key is not None
              else (scan_keys[0] if scan_keys else None))
        parent = hf.get(f'scan/{sk}') if sk is not None else None
        if parent is None:
            print("No scan group found in the HDF5 file.")
            return None, None
        # pass the raw key (str) through: _scan_key(str) is identity, so
        # this hits the same "scan/<key>" path the draws were stored under
        bl_boundary = _baseline_boundary(hf, scan_key if scan_key is not None
                                         else sk)
        for c in sorted(int(k) for k in parent.keys() if k.isdigit()):
            g = parent[str(c)]
            a = g.attrs
            fF = (float(a.get('max_F_drift_pct', np.nan))
                  / (100.0 * float(a.get('inspec_F_max', 0.02))))
            fV = (float(a.get('max_VSC_drift_pct', np.nan))
                  / (100.0 * float(a.get('inspec_VSC_max', 0.02))))
            rms, _mx = _boundary_devs(bl_boundary, g)
            idxs.append(c)
            coil_f.append(np.nanmax([fF, fV]))
            bnd_f.append(rms / rms_max_mm if np.isfinite(rms) else np.nan)
            selected.append(bool(a.get('selected', True)))
    if not idxs:
        print("No draws found in the HDF5 file.")
        return None, None
    coil_f, bnd_f = np.asarray(coil_f), np.asarray(bnd_f)
    sel = np.asarray(selected)
    x = np.arange(len(idxs))
    PASS, FAIL = '#009E73', '#D55E00'

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6),
                             gridspec_kw={'width_ratios': [1.5, 1.0]})
    from matplotlib.lines import Line2D

    ax = axes[0]
    for off, vals, mk in ((-0.16, coil_f, 'o'), (+0.16, bnd_f, 's')):
        col = np.where(vals > 1.0, FAIL, '#0072B2' if off < 0 else _GOLD)
        ax.vlines(x + off, 0, vals, color=col, lw=1.6)
        ax.scatter(x + off, vals, c=col, marker=mk, s=26, zorder=3)
    ax.axhline(1.0, color='black', ls='--', lw=1.0)
    h = [Line2D([], [], color='#0072B2', marker='o', ls='none',
                label='coil drift'),
         Line2D([], [], color=_GOLD, marker='s', ls='none',
                label='boundary RMS'),
         Line2D([], [], color=FAIL, marker='o', ls='none', label='over spec'),
         Line2D([], [], color='black', ls='--', label='spec')]
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in idxs])
    ax.set_xlabel('draw'); ax.set_ylabel('fraction of spec')
    ax.set_ylim(0, max(1.25, np.nanmax([coil_f.max(), np.nanmax(bnd_f)]) * 1.12))
    _framed_legend(ax, handles=h, fontsize=8, loc='upper left', ncol=2)
    ax.set_title(f'Per-draw spec fractions  '
                 f'({int(sel.sum())}/{len(sel)} selected)', fontsize=10)
    ax.grid(ls=':', axis='y')

    ax2 = axes[1]
    lim = max(1.25, np.nanmax([coil_f.max(), np.nanmax(bnd_f)]) * 1.12)
    ax2.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=PASS, alpha=0.12,
                                edgecolor=PASS, lw=1.0))
    ax2.scatter(coil_f[sel], bnd_f[sel], c=PASS, s=34, zorder=3,
                label='selected (filled)')
    ax2.scatter(coil_f[~sel], bnd_f[~sel], facecolor='none', edgecolor=FAIL,
                s=40, zorder=3, label='excluded (open)')
    # annotate every draw on small bouquets; only the failing ones on large
    for i, cf, bf, s in zip(np.asarray(idxs), coil_f, bnd_f, sel):
        if not (np.isfinite(cf) and np.isfinite(bf)):
            continue
        if len(idxs) <= 12 or not s:
            ax2.annotate(str(i), (cf, bf), textcoords='offset points',
                         xytext=(4, 3), fontsize=7)
    ax2.axvline(1.0, color='black', ls='--', lw=0.8)
    ax2.axhline(1.0, color='black', ls='--', lw=0.8)
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel('coil drift / spec'); ax2.set_ylabel('boundary RMS / spec')
    ax2.set_title('Spec box (pass region shaded)', fontsize=10)
    _framed_legend(ax2, fontsize=8, loc='upper right')
    fig.tight_layout()
    return fig, axes


# ====================================================================
#  Auxiliary-profile (rotation / transport / impurity) ensemble + IMAS view
# ====================================================================
_AUX_LABELS = {
    "omega_tor": r"$\omega_{tor}$ [rad/s]", "e_r": r"$E_r$ [V/m]",
    "chi_e": r"$\chi_e$ [m$^2$/s]", "chi_i": r"$\chi_i$ [m$^2$/s]",
    "zeff": r"$Z_{eff}$",
}
_AUX_ORDER = ["omega_tor", "e_r", "chi_e", "chi_i", "zeff"]


def plot_aux_profiles(h5path_or_header, scan_key=None, names=None,
                            selection="all"):
    """Overlay the perturbed transport / rotation / Z_eff profiles per draw.

    These are the switchboard profiles outside the core kinetic set -- toroidal
    rotation ``omega_tor``, radial field ``e_r``, transport diffusivities
    ``chi_e``/``chi_i`` and ``zeff`` -- read from the ``aux_<name>`` datasets
    (legacy ``extra_<name>`` archives are still read),
    one panel per profile in the same style as the kinetic dashboard: black
    input profile, shaded ±1σ and dotted ±2σ envelopes (when the baseline
    aux baselines/sigmas are stored in the archive), gold perturbed family.

    Returns ``(fig, axes)``, or ``(None, None)`` if none are stored (the
    switchboard wasn't used).
    """
    from .utils import _group_path

    h5path = (h5path_or_header if h5path_or_header.endswith(".h5")
              else os.path.abspath(f"{h5path_or_header}.h5"))

    svs = ([scan_key] if scan_key is not None
           else (discover_scan_keys(h5path) or [None]))
    pairs = [(sv, i) for sv in svs
             for i in list_equilibrium_indices(h5path, scan_key=sv)]
    if selection != "all":
        from .filtering import select_indices as _sel
        keep = {(sv, i) for sv in svs
                for i in _sel(h5path, scan_key=sv, selection=selection)}
        pairs = [p for p in pairs if p in keep]

    # dataset names: "aux_<name>" (current) with "extra_<name>" fallback for
    # archives written before the aux rename
    def _aux_ds(grp, nm, sigma=False):
        for key in ((f"sigma_aux_{nm}", f"sigma_extra_{nm}") if sigma
                    else (f"aux_{nm}", f"extra_{nm}")):
            if key in grp:
                return np.asarray(grp[key][()])
        return None

    avail, xgrid = [], None
    with h5py.File(h5path, "r") as hf:
        for sv, i in pairs:
            gp = _group_path(sv, i)
            if gp in hf:
                g = hf[gp]
                avail = sorted({k.split("_", 1)[1] for k in g.keys()
                                if k.startswith(("aux_", "extra_"))})
                if "psi_N_kinetic" in g:
                    xgrid = np.asarray(g["psi_N_kinetic"][()])
                break
    # Auto-selection excludes zeff -- it has its own panel in the main
    # plot_bouquet dashboard, so showing it here too is redundant (this figure is
    # the rotation/transport switchboard). Pass names=['zeff'] to force it.
    names = (names or [n for n in _AUX_ORDER if n in avail and n != "zeff"]
             or [n for n in avail if n != "zeff"])
    if not names:
        print("No aux_* profiles stored (switchboard not used for this run).")
        return None, None

    # baseline + sigma (stored by store_baseline_profiles since the
    # aux-bands feature; older files fall back to draws-only)
    from .utils import _scan_key
    bl_prof, bl_sig, bl_x = {}, {}, None
    with h5py.File(h5path, "r") as hf:
        bkey = _scan_key(pairs[0][0]) if pairs else None
        bl_path = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"
        if bl_path in hf:
            bl = hf[bl_path]
            if "psi_N_kinetic" in bl:
                bl_x = np.asarray(bl["psi_N_kinetic"][()])
            for nm in names:
                v = _aux_ds(bl, nm)
                if v is not None:
                    bl_prof[nm] = v
                s = _aux_ds(bl, nm, sigma=True)
                if s is not None:
                    bl_sig[nm] = s

    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 2.9 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    n_drawn = 0
    with h5py.File(h5path, "r") as hf:
        for j, nm in enumerate(names):
            ax = flat[j]
            n_drawn = 0
            for sv, i in pairs:
                gp = _group_path(sv, i)
                y = _aux_ds(hf[gp], nm) if gp in hf else None
                if y is not None:
                    x = (xgrid if xgrid is not None and len(xgrid) == len(y)
                         else np.linspace(0, 1, len(y)))
                    ax.plot(x, y, color=_GOLD, lw=0.9, alpha=0.8, zorder=2)
                    n_drawn += 1
            # input profile + sigma envelopes, matching the kinetic dashboard
            if nm in bl_prof:
                y0 = bl_prof[nm]
                x0 = (bl_x if bl_x is not None and len(bl_x) == len(y0)
                      else np.linspace(0, 1, len(y0)))
                ax.plot(x0, y0, c="k", lw=2, zorder=3)
                if nm in bl_sig and len(bl_sig[nm]) == len(y0):
                    sg = bl_sig[nm]
                    ax.fill_between(x0, y0 - sg, y0 + sg, alpha=0.25,
                                    color="0.45", zorder=1)
                    ax.plot(x0, y0 + 2 * sg, c="k", ls=":", lw=1.5,
                            alpha=0.5, zorder=1)
                    ax.plot(x0, y0 - 2 * sg, c="k", ls=":", lw=1.5,
                            alpha=0.5, zorder=1)
            ax.set_title(_AUX_LABELS.get(nm, nm), fontsize=10)
            if j // ncols == nrows - 1 or j + ncols >= len(names):
                ax.set_xlabel(r"$\psi_N$")
            ax.grid(ls=":")
        for j in range(len(names), nrows * ncols):
            flat[j].axis("off")
    # one legend for the whole figure (same encoding as the dashboard)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], c="k", lw=2, label="input"),
               Patch(facecolor="0.45", alpha=0.25,
                     label=r"$\pm\,1\sigma_{\rm exp}$"),
               Line2D([], [], c="k", ls=":", lw=1.5, alpha=0.5,
                      label=r"$\pm\,2\sigma_{\rm exp}$"),
               Line2D([], [], color=_GOLD, lw=1.2,
                      label=f"perturbed ({n_drawn})")]
    if not bl_prof:   # legacy file: only the draws are available
        handles = [handles[-1]]
    _framed_legend(flat[0], handles=handles, fontsize=7, loc="best")
    fig.tight_layout()
    return fig, axes


# Alias kept for early class-API notebooks (the figure was first exposed
# as "transport profiles" before the aux rename).
plot_transport_profiles = plot_aux_profiles


def plot_imas_bouquet(h5path_or_header, scan_key=None, selection="all",
                      layout="stack", pub_style=False):
    """**Deprecated alias** for :func:`plot_bouquet` (``mode='all'``).

    :func:`plot_bouquet` now auto-detects the archive type and appends the
    perturbed auxiliary-profile figure for IMAS runs (geqdsk runs skip it), so a
    single ``plot_bouquet(header, mode='all')`` covers both paths. Retained for
    backward compatibility; returns the list of figures.
    """
    warnings.warn(
        "plot_imas_bouquet() is deprecated -- plot_bouquet(mode='all') now "
        "auto-detects IMAS vs geqdsk archives and appends aux profiles itself.",
        DeprecationWarning, stacklevel=2,
    )
    res = plot_bouquet(h5path_or_header, scan_key=scan_key, mode="all",
                       selection=selection, layout=layout, pub_style=pub_style)
    figs = res[0] if isinstance(res, tuple) else res
    return list(figs) if isinstance(figs, (list, tuple)) else [figs]


def plot_bouquet_timeseries(entries, scan_key=None, draws=True, envelopes=True,
                            selection="all", cmap="viridis", time_label="time [s]"):
    r"""Overlay bouquet profile families from MULTIPLE time slices, colored by time.

    One 2x3 dashboard (n_e, T_e, n_i, T_i, pressure, j_phi) in which every slice
    contributes its baseline and its perturbed draws (all one thin weight,
    since the per-slice cluster is too tight to separate), and -- for the
    kinetic / j_phi panels -- its baseline +/-1 sigma envelope (light band). All
    of a slice's curves share one colour, set by its time on a ``viridis``
    colour bar, so the bouquet's *evolution* and its per-slice spread read off a
    single figure.

    Parameters
    ----------
    entries : dict ``{time: header_or_h5path}`` or list of ``(time, header)``
        One bouquet archive per time slice (e.g. the per-slice outputs of a
        ``set_slice`` sweep). Headers and ``.h5`` paths are both accepted.
    scan_key : int/float, optional
        The scan key under which each archive's draws are stored. ``None``
        (default) uses each entry's *time* as its own scan key -- i.e. each
        slice was generated with ``generation.scan_key = time`` -- so the time
        is both the colour value and the storage key. Pass a scalar to use one
        shared key for every archive.
    draws : bool
        Overlay the per-draw curves (default True).
    envelopes : bool
        Shade each slice's baseline +/-1 sigma band (default True).
    selection : {"all", "selected"}
        Which draws to overlay.
    cmap : str
        Colour map for the time axis.
    time_label : str
        Colour-bar label (e.g. ``"time [ms]"`` for a relabelled axis).

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.colors as _mcolors
    from .utils import _scan_key, _group_path

    # keep the ORIGINAL key form (int 1600 -> group "1600", float 2.1 -> "2.1");
    # float() only the numeric used for the colour norm, not the scan-key lookup
    items = list(entries.items()) if isinstance(entries, dict) else list(entries)
    items = sorted(items, key=lambda kv: float(kv[0]))
    if not items:
        print("plot_bouquet_timeseries: no entries given.")
        return None, None
    times = [float(k) for k, _ in items]
    norm = _mcolors.Normalize(vmin=min(times), vmax=max(times)
                              if max(times) > min(times) else min(times) + 1e-9)
    import matplotlib as _mpl
    cm_obj = _mpl.colormaps[cmap]

    # (title, dataset, grid-key, scale, sigma-dataset)
    PANELS = [
        (r"$n_e$ [$10^{19}$ m$^{-3}$]", "n_e", "psi_N_kinetic", 1e-19, "sigma_ne"),
        (r"$T_e$ [keV]",               "T_e",    "psi_N_kinetic", 1e-3,  "sigma_te"),
        (r"$n_i$ [$10^{19}$ m$^{-3}$]", "n_i", "psi_N_kinetic", 1e-19, "sigma_ni"),
        (r"$T_i$ [keV]",               "T_i",    "psi_N_kinetic", 1e-3,  "sigma_ti"),
        (r"$p$ [kPa]",                 "pressure", "psi_N",       1e-3,  None),
        (r"$j_\phi$ [MA/m$^2$]",       "j_phi", "psi_N",      1e-6,  "sigma_jphi"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    flat = axes.ravel()

    for (orig_key, path) in items:
        h5 = path if str(path).endswith(".h5") else os.path.abspath(f"{path}.h5")
        col = cm_obj(norm(float(orig_key)))
        # scan key for THIS archive: explicit scalar, else the entry's own key
        # (preserve its original form so int 1600 -> group "1600")
        key_this = scan_key if scan_key is not None else orig_key
        bkey = _scan_key(key_this)
        bl_path = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"
        try:
            sel_idx = set(select_indices(h5, scan_key=key_this, selection=selection)) \
                if selection != "all" else None
        except Exception:
            sel_idx = None
        with h5py.File(h5, "r") as hf:
            if bl_path not in hf:
                continue
            bl = hf[bl_path]
            grids = {g: np.asarray(bl[g][()]) for g in ("psi_N", "psi_N_kinetic") if g in bl}
            draw_keys = sorted(int(k) for k in hf[f"scan/{bkey}" if bkey is not None else "."].keys()
                               if k.isdigit()) if True else []
            for j, (title, ds, gk, sc, sds) in enumerate(PANELS):
                ax = flat[j]
                x = grids.get(gk)
                if x is None or ds not in bl:
                    continue
                y0 = np.asarray(bl[ds][()]) * sc
                if envelopes and sds and sds in bl:
                    s = np.asarray(bl[sds][()]) * sc
                    if s.shape == y0.shape:
                        # translucent GRAY band (slice-independent), behind everything
                        ax.fill_between(x, y0 - s, y0 + s, color="0.5", alpha=0.18,
                                        lw=0, zorder=1)
                if draws:
                    for c in draw_keys:
                        if sel_idx is not None and c not in sel_idx:
                            continue
                        g = hf[_group_path(key_this, c)]
                        if ds not in g:
                            continue
                        yd = np.asarray(g[ds][()]) * sc
                        xg = grids.get(gk)
                        if len(yd) == len(xg):
                            ax.plot(xg, yd, "-", color=col, lw=1.1, alpha=0.65, zorder=2)
                # baseline at the SAME thin weight as the draws (the per-slice
                # cluster is too tight to distinguish a bold baseline anyway);
                # slightly more opaque so it still reads as the slice's centre
                ax.plot(x, y0, "-", color=col, lw=1.1, alpha=0.9, zorder=3)
                ax.set_title(title, fontsize=10); ax.grid(ls=":")

    for j in (3, 4, 5):
        flat[j].set_xlabel(r"$\psi_N$")
    sm = _cm.ScalarMappable(norm=norm, cmap=cm_obj); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02, label=time_label)
    fig.suptitle("Bouquet time evolution  (lines = baseline + draws, "
                 "band = $\\pm1\\sigma$; colour = time)", fontsize=11)
    return fig, axes


# ====================================================================
#  Trace plots: l_i, boundary deviation across equilibria
# ====================================================================
def plot_traces(h5path_or_header, scan_key="all", li_band=None, rms_max_mm=None):
    r"""Compact per-draw diagnostics: l_i, LCFS deviation, anchor heatmap.

    One row of three panels (per scan value):

      1. **l_i** per draw -- on the estimator the baseline's ``l_i_scale``
         attr names (``l_i(3)`` post-issue-#20, ``l_i(1)`` for legacy
         archives) -- with the recon baseline (black star at draw 0),
         the l_i target (dashed) and, when ``li_band`` is given (fraction,
         e.g. ``0.05``), the acceptance band.
      2. **LCFS deviation** from the recon baseline: RMS (filled) and max
         (open) [mm], with the ``rms_max_mm`` threshold when given.
      3. **Anchor displacement** |Δ| [mm] of the four LCFS reference points
         (inboard/outboard midplane, top, lower X-point) as an
         anchors x draws heatmap. Signed ΔR/ΔZ detail lives in
         :func:`plot_boundary_point_traces`.

    Selected draws are filled; excluded draws are open markers.

    Returns
    -------
    figs : list of Figure
        One figure per scan value.
    """
    from .utils import read_eqdsk_from_bytes, _scan_key
    from .filtering import _baseline_boundary, _boundary_devs

    if not h5path_or_header.endswith(".h5"):
        h5path = os.path.abspath(f"{h5path_or_header}.h5")
    else:
        h5path = os.path.abspath(h5path_or_header)

    if scan_key == "all":
        scan_keys = discover_scan_keys(h5path) or [None]
    else:
        scan_keys = [scan_key]

    figs = []
    for sv in scan_keys:
        bl = load_baseline_profiles(h5path, scan_key=sv)
        li_target = float(bl.get("l_i_target", np.nan))
        # The per-draw l_i must be read on the SAME estimator the target lives
        # on, or the band is meaningless (issue #20).  Archives written after
        # the estimator-consistency fix carry `l_i_scale`; older ones are on
        # the legacy std/li(1) scale.
        _li_scale = str(bl.get("l_i_scale", "std(li1)"))
        if isinstance(_li_scale, bytes):
            _li_scale = _li_scale.decode()
        _li_attr = "l_i(3)" if _li_scale.startswith("iter") else "l_i(1)"
        _li_label = r'$l_i(3)$' if _li_attr == "l_i(3)" else r'$l_i(1)$'

        idxs, li1, sel = [], [], []
        rms_mm, max_mm = [], []
        anchors = []          # per draw: dict name -> (R, Z)
        bl_anchor_pts = None
        anchor_names = ['inboard', 'outboard', 'top', 'bottom']
        xpt_label = 'bottom'

        with h5py.File(h5path, "r") as hf:
            bkey = _scan_key(sv)
            parent = hf[f"scan/{bkey}"] if bkey is not None else hf
            bl_grp = parent.get("_baseline")
            bl_boundary = _baseline_boundary(hf, sv)

            # baseline magnetic axis (fixed anchor reference for ALL draws)
            R_axis = Z_axis = None
            if bl_grp is not None:
                eqk = find_bytes_dataset(bl_grp)
                if eqk is not None:
                    try:
                        eq_bl = read_eqdsk_from_bytes(
                            bytes(bl_grp[eqk][()]), read_geqdsk)
                        R_axis, Z_axis = float(eq_bl.R_mag), float(eq_bl.Z_mag)
                    except Exception:
                        pass
            if R_axis is None and bl_boundary is not None:
                R_axis = float(np.mean(bl_boundary[:, 0]))
                Z_axis = float(np.mean(bl_boundary[:, 1]))

            def _anchor_pts(boundary, grp_or_none):
                if boundary is None or R_axis is None:
                    return None
                xpts = None
                if grp_or_none is not None and "x_points" in grp_or_none:
                    try:
                        xpts = np.asarray(grp_or_none["x_points"][()])
                    except Exception:
                        xpts = None
                # close the polyline: trace_surf starts/ends at the outboard
                # midplane, so the un-closed seam sits exactly on the
                # outboard intersection and the bracket test misses it
                closed = np.vstack([boundary, boundary[:1]])
                pts, has_xpt = _extract_boundary_points(
                    closed[:, 0], closed[:, 1], R_axis, Z_axis,
                    ref_R_axis=R_axis, ref_Z_axis=Z_axis, x_points=xpts)
                return pts, has_xpt

            res = _anchor_pts(bl_boundary, bl_grp)
            if res is not None:
                bl_anchor_pts, bl_has_xpt = res
                if bl_has_xpt.get('lower'):
                    xpt_label = 'X-pt'

            for c in sorted(int(k) for k in parent.keys()
                            if k not in ("_baseline", "scan") and k.isdigit()):
                grp = parent[str(c)]
                idxs.append(c)
                li1.append(float(grp.attrs.get(_li_attr, np.nan)))
                sel.append(bool(grp.attrs.get("selected", True)))
                r, m = _boundary_devs(bl_boundary, grp)
                rms_mm.append(r); max_mm.append(m)
                pb = None
                if "perturbed_lcfs_ref" in grp:
                    try:
                        pb = np.asarray(grp["perturbed_lcfs_ref"][()])
                    except Exception:
                        pb = None
                res = _anchor_pts(pb, grp) if pb is not None else None
                anchors.append(res[0] if res is not None else None)

        if not idxs:
            continue
        # x = the STORED draw indices (same labels as the coil heatmap, the
        # spec summary, and the HDF5 groups); the recon baseline sits one
        # slot to the left of the first draw
        x = np.asarray(idxs, dtype=float)
        x_bl = x[0] - 1.0
        sel = np.asarray(sel)
        li1 = np.asarray(li1)
        rms_mm = np.asarray(rms_mm); max_mm = np.asarray(max_mm)

        # anchor |displacement| matrix [mm]
        D = np.full((len(anchor_names), len(idxs)), np.nan)
        if bl_anchor_pts is not None:
            for j, pts in enumerate(anchors):
                if pts is None:
                    continue
                for i, name in enumerate(anchor_names):
                    p0, p1 = bl_anchor_pts.get(name), pts.get(name)
                    if p0 is not None and p1 is not None and \
                            np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                        D[i, j] = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) * 1e3

        fig, axes = plt.subplots(
            1, 3, figsize=(10.0, 3.3),
            gridspec_kw={'width_ratios': [1.0, 1.0, 1.25]})
        sv_tag = f"  (scan {sv})" if sv is not None and len(scan_keys) > 1 else ""

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        # -- panel 1: l_i --
        ax = axes[0]
        h1 = []
        if np.isfinite(li_target):
            ax.axhline(li_target, color='gray', ls='--', lw=1.0, zorder=1)
            h1.append(Line2D([], [], color='gray', ls='--',
                             label=r'$l_i$ target'))
            if li_band is not None:
                ax.axhspan(li_target * (1 - li_band), li_target * (1 + li_band),
                           color='gray', alpha=0.15, zorder=0)
                h1.append(Patch(facecolor='gray', alpha=0.15,
                                label=f'±{100*li_band:.0f}% band'))
            ax.plot(x_bl, li_target, marker='*', ms=12, color='black', ls='none',
                    zorder=3)
            h1.append(Line2D([], [], color='black', marker='*', ms=11, ls='none',
                             label='baseline'))
            ax.axvline(x_bl + 0.5, color='lightgray', lw=0.8)
        ax.plot(x[sel], li1[sel], 'o', color=_GOLD, ms=5, zorder=3)
        ax.plot(x[~sel], li1[~sel], 'o', mfc='none', mec=_GOLD, ms=5, zorder=3)
        h1 += [Line2D([], [], color=_GOLD, marker='o', ls='none', ms=5,
                      label='selected (filled)'),
               Line2D([], [], mfc='none', mec=_GOLD, color=_GOLD, marker='o',
                      ls='none', ms=5, label='excluded (open)')]
        ax.set_xlabel('draw'); ax.set_ylabel(_li_label)
        ax.set_title(r'$l_i$ per draw' + sv_tag, fontsize=10)
        _framed_legend(ax, handles=h1, fontsize=7, loc='best')
        ax.grid(ls=':')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # -- panel 2: boundary deviation --
        ax = axes[1]
        ax.plot(x[sel], rms_mm[sel], 'o', color=_GOLD, ms=5, zorder=3)
        ax.plot(x[~sel], rms_mm[~sel], 'o', mfc='none', mec=_GOLD, ms=5, zorder=3)
        ax.plot(x, max_mm, 's', mfc='none', mec='black', ms=4.5, zorder=2)
        h2 = [Line2D([], [], color=_GOLD, marker='o', ls='none', ms=5,
                     label='RMS — selected'),
              Line2D([], [], mfc='none', mec=_GOLD, color=_GOLD, marker='o',
                     ls='none', ms=5, label='RMS — excluded'),
              Line2D([], [], mfc='none', mec='black', color='black', marker='s',
                     ls='none', ms=4.5, label='max')]
        if rms_max_mm is not None:
            ax.axhline(rms_max_mm, color='black', ls='--', lw=1.0)
            h2.append(Line2D([], [], color='black', ls='--',
                             label=f'RMS spec {rms_max_mm:g} mm'))
        ax.set_ylim(bottom=0)
        ax.set_xlabel('draw'); ax.set_ylabel('LCFS deviation [mm]')
        ax.set_title('Boundary vs baseline', fontsize=10)
        _framed_legend(ax, handles=h2, fontsize=7, loc='best')
        ax.grid(ls=':')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # -- panel 3: anchor heatmap --
        ax = axes[2]
        row_labels = [n if n != 'bottom' else xpt_label for n in anchor_names]
        im = ax.imshow(D, aspect='auto', cmap='YlOrBr', vmin=0.0)
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xticks(range(len(idxs)))
        ax.set_xticklabels([str(i) for i in idxs], fontsize=7)
        ax.set_xlabel('draw')
        ax.set_title('Anchor displacement', fontsize=10)
        if len(idxs) <= 12:
            vmax_d = np.nanmax(D) if np.any(np.isfinite(D)) else 1.0
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    if np.isfinite(D[i, j]):
                        ax.text(j, i, f"{D[i, j]:.1f}", ha='center', va='center',
                                fontsize=6.5,
                                color=('white' if D[i, j] > 0.7 * vmax_d else 'black'))
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(r'$|\Delta|$ [mm]', fontsize=8)
        fig.tight_layout()
        figs.append(fig)
    return figs
# ====================================================================
#  Boundary-points trace plot
# ====================================================================
def _lcfs_intersect_horizontal(R, Z, R_axis, Z_axis):
    """Return (R_inboard, R_outboard) where LCFS crosses Z=Z_axis.

    Iterates over consecutive contour segments and linearly interpolates
    crossings of the horizontal line ``Z=Z_axis``.  Returns the crossing
    with R < R_axis (inboard) and R > R_axis (outboard).  Returns
    ``(nan, nan)`` if either crossing is not found.
    """
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    crossings = []
    for i in range(len(R) - 1):
        z1, z2 = Z[i], Z[i + 1]
        if (z1 - Z_axis) * (z2 - Z_axis) <= 0 and z1 != z2:
            t = (Z_axis - z1) / (z2 - z1)
            r_cross = R[i] + t * (R[i + 1] - R[i])
            crossings.append(r_cross)
    if not crossings:
        return (np.nan, np.nan)
    crossings = np.asarray(crossings)
    inb = crossings[crossings < R_axis]
    out = crossings[crossings > R_axis]
    r_in = inb.max() if len(inb) else np.nan  # closest inboard to axis
    r_out = out.min() if len(out) else np.nan  # closest outboard to axis
    return (r_in, r_out)


def _lcfs_intersect_vertical(R, Z, R_axis, Z_axis):
    """Return (Z_below, Z_above) where LCFS crosses R=R_axis.

    Same idea as ``_lcfs_intersect_horizontal`` but crossing the
    vertical line ``R=R_axis``.  Returns (Z_below_axis, Z_above_axis).
    """
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    crossings = []
    for i in range(len(R) - 1):
        r1, r2 = R[i], R[i + 1]
        if (r1 - R_axis) * (r2 - R_axis) <= 0 and r1 != r2:
            t = (R_axis - r1) / (r2 - r1)
            z_cross = Z[i] + t * (Z[i + 1] - Z[i])
            crossings.append(z_cross)
    if not crossings:
        return (np.nan, np.nan)
    crossings = np.asarray(crossings)
    below = crossings[crossings < Z_axis]
    above = crossings[crossings > Z_axis]
    z_below = below.max() if len(below) else np.nan  # closest below
    z_above = above.min() if len(above) else np.nan  # closest above
    return (z_below, z_above)


def _detect_xpoint(R, Z, R_axis, Z_axis, half="lower",
                   corner_angle_deg=40.0):
    """Find an X-point candidate on the LCFS in the upper or lower half.

    An X-point on a separatrix shows up as a corner with a tangent
    turn far above what a smooth flux surface would have.  We compute
    the signed exterior turning angle at each contour vertex, look for
    the vertex with the largest |angle| in the requested half-plane,
    and return it if the angle exceeds ``corner_angle_deg``.

    Parameters
    ----------
    R, Z : array_like
        LCFS contour points (closed loop assumed; last == first not
        required, the function handles either).
    R_axis, Z_axis : float
        Magnetic axis position.
    half : 'upper' or 'lower'
        Which side of Z_axis to search.
    corner_angle_deg : float
        Threshold for accepting a vertex as an X-point.  D-shaped
        plasmas have smooth boundaries that don't exceed ~5-10 deg per
        vertex; X-points are ~80-120 deg.

    Returns
    -------
    (R_xpt, Z_xpt) or (nan, nan)
        Coordinates of the detected X-point.
    """
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    n = len(R)
    if n < 6:
        return (np.nan, np.nan)
    # Build closed loop indexing
    angles = np.zeros(n)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        v1 = np.array([R[i] - R[i_prev], Z[i] - Z[i_prev]])
        v2 = np.array([R[i_next] - R[i], Z[i_next] - Z[i]])
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles[i] = np.degrees(np.arccos(cos_t))
    mask_half = (Z > Z_axis) if half == "upper" else (Z < Z_axis)
    if not np.any(mask_half):
        return (np.nan, np.nan)
    cand = np.where(mask_half, angles, -np.inf)
    i_best = int(np.argmax(cand))
    if angles[i_best] < corner_angle_deg:
        return (np.nan, np.nan)
    return (float(R[i_best]), float(Z[i_best]))


def _xpoints_on_lcfs(x_points, R, Z, tol=0.03):
    """Return the TokaMaker X-points that lie on the LCFS polyline.

    ``mygs.get_xpoints()`` can return inactive nulls far from the plasma
    alongside the boundary-defining (active) saddle(s).  Keep only those
    within ``tol`` metres of the boundary contour ``(R, Z)`` -- these are
    the X-points that actually sit on the separatrix and define the
    top/bottom of the diverted boundary.

    Returns a list of ``(R, Z)`` tuples (possibly empty).
    """
    if x_points is None:
        return []
    xp = np.asarray(x_points, dtype=float).reshape(-1, 2)
    if xp.size == 0:
        return []
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    A = np.column_stack([R[:-1], Z[:-1]])
    B = np.column_stack([R[1:], Z[1:]])
    AB = B - A
    denom = np.einsum("ij,ij->i", AB, AB)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    out = []
    for r0, z0 in xp:
        if not (np.isfinite(r0) and np.isfinite(z0)):
            continue
        # point-to-SEGMENT distance: a saddle sitting on an edge between two
        # far-apart boundary vertices is "on" the LCFS even if no vertex is near.
        AP = np.array([r0, z0]) - A
        t = np.clip(np.einsum("ij,ij->i", AP, AB) / denom, 0.0, 1.0)
        proj = A + t[:, None] * AB
        d = float(np.min(np.hypot(proj[:, 0] - r0, proj[:, 1] - z0)))
        if d <= tol:
            out.append((float(r0), float(z0)))
    return out


def _extract_boundary_points(R, Z, R_axis, Z_axis,
                              ref_R_axis=None,
                              ref_Z_axis=None,
                              x_points=None,
                              xpoint_tol=0.05,
                              require_xpoint=None):
    """Extract the four characteristic LCFS points used by the trace.

    Returns ``(pts, has_xpt)`` where ``pts`` is a dict with keys
    ``'inboard'``, ``'outboard'``, ``'top'``, ``'bottom'`` and
    ``has_xpt`` is ``{'upper': bool, 'lower': bool}`` recording which
    halves are anchored on a true X-point.

    If ``ref_R_axis`` / ``ref_Z_axis`` are provided, the axis-intersection
    anchors are computed at those FIXED reference coordinates (typically
    the recon's magnetic axis).  This means the inboard/outboard
    anchors are always at the same Z, and top/bottom anchors are always
    at the same R, across all draws -- so ΔR/ΔZ deviations measure only
    LCFS shape change, not motion of the magnetic axis.  If reference
    coordinates are not provided, fall back to per-draw axis (legacy
    behaviour, includes axis motion in the deviations).

    Top/bottom default to the LCFS ∩ {R=R_anchor} axis-line intersection
    but are replaced by TokaMaker's own X-point (``x_points``, a true
    B_p=0 saddle captured at solve time) whenever a stored null lies on
    the boundary in that half-plane.  This is exact and resolution-
    independent, unlike a geometric corner search on the saved polyline.
    """
    R_anchor = ref_R_axis if ref_R_axis is not None else R_axis
    Z_anchor = ref_Z_axis if ref_Z_axis is not None else Z_axis
    r_in, r_out = _lcfs_intersect_horizontal(R, Z, R_anchor, Z_anchor)
    z_below, z_above = _lcfs_intersect_vertical(R, Z, R_anchor, Z_anchor)
    pts = {
        'inboard':  (r_in,  Z_anchor),
        'outboard': (r_out, Z_anchor),
        'top':      (R_anchor, z_above),
        'bottom':   (R_anchor, z_below),
    }
    has_xpt = {'upper': False, 'lower': False}
    # X-points live in physical space, so they are compared against THIS
    # draw's own axis (not the fixed reference axis) to decide which half
    # each null belongs to.  The detected X-point coords become the
    # anchor for that half, so the trace follows the X-point's motion in
    # absolute space.
    # Anchor top/bottom on the EXACT TokaMaker null that lies on the separatrix
    # (the true B_p=0 saddle). Using the stored null directly -- rather than a
    # geometric corner search -- avoids the ~vertex-spacing quantization that
    # shows up as spurious ~cm ΔR/ΔZ jumps between draws. The on-LCFS filter
    # rejects far inactive nulls, so a non-diverted (round) half keeps the smooth
    # midline crossing below.
    on_lcfs = _xpoints_on_lcfs(x_points, R, Z, tol=xpoint_tol)
    below = [(r, z) for r, z in on_lcfs if z < Z_axis]
    above = [(r, z) for r, z in on_lcfs if z > Z_axis]
    if below:
        r0, z0 = min(below, key=lambda p: p[1])   # lowest-Z = lower X-point
        pts['bottom'] = (r0, z0); has_xpt['lower'] = True
    if above:
        r0, z0 = max(above, key=lambda p: p[1])   # highest-Z = upper X-point
        pts['top'] = (r0, z0); has_xpt['upper'] = True

    # Geometric fallback only where there is no on-LCFS null (e.g. legacy h5
    # files): the 40° corner threshold leaves a genuinely round top/bottom on the
    # smooth midline crossing.
    for half, key in (('lower', 'bottom'), ('upper', 'top')):
        if not has_xpt[half]:
            rx, zx = _detect_xpoint(R, Z, R_axis, Z_axis, half=half)
            if np.isfinite(rx) and np.isfinite(zx):
                pts[key] = (rx, zx)
                has_xpt[half] = True
    return pts, has_xpt


def plot_boundary_point_traces(h5path_or_header, scan_key="all",
                                prefer_xpoint=True,
                                corner_angle_deg=None,
                                axes=None):
    r"""Trace plot of characteristic LCFS points across draws,
    expressed as **signed deviation (mm) from the baseline recon**.

    For each draw (and the baseline reconstruction at index 0) the
    function extracts four boundary points from the stored geqdsk:

      - **inboard midplane**: LCFS where Z = Z_axis, R < R_axis
      - **outboard midplane**: LCFS where Z = Z_axis, R > R_axis
      - **top**: upper X-point if the equilibrium has one on the LCFS,
        else LCFS where R = R_axis, Z > Z_axis
      - **bottom**: lower X-point if present, else LCFS where
        R = R_axis, Z < Z_axis

    The X-points are **TokaMaker's own** (``mygs.get_xpoints()``, true
    B_p=0 saddles) captured at solve time and stored per draw in the
    ``x_points`` dataset.  This replaces the earlier geometric
    turning-angle corner search, which was resolution-sensitive and
    produced erratic X-point hops between neighbouring boundary
    vertices.  For h5 files written before X-points were stored, the
    top/bottom traces fall back to the R=R_axis axis-line intersection.

    Produces one figure with two subplots:

      1. **ΔR [mm] vs draw index** for all four points (signed)
      2. **ΔZ [mm] vs draw index** for all four points (signed)

    The baseline (index 0) sits at zero by construction.  A horizontal
    dashed line at 0 mm marks the reference.  Lets the user see at a
    glance whether bnd-diag RMS is a systematic offset (all points
    drifting in the same direction) or random scatter around the recon
    (points fluctuating symmetrically around 0).

    Parameters
    ----------
    h5path_or_header : str
        Path to ``.h5`` file or header string (``"path/to/X.h5"`` or
        just ``"X"``).
    scan_key : str, float, int, or ``'all'``
        Scan value to plot.  ``'all'`` (default) plots every scan value
        as separate columns of markers.
    prefer_xpoint : bool, default True
        When True, use the stored TokaMaker X-points for the top/bottom
        traces.  When False, always use the R=R_axis axis-line
        intersections.
    corner_angle_deg : float, optional
        **Deprecated and ignored.**  The geometric corner search this
        parameter tuned has been replaced by TokaMaker's X-point finder.
        Retained only so existing calls don't break.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from .utils import (read_eqdsk_from_bytes, _scan_key, _group_path,
                        list_equilibrium_indices)
    if corner_angle_deg is not None:
        warnings.warn(
            "plot_boundary_point_traces: 'corner_angle_deg' is deprecated "
            "and ignored; X-points now come from TokaMaker's get_xpoints() "
            "stored per draw.", DeprecationWarning, stacklevel=2)

    if not h5path_or_header.endswith(".h5"):
        h5path = os.path.abspath(f"{h5path_or_header}.h5")
    else:
        h5path = os.path.abspath(h5path_or_header)

    if scan_key == "all":
        scan_keys = discover_scan_keys(h5path)
        if not scan_keys:
            scan_keys = [None]
    else:
        scan_keys = [scan_key]

    # Draw into caller-provided axes (e.g. plot_traces' combined figure) or make
    # our own compact figure.
    if axes is not None:
        ax_r, ax_z = axes
        fig = ax_r.figure
    else:
        fig, (ax_r, ax_z) = plt.subplots(2, 1, figsize=(7.5, 3.0), sharex=True)
    point_labels = ['inboard', 'outboard', 'top', 'bottom']
    point_colors = {'inboard':  '#1f77b4',
                    'outboard': '#d62728',
                    'top':      '#2ca02c',
                    'bottom':   '#9467bd'}
    point_markers = {'inboard': 'o', 'outboard': 's', 'top': '^', 'bottom': 'v'}

    n_scan = max(len(scan_keys), 1)
    scan_color_offset = (_cm.tab10(np.linspace(0, 0.9, n_scan))
                          if n_scan > 1 else None)

    with h5py.File(h5path, "r") as hf:
        for i_sv, sv in enumerate(scan_keys):
            indices = []
            R_pts = {k: [] for k in point_labels}
            Z_pts = {k: [] for k in point_labels}
            sv_key = _scan_key(sv)
            scan_tag = f"scan={sv}" if sv is not None else "single-scan"

            # Recon's axis -- used as the FIXED reference for all
            # axis-intersection anchors across every draw so axis
            # motion doesn't leak into the deviation signal.  Set when
            # we process the baseline below; if absent, fall back to
            # per-draw axis (legacy behaviour).
            ref_R_axis = None
            ref_Z_axis = None
            # Whether baseline has an upper/lower X-point on the LCFS --
            # used to decide whether the "top"/"bottom" curves are
            # labelled as X-points in the legend.  Determined once on
            # baseline so the legend label is stable across all draws.
            baseline_xpoint = {'upper': False, 'lower': False}

            def _read_xpoints(grp):
                """Stored TokaMaker X-points for a group, or None."""
                if not prefer_xpoint or "x_points" not in grp:
                    return None
                try:
                    return np.asarray(grp["x_points"][()], dtype=float)
                except Exception:
                    return None

            # ---- Baseline at index 0 ----
            bl_grp_path = (f"scan/{sv_key}/_baseline"
                           if sv_key else "_baseline")
            if bl_grp_path in hf:
                bl_grp = hf[bl_grp_path]
                eqdsk_keys = find_bytes_dataset(bl_grp)
                if eqdsk_keys is not None:
                    raw = bytes(bl_grp[eqdsk_keys][()])
                    try:
                        eq_bl = read_eqdsk_from_bytes(raw, read_geqdsk)
                        # Pin reference coordinates from the baseline.
                        ref_R_axis = float(eq_bl.R_mag)
                        ref_Z_axis = float(eq_bl.Z_mag)
                        pts, has_xpt = _extract_boundary_points(
                            eq_bl.boundary_R, eq_bl.boundary_Z,
                            eq_bl.R_mag, eq_bl.Z_mag,
                            ref_R_axis=ref_R_axis,
                            ref_Z_axis=ref_Z_axis,
                            x_points=_read_xpoints(bl_grp))
                        # Legend labels top/bottom as X-points only when
                        # the recon baseline actually has them on the LCFS.
                        baseline_xpoint = has_xpt
                        indices.append(0)
                        for k in point_labels:
                            R_pts[k].append(pts[k][0])
                            Z_pts[k].append(pts[k][1])
                    except Exception as exc:
                        warnings.warn(
                            f"[plot_boundary_point_traces] baseline "
                            f"({scan_tag}): {exc}")

            # ---- Perturbed draws (stored indices may have gaps where
            # band-rejected draws were dropped) ----
            for i in list_equilibrium_indices(h5path, scan_key=sv):
                grp_path = _group_path(sv, i)
                if grp_path not in hf:
                    continue
                grp = hf[grp_path]
                eqdsk_keys = find_bytes_dataset(grp)
                if eqdsk_keys is None:
                    continue
                raw = bytes(grp[eqdsk_keys][()])
                try:
                    eq = read_eqdsk_from_bytes(raw, read_geqdsk)
                    pts, _has_xpt = _extract_boundary_points(
                        eq.boundary_R, eq.boundary_Z,
                        eq.R_mag, eq.Z_mag,
                        ref_R_axis=ref_R_axis,
                        ref_Z_axis=ref_Z_axis,
                        x_points=_read_xpoints(grp),
                        require_xpoint=baseline_xpoint)
                    indices.append(i + 1)
                    for k in point_labels:
                        R_pts[k].append(pts[k][0])
                        Z_pts[k].append(pts[k][1])
                except Exception as exc:
                    warnings.warn(
                        f"[plot_boundary_point_traces] draw {i} "
                        f"({scan_tag}): {exc}")
                    continue

            if not indices:
                continue

            # Convert to signed deviation [mm] from the baseline
            # (index 0 = recon).  If index 0 was successfully
            # extracted, use it as the reference; otherwise warn and
            # skip this scan value.
            indices = np.asarray(indices)
            if indices[0] != 0:
                warnings.warn(
                    f"[plot_boundary_point_traces] no baseline at "
                    f"index 0 for {scan_tag}; cannot compute "
                    f"deviation -- skipping this scan value")
                continue
            # Map raw label -> legend label, surfacing X-point status
            # from the baseline detection.  When the baseline has an
            # X-point on a given half, the "top"/"bottom" curve is
            # actually tracking that X-point (per _extract_boundary_points
            # logic) -- relabel so the user knows.
            legend_label = {
                'inboard':  'inboard midplane',
                'outboard': 'outboard midplane',
                'top':      'X-pt (upper)' if baseline_xpoint['upper']
                            else 'top (axis line)',
                'bottom':   'X-pt (lower)' if baseline_xpoint['lower']
                            else 'bottom (axis line)',
            }
            for k in point_labels:
                R_arr = np.asarray(R_pts[k], dtype=float)
                Z_arr = np.asarray(Z_pts[k], dtype=float)
                R0 = R_arr[0]
                Z0 = Z_arr[0]
                dR_mm = (R_arr - R0) * 1e3
                dZ_mm = (Z_arr - Z0) * 1e3
                # Sanity guard: a few-% kinetic perturbation cannot move a
                # characteristic LCFS point by >10 cm. Anything larger is an
                # anchor-identification artifact (e.g. a missed X-point); drop
                # it rather than letting a spurious spike dominate the y-scale.
                _ANCHOR_MAX_MM = 100.0
                _bad = (np.abs(dR_mm) > _ANCHOR_MAX_MM) | (np.abs(dZ_mm) > _ANCHOR_MAX_MM)
                if np.any(_bad):
                    warnings.warn(
                        f"[plot_boundary_point_traces] {k}: {_bad.sum()} draw(s) "
                        f"with >|{_ANCHOR_MAX_MM:.0f}| mm anchor deviation dropped "
                        f"(likely an X-point identification artifact).")
                    dR_mm = np.where(_bad, np.nan, dR_mm)
                    dZ_mm = np.where(_bad, np.nan, dZ_mm)
                col = point_colors[k]
                mk = point_markers[k]
                lbl_R = legend_label[k]
                lbl_Z = legend_label[k]
                if scan_color_offset is not None:
                    lbl_R += f"  ({scan_tag})"
                    lbl_Z += f"  ({scan_tag})"
                ax_r.plot(indices, dR_mm, marker=mk, color=col,
                          linestyle='-', linewidth=0.7, markersize=5,
                          label=lbl_R if i_sv == 0 else None,
                          alpha=0.9)
                ax_z.plot(indices, dZ_mm, marker=mk, color=col,
                          linestyle='-', linewidth=0.7, markersize=5,
                          label=lbl_Z if i_sv == 0 else None,
                          alpha=0.9)

    # Zero reference: by construction every point's deviation at
    # index 0 is exactly 0, so the dashed line is a useful eye-guide.
    ax_r.axhline(0.0, color='black', linestyle='--',
                 alpha=0.4, linewidth=0.8)
    ax_z.axhline(0.0, color='black', linestyle='--',
                 alpha=0.4, linewidth=0.8)

    ax_r.set_ylabel(r"$\Delta R$ [mm]")
    ax_r.set_title("LCFS reference points: deviation from recon baseline "
                   "(index 0 = recon, signed)")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc='best', fontsize=8, ncol=2)
    ax_z.set_xlabel("Draw index")
    ax_z.set_ylabel(r"$\Delta Z$ [mm]")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc='best', fontsize=8, ncol=2)
    if axes is None:
        fig.tight_layout()

    return fig


def plot_jphi(h5path_or_header, scan_key=None, source=None, source_kind="auto",
              selection="all", save=None):
    r"""Three-panel current decomposition: total / inductive / bootstrap,
    overlaying the raw input (FUSE OMAS or geqdsk), the SWB-reconstructed
    baseline, and the perturbed draws.

    Uses the FAITHFUL components actually summed into TokaMaker --
    ``j_inductive = j_phi - j_BS,edge - j_fixed`` (which closes to the total
    exactly) -- NOT the post-hoc shelf-blend ``j_inductive`` stored in the
    archive (that over-states the core and does not sum back to the total).

    Parameters
    ----------
    h5path_or_header : str
        Bouquet archive (``.h5`` path or header stem).
    scan_key : int/str or None
        Scan group; defaults to the first scan in the file.
    source : str or None
        Raw input to overlay. IMAS ``dd_sim.json`` -> raw FUSE j_tor /
        j_bootstrap / j_ohmic (toroidal). g-file -> its direct j_phi used as
        the input reference in all three panels (no FUSE component split).
    source_kind : {'auto','imas','geqdsk'}
    selection : {'all','selected'}
    save : str or None
        If given, savefig to this path.
    """
    import json
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt

    h5 = h5path_or_header if str(h5path_or_header).endswith(".h5") else f"{h5path_or_header}.h5"
    with h5py.File(h5, "r") as hf:
        sk = str(scan_key) if scan_key is not None else list(hf["scan"].keys())[0]
        g = hf[f"scan/{sk}"]
        psi = np.asarray(g["_baseline/psi_N"][:], float)
        base_total = np.asarray(g["_baseline/j_phi"][:], float)
        base_jBS = (np.asarray(g["_baseline/j_BS"][:], float)
                    if "j_BS" in g["_baseline"] else None)
        ids = [k for k in g if k.isdigit()]
        # IMAS / full-bootstrap archives (isolate_edge_jBS=False) store no edge
        # spike -- fall back to the full j_BS, so j_ind = total - j_BS - fixed
        # closes exactly there too.
        has_spike = bool(ids) and "j_BS,edge" in g[ids[0]]
        _bs_key = "j_BS,edge" if has_spike else "j_BS"
        D = {d: dict(total=np.asarray(g[f"{d}/j_phi"][:], float),
                     spike=(np.asarray(g[f"{d}/{_bs_key}"][:], float)
                            if _bs_key in g[d] else np.zeros_like(psi)))
             for d in ids}

    sel = None
    if selection == "selected":
        try:
            from .filtering import select_indices
            sel = set(str(i) for i in select_indices(
                h5path_or_header, scan_key=int(sk), selection="selected"))
        except Exception:
            sel = None

    fixed = np.zeros_like(psi)
    F = None; Flabel = "input"
    if source is not None:
        kind = source_kind
        if kind == "auto":
            kind = "imas" if (str(source).endswith(".json") or "dd_sim" in str(source)) else "geqdsk"
        try:
            if kind == "imas":
                from .physics import parallel_to_toroidal
                cp = json.load(open(source))["core_profiles"]
                ic = int(np.argmin(np.abs(np.asarray(cp["time"], float) - float(int(sk)) / 1000.0)))
                c = cp["profiles_1d"][ic]
                p = np.asarray(c["grid"]["psi"], float); pN = (p - p[0]) / (p[-1] - p[0])
                jtot = np.asarray(c["j_total"], float); jtor = np.asarray(c["j_tor"], float)
                tt = lambda jp: parallel_to_toroidal(jp, j_parallel_total=jtot, j_tor_total=jtor)
                F = dict(total=np.interp(psi, pN, jtor),
                         jBS=np.interp(psi, pN, tt(np.asarray(c["j_bootstrap"], float))),
                         jind=np.interp(psi, pN, tt(np.asarray(c["j_ohmic"], float))))
                fixed = F["total"] - F["jBS"] - F["jind"]; Flabel = "FUSE"
            else:
                from .io.geqdsk import read_geqdsk
                eq = read_geqdsk(source)
                jg = getattr(eq, "j_tor_averaged", None)
                if jg is None:
                    jg = getattr(eq, "j_tor_averaged_direct", None)
                jg = np.asarray(jg, float).ravel()
                F = dict(total=np.interp(psi, np.asarray(eq.psi_N, float).ravel(), jg),
                         jBS=None, jind=None); Flabel = "geqdsk"
        except Exception as e:
            print(f"plot_jphi: source not read ({e!r}); baseline + draws only")
            F = None

    if base_jBS is None and D:
        base_jBS = np.mean([D[d]["spike"] for d in D], axis=0)
        base_jBS_label = r"SWB baseline $j_{BS}$ (ens.-mean)"
    else:
        base_jBS_label = r"SWB baseline $j_{BS}$"

    OR, FU = _GOLD, "tab:blue"   # gold draws (matches the other plots)
    keep = [d for d in ids if (sel is None or d in sel)]
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.8))
    if F is not None:
        ax[0].plot(psi, F["total"] / 1e6, "--", color=FU, lw=1.6, label=rf"{Flabel} $j_\phi$")
    ax[0].plot(psi, base_total / 1e6, "k-", lw=2.0, zorder=5, label=r"SWB baseline $j_\phi$")
    for i, d in enumerate(keep):
        ax[0].plot(psi, D[d]["total"] / 1e6, "-", color=OR, lw=1.0, alpha=0.55,
                   label="perturbed" if i == 0 else None)
    ax[0].set_title(r"total $j_\phi$")

    if F is not None:
        ax[1].plot(psi, (F["jind"] if F.get("jind") is not None else F["total"]) / 1e6,
                   "--", color=FU, lw=1.6, label=rf"{Flabel} $j_{{ind}}$" if F.get("jind") is not None else rf"{Flabel} $j_\phi$")
    # total j_phi reference, so the component is read against the whole current
    ax[1].plot(psi, base_total / 1e6, "k--", lw=1.4, alpha=0.6, zorder=4,
               label=r"baseline total $j_\phi$")
    for i, d in enumerate(keep):
        ax[1].plot(psi, (D[d]["total"] - D[d]["spike"] - fixed) / 1e6, "-", color=OR,
                   lw=1.0, alpha=0.55, label=r"perturbed $j_{ind}$" if i == 0 else None)
    ax[1].set_title(r"$j_{inductive}$  (= total $-\,%s-$ fixed)"
                    % ("j_{BS,edge}" if has_spike else "j_{BS}"))

    if F is not None:
        ax[2].plot(psi, (F["jBS"] if F.get("jBS") is not None else F["total"]) / 1e6,
                   "--", color=FU, lw=1.6, label=rf"{Flabel} $j_{{BS}}$" if F.get("jBS") is not None else rf"{Flabel} $j_\phi$")
    if base_jBS is not None:
        ax[2].plot(psi, base_jBS / 1e6, "k-", lw=2.0, zorder=5, label=base_jBS_label)
    # total j_phi reference (dashed, to distinguish from the solid baseline j_BS)
    ax[2].plot(psi, base_total / 1e6, "k--", lw=1.4, alpha=0.6, zorder=4,
               label=r"baseline total $j_\phi$")
    for i, d in enumerate(keep):
        ax[2].plot(psi, D[d]["spike"] / 1e6, "-", color=OR, lw=1.0, alpha=0.55,
                   label=r"perturbed $j_{BS}$" if i == 0 else None)
    ax[2].set_title(r"$j_{BS}$  (%s)"
                    % ("isolate-edge spike" if has_spike else "full bootstrap"))

    for a in ax:
        a.axhline(0, color="gray", lw=0.5); a.set_xlim(0, 1); a.grid(alpha=0.3)
        a.set_xlabel(r"$\psi_N$"); a.set_ylabel(r"$j$ [MA/m$^2$]"); a.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig, ax
