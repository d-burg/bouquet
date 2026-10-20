"""OpenFUSIONToolkit's bootstrap entry points under bouquet's names, on any toolkit build.

bouquet uses OFT PSFC_dev's names for the bootstrap options (``scale_jbs``, ``isolate_edge_jbs``,
``taper_edge_jphi``, ...) and results (``jphi_total``, ``jphi_bs``, ...).  Older toolkits spell them
``scale_jBS``, ``isolate_edge_jBS``, ``total_j_phi``, ``isolated_j_BS``, ...: options are translated at
the call and results on return, so the same code runs on upstream OFT and on PSFC_dev.
"""
from __future__ import annotations

import inspect

__all__ = ["OPTION_NAMES", "RESULT_NAMES", "toolkit_options", "toolkit_name", "results",
           "solve_with_bootstrap", "find_optimal_scale"]

#: bouquet's option name -> the spellings a toolkit may use instead.
OPTION_NAMES = {
    "scale_jbs": ("scale_jBS",),
    "isolate_edge_jbs": ("isolate_edge_jBS",),
    "parameterize_jbs": ("parameterize_jBS",),
    "djbs_tol": ("djBS_tol",),
    "taper_edge_jphi": ("taper_edge_jbs", "taper_edge_jBS"),
    "spike_prof": ("jphi_bs",),                       # find_optimal_scale
}

#: bouquet's result key -> the keys a toolkit may return it under instead (OFT PSFC_dev's own
#: ``jphi_saw``, its sawtooth current out, is bouquet's ``jphi_saw_out``).
RESULT_NAMES = {
    "jphi_total": ("total_j_phi",),
    "jphi_bs_raw": ("j_BS",),
    "jphi_bs": ("isolated_j_BS",),
    "jphi_ind": ("j_inductive",),
    "jphi_fixed": ("j_fixed",),
    "jphi_saw_out": ("jphi_saw", "j_saw"),
}

_OPTIONS = {}


def _params(fn):
    return {p.name for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)} - {"self"}


def toolkit_options(fn=None):
    """The keywords the installed toolkit's bootstrap takes: ``fn``'s own, plus, when ``fn`` passes
    ``**kwargs`` on (or ``fn`` is None), those of ``TokaMaker.solve_bootstrap`` / ``set_boot_ops``.
    ``None`` when OpenFUSIONToolkit is not importable.  Cached per entry point."""
    key = fn
    if key in _OPTIONS:
        return _OPTIONS[key]
    try:
        from OpenFUSIONToolkit.TokaMaker._core import TokaMaker
        from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap as _swb
    except Exception:
        return None
    fn = _swb if fn is None else fn
    names = _params(fn)
    if any(p.kind is p.VAR_KEYWORD for p in inspect.signature(fn).parameters.values()) \
            or fn is _swb:
        for f in (getattr(TokaMaker, "solve_bootstrap", None),
                  getattr(TokaMaker, "set_boot_ops", None)):
            if f is not None:
                names |= _params(f)
    _OPTIONS[key] = frozenset(names)
    return _OPTIONS[key]


def toolkit_name(name, options=None):
    """The installed toolkit's spelling of bouquet option ``name``: ``name`` itself when the toolkit
    takes it (or when its options are unknown), else the first of :data:`OPTION_NAMES` it takes."""
    options = toolkit_options() if options is None else options
    if options is None or name in options:
        return name
    return next((alt for alt in OPTION_NAMES.get(name, ()) if alt in options), name)


def results(res):
    """A ``solve_with_bootstrap`` result as a plain dict, with bouquet's keys (:data:`RESULT_NAMES`)
    beside the toolkit's, so nothing is lost when a renamed toolkit's result object is copied."""
    out = dict(res)
    for name, alts in RESULT_NAMES.items():
        if name not in out:
            alt = next((a for a in alts if a in out), None)
            if alt is not None:
                out[name] = out[alt]
    return out


def solve_with_bootstrap(*args, **kw):
    """The toolkit's ``solve_with_bootstrap``, called with bouquet's option names; returns
    :func:`results`."""
    from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap as _swb
    opts = toolkit_options(_swb)
    return results(_swb(*args, **{toolkit_name(k, opts): v for k, v in kw.items()}))


def find_optimal_scale(*args, **kw):
    """The toolkit's ``find_optimal_scale``, called with bouquet's option names."""
    from OpenFUSIONToolkit.TokaMaker.bootstrap import find_optimal_scale as _fos
    opts = toolkit_options(_fos)
    return _fos(*args, **{toolkit_name(k, opts): v for k, v in kw.items()})
