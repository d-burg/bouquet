"""The pre-v4 spellings of bouquet's current components and options, as deprecated aliases.

bouquet's current components and options follow OpenFUSIONToolkit's ``boot_profs`` names (``jphi_total``,
``jphi_bs``, ``scale_jbs``, ...).  Old keyword arguments, attributes and function names keep working, with a
:class:`DeprecationWarning` naming the new one.  Stored configs and archives are handled where they are read
(``config.RENAMED_OPTIONS`` / ``RENAMED_FIELDS``, ``schema.LEGACY_NAMES``).
"""
from __future__ import annotations

import functools
import inspect
import warnings

#: old keyword / attribute name -> new name
OLD_NAMES = {
    "j_phi": "jphi_total", "input_j_phi": "jphi_total_input", "j_inductive": "jphi_ind",
    "input_jinductive": "jphi_ind_input", "j_BS": "jphi_bs", "j_BS_isolated": "jphi_bs_isolated",
    "j_BS_edge": "jphi_bs_edge", "baseline_j_BS": "baseline_jphi_bs", "have_baseline_j_BS": "have_baseline_jphi_bs",
    "j_pressure": "jphi_pd", "j_NBI": "jphi_nbi", "j_RF": "jphi_rf", "j_other": "jphi_other",
    "j_sawteeth": "jphi_saw", "swb_jphi_saw": "jphi_saw_input", "j_saw": "jphi_saw_out",
    "swb_jphi_fixed": "jphi_fixed", "jBS_diff": "jphi_bs_diff", "j_phi_profile": "jphi_total_profile",
    "recalculate_j_BS": "recalculate_jbs", "rescale_j_BS": "rescale_jbs", "floor_j_BS": "floor_jbs",
    "scale_jBS": "scale_jbs", "jBS_scale_range": "jbs_scale_range", "jBS_scale_profile": "jbs_scale_profile",
    "jBS_baseline_mode": "jbs_baseline_mode",
}


def _warn(old, new, what, stacklevel=3):
    warnings.warn(f"{what} {old!r} is renamed {new!r}; the old name will be removed",
                  DeprecationWarning, stacklevel=stacklevel)


def accepts_old_names(fn):
    """``fn`` also accepts the old spelling (:data:`OLD_NAMES`) of each keyword it takes, with a warning."""
    params = set(inspect.signature(fn).parameters)
    old = {o: n for o, n in OLD_NAMES.items() if n in params and o not in params}

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        for o in [k for k in kw if k in old]:
            n = old[o]
            if n in kw:
                raise TypeError(f"{fn.__name__}() got both {o!r} and its new name {n!r}")
            _warn(o, n, f"{fn.__name__}() keyword")
            kw[n] = kw.pop(o)
        return fn(*args, **kw)
    return wrapper


def add_attribute_aliases(cls):
    """Give ``cls`` a read/write property under the old spelling of each of its renamed attributes."""
    fields = set(getattr(cls, "__dataclass_fields__", {}))
    for o, n in OLD_NAMES.items():
        if n in fields and not hasattr(cls, o):
            def get(self, o=o, n=n):
                _warn(o, n, f"{type(self).__name__} attribute")
                return getattr(self, n)

            def put(self, v, o=o, n=n):
                _warn(o, n, f"{type(self).__name__} attribute")
                setattr(self, n, v)
            setattr(cls, o, property(get, put))
    return cls


def deprecated_alias(fn, old):
    """``fn`` under its old name ``old``, with a warning."""
    @functools.wraps(fn)
    def alias(*args, **kw):
        _warn(old, fn.__name__, "function")
        return fn(*args, **kw)
    alias.__name__ = alias.__qualname__ = old
    return alias
