"""Measurement sets from an EFIT k-file (Fortran namelist input).

A k-file holds one time slice of the measurements EFIT fits, as Fortran
namelists (``&IN1``, ``&INWANT``, ``&INS``, ...).  It carries no sensor
geometry: channels are positional, in the order of the device's EFIT green
tables, so geometry and names come from a
:class:`~bouquet.kinrecon.measurements.MachineBundle` whose sensors are listed
in that order.

Variables read (namelist ``IN1`` unless noted; EFIT input-namelist names):

=================  ===========  ==========  =========  =========================
group              value        weight      bit        notes
=================  ===========  ==========  =========  =========================
``flux_loop``      ``COILS``    ``FWTSI``   ``PSIBIT`` [Wb/rad] x ``flux_scale``
``bpol_probe``     ``EXPMP2``   ``FWTMP2``  ``BITMPI`` [T]
``ip``             ``PLASMA``   ``FWTCUR``  ``BITIP``  [A]
``pf_current``     ``BRSP``,    ``FWTFC``,  ``BITFC``,  as written in the file;
                   ``ECURRT``   ``FWTEC``   ``BITEC``   ``BRSP`` truncated to
                                                       len(``FWTFC``)
``diamagnetic_flux`` ``DFLUX``  ``FWTDLC``  --         sigma ``SIGDLC``; [mV s]
``mse`` (``INS``)  ``TGAMMA``   ``FWTGAM``  --         sigma ``SGAMMA``
                                                       (or ``SIGGAM``)
=================  ===========  ==========  =========  =========================

**sigma convention** (sigma model ``"efit_serror_bit"``): for every group
that has a digitizer-bit array,

.. code-block:: text

    sigma = max(SERROR |value|, 10 bit)

with ``SERROR`` from ``IN1`` (default 0.03 when absent).  This is the standard
EFIT magnetics convention (probes with ``BITMPI``, loops with ``PSIBIT``): the
OMAS machine mappings (omas 0.95.2, ``omas/machine_mappings``, flux-loop and
poloidal-probe data functions) build the same max(10 digitizer LSB, 3 %
|value|) and describe it as "how it is treated in EFIT".  Applying the same rule to Ip
(``BITIP``) and the coil currents (``BITFC``, ``BITEC``) extends it by
analogy; where ``10 bit`` is below ``SERROR |value|`` the floor has no
effect.  The diamagnetic flux takes ``SIGDLC`` directly; MSE takes
``SGAMMA`` = sigma(tan gamma) directly.

**Weights.**  The fit weight is the k-file's ``FWT*`` (``weight`` multiplies
the normalized residual, as EFIT folds FWT into 1/sigma and as in
``equilibrium.time_slice[:].constraints.*.chi_squared = weight^2 (rec -
meas)^2 / sigma^2``, IMAS DD).  A channel is valid when its FWT > 0, sigma is
finite and > 0, and (MSE) |TGAMMA| > 1e-8, since EFIT writes 0 for chords
without data.

**Units.**  Flux loops are multiplied by ``flux_scale`` (default 2 pi: EFIT
COILS are poloidal flux per radian, COCOS 1; IMAS ``flux_loop[:].flux`` is
the total flux in Wb, COCOS 11, psi_11 = 2 pi psi_1 with the same sign --
O. Sauter and S. Yu. Medvedev, Comput. Phys. Commun. 184, 293 (2013), Table
I).  The SIGN of the stored flux depends on the producer's conventions
(e.g. the discharge's current direction and the device's flux-loop
calibration), so it is a stated choice: pass a negative ``flux_scale`` when
the file's sign is opposite to the one wanted.  A k-file may also store loop
fluxes RELATIVE to a reference loop (its own entry then reads 0, and EFIT
fits the reference flux separately, cf. ``FITSIREF`` / ``SIREF``); name that
loop with ``flux_reference`` and it is recorded in ``meta`` so the forward
model compares ``psi_i - psi_ref``.  Neither is inferred from the file.

``DFLUX`` and ``SIGDLC`` [mV s] are multiplied by 1e-3 to Wb.  Coil currents
are passed through as written, or divided by the bundle's ``pf_current``
``turns`` for the source arrays named in ``coil_current_per_turn`` (IMAS
``pf_active.coil[:].current`` is per turn; whether a k-file array holds
ampere-turns or per-turn amperes is device convention, so it is stated per
array, e.g. ``("BRSP",)``; ``True`` means every array).

**MSE.**  ``TGAMMA`` is tan(gamma).  The 7 EFIT coefficients
``AA1GAM..AA7GAM`` go into the 9 IMAS ones with
:func:`~bouquet.kinrecon.measurements.efit_to_imas_coefficients`.  Arrays are
truncated to the common length of the core arrays (EFIT zero-pads FWTGAM etc.
to the namelist dimension).  ``MSE_USECER`` = 1 means EFIT has already
applied the E_r correction to ``TGAMMA`` (``TGAMMAUNCOR`` holds the raw
value); ``er_corrected="auto"`` reads that flag and refuses when it is
missing.

**Time.**  ``ITIME`` [ms] -> ``time`` [s].  ``time_window`` defaults to
``IAVEM`` [ms] taken as the full width of EFIT's magnetics averaging window;
that reading of ``IAVEM`` is recorded in ``meta``.
"""

import re

import numpy as np

from ..interfaces import ConstraintGroup, MeasurementSet
from ..measurements import efit_to_imas_coefficients
from . import sigma_from_model

# ---------------------------------------------------------------------------
# Fortran namelist parser
# ---------------------------------------------------------------------------

_BLOCK = re.compile(r"[&$]\s*([A-Za-z_][A-Za-z0-9_]*)(.*?)(?:^\s*/\s*$|[&$]END\b|\Z)",
                    re.S | re.M | re.I)
_KEY = re.compile(r"(?<![\w.'\"])([A-Za-z_][A-Za-z0-9_]*)\s*(\(\s*(\d+)\s*\))?\s*=")
_STR = re.compile(r"'[^']*'|\"[^\"]*\"")


def _value(tok):
    t = tok.strip()
    if not t:
        return None
    if t[0] in "'\"":
        return t[1:-1]
    low = t.lower()
    if low in (".true.", "t", ".t."):
        return True
    if low in (".false.", "f", ".f."):
        return False
    return float(low.replace("d", "e"))


def _values(text):
    out = []
    for tok in re.findall(r"'[^']*'|\"[^\"]*\"|[^\s,]+", text):
        if "*" in tok and tok[0] not in "'\"":
            k, v = tok.split("*", 1)
            out.extend([_value(v)] * int(k))
        else:
            out.append(_value(tok))
    return out


def parse_namelists(text):
    """Parse Fortran namelist text into ``{BLOCK: {KEY: value}}``.

    Block and key names are upper-cased (Fortran names are case-blind).
    Values are lists (numeric entries as float, logicals as bool, strings as
    str); ``n*v`` repeats are expanded.  A key repeated within a block keeps
    its last assignment (Fortran semantics); ``KEY(i) = v ...`` assigns from
    1-based position ``i``.  Text outside ``&NAME ... /`` blocks is ignored.
    """
    out = {}
    for m in _BLOCK.finditer(text):
        name, body = m.group(1).upper(), m.group(2)
        masked = _STR.sub(lambda s: "\0" * len(s.group(0)), body)
        keys = list(_KEY.finditer(masked))
        block = out.setdefault(name, {})
        for i, k in enumerate(keys):
            end = keys[i + 1].start() if i + 1 < len(keys) else len(body)
            vals = _values(body[k.end():end])
            key = k.group(1).upper()
            if k.group(3) is not None:
                start = int(k.group(3)) - 1
                cur = list(block.get(key, []))
                cur.extend([None] * max(0, start + len(vals) - len(cur)))
                cur[start:start + len(vals)] = vals
                vals = cur
            block[key] = vals
    return out


def read_kfile(path):
    """Parse an EFIT k-file into ``{BLOCK: {KEY: values}}``."""
    with open(path) as f:
        return parse_namelists(f.read())


# ---------------------------------------------------------------------------
# k-file -> MeasurementSet
# ---------------------------------------------------------------------------

def _num(block, key, n=None, default=None):
    v = block.get(key)
    if v is None:
        if default is None:
            return None
        v = [default] * (n or 1)
    a = np.array([np.nan if x is None else float(x) for x in v], dtype=float)
    if n is not None:
        if a.size < n:
            a = np.concatenate([a, np.full(n - a.size, np.nan if default is None else default)])
        a = a[:n]
    return a


def _scalar(block, key, default=None):
    a = _num(block, key)
    return default if a is None or a.size == 0 else float(a[0])


ALL_GROUPS = ("flux_loop", "bpol_probe", "ip", "diamagnetic_flux", "pf_current", "mse")


def measurements_from_kfile(kfile, machine, groups=None, time_window=None,
                            er_corrected="auto", flux_scale=2.0 * np.pi,
                            coil_current_per_turn=(), flux_reference=None, meta=None):
    """Build a :class:`MeasurementSet` from an EFIT k-file.

    Parameters
    ----------
    kfile : str or dict
        Path, or the dict returned by :func:`read_kfile`.
    machine : MachineBundle
        Sensor names/geometry in the k-file's channel order:
        ``sensors["flux_loop"]`` for ``COILS``, ``sensors["bpol_probe"]`` for
        ``EXPMP2``, ``sensors["pf_current"]`` for the coil currents.  The
        coil labels are ``channel_maps["BRSP"]`` + ``channel_maps["ECURRT"]``
        when given, else the ``pf_current`` labels in that order.
    groups : sequence, optional
        Groups to build (default: those present in the file).
    time_window : float, optional
        Full window width [s]; default ``IAVEM`` [ms] / 1e3.
    er_corrected : {"auto", True, False}
        Whether ``TGAMMA`` is E_r-corrected; ``auto`` reads ``MSE_USECER``.
    flux_scale : float
        Multiplies ``COILS`` and ``PSIBIT`` (see module docstring).
    coil_current_per_turn : bool or sequence of str
        Source arrays (``"BRSP"``, ``"ECURRT"``) whose currents are divided by
        the bundle's ``pf_current`` ``turns``; True = all.
    flux_reference : str, optional
        Label of the reference loop the ``COILS`` values are relative to.
    """
    nl = read_kfile(kfile) if isinstance(kfile, str) else kfile
    in1 = nl.get("IN1", {})
    ins = nl.get("INS", {})
    serror = _scalar(in1, "SERROR", 0.03)
    itime = _scalar(in1, "ITIME")
    iavem = _scalar(in1, "IAVEM")
    if time_window is None:
        time_window = 0.0 if iavem is None else iavem * 1e-3
        tw_src = "IAVEM [ms] taken as the full magnetics averaging width"
    else:
        tw_src = "caller"
    ms_meta = {"source": "efit_kfile", "SERROR": serror, "ITIME_ms": itime,
               "IAVEM_ms": iavem, "time_window_source": tw_src}
    if isinstance(kfile, str):
        ms_meta["file"] = kfile.rsplit("/", 1)[-1]
    ms_meta.update(meta or {})
    ms = MeasurementSet(machine=machine.machine,
                        time=0.0 if itime is None else itime * 1e-3,
                        time_window=float(time_window), meta=ms_meta)
    requested = groups is not None
    groups = list(groups) if requested else list(ALL_GROUPS)

    def need(name, present):
        if not present and requested:
            raise ValueError(f"group '{name}' requested but absent from the k-file")
        return present

    def efit_group(name, value, fwt, bit, labels, geometry, scale=1.0, extra=None):
        n = len(labels)
        if np.count_nonzero(np.isfinite(value)) < n:
            raise ValueError(f"group '{name}': k-file has "
                             f"{np.count_nonzero(np.isfinite(value))} values for "
                             f"{n} sensors in the machine bundle")
        value, fwt, bit = value[:n] * scale, fwt[:n], bit[:n] * np.abs(scale)
        sigma = sigma_from_model("efit_serror_bit", value, serror=serror, bit=bit)
        valid = (fwt > 0) & np.isfinite(value) & np.isfinite(sigma) & (sigma > 0)
        gm = {"sigma_model": "efit_serror_bit",
              "sigma_params": {"serror": serror, "bit": bit},
              "source": "efit_kfile"}
        gm.update(extra or {})
        return ConstraintGroup(name=name, labels=labels, measured=value, sigma=sigma,
                               valid=valid, weight=np.where(np.isfinite(fwt), fwt, 0.0),
                               geometry=geometry, meta=gm)

    if "flux_loop" in groups and need("flux_loop", "COILS" in in1):
        labels = machine.labels("flux_loop")
        n = len(labels)
        ms.groups["flux_loop"] = efit_group(
            "flux_loop", _num(in1, "COILS", n), _num(in1, "FWTSI", n, 0.0),
            _num(in1, "PSIBIT", n, 0.0), labels, machine.geometry("flux_loop"),
            scale=flux_scale, extra={"flux_scale": flux_scale,
                                     "units": "COILS [Wb/rad] x flux_scale",
                                     "flux_reference": flux_reference})
        if flux_reference is not None and flux_reference not in labels:
            raise ValueError(f"flux_reference {flux_reference!r} is not a flux-loop label")
    if "bpol_probe" in groups and need("bpol_probe", "EXPMP2" in in1):
        labels = machine.labels("bpol_probe")
        n = len(labels)
        ms.groups["bpol_probe"] = efit_group(
            "bpol_probe", _num(in1, "EXPMP2", n), _num(in1, "FWTMP2", n, 0.0),
            _num(in1, "BITMPI", n, 0.0), labels, machine.geometry("bpol_probe"))
    if "ip" in groups and need("ip", "PLASMA" in in1):
        ms.groups["ip"] = efit_group(
            "ip", _num(in1, "PLASMA", 1), _num(in1, "FWTCUR", 1, 0.0),
            _num(in1, "BITIP", 1, 0.0), ["ip"], {})
    if "pf_current" in groups and need("pf_current", "BRSP" in in1 or "ECURRT" in in1):
        ms.groups["pf_current"] = _coils(in1, machine, efit_group, coil_current_per_turn)
    if "diamagnetic_flux" in groups and need("diamagnetic_flux", "DFLUX" in in1):
        v = _num(in1, "DFLUX", 1) * 1e-3
        s = _num(in1, "SIGDLC", 1, np.nan) * 1e-3
        w = _num(in1, "FWTDLC", 1, 0.0)
        ms.groups["diamagnetic_flux"] = ConstraintGroup(
            name="diamagnetic_flux", labels=["diamagnetic_flux"], measured=v, sigma=s,
            valid=(w > 0) & np.isfinite(s) & (s > 0), weight=w,
            meta={"sigma_model": "efit_sigdlc", "sigma_params": {},
                  "source": "efit_kfile", "units": "DFLUX, SIGDLC [mV s] x 1e-3"})
    if "mse" in groups and need("mse", "TGAMMA" in ins):
        ms.groups["mse"] = _mse(ins, machine, er_corrected)
    return ms


def _coils(in1, machine, efit_group, per_turn):
    parts = []
    for key, fkey, bkey in (("BRSP", "FWTFC", "BITFC"), ("ECURRT", "FWTEC", "BITEC")):
        if key in in1:
            parts.append((key, fkey, bkey))
    maps = machine.channel_maps
    if all(k in maps for k, _, _ in parts):
        labels = [s for k, _, _ in parts for s in maps[k]]
        sizes = [len(maps[k]) for k, _, _ in parts]
    else:
        labels = machine.labels("pf_current")
        # the weight array has the coil count; BRSP is dimensioned for all
        # EFIT response columns and padded beyond the F coils
        lens = [len(in1.get(fk) or in1[k]) for k, fk, _ in parts]
        if sum(lens) != len(labels):
            raise ValueError(f"k-file has {sum(lens)} coil currents "
                             f"({', '.join(k for k, _, _ in parts)}), the bundle names "
                             f"{len(labels)}; give channel_maps for each array")
        sizes = lens
    if per_turn is True:
        per_turn = tuple(k for k, _, _ in parts)
    per_turn = tuple(per_turn or ())
    unknown = [k for k in per_turn if k not in ("BRSP", "ECURRT")]
    if unknown:
        raise ValueError(f"coil_current_per_turn names unknown arrays {unknown}")
    val, fwt, bit, divide = [], [], [], []
    for (k, fk, bk), n in zip(parts, sizes):
        val.append(_num(in1, k, n))
        fwt.append(_num(in1, fk, n, 0.0))
        bit.append(_num(in1, bk, n, 0.0))
        divide.append(np.full(n, k in per_turn))
    val, fwt, bit, divide = (np.concatenate(x) for x in (val, fwt, bit, divide))
    geometry = {}
    if "pf_current" in machine.sensors:
        order = machine.labels("pf_current")
        idx = [order.index(s) for s in labels]
        geometry = machine.geometry("pf_current", idx)
    scale = np.ones(len(labels))
    if divide.any():
        turns = geometry.get("turns")
        if turns is None:
            raise ValueError("coil_current_per_turn needs 'turns' in the bundle's pf_current")
        scale = np.where(divide, 1.0 / np.asarray(turns, dtype=float), 1.0)
    extra = {"current_convention": {k: ("k-file value / bundle turns" if k in per_turn
                                        else "as written in the k-file")
                                    for k, _, _ in parts}}
    return efit_group("pf_current", val, fwt, bit, labels, geometry, scale=scale,
                      extra=extra)


def _mse(ins, machine, er_corrected):
    sig_key = "SGAMMA" if ins.get("SGAMMA") else "SIGGAM"
    core = ["TGAMMA", sig_key, "FWTGAM", "RRRGAM", "ZZZGAM",
            "AA1GAM", "AA2GAM", "AA3GAM", "AA4GAM"]
    missing = [k for k in core if not ins.get(k)]
    if missing:
        raise ValueError(f"k-file MSE block lacks {missing}")
    n = min(len(ins[k]) for k in core)
    tg = _num(ins, "TGAMMA", n)
    sg = _num(ins, sig_key, n)
    fw = _num(ins, "FWTGAM", n, 0.0)
    a = {i: _num(ins, f"AA{i}GAM", n, 0.0) for i in range(1, 8)}
    coef = efit_to_imas_coefficients(a[1], a[2], a[3], a[4], a[5], a[6], a[7])
    usecer = ins.get("MSE_USECER")
    usecer = None if not usecer else float(usecer[0])
    if er_corrected == "auto":
        if usecer is None:
            raise ValueError("MSE_USECER missing from the k-file: whether TGAMMA is "
                             "E_r-corrected is unknown; pass er_corrected=True/False")
        erc, how = usecer == 1.0, f"MSE_USECER={usecer:g}"
    else:
        erc, how = bool(er_corrected), "caller override"
    if "mse" in machine.sensors and len(machine.labels("mse")) == n:
        labels = machine.labels("mse")
    else:
        labels = [f"mse_{i:03d}" for i in range(n)]
    valid = (fw > 0) & np.isfinite(sg) & (sg > 0) & np.isfinite(tg) & (np.abs(tg) > 1e-8)
    meta = {"sigma_model": "efit_sgamma", "sigma_params": {}, "source": "efit_kfile",
            "quantity": "tan_polarisation_angle", "er_corrected": bool(erc),
            "er_corrected_source": how, "coefficient_source": "AA1GAM..AA7GAM"}
    if ins.get("TGAMMAUNCOR") and len(ins["TGAMMAUNCOR"]) >= n:
        meta["tan_gamma_uncorrected"] = _num(ins, "TGAMMAUNCOR", n)
    for key in ("DTMSEFULL", "MSE_SPAVE_ON"):
        if ins.get(key):
            meta[key] = _num(ins, key)
    return ConstraintGroup(name="mse", labels=labels, measured=tg, sigma=sg, valid=valid,
                           weight=fw,
                           geometry={"r": _num(ins, "RRRGAM", n), "z": _num(ins, "ZZZGAM", n),
                                     "coefficients": coef},
                           meta=meta)
