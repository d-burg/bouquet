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

**sigma convention** (sigma model ``"efit_vbit"``, the default): for every
group with a digitizer-bit array,

.. code-block:: text

    sigma = max(SERROR |value|, VBIT bit)

with ``SERROR`` (default 0.03 when absent) and ``VBIT`` (default 10 when
absent) both read from ``IN1``.  This is the sigma EFIT itself uses: on
k-file / m-file pairs (same run) it reproduces EFIT's stored sigma arrays
(``sigmpi``, ``sigsil``, ``sigfcc``, ``sigpasma``, ``siggam``, ``sigdia``)
to float32 precision, channel by channel, and the implied multiplier equals
the file's ``VBIT`` wherever the floor binds (``VBIT`` differs between k-file
vintages, so a fixed constant is right for only some files).  EFIT source was
not available; the rule is an empirical match, see the project record's EFIT
conventions check.  Per group:

* probes: ``max(SERROR |EXPMP2|, VBIT BITMPI)``;
* flux loops: ``max(SERROR |COILS|, VBIT PSIBIT)`` on ``COILS`` AS STORED,
  i.e. on the flux relative to the reference loop when the file stores
  relative fluxes (EFIT does not use the absolute reference flux for sigma);
  the reference loop itself (``flux_reference``) gets ``VBIT PSIBIT`` only;
  times ``|flux_scale|``;
* coil currents: ``max(SERROR |BRSP|, VBIT BITFC)`` and ``max(SERROR
  |ECURRT|, VBIT BITEC)`` (the E-coil floor never bound on the files
  checked: untested);
* Ip: ``SERROR |PLASMA|``, no bit floor (what EFIT's ``sigpasma`` shows);
* diamagnetic flux: ``SIGDLC`` x 1e-3 (SERROR not involved);
* MSE: ``SGAMMA`` = sigma(tan gamma).

The VBIT value used, and where it came from, is recorded in each magnetics
group's ``meta`` (``"VBIT"``, ``"VBIT_source"``) and in the set's ``meta``.

``sigma_model="efit_serror_bit"`` keeps the previous convention, a fixed
``max(SERROR |value|, 10 bit)`` for every bit-array group including Ip (the
OMAS machine mappings, omas 0.95.2 ``omas/machine_mappings``, build the same
max(10 digitizer LSB, 3 % |value|) and call it "how it is treated in EFIT";
it matches EFIT only on files whose ``VBIT`` is 10).

**Weights.**  The fit weight is the k-file's ``FWT*`` (``weight`` multiplies
the normalized residual, as EFIT folds FWT into 1/sigma and as in
``equilibrium.time_slice[:].constraints.*.chi_squared = weight^2 (rec -
meas)^2 / sigma^2``, IMAS DD).  A channel is valid when its FWT > 0, sigma is
finite and > 0, and (MSE) |TGAMMA| > 1e-8, since EFIT writes 0 for chords
without data.

**Coils EFIT did not fit.**  ``FWTFC`` / ``FWTEC`` = 0 means EFIT held that
coil current fixed (snap-file template: fit weights, default 0), not that the
measurement is bad.  By default (``score_unfitted_coils=True``) such coils are
valid measurements with weight 1 and are listed in the group's
``meta["not_fitted_by_efit"]``; EFIT's own weights stay in
``meta["efit_fit_weight"]``.  ``score_unfitted_coils=False`` reproduces the
earlier behavior (valid only where FWT > 0, weight = FWT).

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
to the namelist dimension).

Whether ``TGAMMA`` is already E_r-corrected is decided by
:func:`mse_er_status` from ``D = TGAMMA - TGAMMAUNCOR`` on the active chords,
never from ``MSE_USECER`` alone (k-files with ``MSE_USECER = 0`` can carry an
externally pre-corrected ``TGAMMA``, e.g. OMFIT ``MSEerCorrection``, which
writes ``TGAMMA = TGAMMAUNCOR - A5 E_R / (A2 B_phi)`` and keeps
``TGAMMAUNCOR`` raw).  ``er_corrected="auto"`` applies the rule and raises
when it cannot decide; ``True`` / ``False`` states the status explicitly (the
rule's evidence is still recorded).  The AA5..AA7GAM coefficients are
geometry, set whether or not a correction was applied, and are not used as
evidence.

**Time.**  ``ITIME`` [ms] -> ``time`` [s].  EFIT averages the magnetics over
``ITIME +/- IAVEM`` [ms] (``IAVEM`` is a half-width: the EFIT snap-file
template documents it as "+/- this value"), so ``time_window`` defaults to
``2 IAVEM`` [ms] / 1e3, the full width.  The window ``[t0, t1]`` [s] and the
reading of ``IAVEM`` are recorded in ``meta``.
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

#: Magnetics sigma models the k-file adapter can apply (see module docstring).
KFILE_SIGMA_MODELS = ("efit_vbit", "efit_serror_bit")
#: ``VBIT`` used when the k-file's ``IN1`` does not set it.
VBIT_DEFAULT = 10.0


def measurements_from_kfile(kfile, machine, groups=None, time_window=None,
                            er_corrected="auto", flux_scale=2.0 * np.pi,
                            coil_current_per_turn=(), flux_reference=None, meta=None,
                            sigma_model="efit_vbit", score_unfitted_coils=True):
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
        Full window width [s]; default ``2 IAVEM`` [ms] / 1e3 (EFIT averages
        over ``ITIME +/- IAVEM``).
    er_corrected : {"auto", True, False}
        Whether ``TGAMMA`` is E_r-corrected; ``auto`` applies
        :func:`mse_er_status` and raises when the status cannot be decided.
    flux_scale : float
        Multiplies ``COILS`` and ``PSIBIT`` (see module docstring).
    coil_current_per_turn : bool or sequence of str
        Source arrays (``"BRSP"``, ``"ECURRT"``) whose currents are divided by
        the bundle's ``pf_current`` ``turns``; True = all.
    flux_reference : str, optional
        Label of the reference loop the ``COILS`` values are relative to.
    sigma_model : {"efit_vbit", "efit_serror_bit"}
        Magnetics sigma convention (module docstring).  ``efit_vbit`` (default)
        uses the file's ``VBIT``; ``efit_serror_bit`` the fixed 10-bit floor.
    score_unfitted_coils : bool
        True (default): coils with ``FWTFC`` / ``FWTEC`` = 0 are valid
        measurements with weight 1, noted in ``meta``.  False: they are
        invalid (the earlier behavior).
    """
    if sigma_model not in KFILE_SIGMA_MODELS:
        raise ValueError(f"sigma_model must be one of {KFILE_SIGMA_MODELS}, got {sigma_model!r}")
    nl = read_kfile(kfile) if isinstance(kfile, str) else kfile
    in1 = nl.get("IN1", {})
    serror = _scalar(in1, "SERROR", 0.03)
    vbit_file = _scalar(in1, "VBIT")
    if vbit_file is None or not np.isfinite(vbit_file):
        vbit, vbit_src = VBIT_DEFAULT, f"default {VBIT_DEFAULT:g} (VBIT absent from &IN1)"
    else:
        vbit, vbit_src = vbit_file, "k-file &IN1"
    itime = _scalar(in1, "ITIME")
    iavem = _scalar(in1, "IAVEM")
    t_mid = 0.0 if itime is None else itime * 1e-3
    if time_window is None:
        time_window = 0.0 if iavem is None else 2.0 * iavem * 1e-3
        tw_src = ("2 x IAVEM [ms]: EFIT averages the magnetics over ITIME +/- IAVEM "
                  "(IAVEM is a half-width)")
    else:
        tw_src = "caller"
    ms_meta = {"source": "efit_kfile", "SERROR": serror, "ITIME_ms": itime,
               "IAVEM_ms": iavem, "time_window_source": tw_src,
               "time_window_s": [t_mid - 0.5 * float(time_window),
                                 t_mid + 0.5 * float(time_window)],
               "sigma_model": sigma_model}
    if sigma_model == "efit_vbit":
        ms_meta.update({"VBIT": vbit, "VBIT_source": vbit_src})
    if isinstance(kfile, str):
        ms_meta["file"] = kfile.rsplit("/", 1)[-1]
    ms_meta.update(meta or {})
    ms = MeasurementSet(machine=machine.machine, time=t_mid,
                        time_window=float(time_window), meta=ms_meta)
    requested = groups is not None
    groups = list(groups) if requested else list(ALL_GROUPS)

    def need(name, present):
        if not present and requested:
            raise ValueError(f"group '{name}' requested but absent from the k-file")
        return present

    def efit_group(name, value, fwt, bit, labels, geometry, scale=1.0, extra=None,
                   rule="bit", floor_only=None, unfitted_valid=False):
        # rule "bit": max(SERROR |x|, n bit); rule "relative": SERROR |x|
        # (Ip under efit_vbit).  floor_only: channels whose sigma is the bit
        # floor alone (the reference loop).  unfitted_valid: FWT = 0 channels
        # stay valid with weight 1 (coils, R04).
        n = len(labels)
        if np.count_nonzero(np.isfinite(value)) < n:
            raise ValueError(f"group '{name}': k-file has "
                             f"{np.count_nonzero(np.isfinite(value))} values for "
                             f"{n} sensors in the machine bundle")
        value, fwt, bit = value[:n] * scale, fwt[:n], bit[:n] * np.abs(scale)
        if sigma_model == "efit_serror_bit":
            sigma = sigma_from_model("efit_serror_bit", value, serror=serror, bit=bit)
            params = {"serror": serror, "bit": bit}
            text = "max(SERROR |x|, 10 bit)"
        elif rule == "relative":
            sigma = sigma_from_model("relative", value, relative=serror)
            params = {"serror": serror}
            text = "SERROR |x| (no bit floor for this group in EFIT)"
        else:
            sigma = sigma_from_model("efit_vbit", value, serror=serror, bit=bit, vbit=vbit)
            params = {"serror": serror, "bit": bit, "vbit": vbit}
            text = "max(SERROR |x|, VBIT bit)"
            if floor_only is not None and np.any(floor_only):
                sigma = np.where(floor_only, vbit * bit, sigma)
                text += "; reference loop: VBIT bit"
        fwt = np.where(np.isfinite(fwt), fwt, 0.0)
        good = np.isfinite(value) & np.isfinite(sigma) & (sigma > 0)
        unfitted = fwt <= 0
        if unfitted_valid:
            valid, weight = good, np.where(unfitted, 1.0, fwt)
        else:
            valid, weight = good & ~unfitted, fwt
        gm = {"sigma_model": sigma_model, "sigma_params": params, "sigma_rule": text,
              "source": "efit_kfile"}
        if sigma_model == "efit_vbit":
            gm.update({"VBIT": vbit, "VBIT_source": vbit_src})
        if unfitted_valid:
            gm["efit_fit_weight"] = fwt.copy()
            gm["not_fitted_by_efit"] = [labels[i] for i in np.nonzero(unfitted)[0]]
            if np.any(unfitted):
                gm["note"] = ("EFIT did not fit the coils in 'not_fitted_by_efit' "
                              "(fit weight 0: current held fixed); they are scored here "
                              "as valid measurements with weight 1")
        gm.update(extra or {})
        return ConstraintGroup(name=name, labels=labels, measured=value, sigma=sigma,
                               valid=valid, weight=weight, geometry=geometry, meta=gm)

    if "flux_loop" in groups and need("flux_loop", "COILS" in in1):
        labels = machine.labels("flux_loop")
        n = len(labels)
        if flux_reference is not None and flux_reference not in labels:
            raise ValueError(f"flux_reference {flux_reference!r} is not a flux-loop label")
        ref = np.zeros(n, dtype=bool)
        if flux_reference is not None:
            ref[labels.index(flux_reference)] = True
        ms.groups["flux_loop"] = efit_group(
            "flux_loop", _num(in1, "COILS", n), _num(in1, "FWTSI", n, 0.0),
            _num(in1, "PSIBIT", n, 0.0), labels, machine.geometry("flux_loop"),
            scale=flux_scale, floor_only=ref,
            extra={"flux_scale": flux_scale, "units": "COILS [Wb/rad] x flux_scale",
                   "flux_reference": flux_reference,
                   "sigma_basis": "COILS as stored (relative to the reference loop "
                                  "when the file stores relative fluxes)"})
    if "bpol_probe" in groups and need("bpol_probe", "EXPMP2" in in1):
        labels = machine.labels("bpol_probe")
        n = len(labels)
        ms.groups["bpol_probe"] = efit_group(
            "bpol_probe", _num(in1, "EXPMP2", n), _num(in1, "FWTMP2", n, 0.0),
            _num(in1, "BITMPI", n, 0.0), labels, machine.geometry("bpol_probe"))
    if "ip" in groups and need("ip", "PLASMA" in in1):
        ms.groups["ip"] = efit_group(
            "ip", _num(in1, "PLASMA", 1), _num(in1, "FWTCUR", 1, 0.0),
            _num(in1, "BITIP", 1, 0.0), ["ip"], {}, rule="relative")
    if "pf_current" in groups and need("pf_current", "BRSP" in in1 or "ECURRT" in in1):
        ms.groups["pf_current"] = _coils(in1, machine, efit_group, coil_current_per_turn,
                                         score_unfitted_coils)
    if "diamagnetic_flux" in groups and need("diamagnetic_flux", "DFLUX" in in1):
        v = _num(in1, "DFLUX", 1) * 1e-3
        s = _num(in1, "SIGDLC", 1, np.nan) * 1e-3
        w = _num(in1, "FWTDLC", 1, 0.0)
        ms.groups["diamagnetic_flux"] = ConstraintGroup(
            name="diamagnetic_flux", labels=["diamagnetic_flux"], measured=v, sigma=s,
            valid=(w > 0) & np.isfinite(s) & (s > 0), weight=w,
            meta={"sigma_model": "efit_sigdlc", "sigma_params": {},
                  "sigma_rule": "SIGDLC x 1e-3 (SERROR not involved)",
                  "source": "efit_kfile", "units": "DFLUX, SIGDLC [mV s] x 1e-3"})
    if "mse" in groups and need("mse", "TGAMMA" in nl.get("INS", {})):
        ms.groups["mse"] = _mse(nl, machine, er_corrected)
    return ms


def _coils(in1, machine, efit_group, per_turn, score_unfitted):
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
                                    for k, _, _ in parts},
             "score_unfitted_coils": bool(score_unfitted)}
    return efit_group("pf_current", val, fwt, bit, labels, geometry, scale=scale,
                      extra=extra, unfitted_valid=bool(score_unfitted))


# ---------------------------------------------------------------------------
# MSE E_r-correction status
# ---------------------------------------------------------------------------

#: |TGAMMA - TGAMMAUNCOR| <= ER_D_RTOL max(|TGAMMA|, |TGAMMAUNCOR|) counts as D = 0.
ER_D_RTOL = 1e-7
#: An E_r correction from a smooth E_r profile changes the sign of the implied
#: E_R along major radius at most this many times (axis crossing + one zero).
ER_MAX_SIGN_CHANGES = 2
#: Minimum fraction of active chords on which sign(D) sign(TGAMMA) agrees.
ER_MIN_SIGN_CONSISTENCY = 0.9

#: Outcomes of :func:`mse_er_status`.
ER_STATUSES = ("uncorrected", "pre_corrected", "inconsistent", "unknown", "no_active_chords")


def _flag(nl, key, blocks=("INS", "EFITIN", "IN1")):
    """First value of ``key`` found in ``blocks`` (in that order), else None."""
    for b in blocks:
        v = nl.get(b, {}).get(key)
        if v and v[0] is not None:
            return float(v[0])
    return None


def mse_er_status(tgamma, tgamma_uncor, a2, a5, r, active, usecer=None, keecur=None):
    """Decide whether k-file ``TGAMMA`` is already E_r-corrected.

    Rule (project record, EFIT conventions check, rule R25 / R05), on the
    active chords with ``D = TGAMMA - TGAMMAUNCOR``:

    0. No active chord -> ``"no_active_chords"`` (nothing to decide).
    1. ``TGAMMAUNCOR`` absent -> ``"unknown"``.
    2. ``D = 0`` on every active chord (|D| <= :data:`ER_D_RTOL` x
       max(|TGAMMA|, |TGAMMAUNCOR|)) -> ``"uncorrected"``, whatever
       ``MSE_USECER`` says (``MSE_USECER = 1`` is flagged as "requested, not
       applied").
    3. ``D != 0`` on every active chord AND E_r-shaped -> ``"pre_corrected"``,
       route ``"efit_internal"`` when ``MSE_USECER = 1`` (EFIT's CER
       high-rotation correction at k-file generation) and ``"external"``
       otherwise (e.g. OMFIT ``MSEerCorrection``).  E_r-shaped means: the
       implied ``E_R``, which is proportional to ``-D A2 B_phi / A5`` (from
       ``TGAMMA = TGAMMAUNCOR - A5 E_R / (A2 B_phi)``, OMFIT
       ``MSEerCorrection``; B. W. Rice et al., Rev. Sci. Instrum. 70 (1999)),
       changes sign at most :data:`ER_MAX_SIGN_CHANGES` times between
       neighboring chords ordered by major radius (``B_phi`` has one sign,
       so ``D A2 / A5`` is used), and ``sign(D) sign(TGAMMA)`` takes one
       value on at least :data:`ER_MIN_SIGN_CONSISTENCY` of the chords.
    4. Anything else (D mixed zero / nonzero, shape test failed, ``A5 = 0``
       on an active chord, or ``KEECUR > 0`` -- the in-fit E_r route, where
       TGAMMA should be raw -- with ``D != 0``) -> ``"inconsistent"``.

    The nonzero AA5..AA7GAM coefficients are geometry and never used as
    evidence.  ``MSE_USECER`` only selects the route of an established
    correction.  The thresholds (:data:`ER_D_RTOL`,
    :data:`ER_MAX_SIGN_CHANGES`, :data:`ER_MIN_SIGN_CONSISTENCY`) are those
    of the rule as specified; they are a heuristic checked against k-files
    with known provenance, not derived from EFIT source.

    Returns ``{"status", "route", "evidence"}``; ``evidence`` holds counts and
    fractions only.
    """
    active = np.asarray(active, dtype=bool)
    ev = {"rule": "D = TGAMMA - TGAMMAUNCOR on active chords (R25)",
          "MSE_USECER": usecer, "KEECUR": keecur, "n_active": int(active.sum()),
          "d_rtol": ER_D_RTOL, "max_sign_changes": ER_MAX_SIGN_CHANGES,
          "min_sign_consistency": ER_MIN_SIGN_CONSISTENCY}
    if not active.any():
        ev["reason"] = "no active MSE chord"
        return {"status": "no_active_chords", "route": "none", "evidence": ev}
    if tgamma_uncor is None:
        ev["reason"] = "TGAMMAUNCOR absent: the raw pitch angle is not in the file"
        return {"status": "unknown", "route": None, "evidence": ev}
    t, u = np.asarray(tgamma, float)[active], np.asarray(tgamma_uncor, float)[active]
    d = t - u
    zero = np.abs(d) <= ER_D_RTOL * np.maximum(np.abs(t), np.abs(u))
    ev["n_d_nonzero"] = int(np.count_nonzero(~zero))
    in_fit = keecur is not None and keecur > 0
    if zero.all():
        ev["reason"] = "TGAMMA == TGAMMAUNCOR on every active chord"
        if usecer == 1.0:
            ev["flag"] = "MSE_USECER = 1 but TGAMMA == TGAMMAUNCOR: correction requested, not applied"
        if in_fit:
            ev["flag_in_fit"] = "KEECUR > 0: in-fit E_r route; the forward model must include E_r"
        return {"status": "uncorrected", "route": "none", "evidence": ev}
    if not (~zero).all():
        ev["reason"] = "TGAMMA differs from TGAMMAUNCOR on some active chords but not all"
        return {"status": "inconsistent", "route": None, "evidence": ev}
    a2a, a5a = np.asarray(a2, float)[active], np.asarray(a5, float)[active]
    order = np.argsort(np.asarray(r, float)[active], kind="stable")
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (d * a2a / a5a)[order]
    if not np.all(np.isfinite(e)) or np.any(e == 0):
        ev["reason"] = "implied E_R undefined (A5 = 0 or non-finite on an active chord)"
        return {"status": "inconsistent", "route": None, "evidence": ev}
    s = np.sign(e)
    changes = int(np.count_nonzero(s[1:] != s[:-1]))
    prod = np.sign(d) * np.sign(t)
    consistency = float(max(np.mean(prod > 0), np.mean(prod < 0)))
    ev.update({"sign_changes": changes, "sign_consistency": consistency})
    shaped = changes <= ER_MAX_SIGN_CHANGES and consistency >= ER_MIN_SIGN_CONSISTENCY
    if not shaped:
        ev["reason"] = "TGAMMA - TGAMMAUNCOR is not E_r-shaped"
        return {"status": "inconsistent", "route": None, "evidence": ev}
    if in_fit:
        ev["reason"] = ("KEECUR > 0 (in-fit E_r route) but TGAMMA is already altered: "
                        "E_r would be counted twice")
        return {"status": "inconsistent", "route": None, "evidence": ev}
    ev["reason"] = "TGAMMA - TGAMMAUNCOR nonzero on every active chord and E_r-shaped"
    route = "efit_internal" if usecer == 1.0 else "external"
    return {"status": "pre_corrected", "route": route, "evidence": ev}


def _mse(nl, machine, er_corrected):
    ins = nl.get("INS", {})
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
    valid = (fw > 0) & np.isfinite(sg) & (sg > 0) & np.isfinite(tg) & (np.abs(tg) > 1e-8)
    unc = None
    if ins.get("TGAMMAUNCOR") and len(ins["TGAMMAUNCOR"]) >= n:
        unc = _num(ins, "TGAMMAUNCOR", n)
    usecer = _flag(nl, "MSE_USECER")
    keecur = _flag(nl, "KEECUR", ("IN1", "EFITIN", "INS"))
    st = mse_er_status(tg, unc, a[2], a[5], _num(ins, "RRRGAM", n), valid,
                       usecer=usecer, keecur=keecur)
    if er_corrected == "auto":
        if st["status"] in ("unknown", "inconsistent"):
            ev = st["evidence"]
            raise ValueError(
                f"MSE E_r status is {st['status']!r} ({ev.get('reason')}; "
                f"MSE_USECER={usecer}, {ev.get('n_d_nonzero', '?')}/{ev['n_active']} "
                f"chords differ, sign changes {ev.get('sign_changes', '-')}, "
                f"sign consistency {ev.get('sign_consistency', '-')}): state it "
                "explicitly with er_corrected=True/False")
        erc = st["status"] == "pre_corrected"
        how, route = "TGAMMA - TGAMMAUNCOR rule (R25)", st["route"]
    else:
        erc, how, route = bool(er_corrected), "caller override", "caller_stated"
    if "mse" in machine.sensors and len(machine.labels("mse")) == n:
        labels = machine.labels("mse")
    else:
        labels = [f"mse_{i:03d}" for i in range(n)]
    meta = {"sigma_model": "efit_sgamma", "sigma_params": {}, "sigma_rule": "SGAMMA",
            "source": "efit_kfile", "quantity": "tan_polarisation_angle",
            "er_corrected": bool(erc), "er_corrected_source": how, "er_route": route,
            "er_status": st["status"], "er_evidence": st["evidence"],
            "coefficient_source": "AA1GAM..AA7GAM"}
    if keecur is not None and keecur > 0:
        meta["er_in_fit"] = True
    if unc is not None:
        meta["tan_gamma_uncorrected"] = unc
    for key in ("DTMSEFULL", "MSE_SPAVE_ON"):
        if ins.get(key):
            meta[key] = _num(ins, key)
    return ConstraintGroup(name="mse", labels=labels, measured=tg, sigma=sg, valid=valid,
                           weight=fw,
                           geometry={"r": _num(ins, "RRRGAM", n), "z": _num(ins, "ZZZGAM", n),
                                     "coefficients": coef},
                           meta=meta)
