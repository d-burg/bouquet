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

**sigma models.**  The default (``sigma_model="default"``) is chosen from
what the caller supplies, and the model actually used is recorded:

* EFIT's stored sigma (model ``"efit_stored"``) when ``stored_sigma`` is
  given -- the sigma arrays EFIT wrote for the same run (its measurement
  file); the adapter only applies its unit scaling;
* otherwise the data-inferred flux-loop rule: model ``"efit_kersil"`` with
  the ``"siref"`` reading (below).  Its inputs are required, never assumed:
  ``ERRSIL`` and ``SICONT`` (k-file, else ``efit_constants``, else the
  machine bundle's ``meta["efit_sigma_constants"]``), the reference-loop
  number ``NSLREF`` (k-file, else ``nslref``, else
  ``meta["efit_sigma_constants"]["NSLREF"]``) and the loop radii ``r`` of
  the machine bundle; a missing one raises a ``ValueError`` naming it.

Every group records ``meta["sigma_model"]`` (the model name),
``meta["sigma_model_id"]`` (the model with its reading, e.g.
``"efit_kersil:siref"``) and ``meta["sigma_rule"]``; the set records
``meta["sigma_model"]`` (the model used), ``meta["sigma_model_requested"]``,
``meta["sigma_model_selection"]`` (why) and ``meta["sigma_models"]``
(group -> ``sigma_model_id``), so tables built from different models can be
told apart.

**sigma model ``"efit_kersil"``**: EFIT's own rule as read from its
weight-setting step (L. L. Lao et al., Nucl. Fusion 25, 1611 (1985) describe
the code; the rule below was read from its source).  For every group with a
digitizer-bit array other than the flux loops,

.. code-block:: text

    sigma = max(SERROR |value|, VBIT bit)

-- probes ``max(SERROR |EXPMP2|, VBIT BITMPI)``, Ip ``max(SERROR |PLASMA|,
VBIT BITIP)``, F-coils ``max(SERROR |BRSP|, VBIT BITFC)``, E-coils
``max(SERROR |ECURRT|, VBIT BITEC)`` (EFIT forms an E-coil sigma only when
``IECURR = 2``; for other ``IECURR`` the same formula is this package's
scoring convention, recorded in the group's ``meta``) -- the diamagnetic flux
``SIGDLC`` x 1e-3 and MSE ``SGAMMA``.  Flux loops by ``KERSIL``
(:func:`kersil_loop_sigma`)::

    KERSIL = 2, 3:  sigma_m = max(ERRSIL |psi_m - psi_ref|, SICONT R_m |Ip|, VBIT PSIBIT_m)
                    reference loop: KERSIL = 3 -> max(ERRSIL |psi_ref|, SICONT R_ref |Ip|, VBIT PSIBIT_ref)
                                    KERSIL = 2 -> ERSIL8
    otherwise:      sigma_m = max(SERROR |psi_m|, VBIT PSIBIT_m)
                    (reference loop with zero bit: SERROR max |psi|)

with ``R_m`` the loop radii of the machine bundle and the reference loop
``|NSLREF|``.  ``ERRSIL``, ``SICONT``, ``ERSIL8`` are EFIT inputs whose
defaults were not available here: they are taken from the file when present,
else from ``efit_constants`` or the machine bundle's
``meta["efit_sigma_constants"]``, else refused.  What ``psi`` and
``psi_ref`` are at that step is stated with ``kersil_reference``:
``"source"`` (the order of operations as read: ``COILS``, plus ``SIREF`` on
every loop when ``FITSIREF`` is false and ``IECURR > 0`` or ``NSLREF < 0``;
reference = that loop's value) or ``"siref"`` (reference flux = ``SIREF``;
other loops relative to it in stored form).  On a large set of EFIT runs,
with constants fitted to EFIT's stored sigma, ``"siref"`` reproduced every
loop channel of an independent control set and 97 % of the fitted set (the
rest a sub-percent difference on a few discharges), while ``"source"``
reproduced about 60 %; ``"siref"`` is therefore an inference from data, not
a reading of the source, and it is the reading of the default.  Values below
the EFIT floor (1e-10, file units) make a channel invalid, as EFIT zeroes its
weight.  Under an explicit ``sigma_model="efit_kersil"`` the reading must be
stated.

**sigma model ``"efit_vbit"``** (named, no longer the default): the same
``max(SERROR |value|, VBIT bit)`` for every bit-array group, with flux loops
``max(SERROR |COILS|, VBIT PSIBIT)`` on ``COILS`` AS STORED (the reference
loop named by ``flux_reference`` gets ``VBIT PSIBIT`` only) and Ip
``SERROR |PLASMA|`` without the bit floor.  For probes, coil currents, MSE and
the diamagnetic flux it is EFIT's sigma; for flux loops it matched EFIT's
stored sigma on only about 60 % of the loop channels of a large set of EFIT
runs, which is why it was retired as the default.

**``VBIT``.**  Read from ``&IN1``; when absent, given by the caller
(``vbit=``), else taken from the machine bundle's piecewise default by shot,
``meta["vbit_by_shot"] = [[first_shot, value], ...]``: the value of the last
entry whose ``first_shot <= ISHOT`` (EFIT builds set a device- and
era-dependent default when the namelist omits ``VBIT``; the device's rule
belongs to its machine description, not to this module).  When none of the
three gives a value -- or ``ISHOT`` is absent, or below the first entry -- a
model that needs ``VBIT`` raises; models that do not use it are not blocked.
The value used and where it came from are recorded (``"VBIT"``,
``"VBIT_source"``) in each magnetics group's ``meta`` and in the set's
``meta``.

``sigma_model="efit_serror_bit"`` keeps the oldest convention, a fixed
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

**Coil roles in EFIT's fit** (recorded in ``meta["coil_roles"]``, one entry
per coil, and summarized in ``meta["coil_role_rule"]``).  Evidence: m-file /
k-file pairs (calculated = measured exactly for E-coils with ``IECURR = 1``;
F-coil calculated != measured where ``FWTFC = 0``; a k-file with every
``FWTFC = 0`` still has every calculated F current different from the
measured one), and the k-file flags ``IECURR``, ``IFITVS`` (inferred from
EFIT output, not from its source):

* ``"fitted"`` -- an F-coil (``BRSP``) with ``FWTFC > 0``: its current is a
  free fit parameter AND its measurement is a constraint row of EFIT's fit;
* ``"free"`` -- an F-coil with ``FWTFC = 0``: its current is still a free fit
  parameter, but its MEASUREMENT is not a constraint (it is NOT held fixed);
* ``"held"`` -- an E-coil (``ECURRT``) when ``IECURR = 1``: EFIT holds it at
  the measured current;
* ``"fitted"`` / ``"free"`` for E-coils when ``IECURR = 2`` (by ``FWTEC``), and
  ``"undetermined"`` when ``IECURR`` is absent or has another value (the caller
  must state the role).

**Scoring coils EFIT did not constrain.**  By default
(``score_unfitted_coils=True``) coils with ``FWTFC`` / ``FWTEC`` = 0 are valid
measurements with weight 1 for SCORING (a scoring convention of this package) and are listed in
``meta["not_fitted_by_efit"]``; EFIT's own weights stay in
``meta["efit_fit_weight"]``.  Whether a coil measurement is a constraint of a
FIT is a separate question answered by ``coil_roles`` (a fit should use rows
only for ``"fitted"`` coils and hold the ``"held"`` ones).
``score_unfitted_coils=False`` reproduces the earlier behavior (valid only
where FWT > 0, weight = FWT).

**Flux-loop gauge (``SIREF``; settled by direct comparison with m-file
values).**  Three k-file families have been seen:
(A) OMFIT kinetic-EFIT aggregates store ``COILS`` RELATIVE to the reference
loop (its own entry exactly 0) with the measured absolute reference flux in
``SIREF``; EFIT itself fits absolute fluxes, m-file ``silopt = COILS + SIREF``
to 6e-8; (B) kinetic-EFIT files with ``FITSIREF = T`` also store relative
``COILS`` (the reference entry small but not zero) with the MEASURED
reference flux in ``SIREF``, and EFIT fits an additive reference offset
together with its other unknowns (its calculated loop flux equals the
stored relative value plus that offset); whether the fitted offset is
written to a measurement file depends on the code version -- some versions
store it (where present, the closed form below reproduces it exactly),
others write it only to the fit's text output -- so it is never read from
a file here; (C) EFIT re-runs store ABSOLUTE ``COILS`` with ``SIREF = 0``
for some discharges and relative ``COILS`` with ``SIREF != 0`` and
``FITSIREF = T`` for others.  In every family the absolute flux is ``COILS +
SIREF`` (per radian, the file's sign).

The storage form (``meta["coils_form"]``: ``"relative"``, ``"absolute"`` or
``"undetermined"``) is decided per slice by :func:`flux_gauge`, in this order,
and the rule that decided is recorded (``meta["coils_form_rule"]``):

1. ``"fitsiref"`` -- ``FITSIREF = T`` in the file: relative to a reference
   offset that EFIT fits (scored with a fitted offset, below);
2. ``"siref_zero"`` / ``"reference_entry_zero"`` -- what one slice can decide
   without a threshold: ``SIREF = 0`` (or absent) -> absolute; the reference
   loop's entry exactly 0 with ``SIREF != 0`` -> relative;
3. ``"series"`` -- several slices of one discharge
   (:func:`flux_storage_series`, passed in as ``flux_storage``): relative when
   the reference loop's entry stays closer to a constant than to ``SIREF``
   while ``SIREF`` changes, absolute in the opposite case (a comparison, no
   threshold);
4. otherwise ``"undetermined"``: naming a ``flux_reference`` is then refused.

``meta["absolute_offset"]`` = ``flux_scale * SIREF`` [Wb]: ``measured +
absolute_offset`` is the absolute flux in the group's units (``SIREF`` is
the MEASURED reference flux, so the solver's gauge never comes from EFIT's
fit).  Only a solver needs the absolute gauge; scoring stays in the form the
data are stored in (relative to ``flux_reference`` when that is named),
except for a fitted reference below.

**Fitted reference offset.**  When the file's own record says EFIT fitted
the reference (``FITSIREF = T``), the loops are scored as EFIT fits them
(``fitsiref_scoring="fitted_offset"``, default): absolutely modeled plus ONE
additive offset ``c`` per table, minimizing ``sum_m ((model_m - measured_m +
c)/sigma_m)^2 + ((c - c0)/s0)^2`` with the prior row ``c0 = -flux_scale x
SIREF`` (the measured reference flux) and ``s0`` the reference loop's sigma,
the reference loop ALSO an ordinary row; no prior row when the reference loop
is switched off (EFIT weight 0); one extra datum (the prior row) and one
extra parameter (``c``) in the counts.  The group's meta then carries
``flux_offset = "fitted"``, ``flux_offset_prior`` and
``flux_offset_reference`` (the reference loop: ``flux_reference`` when named,
else ``|NSLREF|``), and ``flux_reference`` is None (:func:`..score.score`).
``fitsiref_scoring="reference_loop"`` keeps the earlier reading (relative to
the named reference loop; recorded).

**Uncertainty-model and vessel switches.**  A file whose namelist turns on
EFIT's calibration-error uncertainty model (``IMAGSIGMA > 0``) is refused
under every rule-based sigma model unless
``allow_calibration_sigma_option=True`` (none reproduces it); its state is
recorded in ``meta["calibration_sigma_option"]``.  The vessel-current
switches the file carries (``IFITVS``, ``IVESEL``, ``VLOOP``) are recorded
in ``meta["vessel"]``; when ``IVESEL`` is absent, whether EFIT prescribed
vessel currents is unknown (``prescription_known=False``).

**Units and sign.**  Flux loops are multiplied by ``flux_scale``, default
``-2 pi``.  The k-files tested store ``COILS`` in the sign of the flux
whose ``B_Z = +(1/R) dpsi/dR`` -- TokaMaker's ``psi_TM`` sign -- WHATEVER the
direction of Ip (fit ``COILS = a psi_EFIT + b`` against EFIT's own ``psirz``
and the probe-fitted field sign: ``a = -1.005 .. -1.007`` x 2 pi in the
needed scale on 4 cases with Ip > 0 / < 0 and B_t > 0 / < 0, in an
independent review).  IMAS DD 3.41.0 ``magnetics.flux_loop[:].flux`` is the flux
"over loop in which Z component of normal to loop is directed downwards",
i.e. ``-2 pi psi_TM`` (what :func:`..diagnostics.flux_loop` models), so the
measured value is ``-2 pi COILS``.  (The earlier default ``+2 pi`` with a
COCOS-1 "same sign" argument was wrong for these files.)  Other producers
may differ: ``flux_scale`` stays a stated argument.  A k-file may store loop
fluxes RELATIVE to a reference loop or ABSOLUTE -- decided PER SLICE from the
file (``SIREF`` and the reference entry; "Flux-loop gauge" above).  Naming a
``flux_reference`` on a slice that is not in relative form raises: the
relative model ``psi_i - psi_ref`` would then be compared with absolute data.

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

**Conventions** (documented defaults; the shared list is in :mod:`..adapters`):

* sentinels: MSE ``|TGAMMA| <= 1e-8`` is "no data" (EFIT writes 0); a sigma
  at or below EFIT's floor 1e-10 (file units) makes a channel invalid, as
  EFIT zeroes its weight; namelist nulls are kept positional and recorded;
* exclusions: magnetics and MSE channels with ``FWT* = 0`` are invalid
  (EFIT's own fit selection); coil currents with ``FWT = 0`` are scored with
  weight 1 by default and listed (``score_unfitted_coils``); further default
  exclusions (machine bundle ``channel_exclusions``, ``exclude=``) are
  recorded in the group's ``meta["excluded"]``;
* window reduction: none here -- the file holds values EFIT already averaged
  over ``ITIME +/- IAVEM``; the adapter only records that window;
* nothing is defaulted silently: ``SERROR``, ``ITIME``, ``IAVEM``, the fit
  weights and the bit arrays are required (or given by the caller, recorded),
  and nonzero entries beyond the bundle's channel count are refused.
"""

import re

import numpy as np

from ..interfaces import ConstraintGroup, MeasurementSet
from ..diagnostics import PROBE_N_AVG_DEFAULT
from ..measurements import efit_to_imas_coefficients
from . import apply_exclusions, sigma_from_model

# ---------------------------------------------------------------------------
# Fortran namelist parser
# ---------------------------------------------------------------------------

_BLOCK = re.compile(r"[&$]\s*([A-Za-z_][A-Za-z0-9_]*)(.*?)(?:^\s*/\s*$|[&$]END\b|\Z)",
                    re.S | re.M | re.I)
# a key: NAME, NAME(i), NAME(i:j) or (refused) NAME(i,j,...); the lookbehind keeps names inside numbers out
_KEY = re.compile(r"(?<![\w.'\"])([A-Za-z_][A-Za-z0-9_]*)\s*(\(\s*([^)]*?)\s*\))?\s*=")
_STR = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_TOKEN = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|,|[^\s,]+")
_COMMENT = re.compile(r"!.*$", re.M)


def _value(tok):
    t = tok.strip()
    if not t:
        return None
    if t[0] in "'\"":
        q = t[0]
        return t[1:-1].replace(q + q, q)
    low = t.lower()
    if low in (".true.", "t", ".t."):
        return True
    if low in (".false.", "f", ".f."):
        return False
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        raise ValueError(f"namelist value {t!r} is not a number, logical or string") from None


def _values(text):
    """Values of one namelist assignment, positionally: a null value (two consecutive commas, a leading comma,
    or ``r*`` with nothing after the star) is ``None`` -- Fortran leaves that element unchanged, so the
    positions of later values are kept.  Trailing nulls (a separator before the next name) are dropped."""
    out = []
    prev_comma = True                     # a comma right at the start is a leading null
    for tok in _TOKEN.findall(text):
        if tok == ",":
            if prev_comma:
                out.append(None)
            prev_comma = True
            continue
        prev_comma = False
        if "*" in tok and tok[0] not in "'\"":
            k, v = tok.split("*", 1)
            try:
                count = int(k)
            except ValueError:
                raise ValueError(f"namelist repeat {tok!r} has no integer count") from None
            out.extend([_value(v) if v else None] * count)
        else:
            out.append(_value(tok))
    while out and out[-1] is None and text.rstrip().endswith(","):
        out.pop()
        break
    return out


def parse_namelists(text):
    """Parse Fortran namelist text into ``{BLOCK: {KEY: value}}``.

    Block and key names are upper-cased (Fortran names are case-blind).
    Values are lists (numeric entries as float, logicals as bool, strings as
    str); ``n*v`` repeats are expanded.  A key repeated within a block keeps
    its last assignment (Fortran semantics); ``KEY(i) = v ...`` assigns from
    1-based position ``i`` and ``KEY(i:j) = ...`` assigns positions ``i..j``
    (more values than the range holds raises).  A null value (``a, , b``,
    ``r*``) is kept as ``None`` at its position (Fortran: that element is
    unchanged), never compacted.  ``!`` starts a comment (outside strings).
    Refused with ``ValueError``: multi-dimensional entries ``KEY(i,j)=``, any
    unparsed text before the first key of a block, and tokens that are not
    numbers, logicals or strings (e.g. ``'****'`` overflow fields).  Text
    outside ``&NAME ... /`` blocks is ignored.
    """
    text = _STR.sub(lambda s: s.group(0).replace("!", "\1"), text)
    text = _COMMENT.sub("", text).replace("\1", "!")
    out = {}
    for m in _BLOCK.finditer(text):
        name, body = m.group(1).upper(), m.group(2)
        masked = _STR.sub(lambda s: "\0" * len(s.group(0)), body)
        keys = list(_KEY.finditer(masked))
        head = body[:keys[0].start()] if keys else body
        if head.strip():
            raise ValueError(f"namelist block {name}: unparsed text before the first key: {head.strip()[:40]!r}")
        block = out.setdefault(name, {})
        for i, k in enumerate(keys):
            end = keys[i + 1].start() if i + 1 < len(keys) else len(body)
            vals = _values(body[k.end():end])
            key = k.group(1).upper()
            sub = k.group(3)
            if sub is not None:
                if "," in sub:
                    raise ValueError(f"namelist block {name}: multi-dimensional entry {key}({sub}) is not supported")
                rng = re.fullmatch(r"(\d+)\s*(?::\s*(\d+))?", sub)
                if rng is None:
                    raise ValueError(f"namelist block {name}: cannot read the subscript of {key}({sub})")
                start = int(rng.group(1)) - 1
                if rng.group(2) is not None:
                    stop = int(rng.group(2))
                    if stop <= start:
                        raise ValueError(f"namelist block {name}: empty range {key}({sub})")
                    if len(vals) > stop - start:
                        raise ValueError(f"namelist block {name}: {len(vals)} values for the range {key}({sub})")
                cur = list(block.get(key, []))
                cur.extend([None] * max(0, start + len(vals) - len(cur)))
                for j, v in enumerate(vals):
                    if v is not None:          # a null leaves the element unchanged
                        cur[start + j] = v
                vals = cur
            else:
                prev = block.get(key)
                if prev is not None and any(v is None for v in vals):
                    # nulls leave earlier assignments of this key unchanged
                    vals = [prev[j] if (v is None and j < len(prev)) else v for j, v in enumerate(vals)] \
                        + list(prev[len(vals):])
            block[key] = vals
    return out


def read_kfile(path):
    """Parse an EFIT k-file into ``{BLOCK: {KEY: values}}``."""
    with open(path) as f:
        return parse_namelists(f.read())


# ---------------------------------------------------------------------------
# k-file -> MeasurementSet
# ---------------------------------------------------------------------------

def _num(block, key, n=None, default=None, log=None):
    """Array ``key`` of a namelist block as floats (a null entry -> NaN).

    With ``n``: entries beyond ``n`` are dropped only when every one of them is zero or null (EFIT pads its
    namelist arrays to the namelist dimension); a nonzero entry beyond ``n`` raises (the machine bundle has
    fewer channels than the file -- e.g. a wrong era).  A drop is recorded in ``log`` (a dict) when given.
    A short array is padded with NaN (or ``default`` when the key is absent altogether).
    """
    v = block.get(key)
    if v is None:
        if default is None:
            return None
        v = [default] * (n or 1)
    a = np.array([np.nan if x is None else float(x) for x in v], dtype=float)
    if n is not None:
        if a.size < n:
            a = np.concatenate([a, np.full(n - a.size, np.nan)])
        elif a.size > n:
            extra = a[n:]
            if np.any(np.isfinite(extra) & (extra != 0.0)):
                raise ValueError(f"k-file array {key} has {a.size} entries with nonzero values beyond the "
                                 f"{n} channels of the machine bundle: the bundle does not match the file")
            if log is not None:
                log[key] = int(a.size - n)
            a = a[:n]
    return a


def _scalar(block, key, default=None):
    a = _num(block, key)
    return default if a is None or a.size == 0 or not np.isfinite(a[0]) else float(a[0])


ALL_GROUPS = ("flux_loop", "bpol_probe", "ip", "diamagnetic_flux", "pf_current", "mse")

#: Magnetics sigma models the k-file adapter can apply (see module docstring); ``"default"`` selects one.
KFILE_SIGMA_MODELS = ("default", "efit_stored", "efit_kersil", "efit_vbit", "efit_serror_bit")
#: The adapter's default sigma model (module docstring, "sigma models").
DEFAULT_SIGMA_MODEL = "default"
#: Readings of the reference flux in the ``efit_kersil`` flux-loop rule (module docstring).
KERSIL_REFERENCE_READINGS = ("source", "siref")
#: The reading the default sigma model uses (inferred from EFIT output, module docstring).
DEFAULT_KERSIL_REFERENCE = "siref"
#: EFIT zeroes the weight of a channel whose sigma does not exceed this (file units).
EFIT_SIGMA_FLOOR = 1.0e-10
#: Machine-bundle meta key of the piecewise ``VBIT`` default by shot: ``[[first_shot, value], ...]``.
VBIT_BY_SHOT_KEY = "vbit_by_shot"


def measurements_from_kfile(kfile, machine, groups=None, time_window=None,
                            er_corrected="auto", flux_scale=-2.0 * np.pi,
                            coil_current_per_turn=(), flux_reference=None, meta=None,
                            sigma_model=DEFAULT_SIGMA_MODEL, score_unfitted_coils=True, *,
                            serror=None, time=None, allow_missing_bits=False,
                            efit_constants=None, kersil_reference=None, nslref=None,
                            stored_sigma=None, vbit=None, flux_storage=None, exclude=None,
                            fitsiref_scoring="fitted_offset", allow_calibration_sigma_option=False):
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
        Multiplies ``COILS`` and ``PSIBIT``; default ``-2 pi`` (module
        docstring, "Units and sign").
    coil_current_per_turn : bool or sequence of str
        Source arrays (``"BRSP"``, ``"ECURRT"``) whose currents are divided by
        the bundle's ``pf_current`` ``turns``; True = all.
    flux_reference : str, optional
        Label of the reference loop the ``COILS`` values are relative to.
    sigma_model : {"default", "efit_stored", "efit_kersil", "efit_vbit", "efit_serror_bit"}
        Magnetics sigma convention (module docstring, "sigma models").
        ``"default"``: ``efit_stored`` when ``stored_sigma`` is given, else
        ``efit_kersil`` with the ``"siref"`` reading (its inputs required).
    score_unfitted_coils : bool
        True (default): coils with ``FWTFC`` / ``FWTEC`` = 0 are valid
        measurements with weight 1, noted in ``meta``.  False: they are
        invalid (the earlier behavior).
    serror : float, optional
        Used only when the file has no ``SERROR`` (otherwise the file's value
        is used and this must be None); without either, a ``ValueError`` is
        raised.  The source is recorded in ``meta["SERROR_source"]``.
    time : float, optional
        Slice time [s], used only when the file has no ``ITIME`` (otherwise a
        ``ValueError``); recorded in ``meta["time_source"]``.
    allow_missing_bits : bool
        False (default): a group whose digitizer-bit array (``PSIBIT``,
        ``BITMPI``, ``BITFC``/``BITEC``, ``BITIP`` where used) is absent
        raises.  True: the floor is taken as 0 for that group and recorded in
        its ``meta["bit_source"]``.
    efit_constants : dict, optional
        ``efit_kersil`` (and the default without ``stored_sigma``):
        ``{"ERRSIL": ..., "SICONT": ..., "ERSIL8": ...}``
        (EFIT input-namelist names; ERSIL8 needed only for ``KERSIL = 2``).
        A constant present in the k-file is taken from the file; otherwise
        from this dict, else from ``machine.meta["efit_sigma_constants"]``,
        else a ``ValueError``.  Every value's source is recorded.
    kersil_reference : {"source", "siref"}
        How the reference flux of the ``efit_kersil`` flux-loop rule is read
        (module docstring): required under an explicit
        ``sigma_model="efit_kersil"``; the default uses ``"siref"``.
    nslref : int, optional
        ``efit_kersil`` (and the default without ``stored_sigma``): EFIT's
        reference-loop number (1-based), used when the file has no ``NSLREF``;
        else ``machine.meta["efit_sigma_constants"]["NSLREF"]``; without any,
        a ``ValueError``.
    stored_sigma : dict, optional
        ``efit_stored`` only: EFIT's own sigma per group (``"flux_loop"``,
        ``"bpol_probe"``, ``"ip"``, ``"pf_current"``, ``"diamagnetic_flux"``,
        ``"mse"``) as written by EFIT next to the k-file (m-file ``sigsil``,
        ``sigmpi``, ``sigpasma``, ``sigfcc`` + ``sigecc``, ``sigdia``,
        ``siggam``), in EFIT's units (flux loops Wb/rad, diamagnetic flux Wb);
        the adapter applies ``|flux_scale|`` and the per-turn coil scaling.
        Every built group must be present.
    vbit : float, optional
        Used only when the file has no ``VBIT`` (otherwise a ``ValueError``);
        without it the machine bundle's ``meta["vbit_by_shot"]`` applies
        (module docstring), else models that need VBIT raise.
    flux_storage : dict or str, optional
        Storage-form evidence for slices one file cannot decide: the record
        returned by :func:`flux_storage_series` for the discharge this slice
        belongs to, or ``"relative"`` / ``"absolute"`` stated by the caller
        (recorded).  Never overrides what the file itself decides.
    exclude : dict, optional
        ``{group: {label: reason}}``: channels to exclude by default, added to
        the machine bundle's ``meta["channel_exclusions"]``; applied as
        recorded exclusions (:func:`.apply_exclusions`).
    fitsiref_scoring : {"fitted_offset", "reference_loop"}
        How flux loops are scored when the file's own record says EFIT fitted
        the reference (``FITSIREF = T``; module docstring, "Fitted reference
        offset"): ``"fitted_offset"`` (default; EFIT's convention, one fitted
        additive offset with the reference prior row) or ``"reference_loop"``
        (relative to the reference loop named by ``flux_reference``, the
        earlier reading; non-default, recorded).
    allow_calibration_sigma_option : bool
        False (default): a file whose namelist turns on EFIT's calibration-error
        uncertainty model (``IMAGSIGMA > 0``) is refused under every rule-based
        sigma model (none reproduces it); ``efit_stored`` is allowed.  True
        accepts it (recorded as an override).
    """
    if sigma_model not in KFILE_SIGMA_MODELS:
        raise ValueError(f"sigma_model must be one of {KFILE_SIGMA_MODELS}, got {sigma_model!r}")
    requested_model = sigma_model
    if sigma_model == "default":
        if stored_sigma is not None:
            sigma_model = "efit_stored"
            selection = "default: EFIT's stored sigma supplied (stored_sigma)"
        else:
            if kersil_reference not in (None, DEFAULT_KERSIL_REFERENCE):
                raise ValueError(f"the default sigma model uses the {DEFAULT_KERSIL_REFERENCE!r} reading; for "
                                 f"kersil_reference={kersil_reference!r} name sigma_model='efit_kersil'")
            sigma_model, kersil_reference = "efit_kersil", DEFAULT_KERSIL_REFERENCE
            selection = ("default: no stored sigma given; the data-inferred flux-loop rule (efit_kersil, reading "
                         f"{DEFAULT_KERSIL_REFERENCE!r}) with EFIT's rule for the other groups")
    else:
        selection = "named by the caller"
    if sigma_model == "efit_kersil" and kersil_reference not in KERSIL_REFERENCE_READINGS:
        raise ValueError(f"sigma_model 'efit_kersil' needs kersil_reference in {KERSIL_REFERENCE_READINGS} "
                         "(stated, never defaulted)")
    if sigma_model == "efit_stored" and not isinstance(stored_sigma, dict):
        raise ValueError("sigma_model 'efit_stored' needs stored_sigma={group: EFIT sigma array}")
    if fitsiref_scoring not in FITSIREF_SCORING:
        raise ValueError(f"fitsiref_scoring must be one of {FITSIREF_SCORING}")
    nl = read_kfile(kfile) if isinstance(kfile, str) else kfile
    in1 = nl.get("IN1", {})
    calib = calibration_sigma_option(nl)
    if calib["on"] and sigma_model != "efit_stored":
        if not allow_calibration_sigma_option:
            raise ValueError(f"the k-file turns on EFIT's calibration-error uncertainty model (IMAGSIGMA = "
                             f"{calib['IMAGSIGMA']:g}): no rule-based sigma model of this adapter reproduces it; use "
                             "sigma_model='efit_stored' with EFIT's stored sigma, or pass "
                             "allow_calibration_sigma_option=True (recorded)")
        calib["override"] = "allow_calibration_sigma_option=True: rule-based sigma used although EFIT's run did not"
    serror_file = _scalar(in1, "SERROR")
    if serror_file is not None:
        if serror is not None:
            raise ValueError("serror given by the caller but the k-file sets SERROR: remove one")
        serror, serror_src = serror_file, "k-file &IN1"
    elif serror is not None:
        serror, serror_src = float(serror), "caller (SERROR absent from &IN1)"
    else:
        serror_src = "absent from &IN1 and not given (refused by every group that needs it)"
    vbit, vbit_src, vbit_err = _resolve_vbit(in1, vbit, machine)
    itime = _scalar(in1, "ITIME")
    iavem = _scalar(in1, "IAVEM")
    if itime is not None:
        if time is not None:
            raise ValueError("time given by the caller but the k-file sets ITIME: remove one")
        t_mid, t_src = itime * 1e-3, "k-file &IN1 ITIME [ms]"
    elif time is not None:
        t_mid, t_src = float(time), "caller (ITIME absent from &IN1)"
    else:
        raise ValueError("ITIME is absent from the k-file's &IN1: state the slice time with time=... [s]")
    if time_window is None:
        if iavem is None:
            raise ValueError("IAVEM is absent from the k-file's &IN1: state time_window=... [s] (full width)")
        time_window = 2.0 * iavem * 1e-3
        tw_src = ("2 x IAVEM [ms]: EFIT averages the magnetics over ITIME +/- IAVEM "
                  "(IAVEM is a half-width)")
    else:
        tw_src = "caller"
    truncated = {}
    ms_meta = {"source": "efit_kfile", "SERROR": serror, "SERROR_source": serror_src, "ITIME_ms": itime,
               "IAVEM_ms": iavem, "time_source": t_src, "time_window_source": tw_src,
               "time_window_s": [t_mid - 0.5 * float(time_window),
                                 t_mid + 0.5 * float(time_window)],
               "sigma_model": sigma_model, "sigma_model_requested": requested_model,
               "sigma_model_selection": selection, "truncated_trailing_zeros": truncated,
               "calibration_sigma_option": calib, "vessel": vessel_switches(nl)}
    nulls = {f"{b}.{k}": [i for i, x in enumerate(v) if x is None]
             for b, blk in nl.items() for k, v in blk.items() if isinstance(v, list) and None in v}
    if nulls:
        ms_meta["namelist_null_entries"] = nulls
    if sigma_model in ("efit_vbit", "efit_kersil") and vbit_err is None:
        ms_meta.update({"VBIT": vbit, "VBIT_source": vbit_src})
    if isinstance(kfile, str):
        ms_meta["file"] = kfile.rsplit("/", 1)[-1]
    ms_meta.update(meta or {})
    ms = MeasurementSet(machine=machine.machine, time=t_mid,
                        time_window=float(time_window), meta=ms_meta)
    requested = groups is not None
    groups = list(groups) if requested else list(ALL_GROUPS)
    model_id = f"efit_kersil:{kersil_reference}" if sigma_model == "efit_kersil" else sigma_model

    def need(name, present):
        if not present and requested:
            raise ValueError(f"group '{name}' requested but absent from the k-file")
        return present

    def arrays(name, vkey, fkey, bkey, n, bit_needed=True):
        """Value, weight and bit arrays of one group; missing weight or bit arrays are refused (or, for bits,
        recorded when allow_missing_bits)."""
        if serror is None:
            raise ValueError(f"group '{name}': SERROR is absent from the k-file's &IN1; state it with serror=... "
                             "(no default is assumed)")
        if vbit_err is not None and sigma_model in ("efit_vbit", "efit_kersil"):
            raise ValueError(f"group '{name}': {vbit_err}")
        value = _num(in1, vkey, n, log=truncated)
        if np.count_nonzero(np.isfinite(value)) < n:
            raise ValueError(f"group '{name}': k-file has {np.count_nonzero(np.isfinite(value))} values for "
                             f"{n} sensors in the machine bundle")
        if fkey not in in1:
            raise ValueError(f"group '{name}': {vkey} present but {fkey} (fit weights) absent from &IN1")
        fwt = _num(in1, fkey, n, log=truncated)
        if bkey in in1:
            bit, bsrc = _num(in1, bkey, n, log=truncated), f"k-file &IN1 {bkey}"
        elif not bit_needed:
            bit, bsrc = np.zeros(n), f"{bkey} absent (not used by this sigma rule)"
        elif allow_missing_bits:
            bit, bsrc = np.zeros(n), f"{bkey} absent: floor taken as 0 (allow_missing_bits=True)"
        else:
            raise ValueError(f"group '{name}': digitizer-bit array {bkey} absent from &IN1 (the sigma floor would "
                             "be 0); pass allow_missing_bits=True to accept that explicitly")
        if np.any(~np.isfinite(fwt)):
            raise ValueError(f"group '{name}': {fkey} has null or non-finite entries")
        return value, fwt, bit, bsrc

    def efit_group(name, value, fwt, bit, labels, geometry, scale=1.0, extra=None,
                   rule="bit", floor_only=None, unfitted_valid=False, sigma_override=None,
                   override_text=None, bit_source=None):
        # rule "bit": max(SERROR |x|, n bit); rule "relative": SERROR |x|
        # (Ip under efit_vbit).  floor_only: channels whose sigma is the bit
        # floor alone (the reference loop).  unfitted_valid: FWT = 0 channels
        # stay valid with weight 1 (coils, scoring convention).  sigma_override:
        # sigma in file units from another rule (efit_kersil loops, efit_stored).
        n = len(labels)
        if np.count_nonzero(np.isfinite(value)) < n:
            raise ValueError(f"group '{name}': k-file has "
                             f"{np.count_nonzero(np.isfinite(value))} values for "
                             f"{n} sensors in the machine bundle")
        value, fwt, bit = value[:n] * scale, fwt[:n], bit[:n] * np.abs(scale)
        if sigma_override is not None:
            sigma = np.asarray(sigma_override, dtype=float)[:n] * np.abs(scale)
            floor = EFIT_SIGMA_FLOOR * np.abs(scale)
            sigma = np.where(sigma > floor, sigma, np.nan)
            params = {"serror": serror, "bit": bit, "vbit": vbit}
            text = override_text
        elif sigma_model == "efit_serror_bit":
            sigma = sigma_from_model("efit_serror_bit", value, serror=serror, bit=bit)
            params = {"serror": serror, "bit": bit}
            text = "max(SERROR |x|, 10 bit)"
        elif rule == "relative" and sigma_model == "efit_vbit":
            sigma = sigma_from_model("relative", value, relative=serror)
            params = {"serror": serror}
            text = "SERROR |x| (no bit floor for this group in the efit_vbit model)"
        else:
            sigma = sigma_from_model("efit_vbit", value, serror=serror, bit=bit, vbit=vbit)
            params = {"serror": serror, "bit": bit, "vbit": vbit}
            text = "max(SERROR |x|, VBIT bit)"
            if floor_only is not None and np.any(floor_only):
                sigma = np.where(floor_only, vbit * bit, sigma)
                text += "; reference loop: VBIT bit"
        good = np.isfinite(value) & np.isfinite(sigma) & (sigma > 0)
        unfitted = fwt <= 0
        if unfitted_valid:
            valid, weight = good, np.where(unfitted, 1.0, fwt)
        else:
            valid, weight = good & ~unfitted, fwt
        gm = {"sigma_model": sigma_model, "sigma_model_id": model_id, "sigma_params": params, "sigma_rule": text,
              "source": "efit_kfile", "SERROR_source": serror_src}
        if bit_source is not None:
            gm["bit_source"] = bit_source
        if sigma_model in ("efit_vbit", "efit_kersil"):
            gm.update({"VBIT": vbit, "VBIT_source": vbit_src})
        if unfitted_valid:
            gm["efit_fit_weight"] = fwt.copy()
            gm["not_fitted_by_efit"] = [labels[i] for i in np.nonzero(unfitted)[0]]
            if np.any(unfitted):
                gm["note"] = ("EFIT gave the coils in 'not_fitted_by_efit' fit weight 0: their "
                              "MEASUREMENT is not a constraint of EFIT's fit (an F-coil current is "
                              "then still a free fit parameter; an E-coil with IECURR = 1 is held at "
                              "its measured current -- see meta['coil_roles']); they are scored "
                              "here as valid measurements with weight 1")
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
        coils_raw, fwt_l, bit_l, bsrc = arrays("flux_loop", "COILS", "FWTSI", "PSIBIT", n)
        gauge = flux_gauge(in1, coils_raw, labels.index(flux_reference) if flux_reference else None,
                           flux_scale, series=flux_storage)
        ms.meta["SIREF"] = gauge["siref"]
        if flux_reference is not None and gauge["coils_form"] != "relative":
            raise ValueError(f"flux_reference {flux_reference!r} given, but this slice's loop fluxes are not in "
                             f"relative form (coils_form = {gauge['coils_form']!r}: {gauge['coils_form_evidence']}); "
                             "the storage form is decided per slice -- do not name a reference for absolute data")
        extra = {"flux_scale": flux_scale, "units": "COILS [Wb/rad] x flux_scale",
                 "flux_reference": flux_reference,
                 "sigma_basis": "COILS as stored (relative to the reference loop "
                                "when the file stores relative fluxes)"}
        fitted_offset = gauge["coils_form_rule"] == "fitsiref" and fitsiref_scoring == "fitted_offset"
        if gauge["coils_form_rule"] == "fitsiref":
            extra["fitsiref_scoring"] = fitsiref_scoring
        if fitted_offset:
            # the file's own record says EFIT fitted the reference: one additive offset with the reference prior
            # row (the measured reference flux SIREF and the reference loop's sigma), the reference loop also an
            # ordinary row; flux_reference then only names that loop (the scorer's flux_reference stays None)
            extra.update(flux_reference=None, flux_reference_requested=flux_reference, flux_offset="fitted")
        override, otext = None, None
        if sigma_model == "efit_kersil":
            plasma = _scalar(in1, "PLASMA")
            if plasma is None:
                raise ValueError("efit_kersil flux-loop sigma needs PLASMA (Ip) in &IN1")
            r_m = machine.geometry("flux_loop").get("r")
            nsl_given, nsl_src = nslref, "caller (NSLREF absent from the k-file)"
            bundle_nsl = ((getattr(machine, "meta", None) or {}).get("efit_sigma_constants") or {}).get("NSLREF")
            if nsl_given is None and bundle_nsl is not None:
                nsl_given, nsl_src = bundle_nsl, "machine bundle meta['efit_sigma_constants']['NSLREF']"
            override, krec = kersil_loop_sigma(in1, coils_raw, bit_l, r_m, plasma, vbit, serror,
                                               _efit_constants(in1, efit_constants, machine), kersil_reference,
                                               nsl_given, nslref_source=nsl_src)
            otext = krec["rule"]
            extra.update({"kersil": krec, "sigma_basis": krec["basis"]})
            ref = np.zeros(n, dtype=bool)
        elif sigma_model == "efit_stored":
            override, otext = _stored(stored_sigma, "flux_loop", n), "EFIT's stored sigma (sigsil) x |flux_scale|"
            ref = np.zeros(n, dtype=bool)
        ms.groups["flux_loop"] = efit_group(
            "flux_loop", coils_raw, fwt_l, bit_l, labels, machine.geometry("flux_loop"),
            scale=flux_scale, floor_only=ref, sigma_override=override, override_text=otext, bit_source=bsrc,
            extra=dict(gauge, **extra))
        if fitted_offset:
            _fitted_offset_prior(ms.groups["flux_loop"], in1, labels, flux_reference, flux_scale, gauge, nslref,
                                 machine)
    if "bpol_probe" in groups and need("bpol_probe", "EXPMP2" in in1):
        labels = machine.labels("bpol_probe")
        n = len(labels)
        v, w, b, bsrc = arrays("bpol_probe", "EXPMP2", "FWTMP2", "BITMPI", n)
        ms.groups["bpol_probe"] = efit_group(
            "bpol_probe", v, w, b, labels, machine.geometry("bpol_probe"), bit_source=bsrc,
            sigma_override=_stored(stored_sigma, "bpol_probe", n) if sigma_model == "efit_stored" else None,
            override_text="EFIT's stored sigma (sigmpi)")
    if "ip" in groups and need("ip", "PLASMA" in in1):
        v, w, b, bsrc = arrays("ip", "PLASMA", "FWTCUR", "BITIP", 1, bit_needed=sigma_model != "efit_vbit")
        ms.groups["ip"] = efit_group(
            "ip", v, w, b, ["ip"], {}, rule="relative", bit_source=bsrc,
            sigma_override=_stored(stored_sigma, "ip", 1) if sigma_model == "efit_stored" else None,
            override_text="EFIT's stored sigma (sigpasma)")
    if "pf_current" in groups and need("pf_current", "BRSP" in in1 or "ECURRT" in in1):
        ms.groups["pf_current"] = _coils(in1, machine, efit_group, coil_current_per_turn,
                                         score_unfitted_coils, arrays, truncated,
                                         stored_sigma if sigma_model == "efit_stored" else None)
    if "diamagnetic_flux" in groups and need("diamagnetic_flux", "DFLUX" in in1):
        v = _num(in1, "DFLUX", 1) * 1e-3
        if "FWTDLC" not in in1:
            raise ValueError("group 'diamagnetic_flux': DFLUX present but FWTDLC absent from &IN1")
        w = _num(in1, "FWTDLC", 1)
        if sigma_model == "efit_stored":
            s = _stored(stored_sigma, "diamagnetic_flux", 1)
            dm = {"sigma_model": "efit_stored", "sigma_model_id": "efit_stored",
                  "sigma_rule": "EFIT's stored sigma (sigdia) [Wb]"}
        else:
            if "SIGDLC" not in in1:
                raise ValueError("group 'diamagnetic_flux': DFLUX present but SIGDLC absent from &IN1")
            s = _num(in1, "SIGDLC", 1) * 1e-3
            dm = {"sigma_model": "efit_sigdlc", "sigma_model_id": "efit_sigdlc",
                  "sigma_rule": "SIGDLC x 1e-3 (SERROR not involved)"}
        ms.groups["diamagnetic_flux"] = ConstraintGroup(
            name="diamagnetic_flux", labels=["diamagnetic_flux"], measured=v, sigma=s,
            valid=(w > 0) & np.isfinite(s) & (s > 0), weight=w,
            meta=dict(dm, sigma_params={}, source="efit_kfile", units="DFLUX, SIGDLC [mV s] x 1e-3"))
    if "mse" in groups and need("mse", "TGAMMA" in nl.get("INS", {})):
        ms.groups["mse"] = _mse(nl, machine, er_corrected,
                                _stored(stored_sigma, "mse", None) if sigma_model == "efit_stored" else None)
    if "bpol_probe" in ms.groups:
        ms.groups["bpol_probe"].meta.update(_probe_n_avg(machine))
    excl = _merge_exclusions((getattr(machine, "meta", None) or {}).get("channel_exclusions"), exclude)
    if excl:
        ms.meta["channel_exclusions_applied"] = apply_exclusions(ms, excl)
    ms.meta["sigma_models"] = {g: grp.meta.get("sigma_model_id", grp.meta.get("sigma_model"))
                               for g, grp in ms.groups.items()}
    return ms


def _resolve_vbit(in1, given, machine):
    """(VBIT, source, error): from the file, else the caller, else the machine bundle's piecewise default by
    shot (module docstring).  ``error`` is a message when no value can be given (raised only by models that
    need VBIT)."""
    f = _scalar(in1, "VBIT")
    if f is not None and np.isfinite(f):
        if given is not None:
            raise ValueError("vbit given by the caller but the k-file sets VBIT: remove one")
        return f, "k-file &IN1", None
    if given is not None:
        return float(given), "caller (VBIT absent from &IN1)", None
    table = (getattr(machine, "meta", None) or {}).get(VBIT_BY_SHOT_KEY)
    if not table:
        return None, None, (f"VBIT is absent from the k-file's &IN1, no vbit= was given and the machine bundle "
                            f"has no meta['{VBIT_BY_SHOT_KEY}']: state the value")
    rows = sorted((float(a), float(v)) for a, v in table)
    shot = _scalar(in1, "ISHOT")
    if shot is None:
        return None, None, ("VBIT is absent from the k-file's &IN1 and so is ISHOT: the machine bundle's default "
                            "depends on the shot number; state vbit=...")
    hit = [v for a, v in rows if a <= shot]
    if not hit:
        return None, None, (f"VBIT is absent from the k-file's &IN1 and ISHOT = {shot:g} precedes the first entry "
                            f"of the machine bundle's meta['{VBIT_BY_SHOT_KEY}']: state vbit=...")
    src = (getattr(machine, "meta", None) or {}).get(VBIT_BY_SHOT_KEY + "_source")
    return hit[-1], (f"default {hit[-1]:g}: VBIT absent from &IN1; machine bundle meta['{VBIT_BY_SHOT_KEY}'] "
                     f"with ISHOT = {shot:g}" + (f" ({src})" if src else "")), None


#: Number of averaging points of a finite-length probe when the machine bundle does not state it: the
#: scorer's documented default (:data:`..diagnostics.PROBE_N_AVG_DEFAULT`).
PROBE_N_AVG_FALLBACK = PROBE_N_AVG_DEFAULT


def _probe_n_avg(machine):
    """Probe-averaging point count for the scorer: the machine bundle's ``meta["probe_n_avg"]`` (read from the
    device file's ``NSMP2`` by whoever built the bundle), else the documented default; recorded."""
    meta = getattr(machine, "meta", None) or {}
    if meta.get("probe_n_avg") is not None:
        return {"probe_n_avg": int(meta["probe_n_avg"]),
                "probe_n_avg_source": "machine bundle meta['probe_n_avg'] (" + str(meta.get("probe_n_avg_source",
                                                                                         "source not stated")) + ")"}
    return {"probe_n_avg": PROBE_N_AVG_FALLBACK,
            "probe_n_avg_source": f"default {PROBE_N_AVG_FALLBACK} (diagnostics.PROBE_N_AVG_DEFAULT; not stated in the machine bundle)"}


def _merge_exclusions(*parts):
    out = {}
    for p in parts:
        for g, d in (p or {}).items():
            out.setdefault(g, {}).update({str(k): str(v) for k, v in dict(d).items()})
    return out


def _stored(stored, group, n):
    if group not in stored:
        raise ValueError(f"sigma_model 'efit_stored': stored_sigma has no entry for group '{group}'")
    a = np.asarray(stored[group], dtype=float).ravel()
    if n is not None and a.size < n:
        raise ValueError(f"stored_sigma['{group}'] has {a.size} entries for {n} channels")
    return a if n is None else a[:n]


#: Names of the EFIT constants of the KERSIL flux-loop rule.
KERSIL_CONSTANTS = ("ERRSIL", "SICONT", "ERSIL8")


def _efit_constants(in1, given, machine):
    """ERRSIL, SICONT, ERSIL8: from the k-file, else the caller, else ``machine.meta["efit_sigma_constants"]``.

    Their EFIT defaults were not available to this module; a set of EFIT runs was reproduced with values found
    by fitting EFIT's stored sigma, which is why they are inputs (no value is assumed here)."""
    bundle = (getattr(machine, "meta", None) or {}).get("efit_sigma_constants") or {}
    out, src = {}, {}
    for k in KERSIL_CONSTANTS:
        v = _scalar(in1, k)
        if v is not None:
            out[k], src[k] = v, "k-file &IN1"
        elif given and k in given:
            out[k], src[k] = float(given[k]), "caller (efit_constants)"
        elif k in bundle:
            out[k], src[k] = float(bundle[k]), "machine bundle meta['efit_sigma_constants']"
    out["_source"] = src
    return out


def kersil_loop_sigma(in1, coils, psibit, r_loop, plasma, vbit, serror, consts, reading, nslref=None,
                      nslref_source="caller (NSLREF absent from the k-file)"):
    """EFIT's flux-loop sigma under its ``KERSIL`` switch, in file units (Wb/rad), and a record.

    ``KERSIL`` = 2 or 3 (module docstring, sigma model ``efit_kersil``)::

        sigma_m   = max(ERRSIL |psi_m - psi_ref|, SICONT R_m |Ip|, VBIT PSIBIT_m)      (m != reference)
        sigma_ref = max(ERRSIL |psi_ref|,         SICONT R_ref |Ip|, VBIT PSIBIT_ref)   (KERSIL = 3)
        sigma_ref = ERSIL8                                                               (KERSIL = 2)

    any other ``KERSIL``: ``max(SERROR |psi_m|, VBIT PSIBIT_m)``, and when the reference loop's bit is zero its
    sigma is ``SERROR max_m |psi_m|``.  ``R_m`` [m] are the loop radii of the machine bundle, ``Ip`` = |PLASMA|
    [A].  The reference loop is ``|NSLREF|`` (1-based; from the file, else ``nslref``).

    ``reading`` says what ``psi`` and ``psi_ref`` are (both readings are stated assumptions):

    * ``"source"``: ``psi = COILS``, plus ``SIREF`` on every loop when ``FITSIREF`` is false and (``IECURR > 0``
      or ``NSLREF < 0``) -- the order of operations of the EFIT weight-setting step as read -- and
      ``psi_ref = psi[reference]``;
    * ``"siref"``: the reference loop's flux is ``SIREF`` (the measured reference flux written in the file) and
      every other loop's flux enters relative to it in stored form: ``COILS_m`` when ``SIREF != 0`` (relative
      storage; the file's reference entry is not used) and ``COILS_m - COILS_ref`` when ``SIREF = 0``
      (absolute storage).  This reading reproduced EFIT's stored sigma channel by channel on a set of runs where
      the ``"source"`` reading did not (inferred from data, not read from source).
    """
    kersil = _scalar(in1, "KERSIL")
    if kersil is None:
        raise ValueError("efit_kersil needs KERSIL in the k-file")
    nsl = _scalar(in1, "NSLREF")
    nsl_src = "k-file &IN1"
    if nsl is None:
        if nslref is None:
            raise ValueError("NSLREF is absent from the k-file: state the reference-loop number with nslref=... "
                             "(or the machine bundle's meta['efit_sigma_constants']['NSLREF'])")
        nsl, nsl_src = float(nslref), nslref_source
    iref = int(abs(nsl)) - 1
    n = coils.size
    if not 0 <= iref < n:
        raise ValueError(f"reference loop |NSLREF| = {abs(nsl):g} outside the {n} loops")
    fitsiref = (_flag_bool(in1, "FITSIREF"))
    iecurr = _scalar(in1, "IECURR")
    siref = _scalar(in1, "SIREF")
    ip = abs(float(plasma))
    bit = vbit * np.abs(psibit)
    rec = {"KERSIL": kersil, "NSLREF": nsl, "NSLREF_source": nsl_src, "reading": reading, "FITSIREF": fitsiref,
           "IECURR": iecurr, "SIREF": siref, "constants": {k: v for k, v in consts.items() if k != "_source"},
           "constants_source": consts.get("_source", {}), "reference_index": iref}
    if kersil in (2.0, 3.0):
        for k in ("ERRSIL", "SICONT") + (("ERSIL8",) if kersil == 2.0 else ()):
            if k not in consts:
                raise ValueError(f"efit_kersil with KERSIL = {kersil:g} needs {k}: not in the k-file; give "
                                 "efit_constants={...} or machine.meta['efit_sigma_constants']")
        if r_loop is None:
            raise ValueError("efit_kersil with KERSIL = 2/3 needs the loop radii 'r' in the machine bundle")
        r_loop = np.asarray(r_loop, dtype=float)[:n]
        if reading == "source":
            if fitsiref is None or iecurr is None:
                raise ValueError("the 'source' reading needs FITSIREF and IECURR in the k-file")
            add = (not fitsiref) and (iecurr > 0 or nsl < 0)
            psi = coils + ((siref or 0.0) if add else 0.0)
            psi_ref = psi[iref]
            rel = np.abs(psi - psi_ref)
            ref_mag = abs(psi_ref)
            basis = ("COILS" + (" + SIREF on every loop (FITSIREF false and IECURR > 0 or NSLREF < 0)" if add else "")
                     + "; reference flux = that loop's value")
        else:
            s = siref or 0.0
            rel = np.abs(coils if s != 0.0 else coils - coils[iref])
            ref_mag = abs(s)
            basis = ("reference flux = SIREF; other loops relative to it as stored ("
                     + ("relative storage, SIREF != 0: COILS_m" if s != 0.0 else
                        "absolute storage, SIREF = 0: COILS_m - COILS_ref") + ")")
        sig = np.maximum.reduce([consts["ERRSIL"] * rel, consts["SICONT"] * r_loop * ip, bit])
        if kersil == 3.0:
            sig[iref] = max(consts["ERRSIL"] * ref_mag, consts["SICONT"] * r_loop[iref] * ip, bit[iref])
            rtext = "max(ERRSIL |psi_ref|, SICONT R_ref |Ip|, VBIT PSIBIT_ref)"
        else:
            sig[iref] = consts["ERSIL8"]
            rtext = "ERSIL8"
        rule = (f"KERSIL = {kersil:g}: max(ERRSIL |psi_m - psi_ref|, SICONT R_m |Ip|, VBIT PSIBIT_m); reference loop "
                f"{rtext}; reading '{reading}'")
    else:
        psi = coils
        sig = np.maximum(serror * np.abs(psi), bit)
        if abs(psibit[iref]) <= EFIT_SIGMA_FLOOR:
            sig[iref] = serror * float(np.nanmax(np.abs(psi)))
        basis = "COILS as stored"
        rule = (f"KERSIL = {kersil:g}: max(SERROR |psi_m|, VBIT PSIBIT_m); reference loop SERROR max|psi| when its "
                "bit is zero (this branch was not checked against EFIT output)")
    rec.update(rule=rule, basis=basis)
    return sig, rec


#: Scoring readings of a FITSIREF = T file (``measurements_from_kfile(fitsiref_scoring=...)``).
FITSIREF_SCORING = ("fitted_offset", "reference_loop")


def _fitted_offset_prior(g, in1, labels, flux_reference, flux_scale, gauge, nslref, machine):
    """Set the reference prior row of the fitted loop offset on a FITSIREF = T flux-loop group: ``c0 =
    -flux_scale x SIREF`` (the measured reference flux as an offset of the stored relative values) and ``s0`` =
    the reference loop's sigma in the group's units; the reference loop is ``flux_reference`` when named, else
    ``|NSLREF|`` (file, caller, machine bundle).  When the reference loop is switched off (EFIT weight 0) or
    has no valid sigma there is no prior row (recorded), as in EFIT's fit."""
    ref = flux_reference
    src = "flux_reference (caller)"
    if ref is None:
        nsl = _scalar(in1, "NSLREF")
        src = "k-file &IN1 NSLREF"
        if nsl is None:
            nsl, src = nslref, "caller nslref"
        if nsl is None:
            nsl = ((getattr(machine, "meta", None) or {}).get("efit_sigma_constants") or {}).get("NSLREF")
            src = "machine bundle meta['efit_sigma_constants']['NSLREF']"
        if nsl is not None and 1 <= abs(int(nsl)) <= len(labels):
            ref = labels[abs(int(nsl)) - 1]
    g.meta["flux_offset_reference"] = ref
    g.meta["flux_offset_reference_source"] = src if ref is not None else "not identified"
    if ref is None:
        g.meta["flux_offset_prior_note"] = ("reference loop not identified (no flux_reference, no NSLREF): the offset "
                                            "carries no prior row")
        return
    i = labels.index(ref)
    s0 = float(g.sigma[i])
    if not (bool(g.valid[i]) and np.isfinite(s0) and s0 > 0):
        g.meta["flux_offset_prior_note"] = (f"reference loop {ref!r} switched off (EFIT weight 0) or without a valid "
                                            "sigma: no prior row, as in EFIT's fit")
        return
    g.meta["flux_offset_prior"] = [-float(flux_scale) * float(gauge["siref"]), s0]
    g.meta["flux_offset_prior_source"] = (f"measured reference flux SIREF (k-file) and the sigma of reference loop "
                                          f"{ref!r}")


def calibration_sigma_option(nl):
    """Whether the file turns on EFIT's calibration-error uncertainty model (``IMAGSIGMA > 0`` in any namelist
    block; its thresholds ``ERRMAG`` / ``ERRMAGB`` recorded).  Returns ``{"on", "IMAGSIGMA", "ERRMAG", "ERRMAGB",
    "note"}``."""
    v = _flag(nl, "IMAGSIGMA")
    out = {"on": bool(v is not None and v > 0), "IMAGSIGMA": v, "ERRMAG": _flag(nl, "ERRMAG"),
           "ERRMAGB": _flag(nl, "ERRMAGB")}
    out["note"] = ("IMAGSIGMA absent: the option is off by EFIT's default" if v is None else
                   ("on: EFIT replaced its rule-based sigma by the calibration-error model during the iterations"
                    if out["on"] else "off"))
    return out


def vessel_switches(nl):
    """The vessel-current switches a k-file carries (``IFITVS``, ``IVESEL`` and ``VLOOP`` where present in any
    namelist block, else None).  ``IVESEL`` (EFIT's prescribed eddy currents from the measured loop voltage)
    is not written to every k-file: when it is absent, whether the run prescribed vessel currents is unknown
    (``prescription_known=False``) and comparisons during a current ramp are to be treated as unverified."""
    out = {k: _flag(nl, k) for k in ("IFITVS", "IVESEL", "VLOOP")}
    out["prescription_known"] = out["IVESEL"] is not None
    out["note"] = ("IVESEL not in the file: whether EFIT prescribed vessel currents is unknown (comparisons during "
                   "a current ramp unverified)" if out["IVESEL"] is None else "IVESEL read from the file")
    return out


def _flag_bool(block, key):
    v = block.get(key)
    if not v or v[0] is None:
        return None
    x = v[0]
    return bool(x) if isinstance(x, bool) else bool(float(x))


#: Rules that can decide the flux-loop storage form, in order of precedence (module docstring, "Flux-loop gauge").
FLUX_STORAGE_RULES = ("fitsiref", "siref_zero", "reference_entry_zero", "series", "caller")


def flux_gauge(in1, coils, ref_index, flux_scale, series=None):
    """The k-file's flux-loop gauge: ``SIREF`` and the storage form of ``COILS``.

    Returns ``{"siref", "siref_source", "coils_form", "coils_form_rule", "coils_form_evidence", "FITSIREF",
    "absolute_offset", "absolute_rule"}``.  The form is decided by the first rule that applies (module
    docstring, "Flux-loop gauge"): ``FITSIREF = T``; then what one slice decides exactly (``SIREF = 0`` ->
    absolute; reference entry exactly 0 with ``SIREF != 0`` -> relative); then ``series`` (the record of
    :func:`flux_storage_series`, or ``"relative"`` / ``"absolute"`` stated by the caller); otherwise
    ``"undetermined"`` (rule ``None``).  No ratio threshold is used.  ``absolute_offset = flux_scale * SIREF``
    [Wb]; the absolute flux in the group's units is ``measured + absolute_offset``.
    """
    siref = _scalar(in1, "SIREF")
    src = "k-file &IN1 SIREF" if siref is not None else "absent (taken as 0)"
    s = 0.0 if siref is None or not np.isfinite(siref) else float(siref)
    fitsiref = _flag_bool(in1, "FITSIREF")
    r = None if ref_index is None else float(coils[ref_index])
    form, rule = "undetermined", None
    if fitsiref:
        form, rule = "relative", "fitsiref"
        ev = "FITSIREF = T: loops relative to a reference offset that EFIT fits"
    elif s == 0.0 and (r is None or r != 0.0):
        form, rule = "absolute", "siref_zero"
        ev = ("SIREF = 0 (or absent): absolute storage" + ("" if r is not None else
              "; no flux_reference named, so the reference entry is not checked"))
    elif r is not None and r == 0.0 and s != 0.0:
        form, rule = "relative", "reference_entry_zero"
        ev = "reference-loop entry exactly 0 and SIREF != 0: relative storage"
    else:
        ev = ("SIREF != 0, FITSIREF not true, " + ("no flux_reference named (reference entry cannot be checked)"
                                                   if r is None else "reference entry nonzero")
              + ": one slice cannot decide")
        if series is not None:
            if isinstance(series, str):
                if series not in ("relative", "absolute"):
                    raise ValueError("flux_storage must be 'relative', 'absolute' or a flux_storage_series record")
                form, rule = series, "caller"
                ev += f"; stated by the caller: {series}"
            elif series.get("coils_form") in ("relative", "absolute"):
                form, rule = series["coils_form"], "series"
                ev += f"; series evidence: {series.get('evidence')}"
            else:
                ev += f"; series evidence does not decide ({series.get('evidence')})"
    return {"siref": s, "siref_source": src, "coils_form": form, "coils_form_rule": rule,
            "coils_form_evidence": ev, "FITSIREF": fitsiref,
            "absolute_offset": float(flux_scale) * s,
            "absolute_rule": "absolute flux = (COILS + SIREF) x flux_scale = measured + absolute_offset "
                             "(all k-file families checked against m-file silopt/psiref)"}


def flux_storage_series(kfiles, ref_index):
    """Storage form of the flux loops from several slices of ONE discharge (no threshold).

    ``kfiles`` are paths or :func:`read_kfile` dicts; ``ref_index`` is the 0-based reference loop.  Only the
    slices a single file cannot decide (:func:`flux_gauge` rules 1-2) are used.  With ``c`` the reference
    loop's ``COILS`` entry and ``s`` = ``SIREF`` over those slices: relative storage leaves ``c`` constant
    while ``s`` follows the flux swing; absolute storage puts the reference flux in ``c``, which then follows
    ``s``.  The rule compares the two sums of squares ``sum (c - mean c)^2`` and ``sum (c - s)^2``: relative
    when the first is smaller, absolute when the second is, undetermined when they are equal, when fewer than
    two such slices exist or when ``s`` does not change.  Returns ``{"coils_form", "rule", "evidence",
    "n_slices", "ss_constant", "ss_follows_siref"}``, to be passed to :func:`measurements_from_kfile` as
    ``flux_storage``.
    """
    c, s = [], []
    for k in kfiles:
        nl = read_kfile(k) if isinstance(k, str) else k
        in1 = nl.get("IN1", {})
        coils = _num(in1, "COILS")
        if coils is None or ref_index >= coils.size:
            raise ValueError("flux_storage_series: a slice has no COILS entry for the reference loop")
        g = flux_gauge(in1, coils, ref_index, 1.0)
        if g["coils_form_rule"] is None:
            c.append(float(coils[ref_index]))
            s.append(g["siref"])
    c, s = np.asarray(c), np.asarray(s)
    out = {"rule": "series: sum (c - mean c)^2 vs sum (c - SIREF)^2 over the undecided slices",
           "n_slices": int(c.size), "ss_constant": None, "ss_follows_siref": None, "coils_form": "undetermined"}
    if c.size < 2:
        out["evidence"] = f"{c.size} undecided slice(s): at least two are needed"
        return out
    if np.ptp(s) == 0.0:
        out["evidence"] = "SIREF identical on every undecided slice: the two readings cannot be told apart"
        return out
    a = float(np.sum((c - c.mean()) ** 2))
    b = float(np.sum((c - s) ** 2))
    out.update(ss_constant=a, ss_follows_siref=b)
    if a < b:
        out["coils_form"] = "relative"
    elif b < a:
        out["coils_form"] = "absolute"
    out["evidence"] = (f"{c.size} slices: reference entry spread about its mean {a:.3g}, about SIREF {b:.3g} "
                       f"(SIREF range {np.ptp(s):.3g}) -> {out['coils_form']}")
    return out


def _coil_roles(labels, sizes, parts, fwt, iecurr):
    """Per-coil role in EFIT's fit (module docstring, "Coil roles")."""
    roles, k = {}, 0
    for (key, _, _), n in zip(parts, sizes):
        for i in range(n):
            lab, w = labels[k], float(fwt[k])
            if key == "BRSP":
                roles[lab] = "fitted" if w > 0 else "free"
            elif iecurr == 1.0:
                roles[lab] = "held"
            elif iecurr == 2.0:
                roles[lab] = "fitted" if w > 0 else "free"
            else:
                roles[lab] = "undetermined"
            k += 1
    rule = ("F-coils (BRSP): FWTFC > 0 -> 'fitted' (free current + measurement row), FWTFC = 0 -> "
            "'free' (free current, measurement not a constraint); E-coils (ECURRT): IECURR = 1 -> "
            "'held' at the measured current, IECURR = 2 -> 'fitted'/'free' by FWTEC, else "
            f"'undetermined' (IECURR in this file: {iecurr})")
    return roles, rule


def _coils(in1, machine, efit_group, per_turn, score_unfitted, arrays=None, truncated=None, stored=None):
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
    val, fwt, bit, divide, bsrc = [], [], [], [], []
    for (k, fk, bk), n in zip(parts, sizes):
        if arrays is not None:
            v, w, b, src = arrays("pf_current", k, fk, bk, n)
        else:
            v, w, b, src = _num(in1, k, n), _num(in1, fk, n), _num(in1, bk, n), "k-file"
        val.append(v)
        fwt.append(w)
        bit.append(b)
        bsrc.append(src)
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
    iecurr = _scalar(in1, "IECURR")
    roles, rule = _coil_roles(labels, sizes, parts, fwt, iecurr)
    extra = {"current_convention": {k: ("k-file value / bundle turns" if k in per_turn
                                        else "as written in the k-file")
                                    for k, _, _ in parts},
             "score_unfitted_coils": bool(score_unfitted),
             "coil_roles": roles, "coil_role_rule": rule, "IECURR": iecurr,
             "ecoil_sigma_basis": ("EFIT's own (it forms an E-coil sigma when IECURR = 2)" if iecurr == 2.0 else
                                   "this package's scoring convention: EFIT forms no E-coil sigma unless "
                                   "IECURR = 2; the F-coil formula is applied"),
             "coil_arrays": {k: list(labels[sum(sizes[:j]):sum(sizes[:j + 1])])
                             for j, (k, _, _) in enumerate(parts)}}
    override = None
    if stored is not None:
        override = _stored(stored, "pf_current", len(labels))
    return efit_group("pf_current", val, fwt, bit, labels, geometry, scale=scale,
                      extra=extra, unfitted_valid=bool(score_unfitted), bit_source="; ".join(bsrc),
                      sigma_override=override, override_text="EFIT's stored sigma (sigfcc, sigecc)")


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

    Rule (this package's convention, checked against k-files of known
    provenance), on the active chords with ``D = TGAMMA - TGAMMAUNCOR``:

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
    ev = {"rule": "D = TGAMMA - TGAMMAUNCOR on active chords",
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


def _mse(nl, machine, er_corrected, stored=None):
    ins = nl.get("INS", {})
    sig_key = "SGAMMA" if ins.get("SGAMMA") else "SIGGAM"
    core = ["TGAMMA", sig_key, "FWTGAM", "RRRGAM", "ZZZGAM",
            "AA1GAM", "AA2GAM", "AA3GAM", "AA4GAM"]
    missing = [k for k in core if not ins.get(k)]
    if missing:
        raise ValueError(f"k-file MSE block lacks {missing}")
    n = min(len(ins[k]) for k in core)
    tg = _num(ins, "TGAMMA", n)
    sg = _num(ins, sig_key, n) if stored is None else _stored({"mse": stored}, "mse", n)
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
        how, route = "TGAMMA - TGAMMAUNCOR rule", st["route"]
    else:
        erc, how, route = bool(er_corrected), "caller override", "caller_stated"
    if "mse" in machine.sensors and len(machine.labels("mse")) == n:
        labels, lab_src = machine.labels("mse"), "machine bundle"
    else:
        labels = [f"mse_{i:03d}" for i in range(n)]
        lab_src = ("generic (no 'mse' sensors in the machine bundle)" if "mse" not in machine.sensors else
                   f"generic: the bundle names {len(machine.labels('mse'))} MSE channels, the file has {n}")
    meta = {"sigma_model": "efit_sgamma" if stored is None else "efit_stored",
            "sigma_model_id": "efit_sgamma" if stored is None else "efit_stored", "sigma_params": {},
            "sigma_rule": "SGAMMA" if stored is None else "EFIT's stored sigma (siggam)", "labels_source": lab_src,
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
