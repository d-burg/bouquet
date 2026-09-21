"""
Reader, writer, and interface for Osborne p-files (kinetic profile files).

The p-file format stores 1-D kinetic profiles (densities, temperatures,
rotation frequencies, etc.) on a normalised poloidal flux (``psinorm``)
grid.  Each profile block has a header line followed by rows of
``(psinorm, value, derivative)`` triples.  An optional ``N Z A of ION
SPECIES`` block records the atomic number, charge, and mass of each ion
species.

This module has **no** OMFIT or OMAS dependencies.
"""

import re
import tempfile
import warnings
from collections import OrderedDict

import numpy as np
from scipy import interpolate

# Unit conversion: n [10^20/m^3] * T [keV] -> p [kPa]
# = 1e20 * 1e3 * e / 1e3 = e * 1e20 = 16.0218 kPa per (10^20/m^3 * keV)
_NT_TO_KPA = 1.602176634e-19 * 1e20  # exactly 16.02176634

# ---------------------------------------------------------------------------
# Known profile metadata (adapted from OMFIT OMFITpFile)
# ---------------------------------------------------------------------------

DESCRIPTIONS = OrderedDict([
    ("ne", "Electron density"),
    ("te", "Electron temperature"),
    ("ni", "Ion density"),
    ("ti", "Ion temperature"),
    ("nb", "Fast ion density"),
    ("pb", "Fast ion pressure"),
    ("ptot", "Total pressure"),
    ("omeg", "Toroidal rotation: VTOR/R"),
    ("omegp", "Poloidal rotation: Bt * VPOL / (RBp)"),
    ("omgvb", "VxB rotation term in the ExB rotation frequency"),
    ("omgpp", "Diamagnetic term in the ExB rotation frequency"),
    ("omgeb", "ExB rotation frequency"),
    ("er", "Radial electric field from force balance"),
    ("ommvb", "Main ion VxB term of Er/RBp"),
    ("ommpp", "Main ion pressure term of Er/RBp"),
    ("omevb", "Electron VxB term of Er/RBp"),
    ("omepp", "Electron pressure term of Er/RBp"),
    ("kpol", "KPOL = VPOL/Bp"),
    ("omghb", "Hahm-Burrell ExB velocity shearing rate"),
    ("nz1", "Density of the 1st impurity species"),
    ("vtor1", "Toroidal velocity of the 1st impurity species"),
    ("vpol1", "Poloidal velocity of the 1st impurity species"),
    ("nz2", "Density of the 2nd impurity species"),
    ("vtor2", "Toroidal velocity of the 2nd impurity species"),
    ("vpol2", "Poloidal velocity of the 2nd impurity species"),
])

UNITS = OrderedDict([
    ("ne", "10^20/m^3"),
    ("te", "KeV"),
    ("ni", "10^20/m^3"),
    ("ti", "KeV"),
    ("nb", "10^20/m^3"),
    ("pb", "KPa"),
    ("ptot", "KPa"),
    ("omeg", "kRad/s"),
    ("omegp", "kRad/s"),
    ("omgvb", "kRad/s"),
    ("omgpp", "kRad/s"),
    ("omgeb", "kRad/s"),
    ("er", "kV/m"),
    ("ommvb", ""),
    ("ommpp", ""),
    ("omevb", ""),
    ("omepp", ""),
    ("kpol", "km/s/T"),
    ("omghb", ""),
    ("nz1", "10^20/m^3"),
    ("vtor1", "km/s"),
    ("vpol1", "km/s"),
    ("nz2", "10^20/m^3"),
    ("vtor2", "km/s"),
    ("vpol2", "km/s"),
])

# Header regex: "256 psinorm ne(10^20/m^3) dne/dpsiN"
_HEADER_RE = re.compile(
    r"^(\d+)\s+(\S+)\s+(\S+)\(([^)]*)\)\s+(.*?)\s*$"
)


# ---------------------------------------------------------------------------
# Low-level parser / writer
# ---------------------------------------------------------------------------

def _deepcopy_raw(raw):
    """Copy the parsed-profile mapping, arrays included."""
    out = OrderedDict()
    for k, v in raw.items():
        out[k] = {kk: (np.array(vv, copy=True)
                       if isinstance(vv, np.ndarray) else vv)
                  for kk, vv in v.items()}
    return out


def _read_pfile(filename):
    """Parse an Osborne p-file into an OrderedDict.

    Parameters
    ----------
    filename : str or path-like
        Path to the p-file.

    Returns
    -------
    OrderedDict
        Keyed by profile name (``"ne"``, ``"te"``, ...).  Each value is a
        dict with keys ``"psinorm"``, ``"data"``, ``"derivative"``,
        ``"units"``, and ``"deriv_label"``.

        The special key ``"N Z A"`` (if present) maps to a dict with
        ``"N"``, ``"Z"``, ``"A"`` arrays.
    """
    with open(filename, "r") as f:
        lines = f.read().strip().splitlines()

    profiles = OrderedDict()
    idx = 0
    while idx < len(lines):
        header = lines[idx]
        tokens = header.split()
        if len(tokens) < 2:
            idx += 1
            continue

        count = int(tokens[0])

        # --- Special block: N Z A of ION SPECIES ---
        if "N Z A of ION SPECIES" in header:
            rows = []
            for i in range(idx + 1, idx + 1 + count):
                rows.append(list(map(float, lines[i].split())))
            cols = list(zip(*rows))
            profiles["N Z A"] = {
                "N": np.array(cols[0]),
                "Z": np.array(cols[1]),
                "A": np.array(cols[2]),
            }
            idx += 1 + count
            continue

        # --- Standard profile block ---
        m = _HEADER_RE.match(header)
        if m is None:
            idx += 1
            continue

        _xkey = m.group(2)      # e.g. "psinorm"
        key = m.group(3)        # e.g. "ne"
        units = m.group(4)      # e.g. "10^20/m^3"
        deriv_label = m.group(5)  # e.g. "dne/dpsiN"

        rows = []
        for i in range(idx + 1, idx + 1 + count):
            rows.append(list(map(float, lines[i].split())))
        cols = list(zip(*rows))

        profiles[key] = {
            "psinorm": np.array(cols[0]),
            "data": np.array(cols[1]),
            "derivative": np.array(cols[2]),
            "units": units,
            "deriv_label": deriv_label,
        }

        idx += 1 + count

    return profiles


def _write_pfile(profiles, filename):
    """Write an OrderedDict of profiles to an Osborne p-file.

    Parameters
    ----------
    profiles : OrderedDict
        Same structure as returned by :func:`_read_pfile`.
    filename : str or path-like
        Output path.
    """
    buf = []
    # The "N Z A of ION SPECIES" ion block must be the LAST element of an
    # Osborne p-file (it is the footer, and downstream readers expect it there).
    # The reader detects it by header regardless of position, so an input with
    # N Z A elsewhere is accepted -- but on write we always force it last, rather
    # than emit in dict order, so bouquet's output is canonical no matter where
    # the block sat in the source.
    nza = None
    for key, val in profiles.items():
        if key == "N Z A":
            nza = val
            continue
        n = len(val["data"])
        if n <= 1:
            continue
        units = val.get("units", "")
        deriv_label = val.get("deriv_label", f"d{key}/dpsiN")
        buf.append(f"{n} psinorm {key}({units}) {deriv_label}\n")
        for i in range(n):
            buf.append(
                f" {val['psinorm'][i]:f}   {val['data'][i]:f}"
                f"   {val['derivative'][i]:f}\n"
            )

    if nza is not None:                       # ion species block -> always last
        n = len(nza["A"])
        buf.append(f"{n} N Z A of ION SPECIES\n")
        for i in range(n):
            buf.append(
                f" {nza['N'][i]:f}   {nza['Z'][i]:f}   {nza['A'][i]:f}\n"
            )

    with open(filename, "w") as f:
        f.writelines(buf)


# ---------------------------------------------------------------------------
# PFile class
# ---------------------------------------------------------------------------

class PFile:
    """Interface for Osborne p-file kinetic profiles.

    Parameters
    ----------
    filename : str or path-like
        Path to the p-file.
    remap : bool
        Put every profile on ONE common psinorm grid at load (default
        ``True``).  A p-file stores a separate grid per quantity, and pairing
        one quantity's values with another's grid misplaces the profile; see
        :meth:`remap`.  ``False`` keeps the profiles exactly as read.
    remap_key : str
        Which profile's grid to adopt when *remap* is ``True`` (default
        ``"ne"``).  Ignored when the file has no such channel.
    doRemap : bool, optional
        Deprecated alias for *remap*, kept so existing callers keep working.

    Examples
    --------
    >>> pf = PFile("p123456.01234")
    >>> pf.ne          # electron density array
    >>> pf.te          # electron temperature array
    >>> pf.psinorm_for("ne")  # psinorm grid for ne
    >>> "omgeb" in pf  # check if profile exists
    True
    """

    def __init__(self, filename, remap=True, remap_key="ne", doRemap=None):
        self._raw = _read_pfile(filename)
        # As-read profiles, each on ITS OWN psinorm grid.  Kept so a remap is
        # never destructive: `psinorm_for(k, native=True)` still reports where a
        # quantity was actually measured, and an unmodified file round-trips
        # through save()/to_bytes() byte-for-byte on its original grids.
        self._native = _deepcopy_raw(self._raw)
        self._modified = set()
        self._remapped_to = None
        if doRemap is not None:                 # back-compat alias
            remap = bool(doRemap)
        # A p-file without the reference channel (partial/synthetic files) has
        # no common grid to adopt -- leave it exactly as read rather than
        # raising, so remap-by-default cannot break an otherwise valid file.
        if remap and remap_key in self._raw:
            self.remap(key=remap_key, overwrite=True, warn=True)

    # -- grid bookkeeping --------------------------------------------------
    def native_grid_spread(self, ref_key="ne"):
        """Max |psinorm - psinorm(ref_key)| across profiles, as READ.

        Nonzero means the p-file carries a genuinely different radial grid per
        quantity -- the normal case for DIII-D Osborne files, where pairing one
        quantity's values with another's grid misplaces the profile.
        """
        if self._native is None:
            return {}          # built in memory: no as-read grids to compare
        ref = self._native.get(ref_key)
        if ref is None:
            return {}
        g0 = np.asarray(ref["psinorm"], dtype=float)
        out = {}
        for k, v in self._native.items():
            if k == "N Z A" or k == ref_key:
                continue
            g = np.asarray(v["psinorm"], dtype=float)
            if len(g) != len(g0):
                out[k] = float("inf")
            else:
                d = float(np.abs(g - g0).max())
                if d > 0.0:
                    out[k] = d
        return out

    @classmethod
    def from_bytes(cls, raw_bytes, remap=True, remap_key="ne"):
        """Construct from in-memory bytes.

        Parameters
        ----------
        raw_bytes : bytes
            Raw p-file content.
        """
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pfile", delete=False
        ) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        return cls(tmp_path, remap=remap, remap_key=remap_key)

    # --- Persistence ---

    def _write_source(self, native=None):
        """Which profile set save()/to_bytes() should emit.

        ``native=None`` (default) is the honest choice: a file that has only
        been READ round-trips on its own native grids, while a file whose
        profiles were modified (a perturbed draw, a computed rotation) emits
        the modified state -- which necessarily lives on the common grid.
        Pass ``True``/``False`` to force either.
        """
        if native is None:
            native = not self._modified
        if native and self._native is not None:
            return self._native
        return self._raw

    def save(self, filename, native=None):
        """Write the profiles to *filename* in p-file format.

        See :meth:`_write_source` for what ``native`` selects.
        """
        _write_pfile(self._write_source(native), filename)

    def to_bytes(self, native=None):
        """Serialize to in-memory bytes (round-trip with ``from_bytes``).

        An unmodified file serialises on its NATIVE per-quantity grids, so
        reading and re-writing does not silently rewrite the radial grids of a
        provenance copy.  See :meth:`_write_source`.

        Returns
        -------
        bytes
            Raw p-file content.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pfile", delete=False
        ) as tmp:
            tmp_path = tmp.name
        _write_pfile(self._write_source(native), tmp_path)
        with open(tmp_path, "rb") as fh:
            data = fh.read()
        import os
        os.remove(tmp_path)
        return data

    # --- Dict-like access ---

    @property
    def keys(self):
        """Profile names in file order (list of str)."""
        return list(self._raw.keys())

    def __contains__(self, key):
        return key in self._raw

    def __getitem__(self, key):
        """Return the raw sub-dict for *key*."""
        return self._raw[key]

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)

    # --- Per-profile accessors ---

    def psinorm_for(self, key, native=False):
        """Return the psinorm grid for profile *key*.

        With ``native=True``, the grid the quantity was MEASURED on, even after
        a remap -- so provenance survives the interpolation.

        Returns ``None`` if *key* is not present or is the ``"N Z A"``
        block.
        """
        source = (self._native if native and self._native is not None
                  else self._raw)
        entry = source.get(key)
        if entry is None or key == "N Z A":
            return None
        return entry["psinorm"]

    def derivative_for(self, key):
        """Return the derivative array for profile *key*."""
        entry = self._raw.get(key)
        if entry is None or key == "N Z A":
            return None
        return entry["derivative"]

    def units_for(self, key):
        """Return the units string for profile *key*."""
        entry = self._raw.get(key)
        if entry is None or key == "N Z A":
            return None
        return entry.get("units", "")

    # --- Named properties for common profiles ---

    def _get_data(self, key):
        entry = self._raw.get(key)
        if entry is None:
            return None
        return entry["data"]

    @property
    def ne(self):
        """Electron density [10^20/m^3]."""
        return self._get_data("ne")

    @property
    def te(self):
        """Electron temperature [KeV]."""
        return self._get_data("te")

    @property
    def ni(self):
        """Ion density [10^20/m^3]."""
        return self._get_data("ni")

    @property
    def ti(self):
        """Ion temperature [KeV]."""
        return self._get_data("ti")

    @property
    def nb(self):
        """Fast ion density [10^20/m^3]."""
        return self._get_data("nb")

    @property
    def pb(self):
        """Fast ion pressure [KPa]."""
        return self._get_data("pb")

    @property
    def ptot(self):
        """Total pressure [KPa]."""
        return self._get_data("ptot")

    @property
    def omeg(self):
        """Toroidal rotation VTOR/R [kRad/s]."""
        return self._get_data("omeg")

    @property
    def omegp(self):
        """Poloidal rotation Bt*VPOL/(RBp) [kRad/s]."""
        return self._get_data("omegp")

    @property
    def omgvb(self):
        """VxB rotation term [kRad/s]."""
        return self._get_data("omgvb")

    @property
    def omgpp(self):
        """Diamagnetic rotation term [kRad/s]."""
        return self._get_data("omgpp")

    @property
    def omgeb(self):
        """ExB rotation frequency [kRad/s]."""
        return self._get_data("omgeb")

    @property
    def er(self):
        """Radial electric field [kV/m]."""
        return self._get_data("er")

    @property
    def kpol(self):
        """KPOL = VPOL/Bp [km/s/T]."""
        return self._get_data("kpol")

    @property
    def omghb(self):
        """Hahm-Burrell ExB shearing rate."""
        return self._get_data("omghb")

    @property
    def ion_species(self):
        """Ion species dict with 'N', 'Z', 'A' arrays, or None."""
        return self._raw.get("N Z A")

    # --- Construction helpers ---

    @classmethod
    def new(cls):
        """Create an empty PFile (no profiles loaded from disk).

        Returns
        -------
        PFile
        """
        obj = object.__new__(cls)
        obj._raw = OrderedDict()
        # A file built in memory has no on-disk original, so there is nothing
        # to round-trip back to; everything written is "modified".
        obj._native = None
        obj._modified = set()
        obj._remapped_to = None
        return obj

    def set_profile(self, key, psinorm, data, derivative=None, units=None):
        """Add or replace a profile.

        Parameters
        ----------
        key : str
            Profile name (e.g. ``"ne"``, ``"te"``).
        psinorm : array-like
            Normalised poloidal flux grid.
        data : array-like
            Profile values on *psinorm*.
        derivative : array-like or None
            Derivative d(data)/d(psinorm).  If ``None``, computed via
            ``np.gradient``.
        units : str or None
            Unit label.  If ``None``, looked up from :data:`UNITS`.
        """
        psinorm = np.asarray(psinorm, dtype=float)
        data = np.asarray(data, dtype=float)
        # any write makes this no longer a faithful copy of the file on disk
        getattr(self, "_modified", set()).add(key)
        if derivative is None:
            derivative = np.gradient(data, psinorm)
        else:
            derivative = np.asarray(derivative, dtype=float)
        if units is None:
            units = UNITS.get(key, "")
        self._raw[key] = {
            "psinorm": psinorm,
            "data": data,
            "derivative": derivative,
            "units": units,
            "deriv_label": f"d{key}/dpsiN",
        }

    def set_ion_species(self, N, Z, A):
        """Set the ion species block.

        Parameters
        ----------
        N, Z, A : array-like
            Atomic number, charge state, and mass number for each species.
        """
        self._raw["N Z A"] = {
            "N": np.asarray(N, dtype=float),
            "Z": np.asarray(Z, dtype=float),
            "A": np.asarray(A, dtype=float),
        }

    def compute_derivatives(self):
        """Recompute d(data)/d(psinorm) for all profiles in place."""
        for key, val in self._raw.items():
            if key == "N Z A":
                continue
            val["derivative"] = np.gradient(val["data"], val["psinorm"])

    # --- Physics computations ---

    def compute_pressure(self):
        """Compute total pressure from density and temperature profiles.

        Uses the relation ``ptot = _NT_TO_KPA * (ne*Te + (ni + nz1)*Ti) + pb``
        where the constant converts from (10^20/m^3 * keV) to kPa.

        Requires ``ne``, ``te``, ``ni``, ``ti`` on the same psinorm grid.
        ``nz1`` and ``pb`` default to zero if absent.
        """
        psinorm = self._raw["ne"]["psinorm"]
        ne = self._raw["ne"]["data"]
        te = self._raw["te"]["data"]
        ni = self._raw["ni"]["data"]
        ti = self._raw["ti"]["data"]

        nz1 = self._get_data("nz1")
        if nz1 is None:
            nz1 = np.zeros_like(psinorm)
        pb = self._get_data("pb")
        if pb is None:
            pb = np.zeros_like(psinorm)

        ptot = _NT_TO_KPA * (ne * te + (ni + nz1) * ti) + pb
        self.set_profile("ptot", psinorm, ptot)

    def compute_quasineutrality(self):
        """Compute impurity density nz1 from quasi-neutrality.

        ``nz1 = (ne - ni - Z_beam nb) / Z_impurity``

        ``nb`` is a PARTICLE density -- the same convention
        :meth:`compute_zeff` uses when it weights the block by
        ``Z_beam**2`` -- so it enters quasi-neutrality weighted by the beam
        charge from the species block, not bare.  The two agree for the
        hydrogenic beams these files usually describe; they do not for a
        helium or heavier beam, where charging ``nb`` at Z=1 would push the
        difference into ``nz1``.

        Requires ``ne``, ``ni`` on the same grid and a ``"N Z A"`` block
        with at least one impurity species.  ``nb`` defaults to zero if
        absent.
        """
        nza = self._raw.get("N Z A")
        if nza is None:
            raise ValueError("Ion species (N Z A) block required")
        Z_imp = nza["Z"][0]
        Z_beam = float(nza["Z"][-1]) if len(nza["Z"]) > 2 else 1.0

        psinorm = self._raw["ne"]["psinorm"]
        ne = self._raw["ne"]["data"]
        ni = self._raw["ni"]["data"]
        nb = self._get_data("nb")
        if nb is None:
            nb = np.zeros_like(psinorm)

        nz1 = (ne - ni - Z_beam * nb) / Z_imp
        n_neg = np.count_nonzero(nz1 < 0)
        if n_neg:
            warnings.warn(
                f"Quasi-neutrality produced negative nz1 at {n_neg}/{len(nz1)} "
                f"grid points (min = {nz1.min():.4g}).  This usually means "
                f"the perturbed ne is too low relative to ni + Z_beam*nb.  Consider "
                f"skipping quasi-neutrality recomputation and keeping the "
                f"baseline impurity density instead."
            )
        self.set_profile("nz1", psinorm, nz1)

    def compute_zeff(self):
        """Compute the effective charge profile.

        .. math::

            Z_{\\mathrm{eff}}
            = \\frac{\\sum_s n_s Z_s^2}{n_e}
            = \\frac{n_i Z_{\\mathrm{main}}^2
                   + n_{z1} Z_{\\mathrm{imp}}^2
                   + n_b Z_{\\mathrm{beam}}^2}{n_e}

        Charge states are read from the ``"N Z A"`` block (OMFIT
        convention: impurities first, then main ion, beam ion last).

        Requires ``ne``, ``ni`` on the same grid and an ``"N Z A"`` block.
        ``nz1`` and ``nb`` default to zero if absent.

        Returns
        -------
        psinorm : numpy.ndarray
            Normalised poloidal flux grid.
        zeff : numpy.ndarray
            Effective charge profile (dimensionless).

        Notes
        -----
        This is intentionally **not** written into the p-file (Zeff is
        not a standard p-file key) to preserve OMFIT compatibility.
        """
        nza = self._raw.get("N Z A")
        if nza is None:
            raise ValueError("Ion species (N Z A) block required")
        Z_imp = nza["Z"][0]
        Z_main = nza["Z"][-2]
        Z_beam = nza["Z"][-1]

        psinorm = self._raw["ne"]["psinorm"]
        ne = self._raw["ne"]["data"]
        ni = self._raw["ni"]["data"]
        nz1 = self._get_data("nz1")
        if nz1 is None:
            nz1 = np.zeros_like(psinorm)
        nb = self._get_data("nb")
        if nb is None:
            nb = np.zeros_like(psinorm)

        with np.errstate(divide="ignore", invalid="ignore"):
            zeff = (ni * Z_main**2 + nz1 * Z_imp**2 + nb * Z_beam**2) / ne
        # At the boundary where ne -> 0, Zeff is undefined; default to 1
        np.nan_to_num(zeff, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
        return psinorm, zeff

    def compute_diamagnetic_rotations(self, psi, nI=None, TI=None):
        """Compute diamagnetic rotation frequencies from kinetic profiles.

        The diamagnetic frequency for species *s* is

        .. math::

            \\omega_{\\mathrm{dia},s}
            = \\frac{1}{n_s Z_s e} \\frac{\\mathrm{d}(n_s T_s)}{\\mathrm{d}\\psi}

        In p-file units (n in 10^20/m^3, T in keV, psi in Wb) this
        reduces to ``d(n*T)/dpsi / (n * Z)`` in kRad/s.

        Parameters
        ----------
        psi : array-like
            Poloidal flux in SI (Weber), same length as the profile grids.
            Typically from the g-file: ``psi = psiN * (psi_bdy - psi_axis)
            + psi_axis``.
        nI : array-like or None
            Impurity density in 10^20/m^3.  If ``None``, uses
            ``nz1`` from the pfile (must already be set).
        TI : array-like or None
            Impurity temperature in keV.  If ``None``, uses ``ti``
            (assumes all ions share the same temperature).

        Sets
        ----
        omgpp, ommpp, omepp : profiles in kRad/s
        """
        psi = np.asarray(psi, dtype=float)
        dpsi = np.gradient(psi)
        psinorm = self._raw["ne"]["psinorm"]

        ne = self._raw["ne"]["data"]
        te = self._raw["te"]["data"]
        ni = self._raw["ni"]["data"]
        ti = self._raw["ti"]["data"]

        if nI is None:
            nI = self._raw["nz1"]["data"]
        else:
            nI = np.asarray(nI, dtype=float)
        if TI is None:
            TI = ti
        else:
            TI = np.asarray(TI, dtype=float)

        nza = self._raw.get("N Z A")
        if nza is None:
            raise ValueError("Ion species (N Z A) block required")
        Z_imp = nza["Z"][0]

        # Floor |dpsi| to avoid division-by-zero spikes near the axis
        # where psi is nearly flat.  The floor is set at 1e-4 × max|dpsi|
        # which is small enough to preserve the physics everywhere except
        # the degenerate axis point.
        dpsi_floor = max(1e-4 * np.max(np.abs(dpsi)), 1e-30)
        dpsi_safe = np.where(np.abs(dpsi) > dpsi_floor, dpsi,
                             np.sign(dpsi) * dpsi_floor)
        # Also floor densities to avoid spikes where n → 0 at edges
        # or where quasi-neutrality pushes nz1 negative.
        n_floor = 1e-4 * np.max(np.abs(ne))
        nI_safe = np.maximum(np.abs(nI), n_floor)
        ni_safe = np.maximum(np.abs(ni), n_floor)
        ne_safe = np.maximum(np.abs(ne), n_floor)

        # Impurity diamagnetic (counter-current, negative by convention)
        with np.errstate(divide="ignore", invalid="ignore"):
            omgpp = -np.abs(np.gradient(nI * TI) / dpsi_safe / (nI_safe * Z_imp))
            # Main ion diamagnetic (counter-current, negative by convention)
            ommpp = -np.abs(np.gradient(ni * ti) / dpsi_safe / (ni_safe * 1.0))
            # Electron diamagnetic (co-current, positive by convention)
            omepp = np.abs(np.gradient(ne * te) / dpsi_safe / (ne_safe * 1.0))
        # Replace any remaining NaN/inf
        np.nan_to_num(omgpp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(ommpp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(omepp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        self.set_profile("omgpp", psinorm, omgpp)
        self.set_profile("ommpp", psinorm, ommpp)
        self.set_profile("omepp", psinorm, omepp)

    def compute_rotation_decomposition(self, R=None, Bp=None, Bt=None,
                                       psi=None):
        """Compute ExB and VxB rotation frequencies and derived quantities.

        From the diamagnetic terms (``omgpp``, ``ommpp``, ``omepp``) and
        the impurity VxB rotation (``omgvb``, which must already be set or
        defaults to zero), computes:

        - ``omgeb = omgvb + omgpp``  (ExB rotation)
        - ``ommvb = omgeb - ommpp``  (main ion VxB)
        - ``omevb = omgeb - omepp``  (electron VxB)

        If equilibrium data (*R*, *Bp*, *Bt*, *psi*) are provided, also
        computes:

        - ``er = omgeb * R * Bp``  (radial electric field, kV/m)
        - ``omghb = (R*Bp)^2/Bt * d(omgeb)/dpsi``  (Hahm-Burrell rate)

        Parameters
        ----------
        R, Bp, Bt : array-like or None
            Midplane major radius [m], poloidal field [T], and toroidal
            field [T] on the profile psinorm grid.
        psi : array-like or None
            Poloidal flux in SI (Weber).
        """
        psinorm = self._raw["omgpp"]["psinorm"]

        omgvb = self._get_data("omgvb")
        if omgvb is None:
            omgvb = np.zeros_like(psinorm)

        omgpp = self._raw["omgpp"]["data"]
        ommpp = self._raw["ommpp"]["data"]
        omepp = self._raw["omepp"]["data"]

        omgeb = omgvb + omgpp
        ommvb = omgeb - ommpp
        omevb = omgeb - omepp

        self.set_profile("omgeb", psinorm, omgeb)
        self.set_profile("ommvb", psinorm, ommvb)
        self.set_profile("omevb", psinorm, omevb)

        if R is not None and Bp is not None:
            R = np.asarray(R, dtype=float)
            Bp = np.asarray(Bp, dtype=float)
            er = omgeb * R * Bp
            self.set_profile("er", psinorm, er)

            if Bt is not None and psi is not None:
                Bt = np.asarray(Bt, dtype=float)
                psi = np.asarray(psi, dtype=float)
                Bt_safe = np.where(np.abs(Bt) > 1e-6, Bt,
                                   np.sign(Bt) * 1e-6)
                # The Hahm-Burrell shearing rate is effectively a
                # second derivative of the kinetic profiles
                # (omgeb ~ d(nT)/dpsi, omghb ~ d(omgeb)/dpsi), so
                # grid-scale noise is amplified enormously.  We use a
                # Savitzky-Golay filter to compute the derivative in
                # one step: it fits a local polynomial and analytically
                # differentiates it, which is mathematically optimal for
                # noisy data.  The window is ~3% of the grid (minimum
                # 7 points), wide enough to suppress two-grid-point
                # oscillations while preserving the pedestal gradient.
                from scipy.signal import savgol_filter
                n_pts = len(omgeb)
                win = max(7, int(np.round(n_pts * 0.03)) | 1)
                if win >= n_pts:
                    win = n_pts - (1 - n_pts % 2)
                delta_psi = np.abs(np.median(np.diff(psi)))
                if delta_psi == 0:
                    delta_psi = 1.0  # fallback for degenerate grids
                domgeb_dpsi = savgol_filter(
                    omgeb, win, min(3, win - 1),
                    deriv=1, delta=delta_psi,
                )
                omghb = (R * Bp) ** 2 / Bt_safe * domgeb_dpsi
                np.nan_to_num(omghb, copy=False, nan=0.0,
                              posinf=0.0, neginf=0.0)
                self.set_profile("omghb", psinorm, omghb)

    # --- Remap ---

    def remap(self, psinorm=None, key="ne", overwrite=False, warn=False):
        """Put every profile on ONE common psinorm grid.

        An Osborne p-file stores a separate radial grid per quantity -- on real
        DIII-D files ``te``'s grid can sit ~0.2 in psi_N away from ``ne``'s --
        so consumers that pair one quantity's values with another's grid
        misplace the profile (measured: up to 13% of peak T_e, worst at the
        pedestal) and index-wise arithmetic across quantities, such as the
        quasi-neutrality ``ne - ni - nb``, is not even well posed.

        Parameters
        ----------
        psinorm : array-like, int, or None
            Target grid.  ``None`` uses *key*'s grid; an ``int`` uses
            ``np.linspace(0, 1, psinorm)``.
        key : str
            Profile whose grid to adopt when *psinorm* is ``None``.  ``"ne"``
            by default, because ``compute_quasineutrality`` / ``compute_zeff``
            are written against the electron grid.  Note this resamples finer
            grids DOWN: pass an explicit grid (or an int) to keep edge
            resolution that only one channel carries.
        overwrite : bool
            Remap this object in place (and return it).  Otherwise a new
            :class:`PFile` is returned and this one is untouched.
        warn : bool
            Emit a :class:`UserWarning` naming the channels whose native grid
            actually differed, so an interpolation is never silent.

        Returns
        -------
        PFile
            The remapped file (``self`` when *overwrite*).

        Notes
        -----
        Two documented limits, both measured on real DIII-D files and both
        judged acceptable rather than overlooked:

        * **Derivative consistency.**  The stored ``d(profile)/dpsiN`` column is
          interpolated alongside the data rather than recomputed from it, so
          after a remap the two are internally consistent only to ~13 % of peak
          (vs ~1.4 % natively).  Accuracy against a PCHIP reference is a wash
          either way (2.4 % interpolating the stored column, 3.3 % recomputing
          it), the source column is itself only ~1.7 % consistent with its own
          data, and nothing in bouquet reads the derivative -- but a downstream
          consumer of a written p-file should differentiate the data rather
          than trust this column across a remap.
        * **Range.**  Channels do not share endpoints (``te`` stops at
          psi_N 1.096 where ``ne`` reaches 1.125), so target points beyond a
          channel's own data are held at its boundary value -- 14 of 256 points
          on that file, all in the SOL.  The equilibrium never sees them: the
          reconstruction interpolates kinetics onto the g-file's psi_N grid,
          which ends at 1.0.
        """
        if psinorm is None:
            if key not in self._raw:
                raise KeyError(f"Profile {key!r} not found for grid reference")
            target = np.asarray(self._raw[key]["psinorm"], dtype=float)
        elif isinstance(psinorm, (int, np.integer)):
            target = np.linspace(0, 1, int(psinorm))
        else:
            target = np.asarray(psinorm, dtype=float)

        if warn:
            # Compare against a channel that EXISTS: when an explicit target
            # grid is supplied, `key` may be absent (or irrelevant), and
            # defaulting to a missing "ne" would silence the warning on a file
            # whose native grids genuinely differ.
            ref_key = key if (psinorm is None or key in self._raw) else next(
                (k for k in self._raw if k != "N Z A"), "ne")
            spread = self.native_grid_spread(ref_key)
            if spread:
                worst = sorted(spread.items(), key=lambda kv: -kv[1])[:4]
                warnings.warn(
                    "p-file carries a different psinorm grid per quantity; "
                    "interpolating all profiles onto "
                    + (f"{key!r}'s grid" if psinorm is None else "the supplied grid")
                    + ". Largest native offsets: "
                    + ", ".join(f"{k} {d:.3g}" for k, d in worst)
                    + ". Native grids remain available via "
                    "psinorm_for(key, native=True); pass remap=False to keep "
                    "the profiles as read.",
                    UserWarning, stacklevel=3,
                )

        new_raw = OrderedDict()
        for k, val in self._raw.items():
            if k == "N Z A":
                new_raw[k] = val.copy()
                continue

            # Use boundary values for out-of-range points instead of
            # linear extrapolation, which can produce unphysical values
            # (e.g. negative densities) at the edge.
            f_data = interpolate.interp1d(
                val["psinorm"], val["data"],
                kind="linear", bounds_error=False,
                fill_value=(val["data"][0], val["data"][-1]),
            )
            f_deriv = interpolate.interp1d(
                val["psinorm"], val["derivative"],
                kind="linear", bounds_error=False,
                fill_value=(val["derivative"][0], val["derivative"][-1]),
            )
            new_raw[k] = {
                "psinorm": target.copy(),
                "data": f_data(target),
                "derivative": f_deriv(target),
                "units": val.get("units", ""),
                "deriv_label": val.get("deriv_label", f"d{k}/dpsiN"),
            }

        if overwrite:
            self._raw = new_raw
            self._remapped_to = target.copy()
            return self

        # A detached copy: carry the SAME native profiles and edit state, so
        # the returned object is a first-class PFile (round-trip, native grid
        # reporting) rather than a bare shell.
        obj = object.__new__(PFile)
        obj._raw = new_raw
        obj._native = (None if self._native is None
                       else _deepcopy_raw(self._native))
        obj._modified = set(self._modified)
        obj._remapped_to = target.copy()
        return obj

    def __repr__(self):
        profile_keys = [k for k in self._raw if k != "N Z A"]
        return (
            f"PFile({len(profile_keys)} profiles: "
            f"{', '.join(profile_keys[:6])}"
            f"{'...' if len(profile_keys) > 6 else ''})"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def read_pfile(filename, remap=True, remap_key="ne"):
    """Read an Osborne p-file and return a :class:`PFile` object.

    Parameters
    ----------
    filename : str or path-like
        Path to the p-file.

    Returns
    -------
    PFile
    """
    return PFile(filename, remap=remap, remap_key=remap_key)
