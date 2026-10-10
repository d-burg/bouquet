"""The quarantined IMAS-path fixtures (tests/data/README.md, owner decision
D3): importing a test module that reads them warns, on every run, that they
are owner-flagged as possibly derived from real device data and must be
replaced by the synthetic set specified in tests/data/FIXTURE_SPEC.md before
merge."""
import os
import warnings

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
#: The quarantined files (left unedited until the owner's synthetic set lands).
QUARANTINED = ("dd_synthetic.json.gz", "diiid_profs_synthetic.cdf",
               "g_synthetic.geqdsk")


class QuarantinedFixtureWarning(UserWarning):
    """A test reads a fixture quarantined pending replacement (D3)."""


def warn_quarantined(module):
    """Emit :class:`QuarantinedFixtureWarning` naming *module* and the
    quarantined files that are present."""
    present = [f for f in QUARANTINED if os.path.isfile(os.path.join(_DATA, f))]
    if present:
        warnings.warn(
            f"{module} reads QUARANTINED fixtures {present}: owner-flagged as "
            "possibly derived from real device data; they MUST be replaced "
            "by the synthetic set of tests/data/FIXTURE_SPEC.md before merge "
            "(tests/data/README.md)", QuarantinedFixtureWarning, stacklevel=2)
    return present
