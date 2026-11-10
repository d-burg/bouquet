"""Kinetic equilibrium reconstruction on TokaMaker (self-contained; see ``interfaces``).

``score`` and ``chi2_summary`` are re-exported from :mod:`.score`.  Importing
the package needs only numpy: :mod:`.score` and :mod:`.diagnostics` call
TokaMaker through the equilibrium object they are given and never import the
Open FUSION Toolkit themselves.  Note that the re-exported function shadows
the submodule as an attribute (``bouquet.kinrecon.score`` is the function);
``from bouquet.kinrecon.score import ...`` still reaches the module.
"""

from .interfaces import ConstraintGroup, MeasurementSet, ResidualTable
from .score import chi2_summary, score

__all__ = ["ConstraintGroup", "MeasurementSet", "ResidualTable", "chi2_summary", "score"]
