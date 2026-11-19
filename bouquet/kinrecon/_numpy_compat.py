"""Numpy version compatibility: the package runs on numpy 1 and numpy 2.

numpy 2 renamed the trapezoid-rule integrator ``trapz`` to ``trapezoid``; numpy 1 has only ``trapz``.  The package
calls :func:`trapezoid`, which is numpy's own function under whichever name the installed numpy provides, so the
results do not change.
"""
import numpy as np

__all__ = ["pick_trapezoid", "trapezoid"]


def pick_trapezoid(module):
    """The trapezoid-rule integrator of a numpy-like ``module``: ``trapezoid`` (numpy 2) when it has one, else
    ``trapz`` (numpy 1)."""
    f = getattr(module, "trapezoid", None)
    return f if f is not None else getattr(module, "trapz")


#: numpy's trapezoid-rule integrator (``numpy.trapezoid`` on numpy 2, ``numpy.trapz`` on numpy 1).
trapezoid = pick_trapezoid(np)
