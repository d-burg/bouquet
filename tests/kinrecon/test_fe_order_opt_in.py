"""The opt-in finite-element order :attr:`bouquet.kinrecon.forward.ForwardSettings.fe_order`, without OFT.

* Default 2, left out of ``ForwardSettings.as_dict()`` (records made with the default are unchanged).
* 3 or 4 recorded in ``as_dict()``; only 2, 3, 4 accepted (:func:`forward.check_fe_order`).
* The forward model passes the checked order to the solver's ``setup_model`` (source check of the one call).
"""
import inspect
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon import forward as FW  # noqa: E402


def test_default_is_two_and_not_recorded():
    s = FW.ForwardSettings(vsc={"A": 1.0})
    assert s.fe_order == 2 == FW.FE_ORDER_DEFAULT
    assert "fe_order" not in s.as_dict()


@pytest.mark.parametrize("order", [3, 4])
def test_opt_in_is_recorded(order):
    assert FW.ForwardSettings(vsc={"A": 1.0}, fe_order=order).as_dict()["fe_order"] == order


@pytest.mark.parametrize("value, expected", [(2, 2), (3, 3), (4, 4), (3.0, 3), (np.int64(4), 4)])
def test_check_accepts(value, expected):
    assert FW.check_fe_order(value) == expected


@pytest.mark.parametrize("value", [None, 0, 1, 5, 2.5, float("nan"), True, "3", b"3", [3]])
def test_check_refuses(value):
    with pytest.raises(ValueError):
        FW.check_fe_order(value)


def test_bad_order_refused_in_as_dict():
    with pytest.raises(ValueError):
        FW.ForwardSettings(vsc={"A": 1.0}, fe_order=7).as_dict()


def test_setup_model_receives_the_checked_order():
    src = inspect.getsource(FW.ForwardModel.__init__)
    assert "fe_order = check_fe_order(s.fe_order)" in src
    assert "order=fe_order" in src and "order=2" not in src
