"""``utils.capture_xpoints`` must own its data.

``TokaMaker.get_xpoints()`` returns a *view* onto the Fortran ``gs_equil``
struct, and every ``copy_eq``/``replace_eq`` swap bouquet makes (in
``safe_trace_surf``, ``safe_save_eqdsk``, and once per traced surface inside
``capture_equilibrium_fsa``) frees the struct that view points at.  A captured
view therefore reads freed heap by the time the draw is archived -- which is
how two same-seed runs archived different ``x_points`` datasets.

No solver here: a fake ``gs`` stands in for TokaMaker, returning a view into a
buffer the test then mutates (the cheapest faithful model of the free).
"""
import numpy as np
import pytest

from bouquet.utils import capture_xpoints


class FakeGS:
    """Minimal stand-in for TokaMaker.

    ``buf`` models ``gs_equil%x_points``; ``get_xpoints`` reproduces the
    wrapper's contract exactly -- ``None`` when row 0 has ``R < 0``, otherwise
    a slice *view* up to the first ``R < 0`` sentinel row.
    """

    def __init__(self, rows, nrows=20, diverted=True):
        self.buf = np.zeros((nrows, 2), dtype=np.float64)
        self.buf[:, 0] = -1.0                      # the Fortran sentinel
        for i, (r, z) in enumerate(rows):
            self.buf[i, :] = [r, z]
        self._diverted = diverted

    def get_xpoints(self):
        if self.buf[0, 0] < 0.0:
            return None, False
        for i in range(self.buf.shape[0]):
            if self.buf[i, 0] < 0.0:
                break
        return self.buf[:i, :], self._diverted   # a VIEW, as OFT does


def test_capture_owns_its_buffer():
    gs = FakeGS([(1.27, -1.14), (0.96, 1.17)])
    xp, div = capture_xpoints(gs)
    assert xp is not None and xp.shape == (2, 2)
    assert div is True
    # Guard the guard: the fake really does hand out a view, so a capture that
    # merely did np.asarray() would alias it.
    raw, _ = gs.get_xpoints()
    assert np.shares_memory(raw, gs.buf)
    assert np.asarray(raw, dtype=float).base is not None
    # The capture must not.
    assert not np.shares_memory(xp, gs.buf)
    assert xp.flags["OWNDATA"] and xp.base is None


def test_capture_survives_the_buffer_being_overwritten():
    """This is the bug: the eq swap frees the struct, and the next allocation
    writes over it.  Overwriting ``buf`` models that."""
    gs = FakeGS([(1.27, -1.14)])
    xp, _ = capture_xpoints(gs)
    before = xp.copy()

    # what a later copy_eq/replace_eq + heap reuse does to the freed block
    gs.buf[:, :] = 5.2580877651087e-310
    gs.buf[0, :] = [4.74e-322, 0.0]
    assert np.array_equal(xp, before), \
        "captured X-points changed when the Fortran buffer was overwritten"
    assert np.array_equal(xp, np.array([[1.27, -1.14]]))


def test_none_when_no_xpoints():
    gs = FakeGS([])                                # row 0 keeps R = -1
    xp, div = capture_xpoints(gs)
    assert xp is None
    assert div is False


def test_sentinel_and_nonfinite_rows_are_dropped_with_a_warning():
    gs = FakeGS([(1.27, -1.14)])
    # A wrapper that failed to stop at the sentinel would hand these back.
    gs.get_xpoints = lambda: (np.array([[1.27, -1.14],
                                        [-1.0, 0.0],
                                        [0.0, 0.0],
                                        [np.nan, 1.0]]), True)
    with pytest.warns(RuntimeWarning, match="dropped 3 of 4"):
        xp, _ = capture_xpoints(gs)
    assert np.array_equal(xp, np.array([[1.27, -1.14]]))


def test_small_positive_R_is_kept_not_masked():
    """A denormal R is garbage, but no threshold can prove that without
    inventing a geometry bound -- so it is kept, and the missing-sentinel
    warning below is what flags it."""
    gs = FakeGS([(1.27, -1.14)])
    gs.get_xpoints = lambda: (np.array([[5.2580877651087e-310, 0.0]]), False)
    xp, _ = capture_xpoints(gs)
    assert xp is not None and xp.shape == (1, 2)
    assert xp[0, 0] == 5.2580877651087e-310


def test_full_buffer_return_warns_about_the_missing_sentinel():
    """19 rows out of a 20-row buffer is the wrapper walking off the end: no
    equilibrium has 19 X-points."""
    gs = FakeGS([])
    gs.get_xpoints = lambda: (np.full((19, 2), 1e-310), True)
    with pytest.warns(RuntimeWarning, match="sentinel"):
        xp, _ = capture_xpoints(gs)
    assert xp is not None and xp.shape == (19, 2)   # returned, not discarded


def test_all_rows_dropped_reports_no_xpoints():
    gs = FakeGS([])
    gs.get_xpoints = lambda: (np.array([[0.0, 0.0], [0.0, 0.0]]), False)
    with pytest.warns(RuntimeWarning):
        xp, div = capture_xpoints(gs)
    assert xp is None
    assert div is False


def test_get_xpoints_failure_is_reported_not_raised():
    class Boom:
        def get_xpoints(self):
            raise RuntimeError("no equilibrium")

    with pytest.warns(RuntimeWarning, match="get_xpoints\\(\\) failed"):
        xp, div = capture_xpoints(Boom())
    assert xp is None and div is None
