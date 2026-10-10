"""The build-keyed golden policy (``tests/golden/README.md``, "Build-specific
goldens"), on mocked build records: no OFT, no fixture, no solve.

A golden comparison always runs at its own bar.  The installed OFT build is
compared with the one stamped in the fixture (``provenance.oft``,
``library_sha256`` primary, ``build_id`` informational), and only the
reporting depends on it: a pass on a different build warns, naming both
builds; a failure on a different build fails with the mismatch block first.
A fixture with no library digest is "unstamped", a mismatch on every build.
"""
import json
import warnings

import pytest

import _harness

_SHA_A = "a" * 8 + "1" * 56
_SHA_B = "b" * 8 + "2" * 56


def _manifest(build_id="fixture_build", sha=_SHA_A):
    oft = {"available": True, "build_id": build_id}
    if sha is not None:
        oft["library_sha256"] = sha
    return {"provenance": {"oft": oft}}


def _installed(build_id="installed_build", sha=_SHA_A):
    return {"build_id": build_id, "library_sha256": sha}


def _check(passes, manifest, installed, **kw):
    """Run one comparison under the guard; *passes* picks its outcome."""
    with _harness.golden_comparison(
            manifest, regenerate=_harness.REGENERATE_LEGACY_GOLDEN,
            installed=installed, **kw):
        assert passes, "coil drift 1.4543 % >= 0.3 %"


# ---------------------------------------------------------------------------
#  golden_build_check
# ---------------------------------------------------------------------------
def test_the_library_digest_decides_a_match():
    m = _harness.golden_build_check(_manifest(), installed=_installed())
    assert m.matches and m.stamped
    # the stated names differ: informational, it does not decide the match
    assert m.build_id_matches is False


def test_a_different_library_is_a_mismatch_whatever_the_names_say():
    m = _harness.golden_build_check(
        _manifest(build_id="same"), installed=_installed("same", _SHA_B))
    assert not m.matches and m.stamped
    assert m.build_id_matches is True
    assert m.describe() == ("fixture generated with same/aaaaaaaa, "
                            "installed same/bbbbbbbb")


def test_a_fixture_without_a_library_digest_is_unstamped():
    m = _harness.golden_build_check(_manifest(sha=None),
                                    installed=_installed())
    assert not m.matches and not m.stamped
    assert "fixture generated with fixture_build/unstamped" in m.describe()
    m = _harness.golden_build_check({}, installed=_installed())
    assert not m.matches and not m.stamped
    assert "no build id/unstamped" in m.describe()


def test_an_installed_build_that_cannot_be_measured_is_a_mismatch():
    m = _harness.golden_build_check(_manifest(), installed=_installed(sha=None))
    assert not m.matches and m.stamped
    assert "installed installed_build/unavailable" in m.describe()


def test_the_manifest_can_be_given_as_a_path(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(_manifest()))
    assert _harness.golden_build_check(str(p), installed=_installed()).matches


def test_the_installed_build_names_the_stated_id_and_the_measured_digest(
        monkeypatch):
    from bouquet import jbs_loop
    monkeypatch.setitem(jbs_loop._OFT_BUILD_CACHE, "info", {
        "version": "1.0", "git_hash": None, "library_sha256": _SHA_B,
        "sources_sha256": None, "build_id": "OpenFUSIONToolkit 1.0 lib x"})
    monkeypatch.delenv("BOUQUET_OFT_BUILD_ID", raising=False)
    assert _harness.installed_oft_build() == {
        "build_id": "OpenFUSIONToolkit 1.0 lib x", "library_sha256": _SHA_B}
    monkeypatch.setenv("BOUQUET_OFT_BUILD_ID", "stated_build")
    assert _harness.installed_oft_build() == {
        "build_id": "stated_build", "library_sha256": _SHA_B}


# ---------------------------------------------------------------------------
#  golden_comparison: the four outcomes
# ---------------------------------------------------------------------------
def test_match_and_pass_passes_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(True, _manifest(), _installed())


def test_match_and_fail_fails_unchanged():
    with pytest.raises(AssertionError) as ei:
        _check(False, _manifest(), _installed())
    assert "OFT build mismatch" not in str(ei.value)
    assert "coil drift 1.4543 % >= 0.3 %" in str(ei.value)


def test_mismatch_and_pass_passes_with_a_warning_naming_both_builds():
    with pytest.warns(_harness.GoldenBuildMismatchWarning) as rec:
        _check(True, _manifest(), _installed(sha=_SHA_B))
    msg = str(rec[0].message)
    assert "fixture generated with fixture_build/aaaaaaaa" in msg
    assert "installed installed_build/bbbbbbbb" in msg


def test_mismatch_and_fail_fails_with_the_block_first():
    with pytest.raises(AssertionError) as ei:
        _check(False, _manifest(), _installed(sha=_SHA_B))
    msg = str(ei.value)
    assert msg.startswith(
        "OFT build mismatch: fixture generated with fixture_build/aaaaaaaa, "
        "installed installed_build/bbbbbbbb; goldens are build-specific "
        "(edge FF' from per-node <R> moved the lower coils ~1.4 % between "
        "builds); regenerate with:\n")
    for line in _harness.REGENERATE_LEGACY_GOLDEN.splitlines():
        assert line in msg
    assert "and review the diff before committing." in msg
    # ... then the original comparison failure, unchanged
    assert msg.index("review the diff") < msg.index(
        "coil drift 1.4543 % >= 0.3 %")
    assert "reads the stored fixture only" not in msg


def test_an_unstamped_fixture_fails_with_unstamped_wording():
    with pytest.raises(AssertionError) as ei:
        _check(False, _manifest(sha=None), _installed())
    msg = str(ei.value)
    assert msg.startswith("OFT build mismatch (the fixture is unstamped")
    assert "fixture generated with fixture_build/unstamped" in msg
    with pytest.warns(_harness.GoldenBuildMismatchWarning, match="unstamped"):
        _check(True, _manifest(sha=None), _installed())


def test_a_fixture_only_comparison_says_so_on_a_mismatch():
    with pytest.raises(AssertionError) as ei:
        _check(False, _manifest(), _installed(sha=_SHA_B), fixture_only=True)
    assert "reads the stored fixture only" in str(ei.value)


def test_an_error_that_is_not_a_comparison_passes_through_unchanged():
    with pytest.raises(KeyError):
        with _harness.golden_comparison(
                _manifest(), regenerate="x", installed=_installed(sha=_SHA_B)):
            raise KeyError("mode1")


def test_the_committed_fixtures_are_stamped():
    """Both committed goldens carry the digest the policy keys on."""
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
    for name in ("golden_manifest.json", "D3Dlike_Hmode_legacy_golden.json"):
        path = os.path.join(here, name)
        if not os.path.isfile(path):
            pytest.skip(f"{name} absent")
        assert _harness.golden_build_check(path, installed={}).stamped, name
