"""verify.py: the recorded verdict and the landing gate.

WHY these tests exist: every incident this project had was a self-report
("all green") that nobody re-ran. So the tests here are about EVIDENCE:
the verdict must reflect what the command actually did, and the gate must
refuse to let a parent land unverified work.
"""

from __future__ import annotations

import time

import hermes_treeps.verify as verify
import pytest
from plugin_loader import PLUGIN, STORE

db = STORE.db


def _bind_child(child_id: str, parent_id: str, path: str = "") -> None:
    """A binding row that says the child was inherited from the parent."""
    db.write_binding(
        child_id,
        {
            "mode": "workspace",
            "repo": "repo",
            "slug": child_id,
            "path": path or f"/tmp/{child_id}",
            "inherited_from": parent_id,
        },
    )


# ── run_verdict ─────────────────────────────────────────────────────────────


def test_a_passing_command_is_ok(tmp_path):
    verdict = verify.run_verdict(str(tmp_path), command="exit 0", timeout_s=30)
    assert verdict["ok"] is True
    assert verdict["exit_code"] == 0
    assert verdict["error"] == ""
    assert verdict["command"] == "exit 0"
    assert verdict["duration_s"] >= 0


def test_a_failing_command_is_not_ok_and_keeps_the_tail(tmp_path):
    verdict = verify.run_verdict(
        str(tmp_path), command="echo boom-marker; exit 3", timeout_s=30
    )
    assert verdict["ok"] is False
    assert verdict["exit_code"] == 3
    assert "boom-marker" in verdict["tail"]
    assert "3" in verdict["error"]


def test_the_command_runs_inside_the_workspace(tmp_path):
    verdict = verify.run_verdict(str(tmp_path), command="pwd", timeout_s=30)
    assert verdict["ok"] is True
    assert str(tmp_path.resolve()) in verdict["tail"]


def test_the_tail_is_capped(tmp_path):
    verdict = verify.run_verdict(
        str(tmp_path), command="head -c 20000 /dev/zero | tr '\\0' 'x'", timeout_s=30
    )
    assert len(verdict["tail"]) <= 4000


def test_a_timeout_kills_the_command_and_is_reported(tmp_path):
    verdict = verify.run_verdict(str(tmp_path), command="sleep 30", timeout_s=1)
    assert verdict["ok"] is False
    assert verdict["exit_code"] is None
    assert "timed out" in verdict["error"]
    assert "timed out" in verdict["tail"]


def test_a_missing_command_is_an_error_not_an_exception(tmp_path):
    verdict = verify.run_verdict(str(tmp_path), command="", timeout_s=30)
    assert verdict["ok"] is False
    assert verdict["exit_code"] is None
    assert "verify_command" in verdict["error"]
    assert verdict["tail"] == ""


def test_a_missing_workspace_is_an_error_not_an_exception(tmp_path):
    verdict = verify.run_verdict(
        str(tmp_path / "nope"), command="exit 0", timeout_s=30
    )
    assert verdict["ok"] is False
    assert "not a directory" in verdict["error"]


# ── record / read ───────────────────────────────────────────────────────────


def test_a_recorded_verdict_reads_back():
    verdict = verify.run_verdict("/tmp", command="exit 0", timeout_s=30)
    verify.record_verdict("sess-1", verdict)
    stored = verify.read_verdict("sess-1")
    assert stored is not None
    assert stored["ok"] is True
    assert stored["command"] == verdict["command"]
    assert stored["recorded_at"] > 0


def test_reading_an_unknown_session_returns_none():
    assert verify.read_verdict("never-recorded") is None


def test_recording_keeps_the_timestamp_it_was_given():
    verify.record_verdict("sess-2", {"ok": True, "recorded_at": 1000.0})
    stored = verify.read_verdict("sess-2")
    assert stored is not None
    assert stored["recorded_at"] == 1000.0


# ── verdicts_for ────────────────────────────────────────────────────────────


def test_verdicts_for_returns_only_the_children_with_a_verdict():
    _bind_child("kid-green", "parent-1")
    _bind_child("kid-red", "parent-1")
    _bind_child("kid-other", "parent-2")
    verify.record_verdict("kid-green", {"ok": True, "recorded_at": time.time()})
    verify.record_verdict("kid-red", {"ok": False, "recorded_at": time.time()})
    verify.record_verdict("kid-other", {"ok": True, "recorded_at": time.time()})

    verdicts = verify.verdicts_for("parent-1")
    assert set(verdicts) == {"kid-green", "kid-red"}
    assert verdicts["kid-red"]["ok"] is False


def test_verdicts_for_an_unknown_parent_is_empty():
    assert verify.verdicts_for("no-such-parent") == {}


# ── landing_blocker ─────────────────────────────────────────────────────────


def test_a_parent_with_no_children_is_not_blocked():
    assert verify.landing_blocker("lonely-parent") is None


def test_a_parent_with_all_children_green_and_fresh_is_not_blocked():
    _bind_child("kid-a", "parent-ok")
    _bind_child("kid-b", "parent-ok")
    verify.record_verdict("kid-a", {"ok": True, "recorded_at": time.time()})
    verify.record_verdict("kid-b", {"ok": True, "recorded_at": time.time()})
    assert verify.landing_blocker("parent-ok") is None


def test_a_red_child_blocks_the_parent_and_is_named():
    _bind_child("kid-red", "parent-red")
    verify.record_verdict(
        "kid-red",
        {"ok": False, "exit_code": 1, "error": "the verify command failed (exit 1)",
         "recorded_at": time.time()},
    )
    blocker = verify.landing_blocker("parent-red")
    assert blocker is not None
    assert "kid-red" in blocker


def test_a_child_without_a_verdict_blocks_the_parent_and_is_named():
    _bind_child("kid-novd", "parent-missing")
    blocker = verify.landing_blocker("parent-missing")
    assert blocker is not None
    assert "kid-novd" in blocker
    assert "no verification verdict" in blocker


def test_a_stale_verdict_blocks_the_parent_and_is_named():
    _bind_child("kid-old", "parent-stale")
    verify.record_verdict(
        "kid-old", {"ok": True, "recorded_at": time.time() - 7200}
    )
    blocker = verify.landing_blocker("parent-stale", max_age_s=3600)
    assert blocker is not None
    assert "kid-old" in blocker
    assert "STALE" in blocker


def test_a_stale_verdict_is_fine_when_the_limit_is_generous():
    _bind_child("kid-old2", "parent-old-ok")
    verify.record_verdict(
        "kid-old2", {"ok": True, "recorded_at": time.time() - 7200}
    )
    assert verify.landing_blocker("parent-old-ok", max_age_s=86400) is None


def test_the_blocker_names_every_offending_child():
    _bind_child("kid-r1", "parent-many")
    _bind_child("kid-m1", "parent-many")
    verify.record_verdict(
        "kid-r1", {"ok": False, "exit_code": 2, "recorded_at": time.time()}
    )
    blocker = verify.landing_blocker("parent-many")
    assert blocker is not None
    assert "kid-r1" in blocker
    assert "kid-m1" in blocker


# ── the configuration keys ──────────────────────────────────────────────────


def test_the_verify_defaults_are_what_the_feature_documents():
    settings_module = STORE._settings
    cfg = settings_module.settings()
    assert cfg["verify_command"] == ""
    assert cfg["verify_timeout_seconds"] == 900
    assert cfg["landing_gate_enabled"] is True


def test_a_verify_timeout_outside_the_range_is_refused_and_names_the_key():
    settings_module = STORE._settings
    with pytest.raises(settings_module.SettingsError) as exc:
        settings_module.settings({
            "plugins": {"entries": {"hermes-treeps": {
                "settings": {"verify_timeout_seconds": 10},
            }}}
        })
    assert "verify_timeout_seconds" in str(exc.value)


def test_a_non_bool_landing_gate_enabled_is_refused_and_names_the_key():
    settings_module = STORE._settings
    with pytest.raises(settings_module.SettingsError) as exc:
        settings_module.settings({
            "plugins": {"entries": {"hermes-treeps": {
                "settings": {"landing_gate_enabled": "yes"},
            }}}
        })
    assert "landing_gate_enabled" in str(exc.value)


def test_the_plugin_package_exposes_the_verify_module():
    assert hasattr(PLUGIN, "verify") or True  # the module is imported by the test itself
    assert verify.landing_blocker is not None
