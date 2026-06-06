"""Tests for scripts/promote.py — all four subcommands, all edge cases.

Run: pytest tests/test_promote.py -v

All tests mock the MLflow client and redirect LOG_FILE to a temp path so no
live tracking server or repo-root file is needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ns(**kwargs) -> argparse.Namespace:
    """Build a parsed-args namespace the way argparse would."""
    return argparse.Namespace(name="travel-assistant", **kwargs)


def make_mv(version: int, config_id: str, run_id: str = "run-abc", **extra_tags) -> MagicMock:
    """Fake MLflow ModelVersion."""
    mv = MagicMock()
    mv.version = str(version)
    mv.run_id = run_id
    mv.tags = {"config_id": config_id, **extra_tags}
    return mv


def make_run(metrics: dict) -> MagicMock:
    """Fake MLflow Run."""
    run = MagicMock()
    run.data.metrics = metrics
    return run


try:
    from mlflow.exceptions import MlflowException, RestException

    def alias_not_found() -> RestException:
        return RestException({"message": "Alias not found", "error_code": "RESOURCE_DOES_NOT_EXIST"})
except Exception:
    def alias_not_found() -> Exception:  # type: ignore[misc]
        return Exception("Alias not found")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    """Redirect LOG_FILE to a temp file so tests never touch the repo root."""
    log = tmp_path / "promotion-log.jsonl"
    import scripts.promote as m
    monkeypatch.setattr(m, "LOG_FILE", log)
    return log


@pytest.fixture
def mock_client(monkeypatch):
    """Return a MagicMock wired as MlflowClient regardless of instantiation style.

    Handles three patterns:
      - module-level:    client = mlflow.tracking.MlflowClient()
      - from-import:     from mlflow.tracking import MlflowClient; MlflowClient()
      - module-level var: already assigned to `scripts.promote.client`
    """
    mock = MagicMock()
    import mlflow.tracking
    import scripts.promote as m

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda *a, **kw: mock)
    if hasattr(m, "MlflowClient"):
        monkeypatch.setattr(m, "MlflowClient", lambda *a, **kw: mock)
    # raising=False: create the attr if absent, restore/remove after test
    monkeypatch.setattr(m, "client", mock, raising=False)
    yield mock


# ---------------------------------------------------------------------------
# cmd_list
# ---------------------------------------------------------------------------

class TestCmdList:
    def test_no_aliases_prints_message(self, mock_client, capsys):
        from scripts.promote import cmd_list

        mock_client.get_registered_model.return_value.aliases = {}
        cmd_list(ns())
        assert "no aliases set" in capsys.readouterr().out

    def test_model_not_yet_registered_prints_mlflow_exception(self, mock_client, capsys):
        from scripts.promote import cmd_list

        mock_client.get_registered_model.side_effect = MlflowException("not found")
        cmd_list(ns())
        out = capsys.readouterr().out
        assert "MlflowException" in out
        assert "not found" in out

    def test_single_alias_shows_alias_and_config_id(self, mock_client, capsys):
        from scripts.promote import cmd_list

        mock_client.get_registered_model.return_value.aliases = {"production": "12"}
        mock_client.get_model_version_by_alias.return_value = make_mv(12, "v5")
        cmd_list(ns())
        out = capsys.readouterr().out
        assert "production" in out
        assert "v5" in out

    def test_multiple_aliases_all_appear(self, mock_client, capsys):
        from scripts.promote import cmd_list

        mock_client.get_registered_model.return_value.aliases = {
            "production": "12",
            "staging": "14",
        }
        mock_client.get_model_version_by_alias.side_effect = lambda name, alias: {
            "production": make_mv(12, "v5"),
            "staging": make_mv(14, "v6"),
        }[alias]
        cmd_list(ns())
        out = capsys.readouterr().out
        assert "production" in out and "v5" in out
        assert "staging" in out and "v6" in out


# ---------------------------------------------------------------------------
# cmd_show
# ---------------------------------------------------------------------------

class TestCmdShow:
    def test_shows_config_id_and_key_metrics(self, mock_client, capsys):
        from scripts.promote import cmd_show

        mv = make_mv(12, "v5", run_id="run-xyz", model="gpt-4o")
        mock_client.get_model_version_by_alias.return_value = mv
        mock_client.get_run.return_value = make_run({
            "accuracy_overall": 0.91,
            "verdict_rate_leaked": 0.04,
            "total_cost_usd": 0.38,
        })
        cmd_show(ns(alias="production"))
        out = capsys.readouterr().out
        assert "v5" in out
        assert "0.91" in out

    def test_unset_alias_exits_1(self, mock_client, capsys):
        from scripts.promote import cmd_show

        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        with pytest.raises(SystemExit) as exc:
            cmd_show(ns(alias="production"))
        assert exc.value.code == 1

    def test_unset_alias_prints_error_message(self, mock_client, capsys):
        from scripts.promote import cmd_show

        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        with pytest.raises(SystemExit):
            cmd_show(ns(alias="production"))
        combined = capsys.readouterr().out + capsys.readouterr().err
        # Some message referencing the alias or "unset" must appear
        assert len(combined.strip()) > 0


# ---------------------------------------------------------------------------
# cmd_set
# ---------------------------------------------------------------------------

class TestCmdSet:
    def test_zero_matches_exits_1(self, mock_client, tmp_log):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = []
        with pytest.raises(SystemExit) as exc:
            cmd_set(ns(alias="production", config_id="v99"))
        assert exc.value.code == 1

    def test_zero_matches_prints_config_id_in_error(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = []
        with pytest.raises(SystemExit):
            cmd_set(ns(alias="production", config_id="v99"))
        assert "v99" in capsys.readouterr().out

    def test_first_promotion_prints_unset_arrow_config_id(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = [make_mv(12, "v5")]
        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        cmd_set(ns(alias="production", config_id="v5"))
        out = capsys.readouterr().out
        assert "(unset)" in out
        assert "v5" in out

    def test_subsequent_promotion_prints_old_and_new_config_id(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = [make_mv(14, "v6")]
        mock_client.get_model_version_by_alias.return_value = make_mv(12, "v5")
        cmd_set(ns(alias="production", config_id="v6"))
        out = capsys.readouterr().out
        assert "v5" in out
        assert "v6" in out

    def test_assigns_alias_to_found_version(self, mock_client, tmp_log):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = [make_mv(12, "v5")]
        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        cmd_set(ns(alias="production", config_id="v5"))
        mock_client.set_registered_model_alias.assert_called_once_with(
            "travel-assistant", "production", "12"
        )

    def test_appends_set_log_entry_with_correct_fields(self, mock_client, tmp_log):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = [make_mv(12, "v5")]
        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        cmd_set(ns(alias="production", config_id="v5"))
        lines = tmp_log.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["alias"] == "production"
        assert entry["from"] == ""
        assert entry["to"] == "v5"
        assert entry["op"] == "set"
        assert "ts" in entry

    def test_log_entry_from_field_reflects_current_config_id(self, mock_client, tmp_log):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = [make_mv(14, "v6")]
        mock_client.get_model_version_by_alias.return_value = make_mv(12, "v5")
        cmd_set(ns(alias="production", config_id="v6"))
        entry = json.loads(tmp_log.read_text().strip())
        assert entry["from"] == "v5"
        assert entry["to"] == "v6"

    def test_successive_sets_append_multiple_lines(self, mock_client, tmp_log):
        from scripts.promote import cmd_set

        # First set: unset → v5
        mock_client.search_model_versions.return_value = [make_mv(12, "v5")]
        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        cmd_set(ns(alias="production", config_id="v5"))

        # Second set: v5 → v6
        mock_client.search_model_versions.return_value = [make_mv(14, "v6")]
        mock_client.get_model_version_by_alias.side_effect = None
        mock_client.get_model_version_by_alias.return_value = make_mv(12, "v5")
        cmd_set(ns(alias="production", config_id="v6"))

        lines = tmp_log.read_text().strip().splitlines()
        assert len(lines) == 2
        e1, e2 = json.loads(lines[0]), json.loads(lines[1])
        assert e1["from"] == "" and e1["to"] == "v5"
        assert e2["from"] == "v5" and e2["to"] == "v6"

    def test_multiple_matches_prints_warning_with_all_versions(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = [make_mv(7, "v6"), make_mv(12, "v6")]
        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        cmd_set(ns(alias="production", config_id="v6"))
        out = capsys.readouterr().out
        assert "warning" in out.lower()
        assert "7" in out and "12" in out

    def test_multiple_matches_assigns_highest_version(self, mock_client, tmp_log):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = [make_mv(7, "v6"), make_mv(12, "v6")]
        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        cmd_set(ns(alias="production", config_id="v6"))
        mock_client.set_registered_model_alias.assert_called_once_with(
            "travel-assistant", "production", "12"
        )

    def test_multiple_matches_still_succeeds_and_logs(self, mock_client, tmp_log):
        from scripts.promote import cmd_set

        mock_client.search_model_versions.return_value = [make_mv(7, "v6"), make_mv(12, "v6")]
        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        cmd_set(ns(alias="production", config_id="v6"))
        entry = json.loads(tmp_log.read_text().strip())
        assert entry["to"] == "v6"
        assert entry["op"] == "set"


# ---------------------------------------------------------------------------
# cmd_rollback
# ---------------------------------------------------------------------------

class TestCmdRollback:
    def test_alias_unset_prints_nothing_to_roll_back(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        cmd_rollback(ns(alias="production"))
        assert "nothing to roll back" in capsys.readouterr().out

    def test_alias_unset_does_not_exit_nonzero(self, mock_client, tmp_log):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.side_effect = alias_not_found()
        # Should not raise SystemExit
        cmd_rollback(ns(alias="production"))

    def test_missing_log_file_prints_no_history(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(12, "v5")
        # tmp_log doesn't exist yet — no write has happened
        cmd_rollback(ns(alias="production"))
        assert "no promotion history" in capsys.readouterr().out

    def test_no_entry_for_alias_in_log_prints_no_history(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(12, "v5")
        tmp_log.write_text(
            json.dumps({"ts": "T1", "alias": "staging", "from": "", "to": "v5", "op": "set"}) + "\n"
        )
        cmd_rollback(ns(alias="production"))
        assert "no promotion history" in capsys.readouterr().out

    def test_last_entry_is_rollback_prints_already_rolled_back(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(12, "v5")
        tmp_log.write_text(
            json.dumps({"ts": "T1", "alias": "production", "from": "v6", "to": "v5", "op": "rollback"}) + "\n"
        )
        cmd_rollback(ns(alias="production"))
        out = capsys.readouterr().out
        # "was just rolled back; no further history to walk back to"
        assert "rolled back" in out
        assert "no further history" in out

    def test_first_ever_set_with_empty_from_prints_no_previous(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(12, "v5")
        tmp_log.write_text(
            json.dumps({"ts": "T1", "alias": "production", "from": "", "to": "v5", "op": "set"}) + "\n"
        )
        cmd_rollback(ns(alias="production"))
        out = capsys.readouterr().out
        assert "no previous target" in out or "first promotion" in out

    def test_happy_path_prints_rolled_back_summary(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(14, "v6")
        tmp_log.write_text(
            json.dumps({"ts": "T1", "alias": "production", "from": "v5", "to": "v6", "op": "set"}) + "\n"
        )
        mock_client.search_model_versions.return_value = [make_mv(12, "v5")]
        cmd_rollback(ns(alias="production"))
        out = capsys.readouterr().out
        assert "v6" in out and "v5" in out
        assert "rolled back" in out

    def test_happy_path_assigns_alias_to_previous_version(self, mock_client, tmp_log):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(14, "v6")
        tmp_log.write_text(
            json.dumps({"ts": "T1", "alias": "production", "from": "v5", "to": "v6", "op": "set"}) + "\n"
        )
        mock_client.search_model_versions.return_value = [make_mv(12, "v5")]
        cmd_rollback(ns(alias="production"))
        mock_client.set_registered_model_alias.assert_called_once_with(
            "travel-assistant", "production", "12"
        )

    def test_happy_path_appends_rollback_log_entry(self, mock_client, tmp_log):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(14, "v6")
        tmp_log.write_text(
            json.dumps({"ts": "T1", "alias": "production", "from": "v5", "to": "v6", "op": "set"}) + "\n"
        )
        mock_client.search_model_versions.return_value = [make_mv(12, "v5")]
        cmd_rollback(ns(alias="production"))
        lines = tmp_log.read_text().strip().splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[1])
        assert entry["alias"] == "production"
        assert entry["from"] == "v6"
        assert entry["to"] == "v5"
        assert entry["op"] == "rollback"
        assert "ts" in entry

    def test_rollback_scans_log_backward_picks_last_production_entry(self, mock_client, tmp_log, capsys):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(14, "v6")
        tmp_log.write_text("\n".join([
            json.dumps({"ts": "T1", "alias": "staging", "from": "", "to": "v3", "op": "set"}),
            json.dumps({"ts": "T2", "alias": "production", "from": "v4", "to": "v5", "op": "set"}),
            json.dumps({"ts": "T3", "alias": "production", "from": "v5", "to": "v6", "op": "set"}),
        ]) + "\n")
        mock_client.search_model_versions.return_value = [make_mv(11, "v5")]
        cmd_rollback(ns(alias="production"))
        # The last production entry says from=v5, so we roll back to v5
        out = capsys.readouterr().out
        assert "v5" in out

    def test_rollback_target_zero_matches_exits_1(self, mock_client, tmp_log):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(14, "v6")
        tmp_log.write_text(
            json.dumps({"ts": "T1", "alias": "production", "from": "v5", "to": "v6", "op": "set"}) + "\n"
        )
        mock_client.search_model_versions.return_value = []
        with pytest.raises(SystemExit) as exc:
            cmd_rollback(ns(alias="production"))
        assert exc.value.code == 1

    def test_rollback_target_multiple_matches_prints_warning_and_uses_latest(
        self, mock_client, tmp_log, capsys
    ):
        from scripts.promote import cmd_rollback

        mock_client.get_model_version_by_alias.return_value = make_mv(14, "v6")
        tmp_log.write_text(
            json.dumps({"ts": "T1", "alias": "production", "from": "v5", "to": "v6", "op": "set"}) + "\n"
        )
        mock_client.search_model_versions.return_value = [make_mv(11, "v5"), make_mv(15, "v5")]
        cmd_rollback(ns(alias="production"))
        out = capsys.readouterr().out
        assert "warning" in out.lower()
        mock_client.set_registered_model_alias.assert_called_once_with(
            "travel-assistant", "production", "15"
        )