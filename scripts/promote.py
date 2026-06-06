"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient

from src.config import get_settings

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"

client = MlflowClient(get_settings().mlflow_tracking_uri)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_version(name: str, config_id: str):
    """Resolve config_id → ModelVersion, handling zero/multiple matches."""
    versions = client.search_model_versions(
        f"name = '{name}' AND tags.config_id = '{config_id}'"
    )
    if len(versions) == 0:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)
    if len(versions) > 1:
        sorted_vs = sorted(versions, key=lambda v: int(v.version))
        nums = [int(v.version) for v in sorted_vs]
        latest = sorted_vs[-1]
        print(
            f"warning: multiple versions match config_id={config_id} "
            f"(MLflow versions {nums}); using latest ({latest.version})"
        )
        return latest
    return versions[0]


def _read_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    return [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]


def _append_log(entry: dict) -> None:
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    target = _find_version(args.name, args.config_id)

    try:
        current_mv = client.get_model_version_by_alias(args.name, args.alias)
        current_config_id = current_mv.tags.get("config_id", "")
    except RestException:
        current_config_id = ""

    client.set_registered_model_alias(args.name, args.alias, target.version)

    from_id = "(unset)" if current_config_id == "" else current_config_id
    print(f"{args.alias}: {from_id} → {args.config_id}")

    _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": args.alias,
        "from": current_config_id,
        "to": args.config_id,
        "op": "set",
    })


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    try:
        mv = client.get_model_version_by_alias(args.name, args.alias)
    except RestException:
        print(f"error: alias '{args.alias}' is not set")
        sys.exit(1)

    print(f"{args.name} @ {args.alias}")
    for key, val in mv.tags.items():
        print(f"  {key}: {val}")

    run = client.get_run(mv.run_id)
    for metric in ("accuracy_overall", "verdict_rate_leaked", "total_cost_usd"):
        if metric not in run.data.metrics:
            continue
        val = run.data.metrics[metric]
        if metric == "total_cost_usd":
            print(f"  {metric}: ${val:.2f}")
        else:
            print(f"  {metric}: {val}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""
    try:
        registered_model = client.get_registered_model(args.name)
    except MlflowException as e:
        print(f"MlflowException: {e.message}")
        return
    if not registered_model.aliases:
        print("no aliases set")
        return
    for alias in registered_model.aliases:
        model_version = client.get_model_version_by_alias(args.name, alias)
        print(f"{alias} -> {model_version.tags['config_id']}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    try:
        current_mv = client.get_model_version_by_alias(args.name, args.alias)
    except RestException:
        print("nothing to roll back")
        return

    current_config_id = current_mv.tags.get("config_id", "")

    log = _read_log()
    last_entry = next(
        (e for e in reversed(log) if e.get("alias") == args.alias), None
    )

    if last_entry is None:
        print(f"no promotion history for alias {args.alias}")
        return

    if last_entry["op"] == "rollback":
        print(f"error: {args.alias} was just rolled back; no further history to walk back to")
        return

    if last_entry["from"] == "":
        print(f"{args.alias} has no previous target (first promotion ever)")
        return

    target_config_id = last_entry["from"]
    target = _find_version(args.name, target_config_id)

    client.set_registered_model_alias(args.name, args.alias, target.version)
    print(f"{args.alias}: {current_config_id} → {target_config_id} (rolled back)")

    _append_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": args.alias,
        "from": current_config_id,
        "to": target_config_id,
        "op": "rollback",
    })


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
