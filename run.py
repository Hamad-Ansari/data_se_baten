#!/usr/bin/env python
"""DATA_SE_BATEN command line interface.

Examples
--------
::

    python run.py serve-api                      # REST API on :8000
    python run.py serve-ui                       # Streamlit UI on :8501
    python run.py samples                        # regenerate the sample datasets
    python run.py analyze data/samples/customer_churn.csv --target churn
    python run.py analyze sales.xlsx --task time_series_forecasting
    python run.py runs                           # list recent runs
    python run.py runs --describe <run_id>
    python run.py predict <run_id> --json '{"age": 34, "plan_type": "basic"}'
    python run.py info                           # environment + dependency check
    python run.py cleanup --keep 5               # delete the oldest runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import APP_NAME, APP_TAGLINE, WORKFLOW_STAGES  # noqa: E402
from config.settings import get_settings  # noqa: E402


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> int:
    settings = get_settings()
    from utils.optional_deps import capability_report

    print(f"{APP_NAME} — {APP_TAGLINE}")
    print(f"  version      : {settings.app_version} ({settings.environment})")
    print(f"  project root : {ROOT}")
    print(f"  data dir     : {settings.data_dir}")
    print(f"  stages       : {len(WORKFLOW_STAGES)}")
    print(f"  LLM          : {'enabled' if settings.enable_llm else 'disabled'} "
          f"({settings.ollama_model} @ {settings.ollama_base_url})")
    deps = capability_report()
    available = [name for name, info in deps.items() if info.get("available")]
    missing = sorted(name for name, info in deps.items() if not info.get("available"))
    print(f"  dependencies : {len(available)}/{len(deps)} optional packages available")
    if missing:
        print(f"  optional missing: {', '.join(missing[:12])}")
    return 0


def cmd_serve_api(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    print(f"Serving the {APP_NAME} API on http://{host}:{port} (docs at /docs)")
    uvicorn.run("backend.main:app", host=host, port=int(port), reload=args.reload)
    return 0


def cmd_serve_ui(args: argparse.Namespace) -> int:
    import subprocess

    settings = get_settings()
    port = args.port or 8501
    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "streamlit_app.py"),
        "--server.address", args.host or "0.0.0.0", "--server.port", str(port),
        "--server.headless", "true", "--browser.gatherUsageStats", "false",
    ]
    print(f"Serving the Streamlit UI on http://localhost:{port} (API base: {settings.api_base_url})")
    return subprocess.call(command, cwd=str(ROOT))


def cmd_samples(args: argparse.Namespace) -> int:
    from scripts.make_sample_data import main as make_samples

    result = make_samples()
    print(result if isinstance(result, str) else "Sample datasets regenerated.")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from orchestrator import run_analysis
    from utils.errors import DataSenseError

    path = Path(args.dataset).expanduser()
    if not path.exists():
        print(f"Dataset not found: {path}", file=sys.stderr)
        return 2
    try:
        result = run_analysis(
            path,
            target=args.target,
            task=args.task,
            auto_approve=not args.require_approval,
            constraints={"max_candidates": args.max_candidates} if args.max_candidates else None,
            agent=not args.no_agent,
        )
    except DataSenseError as exc:
        print(exc.user_message, file=sys.stderr)
        return 1
    print(json.dumps(result if isinstance(result, dict) else {"result": str(result)}, indent=2, default=str))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    from orchestrator import describe_run, list_runs

    if args.describe:
        described = describe_run(args.describe)
        print(described if isinstance(described, str)
              else json.dumps(described, indent=2, default=str))
        return 0
    rows = list_runs(limit=args.limit)
    if not rows:
        print("No runs yet — start one with `python run.py analyze <file>`.")
        return 0
    print(f"{'run_id':<44} {'status':<18} {'dataset':<28} {'model':<22} score")
    for row in rows:
        print(
            f"{str(row.get('run_id')):<44} {str(row.get('status')):<18} "
            f"{str(row.get('dataset') or row.get('dataset_name'))[:27]:<28} "
            f"{str(row.get('model') or '')[:21]:<22} {row.get('primary_score') or ''}"
        )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from ml.deployment import get_model_service

    service = get_model_service(args.run_id)
    if args.file:
        import pandas as pd

        path = Path(args.file)
        frame = pd.read_csv(path) if path.suffix.lower() in {".csv", ".tsv"} else pd.read_excel(path)
        payload = service.predict_dataframe(frame)
    else:
        if not args.json:
            print("Pass --json '{\"col\": value}' or --file data.csv", file=sys.stderr)
            return 2
        payload = service.predict([json.loads(args.json)])
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    from orchestrator import cleanup_runs

    print(json.dumps(cleanup_runs(keep=args.keep), indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=f"{APP_NAME} — {APP_TAGLINE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="environment, settings and dependency check")
    p_info.set_defaults(func=cmd_info)

    p_api = sub.add_parser("serve-api", help="start the FastAPI backend")
    p_api.add_argument("--host", default=None)
    p_api.add_argument("--port", type=int, default=None)
    p_api.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    p_api.set_defaults(func=cmd_serve_api)

    p_ui = sub.add_parser("serve-ui", help="start the Streamlit interface")
    p_ui.add_argument("--host", default=None)
    p_ui.add_argument("--port", type=int, default=None)
    p_ui.set_defaults(func=cmd_serve_ui)

    p_samples = sub.add_parser("samples", help="regenerate the bundled sample datasets")
    p_samples.set_defaults(func=cmd_samples)

    p_analyze = sub.add_parser("analyze", help="run the agent on a dataset from the terminal")
    p_analyze.add_argument("dataset", help="path to the dataset")
    p_analyze.add_argument("--target", default=None, help="target column (optional)")
    p_analyze.add_argument("--task", default=None, help="force a task, e.g. clustering")
    p_analyze.add_argument("--max-candidates", type=int, default=None)
    p_analyze.add_argument("--require-approval", action="store_true",
                           help="pause after the cleaning plan instead of auto-approving")
    p_analyze.add_argument("--no-agent", action="store_true", help="run the deterministic pipeline only")
    p_analyze.set_defaults(func=cmd_analyze)

    p_runs = sub.add_parser("runs", help="list or describe runs")
    p_runs.add_argument("--limit", type=int, default=20)
    p_runs.add_argument("--describe", default=None, help="run id to describe in full")
    p_runs.set_defaults(func=cmd_runs)

    p_predict = sub.add_parser("predict", help="score records with a deployed model")
    p_predict.add_argument("run_id")
    p_predict.add_argument("--json", default=None, help="one record as JSON")
    p_predict.add_argument("--file", default=None, help="CSV/Excel file to score in batch")
    p_predict.set_defaults(func=cmd_predict)

    p_cleanup = sub.add_parser("cleanup", help="delete the oldest runs")
    p_cleanup.add_argument("--keep", type=int, default=10, help="how many recent runs to keep")
    p_cleanup.set_defaults(func=cmd_cleanup)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
