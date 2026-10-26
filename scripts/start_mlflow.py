"""Start the local historical tracking UI without touching the production pipeline."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / ".mlflow")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    store = args.store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    os.execv(sys.executable, [sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", "sqlite:///" + str(store / "mlflow.db"),
        "--default-artifact-root", (store / "artifacts").as_uri(),
        "--no-serve-artifacts", "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1"])


if __name__ == "__main__":
    main()
