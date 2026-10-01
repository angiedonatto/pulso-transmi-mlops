"""Switch the champion between Candidate J v1.1 and Candidate K v2.1 (model_versions.is_active only).

Dry-run by default. `--apply` promotes K (J stays intact and inactive); `--rollback --apply` restores
J immediately. See src/candidate_k_promotion.py for the preconditions and verification.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import candidate_k_promotion as promotion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollback", action="store_true", help="K -> J instead of J -> K.")
    parser.add_argument("--apply", action="store_true", help="Write the switch (default: dry-run).")
    parser.add_argument("--env-file", type=Path, help="Load only SUPABASE_URL / SUPABASE_SERVICE_KEY from this file.")
    args = parser.parse_args()
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
                os.environ.setdefault(key, value)
    with promotion.ModelVersionsClient() as client:
        result = (promotion.rollback if args.rollback else promotion.promote)(client, apply=args.apply)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
