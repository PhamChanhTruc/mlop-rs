from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.recommendation import CandidateGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact recommendation candidate artifact from processed RetailRocket events.",
    )
    parser.add_argument(
        "--events-path",
        type=Path,
        default=REPO_ROOT / "data/processed/events_retailrocket.parquet",
        help="Processed events parquet used to derive recent-item and popular-item candidates.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "data/processed/recommendation_candidates.json.gz",
        help="Output path for the compact recommendation candidate artifact.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=90,
        help="Window used for popular-item fallback scoring.",
    )
    parser.add_argument(
        "--max-user-recent-items",
        type=int,
        default=50,
        help="Maximum number of recent unique items retained per user.",
    )
    parser.add_argument(
        "--max-popular-items",
        type=int,
        default=500,
        help="Maximum number of globally popular fallback items retained.",
    )
    parser.add_argument(
        "--candidate-pool-multiplier",
        type=int,
        default=10,
        help="Multiplier used to size the generated candidate pool for top-k requests.",
    )
    parser.add_argument(
        "--min-candidate-pool-size",
        type=int,
        default=50,
        help="Minimum generated candidate pool size regardless of top-k.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.events_path.exists():
        raise FileNotFoundError(
            f"Missing {args.events_path}. Run data/scripts/preprocess_retailrocket.py first."
        )

    generator = CandidateGenerator.from_parquet(
        args.events_path,
        recent_days=args.recent_days,
        max_user_recent_items=args.max_user_recent_items,
        max_popular_items=args.max_popular_items,
        candidate_pool_multiplier=args.candidate_pool_multiplier,
        min_candidate_pool_size=args.min_candidate_pool_size,
    )
    output_path = generator.save_artifact(args.output_path)

    print("Saved recommendation candidate artifact")
    print(f"events_path={args.events_path}")
    print(f"output_path={output_path}")
    print(f"users_indexed={len(generator.user_recent_items):,}")
    print(f"popular_items={len(generator.popular_items):,}")
    print(f"candidate_pool_multiplier={generator.candidate_pool_multiplier}")
    print(f"min_candidate_pool_size={generator.min_candidate_pool_size}")


if __name__ == "__main__":
    main()
