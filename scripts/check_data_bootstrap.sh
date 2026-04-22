#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw/retailrocket"
PROCESSED_DIR="${REPO_ROOT}/data/processed"
REQUIRED_FILE="${RAW_DIR}/events.csv"
OPTIONAL_FILES=(
  "category_tree.csv"
  "item_properties_part1.csv"
  "item_properties_part2.csv"
)

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}"

echo "RetailRocket data bootstrap check"
echo "Repository root: ${REPO_ROOT}"
echo "Expected raw data directory: ${RAW_DIR}"
echo "Expected processed output directory: ${PROCESSED_DIR}"
echo

if [[ ! -f "${REQUIRED_FILE}" ]]; then
  echo "ERROR: Missing required raw input file: ${REQUIRED_FILE}" >&2
  echo "Place RetailRocket events.csv at data/raw/retailrocket/events.csv, then rerun this check." >&2
  exit 1
fi

if [[ ! -s "${REQUIRED_FILE}" ]]; then
  echo "ERROR: Required raw input file is empty: ${REQUIRED_FILE}" >&2
  exit 1
fi

python3 - "${REQUIRED_FILE}" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.reader(handle)
    header = next(reader, None)

if not header:
    raise SystemExit(f"ERROR: {path} has no header row")

required = {"timestamp", "visitorid", "event", "itemid"}
missing = sorted(required - set(header))
if missing:
    raise SystemExit(
        f"ERROR: {path} is missing required columns for preprocess_retailrocket.py: {missing}. "
        f"Found header: {header}"
    )

print(f"Header OK for {path}: {header}")
PY

echo
echo "Optional RetailRocket raw files in the same folder:"
for name in "${OPTIONAL_FILES[@]}"; do
  path="${RAW_DIR}/${name}"
  if [[ -f "${path}" ]]; then
    echo "  present: ${path}"
  else
    echo "  optional missing: ${path}"
  fi
done

echo
echo "Bootstrap check passed."
echo "Next commands:"
echo "  python3 data/scripts/preprocess_retailrocket.py"
echo "  python3 data/scripts/build_trainset_retailrocket.py"
