#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw/retailrocket"
PROCESSED_DIR="${REPO_ROOT}/data/processed"
DATA_URL="${RETAILROCKET_DATA_URL:-}"
EXPECTED_SHA256="$(printf '%s' "${RETAILROCKET_SHA256:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}"

if [[ -z "${DATA_URL}" ]]; then
  cat >&2 <<'EOF'
ERROR: RETAILROCKET_DATA_URL is not configured.
Set the GitHub Actions repository secret RETAILROCKET_DATA_URL to either:
- a direct download URL for RetailRocket events.csv
- or a zip archive URL containing events.csv
EOF
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

DOWNLOAD_PATH="${TMP_DIR}/retailrocket_download"
EXTRACT_DIR="${TMP_DIR}/extracted"

echo "Bootstrapping RetailRocket raw dataset into ${RAW_DIR}"
echo "Downloading dataset asset from configured source..."
curl -fL --retry 3 --retry-delay 5 "${DATA_URL}" -o "${DOWNLOAD_PATH}"

if [[ -n "${EXPECTED_SHA256}" ]]; then
  ACTUAL_SHA256="$(sha256sum "${DOWNLOAD_PATH}" | awk '{print $1}' | tr '[:upper:]' '[:lower:]')"
  if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "ERROR: SHA256 mismatch for downloaded RetailRocket asset." >&2
    echo "Expected: ${EXPECTED_SHA256}" >&2
    echo "Actual:   ${ACTUAL_SHA256}" >&2
    echo "If the dataset asset changed, update RETAILROCKET_SHA256 or remove it." >&2
    exit 1
  fi
  echo "Verified SHA256 checksum for downloaded asset."
fi

python3 - "${DOWNLOAD_PATH}" "${EXTRACT_DIR}" "${RAW_DIR}" <<'PY'
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import zipfile

download_path = Path(sys.argv[1])
extract_dir = Path(sys.argv[2])
raw_dir = Path(sys.argv[3])

expected_names = {
    "events.csv": "events.csv",
    "category_tree.csv": "category_tree.csv",
    "item_properties_part1.csv": "item_properties_part1.csv",
    "item_properties_part2.csv": "item_properties_part2.csv",
}

if zipfile.is_zipfile(download_path):
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(download_path) as archive:
        archive.extractall(extract_dir)

    found: dict[str, Path] = {}
    for path in sorted(extract_dir.rglob("*")):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if lower_name in expected_names and lower_name not in found:
            found[lower_name] = path

    if "events.csv" not in found:
        csv_files = [str(path.relative_to(extract_dir)) for path in sorted(extract_dir.rglob("*.csv"))]
        print(
            "ERROR: Downloaded archive did not contain events.csv. "
            f"CSV files found: {csv_files or 'none'}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    for lower_name, source_path in found.items():
        target_path = raw_dir / expected_names[lower_name]
        shutil.copy2(source_path, target_path)
        print(f"Copied {source_path} -> {target_path}")
else:
    target_path = raw_dir / "events.csv"
    shutil.copy2(download_path, target_path)
    print(f"Copied direct CSV download -> {target_path}")
PY

if [[ ! -f "${RAW_DIR}/events.csv" ]]; then
  echo "ERROR: RetailRocket bootstrap completed without ${RAW_DIR}/events.csv." >&2
  echo "Check RETAILROCKET_DATA_URL and confirm it points to a direct events.csv or a zip containing events.csv." >&2
  exit 1
fi

echo "RetailRocket bootstrap completed successfully."
echo "Available raw files:"
find "${RAW_DIR}" -maxdepth 1 -type f | sort
