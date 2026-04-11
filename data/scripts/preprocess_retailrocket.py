from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/retailrocket")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    events_path = RAW_DIR / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}. Check your download folder.")

    # RetailRocket events.csv columns commonly:
    # timestamp, visitorid, event, itemid, transactionid
    events = pd.read_csv(events_path)

    # Standardize column names
    rename_map = {
        "timestamp": "event_ts",
        "visitorid": "user_id",
        "itemid": "item_id",
        "event": "event_type",
        "transactionid": "transaction_id",
    }
    events = events.rename(columns={k: v for k, v in rename_map.items() if k in events.columns})

    # Convert timestamp (ms) -> datetime UTC
    # (RetailRocket timestamps are usually milliseconds)
    events["event_ts"] = pd.to_datetime(events["event_ts"], unit="ms", utc=True, errors="coerce")

    # Keep only necessary columns
    keep_cols = [c for c in ["user_id", "item_id", "event_type", "event_ts", "transaction_id"] if c in events.columns]
    events = events[keep_cols].dropna(subset=["user_id", "item_id", "event_ts", "event_type"])

    # Sort
    events = events.sort_values("event_ts").reset_index(drop=True)

    out_path = OUT_DIR / "events_retailrocket.parquet"
    events.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(events):,} | columns={list(events.columns)}")
    print(events.head(5))

if __name__ == "__main__":
    main()
