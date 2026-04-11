from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Int64, UnixTimestamp

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"

user = Entity(name="user_id", join_keys=["user_id"], value_type=ValueType.INT64)
item = Entity(name="item_id", join_keys=["item_id"], value_type=ValueType.INT64)
user_item = Entity(
    name="user_item_key",
    join_keys=["user_item_key"],
    value_type=ValueType.STRING,
)

user_stats_source = FileSource(
    path=str(DATA_DIR / "user_stats.parquet"),
    timestamp_field="event_timestamp",
)

item_stats_source = FileSource(
    path=str(DATA_DIR / "item_stats.parquet"),
    timestamp_field="event_timestamp",
)

user_item_stats_source = FileSource(
    path=str(DATA_DIR / "user_item_stats.parquet"),
    timestamp_field="event_timestamp",
)

user_stats = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="user_event_count_prev", dtype=Int64),
        Field(name="user_last_event_ts", dtype=UnixTimestamp),
    ],
    source=user_stats_source,
    online=True,
)

item_stats = FeatureView(
    name="item_stats",
    entities=[item],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="item_event_count_prev", dtype=Int64),
    ],
    source=item_stats_source,
    online=True,
)

user_item_stats = FeatureView(
    name="user_item_stats",
    entities=[user_item],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="user_item_event_count_prev", dtype=Int64),
        Field(name="user_item_last_event_ts", dtype=UnixTimestamp),
    ],
    source=user_item_stats_source,
    online=True,
)

purchase_prediction_v1 = FeatureService(
    name="purchase_prediction_v1",
    features=[user_stats, item_stats, user_item_stats],
)
