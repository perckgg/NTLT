"""
Compare candidate PM2.5 stations by how much of the study period they actually measured.

For every candidate OpenAQ sensor the script downloads hourly PM2.5 over the same period
as crawl_air_quality.py and reports:
  - coverage          share of hours in [START_DATE 00:00, END_DATE 23:00] with a value
  - worst_hour_cov    lowest coverage among the 24 hours of the day; a value far below
                      `coverage` means the gaps fall on particular hours (not at random)
  - worst_month_cov   lowest coverage among calendar months
  - longest_gap_h     longest run of consecutive missing hours
  - first / last      first and last hour with a value inside the period

Output: data/timeseries/station_coverage.csv (shown in section 1.1 of the notebook).

Usage (from the 03_TimeSeries folder):
  OPENAQ_API_KEY=<key> python scripts/check_station_coverage.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawl_air_quality import END_DATE, OUTPUT_DIR, START_DATE, fetch_pm25_openaq  # noqa: E402

# Every OpenAQ PM2.5 sensor within 25 km of each city centre that reported during the
# period (OpenAQ /v3/locations search, 15/11/2026). HCMC has only one such station.
CANDIDATES = [
    ("Hanoi", 13502151, "Cong vien Nhan Chinh - Khuat Duy Tien", "reference"),
    ("Hanoi", 13502154, "Giai Phong - DH Bach Khoa", "reference"),
    ("Hanoi", 7772032, "46 Luu Quang Vu", "reference"),
    ("Hanoi", 13502150, "556 Nguyen Van Cu", "reference"),
    ("Hanoi", 7772087, "Minh Khai - Bac Tu Liem", "reference"),
    ("Hanoi", 14581375, "Dragon (AirGradient)", "low-cost"),
    ("HCMC", 11357424, "CMT8 (AirGradient)", "low-cost"),
]


def longest_run(mask: pd.Series) -> int:
    """Length of the longest run of True values in a boolean series."""
    run_id = (~mask).cumsum()
    runs = mask.groupby(run_id).sum()
    return int(runs.max()) if len(runs) else 0


def coverage_row(city: str, sensor_id: int, name: str, kind: str, api_key: str) -> dict:
    grid = pd.date_range(f"{START_DATE} 00:00", f"{END_DATE} 23:00", freq="h")
    try:
        pm = fetch_pm25_openaq(sensor_id, api_key)["pm25"]
    except RuntimeError:  # sensor returned nothing in the period
        pm = pd.Series(dtype=float)
    s = pm.reindex(grid)
    has = s.notna()
    measured = s.dropna()
    return {
        "city": city,
        "station": name,
        "sensor_id": sensor_id,
        "type": kind,
        "coverage": round(has.mean(), 3),
        "worst_hour_cov": round(has.groupby(grid.hour).mean().min(), 3),
        "worst_month_cov": round(has.groupby(grid.to_period("M")).mean().min(), 3),
        "longest_gap_h": longest_run(~has),
        "first": measured.index.min(),
        "last": measured.index.max(),
    }


def main() -> None:
    api_key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENAQ_API_KEY is not set")
    rows = [coverage_row(*c, api_key) for c in CANDIDATES]
    table = pd.DataFrame(rows).sort_values(["city", "coverage"], ascending=[True, False])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "station_coverage.csv")
    table.to_csv(path, index=False, encoding="utf-8")
    print(table.to_string(index=False))
    print(f"[+] Saved {path}")


if __name__ == "__main__":
    main()
