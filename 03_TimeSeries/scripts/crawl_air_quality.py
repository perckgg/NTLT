"""
Thu thập dữ liệu chuỗi thời gian chất lượng không khí (PM2.5) + thời tiết theo giờ
cho 2 thành phố Hà Nội và TP.HCM, phục vụ bài Time Series của BTL.

Nguồn:
  - PM2.5: OpenAQ v3 — Hà Nội: trạm "Số 46 Lưu Quang Vũ" (Hanoi Air Quality
    Monitoring Network); TP.HCM: trạm "CMT8" (AirGradient). Cần API key miễn phí,
    đặt trong biến môi trường OPENAQ_API_KEY. Nếu không có key hoặc lỗi,
    tự động fallback sang Open-Meteo Air Quality API (dữ liệu mô phỏng CAMS).
  - Thời tiết: Open-Meteo Historical Weather API (không cần key).

Kết quả: mỗi thành phố một file CSV riêng trong data/timeseries/.

Cách chạy (PowerShell, từ thư mục 03_TimeSeries):
  $env:OPENAQ_API_KEY = "<key>"
  py scripts/crawl_air_quality.py
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

# Giai đoạn gần nhất mà cả 2 trạm đều có dữ liệu (trạm CMT8 bắt đầu gửi từ 11/2024).
START_DATE = "2024-12-01"
END_DATE = "2026-09-30"
TIMEZONE = "Asia/Ho_Chi_Minh"
OUTPUT_DIR = os.path.join("data", "timeseries")

CITIES = {
    "hanoi": {"name": "Hà Nội", "lat": 21.0285, "lon": 105.8542, "openaq_sensor_id": 7772032},
    "hcmc": {"name": "TP.HCM", "lat": 10.7769, "lon": 106.7009, "openaq_sensor_id": 11357424},
}

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
]


def http_get_json(url, headers=None, retries=5):
    req = urllib.request.Request(url, headers=headers or {})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(30 if e.code == 429 else 5 * attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(5)
                continue
            raise


def fetch_pm25_openaq(sensor_id, api_key, start=START_DATE, end=END_DATE):
    """Lấy PM2.5 theo giờ từ OpenAQ v3, chia theo từng năm để phân trang nhẹ hơn."""
    headers = {"X-API-Key": api_key}
    rows = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        dt_from = max(f"{year}-01-01", start) + "T00:00:00+07:00"
        dt_to = min(f"{year}-12-31", end) + "T23:59:59+07:00"
        page = 1
        while True:
            params = urllib.parse.urlencode({
                "datetime_from": dt_from, "datetime_to": dt_to, "limit": 1000, "page": page,
            })
            url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours?{params}"
            data = http_get_json(url, headers)
            results = data.get("results", [])
            for r in results:
                rows.append({
                    "datetime": r["period"]["datetimeFrom"]["local"],
                    "pm25": r["value"],
                })
            print(f"    OpenAQ sensor {sensor_id} | {year} | page {page}: {len(results)} bản ghi")
            if len(results) < 1000:
                break
            page += 1
            time.sleep(1.1)  # giới hạn 60 request/phút
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("OpenAQ không trả về dữ liệu")
    df["datetime"] = pd.to_datetime(df["datetime"].str[:19])
    return df.drop_duplicates("datetime").set_index("datetime").sort_index()


def fetch_pm25_openmeteo(lat, lon, start=START_DATE, end=END_DATE):
    """Fallback: PM2.5 mô phỏng (CAMS) từ Open-Meteo Air Quality API."""
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "hourly": "pm2_5",
        "start_date": start, "end_date": end, "timezone": TIMEZONE,
    })
    data = http_get_json(f"https://air-quality-api.open-meteo.com/v1/air-quality?{params}")
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "pm25": data["hourly"]["pm2_5"],
    })
    return df.set_index("datetime")


def fetch_weather(lat, lon, start=START_DATE, end=END_DATE):
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
        "start_date": start, "end_date": end, "timezone": TIMEZONE,
    })
    data = http_get_json(f"https://archive-api.open-meteo.com/v1/archive?{params}")
    df = pd.DataFrame(data["hourly"])
    df["datetime"] = pd.to_datetime(df.pop("time"))
    return df.set_index("datetime")


def build_city_dataset(key, cfg, api_key):
    print(f"[*] {cfg['name']}")
    source = "openaq"
    try:
        if not api_key:
            raise RuntimeError("Chưa đặt OPENAQ_API_KEY")
        pm = fetch_pm25_openaq(cfg["openaq_sensor_id"], api_key)
    except Exception as e:
        print(f"    [!] OpenAQ lỗi ({e}) -> fallback Open-Meteo Air Quality")
        pm = fetch_pm25_openmeteo(cfg["lat"], cfg["lon"])
        source = "open-meteo"

    weather = fetch_weather(cfg["lat"], cfg["lon"])

    # Lưới thời gian đầy đủ theo giờ: giờ nào trạm không đo sẽ thành NaN (missing thật)
    full_index = pd.date_range(f"{START_DATE} 00:00", f"{END_DATE} 23:00", freq="h", name="datetime")
    df = weather.reindex(full_index).join(pm.reindex(full_index))
    df.insert(0, "city", cfg["name"])
    df["pm25_source"] = source
    return df.reset_index()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api_key = os.environ.get("OPENAQ_API_KEY", "").strip()
    for key, cfg in CITIES.items():
        df = build_city_dataset(key, cfg, api_key)
        path = os.path.join(OUTPUT_DIR, f"air_quality_{key}.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"[+] Đã lưu {path}: {len(df)} dòng, missing PM2.5 = {df['pm25'].isna().mean():.1%}")


if __name__ == "__main__":
    main()
