import requests
import pandas as pd
import time

url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

all_records = []

# 按年份分批，每年单独请求
for year in range(2020, 2027):
    start = f"{year}-01-01"
    end = f"{year}-12-31" if year < 2026 else "2026-10-11"
    
    params = {
        "format": "geojson",
        "starttime": start,
        "endtime": end,
        "minmagnitude": 2.5,
        "limit": 20000
    }
    
    print(f"正在抓取 {start} 到 {end} ...")
    response = requests.get(url, params=params, headers=headers, timeout=60)
    
    if response.status_code != 200:
        print(f"  请求失败，状态码：{response.status_code}")
        print(f"  返回内容：{response.text[:200]}")
        continue
    
    if not response.text.strip().startswith("{"):
        print(f"  返回的不是 JSON，跳过。")
        continue
    
    data = response.json()
    features = data.get("features", [])
    print(f"  获取 {len(features)} 条记录")
    
    for eq in features:
        p = eq["properties"]
        geo = eq["geometry"]["coordinates"]
        all_records.append({
            "magnitude": p["mag"],
            "depth": geo[2],
            "longitude": geo[0],
            "latitude": geo[1],
            "place": p["place"],
            "time": p["time"],
            "tsunami": p["tsunami"],
        })
    
    time.sleep(0.5)  # 礼貌爬取

# 保存
df = pd.DataFrame(all_records)
df["time"] = pd.to_datetime(df["time"], unit="ms")
print(f"\n总共获取 {len(df)} 条记录")
df.to_csv("usgs_earthquake.csv", index=False, encoding="utf-8-sig")
print("已保存到 usgs_earthquake.csv")