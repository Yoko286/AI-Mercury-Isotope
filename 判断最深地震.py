# -*- coding: utf-8 -*-
"""
从 USGS 地震数据中找出震源最深的地震
"""

import pandas as pd

# ================= 配置区 =================
FILE_PATH = r"D:\wyc\Programming\Talent Program\AI-Mercury-Isotope\usgs_earthquake.csv"
# =========================================

def find_deepest_earthquake(path):
    # 读取数据
    df = pd.read_csv(path)
    print(f"共读取 {len(df)} 条记录")
    print(f"列名：{list(df.columns)}")

    # 确保 depth 是数值类型
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')

    # 去掉 depth 为空的记录
    df_valid = df.dropna(subset=['depth'])
    print(f"有效深度记录：{len(df_valid)} 条")

    # 找出深度最大的那一条
    deepest = df_valid.loc[df_valid['depth'].idxmax()]

    print("\n" + "=" * 50)
    print("震源最深的地震")
    print("=" * 50)
    print(f"深度   : {deepest['depth']} 公里")
    print(f"震级   : {deepest['magnitude']}")
    print(f"位置   : {deepest['place']}")
    print(f"经度   : {deepest['longitude']}")
    print(f"纬度   : {deepest['latitude']}")
    print(f"时间   : {deepest['time']}")

    # 顺便打印深度前 10 名的记录，方便对比
    print("\n" + "=" * 50)
    print("深度最大的前 10 条记录")
    print("=" * 50)
    top10 = df_valid.nlargest(10, 'depth')[
        ['depth', 'magnitude', 'place', 'time']
    ]
    print(top10.to_string(index=False))

    return deepest


if __name__ == "__main__":
    find_deepest_earthquake(FILE_PATH)