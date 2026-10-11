# -*- coding: utf-8 -*-
"""
深源地震分布可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================= 配置区 =================
FILE_PATH = r"D:\wyc\Programming\Talent Program\AI-Mercury-Isotope\usgs_earthquake.csv"
OUTPUT_DIR = r"D:\wyc\Programming\Talent Program\AI-Mercury-Isotope"
# =========================================

# 中文字体（自动检测）
from matplotlib import font_manager

font_path = r"C:\Windows\Fonts\msyh.ttc"  # 改成你系统里实际存在的路径
my_font = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = [my_font.get_name()]
plt.rcParams['axes.unicode_minus'] = False


def plot_depth_classes(df, out_dir):
    """按深度分类画全球地图：浅源红、中源橙、深源蓝"""

    # 深度分类
    def depth_class(d):
        if pd.isna(d):
            return '未知'
        if d < 70:
            return '浅源'
        elif d < 300:
            return '中源'
        else:
            return '深源'

    df = df.copy()
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df = df.dropna(subset=['depth', 'longitude', 'latitude'])
    df['depth_class'] = df['depth'].apply(depth_class)

    colors = {'浅源': 'red', '中源': 'orange', '深源': 'blue'}

    # 画图
    plt.figure(figsize=(16, 9))
    for cls in ['浅源', '中源', '深源']:
        subset = df[df['depth_class'] == cls]
        plt.scatter(
            subset['longitude'], subset['latitude'],
            c=colors[cls], label=f'{cls}（{len(subset)} 条）',
            s=15, alpha=0.5, edgecolors='none'
        )

    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('全球地震分布（按深度分类）')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.savefig(f"{out_dir}\\09_depth_classes_map.png", dpi=150)
    plt.close()
    print("已保存：09_depth_classes_map.png")

    # 打印各分类数量
    print("\n深度分类统计：")
    print(df['depth_class'].value_counts())


def plot_deep_only(df, out_dir):
    """单独画深源地震（>300 km）分布"""

    df = df.copy()
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df = df.dropna(subset=['depth', 'longitude', 'latitude'])

    deep = df[df['depth'] > 300].copy()

    if len(deep) == 0:
        print("没有深源地震记录，跳过绘图。")
        return

    plt.figure(figsize=(16, 9))
    scatter = plt.scatter(
        deep['longitude'], deep['latitude'],
        c=deep['depth'], cmap='plasma_r',
        s=25, alpha=0.7, edgecolors='none'
    )
    plt.colorbar(scatter, label='深度 (km)')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title(f'深源地震分布（深度 > 300 km，共 {len(deep)} 条）')
    plt.grid(alpha=0.3)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.savefig(f"{out_dir}\\10_deep_earthquakes_map.png", dpi=150)
    plt.close()
    print("已保存：10_deep_earthquakes_map.png")

    # 打印深源地震最多的区域
    print(f"\n深源地震共 {len(deep)} 条，深度范围："
          f"{deep['depth'].min():.1f} ~ {deep['depth'].max():.1f} km")

    # 按 place 粗略统计，看哪些地方深源地震最多
    print("\n深源地震最多的区域（前 10）：")
    # 简单提取 place 里的区域关键词
    deep['region'] = deep['place'].astype(str).str.split(',').str[-1].str.strip()
    print(deep['region'].value_counts().head(10))


if __name__ == "__main__":
    df = pd.read_csv(FILE_PATH)
    plot_depth_classes(df, OUTPUT_DIR)
    plot_deep_only(df, OUTPUT_DIR)