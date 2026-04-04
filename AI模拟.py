import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# 设置Nature风格的全局参数
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['legend.frameon'] = False
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300

# 设置随机种子以保证结果可重现
np.random.seed(42)
mercury_isotopes = {
    '196Hg': {'typical_abundance': 0.15, 'unit': '%', 'color': '#4C72B0'},
    '198Hg': {'typical_abundance': 10.00, 'unit': '%', 'color': '#55A868'},
    '199Hg': {'typical_abundance': 16.87, 'unit': '%', 'color': '#C44E52'},
    '200Hg': {'typical_abundance': 23.10, 'unit': '%', 'color': '#8172B2'},
    '201Hg': {'typical_abundance': 13.18, 'unit': '%', 'color': '#CCB974'},
    '202Hg': {'typical_abundance': 29.86, 'unit': '%', 'color': '#64B5CD'},
    '204Hg': {'typical_abundance': 6.87, 'unit': '%', 'color': '#E28F41'}
}

# 生成模拟数据：假设有15个样品，每个同位素的丰度在典型值附近波动
num_samples = 15
simulated_data = {}

for isotope, info in mercury_isotopes.items():
    typical = info['typical_abundance']
    # 添加 ±5% 的相对波动（正态分布），并保证丰度非负
    abundances = np.random.normal(loc=typical, scale=typical * 0.05, size=num_samples)
    abundances = np.maximum(abundances, 0)  # 确保没有负值
    simulated_data[isotope] = abundances

# 转换为DataFrame
df = pd.DataFrame(simulated_data)

# 计算每个同位素的统计量
mean_abundances = df.mean(axis=0)
std_abundances = df.std(axis=0)
sem_abundances = df.sem(axis=0)  # 标准误
typical_abundances = [mercury_isotopes[iso]['typical_abundance'] for iso in mean_abundances.index]

# 创建图形 - Nature风格通常采用单栏宽度（约8.9cm = 3.5英寸）
fig, ax = plt.subplots(figsize=(3.5, 3.5))

# 设置x轴位置
x_pos = np.arange(len(mean_abundances))
width = 0.7  # 柱宽

# 绘制柱状图（平均丰度）
bars = ax.bar(x_pos, mean_abundances, width, 
              color=[mercury_isotopes[iso]['color'] for iso in mean_abundances.index],
              edgecolor='black', linewidth=0.5, alpha=0.85,
              label='Mean abundance')

# 添加误差棒（标准偏差，反映样品间自然变异）
errorbars = ax.errorbar(x_pos, mean_abundances, yerr=std_abundances,
                        fmt='none', ecolor='black', capsize=3, capthick=0.8,
                        elinewidth=0.8, alpha=0.7, label='±1 s.d. (n=15)')

# 添加理论典型值作为参考点
ax.scatter(x_pos, typical_abundances, marker='o', s=35, 
           facecolors='white', edgecolors='black', linewidths=1,
           zorder=3, label='Typical natural abundance')

# 设置坐标轴标签和标题
ax.set_ylabel('Abundance (%)', fontsize=9, fontweight='normal')
ax.set_xlabel('Mercury isotope', fontsize=9, fontweight='normal')

# 设置x轴刻度标签
ax.set_xticks(x_pos)
ax.set_xticklabels(mean_abundances.index, rotation=45, ha='right', fontsize=8)

# 设置y轴范围（留出一些空间给误差棒）
y_max = max(mean_abundances + std_abundances) * 1.1
ax.set_ylim(0, y_max)

# 添加次要网格线（Nature风格通常有浅色网格）
ax.yaxis.grid(True, linestyle='-', alpha=0.15, linewidth=0.3)
ax.set_axisbelow(True)  # 网格线在图形下方

# 添加图例
legend = ax.legend(loc='upper right', frameon=False, fontsize=7, handlelength=1.5)

# 添加样本量注释
ax.text(0.98, 0.97, f'n = {num_samples} samples', transform=ax.transAxes,
        fontsize=7, verticalalignment='top', horizontalalignment='right',
        fontstyle='italic', color='#555555')

# 可选：添加每个柱子上方的平均丰度数值
for i, (bar, mean_val, std_val) in enumerate(zip(bars, mean_abundances, std_abundances)):
    if mean_val > 1:  # 数值较大时显示整数部分
        label = f'{mean_val:.1f}'
    else:  # 数值较小时显示两位小数
        label = f'{mean_val:.2f}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.2,
            label, ha='center', va='bottom', fontsize=6, color='black')

# 添加一个简短的注释框（Nature风格有时会包含关键统计信息）
stats_text = f"Total abundance sum: {mean_abundances.sum():.1f}%"
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=6,
        verticalalignment='top', bbox=props)

# 设置标题（Nature文章通常不设主标题，用图注代替，但这里添加一个简洁的）
ax.set_title('Mercury isotopic abundances', fontsize=10, fontweight='normal', pad=10)

# 调整布局，确保标签完整显示
plt.tight_layout(pad=0.5)

# 保存为高分辨率图片（适用于Nature投稿）
plt.savefig('mercury_isotope_abundances_nature_style.png', dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('mercury_isotope_abundances_nature_style.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

# 显示图形
plt.show()

# 打印统计摘要
print("\n" + "=" * 60)
print("汞同位素丰度统计分析")
print("=" * 60)
print(f"{'Isotope':<8} {'Mean (%)':<10} {'Std Dev (%)':<12} {'Typical (%)':<12} {'Deviation (%)':<12}")
print("-" * 60)
for iso, mean_val, std_val, typ_val in zip(mean_abundances.index, mean_abundances, std_abundances, typical_abundances):
    print(f"{iso:<8} {mean_val:<10.3f} {std_val:<12.3f} {typ_val:<12.3f} {(mean_val-typ_val):<12.3f}")
print("=" * 60)
print(f"总丰度和: {mean_abundances.sum():.2f}% (理论值应为100%)")
print(f"数据范围: {num_samples} 个独立样品")
print("图形已保存为: mercury_isotope_abundances_nature_style.png/pdf")
print("=" * 60)