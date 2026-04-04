import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

# ========== Nature风格全局设置 ==========
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['legend.frameon'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300

# ========== 数据准备 ==========
mass_numbers = np.array([196, 198, 199, 200, 201, 202, 204])
abundances = np.array([0.15, 10.0, 16.9, 23.1, 13.2, 29.8, 6.9])

# 模拟误差（真实研究中通常有测量误差）
errors = np.array([0.008, 0.15, 0.12, 0.18, 0.11, 0.22, 0.09])

# ========== 创建图形（单栏宽度3.5英寸） ==========
fig, ax = plt.subplots(figsize=(3.5, 3.5))
scatter = ax.scatter(mass_numbers, abundances, 
                    s=80,                    # 点大小
                    c='#4C72B0',            # Nature蓝
                    marker='o',             # 圆形
                    edgecolors='black',     # 黑色边缘
                    linewidths=0.8,         # 边缘线宽
                    alpha=0.85,             # 轻微透明度
                    zorder=2,               # 图层顺序
                    label='Measured abundance')

# ========== 添加误差棒 ==========
errorbars = ax.errorbar(mass_numbers, abundances, yerr=errors,
                        fmt='none',           # 不显示连线
                        ecolor='black',       # 误差线颜色
                        capsize=3,            # 端帽大小
                        capthick=0.8,         # 端帽厚度
                        elinewidth=0.8,       # 误差线宽度
                        alpha=0.6,            # 透明度
                        label='±1 s.d. (n=3)')

# ========== 添加趋势线（可选，展示Nature常见风格） ==========
# 计算线性回归（用于展示整体趋势）
slope, intercept, r_value, p_value, std_err = stats.linregress(mass_numbers, abundances)
x_line = np.array([195, 205])
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 
        color='#CC6677',          # Nature红褐色
        linestyle='--', 
        linewidth=0.8,
        alpha=0.7,
        label=f'Linear fit (R² = {r_value**2:.2f})')

# ========== 坐标轴设置 ==========
ax.set_xlabel('Mass number (A)', fontsize=9, fontweight='normal', labelpad=5)
ax.set_ylabel('Abundance (%)', fontsize=9, fontweight='normal', labelpad=5)

# 设置x轴范围（留出边距）
ax.set_xlim(194.5, 205.5)

# 设置x轴刻度（显示所有质量数）
ax.set_xticks(mass_numbers)
ax.set_xticklabels([f'^{{{m}}}Hg' for m in mass_numbers], rotation=45, ha='right', fontsize=7)

# 设置y轴范围
ax.set_ylim(0, 35)
ax.set_yticks(np.arange(0, 36, 5))
ax.set_yticklabels([f'{y:.0f}' for y in np.arange(0, 36, 5)], fontsize=8)

# ========== 添加网格线（浅色，仅y轴） ==========
ax.yaxis.grid(True, linestyle='-', alpha=0.15, linewidth=0.3, color='gray')
ax.set_axisbelow(True)  # 网格线在数据点下方

# ========== 添加图例 ==========
legend = ax.legend(loc='upper left', fontsize=7, handlelength=1.5, 
                   handletextpad=0.8, borderaxespad=0.5)

# ========== 添加注释和统计信息 ==========
# 注释1：样品信息
ax.text(0.02, 0.97, 'n = 7 isotopes', transform=ax.transAxes,
        fontsize=7, verticalalignment='top', fontstyle='italic',
        color='#555555', bbox=dict(boxstyle='round', facecolor='white', 
                                   edgecolor='none', alpha=0.7))

# 注释2：总丰度和
total_abundance = np.sum(abundances)
ax.text(0.98, 0.97, f'Total: {total_abundance:.1f}%', transform=ax.transAxes,
        fontsize=7, verticalalignment='top', horizontalalignment='right',
        fontstyle='italic', color='#555555', bbox=dict(boxstyle='round', 
                                   facecolor='white', edgecolor='none', alpha=0.7))

# 注释3：主要同位素标注（标注丰度最高的两个点）
max_idx = np.argmax(abundances)
second_idx = np.argsort(abundances)[-2]
for idx in [max_idx, second_idx]:
    ax.annotate(f'{abundances[idx]:.1f}%', 
                xy=(mass_numbers[idx], abundances[idx]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=6, color='black', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='none', alpha=0.7))

# ========== 添加插入小图（可选，展示数据分布） ==========
# 创建子图插入（Nature常见做法）
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
inset = inset_axes(ax, width="30%", height="30%", loc='lower right',
                   bbox_to_anchor=(0.02, 0.02, 0.98, 0.98),
                   bbox_transform=ax.transAxes, borderpad=1)

# 绘制丰度分布直方图
inset.hist(abundances, bins=5, color='#4C72B0', edgecolor='black', 
           linewidth=0.5, alpha=0.7, orientation='horizontal')
inset.set_xlabel('Count', fontsize=6)
inset.set_ylabel('Abundance (%)', fontsize=6)
inset.tick_params(axis='both', labelsize=5)
inset.set_facecolor('#F8F9FA')

# ========== 美化调整 ==========
# 设置坐标轴颜色和粗细
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)

# 调整布局，确保所有元素完整显示
plt.tight_layout(pad=0.5)

# ========== 保存图片 ==========
plt.savefig('mercury_isotopes_nature_scatter.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('mercury_isotopes_nature_scatter.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# ========== 显示图形 ==========
plt.show()

# ========== 打印统计分析 ==========
print("\n" + "="*60)
print("汞同位素丰度统计分析 (Nature风格)")
print("="*60)
print(f"同位素数量: {len(mass_numbers)}")
print(f"丰度范围: {abundances.min():.2f}% - {abundances.max():.2f}%")
print(f"平均丰度: {abundances.mean():.2f}% ± {abundances.std():.2f}%")
print(f"总丰度和: {total_abundance:.2f}%")
print(f"\n线性回归分析:")
print(f"  斜率: {slope:.4f}")
print(f"  截距: {intercept:.2f}")
print(f"  R²: {r_value**2:.3f}")
print(f"  p值: {p_value:.4f}")
print("="*60)
print("图形已保存为:")
print("  - mercury_isotopes_nature_scatter.png (600 DPI)")
print("  - mercury_isotopes_nature_scatter.pdf (矢量格式)")
print("="*60)