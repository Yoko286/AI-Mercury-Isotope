# -*- coding: utf-8 -*-
"""
USGS 地震数据探索性分析
用非线性回归方法，探索经纬度与震级的关系
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm

# ================= 配置区 =================
FILE_PATH = r"D:\wyc\Programming\Talent Program\AI-Mercury-Isotope\usgs_earthquake.csv"
OUTPUT_DIR = r"D:\wyc\Programming\Talent Program\AI-Mercury-Isotope"
# =========================================

# 中文字体
from matplotlib import font_manager

font_path = r"C:\Windows\Fonts\msyh.ttc"  # 改成你系统里实际存在的路径
my_font = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = [my_font.get_name()]
plt.rcParams['axes.unicode_minus'] = False


# ========== 1. 加载和清洗数据 ==========
def load_and_clean(path):
    df = pd.read_csv(path)
    print("=" * 50)
    print("数据加载成功")
    print("=" * 50)
    print(f"数据形状：{df.shape}")
    print(f"\n列名：{list(df.columns)}")
    print(f"\n缺失值：\n{df.isnull().sum()}")

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    def depth_class(d):
        if pd.isna(d):
            return '未知'
        if d < 70:
            return '浅源'
        elif d < 300:
            return '中源'
        else:
            return '深源'

    df['depth_class'] = df['depth'].apply(depth_class)

    print(f"\n深度分类统计：")
    print(df['depth_class'].value_counts())

    return df


# ========== 2. 数据分布可视化 ==========
def plot_distributions(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['magnitude'].dropna(), bins=30,
                 color='steelblue', edgecolor='black')
    axes[0].set_xlabel('震级')
    axes[0].set_ylabel('频次')
    axes[0].set_title('震级分布')

    axes[1].hist(df['depth'].dropna(), bins=30,
                 color='coral', edgecolor='black')
    axes[1].set_xlabel('深度 (km)')
    axes[1].set_ylabel('频次')
    axes[1].set_title('深度分布')

    plt.tight_layout()
    plt.savefig(f"{out_dir}\\01_distributions.png", dpi=150)
    plt.close()
    print("已保存：01_distributions.png")


# ========== 3. 全球地震分布地图 ==========
def plot_global_map(df, out_dir):
    plt.figure(figsize=(15, 8))
    scatter = plt.scatter(
        df['longitude'], df['latitude'],
        c=df['depth'], cmap='viridis_r',
        s=df['magnitude'] ** 2,
        alpha=0.6, edgecolors='none'
    )
    plt.colorbar(scatter, label='深度 (km)')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('全球地震分布（点大小=震级，颜色=深度）')
    plt.grid(alpha=0.3)
    plt.savefig(f"{out_dir}\\02_global_map.png", dpi=150)
    plt.close()
    print("已保存：02_global_map.png")


# ========== 4. 深源 vs 浅源对比 ==========
def plot_depth_comparison(df, out_dir):
    order = ['浅源', '中源', '深源']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(x='depth_class', y='magnitude', data=df,
                order=order, ax=axes[0])
    axes[0].set_title('不同深度类别的震级分布（箱线图）')

    sns.violinplot(x='depth_class', y='magnitude', data=df,
                   order=order, ax=axes[1])
    axes[1].set_title('不同深度类别的震级分布（小提琴图）')

    plt.tight_layout()
    plt.savefig(f"{out_dir}\\03_depth_vs_magnitude.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    colors = {'浅源': 'red', '中源': 'orange', '深源': 'blue'}
    for cls, color in colors.items():
        subset = df[df['depth_class'] == cls]
        if len(subset) > 0:
            plt.scatter(subset['depth'], subset['magnitude'],
                        c=color, label=cls, alpha=0.6, s=30)

    plt.xlabel('深度 (km)')
    plt.ylabel('震级')
    plt.title('深度 vs 震级')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{out_dir}\\04_depth_magnitude_scatter.png", dpi=150)
    plt.close()
    print("已保存：03_depth_vs_magnitude.png, 04_depth_magnitude_scatter.png")


# ========== 5. 相关性分析 ==========
def correlation_analysis(df, out_dir):
    numeric_df = df[['magnitude', 'depth', 'longitude', 'latitude']].dropna()

    corr_matrix = numeric_df.corr(method='pearson')
    print("\n皮尔逊相关系数矩阵：")
    print(corr_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.3f', square=True)
    plt.title('特征相关性热力图')
    plt.savefig(f"{out_dir}\\05_correlation_heatmap.png", dpi=150)
    plt.close()

    return corr_matrix


# ========== 6. 非线性回归：用经纬度预测震级 ==========
def nonlinear_regression(df, out_dir):
    data = df[['magnitude', 'longitude', 'latitude', 'depth']].dropna()

    # 特征：经纬度（可以加上深度）
    X = data[['longitude', 'latitude']]
    y = data['magnitude']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 定义多个非线性模型
    models = {
        # 1. 多项式回归（Ridge 正则化），degree=3
        '多项式回归(3阶+Ridge)': Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ]),
        # 2. 随机森林
        '随机森林': RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        # 3. 梯度提升
        '梯度提升': GradientBoostingRegressor(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, random_state=42
        ),
        # 4. 支持向量回归（RBF 核）
        '支持向量回归(RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=10, epsilon=0.2))
        ]),
    }

    results = {}
    predictions = {}

    print("\n" + "=" * 50)
    print("非线性回归模型对比（特征：经度+纬度）")
    print("=" * 50)

    for name, model in models.items():
        print(f"\n训练 {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # 5 折交叉验证（在训练集上）
        try:
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
        except Exception as e:
            cv_mean = np.nan
            print(f"  交叉验证失败: {e}")

        results[name] = {
            'R2': r2,
            'RMSE': rmse,
            'CV_R2_mean': cv_mean
        }
        predictions[name] = y_pred

        print(f"  测试集 R²   : {r2:.4f}")
        print(f"  测试集 RMSE : {rmse:.4f}")
        print(f"  5折CV R²均值: {cv_mean:.4f}")

    # 汇总表
    print("\n" + "=" * 50)
    print("模型性能汇总")
    print("=" * 50)
    summary = pd.DataFrame(results).T
    summary = summary.sort_values('R2', ascending=False)
    print(summary)

    # ===== 可视化：4 个模型的预测对比 =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, y_pred, alpha=0.4, s=15)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('实际震级')
        ax.set_ylabel('预测震级')
        ax.set_title(f'{name}\nR²={results[name]["R2"]:.3f}, '
                     f'RMSE={results[name]["RMSE"]:.3f}')

    plt.tight_layout()
    plt.savefig(f"{out_dir}\\06_nonlinear_prediction.png", dpi=150)
    plt.close()
    print("\n已保存：06_nonlinear_prediction.png")

    # ===== 可视化：模型性能对比柱状图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    r2_values = [results[n]['R2'] for n in names]
    rmse_values = [results[n]['RMSE'] for n in names]

    axes[0].barh(names, r2_values, color='steelblue')
    axes[0].set_xlabel('R²')
    axes[0].set_title('各模型测试集 R² 对比')
    axes[0].axvline(0, color='black', lw=0.8)

    axes[1].barh(names, rmse_values, color='coral')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('各模型测试集 RMSE 对比')

    plt.tight_layout()
    plt.savefig(f"{out_dir}\\07_model_comparison.png", dpi=150)
    plt.close()
    print("已保存：07_model_comparison.png")

    return results, summary


# ========== 7. 特征重要性（仅树模型） ==========
def feature_importance(df, out_dir):
    data = df[['magnitude', 'longitude', 'latitude', 'depth']].dropna()

    # 加入更多特征
    X = data[['longitude', 'latitude', 'depth']]
    y = data['magnitude']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    features = X.columns

    print("\n随机森林特征重要性：")
    for f, imp in sorted(zip(features, importances),
                         key=lambda x: -x[1]):
        print(f"  {f}: {imp:.4f}")

    plt.figure(figsize=(8, 5))
    plt.barh(features, importances, color='seagreen')
    plt.xlabel('重要性')
    plt.title('特征重要性（随机森林）')
    plt.tight_layout()
    plt.savefig(f"{out_dir}\\08_feature_importance.png", dpi=150)
    plt.close()
    print("已保存：08_feature_importance.png")


# ========== 主流程 ==========
def main():
    df = load_and_clean(FILE_PATH)

    print("\n开始生成图表...")
    plot_distributions(df, OUTPUT_DIR)
    plot_global_map(df, OUTPUT_DIR)
    plot_depth_comparison(df, OUTPUT_DIR)
    correlation_analysis(df, OUTPUT_DIR)
    nonlinear_regression(df, OUTPUT_DIR)
    feature_importance(df, OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("全部分析完成！")
    print("=" * 50)
    print(f"图表已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()