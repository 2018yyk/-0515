import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf
import os
from pandas.errors import ParserError
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def try_read_file(file_path):
    """
    自动判断文件格式并安全读取（支持csv和Excel），尝试多种编码，遇异常打印并退出。
    """
    try:
        if file_path.endswith('.csv'):
            for enc in ['utf-8-sig', 'gbk', 'gb18030', 'latin1']:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    print(f"成功用编码 {enc} 读取文件 {file_path}")
                    return df
                except (UnicodeDecodeError, ParserError):
                    continue
            raise ValueError("所有常见编码均无法解码该 CSV 文件。")
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            print(f"成功读取 Excel 文件 {file_path}")
            return df
        else:
            raise ValueError("仅支持 .csv 或 .xlsx 文件。")
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}，错误信息: {e}")
        exit()


def compare_statistics(real_df, gen_df, feature_columns):
    print("\n===== [1] 数据统计描述对比 =====")
    real_stats = real_df[feature_columns].describe()
    gen_stats = gen_df[feature_columns].describe()
    print("Real Data Statistics:\n", real_stats)
    print("Generated Data Statistics:\n", gen_stats)


def plot_distributions(real_df, gen_df, feature_columns, max_cols=5):
    print("\n===== [2] 特征分布直方图对比 =====")
    for col in feature_columns[:max_cols]:
        plt.figure(figsize=(6, 4))
        sns.histplot(real_df[col], label='Real', color='blue', stat="density", kde=True)
        sns.histplot(gen_df[col], label='Generated', color='red', stat="density", kde=True)
        plt.title(f"Distribution: {col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def compare_time_series(real_df, gen_df, feature_columns, max_cols=3):
    print("\n===== [3] 时间趋势对比图 =====")
    for col in feature_columns[:max_cols]:
        plt.figure(figsize=(10, 4))
        plt.plot(real_df['作业时间'], real_df[col], label='Real', alpha=0.7)
        plt.plot(gen_df['作业时间'], gen_df[col], label='Generated', alpha=0.7)
        plt.title(f"Time Series: {col}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_autocorrelation(df, feature_columns, max_cols=3):
    print("\n===== [4] 自相关函数（ACF）图 =====")
    for col in feature_columns[:max_cols]:
        plt.figure(figsize=(6, 3))
        plot_acf(df[col], lags=40)
        plt.title(f"ACF: {col}")
        plt.tight_layout()
        plt.show()


def pca_visualization(real_df, gen_df, feature_columns):
    print("\n===== [5] PCA 降维可视化 =====")
    real_feats = real_df[feature_columns].values
    gen_feats = gen_df[feature_columns].values
    all_feats = np.vstack([real_feats, gen_feats])
    labels = ['Real'] * len(real_feats) + ['Generated'] * len(gen_feats)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_feats)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, alpha=0.6)
    plt.title("PCA Projection: Real vs Generated")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_all(real_df, gen_df, feature_columns):
    """
    汇总性评估函数，逐项输出所有结果。
    """
    compare_statistics(real_df, gen_df, feature_columns)
    plot_distributions(real_df, gen_df, feature_columns)
    compare_time_series(real_df, gen_df, feature_columns)
    plot_autocorrelation(gen_df, feature_columns)
    pca_visualization(real_df, gen_df, feature_columns)
    print("\n===== ✅ 数据质量评估完成 =====")


if __name__ == "__main__":
    # === [主函数参数设置] ===
    real_data_path = './data/高炉运行参数.xlsx'            # 请替换为你的真实数据路径
    generated_data_path = './generated_minute_data.csv'  # 请替换为你保存的生成数据路径

    if not os.path.exists(real_data_path) or not os.path.exists(generated_data_path):
        print("❌ 未找到数据文件，请检查路径是否正确。")
        exit()

    # === [加载数据文件，无需手动指定编码] ===
    real_df = try_read_file(real_data_path)
    gen_df = try_read_file(generated_data_path)

    # === [作业时间列转时间类型] ===
    try:
        real_df['作业时间'] = pd.to_datetime(real_df['作业时间'])
        gen_df['作业时间'] = pd.to_datetime(gen_df['作业时间'])
    except Exception as e:
        print(f"⚠️ 时间字段转换失败: {e}")
        exit()

    # === [特征列自动识别：排除时间列] ===
    feature_columns = [col for col in real_df.columns if col != '作业时间']

    # === [执行评估函数] ===
    evaluate_all(real_df, gen_df, feature_columns)
