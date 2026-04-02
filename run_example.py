"""
示例脚本：使用 cookie_cats.csv 演示完整 A/B 测试流程。

数据集：Cookie Cats 手游
- gate_30（A 组，对照）：关卡 30 设置关卡
- gate_40（B 组，实验）：关卡 40 设置关卡
- 指标：retention_1（次日留存率，0/1 二值指标）

运行方法：
    python run_example.py
"""
import os
import sys

# 确保从项目根目录导入
sys.path.insert(0, os.path.dirname(__file__))

from ab_testing import ABTestPipeline

DATA_PATH = os.path.join(os.path.dirname(__file__), "cookie_cats.csv")


def run_binary_example():
    """次日留存率：二值指标，同时运行频率派和贝叶斯。"""
    print("=" * 56)
    print("  示例 1：次日留存率（二值指标）")
    print("=" * 56)
    print()

    pipeline = ABTestPipeline(
        metric_type="binary",
        method="both",
        alpha=0.05,
        mde=0.005,              # 最小可检测提升：绝对值 0.5%
        loss_threshold=0.001,   # 期望损失阈值：0.1%
        historical_rate=0.44,   # 历史次日留存率
        prior_strength=100,     # 等效历史样本量
        n_samples=200_000,
    )

    result = pipeline.run_from_csv(
        filepath=DATA_PATH,
        group_col="version",
        metric_col="retention_1",
        control_label="gate_30",
        treatment_label="gate_40",
    )

    result.print_summary()
    print()

    # 保存图表到当前目录
    pipeline.plot(result, metric_label="次日留存率", save_dir=".", show=False)
    print()


def run_simulated_revenue_example():
    """
    模拟收入数据：连续指标贝叶斯（StudentT MCMC）。
    演示收入场景的 API 用法，实际使用时替换为真实数据。
    """
    print("=" * 56)
    print("  示例 2：模拟收入数据（连续指标，频率派）")
    print("=" * 56)
    print()

    import numpy as np
    rng = np.random.default_rng(42)

    # 模拟：A 组均值 50 元，B 组均值 55 元，重尾分布
    revenue_a = rng.exponential(scale=50, size=500)
    revenue_b = rng.exponential(scale=55, size=500)

    # 仅运行频率派（Welch t-test），避免需要安装 PyMC3
    pipeline = ABTestPipeline(
        metric_type="continuous",
        method="frequentist",
        alpha=0.05,
        mde=3.0,              # 最小可检测提升：3 元
        loss_threshold=1.0,
    )

    result = pipeline.run(revenue_a, revenue_b)
    result.print_summary()
    print()

    pipeline.plot(result, metric_label="收入（元）", save_dir=".", show=False)

    # 如需运行贝叶斯（需要 PyMC3），取消注释以下代码：
    # pipeline_bayes = ABTestPipeline(
    #     metric_type="continuous",
    #     method="bayesian",
    #     mde=3.0,
    #     loss_threshold=1.0,
    #     mcmc_draws=2000,
    # )
    # result_bayes = pipeline_bayes.run(revenue_a, revenue_b)
    # result_bayes.print_summary()
    # pipeline_bayes.plot(result_bayes, metric_label="收入（元）")


def run_bayesian_only_example():
    """仅运行贝叶斯分析（适合不关心 p 值、只看概率和损失的场景）。"""
    print("=" * 56)
    print("  示例 3：仅贝叶斯分析（7日留存率）")
    print("=" * 56)
    print()

    pipeline = ABTestPipeline(
        metric_type="binary",
        method="bayesian",
        mde=0.003,
        loss_threshold=0.001,
        historical_rate=0.19,   # 7 日留存率历史值约 19%
        prior_strength=100,
    )

    result = pipeline.run_from_csv(
        filepath=DATA_PATH,
        group_col="version",
        metric_col="retention_7",
        control_label="gate_30",
        treatment_label="gate_40",
    )

    result.print_summary()
    pipeline.plot(result, metric_label="7日留存率", save_dir=".", show=False)


if __name__ == "__main__":
    run_binary_example()
    run_simulated_revenue_example()
    run_bayesian_only_example()
