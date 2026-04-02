"""
A/B 测试分析平台 — Streamlit + streamlit-echarts

运行方式：
    streamlit run streamlit_app.py
"""
import os
import ctypes
import threading
import time
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts


# ── 后台线程管理 ───────────────────────────────────────────────────
def _force_stop_thread(thread: threading.Thread) -> bool:
    """
    通过 ctypes 向目标线程注入 SystemExit 异常实现强制停止。
    对纯 Python 代码（包括 PyMC 的 Python 层回调）有效；
    C 扩展密集区域可能延迟响应，但线程最终会退出。
    """
    if not thread.is_alive():
        return True
    tid = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(SystemExit),
    )
    return res >= 1


def _analysis_worker(pipeline, data_a, data_b, state: dict) -> None:
    """后台 daemon 线程入口，结果写回共享 state dict。"""
    try:
        result = pipeline.run(data_a, data_b)
        state["result"]   = result
        state["pipeline"] = pipeline
        state["status"]   = "done"
    except SystemExit:
        state["status"] = "stopped"
    except Exception as e:
        state["error"]  = str(e)
        state["status"] = "error"

from ab_testing import ABTestPipeline
from ab_testing.visualizer_echarts import (
    posterior_chart,
    delta_chart,
    loss_chart,
    freq_chart,
)

# ── 页面配置 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B 测试分析平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "cookie_cats.csv")

# ── 侧边栏：参数配置 ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 参数配置")

    method = st.radio(
        "分析方法",
        ["both", "bayesian", "frequentist"],
        format_func=lambda x: {
            "both": "两种方法对比",
            "bayesian": "仅贝叶斯",
            "frequentist": "仅频率派",
        }[x],
    )

    metric_type = st.radio(
        "指标类型",
        ["binary", "continuous"],
        format_func=lambda x: {
            "binary": "二值（转化率 / 留存率）",
            "continuous": "连续（收入 / 金额）",
        }[x],
    )

    st.divider()
    st.subheader("统计参数")
    alpha = st.slider("显著性水平 α（频率派）", 0.01, 0.10, 0.05, 0.01)
    mde = st.number_input(
        "MDE 最小可检测提升",
        value=0.005 if metric_type == "binary" else 3.0,
        format="%.4f",
        help="与指标同量纲，二值用绝对差值，连续用金额差",
    )
    loss_threshold = st.number_input(
        "期望损失阈值（贝叶斯停止准则）",
        value=0.001 if metric_type == "binary" else 1.0,
        format="%.4f",
    )

    st.divider()
    st.subheader("贝叶斯先验")
    st.caption(
        "⚠️ 先验必须来自**实验开始前**的独立历史数据或业务经验，"
        "**不能**使用当前 AB 测试数据估算，否则数据被重复使用，后验失真。"
    )

    prior_source = st.radio(
        "先验参数来源",
        ["手动输入", "上传历史数据自动估算"],
        horizontal=True,
        help="历史数据：实验前的同指标数据集（独立于当前 AB 测试）",
    )

    # 先验参数默认值（无信息先验）
    historical_rate = 0.5
    historical_mean = None
    historical_std  = None
    nu_expected     = 30.0
    prior_strength  = 2

    if prior_source == "手动输入":
        if metric_type == "binary":
            historical_rate = st.slider(
                "历史转化率",
                0.01, 0.99, 0.44, 0.01,
                help="来源：历史报表或上一期实验结论，不得使用当前测试数据均值",
            )
        else:  # continuous
            historical_mean = st.number_input(
                "历史均值（mu）",
                value=50.0, format="%.2f",
                help="与指标同量纲，来自实验前历史数据的均值",
            )
            historical_std = st.number_input(
                "历史标准差（sigma）",
                value=30.0, min_value=0.01, format="%.2f",
                help="历史数据的标准差，控制先验的不确定性范围",
            )
            nu_expected = st.slider(
                "自由度先验期望（nu）",
                min_value=3, max_value=100, value=30,
                help=(
                    "控制 StudentT 尾部厚度：\n"
                    "• nu ≈ 3–5：极厚尾，适合高度离群的收入数据\n"
                    "• nu ≈ 10–20：中等厚尾，常见收入场景\n"
                    "• nu ≈ 30+：接近正态分布"
                ),
            )
        prior_strength = st.slider(
            "先验强度（等效历史样本量）",
            min_value=1, max_value=500, value=100,
            help="越大表示越信任历史数据；样本量充足时建议设低（≤10）",
        )

    else:  # 上传历史数据自动估算
        hist_file = st.file_uploader(
            "上传历史数据 CSV（实验前数据，独立数据集）",
            type=["csv"],
            key="hist_upload",
        )
        if hist_file:
            hist_df   = pd.read_csv(hist_file)
            hist_col  = st.selectbox("选择指标列", hist_df.columns.tolist(), key="hist_col")
            hist_vals = hist_df[hist_col].dropna().values.astype(float)

            if metric_type == "binary":
                historical_rate = float(hist_vals.mean())
                st.success(f"估算历史转化率：**{historical_rate:.4f}**（n={len(hist_vals):,}）")
            else:
                historical_mean = float(hist_vals.mean())
                historical_std  = float(hist_vals.std())
                st.success(
                    f"估算历史均值：**{historical_mean:.4f}**  "
                    f"标准差：**{historical_std:.4f}**（n={len(hist_vals):,}）"
                )

            prior_strength = st.slider(
                "先验强度",
                min_value=1, max_value=500,
                value=min(max(int(len(hist_vals) / 10), 1), 200),
                help="建议设为历史样本量的 1/10 左右，避免先验过强压制实验数据",
            )
        else:
            st.info("请上传历史数据文件，将自动估算先验参数")


# ── 主区域 ────────────────────────────────────────────────────────
st.title("🧪 A/B 测试分析平台")
st.caption("频率派（Z-test / Welch t-test）× 贝叶斯（Beta-Bernoulli / StudentT）双方法对比")
st.divider()

# ── 数据加载 ──────────────────────────────────────────────────────
st.subheader("① 加载数据")
data_source = st.radio(
    "数据来源",
    ["示例数据（Cookie Cats）", "上传 CSV"],
    horizontal=True,
)

df = None
group_col = metric_col = control_label = treatment_label = None

if data_source == "示例数据（Cookie Cats）":
    df = pd.read_csv(SAMPLE_CSV)
    group_col = "version"
    metric_col = st.selectbox("选择指标列", ["retention_1", "retention_7"])
    control_label, treatment_label = "gate_30", "gate_40"

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.caption(f"对照组：`{control_label}`  |  实验组：`{treatment_label}`")
    with col_info2:
        st.caption(f"共 {len(df):,} 条记录")
    st.dataframe(df.head(5), use_container_width=True)

else:
    uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(5), use_container_width=True)

        cols = df.columns.tolist()

        # 自动推断分组列：优先选唯一值少（2–10）且为字符串/低基数的列
        def _guess_group_col(dataframe: pd.DataFrame) -> str:
            for c in dataframe.columns:
                n_unique = dataframe[c].nunique()
                if 2 <= n_unique <= 10 and dataframe[c].dtype == object:
                    return c
            # 退而求其次：唯一值最少的列
            return min(dataframe.columns, key=lambda c: dataframe[c].nunique())

        # 自动推断指标列：优先选数值型且不是 ID 类（唯一值多）的列
        def _guess_metric_col(dataframe: pd.DataFrame, exclude: str) -> str:
            numeric_cols = [
                c for c in dataframe.select_dtypes(include="number").columns
                if c != exclude and dataframe[c].nunique() < len(dataframe) * 0.9
            ]
            return numeric_cols[0] if numeric_cols else cols[0]

        default_group  = _guess_group_col(df)
        default_metric = _guess_metric_col(df, exclude=default_group)

        c1, c2 = st.columns(2)
        with c1:
            group_col = st.selectbox(
                "分组列（A/B 标签列）", cols,
                index=cols.index(default_group),
            )
            metric_col = st.selectbox(
                "指标列", [c for c in cols if c != group_col],
                index=max(0, [c for c in cols if c != group_col].index(default_metric))
                if default_metric in cols else 0,
            )
        with c2:
            if group_col:
                groups = df[group_col].unique().tolist()
                control_label   = st.selectbox("对照组标签（A）", groups)
                treatment_label = st.selectbox(
                    "实验组标签（B）", [g for g in groups if g != control_label]
                )

st.divider()

# ── 运行分析 ──────────────────────────────────────────────────────
st.subheader("② 运行分析")

ready   = df is not None and all([group_col, metric_col, control_label, treatment_label])
running = st.session_state.get("running", False)

col_btn, col_status = st.columns([2, 5])

with col_btn:
    if not running:
        run_btn = st.button("🚀 运行分析", type="primary",
                            disabled=not ready, use_container_width=True)
    else:
        stop_btn = st.button("⏹ 停止分析", type="secondary",
                             use_container_width=True)

with col_status:
    status_placeholder = st.empty()

# ── 启动分析 ──────────────────────────────────────────────────────
if not running and ready and "run_btn" in dir() and run_btn:
    data_a = df[df[group_col] == control_label][metric_col].values.astype(float)
    data_b = df[df[group_col] == treatment_label][metric_col].values.astype(float)

    pipeline_kwargs = dict(
        metric_type=metric_type,
        method=method,
        alpha=alpha,
        mde=float(mde),
        loss_threshold=float(loss_threshold),
        prior_strength=prior_strength,
    )
    if metric_type == "binary":
        pipeline_kwargs["historical_rate"] = historical_rate
    else:
        if historical_mean is not None:
            pipeline_kwargs["historical_mean"] = historical_mean
        if historical_std is not None:
            pipeline_kwargs["historical_std"] = historical_std
        pipeline_kwargs["nu_expected"] = float(nu_expected)

    pipeline = ABTestPipeline(**pipeline_kwargs)

    # 共享状态 dict（线程与主线程通信）
    worker_state: dict = {"status": "running", "result": None,
                          "pipeline": None, "error": None}
    thread = threading.Thread(
        target=_analysis_worker,
        args=(pipeline, data_a, data_b, worker_state),
        daemon=True,
    )
    thread.start()

    st.session_state["running"]      = True
    st.session_state["worker_state"] = worker_state
    st.session_state["worker_thread"] = thread
    st.session_state["metric_col"]   = metric_col
    st.rerun()

# ── 停止分析 ──────────────────────────────────────────────────────
if running and "stop_btn" in dir() and stop_btn:
    thread = st.session_state.get("worker_thread")
    if thread:
        _force_stop_thread(thread)
    st.session_state["running"] = False
    st.session_state.pop("worker_state",  None)
    st.session_state.pop("worker_thread", None)
    status_placeholder.warning("⚠️ 分析已强制停止")
    st.rerun()

# ── 轮询运行状态 ──────────────────────────────────────────────────
if running:
    worker_state = st.session_state.get("worker_state", {})
    status = worker_state.get("status", "running")

    if status == "running":
        status_placeholder.info("⏳ 分析运行中...")
        time.sleep(0.5)
        st.rerun()

    elif status == "done":
        st.session_state["result"]   = worker_state["result"]
        st.session_state["pipeline"] = worker_state["pipeline"]
        st.session_state["running"]  = False
        st.session_state.pop("worker_state",  None)
        st.session_state.pop("worker_thread", None)
        status_placeholder.success("✅ 分析完成")
        st.rerun()

    elif status == "error":
        st.session_state["running"] = False
        status_placeholder.error(f"❌ 分析出错：{worker_state.get('error')}")
        st.session_state.pop("worker_state",  None)
        st.session_state.pop("worker_thread", None)
        st.rerun()

    elif status == "stopped":
        st.session_state["running"] = False
        st.session_state.pop("worker_state",  None)
        st.session_state.pop("worker_thread", None)
        status_placeholder.warning("⚠️ 分析已停止")
        st.rerun()

# ── 展示结果 ──────────────────────────────────────────────────────
if "result" in st.session_state:
    result   = st.session_state["result"]
    pipeline = st.session_state["pipeline"]
    m_label  = st.session_state.get("metric_col", "指标")

    st.divider()
    st.subheader("③ 分析结果")

    # 决策 Banner
    def _decision_banner(decision: str, method_name: str):
        if "上线 B" in decision:
            st.success(f"**{method_name}决策：{decision}**  — B 组表现更优")
        elif "保持 A" in decision:
            st.info(f"**{method_name}决策：{decision}**  — A 组表现更优或无显著差异")
        else:
            st.warning(f"**{method_name}决策：{decision}**  — 需要更多数据")

    if result.frequentist and result.decision_freq:
        _decision_banner(result.decision_freq, "【频率派】")
    if result.bayesian and result.decision_bayes:
        _decision_banner(result.decision_bayes, "【贝叶斯】")

    # KPI 卡片（自定义 HTML，字体紧凑，小屏友好）
    def _card(label: str, value: str, sub: str = "", sub_color: str = "#555") -> str:
        return (
            f'<div style="background:#f5f6fa;border-radius:8px;padding:8px 10px;'
            f'border:1px solid #e2e4ea;min-width:0;overflow:hidden">'
            f'<div style="font-size:0.68rem;color:#888;font-weight:500;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>'
            f'<div style="font-size:0.95rem;font-weight:700;color:#1a1a2e;'
            f'margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{value}</div>'
            + (f'<div style="font-size:0.65rem;color:{sub_color};margin-top:1px;'
               f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{sub}</div>' if sub else "")
            + "</div>"
        )

    def _kpi_row(cards: list[tuple]) -> None:
        cols = len(cards)
        html = (
            f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
            f'gap:8px;margin-bottom:10px">'
        )
        for item in cards:
            html += _card(*item)
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("#### 核心指标")
    if result.frequentist:
        f = result.frequentist
        sig_sub   = ("显著 ✓" if f.significant else "不显著 ✗")
        sig_color = "#27ae60" if f.significant else "#888"
        _kpi_row([
            ("A 组均值",  f"{f.mean_a:.4f}"),
            ("B 组均值",  f"{f.mean_b:.4f}", f"delta {f.delta:+.4f}"),
            ("p 值",      f"{f.p_value:.4f}", sig_sub, sig_color),
            ("效应量",    f"{f.effect_size:.4f}"),
            ("95% CI",   f"[{f.ci[0]:+.4f}, {f.ci[1]:+.4f}]"),
        ])
    if result.bayesian:
        b = result.bayesian
        _kpi_row([
            ("A 后验均值",    f"{b.mean_a:.4f}"),
            ("B 后验均值",    f"{b.mean_b:.4f}", f"delta {b.delta_mean:+.4f}"),
            ("P(B > A)",     f"{b.prob_b_better:.1%}"),
            ("P(delta>MDE)", f"{b.prob_practical:.1%}"),
            ("选A期望损失",   f"{b.expected_loss_a:.5f}"),
            ("选B期望损失",   f"{b.expected_loss_b:.5f}"),
        ])

    st.markdown("---")

    # ── 频率派图表 ──────────────────────────────────────────────
    if result.frequentist:
        st.markdown("#### 频率派：均值对比 & 置信区间")
        st_echarts(
            options=freq_chart(result.frequentist, metric_label=m_label),
            height="380px",
            key="chart_freq",
        )

    # ── 贝叶斯图表 ──────────────────────────────────────────────
    if result.bayesian:
        b = result.bayesian
        st.markdown("#### 贝叶斯：后验分布 & 决策指标")

        tab1, tab2, tab3 = st.tabs(["后验分布", "Delta 分布", "期望损失"])

        with tab1:
            st.caption(f"A 组后验均值 {b.mean_a:.4f}  |  B 组后验均值 {b.mean_b:.4f}")
            st_echarts(
                options=posterior_chart(b, metric_label=m_label),
                height="400px",
                key="chart_posterior",
            )

        with tab2:
            st.caption(
                f"P(B > A) = {b.prob_b_better:.1%}  |  "
                f"P(delta > MDE={pipeline.mde}) = {b.prob_practical:.1%}"
            )
            st_echarts(
                options=delta_chart(b, mde=pipeline.mde, metric_label=m_label),
                height="400px",
                key="chart_delta",
            )

        with tab3:
            st.caption(
                f"选 A 的期望损失：{b.expected_loss_a:.6f}  |  "
                f"选 B 的期望损失：{b.expected_loss_b:.6f}  |  "
                f"阈值：{pipeline.loss_threshold}"
            )
            st_echarts(
                options=loss_chart(b, loss_threshold=pipeline.loss_threshold),
                height="380px",
                key="chart_loss",
            )

    # ── 文字摘要（可折叠）──────────────────────────────────────
    with st.expander("📋 查看完整文字摘要"):
        st.code(result.summary(), language=None)
