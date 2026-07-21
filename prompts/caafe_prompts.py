

from __future__ import annotations

from textwrap import dedent
from typing import Sequence

import pandas as pd


PROMPT_VERSION = "caafe_feature_generation_v1.0"

SYSTEM_PROMPT = (
    "你只生成安全的 pandas 特征工程代码。"
    "不要解释，不要输出代码块以外的内容。"
    "只能返回一个 Python 代码块，代码块中必须包含三个候选特征。"
)

CAAFE_USER_TEMPLATE = dedent(
    r"""
    你是一名数据科学专家，正在做 Context-Aware Automated Feature Engineering，也就是 CAAFE。

    任务背景：
    {dataset_description}

    当前任务：
    根据已有特征，为回归预测目标 {target} 一次构造三个新的、有物理或统计意义的候选特征。
    程序会分别验证这三个候选特征，并只保留交叉验证 MAPE 改善最大的一个。

    当前可用特征：
    {used_features}

    当前特征的类型、缺失率和样本值：
    {feature_summary}

    {accepted_part}

    {feedback_part}

    严格要求：
    1. dataframe 名称必须是 df。
    2. 目标列 {target} 不在 df 中，禁止使用目标列。
    3. 一次生成三个候选新特征，名称必须分别是：{feature_names_text}。
    4. 三个候选特征必须彼此不同，且不能互相引用，只能使用当前可用特征。
    5. 只能使用当前可用特征，不能使用不存在的列。
    6. 可以使用 +、-、*、np.log1p、np.abs、np.sqrt，也允许使用平方 **2。
    7. 禁止使用除法 /，避免除零和数值不稳定。
    8. 禁止使用除 **2 以外的其他幂运算。
    9. 如果需要平方，可以写 df['A'] ** 2，也可以写 df['A'] * df['A']。
    10. 不要 import。
    11. 不要定义函数。
    12. 不要读取文件。
    13. 不要输出代码块以外的解释文字。
    14. 每个新特征必须各用一行代码完成，不要使用临时变量。
    15. 代码要尽量稳健，避免 NaN 和 inf。
    16. 返回一个 Python 代码块。
    17. 每个候选特征前都要用注释写明 Feature 和 Usefulness。

    返回格式必须类似：

    ```python
    # Feature: {feature_1}
    # Usefulness: 这个特征为什么可能有助于预测 {target}
    df['{feature_1}'] = np.sqrt(np.abs(df['某个特征'])) + df['另一个特征']

    # Feature: {feature_2}
    # Usefulness: 这个特征为什么可能有助于预测 {target}
    df['{feature_2}'] = np.log1p(np.abs(df['某个特征'])) * df['另一个特征']

    # Feature: {feature_3}
    # Usefulness: 这个特征为什么可能有助于预测 {target}
    df['{feature_3}'] = df['某个特征'] * df['另一个特征']
    ```
    """
).strip()


def summarize_features_for_prompt(
    X_df: pd.DataFrame,
    used_features: Sequence[str],
    sample_size: int = 5,
) -> str:
    """将特征类型、缺失率和样本值整理为 Prompt 文本。"""
    lines = []

    for col in used_features:
        if col not in X_df.columns:
            raise KeyError("数据中不存在特征列：{}".format(col))

        dtype = str(X_df[col].dtype)
        missing_rate = X_df[col].isna().mean() * 100
        values = X_df[col].dropna()

        if len(values) > 0:
            samples = values.sample(
                n=min(sample_size, len(values)),
                random_state=42,
            ).tolist()
        else:
            samples = []

        lines.append(
            "{}: dtype={}, missing={:.2f}%, samples={}".format(
                col,
                dtype,
                missing_rate,
                samples,
            )
        )

    return "\n".join(lines)


def _build_accepted_part(accepted_codes: Sequence[str]) -> str:
    if not accepted_codes:
        return ""

    blocks = ["已经被交叉验证证明有效并接受的特征工程代码如下："]

    for index, code in enumerate(accepted_codes, start=1):
        blocks.append(
            "# Accepted feature code {}\n{}".format(index, code)
        )

    return "\n\n".join(blocks)


def _build_feedback_part(feedback: str) -> str:
    if not feedback:
        return ""
    return "上一轮反馈：\n{}".format(feedback)


def build_caafe_prompt(
    *,
    X_df: pd.DataFrame,
    used_features: Sequence[str],
    new_feature_names: Sequence[str],
    accepted_codes: Sequence[str],
    dataset_description: str,
    target: str,
    feedback: str = "",
) -> str:
    """构造单轮 CAAFE 特征生成 Prompt。"""
    if len(new_feature_names) != 3:
        raise ValueError(
            "当前 Prompt 固定要求生成 3 个特征，但收到 {} 个特征名。".format(
                len(new_feature_names)
            )
        )

    feature_summary = summarize_features_for_prompt(
        X_df=X_df,
        used_features=used_features,
    )

    feature_names_text = ", ".join(
        "df['{}']".format(name) for name in new_feature_names
    )

    return CAAFE_USER_TEMPLATE.format(
        dataset_description=dataset_description.strip(),
        target=target,
        used_features=", ".join(used_features),
        feature_summary=feature_summary,
        accepted_part=_build_accepted_part(accepted_codes),
        feedback_part=_build_feedback_part(feedback),
        feature_names_text=feature_names_text,
        feature_1=new_feature_names[0],
        feature_2=new_feature_names[1],
        feature_3=new_feature_names[2],
    )
