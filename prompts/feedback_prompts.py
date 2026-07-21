# -*- coding: utf-8 -*-
"""根据实验结果构造下一轮大模型反馈。"""

from __future__ import annotations

from typing import Sequence


def all_candidates_failed_feedback() -> str:
    return (
        "上一轮三个候选特征全部失败。"
        "请生成更简单、更稳健、彼此独立的三个候选特征。"
    )


def gp_feature_accepted_feedback(
    *,
    candidate_names: Sequence[str],
    parents: Sequence[str],
    generations: int,
    feature_name: str,
    base_mape: float,
    final_mape: float,
    feature_code: str,
) -> str:
    return (
        "上一轮 DeepSeek 生成了三个候选特征 {}。"
        "其中改善最好的两个候选为 {}。"
        "随后使用 DEAP 遗传编程将这两个候选表达式抽象为树，"
        "经过 {} 代交叉和变异后，得到更优特征 {}。"
        "它使 CV MAPE 从 {:.4f}% 降低到 {:.4f}%，已接受。"
        "后续生成特征时，可以优先结合两个父代特征的结构模式，"
        "例如复合交互、非线性变换和稳健绝对值变换。\n{}"
    ).format(
        list(candidate_names),
        list(parents),
        generations,
        feature_name,
        base_mape,
        final_mape,
        feature_code,
    )


def llm_feature_accepted_feedback(
    *,
    candidate_names: Sequence[str],
    feature_name: str,
    base_mape: float,
    final_mape: float,
) -> str:
    return (
        "上一轮生成了三个候选特征 {}，其中 {} 最优，"
        "使 CV MAPE 从 {:.4f}% 降低到 {:.4f}%，已接受。"
        "DEAP 遗传编程没有找到比该 LLM 原始候选更优的表达式。"
    ).format(
        list(candidate_names),
        feature_name,
        base_mape,
        final_mape,
    )


def candidates_rejected_feedback(
    *,
    candidate_names: Sequence[str],
    best_feature_name: str,
    base_mape: float,
    final_mape: float,
) -> str:
    return (
        "上一轮生成的三个候选特征 {} 以及基于 Top-2 候选进行 DEAP "
        "遗传编程进化得到的特征，都没有带来足够提升。"
        "最佳特征 {} 的 CV MAPE 从 {:.4f}% 变为 {:.4f}%。"
        "请换三种更有意义、彼此不同、并且更可能与目标存在非线性关系的特征。"
    ).format(
        list(candidate_names),
        best_feature_name,
        base_mape,
        final_mape,
    )


def generation_error_feedback(error: Exception) -> str:
    return (
        "上一轮生成三个候选特征失败，错误为：{}。"
        "请生成更简单、更稳健、彼此独立的三个候选特征。"
    ).format(str(error))
