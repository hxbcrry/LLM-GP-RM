# -*- coding: utf-8 -*-
"""


消融开关：
- USE_GP_EVOLUTION=False：仅 LLM 特征生成；
- USE_GP_EVOLUTION=True 且 USE_REFLECTION_GP=False：原始 LLM + 普通 GP；
- USE_GP_EVOLUTION=True 且 USE_REFLECTION_GP=True：本文建议的短期反思增强版本。


"""

import re
import ast
import warnings
import os
import random
import operator
import time
from pathlib import Path
from functools import partial
from textwrap import dedent
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge,Lasso
from deap import base, creator, tools, gp
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
warnings.filterwarnings("ignore")


# ============================================================
# 1. 用户配置区：打开脚本后优先修改这里
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# -------------------- 数据集配置 --------------------
# 数据文件建议与脚本放在同一目录。也可通过环境变量 CAAFE_DATA_PATH 指定。
DATA_PATH = '***********'

# None 表示默认使用数据最后一列作为目标列；也可以填写具体列名，例如 "target"。
TARGET_COLUMN = None

# 为保证多数据集实验公平，主实验建议保持为空。
# 只有明确允许使用领域先验时，才填写数据集专属描述。
CUSTOM_DATASET_DESCRIPTION = ""

# -------------------- DeepSeek 配置 --------------------

# Windows CMD：set DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_API_KEY = "***********************************"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT_SECONDS = 120
DEEPSEEK_MAX_RETRIES = 2

# -------------------- 基础实验配置 --------------------
SEEDS = [1, 3, 5, 7, 9, 11, 28, 42, 43, 46]
TEST_SIZE = 0.2
MAX_ROUNDS = 5
MAX_ACCEPTED_FEATURES = 5
N_CANDIDATES_PER_ROUND = 3

# 消融开关：
# 1) False：仅使用 LLM 候选；
# 2) True + USE_REFLECTION_GP=False：原始 LLM + 普通 GP；
# 3) True + USE_REFLECTION_GP=True：停滞触发短期反思增强版本。
USE_GP_EVOLUTION = True

# -------------------- 普通 GP 参数 --------------------
GP_N_GENERATIONS = 50
GP_POP_SIZE = 20
GP_CXPB = 0.75
GP_MUTPB = 0.25
GP_TOURN_SIZE = 3
GP_ELITE_SIZE = 1
GP_INIT_MIN_HEIGHT = 1
GP_INIT_MAX_HEIGHT = 3
GP_MUTATION_MAX_HEIGHT = 2
GP_MAX_TREE_HEIGHT = 5
GP_PARSIMONY_COEF = 0.001

# CV-MAPE 至少降低该数值（单位：百分点）才接受新特征。
MIN_IMPROVEMENT = 0.020

# -------------------- 反思增强 GP 参数 --------------------
# USE_REFLECTION_GP = True
# USE_LONG_TERM_REFLECTION = False
# REFLECTION_STAGNATION_PATIENCE = 5
# REFLECTION_START_GENERATION = 15       #10
# REFLECTION_MIN_INTERVAL = 5
# REFLECTION_MAX_CALLS_PER_GP = 3
# REFLECTION_CANDIDATE_COUNT = 3
# REFLECTION_INJECTION_COUNT = 2
# REFLECTION_COMPARISON_RANK = 5
# GP_MIN_PROGRESS = 1e-6
# REFLECTION_MAX_RECENT_FAILURES = 5
# MAX_LONG_TERM_REFLECTION_LENGTH = 1600
# REFLECTION_REQUIRE_BETTER_THAN_WORST = True

# 启用停滞触发的短期反思
USE_REFLECTION_GP = True

# 当前主实验先关闭长期反思，减少 Token、变量数量和错误经验累积
USE_LONG_TERM_REFLECTION = False

# 连续 8 代没有实质改善才触发反思
REFLECTION_STAGNATION_PATIENCE = 8

# 普通 GP 至少先运行 15 代，避免反思过早干预
REFLECTION_START_GENERATION = 15

# 两次反思至少间隔 10 代
# 当前最多反思 1 次，但保留该参数便于后续扩展
REFLECTION_MIN_INTERVAL = 10

# 每次 GP 搜索最多触发 1 次反思
REFLECTION_MAX_CALLS_PER_GP = 1

# 每次让 LLM 生成 3 个定向候选，保持一定选择空间
REFLECTION_CANDIDATE_COUNT = 3

# 最多只注入 1 个候选，降低反思对种群的冲击
REFLECTION_INJECTION_COUNT = 1

# 第一阶段继续使用排名第 5 左右的较优个体与精英比较
REFLECTION_COMPARISON_RANK = 5

# Hall of Fame 适应度至少改善该数值，才视为真正进步
GP_MIN_PROGRESS = 1e-6

# 向反思器提供最近 5 个失败表达式
REFLECTION_MAX_RECENT_FAILURES = 5

# 长期反思当前关闭，该参数暂时不会生效
MAX_LONG_TERM_REFLECTION_LENGTH = 1600

# 继续保留基础保护：反思候选至少必须优于种群最差个体
REFLECTION_REQUIRE_BETTER_THAN_WORST = True

# -------------------- 结果保存配置 --------------------
SAVE_RESULTS = True
#RESULTS_FILENAME = "deepseek_caafe_reflection_results.xlsx"
#RESULTS_PATH = PROJECT_ROOT / RESULTS_FILENAME


# ============================================================
# 2. 运行时数据变量与数据读取函数
# ============================================================
# 这些变量在 main() 中通过 load_dataset() 初始化，避免导入脚本时立即读取文件。
DATA_FILE = None
df = None
TARGET = None
ORIGINAL_FEATURES = []
DATASET_DESCRIPTION = ""


def resolve_data_file(data_path):
    """将相对路径解析为脚本同级路径，并返回 Path。"""
    data_file = Path(data_path)
    if not data_file.is_absolute():
        data_file = PROJECT_ROOT / data_file
    return data_file


def load_dataset(data_path, target_column=None):
    """读取 xlsx/xls/csv 数据集，并返回数据、目标列和原始特征列。"""
    data_file = resolve_data_file(data_path)

    if not data_file.exists():
        raise FileNotFoundError(
            "未找到数据文件：{}。请修改 DATA_PATH 或设置 CAAFE_DATA_PATH。".format(
                data_file
            )
        )

    suffix = data_file.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        data_frame = pd.read_excel(data_file)
    elif suffix == ".csv":
        data_frame = pd.read_csv(data_file)
    else:
        raise ValueError(
            "仅支持 .xlsx、.xls 或 .csv 数据文件：{}".format(data_file)
        )

    if data_frame.shape[1] < 2:
        raise ValueError("数据至少需要 1 个输入特征列和 1 个目标列。")

    target = target_column if target_column is not None else data_frame.columns[-1]
    if target not in data_frame.columns:
        raise ValueError("指定的目标列不存在：{}".format(target))

    original_features = [col for col in data_frame.columns if col != target]
    if not original_features:
        raise ValueError("未检测到可用输入特征。")

    return data_file, data_frame, target, original_features



# ============================================================
# 3. 内置数据集描述、Prompt 与反馈构造函数
# 说明：为便于单文件运行，原 prompts/ 和 configs/ 中的内容已整合到本文件。
# ============================================================

# -------------------- 数据集描述 --------------------
# 默认使用数据集无关描述，保证多数据集实验采用一致的 Prompt 结构。
# 若某项实验允许使用领域背景，可在配置区填写 CUSTOM_DATASET_DESCRIPTION；
# 主实验建议保持为空，避免不同数据集获得不等量的人工先验。
GENERIC_DATASET_DESCRIPTION_TEMPLATE = dedent(
    """
*******************************
    """.strip())


# -------------------- CAAFE 特征生成 Prompt --------------------
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
    程序会分别验证三个候选特征，选择 Top-2 初始化 GP，并在 LLM 与 GP 候选中保留真正最优者。

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


# -------------------- 反思增强 GP Prompt --------------------
REFLECTION_PROMPT_VERSION = "reflection_guided_gp_v1.0"

REFLECTOR_SYSTEM_PROMPT = (
    "你是遗传编程特征工程中的反思分析器。"
    "你只分析候选特征表达式、交叉验证性能、复杂度和失败原因。"
    "输出必须具体、简洁、可执行，不要生成代码，不要输出空泛建议。"
)

REFLECTION_GENERATOR_SYSTEM_PROMPT = (
    "你是受约束的遗传编程前缀表达式生成器。"
    "只能使用用户给出的变量和函数集合。"
    "每行只输出一个合法表达式，不要编号、不要解释、不要代码块。"
)

LONG_TERM_REFLECTOR_SYSTEM_PROMPT = (
    "你负责维护遗传编程特征搜索的长期经验。"
    "请压缩重复内容，只保留稳定、可复用且与后续搜索直接相关的规律。"
    "不要生成代码。"
)


def _format_variable_mapping(variable_mapping: Mapping[str, str]) -> str:
    if not variable_mapping:
        return "暂无可用变量"
    return "\n".join(
        "- {} -> {}".format(var_name, feature_name)
        for var_name, feature_name in variable_mapping.items()
    )


def _format_failures(recent_failures: Sequence[Mapping[str, str]]) -> str:
    if not recent_failures:
        return "暂无已记录失败表达式"

    lines = []
    for index, item in enumerate(recent_failures, start=1):
        lines.append(
            "{}. {}：{}".format(
                index,
                item.get("expression", "<unknown>"),
                item.get("reason", "未知原因"),
            )
        )
    return "\n".join(lines)


def build_short_term_reflection_prompt(
    *,
    base_mape: float,
    elite_expr: str,
    elite_mape: float,
    elite_fitness: float,
    elite_size: int,
    comparison_expr: str,
    comparison_mape: float,
    comparison_fitness: float,
    comparison_size: int,
    variable_mapping: Mapping[str, str],
    recent_failures: Sequence[Mapping[str, str]],
) -> str:
    """比较两个较优但不同的 GP 个体，生成一次短期反思。"""
    return dedent(
        """
        当前任务是表格回归的自动特征工程。候选表达式会被加入原始特征集合，
        再由下游模型进行交叉验证；CV-MAPE 越低越好。

        原始特征基线 CV-MAPE：{base_mape:.6f}%

        当前精英个体：
        - GP 前缀表达式：{elite_expr}
        - CV-MAPE：{elite_mape:.6f}%
        - 含简约惩罚的适应度：{elite_fitness:.6f}
        - 树节点数：{elite_size}

        对比个体：
        - GP 前缀表达式：{comparison_expr}
        - CV-MAPE：{comparison_mape:.6f}%
        - 含简约惩罚的适应度：{comparison_fitness:.6f}
        - 树节点数：{comparison_size}

        变量映射：
        {variable_mapping}

        最近失败或无效的表达式：
        {recent_failures}

        请基于两者相对性能，给出一次“短期反思”，必须明确回答：
        1. 精英表达式中最值得保留的结构或变量交互；
        2. 对比表达式中可能冗余、无效或不稳定的结构；
        3. 下一步应优先进行哪类局部替换、重组或非线性变换；
        4. 应避免哪些已出现的失败模式；
        5. 如何在性能提升和表达式复杂度之间保持平衡。

        只输出 5 条以内的具体搜索建议，不要生成表达式或 Python 代码。
        """
    ).format(
        base_mape=base_mape,
        elite_expr=elite_expr,
        elite_mape=elite_mape,
        elite_fitness=elite_fitness,
        elite_size=elite_size,
        comparison_expr=comparison_expr,
        comparison_mape=comparison_mape,
        comparison_fitness=comparison_fitness,
        comparison_size=comparison_size,
        variable_mapping=_format_variable_mapping(variable_mapping),
        recent_failures=_format_failures(recent_failures),
    ).strip()


def build_long_term_reflection_prompt(
    *,
    previous_memory: str,
    short_reflection: str,
    max_chars: int = 1600,
) -> str:
    """把多次短期反思压缩成可复用的长期搜索经验。"""
    previous_memory = previous_memory.strip() or "暂无长期经验"
    return dedent(
        """
        下面是普通 GP 特征搜索的已有长期经验和本轮短期反思。

        已有长期经验：
        {previous_memory}

        本轮短期反思：
        {short_reflection}

        请更新长期经验，只保留：
        1. 多次有效的变量组合或结构模式；
        2. 多次有效的算子使用方式；
        3. 常见失败、冗余和数值不稳定模式；
        4. 下一阶段最值得探索的方向；
        5. 对表达式复杂度的控制原则。

        不要重复，不要生成代码，总长度不超过 {max_chars} 个字符。
        """
    ).format(
        previous_memory=previous_memory,
        short_reflection=short_reflection,
        max_chars=max_chars,
    ).strip()


def build_reflection_candidate_prompt(
    *,
    elite_expr: str,
    comparison_expr: str,
    short_reflection: str,
    long_term_reflection: str,
    variable_mapping: Mapping[str, str],
    candidate_count: int,
    max_tree_height: int,
) -> str:
    """根据反思生成可直接解析为 DEAP GP 树的前缀表达式。"""
    long_term_text = long_term_reflection.strip() or "当前未启用或尚未形成长期经验"

    return dedent(
        """
        当前精英 GP 前缀表达式：
        {elite_expr}

        对比 GP 前缀表达式：
        {comparison_expr}

        本轮短期反思：
        {short_reflection}

        长期搜索经验：
        {long_term_reflection}

        变量映射：
        {variable_mapping}

        允许使用的二元函数只有：
        add(a,b)
        sub(a,b)
        mul(a,b)

        允许使用的一元函数只有：
        square(a)
        safe_abs(a)
        safe_sqrt_abs(a)
        safe_log1p_abs(a)
        neg(a)

        请生成恰好 {candidate_count} 个新的 GP 前缀表达式。

        严格要求：
        1. 只能使用上面列出的变量和函数，不得使用其他符号或函数；
        2. 至少保留或合理重组精英表达式中的一个有效结构；
        3. 根据短期反思修改可能无效的部分，而不是简单复制精英；
        4. 各表达式必须彼此不同，也不能与两个输入表达式完全相同；
        5. 最大树高不得超过 {max_tree_height}；
        6. 避免连续 square、连续多层一元函数和无意义自消结构；
        7. 禁止除法、任意幂、三角函数、条件语句、赋值和临时变量；
        8. 每行只输出一个表达式；
        9. 不要编号、不要解释、不要 Markdown 代码块。
        """
    ).format(
        elite_expr=elite_expr,
        comparison_expr=comparison_expr,
        short_reflection=short_reflection,
        long_term_reflection=long_term_text,
        variable_mapping=_format_variable_mapping(variable_mapping),
        candidate_count=candidate_count,
        max_tree_height=max_tree_height,
    ).strip()


# -------------------- 下一轮反馈文本 --------------------
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
    reflection_calls: int = 0,
    reflection_injected: int = 0,
) -> str:
    reflection_part = ""
    if reflection_calls > 0:
        reflection_part = (
            "GP 搜索停滞期间共触发 {} 次短期反思，"
            "并向种群注入 {} 个通过安全检查和适应度评价的定向候选。"
        ).format(reflection_calls, reflection_injected)

    return (
        "上一轮 DeepSeek 生成了三个候选特征 {}。"
        "其中改善最好的两个候选为 {}。"
        "随后使用 DEAP 遗传编程将这两个候选表达式抽象为树，"
        "经过 {} 代交叉和变异后，得到更优特征 {}。"
        "{}"
        "它使 CV MAPE 从 {:.4f}% 降低到 {:.4f}%，已接受。"
        "后续生成特征时，可以优先结合两个父代特征的结构模式，"
        "例如复合交互、非线性变换和稳健绝对值变换。\n{}"
    ).format(
        list(candidate_names),
        list(parents),
        generations,
        feature_name,
        reflection_part,
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



# ============================================================
# 4. DeepSeek 客户端
# ============================================================

def get_client():
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            '未检测到环境变量 DEEPSEEK_API_KEY。\n'
            'Windows PowerShell：$env:DEEPSEEK_API_KEY="你的密钥"\n'
            'Windows CMD：set DEEPSEEK_API_KEY=你的密钥'
        )

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )


# ============================================================
# 5. 指标函数
# ============================================================

def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))
    ) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    }


# ============================================================
# 5. 下游模型：Ridge
# 主实验中请对所有数据集保持相同模型与参数。
# ============================================================

def build_model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge", DecisionTreeRegressor())
    ])


def cv_score_mape(X_df, y, features, seed):
    X = X_df[features].replace([np.inf, -np.inf], np.nan).copy()
    y = np.asarray(y)

    if len(X) < 3:
        return 999999.0

    n_splits = min(5, len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    y_true_all = []
    y_pred_all = []

    for tr_idx, va_idx in kf.split(X):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        y_tr = y[tr_idx]
        y_va = y[va_idx]

        model = build_model(seed)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_va)

        y_true_all.extend(y_va)
        y_pred_all.extend(pred)

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)

    mape = np.mean(
        np.abs((y_true_all - y_pred_all) / (np.abs(y_true_all) + 1e-8))
    ) * 100

    return mape


# ============================================================
# 6. Prompt 构造
# ============================================================
# Prompt 模板和构造逻辑已拆分到 prompts/caafe_prompts.py。


# ============================================================
# 7. 调用 DeepSeek 生成特征代码
# ============================================================

API_CALL_COUNTER = {
    "feature_generation": 0,
    "short_reflection": 0,
    "long_reflection": 0,
    "reflection_generation": 0,
}


def _call_deepseek_with_system(
    system_prompt,
    user_prompt,
    temperature,
    call_type,
):
    """统一封装 DeepSeek 调用，并进行有限次数重试。"""
    client = get_client()
    last_error = None

    for attempt in range(1, DEEPSEEK_MAX_RETRIES + 2):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
                timeout=DEEPSEEK_TIMEOUT_SECONDS,
            )

            content = response.choices[0].message.content
            if content is None or not content.strip():
                raise ValueError("DeepSeek 返回内容为空")

            API_CALL_COUNTER[call_type] = API_CALL_COUNTER.get(call_type, 0) + 1
            return content.strip()

        except Exception as error:
            last_error = error
            if attempt > DEEPSEEK_MAX_RETRIES:
                break
            wait_seconds = min(2 ** (attempt - 1), 4)
            print(
                "[API重试] {} 调用失败：{}；{} 秒后进行第 {} 次尝试。".format(
                    call_type,
                    error,
                    wait_seconds,
                    attempt + 1,
                )
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        "DeepSeek {} 调用连续失败：{}".format(call_type, last_error)
    )


def call_deepseek(prompt):
    """调用特征生成器。"""
    return _call_deepseek_with_system(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.3,
        call_type="feature_generation",
    )


def call_reflector(prompt):
    """调用短期反思器，温度较低以提高分析稳定性。"""
    return _call_deepseek_with_system(
        system_prompt=REFLECTOR_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.1,
        call_type="short_reflection",
    )


def call_long_term_reflector(prompt):
    """调用长期经验压缩器。"""
    return _call_deepseek_with_system(
        system_prompt=LONG_TERM_REFLECTOR_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.1,
        call_type="long_reflection",
    )


def call_reflection_generator(prompt):
    """根据反思生成受约束的 GP 前缀表达式。"""
    return _call_deepseek_with_system(
        system_prompt=REFLECTION_GENERATOR_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.2,
        call_type="reflection_generation",
    )


# ============================================================
# 8. 清洗、检查和执行 LLM 代码
# ============================================================

def extract_python_code(text):
    blocks = re.findall(
        r"```(?:python)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if blocks:
        return blocks[0].strip()

    return text.strip()


def clean_code(code, expected_feature_name):
    code = extract_python_code(code)

    cleaned_lines = []

    for line in code.splitlines():
        raw = line.rstrip()
        stripped = raw.strip()

        if stripped == "":
            continue

        if stripped.startswith("#"):
            cleaned_lines.append(raw)
            continue

        allowed_prefix_1 = "df['{}']".format(expected_feature_name)
        allowed_prefix_2 = 'df["{}"]'.format(expected_feature_name)

        if stripped.startswith(allowed_prefix_1) or stripped.startswith(allowed_prefix_2):
            cleaned_lines.append(raw)

    return "\n".join(cleaned_lines)


def check_code_safety(code, allowed_input_features, expected_feature_name):
    banned_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.ClassDef,
        ast.With,
        ast.While,
        ast.For,
        ast.Lambda,
        ast.Try,
    )

    banned_names = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
    }

    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            raise ValueError("发现不安全语法：{}".format(type(node).__name__))

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            raise ValueError("禁止使用除法 /")

        # 允许平方 **2，但禁止 **3、**10、**df['x'] 等不稳定幂运算
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and node.right.value == 2):
                raise ValueError("只允许平方 **2，禁止其他幂运算")

        if isinstance(node, ast.Call):
            func = node.func

            if isinstance(func, ast.Name) and func.id in banned_names:
                raise ValueError("发现禁止函数：{}".format(func.id))

            if isinstance(func, ast.Attribute) and func.attr in banned_names:
                raise ValueError("发现禁止方法：{}".format(func.attr))

    used_cols = re.findall(r"df\[['\"](.+?)['\"]\]", code)

    for col in used_cols:
        if col == expected_feature_name:
            continue

        if col not in allowed_input_features:
            raise ValueError("代码使用了不允许的列：{}".format(col))


def apply_code_safe(X_input, code, allowed_input_features, expected_feature_name):
    code = clean_code(code, expected_feature_name)

    if not code:
        raise ValueError("清洗后代码为空，未生成有效特征。")

    check_code_safety(
        code=code,
        allowed_input_features=allowed_input_features,
        expected_feature_name=expected_feature_name
    )

    df_local = X_input.copy()

    env = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "len": len,
        },
        "df": df_local,
        "np": np,
    }

    exec(compile(ast.parse(code), filename="<caafe_code>", mode="exec"), env, env)

    df_new = env["df"]

    if expected_feature_name not in df_new.columns:
        raise ValueError("没有成功生成特征：{}".format(expected_feature_name))

    values = df_new[expected_feature_name].replace([np.inf, -np.inf], np.nan)

    if values.isna().all():
        raise ValueError("{} 全部为 NaN。".format(expected_feature_name))

    df_new[expected_feature_name] = values.fillna(values.median())

    return df_new, code




# ============================================================
# 8.5 DEAP 遗传编程：把 Top-2 候选特征表达式抽象成树并进化
# ============================================================

def gp_add(a, b):
    return a + b


def gp_sub(a, b):
    return a - b


def gp_mul(a, b):
    return a * b


def gp_square(a):
    # 只用于平方非线性，避免开放任意幂运算造成数值爆炸
    return a * a


def gp_abs(a):
    return np.abs(a)


def gp_sqrt_abs(a):
    return np.sqrt(np.abs(a))


def gp_log1p_abs(a):
    return np.log1p(np.abs(a))


def gp_neg(a):
    return -a


def make_gp_creator_classes():
    """DEAP 的 creator 是全局注册的，重复 create 会报错，所以先判断。"""
    if not hasattr(creator, "FitnessMinCAAFE"):
        creator.create("FitnessMinCAAFE", base.Fitness, weights=(-1.0,))

    if not hasattr(creator, "IndividualCAAFE"):
        creator.create("IndividualCAAFE", gp.PrimitiveTree, fitness=creator.FitnessMinCAAFE)


def build_gp_pset(input_features):
    """为当前可用特征构造 GP primitive set。"""
    pset = gp.PrimitiveSet("MAIN", len(input_features))

    rename_map = {}
    feature_to_var = {}
    var_to_feature = {}

    for i, col in enumerate(input_features):
        var_name = "x{}".format(i)
        rename_map["ARG{}".format(i)] = var_name
        feature_to_var[col] = var_name
        var_to_feature[var_name] = col

    pset.renameArguments(**rename_map)

    pset.addPrimitive(gp_add, 2, name="add")
    pset.addPrimitive(gp_sub, 2, name="sub")
    pset.addPrimitive(gp_mul, 2, name="mul")
    pset.addPrimitive(gp_square, 1, name="square")
    pset.addPrimitive(gp_abs, 1, name="safe_abs")
    pset.addPrimitive(gp_sqrt_abs, 1, name="safe_sqrt_abs")
    pset.addPrimitive(gp_log1p_abs, 1, name="safe_log1p_abs")
    pset.addPrimitive(gp_neg, 1, name="neg")

    pset.addEphemeralConstant(
        "rand_const",
        partial(random.uniform, -2.0, 2.0)
    )

    return pset, feature_to_var, var_to_feature


def get_df_column_from_ast(node):
    """识别 df['col'] 或 df[\"col\"]，返回 col。"""
    if not isinstance(node, ast.Subscript):
        return None

    if not isinstance(node.value, ast.Name) or node.value.id != "df":
        return None

    slice_node = node.slice

    if isinstance(slice_node, ast.Constant):
        return slice_node.value

    # 兼容较老 Python AST
    if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
        if isinstance(slice_node.value, ast.Constant):
            return slice_node.value.value

    return None


def get_assignment_rhs_ast(code, expected_feature_name):
    """从 df['feat_x'] = ... 中提取右侧表达式 AST。"""
    cleaned = clean_code(code, expected_feature_name)
    tree = ast.parse(cleaned)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            col = get_df_column_from_ast(target)
            if col == expected_feature_name:
                return node.value

    raise ValueError("未找到 {} 的赋值表达式".format(expected_feature_name))


def get_call_name(func_node):
    if isinstance(func_node, ast.Name):
        return func_node.id

    if isinstance(func_node, ast.Attribute):
        parent = get_call_name(func_node.value)
        if parent:
            return parent + "." + func_node.attr
        return func_node.attr

    return ""


def ast_expr_to_gp_string(node, feature_to_var):
    """把 Python 右侧表达式 AST 转为 DEAP PrimitiveTree.from_string 可解析的前缀表达式。"""
    if isinstance(node, ast.BinOp):
        left = ast_expr_to_gp_string(node.left, feature_to_var)
        right = ast_expr_to_gp_string(node.right, feature_to_var)

        if isinstance(node.op, ast.Add):
            return "add({}, {})".format(left, right)
        if isinstance(node.op, ast.Sub):
            return "sub({}, {})".format(left, right)
        if isinstance(node.op, ast.Mult):
            return "mul({}, {})".format(left, right)

        # 只支持平方 **2，将其映射为 GP 的一元 square primitive
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                return "square({})".format(left)
            else:
                raise ValueError("GP 目前只支持平方 **2，不支持其他幂运算")

        raise ValueError("GP 不支持该二元运算：{}".format(type(node.op).__name__))

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = ast_expr_to_gp_string(node.operand, feature_to_var)
        return "neg({})".format(inner)

    if isinstance(node, ast.Call):
        if len(node.args) != 1:
            raise ValueError("GP 只支持单参数函数调用")

        arg = ast_expr_to_gp_string(node.args[0], feature_to_var)
        call_name = get_call_name(node.func)

        if call_name in {"np.abs", "abs"}:
            return "safe_abs({})".format(arg)

        if call_name == "np.sqrt":
            # 为了稳健，统一转成 sqrt(abs(.))
            return "safe_sqrt_abs({})".format(arg)

        if call_name == "np.log1p":
            # 为了稳健，统一转成 log1p(abs(.))
            return "safe_log1p_abs({})".format(arg)

        raise ValueError("GP 不支持该函数调用：{}".format(call_name))

    col = get_df_column_from_ast(node)
    if col is not None:
        if col not in feature_to_var:
            raise ValueError("表达式使用了不允许的列：{}".format(col))
        return feature_to_var[col]

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return repr(float(node.value))

    raise ValueError("GP 不支持该表达式节点：{}".format(type(node).__name__))


def code_to_gp_individual(code, expected_feature_name, input_features, pset, feature_to_var):
    """把 LLM 生成的单个候选特征代码转为 DEAP GP 个体。"""
    rhs_ast = get_assignment_rhs_ast(code, expected_feature_name)
    gp_expr = ast_expr_to_gp_string(rhs_ast, feature_to_var)
    tree = gp.PrimitiveTree.from_string(gp_expr, pset)
    return creator.IndividualCAAFE(tree)


def _normalize_gp_terminal_token(token, depth=0):
    """
    把 DEAP terminal 的 name/value 递归拆成可判断的原始值。

    不同 DEAP 版本、不同生成路径下，terminal 可能表现为：
    - node.name == "x0"
    - node.value == "x0"
    - node.name == "ARG0" 或 node.value == "ARG0"
    - node.value 是 int/float 常数
    - node.value 本身又是一个 gp.Terminal

    这个函数用于兼容上述情况，避免最优树转 pandas 表达式时失败。
    """
    if depth > 5:
        return token

    if isinstance(token, gp.Terminal):
        name = getattr(token, "name", None)
        value = getattr(token, "value", None)

        # 优先返回更具体的 value；若 value 仍是 Terminal，则继续拆。
        if isinstance(value, gp.Terminal):
            return _normalize_gp_terminal_token(value, depth + 1)

        if value is not None:
            return value

        return name

    return token


def _terminal_token_to_pandas_expr(token, var_to_feature):
    """尝试把一个 terminal token 转成 pandas 表达式；失败则返回 None。"""
    token = _normalize_gp_terminal_token(token)

    # 1. x0、x1 这类重命名后的输入变量
    if isinstance(token, str) and token in var_to_feature:
        col = var_to_feature[token]
        return "df[{}]".format(repr(col))

    # 2. ARG0、ARG1 这类未重命名或部分 DEAP 版本保留的输入变量
    if isinstance(token, str) and token.startswith("ARG"):
        try:
            idx = int(token.replace("ARG", ""))
            var_name = "x{}".format(idx)
            if var_name in var_to_feature:
                col = var_to_feature[var_name]
                return "df[{}]".format(repr(col))
        except Exception:
            pass

    # 3. 常数 terminal
    if isinstance(token, (int, float, np.integer, np.floating)):
        return repr(float(token))

    # 4. 字符串形式的常数
    if isinstance(token, str):
        try:
            return repr(float(token))
        except Exception:
            pass

    return None


def gp_individual_to_python_expr(individual, var_to_feature):
    """把 DEAP GP 个体转回 df['col'] 风格的一行 pandas 表达式。"""
    def rec(pos):
        node = individual[pos]

        if isinstance(node, gp.Primitive):
            args = []
            next_pos = pos + 1
            for _ in range(node.arity):
                arg_expr, next_pos = rec(next_pos)
                args.append(arg_expr)

            if node.name == "add":
                return "({} + {})".format(args[0], args[1]), next_pos
            if node.name == "sub":
                return "({} - {})".format(args[0], args[1]), next_pos
            if node.name == "mul":
                return "({} * {})".format(args[0], args[1]), next_pos
            if node.name == "square":
                return "({} * {})".format(args[0], args[0]), next_pos
            if node.name == "safe_abs":
                return "np.abs({})".format(args[0]), next_pos
            if node.name == "safe_sqrt_abs":
                return "np.sqrt(np.abs({}))".format(args[0]), next_pos
            if node.name == "safe_log1p_abs":
                return "np.log1p(np.abs({}))".format(args[0]), next_pos
            if node.name == "neg":
                return "(-{})".format(args[0]), next_pos

            raise ValueError("未知 GP primitive：{}".format(node.name))

        # Terminal: 可能是输入变量、ARG 输入变量、ephemeral 常数，或者嵌套 Terminal。
        candidate_tokens = [
            node,
            getattr(node, "name", None),
            getattr(node, "value", None),
        ]

        # 某些情况下 str(node) 会返回 x0、ARG0 或常数字符串。
        try:
            node_str = str(node)
            if node_str and not node_str.startswith("<deap.gp.Terminal"):
                candidate_tokens.append(node_str)
        except Exception:
            pass

        for token in candidate_tokens:
            expr = _terminal_token_to_pandas_expr(token, var_to_feature)
            if expr is not None:
                return expr, pos + 1

        raise ValueError(
            "无法解析 GP terminal：type={}, name={}, value={}, str={}".format(
                type(node),
                repr(getattr(node, "name", None)),
                repr(getattr(node, "value", None)),
                repr(str(node))
            )
        )

    expr, end_pos = rec(0)
    if end_pos != len(individual):
        raise ValueError("GP 树没有被完整解析")

    return expr


def gp_individual_to_code(individual, expected_feature_name, var_to_feature):
    expr = gp_individual_to_python_expr(individual, var_to_feature)

    return "\n".join([
        "# Feature: {}".format(expected_feature_name),
        "# Usefulness: 由本轮三个 LLM 候选中特征改善最好的两个作为父代，经 DEAP 遗传编程交叉和变异进化得到。",
        "df['{}'] = {}".format(expected_feature_name, expr)
    ])


def parse_reflection_gp_candidates(raw_text, pset, existing_expressions=None):
    """把 LLM 返回的逐行 GP 前缀表达式解析为普通 GP 个体。"""
    existing_expressions = set(existing_expressions or [])
    candidates = []
    seen = set(existing_expressions)

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("```") or line.startswith("#"):
            continue

        # 兼容模型偶尔输出的项目符号或编号。
        line = re.sub(r"^(?:[-*•]|\d+[.)、])\s*", "", line).strip()
        line = line.strip("`").rstrip(";").strip()

        # 兼容偶尔输出 candidate_1 = add(...) 这类形式。
        if "=" in line:
            left, right = line.split("=", 1)
            if left.strip().lower().startswith(("candidate", "expr", "feature")):
                line = right.strip()

        if not line or line in seen:
            continue

        try:
            tree = gp.PrimitiveTree.from_string(line, pset)
            individual = creator.IndividualCAAFE(tree)

            if individual.height > GP_MAX_TREE_HEIGHT:
                continue

            canonical = str(individual)
            if canonical in seen:
                continue

            seen.add(canonical)
            candidates.append(individual)

        except Exception:
            continue

    return candidates


def get_recent_failure_records(eval_cache, limit):
    """提取最近的失败表达式及原因，供反思 Prompt 使用。"""
    records = []

    for expression, info in reversed(list(eval_cache.items())):
        if info.get("status") != "failed":
            continue

        records.append({
            "expression": expression,
            "reason": info.get("failure_reason", "未知原因")
        })

        if len(records) >= limit:
            break

    return records


def select_reflection_pair(population, comparison_rank):
    """选择当前精英和一个较优但结构不同的对比个体。"""
    successful = [
        ind for ind in population
        if ind.fitness.valid and ind.fitness.values[0] < 999999.0
    ]

    if len(successful) < 2:
        return None, None

    successful.sort(key=lambda ind: ind.fitness.values[0])
    elite = successful[0]

    preferred_index = min(
        max(int(comparison_rank) - 1, 1),
        len(successful) - 1
    )

    search_order = list(range(preferred_index, len(successful))) + list(
        range(1, preferred_index)
    )

    for idx in search_order:
        candidate = successful[idx]
        if str(candidate) != str(elite):
            return elite, candidate

    return elite, successful[1]


def evolve_top2_candidates_with_deap(
    X_current,
    y_train,
    current_features,
    base_mape,
    candidate_results,
    evolved_feature_name,
    seed
):
    """
    从三个 LLM 候选里取改善最好的两个，将其表达式转为 GP 树，
    用 DEAP 交叉/变异迭代若干代，返回最优进化特征。
    """
    if len(candidate_results) < 2:
        print("候选特征少于 2 个，跳过 DEAP 进化。")
        return None

    sorted_candidates = sorted(
        candidate_results,
        key=lambda x: x["improvement"],
        reverse=True
    )
    top2 = sorted_candidates[:2]

    print("\n准备对 Top-2 候选特征进行 DEAP 遗传编程进化：{} 和 {}".format(
        top2[0]["feature_name"],
        top2[1]["feature_name"]
    ))

    random.seed(seed + 10007)
    np.random.seed(seed + 10007)

    make_gp_creator_classes()
    pset, feature_to_var, var_to_feature = build_gp_pset(current_features)

    seed_individuals = []
    for item in top2:
        try:
            ind = code_to_gp_individual(
                code=item["cleaned_code"],
                expected_feature_name=item["feature_name"],
                input_features=current_features,
                pset=pset,
                feature_to_var=feature_to_var
            )
            seed_individuals.append(ind)
            print("已将 {} 转为 GP 树：{}".format(item["feature_name"], ind))
        except Exception as e:
            print("{} 转 GP 树失败：{}".format(item["feature_name"], e))

    if len(seed_individuals) < 2:
        print("可用 GP 种子不足 2 个，跳过 DEAP 进化。")
        return None

    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=GP_INIT_MIN_HEIGHT, max_=GP_INIT_MAX_HEIGHT)
    toolbox.register("individual", tools.initIterate, creator.IndividualCAAFE, toolbox.expr)
    toolbox.register("select", tools.selTournament, tournsize=GP_TOURN_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=GP_MUTATION_MAX_HEIGHT)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=GP_MAX_TREE_HEIGHT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=GP_MAX_TREE_HEIGHT))

    eval_cache = {}

    def evaluate_individual(individual):
        key = str(individual)
        if key in eval_cache:
            return (eval_cache[key]["fitness"],)

        try:
            func = toolbox.compile(expr=individual)
            arrays = [X_current[col].values for col in current_features]
            values = func(*arrays)
            values = np.asarray(values, dtype=float)

            if values.shape == ():
                values = np.full(len(X_current), float(values))

            if len(values) != len(X_current):
                raise ValueError("进化特征长度不等于样本数")

            values = pd.Series(values).replace([np.inf, -np.inf], np.nan)

            if values.isna().all():
                raise ValueError("进化特征全部为 NaN")

            values = values.fillna(values.median()).values

            if np.nanstd(values) < 1e-12:
                raise ValueError("进化特征近似常数")

            X_tmp = X_current.copy()
            X_tmp[evolved_feature_name] = values

            candidate_mape = cv_score_mape(
                X_df=X_tmp,
                y=y_train,
                features=current_features + [evolved_feature_name],
                seed=seed
            )

            fitness = candidate_mape + GP_PARSIMONY_COEF * len(individual)
            eval_cache[key] = {
                "fitness": fitness,
                "mape": candidate_mape,
                "length": len(individual),
                "status": "success",
                "failure_reason": ""
            }

            return (fitness,)

        except Exception as e:
            eval_cache[key] = {
                "fitness": 999999.0,
                "mape": 999999.0,
                "length": len(individual),
                "status": "failed",
                "failure_reason": str(e)
            }
            return (999999.0,)

    toolbox.register("evaluate", evaluate_individual)

    population = []

    # 先放入两个 LLM 高分候选作为精英种子
    for ind in seed_individuals:
        population.append(toolbox.clone(ind))

    # 再用种子变异和随机树补齐种群
    while len(population) < GP_POP_SIZE:
        if random.random() < 0.60:
            ind = toolbox.clone(random.choice(seed_individuals))
            if random.random() < 0.80:
                ind, = toolbox.mutate(ind)
        else:
            ind = toolbox.individual()
        population.append(ind)

    hof = tools.HallOfFame(1)

    invalid_individuals = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit
    hof.update(population)

    # 反思增强状态。HallOfFame 的最优适应度单调不增，适合用于停滞检测。
    best_fitness_so_far = hof[0].fitness.values[0]
    stagnation_count = 0
    last_reflection_generation = -REFLECTION_MIN_INTERVAL
    reflection_calls = 0
    reflection_candidates_evaluated = 0
    reflection_injected = 0
    long_term_reflection = ""

    for gen_idx in range(1, GP_N_GENERATIONS + 1):
        # =====================================================
        # 精英保留：先从 HallOfFame 复制历史最优个体
        # 注意：本问题是最小化 fitness，fitness 越小越好。
        # HallOfFame 只负责记录历史最优；下面会把精英强制放回下一代。
        # =====================================================
        elites = []
        if GP_ELITE_SIZE > 0 and len(hof) > 0:
            elites = [toolbox.clone(ind) for ind in hof[:GP_ELITE_SIZE]]

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GP_CXPB:
                toolbox.mate(child1, child2)
                if child1.fitness.valid:
                    del child1.fitness.values
                if child2.fitness.valid:
                    del child2.fitness.values

        for mutant in offspring:
            if random.random() < GP_MUTPB:
                toolbox.mutate(mutant)
                if mutant.fitness.valid:
                    del mutant.fitness.values

        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # =====================================================
        # 把历史最优个体替换掉本代 offspring 中最差的个体。
        # 因为 creator.FitnessMinCAAFE 使用 weights=(-1.0,)，
        # 所以 fitness.values[0] 越大代表越差。
        # =====================================================
        if elites:
            for elite in elites:
                if not elite.fitness.valid:
                    elite.fitness.values = toolbox.evaluate(elite)

            worst_indices = sorted(
                range(len(offspring)),
                key=lambda i: offspring[i].fitness.values[0],
                reverse=True
            )[:len(elites)]

            for worst_idx, elite in zip(worst_indices, elites):
                offspring[worst_idx] = elite

        population[:] = offspring
        hof.update(population)

        # =====================================================
        # 停滞检测：只有 HallOfFame 最优适应度出现实质下降才重置计数。
        # =====================================================
        current_best_fitness = hof[0].fitness.values[0]
        if best_fitness_so_far - current_best_fitness > GP_MIN_PROGRESS:
            best_fitness_so_far = current_best_fitness
            stagnation_count = 0
        else:
            stagnation_count += 1

        should_reflect = (
            USE_REFLECTION_GP
            and gen_idx >= REFLECTION_START_GENERATION
            and stagnation_count >= REFLECTION_STAGNATION_PATIENCE
            and gen_idx - last_reflection_generation >= REFLECTION_MIN_INTERVAL
            and reflection_calls < REFLECTION_MAX_CALLS_PER_GP
        )

        # =====================================================
        # ReEvo 启发的最小改进版：
        # 保留普通 GP 作为主搜索器，仅在停滞时进行短期反思和定向候选注入。
        # =====================================================
        if should_reflect:
            reflection_calls += 1
            last_reflection_generation = gen_idx

            print(
                "\n[反思触发] 第 {} 代连续 {} 代没有实质提升，"
                "开始第 {} 次短期反思。".format(
                    gen_idx,
                    stagnation_count,
                    reflection_calls
                )
            )

            try:
                elite_ind, comparison_ind = select_reflection_pair(
                    population=population,
                    comparison_rank=REFLECTION_COMPARISON_RANK
                )

                if elite_ind is None or comparison_ind is None:
                    raise ValueError("可用于反思比较的有效 GP 个体不足 2 个")

                elite_expr = str(elite_ind)
                comparison_expr = str(comparison_ind)
                elite_info = eval_cache.get(elite_expr)
                comparison_info = eval_cache.get(comparison_expr)

                if elite_info is None or comparison_info is None:
                    raise ValueError("反思个体缺少适应度缓存信息")

                recent_failures = get_recent_failure_records(
                    eval_cache=eval_cache,
                    limit=REFLECTION_MAX_RECENT_FAILURES
                )

                short_prompt = build_short_term_reflection_prompt(
                    base_mape=base_mape,
                    elite_expr=elite_expr,
                    elite_mape=elite_info["mape"],
                    elite_fitness=elite_info["fitness"],
                    elite_size=len(elite_ind),
                    comparison_expr=comparison_expr,
                    comparison_mape=comparison_info["mape"],
                    comparison_fitness=comparison_info["fitness"],
                    comparison_size=len(comparison_ind),
                    variable_mapping=var_to_feature,
                    recent_failures=recent_failures
                )

                short_reflection = call_reflector(short_prompt)
                print("[短期反思结果]")
                print(short_reflection)

                if USE_LONG_TERM_REFLECTION:
                    long_prompt = build_long_term_reflection_prompt(
                        previous_memory=long_term_reflection,
                        short_reflection=short_reflection,
                        max_chars=MAX_LONG_TERM_REFLECTION_LENGTH
                    )
                    long_term_reflection = call_long_term_reflector(long_prompt)
                    long_term_reflection = long_term_reflection[
                        :MAX_LONG_TERM_REFLECTION_LENGTH
                    ]
                    print("[更新后的长期经验]")
                    print(long_term_reflection)

                candidate_prompt = build_reflection_candidate_prompt(
                    elite_expr=elite_expr,
                    comparison_expr=comparison_expr,
                    short_reflection=short_reflection,
                    long_term_reflection=long_term_reflection,
                    variable_mapping=var_to_feature,
                    candidate_count=REFLECTION_CANDIDATE_COUNT,
                    max_tree_height=GP_MAX_TREE_HEIGHT
                )

                raw_reflection_candidates = call_reflection_generator(
                    candidate_prompt
                )
                print("[反思生成的 GP 前缀表达式]")
                print(raw_reflection_candidates)

                reflected_individuals = parse_reflection_gp_candidates(
                    raw_text=raw_reflection_candidates,
                    pset=pset,
                    existing_expressions=set(eval_cache.keys())
                )

                if not reflected_individuals:
                    print("[反思注入] 未解析到新的合法 GP 表达式，本次不注入。")
                else:
                    for ind in reflected_individuals:
                        if not ind.fitness.valid:
                            ind.fitness.values = toolbox.evaluate(ind)
                            reflection_candidates_evaluated += 1

                    reflected_individuals = [
                        ind for ind in reflected_individuals
                        if ind.fitness.values[0] < 999999.0
                    ]
                    reflected_individuals.sort(
                        key=lambda ind: ind.fitness.values[0]
                    )

                    worst_indices = sorted(
                        range(len(population)),
                        key=lambda i: population[i].fitness.values[0],
                        reverse=True
                    )

                    injected_this_round = 0
                    worst_cursor = 0

                    for new_ind in reflected_individuals:
                        if injected_this_round >= REFLECTION_INJECTION_COUNT:
                            break
                        if worst_cursor >= len(worst_indices):
                            break

                        worst_idx = worst_indices[worst_cursor]
                        worst_cursor += 1
                        worst_fitness = population[worst_idx].fitness.values[0]
                        new_fitness = new_ind.fitness.values[0]

                        if (
                            REFLECTION_REQUIRE_BETTER_THAN_WORST
                            and new_fitness >= worst_fitness
                        ):
                            # 候选已按适应度升序排列；最优候选都无法超过当前最差个体时，
                            # 后续候选也无需继续尝试。
                            break

                        population[worst_idx] = toolbox.clone(new_ind)
                        injected_this_round += 1
                        reflection_injected += 1

                        info = eval_cache.get(str(new_ind), {})
                        print(
                            "[反思注入] 表达式={}，CV MAPE={:.4f}%，"
                            "fitness={:.6f}".format(
                                str(new_ind),
                                info.get("mape", 999999.0),
                                info.get("fitness", 999999.0)
                            )
                        )

                    if injected_this_round == 0:
                        print("[反思注入] 合法候选未优于当前种群最差个体，本次不注入。")
                    else:
                        hof.update(population)
                        reflected_best = hof[0].fitness.values[0]
                        if best_fitness_so_far - reflected_best > GP_MIN_PROGRESS:
                            best_fitness_so_far = reflected_best

                        print(
                            "[反思注入完成] 本次注入 {} 个候选，累计注入 {} 个。".format(
                                injected_this_round,
                                reflection_injected
                            )
                        )

            except Exception as e:
                print("[反思失败] 本次反思或候选注入失败：{}".format(e))

            # 无论本次是否成功注入，都进入冷却期，避免下一代立即重复调用 API。
            stagnation_count = 0

        if gen_idx % 10 == 0 or gen_idx == GP_N_GENERATIONS:
            best_now = hof[0]
            best_info = eval_cache.get(str(best_now), {"mape": 999999.0, "fitness": 999999.0})
            print(
                "DEAP 第 {:02d} 代：best CV MAPE = {:.4f}%, tree_size = {}, "
                "reflection_calls = {}, injected = {}".format(
                    gen_idx,
                    best_info["mape"],
                    len(best_now),
                    reflection_calls,
                    reflection_injected
                )
            )

    best_ind = hof[0]
    best_info = eval_cache.get(str(best_ind))

    if best_info is None:
        toolbox.evaluate(best_ind)
        best_info = eval_cache.get(str(best_ind))

    best_mape = best_info["mape"]
    improvement = base_mape - best_mape

    try:
        evolved_code = gp_individual_to_code(
            individual=best_ind,
            expected_feature_name=evolved_feature_name,
            var_to_feature=var_to_feature
        )

        X_evolved, cleaned_evolved_code = apply_code_safe(
            X_input=X_current,
            code=evolved_code,
            allowed_input_features=current_features,
            expected_feature_name=evolved_feature_name
        )

    except Exception as e:
        print("DEAP 最优树转回 pandas 代码失败：{}".format(e))
        return None

    print("\nDEAP 进化得到的最优特征：{}".format(evolved_feature_name))
    print(cleaned_evolved_code)
    print("DEAP 最优 CV MAPE = {:.4f}%, 改善量 = {:.4f}".format(best_mape, improvement))
    print(
        "反思统计：触发 {} 次，评价定向候选 {} 个，成功注入 {} 个，"
        "唯一 GP 表达式评价总数 {} 个。".format(
            reflection_calls,
            reflection_candidates_evaluated,
            reflection_injected,
            len(eval_cache)
        )
    )

    source_name = "deap_gp_reflection" if reflection_injected > 0 else "deap_gp"

    return {
        "feature_name": evolved_feature_name,
        "X_candidate": X_evolved,
        "cleaned_code": cleaned_evolved_code,
        "candidate_features": current_features + [evolved_feature_name],
        "candidate_mape": best_mape,
        "improvement": improvement,
        "source": source_name,
        "parents": [top2[0]["feature_name"], top2[1]["feature_name"]],
        "reflection_calls": reflection_calls,
        "reflection_candidates_evaluated": reflection_candidates_evaluated,
        "reflection_injected": reflection_injected,
        "gp_unique_evaluations": len(eval_cache),
        "long_term_reflection": long_term_reflection
    }


# ============================================================
# 9. CAAFE 主流程
# ============================================================


def run_caafe_on_train(X_train_base, y_train, seed):
    feature_counter = 1

    X_current = X_train_base.copy()
    current_features = ORIGINAL_FEATURES.copy()

    accepted_codes = []
    accepted_features = []
    feedback = ""

    search_stats = {
        "reflection_calls": 0,
        "reflection_candidates_evaluated": 0,
        "reflection_injected": 0,
        "gp_unique_evaluations": 0,
        "gp_runs": 0,
    }

    for round_idx in range(1, MAX_ROUNDS + 1):
        if len(accepted_features) >= MAX_ACCEPTED_FEATURES:
            print("\n已达到最大接受特征数 {}，提前停止。".format(MAX_ACCEPTED_FEATURES))
            break

        candidate_names = []
        for _ in range(N_CANDIDATES_PER_ROUND):
            candidate_names.append("feat_{}".format(feature_counter))
            feature_counter += 1

        print("\n" + "-" * 60)
        print("CAAFE 第 {} 轮".format(round_idx))
        print("-" * 60)
        print("当前特征：", current_features)
        print("准备生成候选特征：", candidate_names)

        base_mape = cv_score_mape(
            X_df=X_current,
            y=y_train,
            features=current_features,
            seed=seed
        )

        print("生成前 CV MAPE = {:.4f}%".format(base_mape))

        prompt = build_caafe_prompt(
            X_df=X_current,
            used_features=current_features,
            new_feature_names=candidate_names,
            accepted_codes=accepted_codes,
            dataset_description=DATASET_DESCRIPTION,
            target=TARGET,
            feedback=feedback
        )

        try:
            raw_code = call_deepseek(prompt)

            print("\nDeepSeek 原始输出：")
            print(raw_code)

            candidate_results = []

            for candidate_name in candidate_names:
                print("\n正在评估候选特征：{}".format(candidate_name))

                try:
                    X_candidate, cleaned_code = apply_code_safe(
                        X_input=X_current,
                        code=raw_code,
                        allowed_input_features=current_features,
                        expected_feature_name=candidate_name
                    )

                    candidate_features = current_features + [candidate_name]

                    candidate_mape = cv_score_mape(
                        X_df=X_candidate,
                        y=y_train,
                        features=candidate_features,
                        seed=seed
                    )

                    improvement = base_mape - candidate_mape

                    print(
                        "{}: CV MAPE = {:.4f}%, 改善量 = {:.4f}".format(
                            candidate_name,
                            candidate_mape,
                            improvement
                        )
                    )

                    candidate_results.append({
                        "feature_name": candidate_name,
                        "X_candidate": X_candidate,
                        "cleaned_code": cleaned_code,
                        "candidate_features": candidate_features,
                        "candidate_mape": candidate_mape,
                        "improvement": improvement,
                        "source": "llm"
                    })

                except Exception as e:
                    print("{} 评估失败，错误原因：{}".format(candidate_name, e))

            if not candidate_results:
                print("本轮三个候选特征全部失败。")
                feedback = all_candidates_failed_feedback()
                continue

            sorted_candidates = sorted(
                candidate_results,
                key=lambda x: x["improvement"],
                reverse=True
            )

            best_llm_candidate = sorted_candidates[0]
            best_name = best_llm_candidate["feature_name"]
            best_mape = best_llm_candidate["candidate_mape"]
            best_improvement = best_llm_candidate["improvement"]

            print("\n本轮 LLM 原始最佳候选特征：{}".format(best_name))
            print("LLM 最佳候选 CV MAPE = {:.4f}%".format(best_mape))
            print("LLM 最佳改善量 = {:.4f}".format(best_improvement))

            evolved_candidate = None

            if USE_GP_EVOLUTION and len(sorted_candidates) >= 2:
                evolved_feature_name = "feat_{}".format(feature_counter)
                feature_counter += 1

                try:
                    evolved_candidate = evolve_top2_candidates_with_deap(
                        X_current=X_current,
                        y_train=y_train,
                        current_features=current_features,
                        base_mape=base_mape,
                        candidate_results=sorted_candidates,
                        evolved_feature_name=evolved_feature_name,
                        seed=seed
                    )
                except Exception as e:
                    print("DEAP 进化过程失败，退回只使用 LLM 原始候选。错误原因：{}".format(e))
                    evolved_candidate = None

            if evolved_candidate is not None:
                search_stats["gp_runs"] += 1
                search_stats["reflection_calls"] += evolved_candidate.get(
                    "reflection_calls", 0
                )
                search_stats["reflection_candidates_evaluated"] += evolved_candidate.get(
                    "reflection_candidates_evaluated", 0
                )
                search_stats["reflection_injected"] += evolved_candidate.get(
                    "reflection_injected", 0
                )
                search_stats["gp_unique_evaluations"] += evolved_candidate.get(
                    "gp_unique_evaluations", 0
                )

            # 在 LLM 原始最佳和 DEAP 进化最佳之间再次比较，取真正最好的一个。
            final_candidate = best_llm_candidate

            if evolved_candidate is not None:
                if evolved_candidate["improvement"] > final_candidate["improvement"]:
                    final_candidate = evolved_candidate

            final_name = final_candidate["feature_name"]
            final_mape = final_candidate["candidate_mape"]
            final_improvement = final_candidate["improvement"]
            final_source = final_candidate.get("source", "llm")

            print("\n本轮最终最佳特征：{}".format(final_name))
            print("来源：{}".format(final_source))
            print("最终最佳 CV MAPE = {:.4f}%".format(final_mape))
            print("最终最佳改善量 = {:.4f}".format(final_improvement))

            if final_improvement > MIN_IMPROVEMENT:
                print("接受特征：{}".format(final_name))

                X_current = final_candidate["X_candidate"]
                current_features = final_candidate["candidate_features"]
                accepted_features.append(final_name)
                accepted_codes.append(final_candidate["cleaned_code"])

                if final_source.startswith("deap_gp"):
                    feedback = gp_feature_accepted_feedback(
                        candidate_names=candidate_names,
                        parents=final_candidate.get("parents", []),
                        generations=GP_N_GENERATIONS,
                        feature_name=final_name,
                        base_mape=base_mape,
                        final_mape=final_mape,
                        feature_code=final_candidate["cleaned_code"],
                        reflection_calls=final_candidate.get("reflection_calls", 0),
                        reflection_injected=final_candidate.get("reflection_injected", 0)
                    )
                else:
                    feedback = llm_feature_accepted_feedback(
                        candidate_names=candidate_names,
                        feature_name=final_name,
                        base_mape=base_mape,
                        final_mape=final_mape
                    )
            else:
                print("拒绝本轮全部候选特征，包括 DEAP 进化特征。")

                feedback = candidates_rejected_feedback(
                    candidate_names=candidate_names,
                    best_feature_name=final_name,
                    base_mape=base_mape,
                    final_mape=final_mape
                )

        except Exception as e:
            print("本轮 DeepSeek 调用或整体处理失败。")
            print("错误原因：", e)

            feedback = generation_error_feedback(e)

            continue

    return X_current, current_features, accepted_codes, accepted_features, search_stats

def apply_accepted_codes_to_test(X_test_base, accepted_codes, final_features):
    X_test_current = X_test_base.copy()

    allowed_features = ORIGINAL_FEATURES.copy()

    for code in accepted_codes:
        matched = re.findall(r"df\[['\"](feat_\d+)['\"]\]", code)

        if not matched:
            continue

        expected_feature_name = matched[0]

        X_test_current, _ = apply_code_safe(
            X_input=X_test_current,
            code=code,
            allowed_input_features=allowed_features,
            expected_feature_name=expected_feature_name
        )

        allowed_features.append(expected_feature_name)

    return X_test_current[final_features]


# ============================================================
# 10. 主程序
# ============================================================

def main():
    global DATA_FILE, df, TARGET, ORIGINAL_FEATURES, DATASET_DESCRIPTION

    DATA_FILE, df, TARGET, ORIGINAL_FEATURES = load_dataset(
        data_path=DATA_PATH,
        target_column=TARGET_COLUMN,
    )

    DATASET_DESCRIPTION = (
        CUSTOM_DATASET_DESCRIPTION.strip()
        or GENERIC_DATASET_DESCRIPTION_TEMPLATE.format(
            target=TARGET,
            original_features=", ".join(map(str, ORIGINAL_FEATURES)),
        )
    )

    print("\n正在读取数据：{}".format(DATA_FILE))
    print("数据读取成功。")
    print("样本数：", len(df))
    print("目标列：", TARGET)
    print("原始特征：", ORIGINAL_FEATURES)

    results = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print("SEED = {}".format(seed))
        print("=" * 80)

        seed_start_time = time.perf_counter()
        api_calls_before = dict(API_CALL_COUNTER)

        df_train, df_test = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=seed,
            shuffle=True
        )

        X_train_base = df_train[ORIGINAL_FEATURES].reset_index(drop=True)
        y_train = df_train[TARGET].reset_index(drop=True).values

        X_test_base = df_test[ORIGINAL_FEATURES].reset_index(drop=True)
        y_test = df_test[TARGET].reset_index(drop=True).values

        (
            X_train_caafe,
            final_features,
            accepted_codes,
            accepted_features,
            search_stats,
        ) = run_caafe_on_train(
            X_train_base=X_train_base,
            y_train=y_train,
            seed=seed
        )

        print("\n最终接受的新特征：", accepted_features)
        print("最终用于训练的特征：", final_features)

        X_test_caafe = apply_accepted_codes_to_test(
            X_test_base=X_test_base,
            accepted_codes=accepted_codes,
            final_features=final_features
        )

        final_model = build_model(seed)
        final_model.fit(X_train_caafe[final_features], y_train)

        y_pred = final_model.predict(X_test_caafe)

        metrics = calc_metrics(y_test, y_pred)

        elapsed_seconds = time.perf_counter() - seed_start_time
        seed_api_calls = {
            key: API_CALL_COUNTER.get(key, 0) - api_calls_before.get(key, 0)
            for key in API_CALL_COUNTER
        }

        results.append({
            "seed": seed,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            "n_accepted_features": len(accepted_features),
            "accepted_features": ",".join(accepted_features),
            "gp_runs": search_stats["gp_runs"],
            "reflection_calls": search_stats["reflection_calls"],
            "reflection_candidates_evaluated": search_stats[
                "reflection_candidates_evaluated"
            ],
            "reflection_injected": search_stats["reflection_injected"],
            "gp_unique_evaluations": search_stats["gp_unique_evaluations"],
            "feature_generation_calls": seed_api_calls["feature_generation"],
            "short_reflection_calls": seed_api_calls["short_reflection"],
            "reflection_generation_calls": seed_api_calls[
                "reflection_generation"
            ],
            "long_reflection_calls": seed_api_calls["long_reflection"],
            "total_api_calls": sum(seed_api_calls.values()),
            "elapsed_seconds": elapsed_seconds,
        })

        print(
            "\n搜索统计：GP运行 {} 次，反思触发 {} 次，反思候选评价 {} 个，"
            "成功注入 {} 个，唯一GP表达式评价 {} 个。".format(
                search_stats["gp_runs"],
                search_stats["reflection_calls"],
                search_stats["reflection_candidates_evaluated"],
                search_stats["reflection_injected"],
                search_stats["gp_unique_evaluations"],
            )
        )

        print(
            "本次随机种子 API 调用：特征生成 {} 次，短期反思 {} 次，"
            "反思候选生成 {} 次，长期反思 {} 次，总计 {} 次；耗时 {:.2f} 秒。".format(
                seed_api_calls["feature_generation"],
                seed_api_calls["short_reflection"],
                seed_api_calls["reflection_generation"],
                seed_api_calls["long_reflection"],
                sum(seed_api_calls.values()),
                elapsed_seconds,
            )
        )

        print(
            "\n测试集结果：MAE={:.4f}, RMSE={:.4f}, R2={:.4f}, MAPE={:.4f}%"
            .format(
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
                metrics["mape"]
            )
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("{} 次实验结果".format(len(SEEDS)))
    print("=" * 80)
    print(results_df)

    print("\n" + "=" * 80)
    print("最终 {} 次平均结果".format(len(SEEDS)))
    print("=" * 80)

    print("MAE  = {:.4f}".format(results_df["mae"].mean()))
    print("RMSE = {:.4f}".format(results_df["rmse"].mean()))
    print("R²   = {:.4f}".format(results_df["r2"].mean()))
    print("MAPE = {:.4f}%".format(results_df["mape"].mean()))

    print("\n" + "=" * 80)
    print("标准差")
    print("=" * 80)

    print("MAE std  = {:.4f}".format(results_df["mae"].std()))
    print("RMSE std = {:.4f}".format(results_df["rmse"].std()))
    print("R² std   = {:.4f}".format(results_df["r2"].std()))
    print("MAPE std = {:.4f}%".format(results_df["mape"].std()))

    # if SAVE_RESULTS:
    #     results_df.to_excel(RESULTS_PATH, index=False)
    #     print("\n结果已保存到：{}".format(RESULTS_PATH))


if __name__ == "__main__":
    main()
