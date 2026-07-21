
import re
import ast
import warnings
import os
import random
import operator
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd

from openai import OpenAI
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge,Lasso
#from tabpfn.constants import ModelVersion
from deap import base, creator, tools, gp
from sklearn.linear_model import Ridge

from prompts.caafe_prompts import SYSTEM_PROMPT, build_caafe_prompt
from prompts.feedback_prompts import (
    all_candidates_failed_feedback,
    gp_feature_accepted_feedback,
    llm_feature_accepted_feedback,
    candidates_rejected_feedback,
    generation_error_feedback,
)
from configs.dataset_descriptions import STOCK_DATASET_DESCRIPTION

warnings.filterwarnings("ignore")

#from tabpfn import TabPFNRegressor
# ============================================================
# 1. 需要你修改的配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = "xxxxxxxxxx"
DATA_FILE = PROJECT_ROOT / DATA_PATH

# 不要把 API Key 直接写进代码。
# Windows PowerShell：$env:DEEPSEEK_API_KEY="你的新密钥"
# Windows CMD：set DEEPSEEK_API_KEY=你的新密钥
DEEPSEEK_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "未找到数据文件：{}。请把 {} 放到脚本同级目录。".format(
            DATA_FILE, DATA_PATH
        )
    )

df = pd.read_excel(DATA_FILE)

TARGET = df.columns[-1]
ORIGINAL_FEATURES = df.columns[:-1].tolist()

# 数据集描述已经单独放在 configs/dataset_descriptions.py 中。
DATASET_DESCRIPTION = STOCK_DATASET_DESCRIPTION


# ============================================================
# 2. 实验参数
# ============================================================

SEEDS = [1, 3, 5, 7, 9, 11, 28, 42, 43, 46]

TEST_SIZE = 0.2

MAX_ROUNDS =5

MAX_ACCEPTED_FEATURES = 5

# 每一轮让 DeepSeek 一次生成的候选特征数量
N_CANDIDATES_PER_ROUND = 3

# 是否启用 DEAP 遗传编程：从三个候选里选改善最好的两个作为种子表达式
USE_GP_EVOLUTION = True

# GP 进化参数。50 代是你提出的设定；如果运行太慢，可以先改成 10 或 20 调试。
GP_N_GENERATIONS = 50
GP_POP_SIZE = 20
GP_CXPB = 0.75
GP_MUTPB = 0.25
GP_TOURN_SIZE = 3

# 精英保留数量。1 表示每一代强制保留历史最优个体；0 表示关闭精英保留。
GP_ELITE_SIZE = 1
GP_INIT_MIN_HEIGHT = 1
GP_INIT_MAX_HEIGHT = 3
GP_MUTATION_MAX_HEIGHT = 2
GP_MAX_TREE_HEIGHT = 5   #3

# 树越复杂，惩罚越大，避免进化出过长、过拟合、不可解释的表达式。
GP_PARSIMONY_COEF = 0.001

# MAPE 至少降低 0.1 才接受新特征
MIN_IMPROVEMENT = 0.0020


# ============================================================
# 3. DeepSeek 客户端
# ============================================================

def get_client():
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "未检测到环境变量 DEEPSEEK_API_KEY。\n"
            "请先在系统环境变量或当前终端中配置新的 DeepSeek API Key。"
        )

    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )


# ============================================================
# 4. 指标函数
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
# 5. 下游模型 森林  n_estimators=100,
#             random_state=seed,
#             min_samples_leaf=1
# ============================================================

def build_model(seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", Ridge(

        ))
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

def call_deepseek(prompt):
    client = get_client()

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        stream=False
    )

    return response.choices[0].message.content.strip()


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
                "length": len(individual)
            }

            return (fitness,)

        except Exception:
            eval_cache[key] = {
                "fitness": 999999.0,
                "mape": 999999.0,
                "length": len(individual)
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

        if gen_idx % 10 == 0 or gen_idx == GP_N_GENERATIONS:
            best_now = hof[0]
            best_info = eval_cache.get(str(best_now), {"mape": 999999.0, "fitness": 999999.0})
            print(
                "DEAP 第 {:02d} 代：best CV MAPE = {:.4f}%, tree_size = {}".format(
                    gen_idx,
                    best_info["mape"],
                    len(best_now)
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

    return {
        "feature_name": evolved_feature_name,
        "X_candidate": X_evolved,
        "cleaned_code": cleaned_evolved_code,
        "candidate_features": current_features + [evolved_feature_name],
        "candidate_mape": best_mape,
        "improvement": improvement,
        "source": "deap_gp",
        "parents": [top2[0]["feature_name"], top2[1]["feature_name"]]
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

                if final_source == "deap_gp":
                    feedback = gp_feature_accepted_feedback(
                        candidate_names=candidate_names,
                        parents=final_candidate.get("parents", []),
                        generations=GP_N_GENERATIONS,
                        feature_name=final_name,
                        base_mape=base_mape,
                        final_mape=final_mape,
                        feature_code=final_candidate["cleaned_code"]
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

    return X_current, current_features, accepted_codes, accepted_features

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
    print("\n正在读取数据：{}".format(DATA_FILE))
    # df = pd.read_excel(DATA_PATH)

    missing_cols = [c for c in ORIGINAL_FEATURES + [TARGET] if c not in df.columns]

    if missing_cols:
        raise ValueError("数据中缺少这些列：{}".format(missing_cols))

    print("数据读取成功。")
    print("样本数：", len(df))
    print("目标列：", TARGET)
    print("原始特征：", ORIGINAL_FEATURES)

    results = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print("SEED = {}".format(seed))
        print("=" * 80)

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

        X_train_caafe, final_features, accepted_codes, accepted_features = run_caafe_on_train(
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

        results.append({
            "seed": seed,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "mape": metrics["mape"],
            "n_accepted_features": len(accepted_features),
            "accepted_features": ",".join(accepted_features)
        })

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
    print("10 次实验结果")
    print("=" * 80)
    print(results_df)

    print("\n" + "=" * 80)
    print("最终 10 次平均结果")
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

    # results_df.to_excel("deepseek_caafe_results.xlsx", index=False)
    #
    # print("\n结果已保存到 deepseek_caafe_results.xlsx")


if __name__ == "__main__":
    main()
