# -*- coding: utf-8 -*-
"""只测试 Prompt 模块，不调用 DeepSeek，也不读取 stock.xlsx。"""

import pandas as pd

from prompts.caafe_prompts import build_caafe_prompt


def main():
    X = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 4.0, 8.0]})
    prompt = build_caafe_prompt(
        X_df=X,
        used_features=["A", "B"],
        new_feature_names=["feat_1", "feat_2", "feat_3"],
        accepted_codes=[],
        dataset_description="A small regression dataset.",
        target="target",
        feedback="",
    )
    print(prompt)


if __name__ == "__main__":
    main()
