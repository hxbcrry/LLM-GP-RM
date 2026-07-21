# -*- coding: utf-8 -*-
"""各数据集的自然语言描述。更换数据集时主要修改本文件。"""

from textwrap import dedent


STOCK_DATASET_DESCRIPTION = dedent(
    """
    Stock Price Dataset.

    This is a multivariate time series regression/forecasting task for predicting future stock prices
    or identifying trends based on historical price data of 10 companies. The dataset contains
    sequential observations of stock prices, likely recorded at regular intervals (e.g., daily or weekly).

    The input variables include company1 through company10, representing the stock prices of 10
    different companies. The data is arranged in rows, where each row corresponds to a specific
    point in time (e.g., trading day), and each column represents a company's stock price at that time.
    The dataset spans a large number of time steps, exhibiting various market behaviors including
    trends, volatility, and potential regime changes.

    The relationship between past and future prices may be highly nonlinear, with complex temporal
    dependencies, autocorrelation, and cross-correlations among companies. Meaningful features may
    include lagged prices, rolling statistics (e.g., moving averages, volatility), differences
    (e.g., daily returns), relative strength indicators, and company interaction terms.

    Useful engineered features may include log returns, price spread between companies,
    normalized prices, rolling volatility (e.g., standard deviation over a window), momentum indicators
    (e.g., rate of change), and ratio-based features (e.g., price of company i relative to company j).
    Never use future price values to construct features when training predictive models.
    """
).strip()
