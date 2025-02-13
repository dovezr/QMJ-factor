"""
RUNNING STEP THREE

This code file achieve Quality-Based Stock Grouping to help confirm whether the 
Quality score has predictive power for returns.

Input: quality score, profitability score, growth score and safty score
Output: T-statistic and p-value for High-Low in each group.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import ttest_1samp


growth = pd.read_pickle("./features/aqr_growth.pkl")
profitability = pd.read_pickle("./features/aqr_profitability.pkl")
safty = pd.read_pickle("./features/aqr_safety.pkl")
quality = pd.read_pickle("./features/aqr_quality.pkl")
returns = pd.read_pickle("./monthly_returns.pkl")
common_dates = quality.index.intersection(returns.index)
common_stocks = quality.columns.intersection(returns.columns)
quality = quality.loc[common_dates, common_stocks]
returns = returns.loc[common_dates, common_stocks]
profitability = profitability.loc[common_dates, common_stocks]
safty = safty.loc[common_dates, common_stocks]
growth = growth.loc[common_dates, common_stocks]


def h_l(df):
    hl_results = {}

    for date in df.index:
        df_date = df.loc[date]
        returns_date = returns.loc[date]

        df_date = df_date.dropna()
        returns_date = returns_date.loc[df_date.index]

        labels = range(1, 11)
        df_groups = pd.qcut(df_date, q=10, labels=labels)
        group_returns = returns_date.groupby(df_groups, observed=False).mean()

        hl_diff = group_returns[10] - group_returns[1]
        hl_results[date] = hl_diff

    hl_df = pd.DataFrame.from_dict(hl_results, orient="index", columns=["H-L"])

    hl_values = hl_df["H-L"].dropna()
    t_stat, p_value = ttest_1samp(hl_values, popmean=0)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}")


h_l(quality)
h_l(profitability)
h_l(growth)
h_l(safty)
"""
T-statistic: 2.3477, P-value: 1.9502e-02
T-statistic: 3.3062, P-value: 1.0536e-03
T-statistic: -0.1940, P-value: 8.4628e-01
T-statistic: 3.3066, P-value: 1.0522e-03
The H-L difference is significantly non-zero. It can be concluded that after grouping 
    by the quality factor, there is a significant difference in returns between the high-quality 
    group (H) and the low-quality group (L), indicating that the quality factor may have predictive power.
"""
