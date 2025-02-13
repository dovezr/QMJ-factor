"""
RUNNING STEP FIVE

This code file judge whether the excess return for composite factor can 
not be explained by benchmark model(FF3, FF5, HXZ).
Input: qmj_equal, qmj_value
Output: 1. OLS regression results for FF3, FF5 and HXZ as independent variables
        2. T statistic, p-value, Alpha and Sharpe Ratio for 3 regressions.
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np


qmj_e = pd.read_csv("qmj_equal.csv")
qmj_v = pd.read_csv("qmj_value.csv")
ff5 = pd.read_csv("FF5.csv")
hxz = pd.read_csv("HXZ.csv")
qmj_e["date"] = pd.to_datetime(qmj_e["date"]).dt.strftime("%Y-%m-%d")
qmj_v["date"] = pd.to_datetime(qmj_e["date"]).dt.strftime("%Y-%m-%d")
ff5["date"] = pd.to_datetime(ff5["date"]).dt.strftime("%Y-%m-%d")
hxz["date"] = pd.to_datetime(hxz["date"]).dt.strftime("%Y-%m-%d")
qmj_e.set_index("date", inplace=True)
qmj_v.set_index("date", inplace=True)
ff5.set_index("date", inplace=True)
hxz.set_index("date", inplace=True)


def test_alpha(qmj, ff5, hxz):
    # This part is for data processing, making sure the all data sets are consistent
    common_dates = qmj.index.intersection(ff5.index)
    qmj = qmj.loc[common_dates, :]
    ff5 = ff5.loc[common_dates, :]

    # This part is for factor regression analysis using FF3 as benchmark
    X_3f = ff5[["MKT", "SMB", "HML"]]
    X_3f = sm.add_constant(X_3f)
    y = qmj["return"]
    print(X_3f, y)
    model_3f = sm.OLS(y, X_3f).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    alpha_3f = model_3f.params["const"]
    alpha_t_stat = model_3f.tvalues["const"]
    alpha_p_value = model_3f.pvalues["const"]

    print("Using FF3 as benchmark")
    print(f"t statistic:{alpha_t_stat}")
    print(f"p-value:{alpha_p_value}")
    print(f"Alpha: {alpha_3f}")
    print(model_3f.summary())
    sharpe_ratio = qmj["return"].mean() / np.std(y)
    print(f"Sharpe Ratio: {sharpe_ratio}")

    # This part is for factor regression analysis using FF5 as benchmark
    X_5f = ff5[["MKT", "SMB", "HML", "RMW", "CMA"]]
    X_5f = sm.add_constant(X_5f)
    y = qmj["return"]
    model_5f = sm.OLS(y, X_5f).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    alpha_5f = model_5f.params["const"]
    alpha_t_stat = model_5f.tvalues["const"]
    alpha_p_value = model_5f.pvalues["const"]

    print("Using FF5 as benchmark")
    print(f"t statistic:{alpha_t_stat}")
    print(f"p-value:{alpha_p_value}")
    print(f"Alpha: {alpha_5f}")
    print(model_5f.summary())
    sharpe_ratio = qmj["return"].mean() / np.std(y)
    print(f"Sharpe Ratio: {sharpe_ratio}")

    # This part is for data processing, making sure the all data sets are consistent
    common_dates = qmj.index.intersection(hxz.index)
    qmj = qmj.loc[common_dates, :]
    hxz = hxz.loc[common_dates, :]
    print(qmj.shape, ff5.shape, hxz.shape)

    # This part is for factor regression analysis using HXZ as benchmark
    X_4f = hxz[["MKT", "SIZE", "INV", "ROE"]]
    X_4f = sm.add_constant(X_4f)
    y = qmj["return"]
    model_4f = sm.OLS(y, X_4f).fit(cov_type="HAC", cov_kwds={"maxlags": 24})
    alpha_4f = model_4f.params["const"]
    alpha_t_stat = model_4f.tvalues["const"]
    alpha_p_value = model_4f.pvalues["const"]

    print("Using HXZ as benchmark")
    print(f"t statistic:{alpha_t_stat}")
    print(f"p-value:{alpha_p_value}")
    print(f"Alpha: {alpha_4f}")
    print(model_4f.summary())
    sharpe_ratio = qmj["return"].mean() / np.std(y)
    print(f"Sharpe Ratio: {sharpe_ratio}")


test_alpha(qmj_e, ff5, hxz)
test_alpha(qmj_v, ff5, hxz)
