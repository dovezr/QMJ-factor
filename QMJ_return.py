"""
RUNNING STEP FOUR

This code file achieve conditional double sorting for calculating the monthly returns for QMJ factor.

Input: quality score, quality score build after PCA and quality score build without growth score
Output: 1. The T-statistic, P-value and average excess return for
           value-weighted portfolio using quality score without PCA,
           equal-weighted portfolio using quality score without PCA,
           value-weighted portfolio using quality score with PCA,
           equal-weighted portfolio using quality score with PCA,
           value-weighted portfolio using new quality score,
           equal-weighted portfolio using new quality score.
        2. qmj_value.csv, qmj_equal.csv, qmj_value_pca.csv, qmj_equal_pca.csv, qmj_value_n_g.csv, qmj_equal_n_g.csv
        3. The excess return figure against with date for
           value-weighted portfolio using quality score without PCA,
           equal-weighted portfolio using quality score without PCA,
           value-weighted portfolio using quality score with PCA,
           equal-weighted portfolio using quality score with PCA,
           value-weighted portfolio using new quality score,
           equal-weighted portfolio using new quality score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats


growth = pd.read_pickle("./features/aqr_growth.pkl")
profitability = pd.read_pickle("./features/aqr_profitability.pkl")
safty = pd.read_pickle("./features/aqr_safety.pkl")
quality = pd.read_pickle("./features/aqr_quality.pkl")
mcap = pd.read_pickle("./mcap.pkl")
returns = pd.read_pickle("./monthly_returns.pkl")
quality_pca = pd.read_csv("./quality_pca.csv")
quality_pca.set_index("date", inplace=True)
safety_pca = pd.read_csv("./safety_pca.csv")
safety_pca.set_index("date", inplace=True)
quality_n_g = pd.read_csv("./quality_no_g.csv")


# This function is used to finish value-weighted portfolio
# list: The list for the stock name in a sort
# total: In this type, the total market capitalization
# returns: In one date, returns for all the stocks in type of dataframe
# size: In one date, market capitalization for all the stocks in type of dataframe
# return: The return of the value-weighted portfolio
def cal_return(list, total, returns, size):
    num = 0
    for stock in list:
        if pd.isna(returns[stock]) or pd.isna(size[stock]):
            continue
        num += (size[stock] / total) * returns[stock]
    return num


# This part is for data processing, making sure the all data sets are consistent
common_dates = quality.index.intersection(mcap.index).intersection(returns.index)
common_stocks = quality.columns.intersection(mcap.columns).intersection(returns.columns)
quality = quality.loc[common_dates, common_stocks]
mcap = mcap.loc[common_dates, common_stocks]
returns = returns.loc[common_dates, common_stocks]
growth = growth.loc[common_dates, common_stocks]
profitability = profitability.loc[common_dates, common_stocks]
safty = safty.loc[common_dates, common_stocks]


###  This part finished conditional sorts, first sorting on size and then on quality
def conditional_double_sort(mcap, sort_df, returns):
    # Build a dataframe to store the qmj in every dates(except the first day)
    # df_v for value-weighted, df_e for equal-weighted
    df_v = pd.DataFrame(index=returns.index[1:])
    df_v["return"] = None
    df_e = pd.DataFrame(index=returns.index[1:])
    df_e["return"] = None

    for i in range(returns.shape[0] - 1):
        # for i in range(1):
        quality_score = sort_df.iloc[i]
        size = mcap.iloc[i]
        returns_next = returns.iloc[i + 1]

        # First I use the size breakpoint to sort big and small
        size_cutoffs = np.percentile(size.dropna(), 50)
        big = []
        small = []
        for stock, stock_size in size.items():
            if stock_size >= size_cutoffs:
                big.append(stock)
            else:
                small.append(stock)

        # Then I use the quality score breakpoint (the 30th and 70th percentile) to sort quality and junk
        big_quality = []
        big_junk = []
        small_quality = []
        small_junk = []

        big_quality_score = quality_score[big]
        big_quality_cutoffs_low = np.percentile(big_quality_score.dropna(), 30)
        big_quality_cutoffs_high = np.percentile(big_quality_score.dropna(), 70)

        for stock in big:
            stock_quality = quality_score[stock]
            if pd.isna(stock_quality):
                continue
            if stock_quality >= big_quality_cutoffs_high:
                big_quality.append(stock)
            elif stock_quality <= big_quality_cutoffs_low:
                big_junk.append(stock)

        small_quality_score = quality_score[small]
        small_quality_cutoffs_low = np.percentile(small_quality_score.dropna(), 30)
        small_quality_cutoffs_high = np.percentile(small_quality_score.dropna(), 70)

        for stock in small:

            stock_quality = quality_score[stock]
            if pd.isna(stock_quality):
                continue
            if stock_quality >= small_quality_cutoffs_high:
                small_quality.append(stock)
            elif stock_quality <= small_quality_cutoffs_low:
                small_junk.append(stock)

        # Calculate the value-weighted factor return
        big_quality_size = size[big_quality].sum()
        big_junk_size = size[big_junk].sum()
        small_quality_size = size[small_quality].sum()
        small_junk_size = size[small_junk].sum()
        sq = cal_return(small_quality, small_quality_size, returns_next, size)
        bq = cal_return(big_quality, big_quality_size, returns_next, size)
        sj = cal_return(small_junk, small_junk_size, returns_next, size)
        bj = cal_return(big_junk, big_junk_size, returns_next, size)
        df_v.iloc[i, 0] = 0.5 * (sq + bq) - 0.5 * (sj + bj)

        # Calculate the equal-weighted factor return
        sq_e = np.mean(
            [
                returns_next[stock]
                for stock in small_quality
                if not pd.isna(returns_next[stock])
            ]
        )
        bq_e = np.mean(
            [
                returns_next[stock]
                for stock in big_quality
                if not pd.isna(returns_next[stock])
            ]
        )
        sj_e = np.mean(
            [
                returns_next[stock]
                for stock in small_junk
                if not pd.isna(returns_next[stock])
            ]
        )
        bj_e = np.mean(
            [
                returns_next[stock]
                for stock in big_junk
                if not pd.isna(returns_next[stock])
            ]
        )
        df_e.iloc[i, 0] = 0.5 * (sq_e + bq_e) - 0.5 * (sj_e + bj_e)
        # Finally I calculate the QMJ factor return
    return df_v, df_e


qmj_v, qmj_e = conditional_double_sort(mcap, quality, returns)

# Plot the monthly return of the conditional double sort(x:date, y:return)
plt.figure(figsize=(10, 6))
plt.plot(
    qmj_v.index,
    qmj_v["return"],
    color="b",
    label="value-weighted",
)
plt.plot(
    qmj_e.index,
    qmj_e["return"],
    color="r",
    label="equal-weighted",
)
plt.title("Factor Monthly Returns")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Return", fontsize=12)
plt.legend(loc="best")
plt.show()

print("For value-weighted portfolio using quality score without pca")
t_stat, p_value = stats.ttest_1samp(pd.to_numeric(qmj_v["return"]), 0)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print("For equal-weighted portfolio using quality score without pca")
t_stat, p_value = stats.ttest_1samp(pd.to_numeric(qmj_e["return"]), 0)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Import the results into excel
qmj_v.to_csv("qmj_value.csv", index=True)
qmj_e.to_csv("qmj_equal.csv", index=True)

print(quality_pca)
qmj_v_pca, qmj_e_pca = conditional_double_sort(mcap, quality_pca, returns)
plt.figure(figsize=(10, 6))
plt.plot(
    qmj_v_pca.index,
    qmj_v_pca["return"],
    color="b",
    label="value-weighted",
)
plt.plot(
    qmj_e_pca.index,
    qmj_e_pca["return"],
    color="r",
    label="equal-weighted",
)
plt.title("Factor Monthly Returns")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Return", fontsize=12)
plt.legend(loc="best")
plt.show()

print("For value-weighted portfolio using quality score with pca")
t_stat, p_value = stats.ttest_1samp(pd.to_numeric(qmj_v_pca["return"]), 0)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print("For equal-weighted portfolio using quality score with pca")
t_stat, p_value = stats.ttest_1samp(pd.to_numeric(qmj_e_pca["return"]), 0)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# Import the results into excel
qmj_v_pca.to_csv("qmj_value_pca.csv", index=True)
qmj_e_pca.to_csv("qmj_equal_pca.csv", index=True)


print(qmj_v.mean())
print(qmj_e.mean())
print(qmj_v_pca.mean())
print(qmj_e_pca.mean())

"""
For value-weighted portfolio using quality score without pca
T-statistic: 2.3014070250831686
P-value: 0.022014878104763574
For equal-weighted portfolio using quality score without pca
T-statistic: 2.8257198396969363
P-value: 0.005015804834867188
For value-weighted portfolio using quality score with pca
T-statistic: 1.6055306783786103
P-value: 0.10936962464686463
For equal-weighted portfolio using quality score with pca
T-statistic: 2.5786592334521714
P-value: 0.010367729312597955
return    0.004021
dtype: object
return    0.004894
dtype: object
return    0.003316
dtype: object
return    0.004963
dtype: object
"""

qmj_v_n_g, qmj_e_n_g = conditional_double_sort(mcap, quality_n_g, returns)

# Plot the monthly return of the conditional double sort(x:date, y:return)
plt.figure(figsize=(10, 6))
plt.plot(
    qmj_v_n_g.index,
    qmj_v_n_g["return"],
    color="b",
    label="value-weighted",
)
plt.plot(
    qmj_e_n_g.index,
    qmj_e_n_g["return"],
    color="r",
    label="equal-weighted",
)
plt.title("Factor Monthly Returns")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Return", fontsize=12)
plt.legend(loc="best")
plt.show()

print("For value-weighted portfolio using new quality score")
t_stat, p_value = stats.ttest_1samp(pd.to_numeric(qmj_v_n_g["return"]), 0)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print("For equal-weighted portfolio using new quality score")
t_stat, p_value = stats.ttest_1samp(pd.to_numeric(qmj_e_n_g["return"]), 0)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Import the results into excel
qmj_v_n_g.to_csv("qmj_value_n_g.csv", index=True)
qmj_e_n_g.to_csv("qmj_equal_n_g.csv", index=True)

print(qmj_v_n_g.mean())
print(qmj_e_n_g.mean())

"""
For value-weighted portfolio using new quality score
T-statistic: 2.040352491254767
P-value: 0.042140979446496245
For equal-weighted portfolio using new quality score
T-statistic: 3.154158993555601
P-value: 0.001763325435638745
return    0.003763
dtype: object
return    0.005309
dtype: object
"""
