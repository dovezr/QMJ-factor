"""
RUNNING STEP ONE

This code file is for ensuring which indicators are used to build the score using PCA.

For profitability score:
Input: GPOA, ROE, ROA, CFOA, GMAR and ACC
Output: a pictures show total explained variance ratio for profitability
        number of times for each indicator to be the most important to construct principal components 

For growth score:
Input: delta_GPOA, delta_ROE, delta_ROA and delta_CFOA
Output: a pictures show total explained variance ratio for growth
        number of times for each indicator to be the most important to construct principal components

For safety score:
Input: BAB, LEV, O and Z
Output: a pictures show total explained variance ratio for safety
        number of times for each indicator to be the most important to construct principal components

Finally choose two to three indicators to build each of profitability, growth and safety score
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from collections import Counter


# Data import
gpoa = pd.read_pickle("./features/gross_profit_to_asset_ttm.pkl")
roe = pd.read_pickle("./features/net_profit_to_book_value_ttm.pkl")
roa = pd.read_pickle("./features/net_profit_to_asset_ttm.pkl")
cfoa = pd.read_pickle("./features/net_cash_flow_to_asset_ttm.pkl")
gmar = pd.read_pickle("./features/gross_profit_margin_ttm.pkl")
acc = -pd.read_pickle("./features/total_accruals.pkl")
delta_gpoa = pd.read_pickle("./features/gross_profit_growth_to_asset.pkl")
delta_roe = pd.read_pickle("./features/net_profit_growth_to_book_value.pkl")
delta_roa = pd.read_pickle("./features/net_profit_growth_to_asset.pkl")
delta_cfoa = pd.read_pickle("./features/net_cash_flow_growth_to_asset.pkl")
delta_gmar = pd.read_pickle("./features/gross_profit_margin_mra_yoy_chg.pkl")
bab = -pd.read_pickle("./features/market_beta_252.pkl")
lev = -pd.read_pickle("./features/operating_leverage.pkl")
o = -pd.read_pickle("./features/ohlson_score.pkl")
z = -pd.read_pickle("./features/altman_zscore.pkl")


# This function input a dataframe and output a rank z score dataframe (rank in one row)
def compute_rank_z_score(df):
    ranks = df.rank(axis=1, method="average", na_option="bottom")
    rank_z_score = (
        ranks - ranks.mean(axis=1, skipna=True).values[:, None]
    ) / ranks.std(axis=1, skipna=True).values[:, None]
    return rank_z_score


# This part I ensure the indicator for constructing profitabiliy score using PCA
gpoa_z = compute_rank_z_score(gpoa)
roe_z = compute_rank_z_score(roe)
roa_z = compute_rank_z_score(roa)
cfoa_z = compute_rank_z_score(cfoa)
gmar_z = compute_rank_z_score(gmar)
acc_z = compute_rank_z_score(acc)

# pca1 is a list to store the explained variance ratio of PCA1
# pca2 is a list to store the explained variance ratio of PCA2
# explain is a list to store the total xplained variance ratio after PCA (which is PCA1 + PCA2)
# remain is a list to store the index of first contribute indicators for PCA1 and PCA2
explain = []
pca1 = []
pca2 = []
remain = []

# On each date, combine all the feature in one array and using PCA to reduce the dimension
for i in range(gpoa.shape[0]):
    # for i in range(1):
    features = pd.concat(
        [
            gpoa_z.iloc[i],
            roe_z.iloc[i],
            roa_z.iloc[i],
            cfoa_z.iloc[i],
            gmar_z.iloc[i],
            acc_z.iloc[i],
        ],
        axis=1,
    ).fillna(0)

    # Finishing the PCA with dimension 2
    pca = PCA(n_components=2)
    # Store the explained variance ratio for first, second and total component
    pca.fit(features)
    explain.append(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])
    pca1.append(pca.explained_variance_ratio_[0])
    pca2.append(pca.explained_variance_ratio_[1])

    pca_loadings_abs = abs(pca.components_)
    # store the original features with the greatest weight on PCA1
    remain.append(pca_loadings_abs[0].argmax())
    # store the original features with the greatest weight on PCA2
    remain.append(pca_loadings_abs[1].argmax())

    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = pca.components_

    # 创建一个 DataFrame 来查看每个主成分对应的载荷
    loadings_df = pd.DataFrame(
        loadings.T, columns=[f"PC{i+1}" for i in range(loadings.shape[0])]
    )
    contribution_df = loadings_df.apply(lambda x: x**2)  # 计算载荷的平方
    contribution_df = contribution_df.multiply(explained_variance_ratio, axis=1)

    # 显示每个指标对总方差的贡献
    print("Contribution of each variable to the total variance explained by each PC:")
    print(contribution_df)
count = Counter(remain)
print(count)
"""
result: Counter({5: 293, 2: 223, 0: 67, 1: 28, 3: 27, 4: 2})
choose acc_z, roa_z to form profitability score
"""


# Plot the explained variance ratio to ensure the PCA remain most of the part
plt.plot(
    explain, label="Total Explained Variance Ratio For Profitability", color="blue"
)
plt.plot(pca1, label="PCA1", color="green", linestyle="--")
plt.plot(pca2, label="PCA2", color="red", linestyle="-.")
plt.legend(loc="best")
plt.title("Explained Variance and PCA Components")
plt.xlabel("Index (or Date)")
plt.ylabel("Values")
plt.show()


# This part I ensure the indicator for constructing growth score using PCA. Same details as before.
delta_gpoa_z = compute_rank_z_score(delta_gpoa)
delta_roe_z = compute_rank_z_score(delta_roe)
delta_roa_z = compute_rank_z_score(delta_roa)
delta_cfoa_z = compute_rank_z_score(delta_cfoa)
delta_gmar_z = compute_rank_z_score(delta_gmar)


explain = []
pca1 = []
pca2 = []
remain = []
# On each date, combine all the feature in one array and using PCA to reduce the dimension
for i in range(gpoa.shape[0]):
    features = pd.concat(
        [
            delta_gpoa_z.iloc[i],
            delta_roe_z.iloc[i],
            delta_roa_z.iloc[i],
            delta_cfoa_z.iloc[i],
            delta_gmar_z.iloc[i],
        ],
        axis=1,
    ).fillna(0)

    # Finishing the PCA with dimension 2
    pca = PCA(n_components=2)
    # Store the explained variance ratio for first, second and total component
    pca.fit(features)
    explain.append(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])
    pca1.append(pca.explained_variance_ratio_[0])
    pca2.append(pca.explained_variance_ratio_[1])

    pca_loadings_abs = abs(pca.components_)
    remain.append(pca_loadings_abs[0].argmax())
    remain.append(pca_loadings_abs[1].argmax())
count = Counter(remain)
print(count)
"""
result: Counter({2: 295, 4: 292, 3: 28, 1: 13, 0: 12})
choose  delta_roa_z and delta_gmar_z to form growth score

"""
plt.plot(explain, label="Total Explained Variance Ratio For Growth", color="blue")
plt.plot(pca1, label="PCA1", color="green", linestyle="--")
plt.plot(pca2, label="PCA2", color="red", linestyle="-.")
plt.legend(loc="best")
plt.title("Explained Variance and PCA Components")
plt.xlabel("Index (or Date)")
plt.ylabel("Values")
plt.show()


# This part I ensure the indicator for constructing safety score using PCA. Same details as before.
bab_z = compute_rank_z_score(bab)
lev_z = compute_rank_z_score(lev)
o_z = compute_rank_z_score(o)
z_z = compute_rank_z_score(z)


explain = []
pca1 = []
pca2 = []
remain = []
# On each date, combine all the feature in one array and using PCA to reduce the dimension
for i in range(gpoa.shape[0]):
    features = pd.concat(
        [
            bab_z.iloc[i],
            lev_z.iloc[i],
            o_z.iloc[i],
            z_z.iloc[i],
        ],
        axis=1,
    ).fillna(0)

    # Finishing the PCA with dimension 2
    pca = PCA(n_components=2)
    # Store the explained variance ratio for first, second and total component
    pca.fit(features)
    explain.append(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])
    pca1.append(pca.explained_variance_ratio_[0])
    pca2.append(pca.explained_variance_ratio_[1])

    pca_loadings_abs = abs(pca.components_)
    remain.append(pca_loadings_abs[0].argmax())
    remain.append(pca_loadings_abs[1].argmax())
count = Counter(remain)
print(count)
"""
result: Counter({2: 236, 0: 226, 1: 119, 3: 59})
choose bab_z, lev_z, o_z to form safety_score
"""

plt.plot(explain, label="Total Explained Variance Ratio For Safety", color="blue")
plt.plot(pca1, label="PCA1", color="green", linestyle="--")
plt.plot(pca2, label="PCA2", color="red", linestyle="-.")
plt.legend(loc="best")
plt.title("Explained Variance and PCA Components")
plt.xlabel("Index (or Date)")
plt.ylabel("Values")
plt.show()


# Counter({5: 293, 2: 223, 0: 67, 1: 28, 3: 27, 4: 2})
# Counter({2: 295, 4: 292, 3: 28, 1: 13, 0: 12})
# Counter({2: 236, 0: 226, 1: 119, 3: 59})
