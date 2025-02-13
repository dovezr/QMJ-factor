"""
RUNNING STEP TWO

This code file is for building profitability, growth, safety and quality score using the indicators
choosing in the last step. Besides, construct the quality score without growth score.

Input: Different indicators dataframe
Output: profitability_pca.csv, growth_pca.csv, safety_pca.csv, quality_pca.csv, quality_no_g.csv
"""

import pandas as pd
import numpy as np


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
lev = pd.read_pickle("./features/operating_leverage.pkl")
o = pd.read_pickle("./features/ohlson_score.pkl")
z = -pd.read_pickle("./features/altman_zscore.pkl")
quality = pd.read_pickle("./features/aqr_quality.pkl")


common_dates = quality.index.intersection(gpoa.index)
common_stocks = quality.columns.intersection(gpoa.columns)
quality = quality.loc[common_dates, common_stocks]
gpoa = gpoa.loc[common_dates, common_stocks]


# This function input a dataframe and output a rank z score dataframe (rank in one row)
def compute_rank_z_score(df):
    ranks = df.rank(axis=1, method="average", na_option="bottom")
    rank_z_score = (
        ranks - ranks.mean(axis=1, skipna=True).values[:, None]
    ) / ranks.std(axis=1, skipna=True).values[:, None]
    return rank_z_score


gpoa_z = compute_rank_z_score(gpoa)
roe_z = compute_rank_z_score(roe)
roa_z = compute_rank_z_score(roa)
cfoa_z = compute_rank_z_score(cfoa)
gmar_z = compute_rank_z_score(gmar)
acc_z = compute_rank_z_score(acc)
delta_gpoa_z = compute_rank_z_score(delta_gpoa)
delta_roe_z = compute_rank_z_score(delta_roe)
delta_roa_z = compute_rank_z_score(delta_roa)
delta_cfoa_z = compute_rank_z_score(delta_cfoa)
delta_gmar_z = compute_rank_z_score(delta_gmar)
bab_z = compute_rank_z_score(bab)
lev_z = compute_rank_z_score(lev)
o_z = compute_rank_z_score(o)
z_z = compute_rank_z_score(z)


profitability_score = compute_rank_z_score(acc_z.add(roa_z, fill_value=0))
growth_score = compute_rank_z_score(delta_roa_z.add(delta_gmar_z, fill_value=0))
safety_score = compute_rank_z_score(
    bab_z.add(lev_z, fill_value=0).add(o_z, fill_value=0)
)


quality_score = compute_rank_z_score(
    safety_score.add(growth_score).add(profitability_score)
)
quality_score[quality.isna()] = np.nan


quality_score_no_g = compute_rank_z_score(safety_score.add(profitability_score))
quality_score_no_g[quality.isna()] = np.nan


profitability_score.to_csv("profitability_pca.csv", index=True)
growth_score.to_csv("growth_pca.csv", index=True)
safety_score.to_csv("safety_pca.csv", index=True)
quality_score.to_csv("quality_pca.csv", index=True)
quality_score_no_g.to_csv("quality_no_g.csv", index=True)
