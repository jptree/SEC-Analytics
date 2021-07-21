import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import os
from PetriProgramming.PortfolioHelper import plot_cumulative_returns, summary_statistics

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 50)
pd.options.display.width = 0


def get_data2():
    _, _, directories = next(os.walk('../EntireFilePreviousYear/Cosine/CountVectorizerBinary/Quarterly'))
    # _, _, directories = next(os.walk('../EntireFile/Cosine/CountVectorizer/Quarterly'))

    similarity = pd.DataFrame()
    for d in directories:
        df = pd.read_csv(f'../EntireFilePreviousYear/Cosine/CountVectorizerBinary/Quarterly/{d}', header=None, names=['date', 'cik', 'similarity'])
        similarity = similarity.append(df)

    # similarity = pd.read_csv('../EntireFile/Cosine/CountVectorizer/Quarterly', index_col=0)
    similarity['date'] = pd.to_datetime(similarity['date'], format='%Y%m%d')

    returns = pd.read_csv('../../Data/Returns-CRSP.csv')
    returns['date'] = pd.to_datetime(returns['date'], format='%Y%m%d')
    returns['PRC'] = returns['PRC'].apply(lambda x: abs(x))
    returns['RET'] = pd.to_numeric(returns['RET'], errors='coerce')

    CIK_link = pd.read_csv('../../Data/CIK_Link.csv')
    CIK_link['merge_CUSIP'] = CIK_link['cusip'].apply(lambda x: str(x)[:8])
    CIK_link['datadate'] = pd.to_datetime(CIK_link['datadate'], format='%Y%m%d')

    # Don't merge on date, too?
    CIK_merged = pd.merge(returns, CIK_link, how='left', left_on=['date', 'CUSIP'], right_on=['datadate', 'merge_CUSIP'])
    df_all = pd.merge(CIK_merged, similarity, how='left', on=['cik', 'date'])

    df_all['similarity'] = df_all.groupby('PERMNO')['similarity'].ffill()

    df_all = df_all[~df_all['similarity'].isna()]
    df_all = df_all[~df_all['RET'].isna()]

    df_all = df_all[(~df_all['PRC'].isna()) & (~df_all['SHROUT'].isna())]


    df_all = df_all[['PERMNO', 'date', 'SHRCD', 'EXCHCD', 'SICCD', 'TICKER', 'COMNAM', 'SHRCLS',
                     'CUSIP', 'PRC', 'RET', 'SHROUT', 'cik', 'similarity']]

    return df_all


def get_data3():
    _, _, directories = next(os.walk('../EntireFilePreviousYear/Cosine/CountVectorizerBinary/Quarterly'))
    # _, _, directories = next(os.walk('../EntireFile/Cosine/CountVectorizer/Quarterly'))

    similarity = pd.DataFrame()
    for d in directories:
        df = pd.read_csv(f'../EntireFile/Cosine/CountVectorizer/Quarterly/{d}', header=None,
                         names=['date', 'cik', 'similarity'])
        similarity = similarity.append(df)

    # similarity = pd.read_csv('../EntireFile/Cosine/CountVectorizer/Quarterly', index_col=0)
    similarity['date'] = pd.to_datetime(similarity['date'], format='%Y%m%d')
    similarity['month'] = similarity['date'].dt.month
    similarity['year'] = similarity['date'].dt.year
    similarity = similarity.rename(columns={'date': 'filing_date'})

    returns = pd.read_csv('../../Data/Returns-CRSP.csv')
    returns['date'] = pd.to_datetime(returns['date'], format='%Y%m%d')
    returns['PRC'] = returns['PRC'].apply(lambda x: abs(x))
    returns['RET'] = pd.to_numeric(returns['RET'], errors='coerce')
    returns['year'] = returns['date'].dt.year
    returns['month'] = returns['date'].dt.month

    CIK_link = pd.read_csv('../../Data/CIK_Link.csv')
    CIK_link['merge_CUSIP'] = CIK_link['cusip'].apply(lambda x: str(x)[:8])
    CIK_link['datadate'] = pd.to_datetime(CIK_link['datadate'], format='%Y%m%d')
    CIK_link['month'] = CIK_link['datadate'].dt.month
    CIK_link['year'] = CIK_link['datadate'].dt.year

    CIK_merged = pd.merge(returns, CIK_link, how='left', left_on=['year', 'month', 'CUSIP'],
                          right_on=['year', 'month', 'merge_CUSIP'])
    # print(CIK_merged)

    df_all = pd.merge(CIK_merged, similarity, how='left', on=['cik', 'year', 'month'])
    df_all = df_all[df_all['year'] < 2019]

    # df_all['new_filing'] = df_all['similarity'].apply(lambda x: 0 if pd.isna(x) else 1)
    #
    df_all['similarity'] = df_all.groupby('PERMNO')['similarity'].ffill()

    df_all = df_all[~df_all['similarity'].isna()]
    # print(df_all)

    df_all = df_all[~df_all['RET'].isna()]

    df_all = df_all[(~df_all['PRC'].isna()) & (~df_all['SHROUT'].isna())]

    df_all = df_all[['PERMNO', 'date', 'SHRCD', 'EXCHCD', 'SICCD', 'TICKER', 'COMNAM', 'SHRCLS',
                     'CUSIP', 'PRC', 'RET', 'SHROUT', 'cik', 'similarity']]

    return df_all


def get_portfolio_returns2(n_portfolios, factor_column, df):
    """
    Rebalances portfolios monthly to weights determined at previous year's end, reconstitutes at previous year's end
    :param n_portfolios: number of portfolios for sorting
    :param factor_column: column that we are sorting on
    :param df: dataframe of data
    :return: returns monthly returns for all portfolios
    """
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df = df.drop_duplicates(subset=['PERMNO', 'date'])


    df['random'] = np.random.rand(df.shape[0])

    # df['similarity'] = df['similarity'].apply(lambda x: np.nan if x < 0.9 else x)
    # df['similarity'] = df.groupby('PERMNO')['similarity'].ffill()
    # df = df[~df['similarity'].isna()]




    df['market_cap'] = df['PRC'] * df['SHROUT']


    df_eoy = df[df['month'] == 12]
    df_eoy = df_eoy[['PERMNO', 'date', 'year', 'month', factor_column, 'market_cap']]

    df_eoy['group'] = df_eoy.groupby(['year'])[factor_column].transform(
        lambda x: pd.qcut(x, n_portfolios, range(n_portfolios)))

    eoy_group_market_cap = df_eoy.groupby(['group', 'year'])['market_cap'].sum().reset_index()
    eoy_group_market_cap = eoy_group_market_cap.rename(columns={'market_cap': 'group_market_cap'})
    df_eoy = pd.merge(df_eoy, eoy_group_market_cap, how='left', on=['group', 'year'])
    df_eoy['weight'] = df_eoy['market_cap'] / df_eoy['group_market_cap']


    # Merge the group
    df_eoy['merge_year'] = df_eoy['year'] + 1
    df_eoy = df_eoy[['merge_year', 'PERMNO', 'group', 'group_market_cap', 'weight']]
    df = pd.merge(df, df_eoy, how='left', left_on=['PERMNO', 'year'], right_on=['PERMNO', 'merge_year'])




    df = df[~df['weight'].isna()]
    df = df[~df['group'].isna()]

    a = df.groupby(['group', 'date'])['PERMNO'].count()

    for i in range(n_portfolios):
        plt.plot(a[i])

    plt.show()

    df['weighted_return'] = df['weight'] * df['RET']

    # df.to_csv('wtf.csv')

    portfolios = df.groupby(['group', 'date'])['weighted_return'].sum().unstack().T.reset_index()

    # TODO: Actual return here will be messed up.... fix later! ????
    # TODO: Rename columns and stuff
    # TODO: Add Long-Short
    # TODO: Why is weight above 1?

    ff3f = pd.read_csv('../../Data/FF3F.csv')
    ff3f['Mkt-RF'] = ff3f['Mkt-RF'] / 100
    ff3f['SMB'] = ff3f['SMB'] / 100
    ff3f['HML'] = ff3f['HML'] / 100
    ff3f['RF'] = ff3f['RF'] / 100
    ff3f['Date'] = pd.to_datetime(ff3f['Date'], format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
    ff3f = ff3f.rename(columns={'Date': 'date'})

    portfolios['year'] = portfolios['date'].dt.year
    portfolios['month'] = portfolios['date'].dt.month

    ff3f['year'] = ff3f['date'].dt.year
    ff3f['month'] = ff3f['date'].dt.month
    ff3f = ff3f.drop(columns=['date'])

    portfolios = pd.merge(portfolios, ff3f, how='left', on=['year', 'month'])

    portfolios = portfolios.drop(columns=['year', 'month'])

    return portfolios


def get_portfolio_quarterly(n_portfolios, factor_column, df):
    """
       Rebalances portfolios monthly to weights determined at previous year's end, reconstitutes at previous year's end
       :param n_portfolios: number of portfolios for sorting
       :param factor_column: column that we are sorting on
       :param df: dataframe of data
       :return: returns monthly returns for all portfolios
       """
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Gets rid of some multiple stuff...
    df = df.drop_duplicates(subset=['PERMNO', 'date'])

    # df['random'] = np.random.rand(df.shape[0])

    # df['similarity'] = df['similarity'].apply(lambda x: np.nan if x < 0.9 else x)
    # df['similarity'] = df.groupby('PERMNO')['similarity'].ffill()
    # df = df[~df['similarity'].isna()]

    df['market_cap'] = df['PRC'] * df['SHROUT']


    # Start where we have more data...
    df = df[df['year'] > 1996]
    df.groupby('date')['PERMNO'].count().plot()
    plt.show()

    df_eoq = df.drop_duplicates(subset=['PERMNO', 'year', 'quarter'], keep='last')
    df_eoq = df_eoq[['PERMNO', 'date', 'year', 'month', 'quarter', factor_column, 'market_cap']]

    df_eoq['group'] = df_eoq.groupby(['year', 'quarter'])[factor_column].transform(
        lambda x: pd.qcut(x, n_portfolios, range(n_portfolios)))

    eoq_group_market_cap = df_eoq.groupby(['group', 'year', 'quarter'])['market_cap'].sum().reset_index()
    eoq_group_market_cap = eoq_group_market_cap.rename(columns={'market_cap': 'group_market_cap'})
    df_eoq = pd.merge(df_eoq, eoq_group_market_cap, how='left', on=['group', 'year', 'quarter'])
    df_eoq['weight'] = df_eoq['market_cap'] / df_eoq['group_market_cap']

    # Merge the group
    df_eoq['merge_year'] = df_eoq.apply(lambda x: x['year'] if x['quarter'] != 4 else x['year'] + 1, axis=1)
    df_eoq['merge_quarter'] = df_eoq['quarter'].apply(lambda x: x + 1 if x != 4 else 1)
    df_eoq = df_eoq[['merge_year', 'merge_quarter', 'PERMNO', 'group', 'group_market_cap', 'weight']]
    df = pd.merge(df, df_eoq, how='left', left_on=['PERMNO', 'year', 'quarter'], right_on=['PERMNO', 'merge_year', 'merge_quarter'])

    # df = df[~df['weight'].isna()]
    # df = df[~df['group'].isna()]

    # df.groupby('date')['PERMNO'].count().plot()
    # plt.show()

    # Validate sorting methodology
    # a = df.groupby(['group', 'date'])['PERMNO'].count().unstack().T
    # for i in range(n_portfolios):
    #     plt.plot(a[i], label=f'Portfolio {i}')
    # plt.show()
    #
    # a = df.groupby(['group', 'date'])['weight'].sum().unstack().T
    # for i in range(n_portfolios):
    #     plt.plot(a[i], label=f'Portfolio {i}')
    # plt.show()

    df['weighted_return'] = df['weight'] * df['RET']


    portfolios = df.groupby(['group', 'date'])['weighted_return'].sum().unstack().T.reset_index()

    ff3f = pd.read_csv('../../Data/FF3F.csv')
    ff3f['Mkt-RF'] = ff3f['Mkt-RF'] / 100
    ff3f['SMB'] = ff3f['SMB'] / 100
    ff3f['HML'] = ff3f['HML'] / 100
    ff3f['RF'] = ff3f['RF'] / 100
    ff3f['Date'] = pd.to_datetime(ff3f['Date'], format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
    ff3f = ff3f.rename(columns={'Date': 'date'})

    portfolios['year'] = portfolios['date'].dt.year
    portfolios['month'] = portfolios['date'].dt.month

    ff3f['year'] = ff3f['date'].dt.year
    ff3f['month'] = ff3f['date'].dt.month
    ff3f = ff3f.drop(columns=['date'])

    portfolios = pd.merge(portfolios, ff3f, how='left', on=['year', 'month'])

    portfolios = portfolios.drop(columns=['year', 'month'])

    return portfolios


def get_portfolio_monthly(n_portfolios, factor_column, df):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Gets rid of some multiple stuff...
    df = df.drop_duplicates(subset=['PERMNO', 'date'])

    # df['random'] = np.random.rand(df.shape[0])

    # df['similarity'] = df['similarity'].apply(lambda x: np.nan if x < 0.9 else x)
    # df['similarity'] = df.groupby('PERMNO')['similarity'].ffill()
    # df = df[~df['similarity'].isna()]

    df['market_cap'] = df['PRC'] * df['SHROUT']

    # Start where we have more data...
    df = df[df['year'] > 1996]
    df.groupby('date')['PERMNO'].count().plot()
    plt.show()

    df_eoq = df.drop_duplicates(subset=['PERMNO', 'year', 'quarter'], keep='last')
    df_eoq = df_eoq[['PERMNO', 'date', 'year', 'month', 'quarter', factor_column, 'market_cap']]

    df_eoq['group'] = df_eoq.groupby(['year', 'quarter'])[factor_column].transform(
        lambda x: pd.qcut(x, n_portfolios, range(n_portfolios)))

    eoq_group_market_cap = df_eoq.groupby(['group', 'year', 'quarter'])['market_cap'].sum().reset_index()
    eoq_group_market_cap = eoq_group_market_cap.rename(columns={'market_cap': 'group_market_cap'})
    df_eoq = pd.merge(df_eoq, eoq_group_market_cap, how='left', on=['group', 'year', 'quarter'])
    df_eoq['weight'] = df_eoq['market_cap'] / df_eoq['group_market_cap']

    # Merge the group
    df_eoq['merge_year'] = df_eoq.apply(lambda x: x['year'] if x['quarter'] != 4 else x['year'] + 1, axis=1)
    df_eoq['merge_quarter'] = df_eoq['quarter'].apply(lambda x: x + 1 if x != 4 else 1)
    df_eoq = df_eoq[['merge_year', 'merge_quarter', 'PERMNO', 'group', 'group_market_cap', 'weight']]
    df = pd.merge(df, df_eoq, how='left', left_on=['PERMNO', 'year', 'quarter'],
                  right_on=['PERMNO', 'merge_year', 'merge_quarter'])

    # df = df[~df['weight'].isna()]
    # df = df[~df['group'].isna()]

    # df.groupby('date')['PERMNO'].count().plot()
    # plt.show()

    # Validate sorting methodology
    # a = df.groupby(['group', 'date'])['PERMNO'].count().unstack().T
    # for i in range(n_portfolios):
    #     plt.plot(a[i], label=f'Portfolio {i}')
    # plt.show()
    #
    # a = df.groupby(['group', 'date'])['weight'].sum().unstack().T
    # for i in range(n_portfolios):
    #     plt.plot(a[i], label=f'Portfolio {i}')
    # plt.show()

    df['weighted_return'] = df['weight'] * df['RET']

    portfolios = df.groupby(['group', 'date'])['weighted_return'].sum().unstack().T.reset_index()

    ff3f = pd.read_csv('../../Data/FF3F.csv')
    ff3f['Mkt-RF'] = ff3f['Mkt-RF'] / 100
    ff3f['SMB'] = ff3f['SMB'] / 100
    ff3f['HML'] = ff3f['HML'] / 100
    ff3f['RF'] = ff3f['RF'] / 100
    ff3f['Date'] = pd.to_datetime(ff3f['Date'], format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
    ff3f = ff3f.rename(columns={'Date': 'date'})

    portfolios['year'] = portfolios['date'].dt.year
    portfolios['month'] = portfolios['date'].dt.month

    ff3f['year'] = ff3f['date'].dt.year
    ff3f['month'] = ff3f['date'].dt.month
    ff3f = ff3f.drop(columns=['date'])

    portfolios = pd.merge(portfolios, ff3f, how='left', on=['year', 'month'])

    portfolios = portfolios.drop(columns=['year', 'month'])

    return portfolios


if __name__ == "__main__":
    d = get_data3()
    # d.to_csv('entireSimilarityData.csv')
    # d = pd.read_csv('../entireSimilarityData.csv', index_col=0)
    ports = get_portfolio_quarterly(5, 'similarity', d)
    plot_cumulative_returns(ports, 5)
    summary_statistics(ports, 5)
