import pandas as pd
import os
import numpy as np
from PetriProgramming.PortfolioHelper import plot_cumulative_returns, summary_statistics
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 25)
pd.set_option('display.min_rows', 25)
pd.options.display.width = 150


def get_similarity_data(dir):
    _, _, annual_directories = next(os.walk(f'{dir}/Annual'))
    _, _, quarterly_directories = next(os.walk(f'{dir}/Quarterly'))


    df_similarity = pd.DataFrame()

    for d in annual_directories:
        df = pd.read_csv(f'{dir}/Annual/{d}', header=None,
                         names=['date_int', 'cik', 'similarity'])
        df_similarity = df_similarity.append(df)
    for d in quarterly_directories:
        df = pd.read_csv(f'{dir}/Quarterly/{d}', header=None,
                         names=['date_int', 'cik', 'similarity'])
        df_similarity = df_similarity.append(df)

    df_similarity['year'] = df_similarity['date_int'].apply(lambda x: int(str(x)[:4]))
    df_similarity['month'] = df_similarity['date_int'].apply(lambda x: int(str(x)[4:6]))

    return df_similarity

def get_return_data():
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
    return CIK_merged


def get_data():
    CIK_merged = get_return_data()
    similarity = get_similarity_data('../../Cosine Similarity Analysis/EntireFilePreviousYear/Cosine/CountVectorizerBinary')

    df_all = pd.merge(CIK_merged, similarity, how='left', on=['cik', 'year', 'month'])
    df_all = df_all[df_all['year'] < 2019]

    df_all.to_csv('dataset_v2.csv')
    return df_all
    # print(df_all)


def portfolio(df, n_portfolios):
    # print(df)
    df['CAP'] = abs(df['PRC']) * df['SHROUT']
    desired_columns = ['similarity', 'month', 'year', 'PERMNO', 'date', 'RET', 'CAP']
    df = df[desired_columns]
    df = df[df['year'] > 1994]

    df_dropped = df.dropna(subset=['similarity'])

    df_dropped['group'] = df_dropped.groupby(['year', 'month'])['similarity'].transform(
        lambda x: pd.qcut(x, n_portfolios, range(n_portfolios)))
    df_dropped_group_capitalization = df_dropped.groupby(['year', 'month', 'group'])['CAP'].sum().reset_index()
    df_dropped_group_capitalization.columns = ['year', 'month', 'group', 'group_cap']


    df_dropped = pd.merge(df_dropped, df_dropped_group_capitalization, on=['year', 'month', 'group'])
    df_dropped['weight'] = df_dropped['CAP'] / df_dropped['group_cap']

    df_dropped = df_dropped[['date', 'PERMNO', 'weight', 'group']]

    df = pd.merge(df, df_dropped, on=['date', 'PERMNO'], how='left')

    df['date'] = pd.to_datetime(df['date'])

    df['weight'] = df['weight'].ffill(limit=2)

    df = df[~df['weight'].isna()]
    df = df[~df['group'].isna()]

    df.groupby('date')['PERMNO'].count().plot()
    plt.show()

    # Validate sorting methodology
    a = df.groupby(['group', 'date'])['PERMNO'].count().unstack().T
    for i in range(n_portfolios):
        plt.plot(a[i], label=f'Portfolio {i}')
    plt.show()

    a = df.groupby(['group', 'date'])['weight'].sum().unstack().T
    for i in range(n_portfolios):
        plt.plot(a[i], label=f'Portfolio {i}')
    plt.show()



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


    # print(df)
    # print(df.groupby(['year', 'month', 'PERMNO']).mean())
    # print(df.groupby(['year', 'month']).count())


def portfolio_testing():
    df = pd.DataFrame({
        "date": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10] + [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10],
        "stock": ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'] + ['C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D'],
        "weight": [.4, .4, .4, .4, .4, .6, .6, .6, .6, .6, .3, .3, .3, .3, .3, .7, .7, .7, .7, .7] + [.2, .2, .2, .2, .2, .8, .8, .8, .8, .8, .3, .3, .3, .3, .3, .7, .7, .7, .7, .7],
        "return": [0.1, 0.2, 0.1, 0.2, -0.1, -0.05, 0.05, 0.2, -0.05, 0.05, 0.05, -0.05, 0.05, 0.05, 0.2, -0.05, 0.05, -0.05, 0.05, 0.2] + [0.1, 0.2, 0.1, 0.2, -0.1, -0.05, 0.05, 0.2, -0.05, 0.05, 0.05, -0.05, 0.05, 0.05, 0.2, -0.05, 0.05, -0.05, 0.05, 0.2],
        "period": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "group": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    })

    df['cumulative'] = df.groupby(['group', 'stock', 'period']).apply(lambda x: np.cumprod(1 + x))['return']
    df['weighted_cumulative'] = df['cumulative'] * df['weight']
    df_portfolio = df.groupby(['group', 'period', 'date'])['weighted_cumulative'].sum().reset_index()
    df_portfolio['port_returns'] = df_portfolio.groupby(['group', 'period'])['weighted_cumulative'].pct_change()
    df_portfolio['port_returns'] = df_portfolio.apply(lambda x: x['weighted_cumulative'] / 1 - 1 if np.isnan(x['port_returns']) else x['port_returns'], axis=1)

    print(df_portfolio)



if __name__ == '__main__':
    # get_similarity_data('../../Cosine Similarity Analysis/EntireFilePreviousYear/Cosine/CountVectorizerBinary')
    # get_return_data()
    # get_data()
    # portfolio(get_data())

    # n_portfolios = 5
    # p = portfolio(pd.read_csv('dataset_v2.csv', index_col=0), n_portfolios)
    #
    # plot_cumulative_returns(p, n_portfolios)
    # summary_statistics(p, n_portfolios)

    portfolio_testing()

