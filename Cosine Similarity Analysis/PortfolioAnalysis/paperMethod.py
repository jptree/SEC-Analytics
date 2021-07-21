import pandas as pd
import numpy as np
import os
from PetriProgramming.PortfolioHelper import plot_cumulative_returns, summary_statistics
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 25)
pd.set_option('display.min_rows', 25)
pd.options.display.width = 150

# SIMILARITY_PATH_QUARTER = '../EntireFilePreviousYear/Cosine/CountVectorizerBinary/Quarterly'
# SIMILARITY_PATH_ANNUAL = '../EntireFilePreviousYear/Cosine/CountVectorizerBinary/Annual'

# SIMILARITY_PATH_QUARTER = '../MDA/Cosine/CountVectorizerBinary/Quarterly'
# SIMILARITY_PATH_ANNUAL = None


def get_data(path_quarter, path_annual):

    # Quarterly Similarity
    _, _, directories = next(os.walk(path_quarter))
    # TODO: This is all fucked. Multiple 10Q...
    similarity = pd.DataFrame()
    for d in directories:
        df = pd.read_csv(f'{path_quarter}\\{d}', header=None,
                         names=['date', 'cik', 'similarity'])
        similarity = similarity.append(df)


    # Annual Similarity
    if path_annual:
        _, _, directories = next(os.walk(path_annual))
        # TODO: This is all fucked. Multiple 10Q...
        for d in directories:
            df = pd.read_csv(f'{path_annual}\\{d}', header=None,
                             names=['date', 'cik', 'similarity'])
            similarity = similarity.append(df)

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

    df_all = pd.merge(CIK_merged, similarity, how='left', on=['cik', 'year', 'month'])
    df_all = df_all[df_all['year'] < 2019]

    # df_all['new_filing'] = df_all['similarity'].apply(lambda x: 0 if pd.isna(x) else 1)

    df_all['similarity'] = df_all.groupby('PERMNO')['similarity'].ffill()

    df_all = df_all[~df_all['similarity'].isna()]

    df_all = df_all[~df_all['RET'].isna()]

    df_all = df_all[(~df_all['PRC'].isna()) & (~df_all['SHROUT'].isna())]

    df_all = df_all[['PERMNO', 'date', 'SHRCD', 'EXCHCD', 'SICCD', 'TICKER', 'COMNAM', 'SHRCLS',
                     'CUSIP', 'PRC', 'RET', 'SHROUT', 'cik', 'similarity', 'filing_date']]

    return df_all


def get_portfolio_monthly(n_portfolios, factor_column, df):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Gets rid of some multiple stuff...
    # This needs to be fixed
    df = df.drop_duplicates(subset=['PERMNO', 'date'])

    # df['random'] = np.random.rand(df.shape[0])

    # df['similarity'] = df['similarity'].apply(lambda x: np.nan if x < 0.9 else x)
    # df['similarity'] = df.groupby('PERMNO')['similarity'].ffill()
    # df = df[~df['similarity'].isna()]

    df['market_cap'] = df['PRC'] * df['SHROUT']

    # Start where we have more data...
    df = df[df['year'] > 1996]
    # df.groupby('date')['PERMNO'].count().plot()
    # plt.show()

    df['similarity'] = df.groupby('PERMNO')['RET'].shift(-1)
    df = df[df['similarity'].notna()]

    df_eom = df.drop_duplicates(subset=['PERMNO', 'year', 'month'], keep='last')
    df_eom = df_eom[['PERMNO', 'date', 'year', 'month', 'quarter', factor_column, 'market_cap']]

    df_eom['group'] = df_eom.groupby(['year', 'month'])[factor_column].transform(
        lambda x: pd.qcut(x, n_portfolios, range(n_portfolios)))


    # df_eom['group'] = df_eom.groupby(['year', 'month'])[factor_column].transform(
    #     lambda x: pd.cut(x=x, bins=n_portfolios, labels=list(range(n_portfolios))))


    eom_group_market_cap = df_eom.groupby(['group', 'year', 'month'])['market_cap'].sum().reset_index()
    eom_group_market_cap = eom_group_market_cap.rename(columns={'market_cap': 'group_market_cap'})
    df_eom = pd.merge(df_eom, eom_group_market_cap, how='left', on=['group', 'year', 'month'])
    df_eom['weight'] = df_eom['market_cap'] / df_eom['group_market_cap']

    # Merge the group
    df_eom['merge_year'] = df_eom.apply(lambda x: x['year'] if x['month'] != 12 else x['year'] + 1, axis=1)
    df_eom['merge_month'] = df_eom['month'].apply(lambda x: x + 1 if x != 12 else 1)
    df_eom = df_eom[['merge_year', 'merge_month', 'PERMNO', 'group', 'group_market_cap', 'weight']]
    df = pd.merge(df, df_eom, how='left', left_on=['PERMNO', 'year', 'month'],
                  right_on=['PERMNO', 'merge_year', 'merge_month'])

    # df = df[~df['weight'].isna()]
    # df = df[~df['group'].isna()]

    # df.groupby('date')['PERMNO'].count().plot()
    # plt.show()
    #
    # # Validate sorting methodology
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


def get_portfolio_monthly_equal(n_portfolios, factor_column, df):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Gets rid of some multiple stuff...
    # This needs to be fixed
    df = df.drop_duplicates(subset=['PERMNO', 'date'])


    df['market_cap'] = df['PRC'] * df['SHROUT']

    # Start where we have more data...
    df = df[df['year'] > 1996]

    df_eom = df.drop_duplicates(subset=['PERMNO', 'year', 'month'], keep='last')
    df_eom = df_eom[['PERMNO', 'date', 'year', 'month', 'quarter', factor_column, 'market_cap']]


    df_eom['group'] = df_eom.groupby(['year', 'month'])[factor_column].transform(
        lambda x: pd.qcut(x, n_portfolios, range(n_portfolios)))

    # df_eom['group'] = df_eom.groupby(['year', 'month'])[factor_column].transform(
    #     lambda x: pd.cut(x=x, bins=n_portfolios, labels=list(range(n_portfolios))))



    # Merge the group
    df_eom['merge_year'] = df_eom.apply(lambda x: x['year'] if x['month'] != 12 else x['year'] + 1, axis=1)
    df_eom['merge_month'] = df_eom['month'].apply(lambda x: x + 1 if x != 12 else 1)
    df_eom = df_eom[['merge_year', 'merge_month', 'PERMNO', 'group']]
    df = pd.merge(df, df_eom, how='left', left_on=['PERMNO', 'year', 'month'],
                  right_on=['PERMNO', 'merge_year', 'merge_month'])



    df.groupby('date')['PERMNO'].count().plot()
    plt.show()

    # Validate sorting methodology
    a = df.groupby(['group', 'date'])['PERMNO'].count().unstack().T
    for i in range(n_portfolios):
        plt.plot(a[i], label=f'Portfolio {i}')
    plt.show()




    portfolios = df.groupby(['group', 'date'])['RET'].mean().unstack().T.reset_index()

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
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Gets rid of some multiple stuff...
    # This needs to be fixed
    df = df.drop_duplicates(subset=['PERMNO', 'date'])

    # df['random'] = np.random.rand(df.shape[0])

    # df['similarity'] = df['similarity'].apply(lambda x: np.nan if x < 0.9 else x)
    # df['similarity'] = df.groupby('PERMNO')['similarity'].ffill()
    # df = df[~df['similarity'].isna()]

    df['market_cap'] = df['PRC'] * df['SHROUT']

    # Start where we have more data...
    df = df[df['year'] > 1996]
    # df.groupby('date')['PERMNO'].count().plot()
    # plt.show()

    df_eom = df.drop_duplicates(subset=['PERMNO', 'year', 'month'], keep='last')
    df_eom = df_eom[['PERMNO', 'date', 'year', 'month', 'quarter', factor_column, 'market_cap']]

    df_eom['group'] = df_eom.groupby(['year', 'quarter'])[factor_column].transform(
        lambda x: pd.qcut(x, n_portfolios, range(n_portfolios)))

    eom_group_market_cap = df_eom.groupby(['group', 'year', 'quarter'])['market_cap'].sum().reset_index()
    eom_group_market_cap = eom_group_market_cap.rename(columns={'market_cap': 'group_market_cap'})
    df_eom = pd.merge(df_eom, eom_group_market_cap, how='left', on=['group', 'year', 'quarter'])
    df_eom['weight'] = df_eom['market_cap'] / df_eom['group_market_cap']

    # Merge the group
    df_eom['merge_year'] = df_eom.apply(lambda x: x['year'] if x['quarter'] != 4 else x['year'] + 1, axis=1)
    df_eom['merge_quarter'] = df_eom['quarter'].apply(lambda x: x + 1 if x != 4 else 1)
    df_eom = df_eom[['merge_year', 'merge_quarter', 'PERMNO', 'group', 'group_market_cap', 'weight']]
    df = pd.merge(df, df_eom, how='left', left_on=['PERMNO', 'year', 'quarter'],
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


def explore(df):
    print(df.columns)

    # ff3f = pd.read_csv('../../Data/FF3F.csv')
    # ff3f['Mkt-RF'] = ff3f['Mkt-RF'] / 100
    # ff3f['SMB'] = ff3f['SMB'] / 100
    # ff3f['HML'] = ff3f['HML'] / 100
    # ff3f['RF'] = ff3f['RF'] / 100
    # ff3f['Date'] = pd.to_datetime(ff3f['Date'], format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
    # ff3f = ff3f.rename(columns={'Date': 'date'})
    #
    # df['year'] = df['date'].dt.year
    # df['month'] = df['date'].dt.month
    #
    # ff3f['year'] = ff3f['date'].dt.year
    # ff3f['month'] = ff3f['date'].dt.month
    # ff3f = ff3f.drop(columns=['date'])
    #
    # df = pd.merge(df, ff3f, how='left', on=['year', 'month'])
    #
    # df = df.drop(columns=['year', 'month'])
    #
    # df['RET_M'] = df['RET'] - (df['Mkt-RF'] + df['RF'])


    df['RET_t_1'] = df.groupby('PERMNO')['RET'].shift(-1)
    a = df[['RET_t_1', 'similarity', 'RET']]
    a = a.dropna()
    a['random'] = np.random.rand(a.shape[0])
    # print(a.head())
    # plt.scatter(x=a['similarity'], y=a['RET_t_1'], s=10)
    # plt.xlabel('Similarity')
    # plt.ylabel('Return')
    #
    # plt.show()

    signal = a['RET_t_1']
    pos_signal = signal.copy()
    neg_signal = signal.copy()

    # pos_signal[pos_signal <= 0] = np.nan
    # neg_signal[neg_signal > 0] = np.nan

    pos_signal = []
    neg_signal = []
    returns = list(a['RET_t_1'])
    similarities = list(a['similarity'])
    randoms = list(a['random'])
    for i, r in enumerate(a['RET_t_1']):
        if r < 0:
            neg_signal.append(randoms[i])
            # neg_signal.append(similarities[i])
        else:
            # pos_signal.append(similarities[i])
            pos_signal.append(randoms[i])

    # plt.hist(pos_signal, bins=100)
    # plt.show()
    #
    # plt.hist(neg_signal, bins=100)
    # plt.show()



    bins = np.linspace(0, 1, 1000)

    plt.hist(neg_signal, bins, alpha=0.5, label='neg')
    plt.hist(pos_signal, bins, alpha=0.5, label='pos')
    plt.legend(loc='upper right')
    plt.show()


    # plotting
    # plt.style.use('fivethirtyeight')
    # plt.scatter(a['similarity'], pos_signal, color='r', s=10)
    # plt.scatter(a['similarity'], neg_signal, color='b', s=10)
    # # plt.savefig('pos_neg.png', dpi=200)
    # plt.show()


    # plt.scatter(x=a['RET'], y=a['RET_t_1'], s=10)
    # plt.xlabel('Current Return')
    # plt.ylabel('Next Period Return')
    #
    # plt.show()


if __name__ == "__main__":

    d = get_data(path_quarter='../EntireFilePreviousYear/Cosine/CountVectorizerBinary/Quarterly',
                 path_annual='../EntireFilePreviousYear/Cosine/CountVectorizerBinary/Annual')

    # d = get_data(path_quarter='../EntireFile/Cosine/CountVectorizer/Quarterly',
    #              path_annual=None)
    #
    # explore(d)


    n_portfolios = 10

    # p = get_portfolio_quarterly(5, 'similarity', d)
    p = get_portfolio_monthly(n_portfolios, 'similarity', d)
    # p.to_csv('test.csv')
    # p = get_portfolio_monthly(5, 'similarity', d)

    plot_cumulative_returns(p, n_portfolios)
    summary_statistics(p, n_portfolios)

    # Without annual statement similarity
    #  Model Portfolio  Alpha  t-Stat  Return    Risk  Sharpe
    #  0  FF3F         0  1.11%  1.3338   7.66%  15.44%  0.4962
    #  0  CAPM         0  1.27%  1.5123   7.66%  15.44%  0.4962
    #  0  FF3F         1  1.31%  1.5722   7.69%  15.19%   0.506
    #  0  CAPM         1  1.41%   1.667   7.69%  15.19%   0.506
    #  0  FF3F         2  1.41%  1.8999   8.08%  15.64%  0.5163
    #  0  CAPM         2  1.56%  2.0172   8.08%  15.64%  0.5163
    #  0  FF3F         3  3.34%  3.7504   10.1%  15.97%  0.6325
    #  0  CAPM         3  3.52%   3.831   10.1%  15.97%  0.6325
    #  0  FF3F         4  4.34%  4.5655  10.91%  16.03%  0.6808
    #  0  CAPM         4  4.32%  4.5679  10.91%  16.03%  0.6808
    #  0  FF3F      L4S0  3.22%  2.3094   3.25%   6.48%  0.5019
    #  0  CAPM      L4S0  3.05%  2.1854   3.25%   6.48%  0.5019
    #  0          Market                  8.63%  15.43%  0.5593