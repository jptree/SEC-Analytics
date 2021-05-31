import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    similarity = pd.read_csv('similarity.csv', header=None, names=['cik', 'date', 'similarity'])
    ff3f = pd.read_csv('ff3f.csv')
    security = pd.read_csv('security.csv')

    df_all = pd.merge(similarity, ff3f, how='left', on='date')
    df_all = pd.merge(security, df_all, how='left', on=['cik', 'date'])

    df_all['similarity'] = df_all.groupby('cik')['similarity'].ffill()
    df_all['date'] = pd.to_datetime(df_all['date'])

    return df_all


def get_portfolio_returns(n_portfolios, factor_column, df):
    """
    Rebalances portfolios monthly to weights determined at previous year's end, reconstitutes at previous year's end
    :param n_portfolios: number of portfolios for sorting
    :param factor_column: column that we are sorting on
    :param df: dataframe of data
    :return: returns monthly returns for all portfolios
    """
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df_eoy = df[df['month'] != 12]
    df_eoy = df_eoy[['cik', 'date', 'year', 'month', 'similarity']]

    df_eoy['group'] = df_eoy.groupby(['year'])[factor_column].transform(
        lambda x: pd.qcut(x, n_portfolios, range(n_portfolios)))

    df_eoy['group_market_cap'] = df_eoy.group('group')['market_cap'].sum()
    df_eoy['weight'] = df_eoy['market_cap'] / df_eoy['group_market_cap']

    # Merge the group
    df_eoy['merge_year'] = df_eoy['year'] + 1
    df_eoy = df_eoy[['merge_year', 'cik', 'group', 'group_market_cap']]

    df = pd.merge(df, df_eoy, how='left', left_on=['cik', 'year'], right_on=['cik', 'merge_year'])

    df['weighted_return'] = df['weight'] * df['return']

    portfolios = df.groupby(['group', 'date'])['weighted_return'].sum().reset_index()

    # TODO: Actual return here will be messed up.... fix later!
    # TODO: Attach FF3F here
    return portfolios


def plot_cumulative_returns(df, n_portfolios):

    # Cumulative returns for portfolios
    for i in range(n_portfolios):
        cumulative_returns = np.cumprod(1 + df[i].values) - 1
        plt.plot(df['date'], cumulative_returns, label=f'Portfolio {i}')

    # Cumulative return for the S&P 500
    sp500 = np.cumprod(1 + (df['Mkt-RF'] + df['RF']).values) - 1
    plt.plot(df['date'], sp500, label='S&P 500')

    # Cumulative return for Long-Short
    long_short = np.cumprod(1 + (df[n_portfolios - 1] - df[0]).values) - 1
    plt.plot(df['date'], long_short, label=f'Long {n_portfolios}, Short 0')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(f'Cumulative Return of $1 for {n_portfolios} Portfolios')
    plt.show()


def summary_statistics(df, n_portfolios):



if __name__ == "__main__":
    d = get_data()
    portfolios = get_portfolio_returns(5, 'similarity', d)
    plot_cumulative_returns(portfolios)