import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import math


def plot_cumulative_returns(df, n_portfolios, log=False):

    # Cumulative returns for portfolios
    for i in range(n_portfolios):
        cumulative_returns = np.cumprod(1 + df[i].values) - 1
        if log:
            plt.plot(df['date'], np.log(cumulative_returns), label=f'Portfolio {i}')
        else:
            plt.plot(df['date'], cumulative_returns, label=f'Portfolio {i}')

    # Cumulative return for the S&P 500
    sp500 = np.cumprod(1 + (df['Mkt-RF'] + df['RF']).values) - 1
    if log:
        plt.plot(df['date'], np.log(sp500), '--', label=f'S&P 500')
    else:
        plt.plot(df['date'], sp500, '--', label=f'S&P 500')


    # Cumulative return for Long-Short
    long_short = np.cumprod(1 + (df[n_portfolios - 1] - df[0]).values) - 1
    if log:

        plt.plot(df['date'], np.log(long_short), '--', label=f'Long {n_portfolios - 1}, Short 0')
    else:
        plt.plot(df['date'], long_short, '--', label=f'Long {n_portfolios - 1}, Short 0')



    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(f'Cumulative Return{": Log Returns" if log else ""}')
    plt.title(f'Cumulative Return of $1 for {n_portfolios} Portfolios')
    plt.show()


def ols_helper(est, model_type, portfolio):


    coefs = dict(est.params)
    # standard_error = est.bse
    r2 = est.rsquared
    r2_adj = est.rsquared_adj
    t_values = est.tvalues
    p_values = est.pvalues

    data = []
    for x in range(len(coefs)):
        d = [model_type, portfolio, r2, r2_adj, list(coefs.keys())[x], list(coefs.values())[x],
                t_values[x], p_values[x]]
        data.append(d)

    result = pd.DataFrame(np.array(data), columns=['Model', 'Portfolio', 'R^2', 'Adj. R^2', 'Factor', 'Coef.', 't-Value', 'p-Value'])
    return result


def summary_statistics(df, n_portfolios):

    d = pd.DataFrame()
    perf = pd.DataFrame()
    FF3F_X = sm.add_constant(df[['Mkt-RF', 'SMB', 'HML']])
    CAPM_X = sm.add_constant(df[['Mkt-RF']])
    for i in range(n_portfolios):

        y = df[i]

        FF3F_est = sm.OLS(y, FF3F_X).fit()
        CAPM_est = sm.OLS(y, CAPM_X).fit()

        ff3f = ols_helper(est=FF3F_est, model_type='FF3F', portfolio=i)
        capm = ols_helper(est=CAPM_est, model_type='CAPM', portfolio=i)

        d = d.append(ff3f)
        d = d.append(capm)

        perf = perf.append(portfolio_performance(y, FF3F_est, 'FF3F', i))
        perf = perf.append(portfolio_performance(y, CAPM_est, 'CAPM', i))


    # Long short
    y = (df[n_portfolios - 1] - df[0])

    FF3F_est = sm.OLS(y, FF3F_X).fit()
    CAPM_est = sm.OLS(y, CAPM_X).fit()

    ff3f = ols_helper(est=FF3F_est, model_type='FF3F', portfolio=f'L{n_portfolios - 1}S0')
    capm = ols_helper(est=CAPM_est, model_type='CAPM', portfolio=f'L{n_portfolios - 1}S0')

    d = d.append(ff3f)
    d = d.append(capm)

    perf = perf.append(portfolio_performance(y, FF3F_est, 'FF3F', f'L{n_portfolios - 1}S0'))
    perf = perf.append(portfolio_performance(y, CAPM_est, 'CAPM', f'L{n_portfolios - 1}S0'))

    # Market
    y = (df['Mkt-RF'] + df['RF'])
    CAPM_est = sm.OLS(y, CAPM_X).fit()
    perf = perf.append(portfolio_performance(y, CAPM_est, 'CAPM', f'Market', is_market=True))

    print(d)
    print(perf)

    # for i in range(n_portfolios):
    #     r = df[i].mean() * 12
    #     risk = df[i].std() * math.sqrt(12)
    #     sharpe = r / risk
    #
    #     print(f'Sharpe for Portfolio {i}: {sharpe}')
    #     print(f'Risk   for Portfolio {i}: {risk}')
    #     print(f'Return for Portfolio {i}: {r}')
    #
    # r = (df['Mkt-RF'] + df['RF']).mean() * 12
    # risk = (df['Mkt-RF'] + df['RF']).std() * math.sqrt(12)
    # sharpe = r / risk
    #
    # print(f'Sharpe for Market: {sharpe}')
    # print(f'Risk   for Market: {risk}')
    # print(f'Return for Market: {r}')

    # r = (df[n_portfolios - 1] - df[0]).mean() * 12
    # risk = (df[n_portfolios - 1] - df[0]).std() * math.sqrt(12)
    # sharpe = r / risk
    #
    # print(f'Sharpe for L{n_portfolios - 1}S{0}: {sharpe}')
    # print(f'Risk   for L{n_portfolios - 1}S{0}: {risk}')
    # print(f'Return for L{n_portfolios - 1}S{0}: {r}')


def portfolio_performance(returns, est, model_type, portfolio, is_market=False):
    coefs = dict(est.params)
    t_values = est.tvalues

    if not is_market:
        d = [
            model_type,
            portfolio,
            f'{round(list(coefs.values())[0] * 12 * 100, 2)}%',
            round(t_values[0], 4),
            f'{round(returns.mean() * 12 * 100, 2)}%',
            f'{round(returns.std() * math.sqrt(12) * 100, 2)}%',
            round((returns.mean() * 12) / (returns.std() * math.sqrt(12)), 4)
        ]
    else:
        d = [
            '',
            'Market',
            '',
            '',
            f'{round(returns.mean() * 12 * 100, 2)}%',
            f'{round(returns.std() * math.sqrt(12) * 100, 2)}%',
            round((returns.mean() * 12) / (returns.std() * math.sqrt(12)), 4)
        ]

    result = pd.DataFrame(np.array([d]),
                          columns=['Model', 'Portfolio', 'Alpha', 't-Stat', 'Return', 'Risk', 'Sharpe'])
    return result
