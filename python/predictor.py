import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ta import add_all_ta_features
import pandas as pd
from seaborn import heatmap
import pandas



def gen_model(df, attributes):
    # model = Ridge()
    # model = Lasso()
    model = LinearRegression()
    test_size = 0.3
    train, test = train_test_split(df, shuffle=False, test_size=test_size, random_state=0)

    model.fit(train[attributes], train['returns'])

    test['prediction_LR'] = model.predict(test[attributes])
    test['direction_LR'] = [1 if i > 0 else -1 for i in test.prediction_LR]
    test['strat_LR'] = test['direction_LR'] * test['returns']

    test['buy_sell'] = test['direction_LR'].diff().fillna(0).apply(lambda x: 1 if x == 2 else (-1 if x == -2 else 0))

    return test, model


def returns_plot(test, title='Market Return Compared to Strategy Returns'):

    plt.plot(test.index, np.exp(test['returns'].cumsum()), color='blue', linewidth=1.5, label='Market Return')
    plt.plot(test.index, np.exp(test['strat_LR'].cumsum()), color='orange', linewidth=1.5, label='Strategy Return')

    # add buy/sell vertical indicator
    # for index, row in test.iterrows():
    #     if row['buy_sell'] == 1:  # Buy signal
    #         plt.axvline(x=index, color='green', linewidth=0.1, alpha=1,
    #                     label='Buy' if 'Buy' not in plt.gca().get_legend_handles_labels()[1] else None)
    #     elif row['buy_sell'] == -1:  # Sell signal
    #         plt.axvline(x=index, color='red', linewidth=0.1, alpha=1,
    #                     label='Sell' if 'Sell' not in plt.gca().get_legend_handles_labels()[1] else None)
    plt.legend()

    plt.title(title)
    plt.ylabel("Principal/Loss (percent)")
    plt.show()

    plt.close()


def predicted_actual_plot(test, title='Actual vs Predicted'):
    plt.figure(figsize=(12, 6))

    plt.plot(test.index, test['returns'], label='Actual Returns', color='blue', linewidth=1.5)
    plt.plot(test.index, test['prediction_LR'], label='Predicted Returns (LR)', color='orange', linewidth=1.5)

    plt.title(title)
    plt.ylabel('Percentage Change')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()
    plt.close()

# calculating VIF, checking for which attributes have high multicollinearity
# https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def vif_values(df):
    X = add_constant(df)
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)

def coefficent_plot(attributes, model, title='Attribute Coefficients'):
    # attribute coefficient plot, showing importance of attribute on prediction
    plt.figure(figsize=(15, 15))
    plt.barh(attributes, model.coef_, color='green')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Attribute')
    plt.show()
    plt.close()


def correlation_matrix(df):
    # correlation matrix
    plt.figure(figsize=(20, 20))
    heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()
    plt.close()


def r2_error(test):
    return r2_score(test['returns'], test['prediction_LR'])


def sse(test):
    return np.sum((test['returns'] - test['prediction_LR']) ** 2)


def main():
    df = yf.download("SPY", start="2010-01-01", end="2024-12-16", multi_level_index=False)
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    df['returns'] = df.Close.pct_change()
    # df['Volume_pct_change'] = df['Volume'].pct_change()

    lags = 32
    # lags = 10

    def gen_lags(df):
        added_attributes = []
        for i in range(1, lags + 1):
            df['Lag_' + str(i)] = df['returns'].shift(i)
            added_attributes.append('Lag_' + str(i))
        return added_attributes

    attributes = gen_lags(df)


    # considered attributes to determine final attributes after correlation matrix and VIF
    considered_attributes = []
    considered_attributes.extend(attributes)
    considered_attributes.append('Volume')
    # considered_attributes.append('Volume_pct_change')
    # considered_attributes.append('volume_em')
    # considered_attributes.append('volume_cmf')
    # considered_attributes.append('trend_macd')
    considered_attributes.append('trend_macd_signal')
    # considered_attributes.append('trend_sma_slow')
    # considered_attributes.append('trend_sma_fast')
    considered_attributes.append('trend_cci')
    # considered_attributes.append('momentum_rsi')

    # lag ta attributes so there is no look ahead bias
    shifted_attributes = {}
    for considered_attribute in considered_attributes:
        if considered_attribute not in attributes:
            shifted_attributes[considered_attribute + '_lagged'] = df[considered_attribute].shift(1)

    df = pd.concat([df, pd.DataFrame(shifted_attributes, index=df.index)], axis=1)

    considered_attributes = [attr + '_lagged' if attr not in attributes else attr
        for attr in considered_attributes]

    considered_attributes.append('returns')

    considered_attributes_df = df.copy()
    considered_attributes_df = considered_attributes_df.drop(columns=[col for col in df.columns if col not in considered_attributes])

    considered_attributes_df.dropna(inplace=True)

    correlation_matrix(considered_attributes_df)


    vif_values(considered_attributes_df)
    considered_attributes.remove('returns')

    test_df, model = gen_model(considered_attributes_df, considered_attributes)
    returns_plot(test_df)
    predicted_actual_plot(test_df)
    coefficent_plot(considered_attributes, model)

    with pandas.option_context('display.max_columns', None):
        print(test_df)

    print(f'normal return: {np.exp(test_df[['returns', 'strat_LR']].sum())[0]}')
    print(f'strategy return: {np.exp(test_df[['returns', 'strat_LR']].sum())[1]}')

    print(f"sse: {sse(test_df):.5f}")
    print(f"r^2: {r2_error(test_df)}")

main()