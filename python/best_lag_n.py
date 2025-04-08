import matplotlib.pyplot as plt

from predictor import *

data = yf.download("SPY", start="2010-01-01", multi_level_index=False)

values = []
errors = []

with open('lag_n.txt', 'w') as f:
    f.write('')

for lags in range(5, 50):
    df = data.copy()
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    df['returns'] = df.Close.pct_change()
    # df['Volume_pct_change'] = df['Volume'].pct_change()

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

    considered_attributes.remove('returns')

    test_df, model = gen_model(considered_attributes_df, considered_attributes)
    # returns_plot(test_df, f"returns: lags {lags}")
    # predicted_actual_plot(test_df, f"predicted: lags {lags}")
    # coefficent_plot(considered_attributes, model, f'coefficent: lags {lags}')

    exp_values = np.exp(test_df[['returns', 'strat_LR']].sum())
    sse_value = sse(test_df)
    r2_value = r2_error(test_df)

    values.append([exp_values[0], exp_values[1]])
    errors.append([sse_value, r2_value])

    with open('lag_n.txt', 'a') as f:
        f.write(f"Lags: {lags}\n")
        f.write(f"Original Returns:\n{exp_values[0]}\n")
        f.write(f"Model Returns:\n{exp_values[1]}\n")
        f.write(f"SSE: {sse_value:.5f}\n")
        f.write(f"R^2: {r2_value}\n")
        f.write("-" * 30 + "\n")


values = np.array(values)
index = np.arange(5, 5 + len(values))
plt.plot(index, values[:, 0], label='Normal Returns', color='blue')
plt.plot(index, values[:, 1], label='Model Returns', color='red')
plt.title('n lags return')
plt.xlabel('n lags')
plt.ylabel('return')
plt.legend()
plt.grid(True)
plt.show()

plt.close()

errors = np.array(errors)
index = np.arange(5, 5 + len(values))
plt.plot(index, errors[:, 0], label='SSE', color='blue')
plt.plot(index, errors[:, 1], label='R^2', color='red')
plt.title('n lags return')
plt.xlabel('n lags')
plt.ylabel('value')
plt.legend()
plt.grid(True)
plt.show()