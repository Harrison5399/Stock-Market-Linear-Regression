<h3>Linear Regression to Predict Stock Price Movement</h2>
<p>
  In collaboration with Chinmay and Noah
</p>

<p>
  Our goal was to make a linear regression model that would out perform the S&P500 ETF
  Uses a linear regression model from sklearn to determine if next day would be positive/negative
  Lag day hyper parameter optimization to determine models memory of previous day's prices
  Using model created an stock indicator on a graph for buy/sell
</p>

<p>
  To run, download python/requirements.txt and run pip install -r requirements.txt
</p>
<p>
  run python/predictions.py to run model and generate graphs
</p>
<p>
  configure lag days and stock ticker in file
</p>
<p>
  run python/best_lag_n.py to generate graph of lag n to returns, warning: slow to run if range is high
</p>
