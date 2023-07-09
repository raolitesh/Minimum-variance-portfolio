```python
'''This program attempts to optimize a users portfolio of 8 securities'''
#Import the python libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```
# List of the eight securities
```python
#Get the stock tickers in the portfolio 
assets = ['AAL','DIS','GM','JPM', 'PEP', 'AMD', 'NFLX', 'AAPL']
# Get the stock/ portfolio starting date
stockStartDate = '2021-11-03'
# Get the stocks' ending date (today)
today = datetime.today().strftime('%Y-%m-%d')
today
```
# Getting stock prices from Yahoo finance
```python
# Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()

# Store the adjusted close price of the sock into the df
for stock in assets:
  df[stock] = web.DataReader(stock,data_source='yahoo',start=stockStartDate,end=today)['Adj Close']
```
# calculating daily stock returns
```python
# show the daily simple return
returns = df.pct_change()
returns
```
# calculating mean stock returns
```python
mean = returns.mean()
mean
```
# calculating std. deviation stock returns
```python
stdev = returns.std()
stdev
```
# calculating covariance matrix stock returns
```python
# Create and show the annualized covariance matrix
cov_matrix_annual = returns.cov()*252
cov_matrix_annual
```
# calculating correlation matrix stock returns
```python
print(pd.DataFrame(returns).corr())
```
# Portfolio optimization algorithm
```python
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
```

```python
# Portfolio Optimization

# Calculate the expected returns and the annualized sample covariance matrix of asset returns
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimize for maximum sharpe ratio
ef = EfficientFrontier(mu,S,weight_bounds=(None,None))
ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3]+w[4]+w[5]+w[6]+w[7] == 1)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights() 
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

```

# Plotting the efficient frontier
```python
ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
ef.add_constraint(lambda w: w[0]+w[1]+w[2]+w[3]+w[4]+w[5]+w[6]+w[7] == 1)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
plt.title('Risk and return of portfolio')
plt.savefig('fig.png',bbox_inches='tight',dpi=300, transparent=False)
#plt.show()


```
