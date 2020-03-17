# %% markdown
# # Stock Market Analysis
# %%
# This analysis consists of looking at data from the stock market.
# I'm gonna look specifically into technology stocks.

# By looking at the data, that are some questions that I came up with:

# 1. What is the change in price of the stock over time?
# 2. What is the daily return of the stock on average?
# 3. What is the moving average of the various stocks?
# 4.1. What is the correlation between different stocks' closing prices?
# 4.2. What is the correlation between different stocks' daily returns?
# 5. How much value is at risk if we invest in a particular stock?
# 6. How can we attempt to predict future stock behavior?
# %%
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# %%
from pandas_datareader import DataReader
# %%
from datetime import datetime
# %%
from __future__ import division
# %%
# This is a tech list of the big companies: Apple, Google, Microsoft, and Amazon

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
# %%
end = datetime.now()

start = datetime(end.year-1, end.month, end.day)
# %%
# This is a for loop for grabbing some financial data and setting it as a Dataframe

for stock in tech_list:
    globals()[stock] = DataReader(stock, 'yahoo', start, end)
# %%
AAPL.describe()
# %%
AAPL.info()
# %%
AAPL['Adj Close'].plot(legend=True, figsize=(14,7))
# %%
AAPL['Volume'].plot(legend=True, figsize=(14,7))
# %%
# Now let's calculate the moving average for the stock
# %%
# Calculating 3 different moving averages: 10 days, 20 days, and 50 days

ma_day = [10,20,50]

for ma in ma_day:
    column_name = 'MA for %s days' %(str(ma))

    AAPL[column_name] = pd.Series(AAPL['Adj Close']).rolling(window=ma).mean()
# %%
AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False, figsize=(14,7))
# %%
# If we get a moving average for more days at a time, we get a smoother line, and it's not gonna rely much on the daily
# fluctuation changes.
# %%
# Now retrieving the daily returns for Apple
# What that means is: for any given day, what is your percent return on your money?

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(14,7), legend=True, linestyle='--', marker='o')
# %%
# This is a histogram of the daily returns for the past year.

sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')
# %%
# It looks like the above histogram is skewed a little more negatively, but we need to do some more analysis to check
# that out.
# The following graph is just another way to see it.

AAPL['Daily Return'].hist(bins=50)
plt.gcf().set_size_inches(15, 8)
# %%
# Now building up another Dataframe with all the adjusted close columns for each of the stocks Dataframes in order to
# analyse the return of all the stocks in our data list.

closing_df = DataReader(tech_list, 'yahoo', start, end)['Adj Close']
# %%
closing_df.head()
# %%
tech_returns = closing_df.pct_change()
# %%
tech_returns.head()
# %%
# Now let's compare google to itself

sns.jointplot('GOOG', 'GOOG', tech_returns, kind='scatter', color='seagreen')
# %%
# That's a perfect linear relationship, and that makes sense, since we are comparing google to google.
# %%
# Now let's check if there are relationships between different tech stocks

sns.jointplot('GOOG', 'MSFT', tech_returns, kind='scatter', color='seagreen')
# %%
# Now let's do some plots that will make it easy to compare the tech stocks on our list

tech_returns.head()
# %%
sns.pairplot(tech_returns.dropna())
# %%
sns.pairplot(tech_returns.dropna(), kind="reg")
# %%
sns.pairplot(tech_returns.dropna(), kind="reg", diag_kind='kde')
# %%
# Just so we can have an idea on how to interpret these graphs:

from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')
# %%
# The above visualizations show a an interesting correlation between Google and Amazon daily returns
# We can dig a little deeper and use a PairGrid to see a more detailed and controled plot between those two.
# %%
returns_fig = sns.PairGrid(tech_returns.dropna())
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)
# %%
# Now, correlations between the closing prices

returns_fig = sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter, color='purple')
returns_fig.map_lower(sns.kdeplot, cmap='cool_d')
returns_fig.map_diag(plt.hist, bins=30)
# %%
# The above plot shows us an interesting correlation betweenMicrosoft and Apple.
# Let's see a correlation plot for the daily returns

sns.heatmap(tech_returns.dropna().corr(), annot=True)
plt.gcf().set_size_inches(13, 6)
# %%
sns.heatmap(closing_df.dropna().corr(), annot=True)
plt.gcf().set_size_inches(13, 6)
# %%
# By looking at all the built visualizations, we can conclude that all the big tech companies considered are correlated somehow.
# We can also conclude that there is a strong correlation of daily stock return between Amazon and Google.
# %%
# Let's see how we can quantify risk
# %%
# There are many ways of doing that, but I will use the information gathered on daily percentage returns and compare
# the expected return with the standard deviation of the daily returns.
# %%
rets = tech_returns.dropna()
# %%
area = np.pi*20
plt.scatter(rets.mean(), rets.std(), s = area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy = (x,y), xytext = (50,50),
                textcoords = 'offset points', ha = 'right', va= 'bottom',
                arrowprops = dict(arrowstyle = '-', color = 'black', connectionstyle = 'arc3,rad=-0.3'))

plt.gcf().set_size_inches(15, 8)
# %%
# We want a strong expect return with a lower risk.
# Based on the graph I would choose Microsoft. It has almost the same expected return as Apple (the highest one), but
# it would have a considerable lower risk.
# %%
# Now, let's take a look at value at risk (the amount of money we would expect to lose for a given confidence interval).
# %%
# This is the histogram about Apple Daily Returns

sns.distplot(AAPL['Daily Return'].dropna(), bins = 100, color='purple')
plt.gcf().set_size_inches(15, 8)
# %%
rets.head()
# %%
rets['AAPL'].quantile(0.05)
# %%
# This means with 95% confidence, or 95% of the simulations made with this, the worst
# daily loss would not exceed 2.07%.
# %%
# There is another way to quantify value at risk, which is by using the Monte Carlo method.
# The method consists on running many trials with random market conditions, then calculating portfolio losses for each trial.
# After that, we can use aggregation on all these simulations to stablish how risky the stock is.
# %%
days = 365
dt = 1/days
mu = rets.mean()['GOOG']
sigma = rets.std()['GOOG']
# %%
def stock_monte_carlo(start_price, days, mu, sigma):
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''

    # Define a price array
    price = np.zeros(days)
    price[0] = start_price
    # Schok and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)

    # Run price array for number of days
    for x in range(1,days):

        # Calculate Schock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))

    return price
# %%
GOOG.head()
# %%
# We can use the first opening price (1112.66) as the start_price

start_price = 1112.66

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')
plt.gcf().set_size_inches(15, 8)
# %%
# Now running a 1000 times and getting an array of the end points

# Set a large numebr of runs
runs = 10000

# Create an empty matrix to hold the end price data
simulations = np.zeros(runs)

# Set the print options of numpy to only display 0-5 points from an array to suppress output
np.set_printoptions(threshold=5)

for run in range(runs):
    # Set the simulation data point as the last stock price for that run
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];
# %%
# Now we'll define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)

# Now let's plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold')

# Making the plot bigger
plt.gcf().set_size_inches(15, 8)
# %%
# So, the value at risk is $35.63. This is means that, 99% of the time we run this MOnte Carlo simulation,
# the amount of money we would lose at most is $35.63. It doesn't look like a huge risk, given the initial investment ($1112.66).
