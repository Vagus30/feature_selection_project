# %load q01_plot_corr/build.py
# Default imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
plt.switch_backend('agg')
data = pd.read_csv('data/house_prices_multivariate.csv')


# Write your solution here:
def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = subplots(figsize=(size, size))
    set_cmap('YlOrRd')
    ax.matshow(corr)
    xticks(range(len(corr.columns)), corr.columns, rotation=90)
    yticks(range(len(corr.columns)), corr.columns)
    return ax,plt.matshow(df.corr())

