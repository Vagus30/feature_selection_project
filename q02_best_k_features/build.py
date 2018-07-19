# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    X = data.drop(['SalePrice'],axis=1)
    y = data['SalePrice']
    model = f_regression
    #f_regression.fit_transform(X,y)
    skb = SelectPercentile(model,percentile=20)
    predictors = skb.fit_transform(X,y)
    scores = list(skb.scores_)
    top_k_index = sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:predictors.shape[1]]
    top_k_predictores = [X.columns[i] for i in top_k_index]
    return top_k_predictores

