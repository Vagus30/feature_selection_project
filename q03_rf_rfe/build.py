# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    model = RandomForestClassifier()
    X,y = df.iloc[:,:-1], df.iloc[:,-1]
    rfe = RFE(model)
    rfe = rfe.fit(X,y)
    ranking = list(rfe.ranking_)
    top_features = []

    for i in range(len(ranking)):
        if rfe.ranking_[i] == 1:
            top_features.append(X.columns[i])

    return top_features
rf_rfe(data)

