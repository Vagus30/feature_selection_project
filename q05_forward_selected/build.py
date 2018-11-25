# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


# Your solution code here
def forward_selected(data,model):
    X = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    current_selection = []
    feature_set = list(X.columns)
    best_scores = []
    while len(feature_set) > 0:
        scores_with_candidates = []
        for feature in feature_set:
            current_selection.append(feature)
            model.fit(X[current_selection], target)
            score = model.score(X[current_selection], target)
            scores_with_candidates.append((score, feature))
            current_selection.remove(feature)
        scores_with_candidates.sort()
        best_score, best_candidate = scores_with_candidates.pop()
        feature_set.remove(best_candidate)
        current_selection.append(best_candidate)
        best_scores.append(best_score)
    return current_selection, best_scores
forward_selected(data,model)
    

