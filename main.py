from random import random
import pandas as pd
import adaboost_decision_stumps as ab
from sklearn.ensemble import AdaBoostClassifier


def my_adaboost(X_training, y_training, X_test, y_test):
    adaboost = ab.AdaboostDecisionStumps(10, weighted_gini=False)
    adaboost.fit(X_training, y_training)
    return adaboost.score(X_test, y_test)


def my_adaboost_weighted_gini(X_training, y_training, X_test, y_test):
    adaboost = ab.AdaboostDecisionStumps(10, weighted_gini=True)
    adaboost.fit(X_training, y_training)
    return adaboost.score(X_test, y_test)


def scikit_adaboost(X_training, y_training, X_test, y_test):
    adaboost = AdaBoostClassifier(n_estimators=10)
    adaboost.fit(X_training, y_training)
    return adaboost.score(X_test, y_test)


my_score_weighted_gini = []
my_score = []
scikit_score = []

beer = pd.read_csv("beer.txt", sep="\t", index_col=False)
beer = beer.drop("beer_id", axis=1)

for i in range(10):
    data = beer
    training = pd.DataFrame(columns=data.columns)
    test = pd.DataFrame(columns=data.columns)

    for j in range(0, int(len(data) / 3)):
        rand = int(random() * len(data))
        test = test.append(data.iloc[rand], ignore_index=True)
        data = data.drop(rand, axis=0)
        data = data.reset_index(drop=True)

    training = data
    X_training = training.drop("style", axis=1)
    y_training = training["style"]

    X_test = test.drop("style", axis=1)
    y_test = test["style"]

    print("Test", i+1)

    score = my_adaboost(X_training, y_training, X_test, y_test)
    print("\tMy score (weighted_gini = False):\t", score)
    my_score.append(score)

    score = my_adaboost_weighted_gini(X_training, y_training, X_test, y_test)
    print("\tMy score (weighted_gini = True):\t", score)
    my_score_weighted_gini.append(score)

    score = scikit_adaboost(X_training, y_training, X_test, y_test)
    print("\tScikit score:\t\t\t\t\t\t", score)
    scikit_score.append(score)

print("My AdaBoost (weighted_gini = False):",
      "\n\tMax:\t\t", max(my_score),
      "\n\tMin:\t\t", min(my_score),
      "\n\tAverage:\t", sum(my_score) / len(my_score))

print("My AdaBoost (weighted_gini = True):",
      "\n\tMax:\t\t", max(my_score_weighted_gini),
      "\n\tMin:\t\t", min(my_score_weighted_gini),
      "\n\tAverage:\t", sum(my_score_weighted_gini) / len(my_score_weighted_gini))

print("Scikit:",
      "\n\tMax:\t\t", max(scikit_score),
      "\n\tMin:\t\t", min(scikit_score),
      "\n\tAverage:\t", sum(scikit_score) / len(scikit_score))
