# Importing Libraries.
# -------------------------------------

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


# Getting the dataset
# -------------------------------------

df = pd.read_csv("deploy_df.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

for i in df:
    df.replace("New Delhi", "Delhi", inplace=True)


# Feature Selection.
# -------------------------------------

X = df.drop("Price", axis=1)
y = df["Price"]


# Train-Test Split.
# -------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)


# Model Building.
# -------------------------------------

best_params = {
    "bagging_temperature": 0.2,
    "border_count": 32,
    "depth": 8,
    "iterations": 200,
    "l2_leaf_reg": 1,
    "learning_rate": 0.1,
    "random_strength": 0.2,
}


cat = CatBoostRegressor(**best_params)
cat.fit(X_train, y_train)

print("\n\nFinished TRAINING CatBoost model !!")

y_pred = cat.predict(X_test)


# Saving model (using pickle) & dataframe.
# -------------------------------------

import pickle

pickle.dump(cat, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))

print("\nFinished SAVING model !!\n\n")
