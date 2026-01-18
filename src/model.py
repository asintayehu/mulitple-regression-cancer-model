from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Model():
    def __init__(self):
        file_path = r"~/poincare/cancer_regression/data/cancer_reg.csv"
        cancer_df = pd.read_csv(file_path)
        self.train_set, self.test_set = train_test_split(cancer_df, test_size = 0.2) # now have a training set and a testing set! Yay!
        self.cancer_df_train_set = self.train_set.drop("TARGET_deathRate", axis=1)[["povertyPercent", "medIncome","PctPublicCoverage"]].copy()
        self.cancer_labels = self.train_set["TARGET_deathRate"].copy()

    def predict(self):
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.clean_data(self.cancer_df))

    def clean_data(self):
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        cancer_numerical = self.cancer_df_train_set
        imputer.fit_transform(cancer_numerical)
        numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
        full_pipeline = Pipeline([('num', numeric_pipeline)])
        self.cancer_prepped = full_pipeline.fit_transform(cancer_numerical)
        return self.cancer_prepped
    
    def evaluate(self):
        pass
    
