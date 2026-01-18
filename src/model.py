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

class Model():
    def __init__(self):
        _file_path = '~/poincare/cancer_regression/data/cancer_reg.csv'

        self.cancer_df = pd.read_csv(_file_path) # maybe revise file names

        self.train_set, self.test_set = train_test_split(self.cancer_df, test_size=0.2)

        # only using values which are relevant to us, drop labels
        self.cancer_df = self.train_set.drop("TARGET_deathRate", axis=1)[["medIncome", "povertyPercent", "PctPublicCoverage"]].copy()
        self.cancer_labels = self.train_set["TARGET_deathRate"].copy()

    def predict(self):
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.clean_data(self.cancer_df))

    def clean_data(self):
        self.imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        self.cancer_df = self.cancer_df.dropna()
        self.imputer.fit_transform(self.cancer_df)
        numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('std_scaler', StandardScaler())])
        numeric_attribs = list(self.cancer_df)
        full_pipeline = ColumnTransformer([('num', numeric_pipeline, numeric_attribs)])
        self.cancer_final = full_pipeline.fit_transform(self.cancer_df)
        return self.cancer_final


    
