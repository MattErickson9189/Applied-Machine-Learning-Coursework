import os
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
DATASET_BASE_PATH = './datasets/PM25DataSet/'
DESTINATION_PATH =  DATASET_BASE_PATH + 'original_data_set.csv'


def fetch_data(fetchUrl):

    #make any parent directories if needed
    index = DESTINATION_PATH.rindex('/')
    os.makedirs(DESTINATION_PATH[:index], exist_ok=True)

    urllib.request.urlretrieve(fetchUrl, DESTINATION_PATH)
  
def load_data(dataSet):
    pm25DF = pd.read_csv(dataSet)
    return pm25DF

def remove_features(dataSet, features):
    for feature in features:
        dataSet = dataSet.drop(feature, axis=1)
    dataSet = dataSet.dropna(axis=0)
    return dataSet


class CombineAttributes(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.Is_ix = 4
        self.Ir_ix = 5 

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        total_hours_rain_and_snow = (x[:, self.Is_ix] + x[:,self.Ir_ix]).reshape(-1,1)
        return np.hstack([x, total_hours_rain_and_snow])

class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, x, y=None):
        return self

    def transform(self, dataFrame):
        return dataFrame[self.attribute_names].values


fetch_data(DOWNLOAD_URL)


pm25DF = load_data(DESTINATION_PATH)

numeric_feature_names = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
categorical_feature_names = ["cbwd"]
target_variable = ["pm2.5"]

features_to_drop = ["No", "year", "month", "day", "hour"]

pm25DF = remove_features(pm25DF, features_to_drop)

print(pm25DF.info())
print()

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(numeric_feature_names)),
    ('attribs_adder', CombineAttributes()),
    ('std_scaler', StandardScaler()),
    ])
 
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(categorical_feature_names)),
    ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

target_pipeline = Pipeline([
    ('selector', DataFrameSelector(target_variable)),
    ])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ("target_pipeline", target_pipeline)
    ])

pm25DF_Processed = full_pipeline.fit_transform(pm25DF)

print(pm25DF_Processed.shape)
print(pm25DF_Processed[0,:])
print()

train_set, test_set = train_test_split(pm25DF_Processed, test_size=.2, random_state=10)
print(train_set.shape)
print(test_set.shape)

pickle_path = os.path.join(DATASET_BASE_PATH, "mp25.pickle")

with open(pickle_path, 'wb') as f:
    pickle.dump([train_set,test_set],f)

