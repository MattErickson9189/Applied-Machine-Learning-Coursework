import sys
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


DOWNLOAD_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
DATASET_BASE_PATH = './datasets/PM25DataSet/'
DESTINATION_PATH =  DATASET_BASE_PATH + 'original_data_set.csv'

np.set_printoptions(suppress=True)

#Downloads the data from a URL
def fetch_data(fetchUrl):

    #make any parent directories if needed
    index = DESTINATION_PATH.rindex('/')
    os.makedirs(DESTINATION_PATH[:index], exist_ok=True)

    urllib.request.urlretrieve(fetchUrl, DESTINATION_PATH)
 
# Loads in data from a csv and returns the dataFrame
def load_data(dataSet):
    pm25DF = pd.read_csv(dataSet)
    return pm25DF

#Function takes in a dataFrame and a list of features and removes all the features and na rows
def remove_features(dataSet, features):
    for feature in features:
        dataSet = dataSet.drop(feature, axis=1)
    dataSet = dataSet.dropna(axis=0)
    return dataSet


#Class to combine the Is and Ir features
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

numeric_feature_names = ["month", "day", "hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
categorical_feature_names = ["cbwd"]
target_variable = ["pm2.5"]

features_to_drop = ["No", "year"]

pm25DF = remove_features(pm25DF, features_to_drop)

print(pm25DF.info())
print()
print(pm25DF.head())

# Preprocessing pipeline

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

# Splitting the data into train and test sections

train_set, test_set = train_test_split(pm25DF_Processed, test_size=.2, random_state=10)
print(train_set.shape)
print(test_set.shape)

pickle_path = os.path.join(DATASET_BASE_PATH, "mp25.pickle")

with open(pickle_path, 'wb') as f:
    pickle.dump([train_set,test_set],f)


# Experimenting with different models


#TODO Slice out 1 column and put it into the y_train and y_test
x_train = train_set[:,:-1]
y_train = train_set[:,-1:]
x_test = test_set[:,:-1]
y_test = test_set[:,-1:]

print("\nTrain Set\n")
print(train_set[:3,:])
print()
print("\nx_train\n")
print(x_train[:3,:])
print()
print("\ny_train\n")
print(y_train[:3,:])
print()


print()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Create polynomial features from train set
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly_features.fit_transform(x_train)

linear_reg = LinearRegression()
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()
svr_reg = SVR()
gradient_boost_reg = GradientBoostingRegressor()

print("\nTraining Models...")

print("Training Linear Model")
linear_scores = cross_val_score(linear_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
print("Done Training Linear Model")

print("Training Elastic Net Model")
elastic_scores = cross_val_score(elastic_net, x_train_poly, y_train, scoring="neg_mean_squared_error", cv=10)
print("Done Training Elastic Net Model")

print("Training Decision Tree Model")
tree_scores = cross_val_score(tree_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
print("Done Training Decision Tree Model")

print("Training Random Forest Model")
forest_scores = cross_val_score(forest_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
print("Done Training Randome Forest Model")

print("Training Gradient Boosting Model")
gradient_boost_scores = cross_val_score(gradient_boost_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
print("Done Training Gradient Boosting Model")

print("Training SVR Model")
svr_scores = cross_val_score(svr_reg, x_train, y_train, scoring="neg_mean_squared_error", cv=10)
print("Done Training SVR Model")

#Calculate the average RMSE and make it positive

target_range = y_train.max()-y_train.min()
linear_average_rmse = np.sqrt(-linear_scores).mean()/target_range
elastic_average_rmse = np.sqrt(-elastic_scores).mean()/target_range
tree_average_rmse = np.sqrt(-tree_scores).mean()/target_range
forest_average_rmse = np.sqrt(-forest_scores).mean()/target_range
svr_average_rmse = np.sqrt(-svr_scores).mean()/target_range
gradient_boost_average_rmse = np.sqrt(-gradient_boost_scores).mean()/target_range

#Plot the RMSE
print("\nPlotting Average RMSEs...")
x = [1,2,3,4,5,6]

model_names = ["Linear", "ElasticNet", "Decision Tree", "Random Forest", "Gradient Boosting", "SVR"]
average_rmses = [linear_average_rmse, elastic_average_rmse, tree_average_rmse, forest_average_rmse, gradient_boost_average_rmse, svr_average_rmse]
print(['{:.2%}'.format(item) for item in average_rmses])

plt.figure(figsize=(10,5))
plt.bar(x,average_rmses)
plt.xticks(x,model_names)
plt.ylim(ymax=0.08)
plt.show()

print("\nTuning Hyperparameters...\n")

param_grid = [
        {'n_estimators': [5,10,15,20], 'min_samples_split':[2,4,6,10]}
        ]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(x_train, np.squeeze(y_train))
print(grid_search.best_params_)

final_model = grid_search.best_estimator_
final_predictions = final_model.predict(x_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_normalized_rmse = final_rmse/target_range

print("RMSE: {0:.0f}".format(final_rmse))
print("Normalized RMSE: {0:.2%}".format(final_normalized_rmse))

