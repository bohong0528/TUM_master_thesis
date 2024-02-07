import sys
import json
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, make_scorer, max_error
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, RandomizedSearchCV, ShuffleSplit, cross_validate, train_test_split
from scipy.stats import expon, reciprocal, uniform
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ExpSineSquared, RationalQuadratic
import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel, RFECV, SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from mango import Tuner, scheduler
import xgboost as xgb
from skopt  import BayesSearchCV
import lightgbm as lgb
from sklearn.cluster import OPTICS, MiniBatchKMeans
from pyGRNN import GRNN
from skopt.space import Categorical
from loading import load_data

numerical_features = ['link_id', 'link_from', 'link_to', 'start_node_x', 'start_node_y', 'end_node_x', 'end_node_y',
                      'link_length', 'link_freespeed', 'link_capacity', 'link_permlanes', 'start_count', 'end_count',
                      'go_to_sum', 'rush_hour', 'max_dur', 'cemdapStopDuration_s', 'length_per_capacity_ratio', 'speed_capacity_ratio',
                      'length_times_lanes', 'speed_times_capacity', 'length_times', 'capacity_divided_by_lanes'
                     ]
category_feature = ['type']
scaler = StandardScaler()
le = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False)
ct = ColumnTransformer(
     [("num_preprocess", scaler, numerical_features),
      ("text_preprocess", ohe, category_feature)], remainder='passthrough').set_output(transform="pandas")
model_space = {
    'SVR': SVR(),
    'KNN': KNeighborsRegressor(),
    'XGB': xgb.XGBRegressor(random_state=101),
    'LGBM': lgb.LGBMRegressor(random_state=101),
    'RF': RandomForestRegressor(random_state=101),
    'GB': GradientBoostingRegressor(random_state=101),
    'ANN': MLPRegressor(hidden_layer_sizes=(50,50) ,random_state=101),
    'GRNN': GRNN(seed=101)
}
param_space = {
'SVR': {
    'C': np.logspace(-3, 3),
    'gamma': Categorical(['scale', 'auto']),
    'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
    'epsilon': np.arange(0.01, 1, 0.01),
},
'RF':  {
    'max_features': Categorical(['sqrt', 'log2']),
    'n_estimators': np.arange(50, 10001, 50),
    'max_depth': np.arange(1, 20),
    'min_samples_leaf': np.arange(1, 20),
    'criterion': Categorical(['absolute_error', 'friedman_mse'])
},
'GB':{
    'learning_rate': np.arange(0.01, 1.0, 0.01),
    'n_estimators': np.arange(50, 3001, 50),
    'max_depth': np.arange(1, 200),
    'min_samples_split': np.arange(2, 11, 1),
    'min_samples_leaf': np.arange(1, 10),
    'subsample': np.arange(0.1, 1.0, 0.1),
},
'ANN': {
    'activation': Categorical(['tanh', 'relu', 'identity', 'logistic']),
    'solver': Categorical(['sgd', 'adam']),
    'alpha': np.logspace(-4, 7),
},
'KNN':{
    'n_neighbors': np.arange(1, 50),
    'weights': Categorical(['uniform', 'distance']),
    'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'])
},
'LGBM': {
    'learning_rate': np.arange(0.01, 1.0, 0.01),
    'n_estimators': np.arange(50, 2001, 50),
    'max_depth': np.arange(1, 50),
    'num_leaves': np.arange(2, 50),
    'min_child_samples': np.arange(1, 20),
    'subsample': np.arange(0.1, 1.0, 0.1),
    'colsample_bytree': np.arange(0.1, 1.0, 0.1),
},
'XGB': {
    'learning_rate': np.arange(0.01, 1.0, 0.01),
    'n_estimators': np.arange(50, 2001, 50),
    'max_depth': np.arange(1, 20),
    'max_leaves': np.arange(2, 50),
    'max_bin': np.arange(2, 50),
    'gamma': np.arange(1, 20),
},
'GRNN':{
    'sigma' : np.arange(0.1, 4, 0.01)
}
}

train_files = ['s-0.json', 's-1.json', 's-2.json', 's-3.json', 's-4.json','s-5.json', 's-6.json', 's-7.json', 's-8.json', 's-9.json']
test_files = ['s-15.json', 's-16.json', 's-17.json', 's-18.json','s-19.json']
validate_files = ['s-10.json', 's-11.json', 's-12.json', 's-13.json','s-14.json']
train_files = ['Data/cutoutWorlds/Train/po-1_pn-1.0_sn-1/' + i for i in train_files]
test_files = ['Data/cutoutWorlds/Test/po-1_pn-1.0_sn-1/' + j for j in test_files]
validate_files = ['Data/cutoutWorlds/Validate/po-1_pn-1.0_sn-1/' + k for k in validate_files]
df_activities = pd.read_pickle("Data/cutoutWorlds/Train/po-1_pn-1.0_sn-1/df_activities.pkl")
df_links_network = pd.read_pickle("Data/cutoutWorlds/Train/po-1_pn-1.0_sn-1/df_links_network.pkl")
train_data = load_data(train_files, df_activities, df_links_network)
validate_data = load_data(validate_files, df_activities, df_links_network)
test_data = load_data(test_files, df_activities, df_links_network)

train_data['dataset'] = 'train'
validate_data['dataset'] = 'validate'
test_data['dataset'] = 'test'
Big_data = pd.concat([train_data, validate_data, test_data], ignore_index=True)
Big_data_tr = ct.fit_transform(Big_data)
train_data_tr = Big_data_tr[Big_data_tr['remainder__dataset']=='train']
validate_data_tr = Big_data_tr[Big_data_tr['remainder__dataset']=='validate']
test_data_tr = Big_data_tr[Big_data_tr['remainder__dataset']=='test']
temp = pd.concat([train_data_tr, validate_data_tr], ignore_index=True)
X_t = temp.drop(columns=['remainder__dataset', 'remainder__link_counts'])
y_t = temp['remainder__link_counts']

X_te = test_data_tr.drop(columns=['remainder__dataset', 'remainder__link_counts'])
y_te = test_data_tr['remainder__link_counts']

rscv = sys.argv[2]
model = model_space[sys.argv[1]]
params_grid = param_space[sys.argv[1]]
if rscv:
    grid_model = RandomizedSearchCV(model, params_grid,
                                scoring='neg_mean_absolute_error',
                                cv=ShuffleSplit(test_size=0.33, n_splits=3),
                                n_iter=200,
                          )
    grid_model.fit(X_t, y_t)
    best_model = grid_model.best_estimator_
    y_pred = best_model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    me = max_error(y_te, y_pred)
    print([best_model, mae, mse, me])
else:
    opt = BayesSearchCV(
    model,
    params_grid,
    n_iter=200,
    cv=ShuffleSplit(test_size=0.33, n_splits=3),
    scoring='neg_mean_absolute_error',
    )
    opt.fit(X_t, y_t)
    best_model = opt.best_estimator_
    y_pred = best_model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    me = max_error(y_te, y_pred)
    print([best_model, mae, mse, me])
