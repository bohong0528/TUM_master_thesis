import json
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, make_scorer, max_error, explained_variance_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from scipy.stats import expon, reciprocal, uniform
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, ExpSineSquared, RationalQuadratic
import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel, RFECV
from sklearn.compose import ColumnTransformer
from mango import Tuner, scheduler
import xgboost as xgb
from skopt  import BayesSearchCV
import lightgbm as lgb
import sys
from skopt.space import Categorical
import pickle
def load_data(file_list, df_activities, df_links_network):
    data_frames = []
    for file in file_list:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data['link_counts'], dict):
                data['link_counts'] = data['link_counts'].values()
            df_links = pd.DataFrame({
                'link_id': data['links_id'],
                'link_from': data['link_from'],
                'link_to': data['link_to'],
                'link_length': data['link_length'],
                'link_freespeed': data['link_freespeed'],
                'link_capacity': data['link_capacity'],
                'link_permlanes': data['link_permlanes'],
                'link_counts': data['link_counts']
            })
            df_nodes = pd.DataFrame({
                'node_id': data['nodes_id'],
                'node_x': data['nodes_x'],
                'node_y': data['nodes_y']
            })
            df_od_pairs = pd.DataFrame(data['o_d_pairs'], columns=['origin', 'destination'])

            df_work = pd.DataFrame({
                'work_x': data['work_x'],
                'work_y': data['work_y'],
                'go_to_work': data['go_to_work']
            })
            df_home = pd.DataFrame({
                'home_x': data['home_x'],
                'home_y': data['home_y'],
                'go_to_home': data['go_to_home']
            })

            df_links = df_links.merge(df_nodes, how='left', left_on='link_from', right_on='node_id')
            df_links = df_links.rename(columns={'node_x': 'start_node_x', 'node_y': 'start_node_y'})
            df_links.drop('node_id', axis=1, inplace=True)
            df_links = df_links.merge(df_nodes, how='left', left_on='link_to', right_on='node_id')
            df_links = df_links.rename(columns={'node_x': 'end_node_x', 'node_y': 'end_node_y'})
            df_links.drop('node_id', axis=1, inplace=True)

            origin_counts = df_od_pairs['origin'].value_counts()
            df_origin_counts = origin_counts.reset_index()
            df_origin_counts.columns = ['origin', 'start_count']
            destination_counts = df_od_pairs['destination'].value_counts()
            df_destination_counts = destination_counts.reset_index()
            df_destination_counts.columns = ['destination', 'end_count']
            df_links = df_links.merge(df_origin_counts, how='left', left_on='link_from', right_on='origin')
            df_links.drop('origin', axis=1, inplace=True)
            df_links = df_links.merge(df_destination_counts, how='left', left_on='link_to', right_on='destination')
            df_links.drop('destination', axis=1, inplace=True)
            df_links[['start_count', 'end_count']] = df_links[['start_count', 'end_count']].fillna(-1)

            # Calculate time of go_to_work and go_to_sum
            df_act_work = df_activities[df_activities['activity_type_main'] == 'work'].drop(['end_time'], axis=1)
            df_act_work = df_act_work.merge(df_work, how='left', left_on=['x', 'y'], right_on=['work_x', 'work_y'])
            df_act_work.drop(['x', 'y'], axis=1, inplace=True)
            df_act_work_agg = df_act_work.groupby(by="link")['go_to_work'].sum().reset_index(drop=False)
            df_act_home = df_activities[df_activities['activity_type_main'] == 'home'].drop(['end_time'], axis=1)
            df_act_home = df_act_home.merge(df_home, how='left', left_on=['x', 'y'], right_on=['home_x', 'home_y'])
            df_act_home.drop(['x', 'y'], axis=1, inplace=True)
            df_act_home_agg = df_act_home.groupby(by="link")['go_to_home'].sum().reset_index(drop=False)
            df_act_agg = df_act_home_agg.merge(df_act_work_agg, how='outer', on='link')
            df_act_agg.fillna(0, inplace=True)
            df_act_agg['go_to_sum'] = df_act_agg['go_to_home'] + df_act_agg['go_to_work']

            df_rushhr = df_activities[df_activities['end_time'] != -1]
            df_rushhr.loc[:, 'rush_hour'] = 0
            df_rushhr.loc[df_rushhr['end_time'].between(pd.to_timedelta('08:00:00'), pd.to_timedelta('10:00:00'),
                                                        inclusive='both'), 'rush_hour'] = 1
            df_rushhr.loc[df_rushhr['end_time'].between(pd.to_timedelta('16:00:00'), pd.to_timedelta('19:00:00'),
                                                        inclusive='both'), 'rush_hour'] = 1
            df_rushhr.drop(['end_time', 'max_dur', 'zoneId', 'cemdapStopDuration_s'], axis=1, inplace=True)
            df_rushhragg = df_rushhr.groupby(by="link").sum()['rush_hour'].reset_index(drop=False)

            df_maxduragg = df_activities[df_activities['max_dur'] != -1].groupby(by='link')[
                'max_dur'].sum().reset_index(drop=False)

            df_activities['cemdapStopDuration_s'] = df_activities['cemdapStopDuration_s'].astype(float)
            df_cemagg = df_activities[df_activities['cemdapStopDuration_s'] != -1].groupby(by='link')[
                'cemdapStopDuration_s'].sum().reset_index(drop=False)

            df_temp = df_links.merge(df_links_network, how='left',
                                     on=['start_node_x', 'start_node_y', 'end_node_x', 'end_node_y'])
            df_temp = df_temp[['link_id_x', 'link_from', 'link_to', 'link_id_y', 'from', 'to', 'type']]
            df_temp = df_temp.merge(df_act_agg, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)
            df_temp = df_temp.merge(df_rushhragg, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)
            df_temp = df_temp.merge(df_maxduragg, how='left', left_on='link_id_y', right_on='link')
            df_temp.drop('link', axis=1, inplace=True)
            df_temp = df_temp.merge(df_cemagg, how='left', left_on='link_id_y', right_on='link')
            df_temp.fillna({'cemdapStopDuration_s': -1, 'max_dur': -1, 'rush_hour': -1, 'go_to_sum': -1}, inplace=True)
            df_temp = df_temp[['link_id_x', 'go_to_sum', 'rush_hour', 'max_dur', 'cemdapStopDuration_s', 'type']]

            df_links = df_links.merge(df_temp, how='left', left_on='link_id', right_on='link_id_x')
            df_links.drop('link_id_x', axis=1, inplace=True)
            df_links['length_per_capacity_ratio'] = df_links['link_length'] / df_links['link_capacity']
            df_links['speed_capacity_ratio'] = df_links['link_freespeed'] / df_links['link_capacity']
            df_links['length_times_lanes'] = df_links['link_length'] * df_links['link_permlanes']
            df_links['speed_times_capacity'] = df_links['link_freespeed'] * df_links['link_capacity']
            df_links['length_times'] = df_links['link_length'] / df_links['link_freespeed']
            df_links['capacity_divided_by_lanes'] = df_links['link_capacity'] / df_links['link_permlanes']

        data_frames.append(df_links)
    return pd.concat(data_frames, ignore_index=True)
test = pd.read_pickle("result_LGBM.pkl")
train_files = ['s-0.json', 's-1.json', 's-2.json', 's-3.json', 's-4.json', 's-5.json', 's-6.json', 's-7.json',
               's-8.json', 's-9.json']
test_files = ['s-15.json', 's-16.json', 's-17.json', 's-18.json', 's-19.json']
validate_files = ['s-10.json', 's-11.json', 's-12.json', 's-13.json', 's-14.json']
train_files = ['data/Data/cutoutWorlds/Train/po-1_pn-1.0_sn-1/' + i for i in train_files]
test_files = ['data/Data/cutoutWorlds/Test/po-1_pn-1.0_sn-1/' + j for j in test_files]
validate_files = ['data/Data/cutoutWorlds/Validate/po-1_pn-1.0_sn-1/' + k for k in validate_files]
df_activities = pd.read_pickle("data/Data/cutoutWorlds/Train/po-1_pn-1.0_sn-1/df_activities.pkl")
df_links_network = pd.read_pickle("data/Data/cutoutWorlds/Train/po-1_pn-1.0_sn-1/df_links_network.pkl")
train_data = load_data(train_files, df_activities, df_links_network)
test_data = load_data(test_files, df_activities, df_links_network)
validate_data = load_data(validate_files, df_activities, df_links_network)
Big_train_data = pd.concat([train_data, validate_data], ignore_index=True)

# %%
numerical_features = ['link_id', 'link_from', 'link_to', 'start_node_x', 'start_node_y', 'end_node_x', 'end_node_y',
                      'link_length', 'link_freespeed', 'link_capacity', 'link_permlanes', 'start_count', 'end_count',
                      'go_to_sum', 'rush_hour', 'max_dur', 'cemdapStopDuration_s', 'length_per_capacity_ratio',
                      'speed_capacity_ratio',
                      'length_times_lanes', 'speed_times_capacity', 'length_times', 'capacity_divided_by_lanes'
                      ]
category_feature = ['type']
X_t = Big_train_data.drop(columns=['link_counts'])
y_t = Big_train_data['link_counts']
X_te = test_data.drop(columns=['link_counts'])
y_te = test_data['link_counts']
scaler = StandardScaler()
le = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False)
ct = ColumnTransformer(
    [("num_preprocess", scaler, numerical_features),
     ("text_preprocess", ohe, category_feature)], remainder='passthrough').set_output(transform="pandas")
X_t = ct.fit_transform(X_t)
X_te = ct.fit_transform(X_te)

model_arg = sys.argv[1]
if model_arg == 'LGBM':
    param_space = {
        'learning_rate': np.arange(0.01, 1.0, 0.01),
        'n_estimators': np.arange(50, 2001, 50),
        'max_depth': np.arange(1, 200),
        'num_leaves': np.arange(2, 200),
        'min_child_samples': np.arange(1, 20),
        'subsample': np.arange(0.1, 1.0, 0.1),
        'colsample_bytree': np.arange(0.1, 1.0, 0.1),
        'reg_alpha': np.arange(0, 100),
        'reg_lambda': np.arange(0, 20, 0.01),
    }

    # Define the LGBMRegressor model
    lgbm_model = lgb.LGBMRegressor(random_state=101)

    # Create a Bayesian optimization object
    opt = BayesSearchCV(
        lgbm_model,
        param_space,
        n_iter=100,  # Adjust the number of iterations based on your computational resources
        cv=3,  # Adjust the number of cross-validation folds
        scoring='neg_mean_absolute_error',  # Use a suitable regression metric
        n_jobs=-1,
    )

    # Perform Bayesian optimization
    np.int = int
    opt.fit(X_t, y_t)

    # Get the best hyperparameters
    best_params = opt.best_params_
    print(opt.best_estimator_)

    results = {"para": opt.best_params_, "estimator": opt.best_estimator_}

    with open(f'results/result_{model_arg}.pkl', 'wb') as file:
        pickle.dump(results, file)

    # Train the final model with the best hyperparameters
    final_model = lgb.LGBMRegressor(**best_params, random_state=101)
    final_model.fit(X_t, y_t)

    print("With LGB Pred")
    y_t_pred = final_model.predict(X_t)
    print(mean_absolute_error(y_t, y_t_pred))

    y_te_pred = final_model.predict(X_te)
    print(mean_absolute_error(y_te, y_te_pred))

    print("---------------")

    onehot = OneHotEncoder()
    X_t_leaves = onehot.fit_transform(final_model.predict(X_t, pred_leaf=True))
    xgb_lr = LogisticRegression()
    xgb_lr.fit(X_t_leaves, y_t)

    print("With LGB + LR Pred")
    X_t_leaves = onehot.transform(final_model.predict(X_t, pred_leaf=True))
    y_t_pred_xgb_lr = xgb_lr.predict(X_t_leaves)
    print(mean_absolute_error(y_t, y_t_pred_xgb_lr))

    X_te_leaves = onehot.transform(final_model.predict(X_te, pred_leaf=True))
    y_te_pred_xgb_lr = xgb_lr.predict(X_te_leaves)
    print(mean_absolute_error(y_te, y_te_pred_xgb_lr))

elif model_arg == 'XGB':
    param_space = {
        'learning_rate': np.arange(0.01, 1.0, 0.01),
        'n_estimators': np.arange(50, 2001, 50),
        'max_depth': np.arange(1, 20),
        'max_leaves': np.arange(2, 50),
        'max_bin': np.arange(2, 50),
        'gamma': np.arange(1, 20),
        'min_child_weight': np.arange(0, 20),
        'subsample': np.arange(0.1, 1.0, 0.1),
        'colsample_bytree': np.arange(0.1, 1.0, 0.1),
        'reg_alpha': np.arange(0, 100),
        'reg_lambda': np.arange(0, 10, 0.01),
    }

    # Define the LGBMRegressor model
    xgb_model = xgb.XGBRegressor(random_state=101)

    # Create a Bayesian optimization object
    opt = BayesSearchCV(
        xgb_model,
        param_space,
        n_iter=100,  # Adjust the number of iterations based on your computational resources
        cv=3,  # Adjust the number of cross-validation folds
        scoring='neg_mean_absolute_error',  # Use a suitable regression metric
        n_jobs=-1,
    )

    # Perform Bayesian optimization
    np.int = int
    opt.fit(X_t, y_t)

    # Get the best hyperparameters
    best_params = opt.best_params_
    print(opt.best_estimator_)

    results = {"para": opt.best_params_, "estimator": opt.best_estimator_}

    with open(f'results/result_{model_arg}.pkl', 'wb') as file:
        pickle.dump(results, file)

    # Train the final model with the best hyperparameters
    final_model = xgb.XGBRegressor(**best_params, random_state=101)
    final_model.fit(X_t, y_t)

    print("With XGB Pred")
    y_t_pred = final_model.predict(X_t)
    print(mean_absolute_error(y_t, y_t_pred))

    y_te_pred = final_model.predict(X_te)
    print(mean_absolute_error(y_te, y_te_pred))

    print("---------------")

    onehot = OneHotEncoder()
    X_t_leaves = onehot.fit_transform(final_model.apply(X_t))
    xgb_lr = LogisticRegression()
    xgb_lr.fit(X_t_leaves, y_t)

    print("With XGB + LR Pred")
    X_t_leaves = onehot.transform(final_model.apply(X_t))
    y_t_pred_xgb_lr = xgb_lr.predict(X_t_leaves)
    print(mean_absolute_error(y_t, y_t_pred_xgb_lr))

    X_te_leaves = onehot.transform(final_model.apply(X_te))
    y_te_pred_xgb_lr = xgb_lr.predict(X_te_leaves)
    print(mean_absolute_error(y_te, y_te_pred_xgb_lr))

elif model_arg == 'KNN':

    param_space = {
        'n_neighbors': np.arange(1, 50),
        'weights': Categorical(['uniform', 'distance']),
        'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'])
    }

    knn = KNeighborsRegressor()
    # Create a Bayesian optimization object
    opt = BayesSearchCV(
        knn,
        param_space,
        n_iter=100,  # Adjust the number of iterations based on your computational resources
        cv=3,  # Adjust the number of cross-validation folds
        scoring='neg_mean_absolute_error',  # Use a suitable regression metric
        n_jobs=-1,
    )

    # Perform Bayesian optimization
    np.int = int
    opt.fit(X_t, y_t)

    # Get the best hyperparameters
    best_params = opt.best_params_
    print(opt.best_estimator_)

    results = {"para": opt.best_params_, "estimator": opt.best_estimator_}

    with open(f'results/result_{model_arg}.pkl', 'wb') as file:
        pickle.dump(results, file)

    # Train the final model with the best hyperparameters
    final_model = KNeighborsRegressor(**best_params)
    final_model.fit(X_t, y_t)

    print("With KNN Pred")
    y_t_pred = final_model.predict(X_t)
    print(mean_absolute_error(y_t, y_t_pred))

    y_te_pred = final_model.predict(X_te)
    print(mean_absolute_error(y_te, y_te_pred))

elif model_arg == 'SVR':
    param_space = {
        'C': np.logspace(-3, 2, 6),
        'gamma': Categorical(['scale', 'auto']),  # Including specific gamma values
        'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),  # Focusing on RBF kernel
        'epsilon': np.arange(0.01, 1, 0.01),  # Epsilon in the epsilon-SVR model
    }
    # Define the LGBMRegressor model
    svr = SVR()

    # Create a Bayesian optimization object
    opt = BayesSearchCV(
        svr,
        param_space,
        n_iter=100,  # Adjust the number of iterations based on your computational resources
        cv=3,  # Adjust the number of cross-validation folds
        scoring='neg_mean_absolute_error',  # Use a suitable regression metric
        n_jobs=-1,
    )

    # Perform Bayesian optimization
    np.int = int
    opt.fit(X_t, y_t)

    # Get the best hyperparameters
    best_params = opt.best_params_
    print(opt.best_estimator_)

    results = {"para": opt.best_params_, "estimator": opt.best_estimator_}

    with open(f'results/result_{model_arg}.pkl', 'wb') as file:
        pickle.dump(results, file)

    final_model = SVR(**best_params)
    final_model.fit(X_t, y_t)

    print("With SVR Pred")
    y_t_pred = final_model.predict(X_t)
    print(mean_absolute_error(y_t, y_t_pred))

    y_te_pred = final_model.predict(X_te)
    print(mean_absolute_error(y_te, y_te_pred))

elif model_arg == 'RF':
    param_space = {
        'max_features': Categorical(['sqrt', 'log2']),
        'n_estimators': np.arange(50, 2001, 50),
        'bootstrap': Categorical([True, False]),
        'max_depth': np.arange(1, 20),
        'min_samples_leaf': np.arange(1, 20),
        'criterion': Categorical(['squared_error', 'absolute_error', 'friedman_mse', 'poisson'])
    }
    # Define the LGBMRegressor model
    rf_model = RandomForestRegressor(random_state=101)

    # Create a Bayesian optimization object
    opt = BayesSearchCV(
        rf_model,
        param_space,
        n_iter=100,  # Adjust the number of iterations based on your computational resources
        cv=3,  # Adjust the number of cross-validation folds
        scoring='neg_mean_absolute_error',  # Use a suitable regression metric
        n_jobs=-1,
    )

    # Perform Bayesian optimization
    np.int = int
    opt.fit(X_t, y_t)

    # Get the best hyperparameters
    best_params = opt.best_params_
    print(opt.best_estimator_)

    results = {"para": opt.best_params_, "estimator": opt.best_estimator_}

    with open(f'results/result_{model_arg}.pkl', 'wb') as file:
        pickle.dump(results, file)

    # Train the final model with the best hyperparameters
    final_model = RandomForestRegressor(**best_params, random_state=101)
    final_model.fit(X_t, y_t)

    print("With RF Pred")
    y_t_pred = final_model.predict(X_t)
    print(mean_absolute_error(y_t, y_t_pred))

    y_te_pred = final_model.predict(X_te)
    print(mean_absolute_error(y_te, y_te_pred))

    print("---------------")

    onehot = OneHotEncoder()
    X_t_leaves = onehot.fit_transform(final_model.apply(X_t))
    xgb_lr = LogisticRegression()
    xgb_lr.fit(X_t_leaves, y_t)

    print("With RF + LR Pred")
    X_t_leaves = onehot.transform(final_model.apply(X_t))
    y_t_pred_xgb_lr = xgb_lr.predict(X_t_leaves)
    print(mean_absolute_error(y_t, y_t_pred_xgb_lr))

    X_te_leaves = onehot.transform(final_model.apply(X_te))
    y_te_pred_xgb_lr = xgb_lr.predict(X_te_leaves)
    print(mean_absolute_error(y_te, y_te_pred_xgb_lr))

elif model_arg == 'GB':
    param_space = {
        'learning_rate': np.arange(0.01, 1.0, 0.01),
        'n_estimators': np.arange(50, 2001, 50),
        'max_depth': np.arange(1, 200),
        'min_samples_split': np.arange(2, 11, 1),
        'min_samples_leaf': np.arange(1, 10),
        'subsample': np.arange(0.1, 1.0, 0.1),
    }
    # Define the LGBMRegressor model
    gb_model = GradientBoostingRegressor(random_state=101)

    # Create a Bayesian optimization object
    opt = BayesSearchCV(
        gb_model,
        param_space,
        n_iter=100,  # Adjust the number of iterations based on your computational resources
        cv=3,  # Adjust the number of cross-validation folds
        scoring='neg_mean_absolute_error',  # Use a suitable regression metric
        n_jobs=-1,
    )

    # Perform Bayesian optimization
    np.int = int
    opt.fit(X_t, y_t)

    # Get the best hyperparameters
    best_params = opt.best_params_
    print(opt.best_estimator_)

    results = {"para": opt.best_params_, "estimator": opt.best_estimator_}

    with open(f'results/result_{model_arg}.pkl', 'wb') as file:
        pickle.dump(results, file)

    final_model = GradientBoostingRegressor(**best_params, random_state=101)
    final_model.fit(X_t, y_t)

    print("With GB Pred")
    y_t_pred = final_model.predict(X_t)
    print(mean_absolute_error(y_t, y_t_pred))

    y_te_pred = final_model.predict(X_te)
    print(mean_absolute_error(y_te, y_te_pred))

    print("---------------")

    onehot = OneHotEncoder()
    X_t_leaves = onehot.fit_transform(final_model.apply(X_t))
    xgb_lr = LogisticRegression()
    xgb_lr.fit(X_t_leaves, y_t)

    print("With GB + LR Pred")
    X_t_leaves = onehot.transform(final_model.apply(X_t))
    y_t_pred_xgb_lr = xgb_lr.predict(X_t_leaves)
    print(mean_absolute_error(y_t, y_t_pred_xgb_lr))

    X_te_leaves = onehot.transform(final_model.apply(X_te))
    y_te_pred_xgb_lr = xgb_lr.predict(X_te_leaves)
    print(mean_absolute_error(y_te, y_te_pred_xgb_lr))

