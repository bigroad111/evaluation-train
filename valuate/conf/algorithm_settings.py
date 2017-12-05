# 随机种子数
SEED = 0
# K交叉验证数
NFOLDS = 5
# 训练测试分割比例
TEST_SIZE = 0.2

#########################
# final parameters
#########################
# KMeans parameters
km_params = {
    'n_clusters': 4,
    'max_iter': 800
}

# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 3500,
     # 'warm_start': True,
     # 'max_features': 0.2,
    'max_depth': 10,
     # 'min_samples_leaf': 2,
     # 'max_features': 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 5000,
    # 'max_features': 0.5,
    'max_depth': 10,
    # 'min_samples_leaf': 2,
    'verbose': 0
}

# Gradient Tree Boosting
gtb_params = {
    'loss': 'huber',
    'n_estimators': 3000,
    'max_depth': 6,
    'learning_rate':  0.1,
    'verbose': 0
}

# XGBOOST level1 parameters
xgb_level1_params = {
    'learning_rate': 0.1,
    'n_estimators': 5000,
    'max_depth': 10,
    # 'subsample': 0.8,
    # 'lamda': 0.8,
    # 'alpha': 0.4,
    # 'base_score': 0.16,
    # 'colsample_bytree': 0.7,
    # 'silent': 1
}

# XGBOOST level2 parameters
NUM_BOOST_ROUND = 5000
xgb_level2_params = {
    'eta': 0.015,
    'max_depth': 8,
    #    'subsample': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    # 'subsample': 0.9,
    # 'colsample_bytree': 0.9,
    #    'lamda': 0.8,
    #    'alpha': 0.4,
    #    'base_score': 0.16,
    #    'colsample_bytree': 0.7,
    'silent': 1
}
xgb_gpu_params = {
    'eta': 0.015,
    'max_depth': 8,
    #    'subsample': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    #    'lamda': 0.8,
    #    'alpha': 0.4,
    #    'base_score': 0.16,
    #    'colsample_bytree': 0.7,
    'gpu_id': 0,
    'max_bin': 16,
    'tree_method': 'gpu_hist',
    'silent': 1
}

# lightGBM parameters
lgb_params = {
    'max_depth': 10,
    'objective': 'regression',
    'num_leaves': 50,
    'learning_rate': 0.1,
    'n_estimators': 5000
}

###########################
# grid search cv parameters
###########################
# rf_grid_params = {
#     'n_estimators': [1000, 2000, 3000],
#     'max_depth': [6, 8, 10]
# }

rf_grid_params = {
    'n_estimators': [3000, 3500, 4000],
    'max_depth': [10]
}

et_grid_params = {
    'n_estimators': [4000, 5000, 6000],
    'max_depth': [10]
}

gtb_grid_params = {
    'loss': ('ls', 'lad', 'huber', 'quantile'),
    'learning_rate': [0.1, 0.01, 0.005],
    'n_estimators': [1000, 2000, 3000],
    'max_depth': [6, 8, 10]
}

# xgb_grid_leve1_params = {
#     'learning_rate': [0.1, 0.05, 0.01],
#     'n_estimators': [1000, 2000, 3000],
#     'max_depth': [6, 8, 10]
# }

xgb_grid_leve1_params = {
    'learning_rate': [0.1],
    # 'subsample': [0.9],
    # 'colsample_bytree': [0.9],
    'n_estimators': [5000, 7000, 9000],
    'max_depth': [3],
    'seed': [7]
}

# lgb_grid_params = {
#     'objective': ['regression'],
#     'learning_rate': [0.1, 0.05, 0.01],
#     'n_estimators': [1000, 2000, 3000],
#     'num_leaves': [31, 40, 50],
#     'max_depth': [6, 8, 10]
# }

lgb_grid_params = {
    'objective': ['regression'],
    'learning_rate': [0.1],
    'n_estimators': [2000, 3000, 4000],
    'num_leaves': [50],
    'max_depth': [10]
}
