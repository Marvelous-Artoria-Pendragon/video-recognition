import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import time
import joblib

def dataset(feature_path):
    for file_name in os.listdir(feature_path):
        if file_name[-3:] == 'npy':
            process, model_name, dataset, p = file_name[:-4].split('_')
            data = np.load(os.path.join(feature_path, file_name))
            if process == 'test':
                if p == 'label':
                    y_test = data
                else: 
                    x_test = data
            elif process == 'train':
                if p == 'label':
                    y_train = data
                else:
                    x_train = data
            elif process == 'val':
                if p == 'label':
                    y_val = data
                else:
                    x_val = data
    return x_train, x_test, y_train, y_test, x_val, y_val

def GBDT(feature_path, lr = 1e-1, n_estimators = 100, n_iter_no_change = None, model_save_path = '.', save_name = 'gbdt',
        loss = 'deviance', min_samples_split = 2, min_samples_leaf = 1, max_depth = 3):
    
    x_train, x_test, y_train, y_test, x_val, y_val = dataset(feature_path)
    clf = GradientBoostingClassifier(learning_rate=lr, n_estimators = n_estimators, n_iter_no_change = n_iter_no_change, loss = loss,
                                    min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_depth = max_depth)
    print('GBDT starts training...')
    start_time = time.time()
    gbdt = clf.fit(x_train, y_train)
    end_time = time.time()
    
    train_score = gbdt.score(x_train, y_train)
    test_score = gbdt.score(x_test, y_test)
    val_score = gbdt.score(x_val, y_val)

    print('train score: ', train_score)
    print('val score: ', val_score)
    print('test score: ', test_score)
    print(gbdt)
    print('time: {}s'.format(end_time - start_time))
    
    # save gbdt model
    joblib.dump(gbdt, os.path.join(model_save_path, save_name))

def XGBoost(feature_path, num_classes, lr = 1e-1, max_depth = 10, n_estimators = 100, gamma = 0.1, n_thread = 1, eta = 0.007,
            model_save_path = '.', save_name = 'xgb'):
    xgb = XGBClassifier(max_depth = max_depth, learning_rate = lr, n_estimators = n_estimators,
                        objective = 'multi:softproba', gamma = gamma, n_jobs = n_thread, eta = eta, eval_metric = 'mlogloss')
    x_train, x_test, y_train, y_test, x_val, y_val = dataset(feature_path)
    print('XGBoost starts training...')
    start_time = time.time()
    xgb.fit(x_train, y_train)
    end_time = time.time()
    train_score = xgb.score(x_train, y_train)
    test_score = xgb.score(x_test, y_test)
    val_score = xgb.score(x_val, y_val)

    print('train score: ', train_score)
    print('val score: ', val_score)
    print('test score: ', test_score)
    print(xgb)

    print('time: {}s'.format(end_time - start_time))
    
    # save xgboost model
    joblib.dump(xgb, os.path.join(model_save_path, save_name))