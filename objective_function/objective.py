import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from botorch.utils.transforms import unnormalize


def insurance_init():

    '''
    link to download the dataset

    https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/

    '''

    train_path = 'data/ticdata2000.txt'
    test_path = 'data/ticeval2000.txt'
    test_label_path = 'data/tictgts2000.txt'
    
    train_df = pd.read_csv(train_path, delimiter = "\t", header = None)
    train = train_df.iloc[:, :-1]
    label_train = train_df.iloc[:, 85]

    test = pd.read_csv(test_path, delimiter = "\t", header = None)
    label_test = pd.read_csv(test_label_path, delimiter = "\t", header = None)

    X_train, X_val, y_train, y_val = train_test_split(train, label_train, 
                                                  test_size = 0.3,
                                                 random_state = 1,
                                                 stratify = label_train)
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    
    return dtrain, dval, y_val, test, label_test

def insurance(x, param_names, min_val, max_val, dtrain, dval, y_val):
    
    bounds = torch.Tensor([min_val, max_val])
    
     ## taking input
    un = unnormalize(x, bounds)
    param_values = ['binary:logistic', ['logloss']] + un.squeeze().tolist()
    
    param = dict(zip(param_names, param_values))
    
    ## processing input
    num_round = int(param['num_round'])
    del param['num_round']

    param['max_depth'] = int(param['max_depth'])
 
    evallist = [(dtrain, 'dtrain'), (dval, 'dval')]
    evals_result = {}
    model = xgb.train(param, 
                      dtrain, 
                      num_round, 
                      evallist, 
                      evals_result=evals_result,
                      verbose_eval = False)
    
    
    val_preds = model.predict(dval)
    
    val_preds[val_preds > 0.7] = 1
    val_preds[val_preds <= 0.7] = 0
    
    return f1_score(y_val, val_preds) 

def eval_objective(x, hyperparams, lb, ub, time_precision = 2):
    
    '''
    '''
    
    start = time.time()
  
    dtrain, dval, y_val, test, label_test = insurance_init()
    f1 = insurance(x, hyperparams, lb, ub, dtrain, dval, y_val)
    
    time_taken = round(time.time() - start, time_precision)
    return f1, time_taken

def get_best_params(best_x, param_names, algo_values, lb, ub, device):

    bounds = torch.tensor([lb, ub], device = device)
    un = unnormalize(best_x, bounds)

    param_values = algo_values + un.squeeze().tolist()
    best_param = dict(zip(param_names, param_values))

    return best_param