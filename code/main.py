'''
driver code 

'''
import json
from trust_region import TrustRegion
from utils import get_best_params
import os

if __name__== "__main__":

    NUM_RESTARTS = 3
    RESULTS_PATH = 'results/'
    
    try:
        os.mkdir(RESULTS_PATH)
    except OSError as e:
        pass
    
    algoparams = ['objective', 'eval_metric']
    algo_values  = ['binary:logistic', ['logloss']]

    hyperparams = [
                   'num_round',
                   'learning_rate',
                   'gamma',
                   'subsample',
                   'max_depth']

    param_names = algoparams + hyperparams
    lower_bounds = [1,  0.01, 0,  0.1, 1]
    upper_bounds = [200, 1,  0.1,  1,   16]

    objective_params = {'hyperparams': param_names,
                        'lb': lower_bounds,
                        'ub': upper_bounds }
    
    results = {}
    for r in range(NUM_RESTARTS):

        tr = TrustRegion(dim = 5, 
                        batch_size  = 1, 
                        lb = 0, 
                        ub = 1, 
                        objective_params = objective_params,
                        n_init = 10,
                        success_tolerance = 2,
                        failure_tolerance = 5)

        # print(tr)
        tr.initalize()

        print('*' * 20,"RESTART %d" % r,'*' * 20)
        while not tr.restart_triggered:
            tr.step(lmbda=0.7)
            print('best value %.2f, TR Length %.2e' % (tr.best_value, tr.length))

        best_x = tr.get_best_x()

        results[r] = tr.best_value

        best_params = get_best_params(best_x, param_names, algo_values, 
                                    lower_bounds, upper_bounds,
                                    tr.device)

        with open(RESULTS_PATH + "%d.json"%r, "w") as out:
            json.dump(best_params, out)


        
    best_r = max(results, key = lambda x: results[x])
    
    with open(RESULTS_PATH + "%d.json" % best_r, "r") as read_file:
        best_result = json.load(read_file)
    print(best_result)