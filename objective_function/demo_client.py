
import xmlrpc.client
from torch.quasirandom import SobolEngine
import torch

## Sample Code to demo use of the objective function server

def get_random_points(dim, n_pts, dtype, device, lb = 0, ub = 1):
    
    sobol = SobolEngine(dimension = dim, scramble = True)
    x_init = sobol.draw(n = n_pts).to(dtype = dtype, 
                                        device = device)
    
    
    x_init = lb + (ub - lb) * x_init
    
    return x_init
    

s = xmlrpc.client.ServerProxy('http://localhost:5000')

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


x_init  = get_random_points(len(hyperparams),
                                    2,
                                    torch.float,
                                    torch.device("cpu"),
                                    0,
                                    1)

print(x_init[0])
eval_result = s.evaluate(x_init[0].tolist(), param_names, lower_bounds, upper_bounds)

print(eval_result)


print(s.system.listMethods())