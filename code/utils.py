from typing import DefaultDict
import torch
from torch.quasirandom import SobolEngine
from botorch.acquisition import ExpectedImprovement
from botorch.utils.transforms import unnormalize

def get_random_points(dim, n_pts, dtype, device, lb = 0, ub = 1):
    
    sobol = SobolEngine(dimension = dim, scramble = True)
    x_init = sobol.draw(n = n_pts).to(dtype = dtype, 
                                        device = device)
    
    
    x_init = lb + (ub - lb) * x_init
    
    return x_init

def cost_aware_ei(lambda_, model, x_candidates, cost_model, y_max, device):



    ei = ExpectedImprovement(model, y_max)
   
    ei_vals = ei(x_candidates.unsqueeze(1))
    ei_max  = ei_vals.max().item()
    
    is_greater = ei_vals >= (1 - lambda_)*ei_max
    
    cei = torch.tensor([-float("inf")] * len(ei_vals), device = device)
    cost_preds = torch.tensor(cost_model.predict(x_candidates), 
                              dtype=torch.float,
                             device = device) 
    # cost_preds.squeeze_(1)
    cost_preds = -1 * cost_preds
    cei[is_greater] = cost_preds[is_greater]
    
    return cei

def get_best_params(best_x, param_names, algo_values, lb, ub, device):

    bounds = torch.tensor([lb, ub], device = device)
    un = unnormalize(best_x, bounds)

    param_values = algo_values + un.squeeze().tolist()
    best_param = dict(zip(param_names, param_values))

    return best_param