import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.utils.transforms import unnormalize

from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.linear_model import LinearRegression
from torch._C import dtype
from torch.quasirandom import SobolEngine

from utils import cost_aware_ei, get_random_points, get_best_params

from xmlrpc.client import ServerProxy


@dataclass
class TrustRegion:
    
    dim: int
    batch_size: int
    lb: int
    ub: int
    objective_params: dict
    cost_model: LinearRegression = LinearRegression()
    center: torch.Tensor = None
    n_init: int = 5
    model: SingleTaskGP = None
    X: torch.Tensor = None
    y: torch.Tensor = None
    y_time: torch.Tensor = None
    dtype: torch.dtype = torch.float
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length: float = 0.4
    length_min: float = 0.5 **7
    length_max: float = 1.2
    failure_counter: int = 0
    success_counter: int = 0
    failure_tolerance: int = float("nan")
    success_tolerance: int = 2

    best_value: float = -float("inf")
    restart_triggered: bool = False

    remote: ServerProxy = None
    remote_location: str = "localhost"
    remote_port: str = "5000"

    # def __post_init__(self):
        
    #     self.failure_tolerance = math.ceil(
    #         max(4.0/self.batch_size, float(self.dim) / self.batch_size)
    #     )

    def update_counters(self, y_next):
        
        best = self.best_value 
        if max(y_next) > best + 1e-3 * math.fabs(best):
            self.success_counter += 1
            self.failure_counter = 0
        
        else:
            self.success_counter = 0
            self.failure_counter += 1
        
        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2
            self.failure_counter = 0
        
        if self.length < self.length_min:
            self.restart_triggered = True
    
    def get_raw_candidates(self, n):
    
        '''
        n - number of candidates
        dim - dimension of each candidate
        x_center - center point with dimension dim
        weights - weights vector of dimension dim
        
        '''
        weights = self.model.covar_module.base_kernel.lengthscale.\
            squeeze().detach()
        
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        
        # shape -1 x dim 
        tr_lb = torch.clamp(self.center - weights * self.length / 2.0, 0, 1)
        # shape -1 x dim
        tr_ub = torch.clamp(self.center + weights * self.length / 2.0, 0, 1)
        
        ## generate candidates
        sobol = SobolEngine(self.dim, scramble=True)
        # shape n x dim
        candidates = sobol.draw(n).to(dtype = self.dtype, device = self.device)
        
        ## tr_lb , tr_ub both are broadcasted to (n x dim)
        
        # n  x dim
        candidates = tr_lb + (tr_ub - tr_lb) * candidates
    
        return candidates

    def get_perturb_candidates(self, candidates):
        
        n = len(candidates)
        prob_perturb = min(20.0 / self.dim, 1.0)

        mask = (
                    torch.rand(n, self.dim, dtype = self.dtype, device = self.device)
                    <= prob_perturb
                )

        ind = torch.where(mask.sum(dim = 1) == 0)[0] ## sum == False

        ## randomly selecting one dimension to perturb, 
        #  if all dim == False in the mask

        mask[ind, torch.randint(0, self.dim-1, 
                               size = (len(ind),), 
                               device = self.device)] = 1 ## equivalent to = True

        x_candidates = self.center.expand(n, self.dim).clone()
        x_candidates[mask] = candidates[mask]

        return x_candidates

    def get_candidates(self, n):
        
        raw_candidates = self.get_raw_candidates(n)
        perturb_candidates = self.get_perturb_candidates(raw_candidates)

        return perturb_candidates

    def fit_gp(self):
        
        train_y = (self.y - self.y.mean() ) /  self.y.std()
        likelihood = GaussianLikelihood(noise_constraint  = Interval(1e-8, 1e-3))
        self.model = SingleTaskGP(self.X, train_y, likelihood = likelihood)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def initalize(self):
        ## getting remote server object
        remote_address = 'http://' + self.remote_location + \
                        ":" + self.remote_port 
        
        self.remote = ServerProxy(remote_address)
        
        x_init  = get_random_points(self.dim,
                                    self.n_init,
                                    self.dtype,
                                    self.device,
                                    self.lb,
                                    self.ub)

        self.X = x_init
        hyperparams = self.objective_params['hyperparams']
        lb = self.objective_params['lb']
        ub = self.objective_params['ub']

        eval_result = [self.remote.evaluate(x.tolist(), hyperparams, lb, ub) \
            for x in x_init]

        self.y = torch.tensor(
                [val for val, _ in eval_result], 
                dtype = self.dtype,
                device = self.device
        ).unsqueeze(-1)

        self.y_time = np.array([t for _, t in eval_result])
        
    def fit_cost_model(self):
        self.cost_model.fit(self.X, self.y_time)

    def get_acquisition_points(self, lmbda, cost_model, n_candidates = None):

        if n_candidates == None:
            n_candidates = min(5000, max(2000, 200 * self.dim))
        
        self.center = self.X[self.y.argmax(), :]
        
        X_candidates = self.get_candidates(n_candidates)

        y_max = self.y.max().item()
        
        cei = cost_aware_ei(lmbda, 
                            self.model, 
                            X_candidates, 
                            cost_model, 
                            y_max, 
                            self.device)
        
        X_next = X_candidates[cei.argmax(), :].unsqueeze(0)

        return X_next
        
    def step(self, lmbda, n_candidates = None):
        '''
        '''
        
        self.fit_gp()

        self.fit_cost_model()

        x_next = self.get_acquisition_points(lmbda, self.cost_model, n_candidates)

        hyperparams = self.objective_params['hyperparams'] 
        lb = self.objective_params['lb']
        ub = self.objective_params['ub']

        eval_result = [self.remote.evaluate(x.tolist(), hyperparams, lb, ub) \
            for x in x_next]
        
        y_next =  torch.tensor(
            [val for val, _ in eval_result], dtype = self.dtype, device = self.device
        ).unsqueeze(-1)

        self.update_counters(y_next)

        y_time_next = np.array([t for _, t in eval_result])

        self.X = torch.cat((self.X, x_next), dim = 0)
        self.y = torch.cat((self.y, y_next), dim = 0)
        
        self.y_time = np.hstack((self.y_time, y_time_next))

        self.best_value = max(self.best_value, max(y_next).item())
    
    def get_best_x(self):

        best_x = self.X[self.y.argmax()]
        
        return best_x

# if __name__ == "__main__":

#     ## Testing code
    
#     algoparams = ['objective', 'eval_metric']
#     algo_values  = ['binary:logistic', ['logloss']]

#     hyperparams = [
#                    'num_round',
#                    'learning_rate',
#                    'gamma',
#                    'subsample',
#                    'max_depth']

#     param_names = algoparams + hyperparams
#     lower_bounds = [1,  0.01, 0,  0.1, 1]
#     upper_bounds = [200, 1,  0.1,  1,   16]

#     objective_params = {'hyperparams': param_names,
#                         'lb': lower_bounds,
#                         'ub': upper_bounds }

    
    
#     tr = TrustRegion(dim = 5, 
#                      batch_size  = 1, 
#                      lb = 0.2, 
#                      ub = 0.6, 
#                      objective_params = objective_params,
#                      n_init = 20)

#     # print(tr)
#     tr.initalize()

#     while not tr.restart_triggered:
#         tr.step(lmbda=0.7)
#         print('best value %.2f, TR Length %.2e' % (tr.best_value, tr.length))

#     best_x = tr.get_best_x()

#     best_params = get_best_params(best_x, param_names, algo_values, lower_bounds, upper_bounds,
#                                   tr.device)

#     print(best_params)