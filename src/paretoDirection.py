import numpy as np
from scipy.optimize import minimize

def paretoDirection(jacobian):
    # PARETODIRECTION Computes the minimum-norm Pareto-ascent direction.

    N_obj = jacobian.shape[1]
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    res = minimize(lambda x: np.linalg.norm(jacobian @ x), np.ones(N_obj)/N_obj,
                   method='SLSQP', constraints=cons)
    lambda_ = res.x
    dir_ = jacobian @ lambda_

    return dir_, lambda_

if __name__ == '__main__':

    dir, _ = paretoDirection(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]))
    print(np.linalg.norm(dir))