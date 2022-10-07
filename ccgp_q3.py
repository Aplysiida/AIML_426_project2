from deap import base

import numpy as np

def fitness_func(x):
    return  (1/x) + np.sin(x) if (x > 0) else 2*x + pow(x,2) + 3.0 

"""
Setup the DEAP toolbox for GP
"""
def setup_toolbox():
    toolbox = base.Toolbox()
    return toolbox()

def run_CCGP(seeds):
    print('hi')

if __name__ == "__main__":
    print('fitness = ',fitness_func(-1))