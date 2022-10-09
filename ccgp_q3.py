from deap import base
from deap import creator
from deap import tools
from deap import gp

import numpy as np

import operator

def fitness_func(x):
    return  (1/x) + np.sin(x) if (x > 0) else 2*x + pow(x,2) + 3.0 

"""
Protected division to avoid program crashing when dividing by 0
"""
def protected_div(x,y):
    try:
        return x/y
    except (ZeroDivisionError, FloatingPointError):
        return 1.0


def fitness(individual, x_values):
    return 0.0

"""
Create function and terminal sets for GP
"""
def create_primitive_set():
    pset = gp.PrimitiveSet("main", 1)
    #add function set
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    #add terminal set
    pset.addTerminal(1.0)
    #rename argument ARG0 to x
    pset.renameArguments(ARG0="x")
    return pset

"""
Setup the DEAP toolbox for GP
"""
def setup_toolbox(pset):
    toolbox = base.Toolbox()

    #population
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.individual, toolbox.expr)
    toolbox.register("species", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    #genetic operators
    toolbox.register("evaluate",fitness,x_values=x_values)
    toolbox.register("mut_expr", gp.genHalfAndHalf, min_=0,max_=1)
    toolbox.register("mutate", gp.mutUniform, exp=toolbox.mut_expr, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)

    #defined to avoid too complicated trees, max height recommended by DEAP documentation
    toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=17))

    return toolbox()

def run_CCGP(toolbox): 
    species = [toolbox.species() for _ in range(2)]
    print('hi')

if __name__ == "__main__":
    print('fitness = ',fitness_func(-1))