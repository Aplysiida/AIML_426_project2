from deap import algorithms
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
    #compute coopertive fitness

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
def setup_toolbox(x_values, pset, pop_size):
    toolbox = base.Toolbox()

    #population
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.individual, toolbox.expr)
    toolbox.register("species", tools.initRepeat, list, toolbox.individual, pop_size)
    toolbox.register("context vector", tools.initIterate, list, """Add function that gets random unique indices from species""")
    toolbox.register("compile", gp.compile, pset=pset)

    #genetic operators
    toolbox.register("evaluate",fitness,x_values=x_values)
    toolbox.register("mut_expr", gp.genHalfAndHalf, min_=0,max_=1)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.mut_expr, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize = 5)

    #defined to avoid too complicated trees, max height recommended by DEAP documentation
    max_tree_depth = 3
    toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), max_value=max_tree_depth))
    toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=max_tree_depth))

    return toolbox

def run_CCGP(toolbox, crossover_rate, mutation_rate, max_iterations=1000): 
    gp_species = [toolbox.species() for _ in range(2)]
    #evaluate species

    for i in range(max_iterations):
        for i,species in enumerate(gp_species):
            #apply genetic operators
            offspring = algorithms.varAnd(species, toolbox, crossover_rate, mutation_rate)
            #evaluate fitness
            for ind in offspring:
                ind.fitness.values = toolbox.evaluate([ind])
            gp_species[i] = toolbox.select(offspring, len(offspring))   #select individuals for next generation

if __name__ == "__main__":
    #x values for fitness evaluation
    x_num = 30 #number of instances
    x_values = np.linspace(start=-6.0, stop=15.0, num=x_num)

    pset = create_primitive_set()
    creator.create("fitnessmin", base.Fitness, weights=(-1.0,))
    creator.create("individual", gp.PrimitiveTree, fitness=creator.fitnessmin, pset=pset)

    toolbox = setup_toolbox(x_values=x_values, pset=pset, pop_size=5)
    run_CCGP(toolbox, crossover_rate=0.1, mutation_rate=0.05)