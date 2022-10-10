from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pygraphviz as pgv

import random
import operator

def problem_func(x):
    return  (1/x) + np.sin(x) if (x > 0) else 2*x + pow(x,2) + 3.0 

"""
Protected division to avoid program crashing when dividing by 0
"""
def protected_div(x,y):
    return 1.0 if(y == 0.0) else x/y

"""
Calculate fitness of individual using context vector
"""
def fitness(individual, species_index, context_vec, toolbox):
    context_vec[species_index] = individual #replace individual in context vector with individual
    #compute coopertive fitness
    return toolbox.calc_err(context_vec)

"""
Calculate MSE between vector of functions [f1(x), f2(x)] and actual function
"""
def calc_err(vec, toolbox, x_values, y_values):
    functions = [toolbox.compile(expr=i) for i in vec]
    complete_func = lambda x : functions[0](x) if(x > 0.0) else functions[1](x) #complete solution
    error_values = [pow(y_values[i] - complete_func(x), 2.0) for i,x in enumerate(x_values)]
    return [np.sum(error_values)/len(error_values)] #return mse

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
def setup_toolbox(x_values, y_values, pset, species_num, pop_size, rng):
    toolbox = base.Toolbox()

    #population
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.individual, toolbox.expr)
    toolbox.register("species", tools.initRepeat, list, toolbox.individual, pop_size)
    toolbox.register("context_vec", tools.initRepeat, list, lambda : rng.integers(low=0, high=pop_size), species_num)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("calc_err", calc_err, toolbox=toolbox, x_values=x_values, y_values=y_values)

    #genetic operators
    toolbox.register("evaluate",fitness, toolbox=toolbox)
    toolbox.register("mut_expr", gp.genHalfAndHalf, min_=0,max_=1)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.mut_expr, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize=4)

    #defined to avoid too complicated trees, max height recommended by DEAP documentation
    max_tree_depth = 17
    toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), max_value=max_tree_depth))
    toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=max_tree_depth))

    return toolbox

"""
Run the CCGP to get best averages across generations and best final GP trees
"""
def CCGP_algo(toolbox, crossover_rate, mutation_rate, max_iterations=100): 
    gp_species = [toolbox.species() for _ in range(2)]
    gp_context_vec = toolbox.context_vec()
    #evaluate species
    for i,species in enumerate(gp_species):
        for ind in species:
            context_vec_trees = [gp_species[j][context_index] for j, context_index in enumerate(gp_context_vec)]
            ind.fitness.values = toolbox.evaluate(ind, i, context_vec_trees)

    best_avgs = []
    best = None

    iter = 0
    while (iter < max_iterations):
    #for iter in range(max_iterations):
        for i,species in enumerate(gp_species):
            #select parents
            children_pop = toolbox.select(species, k=len(species))
            #generate children
            children_pop = algorithms.varAnd(children_pop, toolbox, cxpb=crossover_rate, mutpb=mutation_rate)
            #recalculate fitness
            invalid_ind = [ind for ind in children_pop if not ind.fitness.valid]
            context_vec_trees = [gp_species[j][context_index] for j, context_index in enumerate(gp_context_vec)]
            fitnesses = toolbox.map(lambda ind : toolbox.evaluate(ind, i, context_vec_trees), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            #select next gen pop
            gp_species[i] = toolbox.select(species + children_pop, k=len(species))

            #update cv with best individual
            best_index = sorted(range(len(gp_species[i])), key=lambda j : gp_species[i][j].fitness, reverse=True)[0]
            gp_context_vec[i] = best_index

        best_num = 5
        bests = [sorted(species, key=operator.attrgetter("fitness"), reverse=True)[:best_num] for species in gp_species]
        best = [bests[0][0], bests[1][0]]   #get the best trees from each species

        best_avg = [toolbox.calc_err([bests[0][i], bests[1][i]]) for i in range(best_num)]
        best_avg = np.sum(best_avg)/len(best_avg)
        best_avgs.append(best_avg)
        iter += 1

    return best, best_avgs, iter

def draw_tree_img(seed, best, name):
    nodes, edges, labels = gp.graph(best)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog='dot')

    for n in nodes:
            node = g.get_node(n)
            node.attr["label"] = labels[n]

    g.draw(name)

"""
Run CCGP using various seeds
"""
def run_ccgp(seeds, x_values, y_values, crossover_rate=0.95, mutation_rate=0.15):

    #set up CCGP
    pset = create_primitive_set()
    creator.create("fitnessmin", base.Fitness, weights=(-1.0,))
    creator.create("individual", gp.PrimitiveTree, fitness=creator.fitnessmin, pset=pset)

    print('Running CCGP')
    for seed in seeds:
        print('Seed = ', seed)
        rng = np.random.default_rng(seed=seed)
        random.seed(int(seed))  #define seed

        toolbox = setup_toolbox(x_values=x_values, y_values=y_values, pset=pset, species_num=2, pop_size=20, rng=rng)
        best, best_avgs, iter_num = CCGP_algo(toolbox, crossover_rate=crossover_rate, mutation_rate=mutation_rate)

        #draw best GP trees
        name = 'ccgp_best_tree_'+str(seed)+'_1.png'
        draw_tree_img(seed, best[0], name)
        name = 'ccgp_best_tree_'+str(seed)+'_2.png'
        draw_tree_img(seed, best[1], name)

        #draw convergence curve for this run
        fig, _ = plt.subplots(1, 1)
        fig.set_figwidth(20)
        fig.suptitle('Convergence Curve for seed '+str(seed))
        sns.lineplot(x=range(iter_num), y=best_avgs)
        fig.savefig('ccgp_curve_'+str(seed)+'.png')

        #draw output of best tree vs actual output
        functions = [toolbox.compile(expr=i) for i in [best[0], best[1]]]
        best_func = lambda x : functions[0](x) if(x > 0.0) else functions[1](x)

        print('\tBest fitness = ',toolbox.calc_err(best))

        fy_values = [best_func(x) for x in x_values]
        fig, _ = plt.subplots(1, 1)
        fig.set_figwidth(20)
        fig.suptitle('Points Comparison for seed '+str(seed))
        sns.scatterplot(x=x_values,y=y_values)
        sns.scatterplot(x=x_values,y=fy_values)
        fig.savefig('ccgp_chart_'+str(seed)+'.png')

if __name__ == "__main__":
    #x values for fitness evaluation
    x_num = 30 #number of instances
    x_values = np.linspace(start=-6.0, stop=15.0, num=x_num)
    x_values = np.concatenate([np.linspace(start=-6.0, stop=0.0, num=x_num), np.linspace(start=0.0, stop=15.0, num=x_num)])
    y_values = [problem_func(x) for x in x_values]

    seeds = np.random.default_rng(seed=5).integers(low=0, high=200, size=5)
    run_ccgp(seeds=seeds, x_values=x_values, y_values=y_values)