from os import DirEntry
import pandas as pd
import numpy as np

import sys

rng = np.random.default_rng(seed=50)

"""
Parse files into tuple (num of items, bag capacity, dataframe)
"""
def parse_data(filepath):
    df = pd.read_table(filepath, sep=' ')
    M = df.columns[0]
    Q = df.columns[1]
    df.columns.values[0], df.columns.values[1] = 'value','weight'
    return (M,Q,df)  

"""
Generate one individual for pop
    p = probability vector
    feature_num = length of individual
"""
def gen_individual(p, feature_num):
    return np.array([1 if (rng.random() < p[k]) else 0 for k in range(feature_num)])

"""
Update probability vector using selected individuals
    p = probability vector
    learning_rate = +learning_rate if best, -learning_rate if worst
    individuals = best or worst individuals
"""
def update_prob(p, learning_rate, individuals):
    for ind in individuals:
        p = p + learning_rate * (ind - p)
    return p

def PBIL(fitness_func, feature_num = 5, pop_size = 50, max_iter = 100, num_best = 5, num_worst = 5, max_p = 0.9, min_p = 0.1, learning_rate=0.1):
    p = np.linspace(start=0.5, stop=0.5, num=feature_num) #initialise prob vector
    #pop = [gen_individual(p, feature_num) for _ in range(pop_size)] 

    #for i in range(max_iter):  #while not stop, todo: add convergence
    pop = [gen_individual(p, feature_num) for _ in range(pop_size)] #use p vector to gen pop
    pop.sort(key=fitness_func, reverse=True)  #sort individuals by fitness

    best_individuals = pop[:num_best]
    worst_individuals = pop[pop_size - num_worst:]
    #check if update function is correct
    p = update_prob(p, learning_rate, best_individuals)   #for best individuals
    p = update_prob(p, (-1.0 * learning_rate), worst_individuals)   #for worst individuals

    #mutate p
    p = np.array([np.maximum(np.minimum(p_i, max_p),min_p) for p_i in p])   #clamp p
    
    #return best individual


if __name__ == "__main__":
    filepaths = sys.argv[1:]
    #store dataset as (num of items, bag capacity, data)
    datasets = [parse_data(filepath=filepath) for filepath in filepaths] #parse files

    fit_func = lambda ind : np.sum(ind)

    PBIL(fitness_func=fit_func)