import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import random

"""
Fitness functions to use
"""
def rosenbrock(x_values):
    sum_value = 0.0
    for i in range(len(x_values)-1):
        a = x_values[i]
        b = x_values[i+1]
        sum_value += 100.0 * pow(pow(a,2) - b, 2) + pow(a - 1.0 ,2)
    return sum_value

def griewanks(x_values):
    sum_value = 0.0
    product = 1.0
    for i,x in enumerate(x_values):
        sum_value += (pow(x,2))/4000.0
        product *= np.cos(x/(np.sqrt(i+1))) #i+1 since python starts at 0
    return sum_value - product + 1.0

"""
Generate a vector that could represent an individual or variance
"""
def gen_vector(feature_num, rng, min_value, max_value):
    x_range = np.abs(max_value)+np.abs(min_value)
    return np.array([(rng.random()*x_range)+min_value for i in range(feature_num)])

"""
Tournment selection used for selecting best individuals for next generation
"""
def tournament_sel(pop, select_num, opponents_num, fitness_func, rng):
    pop_copy = pop.copy()
    selected = []
    for _ in range(select_num):
        indices = rng.integers(low=0, high=(len(pop_copy)-1), size=opponents_num)
        opponents = [ (pop_copy[i], i) for i in indices]
        
        opponents.sort(key=lambda ind : fitness_func(ind[0][0]))
        to_select = opponents[0]
        selected.append(to_select[0]) 
        pop_copy.pop(to_select[1])   #to avoid selecting the same object
    return selected

"""
Combination of Fast-EP and Improved-EP
"""
def EP(fitness_func, feature_num, rng, min_x=-30, max_x=30, variance_range=6.0, variance_threshold=0.6, c=1.0, pop_size=50, max_iter=1000, max_convergence_iter = 20):
    #variance_threshold_vector = np.full((feature_num,1), variance_threshold)
    pop_var = [ 
        (gen_vector(feature_num, rng, min_value=min_x, max_value=max_x), gen_vector(feature_num, rng, min_value=0.0, max_value=variance_range))
        for _ in range(pop_size)]    #gen pop and variance

    best_avg = [] #average fitness of num_best individuals for each generation
    best_individual = pop_var[0][0] #get random individual for initial best

    iter_num = 0
    convergence_iter = 0
    prev_best_avg = -1.0 
    while((iter_num < max_iter) & (convergence_iter < max_convergence_iter)):    #while not reached stopping criteria
        #pop_fitness = [fitness_func(ind) for (ind, _) in pop_var]  #calc fitness for each individual
        mutated_pop_var = []
        for x,v in pop_var: #generate mutated pop
            rand_x = np.array([rng.standard_cauchy() for j in range(feature_num)])
            rand_v = np.array([rng.standard_cauchy() for j in range(feature_num)])

            new_x = x + rand_x*np.sqrt(v)
            new_v = v + rand_v*np.sqrt(c*v)
            #clamp v values between 0.0 and threshold
            new_v = np.array([np.min([np.max([v_value, 0.0]), variance_threshold]) for v_value in new_v])

            mutated_pop_var.append((new_x, new_v))
        #get best individuals from both parent pop and mutated pop and their variences to get next generation's parent pop
        combined_pop_var = pop_var+mutated_pop_var
        #select using tournament selection here
        opponents_num = 4
        pop_var = tournament_sel(pop=combined_pop_var, select_num=pop_size, opponents_num=opponents_num, fitness_func=fitness_func, rng=rng)
        
        #combined_pop_var.sort(key= lambda ind : fitness_func(ind[0]))    #sort by minimum fitness
        #pop_var = combined_pop_var[:pop_size]  

        #calc best average from top 5 individuals and check for convergence
        current_best_avg = np.average([fitness_func(x) for x,_ in pop_var[:6]])
        best_avg.append(current_best_avg)
        if(np.abs(current_best_avg - prev_best_avg) < 0.0000000000001):
            convergence_iter += 1
        else: convergence_iter = 0
        prev_best_avg = current_best_avg
        #get best individual
        best_individual = pop_var[0][0]

        iter_num += 1
    
    #return best
    return best_individual, fitness_func(best_individual), best_avg, iter_num

if __name__ == "__main__":
    #run EP
    D = 20
    seeds = np.random.default_rng(seed=5).integers(low=0,high=200,size=3)
    fitness_functions = [rosenbrock , griewanks]
    fitness_functions_names = ['Rosenbrock','Griewanks']

    value_range = 60.0 #from -30.0 to 30.0
    variance_range = value_range/10.0
    variance_threshold = variance_range/10.0

    for i,fitness_function in enumerate(fitness_functions):
        print('At function ',fitness_functions_names[i])
        best_fitnesses = [] #store all best fitnesses from all the seeds

        fig, axis = plt.subplots(1, len(seeds))
        fig.set_figwidth(20)
        fig.suptitle(fitness_functions_names[i]+' Convergence Curve')

        for j,seed in enumerate(seeds):
            print('seed = ',seed)
            best, best_fitness, best_avg, num_iter = EP(
                fitness_func=fitness_function, 
                feature_num=D, 
                variance_range = variance_range,
                variance_threshold=variance_threshold, 
                rng=np.random.default_rng(seed=seed)
            )
            best_fitnesses.append(best_fitness)
            print('best = ',best,' best fitness = ',best_fitness,' iterations = ',num_iter)
            sns.lineplot(x=range(num_iter), y=best_avg, ax=axis[j])
        print('Mean = ',np.average(best_fitnesses),' Standard Deviation = ', np.std(best_fitnesses))
        plt.show()

    D = 50
    fitness_functions = fitness_functions[1:]
    fitness_functions_names = fitness_functions_names[1:]