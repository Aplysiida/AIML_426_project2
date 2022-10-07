import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

import fitness_q1

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
def EP(fitness_func, feature_num, rng, min_x=-30, max_x=30, variance_range=6.0, variance_threshold=0.6, pop_size=50, max_iter=2000, max_convergence_iter = 20):
    #warnings.filterwarnings("ignore")
    pop_var = [ 
        (gen_vector(feature_num, rng, min_value=min_x, max_value=max_x), gen_vector(feature_num, rng, min_value=0.0, max_value=variance_range))
        for _ in range(pop_size)]    #gen pop and variance

    best_avg = [] #average fitness of num_best individuals for each generation
    best_individual = pop_var[0][0] #get random individual for initial best

    #used for updating mutation/variance level
    tau = 1.0/(np.sqrt(2.0*np.sqrt(feature_num)))
    tau_prime = 1.0/np.sqrt(2.0*feature_num)

    iter_num = 0
    convergence_iter = 0
    prev_best_avg = -1.0 
    while((iter_num < max_iter) & (convergence_iter < max_convergence_iter)):    #while not reached stopping criteria
        #pop_fitness = [fitness_func(ind) for (ind, _) in pop_var]  #calc fitness for each individual
        mutated_pop_var = []
        for x,m in pop_var: #generate mutated pop
            #create distribution vectors for x and mutation separately
            rand_x = np.array([rng.standard_cauchy() for j in range(feature_num)])
            rand_m = np.array([rng.standard_cauchy() for j in range(feature_num)], dtype=np.float128)            

            rand = rng.standard_cauchy()

            new_x = x + rand_x*m
            #if value in np.exp is too big will cause overflow error so just replace with variance threshold
            new_m = np.array([variance_threshold if (np.isinf(np.exp(tau_prime*rand + tau*r))) else np.exp(tau_prime*rand + tau*r) for r in rand_m])

            mutated_pop_var.append((new_x, new_m))
        #get best individuals from both parent pop and mutated pop and their variences to get next generation's parent pop
        combined_pop_var = pop_var+mutated_pop_var
        #select using tournament selection here
        opponents_num = 10
        pop_var = tournament_sel(pop=combined_pop_var, select_num=pop_size, opponents_num=opponents_num, fitness_func=fitness_func, rng=rng)

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

def run_EP(hyperparameters, D, seeds, fitness_functions, fitness_functions_names):
    for i,fitness_function in enumerate(fitness_functions):
        print('At function ',fitness_functions_names[i])
        best_fitnesses = [] #store all best fitnesses from all the seeds
        iterations = []
        hyperparameter = hyperparameters[i] 

        for j,seed in enumerate(seeds):
            print('\tseed = ',seed)
            best, best_fitness, best_avg, num_iter = EP(
                fitness_func=fitness_function, 
                feature_num=D, 
                variance_range=hyperparameter[0],
                variance_threshold=hyperparameter[1],
                rng=np.random.default_rng(seed=seed)
            )
            print(' fitness = ',best_fitness,' iter num = ',num_iter)
            best_fitnesses.append(best_fitness)
            iterations.append(num_iter)
        print('Mean = ',np.average(best_fitnesses),' Standard Deviation = ', np.std(best_fitnesses),' Average Iterations Taken = ',np.average(iterations))

if __name__ == "__main__":
    D = 20
    seeds = np.random.default_rng(seed=5).integers(low=1,high=200,size=30)
    fitness_functions = [fitness_q1.rosenbrock , fitness_q1.griewanks]
    fitness_functions_names = ['Rosenbrock','Griewanks']

    value_range = 60.0 #from -30.0 to 30.0
    #rosenbrock
    variance_range = value_range/10.0
    variance_threshold = variance_range/10.0
    hyperparameters = [(variance_range, variance_threshold)]
    #griewank
    variance_range = value_range/10.0
    variance_threshold = variance_range/10.0
    hyperparameters.append((variance_range, variance_threshold))

    print('D = 20')
    run_EP(hyperparameters=hyperparameters, D=D, seeds=seeds, fitness_functions=fitness_functions, fitness_functions_names=fitness_functions_names)

    D = 50
    print('D = 50')
    fitness_functions = fitness_functions[:1]
    fitness_functions_names = fitness_functions_names[1:]
    hyperparameters = hyperparameters[:1]
    run_EP(hyperparameters=hyperparameters, D=D, seeds=seeds, fitness_functions=fitness_functions, fitness_functions_names=fitness_functions_names)