import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def rosenbrock(points):
    sum = 0.0
    for i in range(len(points)-1):
        a = points[i]
        b = points[i+1]
        sum += 100.0 * pow(pow(a,2) - b, 2) + pow(a - 1.0 ,2)
    return sum

def griewanks(points):
    sum = 0.0
    product = 1.0
    for i,x in enumerate(points):
        sum += (pow(x,2))/4000.0
        product *= np.cos(x/(np.sqrt(i+1))) #i+1 since python starts at 0
    return sum - product + 1.0

def gen_vector(feature_num, rng, min_value, max_value):
    x_range = np.abs(max_value)+np.abs(min_value)
    return np.array([(rng.random()*x_range)+min_value for i in range(feature_num)])

def EP(fitness_func, feature_num, rng, min_x=-30, max_x=30, variance_threshold=2.0, c=0.2, pop_size=50, max_iter=1000):
    variance_threshold = np.full((feature_num,1), variance_threshold)
    pop_var = [ 
        (gen_vector(feature_num, rng, min_value=min_x, max_value=max_x), gen_vector(feature_num, rng, min_value=0.0, max_value=(max_x-min_x))) 
        for _ in range(pop_size)]    #gen pop and variance

    best_avg = [] #average fitness of num_best individuals for each generation
    best_individual = pop_var[0][0] #get random individual for initial best

    i = 0
    while(i < max_iter):    #while not stopping
        #pop_fitness = [fitness_func(ind) for (ind, _) in pop_var]  #calc fitness for each individual
        mutated_pop_var = []
        for x,v in pop_var: #generate mutated pop
            rand_x = np.array([rng.normal(loc=0.0, scale=1.0) for j in range(feature_num)])
            rand_v = np.array([rng.normal(loc=0.0, scale=1.0) for j in range(feature_num)])

            new_x = x + rand_x*np.sqrt(v)
            new_v = v + rand_v*np.sqrt(c*v)
            new_v = np.minimum(np.maximum(new_v, np.full((feature_num,1), 0.0)), variance_threshold)    #clamp v values between 0.0 and threshold

            mutated_pop_var.append((new_x, new_v))
        #get best individuals from both parent pop and mutated pop and their variences to get next generation's parent pop
        combined_pop_var = pop_var+mutated_pop_var
        combined_pop_var.sort(key= lambda ind : fitness_func(ind[0]), reverse=True)
        pop_var = combined_pop_var[:pop_size]  

        #calc best average from top 5 individuals
        current_best_avg = np.average([fitness_func(x) for x,_ in pop_var[:6]])
        best_avg.append(current_best_avg)
        #get best individual
        best_individual = pop_var[0][0]

        i += 1
    
    #return best
    return best_individual, fitness_func(best_individual), best_avg, max_iter

if __name__ == "__main__":
    #run EP
    D = 2
    seed = 5#np.random.default_rng().integers(low=0,high=200)
    fitness_function = lambda ind : np.sum(ind)
    
    best, best_fitness, best_avg, num_iter = EP(fitness_func=fitness_function, feature_num=D, rng=np.random.default_rng(seed=seed))
    print('best = ',best,' best fitness = ',best_fitness,' iterations = ',num_iter)
    sns.lineplot(x=range(num_iter), y=best_avg)
    plt.show()