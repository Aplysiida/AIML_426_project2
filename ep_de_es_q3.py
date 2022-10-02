import numpy as np

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

def EP(pop_size=50, max_iter=1000):
    #gen pop
    #gen variance
    #while not stopping
        #calc fitness for each individual
        #for each individual in pop
            #do something
        #get best individuals from both parent pop and mutated pop and their variences to get next generation's parent pop
    #return best
    print('hello')

if __name__ == "__main__":
    #generate data 

    #run EP
    print('here')