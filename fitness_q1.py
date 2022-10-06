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