def find_min(learning, steps):
    '''Finds the minimum of a given function using gradient descent'''
    x = 10
    function = x^2
    for i in range(steps):
        gradient = 2*x
        x = x - learning*gradient
        print(x)
    return x

find_min(.1,100)
