import plotly.plotly as plotly
import plotly.graph_objs as graphly
import numpy as np
def myFunc(x, y, z):
    return (2*x, 2*y, 2*z)

def find_min(gradient, init, learning, steps):
    '''Finds the minimum of a given function using gradient descent'''
    x, y, z = init
    coords = np.zeros((steps,3))
    for i in range(steps):
        dx, dy, dz = gradient(x, y, z)
        x = x - dx*learning
        y = y - dy*learning
        z = z - dz*learning
        coords[i,:] = [x,y,z]
    return coords

def graph_it(gradient, init, learning, steps):
    data = find_min(myFunc, (10, 10, 10), .1, 3)
    dadata = [graphly.Surface(data)]
    layout = plotly.Layout(
        title = 'Gradient Descent',
        autosize = False,
        width = 500,
        height = 500,
        margin = dict(l=65, r=50, b=65, t=90))
    fig = plotly.Figure(data=data, layout=layout)
    plotly.iplot(fig, filename = 'Gradient Descent Test')


print(graph_it(myFunc, (10, 10, 10), .1, 3))
