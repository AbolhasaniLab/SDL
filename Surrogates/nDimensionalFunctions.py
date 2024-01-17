"""
Adapted from https://www.sfu.ca/~ssurjano/index.html

X is a 1xN dimensional array of values with range 0 to 1
noise is the magnitude of a gaussian noise distribution applied to the
    final output.
Y is the function output scaled from ?? to ??
TODO: Scale Y from 0 to 1

"""

import numpy as np
import math
from scipy import optimize
from Surrogates import nDimensionalFunctions as ND


def findSpaceBounds():
    ndSurrogateList = ['ackley', 'griewank', 'levy', 'rastrigin', 'michalewicz']
    dict = {}
    dimensions = [2, 4, 6]
    for surr in ndSurrogateList:
        dict[surr] = {}
        for dim in dimensions:
            dict[surr][dim] = {}
            for bound in ['Min', 'Max']:
                dict[surr][dim][bound] = {
                    'X': [],
                    'Y': []
                }

    for surr in ndSurrogateList:
        for dim in dimensions:
            bnds = []
            for ii in range(dim):
                bnds.append((0,1))
            surrfunc = getattr(ND, surr)

            minfuncs = ['BH', 'DE', 'SH', 'DA', 'DI']


            f = lambda x: surrfunc(x, normY=False)
            result = {}
            result['BH'] = optimize.basinhopping(f, np.random.rand(dim), minimizer_kwargs={'bounds': bnds})
            result['DE'] = optimize.differential_evolution(f, bnds)
            result['SH'] = optimize.shgo(f, bnds)
            result['DA'] = optimize.dual_annealing(f, bnds)
            result['DI'] = optimize.direct(f, bnds)

            #.fun == value of function at solution
            bestfun = 'BH'
            for minfunc in minfuncs:
                if result[minfunc].fun < result[bestfun].fun:
                    bestfun = minfunc
                #elif result[minfunc].fun == results[bestfun].fun:
                #    bestfun=

            print('\n', surr, dim, 'min', bestfun)
            for minfunc in minfuncs:
                print(minfunc, result[minfunc].fun)

            dict[surr][dim]['Min']['X'] = [x for x in result[bestfun].x]
            dict[surr][dim]['Min']['Y'] = result[bestfun].fun


            f = lambda x: (-1)*surrfunc(x, normY=False)
            result = {}
            result['BH'] = optimize.basinhopping(f, np.random.rand(dim), minimizer_kwargs={'bounds': bnds})
            result['DE'] = optimize.differential_evolution(f, bnds)
            result['SH'] = optimize.shgo(f, bnds)
            result['DA'] = optimize.dual_annealing(f, bnds)
            result['DI'] = optimize.direct(f, bnds)

            bestfun = 'BH'
            for minfunc in minfuncs:
                if result[minfunc].fun < result[bestfun].fun:
                    bestfun = minfunc

            print('\n', surr, dim, 'max', bestfun)
            for minfunc in minfuncs:
                print(minfunc, result[minfunc].fun)

            dict[surr][dim]['Max']['X'] = [x for x in result[bestfun].x]
            dict[surr][dim]['Max']['Y'] = -result[bestfun].fun


        print(dict)
    return dict


def getBounds(surrogate, dim):
    dict = {
        'ackley': {
            2: {'Min': {'X': [0.9937512717150051, 0.006248721980131124],
                        'Y': -22.342987592189562},
                'Max': {'X': [0.5, 0.5],
                        'Y': -4.440892098500626e-16}},
            4: {'Min': {'X': [0.9938271604938271, 0.9938271604938271, 0.9938271604938271, 0.9938271604938271],
                        'Y': -22.34271989812117},
                'Max': {'X': [0.5, 0.5, 0.5, 0.5],
                        'Y': -4.440892098500626e-16}},
            6: {'Min': {'X': [0.993751319658117, 0.0062486703263544075, 0.9937513196657057, 0.9937513196540958, 0.9937513196621846, 0.9812512861330894], 'Y': -22.342738975915964}, 'Max': {'X': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        'Y': -4.440892098500626e-16}}},
        'griewank': {
            2: {'Min': {'X': [0.002325541034839568, 0.0],
                        'Y': -181.10750183468977},
                'Max': {'X': [0.5, 0.5],
                        'Y': 0.0}},
            4: {'Min': {'X': [0.0, 1.0, 0.0, 0.9992988451156694],
                        'Y': -361.0318791323699},
                'Max': {'X': [0.5, 0.5, 0.5, 0.5],
                        'Y': 0.0}},
            6: {'Min': {'X': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                        'Y': -540.995996902623},
                'Max': {'X': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        'Y': 0.0}}},
        'levy': {
            2: {'Min': {'X': [0.0, 0.0],
                        'Y': -175.14061790369217},
                'Max': {'X': [0.55, 0.55],
                        'Y': -1.4997597826618576e-32}},
            4: {'Min': {'X': [0.0, 0.0, 0.0, 0.0],
                        'Y': -334.65623580738435},
                'Max': {'X': [0.55, 0.55, 0.55, 0.55],
                        'Y': -1.4997597826618576e-32}},
            6: {'Min': {'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        'Y': -494.1718537110765},
                'Max': {'X': [0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
                        'Y': -1.4997597826618576e-32}}},
        'rastrigin': {
            2: {'Min': {'X': [0.9416985938196485, 0.05830140028483433],
                        'Y': -80.70658038767723},
                'Max': {'X': [0.5000000000093624, 0.4999999999614362],
                        'Y': 0.0}},
            4: {'Min': {'X': [0.05830139542662738, 0.941698594499639, 0.058301395453185766, 0.05830139544562652],
                        'Y': -161.4131607753538},
                'Max': {'X': [0.5000000000293175, 0.4999999996594937, 0.5000000002456968, 0.5000000004765068],
                        'Y': -0.0}},
            6: {'Min': {'X': [0.058301395654848834, 0.05830139552437053, 0.9416985944014051, 0.941698594317194, 0.9416985945374454, 0.058301395582010127],
                        'Y': -242.11974116303065},
                'Max': {'X': [0.4999999997591995, 0.4999999999475261, 0.49999999970975245, 0.5000000001385585, 0.49999999987189236, 0.499999999616912],
                        'Y': 0.0}}},
        'michalewicz': {
            2: {'Min': {'X': [0.0, 1.0],
                        'Y': 0.0},
                'Max': {'X': [0.7012065995292605, 0.4999999986689657],
                        'Y': 1.8013034100985466}},
            4: {'Min': {'X': [0.0, 1.0, 0.0, 0.0],
                        'Y': 0.0},
                'Max': {'X': [0.7012066029404929, 0.49999999548433405, 0.40902552695663236, 0.6121285154466356],
                        'Y': 3.698857098466492}},
            6: {'Min': {'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        'Y': 0.0},
                'Max': {'X': [0.7012059893343676, 0.5000003998695622, 0.40902510281590176, 0.6121284591080173, 0.5476425010351482, 0.5000000545527541],
                        'Y': 5.687658178839291}
                }
        }
    }

    lowerBnd = dict[surrogate][dim]['Min']['Y']
    upperBnd = dict[surrogate][dim]['Max']['Y']

    return lowerBnd, upperBnd

def scaleY(surr, dim, y):
    lowerBnd, upperBnd = getBounds(surr, dim)
    Y = (y-lowerBnd) / (upperBnd - lowerBnd)
    return Y

def ackley(X, noise=0, normY=True):
    xrange = [-40, 40]
    xx = [xrange[0] + x*(xrange[1] - xrange[0]) for x in X]

    d = len(xx)
    c = 2 * math.pi
    b = 0.2
    a = 20

    sum1 = 0
    sum2 = 0
    for ii in range(d):
        xi = xx[ii]
        sum1 = sum1 + xi ** 2
        sum2 = sum2 + math.cos(c * xi)

    term1 = -a * math.exp(-b * math.sqrt(sum1 / d))
    term2 = -math.exp(sum2 / d)

    y = term1 + term2 + a + math.exp(1)
    y = -y
    if normY:
        y = scaleY('ackley', d, y)

    return y + noise*np.random.normal()

def griewank(X, noise=0, normY=True):
    xrange = [-600, 600]
    xx = [xrange[0] + x * (xrange[1] - xrange[0]) for x in X]
    d = len(xx)
    S = 0
    prod = 1

    for ii in range(d):
        xi = xx[ii]
        S = S + xi ** 2 / 4000
        prod = prod * math.cos(xi / math.sqrt(ii+1))

    y = S - prod + 1
    y = -y

    if normY:
        y = scaleY('griewank', d, y)

    return y + noise*np.random.normal()

def levy(X, noise=0, normY=True):
    xrange = [-10, 10]
    xx = [xrange[0] + x * (xrange[1] - xrange[0]) for x in X]
    d = len(xx)

    w = np.zeros((d, 1))
    for ii in range(d):
        w[ii] = 1 + (xx[ii] - 1) / 4

    term1 = (math.sin(math.pi * w[0])) ** 2
    term3 = (w[d-1] - 1) ** 2 * (1 + (math.sin(2 * math.pi * w[d-1])) ** 2)

    S = 0
    for ii in range(d):
        wi = w[ii]
        new = (wi - 1) ** 2 * (1 + 10 * (math.sin(math.pi * wi + 1)) ** 2)
        S = S + new

    y = term1 + S + term3
    y = y[0]
    y = -y

    if normY:
        y = scaleY('levy', d, y)

    return y + noise*np.random.normal()

def rastrigin(X, noise=0, normY=True):
    xrange = [-5.12, 5.12]
    xx = [xrange[0] + x * (xrange[1] - xrange[0]) for x in X]
    d = len(xx)

    S = 0
    for ii in range(d):
        xi = xx[ii]
        S = S + (xi ** 2 - 10 * math.cos(2 * math.pi * xi))

    y = 10 * d + S
    y = -y

    if normY:
        y = scaleY('rastrigin', d, y)

    return y + noise*np.random.normal()

def michalewicz(X, noise=0, normY=True):
    xrange = [0, math.pi]
    xx = [xrange[0] + x * (xrange[1] - xrange[0]) for x in X]
    d = len(xx)

    m = 10

    S = 0

    for ii in range(d):
        xi = xx[ii]
        new = math.sin(xi) * (math.sin((ii+1) * xi ** 2 / math.pi)) ** (2 * m)
        S = S + new

    y = -S
    y = -y

    if normY:
        y = scaleY('michalewicz', d, y)

    return y + noise*np.random.normal()