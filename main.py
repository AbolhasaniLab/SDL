import numpy as np
from Functions import BayesianOptimization as BO
from time import time
import os

folderpath = os.getcwd()

ndSurrogateList = ['ackley', 'griewank', 'levy', 'rastrigin', 'michalewicz', ]

noisevals = [0, 0.1, 0.2]
randsampnumb = [5]
dimensions2 = [2, 4, 6]
nreps = 10

for i in range(len(ndSurrogateList)):
    surrogatename=ndSurrogateList[i]
    for j in range(len(noisevals)):
        noise=noisevals[j]
        for k in range(len(randsampnumb)):
            startRandSamples = randsampnumb[k]
            tic = time()
            for l in range(len(dimensions2)):
                dimensions = dimensions2[l]
                for rep in range(nreps):
                    beliefmodeltype = 'BRMLPR_EGS'
                    # beliefmodeltype = 'GPR'
                    BO2 = BO.run(modeltype=beliefmodeltype, policy='UCB', surrogate=surrogatename, noise = noise,  runlength = 20,
                                 folderpath=folderpath,startRandSamples = startRandSamples,
                                 dimensions=dimensions).singleOptimization()
                    BO22 = np.array(BO2.X)
                    BO22Y = np.array(BO2.Y)
                    BO22MSE = np.array(BO2.MSE)

                    BO4 = np.c_[BO22, BO22Y, BO22MSE]

                    np.savetxt('./save data/%s %s %.4s %s %s %s xydata.txt' % (beliefmodeltype, ndSurrogateList[i],
                                                                               noisevals[j], randsampnumb[k],
                                                                               dimensions2[l], rep), BO4, delimiter=',', fmt='%1.6f')

            toc = time()
            print(toc-tic)