import numpy as np
from Functions import Plotting
from Surrogates import nDimensionalFunctions as ND
import os

folderpath = os.getcwd()
ndSurrogateList = ['ackley', 'griewank', 'levy', 'rastrigin', 'michalewicz']

beliefmodel = 'GPR'
noisevals = [0, 0.1, 0.2]
randsampnumb = [5]
dimensions = [2, 4, 6]
nreps = 10

for i in range(len(ndSurrogateList)):
    surrogatename=ndSurrogateList[i]
    surrfunc = getattr(ND, surrogatename)
    for k in range(len(randsampnumb)):
        startRandSamples = randsampnumb[k]
        for l in range(len(dimensions)):
            dimension = dimensions[l]
            noiseMedbestY = []
            noiseMedMSE = []
            for j in range(len(noisevals)):
                noise = noisevals[j]
                repMSE = []
                repY = []
                repbestY = []
                repX = []
                repHV = []
                for rep in range(nreps):
                    tempdata = np.loadtxt('./save data/%s %s %.4s %s %s %s xydata.txt' % (beliefmodel,
                        ndSurrogateList[i], noisevals[j], randsampnumb[k], dimensions[l], rep), delimiter=',')

                    tempMSE = tempdata[:, -1]
                    tempY = tempdata[:, -2]
                    tempX = tempdata[:, :-2]

                    tempYNoNoise = []
                    for tempx in tempX:
                        tempYNoNoise.append(surrfunc(tempx, 0))
                    repY.append([tempYNoNoise])
                    repX.append([tempX])
                    tempY = tempYNoNoise

                    repMSE.append([ii for ii in tempMSE])
                    repY.append([tempY])
                    repX.append([tempX])


                    besty = tempY[0]
                    bestY = []
                    for tempy in tempY:
                        if tempy > besty:
                            besty = tempy
                        bestY.append(besty)
                    repbestY.append(bestY)

                medbestY = []
                medMSE = []
                medHV = []
                for ii in range(len(tempY)):
                    medbestY.append(np.median([jj[ii] for jj in repbestY]))
                    medMSE.append(np.median([jj[ii] for jj in repMSE]))

                noiseMedbestY.append(medbestY)
                noiseMedMSE.append(medMSE)

            Plotting.saveNoiseLinePlots(Y=noiseMedbestY, ylabel='Median Best Response', dimension=dimensions[l],
                                        randsampnum=randsampnumb[k], surrogate=ndSurrogateList[i],
                                        filepath='./save plots/%s %s %s bestY' % (ndSurrogateList[i], randsampnumb[k], dimensions[l]),
                                        noises=noisevals)
            np.savetxt('./save plots/%s %s %s bestY.txt' % (ndSurrogateList[i], randsampnumb[k], dimensions[l]),
                       noiseMedbestY, delimiter=',', fmt='%1.6f')

            Plotting.saveNoiseLinePlots(Y=noiseMedMSE, ylabel='Median MSE', dimension=dimensions[l],
                                        randsampnum=randsampnumb[k], surrogate=ndSurrogateList[i],
                                        filepath='./save plots/%s %s %s MSE' % (ndSurrogateList[i], randsampnumb[k], dimensions[l]),
                                        noises=noisevals)
            np.savetxt('./save plots/%s %s %s MSE.txt' % (ndSurrogateList[i], randsampnumb[k], dimensions[l]),
                       noiseMedMSE, delimiter=',', fmt='%1.6f')
