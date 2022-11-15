import numpy as np
import time
import idx2numpy
import os

from math import exp, sqrt, log
from numpy.random import normal
from scipy.special import erf
from PIL import Image
from numpy import asarray

# INITIALISING START TIME AND SEED FOR RANDOM SAMPLING
startTime = time.perf_counter
print("\nStarting...")
np.random.seed(3820672)

# ARRAYS STORING SETS OF VALUES OF EACH VARIABLE WITH OPTIMA CHOSEN AS CONSTANTS
epsset = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
epsconst = epsset[1]

# VECTOR DIMENSION CHOSEN TO MATCH THAT OF CONVERTED IMAGES ABOVE AND NUMBER OF CLIENTS CHOSEN TO GIVE SENSIBLE GS
dtaset = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
dtaconst = dtaset[1]

dsetCifar = [128, 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072]
dsetFashion = [147, 196, 245, 294, 392, 448, 490, 588, 672, 784]
dsetFlair = [768, 1536, 2304, 3072, 4608, 6144, 8192, 9216, 9984, 12288]

dset = [dsetCifar, dsetFashion, dsetFlair]
dconst = maxDim = [arr[9] for arr in dset]

nsetMost = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
nsetFashion = [15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]

nset = [nsetMost, nsetFashion]
nconst = [arr[8] for arr in nset]
maxNum = [arr[9] for arr in nset]

pairsArr = [(dconst[0], nconst[0]), (dconst[1], nconst[1]), (dconst[2], nconst[0])]
GS = [float(sqrt(d))/n for d, n in pairsArr]
maxArraySize = [d*n for d, n in pairsArr]

# INITIALISING OTHER PARAMETERS/CONSTANTS
parset = ['eps', 'dta', 'd', 'n']
rset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
R = len(rset)
mseSum = 0
mseList = list()

# IN THEORY TWO NOISE TERMS ARE ADDED WITH EACH USING EPS AND DTA HALF THE SIZE OF IN EXPERIMENTS
epsTheory = epsconst/2
dtaTheory = dtaconst/2
xiTheory = [(2*d*log(1.25/dtaTheory))/((n**2)*(epsTheory**2)) for d, n in pairsArr]

# FOR ML-FLAIR THE NUMBER OF SMALL IMAGES IS LARGE
smallImages = 429078

# ADAPTATION OF UNPICKLING OF CIFAR-10 FILES BY KRIZHEVSKY
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# ADAPTATION OF LOADING AND SPLITTING FIVE FILES OF TRAINING DATA BY RHYTHM
def loadCifar10():
    for i in range(1, 6):
        filename = 'data_batch_' + str(i)
        dict = unpickle(filename)
        xData = dict[b'data']

        # CONCATENATE X AND Y DATA FROM ALL FILES INTO RELEVANT VARIABLE
        if i == 1:
            xTrain = xData
        else:
            xTrain = np.concatenate((xTrain, xData), axis=0)
    return xTrain

# LOADING AND SPLITTING CIFAR-100 DATA
def loadCifar100():
    filename = 'train'
    dict = unpickle(filename)
    xTrain = dict[b'data']
    return xTrain

# LOADING FASHION-MNIST DATA
def loadFashion(maxNum, maxDim):
    filename = 'train-images-idx3-ubyte'
    dict = idx2numpy.convert_from_file(filename)
    xTrain = dict.reshape((maxNum, maxDim))
    return xTrain

# LOADING ML-FLAIR DATA BY GEEKSFORGEEKS
def loadFlair(maxDim):
    path = 'small_images'
    os.chdir(path)
    xList = []

    from progress.bar import FillingSquaresBar
    bar = FillingSquaresBar(max=smallImages-1, suffix = '%(percent) d%% : %(elapsed)ds elapsed')
 
    for file in os.listdir():

        if file.endswith('.jpg'):
            img = Image.open(file)
            dict = asarray(img)
            vector = dict.reshape((1, maxDim))
            xList.append(vector)
        
        bar.next()
    bar.finish()

    xTrain = np.array(xList)
    return xTrain

# ADAPTATION OF TRANSFORMATION OF LABEL INDICES TO ONE-HOT ENCODED VECTORS AND IMAGES TO 3072-DIMENSIONAL VECTORS BY HADHAZI
def transformValues(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

# CALL ALL THE ABOVE METHODS
print("Loading data...")
xTrainCifar10 = loadCifar10()
xTrainCifar100 = loadCifar100()
xTrainFashion = loadFashion(maxNum[1], maxDim[1])
xTrainFlair = loadFlair(maxDim[2])

xTrain = [xTrainCifar10, xTrainCifar100, xTrainFashion, xTrainFlair]
xTrainNew = [transformValues(data) for data in xTrain]
xTrainSimple = [np.full((n, d), 0.5) for d, n in pairsArr]

def runLoop(dataIndex, xTrainChoice, index, var, epschoice, dtachoice, dchoice, nchoice):

    if dataIndex == 0:
        if np.all(element == 0.5 for element in xTrainChoice[dataIndex]):
            datafile = open("cifar10_simple_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
        else:
            datafile = open("cifar10_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    elif dataIndex == 1:
        if np.all(element == 0.5 for element in xTrainChoice[dataIndex]):
            datafile = open("cifar100_simple_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
        else:
            datafile = open("cifar100_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    elif dataIndex == 2:
        if np.all(element == 0.5 for element in xTrainChoice[dataIndex]):
            datafile = open("fashion_simple_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
        else:
            datafile = open("fashion_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    else:
        if np.all(element == 0.5 for element in xTrainChoice[dataIndex]):
            datafile = open("flair_simple_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
        else:
            datafile = open("flair_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")

    datafile.write("Statistics from Theory and Binary Search in AGM")

    if dataIndex == 0 or dataIndex == 1:
        datafile.write(f"\n\nxiTheory: {round(xiTheory[0], 7):>21}")
    elif dataIndex == 2:
        datafile.write(f"\n\nxiTheory: {round(xiTheory[1], 7):>21}")
    else:
        datafile.write(f"\n\nxiTheory: {round(xiTheory[2], 7):>21}")

    def calibrateAGM(eps, dta, GS, tol=1.e-12):
        """ Calibrate a Gaussian perturbation for DP using the AGM of [Balle and Wang, ICML'18]
        Arguments:
        eps : target epsilon (eps > 0)
        dta : target delta (0 < dta < 1)
        GS : upper bound on L2 global sensitivity (GS >= 0)
        tol : error tolerance for binary search (tol > 0)
        Output:
        sig : s.d. of Gaussian noise needed to achieve (eps,dta)-DP under global sensitivity GS
        """
        loopTime = time.perf_counter()

        # DEFINE GAUSSIAN CUMULATIVE DISTRIBUTION FUNCTION PHI WHERE ERF IS STANDARD ERROR FUNCTION
        def Phi(t):
            return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

        # VALUE V STAR IS LARGEST SUCH THAT THIS EXPRESSION IS LESS THAN OR EQUAL TO DTA
        def caseA(eps, u):
            return Phi(sqrt(eps*u)) - exp(eps)*Phi(-sqrt(eps*(u+2.0)))

        # VALUE U STAR IS SMALLEST SUCH THAT THIS EXPRESSION IS LESS THAN OR EQUAL TO DTA
        def caseB(eps, u):
            return Phi(-sqrt(eps*u)) - exp(eps)*Phi(-sqrt(eps*(u+2.0)))

        # IF INF AND SUP NOT LARGE ENOUGH THEN TRY DOUBLE NEXT TIME
        def doublingTrick(predicateStop, uInf, uSup):
            while (not predicateStop(uSup)):
                uInf = uSup
                uSup = 2.0*uInf
            return uInf, uSup

        # SIMPLE BINARY SEARCH TO FIND MIDPOINT BETWEEN SUP AND INF
        def binarySearch(predicateStop, predicateLeft, uInf, uSup):
            uMid = uInf + (uSup - uInf)/2.0
            while (not predicateStop(uMid)):
                if (predicateLeft(uMid)):
                    uSup = uMid
                else:
                    uInf = uMid
                uMid = uInf + (uSup - uInf)/2.0
            datafile.write(f"\nbinary root: {round(uMid, 4):>16}")
            return uMid

        # INITIAL GUESS FOR DTA
        dtaZero = caseA(eps, 0.0)

        if (dta == dtaZero):
            alpha = 1.0

        # IF GUESS IS NOT CORRECT THEN RUN ONE OF TWO LOOPS BASED ON WHETHER DTA IS LARGER OR SMALLER THAN GUESS
        else:
            if (dta > dtaZero):
                predicateStopDT = lambda u : caseA(eps, u) >= dta
                functionDta = lambda u : caseA(eps, u)
                predicateLeftBS = lambda u : functionDta(u) > dta
                functionAlpha = lambda u : sqrt(1.0 + u/2.0) - sqrt(u/2.0)

            else:
                predicateStopDT = lambda u : caseB(eps, u) <= dta
                functionDta = lambda u : caseB(eps, u)
                predicateLeftBS = lambda u : functionDta(u) < dta
                functionAlpha = lambda u : sqrt(1.0 + u/2.0) + sqrt(u/2.0)

            predicateStopBS = lambda u : abs(functionDta(u) - dta) <= tol

            uInf, uSup = doublingTrick(predicateStopDT, 0.0, 1.0)
            uFinal = binarySearch(predicateStopBS, predicateLeftBS, uInf, uSup)
            alpha = functionAlpha(uFinal)

        casetime = time.perf_counter() - loopTime
        datafile.write(f"\ncalibration: {round(casetime, 6):>18} seconds\n")

        if dataIndex == 0 or dataIndex == 1:
            sigma = alpha*GS[0]/sqrt(2.0*eps)
        elif dataIndex == 2:
            sigma = alpha*GS[1]/sqrt(2.0*eps)
        else:
            sigma = alpha*GS[2]/sqrt(2.0*eps)

        return sigma

    # CALL ALGORITHM FOR AGM TO FIND SIGMA GIVEN EPS AND DTA AS INPUT
    sigma = calibrateAGM(epschoice, dtachoice, GS, tol=1.e-12)
    print("Calibrating AGM...")
    datafile.write("\nStatistics from AGM and computation of MSE")
    datafile.write(f"\n\nsigma from AGM: {round(sigma, 4):>13}")
    datafile.write(f"\nsquare: {round(sigma**2, 8):>25}")

    # FUNCTION BY SCOTT BASED ON OWN LEMMAS THEOREMS AND COROLLARIES IN PAPER
    def computeMSE(xTrainChoice, dchoice, sigma, nchoice, mseSum):

        loopTime = time.perf_counter()
        varSum = 0

        if (dchoice != maxDim):
            if dataIndex == 0 or dataIndex == 1:
                xTrainCrop = xTrainChoice[dataIndex].reshape((int(maxArraySize[0]/dchoice), dchoice))
            elif dataIndex == 2:
                xTrainCrop = xTrainChoice[dataIndex].reshape((int(maxArraySize[1]/dchoice), dchoice))
            else:
                xTrainCrop = xTrainChoice[dataIndex].reshape((int(maxArraySize[2]/dchoice), dchoice))

            xTrainChoice[dataIndex] = xTrainCrop

        
        mu = np.mean(xTrainChoice[dataIndex], axis=0)
        datafile.write(f"\nmu: {str(np.round((sum(mu))/dchoice, 5)):>26}")
        muSquares = [a**2 for a in mu]
        datafile.write(f"\nsum of squares: {str(np.round((sum(muSquares))/dchoice, 5)):>14}")

        noisyMu = [0]*dchoice

        # ADDING FIRST NOISE TERM TO MU DERIVED FROM GAUSSIAN DISTRIBUTION WITH MEAN 0 AND VARIANCE SIGMA SQUARED
        for i in range(0, dchoice):
            xi1 = normal(0, sigma**2)
            noisyMu[i] = mu[i] + xi1
        datafile.write(f"\nmu + noise: {round((sum(noisyMu))/dchoice, 5):>18}")

        # FIRST SUBTRACTION BETWEEN CIFAR-10 VECTOR OF EACH CLIENT AND NOISY MEAN ACCORDING TO THEOREM FOR DISPERSION
        for j in range(0, nchoice):
            noisySigma = np.subtract(xTrainChoice[dataIndex][j], noisyMu)
        datafile.write(f"\nsigma + noise: {round((sum(noisySigma))/nchoice, 4):>14}")

        # PREPARING EXPRESSION OF DISPERSION FOR ADDITION OF SECOND NOISE TERM
        twiceNoisySigma = np.multiply(noisySigma, 2)
        noisyVar = np.power(noisySigma, 2)
        datafile.write(f"\nsigma + twice noise: {round((sum(twiceNoisySigma))/nchoice, 4):>8}")
        datafile.write(f"\nvar + noise: {round((sum(noisyVar))/nchoice, 6):>18}")

        # ADDING SECOND NOISE TERM TO EXPRESSION OF DISPERSION AND COMPUTING MSE USING VARIABLES DEFINED ABOVE
        for j in range(0, nchoice):
            xi2 = normal(0, sigma**2)
            doubleNoisyVar = noisyVar + xi2
            varSum += sum(doubleNoisyVar)
            bracket = np.subtract(noisyMu, twiceNoisySigma)
            outside = np.multiply(bracket, noisyMu)
            mse = np.add(outside, xi2)
            mseSum += sum(mse)

        mseList.append(mseSum/nchoice)

        datafile.write(f"\nvar + twice noise: {round(varSum/nchoice, 4):>11}")
        datafile.write(f"\nmse: {round(mseSum/nchoice, 4):>26}")

        # COMPARISON / CONSOLIDATION OF THEORETICAL RESULTS

        # EXTENSION TO Q, I^2 AND CONFIDENCE INTERVALS

        casetime = time.perf_counter() - loopTime
        datafile.write(f"\ncalibration: {round(casetime, 2):>15} seconds\n")

    # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM ANALYTIC GAUSSIAN MECHANISM
    computeMSE(xTrainChoice, dchoice, sigma, nchoice, mseSum)
    print("Computing MSE...")

    # COMPUTE SIGMA USING CLASSIC GAUSSIAN MECHANISM FOR COMPARISON BETWEEN DISPERSION AND MSE OF BOTH
    if dataIndex == 0 or dataIndex == 1:
        classicSigma = (GS[0]*sqrt(2*log(1.25/dtachoice)))/epschoice
    elif dataIndex == 2:
        classicSigma = (GS[1]*sqrt(2*log(1.25/dtachoice)))/epschoice
    else:
        classicSigma = (GS[2]*sqrt(2*log(1.25/dtachoice)))/epschoice
    
    datafile.write("\nStatistics from classic GM and computation of MSE")
    datafile.write(f"\n\nsigma from classic GM: {round(classicSigma, 4)}")
    datafile.write(f"\nsquare: {round(classicSigma**2, 8):>22}")

    # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM CLASSIC GAUSSIAN MECHANISM
    computeMSE(xTrainChoice, dchoice, classicSigma, nchoice, mseSum)

    # EXPERIMENT 2: AGM VS CGM
    mseA = mseList.pop(0)
    mseB = mseList.pop(0)
    msediff = abs(mseB - mseA)
    print(msediff)
    decdiff = abs(msediff/mseB)
    print(decdiff)
    percdiff = decdiff*100
    print(percdiff)
    datafile.write("\nPercentages comparing AGM and classic GM")
    datafile.write(f"\n\ndifference between mse: {round(percdiff, 8)}%")

    # EXTENSION TO Q, I^2 AND CONFIDENCE INTERVALS

def runLoopVaryEps(dataIndex):
    for eps in epsset:
        print(f"\nProcessing the main loop for the value eps = {eps}.")
        runLoop(dataIndex, xTrainNew, 0, eps, eps, dtaconst, dconst, nconst)

def runLoopVaryDta(dataIndex):
    for dta in dtaset:
        print(f"\nProcessing the main loop for the value dta = {dta}.")
        runLoop(dataIndex, xTrainNew, 1, dta, epsconst, dta, dconst, nconst)

def runLoopVaryD(dataIndex):
    for d in dset:
        print(f"\nProcessing the main loop for the value d = {d}.")
        runLoop(dataIndex, xTrainNew, 2, d, epsconst, dtaconst, d, nconst)

def runLoopVaryN(dataIndex):
    for n in nset:
        print(f"\nSimple case for the value n = {n}.")
        runLoop(dataIndex, xTrainNew, 3, n, epsconst, dtaconst, dconst, n)

def simpleVaryEps(dataIndex):
    for eps in epsset:
        print(f"\nSimple case for the value eps = {eps}.")
        runLoop(dataIndex, xTrainSimple, 0, eps, eps, dtaconst, dconst, nconst)

def simpleVaryDta(dataIndex):
    for dta in dtaset:
        print(f"\nSimple case for the value dta = {dta}.")
        runLoop(dataIndex, xTrainSimple, 1, dta, epsconst, dta, dconst, nconst)

def simpleVaryD(dataIndex):
    for d in dset:
        print(f"\nSimple case for the value d = {d}.")
        runLoop(dataIndex, xTrainSimple, 2, d, epsconst, dtaconst, d, nconst)

def simpleVaryN(dataIndex):
    for n in nset:
        print(f"\nProcessing the main loop for the value n = {n}.")
        runLoop(dataIndex, xTrainSimple, 3, n, epsconst, dtaconst, dconst, n)

# EXPERIMENT 1: BEHAVIOUR OF VARIABLES AT DIFFERENT SETTINGS
for i in range(4):
    runLoopVaryEps(i)
    runLoopVaryDta(i)
    runLoopVaryD(i)
    runLoopVaryN(i)

# COMPARISON WITH SIMPLE DATASETS WITH ALL VALUES EQUAL TO 0.5
for i in range(4):
    simpleVaryEps(i)
    simpleVaryDta(i)
    simpleVaryD(i)
    simpleVaryN(i)

# EXPERIMENT 3: WHAT IS THE COST OF PRIVACY?

# BASELINE SETTING FOR BASIC AND ADVANCED APPLICATIONS
# BIG OR SMALL COST? DRAW A CONCLUSION

# EXPERIMENT 4: WHAT IS THE COST OF A FEDERATED SETTING?

# COMPARE TO CENTRALISED SETTING AND CONCLUDE IN SOME / ALL APPLICATIONS AS IN 3
# NOT SURE HOW TO DO THIS YET, LOOK AT FEDACS / FEDPROX ETC.

# EXPERIMENT 5: VECTOR ALLOCATIONS TESTING ROBUSTNESS OF FEDERATED CASE

# DIFFERENT LEVELS OF HETEROGENEITY
# DO THESE LEVELS AFFECT THE METHOD? HYPOTHESIS: NOT MUCH
# MEANS NOTHING IN CENTRALISED CASE BECAUSE SERVER HAS ALL DATA

print("Finished.\n")