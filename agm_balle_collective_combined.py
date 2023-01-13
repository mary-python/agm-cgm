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
epsset = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
epsconst = epsset[1]

# VECTOR DIMENSION CHOSEN TO MATCH THAT OF CONVERTED IMAGES ABOVE AND NUMBER OF CLIENTS CHOSEN TO GIVE SENSIBLE GS
dtaset = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])
dtaconst = dtaset[1]

dsetCifar = np.array([128, 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072])
dsetFashion = np.array([147, 196, 245, 294, 392, 448, 490, 588, 672, 784])
dsetFlair = np.array([768, 1536, 3072, 4800, 6144, 7680, 8192, 9375, 10240, 12288])

# dset = np.array([dsetCifar, dsetFashion, dsetFlair], dtype=object)
dconstCifar = maxDimCifar = dsetCifar[9]
dconstFashion = maxDimFashion = dsetFashion[9]
dconstFlair = maxDimFlair = dsetFashion[9]

nsetCifar = np.array([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000])
nsetFashion = np.array([15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000])
nsetFlair = np.array([25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000])

# nset = np.array([nsetCifar, nsetFashion, nsetFlair], dtype=object)
nconstCifar = nsetCifar[8]
nconstFashion = nsetFashion[8]
nconstFlair = nsetFlair[8]

maxNumCifar = nsetCifar[9]
maxNumFashion = nsetFashion[9]
maxNumFlair = nsetFlair[9]

# pairsArr = [(dconst[0], nconst[0]), (dconst[1], nconst[1]), (dconst[2], nconst[2])]
# GS = [float(sqrt(d))/n for d, n in pairsArr]

GSCifar = float(sqrt(dconstCifar))/nconstCifar
GSFashion = float(sqrt(dconstFashion))/nconstFashion
GSFlair = float(sqrt(dconstFlair))/nconstFlair

# maxPairsArr = [(dconst[0], maxNum[0]), (dconst[1], maxNum[1]), (dconst[2], maxNum[2])]
# maxArraySize = [d*n for d, n in maxPairsArr]

maxArraySizeCifar = dconstCifar*nconstCifar
maxArraySizeFashion = dconstFashion*nconstFashion
maxArraySizeFlair = dconstFlair*nconstFlair

# INITIALISING OTHER PARAMETERS/CONSTANTS
parset = np.array(['eps', 'dta', 'd', 'n'])
rset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
R = len(rset)
mseEList = np.zeros(nconstFlair)
mseTList = np.zeros(nconstFlair)

# IN THEORY TWO NOISE TERMS ARE ADDED WITH EACH USING EPS AND DTA HALF THE SIZE OF IN EXPERIMENTS
epsTheory = epsconst/2
dtaTheory = dtaconst/2

# xiTheory = [(2*d*log(1.25/dtaTheory))/((n**2)*(epsTheory**2)) for d, n in pairsArr]
xiTheoryCifar = (2*dconstCifar*log(1.25/dtaTheory))/((nconstCifar**2)*(epsTheory**2))
xiTheoryFashion = (2*dconstFashion*log(1.25/dtaTheory))/((nconstFashion**2)*(epsTheory**2))
xiTheoryFlair = (2*dconstFlair*log(1.25/dtaTheory))/((nconstFlair**2)*(epsTheory**2))

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
            xTrainCifar10 = xData
        else:
            xTrainCifar10 = np.concatenate((xTrainCifar10, xData), axis=0)
    return xTrainCifar10

# LOADING AND SPLITTING CIFAR-100 DATA
def loadCifar100():
    filename = 'train'
    dict = unpickle(filename)
    xData = dict[b'data']

    names = np.array(['id,', 'data'])
    formats = np.array(['f8,', 'f8'])
    dtype = dict(names = names, formats = formats)
    xTrainCifar100 = np.array(list(xData.items()), dtype = dtype)
    return xTrainCifar100

# LOADING FASHION-MNIST DATA
def loadFashion():
    filename = 'train-images-idx3-ubyte'
    dict = idx2numpy.convert_from_file(filename)
    xTrainFashion = dict.reshape((maxNumFashion, maxDimFashion))
    return xTrainFashion

# LOADING ML-FLAIR DATA BY GEEKSFORGEEKS
def loadFlair():
    path = 'small_images'
    os.chdir(path)
    xTrainFlair = np.zeros(maxNumFlair, maxDimFlair)
    count = 0

    from alive_progress import alive_bar
    for file in os.listdir():
        with alive_bar(maxNumFlair) as bar:
            while count < maxNumFlair:
                img = Image.open(file)
                dict = asarray(img)
                vector = dict.reshape((1, maxDimFlair))
                np.append(xTrainFlair, vector, axis=0)
                count += 1
                bar()
        break

    return xTrainFlair

# ADAPTATION OF TRANSFORMATION OF LABEL INDICES TO ONE-HOT ENCODED VECTORS AND IMAGES TO 3072-DIMENSIONAL VECTORS BY HADHAZI
def transformValues(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

# CALL ALL THE ABOVE METHODS
print("Loading data...\n")
xTrainCifar10 = loadCifar10()
xTrainCifar100 = loadCifar100()
xTrainFashion = loadFashion()
xTrainFlair = loadFlair()

# xTrain = np.array([xTrainCifar10, xTrainCifar100, xTrainFashion, xTrainFlair], dtype=object)
xTrainNewCifar10 = transformValues(xTrainCifar10)
xTrainNewCifar100 = transformValues(xTrainCifar100)
xTrainNewFashion = transformValues(xTrainFashion)
xTrainNewFlair = transformValues(xTrainFlair)

os.chdir('..')

def runLoop(dataIndex, index, var, dchoice, nchoice, epschoice, dtachoice, xTrainNew, GS, maxArraySize, xiTheory):

    if dataIndex == 0:
        datafile = open("cifar10_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    elif dataIndex == 1:
        datafile = open("cifar100_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    elif dataIndex == 2:
        datafile = open("fashion_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    else:
        datafile = open("flair_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")

    datafile.write("Statistics from Theory and Binary Search in AGM")
    datafile.write(f"\n\nxiTheory: {round(xiTheory, 7):>21}")

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

        sigma = alpha*GS/sqrt(2.0*eps)
        return sigma

    # CALL ALGORITHM FOR AGM TO FIND SIGMA GIVEN EPS AND DTA AS INPUT
    sigma = calibrateAGM(epschoice, dtachoice, GS, tol=1.e-12)
    print("Calibrating AGM...")
    datafile.write("\nStatistics from AGM and computation of MSE")
    datafile.write(f"\n\nsigma from AGM: {round(sigma, 6):>15}")
    datafile.write(f"\nsquare: {round(sigma**2, 10):>27}")

    # FUNCTION BY SCOTT BASED ON OWN LEMMAS THEOREMS AND COROLLARIES IN PAPER
    def computeMSE(xTrainNew, sigma):

        loopTime = time.perf_counter()

        if index == 2:
            xTrainCrop = xTrainNew.reshape((int(maxArraySize/dchoice), dchoice))
        else:
            xTrainCrop = xTrainNew.reshape((int(maxArraySize/dchoice), dchoice))
        xTrainNew = xTrainCrop
        
        # INITIAL COMPUTATION OF WEIGHTED MEAN FOR Q BASED ON VECTOR VARIANCE
        wVector = np.var(xTrainNew, axis=1)
        datafile.write(f"\nwithin-vector: {str(round((sum(wVector))/nchoice, 8)):>18}")

        weight = np.zeros(dchoice)
        for j in range(0, nchoice):
            weight[j] = 1.0/(wVector[j])

        # MULTIPLYING EACH VECTOR BY ITS CORRESPONDING WEIGHTED MEAN
        wxTrainNew = np.zeros(nchoice, dchoice)
        for j in range(0, nchoice):
            wxTrainNew[j] = (weight[j])*(xTrainNew[j])

        print(wxTrainNew)
        mu = np.mean(xTrainNew, axis=0)
        wMu = np.mean(wxTrainNew, axis=0)
        print(mu)
        print(wMu)
        datafile.write(f"\n\nmu: {str(round((sum(mu))/dchoice, 8)):>29}")
        datafile.write(f"\nweighted mu: {str(round((np.sum(wMu))/dchoice, 8)):>20}")
        muSquares = np.power(mu, 2)
        wMuSquares = np.power(wMu, 2)
        datafile.write(f"\nsum of squares: {str(round((sum(muSquares))/dchoice, 5)):>14}")   
        datafile.write(f"\nsum of w squares: {str(round((np.sum(wMuSquares))/dchoice, 5)):>12}")

        noisyMu = np.zeros(dchoice)
        wNoisyMu = np.zeros(dchoice)
        xiSum1 = 0
        xiSum2 = 0

        mseEList = np.zeros(nchoice)
        trueEList = np.zeros(nchoice)
        mseQEList = np.zeros(nchoice)
        trueQEList = np.zeros(nchoice)
        mseTList = np.zeros(nchoice)
        mseQTList = np.zeros(nchoice)

        # ADDING FIRST NOISE TERM TO MU DERIVED FROM GAUSSIAN DISTRIBUTION WITH MEAN 0 AND VARIANCE SIGMA SQUARED
        for i in range(0, dchoice):
            xi1 = normal(0, sigma**2)
            noisyMu[i] = mu[i] + xi1
            wNoisyMu[i] = wMu[i] + xi1
            xiSum1 += xi1
        datafile.write(f"\n\nnoise 1: {round(xiSum1/dchoice, 8):>22}")

        # FIRST SUBTRACTION BETWEEN CIFAR-10 VECTOR OF EACH CLIENT AND NOISY MEAN ACCORDING TO THEOREM FOR DISPERSION
        for j in range(0, nchoice):
            trueDiff = np.subtract(xTrainNew[j], mu)
            wTrueDiff = np.subtract(xTrainNew[j], wMu)
            noisyDiff = np.subtract(xTrainNew[j], noisyMu)
            wNoisyDiff = np.subtract(xTrainNew[j], wNoisyMu)

            # INCORPORATING WEIGHTS FOR STATISTICS ON Q
            trueDisp = np.power(trueDiff, 2)
            wTrueDisp = np.power(wTrueDiff, 2)
            weightedTrueDisp = (weight[j])*(wTrueDisp)
            noisyVar = np.power(noisyDiff, 2)
            wNoisyVar = np.power(wNoisyDiff, 2)
            weightedNoisyVar = (weight[j])*(wNoisyVar)

            xi2 = normal(0, sigma**2)
            noisyDisp = noisyVar + xi2
            noisyQ = weightedNoisyVar + xi2
            xiSum2 += xi2

            mseEList[j] = sum(noisyDisp)
            trueEList[j] = sum(trueDisp)
            mseQEList[j] = sum(noisyQ)
            trueQEList[j] = sum(weightedTrueDisp)

            # ADDING SECOND NOISE TERM TO EXPRESSION OF DISPERSION AND COMPUTING THEORETICAL MSE USING VARIABLES DEFINED ABOVE
            doubleTrueDiff = 2*trueDiff
            wDoubleTrueDiff = 2*wTrueDiff
            bracket = np.subtract(xi1, doubleTrueDiff)
            wBracket = np.subtract(xi1, wDoubleTrueDiff)
            multiply = np.multiply(xi1, bracket)
            wMultiply = np.multiply(xi1, wBracket)
            weightedMult = (weight[j])*(wMultiply)

            mseTheoretical = np.add(multiply, xi2)
            mseQTheoretical = np.add(weightedMult, xi2)
            mseTList[j] = sum(mseTheoretical)
            mseQTList[j] = sum(mseQTheoretical)

        datafile.write(f"\ntrue dispersion: {round((sum(trueEList))/nchoice, 8):>16}")
        datafile.write(f"\ntrue q: {round((sum(trueEList))/nchoice, 8):>25}")
        datafile.write(f"\nnoise 2: {round(xiSum2/nchoice, 8):>18}")

        # EMPIRICAL MSE = THE ABOVE UNROUNDED STATISTIC MINUS THE TRUE DISPERSION
        diffELists = np.subtract(mseEList, trueEList)
        diffQELists = np.subtract(mseQEList, trueQEList)
        squaredDiffELists = np.power(diffELists, 2)
        squaredDiffQELists = np.power(diffQELists, 2)
        mseEmpirical = np.sqrt(squaredDiffELists)
        mseQEmpirical = np.sqrt(squaredDiffQELists)
        datafile.write(f"\nempirical mse: {round((sum(mseEmpirical))/nchoice, 10):>18}")
        datafile.write(f"\ntheoretical mse: {round((sum(mseTList))/nchoice, 10):>17}")
        datafile.write(f"\nempirical q: {round((sum(mseQEmpirical))/nchoice, 10):>20}")
        datafile.write(f"\ntheoretical q: {round((sum(mseQTList))/nchoice, 10):>19}")

        # COMPUTE I^2'' and I^2 USING SIMPLE FORMULA AT BOTTOM OF LEMMA 6.2
        iSquaredPrep = np.divide(nchoice-1, mseQEList)
        trueISquaredPrep = np.divide(nchoice-1, trueQEList)
        iSquared = np.subtract(1, iSquaredPrep)
        trueISquared = np.subtract(1, trueISquaredPrep)
        datafile.write(f"isquared: {round(sum(iSquared), 8):>17}")
        datafile.write(f"true isquared: {round(sum(trueISquared), 8):>12}")

        # ADD THIRD NOISE TERM BASED ON LEMMA 6.2
        xi3 = normal(0, sigma**2)
        noisyISquared = iSquared + xi3
        datafile.write(f"noise 3: {round(xi3, 8):>18}")

        diffEISquared = np.subtract(noisyISquared, trueISquared)
        squaredDEIS = np.power(diffEISquared, 2)
        mseEISquared = np.sqrt(squaredDEIS)
        mseTISquaredPrep = np.divide(nchoice-1, mseQTList)
        mseTISquared = np.subtract(1, mseTISquaredPrep)
        datafile.write(f"\nempirical isquared: {round(sum(mseEISquared), 10):>13}")
        datafile.write(f"\ntheoretical isquared: {round((sum(mseTISquared))/nchoice, 10):>12}")

        # COMPARISON / CONSOLIDATION OF THEORETICAL RESULTS IF GRAPHS NOT ADEQUATE

        # 95% CONFIDENCE INTERVALS USING SIGMA, Z-SCORE AND WEIGHTS IF RELEVANT
        confInt = (7.84*(sqrt(6))*(sigma**2))/(sqrt(nchoice))
        qConfInt = sum((7.84*weight*(sqrt(6))*(sigma**2))/(sqrt(nchoice)))
        iSquaredConfInt = sum((7.84*(sqrt(2*(nchoice-1))))/(3*(sqrt(35*weight*nchoice))*(sigma**2)))
        datafile.write(f"\n95% CI for dispersion: \u00B1 {confInt}")
        datafile.write(f"\n95% CI for q: \u00B1 {qConfInt}")
        datafile.write(f"\n95% CI for isquared: \u00B1 {iSquaredConfInt}")

        casetime = time.perf_counter() - loopTime
        datafile.write(f"\n\ncalibration: {round(casetime, 2):>14} seconds\n")

    # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM ANALYTIC GAUSSIAN MECHANISM
    computeMSE(xTrainNew, sigma)
    print("Computing empirical and theoretical MSEs...")

    # COMPUTE SIGMA USING CLASSIC GAUSSIAN MECHANISM FOR COMPARISON BETWEEN DISPERSION AND MSE OF BOTH
    classicSigma = (GS*sqrt(2*log(1.25/dtachoice)))/epschoice
    datafile.write("\nStatistics from classic GM and computation of MSE")
    datafile.write(f"\n\nsigma from classic GM: {round(classicSigma, 6):>8}")
    datafile.write(f"\nsquare: {round(classicSigma**2, 10):>28}")

    # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM CLASSIC GAUSSIAN MECHANISM
    computeMSE(xTrainNew, classicSigma)

    # EXPERIMENT 2: AGM VS CGM
    def agmVScgm(mseList):
        mseA = mseList.pop(0)
        mseB = mseList.pop(0)
        msediff = float(mseB)/float(mseA)
        return msediff
    
    datafile.write("\nPercentages comparing AGM and classic GM")
    msediff1 = agmVScgm(mseEList)
    datafile.write(f"\n\nempirical mse comparison: {round(msediff1, 8):>10}x")
    msediff2 = agmVScgm(mseTList)
    datafile.write(f"\ntheoretical mse comparison: {round(msediff2, 8):>4}x")

    # COLLECT SIMILAR LISTS IN COMPUTEMSE FOR Q AND I^2 THEN WRITE COMPARISONS HERE

def runLoopVaryEps(dataIndex, index, varset, dconst, nconst, xTrainNew, GS, maxArraySize, xiTheory):

    for eps in varset:
        print(f"\nProcessing dataset {dataIndex+1} for the value eps = {eps}.")
        runLoop(dataIndex, index, eps, eps, dtaconst, dconst, nconst, xTrainNew, GS, maxArraySize, xiTheory)

def runLoopVaryDta(dataIndex, index, varset, dconst, nconst, xTrainNew, GS, maxArraySize, xiTheory):

    for dta in varset:
        print(f"\nProcessing dataset {dataIndex+1} for the value dta = {dta}.")
        runLoop(dataIndex, index, dta, epsconst, dta, dconst, nconst, xTrainNew, GS, maxArraySize, xiTheory)

def runLoopVaryD(dataIndex, index, varset, dset, nconst, xTrainNew, GS, maxArraySize, xiTheory):

    for d in varset:
        print(f"\nProcessing dataset {dataIndex+1} for the value d = {d}.")
        runLoop(dataIndex, index, d, epsconst, dtaconst, d, nconst, xTrainNew, GS, maxArraySize, xiTheory)

def runLoopVaryN(dataIndex, index, varset, dconst, nset, xTrainNew, GS, maxArraySize, xiTheory):

    for n in varset:
        print(f"\nProcessing dataset {dataIndex+1} for the value n = {n}.")
        runLoop(dataIndex, index, n, epsconst, dtaconst, dconst, n, xTrainNew, GS, maxArraySize, xiTheory)

# EXPERIMENT 1: BEHAVIOUR OF VARIABLES AT DIFFERENT SETTINGS
runLoopVaryEps(0, 0, epsset, dconstCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar, xiTheoryCifar)
runLoopVaryDta(0, 1, dtaset, dconstCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar, xiTheoryCifar)
runLoopVaryD(0, 2, dsetCifar, dsetCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar, xiTheoryCifar)
runLoopVaryN(0, 3, nsetCifar, dconstCifar, nsetCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar, xiTheoryCifar)

runLoopVaryEps(1, 0, epsset, dconstCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar, xiTheoryCifar)
runLoopVaryDta(1, 0, dtaset, dconstCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar, xiTheoryCifar)
runLoopVaryD(1, 0, dsetCifar, dsetCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar, xiTheoryCifar)
runLoopVaryN(1, 0, dconstCifar, dconstCifar, nsetCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar, xiTheoryCifar)

runLoopVaryEps(2, 1, epsset, dconstFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion, xiTheoryFashion)
runLoopVaryDta(2, 1, dtaset, dconstFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion, xiTheoryFashion)
runLoopVaryD(2, 1, dsetFashion, dsetFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion, xiTheoryFashion)
runLoopVaryN(2, 1, dconstFashion, dconstFashion, nsetFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion, xiTheoryFashion)

runLoopVaryEps(3, 2, dconstFlair, dconstFlair, nsetFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair, xiTheoryFlair)
runLoopVaryDta(3, 2, dconstFlair, dconstFlair, nsetFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair, xiTheoryFlair)
runLoopVaryD(3, 2, dconstFlair, dconstFlair, nsetFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair, xiTheoryFlair)
runLoopVaryN(3, 2, dconstFlair, dconstFlair, nsetFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair, xiTheoryFlair)

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