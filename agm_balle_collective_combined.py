import time
import numpy as np
import idx2numpy
import os
import mpmath as mp

from math import erf
from numpy.random import normal
from PIL import Image
from numpy import asarray

# INITIALISING START TIME AND SEED FOR RANDOM SAMPLING
startTime = time.perf_counter()
print("\nStarting...")
np.random.seed(3820672)

# ARRAYS STORING SETS OF VALUES OF EACH VARIABLE WITH OPTIMA CHOSEN AS CONSTANTS
epsset = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
epsconst = float(epsset[1])

# VECTOR DIMENSION CHOSEN TO MATCH THAT OF CONVERTED IMAGES ABOVE AND NUMBER OF CLIENTS CHOSEN TO GIVE SENSIBLE GS
dtaset = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])
dtaconst = float(dtaset[1])

dsetCifar = np.array([512, 768, 1024, 1280, 1536, 1875, 2048, 2400, 2560, 3072])
dsetFashion = np.array([392, 525, 588, 600, 625, 640, 672, 700, 735, 784])
dsetFlair = np.array([768, 1536, 3072, 4800, 6144, 7680, 8192, 9375, 10240, 12288], dtype = np.int64)

# dset = np.array([dsetCifar, dsetFashion, dsetFlair], dtype=object)
dconstCifar = maxDimCifar = int(dsetCifar[9])
dconstFashion = maxDimFashion = int(dsetFashion[9])
dconstFlair = maxDimFlair = int(dsetFlair[9])

nsetCifar = np.array([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000])
nsetFashion = np.array([15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000])
nsetFlair = np.array([25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000], dtype = np.int64)

# nset = np.array([nsetCifar, nsetFashion, nsetFlair], dtype=object)
nconstCifar = int(nsetCifar[8])
nconstFashion = int(nsetFashion[8])
nconstFlair = int(nsetFlair[8])

maxNumCifar = int(nsetCifar[9])
maxNumFashion = int(nsetFashion[9])
maxNumFlair = int(nsetFlair[9])

# pairsArr = [(dconst[0], nconst[0]), (dconst[1], nconst[1]), (dconst[2], nconst[2])]
# GS = [float(sqrt(d))/n for d, n in pairsArr]

GSCifar = float(mp.sqrt(dconstCifar))/nconstCifar
GSFashion = float(mp.sqrt(dconstFashion))/nconstFashion
GSFlair = float(mp.sqrt(dconstFlair))/nconstFlair

# maxPairsArr = [(dconst[0], maxNum[0]), (dconst[1], maxNum[1]), (dconst[2], maxNum[2])]
# maxArraySize = [d*n for d, n in maxPairsArr]

maxArraySizeCifar = dconstCifar*maxNumCifar
maxArraySizeFashion = dconstFashion*maxNumFashion
maxArraySizeFlair = dconstFlair*maxNumFlair

# INITIALISING OTHER PARAMETERS/CONSTANTS
parset = np.array(['eps', 'dta', 'd', 'n'])
rset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
R = len(rset)

# IN THEORY TWO NOISE TERMS ARE ADDED WITH EACH USING EPS AND DTA HALF THE SIZE OF IN EXPERIMENTS
epsTheory = epsconst/2
dtaTheory = dtaconst/2

# xiTheory = [(2*d*log(1.25/dtaTheory))/((n**2)*(epsTheory**2)) for d, n in pairsArr]
# xiTheoryCifar = (2*dconstCifar*log(1.25/dtaTheory))/((nconstCifar**2)*(epsTheory**2))
# xiTheoryFashion = (2*dconstFashion*log(1.25/dtaTheory))/((nconstFashion**2)*(epsTheory**2))
# xiTheoryFlair = (2*dconstFlair*log(1.25/dtaTheory))/((nconstFlair**2)*(epsTheory**2))

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

    # names = np.array(['id,', 'data'])
    # formats = np.array(['f8,', 'f8'])
    # dtype = dict(names = names, formats = formats)
    # dtype = {'id,': 'f8,', 'data': 'f8'}
    # xTrainCifar100 = np.array(list(xData.items()), dtype = dtype)

    xTrainCifar100 = xData
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
    xTrainFlair = np.zeros((maxNumFlair, maxDimFlair))
    count = 0

    from alive_progress import alive_bar
    for file in os.listdir():
        with alive_bar(maxNumFlair) as bar:
            while count < maxNumFlair:
                img = Image.open(file)
                dict = asarray(img)
                vector = dict.reshape((1, maxDimFlair))
                # np.append(xTrainFlair, vector, axis=0)
                xTrainFlair[count] = vector
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

def runLoop(dataIndex, index, var, dchoice, nchoice, epschoice, dtachoice, xTrainNew, GS, maxArraySize):

    if dataIndex == 0:
        datafile = open("cifar10_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    elif dataIndex == 1:
        datafile = open("cifar100_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    elif dataIndex == 2:
        datafile = open("fashion_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    else:
        datafile = open("flair_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")

    datafile.write("Statistics from Binary Search in AGM")
    # datafile.write(f"\n\nxiTheory: {round(xiTheory, 7):>21}")

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
            return 0.5*(1.0 + erf(float(t)/mp.sqrt(2.0)))

        # VALUE V STAR IS LARGEST SUCH THAT THIS EXPRESSION IS LESS THAN OR EQUAL TO DTA
        def caseA(eps, u):
            return Phi(mp.sqrt(eps*u)) - mp.exp(eps)*Phi(-mp.sqrt(eps*(u+2.0)))

        # VALUE U STAR IS SMALLEST SUCH THAT THIS EXPRESSION IS LESS THAN OR EQUAL TO DTA
        def caseB(eps, u):
            return Phi(-mp.sqrt(eps*u)) - mp.exp(eps)*Phi(-mp.sqrt(eps*(u+2.0)))

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
                functionAlpha = lambda u : mp.sqrt(1.0 + u/2.0) - mp.sqrt(u/2.0)

            else:
                predicateStopDT = lambda u : caseB(eps, u) <= dta
                functionDta = lambda u : caseB(eps, u)
                predicateLeftBS = lambda u : functionDta(u) < dta
                functionAlpha = lambda u : mp.sqrt(1.0 + u/2.0) + mp.sqrt(u/2.0)

            predicateStopBS = lambda u : abs(functionDta(u) - dta) <= tol

            uInf, uSup = doublingTrick(predicateStopDT, 0.0, 1.0)
            uFinal = binarySearch(predicateStopBS, predicateLeftBS, uInf, uSup)
            alpha = functionAlpha(uFinal)

        casetime = time.perf_counter() - loopTime
        datafile.write(f"\ncalibration: {round(casetime, 6):>18} seconds\n")

        sigma = alpha*GS/mp.sqrt(2.0*eps)
        return sigma

    # CALL ALGORITHM FOR AGM TO FIND SIGMA GIVEN EPS AND DTA AS INPUT
    sigma = calibrateAGM(epschoice, dtachoice, GS, tol=1.e-12)
    print("Calibrating AGM...")
    datafile.write("\nStatistics from AGM and computation of MSE")
    datafile.write(f"\n\nsigma from AGM: {round(sigma, 6):>13}")
    datafile.write(f"\nsquare: {round(sigma**2, 10):>27}")
        
    compareEListA = np.zeros(nchoice)
    compareQEListA = np.zeros(nchoice)
    compareEListB = np.zeros(nchoice)
    compareQEListB = np.zeros(nchoice)
    compareTListA = np.zeros(nchoice)
    compareQTListA = np.zeros(nchoice)
    compareTListB = np.zeros(nchoice)
    compareQTListB = np.zeros(nchoice)

    # FUNCTION BY SCOTT BASED ON OWN LEMMAS THEOREMS AND COROLLARIES IN PAPER
    def computeMSE(xTrainNew, sigma, index):

        loopTime = time.perf_counter()

        if index == 2:
            xTrainCrop = xTrainNew.reshape((int(maxArraySize/dchoice), dchoice))
        else:
            xTrainCrop = xTrainNew.reshape((int(maxArraySize/dchoice), dchoice))
        xTrainNew = xTrainCrop
        
        # INITIAL COMPUTATION OF WEIGHTED MEAN FOR Q BASED ON VECTOR VARIANCE
        wVector = np.var(xTrainNew, axis=1)
        datafile.write(f"\nwithin-vector: {str(round((np.sum(wVector))/nchoice, 6)):>16}")

        weight = np.zeros(nchoice)
        for j in range(0, nchoice):
            weight[j] = 1.0/(wVector[j])

        # MULTIPLYING EACH VECTOR BY ITS CORRESPONDING WEIGHTED MEAN
        wxTrainNew = np.zeros((nchoice, dchoice))
        for j in range(0, nchoice):
            wxTrainNew[j] = (weight[j])*(xTrainNew[j])

        mu = np.mean(xTrainNew, axis=0)
        wMu = np.mean(wxTrainNew, axis=0)
        datafile.write(f"\n\nmu: {str(round((np.sum(mu))/dchoice, 6)):>27}")
        datafile.write(f"\nweighted mu: {str(round((np.sum(wMu))/dchoice, 4)):>16}")
        muSquares = np.power(mu, 2)
        wMuSquares = np.power(wMu, 2)
        datafile.write(f"\nsum of squares: {str(round((np.sum(muSquares))/dchoice, 5)):>14}")   
        datafile.write(f"\nsum of w squares: {str(round((np.sum(wMuSquares))/dchoice, 2)):>8}")

        noisyMu = np.zeros(dchoice)
        wNoisyMu = np.zeros(dchoice)
        xiSum1 = 0
        xiSum2 = 0

        mseEList = np.zeros(nchoice)
        trueEList = np.zeros(nchoice, dtype = np.float64)
        mseQEList = np.zeros(nchoice)
        trueQEList = np.zeros(nchoice, dtype = np.float64)
        mseTList = np.zeros(nchoice, dtype = np.float64)
        mseQTList = np.zeros(nchoice, dtype = np.float64)

        # ADDING FIRST NOISE TERM TO MU DERIVED FROM GAUSSIAN DISTRIBUTION WITH MEAN 0 AND VARIANCE SIGMA SQUARED
        for i in range(0, dchoice):
            xi1 = normal(0, sigma**2)
            noisyMu[i] = mu[i] + xi1
            wNoisyMu[i] = wMu[i] + xi1
            xiSum1 += xi1
        datafile.write(f"\n\nnoise 1: {round(xiSum1/dchoice, 8):>21}")

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

            mseEList[j] = np.sum(noisyDisp)
            trueEList[j] = np.sum(trueDisp)
            mseQEList[j] = np.sum(noisyQ)
            trueQEList[j] = np.sum(weightedTrueDisp)

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
            mseTheoreticalSquared = np.power(mseTheoretical, 2)
            mseQTheoreticalSquared = np.power(mseQTheoretical, 2)
            mseTList[j] = np.sum(mseTheoreticalSquared)
            mseQTList[j] = np.sum(mseQTheoreticalSquared)

        if index == 0:
            np.copyto(compareEListA, mseEList)
            np.copyto(compareQEListA, mseQEList)
            np.copyto(compareTListA, mseTList)
            np.copyto(compareQTListA, mseQTList)
        else:
            np.copyto(compareEListB, mseEList)
            np.copyto(compareQEListB, mseQEList)
            np.copyto(compareTListB, mseTList)
            np.copyto(compareQTListB, mseQTList)

        datafile.write(f"\ntrue dispersion: {round((np.sum(trueEList))/(nchoice*dchoice), 8):>16}")
        datafile.write(f"\ntrue q: {round((np.sum(trueQEList))/(nchoice*dchoice), 8):>25}")
        datafile.write(f"\nnoise 2: {round(xiSum2/nchoice, 8):>22}")

        # EMPIRICAL MSE = THE ABOVE UNROUNDED STATISTIC MINUS THE TRUE DISPERSION
        diffELists = np.subtract(mseEList, trueEList)
        diffQELists = np.subtract(mseQEList, trueQEList)
        squaredDiffELists = np.power(diffELists, 2)
        squaredDiffQELists = np.power(diffQELists, 2)
        mseEmpirical = np.sqrt(squaredDiffELists)
        mseQEmpirical = np.sqrt(squaredDiffQELists)
        datafile.write(f"\nempirical mse: {round((np.sum(mseEmpirical))/(nchoice*dchoice), 8):>18}")
        datafile.write(f"\ntheoretical mse: {round((np.sum(mseTList))/(nchoice*dchoice), 10):>14}")
        datafile.write(f"\nempirical q: {round(np.sum((mseQEmpirical))/(nchoice*dchoice), 8):>20}")
        datafile.write(f"\ntheoretical q: {round((np.sum(mseQTList))/(nchoice*dchoice), 6):>16}")

        # COMPUTE I^2'' and I^2 USING SIMPLE FORMULA AT BOTTOM OF LEMMA 6.2
        iSquaredPrep = np.divide(nchoice-1, mseQEList)
        trueISquaredPrep = np.divide(nchoice-1, trueQEList)
        iSquared = np.subtract(1, iSquaredPrep)
        trueISquared = np.subtract(1, trueISquaredPrep)
        datafile.write(f"\nisquared: {round(np.sum(iSquared), 10):>19}")
        datafile.write(f"\ntrue isquared: {round(np.sum(trueISquared), 10):>14}")

        # ADD THIRD NOISE TERM BASED ON LEMMA 6.2
        xi3 = normal(0, sigma**2)
        noisyISquared = iSquared + xi3
        datafile.write(f"\nnoise 3: {round(xi3, 8):>24}")

        diffEISquared = np.subtract(noisyISquared, trueISquared)
        squaredDEIS = np.power(diffEISquared, 2)
        mseEISquared = np.sqrt(squaredDEIS)

        # AVOID DIVIDE BY ZERO ERRORS
        # with np.errstate(divide='ignore'):

        mseTISquaredPrep = np.divide(nchoice-1, mseQTList)
        mseTISquared = np.subtract(1, mseTISquaredPrep)
        mseTISquaredSquared = np.power(mseTISquared, 2)
        datafile.write(f"\nempirical isquared: {round(np.sum(mseEISquared), 4):>9}")
        datafile.write(f"\ntheoretical isquared: {round((np.sum(mseTISquaredSquared))/(nchoice*dchoice), 4):>7}")

        # COMPARISON / CONSOLIDATION OF THEORETICAL RESULTS IF GRAPHS NOT ADEQUATE

        # 95% CONFIDENCE INTERVALS USING SIGMA, Z-SCORE AND WEIGHTS IF RELEVANT
        confInt = (7.84*(mp.sqrt(6))*(sigma**2))/(mp.sqrt(nchoice))
        qConfInt = np.sum((7.84*weight*(mp.sqrt(6))*(sigma**2))/(mp.sqrt(nchoice)))        
        iSquaredConfInt = np.sum((7.84*(mp.sqrt(2*(nchoice-1))))/(3*(np.sqrt(35*weight*nchoice))*(sigma**2)))
        datafile.write(f"\n95% CI for dispersion: \u00B1 {round(confInt, 8)}")
        datafile.write(f"\n95% CI for q: \u00B1 {round(qConfInt, 4):>15}")
        datafile.write(f"\n95% CI for isquared: \u00B1 {round(iSquaredConfInt)}")

        casetime = time.perf_counter() - loopTime
        datafile.write(f"\n\ncalibration: {round(casetime, 2):>14} seconds\n")

    # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM ANALYTIC GAUSSIAN MECHANISM
    computeMSE(xTrainNew, sigma, 0)
    print("Computing empirical and theoretical MSEs...")

    # COMPUTE SIGMA USING CLASSIC GAUSSIAN MECHANISM FOR COMPARISON BETWEEN DISPERSION AND MSE OF BOTH
    classicSigma = (GS*mp.sqrt(2*mp.log(1.25/dtachoice)))/epschoice
    datafile.write("\nStatistics from classic GM and computation of MSE")
    datafile.write(f"\n\nsigma from classic GM: {round(classicSigma, 6):>8}")
    datafile.write(f"\nsquare: {round(classicSigma**2, 8):>27}")

    # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM CLASSIC GAUSSIAN MECHANISM
    computeMSE(xTrainNew, classicSigma, 1)

    # EXPERIMENT 2: AGM VS CGM
    datafile.write("\nPercentages comparing AGM and classic GM")
    comparelists1 = np.subtract(compareEListA, compareEListB)
    compareqlists1 = np.subtract(compareQEListA, compareQEListB)
    sumdiff1 = np.sum(comparelists1)
    sumqdiff1 = np.sum(compareqlists1)
    datafile.write(f"\n\nempirical mse comparison: {round(sumdiff1, 4):>6}x")
    datafile.write(f"\nempirical q comparison: {round(sumqdiff1, 2):>8}x")
    comparelists2 = np.subtract(compareTListA, compareTListB)
    compareqlists2 = np.subtract(compareQTListA, compareQTListB)
    sumdiff2 = np.sum(comparelists2)
    sumqdiff2 = np.sum(compareqlists2)
    datafile.write(f"\ntheoretical mse comparison: {round(sumdiff2, 4):>4}x")
    datafile.write(f"\ntheoretical q comparison: {round(sumqdiff2, 2):>6}x")

    # COMPUTE SIMILAR LISTS IN COMPUTEMSE FOR I^2 (USING FORMULA IN TERMS OF Q) THEN WRITE COMPARISONS HERE

def runLoopVaryEps(dataIndex, index, varset, dconst, nconst, xTrainNew, GS, maxArraySize):

    for eps in varset:
        print(f"\nProcessing dataset {dataIndex+1} for the value eps = {eps}.")
        runLoop(dataIndex, index, eps, dconst, nconst, eps, dtaconst, xTrainNew, GS, maxArraySize)

def runLoopVaryDta(dataIndex, index, varset, dconst, nconst, xTrainNew, GS, maxArraySize):

    for dta in varset:
        print(f"\nProcessing dataset {dataIndex+1} for the value dta = {dta}.")
        runLoop(dataIndex, index, dta, dconst, nconst, epsconst, dta, xTrainNew, GS, maxArraySize)

def runLoopVaryD(dataIndex, index, varset, dset, nconst, xTrainNew, GS, maxArraySize):

    for d in varset:
        print(f"\nProcessing dataset {dataIndex+1} for the value d = {d}.")
        runLoop(dataIndex, index, d, d, nconst, epsconst, dtaconst, xTrainNew, GS, maxArraySize)

def runLoopVaryN(dataIndex, index, varset, dconst, nset, xTrainNew, GS, maxArraySize):

    for n in varset:
        print(f"\nProcessing dataset {dataIndex+1} for the value n = {n}.")
        runLoop(dataIndex, index, n, dconst, n, epsconst, dtaconst, xTrainNew, GS, maxArraySize)

# EXPERIMENT 1: BEHAVIOUR OF VARIABLES AT DIFFERENT SETTINGS
runLoopVaryEps(0, 0, epsset, dconstCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar)
runLoopVaryDta(0, 1, dtaset, dconstCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar)
runLoopVaryD(0, 2, dsetCifar, dsetCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar)
runLoopVaryN(0, 3, nsetCifar, dconstCifar, nsetCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar)

runLoopVaryEps(1, 0, epsset, dconstCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar)
runLoopVaryDta(1, 1, dtaset, dconstCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar)
runLoopVaryD(1, 2, dsetCifar, dsetCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar)
runLoopVaryN(1, 3, nsetCifar, dconstCifar, nsetCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar)

runLoopVaryEps(2, 0, epsset, dconstFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion)
runLoopVaryDta(2, 1, dtaset, dconstFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion)
runLoopVaryD(2, 2, dsetFashion, dsetFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion)
runLoopVaryN(2, 3, nsetFashion, dconstFashion, nsetFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion)

runLoopVaryEps(3, 0, epsset, dconstFlair, nconstFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair)
runLoopVaryDta(3, 1, dtaset, dconstFlair, nconstFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair)
runLoopVaryD(3, 2, dsetFlair, dsetFlair, nconstFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair)
runLoopVaryN(3, 3, nsetFlair, dconstFlair, nsetFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair)

# EXPERIMENT 3: WHAT IS THE COST OF PRIVACY?

# WHAT IS THE ABSOLUTE DIFFERENCE AND MSE BETWEEN TRUE AND NOISY DISPERSION?
# BIG OR SMALL COST COMPARED TO SIZE OF DISPERSION? DRAW A CONCLUSION
# EXTEND TO Q, I SQUARED AND CONFIDENCE INTERVALS

# EXPERIMENT 4: WHAT IS THE COST OF A DISTRIBUTED SETTING?

# HYPOTHESIS: (CLOSE TO) ZERO COST, TO MATCH THEORY
# COMPARE TO CENTRALISED SETTING AND CONCLUDE IN SOME / ALL APPLICATIONS AS IN 3
# TRY ADDING PRIVACY TO BOTH, THEN REPEAT COST OF PRIVACY EXPERIMENTS

# EXPERIMENT 5: VECTOR ALLOCATIONS TESTING ROBUSTNESS OF DISTRIBUTED CASE

# USE IDEAS FROM SPLITTING EMNIST DATASET BY DIGIT (OR FEMNIST BY WRITER) AND MEASURING PIXEL FUNCTION
# DIFFERENT LEVELS OF HETEROGENEITY: USE DIFFERENT SIZED ARRAYS OF CLIENTS
# DO THESE LEVELS AFFECT THE METHOD? HYPOTHESIS: NOT MUCH
# MEANS NOTHING IN CENTRALISED CASE BECAUSE SERVER HAS ALL DATA

print("Finished.\n")