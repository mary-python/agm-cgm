import numpy as np
import idx2numpy
import os
import mpmath as mp
import matplotlib.pyplot as plt

from math import erf
from numpy.random import normal

# INITIALISING SEED FOR RANDOM SAMPLING
print("\nStarting...")
np.random.seed(3820672)

# ARRAYS STORING SETS OF VALUES OF EACH VARIABLE WITH OPTIMA CHOSEN AS CONSTANTS
epsset = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
epsconst = float(epsset[0])

# VECTOR DIMENSION CHOSEN TO MATCH THAT OF CONVERTED IMAGES ABOVE AND NUMBER OF CLIENTS CHOSEN TO GIVE SENSIBLE GS
dtaset = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
dtaconst = float(dtaset[0])

dimCifar = 3072
dimFashion = 784

numCifar = 50000
numFashion = 60000

GSCifar = float(mp.sqrt(dimCifar))/numCifar
GSFashion = float(mp.sqrt(dimFashion))/numFashion

# INITIALISING OTHER PARAMETERS/CONSTANTS
dataset = np.array(['Cifar10', 'Cifar100', 'Fashion'])
parset = np.array(['eps', 'dta'])
graphset = np.array(['$\mathit{\u03b5}$', '$\mathit{\u03b4}$'])
freqset = np.array(['10 (equal)', '10 (unequal)', '5 (equal)', '5 (unequal)', '2 (equal)', '2 (unequal)'])
R = 10

# IN THEORY TWO NOISE TERMS ARE ADDED WITH EACH USING EPS AND DTA HALF THE SIZE OF IN EXPERIMENTS
epsTheory = epsconst/2
dtaTheory = dtaconst/2

# ADAPTATION OF UNPICKLING OF CIFAR-10 FILES BY KRIZHEVSKY
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# ADAPTATION OF LOADING AND SPLITTING FIVE FILES OF TRAINING DATA BY RHYTHM
def loadCifar10(type):
    for i in range(1, 6):
        dict = unpickle('data_batch_' + str(i))
        batch = dict[type]

        # CONCATENATE X AND Y DATA FROM ALL FILES INTO RELEVANT VARIABLE
        if i == 1:
            trainCifar10 = batch
        else:
            trainCifar10 = np.concatenate((trainCifar10, batch), axis=0)
    return trainCifar10

# LOADING AND SPLITTING CIFAR-100 DATA
def loadCifar100(property):
    dict = unpickle('train')
    trainCifar100 = dict[property]
    return trainCifar100

# LOADING FASHION-MNIST DATA
def loadFashion(filename):
    dataFashion = idx2numpy.convert_from_file(filename)
    return dataFashion

# ADAPTATION OF TRANSFORMATION OF LABEL INDICES TO ONE-HOT ENCODED VECTORS AND IMAGES TO 3072-DIMENSIONAL VECTORS BY HADHAZI
def transformValues(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

# CALL ALL THE ABOVE METHODS
print("Loading data...")
imagesCifar10 = loadCifar10(b'data')
labelsCifar10 = loadCifar10(b'labels')
imagesCifar100 = loadCifar100(b'data')
labelsCifar100 = loadCifar100(b'coarse_labels')
imagesFashion = loadFashion('train-images-idx3-ubyte')
labelsFashion = loadFashion('train-labels-idx1-ubyte')

newImagesCifar10 = transformValues(imagesCifar10)
newImagesCifar100 = transformValues(imagesCifar100)
newImagesFashion = transformValues(imagesFashion)

os.chdir('..')

def runLoop(dataIndex, index, freqIndex, varset, dim, num, eps, dta, newImages, labels, GS):

    # STORING STATISTICS REQUIRED FOR GRAPHS
    V = len(varset)
    mseDispEPlotVA = np.zeros(V)
    mseQEPlotVA = np.zeros(V)
    mseDispEPlotVC = np.zeros(V)
    mseQEPlotVC = np.zeros(V)
    mseDispTPlotVA = np.zeros(V)
    mseQTPlotVA = np.zeros(V)
    mseDispTPlotVC = np.zeros(V)
    mseQTPlotVC = np.zeros(V)
    mseISquaredEPlotVA = np.zeros(V)
    mseISquaredEPlotVC = np.zeros(V)
    mseISquaredTPlotVA = np.zeros(V)
    mseISquaredTPlotVC = np.zeros(V)
    mseCentralPlotV = np.zeros(V)
    acDispEPlotV = np.zeros(V)
    acDispTPlotV = np.zeros(V)
    acQEPlotV = np.zeros(V)
    acQTPlotV = np.zeros(V)
    acISquaredEPlotV = np.zeros(V)
    acISquaredTPlotV = np.zeros(V)

    mseDispEPlotVARange = np.zeros(V)
    mseQEPlotVARange = np.zeros(V)
    mseDispEPlotVCRange = np.zeros(V)
    mseQEPlotVCRange = np.zeros(V)
    mseDispTPlotVARange = np.zeros(V)
    mseQTPlotVARange = np.zeros(V)
    mseDispTPlotVCRange = np.zeros(V)
    mseQTPlotVCRange = np.zeros(V)
    mseISquaredEPlotVARange = np.zeros(V)
    mseISquaredEPlotVCRange = np.zeros(V)
    mseISquaredTPlotVARange = np.zeros(V)
    mseISquaredTPlotVCRange = np.zeros(V)
    mseCentralPlotVRange = np.zeros(V)
    acDispEPlotVRange = np.zeros(V)
    acDispTPlotVRange = np.zeros(V)
    acQEPlotVRange = np.zeros(V)
    acQTPlotVRange = np.zeros(V)
    acISquaredEPlotVRange = np.zeros(V)
    acISquaredTPlotVRange = np.zeros(V)

    F = len(freqset)
    mseDispEPlotFA = np.zeros(F)
    mseQEPlotFA = np.zeros(F)
    mseDispEPlotFC = np.zeros(F)
    mseQEPlotFC = np.zeros(F)
    mseDispTPlotFA = np.zeros(F)
    mseQTPlotFA = np.zeros(F)
    mseDispTPlotFC = np.zeros(F)
    mseQTPlotFC = np.zeros(F)
    mseISquaredEPlotFA = np.zeros(F)
    mseISquaredEPlotFC = np.zeros(F)
    mseISquaredTPlotFA = np.zeros(F)
    mseISquaredTPlotFC = np.zeros(F)
    mseCentralPlotF = np.zeros(F)
    acDispEPlotF = np.zeros(F)
    acDispTPlotF = np.zeros(F)
    acQEPlotF = np.zeros(F)
    acQTPlotF = np.zeros(F)
    acISquaredEPlotF = np.zeros(F)
    acISquaredTPlotF = np.zeros(F)

    mseDispEPlotFARange = np.zeros(F)
    mseQEPlotFARange = np.zeros(F)
    mseDispEPlotFCRange = np.zeros(F)
    mseQEPlotFCRange = np.zeros(F)
    mseDispTPlotFARange = np.zeros(F)
    mseQTPlotFARange = np.zeros(F)
    mseDispTPlotFCRange = np.zeros(F)
    mseQTPlotFCRange = np.zeros(F)
    mseISquaredEPlotFARange = np.zeros(F)
    mseISquaredEPlotFCRange = np.zeros(F)
    mseISquaredTPlotFARange = np.zeros(F)
    mseISquaredTPlotFCRange = np.zeros(F)
    mseCentralPlotFRange = np.zeros(F)
    acDispEPlotFRange = np.zeros(F)
    acDispTPlotFRange = np.zeros(F)
    acQEPlotFRange = np.zeros(F)
    acQTPlotFRange = np.zeros(F)
    acISquaredEPlotFRange = np.zeros(F)
    acISquaredTPlotFRange = np.zeros(F)

    for val in range(10):

        mseDispEPlotVATemp = np.zeros(R)
        mseQEPlotVATemp = np.zeros(R)
        mseDispEPlotVCTemp = np.zeros(R)
        mseQEPlotVCTemp = np.zeros(R)
        mseDispTPlotVATemp = np.zeros(R)
        mseQTPlotVATemp = np.zeros(R)
        mseDispTPlotVCTemp = np.zeros(R)
        mseQTPlotVCTemp = np.zeros(R)
        mseISquaredEPlotVATemp = np.zeros(R)
        mseISquaredEPlotVCTemp = np.zeros(R)
        mseISquaredTPlotVATemp = np.zeros(R)
        mseISquaredTPlotVCTemp = np.zeros(R)
        mseCentralPlotVTemp = np.zeros(R)
        acDispEPlotVTemp = np.zeros(R)
        acDispTPlotVTemp = np.zeros(R)
        acQEPlotVTemp = np.zeros(R)
        acQTPlotVTemp = np.zeros(R)
        acISquaredEPlotVTemp = np.zeros(R)
        acISquaredTPlotVTemp = np.zeros(R)

        mseDispEPlotFATemp = np.zeros(R)
        mseQEPlotFATemp = np.zeros(R)
        mseDispEPlotFCTemp = np.zeros(R)
        mseQEPlotFCTemp = np.zeros(R)
        mseDispTPlotFATemp = np.zeros(R)
        mseQTPlotFATemp = np.zeros(R)
        mseDispTPlotFCTemp = np.zeros(R)
        mseQTPlotFCTemp = np.zeros(R)
        mseISquaredEPlotFATemp = np.zeros(R)
        mseISquaredEPlotFCTemp = np.zeros(R)
        mseISquaredTPlotFATemp = np.zeros(R)
        mseISquaredTPlotFCTemp = np.zeros(R)
        mseCentralPlotFTemp = np.zeros(R)
        acDispEPlotFTemp = np.zeros(R)
        acDispTPlotFTemp = np.zeros(R)
        acQEPlotFTemp = np.zeros(R)
        acQTPlotFTemp = np.zeros(R)
        acISquaredEPlotFTemp = np.zeros(R)
        acISquaredTPlotFTemp = np.zeros(R)

        var = varset[val]
        print(f"Processing dataset {dataIndex+1} for the value {parset[index]} = {var}.")

        if eps == -1:
            eps = var
        elif dta == -1:
            dta = var

        def calibrateAGM(eps, dta, tol=1.e-12):
            """ Calibrate a Gaussian perturbation for DP using the AGM of [Balle and Wang, ICML'18]
            Arguments:
            eps : target epsilon (eps > 0)
            dta : target delta (0 < dta < 1)
            GS : upper bound on L2 global sensitivity (GS >= 0)
            tol : error tolerance for binary search (tol > 0)
            Output:
            sig : s.d. of Gaussian noise needed to achieve (eps,dta)-DP under global sensitivity GS
            """

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

            centralSigma = alpha/mp.sqrt(2.0*eps)
            return centralSigma

        # CALL ALGORITHM FOR AGM TO FIND SIGMA GIVEN EPS AND DTA AS INPUT
        centralSigma = calibrateAGM(eps, dta, tol=1.e-12)
        sigma = GS*centralSigma

        sample = 0.02
        sampleSize = int(num*sample)
        compareEListA = np.zeros(sampleSize)
        compareQEListA = np.zeros(sampleSize)
        compareISEListA = np.zeros(sampleSize)
        compareEListC = np.zeros(sampleSize)
        compareQEListC = np.zeros(sampleSize)
        compareISEListC = np.zeros(sampleSize)
        compareTListA = np.zeros(sampleSize)
        compareQTListA = np.zeros(sampleSize)
        compareISTListA = np.zeros(sampleSize)
        compareTListC = np.zeros(sampleSize)
        compareQTListC = np.zeros(sampleSize)
        compareISTListC = np.zeros(sampleSize)

        def computeMSE(ACindex, rep, imageArray, sigma, centralSigma):

            # INITIAL COMPUTATION OF WEIGHTED MEAN FOR Q BASED ON VECTOR VARIANCE
            wVector = np.var(imageArray, axis=1)
            weight = np.zeros(sampleSize)
            wImageArray = np.zeros((sampleSize, dim))       

            for j in range(0, sampleSize):
                wVectorSquared = np.power(wVector[j], 2)
                weight[j] = 1.0/(wVectorSquared)

                # MULTIPLYING EACH VECTOR BY ITS CORRESPONDING WEIGHTED MEAN
                wImageArray[j] = (weight[j])*(imageArray[j])

            mu = np.mean(imageArray, axis=0)
            wSumMu = np.sum(wImageArray, axis=0)

            # DIVIDING SUM OF WEIGHTED VECTORS BY SUM OF WEIGHTS
            sumWeight = np.sum(weight)
            wMu = (wSumMu)/sumWeight

            noisyMu = np.zeros(dim)
            wNoisyMu = np.zeros(dim)

            noisyEList = np.zeros(sampleSize)
            noisyQEList = np.zeros(sampleSize)
            trueEList = np.zeros(sampleSize, dtype = np.float64)
            trueQEList = np.zeros(sampleSize, dtype = np.float64)
            mseEList = np.zeros(sampleSize)
            mseQEList = np.zeros(sampleSize)
            mseTList = np.zeros(sampleSize, dtype = np.float64)
            mseQTList = np.zeros(sampleSize, dtype = np.float64)

            # ADDING FIRST NOISE TERM TO MU DERIVED FROM GAUSSIAN DISTRIBUTION WITH MEAN 0 AND VARIANCE SIGMA SQUARED
            for i in range(0, dim):
                xi1 = normal(0, sigma**2)
                noisyMu[i] = mu[i] + xi1
                wNoisyMu[i] = wMu[i] + xi1

            # FIRST SUBTRACTION BETWEEN CIFAR-10 VECTOR OF EACH CLIENT AND NOISY MEAN ACCORDING TO THEOREM FOR DISPERSION
            for j in range(0, sampleSize):
                trueDiff = np.subtract(imageArray[j], mu)
                wTrueDiff = np.subtract(imageArray[j], wMu)
                noisyDiff = np.subtract(imageArray[j], noisyMu)
                wNoisyDiff = np.subtract(imageArray[j], wNoisyMu)

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

                noisyEList[j] = np.sum(noisyDisp)
                noisyQEList[j] = np.sum(noisyQ)
                trueEList[j] = np.sum(trueDisp)
                trueQEList[j] = np.sum(weightedTrueDisp)

                # EMPIRICAL MSE = THE SQUARE OF THE ABOVE UNROUNDED STATISTIC MINUS THE TRUE DISPERSION
                mseEList[j] = np.power((noisyEList[j] - trueEList[j]), 2)
                mseQEList[j] = np.power((noisyQEList[j] - trueQEList[j]), 2)

                # ADDING SECOND NOISE TERM TO EXPRESSION OF DISPERSION AND COMPUTING THEORETICAL MSE USING VARIABLES DEFINED ABOVE
                doubleTrueDiff = 2*trueDiff
                wDoubleTrueDiff = 2*wTrueDiff
                bracket = np.subtract(xi1, doubleTrueDiff)
                wBracket = np.subtract(xi1, wDoubleTrueDiff)
                multiply = np.multiply(xi1, bracket)
                wMultiply = np.multiply(xi1, wBracket)
                weightedMult = (weight[j])*(wMultiply)

                extraTerm = np.add(multiply, xi2)
                wExtraTerm = np.add(weightedMult, xi2)
                extraTermSquared = np.power(extraTerm, 2)
                wExtraTermSquared = np.power(wExtraTerm, 2)
                mseTList[j] = np.sum(extraTermSquared)
                mseQTList[j] = np.sum(wExtraTermSquared)

            mseEmpirical = np.sum(mseEList)
            mseTheoretical = np.sum(mseTList)
            mseQEmpirical = np.sum(mseQEList)
            mseQTheoretical = np.sum(mseQTList)

            if ACindex == 0:
                np.copyto(compareEListA, mseEList)
                np.copyto(compareQEListA, mseQEList)
                np.copyto(compareTListA, mseTList)
                np.copyto(compareQTListA, mseQTList)

                # EXPERIMENT 1 (BEHAVIOUR OF EPS AND DTA) ASSUMES UNIFORM DATA
                if freqIndex == 0:
                    mseDispEPlotVATemp[rep] = mseEmpirical
                    mseQEPlotVATemp[rep] = mseQEmpirical
                    mseDispTPlotVATemp[rep] = mseTheoretical
                    mseQTPlotVATemp[rep] = mseQTheoretical

                # EXPERIMENT 4 (STATISTICAL HETEROGENEITY) ASSUMES EPS IS CONSTANT
                if index == 0 and val == 0:
                    mseDispEPlotFATemp[rep] = mseEmpirical
                    mseQEPlotFATemp[rep] = mseQEmpirical
                    mseDispTPlotFATemp[rep] = mseTheoretical
                    mseQTPlotFATemp[rep] = mseQTheoretical
            else:
                np.copyto(compareEListC, mseEList)
                np.copyto(compareQEListC, mseQEList)
                np.copyto(compareTListC, mseTList)
                np.copyto(compareQTListC, mseQTList)

                if freqIndex == 0:
                    mseDispEPlotVCTemp[rep] = mseEmpirical
                    mseQEPlotVCTemp[rep] = mseQEmpirical
                    mseDispTPlotVCTemp[rep] = mseTheoretical
                    mseQTPlotVCTemp[rep] = mseQTheoretical

                if index == 0 and val == 0:
                    mseDispEPlotFCTemp[rep] = mseEmpirical
                    mseQEPlotFCTemp[rep] = mseQEmpirical
                    mseDispTPlotFCTemp[rep] = mseTheoretical
                    mseQTPlotFCTemp[rep] = mseQTheoretical

            trueISquaredList = np.zeros(sampleSize)
            iSquaredList = np.zeros(sampleSize)
            mseISEList = np.zeros(sampleSize)
            mseISTList = np.zeros(sampleSize)

            for j in range(0, sampleSize):

                # COMPUTE I^2'' and I^2 USING SIMPLE FORMULA AT BOTTOM OF LEMMA 6.2
                trueISquaredPrep = np.divide(sampleSize-1, trueQEList[j])
                trueISquaredList[j] = np.subtract(1, trueISquaredPrep)
                iSquaredPrep = np.divide(sampleSize-1, noisyQEList[j])
                iSquaredList[j] = np.subtract(1, iSquaredPrep)

                # ADD THIRD NOISE TERM BASED ON LEMMA 6.2
                xi3 = normal(0, sigma**2)
                noisyISquared = np.add(iSquaredList[j], xi3)

                # COMPUTE EMPIRICAL AND THEORETICAL MSE
                diffEISquared = np.subtract(noisyISquared, trueISquaredList[j])
                mseISEList[j] = np.power(diffEISquared, 2)
                diffTISquaredPrep = np.subtract(xi3, iSquaredList[j])
                diffTISquared = np.add(diffTISquaredPrep, trueISquaredList[j])
                mseISTList[j] = np.power(diffTISquared, 2)

            mseISquaredEmpirical = np.sum(mseISEList)
            mseISquaredTheoretical = np.sum(mseISTList)

            # EXPERIMENT 3: WHAT IS THE COST OF A DISTRIBUTED SETTING?
            xiCentral = normal(0, centralSigma**2)
            mseCentral = xiCentral**2
            
            if freqIndex == 0:
                mseCentralPlotVTemp[rep] = mseCentral
            if index == 0 and val == 0:
                mseCentralPlotFTemp[rep] = mseCentral

            if ACindex == 0:
                np.copyto(compareISEListA, mseISEList)
                np.copyto(compareISTListA, mseISTList)

                if freqIndex == 0:
                    mseISquaredEPlotVATemp[rep] = mseISquaredEmpirical
                    mseISquaredTPlotVATemp[rep] = mseISquaredTheoretical

                if index == 0 and val == 0:
                    mseISquaredEPlotFATemp[rep] = mseISquaredEmpirical
                    mseISquaredTPlotFATemp[rep] = mseISquaredTheoretical
            else:
                np.copyto(compareISEListC, mseISEList)
                np.copyto(compareISTListC, mseISTList)

                if freqIndex == 0:
                    mseISquaredEPlotVCTemp[rep] = mseISquaredEmpirical
                    mseISquaredTPlotVCTemp[rep] = mseISquaredTheoretical

                if index == 0 and val == 0:
                    mseISquaredEPlotFCTemp[rep] = mseISquaredEmpirical
                    mseISquaredTPlotFCTemp[rep] = mseISquaredTheoretical

        # EXPERIMENT 4: SAMPLE APPROX 2% OF CLIENTS THEN SPLIT INTO CASES BY STATISTICAL HETEROGENEITY
        # 1. EQUAL NUMBERS OF EACH OF 10 LABELS [1:1:1:1:1:1:1:1:1:1]
        # 2. UNEQUAL NUMBERS OF EACH OF 10 LABELS [11:1:1:1:1:1:1:1:1:1]
        # 3. EQUAL NUMBERS OF EACH OF 5 LABELS [1:1:1:1:1:0:0:0:0:0]
        # 4. UNEQUAL NUMBERS OF EACH OF 5 LABELS [6:1:1:1:1:0:0:0:0:0]
        # 5. EQUAL NUMBERS OF EACH OF 2 LABELS [1:1:0:0:0:0:0:0:0:0]
        # 6. UNEQUAL NUMBERS OF EACH OF 2 LABELS [9:1:0:0:0:0:0:0:0:0].
    
        numLabels = 10
        lsize = sampleSize/numLabels
        freqArray = np.zeros(numLabels)
        if dataIndex == 2:
            imageArray = np.zeros((sampleSize, np.sqrt(dim), np.sqrt(dim)))
        else:
            imageArray = np.zeros((sampleSize, dim))
        freqOne = np.array([lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize])
        freqTwo = np.array([5.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize])
        freqThree = np.array([2*lsize, 2*lsize, 2*lsize, 2*lsize, 2*lsize, 0, 0, 0, 0, 0])
        freqFour = np.array([6*lsize, lsize, lsize, lsize, lsize, 0, 0, 0, 0, 0])
        freqFive = np.array([5*lsize, 5*lsize, 0, 0, 0, 0, 0, 0, 0, 0])
        freqSix = np.array([9*lsize, lsize, 0, 0, 0, 0, 0, 0, 0, 0])

        if freqIndex == 0:
            freqSpec = freqOne
        if freqIndex == 1:
            freqSpec = freqTwo
        if freqIndex == 2:
            freqSpec = freqThree
        if freqIndex == 3:
            freqSpec = freqFour
        if freqIndex == 4:
            freqSpec = freqFive
        if freqIndex == 5:
            freqSpec = freqSix
     
        LAB_COUNT = 0
        INDEX_COUNT = 0

        while LAB_COUNT < sampleSize:
            for lab in labels:

                # CIFAR-100 HAS 20 COARSE LABELS THAT CAN BE MERGED INTO 10     
                if dataIndex == 1:
                    lab = lab//2

                if freqArray[lab] < freqSpec[lab]:
                    freqArray[lab] = freqArray[lab] + 1
                    sampledImage = newImages[LAB_COUNT]
                    imageArray[INDEX_COUNT] = sampledImage
                    INDEX_COUNT = INDEX_COUNT + 1

                LAB_COUNT = LAB_COUNT + 1

        # COMPUTE SIGMA USING CLASSIC GAUSSIAN MECHANISM FOR COMPARISON BETWEEN MSE AND DISTRIBUTED/CENTRALISED SETTING
        classicSigma = (GS*mp.sqrt(2*mp.log(1.25/dta)))/eps
        classicCentralSigma = (mp.sqrt(2*mp.log(1.25/dta)))/eps

        # REPEATS FOR EACH FREQUENCY SPECIFICATION
        for rep in range(R):
            computeMSE(0, rep, imageArray, sigma, centralSigma)
            computeMSE(1, rep, imageArray, classicSigma, classicCentralSigma)

            # COMPARING AGM AND CGM
            comparelists1 = np.divide(compareEListA, compareEListC)
            compareqlists1 = np.divide(compareQEListA, compareQEListC)
            compareislists1 = np.divide(compareISEListA, compareISEListC)
            sumdiff1 = abs(np.mean(comparelists1))
            sumqdiff1 = abs(np.mean(compareqlists1))
            sumisdiff1 = abs(np.mean(compareislists1))

            # EXPERIMENT 2 (AGM VS CGM) ASSUMES UNIFORM DATA
            if freqIndex == 0:
                acDispEPlotVTemp[rep] = sumdiff1
                acQEPlotVTemp[rep] = sumqdiff1
                acISquaredEPlotVTemp[rep] = sumisdiff1

            # EXPERIMENT 4 (STATISTICAL HETEROGENEITY) ASSUMES EPS IS CONSTANT
            if index == 0 and val == 0:
                acDispEPlotFTemp[rep] = sumdiff1
                acQEPlotFTemp[rep] = sumqdiff1
                acISquaredEPlotFTemp[rep] = sumisdiff1

            comparelists2 = np.divide(compareTListA, compareTListC)
            compareqlists2 = np.divide(compareQTListA, compareQTListC)
            compareislists2 = np.divide(compareISTListA, compareISTListC)
            sumdiff2 = abs(np.mean(comparelists2))
            sumqdiff2 = abs(np.mean(compareqlists2))
            sumisdiff2 = abs(np.mean(compareislists2))

            if freqIndex == 0:
                acDispTPlotVTemp[rep] = sumdiff2
                acQTPlotVTemp[rep] = sumqdiff2
                acISquaredTPlotVTemp[rep] = sumisdiff2

            if index == 0 and val == 0:
                acDispTPlotFTemp[rep] = sumdiff2
                acQTPlotFTemp[rep] = sumqdiff2
                acISquaredTPlotFTemp[rep] = sumisdiff2

        if freqIndex == 0:
            mseDispEPlotVA[val] = np.mean(mseDispEPlotVATemp)
            mseQEPlotVA[val] = np.mean(mseQEPlotVATemp)
            mseDispEPlotVC[val] = np.mean(mseDispEPlotVCTemp)
            mseQEPlotVC[val] = np.mean(mseQEPlotVCTemp)
            mseDispTPlotVA[val] = np.mean(mseDispTPlotVATemp)
            mseQTPlotVA[val] = np.mean(mseQTPlotVATemp)
            mseDispTPlotVC[val] = np.mean(mseDispTPlotVCTemp)
            mseQTPlotVC[val] = np.mean(mseQTPlotVCTemp)
            mseISquaredEPlotVA[val] = np.mean(mseISquaredEPlotVATemp)
            mseISquaredEPlotVC[val] = np.mean(mseISquaredEPlotVCTemp)
            mseISquaredTPlotVA[val] = np.mean(mseISquaredTPlotVATemp)
            mseISquaredTPlotVC[val] = np.mean(mseISquaredTPlotVCTemp)
            mseCentralPlotV[val] = np.mean(mseCentralPlotVTemp)
            acDispEPlotV[val] = np.mean(acDispEPlotVTemp)
            acDispTPlotV[val] = np.mean(acDispTPlotVTemp)
            acQEPlotV[val] = np.mean(acQEPlotVTemp)
            acQTPlotV[val] = np.mean(acQTPlotVTemp)
            acISquaredEPlotV[val] = np.mean(acISquaredEPlotVTemp)
            acISquaredTPlotV[val] = np.mean(acISquaredTPlotVTemp)

            mseDispEPlotVARange[val] = np.std(mseDispEPlotVATemp)
            mseQEPlotVARange[val] = np.std(mseQEPlotVATemp)
            mseDispEPlotVCRange[val] = np.std(mseDispEPlotVCTemp)
            mseQEPlotVCRange[val] = np.std(mseQEPlotVCTemp)
            mseDispTPlotVARange[val] = np.std(mseDispTPlotVATemp)
            mseQTPlotVARange[val] = np.std(mseQTPlotVATemp)
            mseDispTPlotVCRange[val] = np.std(mseDispTPlotVCTemp)
            mseQTPlotVCRange[val] = np.std(mseQTPlotVCTemp)
            mseISquaredEPlotVARange[val] = np.std(mseISquaredEPlotVATemp)
            mseISquaredEPlotVCRange[val] = np.std(mseISquaredEPlotVCTemp)
            mseISquaredTPlotVARange[val] = np.std(mseISquaredTPlotVATemp)
            mseISquaredTPlotVCRange[val] = np.std(mseISquaredTPlotVCTemp)
            mseCentralPlotVRange[val] = np.std(mseCentralPlotVTemp)
            acDispEPlotVRange[val] = np.std(acDispEPlotVTemp)
            acDispTPlotVRange[val] = np.std(acDispTPlotVTemp)
            acQEPlotVRange[val] = np.std(acQEPlotVTemp)
            acQTPlotVRange[val] = np.std(acQTPlotVTemp)
            acISquaredEPlotVRange[val] = np.std(acISquaredEPlotVTemp)
            acISquaredTPlotVRange[val] = np.std(acISquaredTPlotVTemp)

        if index == 0 and val == 0:
            mseDispEPlotFA[freqIndex] = np.mean(mseDispEPlotFATemp)
            mseQEPlotFA[freqIndex] = np.mean(mseQEPlotFATemp)
            mseDispEPlotFC[freqIndex] = np.mean(mseDispEPlotFCTemp)
            mseQEPlotFC[freqIndex] = np.mean(mseQEPlotFCTemp)
            mseDispTPlotFA[freqIndex] = np.mean(mseDispTPlotFATemp)
            mseQTPlotFA[freqIndex] = np.mean(mseQTPlotFATemp)
            mseDispTPlotFC[freqIndex] = np.mean(mseDispTPlotFCTemp)
            mseQTPlotFC[freqIndex] = np.mean(mseQTPlotFCTemp)
            mseISquaredEPlotFA[freqIndex] = np.mean(mseISquaredEPlotFATemp)
            mseISquaredEPlotFC[freqIndex] = np.mean(mseISquaredEPlotFCTemp)
            mseISquaredTPlotFA[freqIndex] = np.mean(mseISquaredTPlotFATemp)
            mseISquaredTPlotFC[freqIndex] = np.mean(mseISquaredTPlotFCTemp)
            mseCentralPlotF[freqIndex] = np.mean(mseCentralPlotFTemp)
            acDispEPlotF[freqIndex] = np.mean(acDispEPlotFTemp)
            acDispTPlotF[freqIndex] = np.mean(acDispTPlotFTemp)
            acQEPlotF[freqIndex] = np.mean(acQEPlotFTemp)
            acQTPlotF[freqIndex] = np.mean(acQTPlotFTemp)
            acISquaredEPlotF[freqIndex] = np.mean(acISquaredEPlotFTemp)
            acISquaredTPlotF[freqIndex] = np.mean(acISquaredTPlotFTemp)

            mseDispEPlotFARange[freqIndex] = np.std(mseDispEPlotFATemp)
            mseQEPlotFARange[freqIndex] = np.std(mseQEPlotFATemp)
            mseDispEPlotFCRange[freqIndex] = np.std(mseDispEPlotFCTemp)
            mseQEPlotFCRange[freqIndex] = np.std(mseQEPlotFCTemp)
            mseDispTPlotFARange[freqIndex] = np.std(mseDispTPlotFATemp)
            mseQTPlotFARange[freqIndex] = np.std(mseQTPlotFATemp)
            mseDispTPlotFCRange[freqIndex] = np.std(mseDispTPlotFCTemp)
            mseQTPlotFCRange[freqIndex] = np.std(mseQTPlotFCTemp)
            mseISquaredEPlotFARange[freqIndex] = np.std(mseISquaredEPlotFATemp)
            mseISquaredEPlotFCRange[freqIndex] = np.std(mseISquaredEPlotFCTemp)
            mseISquaredTPlotFARange[freqIndex] = np.std(mseISquaredTPlotFATemp)
            mseISquaredTPlotFCRange[freqIndex] = np.std(mseISquaredTPlotFCTemp)
            mseCentralPlotFRange[freqIndex] = np.std(mseCentralPlotFTemp)
            acDispEPlotFRange[freqIndex] = np.std(acDispEPlotFTemp)
            acDispTPlotFRange[freqIndex] = np.std(acDispTPlotFTemp)
            acQEPlotFRange[freqIndex] = np.std(acQEPlotFTemp)
            acQTPlotFRange[freqIndex] = np.std(acQTPlotFTemp)
            acISquaredEPlotFRange[freqIndex] = np.std(acISquaredEPlotFTemp)
            acISquaredTPlotFRange[freqIndex] = np.std(acISquaredTPlotFTemp)

    # EXPERIMENT 1: BEHAVIOUR OF (EPSILON, DELTA) GIVEN UNIFORM DATA
    if freqIndex == 0:
        plt.errorbar(varset, mseDispEPlotVA, yerr = np.minimum(mseDispEPlotVARange, np.sqrt(mseDispEPlotVA), np.divide(mseDispEPlotVA, 2)), color = 'blue', marker = 'o', label = "Empirical Analytic")
        plt.errorbar(varset, mseDispTPlotVA, yerr = np.minimum(mseDispTPlotVARange, np.sqrt(mseDispTPlotVA), np.divide(mseDispTPlotVA, 2)), color = 'green', marker = 'o', label = "Theoretical Analytic")
        plt.errorbar(varset, mseDispEPlotVC, yerr = np.minimum(mseDispEPlotVCRange, np.sqrt(mseDispEPlotVC), np.divide(mseDispEPlotVC, 2)), color = 'orange', marker = 'x', label = "Empirical Classic")
        plt.errorbar(varset, mseDispTPlotVC, yerr = np.minimum(mseDispTPlotVCRange, np.sqrt(mseDispTPlotVC), np.divide(mseDispTPlotVC, 2)), color = 'pink', marker = 'x', label = "Theoretical Classic")
        plt.errorbar(varset, mseCentralPlotV, yerr = np.minimum(mseCentralPlotVRange, np.sqrt(mseCentralPlotV), np.divide(mseCentralPlotV, 2)), color = 'red', marker = '*', label = "Centralized")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[index])
        plt.ylabel("MSE of Gaussian Mechanism")
        plt.savefig("Exp1_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_disp.png")
        plt.clf()

        plt.errorbar(varset, mseQEPlotVA, yerr = np.minimum(mseQEPlotVARange, np.sqrt(mseQEPlotVA), np.divide(mseQEPlotVA, 2)), color = 'blue', marker = 'o', label = "Empirical Analytic")
        plt.errorbar(varset, mseQTPlotVA, yerr = np.minimum(mseQTPlotVARange, np.sqrt(mseQTPlotVA), np.divide(mseQTPlotVA, 2)), color = 'green', marker = 'o', label = "Theoretical Analytic")
        plt.errorbar(varset, mseQEPlotVC, yerr = np.minimum(mseQEPlotVCRange, np.sqrt(mseQEPlotVC), np.divide(mseQEPlotVC, 2)), color = 'orange', marker = 'x', label = "Empirical Classic")
        plt.errorbar(varset, mseQTPlotVC, yerr = np.minimum(mseQTPlotVCRange, np.sqrt(mseQTPlotVC), np.divide(mseQTPlotVC, 2)), color = 'pink', marker = 'x', label = "Theoretical Classic")
        plt.errorbar(varset, mseCentralPlotV, yerr = np.minimum(mseCentralPlotVRange, np.sqrt(mseCentralPlotV), np.divide(mseCentralPlotV, 2)), color = 'red', marker = '*', label = "Centralized")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[index])
        plt.ylabel("MSE of Gaussian Mechanism")
        plt.savefig("Exp1_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_q.png")
        plt.clf()

        plt.errorbar(varset, mseISquaredEPlotVA, yerr = np.minimum(mseISquaredEPlotVARange, np.sqrt(mseISquaredEPlotVA), np.divide(mseISquaredEPlotVA, 2)), color = 'blue', marker = 'o', label = "Empirical Analytic")
        plt.errorbar(varset, mseISquaredTPlotVA, yerr = np.minimum(mseISquaredTPlotVARange, np.sqrt(mseISquaredTPlotVA), np.divide(mseISquaredTPlotVA, 2)), color = 'green', marker = 'o', label = "Theoretical Analytic")
        plt.errorbar(varset, mseISquaredEPlotVC, yerr = np.minimum(mseISquaredEPlotVCRange, np.sqrt(mseISquaredEPlotVC), np.divide(mseISquaredEPlotVC, 2)), color = 'orange', marker = 'x', label = "Empirical Classic")
        plt.errorbar(varset, mseISquaredTPlotVC, yerr = np.minimum(mseISquaredTPlotVCRange, np.sqrt(mseISquaredTPlotVC), np.divide(mseISquaredTPlotVC, 2)), color = 'pink', marker = 'x', label = "Theoretical Classic")
        plt.errorbar(varset, mseCentralPlotV, yerr = np.minimum(mseCentralPlotVRange, np.sqrt(mseCentralPlotV), np.divide(mseCentralPlotV, 2)), color = 'red', marker = '*', label = "Centralized")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[index])
        plt.ylabel("MSE of Gaussian Mechanism")
        plt.savefig("Exp1_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_isquared.png")
        plt.clf()

        # EXPERIMENT 2: AGM VS CGM (UNIFORM DATA)
        plt.errorbar(varset, acDispEPlotV, yerr = np.minimum(acDispEPlotVRange, np.sqrt(acDispEPlotV), np.divide(acDispEPlotV, 2)), color = 'blue', marker = 'o', label = "Empirical")
        plt.errorbar(varset, acDispTPlotV, yerr = np.minimum(acDispTPlotVRange, np.sqrt(acDispTPlotV), np.divide(acDispTPlotV, 2)), color = 'red', marker = 'x', label = "Theoretical")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[index])
        plt.ylabel("Multiplication factor")
        plt.savefig("Exp2_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_disp.png")
        plt.clf()

        plt.errorbar(varset, acQEPlotV, yerr = np.minimum(acQEPlotVRange, np.sqrt(acQEPlotV), np.divide(acQEPlotV, 2)), color = 'blue', marker = 'o', label = "Empirical")
        plt.errorbar(varset, acQTPlotV, yerr = np.minimum(acQTPlotVRange, np.sqrt(acQTPlotV), np.divide(acQTPlotV, 2)), color = 'red', marker = 'x', label = "Theoretical")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[index])
        plt.ylabel("Multiplication factor")
        plt.savefig("Exp2_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_q.png")
        plt.clf()

        plt.errorbar(varset, acISquaredEPlotV, yerr = np.minimum(acISquaredEPlotVRange, np.sqrt(acISquaredEPlotV), np.divide(acISquaredEPlotV, 2)), color = 'blue', marker = 'o', label = "Empirical")
        plt.errorbar(varset, acISquaredTPlotV, yerr = np.minimum(acISquaredTPlotVRange, np.sqrt(acISquaredTPlotV), np.divide(acISquaredTPlotV, 2)), color = 'red', marker = 'x', label = "Theoretical")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[index])
        plt.ylabel("Multiplication factor")
        plt.savefig("Exp2_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_isquared.png")
        plt.clf()

    if index == 0:
        # EXPERIMENT 4: STATISTICAL HETEROGENEITY (CONSTANT EPS OR DTA)
        plt.errorbar(freqset, mseDispEPlotFA, yerr = np.minimum(mseDispEPlotFARange, np.sqrt(mseDispEPlotFA), np.divide(mseDispEPlotFA, 2)), color = 'blue', marker = 'o', label = "Empirical Analytic")
        plt.errorbar(freqset, mseDispTPlotFA, yerr = np.minimum(mseDispTPlotFARange, np.sqrt(mseDispTPlotFA), np.divide(mseDispTPlotFA, 2)), color = 'green', marker = 'o', label = "Theoretical Analytic")
        plt.errorbar(freqset, mseDispEPlotFC, yerr = np.minimum(mseDispEPlotFCRange, np.sqrt(mseDispEPlotFC), np.divide(mseDispEPlotFC, 2)), color = 'orange', marker = 'x', label = "Empirical Classic")
        plt.errorbar(freqset, mseDispTPlotFC, yerr = np.minimum(mseDispTPlotFCRange, np.sqrt(mseDispTPlotFC), np.divide(mseDispTPlotFC, 2)), color = 'pink', marker = 'x', label = "Theoretical Classic")
        plt.errorbar(freqset, mseCentralPlotF, yerr = np.minimum(mseCentralPlotFRange, np.sqrt(mseCentralPlotF), np.divide(mseCentralPlotF, 2)), color = 'red', marker = '*', label = "Centralized")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Labels")
        plt.ylabel("MSE of Gaussian Mechanism")
        plt.savefig("Exp41_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_disp.png")
        plt.clf()

        plt.errorbar(freqset, mseQEPlotFA, yerr = np.minimum(mseQEPlotFARange, np.sqrt(mseQEPlotFA), np.divide(mseQEPlotFA, 2)), color = 'blue', marker = 'o', label = "Empirical Analytic")
        plt.errorbar(freqset, mseQTPlotFA, yerr = np.minimum(mseQTPlotFARange, np.sqrt(mseQTPlotFA), np.divide(mseQTPlotFA, 2)), color = 'green', marker = 'o', label = "Theoretical Analytic")
        plt.errorbar(freqset, mseQEPlotFC, yerr = np.minimum(mseQEPlotFCRange, np.sqrt(mseQEPlotFC), np.divide(mseQEPlotFC, 2)), color = 'orange', marker = 'x', label = "Empirical Classic")
        plt.errorbar(freqset, mseQTPlotFC, yerr = np.minimum(mseQTPlotFCRange, np.sqrt(mseQTPlotFC), np.divide(mseQTPlotFC, 2)), color = 'pink', marker = 'x', label = "Theoretical Classic")
        plt.errorbar(freqset, mseCentralPlotF, yerr = np.minimum(mseCentralPlotFRange, np.sqrt(mseCentralPlotF), np.divide(mseCentralPlotF, 2)), color = 'red', marker = '*', label = "Centralized")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Labels")
        plt.ylabel("MSE of Gaussian Mechanism")
        plt.savefig("Exp41_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_q.png")
        plt.clf()

        plt.errorbar(freqset, mseISquaredEPlotFA, yerr = np.minimum(mseISquaredEPlotFARange, np.sqrt(mseISquaredEPlotFA), np.divide(mseISquaredEPlotFA, 2)), color = 'blue', marker = 'o', label = "Empirical Analytic")
        plt.errorbar(freqset, mseISquaredTPlotFA, yerr = np.minimum(mseISquaredTPlotFARange, np.sqrt(mseISquaredTPlotFA), np.divide(mseISquaredTPlotFA, 2)), color = 'green', marker = 'o', label = "Theoretical Analytic")
        plt.errorbar(freqset, mseISquaredEPlotFC, yerr = np.minimum(mseISquaredEPlotFCRange, np.sqrt(mseISquaredEPlotFC), np.divide(mseISquaredEPlotFC, 2)), color = 'orange', marker = 'x', label = "Empirical Classic")
        plt.errorbar(freqset, mseISquaredTPlotFC, yerr = np.minimum(mseISquaredTPlotFCRange, np.sqrt(mseISquaredTPlotFC), np.divide(mseISquaredTPlotFC, 2)), color = 'pink', marker = 'x', label = "Theoretical Classic")
        plt.errorbar(freqset, mseCentralPlotF, yerr = np.minimum(mseCentralPlotFRange, np.sqrt(mseCentralPlotF), np.divide(mseCentralPlotF, 2)), color = 'red', marker = '*', label = "Centralized")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Labels")
        plt.ylabel("MSE of Gaussian Mechanism")
        plt.savefig("Exp41_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_isquared.png")
        plt.clf()

        plt.errorbar(freqset, acDispEPlotF, yerr = np.minimum(acDispEPlotFRange, np.sqrt(acDispEPlotF), np.divide(acDispEPlotF, 2)), color = 'blue', marker = 'o', label = "Empirical")
        plt.errorbar(freqset, acDispTPlotF, yerr = np.minimum(acDispTPlotFRange, np.sqrt(acDispTPlotF), np.divide(acDispTPlotF, 2)), color = 'red', marker = 'x', label = "Theoretical")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Labels")
        plt.ylabel("Multiplication factor")
        plt.savefig("Exp42_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_disp.png")
        plt.clf()

        plt.errorbar(freqset, acQEPlotF, yerr = np.minimum(acQEPlotFRange, np.sqrt(acQEPlotF), np.divide(acQEPlotF, 2)), color = 'blue', marker = 'o', label = "Empirical")
        plt.errorbar(freqset, acQTPlotF, yerr = np.minimum(acQTPlotFRange, np.sqrt(acQTPlotF), np.divide(acQTPlotF, 2)), color = 'red', marker = 'x', label = "Theoretical")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Labels")
        plt.ylabel("Multiplication factor")
        plt.savefig("Exp42_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_q.png")
        plt.clf()

        plt.errorbar(freqset, acISquaredEPlotF, yerr = np.minimum(acISquaredEPlotFRange, np.sqrt(acISquaredEPlotF), np.divide(acISquaredEPlotF, 2)), color = 'blue', marker = 'o', label = "Empirical")
        plt.errorbar(freqset, acISquaredTPlotF, yerr = np.minimum(acISquaredTPlotFRange, np.sqrt(acISquaredTPlotF), np.divide(acISquaredTPlotF, 2)), color = 'red', marker = 'x', label = "Theoretical")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Labels")
        plt.ylabel("Multiplication factor")
        plt.savefig("Exp42_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_isquared.png")
        plt.clf()

def runLoopVaryEps(dataIndex, index, freqIndex, dim, num, newImages, labels, GS):
    runLoop(dataIndex, index, freqIndex, epsset, dim, num, -1, dtaconst, newImages, labels, GS)

def runLoopVaryDta(dataIndex, index, freqIndex, dim, num, newImages, labels, GS):
    runLoop(dataIndex, index, freqIndex, dtaset, dim, num, epsconst, -1, newImages, labels, GS)

for fi in range(6):
    runLoopVaryEps(0, 0, fi, dimCifar, numCifar, newImagesCifar10, labelsCifar10, GSCifar)
    runLoopVaryDta(0, 1, fi, dimCifar, numCifar, newImagesCifar10, labelsCifar10, GSCifar)

    runLoopVaryEps(1, 0, fi, dimCifar, numCifar, newImagesCifar100, labelsCifar100, GSCifar)
    runLoopVaryDta(1, 1, fi, dimCifar, numCifar, newImagesCifar100, labelsCifar100, GSCifar)

    runLoopVaryEps(2, 0, fi, dimFashion, numFashion, newImagesFashion, labelsFashion, GSFashion)
    runLoopVaryDta(2, 1, fi, dimFashion, numFashion, newImagesFashion, labelsFashion, GSFashion)

print("Finished.\n")