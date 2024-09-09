import numpy as np
import idx2numpy
import os
import mpmath as mp
import matplotlib.pyplot as plt

from math import erf
from numpy.random import normal
from PIL import Image
from numpy import asarray

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
    dict = idx2numpy.convert_from_file(filename)
    dataFashion = dict.reshape((numFashion, dimFashion))
    return dataFashion

# ADAPTATION OF TRANSFORMATION OF LABEL INDICES TO ONE-HOT ENCODED VECTORS AND IMAGES TO 3072-DIMENSIONAL VECTORS BY HADHAZI
def transformValues(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

# CALL ALL THE ABOVE METHODS
print("Loading data...\n")
imagesCifar10 = loadCifar10(b'data')
labelsCifar10 = loadCifar10(b'labels')
imagesCifar100 = loadCifar100(b'data')
labelsCifar100 = loadCifar100(b'labels')
imagesFashion = loadFashion('train-images-idx3-ubyte')
labelsFashion = loadFashion('train-labels-idx1-ubyte')

newImagesCifar10 = transformValues(imagesCifar10)
newImagesCifar100 = transformValues(imagesCifar100)
newImagesFashion = transformValues(imagesFashion)

os.chdir('..')

def runLoop(dataIndex, index, freqIndex, varset, dim, num, eps, dta, newImages, labels, GS):

    # STORING STATISTICS REQUIRED FOR GRAPHS
    V = len(varset)
    mseDispEPlotA = np.zeros(V)
    mseQEPlotA = np.zeros(V)
    mseDispEPlotC = np.zeros(V)
    mseQEPlotC = np.zeros(V)
    mseDispTPlotA = np.zeros(V)
    mseQTPlotA = np.zeros(V)
    mseDispTPlotC = np.zeros(V)
    mseQTPlotC = np.zeros(V)
    mseISquaredEPlotA = np.zeros(V)
    mseISquaredEPlotC = np.zeros(V)
    mseISquaredTPlotA = np.zeros(V)
    mseISquaredTPlotC = np.zeros(V)
    mseCentralPlot = np.zeros(V)
    acDispEPlot = np.zeros(V)
    acDispTPlot = np.zeros(V)
    acQEPlot = np.zeros(V)
    acQTPlot = np.zeros(V)
    acISquaredEPlot = np.zeros(V)
    acISquaredTPlot = np.zeros(V)

    for val in range(10):

        var = varset[val]
        print(f"\nProcessing dataset {dataIndex+1} for the value {parset[index]} = {var}.")

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
        print("Calibrating AGM...")

        sample = 0.02
        sampleSize = num*sample
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

        # EXPERIMENT 1: BEHAVIOUR OF VARIABLES AT DIFFERENT SETTINGS
        def computeMSE(ACindex, imageArray, sigma, centralSigma):

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
                mseDispEPlotA[val] = mseDispEPlotA[val] + (mseEmpirical/R)
                mseQEPlotA[val] = mseQEPlotA[val] + (mseQEmpirical/R)
                mseDispTPlotA[val] = mseDispTPlotA[val] + (mseTheoretical/R)
                mseQTPlotA[val] = mseQTPlotA[val] + (mseQTheoretical/R)
            else:
                np.copyto(compareEListC, mseEList)
                np.copyto(compareQEListC, mseQEList)
                np.copyto(compareTListC, mseTList)
                np.copyto(compareQTListC, mseQTList)
                mseDispEPlotC[val] = mseDispEPlotC[val] + (mseEmpirical/R)
                mseQEPlotC[val] = mseQEPlotC[val] + (mseQEmpirical/R)
                mseDispTPlotC[val] = mseDispTPlotC[val] + (mseTheoretical/R)
                mseQTPlotC[val] = mseQTPlotC[val] + (mseQTheoretical/R)

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
            mseCentralPlot[val] = mseCentral

            if ACindex == 0:
                np.copyto(compareISEListA, mseISEList)
                np.copyto(compareISTListA, mseISTList)
                mseISquaredEPlotA[val] = mseISquaredEPlotA[val] + (mseISquaredEmpirical/R)
                mseISquaredTPlotA[val] = mseISquaredTPlotA[val] + (mseISquaredTheoretical/R)
            else:
                np.copyto(compareISEListC, mseISEList)
                np.copyto(compareISTListC, mseISTList)
                mseISquaredEPlotC[val] = mseISquaredEPlotC[val] + (mseISquaredEmpirical/R)
                mseISquaredTPlotC[val] = mseISquaredTPlotC[val] + (mseISquaredTheoretical/R)

        # EXPERIMENT 4: SAMPLE APPROX 2% OF CLIENTS THEN SPLIT INTO CASES BY STATISTICAL HETEROGENEITY
        # 1. EQUAL NUMBERS OF EACH OF 10 LABELS [1:1:1:1:1:1:1:1:1:1]
        # 2. UNEQUAL NUMBERS OF EACH OF 10 LABELS [11:1:1:1:1:1:1:1:1:1]
        # 3. EQUAL NUMBERS OF EACH OF 5 LABELS [1:1:1:1:1:0:0:0:0:0]
        # 4. UNEQUAL NUMBERS OF EACH OF 5 LABELS [6:1:1:1:1:0:0:0:0:0]
        # 5. EQUAL NUMBERS OF EACH OF 2 LABELS [1:1:0:0:0:0:0:0:0:0]
        # 6. UNEQUAL NUMBERS OF EACH OF 2 LABELS [9:1:0:0:0:0:0:0:0:0]
    
        numLabels = 10
        lsize = sampleSize/numLabels
        freqArray = np.zeros(numLabels)
        imageArray = np.zeros(num*sample)
        freqOne = np.array([lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize])
        freqTwo = np.array([5.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize])
        freqThree = np.array([2*lsize, 2*lsize, 2*lsize, 2*lsize, 2*lsize, 0, 0, 0, 0, 0])
        freqFour = np.array[6*lsize, lsize, lsize, lsize, lsize, 0, 0, 0, 0, 0]
        freqFive = np.array[5*lsize, 5*lsize, 0, 0, 0, 0, 0, 0, 0, 0]
        freqSix = np.array[9*lsize, lsize, 0, 0, 0, 0, 0, 0, 0, 0]

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
            
        while LAB_COUNT < sampleSize:
            for lab in labels:
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
            print(f"\nRepeat {rep + 1}.")
            computeMSE(0, imageArray, sigma, centralSigma)
            computeMSE(1, imageArray, classicSigma, classicCentralSigma)

            # EXPERIMENT 2: AGM VS CGM
            comparelists1 = np.divide(compareEListA, compareEListC)
            compareqlists1 = np.divide(compareQEListA, compareQEListC)
            compareislists1 = np.divide(compareISEListA, compareISEListC)
            sumdiff1 = abs(np.mean(comparelists1))
            sumqdiff1 = abs(np.mean(compareqlists1))
            sumisdiff1 = abs(np.mean(compareislists1))
            acDispEPlot[val] = acDispEPlot[val] + (sumdiff1/R)
            acQEPlot[val] = acQEPlot[val] + (sumqdiff1/R)
            acISquaredEPlot[val] = acISquaredEPlot[val] + (sumisdiff1/R)

            comparelists2 = np.divide(compareTListA, compareTListC)
            compareqlists2 = np.divide(compareQTListA, compareQTListC)
            compareislists2 = np.divide(compareISTListA, compareISTListC)
            sumdiff2 = abs(np.mean(comparelists2))
            sumqdiff2 = abs(np.mean(compareqlists2))
            sumisdiff2 = abs(np.mean(compareislists2))
            acDispTPlot[val] = acDispTPlot[val] + (sumdiff2/R)
            acQTPlot[val] = acQTPlot[val] + (sumqdiff2/R)
            acISquaredTPlot[val] = acISquaredTPlot[val] + (sumisdiff2/R)

    # EXPERIMENT 1: BEHAVIOUR OF (EPSILON, DELTA)
    plt.errorbar(varset, mseDispEPlotA, color = 'blue', marker = 'o', label = "Empirical Analytic")
    plt.errorbar(varset, mseDispTPlotA, color = 'green', marker = 'o', label = "Theoretical Analytic")
    plt.errorbar(varset, mseDispEPlotC, color = 'orange', marker = 'x', label = "Empirical Classic")
    plt.errorbar(varset, mseDispTPlotC, color = 'pink', marker = 'x', label = "Theoretical Classic")
    plt.errorbar(varset, mseCentralPlot, color = 'red', marker = '*', label = "Centralized")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Exp1_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_disp.png")
    plt.clf()

    plt.errorbar(varset, mseQEPlotA, color = 'blue', marker = 'o', label = "Empirical Analytic")
    plt.errorbar(varset, mseQTPlotA, color = 'green', marker = 'o', label = "Theoretical Analytic")
    plt.errorbar(varset, mseQEPlotC, color = 'orange', marker = 'x', label = "Empirical Classic")
    plt.errorbar(varset, mseQTPlotC, color = 'pink', marker = 'x', label = "Theoretical Classic")
    plt.errorbar(varset, mseCentralPlot, color = 'red', marker = '*', label = "Centralized")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Exp1_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_q.png")
    plt.clf()

    plt.errorbar(varset, mseISquaredEPlotA, color = 'blue', marker = 'o', label = "Empirical Analytic")
    plt.errorbar(varset, mseISquaredTPlotA, color = 'green', marker = 'o', label = "Theoretical Analytic")
    plt.errorbar(varset, mseISquaredEPlotC, color = 'orange', marker = 'x', label = "Empirical Classic")
    plt.errorbar(varset, mseISquaredTPlotC, color = 'pink', marker = 'x', label = "Theoretical Classic")
    plt.errorbar(varset, mseCentralPlot, color = 'red', marker = '*', label = "Centralized")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Exp1_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_isquared.png")
    plt.clf()

    # EXPERIMENT 2: AGM VS CGM
    plt.errorbar(varset, acDispEPlot, color = 'blue', marker = 'o', label = "Empirical")
    plt.errorbar(varset, acDispTPlot, color = 'red', marker = 'x', label = "Theoretical")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("Multiplication factor")
    plt.savefig("Exp2_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_disp.png")
    plt.clf()

    plt.errorbar(varset, acQEPlot, color = 'blue', marker = 'o', label = "Empirical")
    plt.errorbar(varset, acQTPlot, color = 'red', marker = 'x', label = "Theoretical")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("Multiplication factor")
    plt.savefig("Exp2_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_q.png")
    plt.clf()

    plt.errorbar(varset, acISquaredEPlot, color = 'blue', marker = 'o', label = "Empirical")
    plt.errorbar(varset, acISquaredTPlot, color = 'red', marker = 'x', label = "Theoretical")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("Multiplication factor")
    plt.savefig("Exp2_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_isquared.png")
    plt.clf()

    # ADD GRAPHS FOR EXPERIMENT 4 WHEN READY

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