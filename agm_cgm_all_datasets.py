import numpy as np
import idx2numpy
import mpmath as mp
import matplotlib.pyplot as plt

from math import erf
from numpy.random import normal
from prettytable import PrettyTable
from sigfig import round

# INITIALISING SEED FOR RANDOM SAMPLING
print("\nStarting...")
np.random.seed(3820672)

# ARRAYS STORING SETS OF VALUES OF EACH VARIABLE WITH OPTIMA CHOSEN AS CONSTANTS
epsset = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5])
epsconst = float(epsset[0])

# VECTOR DIMENSION CHOSEN TO MATCH THAT OF CONVERTED IMAGES ABOVE AND NUMBER OF CLIENTS CHOSEN TO GIVE SENSIBLE GS
dtaset = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5])
dtaconst = float(dtaset[0])

# SETTING DIMENSIONS OF DATASETS
dimCifar = 3072
dimFashion = 784

numCifar = 50000
numFashion = 60000

GSCifar = float(mp.sqrt(dimCifar))/numCifar
GSFashion = float(mp.sqrt(dimFashion))/numFashion

# INITIALISING OTHER PARAMETERS AND CONSTANTS
cifarset = np.array(['Cifar-10', 'Cifar-100', 'Fashion-MNIST'])
parset = np.array(['eps', 'dta'])
graphset = np.array(['$\mathit{\u03b5}$', '$\mathit{\u03b4}$'])
labelset = np.array(['10_labels', '5_labels', '2_labels'])
R = 10

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
def loadFashion(filename, reshape):
    dict = idx2numpy.convert_from_file(filename)
    if reshape == 1:
        dataFashion = dict.reshape((numFashion, dimFashion))
    else:
        dataFashion = dict
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
imagesFashion = loadFashion('train-images-idx3-ubyte', 1)
labelsFashion = loadFashion('train-labels-idx1-ubyte', 0)

newImagesCifar10 = transformValues(imagesCifar10)
newImagesCifar100 = transformValues(imagesCifar100)
newImagesFashion = transformValues(imagesFashion)

# PREPARING GLOBAL ARRAYS
L = len(labelset)
E = len(epsset)
C = len(cifarset)
T1 = 4
T2 = 7

mseDispTable = np.zeros((C, 3*T1))
mseQTable = np.zeros((C, 3*T1))
mseI2Table = np.zeros((C, 3*T1))
mseCentralTable = np.zeros((C, C))

stdDispTable = np.zeros((C, T1))
stdQTable = np.zeros((C, T1))
stdI2Table = np.zeros((C, T1))
stdCentralTable = np.zeros((C, C-1))

mseDisp = np.zeros((2*C, 2*L, E))
mseQ = np.zeros((2*C, 2*L, E))
mseI2 = np.zeros((2*C, 2*L, E))
stdDisp = np.zeros((2*C, 2*L, E))
stdQ = np.zeros((2*C, 2*L, E))
stdI2 = np.zeros((2*C, 2*L, E))

percLossDisp = np.zeros((2*C, T2))
percLossQ = np.zeros((2*C, T2))
percLossI2 = np.zeros((2*C, T2))

def runLoop(dataIndex, idx, varset, dim, num, eps, dta, newImages, labels, GS):

    V = len(varset)

    mseDispEPlotA = np.zeros((2*L, V))
    mseQEPlotA = np.zeros((2*L, V))
    mseI2EPlotA = np.zeros((2*L, V))
    stdDispEPlotA = np.zeros((2*L, V))
    stdQEPlotA = np.zeros((2*L, V))
    stdI2EPlotA = np.zeros((2*L, V))
    percLossDispTable = np.zeros(T2)
    percLossQTable = np.zeros(T2)
    percLossI2Table = np.zeros(T2)

    for val in range(len(varset)):

        mseDispETableATemp = np.zeros(R)
        mseDispETableCTemp = np.zeros(R)
        mseDispTTableATemp = np.zeros(R)
        mseDispTTableCTemp = np.zeros(R)
        mseQETableATemp = np.zeros(R)
        mseQETableCTemp = np.zeros(R)
        mseQTTableATemp = np.zeros(R)
        mseQTTableCTemp = np.zeros(R)
        mseI2ETableATemp = np.zeros(R)
        mseI2ETableCTemp = np.zeros(R)
        mseI2TTableATemp = np.zeros(R)
        mseI2TTableCTemp = np.zeros(R)
        mseCTableATemp = np.zeros(R)
        mseCTableCTemp = np.zeros(R)

        mseDispEPlotATemp = np.zeros((2*L, V, R))
        mseQEPlotATemp = np.zeros((2*L, V, R))
        mseI2EPlotATemp = np.zeros((2*L, V, R))

        var = varset[val]
        print(f"Processing dataset {cifarset[dataIndex]} for the value {parset[idx]} = {var}.")

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

        def computeMSE(ACindex, rep, fi, imageArray, sigma, centralSigma):

            # INITIAL COMPUTATION OF WEIGHTED MEAN FOR Q BASED ON VECTOR VARIANCE
            wVector = np.var(imageArray, axis=1)
            weight = np.zeros(sampleSize)
            wImageArray = np.zeros((sampleSize, dim))

            for j in range(0, sampleSize):
                wVectorSquared = np.power(wVector[j], 2)
                weight[j] = np.divide(1.0, wVectorSquared)

                # MULTIPLYING EACH VECTOR BY ITS CORRESPONDING WEIGHTED MEAN
                wImageArray[j] = np.multiply(weight[j], imageArray[j])

            mu = np.mean(imageArray, axis=0)
            wSumMu = np.sum(wImageArray, axis=0)

            # DIVIDING SUM OF WEIGHTED VECTORS BY SUM OF WEIGHTS
            sumWeight = np.sum(weight)
            wMu = (wSumMu)/sumWeight

            noisyMu = np.zeros(dim)
            wNoisyMu = np.zeros(dim)

            mseEList = np.zeros(sampleSize)
            mseQEList = np.zeros(sampleSize)
            mseTList = np.zeros(sampleSize, dtype = np.float64)
            mseQTList = np.zeros(sampleSize, dtype = np.float64)
            weightedTrueDisp = np.zeros(sampleSize)
            noisyQ = np.zeros(sampleSize)

            # ADDING FIRST NOISE TERM TO MU DERIVED FROM GAUSSIAN DISTRIBUTION WITH MEAN 0 AND VARIANCE SIGMA SQUARED
            for i in range(0, dim):
                xi1 = normal(0, sigma**2)
                noisyMu[i] = np.add(mu[i], xi1)
                wNoisyMu[i] = np.add(wMu[i], xi1)

            # FIRST SUBTRACTION BETWEEN CIFAR-10 VECTOR OF EACH CLIENT AND NOISY MEAN ACCORDING TO THEOREM FOR DISPERSION
            for j in range(0, sampleSize):
                trueDiff = np.sum(np.subtract(imageArray[j], mu))
                wTrueDiff = np.sum(np.subtract(imageArray[j], wMu))
                noisyDiff = np.sum(np.subtract(imageArray[j], noisyMu))
                wNoisyDiff = np.sum(np.subtract(imageArray[j], wNoisyMu))

                # INCORPORATING WEIGHTS FOR STATISTICS ON Q
                trueDisp = np.power(trueDiff, 2)
                wTrueDisp = np.power(wTrueDiff, 2)
                weightedTrueDisp[j] = np.multiply(weight[j], wTrueDisp)
                noisyVar = np.power(noisyDiff, 2)
                wNoisyVar = np.power(wNoisyDiff, 2)
                weightedNoisyVar = np.multiply(weight[j], wNoisyVar)

                xi2 = normal(0, sigma**2)
                noisyDisp = np.add(noisyVar, xi2)
                noisyQ[j] = np.add(weightedNoisyVar, xi2)

                # EMSE = MSE OF FORMULA OF DISPERSION OR Q
                mseEList[j] = np.power(np.subtract(noisyDisp, trueDisp), 2)
                mseQEList[j] = np.power(np.subtract(noisyQ[j], weightedTrueDisp[j]), 2)

                # ADDING SECOND NOISE TERM TO EXPRESSION OF DISPERSION AND COMPUTING TMSE USING VARIABLES DEFINED ABOVE
                doubleTrueDiff = np.multiply(2, trueDiff)
                wDoubleTrueDiff = np.multiply(2, wTrueDiff)
                bracket = np.subtract(xi1, doubleTrueDiff)
                wBracket = np.subtract(xi1, wDoubleTrueDiff)
                multiply = np.multiply(xi1, bracket)
                wMultiply = np.multiply(xi1, wBracket)
                weightedMult = np.multiply(weight[j], wMultiply)

                extraTerm = np.add(multiply, xi2)
                wExtraTerm = np.add(weightedMult, xi2)
                extraTermSquared = np.power(extraTerm, 2)
                wExtraTermSquared = np.power(wExtraTerm, 2)
                mseTList[j] = np.sum(extraTermSquared)
                mseQTList[j] = np.sum(wExtraTermSquared)

            wTDSum = np.sum(weightedTrueDisp)
            noisyQSum = np.sum(noisyQ)
            mseE = np.sum(mseEList)
            mseT = np.sum(mseTList)
            mseQE = np.sum(mseQEList)
            mseQT = np.sum(mseQTList)

            if ACindex == 0:

                # TABLES ASSUME UNIFORM DATA
                if fi == 0 and val == 0 and idx == 1:
                    mseDispETableATemp[rep] = mseE
                    mseDispTTableATemp[rep] = mseT
                    mseQETableATemp[rep] = mseQE
                    mseQTTableATemp[rep] = mseQT

                # STATISTICAL HETEROGENEITY GRAPHS
                mseDispEPlotATemp[fi, val, rep] = mseE
                mseQEPlotATemp[fi, val, rep] = mseQE

            else:
                if fi == 0 and val == 0 and idx == 1:
                    mseDispETableCTemp[rep] = mseE
                    mseDispTTableCTemp[rep] = mseT
                    mseQETableCTemp[rep] = mseQE
                    mseQTTableCTemp[rep] = mseQT

            # COMPUTE I^2'' and I^2 USING SIMPLE FORMULA AT BOTTOM OF LEMMA 6.2
            trueI2Prep = np.divide(sampleSize-1, wTDSum)
            trueI2 = np.subtract(1, trueI2Prep)
            I2Prep = np.divide(sampleSize-1, noisyQSum)
            I2Noise = np.subtract(1, I2Prep)

            # ADD THIRD NOISE TERM BASED ON LEMMA 6.2
            xi3 = normal(0, sigma**2)
            noisyI2 = np.add(I2Noise, xi3)

            # COMPUTE EMSE AND TMSE
            diffEI2 = np.subtract(noisyI2, trueI2)
            mseI2E = np.power(diffEI2, 2)
            diffTI2Prep = np.subtract(xi3, I2Noise)
            diffTI2 = np.add(diffTI2Prep, trueI2)
            mseI2T = np.power(diffTI2, 2)

            # EXPERIMENT 2: WHAT IS THE COST OF A DISTRIBUTED SETTING?
            xiCentral = normal(0, centralSigma**2)
            mseC = xiCentral**2

            if ACindex == 0:
                if fi == 0 and val == 0 and idx == 1: 
                    mseI2ETableATemp[rep] = mseI2E
                    mseI2TTableATemp[rep] = mseI2T
                    mseCTableATemp[rep] = mseC

                mseI2EPlotATemp[fi, val, rep] = mseI2E

            else:
                if fi == 0 and val == 0 and idx == 1: 
                    mseI2ETableCTemp[rep] = mseI2E
                    mseI2TTableCTemp[rep] = mseI2T
                    mseCTableCTemp[rep] = mseC

        # EXPERIMENT 3: SAMPLE APPROX 2% OF CLIENTS THEN SPLIT INTO CASES BY STATISTICAL HETEROGENEITY
        # 1. EQUAL NUMBERS OF EACH OF 10 LABELS [1:1:1:1:1:1:1:1:1:1]
        # 2. UNEQUAL NUMBERS OF EACH OF 10 LABELS [91:1:1:1:1:1:1:1:1:1]
        # 3. EQUAL NUMBERS OF EACH OF 5 LABELS [1:1:1:1:1:0:0:0:0:0]
        # 4. UNEQUAL NUMBERS OF EACH OF 5 LABELS [96:1:1:1:1:0:0:0:0:0]
        # 5. EQUAL NUMBERS OF EACH OF 2 LABELS [1:1:0:0:0:0:0:0:0:0]
        # 6. UNEQUAL NUMBERS OF EACH OF 2 LABELS [99:1:0:0:0:0:0:0:0:0].
    
        for fi in range(6):
            numLabels = 10
            lsize = sampleSize/numLabels
            freqArray = np.zeros(numLabels)
            imageArray = np.zeros((sampleSize, dim))
            freqOne = np.array([lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize])
            freqTwo = np.array([9.1*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize])
            freqThree = np.array([2*lsize, 2*lsize, 2*lsize, 2*lsize, 2*lsize, 0, 0, 0, 0, 0])
            freqFour = np.array([9.6*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize, 0.1*lsize, 0, 0, 0, 0, 0])
            freqFive = np.array([5*lsize, 5*lsize, 0, 0, 0, 0, 0, 0, 0, 0])
            freqSix = np.array([9.9*lsize, 0.1*lsize, 0, 0, 0, 0, 0, 0, 0, 0])

            if fi == 0:
                freqSpec = freqOne
            if fi == 1:
                freqSpec = freqTwo
            if fi == 2:
                freqSpec = freqThree
            if fi == 3:
                freqSpec = freqFour
            if fi == 4:
                freqSpec = freqFive
            if fi == 5:
                freqSpec = freqSix

            LAB_COUNT = 0
            INDEX_COUNT = 0

            for lab in labels:
            
                # CIFAR-100 HAS 20 COARSE LABELS THAT CAN BE MERGED INTO 10     
                if dataIndex == 1:
                    lab = lab//2
                
                if freqArray[lab] < freqSpec[lab]:
                    freqArray[lab] = freqArray[lab] + 1
                    sampledImage = newImages[LAB_COUNT]
                    imageArray[INDEX_COUNT] = sampledImage
                    INDEX_COUNT = INDEX_COUNT + 1

            # COMPUTE SIGMA USING CLASSIC GAUSSIAN MECHANISM FOR COMPARISON BETWEEN MSE AND DISTRIBUTED OR CENTRALIZED SETTING
            classicSigma = (GS*mp.sqrt(2*mp.log(1.25/dta)))/eps
            classicCentralSigma = (mp.sqrt(2*mp.log(1.25/dta)))/eps

            # REPEATS FOR EACH FREQUENCY SPECIFICATION
            for rep in range(R):
                computeMSE(0, rep, fi, imageArray, sigma, centralSigma)
                computeMSE(1, rep, fi, imageArray, classicSigma, classicCentralSigma)

            if fi == 0 and val == 0 and idx == 1:
                mseDispETableA = np.mean(mseDispETableATemp)
                mseDispETableC = np.mean(mseDispETableCTemp)
                mseDispTTableA = np.mean(mseDispTTableATemp)
                mseDispTTableC = np.mean(mseDispTTableCTemp)
                mseQETableA = np.mean(mseQETableATemp)
                mseQETableC = np.mean(mseQETableCTemp)
                mseQTTableA = np.mean(mseQTTableATemp)
                mseQTTableC = np.mean(mseQTTableCTemp)
                mseI2ETableA = np.mean(mseI2ETableATemp)
                mseI2ETableC = np.mean(mseI2ETableCTemp)
                mseI2TTableA = np.mean(mseI2TTableATemp)
                mseI2TTableC = np.mean(mseI2TTableCTemp)
                mseCentralTableA = np.mean(mseCTableATemp)
                mseCentralTableC = np.mean(mseCTableCTemp)

                mseDispTable[dataIndex, 0] = round(mseDispETableA, 4)
                mseDispTable[dataIndex, 1] = round(mseDispETableC, 4)
                mseDispTable[dataIndex, 2] = round(mseDispTTableA, 4)
                mseDispTable[dataIndex, 3] = round(mseDispTTableC, 4)
                mseQTable[dataIndex, 0] = round(mseQETableA, 4)
                mseQTable[dataIndex, 1] = round(mseQETableC, 4)
                mseQTable[dataIndex, 2] = round(mseQTTableA, 4)
                mseQTable[dataIndex, 3] = round(mseQTTableC, 4)
                mseI2Table[dataIndex, 0] = round(mseI2ETableA, 4)
                mseI2Table[dataIndex, 1] = round(mseI2ETableC, 4)
                mseI2Table[dataIndex, 2] = round(mseI2TTableA, 4)
                mseI2Table[dataIndex, 3] = round(mseI2TTableC, 4)
                mseCentralTable[dataIndex, 0] = round(mseCentralTableA, 4)
                mseCentralTable[dataIndex, 1] = round(mseCentralTableC, 4)

                mseDispTable[dataIndex, 4] = round(np.divide(mseDispETableA, mseDispETableC), 4)
                mseDispTable[dataIndex, 5] = round(np.divide(mseDispTTableA, mseDispTTableC), 4)
                mseDispTable[dataIndex, 6] = round(np.divide(mseDispETableA, mseDispTTableA), 4)
                mseDispTable[dataIndex, 7] = round(np.divide(mseDispETableC, mseDispTTableC), 4)
                mseQTable[dataIndex, 4] = round(np.divide(mseQETableA, mseQETableC), 4)
                mseQTable[dataIndex, 5] = round(np.divide(mseQTTableA, mseQTTableC), 4)
                mseQTable[dataIndex, 6] = round(np.divide(mseQETableA, mseQTTableA), 4)
                mseQTable[dataIndex, 7] = round(np.divide(mseQETableC, mseQTTableC), 4)
                mseI2Table[dataIndex, 4] = round(np.divide(mseI2ETableA, mseI2ETableC), 4)
                mseI2Table[dataIndex, 5] = round(np.divide(mseI2TTableA, mseI2TTableC), 4)
                mseI2Table[dataIndex, 6] = round(np.divide(mseI2ETableA, mseI2TTableA), 4)
                mseI2Table[dataIndex, 7] = round(np.divide(mseI2ETableC, mseI2TTableC), 4)
                mseCentralTable[dataIndex, 2] = round(np.divide(mseCentralTableA, mseCentralTableC), 4)

                stdDispTable[dataIndex, 0] = round(np.std(mseDispETableATemp), 4)
                stdDispTable[dataIndex, 1] = round(np.std(mseDispETableCTemp), 4)
                stdDispTable[dataIndex, 2] = round(np.std(mseDispTTableATemp), 4)
                stdDispTable[dataIndex, 3] = round(np.std(mseDispTTableCTemp), 4)
                stdQTable[dataIndex, 0] = round(np.std(mseQETableATemp), 4)
                stdQTable[dataIndex, 1] = round(np.std(mseQETableCTemp), 4)
                stdQTable[dataIndex, 2] = round(np.std(mseQTTableATemp), 4)
                stdQTable[dataIndex, 3] = round(np.std(mseQTTableCTemp), 4)
                stdI2Table[dataIndex, 0] = round(np.std(mseI2ETableATemp), 4)
                stdI2Table[dataIndex, 1] = round(np.std(mseI2ETableCTemp), 4)
                stdI2Table[dataIndex, 2] = round(np.std(mseI2TTableATemp), 4)
                stdI2Table[dataIndex, 3] = round(np.std(mseI2TTableCTemp), 4)
                stdCentralTable[dataIndex, 0] = round(np.std(mseCTableATemp), 4)
                stdCentralTable[dataIndex, 1] = round(np.std(mseCTableCTemp), 4)

            mseDispEPlotA[fi, val] = np.mean(mseDispEPlotATemp[fi, val])
            mseQEPlotA[fi, val] = np.mean(mseQEPlotATemp[fi, val])
            mseI2EPlotA[fi, val] = np.mean(mseI2EPlotATemp[fi, val])
            stdDispEPlotA[fi, val] = np.std(mseDispEPlotATemp[fi, val])
            stdQEPlotA[fi, val] = np.std(mseQEPlotATemp[fi, val])
            stdI2EPlotA[fi, val] = np.std(mseI2EPlotATemp[fi, val])

    # EXPERIMENT 1: COMPARISON OF AGM/CGM, EMSE/TMSE AND CMSE
    if idx == 1:
        DispTable = PrettyTable(["Dispersion", "AGM", "CGM", "SD AGM", "SD CGM"])
        DispTable.add_row(["Cifar-10", "", "", "", ""])
        DispTable.add_row(["EMSE", mseDispTable[0, 0], mseDispTable[0, 1], stdDispTable[0, 0], stdDispTable[0, 1]])
        DispTable.add_row(["TMSE", mseDispTable[0, 2], mseDispTable[0, 3], stdDispTable[0, 2], stdDispTable[0, 3]])
        DispTable.add_row(["CMSE", mseCentralTable[0, 0], mseCentralTable[0, 1], stdCentralTable[0, 0], stdCentralTable[0, 1]])
        DispTable.add_row(["Cifar-100", "", "", "", ""])
        DispTable.add_row(["EMSE", mseDispTable[1, 0], mseDispTable[1, 1], stdDispTable[1, 0], stdDispTable[1, 1]])
        DispTable.add_row(["TMSE", mseDispTable[1, 2], mseDispTable[1, 3], stdDispTable[1, 2], stdDispTable[1, 3]])
        DispTable.add_row(["CMSE", mseCentralTable[1, 0], mseCentralTable[1, 1], stdCentralTable[1, 0], stdCentralTable[1, 1]])
        DispTable.add_row(["Fashion-MNIST", "", "", "", ""])
        DispTable.add_row(["EMSE", mseDispTable[2, 0], mseDispTable[2, 1], stdDispTable[2, 0], stdDispTable[2, 1]])
        DispTable.add_row(["TMSE", mseDispTable[2, 2], mseDispTable[2, 3], stdDispTable[2, 2], stdDispTable[2, 3]])
        DispTable.add_row(["CMSE", mseCentralTable[2, 0], mseCentralTable[2, 1], stdCentralTable[2, 0], stdCentralTable[2, 1]])
        
        DispData = DispTable.get_string()
        with open("Table_1_disp.txt", "w") as table1:
            table1.write(DispData)

        QTable = PrettyTable(["Q", "AGM", "CGM", "SD AGM", "SD CGM"])
        QTable.add_row(["Cifar-10", "", "", "", ""])
        QTable.add_row(["EMSE", mseQTable[0, 0], mseQTable[0, 1], stdQTable[0, 0], stdQTable[0, 1]])
        QTable.add_row(["TMSE", mseQTable[0, 2], mseQTable[0, 3], stdQTable[0, 2], stdQTable[0, 3]])
        QTable.add_row(["CMSE", mseCentralTable[0, 0], mseCentralTable[0, 1], stdCentralTable[0, 0], stdCentralTable[0, 1]])
        QTable.add_row(["Cifar-100", "", "", "", ""])
        QTable.add_row(["EMSE", mseQTable[1, 0], mseQTable[1, 1], stdQTable[1, 0], stdQTable[1, 1]])
        QTable.add_row(["TMSE", mseQTable[1, 2], mseQTable[1, 3], stdQTable[1, 2], stdQTable[1, 3]])
        QTable.add_row(["CMSE", mseCentralTable[1, 0], mseCentralTable[1, 1], stdCentralTable[1, 0], stdCentralTable[1, 1]])
        QTable.add_row(["Fashion-MNIST", "", "", "", ""])
        QTable.add_row(["EMSE", mseQTable[2, 0], mseQTable[2, 1], stdQTable[2, 0], stdQTable[2, 1]])
        QTable.add_row(["TMSE", mseQTable[2, 2], mseQTable[2, 3], stdQTable[2, 2], stdQTable[2, 3]])
        QTable.add_row(["CMSE", mseCentralTable[2, 0], mseCentralTable[2, 1], stdCentralTable[2, 0], stdCentralTable[2, 1]])
        
        QData = QTable.get_string()
        with open("Table_2_q.txt", "w") as table2:
            table2.write(QData)

        I2Table = PrettyTable(["I\u00B2", "AGM", "CGM", "SD AGM", "SD CGM"])
        I2Table.add_row(["Cifar-10", "", "", "", ""])
        I2Table.add_row(["EMSE", mseI2Table[0, 0], mseI2Table[0, 1], stdI2Table[0, 0], stdI2Table[0, 1]])
        I2Table.add_row(["TMSE", mseI2Table[0, 2], mseI2Table[0, 3], stdI2Table[0, 2], stdI2Table[0, 3]])
        I2Table.add_row(["CMSE", mseCentralTable[0, 0], mseCentralTable[0, 1], stdCentralTable[0, 0], stdCentralTable[0, 1]])
        I2Table.add_row(["Cifar-100", "", "", "", ""])
        I2Table.add_row(["EMSE", mseI2Table[1, 0], mseI2Table[1, 1], stdI2Table[1, 0], stdI2Table[1, 1]])
        I2Table.add_row(["TMSE", mseI2Table[1, 2], mseI2Table[1, 3], stdI2Table[1, 2], stdI2Table[1, 3]])
        I2Table.add_row(["CMSE", mseCentralTable[1, 0], mseCentralTable[1, 1], stdCentralTable[1, 0], stdCentralTable[1, 1]])
        I2Table.add_row(["Fashion-MNIST", "", "", "", ""])
        I2Table.add_row(["EMSE", mseI2Table[2, 0], mseI2Table[2, 1], stdI2Table[2, 0], stdI2Table[2, 1]])
        I2Table.add_row(["TMSE", mseI2Table[2, 2], mseI2Table[2, 3], stdI2Table[2, 2], stdI2Table[2, 3]])
        I2Table.add_row(["CMSE", mseCentralTable[2, 0], mseCentralTable[2, 1], stdCentralTable[2, 0], stdCentralTable[2, 1]])
        
        I2Data = I2Table.get_string()
        with open("Table_3_i2.txt", "w") as table3:
            table3.write(I2Data)

        ACTable = PrettyTable(["AGM/CGM", "Dispersion", "Q", "I\u00B2"])
        ACTable.add_row(["Cifar-10", "", "", ""])
        ACTable.add_row(["EMSE", mseDispTable[0, 0], mseQTable[0, 0], mseI2Table[0, 0]])
        ACTable.add_row(["TMSE", mseDispTable[0, 1], mseQTable[0, 1], mseI2Table[0, 1]])
        ACTable.add_row(["CMSE", mseCentralTable[0, 2], mseCentralTable[0, 2], mseCentralTable[0, 2]])
        ACTable.add_row(["Cifar-100", "", "", ""])
        ACTable.add_row(["EMSE", mseDispTable[1, 0], mseQTable[1, 0], mseI2Table[1, 0]])
        ACTable.add_row(["TMSE", mseDispTable[1, 1], mseQTable[1, 1], mseI2Table[1, 1]])
        ACTable.add_row(["CMSE", mseCentralTable[1, 2], mseCentralTable[1, 2], mseCentralTable[1, 2]])
        ACTable.add_row(["Fashion-MNIST", "", "", ""])
        ACTable.add_row(["EMSE", mseDispTable[2, 0], mseQTable[2, 0], mseI2Table[2, 0]])
        ACTable.add_row(["TMSE", mseDispTable[2, 1], mseQTable[2, 1], mseI2Table[2, 1]])
        ACTable.add_row(["CMSE", mseCentralTable[2, 2], mseCentralTable[2, 2], mseCentralTable[2, 2]])
        
        ACData = ACTable.get_string()
        with open("Table_4_ac.txt", "w") as table4:
            table4.write(ACData)

        ETTable = PrettyTable(["EMSE/TMSE", "Dispersion", "Q", "I\u00B2"])
        ETTable.add_row(["Cifar-10", "", "", ""])
        ETTable.add_row(["AGM", mseDispTable[0, 2], mseQTable[0, 2], mseI2Table[0, 2]])
        ETTable.add_row(["CGM", mseDispTable[0, 3], mseQTable[0, 3], mseI2Table[0, 3]])
        ETTable.add_row(["Cifar-100", "", "", ""])
        ETTable.add_row(["AGM", mseDispTable[1, 2], mseQTable[1, 2], mseI2Table[1, 2]])
        ETTable.add_row(["CGM", mseDispTable[1, 3], mseQTable[1, 3], mseI2Table[1, 3]])
        ETTable.add_row(["Fashion-MNIST", "", "", ""])
        ETTable.add_row(["AGM", mseDispTable[2, 2], mseQTable[2, 2], mseI2Table[2, 2]])
        ETTable.add_row(["CGM", mseDispTable[2, 3], mseQTable[2, 3], mseI2Table[2, 3]])
        
        ETData = ETTable.get_string()
        with open("Table_5_et.txt", "w") as table5:
            table5.write(ETData)

    # EXPERIMENT 3: STATISTICAL HETEROGENEITY
    def computePercLoss(a, b):
        percLoss = np.divide(np.subtract(a, b), a)*100
        return np.round(np.mean(percLoss), 4)

    percLossDispTable[0] = computePercLoss(mseDispEPlotA[4], mseDispEPlotA[5])
    percLossDispTable[1] = computePercLoss(mseDispEPlotA[2], mseDispEPlotA[3])
    percLossDispTable[2] = computePercLoss(mseDispEPlotA[0], mseDispEPlotA[1])
    percLossDispTable[3] = computePercLoss(mseDispEPlotA[0], mseDispEPlotA[2])
    percLossDispTable[4] = computePercLoss(mseDispEPlotA[2], mseDispEPlotA[4])
    percLossDispTable[5] = computePercLoss(mseDispEPlotA[1], mseDispEPlotA[3])
    percLossDispTable[6] = computePercLoss(mseDispEPlotA[3], mseDispEPlotA[5])

    percLossQTable[0] = computePercLoss(mseQEPlotA[4], mseQEPlotA[5])
    percLossQTable[1] = computePercLoss(mseQEPlotA[2], mseQEPlotA[3])
    percLossQTable[2] = computePercLoss(mseQEPlotA[0], mseQEPlotA[1])
    percLossQTable[3] = computePercLoss(mseQEPlotA[0], mseQEPlotA[2])
    percLossQTable[4] = computePercLoss(mseQEPlotA[2], mseQEPlotA[4])
    percLossQTable[5] = computePercLoss(mseQEPlotA[1], mseQEPlotA[3])
    percLossQTable[6] = computePercLoss(mseQEPlotA[3], mseQEPlotA[5])

    percLossI2Table[0] = computePercLoss(mseI2EPlotA[4], mseI2EPlotA[5])
    percLossI2Table[1] = computePercLoss(mseI2EPlotA[2], mseI2EPlotA[3])
    percLossI2Table[2] = computePercLoss(mseI2EPlotA[0], mseI2EPlotA[1])
    percLossI2Table[3] = computePercLoss(mseI2EPlotA[0], mseI2EPlotA[2])
    percLossI2Table[4] = computePercLoss(mseI2EPlotA[2], mseI2EPlotA[4])
    percLossI2Table[5] = computePercLoss(mseI2EPlotA[1], mseI2EPlotA[3])
    percLossI2Table[6] = computePercLoss(mseI2EPlotA[3], mseI2EPlotA[5])

    copyIndex = (2*dataIndex) + idx

    mseDisp[copyIndex] = np.copy(mseDispEPlotA)
    mseQ[copyIndex] = np.copy(mseQEPlotA)
    mseI2[copyIndex] = np.copy(mseI2EPlotA)
    stdDisp[copyIndex] = np.copy(stdDispEPlotA)
    stdQ[copyIndex] = np.copy(stdQEPlotA)
    stdI2[copyIndex] = np.copy(stdI2EPlotA)
    percLossDisp[copyIndex] = np.copy(percLossDispTable)
    percLossQ[copyIndex] = np.copy(percLossQTable)
    percLossI2[copyIndex] = np.copy(percLossI2Table)

def runLoopVaryEps(dataIndex, idx, dim, num, newImages, labels, GS):
    runLoop(dataIndex, idx, epsset, dim, num, -1, dtaconst, newImages, labels, GS)

def runLoopVaryDta(dataIndex, idx, dim, num, newImages, labels, GS):
    runLoop(dataIndex, idx, dtaset, dim, num, epsconst, -1, newImages, labels, GS)

runLoopVaryEps(0, 0, dimCifar, numCifar, newImagesCifar10, labelsCifar10, GSCifar)
runLoopVaryDta(0, 1, dimCifar, numCifar, newImagesCifar10, labelsCifar10, GSCifar)

runLoopVaryEps(1, 0, dimCifar, numCifar, newImagesCifar100, labelsCifar100, GSCifar)
runLoopVaryDta(1, 1, dimCifar, numCifar, newImagesCifar100, labelsCifar100, GSCifar)

runLoopVaryEps(2, 0, dimFashion, numFashion, newImagesFashion, labelsFashion, GSFashion)
runLoopVaryDta(2, 1, dimFashion, numFashion, newImagesFashion, labelsFashion, GSFashion)

for idx in range(2):

    PLTable1 = PrettyTable(["Privacy Loss", "Dispersion", "Q", "I\u00B2"])
    PLTable1.add_row(["Cifar-10", "", "", ""])
    PLTable1.add_row(["10 labels", percLossDisp[idx, 0], percLossQ[idx, 0], percLossI2[idx, 0]])
    PLTable1.add_row(["5 labels", percLossDisp[idx, 1], percLossQ[idx, 1], percLossI2[idx, 1]])
    PLTable1.add_row(["2 labels", percLossDisp[idx, 2], percLossQ[idx, 2], percLossI2[idx, 2]])
    PLTable1.add_row(["Cifar-100", "", "", ""])
    PLTable1.add_row(["10 labels", percLossDisp[2+idx, 0], percLossQ[2+idx, 0], percLossI2[2+idx, 0]])
    PLTable1.add_row(["5 labels", percLossDisp[2+idx, 1], percLossQ[2+idx, 1], percLossI2[2+idx, 1]])
    PLTable1.add_row(["2 labels", percLossDisp[2+idx, 2], percLossQ[2+idx, 2], percLossI2[2+idx, 2]])
    PLTable1.add_row(["Fashion-MNIST", "", "", ""])
    PLTable1.add_row(["10 labels", percLossDisp[4+idx, 0], percLossQ[4+idx, 0], percLossI2[4+idx, 0]])
    PLTable1.add_row(["5 labels", percLossDisp[4+idx, 1], percLossQ[4+idx, 1], percLossI2[4+idx, 1]])
    PLTable1.add_row(["2 labels", percLossDisp[4+idx, 2], percLossQ[4+idx, 2], percLossI2[4+idx, 2]])

    PLData1 = PLTable1.get_string()
    with open("Table_6_" + "%s" % parset[idx] + "_pl1.txt", "w") as table6:
        table6.write(PLData1)

    PLTable2 = PrettyTable(["Privacy Loss", "Dispersion", "Q", "I\u00B2"])
    PLTable2.add_row(["Cifar-10", "", "", ""])
    PLTable2.add_row(["SH: 10v5", percLossDisp[idx, 3], percLossQ[idx, 3], percLossI2[idx, 3]])
    PLTable2.add_row(["SH: 5v2", percLossDisp[idx, 4], percLossQ[idx, 4], percLossI2[idx, 4]])
    PLTable2.add_row(["Non-SH: 10v5", percLossDisp[idx, 5], percLossQ[idx, 5], percLossI2[idx, 5]])
    PLTable2.add_row(["Non-SH: 5v2", percLossDisp[idx, 6], percLossQ[idx, 6], percLossI2[idx, 6]])
    PLTable2.add_row(["Cifar-100", "", "", ""])
    PLTable2.add_row(["SH: 10v5", percLossDisp[2+idx, 3], percLossQ[2+idx, 3], percLossI2[2+idx, 3]])
    PLTable2.add_row(["SH: 5v2", percLossDisp[2+idx, 4], percLossQ[2+idx, 4], percLossI2[2+idx, 4]])
    PLTable2.add_row(["Non-SH: 10v5", percLossDisp[2+idx, 5], percLossQ[2+idx, 5], percLossI2[2+idx, 5]])
    PLTable2.add_row(["Non-SH: 5v2", percLossDisp[2+idx, 6], percLossQ[2+idx, 6], percLossI2[2+idx, 6]])
    PLTable2.add_row(["Fashion-MNIST", "", "", ""])
    PLTable2.add_row(["SH: 10v5", percLossDisp[4+idx, 3], percLossQ[4+idx, 3], percLossI2[4+idx, 3]])
    PLTable2.add_row(["SH: 5v2", percLossDisp[4+idx, 4], percLossQ[4+idx, 4], percLossI2[4+idx, 4]])
    PLTable2.add_row(["Non-SH: 10v5", percLossDisp[4+idx, 5], percLossQ[4+idx, 5], percLossI2[4+idx, 5]])
    PLTable2.add_row(["Non-SH: 5v2", percLossDisp[4+idx, 6], percLossQ[4+idx, 6], percLossI2[4+idx, 6]])

    PLEpsData2 = PLTable2.get_string()
    with open("Table_7_" + "%s" % parset[idx] + "_pl2.txt", "w") as table7:
        table7.write(PLEpsData2)

    if idx == 0:
        varset = epsset
    else:
        varset = dtaset

    for li in range(3):
        eq = 2*li
        uneq = (2*li) + 1

        plt.errorbar(varset, mseDisp[idx, eq], yerr = np.minimum(stdDisp[idx, eq], np.sqrt(mseDisp[idx, eq]), np.divide(mseDisp[idx, eq], 2)), color = 'blue', marker = 'o', label = f"{cifarset[0]}: equal")
        plt.errorbar(varset, mseDisp[idx, uneq], yerr = np.minimum(stdDisp[idx, uneq], np.sqrt(mseDisp[idx, uneq]), np.divide(mseDisp[idx, uneq], 2)), color = 'blueviolet', marker = 'x', label = f"{cifarset[0]}: unequal")
        plt.errorbar(varset, mseDisp[2+idx, eq], yerr = np.minimum(stdDisp[2+idx, eq], np.sqrt(mseDisp[2+idx, eq]), np.divide(mseDisp[2+idx, eq], 2)), color = 'green', marker = 'o', label = f"{cifarset[1]}: equal")
        plt.errorbar(varset, mseDisp[2+idx, uneq], yerr = np.minimum(stdDisp[2+idx, uneq], np.sqrt(mseDisp[2+idx, uneq]), np.divide(mseDisp[2+idx, uneq], 2)), color = 'lime', marker = 'x', label = f"{cifarset[1]}: unequal")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[idx])
        plt.ylabel("EMSE of Gaussian Mechanism")
        plt.savefig("Graph_Cifars_" + "%s" % labelset[li] + "_vary_" + "%s" % parset[idx] + "_disp.png")
        plt.clf()

        plt.errorbar(varset, mseQ[idx, eq], yerr = np.minimum(stdQ[idx, eq], np.sqrt(mseQ[idx, eq]), np.divide(mseQ[idx, eq], 2)), color = 'blue', marker = 'o', label = f"{cifarset[0]}: equal")
        plt.errorbar(varset, mseQ[idx, uneq], yerr = np.minimum(stdQ[idx, uneq], np.sqrt(mseQ[idx, uneq]), np.divide(mseQ[idx, uneq], 2)), color = 'blueviolet', marker = 'x', label = f"{cifarset[0]}: unequal")
        plt.errorbar(varset, mseQ[2+idx, eq], yerr = np.minimum(stdQ[2+idx, eq], np.sqrt(mseQ[2+idx, eq]), np.divide(mseQ[2+idx, eq], 2)), color = 'green', marker = 'o', label = f"{cifarset[1]}: equal")
        plt.errorbar(varset, mseQ[2+idx, uneq], yerr = np.minimum(stdQ[2+idx, uneq], np.sqrt(mseQ[2+idx, uneq]), np.divide(mseQ[2+idx, uneq], 2)), color = 'lime', marker = 'x', label = f"{cifarset[1]}: unequal")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[idx])
        plt.ylabel("EMSE of Gaussian Mechanism")
        plt.savefig("Graph_Cifars_" + "%s" % labelset[li] + "_vary_" + "%s" % parset[idx] + "_q.png")
        plt.clf()

        plt.errorbar(varset, mseI2[idx, eq], yerr = np.minimum(stdI2[idx, eq], np.sqrt(mseI2[idx, eq]), np.divide(mseI2[idx, eq], 2)), color = 'blue', marker = 'o', label = f"{cifarset[0]}: equal")
        plt.errorbar(varset, mseI2[idx, uneq], yerr = np.minimum(stdI2[idx, uneq], np.sqrt(mseI2[idx, uneq]), np.divide(mseI2[idx, uneq], 2)), color = 'blueviolet', marker = 'x', label = f"{cifarset[0]}: unequal")
        plt.errorbar(varset, mseI2[2+idx, eq], yerr = np.minimum(stdI2[2+idx, eq], np.sqrt(mseI2[2+idx, eq]), np.divide(mseI2[2+idx, eq], 2)), color = 'green', marker = 'o', label = f"{cifarset[1]}: equal")
        plt.errorbar(varset, mseI2[2+idx, uneq], yerr = np.minimum(stdI2[2+idx, uneq], np.sqrt(mseI2[2+idx, uneq]), np.divide(mseI2[2+idx, uneq], 2)), color = 'lime', marker = 'x', label = f"{cifarset[1]}: unequal")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[idx])
        plt.ylabel("EMSE of Gaussian Mechanism")
        plt.savefig("Graph_Cifars_" + "%s" % labelset[li] + "_vary_" + "%s" % parset[idx] + "_i2.png")
        plt.clf()

        plt.errorbar(varset, mseDisp[4+idx, eq], yerr = np.minimum(stdDisp[4+idx, eq], np.sqrt(mseDisp[4+idx, eq]), np.divide(mseDisp[4+idx, eq], 2)), color = 'orange', marker = 'o', label = f"{cifarset[2]}: equal")
        plt.errorbar(varset, mseDisp[4+idx, uneq], yerr = np.minimum(stdDisp[4+idx, uneq], np.sqrt(mseDisp[4+idx, uneq]), np.divide(mseDisp[4+idx, uneq], 2)), color = 'gold', marker = 'x', label = f"{cifarset[2]}: unequal")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[idx])
        plt.ylabel("EMSE of Gaussian Mechanism")
        plt.savefig("Graph_Fashion_" + "%s" % labelset[li] + "_vary_" + "%s" % parset[idx] + "_disp.png")
        plt.clf()

        plt.errorbar(varset, mseQ[4+idx, eq], yerr = np.minimum(stdQ[4+idx, eq], np.sqrt(mseQ[4+idx, eq]), np.divide(mseQ[4+idx, eq], 2)), color = 'orange', marker = 'o', label = f"{cifarset[2]}: equal")
        plt.errorbar(varset, mseQ[4+idx, uneq], yerr = np.minimum(stdQ[4+idx, uneq], np.sqrt(mseQ[4+idx, uneq]), np.divide(mseQ[4+idx, uneq], 2)), color = 'gold', marker = 'x', label = f"{cifarset[2]}: unequal")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[idx])
        plt.ylabel("EMSE of Gaussian Mechanism")
        plt.savefig("Graph_Fashion_" + "%s" % labelset[li] + "_vary_" + "%s" % parset[idx] + "_q.png")
        plt.clf()

        plt.errorbar(varset, mseI2[4+idx, eq], yerr = np.minimum(stdI2[4+idx, eq], np.sqrt(mseI2[4+idx, eq]), np.divide(mseI2[4+idx, eq], 2)), color = 'orange', marker = 'o', label = f"{cifarset[2]}: equal")
        plt.errorbar(varset, mseI2[4+idx, uneq], yerr = np.minimum(stdI2[4+idx, uneq], np.sqrt(mseI2[4+idx, uneq]), np.divide(mseI2[4+idx, uneq], 2)), color = 'gold', marker = 'x', label = f"{cifarset[2]}: unequal")
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.xlabel("Value of " + "%s" % graphset[idx])
        plt.ylabel("EMSE of Gaussian Mechanism")
        plt.savefig("Graph_Fashion_" + "%s" % labelset[li] + "_vary_" + "%s" % parset[idx] + "_i2.png")
        plt.clf()

print("Finished.\n")