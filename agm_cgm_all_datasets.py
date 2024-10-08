import numpy as np
import idx2numpy
import mpmath as mp
import matplotlib.pyplot as plt

from math import erf
from numpy.random import normal
from prettytable import PrettyTable

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
cifarset = np.array(['Cifar_10', 'Cifar_100', 'Fashion'])
parset = np.array(['eps', 'dta'])
graphset = np.array(['$\mathit{\u03b5}$', '$\mathit{\u03b4}$'])
labelset = np.array(['10 labels', '5 labels', '2 labels'])
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
percChangeDisp = np.zeros((2*C, T2))
percChangeQ = np.zeros((2*C, T2))
percChangeI2 = np.zeros((2*C, T2))

mseDispETableA = np.zeros(2*R)
mseDispETableC = np.zeros(2*R)
mseDispTTableA = np.zeros(2*R)
mseDispTTableC = np.zeros(2*R)
mseQETableA = np.zeros(2*R)
mseQETableC = np.zeros(2*R)
mseQTTableA = np.zeros(2*R)
mseQTTableC = np.zeros(2*R)
mseI2ETableA = np.zeros(2*R)
mseI2ETableC = np.zeros(2*R)
mseI2TTableA = np.zeros(2*R)
mseI2TTableC = np.zeros(2*R)
mseCTableA = np.zeros(2*R)
mseCTableC = np.zeros(2*R)

def runLoop(dataIndex, idx, varset, dim, num, eps, dta, newImages, labels, GS):

    V = len(varset)

    mseDispEPlotA = np.zeros((2*L, V))
    mseQEPlotA = np.zeros((2*L, V))
    mseI2EPlotA = np.zeros((2*L, V))
    stdDispEPlotA = np.zeros((2*L, V))
    stdQEPlotA = np.zeros((2*L, V))
    stdI2EPlotA = np.zeros((2*L, V))
    percChangeDispTable = np.zeros(T2)
    percChangeQTable = np.zeros(T2)
    percChangeI2Table = np.zeros(T2)

    for val in range(len(varset)):

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
            wVector = np.var(imageArray, axis = 1)
            weight = np.zeros(sampleSize)
            wImageArray = np.zeros((sampleSize, dim))

            for j in range(0, sampleSize):
                weight[j] = np.divide(1.0, wVector[j])

                # MULTIPLYING EACH VECTOR BY ITS CORRESPONDING WEIGHTED MEAN
                wImageArray[j] = np.multiply(weight[j], imageArray[j])

            mu = np.mean(imageArray, axis = 0)
            wSumMu = np.mean(wImageArray, axis = 0)

            # DIVIDING SUM OF WEIGHTED VECTORS BY SUM OF WEIGHTS
            sumWeight = np.sum(weight)
            wMu = np.divide(wSumMu, sumWeight)

            noisyMu = np.zeros(dim)
            wNoisyMu = np.zeros(dim)

            mseEList = np.zeros(sampleSize)
            mseQEList = np.zeros(sampleSize)
            mseTList = np.zeros(sampleSize, dtype = np.float64)
            mseQTList = np.zeros(sampleSize, dtype = np.float64)
            I2TrueDenom = np.zeros(sampleSize)
            noisyQ = np.zeros(sampleSize)

            # ADDING FIRST NOISE TERM TO MU DERIVED FROM GAUSSIAN DISTRIBUTION WITH MEAN 0 AND VARIANCE SIGMA SQUARED
            for i in range(0, dim):
                xi1 = normal(0, sigma**2)
                noisyMu[i] = np.add(mu[i], xi1)
                wNoisyMu[i] = np.add(wMu[i], xi1)

            # FIRST SUBTRACTION BETWEEN CIFAR-10 VECTOR OF EACH CLIENT AND NOISY MEAN ACCORDING TO THEOREM FOR DISPERSION
            for j in range(0, sampleSize):
                trueDiff = np.mean(np.subtract(imageArray[j], mu))
                wTrueDiff = np.mean(np.subtract(imageArray[j], wMu))
                noisyDiff = np.mean(np.subtract(imageArray[j], noisyMu))
                wNoisyDiff = np.mean(np.subtract(imageArray[j], wNoisyMu))

                # INCORPORATING WEIGHTS FOR STATISTICS ON Q
                trueDisp = np.power(trueDiff, 2)
                wTrueDisp = np.power(wTrueDiff, 2)
                I2TrueDenom[j] = np.multiply(weight[j], wTrueDisp)
                noisyVar = np.power(noisyDiff, 2)
                wNoisyVar = np.power(wNoisyDiff, 2)
                weightedNoisyVar = np.multiply(weight[j], wNoisyVar)

                xi2 = normal(0, sigma**2)
                noisyDisp = np.add(noisyVar, xi2)
                noisyQ[j] = np.add(weightedNoisyVar, xi2)

                # EMSE = MSE OF FORMULA OF DISPERSION OR Q
                mseEList[j] = np.subtract(noisyDisp, trueDisp)
                mseQEList[j] = np.subtract(noisyQ[j], I2TrueDenom[j])

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
                mseTList[j] = np.mean(extraTerm)
                mseQTList[j] = np.mean(wExtraTerm) 

            wTDSum = np.mean(I2TrueDenom)
            noisyQSum = np.mean(noisyQ)
            mseE = np.power(np.mean(mseEList), 2)
            mseT = np.power(np.mean(mseTList), 2)
            mseQE = np.power(np.mean(mseQEList), 2)
            mseQT = np.power(np.mean(mseQTList), 2)
            doublerep = (R*idx) + rep

            if ACindex == 0:

                # TABLES ASSUME UNIFORM DATA
                if fi == 0 and val == 0:
                    mseDispETableA[doublerep] = mseE
                    mseDispTTableA[doublerep] = mseT
                    mseQETableA[doublerep] = mseQE
                    mseQTTableA[doublerep] = mseQT

                # STATISTICAL HETEROGENEITY GRAPHS
                mseDispEPlotATemp[fi, val, rep] = mseE
                mseQEPlotATemp[fi, val, rep] = mseQE

            else:
                if fi == 0 and val == 0:
                    mseDispETableC[doublerep] = mseE
                    mseDispTTableC[doublerep] = mseT
                    mseQETableC[doublerep] = mseQE
                    mseQTTableC[doublerep] = mseQT

            # COMPUTE I^2'' and I^2 USING SIMPLE FORMULA AT BOTTOM OF LEMMA 6.2
            trueI2Prep = np.divide(sampleSize-1, wTDSum)
            trueI2 = np.subtract(1, trueI2Prep)
            I2Prep = np.divide(sampleSize-1, noisyQSum)
            I2Noise = np.subtract(1, I2Prep)

            # ADD THIRD NOISE TERM BASED ON LEMMA 6.2
            xi3 = normal(0, sigma**2)
            noisyI2 = np.add(I2Noise, xi3)

            mseI2EList = np.zeros(sampleSize)
            mseI2TList = np.zeros(sampleSize)

            # COMPUTE EMSE AND TMSE
            for j in range(0, sampleSize):
                diffEI2 = np.subtract(noisyI2, trueI2)
                mseI2EList[j] = np.divide(diffEI2, dim)
                diffTI2Prep = np.subtract(xi3, I2Noise)
                diffTI2 = np.add(diffTI2Prep, trueI2)
                mseI2TList[j] = np.divide(diffTI2, dim)

            mseI2E = np.power(np.mean(mseI2EList), 2)
            mseI2T = np.power(np.mean(mseI2TList), 2)

            # EXPERIMENT 2: WHAT IS THE COST OF A DISTRIBUTED SETTING?
            xiCentral = normal(0, centralSigma**2)
            mseC = xiCentral**2

            if ACindex == 0:
                if fi == 0 and val == 0:
                    mseI2ETableA[doublerep] = mseI2E
                    mseI2TTableA[doublerep] = mseI2T
                    mseCTableA[doublerep] = mseC

                mseI2EPlotATemp[fi, val, rep] = mseI2E

            else:
                if fi == 0 and val == 0:
                    mseI2ETableC[doublerep] = mseI2E
                    mseI2TTableC[doublerep] = mseI2T
                    mseCTableC[doublerep] = mseC

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
                mseDispTable[dataIndex, 0] = np.mean(mseDispETableA)
                mseDispTable[dataIndex, 1] = np.mean(mseDispETableC)
                mseDispTable[dataIndex, 2] = np.mean(mseDispTTableA)
                mseDispTable[dataIndex, 3] = np.mean(mseDispTTableC)
                mseQTable[dataIndex, 0] = np.mean(mseQETableA)
                mseQTable[dataIndex, 1] = np.mean(mseQETableC)
                mseQTable[dataIndex, 2] = np.mean(mseQTTableA)
                mseQTable[dataIndex, 3] = np.mean(mseQTTableC)
                mseI2Table[dataIndex, 0] = np.mean(mseI2ETableA)
                mseI2Table[dataIndex, 1] = np.mean(mseI2ETableC)
                mseI2Table[dataIndex, 2] = np.mean(mseI2TTableA)
                mseI2Table[dataIndex, 3] = np.mean(mseI2TTableC)
                mseCentralTable[dataIndex, 0] = np.mean(mseCTableA)
                mseCentralTable[dataIndex, 1] = np.mean(mseCTableC)

                mseDispTable[dataIndex, 4] = np.divide(mseDispTable[dataIndex, 0], mseDispTable[dataIndex, 1])
                mseDispTable[dataIndex, 5] = np.divide(mseDispTable[dataIndex, 2], mseDispTable[dataIndex, 3])
                mseDispTable[dataIndex, 6] = np.divide(mseDispTable[dataIndex, 0], mseDispTable[dataIndex, 2])
                mseDispTable[dataIndex, 7] = np.divide(mseDispTable[dataIndex, 1], mseDispTable[dataIndex, 3])
                mseQTable[dataIndex, 4] = np.divide(mseQTable[dataIndex, 0], mseQTable[dataIndex, 1])
                mseQTable[dataIndex, 5] = np.divide(mseQTable[dataIndex, 2], mseQTable[dataIndex, 3])
                mseQTable[dataIndex, 6] = np.divide(mseQTable[dataIndex, 0], mseQTable[dataIndex, 2])
                mseQTable[dataIndex, 7] = np.divide(mseQTable[dataIndex, 1], mseQTable[dataIndex, 3])
                mseI2Table[dataIndex, 4] = np.divide(mseI2Table[dataIndex, 0], mseI2Table[dataIndex, 1])
                mseI2Table[dataIndex, 5] = np.divide(mseI2Table[dataIndex, 2], mseI2Table[dataIndex, 3])
                mseI2Table[dataIndex, 6] = np.divide(mseI2Table[dataIndex, 0], mseI2Table[dataIndex, 2])
                mseI2Table[dataIndex, 7] = np.divide(mseI2Table[dataIndex, 1], mseI2Table[dataIndex, 3])
                mseCentralTable[dataIndex, 2] = np.divide(mseCentralTable[dataIndex, 0], mseCentralTable[dataIndex, 1])

                stdDispTable[dataIndex, 0] = np.std(mseDispETableA)
                stdDispTable[dataIndex, 1] = np.std(mseDispETableC)
                stdDispTable[dataIndex, 2] = np.std(mseDispTTableA)
                stdDispTable[dataIndex, 3] = np.std(mseDispTTableC)
                stdQTable[dataIndex, 0] = np.std(mseQETableA)
                stdQTable[dataIndex, 1] = np.std(mseQETableC)
                stdQTable[dataIndex, 2] = np.std(mseQTTableA)
                stdQTable[dataIndex, 3] = np.std(mseQTTableC)
                stdI2Table[dataIndex, 0] = np.std(mseI2ETableA)
                stdI2Table[dataIndex, 1] = np.std(mseI2ETableC)
                stdI2Table[dataIndex, 2] = np.std(mseI2TTableA)
                stdI2Table[dataIndex, 3] = np.std(mseI2TTableC)
                stdCentralTable[dataIndex, 0] = np.std(mseCTableA)
                stdCentralTable[dataIndex, 1] = np.std(mseCTableC)

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
        DispTable.add_row(["EMSE", "%.4f" % mseDispTable[0, 0], "%.4f" % mseDispTable[0, 1], "%.4f" % stdDispTable[0, 0], "%.4f" % stdDispTable[0, 1]])
        DispTable.add_row(["TMSE", "%.4f" % mseDispTable[0, 2], "%.4f" % mseDispTable[0, 3], "%.4f" % stdDispTable[0, 2], "%.4f" % stdDispTable[0, 3]])
        DispTable.add_row(["CMSE", "%.4f" % mseCentralTable[0, 0], "%d" % mseCentralTable[0, 1], "%.4f" % stdCentralTable[0, 0], "%d" % stdCentralTable[0, 1]])
        DispTable.add_row(["Cifar-100", "", "", "", ""])
        DispTable.add_row(["EMSE", "%.4f" % mseDispTable[1, 0], "%.4f" % mseDispTable[1, 1], "%.4f" % stdDispTable[1, 0], "%.4f" % stdDispTable[1, 1]])
        DispTable.add_row(["TMSE", "%.4f" % mseDispTable[1, 2], "%.4f" % mseDispTable[1, 3], "%.4f" % stdDispTable[1, 2], "%.4f" % stdDispTable[1, 3]])
        DispTable.add_row(["CMSE", "%.4f" % mseCentralTable[1, 0], "%d" % mseCentralTable[1, 1], "%.4f" % stdCentralTable[1, 0], "%d" % stdCentralTable[1, 1]])
        DispTable.add_row(["Fashion-MNIST", "", "", "", ""])
        DispTable.add_row(["EMSE", "%.4f" % mseDispTable[2, 0], "%.4f" % mseDispTable[2, 1], "%.4f" % stdDispTable[2, 0], "%.4f" % stdDispTable[2, 1]])
        DispTable.add_row(["TMSE", "%.4f" % mseDispTable[2, 2], "%.4f" % mseDispTable[2, 3], "%.4f" % stdDispTable[2, 2], "%.4f" % stdDispTable[2, 3]])
        DispTable.add_row(["CMSE", "%.4f" % mseCentralTable[2, 0], "%d" % mseCentralTable[2, 1], "%.4f" % stdCentralTable[2, 0], "%d" % stdCentralTable[2, 1]])
        
        DispData = DispTable.get_string()
        with open("Table_1_disp.txt", "w") as table1:
            table1.write(DispData)

        QTable = PrettyTable(["Q", "AGM", "CGM", "SD AGM", "SD CGM"])
        QTable.add_row(["Cifar-10", "", "", "", ""])
        QTable.add_row(["EMSE", "%.2f" % mseQTable[0, 0], "%.2f" % mseQTable[0, 1], "%.2f" % stdQTable[0, 0], "%.2f" % stdQTable[0, 1]])
        QTable.add_row(["TMSE", "%.2f" % mseQTable[0, 2], "%.2f" % mseQTable[0, 3], "%.2f" % stdQTable[0, 2], "%.2f" % stdQTable[0, 3]])
        QTable.add_row(["CMSE", "%.2f" % mseCentralTable[0, 0], "%d" % mseCentralTable[0, 1], "%.2f" % stdCentralTable[0, 0], "%d" % stdCentralTable[0, 1]])
        QTable.add_row(["Cifar-100", "", "", "", ""])
        QTable.add_row(["EMSE", "%.2f" % mseQTable[1, 0], "%.2f" % mseQTable[1, 1], "%.2f" % stdQTable[1, 0], "%.2f" % stdQTable[1, 1]])
        QTable.add_row(["TMSE", "%.2f" % mseQTable[1, 2], "%.2f" % mseQTable[1, 3], "%.2f" % stdQTable[1, 2], "%.2f" % stdQTable[1, 3]])
        QTable.add_row(["CMSE", "%.2f" % mseCentralTable[1, 0], "%d" % mseCentralTable[1, 1], "%.2f" % stdCentralTable[1, 0], "%d" % stdCentralTable[1, 1]])
        QTable.add_row(["Fashion-MNIST", "", "", "", ""])
        QTable.add_row(["EMSE", "%.2f" % mseQTable[2, 0], "%.2f" % mseQTable[2, 1], "%.2f" % stdQTable[2, 0], "%.2f" % stdQTable[2, 1]])
        QTable.add_row(["TMSE", "%.2f" % mseQTable[2, 2], "%.2f" % mseQTable[2, 3], "%.2f" % stdQTable[2, 2], "%.2f" % stdQTable[2, 3]])
        QTable.add_row(["CMSE", "%.2f" % mseCentralTable[2, 0], "%d" % mseCentralTable[2, 1], "%.2f" % stdCentralTable[2, 0], "%d" % stdCentralTable[2, 1]])
        
        QData = QTable.get_string()
        with open("Table_2_q.txt", "w") as table2:
            table2.write(QData)

        I2Table = PrettyTable(["I\u00B2", "AGM", "CGM", "SD AGM", "SD CGM"])
        I2Table.add_row(["Cifar-10", "", "", "", ""])
        I2Table.add_row(["EMSE", "%.2f" % mseI2Table[0, 0], "%.2f" % mseI2Table[0, 1], "%.2f" % [0, 0], "%.2f" % stdI2Table[0, 1]])
        I2Table.add_row(["TMSE", "%.2f" % mseI2Table[0, 2], "%.2f" % mseI2Table[0, 3], "%.2f" % stdI2Table[0, 2], "%.2f" % stdI2Table[0, 3]])
        I2Table.add_row(["CMSE", "%.2f" % mseCentralTable[0, 0], "%d" % mseCentralTable[0, 1], "%.2f" % stdCentralTable[0, 0], "%d" % stdCentralTable[0, 1]])
        I2Table.add_row(["Cifar-100", "", "", "", ""])
        I2Table.add_row(["EMSE", "%.2f" % mseI2Table[1, 0], "%.2f" % mseI2Table[1, 1], "%.2f" % stdI2Table[1, 0], "%.2f" % stdI2Table[1, 1]])
        I2Table.add_row(["TMSE", "%.2f" % mseI2Table[1, 2], "%.2f" % mseI2Table[1, 3], "%.2f" % stdI2Table[1, 2], "%.2f" % stdI2Table[1, 3]])
        I2Table.add_row(["CMSE", "%.2f" % mseCentralTable[1, 0], "%d" % mseCentralTable[1, 1], "%.2f" % stdCentralTable[1, 0], "%d" % stdCentralTable[1, 1]])
        I2Table.add_row(["Fashion-MNIST", "", "", "", ""])
        I2Table.add_row(["EMSE", "%.2f" % mseI2Table[2, 0], "%.2f" % mseI2Table[2, 1], "%.2f" % stdI2Table[2, 0], "%.2f" % stdI2Table[2, 1]])
        I2Table.add_row(["TMSE", "%.2f" % mseI2Table[2, 2], "%.2f" % mseI2Table[2, 3], "%.2f" % stdI2Table[2, 2], "%.2f" % stdI2Table[2, 3]])
        I2Table.add_row(["CMSE", "%.2f" % mseCentralTable[2, 0], "%d" % mseCentralTable[2, 1], "%.2f" % stdCentralTable[2, 0], "%d" % stdCentralTable[2, 1]])
        
        I2Data = I2Table.get_string()
        with open("Table_3_i2.txt", "w") as table3:
            table3.write(I2Data)

        ACTable = PrettyTable(["AGM/CGM", "Dispersion", "Q", "I\u00B2"])
        ACTable.add_row(["Cifar-10", "", "", ""])
        ACTable.add_row(["EMSE", "%.4f" % mseDispTable[0, 4], "%.4f" % mseQTable[0, 4], "%.4f" % mseI2Table[0, 4]])
        ACTable.add_row(["TMSE", "%.4f" % mseDispTable[0, 5], "%.4f" % mseQTable[0, 5], "%.4f" % mseI2Table[0, 5]])
        ACTable.add_row(["CMSE", "%.4f" % mseCentralTable[0, 2], "%.4f" % mseCentralTable[0, 2], "%.4f" % mseCentralTable[0, 2]])
        ACTable.add_row(["Cifar-100", "", "", ""])
        ACTable.add_row(["EMSE", "%.4f" % mseDispTable[1, 4], "%.4f" % mseQTable[1, 4], "%.4f" % mseI2Table[1, 4]])
        ACTable.add_row(["TMSE", "%.4f" %  mseDispTable[1, 5], "%.4f" % mseQTable[1, 5], "%.4f" % mseI2Table[1, 5]])
        ACTable.add_row(["CMSE", "%.4f" % mseCentralTable[1, 2], "%.4f" % mseCentralTable[1, 2], "%.4f" % mseCentralTable[1, 2]])
        ACTable.add_row(["Fashion-MNIST", "", "", ""])
        ACTable.add_row(["EMSE", "%.4f" % mseDispTable[2, 4], "%.4f" % mseQTable[2, 4], "%.4f" % mseI2Table[2, 4]])
        ACTable.add_row(["TMSE", "%.4f" % mseDispTable[2, 5], "%.4f" % mseQTable[2, 5], "%.4f" % mseI2Table[2, 5]])
        ACTable.add_row(["CMSE", "%.4f" % mseCentralTable[2, 2], "%.4f" % mseCentralTable[2, 2], "%.4f" % mseCentralTable[2, 2]])
        
        ACData = ACTable.get_string()
        with open("Table_4_ac.txt", "w") as table4:
            table4.write(ACData)

        ETTable = PrettyTable(["EMSE/TMSE", "Dispersion", "Q", "I\u00B2"])
        ETTable.add_row(["Cifar-10", "", "", ""])
        ETTable.add_row(["AGM", "%.4f" % mseDispTable[0, 6], "%.4f" % mseQTable[0, 6], "%.4f" % mseI2Table[0, 6]])
        ETTable.add_row(["CGM", "%.4f" % mseDispTable[0, 7], "%.4f" % mseQTable[0, 7], "%.4f" % mseI2Table[0, 7]])
        ETTable.add_row(["Cifar-100", "", "", ""])
        ETTable.add_row(["AGM", "%.4f" % mseDispTable[1, 6], "%.4f" % mseQTable[1, 6], "%.4f" % mseI2Table[1, 6]])
        ETTable.add_row(["CGM", "%.4f" % mseDispTable[1, 7], "%.4f" % mseQTable[1, 7], "%.4f" % mseI2Table[1, 7]])
        ETTable.add_row(["Fashion-MNIST", "", "", ""])
        ETTable.add_row(["AGM", "%.4f" % mseDispTable[2, 6], "%.4f" % mseQTable[2, 6], "%.4f" % mseI2Table[2, 6]])
        ETTable.add_row(["CGM", "%.4f" % mseDispTable[2, 7], "%.4f" % mseQTable[2, 7], "%.4f" % mseI2Table[2, 7]])
        
        ETData = ETTable.get_string()
        with open("Table_5_et.txt", "w") as table5:
            table5.write(ETData)
    
    # EXPERIMENT 3: STATISTICAL HETEROGENEITY
    def computePercChange(a, b):
        percChange = np.divide(np.subtract(a, b), a)*100
        return np.mean(percChange)

    percChangeDispTable[0] = computePercChange(mseDispEPlotA[4], mseDispEPlotA[5])
    percChangeDispTable[1] = computePercChange(mseDispEPlotA[2], mseDispEPlotA[3])
    percChangeDispTable[2] = computePercChange(mseDispEPlotA[0], mseDispEPlotA[1])
    percChangeDispTable[3] = computePercChange(mseDispEPlotA[0], mseDispEPlotA[2])
    percChangeDispTable[4] = computePercChange(mseDispEPlotA[2], mseDispEPlotA[4])
    percChangeDispTable[5] = computePercChange(mseDispEPlotA[1], mseDispEPlotA[3])
    percChangeDispTable[6] = computePercChange(mseDispEPlotA[3], mseDispEPlotA[5])

    percChangeQTable[0] = computePercChange(mseQEPlotA[4], mseQEPlotA[5])
    percChangeQTable[1] = computePercChange(mseQEPlotA[2], mseQEPlotA[3])
    percChangeQTable[2] = computePercChange(mseQEPlotA[0], mseQEPlotA[1])
    percChangeQTable[3] = computePercChange(mseQEPlotA[0], mseQEPlotA[2])
    percChangeQTable[4] = computePercChange(mseQEPlotA[2], mseQEPlotA[4])
    percChangeQTable[5] = computePercChange(mseQEPlotA[1], mseQEPlotA[3])
    percChangeQTable[6] = computePercChange(mseQEPlotA[3], mseQEPlotA[5])

    percChangeI2Table[0] = computePercChange(mseI2EPlotA[4], mseI2EPlotA[5])
    percChangeI2Table[1] = computePercChange(mseI2EPlotA[2], mseI2EPlotA[3])
    percChangeI2Table[2] = computePercChange(mseI2EPlotA[0], mseI2EPlotA[1])
    percChangeI2Table[3] = computePercChange(mseI2EPlotA[0], mseI2EPlotA[2])
    percChangeI2Table[4] = computePercChange(mseI2EPlotA[2], mseI2EPlotA[4])
    percChangeI2Table[5] = computePercChange(mseI2EPlotA[1], mseI2EPlotA[3])
    percChangeI2Table[6] = computePercChange(mseI2EPlotA[3], mseI2EPlotA[5])
        
    copyIndex = (2*dataIndex) + idx

    mseDisp[copyIndex] = np.copy(mseDispEPlotA)
    mseQ[copyIndex] = np.copy(mseQEPlotA)
    mseI2[copyIndex] = np.copy(mseI2EPlotA)
    stdDisp[copyIndex] = np.copy(stdDispEPlotA)
    stdQ[copyIndex] = np.copy(stdQEPlotA)
    stdI2[copyIndex] = np.copy(stdI2EPlotA)
    percChangeDisp[copyIndex] = np.copy(percChangeDispTable)
    percChangeQ[copyIndex] = np.copy(percChangeQTable)
    percChangeI2[copyIndex] = np.copy(percChangeI2Table)

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

    PCTable1 = PrettyTable(["Change in MSE (%)", "Dispersion", "Q", "I\u00B2"])
    PCTable1.add_row(["Cifar-10", "", "", ""])
    PCTable1.add_row(["10 labels", "%+.1f" % percChangeDisp[idx, 0], "%+.1f" % percChangeQ[idx, 0], "%+.1f" % percChangeI2[idx, 0]])
    PCTable1.add_row(["5 labels", "%+.1f" % percChangeDisp[idx, 1], "%+.1f" % percChangeQ[idx, 1], "%+.1f" % percChangeI2[idx, 1]])
    PCTable1.add_row(["2 labels", "%+.1f" % percChangeDisp[idx, 2], "%+.1f" % percChangeQ[idx, 2], "%+.1f" % percChangeI2[idx, 2]])
    PCTable1.add_row(["Cifar-100", "", "", ""])
    PCTable1.add_row(["10 labels", "%+.1f" % percChangeDisp[2+idx, 0], "%+.1f" % percChangeQ[2+idx, 0], "%+.1f" % percChangeI2[2+idx, 0]])
    PCTable1.add_row(["5 labels", "%+.1f" % percChangeDisp[2+idx, 1], "%+.1f" % percChangeQ[2+idx, 1], "%+.1f" % percChangeI2[2+idx, 1]])
    PCTable1.add_row(["2 labels", "%+.1f" % percChangeDisp[2+idx, 2], "%+.1f" % percChangeQ[2+idx, 2], "%+.1f" % percChangeI2[2+idx, 2]])
    PCTable1.add_row(["Fashion-MNIST", "", "", ""])
    PCTable1.add_row(["10 labels", "%+.1f" % percChangeDisp[4+idx, 0], "%+.1f" % percChangeQ[4+idx, 0], "%+.1f" % percChangeI2[4+idx, 0]])
    PCTable1.add_row(["5 labels", "%+.1f" % percChangeDisp[4+idx, 1], "%+.1f" % percChangeQ[4+idx, 1], "%+.1f" % percChangeI2[4+idx, 1]])
    PCTable1.add_row(["2 labels", "%+.1f" % percChangeDisp[4+idx, 2], "%+.1f" % percChangeQ[4+idx, 2], "%+.1f" % percChangeI2[4+idx, 2]])

    PCData1 = PCTable1.get_string()
    with open("Table_6_" + "%s" % parset[idx] + "_pl1.txt", "w") as table6:
        table6.write(PCData1)

    PCTable2 = PrettyTable(["Change in MSE (%)", "Dispersion", "Q", "I\u00B2"])
    PCTable2.add_row(["Cifar-10", "", "", ""])
    PCTable2.add_row(["SH: 10v5", "%+.1f" % percChangeDisp[idx, 3], "%+.1f" % percChangeQ[idx, 3], "%+.1f" % percChangeI2[idx, 3]])
    PCTable2.add_row(["SH: 5v2", "%+.1f" % percChangeDisp[idx, 4], "%+.1f" % percChangeQ[idx, 4], "%+.1f" % percChangeI2[idx, 4]])
    PCTable2.add_row(["Non-SH: 10v5", "%+.1f" % percChangeDisp[idx, 5], "%+.1f" % percChangeQ[idx, 5], "%+.1f" % percChangeI2[idx, 5]])
    PCTable2.add_row(["Non-SH: 5v2", "%+.1f" % percChangeDisp[idx, 6], "%+.1f" % percChangeQ[idx, 6], "%+.1f" % percChangeI2[idx, 6]])
    PCTable2.add_row(["Cifar-100", "", "", ""])
    PCTable2.add_row(["SH: 10v5", "%+.1f" % percChangeDisp[2+idx, 3], "%+.1f" % percChangeQ[2+idx, 3], "%+.1f" % percChangeI2[2+idx, 3]])
    PCTable2.add_row(["SH: 5v2", "%+.1f" % percChangeDisp[2+idx, 4], "%+.1f" % percChangeQ[2+idx, 4], "%+.1f" % percChangeI2[2+idx, 4]])
    PCTable2.add_row(["Non-SH: 10v5", "%+.1f" % percChangeDisp[2+idx, 5], "%+.1f" % percChangeQ[2+idx, 5], "%+.1f" % percChangeI2[2+idx, 5]])
    PCTable2.add_row(["Non-SH: 5v2", "%+.1f" % percChangeDisp[2+idx, 6], "%+.1f" % percChangeQ[2+idx, 6], "%+.1f" % percChangeI2[2+idx, 6]])
    PCTable2.add_row(["Fashion-MNIST", "", "", ""])
    PCTable2.add_row(["SH: 10v5", "%+.1f" % percChangeDisp[4+idx, 3], "%+.1f" % percChangeQ[4+idx, 3], "%+.1f" % percChangeI2[4+idx, 3]])
    PCTable2.add_row(["SH: 5v2", "%+.1f" % percChangeDisp[4+idx, 4], "%+.1f" % percChangeQ[4+idx, 4], "%+.1f" % percChangeI2[4+idx, 4]])
    PCTable2.add_row(["Non-SH: 10v5", "%+.1f" % percChangeDisp[4+idx, 5], "%+.1f" % percChangeQ[4+idx, 5], "%+.1f" % percChangeI2[4+idx, 5]])
    PCTable2.add_row(["Non-SH: 5v2", "%+.1f" % percChangeDisp[4+idx, 6], "%+.1f" % percChangeQ[4+idx, 6], "%+.1f" % percChangeI2[4+idx, 6]])

    PCEpsData2 = PCTable2.get_string()
    with open("Table_7_" + "%s" % parset[idx] + "_pl2.txt", "w") as table7:
        table7.write(PCEpsData2)

    if idx == 0:
        varset = epsset
    else:
        varset = dtaset

    plt.errorbar(varset, mseDisp[idx, 0], yerr = np.minimum(stdDisp[idx, 0], np.sqrt(mseDisp[idx, 0]), np.divide(mseDisp[idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseDisp[idx, 1], yerr = np.minimum(stdDisp[idx, 1], np.sqrt(mseDisp[idx, 1]), np.divide(mseDisp[idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseDisp[idx, 2], yerr = np.minimum(stdDisp[idx, 2], np.sqrt(mseDisp[idx, 2]), np.divide(mseDisp[idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseDisp[idx, 3], yerr = np.minimum(stdDisp[idx, 3], np.sqrt(mseDisp[idx, 3]), np.divide(mseDisp[idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[0] + "_vary_" + "%s" % parset[idx] + "_disp_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseDisp[idx, 2], yerr = np.minimum(stdDisp[idx, 2], np.sqrt(mseDisp[idx, 2]), np.divide(mseDisp[idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseDisp[idx, 3], yerr = np.minimum(stdDisp[idx, 3], np.sqrt(mseDisp[idx, 3]), np.divide(mseDisp[idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseDisp[idx, 4], yerr = np.minimum(stdDisp[idx, 4], np.sqrt(mseDisp[idx, 4]), np.divide(mseDisp[idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseDisp[idx, 5], yerr = np.minimum(stdDisp[idx, 5], np.sqrt(mseDisp[idx, 5]), np.divide(mseDisp[idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[0] + "_vary_" + "%s" % parset[idx] + "_disp_5v2.png")
    plt.clf()

    plt.errorbar(varset, mseQ[idx, 0], yerr = np.minimum(stdQ[idx, 0], np.sqrt(mseQ[idx, 0]), np.divide(mseQ[idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseQ[idx, 1], yerr = np.minimum(stdQ[idx, 1], np.sqrt(mseQ[idx, 1]), np.divide(mseQ[idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseQ[idx, 2], yerr = np.minimum(stdQ[idx, 2], np.sqrt(mseQ[idx, 2]), np.divide(mseQ[idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseQ[idx, 3], yerr = np.minimum(stdQ[idx, 3], np.sqrt(mseQ[idx, 3]), np.divide(mseQ[idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[0] + "_vary_" + "%s" % parset[idx] + "_q_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseQ[idx, 2], yerr = np.minimum(stdQ[idx, 2], np.sqrt(mseQ[idx, 2]), np.divide(mseQ[idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseQ[idx, 3], yerr = np.minimum(stdQ[idx, 3], np.sqrt(mseQ[idx, 3]), np.divide(mseQ[idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseQ[idx, 4], yerr = np.minimum(stdQ[idx, 4], np.sqrt(mseQ[idx, 4]), np.divide(mseQ[idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseQ[idx, 5], yerr = np.minimum(stdQ[idx, 5], np.sqrt(mseQ[idx, 5]), np.divide(mseQ[idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[0] + "_vary_" + "%s" % parset[idx] + "_q_5v2.png")
    plt.clf()

    plt.errorbar(varset, mseI2[idx, 0], yerr = np.minimum(stdI2[idx, 0], np.sqrt(mseI2[idx, 0]), np.divide(mseI2[idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseI2[idx, 1], yerr = np.minimum(stdI2[idx, 1], np.sqrt(mseI2[idx, 1]), np.divide(mseI2[idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseI2[idx, 2], yerr = np.minimum(stdI2[idx, 2], np.sqrt(mseI2[idx, 2]), np.divide(mseI2[idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseI2[idx, 3], yerr = np.minimum(stdI2[idx, 3], np.sqrt(mseI2[idx, 3]), np.divide(mseI2[idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[0] + "_vary_" + "%s" % parset[idx] + "_i2_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseI2[idx, 2], yerr = np.minimum(stdI2[idx, 2], np.sqrt(mseI2[idx, 2]), np.divide(mseI2[idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseI2[idx, 3], yerr = np.minimum(stdI2[idx, 3], np.sqrt(mseI2[idx, 3]), np.divide(mseI2[idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseI2[idx, 4], yerr = np.minimum(stdI2[idx, 4], np.sqrt(mseI2[idx, 4]), np.divide(mseI2[idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseI2[idx, 5], yerr = np.minimum(stdI2[idx, 5], np.sqrt(mseI2[idx, 5]), np.divide(mseI2[idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[0] + "_vary_" + "%s" % parset[idx] + "_i2_5v2.png")
    plt.clf()

    plt.errorbar(varset, mseDisp[2+idx, 0], yerr = np.minimum(stdDisp[2+idx, 0], np.sqrt(mseDisp[2+idx, 0]), np.divide(mseDisp[2+idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseDisp[2+idx, 1], yerr = np.minimum(stdDisp[2+idx, 1], np.sqrt(mseDisp[2+idx, 1]), np.divide(mseDisp[2+idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseDisp[2+idx, 2], yerr = np.minimum(stdDisp[2+idx, 2], np.sqrt(mseDisp[2+idx, 2]), np.divide(mseDisp[2+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseDisp[2+idx, 3], yerr = np.minimum(stdDisp[2+idx, 3], np.sqrt(mseDisp[2+idx, 3]), np.divide(mseDisp[2+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[1] + "_vary_" + "%s" % parset[idx] + "_disp_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseDisp[2+idx, 2], yerr = np.minimum(stdDisp[2+idx, 2], np.sqrt(mseDisp[2+idx, 2]), np.divide(mseDisp[2+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseDisp[2+idx, 3], yerr = np.minimum(stdDisp[2+idx, 3], np.sqrt(mseDisp[2+idx, 3]), np.divide(mseDisp[2+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseDisp[2+idx, 4], yerr = np.minimum(stdDisp[2+idx, 4], np.sqrt(mseDisp[2+idx, 4]), np.divide(mseDisp[2+idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseDisp[2+idx, 5], yerr = np.minimum(stdDisp[2+idx, 5], np.sqrt(mseDisp[2+idx, 5]), np.divide(mseDisp[2+idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[1] + "_vary_" + "%s" % parset[idx] + "_disp_5v2.png")
    plt.clf()

    plt.errorbar(varset, mseQ[2+idx, 0], yerr = np.minimum(stdQ[2+idx, 0], np.sqrt(mseQ[2+idx, 0]), np.divide(mseQ[2+idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseQ[2+idx, 1], yerr = np.minimum(stdQ[2+idx, 1], np.sqrt(mseQ[2+idx, 1]), np.divide(mseQ[2+idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseQ[2+idx, 2], yerr = np.minimum(stdQ[2+idx, 2], np.sqrt(mseQ[2+idx, 2]), np.divide(mseQ[2+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseQ[2+idx, 3], yerr = np.minimum(stdQ[2+idx, 3], np.sqrt(mseQ[2+idx, 3]), np.divide(mseQ[2+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[1] + "_vary_" + "%s" % parset[idx] + "_q_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseQ[2+idx, 2], yerr = np.minimum(stdQ[2+idx, 2], np.sqrt(mseQ[2+idx, 2]), np.divide(mseQ[2+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseQ[2+idx, 3], yerr = np.minimum(stdQ[2+idx, 3], np.sqrt(mseQ[2+idx, 3]), np.divide(mseQ[2+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseQ[2+idx, 4], yerr = np.minimum(stdQ[2+idx, 4], np.sqrt(mseQ[2+idx, 4]), np.divide(mseQ[2+idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseQ[2+idx, 5], yerr = np.minimum(stdQ[2+idx, 5], np.sqrt(mseQ[2+idx, 5]), np.divide(mseQ[2+idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[1] + "_vary_" + "%s" % parset[idx] + "_q_5v2.png")
    plt.clf()

    plt.errorbar(varset, mseI2[2+idx, 0], yerr = np.minimum(stdI2[2+idx, 0], np.sqrt(mseI2[2+idx, 0]), np.divide(mseI2[2+idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseI2[2+idx, 1], yerr = np.minimum(stdI2[2+idx, 1], np.sqrt(mseI2[2+idx, 1]), np.divide(mseI2[2+idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseI2[2+idx, 2], yerr = np.minimum(stdI2[2+idx, 2], np.sqrt(mseI2[2+idx, 2]), np.divide(mseI2[2+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseI2[2+idx, 3], yerr = np.minimum(stdI2[2+idx, 3], np.sqrt(mseI2[2+idx, 3]), np.divide(mseI2[2+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[1] + "_vary_" + "%s" % parset[idx] + "_i2_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseI2[2+idx, 2], yerr = np.minimum(stdI2[2+idx, 2], np.sqrt(mseI2[2+idx, 2]), np.divide(mseI2[2+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseI2[2+idx, 3], yerr = np.minimum(stdI2[2+idx, 3], np.sqrt(mseI2[2+idx, 3]), np.divide(mseI2[2+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseI2[2+idx, 4], yerr = np.minimum(stdI2[2+idx, 4], np.sqrt(mseI2[2+idx, 4]), np.divide(mseI2[2+idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseI2[2+idx, 5], yerr = np.minimum(stdI2[2+idx, 5], np.sqrt(mseI2[2+idx, 5]), np.divide(mseI2[2+idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[1] + "_vary_" + "%s" % parset[idx] + "_i2_5v2.png")
    plt.clf()

    plt.errorbar(varset, mseDisp[4+idx, 0], yerr = np.minimum(stdDisp[4+idx, 0], np.sqrt(mseDisp[4+idx, 0]), np.divide(mseDisp[4+idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseDisp[4+idx, 1], yerr = np.minimum(stdDisp[4+idx, 1], np.sqrt(mseDisp[4+idx, 1]), np.divide(mseDisp[4+idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseDisp[4+idx, 2], yerr = np.minimum(stdDisp[4+idx, 2], np.sqrt(mseDisp[4+idx, 2]), np.divide(mseDisp[4+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseDisp[4+idx, 3], yerr = np.minimum(stdDisp[4+idx, 3], np.sqrt(mseDisp[4+idx, 3]), np.divide(mseDisp[4+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[2] + "_vary_" + "%s" % parset[idx] + "_disp_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseDisp[4+idx, 2], yerr = np.minimum(stdDisp[4+idx, 2], np.sqrt(mseDisp[4+idx, 2]), np.divide(mseDisp[4+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseDisp[4+idx, 3], yerr = np.minimum(stdDisp[4+idx, 3], np.sqrt(mseDisp[4+idx, 3]), np.divide(mseDisp[4+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseDisp[4+idx, 4], yerr = np.minimum(stdDisp[4+idx, 4], np.sqrt(mseDisp[4+idx, 4]), np.divide(mseDisp[4+idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseDisp[4+idx, 5], yerr = np.minimum(stdDisp[4+idx, 5], np.sqrt(mseDisp[4+idx, 5]), np.divide(mseDisp[4+idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[2] + "_vary_" + "%s" % parset[idx] + "_disp_5v2.png")
    plt.clf()

    plt.errorbar(varset, mseQ[4+idx, 0], yerr = np.minimum(stdQ[4+idx, 0], np.sqrt(mseQ[4+idx, 0]), np.divide(mseQ[4+idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseQ[4+idx, 1], yerr = np.minimum(stdQ[4+idx, 1], np.sqrt(mseQ[4+idx, 1]), np.divide(mseQ[4+idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseQ[4+idx, 2], yerr = np.minimum(stdQ[4+idx, 2], np.sqrt(mseQ[4+idx, 2]), np.divide(mseQ[4+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseQ[4+idx, 3], yerr = np.minimum(stdQ[4+idx, 3], np.sqrt(mseQ[4+idx, 3]), np.divide(mseQ[4+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[2] + "_vary_" + "%s" % parset[idx] + "_q_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseQ[4+idx, 2], yerr = np.minimum(stdQ[4+idx, 2], np.sqrt(mseQ[4+idx, 2]), np.divide(mseQ[4+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseQ[4+idx, 3], yerr = np.minimum(stdQ[4+idx, 3], np.sqrt(mseQ[4+idx, 3]), np.divide(mseQ[4+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseQ[4+idx, 4], yerr = np.minimum(stdQ[4+idx, 4], np.sqrt(mseQ[4+idx, 4]), np.divide(mseQ[4+idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseQ[4+idx, 5], yerr = np.minimum(stdQ[4+idx, 5], np.sqrt(mseQ[4+idx, 5]), np.divide(mseQ[4+idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[2] + "_vary_" + "%s" % parset[idx] + "_q_5v2.png")
    plt.clf()

    plt.errorbar(varset, mseI2[4+idx, 0], yerr = np.minimum(stdI2[4+idx, 0], np.sqrt(mseI2[4+idx, 0]), np.divide(mseI2[4+idx, 0], 2)), color = 'blue', marker = 'o', label = f"{labelset[0]}: equal")
    plt.errorbar(varset, mseI2[4+idx, 1], yerr = np.minimum(stdI2[4+idx, 1], np.sqrt(mseI2[4+idx, 1]), np.divide(mseI2[4+idx, 1], 2)), color = 'blueviolet', marker = 'x', label = f"{labelset[0]}: unequal")
    plt.errorbar(varset, mseI2[4+idx, 2], yerr = np.minimum(stdI2[4+idx, 2], np.sqrt(mseI2[4+idx, 2]), np.divide(mseI2[4+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseI2[4+idx, 3], yerr = np.minimum(stdI2[4+idx, 3], np.sqrt(mseI2[4+idx, 3]), np.divide(mseI2[4+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[2] + "_vary_" + "%s" % parset[idx] + "_i2_10v5.png")
    plt.clf()

    plt.errorbar(varset, mseI2[4+idx, 2], yerr = np.minimum(stdI2[4+idx, 2], np.sqrt(mseI2[4+idx, 2]), np.divide(mseI2[4+idx, 2], 2)), color = 'green', marker = 'o', label = f"{labelset[1]}: equal")
    plt.errorbar(varset, mseI2[4+idx, 3], yerr = np.minimum(stdI2[4+idx, 3], np.sqrt(mseI2[4+idx, 3]), np.divide(mseI2[4+idx, 3], 2)), color = 'lime', marker = 'x', label = f"{labelset[1]}: unequal")
    plt.errorbar(varset, mseI2[4+idx, 4], yerr = np.minimum(stdI2[4+idx, 4], np.sqrt(mseI2[4+idx, 4]), np.divide(mseI2[4+idx, 4], 2)), color = 'orange', marker = 'o', label = f"{labelset[2]}: equal")
    plt.errorbar(varset, mseI2[4+idx, 5], yerr = np.minimum(stdI2[4+idx, 5], np.sqrt(mseI2[4+idx, 5]), np.divide(mseI2[4+idx, 5], 2)), color = 'gold', marker = 'x', label = f"{labelset[2]}: unequal")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[idx])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % cifarset[2] + "_vary_" + "%s" % parset[idx] + "_i2_5v2.png")
    plt.clf()

print("Finished.\n")