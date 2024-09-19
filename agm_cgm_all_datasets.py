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

# SETTING PRIVACY PARAMETERS
eps = 0.5
dta = 0.1

# SETTING DIMENSIONS OF DATASETS
dimCifar = 3072
dimFashion = 784

numCifar = 50000
numFashion = 60000

GSCifar = float(mp.sqrt(dimCifar))/numCifar
GSFashion = float(mp.sqrt(dimFashion))/numFashion

# INITIALISING OTHER PARAMETERS AND CONSTANTS
dataset = np.array(['Cifar10', 'Cifar100', 'Fashion'])
freqset = np.array(['10 (equal)', '10 (unequal)', '5 (equal)', '5 (unequal)', '2 (equal)', '2 (unequal)'])
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

def runLoop(dataIndex, dim, num, newImages, labels, GS):

    F = len(freqset)
    mseDispEPlotA = np.zeros(F)
    mseDispEPlotC = np.zeros(F)
    mseDispTPlotA = np.zeros(F)
    mseDispTPlotC = np.zeros(F)
    mseQEPlotA = np.zeros(F)
    mseQEPlotC = np.zeros(F)   
    mseQTPlotA = np.zeros(F)
    mseQTPlotC = np.zeros(F)
    mseI2EPlotA = np.zeros(F)
    mseI2EPlotC = np.zeros(F)
    mseI2TPlotA = np.zeros(F)
    mseI2TPlotC = np.zeros(F)
    mseCentralPlotA = np.zeros(F)
    mseCentralPlotC = np.zeros(F)

    mseDispEPlotASD = np.zeros(F)
    mseDispEPlotCSD = np.zeros(F)
    mseDispTPlotASD = np.zeros(F)
    mseDispTPlotCSD = np.zeros(F)
    mseQEPlotASD = np.zeros(F)
    mseQEPlotCSD = np.zeros(F)
    mseQTPlotASD = np.zeros(F)
    mseQTPlotCSD = np.zeros(F)
    mseI2EPlotASD = np.zeros(F)
    mseI2EPlotCSD = np.zeros(F)
    mseI2TPlotASD = np.zeros(F)
    mseI2TPlotCSD = np.zeros(F)
    mseCentralPlotASD = np.zeros(F)
    mseCentralPlotCSD = np.zeros(F)

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

    mseDispEPlotATemp = np.zeros((F, R))
    mseDispEPlotCTemp = np.zeros((F, R))
    mseDispTPlotATemp = np.zeros((F, R))
    mseDispTPlotCTemp = np.zeros((F, R))
    mseQEPlotATemp = np.zeros((F, R))
    mseQEPlotCTemp = np.zeros((F, R))
    mseQTPlotATemp = np.zeros((F, R))
    mseQTPlotCTemp = np.zeros((F, R))
    mseI2EPlotATemp = np.zeros((F, R))
    mseI2EPlotCTemp = np.zeros((F, R))
    mseI2TPlotATemp = np.zeros((F, R))
    mseI2TPlotCTemp = np.zeros((F, R))
    mseCPlotATemp = np.zeros((F, R))
    mseCPlotCTemp = np.zeros((F, R))

    print(f"Processing dataset {dataIndex+1}.")

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
            weightedTrueDisp = np.multiply(weight[j], wTrueDisp)
            noisyVar = np.power(noisyDiff, 2)
            wNoisyVar = np.power(wNoisyDiff, 2)
            weightedNoisyVar = np.multiply(weight[j], wNoisyVar)

            xi2 = normal(0, sigma**2)
            noisyDisp = np.add(noisyVar, xi2)
            noisyQ = np.add(weightedNoisyVar, xi2)

            # EMSE = MSE OF FORMULA OF DISPERSION OR Q
            mseEList[j] = np.power(np.subtract(noisyDisp, trueDisp), 2)
            mseQEList[j] = np.power(np.subtract(noisyQ, weightedTrueDisp), 2)

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

        mseE = np.sum(mseEList)
        mseT = np.sum(mseTList)
        mseQE = np.sum(mseQEList)
        mseQT = np.sum(mseQTList)

        if ACindex == 0:

            # EXPERIMENT 1 (EMSE VS TMSE) ASSUMES UNIFORM DATA
            if fi == 0:
                mseDispETableATemp[rep] = mseE
                mseDispTTableATemp[rep] = mseT
                mseQETableATemp[rep] = mseQE
                mseQTTableATemp[rep] = mseQT

            # EXPERIMENT 3 (STATISTICAL HETEROGENEITY)
            mseDispEPlotATemp[fi, rep] = mseE
            mseDispTPlotATemp[fi, rep] = mseT
            mseQEPlotATemp[fi, rep] = mseQE
            mseQTPlotATemp[fi, rep] = mseQT

        else:
            if fi == 0:
                mseDispETableCTemp[rep] = mseE
                mseDispTTableCTemp[rep] = mseT
                mseQETableCTemp[rep] = mseQE
                mseQTTableCTemp[rep] = mseQT

            mseDispEPlotCTemp[fi, rep] = mseE
            mseDispTPlotCTemp[fi, rep] = mseT
            mseQEPlotCTemp[fi, rep] = mseQE
            mseQTPlotCTemp[fi, rep] = mseQT

        mseISEList = np.zeros(sampleSize)
        mseISTList = np.zeros(sampleSize)

        for j in range(0, sampleSize):

            # COMPUTE I^2'' and I^2 USING SIMPLE FORMULA AT BOTTOM OF LEMMA 6.2
            trueI2Prep = np.divide(sampleSize-1, weightedTrueDisp)
            trueI2 = np.subtract(1, trueI2Prep)
            I2Prep = np.divide(sampleSize-1, noisyQ)
            I2 = np.subtract(1, I2Prep)

            # ADD THIRD NOISE TERM BASED ON LEMMA 6.2
            xi3 = normal(0, sigma**2)
            noisyI2 = np.add(I2, xi3)

            # COMPUTE EMSE AND TMSE
            diffEI2 = np.subtract(noisyI2, trueI2)
            mseISEList[j] = np.power(diffEI2, 2)
            diffTI2Prep = np.subtract(xi3, I2)
            diffTI2 = np.add(diffTI2Prep, trueI2)
            mseISTList[j] = np.power(diffTI2, 2)

        mseI2E = np.sum(mseISEList)
        mseI2T = np.sum(mseISTList)

        # EXPERIMENT 2: WHAT IS THE COST OF A DISTRIBUTED SETTING?
        xiCentral = normal(0, centralSigma**2)
        mseC = xiCentral**2

        if ACindex == 0:
            if fi == 0:
                mseI2ETableATemp[rep] = mseI2E
                mseI2TTableATemp[rep] = mseI2T
                mseCTableATemp[rep] = mseC

            mseI2EPlotATemp[fi, rep] = mseI2E
            mseI2TPlotATemp[fi, rep] = mseI2T
            mseCPlotATemp[fi, rep] = mseC

        else:
            if fi == 0:
                mseI2ETableCTemp[rep] = mseI2E
                mseI2TTableCTemp[rep] = mseI2T
                mseCTableCTemp[rep] = mseC

            mseI2EPlotCTemp[fi, rep] = mseI2E
            mseI2TPlotCTemp[fi, rep] = mseI2T
            mseCPlotCTemp[fi, rep] = mseC

    # EXPERIMENT 3: SAMPLE APPROX 2% OF CLIENTS THEN SPLIT INTO CASES BY STATISTICAL HETEROGENEITY
    # 1. EQUAL NUMBERS OF EACH OF 10 LABELS [1:1:1:1:1:1:1:1:1:1]
    # 2. UNEQUAL NUMBERS OF EACH OF 10 LABELS [11:1:1:1:1:1:1:1:1:1]
    # 3. EQUAL NUMBERS OF EACH OF 5 LABELS [1:1:1:1:1:0:0:0:0:0]
    # 4. UNEQUAL NUMBERS OF EACH OF 5 LABELS [6:1:1:1:1:0:0:0:0:0]
    # 5. EQUAL NUMBERS OF EACH OF 2 LABELS [1:1:0:0:0:0:0:0:0:0]
    # 6. UNEQUAL NUMBERS OF EACH OF 2 LABELS [9:1:0:0:0:0:0:0:0:0].
    
    for fi in range(6):
        numLabels = 10
        lsize = sampleSize/numLabels
        freqArray = np.zeros(numLabels)
        imageArray = np.zeros((sampleSize, dim))
        freqOne = np.array([lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize, lsize])
        freqTwo = np.array([5.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize, 0.5*lsize])
        freqThree = np.array([2*lsize, 2*lsize, 2*lsize, 2*lsize, 2*lsize, 0, 0, 0, 0, 0])
        freqFour = np.array([6*lsize, lsize, lsize, lsize, lsize, 0, 0, 0, 0, 0])
        freqFive = np.array([5*lsize, 5*lsize, 0, 0, 0, 0, 0, 0, 0, 0])
        freqSix = np.array([9*lsize, lsize, 0, 0, 0, 0, 0, 0, 0, 0])

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

        if fi == 0:
            mseDispETableA = np.mean(mseDispETableATemp)
            mseDispETableC = np.mean(mseDispETableCTemp)
            mseDispTTableA = np.mean(mseDispTTableATemp)
            mseDispTTableC = np.mean(mseDispTTableCTemp)
            mseQETableA = np.mean(mseQETableATemp)
            mseQETableC = np.mean(mseQETableCTemp)
            mseQTTableA = np.mean(mseQTTableATemp)
            mseQTTableC = np.mean(mseQTTableCTemp)
            mseCentralTableA = np.mean(mseCTableATemp)
            mseCentralTableC = np.mean(mseCTableCTemp)
            mseI2ETableA = np.mean(mseI2ETableATemp)
            mseI2ETableC = np.mean(mseI2ETableCTemp)
            mseI2TTableA = np.mean(mseI2TTableATemp)
            mseI2TTableC = np.mean(mseI2TTableCTemp)

            mseDispETableARound = round(mseDispETableA, 16)
            mseDispETableCRound = round(mseDispETableC, 12)
            mseDispTTableARound = round(mseDispTTableA, 16)
            mseDispTTableCRound = round(mseDispTTableC, 12)
            mseQETableARound = round(mseQETableA, 16)
            mseQETableCRound = round(mseQETableC, 12)
            mseQTTableARound = round(mseQTTableA, 16)
            mseQTTableCRound = round(mseQTTableC, 12)
            mseCentralTableARound = round(mseCentralTableA, 3)
            mseCentralTableCRound = round(mseCentralTableC, 2)

            mseDispETableASD = round(np.std(mseDispETableATemp), 16)
            mseDispETableCSD = round(np.std(mseDispETableCTemp), 14)
            mseDispTTableASD = round(np.std(mseDispTTableATemp), 16)
            mseDispTTableCSD = round(np.std(mseDispTTableCTemp), 14)
            mseQETableASD = round(np.std(mseQETableATemp), 16)
            mseQETableCSD = round(np.std(mseQETableCTemp), 14)
            mseQTTableASD = round(np.std(mseQTTableATemp), 16)
            mseQTTableCSD = round(np.std(mseQTTableCTemp), 14)
            mseCentralTableASD = round(np.std(mseCTableATemp), 3)
            mseCentralTableCSD = round(np.std(mseCTableCTemp), 2)

            mseDispETableAC = np.round(np.divide(mseDispETableA, mseDispETableC), 7)
            mseDispTTableAC = np.round(np.divide(mseDispTTableA, mseDispTTableC), 6)
            mseDispETTableA = np.round(np.divide(mseDispETableA, mseDispTTableA), 7)
            mseDispETTableC = np.round(np.divide(mseDispETableC, mseDispTTableC), 7)
            mseQETableAC = np.round(np.divide(mseQETableA, mseQETableC), 6)
            mseQTTableAC = np.round(np.divide(mseQTTableA, mseQTTableC), 6)
            mseQETTableA = np.round(np.divide(mseQETableA, mseQTTableA), 6)
            mseQETTableC = np.round(np.divide(mseQETableC, mseQTTableC), 6)
            mseCentralTableAC = np.round(np.divide(mseCentralTableA, mseCentralTableC), 6)
            mseI2ETableAC = np.round(np.divide(mseI2ETableA, mseI2ETableC), 5)
            mseI2TTableAC = np.round(np.divide(mseI2TTableA, mseI2TTableC), 5)
            mseI2ETTableA = np.round(np.divide(mseI2ETableA, mseI2TTableA), 5)
            mseI2ETTableC = np.round(np.divide(mseI2ETableC, mseI2TTableC), 5)

        mseDispEPlotA[fi] = np.mean(mseDispEPlotATemp[fi])
        mseDispEPlotC[fi] = np.mean(mseDispEPlotCTemp[fi])
        mseDispTPlotA[fi] = np.mean(mseDispTPlotATemp[fi])
        mseDispTPlotC[fi] = np.mean(mseDispTPlotCTemp[fi])
        mseQEPlotA[fi] = np.mean(mseQEPlotATemp[fi])
        mseQEPlotC[fi] = np.mean(mseQEPlotCTemp[fi])
        mseQTPlotA[fi] = np.mean(mseQTPlotATemp[fi])
        mseQTPlotC[fi] = np.mean(mseQTPlotCTemp[fi])
        mseCentralPlotA[fi] = np.mean(mseCPlotATemp[fi])
        mseCentralPlotC[fi] = np.mean(mseCPlotCTemp[fi])

        mseDispEPlotASD[fi] = np.std(mseDispEPlotATemp[fi])
        mseDispEPlotCSD[fi] = np.std(mseDispEPlotCTemp[fi])
        mseDispTPlotASD[fi] = np.std(mseDispTPlotATemp[fi])
        mseDispTPlotCSD[fi] = np.std(mseDispTPlotCTemp[fi])
        mseQEPlotASD[fi] = np.std(mseQEPlotATemp[fi])
        mseQEPlotCSD[fi] = np.std(mseQEPlotCTemp[fi])
        mseQTPlotASD[fi] = np.std(mseQTPlotATemp[fi])
        mseQTPlotCSD[fi] = np.std(mseQTPlotCTemp[fi])
        mseCentralPlotASD[fi] = np.std(mseCPlotATemp[fi])
        mseCentralPlotCSD[fi] = np.std(mseCPlotCTemp[fi])

        mseI2EPlotA[fi] = np.mean(mseI2EPlotATemp[fi])
        mseI2EPlotC[fi] = np.mean(mseI2EPlotCTemp[fi])
        mseI2TPlotA[fi] = np.mean(mseI2TPlotATemp[fi])
        mseI2TPlotC[fi] = np.mean(mseI2TPlotCTemp[fi])
        mseI2EPlotASD[fi] = np.std(mseI2EPlotATemp[fi])
        mseI2EPlotCSD[fi] = np.std(mseI2EPlotCTemp[fi])
        mseI2TPlotASD[fi] = np.std(mseI2TPlotATemp[fi])
        mseI2TPlotCSD[fi] = np.std(mseI2TPlotCTemp[fi])

    # EXPERIMENT 1: COMPARISON OF AGM/CGM, EMSE/TMSE AND CMSE
    DispTable = PrettyTable(["Dispersion", "AGM", "CGM", "SD AGM", "SD CGM"])
    DispTable.add_row(["EMSE", mseDispETableARound, mseDispETableCRound, mseDispETableASD, mseDispETableCSD])
    DispTable.add_row(["TMSE", mseDispTTableARound, mseDispTTableCRound, mseDispTTableASD, mseDispTTableCSD])
    DispTable.add_row(["CMSE", mseCentralTableARound, mseCentralTableCRound, mseCentralTableASD, mseCentralTableCSD])
    print(DispTable)

    QTable = PrettyTable(["Q", "AGM", "CGM", "SD AGM", "SD CGM"])
    QTable.add_row(["EMSE", mseQETableARound, mseQETableCRound, mseQETableASD, mseQETableCSD])
    QTable.add_row(["TMSE", mseQTTableARound, mseQTTableCRound, mseQTTableASD, mseQTTableCSD])
    QTable.add_row(["CMSE", mseCentralTableARound, mseCentralTableCRound, mseCentralTableASD, mseCentralTableCSD])
    print(QTable)

    ACTable = PrettyTable(["AGM/CGM", "Dispersion", "Q", "I\u00B2"])
    ACTable.add_row(["EMSE", mseDispETableAC, mseQETableAC, mseI2ETableAC])
    ACTable.add_row(["TMSE", mseDispTTableAC, mseQTTableAC, mseI2TTableAC])
    ACTable.add_row(["CMSE", mseCentralTableAC, mseCentralTableAC, mseCentralTableAC])
    print(ACTable)

    ETTable = PrettyTable(["EMSE/TMSE", "Dispersion", "Q", "I\u00B2"])
    ETTable.add_row(["AGM", mseDispETTableA, mseQETTableA, mseI2ETTableA])
    ETTable.add_row(["CGM", mseDispETTableC, mseQETTableC, mseI2ETTableC])
    print(ETTable)

    # EXPERIMENT 3: STATISTICAL HETEROGENEITY
    plt.errorbar(freqset, mseDispEPlotA, yerr = np.minimum(mseDispEPlotASD, np.sqrt(mseDispEPlotA), np.divide(mseDispEPlotA, 2)), color = 'blue', marker = 'o', label = "Empirical AGM")
    plt.errorbar(freqset, mseDispEPlotC, yerr = np.minimum(mseDispEPlotCSD, np.sqrt(mseDispEPlotC), np.divide(mseDispEPlotC, 2)), color = 'blue', marker = 'x', label = "Empirical CGM")
    plt.errorbar(freqset, mseDispTPlotA, yerr = np.minimum(mseDispTPlotASD, np.sqrt(mseDispTPlotA), np.divide(mseDispTPlotA, 2)), color = 'green', marker = 'o', label = "Theoretical AGM")
    plt.errorbar(freqset, mseDispTPlotC, yerr = np.minimum(mseDispTPlotCSD, np.sqrt(mseDispTPlotC), np.divide(mseDispTPlotC, 2)), color = 'green', marker = 'x', label = "Theoretical CGM")
    plt.errorbar(freqset, mseCentralPlotA, yerr = np.minimum(mseCentralPlotASD, np.sqrt(mseCentralPlotA), np.divide(mseCentralPlotA, 2)), color = 'red', marker = 'o', label = "Centralized AGM")
    plt.errorbar(freqset, mseCentralPlotC, yerr = np.minimum(mseCentralPlotCSD, np.sqrt(mseCentralPlotC), np.divide(mseCentralPlotC, 2)), color = 'red', marker = 'x', label = "Centralized CGM")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Labels")
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % dataset[dataIndex] + "_disp.png")
    plt.clf()

    plt.errorbar(freqset, mseQEPlotA, yerr = np.minimum(mseQEPlotASD, np.sqrt(mseQEPlotA), np.divide(mseQEPlotA, 2)), color = 'blue', marker = 'o', label = "Empirical AGM")
    plt.errorbar(freqset, mseQEPlotC, yerr = np.minimum(mseQEPlotCSD, np.sqrt(mseQEPlotC), np.divide(mseQEPlotC, 2)), color = 'blue', marker = 'x', label = "Empirical CGM")
    plt.errorbar(freqset, mseQTPlotA, yerr = np.minimum(mseQTPlotASD, np.sqrt(mseQTPlotA), np.divide(mseQTPlotA, 2)), color = 'green', marker = 'o', label = "Theoretical AGM")
    plt.errorbar(freqset, mseQTPlotC, yerr = np.minimum(mseQTPlotCSD, np.sqrt(mseQTPlotC), np.divide(mseQTPlotC, 2)), color = 'green', marker = 'x', label = "Theoretical CGM")
    plt.errorbar(freqset, mseCentralPlotA, yerr = np.minimum(mseCentralPlotASD, np.sqrt(mseCentralPlotA), np.divide(mseCentralPlotA, 2)), color = 'red', marker = 'o', label = "Centralized AGM")
    plt.errorbar(freqset, mseCentralPlotC, yerr = np.minimum(mseCentralPlotCSD, np.sqrt(mseCentralPlotC), np.divide(mseCentralPlotC, 2)), color = 'red', marker = 'x', label = "Centralized CGM")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Labels")
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % dataset[dataIndex] + "_q.png")
    plt.clf()

    plt.errorbar(freqset, mseI2EPlotA, yerr = np.minimum(mseI2EPlotASD, np.sqrt(mseI2EPlotA), np.divide(mseI2EPlotA, 2)), color = 'blue', marker = 'o', label = "Empirical AGM")
    plt.errorbar(freqset, mseI2EPlotC, yerr = np.minimum(mseI2EPlotCSD, np.sqrt(mseI2EPlotC), np.divide(mseI2EPlotC, 2)), color = 'blue', marker = 'x', label = "Empirical CGM")
    plt.errorbar(freqset, mseI2TPlotA, yerr = np.minimum(mseI2TPlotASD, np.sqrt(mseI2TPlotA), np.divide(mseI2TPlotA, 2)), color = 'green', marker = 'o', label = "Theoretical AGM")
    plt.errorbar(freqset, mseI2TPlotC, yerr = np.minimum(mseI2TPlotCSD, np.sqrt(mseI2TPlotC), np.divide(mseI2TPlotC, 2)), color = 'green', marker = 'x', label = "Theoretical CGM")
    plt.errorbar(freqset, mseCentralPlotA, yerr = np.minimum(mseCentralPlotASD, np.sqrt(mseCentralPlotA), np.divide(mseCentralPlotA, 2)), color = 'red', marker = 'o', label = "Centralized AGM")
    plt.errorbar(freqset, mseCentralPlotC, yerr = np.minimum(mseCentralPlotCSD, np.sqrt(mseCentralPlotC), np.divide(mseCentralPlotC, 2)), color = 'red', marker = 'x', label = "Centralized CGM")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Labels")
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % dataset[dataIndex] + "_i2.png")
    plt.clf()

def runLoopVaryEps(dataIndex, dim, num, newImages, labels, GS):
    runLoop(dataIndex, dim, num, newImages, labels, GS)

def runLoopVaryDta(dataIndex, dim, num, newImages, labels, GS):
    runLoop(dataIndex, dim, num, newImages, labels, GS)

runLoopVaryEps(0, dimCifar, numCifar, newImagesCifar10, labelsCifar10, GSCifar)
runLoopVaryEps(1, dimCifar, numCifar, newImagesCifar100, labelsCifar100, GSCifar)
runLoopVaryEps(2, dimFashion, numFashion, newImagesFashion, labelsFashion, GSFashion)

print("Finished.\n")