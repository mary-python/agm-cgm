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

# INITIALISING OTHER PARAMETERS/CONSTANTS
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
    mseDispEPlotFA = np.zeros(F)
    mseDispEPlotFC = np.zeros(F)
    mseDispTPlotFA = np.zeros(F)
    mseDispTPlotFC = np.zeros(F)
    mseQEPlotFA = np.zeros(F)
    mseQEPlotFC = np.zeros(F)   
    mseQTPlotFA = np.zeros(F)
    mseQTPlotFC = np.zeros(F)
    mseISquaredEPlotFA = np.zeros(F)
    mseISquaredEPlotFC = np.zeros(F)
    mseISquaredTPlotFA = np.zeros(F)
    mseISquaredTPlotFC = np.zeros(F)
    mseCentralPlotFA = np.zeros(F)
    mseCentralPlotFC = np.zeros(F)

    mseDispEPlotFARange = np.zeros(F)
    mseDispEPlotFCRange = np.zeros(F)
    mseDispTPlotFARange = np.zeros(F)
    mseDispTPlotFCRange = np.zeros(F)
    mseQEPlotFARange = np.zeros(F)
    mseQEPlotFCRange = np.zeros(F)
    mseQTPlotFARange = np.zeros(F)
    mseQTPlotFCRange = np.zeros(F)
    mseISquaredEPlotFARange = np.zeros(F)
    mseISquaredEPlotFCRange = np.zeros(F)
    mseISquaredTPlotFARange = np.zeros(F)
    mseISquaredTPlotFCRange = np.zeros(F)
    mseCentralPlotFARange = np.zeros(F)
    mseCentralPlotFCRange = np.zeros(F)

    mseDispEPlotVATemp = np.zeros(R)
    mseDispEPlotVCTemp = np.zeros(R)
    mseDispTPlotVATemp = np.zeros(R)
    mseDispTPlotVCTemp = np.zeros(R)
    mseQEPlotVATemp = np.zeros(R)
    mseQEPlotVCTemp = np.zeros(R)
    mseQTPlotVATemp = np.zeros(R)
    mseQTPlotVCTemp = np.zeros(R)
    mseISquaredEPlotVATemp = np.zeros(R)
    mseISquaredEPlotVCTemp = np.zeros(R)
    mseISquaredTPlotVATemp = np.zeros(R)
    mseISquaredTPlotVCTemp = np.zeros(R)
    mseCentralPlotVATemp = np.zeros(R)
    mseCentralPlotVCTemp = np.zeros(R)

    mseDispEPlotFATemp = np.zeros((F, R))
    mseDispEPlotFCTemp = np.zeros((F, R))
    mseDispTPlotFATemp = np.zeros((F, R))
    mseDispTPlotFCTemp = np.zeros((F, R))
    mseQEPlotFATemp = np.zeros((F, R))
    mseQEPlotFCTemp = np.zeros((F, R))
    mseQTPlotFATemp = np.zeros((F, R))
    mseQTPlotFCTemp = np.zeros((F, R))
    mseISquaredEPlotFATemp = np.zeros((F, R))
    mseISquaredEPlotFCTemp = np.zeros((F, R))
    mseISquaredTPlotFATemp = np.zeros((F, R))
    mseISquaredTPlotFCTemp = np.zeros((F, R))
    mseCentralPlotFATemp = np.zeros((F, R))
    mseCentralPlotFCTemp = np.zeros((F, R))

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

            # EXPERIMENT 1 (EMPIRICAL VS THEORETICAL) ASSUMES UNIFORM DATA
            if fi == 0:
                mseDispEPlotVATemp[rep] = mseEmpirical
                mseDispTPlotVATemp[rep] = mseTheoretical
                mseQEPlotVATemp[rep] = mseQEmpirical
                mseQTPlotVATemp[rep] = mseQTheoretical

            # EXPERIMENT 3 (STATISTICAL HETEROGENEITY)
            mseDispEPlotFATemp[fi, rep] = mseEmpirical
            mseDispTPlotFATemp[fi, rep] = mseTheoretical
            mseQEPlotFATemp[fi, rep] = mseQEmpirical
            mseQTPlotFATemp[fi, rep] = mseQTheoretical

        else:
            if fi == 0:
                mseDispEPlotVCTemp[rep] = mseEmpirical
                mseDispTPlotVCTemp[rep] = mseTheoretical
                mseQEPlotVCTemp[rep] = mseQEmpirical
                mseQTPlotVCTemp[rep] = mseQTheoretical

            mseDispEPlotFCTemp[fi, rep] = mseEmpirical
            mseDispTPlotFCTemp[fi, rep] = mseTheoretical
            mseQEPlotFCTemp[fi, rep] = mseQEmpirical
            mseQTPlotFCTemp[fi, rep] = mseQTheoretical

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

        # EXPERIMENT 2: WHAT IS THE COST OF A DISTRIBUTED SETTING?
        xiCentral = normal(0, centralSigma**2)
        mseCentral = xiCentral**2

        if ACindex == 0:

            if fi == 0:
                mseISquaredEPlotVATemp[rep] = mseISquaredEmpirical
                mseISquaredTPlotVATemp[rep] = mseISquaredTheoretical
                mseCentralPlotVATemp[rep] = mseCentral

            mseISquaredEPlotFATemp[fi, rep] = mseISquaredEmpirical
            mseISquaredTPlotFATemp[fi, rep] = mseISquaredTheoretical
            mseCentralPlotFATemp[fi, rep] = mseCentral

        else:
            if fi == 0:
                mseISquaredEPlotVCTemp[rep] = mseISquaredEmpirical
                mseISquaredTPlotVCTemp[rep] = mseISquaredTheoretical
                mseCentralPlotVCTemp[rep] = mseCentral

            mseISquaredEPlotFCTemp[fi, rep] = mseISquaredEmpirical
            mseISquaredTPlotFCTemp[fi, rep] = mseISquaredTheoretical
            mseCentralPlotFCTemp[fi, rep] = mseCentral

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

        # COMPUTE SIGMA USING CLASSIC GAUSSIAN MECHANISM FOR COMPARISON BETWEEN MSE AND DISTRIBUTED/CENTRALIZED SETTING
        classicSigma = (GS*mp.sqrt(2*mp.log(1.25/dta)))/eps
        classicCentralSigma = (mp.sqrt(2*mp.log(1.25/dta)))/eps

        # REPEATS FOR EACH FREQUENCY SPECIFICATION
        for rep in range(R):
            computeMSE(0, rep, fi, imageArray, sigma, centralSigma)
            computeMSE(1, rep, fi, imageArray, classicSigma, classicCentralSigma)

        if fi == 0:
            mseDispEPlotVA = round(np.mean(mseDispEPlotVATemp), 2)
            mseDispEPlotVC = round(np.mean(mseDispEPlotVCTemp), 2)
            mseDispTPlotVA = round(np.mean(mseDispTPlotVATemp), 2)
            mseDispTPlotVC = round(np.mean(mseDispTPlotVCTemp), 2)
            mseQEPlotVA = round(np.mean(mseQEPlotVATemp), 2)
            mseQEPlotVC = round(np.mean(mseQEPlotVCTemp), 2)
            mseQTPlotVA = round(np.mean(mseQTPlotVATemp), 2)
            mseQTPlotVC = round(np.mean(mseQTPlotVCTemp), 2)
            mseISquaredEPlotVA = round(np.mean(mseISquaredEPlotVATemp))
            mseISquaredEPlotVC = round(np.mean(mseISquaredEPlotVCTemp))
            mseISquaredTPlotVA = round(np.mean(mseISquaredTPlotVATemp))
            mseISquaredTPlotVC = round(np.mean(mseISquaredTPlotVCTemp))
            mseCentralPlotVA = round(np.mean(mseCentralPlotVATemp), 2)
            mseCentralPlotVC = round(np.mean(mseCentralPlotVCTemp), 2)

            mseDispEPlotVARange = round(np.std(mseDispEPlotVATemp), 10)
            mseDispEPlotVCRange = round(np.std(mseDispEPlotVCTemp), 10)
            mseDispTPlotVARange = round(np.std(mseDispTPlotVATemp), 10)
            mseDispTPlotVCRange = round(np.std(mseDispTPlotVCTemp), 10)
            mseQEPlotVARange = round(np.std(mseQEPlotVATemp), 10)
            mseQEPlotVCRange = round(np.std(mseQEPlotVCTemp), 10)
            mseQTPlotVARange = round(np.std(mseQTPlotVATemp), 10)
            mseQTPlotVCRange = round(np.std(mseQTPlotVCTemp), 10)
            mseISquaredEPlotVARange = round(np.std(mseISquaredEPlotVATemp), 10)
            mseISquaredEPlotVCRange = round(np.std(mseISquaredEPlotVCTemp), 10)
            mseISquaredTPlotVARange = round(np.std(mseISquaredTPlotVATemp), 10)
            mseISquaredTPlotVCRange = round(np.std(mseISquaredTPlotVCTemp), 10)
            mseCentralPlotVARange = round(np.std(mseCentralPlotVATemp), 2)
            mseCentralPlotVCRange = round(np.std(mseCentralPlotVCTemp), 2)

        mseDispEPlotFA[fi] = np.mean(mseDispEPlotFATemp[fi])
        mseQEPlotFA[fi] = np.mean(mseQEPlotFATemp[fi])
        mseDispEPlotFC[fi] = np.mean(mseDispEPlotFCTemp[fi])
        mseQEPlotFC[fi] = np.mean(mseQEPlotFCTemp[fi])
        mseDispTPlotFA[fi] = np.mean(mseDispTPlotFATemp[fi])
        mseQTPlotFA[fi] = np.mean(mseQTPlotFATemp[fi])
        mseDispTPlotFC[fi] = np.mean(mseDispTPlotFCTemp[fi])
        mseQTPlotFC[fi] = np.mean(mseQTPlotFCTemp[fi])
        mseISquaredEPlotFA[fi] = np.mean(mseISquaredEPlotFATemp[fi])
        mseISquaredEPlotFC[fi] = np.mean(mseISquaredEPlotFCTemp[fi])
        mseISquaredTPlotFA[fi] = np.mean(mseISquaredTPlotFATemp[fi])
        mseISquaredTPlotFC[fi] = np.mean(mseISquaredTPlotFCTemp[fi])
        mseCentralPlotFA[fi] = np.mean(mseCentralPlotFATemp[fi])
        mseCentralPlotFC[fi] = np.mean(mseCentralPlotFCTemp[fi])

        mseDispEPlotFARange[fi] = np.std(mseDispEPlotFATemp[fi])
        mseQEPlotFARange[fi] = np.std(mseQEPlotFATemp[fi])
        mseDispEPlotFCRange[fi] = np.std(mseDispEPlotFCTemp[fi])
        mseQEPlotFCRange[fi] = np.std(mseQEPlotFCTemp[fi])
        mseDispTPlotFARange[fi] = np.std(mseDispTPlotFATemp[fi])
        mseQTPlotFARange[fi] = np.std(mseQTPlotFATemp[fi])
        mseDispTPlotFCRange[fi] = np.std(mseDispTPlotFCTemp[fi])
        mseQTPlotFCRange[fi] = np.std(mseQTPlotFCTemp[fi])
        mseISquaredEPlotFARange[fi] = np.std(mseISquaredEPlotFATemp[fi])
        mseISquaredEPlotFCRange[fi] = np.std(mseISquaredEPlotFCTemp[fi])
        mseISquaredTPlotFARange[fi] = np.std(mseISquaredTPlotFATemp[fi])
        mseISquaredTPlotFCRange[fi] = np.std(mseISquaredTPlotFCTemp[fi])
        mseCentralPlotFARange[fi] = np.std(mseCentralPlotFATemp[fi])
        mseCentralPlotFCRange[fi] = np.std(mseCentralPlotFCTemp[fi])

    # EXPERIMENT 1: COMPARISON OF AGM/CGM, EMPIRICAL/THEORETICAL AND DISTRIBUTED/CENTRALIZED
    DispTable = PrettyTable(["Dispersion", "AGM", "CGM", "Range AGM", "Range CGM"])
    DispTable.add_row(["Empirical", mseDispEPlotVA, mseDispEPlotVC, mseDispEPlotVARange, mseDispEPlotVCRange])
    DispTable.add_row(["Theoretical", mseDispTPlotVA, mseDispTPlotVC, mseDispTPlotVARange, mseDispTPlotVCRange])
    DispTable.add_row(["Centralized", mseCentralPlotVA, mseCentralPlotVC, mseCentralPlotVARange, mseCentralPlotVCRange])
    print(DispTable)

    QTable = PrettyTable(["Q", "AGM", "CGM", "Range AGM", "Range CGM"])
    QTable.add_row(["Empirical", mseQEPlotVA, mseQEPlotVC, mseQEPlotVARange, mseQEPlotVCRange])
    QTable.add_row(["Theoretical", mseQTPlotVA, mseQTPlotVC, mseQTPlotVARange, mseQTPlotVCRange])
    QTable.add_row(["Centralized", mseCentralPlotVA, mseCentralPlotVC, mseCentralPlotVARange, mseCentralPlotVCRange])
    print(QTable)

    ISquaredTable = PrettyTable(["ISquared", "AGM", "CGM", "Range AGM", "Range CGM"])
    ISquaredTable.add_row(["Empirical", mseISquaredEPlotVA, mseISquaredEPlotVC, mseISquaredEPlotVARange, mseISquaredEPlotVCRange])
    ISquaredTable.add_row(["Theoretical", mseISquaredTPlotVA, mseISquaredTPlotVC, mseISquaredTPlotVARange, mseISquaredTPlotVCRange])
    ISquaredTable.add_row(["Centralized", mseCentralPlotVA, mseCentralPlotVC, mseCentralPlotVARange, mseCentralPlotVCRange])
    print(ISquaredTable)

    # EXPERIMENT 3: STATISTICAL HETEROGENEITY
    plt.errorbar(freqset, mseDispEPlotFA, yerr = np.minimum(mseDispEPlotFARange, np.sqrt(mseDispEPlotFA), np.divide(mseDispEPlotFA, 2)), color = 'blue', marker = 'o', label = "Empirical AGM")
    plt.errorbar(freqset, mseDispEPlotFC, yerr = np.minimum(mseDispEPlotFCRange, np.sqrt(mseDispEPlotFC), np.divide(mseDispEPlotFC, 2)), color = 'blue', marker = 'x', label = "Empirical CGM")
    plt.errorbar(freqset, mseDispTPlotFA, yerr = np.minimum(mseDispTPlotFARange, np.sqrt(mseDispTPlotFA), np.divide(mseDispTPlotFA, 2)), color = 'green', marker = 'o', label = "Theoretical AGM")
    plt.errorbar(freqset, mseDispTPlotFC, yerr = np.minimum(mseDispTPlotFCRange, np.sqrt(mseDispTPlotFC), np.divide(mseDispTPlotFC, 2)), color = 'green', marker = 'x', label = "Theoretical CGM")
    plt.errorbar(freqset, mseCentralPlotFA, yerr = np.minimum(mseCentralPlotFARange, np.sqrt(mseCentralPlotFA), np.divide(mseCentralPlotFA, 2)), color = 'red', marker = 'o', label = "Centralized AGM")
    plt.errorbar(freqset, mseCentralPlotFC, yerr = np.minimum(mseCentralPlotFCRange, np.sqrt(mseCentralPlotFC), np.divide(mseCentralPlotFC, 2)), color = 'red', marker = 'x', label = "Centralized CGM")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Labels")
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % dataset[dataIndex] + "_disp.png")
    plt.clf()

    plt.errorbar(freqset, mseQEPlotFA, yerr = np.minimum(mseQEPlotFARange, np.sqrt(mseQEPlotFA), np.divide(mseQEPlotFA, 2)), color = 'blue', marker = 'o', label = "Empirical AGM")
    plt.errorbar(freqset, mseQEPlotFC, yerr = np.minimum(mseQEPlotFCRange, np.sqrt(mseQEPlotFC), np.divide(mseQEPlotFC, 2)), color = 'blue', marker = 'x', label = "Empirical CGM")
    plt.errorbar(freqset, mseQTPlotFA, yerr = np.minimum(mseQTPlotFARange, np.sqrt(mseQTPlotFA), np.divide(mseQTPlotFA, 2)), color = 'green', marker = 'o', label = "Theoretical AGM")
    plt.errorbar(freqset, mseQTPlotFC, yerr = np.minimum(mseQTPlotFCRange, np.sqrt(mseQTPlotFC), np.divide(mseQTPlotFC, 2)), color = 'green', marker = 'x', label = "Theoretical CGM")
    plt.errorbar(freqset, mseCentralPlotFA, yerr = np.minimum(mseCentralPlotFARange, np.sqrt(mseCentralPlotFA), np.divide(mseCentralPlotFA, 2)), color = 'red', marker = 'o', label = "Centralized AGM")
    plt.errorbar(freqset, mseCentralPlotFC, yerr = np.minimum(mseCentralPlotFCRange, np.sqrt(mseCentralPlotFC), np.divide(mseCentralPlotFC, 2)), color = 'red', marker = 'x', label = "Centralized CGM")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Labels")
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % dataset[dataIndex] + "_q.png")
    plt.clf()

    plt.errorbar(freqset, mseISquaredEPlotFA, yerr = np.minimum(mseISquaredEPlotFARange, np.sqrt(mseISquaredEPlotFA), np.divide(mseISquaredEPlotFA, 2)), color = 'blue', marker = 'o', label = "Empirical AGM")
    plt.errorbar(freqset, mseISquaredEPlotFC, yerr = np.minimum(mseISquaredEPlotFCRange, np.sqrt(mseISquaredEPlotFC), np.divide(mseISquaredEPlotFC, 2)), color = 'blue', marker = 'x', label = "Empirical CGM")
    plt.errorbar(freqset, mseISquaredTPlotFA, yerr = np.minimum(mseISquaredTPlotFARange, np.sqrt(mseISquaredTPlotFA), np.divide(mseISquaredTPlotFA, 2)), color = 'green', marker = 'o', label = "Theoretical AGM")
    plt.errorbar(freqset, mseISquaredTPlotFC, yerr = np.minimum(mseISquaredTPlotFCRange, np.sqrt(mseISquaredTPlotFC), np.divide(mseISquaredTPlotFC, 2)), color = 'green', marker = 'x', label = "Theoretical CGM")
    plt.errorbar(freqset, mseCentralPlotFA, yerr = np.minimum(mseCentralPlotFARange, np.sqrt(mseCentralPlotFA), np.divide(mseCentralPlotFA, 2)), color = 'red', marker = 'o', label = "Centralized AGM")
    plt.errorbar(freqset, mseCentralPlotFC, yerr = np.minimum(mseCentralPlotFCRange, np.sqrt(mseCentralPlotFC), np.divide(mseCentralPlotFC, 2)), color = 'red', marker = 'x', label = "Centralized CGM")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Labels")
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % dataset[dataIndex] + "_isquared.png")
    plt.clf()

def runLoopVaryEps(dataIndex, dim, num, newImages, labels, GS):
    runLoop(dataIndex, dim, num, newImages, labels, GS)

def runLoopVaryDta(dataIndex, dim, num, newImages, labels, GS):
    runLoop(dataIndex, dim, num, newImages, labels, GS)

runLoopVaryEps(0, dimCifar, numCifar, newImagesCifar10, labelsCifar10, GSCifar)
runLoopVaryEps(1, dimCifar, numCifar, newImagesCifar100, labelsCifar100, GSCifar)
runLoopVaryEps(2, dimFashion, numFashion, newImagesFashion, labelsFashion, GSFashion)

print("Finished.\n")