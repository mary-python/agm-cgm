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
dataset = np.array(['Cifar10', 'Cifar100', 'Fashion'])
parset = np.array(['eps', 'dta'])
graphset = np.array(['$\mathit{\u03b5}$', '$\mathit{\u03b4}$'])
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

def runLoop(dataIndex, index, varset, dim, num, eps, dta, newImages, labels, GS):
    
    F = len(freqset)
    V = len(varset)

    mseDispEPlotA = np.zeros((F, V))
    mseQEPlotA = np.zeros((F, V))
    mseDispEPlotASD = np.zeros((F, V))
    mseQEPlotASD = np.zeros((F, V))

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

        mseDispEPlotATemp = np.zeros((F, V, R))
        mseQEPlotATemp = np.zeros((F, V, R))

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

                # TABLES ASSUME UNIFORM DATA
                if fi == 0 and val == 0 and index == 1:
                    mseDispETableATemp[rep] = mseE
                    mseDispTTableATemp[rep] = mseT
                    mseQETableATemp[rep] = mseQE
                    mseQTTableATemp[rep] = mseQT

                # STATISTICAL HETEROGENEITY GRAPHS
                mseDispEPlotATemp[fi, val, rep] = mseE
                mseQEPlotATemp[fi, val, rep] = mseQE

            else:
                if fi == 0 and val == 0 and index == 1:
                    mseDispETableCTemp[rep] = mseE
                    mseDispTTableCTemp[rep] = mseT
                    mseQETableCTemp[rep] = mseQE
                    mseQTTableCTemp[rep] = mseQT

            mseISEList = np.zeros(sampleSize)
            mseISTList = np.zeros(sampleSize)

            for j in range(0, sampleSize):

                # COMPUTE I^2'' and I^2 USING SIMPLE FORMULA AT BOTTOM OF LEMMA 6.2
                trueI2Prep = np.divide(sampleSize-1, weightedTrueDisp)
                trueI2 = np.subtract(1, trueI2Prep)
                I2Prep = np.divide(sampleSize-1, noisyQ)
                I2True = np.subtract(1, I2Prep)

                # ADD THIRD NOISE TERM BASED ON LEMMA 6.2
                xi3 = normal(0, sigma**2)
                noisyI2 = np.add(I2True, xi3)

                # COMPUTE EMSE AND TMSE
                diffEI2 = np.subtract(noisyI2, trueI2)
                mseISEList[j] = np.power(diffEI2, 2)
                diffTI2Prep = np.subtract(xi3, I2True)
                diffTI2 = np.add(diffTI2Prep, trueI2)
                mseISTList[j] = np.power(diffTI2, 2)

            mseI2E = np.sum(mseISEList)
            mseI2T = np.sum(mseISTList)

            # EXPERIMENT 2: WHAT IS THE COST OF A DISTRIBUTED SETTING?
            xiCentral = normal(0, centralSigma**2)
            mseC = xiCentral**2

            if fi == 0 and val == 0 and index == 1:
                if ACindex == 0:           
                    mseI2ETableATemp[rep] = mseI2E
                    mseI2TTableATemp[rep] = mseI2T
                    mseCTableATemp[rep] = mseC

                else:
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

            if fi == 0 and val == 0 and index == 1:
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

                mseDispETableARound = round(mseDispETableA, 14)
                mseDispETableCRound = round(mseDispETableC, 12)
                mseDispTTableARound = round(mseDispTTableA, 14)
                mseDispTTableCRound = round(mseDispTTableC, 12)
                mseQETableARound = round(mseQETableA, 14)
                mseQETableCRound = round(mseQETableC, 12)
                mseQTTableARound = round(mseQTTableA, 14)
                mseQTTableCRound = round(mseQTTableC, 11)
                mseCentralTableARound = round(mseCentralTableA, 3)
                mseCentralTableCRound = round(mseCentralTableC, 1)

                mseDispETableASD = round(np.std(mseDispETableATemp), 16)
                mseDispETableCSD = round(np.std(mseDispETableCTemp), 13)
                mseDispTTableASD = round(np.std(mseDispTTableATemp), 16)
                mseDispTTableCSD = round(np.std(mseDispTTableCTemp), 13)
                mseQETableASD = round(np.std(mseQETableATemp), 16)
                mseQETableCSD = round(np.std(mseQETableCTemp), 13)
                mseQTTableASD = round(np.std(mseQTTableATemp), 16)
                mseQTTableCSD = round(np.std(mseQTTableCTemp), 13)
                mseCentralTableASD = round(np.std(mseCTableATemp), 3)
                mseCentralTableCSD = round(np.std(mseCTableCTemp), 1)

                mseDispETableAC = np.round(np.divide(mseDispETableA, mseDispETableC), 7)
                mseDispTTableAC = np.round(np.divide(mseDispTTableA, mseDispTTableC), 10)
                mseDispETTableA = np.round(np.divide(mseDispETableA, mseDispTTableA), 4)
                mseDispETTableC = np.round(np.divide(mseDispETableC, mseDispTTableC), 4)
                mseQETableAC = np.round(np.divide(mseQETableA, mseQETableC), 9)
                mseQTTableAC = np.round(np.divide(mseQTTableA, mseQTTableC), 7)
                mseQETTableA = np.round(np.divide(mseQETableA, mseQTTableA), 4)
                mseQETTableC = np.round(np.divide(mseQETableC, mseQTTableC), 4)
                mseCentralTableAC = np.round(np.divide(mseCentralTableA, mseCentralTableC), 7)
                mseI2ETableAC = np.round(np.divide(mseI2ETableA, mseI2ETableC), 7)
                mseI2TTableAC = np.round(np.divide(mseI2TTableA, mseI2TTableC), 7)
                mseI2ETTableA = np.round(np.divide(mseI2ETableA, mseI2TTableA), 4)
                mseI2ETTableC = np.round(np.divide(mseI2ETableC, mseI2TTableC), 4)

            mseDispEPlotA[fi, val] = np.mean(mseDispEPlotATemp[fi, val])
            mseQEPlotA[fi, val] = np.mean(mseQEPlotATemp[fi, val])
            mseDispEPlotASD[fi, val] = np.std(mseDispEPlotATemp[fi, val])
            mseQEPlotASD[fi, val] = np.std(mseQEPlotATemp[fi, val])

    # EXPERIMENT 1: COMPARISON OF AGM/CGM, EMSE/TMSE AND CMSE
    if index == 1:
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
    plt.errorbar(varset, mseDispEPlotA[0], yerr = np.minimum(mseDispEPlotASD[0], np.sqrt(mseDispEPlotA[0]), np.divide(mseDispEPlotA[0], 2)), color = 'blue', marker = 'o', label = freqset[0])
    plt.errorbar(varset, mseDispEPlotA[1], yerr = np.minimum(mseDispEPlotASD[1], np.sqrt(mseDispEPlotA[1]), np.divide(mseDispEPlotA[1], 2)), color = 'blueviolet', marker = 'x', label = freqset[1])
    plt.errorbar(varset, mseDispEPlotA[2], yerr = np.minimum(mseDispEPlotASD[2], np.sqrt(mseDispEPlotA[2]), np.divide(mseDispEPlotA[2], 2)), color = 'green', marker = 'o', label = freqset[2])
    plt.errorbar(varset, mseDispEPlotA[3], yerr = np.minimum(mseDispEPlotASD[3], np.sqrt(mseDispEPlotA[3]), np.divide(mseDispEPlotA[3], 2)), color = 'lime', marker = 'x', label = freqset[3])
    plt.errorbar(varset, mseDispEPlotA[4], yerr = np.minimum(mseDispEPlotASD[4], np.sqrt(mseDispEPlotA[4]), np.divide(mseDispEPlotA[4], 2)), color = 'orange', marker = 'o', label = freqset[4])
    plt.errorbar(varset, mseDispEPlotA[5], yerr = np.minimum(mseDispEPlotASD[5], np.sqrt(mseDispEPlotA[5]), np.divide(mseDispEPlotA[5], 2)), color = 'gold', marker = 'x', label = freqset[5])
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_disp.png")
    plt.clf()

    plt.errorbar(varset, mseQEPlotA[0], yerr = np.minimum(mseQEPlotASD[0], np.sqrt(mseQEPlotA[0]), np.divide(mseQEPlotA[0], 2)), color = 'blue', marker = 'o', label = freqset[0])
    plt.errorbar(varset, mseQEPlotA[1], yerr = np.minimum(mseQEPlotASD[1], np.sqrt(mseQEPlotA[1]), np.divide(mseQEPlotA[1], 2)), color = 'blueviolet', marker = 'x', label = freqset[1])
    plt.errorbar(varset, mseQEPlotA[2], yerr = np.minimum(mseQEPlotASD[2], np.sqrt(mseQEPlotA[2]), np.divide(mseQEPlotA[2], 2)), color = 'green', marker = 'o', label = freqset[2])
    plt.errorbar(varset, mseQEPlotA[3], yerr = np.minimum(mseQEPlotASD[3], np.sqrt(mseQEPlotA[3]), np.divide(mseQEPlotA[3], 2)), color = 'lime', marker = 'x', label = freqset[3])
    plt.errorbar(varset, mseQEPlotA[4], yerr = np.minimum(mseQEPlotASD[4], np.sqrt(mseQEPlotA[4]), np.divide(mseQEPlotA[4], 2)), color = 'orange', marker = 'o', label = freqset[4])
    plt.errorbar(varset, mseQEPlotA[5], yerr = np.minimum(mseQEPlotASD[5], np.sqrt(mseQEPlotA[5]), np.divide(mseQEPlotA[5], 2)), color = 'gold', marker = 'x', label = freqset[5])
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("EMSE of Gaussian Mechanism")
    plt.savefig("Graph_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_q.png")
    plt.clf()

def runLoopVaryEps(dataIndex, index, dim, num, newImages, labels, GS):
    runLoop(dataIndex, index, epsset, dim, num, -1, dtaconst, newImages, labels, GS)

def runLoopVaryDta(dataIndex, index, dim, num, newImages, labels, GS):
    runLoop(dataIndex, index, dtaset, dim, num, epsconst, -1, newImages, labels, GS)

runLoopVaryEps(0, 0, dimCifar, numCifar, newImagesCifar10, labelsCifar10, GSCifar)
runLoopVaryDta(0, 1, dimCifar, numCifar, newImagesCifar10, labelsCifar10, GSCifar)

runLoopVaryEps(1, 0, dimCifar, numCifar, newImagesCifar100, labelsCifar100, GSCifar)
runLoopVaryDta(1, 1, dimCifar, numCifar, newImagesCifar100, labelsCifar100, GSCifar)

runLoopVaryEps(2, 0, dimFashion, numFashion, newImagesFashion, labelsFashion, GSFashion)
runLoopVaryDta(2, 1, dimFashion, numFashion, newImagesFashion, labelsFashion, GSFashion)

print("Finished.\n")