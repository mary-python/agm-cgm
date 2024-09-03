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
epsset = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
epsconst = float(epsset[0])

# VECTOR DIMENSION CHOSEN TO MATCH THAT OF CONVERTED IMAGES ABOVE AND NUMBER OF CLIENTS CHOSEN TO GIVE SENSIBLE GS
dtaset = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05])
dtaconst = float(dtaset[1])

dsetCifar = np.array([512, 768, 1024, 1280, 1536, 1875, 2048, 2400, 2560, 3072])
dsetFashion = np.array([392, 525, 588, 600, 625, 640, 672, 700, 735, 784])
dsetFlair = np.array([768, 1536, 3072, 4800, 6144, 7680, 8192, 9375, 10240, 12288], dtype = np.int64)

dconstCifar = maxDimCifar = int(dsetCifar[9])
dconstFashion = maxDimFashion = int(dsetFashion[9])
dconstFlair = maxDimFlair = int(dsetFlair[9])

nsetCifar = np.array([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000])
nsetFashion = np.array([15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000])
nsetFlair = np.array([25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000], dtype = np.int64)

nconstCifar = int(nsetCifar[8])
nconstFashion = int(nsetFashion[8])
nconstFlair = int(nsetFlair[8])

maxNumCifar = int(nsetCifar[9])
maxNumFashion = int(nsetFashion[9])
maxNumFlair = int(nsetFlair[9])

GSCifar = float(mp.sqrt(dconstCifar))/nconstCifar
GSFashion = float(mp.sqrt(dconstFashion))/nconstFashion
GSFlair = float(mp.sqrt(dconstFlair))/nconstFlair

maxArraySizeCifar = dconstCifar*maxNumCifar
maxArraySizeFashion = dconstFashion*maxNumFashion
maxArraySizeFlair = dconstFlair*maxNumFlair

# INITIALISING OTHER PARAMETERS/CONSTANTS
dataset = np.array(['Cifar10', 'Cifar100', 'Fashion', 'Flair'])
parset = np.array(['eps', 'dta', 'd', 'n'])
graphset = np.array(['$\mathit{\u03b5}$', '$\mathit{\u03b4}$', 'd', 'n']) 
rset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
R = len(rset)

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

xTrainNewCifar10 = transformValues(xTrainCifar10)
xTrainNewCifar100 = transformValues(xTrainCifar100)
xTrainNewFashion = transformValues(xTrainFashion)
xTrainNewFlair = transformValues(xTrainFlair)

os.chdir('..')

def runLoop(dataIndex, index, varset, dchoice, nchoice, epschoice, dtachoice, xTrainNew, GS, maxArraySize):

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
    acDispEPlot = np.zeros(V)
    acDispTPlot = np.zeros(V)
    acQEPlot = np.zeros(V)
    acQTPlot = np.zeros(V)
    acISquaredEPlot = np.zeros(V)
    acISquaredTPlot = np.zeros(V)

    for rep in range(10):

        var = varset[rep]
        print(f"\nProcessing dataset {dataIndex+1} for the value {parset[index]} = {var}.")

        if epschoice == -1:
            epschoice = var
        elif dtachoice == -1:
            dtachoice = var
        elif dchoice == -1:
            dchoice = var
        elif nchoice == -1:
            nchoice = var

        if dataIndex == 0:
            datafile = open("cifar10_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
        elif dataIndex == 1:
            datafile = open("cifar100_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
        elif dataIndex == 2:
            datafile = open("fashion_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
        else:
            datafile = open("flair_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")

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

            sigma = alpha*GS/mp.sqrt(2.0*eps)
            return sigma

        # CALL ALGORITHM FOR AGM TO FIND SIGMA GIVEN EPS AND DTA AS INPUT
        sigma = calibrateAGM(epschoice, dtachoice, GS, tol=1.e-12)
        print("Calibrating AGM...")

        compareEListA = np.zeros(nchoice)
        compareQEListA = np.zeros(nchoice)
        compareISEListA = np.zeros(nchoice)
        compareEListC = np.zeros(nchoice)
        compareQEListC = np.zeros(nchoice)
        compareISEListC = np.zeros(nchoice)
        compareTListA = np.zeros(nchoice)
        compareQTListA = np.zeros(nchoice)
        compareISTListA = np.zeros(nchoice)
        compareTListC = np.zeros(nchoice)
        compareQTListC = np.zeros(nchoice)
        compareISTListC = np.zeros(nchoice)

        # EXPERIMENT 1: BEHAVIOUR OF VARIABLES AT DIFFERENT SETTINGS
        def computeMSE(xTrainNew, sigma, ACindex):

            if index == 2:
                xTrainCrop = xTrainNew.reshape((int(maxArraySize/dchoice), dchoice))
            else:
                xTrainCrop = xTrainNew.reshape((int(maxArraySize/dchoice), dchoice))
            xTrainNew = xTrainCrop
        
            # INITIAL COMPUTATION OF WEIGHTED MEAN FOR Q BASED ON VECTOR VARIANCE
            wVector = np.var(xTrainNew, axis=1)
            weight = np.zeros(nchoice)
            wxTrainNew = np.zeros((nchoice, dchoice))       

            for j in range(0, nchoice):
                wVectorSquared = np.power(wVector[j], 2)
                weight[j] = 1.0/(wVectorSquared)

                # MULTIPLYING EACH VECTOR BY ITS CORRESPONDING WEIGHTED MEAN
                wxTrainNew[j] = (weight[j])*(xTrainNew[j])

            mu = np.mean(xTrainNew, axis=0)
            wSumMu = np.sum(wxTrainNew, axis=0)

            # DIVIDING SUM OF WEIGHTED VECTORS BY SUM OF WEIGHTS
            sumWeight = np.sum(weight)
            wMu = (wSumMu)/sumWeight

            noisyMu = np.zeros(dchoice)
            wNoisyMu = np.zeros(dchoice)

            noisyEList = np.zeros(nchoice)
            noisyQEList = np.zeros(nchoice)
            trueEList = np.zeros(nchoice, dtype = np.float64)
            trueQEList = np.zeros(nchoice, dtype = np.float64)
            mseEList = np.zeros(nchoice)
            mseQEList = np.zeros(nchoice)
            mseTList = np.zeros(nchoice, dtype = np.float64)
            mseQTList = np.zeros(nchoice, dtype = np.float64)

            # ADDING FIRST NOISE TERM TO MU DERIVED FROM GAUSSIAN DISTRIBUTION WITH MEAN 0 AND VARIANCE SIGMA SQUARED
            for i in range(0, dchoice):
                xi1 = normal(0, sigma**2)
                noisyMu[i] = mu[i] + xi1
                wNoisyMu[i] = wMu[i] + xi1

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
            datafile.write(f"\n\ndispersion empirical mse: {round(mseEmpirical)}")
            datafile.write(f"\ndispersion theoretical mse: {round(mseTheoretical, 3)}")
            datafile.write(f"\n\nq empirical mse: {round(mseQEmpirical)}")
            datafile.write(f"\nq theoretical mse: {round(mseQTheoretical)}")

            if ACindex == 0:
                np.copyto(compareEListA, mseEList)
                np.copyto(compareQEListA, mseQEList)
                np.copyto(compareTListA, mseTList)
                np.copyto(compareQTListA, mseQTList)
                mseDispEPlotA[rep] = mseEmpirical
                mseQEPlotA[rep] = mseQEmpirical
                mseDispTPlotA[rep] = mseTheoretical
                mseQTPlotA[rep] = mseQTheoretical
            else:
                np.copyto(compareEListC, mseEList)
                np.copyto(compareQEListC, mseQEList)
                np.copyto(compareTListC, mseTList)
                np.copyto(compareQTListC, mseQTList)
                mseDispEPlotC[rep] = mseEmpirical
                mseQEPlotC[rep] = mseQEmpirical
                mseDispTPlotC[rep] = mseTheoretical
                mseQTPlotC[rep] = mseQTheoretical

            trueISquaredList = np.zeros(nchoice)
            iSquaredList = np.zeros(nchoice)
            mseISEList = np.zeros(nchoice)
            mseISTList = np.zeros(nchoice)

            for j in range(0, nchoice):

                # COMPUTE I^2'' and I^2 USING SIMPLE FORMULA AT BOTTOM OF LEMMA 6.2
                trueISquaredPrep = np.divide(nchoice-1, trueQEList[j])
                trueISquaredList[j] = np.subtract(1, trueISquaredPrep)
                iSquaredPrep = np.divide(nchoice-1, noisyQEList[j])
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
            datafile.write(f"\n\nisquared empirical mse: {round(mseISquaredEmpirical, 5)}")
            datafile.write(f"\nisquared theoretical mse: {round(mseISquaredTheoretical, 5)}")

            if ACindex == 0:
                np.copyto(compareISEListA, mseISEList)
                np.copyto(compareISTListA, mseISTList)
                mseISquaredEPlotA[rep] = mseISquaredEmpirical
                mseISquaredTPlotA[rep] = mseISquaredTheoretical
            else:
                np.copyto(compareISEListC, mseISEList)
                np.copyto(compareISTListC, mseISTList)
                mseISquaredEPlotC[rep] = mseISquaredEmpirical
                mseISquaredTPlotC[rep] = mseISquaredTheoretical

            # 95% CONFIDENCE INTERVALS USING SIGMA, Z-SCORE AND WEIGHTS IF RELEVANT
            confInt = (7.84*(mp.sqrt(6))*(sigma**4))/(mp.sqrt(nchoice))
            qConfInt = np.sum((7.84*weight*(mp.sqrt(6))*(sigma**4))/(mp.sqrt(nchoice)))        
            iSquaredConfInt = np.sum((7.84*(mp.sqrt(2*(nchoice-1))))/(3*(np.sqrt(35*weight*nchoice))*(sigma**4)))
            datafile.write(f"\n\n95% CI for dispersion: \u00B1 {round(confInt, 10)}")
            datafile.write(f"\n95% C Interval for q: \u00B1 {round(qConfInt, 4)}")
            datafile.write(f"\n95% CI for isquared: \u00B1 {round(iSquaredConfInt)}")

        # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM ANALYTIC GAUSSIAN MECHANISM
        datafile.write("EXPERIMENT 1: MSE OF ANALYTIC GAUSSIAN MECHANISM")
        computeMSE(xTrainNew, sigma, 0)
        print("Computing empirical and theoretical MSEs...")

        # COMPUTE SIGMA USING CLASSIC GAUSSIAN MECHANISM FOR COMPARISON BETWEEN DISPERSION AND MSE OF BOTH
        classicSigma = (GS*mp.sqrt(2*mp.log(1.25/dtachoice)))/epschoice
    
        # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM CLASSIC GAUSSIAN MECHANISM
        datafile.write("\n\nEXPERIMENT 1: MSE OF CLASSIC GAUSSIAN MECHANISM")
        computeMSE(xTrainNew, classicSigma, 1)

        # EXPERIMENT 2: AGM VS CGM
        datafile.write("\n\nEXPERIMENT 2: ANALYTIC VS CLASSIC GAUSSIAN MECHANISM")
        comparelists1 = np.divide(compareEListA, compareEListC)
        compareqlists1 = np.divide(compareQEListA, compareQEListC)
        compareislists1 = np.divide(compareISEListA, compareISEListC)
        sumdiff1 = abs(np.mean(comparelists1))
        sumqdiff1 = abs(np.mean(compareqlists1))
        sumisdiff1 = abs(np.mean(compareislists1))
        acDispEPlot[rep] = sumdiff1
        acQEPlot[rep] = sumqdiff1
        acISquaredEPlot[rep] = sumisdiff1
        datafile.write(f"\n\nempirical mse comparison: {round(sumdiff1)}x")
        datafile.write(f"\nempirical q comparison: {round(sumqdiff1)}x")
        datafile.write(f"\nempirical isquared comparison: {round(sumisdiff1)}x")

        comparelists2 = np.divide(compareTListA, compareTListC)
        compareqlists2 = np.divide(compareQTListA, compareQTListC)
        compareislists2 = np.divide(compareISTListA, compareISTListC)
        sumdiff2 = abs(np.mean(comparelists2))
        sumqdiff2 = abs(np.mean(compareqlists2))
        sumisdiff2 = abs(np.mean(compareislists2))
        acDispTPlot[rep] = sumdiff2
        acQTPlot[rep] = sumqdiff2
        acISquaredTPlot[rep] = sumisdiff2
        datafile.write(f"\n\ntheoretical mse comparison: {round(sumdiff2, 4)}x")
        datafile.write(f"\ntheoretical q comparison: {round(sumqdiff2, 4)}x")
        datafile.write(f"\ntheoretical isquared comparison: {round(sumisdiff2)}x")

        # EXPERIMENT 3: WHAT IS THE COST OF A DISTRIBUTED SETTING?

        # HYPOTHESIS: (CLOSE TO) ZERO COST, TO MATCH THEORY
        # COMPARE TO CENTRALISED SETTING
        # APPLY TO DISPERSION, Q, I SQUARED AND CONFIDENCE INTERVALS

        # EXPERIMENT 4: VECTOR ALLOCATIONS TESTING ROBUSTNESS OF DISTRIBUTED CASE

        # USE IDEAS FROM SPLITTING EMNIST DATASET BY DIGIT (OR FEMNIST BY WRITER) AND MEASURING PIXEL FUNCTION
        # DIFFERENT LEVELS OF HETEROGENEITY: USE DIFFERENT SIZED ARRAYS OF CLIENTS
        # DO THESE LEVELS AFFECT THE METHOD? HYPOTHESIS: NOT MUCH
        # MEANS NOTHING IN CENTRALISED CASE BECAUSE SERVER HAS ALL DATA

    # EXPERIMENT 1: BEHAVIOUR OF (EPSILON, DELTA, D, N)
    plt.errorbar(varset, mseDispEPlotA, color = 'blue', marker = 'o', label = "Empirical Analytic")
    plt.errorbar(varset, mseDispEPlotA, color = 'green', marker = 'o', label = "Theoretical Analytic")
    plt.errorbar(varset, mseDispTPlotC, color = 'orange', marker = 'x', label = "Empirical Classic")
    plt.errorbar(varset, mseDispTPlotC, color = 'red', marker = 'x', label = "Theoretical Classic")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Exp1_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_disp.png")
    plt.clf()

    plt.errorbar(varset, mseQEPlotA, color = 'blue', marker = 'o', label = "Empirical Analytic")
    plt.errorbar(varset, mseQEPlotA, color = 'green', marker = 'o', label = "Theoretical Analytic")
    plt.errorbar(varset, mseQTPlotC, color = 'orange', marker = 'x', label = "Empirical Classic")
    plt.errorbar(varset, mseQTPlotC, color = 'red', marker = 'x', label = "Theoretical Classic")
    plt.legend(loc = 'best')
    plt.yscale('log')
    plt.xlabel("Value of " + "%s" % graphset[index])
    plt.ylabel("MSE of Gaussian Mechanism")
    plt.savefig("Exp1_" + "%s" % dataset[dataIndex] + "_vary_" + "%s" % parset[index] + "_q.png")
    plt.clf()

    plt.errorbar(varset, mseISquaredEPlotA, color = 'blue', marker = 'o', label = "Empirical Analytic")
    plt.errorbar(varset, mseISquaredEPlotA, color = 'green', marker = 'o', label = "Theoretical Analytic")
    plt.errorbar(varset, mseISquaredTPlotC, color = 'orange', marker = 'x', label = "Empirical Classic")
    plt.errorbar(varset, mseISquaredTPlotC, color = 'red', marker = 'x', label = "Theoretical Classic")
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

    # ADD GRAPHS FOR EXPERIMENTS 3 AND 4 WHEN READY

def runLoopVaryEps(dataIndex, index, dconst, nconst, xTrainNew, GS, maxArraySize):
    runLoop(dataIndex, index, epsset, dconst, nconst, -1, dtaconst, xTrainNew, GS, maxArraySize)

def runLoopVaryDta(dataIndex, index, dconst, nconst, xTrainNew, GS, maxArraySize):
    runLoop(dataIndex, index, dtaset, dconst, nconst, epsconst, -1, xTrainNew, GS, maxArraySize)

def runLoopVaryD(dataIndex, index, dset, nconst, xTrainNew, GS, maxArraySize):
    runLoop(dataIndex, index, dset, -1, nconst, epsconst, dtaconst, xTrainNew, GS, maxArraySize)

def runLoopVaryN(dataIndex, index, dconst, nset, xTrainNew, GS, maxArraySize):
    runLoop(dataIndex, index, nset, dconst, -1, epsconst, dtaconst, xTrainNew, GS, maxArraySize)

runLoopVaryEps(0, 0, dconstCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar)
runLoopVaryDta(0, 1, dconstCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar)
runLoopVaryD(0, 2, dsetCifar, nconstCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar)
runLoopVaryN(0, 3, dconstCifar, nsetCifar, xTrainNewCifar10, GSCifar, maxArraySizeCifar)

runLoopVaryEps(1, 0, dconstCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar)
runLoopVaryDta(1, 1, dconstCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar)
runLoopVaryD(1, 2, dsetCifar, nconstCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar)
runLoopVaryN(1, 3, dconstCifar, nsetCifar, xTrainNewCifar100, GSCifar, maxArraySizeCifar)

runLoopVaryEps(2, 0, dconstFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion)
runLoopVaryDta(2, 1, dconstFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion)
runLoopVaryD(2, 2, dsetFashion, nconstFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion)
runLoopVaryN(2, 3, dconstFashion, nsetFashion, xTrainNewFashion, GSFashion, maxArraySizeFashion)

runLoopVaryEps(3, 0, dconstFlair, nconstFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair)
runLoopVaryDta(3, 1, dconstFlair, nconstFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair)
runLoopVaryD(3, 2, dsetFlair, nconstFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair)
runLoopVaryN(3, 3, dconstFlair, nsetFlair, xTrainNewFlair, GSFlair, maxArraySizeFlair)

print("Finished.\n")