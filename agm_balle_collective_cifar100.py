import numpy as np
import time

# INITIALISING START TIME AND SEED FOR RANDOM SAMPLING
startTime = time.perf_counter
print("\nStarting...")
np.random.seed(3820672)

# ADAPTATION OF UNPICKLING OF CIFAR-10 FILES BY KRIZHEVSKY
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

# LOADING AND SPLITTING CIFAR-100 DATA
def loadData():
    print("Loading data...")
    filename = 'train'
    dict = unpickle(filename)
    xTrain = dict[b'data']
    return xTrain

# ADAPTATION OF TRANSFORMATION OF LABEL INDICES TO ONE-HOT ENCODED VECTORS AND IMAGES TO 3072-DIMENSIONAL VECTORS BY HADHAZI
def transformValues(x):
    x = x.astype('float32')
    x = x / 255.0
    return x

# CALL ALL THE ABOVE METHODS
xTrain = loadData()
xTrainNew = transformValues(xTrain)
xTrainSimple = np.full((50000, 3072), 0.5)

# ADAPTATION OF ANALYTIC GAUSSIAN MECHANISM BY BALLE AND WANG
from math import exp, sqrt, log
from scipy.special import erf
from numpy.random import normal

# ARRAYS STORING SETS OF VALUES OF EACH VARIABLE WITH OPTIMA CHOSEN AS CONSTANTS
epsset = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
epsconst = epsset[1]

# VECTOR DIMENSION CHOSEN TO MATCH THAT OF CONVERTED IMAGES ABOVE AND NUMBER OF CLIENTS CHOSEN TO GIVE SENSIBLE GS
dtaset = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
dtaconst = dtaset[1]
dset = [128, 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072]
dconst = dset[9]
nset = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
nconst = nset[8]
GS = float(sqrt(dconst))/nconst

# INITIALISING OTHER PARAMETERS/CONSTANTS
parset = ['eps', 'dta', 'd', 'n']
rset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
R = len(rset)
mseSum = 0
mseList = list()
maxDim = dset[9]
maxArraySize = nset[9]*maxDim

# IN THEORY TWO NOISE TERMS ARE ADDED WITH EACH USING EPS AND DTA HALF THE SIZE OF IN EXPERIMENTS
epsTheory = epsconst/2
dtaTheory = dtaconst/2
xiTheory = (2*dconst*log(1.25/dtaTheory))/((nconst**2)*(epsTheory**2))

def runLoop(xTrainChoice, index, var, epschoice, dtachoice, dchoice, nchoice):

    if np.all(element == 0.5 for element in xTrainChoice):
        datafile = open("c100_simple_file_" + "%s" % parset[index] + str(var) + ".txt", "w")
    else:
        datafile = open("c100_data_file_" + "%s" % parset[index] + str(var) + ".txt", "w")

    datafile.write("Statistics from Theory and Binary Search in AGM")
    datafile.write(f"\n\nxiTheory: {round(xiTheory, 7):>21}")

    def calibrateAGM(eps, dta, GS, tol = 1.e-12):
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
    sigma = calibrateAGM(epschoice, dtachoice, GS, tol = 1.e-12)
    print("Calibrating AGM...")
    datafile.write("\nStatistics from AGM and computation of MSE")
    datafile.write(f"\n\nsigma from AGM: {round(sigma, 4):>13}")
    datafile.write(f"\nsquare: {round(sigma**2, 8):>25}")

    # FUNCTION BY SCOTT BASED ON OWN LEMMAS THEOREMS AND COROLLARIES IN PAPER
    def computeMSE(xTrainChoice, dchoice, sigma, nchoice, mseSum):

        loopTime = time.perf_counter()
        varSum = 0

        if (dchoice != maxDim):
            xTrainCrop = xTrainChoice.reshape((int(maxArraySize/dchoice), dchoice))
            xTrainChoice = xTrainCrop

        mu = np.mean(xTrainChoice, axis = 0)
        datafile.write(f"\nmu: {str(round((sum(mu))/dchoice, 5)):>26}")
        muSquares = [a**2 for a in mu]
        datafile.write(f"\nsum of squares: {str(round((sum(muSquares))/dchoice, 5)):>14}")

        noisyMu = [0]*dchoice

        # ADDING FIRST NOISE TERM TO MU DERIVED FROM GAUSSIAN DISTRIBUTION WITH MEAN 0 AND VARIANCE SIGMA SQUARED
        for i in range(0, dchoice):
            xi1 = normal(0, sigma**2)
            noisyMu[i] = mu[i] + xi1
        datafile.write(f"\nmu + noise: {round((sum(noisyMu))/dchoice, 5):>18}")

        # FIRST SUBTRACTION BETWEEN CIFAR-10 VECTOR OF EACH CLIENT AND NOISY MEAN ACCORDING TO THEOREM FOR DISPERSION
        for j in range(0, nchoice):
            noisySigma = np.subtract(xTrainChoice[j], noisyMu)  
        datafile.write(f"\nsigma + noise: {round((sum(noisySigma))/nchoice, 4):>14}")

        # PREPARING EXPRESSION OF DISPERSION FOR ADDITION OF SECOND NOISE TERM
        twiceNoisySigma = np.multiply(noisySigma, 2)
        noisyVar = np.power(noisySigma, 2)
        datafile.write(f"\nsigma + twice noise: {round((sum(twiceNoisySigma))/nchoice, 4):>8}")
        datafile.write(f"\nvar + noise: {round((sum(noisyVar))/nchoice, 6):>18}")

        # ADDING SECOND NOISE TERM TO EXPRESSION OF DISPERSION AND COMPUTING MSE USING VARIABLES DEFINED ABOVE
        for j in range(0, nchoice):
            xi2 = normal(0, sigma**2)
            doubleNoisyVar = noisyVar + xi2
            varSum += sum(doubleNoisyVar)
            bracket = np.subtract(noisyMu, twiceNoisySigma)
            outside = np.multiply(bracket, noisyMu)
            mse = np.add(outside, xi2)          
            mseSum += sum(mse)
        
        mseList.append(mseSum/nchoice)

        datafile.write(f"\nvar + twice noise: {round(varSum/nchoice, 4):>11}")
        datafile.write(f"\nmse: {round(mseSum/nchoice, 4):>26}")

        casetime = time.perf_counter() - loopTime
        datafile.write(f"\ncalibration: {round(casetime, 2):>15} seconds\n")

    # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM ANALYTIC GAUSSIAN MECHANISM
    computeMSE(xTrainChoice, dchoice, sigma, nchoice, mseSum)
    print("Computing MSE...")

    # COMPUTE SIGMA USING CLASSIC GAUSSIAN MECHANISM FOR COMPARISON BETWEEN DISPERSION AND MSE OF BOTH
    classicSigma = (GS*sqrt(2*log(1.25/dtachoice)))/epschoice
    datafile.write("\nStatistics from classic GM and computation of MSE")
    datafile.write(f"\n\nsigma from classic GM: {round(classicSigma, 4)}")
    datafile.write(f"\nsquare: {round(classicSigma**2, 8):>22}")

    # CALL ALGORITHM TO COMPUTE MSE BASED ON SIGMA FROM CLASSIC GAUSSIAN MECHANISM
    computeMSE(xTrainChoice, dchoice, classicSigma, nchoice, mseSum)

    mseA = mseList.pop(0)
    mseB = mseList.pop(0)
    msediff = abs(mseB - mseA)
    print(msediff)
    decdiff = abs(msediff/mseB)
    print(decdiff)
    percdiff = decdiff*100
    print(percdiff)
    datafile.write("\nPercentages comparing AGM and classic GM")
    datafile.write(f"\n\ndifference between mse: {round(percdiff, 8)}%")

def runLoopVaryEps():
    for eps in epsset:
        print(f"\nProcessing the main loop for the value eps = {eps}.")
        runLoop(xTrainNew, 0, eps, eps, dtaconst, dconst, nconst)

def runLoopVaryDta():
    for dta in dtaset:
        print(f"\nProcessing the main loop for the value dta = {dta}.")
        runLoop(xTrainNew, 1, dta, epsconst, dta, dconst, nconst)

def runLoopVaryD():
    for d in dset:
        print(f"\nProcessing the main loop for the value d = {d}.")
        runLoop(xTrainNew, 2, d, epsconst, dtaconst, d, nconst)

def runLoopVaryN():
    for n in nset:
        print(f"\nProcessing the main loop for the value n = {n}.")
        runLoop(xTrainNew, 3, n, epsconst, dtaconst, dconst, n)

def simpleVaryEps():
    for eps in epsset:
        print(f"\nSimple case for the value eps = {eps}.")
        runLoop(xTrainSimple, 0, eps, eps, dtaconst, dconst, nconst)

def simpleVaryDta():
    for dta in dtaset:
        print(f"\nSimple case for the value dta = {dta}.")
        runLoop(xTrainSimple, 1, dta, epsconst, dta, dconst, nconst)

def simpleVaryD():
    for d in dset:
        print(f"\nSimple case for the value d = {d}.")
        runLoop(xTrainSimple, 2, d, epsconst, dtaconst, d, nconst)

def simpleVaryN():
    for n in nset:
        print(f"\nProcessing the main loop for the value n = {n}.")
        runLoop(xTrainSimple, 3, n, epsconst, dtaconst, dconst, n)

runLoopVaryEps()
runLoopVaryDta()
runLoopVaryD()
runLoopVaryN()

simpleVaryEps()
simpleVaryDta()
simpleVaryD()
simpleVaryN()

print("Finished.\n")