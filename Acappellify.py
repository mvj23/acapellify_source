from importstatements import *
import helperFuncs as utils
import GeneralAcapellify
class Acappellify(GeneralAcapellify.GeneralAcapellify):
    """
    musicMagnitudeMatrix - the magnitude spectrogram of the original audio file, an n (# freqs) by m (#time stamps)
    basisMatrix - the concatenated basis, an n (# freqs) by r (# basis)
    timeDomainMusic - the original audio file in the time domain
    timeDomainBases - a list of the original audio files of the basis in the time domain
    """

    def __init__(self, musicMagnitudeMatrix, basisMatrix, timeDomainMusic, timeDomainBases, windowSize, hopSize):
        super(Acappellify, self).__init__(musicMagnitudeMatrix, basisMatrix, timeDomainMusic, timeDomainBases, windowSize, hopSize)

    def doPerSpectroBin(self, func, totalSpectroLength, *params):
        """
        Applies the provided function to for every spectroBinLength elements
        :param totalSpectroLength - The length of the spectrogram (# of bins)
        :param func - a function of (params, timeLocation, endLocation, binLocation, result from last call to func)
        """
        totalOriginalLength = self.timeDomainMusic.size
        #timePerSpectroBin = totalOriginalLength // totalSpectroLength
        lastCall = None
        for n in range(self.num_hops):
            start = n * self.hopSize
            end = start + self.window_size
            lastCall = func(*params, timeLocation=start, endLoc=end, binLocation=n, lastCall=lastCall)

        # for timeLocation in range(0, totalOriginalLength, timePerSpectroBin):
        #     binLocation = Acappellify.__getBin(timeLocation, timePerSpectroBin)
        #     endLoc = min(timeLocation + timePerSpectroBin, totalOriginalLength)
        #     lastCall = func(*params, timeLocation=timeLocation, endLoc=endLoc, binLocation=binLocation, lastCall=lastCall)

    def createAcappella(self):
        """
        Creates the accepalafied music
        Returns
        -------
        the audio file that is the acappelafied music
        """
        def calcResultingAudio(resultingAudio, normalizedBasisMags, repeatedBasis, timeLocation, endLoc, binLocation, lastCall):
            if binLocation < normalizedBasisMags[basisIndex].size:
                magnitude = normalizedBasisMags[basisIndex, binLocation]
            else:
                magnitude = lastCall
            repeatedBasisSection = repeatedBasis[timeLocation:endLoc]
            hannWindow = sp.signal.windows.tukey(repeatedBasisSection.size, alpha=0.1)
            repeatedBasisSection = repeatedBasisSection * hannWindow
            resultingAudio[timeLocation:endLoc] += magnitude * repeatedBasisSection
            return magnitude

        basisMags = (self.getBasisMagnitudeMatrix())
        totalSpectroLength = basisMags.shape[1]

        normalizedBasisMags = self.__normalize(basisMags, totalSpectroLength)

        totalOriginalLength = self.timeDomainMusic.size

        resultingAudio = np.zeros(totalOriginalLength)

        for basisIndex, basis in enumerate(self.getTimeDomainBases()):
            repeatedBasis = self.__getRepeatedBasis(basis)
            self.doPerSpectroBin(calcResultingAudio, totalSpectroLength, resultingAudio, normalizedBasisMags, repeatedBasis)
        return resultingAudio

    def __getRepeatedBasis(self, basis):
        totalOriginalLength = self.timeDomainMusic.size
        repeatedBasis = np.tile(basis, totalOriginalLength // basis.size)
        repeatedBasis = np.concatenate((repeatedBasis, repeatedBasis[:totalOriginalLength % basis.size]))
        return repeatedBasis

    @staticmethod
    def __getBin(realTimeStamp, timePerSpectroBin):
        return realTimeStamp // timePerSpectroBin

    def getTimeDomainBases(self):
        """returns the basis that will be used to recontrusct the signal"""
        return self.timeDomainBases

    def getBasisMagnitudeMatrix(self):
        """
        Divides the music matrix (an nxm matrix) (n= #freqs, m=#time bins) and the basis matrix (an nxr matrix, r is the # of basis)

        Returns
        -------
        basisMagnitudeMatrix (H) (rxm) the solution to V = WH
       """
        print(self.musicMat.shape)
        print("\n")
        foundSolutions = []
        for timeColumn in self.musicMat.T:
            sol = sp.optimize.lsq_linear(A=self.basisMat, b=timeColumn, bounds=(0,np.inf)).x
            sol = np.reshape(sol, (sol.size, 1))
            foundSolutions.append(sol)
        basisMagnitudeMatrix = np.hstack(foundSolutions)
        return basisMagnitudeMatrix

    @staticmethod
    def findMedianAmplitudeAt(medianList, timeDomainMusic, timeLocation, endLoc, binLocation, lastCall):
        medianList.append(np.median(np.abs(timeDomainMusic[timeLocation:endLoc])))

    def __normalize(self, basisMagnitudeMatrix, totalSpectroLength, useAmplitude=True):
        """
        Normalizes the H matrix based on the amplitude of the original music

        Returns
        -------
        H but with all the values normalized
        """
        def findMedianAt(medianAtList, timeDomainMusic, timeLocation, endLoc, binLocation, lastCall):
            #finds the median DB for each bin
            medianAtList.append(utils.median_db(timeDomainMusic[timeLocation:endLoc], 1, 1))


        originalSongMedians = [] #the median amplitude of the original song at each of the bins
        if useAmplitude:
            self.doPerSpectroBin(Acappellify.findMedianAmplitudeAt, totalSpectroLength, originalSongMedians, self.timeDomainMusic)
        else:
            totalMedianDb = utils.median_db(self.timeDomainMusic, 1,1)
            self.doPerSpectroBin(findMedianAt, totalSpectroLength, originalSongMedians, self.timeDomainMusic)
            originalSongMedians /= totalMedianDb
        originalSongMedians = np.array(originalSongMedians)


        medianBasisAmplitudeSum = [] #the matrix (rxk), r=#basis, k=#bins of the median amplitudes of the repeated basis at each bin
        for base in self.getTimeDomainBases():
            medianBasis = []
            self.doPerSpectroBin(Acappellify.findMedianAmplitudeAt, totalSpectroLength, medianBasis, self.__getRepeatedBasis(base))
            medianBasisAmplitudeSum.append(medianBasis)
        medianBasisAmplitudeSum = np.array(medianBasisAmplitudeSum)


        summedColumns = np.multiply(medianBasisAmplitudeSum[:, :basisMagnitudeMatrix.shape[1]], basisMagnitudeMatrix)
        summedColumns = np.sum(summedColumns, axis=0)

        #summedColumns = np.sum(basisMagnitudeMatrix, axis=0)


        #denominator = np.multiply(summedColumns, medianBasisAmplitudeSum[:summedColumns.size])
        numerator = originalSongMedians[:summedColumns.size]
        denom = summedColumns
        normalizationFactor = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom!=0)
        finalResult = basisMagnitudeMatrix * normalizationFactor
        return finalResult


#Uncomment to debug
if __name__== "__main__":
    basis = []
    basis.append(np.array([1,2,3,4,5,6,7]))
    basis.append(np.array([5,6,7]))
    ac = Acappellify(np.array([[1,2,3,4], [4,5,6,4], [7,8,9,4], [1,8,3,4]]),
                     np.array([[1,2], [4,5], [6,7], [8,9]]),
                     np.array(np.random.rand(10)),
                     basis, 2)
    x = ac.createAcappella()
    print(x)




