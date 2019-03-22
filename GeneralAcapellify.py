class GeneralAcapellify(object):
    def __init__(self, musicMagnitudeMatrix, basisMatrix, timeDomainMusic, timeDomainBases, windowSize, hopSize):
       self.musicMat = musicMagnitudeMatrix
       self.basisMat = basisMatrix
       self.timeDomainMusic = timeDomainMusic
       self.timeDomainBases = timeDomainBases
       self.hopSize = hopSize
       self.window_size = windowSize
       _, self.num_hops = self.musicMat.shape

    def createAcappella(self):
        """
        Creates the accepalafied music
        Returns
        -------
        the audio file that is the acappelafied music
        """
        raise NotImplementedError("The method was not implemented")
