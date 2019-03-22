from importstatements import *
from scipy.fftpack import fft

def abs_stft(signal, window_size, hop_size, window_type='hann'):
    """
    Computes the short term fourier transform of a 1-D numpy array, where the array
    is windowed into a set of subarrays, each of length window_size. The distance between
    window centers (in samples) is given by hop_size. The type of window applied is
    determined by window_type. This returns a 2-D numpy array where the ith column
    is the FFT of the ith window. Each column contains an array of complex values.

    Input Parameters
    ----------------
    signal: The 1-d (complex or real) numpy array containing the signal
    window_size: an integer scalar specifying the number of samples in a window
    hop_size: an integer specifying the number of samples between the start of adjacent windows
    window_type: a string specifying one of two "hann" or "rectangular"

    Returns
    -------
    a 2D numpy array of complex numbers where the array column is the FFT of the ith window,
    and the jth element in the ith column is the jth frequency of analysis.
    """

    # figure out how many hops
    length_to_cover_with_hops = len(signal) - window_size;
    assert (length_to_cover_with_hops >= 0), "window_size cannot be longer than the signal to be windowed"
    num_hops = int(1 + np.floor(length_to_cover_with_hops / hop_size));

    # make our window function
    if (window_type == 'hann'):
        window = sp.signal.hann(window_size, sym=False)
    else:
        window = np.ones(window_size)  # rectangular window

    stft = [0] * num_hops
    # fill the array with values
    for hop in range(num_hops):
        start = hop * hop_size
        end = start + window_size
        unwindowed_sound = signal[start:end]
        windowed_sound = unwindowed_sound * window
        stft[hop] = fft(windowed_sound, window_size)
    X = np.array(stft).T
    #X = np.abs(X)
    Nf, Nt = np.shape(X)
    X = X[0:int(Nf / 2) + 1]
    return X

def matrix_gen(signal, sr, window_size, hop_size):
    # return abs_stft(signal, window_size, hop_size)
    #return librosa.feature.melspectrogram(signal, sr, n_mels=256)  # , n_fft=window_size, hop_length=hop_size)
    # return librosa.feature.melspectrogram(signal, sr, n_mels=256)#, n_fft=window_size, hop_length=hop_size)
    S, _ = librosa.core.spectrum._spectrogram(y=signal, S=None, n_fft=window_size, hop_length=hop_size,
                                              power=1)
    return S
    # return librosa.feature.mfcc(signal, sr)
def __dbs(y, window_size, hop_size, minDB=None):
    y_mono = librosa.core.to_mono(y)

    # Compute the MSE for the signal
    mse = librosa.feature.rmse(y=y_mono,
                      frame_length=window_size,
                      hop_length=hop_size) ** 2

    dbs = librosa.core.power_to_db(mse.squeeze(),ref=np.max, top_db=None)
    dbs = -dbs
    if (minDB):
        filteredDbs = dbs[dbs>minDB]
    else:
        filteredDbs = dbs
    return filteredDbs, dbs

def median_db(y, window_size, hop_size, minDB=None):
    filteredDbs, dbs = __dbs(y, window_size, hop_size, minDB)
    return np.median(filteredDbs) if filteredDbs.size > 0 else np.median(dbs)

def min_db(y, window_size, hop_size, minDB=None):
    filteredDbs, dbs = __dbs(y, window_size, hop_size, minDB)
    return np.min(filteredDbs) if filteredDbs.size > 0 else np.min(dbs)

def wavwrite(filepath, data, sr, norm=True, dtype='int16', ):
    '''
    Write wave file using scipy.io.wavefile.write, converting from a float (-1.0 : 1.0) numpy array to an integer array

    Parameters
    ----------
    filepath : str
        The path of the output .wav file
    data : np.array
        The float-type audio array
    sr : int
        The sampling rate
    norm : bool
        If True, normalize the audio to -1.0 to 1.0 before converting integer
    dtype : str
        The output type. Typically leave this at the default of 'int16'.
    '''
    if norm:
        data /= np.max(np.abs(data))
    data = data * np.iinfo(dtype).max
    data = data.astype(dtype)
    sp.io.wavfile.write(filepath, sr, data)


def plot_audio(x, sr, figsize=(16, 4)):
    """
    A simple audio plotting function

    Parameters
    ----------
    x: np.ndarray
        Audio signal to plot
    sr: int
        Sample rate
    figsize: tuple
        A duple representing the figure size (xdim,ydim)
    """
    length = x.shape[0]
    t = np.arange(0, float(length) / sr, 1 / sr)
    plt.figure(figsize=figsize)
    plt.plot(t, x)
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.show()
