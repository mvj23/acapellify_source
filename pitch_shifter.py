from importstatements import *
        
def normalize(bases, sr, WINDOW_SIZE, HOP_SIZE):
    rms_bases = []
    for basis in bases:
        rms_bases.append(np.mean(librosa.feature.rmse(y=basis, frame_length=WINDOW_SIZE, hop_length=HOP_SIZE)))
    print(rms_bases)
    mean_rms = np.mean(rms_bases)
    print(mean_rms)

    for i, basis in enumerate(bases):
        ratio = mean_rms/rms_bases[i]
        bases[i] = basis * ratio
    
    return bases

def tune(audio, sr):
    """
    Autotunes an ndarray representing an audio file to the closest half step

    Input Parameters
    ----------------
    audio: The 1d numpy array containing the time series signal
    sr: the sample rate of the audio signal

    Returns
    -------
    tuned_audio: the 1d numpy array comtaining the autotuned audio file
    """
    #determines how far off from a half note the audio file is in fractions of a bin
    tuning_deviation = -librosa.core.estimate_tuning(audio, sr=sr, S=None, n_fft=2048, resolution=0.01, bins_per_octave=12)

    tuned_audio = librosa.effects.pitch_shift(audio, sr, tuning_deviation, bins_per_octave=12)


    return tuned_audio





def pitch_shifter(audio, sr, lower_bound, upper_bound):
    """
    Takes a ndarray array representing an audio file, autotunes it to the closest note, pitch shifts it to various frequencies,
    and returns an ndarray of all the pitch shifted sounds
    
    Input Parameters
    ----------------
    audio: The 1d numpy array containing the time series signal
    sr: the sample rate of the audio signal
    half_steps: integer, the number of half steps up and down we want to create
                
    Returns
    -------
    pitchShifted: an list containing all the pitch shifted audio ndarrays
    """

    pitch_shifted = []

    tuned_audio = tune(audio, sr)
    
    #cycles through all the half steps, constructing the pitch shifted sounds
    for n_steps in range(lower_bound, upper_bound + 1):
        pitch_shifted.append(librosa.effects.pitch_shift(tuned_audio, sr, n_steps))

    return pitch_shifted

if __name__== "__main__":
    freq435 = librosa.core.tone(435, sr=22050, duration=5)

    pitch_list = pitch_shifter(freq435, 22050, -10, 10)
    sum_pitches = sum(pitch_list)
    fft_tuned = sp.fftpack.fft(sum_pitches)

    sr = 22050
    frequency_range = np.linspace(0,sr,fft_tuned.shape[0])
    plt.figure(figsize = (16,4))
    plt.plot(frequency_range,abs(fft_tuned))

    plt.title('Frequency Spectrum of Balloon (stage)')
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.show()
