import helperFuncs as utils
from importstatements import *
import pitch_shifter as shifter


class ScaleBasesCalc:
    SILENCE_TIME_AT_END_VAL = .5

    def __init__(self, basis_audio, basis_SR, window_size, hop_size, graph_it=False):
        self.basis_audio = basis_audio
        self.basis_SR = basis_SR
        self.window_size = window_size
        self.hop_size = hop_size
        self.graph_it = graph_it


    @staticmethod
    def get_percussive_basis(basis_audio, window_size, hop_size):
        return ScaleBasesCalc.__trimBase(basis_audio, window_size, hop_size)

    def get_scale_bases(self):
        bases = self.__split_into_bases_threshold()
        return self.__get_non_spectral_outliers(bases)


    @staticmethod
    def __trimBase(basis, window_size, hop_size):
        top_db = utils.median_db(basis, window_size, hop_size)
        returnVal, _ = librosa.effects.trim(basis, frame_length=window_size, hop_length=hop_size,
                                           top_db=top_db)
        return returnVal


    def __get_non_spectral_outliers(self, bases):
        vals = []
        newBases = []
        if self.graph_it:
            plt.figure()
        for base in bases:
            spec = np.mean(librosa.feature.spectral_centroid(y=base, sr=self.basis_SR, n_fft=self.window_size,
                                                             hop_length=self.hop_size))
            vals.append(spec)
        lower_bound, upper_bound = ScaleBasesCalc.__getIQR(vals)
        for index, spec in enumerate(vals):
            if spec < lower_bound or spec > upper_bound:
                if self.graph_it:
                    plt.plot(index, spec, 'ro')
            else:
                newBases.append(bases[index])
                if self.graph_it:
                    plt.plot(index, spec, 'bo')

        plt.xlabel('basis index')
        plt.ylabel('mean spectral centroid')
        plt.title('Using IQR to identify spectral centroid outliers')
        plt.legend(('outlier', 'acceptable value'))
        if self.graph_it:
            plt.show()
        return newBases

    @staticmethod
    def __getIQR(data):
        sortedVals = sorted(data)
        q1, q3 = np.percentile(sortedVals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return lower_bound, upper_bound

    def __split_into_bases_threshold(self):
        y = self.basis_audio
        sr = self.basis_SR
        graphit = self.graph_it
        idx = np.arange(len(y))
        myHop = 512
        # Calcualte the onset frames in the usual way
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=myHop)
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=myHop)
        onstm = librosa.frames_to_time(onset_frames, sr=sr, hop_length=myHop)

        # Calculate RMS energy per frame.  I shortened the frame length from the
        # default value in order to avoid ending up with too much smoothing
        rmse = librosa.feature.rmse(y=y, frame_length=512, hop_length=myHop)[0,]
        rmse = np.array(rmse)
        envtm = librosa.frames_to_time(np.arange(len(rmse)), sr=sr, hop_length=myHop)
        # Use final second of recording in order to estimate median noise level
        # and typical variation
        noiseidx = [envtm > envtm[-1] - ScaleBasesCalc.SILENCE_TIME_AT_END_VAL]  # THIS METHOD DEPENDS ON THERE BEING AT LEAST SILENCE_TIME_AT_END_VAL seconds of silence at the end, otherwise it doesn't work at all
        noiseidx = noiseidx[0]
        noisemedian = np.percentile(rmse[noiseidx], 50)
        sigma = np.percentile(rmse[noiseidx], 84.1) - noisemedian
        # Set the minimum RMS energy threshold that is needed in order to declare
        # an "onset" event to be equal to 5 sigma above the median
        threshold = noisemedian + 5 * sigma
        threshidx = [rmse > threshold]
        threshidx = threshidx[0]
        # Choose the corrected onset times as only those which meet the RMS energy
        # minimum threshold requirement
        thresholdEnvtm = envtm[threshidx]
        correctedonstm = []
        CHECK_REGION = 10
        stop_places_index_lookup = dict()
        stop_places = []
        onstm_index = 0
        stop_places_index = 0
        for i in range(len(threshidx)):
            lowerCheck = max(0, i - CHECK_REGION)
            upperCheck = min(len(threshidx), i + CHECK_REGION)
            if i > 0 and all(threshidx[i:upperCheck]) and not threshidx[i - 1]:
                for j in range(lowerCheck, upperCheck):
                    if j in onset_frames:
                        correctedonstm.append(i)
                        onstm_index += 1
                        break
            if i > 0 and not any(threshidx[i:upperCheck]) and threshidx[i - 1] and onstm_index >= 1:
                stop_places.append(i)
                stop_places_index_lookup[onstm_index-1] = stop_places_index
                stop_places_index += 1

        correctedonstm = np.array(correctedonstm)
        correctedonstm = librosa.frames_to_samples(correctedonstm, hop_length=myHop)  # , sr=sr, hop_length=myHop)
        stop_places = np.array(stop_places)
        stop_places = librosa.frames_to_samples(stop_places, hop_length=myHop)


        # correctedonstm = onstm[[tm in correctedonstm for tm in onstm]]

        bases = []
        sampleCorrected = correctedonstm
        end_samples = sampleCorrected[1:]
        end_samples = np.append(end_samples, y.size)
        intervals = list(zip(sampleCorrected, end_samples))
        for onstm_index, interval in enumerate(intervals):
            start = interval[0]
            end = interval[1]
            if onstm_index in stop_places_index_lookup:
                end = stop_places[stop_places_index_lookup[onstm_index]]
                if end < start:
                    end = interval[1]
            curr_basis = y[start:end]
            curr_basis = shifter.tune(curr_basis, sr)
            bases.append(curr_basis)

        if graphit:
            start = 0
            fg = plt.figure(figsize=[12, 8])

            # Print the waveform together with onset times superimposed in red
            ax1 = fg.add_subplot(2, 1, 1)
            ax1.plot(idx + start, y)
            for index, ii in enumerate(sampleCorrected):
                ax1.axvline(ii, color='g')
                if index in stop_places_index_lookup:
                    ax1.axvline(stop_places[stop_places_index_lookup[index]], color='r')

            # for ii in onset_samples:
            #     ax1.axvline(ii, color='g')
            ax1.set_ylabel('Amplitude', fontsize=16)

            # Print the RMSE together with onset times superimposed in red
            ax2 = fg.add_subplot(2, 1, 2, sharex=ax1)
            ax2.plot(envtm * sr + start, rmse)
            for index, ii in enumerate(sampleCorrected):
                ax2.axvline(ii, color='g')
                if index in stop_places_index_lookup:
                    ax2.axvline(stop_places[stop_places_index_lookup[index]], color='r')

            # Plot threshold value superimposed as a black dotted line
            ax2.axhline(threshold, linestyle=':', color='k')
            ax2.set_ylabel("RMS", fontsize=16)
            ax2.set_xlabel("Sample Number", fontsize=16)

            fg.show()
            plt.show()
        return bases
