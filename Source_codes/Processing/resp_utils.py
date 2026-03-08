# resp_utils.py




"""
=======================================================================================
RESPIRATORY SIGNAL QUALITY INDEX AND BREATHING RATE ESTIMATION MODULE

Inspired by:
Charlton, P. H., et al. (2021). "An impedance pneumography signal quality index..."
https://doi.org/10.1088/1361-6579/ac3b8c

Implements:
- listen_resp_sqi(): Evaluates PPG-derived or impedance respiratory signal quality
=======================================================================================
"""

from scipy.signal import butter, filtfilt, detrend
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
import numpy as np
import neurokit2 as nk

def listen_resp_sqi(resp_signal, fs):
    """
    Computes a Signal Quality Index (SQI) for respiratory signals using morphology and consistency.
    "Only morphologically consistent and physiologically plausible breaths are retained to compute the SQI"
    Returns:
    - sqi (int): 1 = good, 0 = poor, -1 = unusable
    - peaks (list)
    - troughs (list)
    - resp_signal_interp (np.ndarray): cleaned & normalized signal
    - t_interp (np.ndarray): interpolated time vector (at 5 Hz)
    """
    try:
        if isinstance(resp_signal, list):
            resp_signal = np.asarray(resp_signal)

        # Minimum 30-second signal
        min_length = int(fs * 30)
        if len(resp_signal) < min_length:
            print(f"❌ Signal too short: {len(resp_signal)} samples. Minimum {min_length} required for 30 seconds.")
            return -10, [], [], [], [], [],np.nan

        resp_signal = np.nan_to_num(resp_signal)
        t = np.arange(len(resp_signal)) / fs
        duration_of_signal = t[-1] - t[0]
        tukey_prop = min(1.0, 2 * 2 / duration_of_signal)

        # Apply Tukey window
        resp_signal -= np.mean(resp_signal)
        resp_signal *= tukey(len(resp_signal), tukey_prop)

        # IIR Butterworth low-pass filter
        b, a = butter(N=4, Wn=1.2 / (fs / 2), btype='low')
        resp_signal = filtfilt(b, a, resp_signal)
        


        # Interpolate to 5 Hz
        downsample_freq = 5
        num_interp_samples = int((t[-1] - t[0]) * downsample_freq) + 1
        t_interp = np.linspace(t[0], t[-1], num=num_interp_samples)
        interp_func = interp1d(t, resp_signal, kind='linear', fill_value="extrapolate")
        resp_signal_interp = interp_func(t_interp)
        resp_signal_interp = (resp_signal_interp - np.mean(resp_signal_interp)) / np.std(resp_signal_interp)
        
        
        
        # inversion, and Peak and trough detection
        v_n = -1* detrend(resp_signal_interp)
        #print("v_n",v_n)
        diffs = np.diff(v_n)
        # True peaks: v_n[i] > v_n[i-1] and v_n[i] > v_n[i+1]
        peaks = np.where((v_n[1:-1] > v_n[:-2]) & (v_n[1:-1] > v_n[2:]))[0] + 1

        # True troughs: v_n[i] < v_n[i-1] and v_n[i] < v_n[i+1]
        troughs = np.where((v_n[1:-1] < v_n[:-2]) & (v_n[1:-1] < v_n[2:]))[0] + 1
        
        #print("original_peaks",peaks)
        
        if len(peaks) == 0 or len(troughs) == 0:
            #print("❌ No peaks or troughs detected.")
            return -11, [], [], [], [], [], np.nan

        # Apply thresholding rule from the paper
        peak_amplitudes = v_n[peaks]
        trough_amplitudes = v_n[troughs]
        peak_thresh = 0.2 * np.percentile(peak_amplitudes, 75)
        trough_thresh = 0.2 * np.percentile(trough_amplitudes, 25)
        peaks = peaks[peak_amplitudes > peak_thresh]
        troughs = troughs[trough_amplitudes < trough_thresh]

        # Select valid peaks between trough pairs
        final_peaks = []
        for i in range(len(troughs) - 1):
            t1, t2 = troughs[i], troughs[i + 1]
            peaks_between = peaks[(peaks > t1) & (peaks < t2)]
            if len(peaks_between) > 0:
                max_peak_idx = peaks_between[np.argmax(v_n[peaks_between])]
                final_peaks.append(max_peak_idx)

        # Add initial peak if available
        initial_peaks = peaks[peaks < troughs[0]]
        if len(initial_peaks) > 0:
            max_initial = initial_peaks[np.argmax(v_n[initial_peaks])]
            final_peaks = [max_initial] + final_peaks

        # Validate peaks with surrounding troughs
        valid_peaks = []
        for i in range(1, len(final_peaks)):
            prev, curr = final_peaks[i - 1], final_peaks[i]
            if np.any((troughs > prev) & (troughs < curr)):
                valid_peaks.append(prev)
        if len(final_peaks) > 0:
            valid_peaks.append(final_peaks[-1])
        valid_peaks = np.asarray(valid_peaks)

        if len(valid_peaks) < 2:
            #print("❌ Not enough valid peaks for breath cycles.")
            return -12, [], [], [], [], [], np.nan

        # Calculate breath cycle metrics
        cycle_durations = np.diff(t_interp[valid_peaks])
        median_dur = np.median(cycle_durations)
        bad_cycles = (cycle_durations > 1.5 * median_dur) | (cycle_durations < 0.5 * median_dur)
        prop_bad_breaths = 100 * np.sum(bad_cycles) / len(bad_cycles)
        norm_duration = np.sum(cycle_durations[~bad_cycles])
        win_length = t_interp[-1] - t_interp[0]
        prop_norm_dur = 100 * norm_duration / win_length
        R2min = np.std(cycle_durations) / np.mean(cycle_durations)

        # Template matching
        mean_rr_samples = int(np.floor(np.mean(np.diff(valid_peaks))))
        

        segments = []
        for idx in valid_peaks:
            
            half = mean_rr_samples // 2
            if mean_rr_samples % 2 == 0:
                start = idx - half
                end = idx + half  # excludes end → end - start = mean_rr_samples
            else:
                start = idx - half
                end = idx + half + 1  # extra 1 to include center → end - start = mean_rr_samples
            
            if start >= 0 and end < len(v_n):
                seg = v_n[start:end]
                if len(seg) == mean_rr_samples and np.linalg.norm(seg) > 0:
                    segments.append(seg / np.linalg.norm(seg))
       
        if len(segments) < 2:
            #print("❌ Not enough segments for RESP template matching.")
            return -13, [], [], [], [], [], np.nan

        segments = np.array(segments)
        template = np.mean(segments, axis=0)
        correlations = [np.corrcoef(template, seg)[0, 1] for seg in segments]
        R2 = np.mean(np.abs(correlations))

        sqi = 1 if (prop_norm_dur > 60 and prop_bad_breaths < 15 and R2 >= 0.75 and R2min < 0.25) else 0

        #print(f"✅ SQI computed: SQI={sqi}, R2={R2:.2f}, prop_norm_dur={prop_norm_dur:.1f}%, prop_bad_breaths={prop_bad_breaths:.1f}%, R2min={R2min:.2f}")
        
        return sqi, peaks.tolist(), troughs.tolist(), cycle_durations.tolist(),  resp_signal_interp, t_interp

    except Exception as e:
        print(f"❌ Error in listen_resp_sqi: {e}")
        return -21, [], [], [], [], [], np.nan
