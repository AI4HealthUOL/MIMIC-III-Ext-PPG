# sqi_utils.py

import numpy as np
import neurokit2 as nk
from scipy.signal import kaiserord, firwin, filtfilt, butter
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from abp_utils import detect_abp_beats, extract_abp_features, listen_abp_sqi
from resp_utils import listen_resp_sqi
from utils import validate_window
import matplotlib.pyplot as plt




"Dispatcher function for evaluating signal quality of ABP, PPG, ECG, or RESP."
"Calls specific extractors or functions like listen_abp_sqi, listen_resp_sqi, etc."
"The algorithms are described in Peter Charlton's Ph.D. thesis (https://theses.eurasip.org/theses/829/continuous-respiratory-rate-monitoring-to-detect/read/) "
"Also please check Orphanidou, C., et al. (2014). Signal-Quality Indices for ECG and PPG..."



def listen_sqi2(window_signals, signal_type, fs):

    
    

    signal_type = signal_type.upper()
    
    
    
    # Validate the signal with signal type awareness (e.g., custom RESP rule)
    validity_code = validate_window(window_signals, fs, signal_type=signal_type)
    if validity_code != 4:
        
        return {
            "signal_type": signal_type,
            "sqi": validity_code
        }

    try:
        # === ABP
        if signal_type == "ABP":

            onesets,abp_filtered  = detect_abp_beats(window_signals, fs)

            if onesets.size < 2 or np.any(np.isnan(onesets)):
                return {"signal_type": signal_type, "sqi": -19,"filtered_signal": abp_filtered}

            bp_feats = extract_abp_features(window_signals, onesets, fs)
            
            sqi, frac_good = listen_abp_sqi(bp_feats)
            return {
                "signal_type": signal_type,
                "sqi": int(sqi),
                "filtered_signal": abp_filtered
            }

        # === RESP
        elif signal_type == "RESP":
            
            sqi, inv_peaks, inv_troughs, cycle_durations, signal_interp,t_interp  = listen_resp_sqi(window_signals, fs)
                       
            rr_median, rr_iqr , f_peak = np.nan, np.nan, np.nan
            
            if sqi> -1:
                durations = cycle_durations
                rr_median = round(60 / np.median(durations), 2)
                q25 = np.percentile(durations, 25)
                q75 = np.percentile(durations, 75)
                rr_iqr = round((60 / q25) - (60 / q75), 2)
                if len(durations) > 0:
                    f_peak = 1 / np.median(durations)  # In Hz

            
            # Swap them just for better plot sicne we use the inverse inresp_utils
            peaks = inv_troughs
            troughs = inv_peaks
            #print("peaks", peaks)
            #print("troughs", troughs)
            return {
                "signal_type": signal_type,
                "sqi": int(sqi),
                "peaks": peaks,
                "troughs": troughs,
                "interp_signal": signal_interp,
                "rr_median": rr_median,
                "rr_iqr": rr_iqr,
                "t_interp": t_interp,
                "f_peak_rr": f_peak
            }

        # === ECG
        elif signal_type in ["ECG", "II"]:
            
            ecg_filtered = nk.ecg_clean(window_signals, sampling_rate=fs, method="neurokit")

            
            
            
            rpeaks = nk.ecg_findpeaks(ecg_filtered, sampling_rate=fs)["ECG_R_Peaks"]
            
    
            
            if len(rpeaks) < 2 or np.any(np.isnan(rpeaks)):
                return {"signal_type": signal_type, "sqi": -17}


            
            
            deriv_stats_ecg = compute_derivative_stats(ecg_filtered, fs)
            ecg_derivative = deriv_stats_ecg["derivative"]
            rpeak_derivatives = ecg_derivative[rpeaks]
            
            #print("Derivative Stats_ECG:")
            #print("  Mean: ", deriv_stats_ecg["mean_derivative"])
            #print("  Std:  ", deriv_stats_ecg["std_derivative"])
            #print("  Max:  ", deriv_stats_ecg["max_derivative"])
            #print("Derivative at ECG R-peaks:")
            #print("  Mean: ", np.mean(rpeak_derivatives))
            #print("  Std:  ", np.std(rpeak_derivatives))
            #print("  Max: ", np.max(rpeak_derivatives))
            #print("  Values:", rpeak_derivatives[:5])
            
            
            #plt.figure(figsize=(6, 4))
            #plt.hist(deriv_stats_ecg["derivative"], bins=50, color='blue', alpha=0.7)
            #plt.title("ECG First Derivative Histogram")
            #plt.xlabel("Derivative Value")
            #plt.ylabel("Frequency")
            #plt.grid(True)
            #plt.tight_layout()
            #plt.show()
                        
            sqi, R2, corrs = sqi_calculator(ecg_filtered, np.array(rpeaks), fs, threshold=0.66)
            return {
                "signal_type": signal_type,
                "sqi": int(sqi),
                "R2": round(R2, 3),
                "n_beats": len(rpeaks),
                "correlations": corrs,
                "rpeaks": rpeaks,
                "ecg_filtered":ecg_filtered,
                "mean_rpeak_derivatives":np.mean(rpeak_derivatives)
                
            }

        # === PPG (PLETH)
        elif signal_type in ["PPG", "PLETH"]:
          
            #print("window_signals",window_signals)
           
            ppg_filtered = nk.ppg_clean(window_signals, sampling_rate=fs, method="elgendi")
            
            

            
            # just for finding the mislabeled signals
            ppg_filtered_ecg_rule= nk.ecg_clean(window_signals, sampling_rate=fs, method="neurokit")

            
          
            #print("ppg_filtered",ppg_filtered)
            try:
                peaks = nk.ppg_findpeaks(ppg_filtered, sampling_rate=fs)["PPG_Peaks"]
                # just for finding the mislabeled signals
                rpeak_ppg= nk.ecg_findpeaks(ppg_filtered_ecg_rule, sampling_rate=fs)["ECG_R_Peaks"]
                
                if peaks.size < 2 or np.any(np.isnan(peaks)):
                    raise ValueError("Invalid or insufficient peaks")
            except Exception:
                return {"signal_type": signal_type, "sqi": -18}
                
            
            deriv_stats_ppg = compute_derivative_stats(ppg_filtered, fs)
            ppg_derivative = deriv_stats_ppg["derivative"]
            peak_derivatives = ppg_derivative[peaks]
            # just for finding the mislabeled signals
            if len(rpeak_ppg) < 2 or np.any(np.isnan(rpeak_ppg)):
                rpeak_derivatives_ppg = np.array([])
            else:
                rpeak_derivatives_ppg = ppg_derivative[rpeak_ppg]

            
            #print("Derivative Stats_PPG:")
            #print("  Mean: ", deriv_stats_ppg["mean_derivative"])
            #print("  Std:  ", deriv_stats_ppg["std_derivative"])
            #print("  Max:  ", deriv_stats_ppg["max_derivative"])
            #print("Derivative at PPG Peaks:")
            #print("  Mean: ", np.mean(peak_derivatives))
            #print("  Std:  ", np.std(peak_derivatives))
            #print("  Max: ", np.max(peak_derivatives))
            #print("  Values:", peak_derivatives[:5])
            
            #plt.figure(figsize=(6, 4))
            #plt.hist(deriv_stats_ppg["derivative"], bins=50, color='green', alpha=0.7)
            #plt.title("PPG First Derivative Histogram")
            #plt.xlabel("Derivative Value")
            #plt.ylabel("Frequency")
            #plt.grid(True)
            #plt.tight_layout()
            #plt.show()

            sqi, R2, corrs = sqi_calculator(ppg_filtered, peaks, fs, threshold=0.86)
            return {
                "signal_type": signal_type,
                "sqi": int(sqi),
                "R2": round(R2, 3),
                "n_beats": len(peaks),
                "correlations": corrs,
                "ppg_filtered": ppg_filtered,
                "peaks": peaks,
                "rpeak_ppg": rpeak_ppg,
                "mean_peak_derivatives": np.mean(peak_derivatives),
                "mean_rpeak_derivatives_ppg": np.mean(rpeak_derivatives_ppg) if rpeak_derivatives_ppg.size > 0 else np.nan
            }


        return {"signal_type": signal_type, "sqi": -20}  # Unknown type

    except Exception:
        return {"signal_type": signal_type, "sqi": -21}





"Pelase see Fig1 of Orphanidou, C., et al. (2014). Signal-Quality Indices for ECG and PPG..."

def sqi_calculator(signal, beat_indices, fs, threshold=0.8):
    """
    Computes the Signal Quality Index (SQI) for ECG/PPG signals based on:
    1. Heart rate plausibility: Ensures mean HR is between 40 and 180 bpm.
    2. RR interval threshold: Rejects if any RR interval exceeds 3 seconds.
    3. RR interval regularity: Ensures max RR / min RR < 2.2.
    4. Morphological consistency: Correlation between beats and average template.

    Parameters:
    - signal (np.ndarray): 1D array of the ECG or PPG signal.
    - beat_indices (list or np.ndarray): Detected beat positions.
    - fs (int): Sampling frequency in Hz.
    - threshold (float): Minimum acceptable mean correlation (R²).

    Returns:
    - sqi (int): 1 if all checks pass (GOOD), else 0 (BAD).
    - R2 (float): Mean absolute correlation of beats vs. template.
    - correlations (list): Individual correlation values.
    """
    
    # Compute RR intervals in seconds
    rr_intervals = np.diff(beat_indices) / fs
    if len(rr_intervals) == 0:
        return -14, 0.0, []

    # Step 1: HR between 40–180 bpm
    mean_hr = 60 / np.mean(rr_intervals)
    if not (40 <= mean_hr <= 180):
        return 0, 0.0, []

    # Step 2: All RR intervals <= 3s
    if np.any(rr_intervals > 3.0):
        return 0, 0.0, []

    # Step 3: RR interval regularity
    if np.max(rr_intervals) / np.min(rr_intervals) > 2.2:
        return 0, 0.0, []

    # Step 4: Template correlation
    mean_rr_samples = int(np.floor(np.mean(np.diff(beat_indices))))
    #print(f"➡️ Total beats: {len(beat_indices)}")
    #print(f"➡️ mean_rr_samples: {mean_rr_samples}")
    if mean_rr_samples < 1:
        return -15, 0.0, []

    segments = []
    for idx in beat_indices:
        
        half = mean_rr_samples // 2
        if mean_rr_samples % 2 == 0:
            start = idx - half
            end = idx + half  # excludes end → end - start = mean_rr_samples
        else:
            start = idx - half
            end = idx + half + 1  # extra 1 to include center → end - start = mean_rr_samples
        if start >= 0 and end < len(signal):
           
            seg = signal[start:end]
            norm = np.linalg.norm(seg)
     
            #print("len(seg)",len(seg))    
            if norm != 0 and len(seg) == mean_rr_samples:
                segments.append(seg / norm)
    #print(f"✅ Valid segments: {len(segments)}")
    if len(segments) < 2:
        #print("❌ Not enough segments for PPG/ECG template matching.")
        return -16, 0.0, []

    segments = np.array(segments)
    template = np.mean(segments, axis=0)
    correlations = [np.corrcoef(template, seg)[0, 1] for seg in segments]
    R2 = np.mean(np.abs(correlations))

    sqi = int(R2 >= threshold)
    return sqi, R2, correlations


import numpy as np

def compute_derivative_stats(signal, fs):
    """
    Compute the first derivative of a signal and return its mean, standard deviation, and maximum value.
    
    Parameters:
    - signal: The input signal (1D array-like).
    - fs: Sampling frequency in Hz.
    
    Returns:
    - A dictionary containing mean, standard deviation, and maximum of the derivative.
    """
    derivative = np.diff(signal) * fs  # Multiply by fs to account for sampling rate
    return {
        'mean_derivative': np.mean(derivative),
        'std_derivative': np.std(derivative),
        'max_derivative': np.max(derivative),
        'derivative': derivative
    }

