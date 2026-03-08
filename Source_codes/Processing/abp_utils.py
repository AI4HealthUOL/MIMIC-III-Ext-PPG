# abp_utils.py
import pandas as pd
import numpy as np
from scipy.signal import lfilter



"Calculates an ABP signal quality index based on physiological plausibility and beat-to-beat consistency."
"for more information please check https://physionet.org/content/cardiac-output/1.0.0/code/2analyze/jSQI.m"

def listen_abp_sqi(bp_feats):
    import numpy as np

    rangeP, rangeMAP, rangeHR, rangePP = [20, 300], [30, 200], [20, 200], [20, np.inf]
    dPsys, dPdias, dPeriod = 20, 20, 0.5 * bp_feats['sampling_frequency']
    noise_threshold = -3

    # Ensure MAP and HR are scalars
    MAP = bp_feats.get('MAP', np.nan)
    if isinstance(MAP, (np.ndarray, list)):
        MAP = float(np.mean(MAP))

    HR = bp_feats.get('HR', np.nan)
    if isinstance(HR, (np.ndarray, list)):
        HR = float(np.mean(HR))

    # Beat-wise arrays
    P_dias = np.atleast_1d(bp_feats.get('P_dias', []))
    P_sys = np.atleast_1d(bp_feats.get('P_sys', []))
    PP = np.atleast_1d(bp_feats.get('PP', []))
    BeatPeriod = np.atleast_1d(bp_feats.get('BeatPeriod', []))
    mean_dyneg = bp_feats.get('mean_dyneg', np.nan)

    n_beats = P_sys.size
    bq = np.zeros((n_beats, 9)) if n_beats > 0 else np.zeros((1, 9))

    try:
        # Rule 1: Systolic and Diastolic plausible ranges
        badP = np.union1d(np.where(P_dias < rangeP[0])[0], np.where(P_sys > rangeP[1])[0])
        if badP.size > 0 and bq.shape[0] > 0 and np.max(badP) < bq.shape[0]:
            bq[badP, 1] = 1

        # Rule 2: MAP plausibility
        if (MAP < rangeMAP[0] or MAP > rangeMAP[1]) and not np.isnan(MAP):
            bq[0, 2] = 1

        # Rule 3: HR plausibility
        if (HR < rangeHR[0] or HR > rangeHR[1]) and not np.isnan(HR):
            bq[0, 3] = 1

        # Rule 4: Pulse Pressure plausibility
        badPP = np.where(PP < rangePP[0])[0]
        if badPP.size > 0 and np.max(badPP) < bq.shape[0]:
            bq[badPP, 4] = 1

        # Rule 5: Large jumps in systolic BP
        jerkPsys = 1 + np.where(np.abs(np.diff(P_sys)) > dPsys)[0] if P_sys.size > 1 else np.array([])
        if jerkPsys.size > 0 and np.max(jerkPsys) < bq.shape[0]:
            bq[jerkPsys, 5] = 1

        # Rule 6: Large jumps in diastolic BP
        jerkPdias = 1 + np.where(np.abs(np.diff(P_dias)) > dPdias)[0] if P_dias.size > 1 else np.array([])
        if jerkPdias.size > 0 and np.max(jerkPdias) < bq.shape[0]:
            bq[jerkPdias, 6] = 1

        # Rule 7: Large jumps in BeatPeriod
        jerkPeriod = 1 + np.where(np.abs(np.diff(BeatPeriod)) > dPeriod)[0] if BeatPeriod.size > 1 else np.array([])
        if jerkPeriod.size > 0 and np.max(jerkPeriod) < bq.shape[0]:
            bq[jerkPeriod, 7] = 1

        # Rule 8: High noise in down slopes
        if isinstance(mean_dyneg, np.ndarray):
            mean_dyneg_val = float(np.mean(mean_dyneg))
        else:
            mean_dyneg_val = mean_dyneg
        if mean_dyneg_val < noise_threshold and not np.isnan(mean_dyneg_val):
            bq[0, 8] = 1

        # Rule 0: Any rule violated
        bq[:, 0] = np.any(bq[:, 1:], axis=1)

        total_beats = bq.shape[0]
        good_beats = np.sum(bq[:, 0] == 0)
        sqi = int(good_beats == total_beats)
        fraction_good_beats = good_beats / total_beats

        
        #print(f"Total beats: {total_beats}")
        #print(f"Good beats: {good_beats}")
        #print("bq matrix:\n", bq)

        return sqi, round(fraction_good_beats, 2)

    except Exception as e:
        print(f"[ERROR in listen_abp_sqi] {e}")
        return -21, 0.0


"Detects beat onset indices in ABP using slope sum function (SSF) logic."
def detect_abp_beats(abp_signal, fs=125):
    abp_signal = np.asarray(abp_signal).flatten()
    offset, scale = 1600, 20
    Araw = abp_signal * scale - offset
    b = np.array([1, 0, 0, 0, 0, -2, 0, 0, 0, 0, 1], dtype=float)
    a = np.array([1, -2, 1], dtype=float)
    A_filt = lfilter(b, a, Araw) / 24.0 + 30
    if len(A_filt) <= 3:
        return np.array([]), abp_signal 
    A = (A_filt[3:] + offset) / scale
    
    dy = np.diff(A)
    dypos = np.where(dy < 0, 0, dy)
    ssf_conv = np.convolve(dypos, np.ones(16), mode='full')
    ssf = np.concatenate(([0, 0], ssf_conv))
    avg0, threshold0 = np.mean(ssf[:fs * 10]), 3 * np.mean(ssf[:fs * 10])
    lockout, timer, beat_indices = 0, 0, []
    for t in range(50, len(ssf) - 19):
        lockout = max(lockout - 1, 0)
        timer += 1
        if lockout < 1 and ssf[t] > avg0:
            timer = 0
            maxSSF = np.max(ssf[t:t + 20])
            minSSF = np.min(ssf[max(0, t - 19):t + 1])
            if maxSSF > (minSSF + 10):
                onset = 0.01 * maxSSF
                tt_start = max(0, t - 16)
                tt = np.arange(tt_start, t + 1)
                if len(tt) >= 2:
                    dssf = np.diff(ssf[tt])
                    indices = np.where(dssf < onset)[0]
                    if indices.size > 0:
                        last_idx = indices[-1]
                        beat_time = last_idx + t - 17
                        beat_indices.append(beat_time)
                        threshold0 += 0.1 * (maxSSF - threshold0)
                        avg0 = threshold0 / 3
                        lockout = 32
        if timer > 312:
            threshold0 -= 1
            avg0 = threshold0 / 3
    
    final_onset= np.array(beat_indices) - 2
    
    #if len(beat_indices) == 0:
        #print("⚠️ No beat onsets detected.")

   
    return final_onset, A 
    # A is the filterd ABP signal



"Extracts MAP, HR, pulse pressure, beat period, and slopes from the ABP signal."
def extract_abp_features(signal_data, onset_indices, fs=125, window=40):
    import numpy as np

    signal_data = np.asarray(signal_data).flatten()
    onset_indices = np.asarray(onset_indices).flatten()
    n_beats = len(onset_indices) - 1  # exclude last onset like in MATLAB

    if n_beats < 1:
        return {
            'P_sys': np.array([]),
            'P_dias': np.array([]),
            'PP': np.array([]),
            'MAP': np.array([]),
            'HR': np.nan,
            'BeatPeriod': np.array([]),
            'mean_dyneg': np.array([]),
            'sampling_frequency': fs
        }

    P_sys = np.zeros(n_beats)
    P_dias = np.zeros(n_beats)
    PP = np.zeros(n_beats)
    MAP = np.zeros(n_beats)
    BeatPeriod = np.zeros(n_beats)
    mean_dyneg = np.zeros(n_beats)

    for i in range(n_beats):
        onset = onset_indices[i]
        next_onset = onset_indices[i + 1]

        # ±40-sample windows for sys/dias
        sys_start = max(0, onset)
        sys_end = min(len(signal_data), onset + window)
        dias_start = max(0, onset - window)
        dias_end = max(1, onset)

        P_sys[i] = np.max(signal_data[sys_start:sys_end])
        P_dias[i] = np.min(signal_data[dias_start:dias_end])
        PP[i] = P_sys[i] - P_dias[i]

        # Beat period in samples
        BeatPeriod[i] = next_onset - onset

        # MAP between onsets
        MAP[i] = np.mean(signal_data[onset:next_onset])

        # Mean of negative slopes (dy < 0)
        dy = np.diff(signal_data[onset:next_onset])
        neg_slopes = dy[dy < 0]
        mean_dyneg[i] = np.mean(neg_slopes) if neg_slopes.size > 0 else 0

    # HR from median beat period
    HR = 60 / np.median(BeatPeriod / fs) if n_beats > 1 else np.nan

    return {
        'P_sys': P_sys,
        'P_dias': P_dias,
        'PP': PP,
        'MAP': MAP,
        'HR': HR,
        'BeatPeriod': BeatPeriod,
        'mean_dyneg': mean_dyneg,
        'sampling_frequency': fs
    }




"Iterates over detected beats, finds systolic and diastolic points within each beat window, and returns their indices and values."
def calculate_bp_from_abp(signal, fs):
    if isinstance(signal, pd.Series):
        signal = signal.values
    if not isinstance(signal, np.ndarray) or len(signal) == 0 or len(signal) < fs:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    beat_onsets = detect_abp_beats(signal, fs)[0]
    
    if beat_onsets.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    sbp_indices, dbp_indices = [], []

    for i, onset in enumerate(beat_onsets):
        window_end = beat_onsets[i+1] if i < len(beat_onsets) - 1 else min(len(signal), onset + int(0.5 * fs))
        seg = signal[onset:window_end]
        if len(seg) < 3:
            continue
        rel_max_idx = np.argmax(seg)
        sbp_index = min(onset + rel_max_idx, len(signal) - 1)
        min_separation = 3
        seg_dbp = signal[sbp_index:window_end] if (sbp_index + min_separation < window_end) else signal[onset:window_end]
        if len(seg_dbp) < 2:
            continue
        rel_min_idx = np.argmin(seg_dbp)
        dbp_index = min((sbp_index + rel_min_idx) if (sbp_index + rel_min_idx) > sbp_index else (onset + rel_min_idx), len(signal) - 1)
        if dbp_index <= sbp_index:
            continue
        sbp_indices.append(sbp_index)
        dbp_indices.append(dbp_index)

    if not sbp_indices or not dbp_indices:
        return np.array([]), np.array([]), np.array([]), np.array([])

    sbp_indices = np.array(sbp_indices)
    dbp_indices = np.array(dbp_indices)
    sbp_values = signal[sbp_indices]
    dbp_values = signal[dbp_indices]
    return sbp_values, dbp_values, sbp_indices, dbp_indices
