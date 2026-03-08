# utils.py (merged validation and IO functions)

import os
import numpy as np
import pandas as pd
import wfdb

"Checks signal duration, flatline sections, or extreme values."
def validate_window(signal_data, fs, signal_type="generic"):
    """
    Validates a window of physiological signal data.

    Parameters:
    signal_data : array-like
        Input signal data (e.g., ECG, PPG, RESP) for a window.
    fs : int
        Sampling frequency in Hz.
    signal_type : str
        Type of signal. If "RESP", uses a custom flatline rule.

    Returns:
    int
        -4 : signal too short (<10s)
        -3 : signal is None or contains NaNs
        -2 : flatline or extreme repeat detected
         4 : valid signal
    """

    if signal_data is None or len(signal_data) < 10 * fs:
        return -4  # Signal too short

    if np.isnan(signal_data).any():
        return -3  # Invalid signal: NaN present

    if signal_type.upper() == "RESP":
        # Custom flatline detection: if >5s of constant value
        count = 1
        for i in range(1, len(signal_data)):
            if signal_data[i] == signal_data[i - 1]:
                count += 1
                if count > 5 * fs:
                    return -2  # Flatline detected in RESP
            else:
                count = 1
        return 4  # Valid RESP

    # General case (non-RESP): flatline + extreme repeat check
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    consecutive_min = (signal_data == min_val).astype(int)
    consecutive_max = (signal_data == max_val).astype(int)

    if (np.max(np.convolve(consecutive_min, np.ones(4, dtype=int), mode='valid')) >= 4 or
        np.max(np.convolve(consecutive_max, np.ones(4, dtype=int), mode='valid')) >= 4):
        return -2  # Repeated extreme values

    count = 1
    for i in range(1, len(signal_data)):
        if signal_data[i] == signal_data[i - 1]:
            count += 1
            if count > fs:
                return -2  # Flatline (>1s) for non-RESP
        else:
            count = 1

    return 4  # Signal passed all checks



"Interpolates and optionally clips the signal."
def fix_nans_and_clip(signal, clip_amp=3):
    tmp = pd.DataFrame(signal).interpolate(limit_direction='both').values.ravel()
    tmp = pd.DataFrame(tmp).bfill().ffill().values.ravel()
    signal = np.clip(tmp, -clip_amp, clip_amp) if clip_amp > 0 else tmp
    return signal

"Returns NaN percentages per channel."
def calculate_nan_percentages(window_signals, channels_to_check):
    nan_percentages = {}
    for alias, channel in channels_to_check.items():
        if channel in window_signals.columns:
            nan_count = window_signals[channel].isna().sum()
            total_count = len(window_signals)
            nan_percentages[f"{alias}_NaN"] = round((nan_count / total_count) * 100, 2)
        else:
            nan_percentages[f"{alias}_NaN"] = np.nan
    return nan_percentages


def pad_vector(vec, target_length=3):
    return vec + [np.nan] * (target_length - len(vec)) if len(vec) < target_length else vec


def load_wfdb_signal(signal_file):
    dat_file, hea_file = f"{signal_file}.dat", f"{signal_file}.hea"
    if not (os.path.exists(dat_file) and os.path.exists(hea_file)):
        print(f"File not found: {dat_file} or {hea_file}")
        return None, None
    try:
        record = wfdb.rdsamp(signal_file)
        signals, metadata = record
        return signals, metadata
    except Exception as e:
        print(f"Error loading WFDB file {signal_file}: {e}")
        return None, None



def round_all_numeric(df, decimals=2):
    def round_if_number(val):
        if isinstance(val, list):
            return [round(v, decimals) if isinstance(v, (int, float)) and not pd.isna(v) else v for v in val]
        elif isinstance(val, (int, float)) and not pd.isna(val):
            return round(val, decimals)
        return val

    return df.applymap(round_if_number)

