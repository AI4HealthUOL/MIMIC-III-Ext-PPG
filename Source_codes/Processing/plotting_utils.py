# plotting_utils.py

import numpy as np
import matplotlib.pyplot as plt


def plot_abp_signal(abp_signal, fs, sbp_indices=None, dbp_indices=None, title="ABP Signal"):


    t = np.arange(len(abp_signal)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, abp_signal, label="ABP Signal", lw=2)

    if sbp_indices is not None and len(sbp_indices) > 0:
        sbp_indices = np.array(sbp_indices)
        valid_sbp = sbp_indices[sbp_indices < len(abp_signal)]
        plt.plot(valid_sbp / fs, abp_signal[valid_sbp], 'rv', markersize=8, label="SBP")

    if dbp_indices is not None and len(dbp_indices) > 0:
        dbp_indices = np.array(dbp_indices)
        valid_dbp = dbp_indices[dbp_indices < len(abp_signal)]
        plt.plot(valid_dbp / fs, abp_signal[valid_dbp], 'g^', markersize=8, label="DBP")

    plt.xlabel("Time (s)")
    plt.ylabel("Blood Pressure (mmHg)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_ecg_only(signal, fs, rpeaks, title="ECG"):
    """
    Plots the ECG signal with R-peaks.

    Parameters:
    - signal : np.ndarray
        ECG signal (1D array)
    - fs : int
        Sampling frequency
    - rpeaks : list or np.ndarray
        Indices of R-peaks
    - title : str
        Title of the plot
    """
    signal = np.asarray(signal)
    rpeaks = np.asarray(rpeaks)
    time = np.arange(len(signal)) / fs

    plt.figure(figsize=(15, 4))
    plt.plot(time, signal, label="ECG", color="blue", linewidth=1)
    
    if rpeaks.size > 0:
        rpeaks = rpeaks[rpeaks < len(signal)]  # safeguard
        plt.plot(time[rpeaks], signal[rpeaks], 'v', color="red", label="R-peaks", markersize=8)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"{title}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ppg_only(signal, fs, peaks, rpeaks=None, title="PPG"):
    """
    Plots the PPG signal with peaks and optionally ECG-style R-peaks.

    Parameters:
    - signal : np.ndarray
        PPG signal (1D array)
    - fs : int
        Sampling frequency
    - peaks : list or np.ndarray
        Indices of PPG peaks
    - rpeaks : list or np.ndarray (optional)
        ECG-style R-peaks from PPG if mislabeling is suspected
    - title : str
        Title of the plot
    """
    signal = np.asarray(signal)
    peaks = np.asarray(peaks)
    time = np.arange(len(signal)) / fs

    plt.figure(figsize=(15, 4))
    plt.plot(time, signal, label="PPG", color="blue", linewidth=1)

    if peaks.size > 0:
        peaks = peaks[peaks < len(signal)]
        plt.plot(time[peaks], signal[peaks], 'v', color="green", label="PPG Peaks", markersize=8)

    if rpeaks is not None and len(rpeaks) > 0:
        rpeaks = np.asarray(rpeaks)
        rpeaks = rpeaks[rpeaks < len(signal)]
        plt.plot(time[rpeaks], signal[rpeaks], 'x', color="red", label="ECG-style R-peaks", markersize=8)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_resp_signal(resp_signal, t_interp, peaks=None, troughs=None, title="RESP Signal"):
    """
    Plots the interpolated RESP signal with physiological peak/trough markers.

    Parameters:
    - resp_signal: Interpolated RESP signal (e.g., at 5 Hz)
    - t_interp: Time vector (must match length of resp_signal)
    - peaks: Indices of physiological inspiration peaks
    - troughs: Indices of physiological expiration troughs
    - title: Title for the plot
    """
    assert len(resp_signal) == len(t_interp), "Length mismatch between signal and time"

    plt.figure(figsize=(12, 4))
    plt.plot(t_interp, resp_signal, label="Normalized RESP (5 Hz)", lw=2)

    if peaks is not None and len(peaks) > 0:
        plt.plot(t_interp[peaks], resp_signal[peaks], 'rv', markersize=8, label="Inspiration Peaks")

    if troughs is not None and len(troughs) > 0:
        plt.plot(t_interp[troughs], resp_signal[troughs], 'g^', markersize=8, label="Expiration Troughs")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


