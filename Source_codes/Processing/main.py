# === main.py ===
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk

from utils import pad_vector, load_wfdb_signal, round_all_numeric, validate_window
from sqi_utils import listen_sqi2
from abp_utils import calculate_bp_from_abp
from plotting_utils import plot_abp_signal, plot_ecg_only, plot_resp_signal, plot_ppg_only

def process_10s_window(sub_window, fs, file_name=None, segment_idx=None, plot_flag=False):
    results = {
        "sbp": np.nan,
        "dbp": np.nan,
        "iqr_sbp": np.nan,
        "iqr_dbp": np.nan,
        "pleth_sqi": np.nan,
        "abp_sqi": np.nan,
        "ecg_sqi": np.nan,
        "hr": np.nan,
        "mean_peak_deriv_pleth": np.nan,
        "mean_peak_deriv_ecg":np.nan,
        "median_abp": np.nan,
        "median_pleth": np.nan,
        "median_ecg": np.nan,
    }

    if "ABP" in sub_window.columns:
        abp = sub_window["ABP"].values
        abp_result = listen_sqi2(abp, "ABP", fs)
        results["abp_sqi"] = abp_result.get("sqi")
        if results["abp_sqi"] > -1:
            results["median_abp"] = np.median(abp)
            sbp_vals, dbp_vals, sbp_idx, dbp_idx = calculate_bp_from_abp(abp, fs)
            if sbp_vals.size > 0:
                results["sbp"] = np.median(sbp_vals)
                results["iqr_sbp"] = np.percentile(sbp_vals, 75) - np.percentile(sbp_vals, 25)
            if dbp_vals.size > 0:
                results["dbp"] = np.median(dbp_vals)
                results["iqr_dbp"] = np.percentile(dbp_vals, 75) - np.percentile(dbp_vals, 25)
            if plot_flag and sbp_idx.size > 0 and dbp_idx.size > 0:
                filtered_abp = abp_result.get("filtered_signal")
                plot_abp_signal(filtered_abp, fs, sbp_idx, dbp_idx, title=f"ABP: {file_name}_segment{segment_idx}")

    if "PLETH" in sub_window.columns:
        
        pleth = sub_window["PLETH"].values
        
        ppg_result= listen_sqi2(pleth, "PLETH", fs)
        results["pleth_sqi"] = ppg_result.get("sqi")
        mean_peak_derivatives_ppg=ppg_result.get("mean_peak_derivatives")
        #print("mean_peak_derivatives_ppg",mean_peak_derivatives_ppg)
        if isinstance(mean_peak_derivatives_ppg, (int, float)) and not np.isnan(mean_peak_derivatives_ppg):
            results["mean_peak_deriv_pleth"] = mean_peak_derivatives_ppg
            
        
        if results["pleth_sqi"] > -1:
            results["median_pleth"] = np.median(pleth)
            if plot_flag:
                peaks = ppg_result.get("peaks")
                ppg_clean = ppg_result.get("ppg_filtered")
                plot_ppg_only(ppg_clean, fs, peaks, title=f"PPG: {file_name}_segment{segment_idx}")
        
        
    if "II" in sub_window.columns:
        ecg = sub_window["II"].values
        ecg_result= listen_sqi2(ecg, "II", fs)
        results["ecg_sqi"] = ecg_result.get("sqi")
        mean_peak_derivatives_ecg=ecg_result.get("mean_rpeak_derivatives")
        #print("mean_peak_derivatives_ecg",mean_peak_derivatives_ecg)
        if isinstance(mean_peak_derivatives_ecg, (int, float)) and not np.isnan(mean_peak_derivatives_ecg):
            results["mean_peak_deriv_ecg"] = mean_peak_derivatives_ecg
        
        if results["ecg_sqi"] > -1:
            results["median_ecg"] = np.median(ecg)
            rpeaks = ecg_result.get("rpeaks")
            rr = np.diff(rpeaks) / fs
            results["hr"] = round(60 / np.median(rr), 2)
            if plot_flag:
                ecg_clean=ecg_result.get("ecg_filtered")
                plot_ecg_only(ecg_clean, fs, rpeaks, title=f"ECG: {file_name}_segment{segment_idx}")


    return results

def process_30s_window(signal_df, fs, file_name=None, plot_flag=False):
    vectors = {
        "vector_10s_sbp": [], "vector_10s_dbp": [],
        "vector_10s_iqr_sbp": [], "vector_10s_iqr_dbp": [],
        "vector_10s_pleth_sqi": [], "vector_10s_abp_sqi": [],
        "vector_10s_ecg_sqi": [], "vector_10s_hr": [],
        "vector_10s_mean_peak_deriv_pleth": [],  "vector_10s_mean_peak_deriv_ecg": [],
        "vector_10s_median_abp": [],
        "vector_10s_median_pleth": [],
        "vector_10s_median_ecg": []
    }

    for i in range(3):
        sub_window = signal_df.iloc[i * 10 * fs : (i + 1) * 10 * fs]
        if sub_window.empty:
            for key in vectors: vectors[key].append(np.nan)
            continue
        results = process_10s_window(sub_window, fs, file_name, i, plot_flag)
        for key in vectors:
            result_key = key.replace("vector_10s_", "")
            vectors[key].append(results.get(result_key, np.nan))


    resp_sqi, rr_median, rr_iqr, median_resp,f_peak = np.nan, np.nan, np.nan, np.nan, np.nan
    if "RESP" in signal_df.columns:
        resp = signal_df["RESP"].values
        resp_result = listen_sqi2(resp, "RESP", fs)
        resp_sqi = resp_result.get("sqi")
        
        if resp_sqi > -1:
            median_resp = np.median(resp)
            rr_median = resp_result.get("rr_median", np.nan)
            rr_iqr = resp_result.get("rr_iqr", np.nan)
            f_peak= resp_result.get("f_peak")
            #print("rr_median",rr_median)
            if plot_flag:
                interp = resp_result.get("interp_signal", [])
                t_interp= resp_result.get("t_interp", [])
                peaks = resp_result.get("peaks", [])
                troughs = resp_result.get("troughs", [])
                plot_resp_signal(interp, t_interp, peaks=peaks, troughs=troughs, title=f"RESP: {file_name}")

    vectors.update({
        "resp_sqi": resp_sqi,
        "median_30s_rr": rr_median,
        "iqr_30s_rr": rr_iqr,
        "f_peak":f_peak,
        "median_resp": median_resp
    })
    return vectors

def process_chunk_of_30s_segments(meta_data_path, fs, start_index, end_index, plot_flag=False, wfdb_folder="WFDB_folder", save_ppg_segments=False, ppg_output_folder="PPG_segments"):
    results = []
    df_meta = pd.read_pickle(meta_data_path)
    df_chunk = df_meta.iloc[start_index:end_index].copy()

    for _, row in tqdm(df_chunk.iterrows(), total=len(df_chunk), desc="Processing Segments"):
        file_name = row["signal_file_name"]
       
        wfdb_data, wfdb_metadata = load_wfdb_signal(os.path.join(wfdb_folder, file_name))
        if wfdb_data is None:
            continue

        channels = wfdb_metadata.get("sig_name", [])
        if wfdb_data.shape[1] != len(channels):
            continue

        signal_df = pd.DataFrame({ch: wfdb_data[:, i] for i, ch in enumerate(channels)})

        if "PLETH" not in signal_df.columns:
            print(f"Skipping: PLETH not found in {file_name}")
            continue

        if save_ppg_segments:
            os.makedirs(ppg_output_folder, exist_ok=True)
            np.save(os.path.join(ppg_output_folder, f"{file_name}.npy"), signal_df["PLETH"].values)

        seg_result = {"signal_name": file_name, "event_rhythm": row.get("event_rhythm", np.nan)}
            
        seg_result.update(process_30s_window(signal_df, fs, file_name, plot_flag))
       
        results.append(seg_result)
       
         
    return pd.DataFrame(results)

if __name__ == "__main__":
    meta_data_path = "final_pkl_files_new_4/final_memmap_files/merged_WFDB_with_metadata_4.pkl"
    fs = 125
    # we have around 7000000 cases
    start_index = 0
    end_index = 500000
    plot_flag = False
    wfdb_folder = "WFDB_files_with_utils_5_15min_30s_segmented_no_Flat_PPG_new_4"
    save_ppg_segments = True
    ppg_output_folder = "npy_PPG_segments_30s_15min_no_Flat_interpolated_NaN_new_4"

    results_df = process_chunk_of_30s_segments(meta_data_path, fs, start_index, end_index, plot_flag, wfdb_folder, save_ppg_segments, ppg_output_folder)
    results_df = round_all_numeric(results_df, decimals=2)
    results_df.to_pickle(f"df_{start_index}_{end_index}_4.pkl")
    print(f"✅ Saved results for rows {start_index} to {end_index}.")
