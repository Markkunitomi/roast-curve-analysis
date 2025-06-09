#!/usr/bin/env python3
"""
Roast Curve Analysis - Consolidated Main Script

A complete Python script for analyzing coffee roast curves from Artisan .alog files.
This consolidated version contains all functionality in a single file for easy deployment.

Author: Coffee Analytics Team
Version: 0.1.0
"""

import argparse
import ast
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import uniform_filter1d


# =============================================================================
# ALOG PARSER MODULE
# =============================================================================

def extract_roast_data(alog_path: Union[str, Path], window: int = 30) -> Dict[str, Any]:
    """
    Extract roast variables from log file.
    
    Args:
        alog_path: Path to .alog file
        window: Window size for RoR calculation in seconds
        
    Returns:
        Dictionary containing time, temperatures, RoR, and event indices
        
    Raises:
        FileNotFoundError: If the alog file doesn't exist
        ValueError: If the alog file format is invalid
        KeyError: If required fields are missing from the alog file
    """
    alog_path = Path(alog_path)
    if not alog_path.exists():
        raise FileNotFoundError(f"ALOG file not found: {alog_path}")
    
    try:
        with open(alog_path, "r", encoding="utf-8") as f:
            data = ast.literal_eval(f.read())
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid ALOG file format: {e}") from e
    
    try:
        bt = data["temp2"]  # Bean temperature
        et = data["temp1"]  # Environment temperature
        comp = data["computed"]
        
        charge_bt = comp["CHARGE_BT"]
        charge_time = next(i for i, t in enumerate(bt) if t >= charge_bt)
        
        tp_time = comp["TP_time"] + charge_time + window
        fcs_time = comp["FCs_time"] + charge_time + window
        drop_time = comp["DROP_time"] + charge_time + window
        
        start = max(charge_time - window, 0)
        end = int(drop_time)
        
        bt_win = np.array(bt[start:end])
        et_win = np.array(et[start:end])
        time_win = np.arange(start, end) - charge_time
        
        # Calculate Rate of Rise (RoR)
        ror = np.zeros_like(bt_win)
        for i in range(window, len(bt_win)):
            ror[i] = (bt_win[i] - bt_win[i-window]) / window * 60
        
        # Smooth RoR
        ror_sm = uniform_filter1d(ror, size=15)
        
        return {
            "time": time_win,
            "bt": bt_win,
            "et": et_win,
            "ror": ror_sm,
            "indices": {
                "tp": int(tp_time - charge_time),
                "fcs": int(fcs_time - charge_time),
                "drop": int(drop_time - charge_time),
            },
            "charge_bt": charge_bt,
            "charge_time": charge_time
        }
        
    except KeyError as e:
        raise KeyError(f"Required field missing from ALOG file: {e}") from e


def load_roast_data_as_dataframe(folder: Union[str, Path], window: int = 30) -> pd.DataFrame:
    """
    Load .alog files into a DataFrame, skipping any files with missing
    required fields and reporting which files failed and what was missing.
    
    Args:
        folder: Path to folder containing .alog files
        window: Window size for RoR calculation
        
    Returns:
        DataFrame with roast data from all valid files
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    rows: List[Dict[str, Any]] = []
    failures: List[Tuple[str, List[str]]] = []
    
    required_data = ["timex", "temp1", "temp2"]
    required_events = [
        "CHARGE_ET", "CHARGE_BT",
        "TP_time", "TP_ET", "TP_BT",
        "DROP_time", "DROP_ET", "DROP_BT"
    ]
    
    for file_path in folder.glob("*.alog"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = ast.literal_eval(f.read())
            
            events = data.get("computed", {})
            missing: List[str] = []
            
            # Check required data fields
            for key in required_data:
                if key not in data:
                    missing.append(key)
            
            # Check required event fields
            for key in required_events:
                if key not in events:
                    missing.append(key)
            
            # Check for first crack events (either start or end)
            if not ("FCs_time" in events or "FCe_time" in events):
                missing.append("FCs_time/FCe_time")
            if not ("FCs_ET" in events or "FCe_ET" in events):
                missing.append("FCs_ET/FCe_ET")
            if not ("FCs_BT" in events or "FCe_BT" in events):
                missing.append("FCs_BT/FCe_BT")
            
            if missing:
                failures.append((file_path.name, missing))
                continue
            
            # Calculate RoR for the full data
            bt = data["temp2"]
            ror = np.zeros(len(bt))
            for i in range(window, len(bt)):
                ror[i] = (bt[i] - bt[i-window]) / window * 60
            bt_ror = uniform_filter1d(ror, size=15).tolist()
            
            # Create row data
            rows.append({
                "filename": file_path.stem,
                "charge_et": events["CHARGE_ET"],
                "charge_bt": events["CHARGE_BT"],
                "tp_time": events["TP_time"],
                "tp_et": events["TP_ET"],
                "tp_bt": events["TP_BT"],
                "fcs_time": events.get("FCs_time", events.get("FCe_time")),
                "fcs_et": events.get("FCs_ET", events.get("FCe_ET")),
                "fcs_bt": events.get("FCs_BT", events.get("FCe_BT")),
                "drop_time": events["DROP_time"],
                "drop_et": events["DROP_ET"],
                "drop_bt": events["DROP_BT"],
                "time": data["timex"],
                "bt": bt,
                "et": data["temp1"],
                "bt_ror": bt_ror,
            })
            
        except Exception as e:
            failures.append((file_path.name, [f"parsing error: {str(e)}"]))
    
    if failures:
        print("Failed files summary:")
        for fname, keys in failures:
            print(f"  {fname}: missing {keys}")
    
    return pd.DataFrame(rows)


def get_roast_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of the loaded roast data.
    
    Args:
        df: DataFrame from load_roast_data_as_dataframe()
        
    Returns:
        Summary statistics
    """
    if df.empty:
        return {"total_roasts": 0}
    
    return {
        "total_roasts": len(df),
        "avg_duration": float(df["drop_time"].mean()),
        "avg_drop_temp": float(df["drop_bt"].mean()),
        "temp_range": {
            "min_drop": float(df["drop_bt"].min()),
            "max_drop": float(df["drop_bt"].max()),
            "min_charge": float(df["charge_bt"].min()),
            "max_charge": float(df["charge_bt"].max()),
        }
    }


# =============================================================================
# DATA PROCESSOR MODULE
# =============================================================================

def annotate_roast_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute charge_time, event observed times, OK flags, and temp diffs for TP, FCs, and DROP.
    
    Args:
        df: DataFrame from load_roast_data_as_dataframe()
        
    Returns:
        Original DataFrame with additional event annotation columns
    """
    def _process(row: pd.Series) -> pd.Series:
        # Default outputs when charge match is missing
        defaults: Dict[str, Any] = {"charge_time": None}
        for evt in ("tp", "fcs", "drop"):
            defaults.update({
                f"{evt}_time_obs": None,
                f"{evt}_et_ok": False,
                f"{evt}_bt_ok": False,
                f"{evt}_et_diff": None,
                f"{evt}_bt_diff": None,
            })
        
        # Find charge index
        et_list = row["et"]
        bt_list = row["bt"]
        charge_et = row["charge_et"]
        charge_bt = row["charge_bt"]
        
        idx_c = next(
            (i for i, (e, b) in enumerate(zip(et_list, bt_list)) 
             if e == charge_et and b == charge_bt), 
            None
        )
        
        if idx_c is None:
            warnings.warn(f"No charge match in {row.get('filename', 'row')}")
            return pd.Series(defaults)
        
        charge_time = row["time"][idx_c]
        result: Dict[str, Any] = {"charge_time": charge_time}
        
        # Annotate each event
        for evt in ("tp", "fcs", "drop"):
            abs_t = charge_time + row.get(f"{evt}_time", 0)
            time_list = row["time"]
            idx_e = next((i for i, t in enumerate(time_list) if t >= abs_t), None)
            
            if idx_e is None:
                warnings.warn(f"{evt.upper()} not found in {row.get('filename', 'row')}")
                # Use defaults if missing
                result.update({k: defaults[k] for k in defaults if k.startswith(evt)})
            else:
                obs_et = et_list[idx_e]
                obs_bt = bt_list[idx_e]
                exp_et = row[f"{evt}_et"]
                exp_bt = row[f"{evt}_bt"]
                result.update({
                    f"{evt}_time_obs": time_list[idx_e],
                    f"{evt}_et_ok": obs_et == exp_et,
                    f"{evt}_bt_ok": obs_bt == exp_bt,
                    f"{evt}_et_diff": obs_et - exp_et,
                    f"{evt}_bt_diff": obs_bt - exp_bt,
                })
        
        return pd.Series(result)
    
    return pd.concat([df, df.apply(_process, axis=1)], axis=1)


def add_charge_drop_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slice time, ET, BT, and ROR between charge and drop; handle missing charge_time.
    
    Args:
        df: DataFrame with annotated events
        
    Returns:
        DataFrame with additional segment columns
    """
    def _slice(row: pd.Series) -> pd.Series:
        times = row["time"]
        ct = row.get("charge_time")
        
        # If charge_time missing or not found, return empty segments
        if ct is None or not isinstance(times, list) or ct not in times:
            return pd.Series({
                "time_seg": [],
                "et_seg": [],
                "bt_seg": [],
                "ror_seg": []
            })
        
        # Normal slicing
        start = times.index(ct)
        end_t = ct + row["drop_time"]
        end = next(i for i, t in enumerate(times) if t >= end_t)
        
        t_seg = times[start:end+1]
        et_seg = row["et"][start:end+1]
        bt_seg = row["bt"][start:end+1]
        ror_full = row["bt_ror"][start:end+1]
        
        # Zero ROR before TP
        tp_abs = ct + row["tp_time"]
        tp_idx = next((i for i, t in enumerate(times) if t >= tp_abs), start) - start
        ror_seg = [0] * max(tp_idx, 0) + ror_full[max(tp_idx, 0):]
        
        return pd.Series({
            "time_seg": t_seg,
            "et_seg": et_seg,
            "bt_seg": bt_seg,
            "ror_seg": ror_seg
        })
    
    return df.join(df.apply(_slice, axis=1))


def hampel_filter(data: np.ndarray, window_size: int = 10, n_sigmas: float = 3) -> np.ndarray:
    """
    Remove spike outliers from a 1D array via Hampel filter.
    
    Args:
        data: Input data array
        window_size: Size of the sliding window
        n_sigmas: Number of standard deviations for outlier detection
        
    Returns:
        Filtered data with outliers replaced by local median
    """
    data = np.asarray(data)
    new_data = data.copy()
    k = window_size
    L = len(data)
    
    for i in range(L):
        start = max(i - k, 0)
        end = min(i + k, L-1)
        window = data[start:end+1]
        med = np.median(window)
        mad = np.mean(np.abs(window - med))
        threshold = n_sigmas * 1.4826 * mad
        
        if abs(data[i] - med) > threshold:
            new_data[i] = med
    
    return new_data


def clean_segments_and_recalc_ror(df: pd.DataFrame, window: int = 30, smooth_size: int = 15) -> pd.DataFrame:
    """
    Clean segments using Hampel filter and recalculate RoR.
    
    Args:
        df: DataFrame with segmented data
        window: Window size for RoR calculation
        smooth_size: Size of smoothing filter
        
    Returns:
        DataFrame with additional cleaned columns
    """
    df2 = df.copy()
    
    def _process(row: pd.Series) -> pd.Series:
        bt = np.array(row["bt_seg"])
        et = np.array(row["et_seg"])
        
        # Apply Hampel filter
        bt_clean = hampel_filter(bt)
        et_clean = hampel_filter(et)
        
        # Recalculate RoR
        ror = np.zeros(len(bt_clean))
        for i in range(window, len(bt_clean)):
            ror[i] = (bt_clean[i] - bt_clean[i-window]) / window * 60
        
        # Smooth RoR
        ror_sm = uniform_filter1d(ror, size=smooth_size).tolist()
        
        return pd.Series({
            "bt_seg_clean": bt_clean.tolist(),
            "et_seg_clean": et_clean.tolist(),
            "ror_seg_clean": ror_sm
        })
    
    # Add cleaned columns
    cleaned_data = df2.apply(_process, axis=1, result_type='expand')
    df2 = pd.concat([df2, cleaned_data], axis=1)
    
    return df2


def regularize_segments(df: pd.DataFrame, n_points: int = 200) -> pd.DataFrame:
    """
    Resample cleaned segments onto a fixed 0→1 grid.
    
    Args:
        df: DataFrame with cleaned segments
        n_points: Number of points in the regularized grid
        
    Returns:
        DataFrame with additional regularized columns
    """
    grid = np.linspace(0, 1, n_points).tolist()
    
    def _interp(row: pd.Series) -> pd.Series:
        ts_list = row.get("time_seg", []) or []
        
        if len(ts_list) < 2:
            # Not enough points to interpolate
            return pd.Series({
                "time_reg": grid,
                "bt_reg": [np.nan] * n_points,
                "et_reg": [np.nan] * n_points,
                "ror_reg": [np.nan] * n_points
            })
        
        ts = np.array(ts_list)
        # Normalize time to 0→1
        t_norm = (ts - ts[0]) / (ts[-1] - ts[0])
        
        # Perform interpolation
        bt_vals = np.interp(grid, t_norm, row["bt_seg_clean"])
        et_vals = np.interp(grid, t_norm, row["et_seg_clean"])
        ror_vals = np.interp(grid, t_norm, row["ror_seg_clean"])
        
        return pd.Series({
            "time_reg": grid,
            "bt_reg": bt_vals.tolist(),
            "et_reg": et_vals.tolist(),
            "ror_reg": ror_vals.tolist()
        })
    
    regs = df.apply(_interp, axis=1, result_type='expand')
    return pd.concat([df.reset_index(drop=True), regs.reset_index(drop=True)], axis=1)


def calculate_development_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate development ratio (time from FCs to Drop as percentage of total time).
    
    Args:
        df: DataFrame with timing data
        
    Returns:
        DataFrame with development ratio column
    """
    df = df.copy()
    df["dev_ratio"] = (df["drop_time"] - df["fcs_time"]) / df["drop_time"] * 100
    return df


def get_roast_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract timing and temperature data for different roast phases.
    
    Args:
        df: DataFrame with event data
        
    Returns:
        DataFrame with phase timing and temperature rise data
    """
    df = df.copy()
    
    # Phase durations
    df["drying_time"] = df["tp_time"]
    df["maillard_time"] = df["fcs_time"] - df["tp_time"]
    df["development_time"] = df["drop_time"] - df["fcs_time"]
    
    # Temperature rises
    df["drying_temp_rise"] = df["tp_bt"] - df["charge_bt"]
    df["maillard_temp_rise"] = df["fcs_bt"] - df["tp_bt"]
    df["development_temp_rise"] = df["drop_bt"] - df["fcs_bt"]
    
    # Rates
    df["drying_rate"] = df["drying_temp_rise"] / df["drying_time"] * 60
    df["maillard_rate"] = df["maillard_temp_rise"] / df["maillard_time"] * 60
    df["development_rate"] = df["development_temp_rise"] / df["development_time"] * 60
    
    return df


def filter_valid_roasts(
    df: pd.DataFrame, 
    min_duration: int = 300, 
    max_duration: int = 1200,
    min_drop_temp: float = 350, 
    max_drop_temp: float = 450
) -> pd.DataFrame:
    """
    Filter out roasts with unrealistic parameters.
    
    Args:
        df: Input DataFrame
        min_duration: Minimum roast duration in seconds
        max_duration: Maximum roast duration in seconds
        min_drop_temp: Minimum drop temperature in °F
        max_drop_temp: Maximum drop temperature in °F
        
    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    
    # Filter based on duration
    df = df[(df["drop_time"] >= min_duration) & (df["drop_time"] <= max_duration)]
    
    # Filter based on drop temperature
    df = df[(df["drop_bt"] >= min_drop_temp) & (df["drop_bt"] <= max_drop_temp)]
    
    # Filter out roasts with missing charge_time
    df = df[df["charge_time"].notna()]
    
    filtered_count = len(df)
    print(f"Filtered {initial_count - filtered_count} roasts, {filtered_count} remaining")
    
    return df.reset_index(drop=True)


def process_roast_data(df: pd.DataFrame, filter_data: bool = True, n_points: int = 200) -> pd.DataFrame:
    """
    Complete processing pipeline for roast data.
    
    Args:
        df: Raw roast data from load_roast_data_as_dataframe()
        filter_data: Whether to filter out invalid roasts
        n_points: Number of points for regularization
        
    Returns:
        Fully processed DataFrame
    """
    print("Starting data processing pipeline...")
    
    # Step 1: Annotate events
    print("1. Annotating roast events...")
    df = annotate_roast_events(df)
    
    # Step 2: Filter valid roasts (optional)
    if filter_data:
        print("2. Filtering valid roasts...")
        df = filter_valid_roasts(df)
    
    # Step 3: Add segments
    print("3. Adding charge-drop segments...")
    df = add_charge_drop_segments(df)
    
    # Step 4: Clean data and recalculate RoR
    print("4. Cleaning data and recalculating RoR...")
    df = clean_segments_and_recalc_ror(df)
    
    # Step 5: Regularize segments
    print("5. Regularizing segments...")
    df = regularize_segments(df, n_points=n_points)
    
    # Step 6: Calculate additional metrics
    print("6. Calculating development ratios and phases...")
    df = calculate_development_ratio(df)
    df = get_roast_phases(df)
    
    print(f"Processing complete! Final dataset has {len(df)} roasts.")
    return df


# =============================================================================
# SCORING ENGINE MODULE
# =============================================================================

def score_ror_curve(ror_data: Union[List[float], np.ndarray], max_score: float = 100) -> Dict[str, Any]:
    """
    Score RoR curve based on peak value, decline ratio, and flick magnitude.
    
    Args:
        ror_data: Rate of rise data
        max_score: Maximum possible score
        
    Returns:
        Score breakdown with individual components
    """
    ror = np.array(ror_data)
    
    if ror.size == 0 or np.all(np.isnan(ror)):
        return {
            "ror_score": 0.0,
            "peak_ror": np.nan,
            "decline_ratio": np.nan,
            "flick_magnitude": np.nan,
            "details": "No valid RoR data"
        }
    
    # Calculate metrics
    peak_value = float(np.nanmax(ror))
    
    # Calculate decline ratio (percentage of points where RoR is increasing)
    diffs = np.diff(ror[~np.isnan(ror)])
    if len(diffs) > 0:
        decline_ratio = float(np.sum(diffs > 0) / len(diffs))
    else:
        decline_ratio = 0.0
    
    # Calculate flick magnitude (max increase in final portion)
    if ror.size >= 30:
        final_portion = ror[-30:]
        final_diffs = np.diff(final_portion[~np.isnan(final_portion)])
        flick_magnitude = float(np.max(final_diffs)) if len(final_diffs) > 0 else 0.0
    else:
        flick_magnitude = 0.0
    
    # Scoring logic
    score = max_score
    
    # Penalize high decline ratio (RoR should generally decrease)
    score -= decline_ratio * 50
    
    # Penalize significant flicking at the end
    flick_penalty = max(0, flick_magnitude - 0.5) * 30
    score -= flick_penalty
    
    # Penalize excessively high peak RoR
    peak_penalty = max(0, peak_value - 40) * 0.5
    score -= peak_penalty
    
    # Ensure score is within bounds
    score = max(0, min(max_score, score))
    
    return {
        "ror_score": round(score, 1),
        "peak_ror": round(peak_value, 2),
        "decline_ratio": round(decline_ratio * 100, 2),
        "flick_magnitude": round(flick_magnitude, 2),
        "decline_penalty": round(decline_ratio * 50, 1),
        "flick_penalty": round(flick_penalty, 1),
        "peak_penalty": round(peak_penalty, 1)
    }


def score_roast_timing(duration: float, target_duration: float = 630, max_score: float = 20) -> Dict[str, Any]:
    """
    Score roast based on timing (duration).
    
    Args:
        duration: Roast duration in seconds
        target_duration: Ideal duration in seconds
        max_score: Maximum possible score
        
    Returns:
        Score and details
    """
    if pd.isna(duration):
        return {"timing_score": 0.0, "duration_penalty": max_score}
    
    # Calculate penalty based on deviation from target
    duration_penalty = abs(duration - target_duration) / 90
    score = max(0, max_score * (1 - duration_penalty))
    
    return {
        "timing_score": round(score, 1),
        "duration_penalty": round(duration_penalty * max_score, 1),
        "duration": duration
    }


def score_temperature_profile(drop_temp: float, target_temp: float = 388, max_score: float = 20) -> Dict[str, Any]:
    """
    Score roast based on drop temperature.
    
    Args:
        drop_temp: Final drop temperature in °F
        target_temp: Ideal drop temperature in °F
        max_score: Maximum possible score
        
    Returns:
        Score and details
    """
    if pd.isna(drop_temp):
        return {"temp_score": 0.0, "temp_penalty": max_score}
    
    # Calculate penalty based on deviation from target
    temp_penalty = abs(drop_temp - target_temp) / 4
    score = max(0, max_score * (1 - temp_penalty))
    
    return {
        "temp_score": round(score, 1),
        "temp_penalty": round(temp_penalty * max_score, 1),
        "drop_temp": drop_temp
    }


def score_development_ratio(dev_ratio: float, target_ratio: float = 22.5, max_score: float = 20) -> Dict[str, Any]:
    """
    Score roast based on development ratio (FCs to Drop time as % of total).
    
    Args:
        dev_ratio: Development ratio as percentage
        target_ratio: Ideal development ratio
        max_score: Maximum possible score
        
    Returns:
        Score and details
    """
    if pd.isna(dev_ratio):
        return {"dev_score": 0.0, "dev_penalty": max_score}
    
    # Calculate penalty based on deviation from target
    dev_penalty = abs(dev_ratio - target_ratio) / 2.5
    score = max(0, max_score * (1 - dev_penalty))
    
    return {
        "dev_score": round(score, 1),
        "dev_penalty": round(dev_penalty * max_score, 1),
        "dev_ratio": dev_ratio
    }


def calculate_composite_score(
    ror_data: Union[List[float], np.ndarray], 
    duration: float, 
    drop_temp: float, 
    dev_ratio: float, 
    ror_weight: float = 0.4, 
    timing_weight: float = 0.2, 
    temp_weight: float = 0.2, 
    dev_weight: float = 0.2
) -> Dict[str, Any]:
    """
    Calculate overall composite roast score.
    
    Args:
        ror_data: Rate of rise data
        duration: Roast duration in seconds
        drop_temp: Drop temperature in °F
        dev_ratio: Development ratio as percentage
        ror_weight: Weight for RoR score
        timing_weight: Weight for timing score
        temp_weight: Weight for temperature score
        dev_weight: Weight for development score
        
    Returns:
        Complete score breakdown
    """
    # Calculate individual scores
    ror_results = score_ror_curve(ror_data, max_score=100)
    timing_results = score_roast_timing(duration, max_score=100)
    temp_results = score_temperature_profile(drop_temp, max_score=100)
    dev_results = score_development_ratio(dev_ratio, max_score=100)
    
    # Calculate weighted composite score
    composite = (
        ror_results["ror_score"] * ror_weight +
        timing_results["timing_score"] * timing_weight +
        temp_results["temp_score"] * temp_weight +
        dev_results["dev_score"] * dev_weight
    )
    
    return {
        "composite_score": round(composite, 1),
        "ror_component": ror_results,
        "timing_component": timing_results,
        "temp_component": temp_results,
        "dev_component": dev_results,
        "weights": {
            "ror": ror_weight,
            "timing": timing_weight,
            "temp": temp_weight,
            "dev": dev_weight
        }
    }


def score_roast_dataframe(df: pd.DataFrame, ror_column: str = "ror_reg") -> pd.DataFrame:
    """
    Apply scoring to an entire DataFrame of roasts.
    
    Args:
        df: DataFrame with roast data
        ror_column: Column name containing RoR data
        
    Returns:
        DataFrame with added scoring columns
    """
    scores = []
    
    for _, row in df.iterrows():
        # Extract data
        ror_data = row.get(ror_column, [])
        duration = row.get("drop_time", np.nan)
        drop_temp = row.get("drop_bt", np.nan)
        
        # Calculate development ratio if not already present
        if "dev_ratio" in row:
            dev_ratio = row["dev_ratio"]
        else:
            fcs_time = row.get("fcs_time", np.nan)
            if not pd.isna(fcs_time) and not pd.isna(duration):
                dev_ratio = (duration - fcs_time) / duration * 100
            else:
                dev_ratio = np.nan
        
        # Calculate scores
        score_data = calculate_composite_score(ror_data, duration, drop_temp, dev_ratio)
        
        # Flatten the results
        scores.append({
            "filename": row.get("filename", ""),
            "composite_score": score_data["composite_score"],
            "ror_score": score_data["ror_component"]["ror_score"],
            "timing_score": score_data["timing_component"]["timing_score"],
            "temp_score": score_data["temp_component"]["temp_score"],
            "dev_score": score_data["dev_component"]["dev_score"],
            "peak_ror": score_data["ror_component"]["peak_ror"],
            "decline_ratio": score_data["ror_component"]["decline_ratio"],
            "flick_magnitude": score_data["ror_component"]["flick_magnitude"],
            "duration": duration,
            "drop_temp": drop_temp,
            "dev_ratio": dev_ratio
        })
    
    # Convert to DataFrame and merge with original
    scores_df = pd.DataFrame(scores)
    return pd.concat([df.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1)


def get_scoring_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for scored roasts.
    
    Args:
        df: DataFrame with scoring columns
        
    Returns:
        Summary statistics
    """
    return {
        "total_roasts": len(df),
        "average_scores": {
            "composite": df["composite_score"].mean(),
            "ror": df["ror_score"].mean(),
            "timing": df["timing_score"].mean(),
            "temperature": df["temp_score"].mean(),
            "development": df["dev_score"].mean()
        },
        "top_roasts": df.nlargest(5, "composite_score")[["filename", "composite_score"]].to_dict('records'),
        "score_distribution": {
            "excellent": len(df[df["composite_score"] >= 80]),
            "good": len(df[(df["composite_score"] >= 60) & (df["composite_score"] < 80)]),
            "fair": len(df[(df["composite_score"] >= 40) & (df["composite_score"] < 60)]),
            "poor": len(df[df["composite_score"] < 40])
        }
    }


# =============================================================================
# PLOTTING MODULE
# =============================================================================

def plot_roast(data: Dict[str, Any], title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    Plot roast curve with raw ET/BT and RoR between TP and DROP.
    
    Args:
        data: Roast data from extract_roast_data()
        title: Optional title for the plot
        save_path: Optional path to save the plot
    """
    time = np.array(data["time"])
    bt = np.array(data["bt"])
    et = np.array(data["et"])
    ror = np.array(data["ror"])
    idx = data["indices"]
    charge_temp = data["charge_bt"]

    # Clamp drop index
    drop_idx = min(idx["drop"], len(time) - 1)

    # Trim data to DROP
    t_raw = time[:drop_idx + 1]
    b_raw = bt[:drop_idx + 1]
    e_raw = et[:drop_idx + 1]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Color palette
    et_color = "#ea5545"  # Red
    bt_color = "#27aeef"  # Blue

    ax1.plot(t_raw, e_raw, color=et_color, label='ET')
    ax1.plot(t_raw, b_raw, color=bt_color, label='BT')

    # Label offset
    label_offset = 5

    # Plot and label charge at time zero
    ax1.scatter(0, charge_temp, color='black', zorder=5)
    ax1.text(0, charge_temp + label_offset,
             f"CHARGE\n{charge_temp:.1f}°F",
             ha="center", va="bottom")

    # Label TP, FCS, DROP on raw data
    for event in ["tp", "fcs", "drop"]:
        ix = min(idx[event], drop_idx)
        x_evt, y_evt = time[ix], bt[ix]
        ax1.scatter(x_evt, y_evt, color='black', zorder=5)
        ax1.text(x_evt, y_evt + label_offset,
                 f"{event.upper()}\n{int(x_evt//60)}:{int(x_evt%60):02d}\n{y_evt:.1f}°F",
                 ha="center", va="bottom")

    # RoR between TP and DROP
    tp_idx = min(idx["tp"], drop_idx)
    r_time = time[tp_idx:drop_idx+1]
    r_segment = ror[tp_idx:drop_idx+1]
    ax2 = ax1.twinx()
    ax2.plot(r_time, r_segment, linestyle="--", color='lightgray', label="RoR (°F/min)")
    ax2.set_ylabel("Rate of Rise (°F/min)")
    ax2.set_ylim(0, 50)

    # Dynamic x-ticks every ~30s
    max_ticks = 10
    interval = ((drop_idx + max_ticks*30 - 1) // (max_ticks*30)) * 30
    ax1.xaxis.set_major_locator(MultipleLocator(interval))
    ax1.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{int(x//60)}:{int(x%60):02d}")
    )
    ax1.set_xlabel("Time since Charge (mm:ss)")
    ax1.set_ylabel("Temperature (°F)")

    # Legend outside
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left",
               bbox_to_anchor=(1.1, .6), borderaxespad=0)

    # Only horizontal gridlines on temperature axis
    ax1.grid(True, axis='y')
    ax1.grid(False, axis='x')
    ax2.grid(False)

    if title:
        plt.title(title)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_all_curves(
    df: pd.DataFrame, 
    curve_type: str = "bt", 
    title: Optional[str] = None, 
    save_path: Optional[str] = None
) -> None:
    """
    Plot all roast curves on a single figure.
    
    Args:
        df: DataFrame with segmented data
        curve_type: 'bt', 'et', or 'ror' to specify which curves to plot
        title: Optional title
        save_path: Optional path to save
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if curve_type == "bt":
        data_col = "bt_seg"
        ylabel = "BT (°F)"
    elif curve_type == "et":
        data_col = "et_seg"
        ylabel = "ET (°F)"
    elif curve_type == "ror":
        data_col = "ror_seg"
        ylabel = "BT Rate of Rise (°F/min)"
        ax.set_ylim(bottom=0)
    else:
        raise ValueError("curve_type must be 'bt', 'et', or 'ror'")
    
    for _, row in df.iterrows():
        if row["charge_time"] is not None:
            t = [tt - row["charge_time"] for tt in row["time_seg"]]
            ax.plot(t, row[data_col], label=row["filename"], alpha=0.7)
    
    ax.set_xlabel("Time since Charge (mm:ss)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x//60)}:{int(x%60):02d}"))
    ax.grid(True, axis="y")
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_score_analysis(df: pd.DataFrame, score_column: str = "composite_score", title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    Create analysis plots for roast scores.
    
    Args:
        df: DataFrame with scores
        score_column: Column name containing scores
        title: Optional title
        save_path: Optional path to save
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Score distribution
    axes[0, 0].hist(df[score_column], bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].set_title("Score Distribution")
    axes[0, 0].set_xlabel(score_column.replace('_', ' ').title())
    axes[0, 0].set_ylabel("Frequency")
    
    # Score vs Duration
    axes[0, 1].scatter(df["drop_time"], df[score_column], alpha=0.7)
    axes[0, 1].set_title("Score vs Roast Duration")
    axes[0, 1].set_xlabel("Duration (seconds)")
    axes[0, 1].set_ylabel(score_column.replace('_', ' ').title())
    
    # Score vs Drop Temperature
    axes[1, 0].scatter(df["drop_bt"], df[score_column], alpha=0.7)
    axes[1, 0].set_title("Score vs Drop Temperature")
    axes[1, 0].set_xlabel("Drop Temperature (°F)")
    axes[1, 0].set_ylabel(score_column.replace('_', ' ').title())
    
    # Top scores bar chart
    top_scores = df.nlargest(10, score_column)
    if len(top_scores) > 0:
        y_positions = range(len(top_scores))
        axes[1, 1].barh(y_positions, top_scores[score_column])
        axes[1, 1].set_yticks(y_positions)
        axes[1, 1].set_yticklabels([f[:20] + "..." if len(f) > 20 else f 
                                   for f in top_scores["filename"]], fontsize=8)
    axes[1, 1].set_title("Top 10 Scores")
    axes[1, 1].set_xlabel(score_column.replace('_', ' ').title())
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_roasts(data_folder: str, options: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Complete roast analysis pipeline.
    
    Args:
        data_folder: Path to folder containing .alog files
        options: Analysis options
        
    Returns:
        Processed DataFrame or None if error
    """
    print("=" * 60)
    print("ROAST CURVE ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print(f"\n1. Loading roast data from {data_folder}...")
    try:
        df = load_roast_data_as_dataframe(data_folder)
        if len(df) == 0:
            print("ERROR: No valid roast files found!")
            return None
        
        print(f"✓ Loaded {len(df)} roasts successfully")
        summary = get_roast_summary(df)
        print(f"  - Average duration: {summary['avg_duration']:.1f} seconds")
        print(f"  - Average drop temp: {summary['avg_drop_temp']:.1f}°F")
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None
    
    # Step 2: Process data
    print("\n2. Processing roast data...")
    try:
        processed_df = process_roast_data(df, filter_data=True, n_points=200)
        print("✓ Data processing complete")
        
    except Exception as e:
        print(f"ERROR processing data: {e}")
        return None
    
    # Step 3: Calculate scores
    print("\n3. Calculating roast scores...")
    try:
        scored_df = score_roast_dataframe(processed_df)
        print("✓ Scoring complete")
        
        scoring_summary = get_scoring_summary(scored_df)
        print(f"  - Average composite score: {scoring_summary['average_scores']['composite']:.1f}")
        print("  - Score distribution:")
        for category, count in scoring_summary['score_distribution'].items():
            print(f"    {category.capitalize()}: {count} roasts")
        
    except Exception as e:
        print(f"ERROR calculating scores: {e}")
        return None
    
    # Step 4: Generate reports
    print("\n4. Generating analysis reports...")
    
    # Top roasts
    top_roasts = scored_df.nlargest(5, "composite_score")
    print("  Top 5 roasts:")
    for _, roast in top_roasts.iterrows():
        filename = roast['filename'] if isinstance(roast['filename'], str) else str(roast['filename'])
        score = roast['composite_score']
        print(f"    {filename}: {score:.1f}")
    
    # Step 5: Export data (if requested)
    if options.get('export_csv'):
        print("\n5. Exporting data to CSV...")
        output_file = "roast_analysis_results.csv"
        export_columns = [
            'filename', 'composite_score', 'ror_score', 'timing_score', 
            'temp_score', 'dev_score', 'drop_time', 'drop_bt', 'dev_ratio',
            'peak_ror', 'decline_ratio', 'flick_magnitude'
        ]
        scored_df[export_columns].to_csv(output_file, index=False)
        print(f"✓ Results exported to {output_file}")
    
    # Step 6: Generate plots (if requested)
    if options.get('plot') or options.get('save_plots'):
        print("\n6. Generating visualizations...")
        plot_output_dir = options.get('save_plots')
        
        if plot_output_dir:
            Path(plot_output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Plot 1: All BT curves
            print("  - Plotting all BT curves...")
            save_path = os.path.join(plot_output_dir, "all_bt_curves.png") if plot_output_dir else None
            plot_all_curves(scored_df, curve_type="bt", 
                           title="All Roast BT Curves", save_path=save_path)
            
            # Plot 2: Score analysis
            print("  - Plotting score analysis...")
            save_path = os.path.join(plot_output_dir, "score_analysis.png") if plot_output_dir else None
            plot_score_analysis(scored_df, score_column="composite_score",
                               title="Roast Score Analysis", save_path=save_path)
            
            print("✓ Visualizations complete")
            
        except Exception as e:
            print(f"ERROR generating plots: {e}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return scored_df


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze coffee roast curves from .alog files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Analyze data/ folder
  python main.py data --plot              # Include visualizations  
  python main.py data --save-plots plots/ # Save plots to plots/ folder
  python main.py custom_data --export-csv # Export results to CSV
        """
    )
    
    parser.add_argument(
        "data_folder", 
        nargs="?", 
        default="data",
        help="Path to folder containing .alog files (default: data)"
    )
    
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Display plots during analysis"
    )
    
    parser.add_argument(
        "--save-plots", 
        metavar="DIR",
        help="Save plots to specified directory"
    )
    
    parser.add_argument(
        "--export-csv", 
        action="store_true",
        help="Export results to CSV file"
    )
    
    parser.add_argument(
        "--filter", 
        action="store_true",
        default=True,
        help="Filter out invalid roasts (default: True)"
    )
    
    parser.add_argument(
        "--points", 
        type=int,
        default=200,
        help="Number of points for curve normalization (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Check if data folder exists
    if not os.path.exists(args.data_folder):
        print(f"ERROR: Data folder '{args.data_folder}' does not exist!")
        sys.exit(1)
    
    # Check if data folder contains .alog files
    alog_files = list(Path(args.data_folder).glob("*.alog"))
    if not alog_files:
        print(f"ERROR: No .alog files found in '{args.data_folder}'!")
        sys.exit(1)
    
    print(f"Found {len(alog_files)} .alog files in {args.data_folder}")
    
    # Setup options
    options = {
        'plot': args.plot,
        'save_plots': args.save_plots,
        'export_csv': args.export_csv,
        'filter_data': args.filter,
        'n_points': args.points
    }
    
    # Run analysis
    try:
        result_df = analyze_roasts(args.data_folder, options)
        
        if result_df is not None:
            print(f"\nAnalysis successful! Processed {len(result_df)} roasts.")
            
            # Show quick summary
            avg_score = result_df['composite_score'].mean()
            max_score = result_df['composite_score'].max()
            best_roast_row = result_df.loc[result_df['composite_score'].idxmax()]
            best_roast = best_roast_row['filename'] if isinstance(best_roast_row['filename'], str) else str(best_roast_row['filename'])
            
            print(f"Average score: {avg_score:.1f}")
            print(f"Best roast: {best_roast} ({max_score:.1f})")
        else:
            print("\nAnalysis failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()