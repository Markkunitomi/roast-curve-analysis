#!/usr/bin/env python3
"""Test script to verify all imports work correctly"""

print("Testing imports from main.py...")

try:
    # Test standard imports
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    print("✓ Standard imports work")
    
    # Test main.py imports
    from main import (
        # ALOG Parser functions
        extract_roast_data,
        load_roast_data_as_dataframe,
        get_roast_summary,
        
        # Data Processing functions
        annotate_roast_events,
        add_charge_drop_segments,
        hampel_filter,
        clean_segments_and_recalc_ror,
        regularize_segments,
        calculate_development_ratio,
        get_roast_phases,
        filter_valid_roasts,
        process_roast_data,
        
        # Scoring Engine functions
        score_ror_curve,
        score_roast_timing,
        score_temperature_profile,
        score_development_ratio,
        calculate_composite_score,
        score_roast_dataframe,
        get_scoring_summary,
        
        # Plotting functions
        plot_roast,
        plot_all_curves,
        plot_score_analysis,
        
        # Main pipeline
        analyze_roasts
    )
    print("✓ All main.py imports work")
    
    # Test basic functionality
    data_folder = Path("data")
    if data_folder.exists():
        alog_files = list(data_folder.glob("*.alog"))
        print(f"✓ Found {len(alog_files)} .alog files")
        
        if alog_files:
            # Test single file extraction
            sample_file = alog_files[0]
            roast_data = extract_roast_data(sample_file)
            print(f"✓ Successfully extracted data from {sample_file.name}")
            print(f"  - Duration: {roast_data['indices']['drop'] / 60:.1f} minutes")
            print(f"  - Charge temp: {roast_data['charge_bt']:.1f}°F")
    else:
        print("⚠ Data folder not found")
    
    print("\n✅ All tests passed! The notebook should work correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()