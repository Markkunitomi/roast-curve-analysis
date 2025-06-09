#!/usr/bin/env python3
"""
Tests for main.py roast analysis functionality
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Import functions from main.py
from main import (
    extract_roast_data,
    get_roast_summary,
    annotate_roast_events,
    hampel_filter,
    score_ror_curve,
    score_roast_timing,
    calculate_composite_score,
    process_roast_data,
    score_roast_dataframe
)


class TestAlogParser(unittest.TestCase):
    """Test ALOG parsing functions"""
    
    def setUp(self):
        """Create test data"""
        self.test_alog_data = {
            'temp1': [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390],
            'temp2': [180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370],
            'timex': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570],
            'computed': {
                'CHARGE_BT': 180,
                'CHARGE_ET': 200,
                'TP_time': 120,
                'TP_BT': 230,
                'TP_ET': 250,
                'FCs_time': 300,
                'FCs_BT': 280,
                'FCs_ET': 300,
                'DROP_time': 450,
                'DROP_BT': 330,
                'DROP_ET': 350
            }
        }
    
    def test_extract_roast_data_with_valid_file(self):
        """Test extracting data from a valid alog file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.alog', delete=False) as f:
            f.write(str(self.test_alog_data))
            temp_path = f.name
        
        try:
            data = extract_roast_data(temp_path)
            self.assertIn('time', data)
            self.assertIn('bt', data)
            self.assertIn('et', data)
            self.assertIn('ror', data)
            self.assertIn('indices', data)
            self.assertEqual(data['charge_bt'], 180)
        finally:
            Path(temp_path).unlink()
    
    def test_extract_roast_data_file_not_found(self):
        """Test error handling for missing file"""
        with self.assertRaises(FileNotFoundError):
            extract_roast_data('nonexistent.alog')
    
    def test_get_roast_summary_empty_dataframe(self):
        """Test summary with empty DataFrame"""
        df = pd.DataFrame()
        summary = get_roast_summary(df)
        self.assertEqual(summary['total_roasts'], 0)
    
    def test_get_roast_summary_with_data(self):
        """Test summary with actual data"""
        df = pd.DataFrame({
            'drop_time': [600, 700, 800],
            'drop_bt': [380, 390, 400],
            'charge_bt': [180, 190, 200]
        })
        summary = get_roast_summary(df)
        self.assertEqual(summary['total_roasts'], 3)
        self.assertAlmostEqual(summary['avg_duration'], 700.0)
        self.assertAlmostEqual(summary['avg_drop_temp'], 390.0)


class TestDataProcessor(unittest.TestCase):
    """Test data processing functions"""
    
    def test_hampel_filter_basic(self):
        """Test Hampel filter with outliers"""
        data = np.array([1, 2, 3, 100, 5, 6, 7])  # 100 is an outlier
        filtered = hampel_filter(data, window_size=3, n_sigmas=2)
        
        # The outlier should be replaced with a more reasonable value
        self.assertLess(filtered[3], 50)  # Should be much less than 100
        self.assertEqual(filtered[0], 1)  # Non-outliers should remain
        self.assertEqual(filtered[-1], 7)
    
    def test_hampel_filter_no_outliers(self):
        """Test Hampel filter with clean data"""
        data = np.array([1, 2, 3, 4, 5, 6, 7])
        filtered = hampel_filter(data)
        
        # Should be nearly identical to input
        np.testing.assert_array_almost_equal(data, filtered)
    
    def test_annotate_roast_events(self):
        """Test event annotation"""
        df = pd.DataFrame({
            'et': [[200, 210, 220, 230, 240, 250]],
            'bt': [[180, 190, 200, 210, 220, 230]],
            'time': [[0, 30, 60, 90, 120, 150]],
            'charge_et': [200],
            'charge_bt': [180],
            'tp_time': [60],
            'tp_et': [220],
            'tp_bt': [200],
            'fcs_time': [120],
            'fcs_et': [240],
            'fcs_bt': [220],
            'drop_time': [150],
            'drop_et': [250],
            'drop_bt': [230],
            'filename': ['test_roast']
        })
        
        result = annotate_roast_events(df)
        self.assertIn('charge_time', result.columns)


class TestScoringEngine(unittest.TestCase):
    """Test scoring functions"""
    
    def test_score_ror_curve_empty_data(self):
        """Test RoR scoring with empty data"""
        result = score_ror_curve([])
        self.assertEqual(result['ror_score'], 0.0)
        self.assertTrue(np.isnan(result['peak_ror']))
    
    def test_score_ror_curve_valid_data(self):
        """Test RoR scoring with valid declining curve"""
        # Simulate a good RoR curve that peaks and declines
        ror_data = [0, 5, 10, 15, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1]
        result = score_ror_curve(ror_data)
        
        self.assertGreater(result['ror_score'], 0)
        self.assertLessEqual(result['ror_score'], 100)
        self.assertEqual(result['peak_ror'], 20.0)
        self.assertIn('decline_ratio', result)
    
    def test_score_roast_timing_perfect(self):
        """Test timing score with perfect duration"""
        result = score_roast_timing(630, target_duration=630)  # Perfect match
        self.assertEqual(result['timing_score'], 20.0)
        self.assertEqual(result['duration_penalty'], 0.0)
    
    def test_score_roast_timing_off_target(self):
        """Test timing score with off-target duration"""
        result = score_roast_timing(720, target_duration=630)  # 90 seconds over
        self.assertLess(result['timing_score'], 20.0)
        self.assertGreater(result['duration_penalty'], 0)
    
    def test_calculate_composite_score(self):
        """Test composite scoring"""
        ror_data = [0, 10, 15, 12, 10, 8, 6, 4, 2]
        duration = 630
        drop_temp = 388
        dev_ratio = 22.5
        
        result = calculate_composite_score(ror_data, duration, drop_temp, dev_ratio)
        
        self.assertIn('composite_score', result)
        self.assertIn('ror_component', result)
        self.assertIn('timing_component', result)
        self.assertIn('temp_component', result)
        self.assertIn('dev_component', result)
        self.assertIn('weights', result)
        
        # Should be a good score since all parameters are at target
        self.assertGreater(result['composite_score'], 70)
    
    def test_score_roast_dataframe(self):
        """Test scoring a DataFrame of roasts"""
        df = pd.DataFrame({
            'ror_reg': [[0, 10, 15, 12, 10, 8, 6, 4, 2], [0, 8, 12, 10, 8, 6, 4, 2, 1]],
            'drop_time': [630, 660],
            'drop_bt': [388, 395],
            'fcs_time': [480, 500],
            'filename': ['roast1', 'roast2']
        })
        
        scored_df = score_roast_dataframe(df)
        
        self.assertIn('composite_score', scored_df.columns)
        self.assertIn('ror_score', scored_df.columns)
        self.assertIn('timing_score', scored_df.columns)
        self.assertEqual(len(scored_df), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_with_mock_data(self):
        """Test complete pipeline with minimal mock data"""
        # Create minimal test DataFrame
        df = pd.DataFrame({
            'filename': ['test1', 'test2'],
            'charge_et': [200, 210],
            'charge_bt': [180, 190],
            'tp_time': [120, 130],
            'tp_et': [220, 230],
            'tp_bt': [200, 210],
            'fcs_time': [300, 310],
            'fcs_et': [280, 290],
            'fcs_bt': [260, 270],
            'drop_time': [450, 460],
            'drop_et': [320, 330],
            'drop_bt': [300, 310],
            'time': [[i*30 for i in range(20)], [i*30 for i in range(20)]],
            'bt': [[180+i*7 for i in range(20)], [190+i*6 for i in range(20)]],
            'et': [[200+i*6 for i in range(20)], [210+i*5 for i in range(20)]],
            'bt_ror': [[0]*5 + [10-i*0.5 for i in range(15)], [0]*5 + [8-i*0.4 for i in range(15)]]
        })
        
        # Test processing pipeline
        try:
            processed_df = process_roast_data(df, filter_data=False)
            self.assertGreater(len(processed_df.columns), len(df.columns))
            
            # Test scoring
            scored_df = score_roast_dataframe(processed_df)
            self.assertIn('composite_score', scored_df.columns)
            
        except Exception as e:
            self.fail(f"Integration test failed with error: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)