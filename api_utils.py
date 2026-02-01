"""
API Utility Functions for Energy-Accuracy Tradeoff Web Application

Provides helper functions for the Flask REST API including:
- Feature extraction wrappers
- Data loading and caching
- Prediction pipeline
- Result formatting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
from scipy.fft import fft, dct
from sklearn.ensemble import RandomForestClassifier
import pickle

# Constants (matching research script)
WINDOW_SIZE = 128
N_AXES = 6
SAMPLING_RATE = 50
N_CLASSES = 6
CLASS_NAMES = ['Walking', 'Walking_Upstairs', 'Walking_Downstairs', 'Sitting', 'Standing', 'Laying']
ENERGY_PER_FLOP = 3.7e-12  # joules
ENERGY_PER_BYTE = 12e-6     # joules

# Cache for loaded models and results
_cache = {
    'results': None,
    'models': {}
}

def extract_raw_features(window: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Extract raw features (baseline method)"""
    features = window.flatten()
    flops = 0
    bytes_transmitted = features.shape[0] * 4
    return features, flops, bytes_transmitted

def extract_time_domain_features(window: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Extract time-domain statistical features"""
    features = []
    flops = 0
    
    for axis in range(N_AXES):
        signal = window[:, axis]
        n = len(signal)
        
        # Mean
        mean_val = np.mean(signal)
        flops += n
        
        # Standard deviation
        std_val = np.std(signal)
        flops += 2 * n
        
        # Min and max
        min_val = np.min(signal)
        max_val = np.max(signal)
        flops += 2 * n
        
        # RMS
        rms_val = np.sqrt(np.mean(signal ** 2))
        flops += 2 * n
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        zcr = zero_crossings / n
        flops += 2 * n
        
        features.extend([mean_val, std_val, min_val, max_val, rms_val, zcr])
    
    features = np.array(features)
    bytes_transmitted = features.shape[0] * 4
    return features, flops, bytes_transmitted

def extract_fft_features(window: np.ndarray, top_k: int = 20) -> Tuple[np.ndarray, int, int]:
    """Extract FFT spectral features"""
    features = []
    n = window.shape[0]
    flops = 0
    
    for axis in range(N_AXES):
        signal = window[:, axis]
        
        # FFT computation
        fft_vals = fft(signal)
        flops += int(5 * n * np.log2(n))
        
        # Magnitude
        magnitudes = np.abs(fft_vals[:n//2])
        flops += n // 2
        
        # Keep top-k
        top_indices = np.argsort(magnitudes)[-top_k:]
        top_magnitudes = magnitudes[top_indices]
        flops += n // 2
        
        features.extend(top_magnitudes)
    
    features = np.array(features)
    bytes_transmitted = features.shape[0] * 4
    return features, flops, bytes_transmitted

def extract_dct_features(window: np.ndarray, compression_ratio: int = 4) -> Tuple[np.ndarray, int, int]:
    """Extract DCT compression features"""
    n = window.shape[0]
    coeffs_per_axis = n // compression_ratio
    features = []
    flops = 0
    
    for axis in range(N_AXES):
        signal = window[:, axis]
        
        # DCT computation
        dct_vals = dct(signal, norm='ortho')
        flops += int(5 * n * np.log2(n))
        
        # Keep top coefficients
        top_coeffs = dct_vals[:coeffs_per_axis]
        features.extend(top_coeffs)
    
    features = np.array(features)
    bytes_transmitted = features.shape[0] * 4
    return features, flops, bytes_transmitted

def extract_hybrid_features(window: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Extract hybrid statistical-spectral features"""
    # Time-domain features
    time_features, time_flops, _ = extract_time_domain_features(window)
    
    # FFT features (top-10 per axis)
    fft_features, fft_flops, _ = extract_fft_features(window, top_k=10)
    
    # Combine
    features = np.concatenate([time_features, fft_features])
    flops = time_flops + fft_flops
    bytes_transmitted = features.shape[0] * 4
    
    return features, flops, bytes_transmitted

def calculate_energy(flops: int, bytes_transmitted: int) -> Dict[str, float]:
    """Calculate energy consumption breakdown"""
    computation_energy_uj = flops * ENERGY_PER_FLOP * 1e6
    transmission_energy_uj = bytes_transmitted * ENERGY_PER_BYTE * 1e6
    total_energy_uj = computation_energy_uj + transmission_energy_uj
    
    return {
        'total_uj': round(total_energy_uj, 4),
        'computation_uj': round(computation_energy_uj, 4),
        'transmission_uj': round(transmission_energy_uj, 4)
    }

def get_feature_extractor(method_name: str):
    """Get feature extraction function by method name"""
    extractors = {
        'Raw': lambda w: extract_raw_features(w),
        'Time-Domain': lambda w: extract_time_domain_features(w),
        'FFT': lambda w: extract_fft_features(w, top_k=20),
        'DCT-2x': lambda w: extract_dct_features(w, compression_ratio=2),
        'DCT-4x': lambda w: extract_dct_features(w, compression_ratio=4),
        'DCT-8x': lambda w: extract_dct_features(w, compression_ratio=8),
        'DCT-16x': lambda w: extract_dct_features(w, compression_ratio=16),
        'Hybrid': lambda w: extract_hybrid_features(w)
    }
    return extractors.get(method_name)

def get_method_info() -> List[Dict]:
    """Get information about all available methods"""
    return [
        {
            'name': 'Raw',
            'description': 'Raw transmission baseline - transmit all sensor samples',
            'features': 768,
            'type': 'baseline'
        },
        {
            'name': 'Time-Domain',
            'description': 'Statistical features: mean, std, min, max, RMS, zero-crossing rate',
            'features': 36,
            'type': 'statistical'
        },
        {
            'name': 'FFT',
            'description': 'Fast Fourier Transform - top-20 frequency coefficients per axis',
            'features': 120,
            'type': 'spectral'
        },
        {
            'name': 'DCT-2x',
            'description': 'Discrete Cosine Transform with 2× compression',
            'features': 384,
            'type': 'compression'
        },
        {
            'name': 'DCT-4x',
            'description': 'Discrete Cosine Transform with 4× compression',
            'features': 192,
            'type': 'compression'
        },
        {
            'name': 'DCT-8x',
            'description': 'Discrete Cosine Transform with 8× compression',
            'features': 96,
            'type': 'compression'
        },
        {
            'name': 'DCT-16x',
            'description': 'Discrete Cosine Transform with 16× compression',
            'features': 48,
            'type': 'compression'
        },
        {
            'name': 'Hybrid',
            'description': 'Combined time-domain statistics and spectral features',
            'features': 96,
            'type': 'hybrid'
        }
    ]

def generate_sample_data() -> np.ndarray:
    """Generate sample sensor data for testing (walking activity)"""
    t = np.linspace(0, WINDOW_SIZE/SAMPLING_RATE, WINDOW_SIZE)
    window = np.zeros((WINDOW_SIZE, N_AXES))
    
    # Walking parameters
    accel_mean = [0.5, 0.2, 9.8]
    accel_std = [2.0, 2.0, 1.5]
    gyro_mean = [0.1, 0.0, 0.0]
    gyro_std = [0.5, 0.5, 0.3]
    freq = 2.0
    
    # Accelerometer data
    for axis in range(3):
        base_signal = np.random.normal(accel_mean[axis], accel_std[axis], WINDOW_SIZE)
        periodic = 0.5 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        window[:, axis] = base_signal + periodic
    
    # Gyroscope data
    for axis in range(3):
        base_signal = np.random.normal(gyro_mean[axis], gyro_std[axis], WINDOW_SIZE)
        periodic = 0.2 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        window[:, axis + 3] = base_signal + periodic
    
    return window

def process_prediction(sensor_data: np.ndarray, method_name: str) -> Dict:
    """
    Process sensor data and return prediction with energy metrics
    
    Args:
        sensor_data: Shape (128, 6) sensor window
        method_name: Feature extraction method to use
        
    Returns:
        Dictionary with prediction results and energy metrics
    """
    # Validate input
    if sensor_data.shape != (WINDOW_SIZE, N_AXES):
        raise ValueError(f"Expected shape ({WINDOW_SIZE}, {N_AXES}), got {sensor_data.shape}")
    
    # Extract features
    extractor = get_feature_extractor(method_name)
    if extractor is None:
        raise ValueError(f"Unknown method: {method_name}")
    
    features, flops, bytes_tx = extractor(sensor_data)
    
    # Calculate energy
    energy = calculate_energy(flops, bytes_tx)
    
    # For demo purposes, predict based on simple heuristics
    # In production, you would load a trained model
    accel_magnitude = np.mean(np.sqrt(np.sum(sensor_data[:, :3]**2, axis=1)))
    gyro_magnitude = np.mean(np.sqrt(np.sum(sensor_data[:, 3:]**2, axis=1)))
    
    # Simple heuristic classification
    if accel_magnitude > 11:
        predicted_class = 1  # Walking_Upstairs
    elif accel_magnitude > 10 and gyro_magnitude > 0.3:
        predicted_class = 0  # Walking
    elif accel_magnitude < 9.5:
        predicted_class = 2  # Walking_Downstairs
    elif gyro_magnitude < 0.15:
        predicted_class = 5  # Laying
    elif gyro_magnitude < 0.2:
        predicted_class = 3  # Sitting
    else:
        predicted_class = 4  # Standing
    
    return {
        'predicted_activity': CLASS_NAMES[predicted_class],
        'predicted_class_id': int(predicted_class),
        'confidence': 0.85 + np.random.uniform(0, 0.15),  # Simulated confidence
        'method': method_name,
        'features_extracted': int(len(features)),
        'flops': int(flops),
        'bytes_transmitted': int(bytes_tx),
        'energy': energy,
        'sensor_stats': {
            'accel_magnitude_mean': round(float(accel_magnitude), 3),
            'gyro_magnitude_mean': round(float(gyro_magnitude), 3)
        }
    }

def load_results_data() -> Optional[pd.DataFrame]:
    """Load experimental results (would parse from research script output)"""
    # This would ideally load from a saved results file
    # For now, return sample data structure
    results = [
        {
            'representation_name': 'Raw',
            'feature_dim': 768,
            'accuracy_mean': 0.8600,
            'accuracy_std': 0.0050,
            'f1_macro_mean': 0.8580,
            'energy_total_uj': 9216.00,
            'energy_computation_uj': 0.00,
            'energy_transmission_uj': 9216.00,
            'bytes_per_window': 768,
            'flops_per_window': 0,
            'compression_ratio': 1.0,
            'energy_savings_percent': 0.0
        },
        {
            'representation_name': 'Time-Domain',
            'feature_dim': 36,
            'accuracy_mean': 0.9980,
            'accuracy_std': 0.0010,
            'f1_macro_mean': 0.9978,
            'energy_total_uj': 433.15,
            'energy_computation_uj': 1.15,
            'energy_transmission_uj': 432.00,
            'bytes_per_window': 36,
            'flops_per_window': 9216,
            'compression_ratio': 21.3,
            'energy_savings_percent': 95.3
        },
        # Add more methods as needed
    ]
    
    return pd.DataFrame(results)

def get_visualization_files() -> List[Dict]:
    """Get list of available visualization files"""
    base_path = 'd:/energy-accuracy-tradeoff-iot-activity-recognition'
    plots = [
        {'name': 'Pareto Frontier', 'file': 'plot1_pareto_frontier.png', 'description': 'Energy-Accuracy trade-off frontier'},
        {'name': 'Accuracy vs Bytes', 'file': 'plot2_accuracy_vs_bytes.png', 'description': 'Classification accuracy vs communication cost'},
        {'name': 'Accuracy vs FLOPs', 'file': 'plot3_accuracy_vs_flops.png', 'description': 'Classification accuracy vs computational cost'},
        {'name': 'F1 Score Heatmap', 'file': 'plot4_f1_heatmap.png', 'description': 'Per-activity classification performance'},
        {'name': 'Energy Breakdown', 'file': 'plot5_energy_breakdown.png', 'description': 'Energy consumption breakdown by method'},
        {'name': 'Confusion Matrices', 'file': 'plot6_confusion_matrices.png', 'description': 'Confusion matrices comparison'}
    ]
    
    # Check which files exist
    available_plots = []
    for plot in plots:
        file_path = os.path.join(base_path, plot['file'])
        if os.path.exists(file_path):
            plot['path'] = file_path
            available_plots.append(plot)
    
    return available_plots
