"""
Flask Web Application for Energy-Accuracy Tradeoff IoT Activity Recognition

A REST API and web interface for exploring research results and testing
different feature extraction methods for activity recognition.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import json
import os
from api_utils import (
    get_method_info,
    get_feature_extractor,
    process_prediction,
    generate_sample_data,
    load_results_data,
    get_visualization_files,
    calculate_energy,
    CLASS_NAMES
)

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# WEB ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page with project overview"""
    return render_template('index.html')

@app.route('/results')
def results_page():
    """Results dashboard with visualizations"""
    return render_template('results.html')

@app.route('/predict')
def predict_page():
    """Interactive prediction interface"""
    return render_template('predict.html')

@app.route('/compare')
def compare_page():
    """Method comparison page"""
    return render_template('compare.html')

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Energy-Accuracy Tradeoff API is running'
    })

@app.route('/api/methods', methods=['GET'])
def get_methods():
    """Get list of all available feature extraction methods"""
    try:
        methods = get_method_info()
        return jsonify({
            'success': True,
            'count': len(methods),
            'methods': methods
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/method/<method_name>', methods=['GET'])
def get_method_details(method_name):
    """Get detailed information about a specific method"""
    try:
        methods = get_method_info()
        method = next((m for m in methods if m['name'] == method_name), None)
        
        if method is None:
            return jsonify({
                'success': False,
                'error': f'Method "{method_name}" not found'
            }), 404
        
        return jsonify({
            'success': True,
            'method': method
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get experimental results for all methods"""
    try:
        results_df = load_results_data()
        
        if results_df is None:
            return jsonify({
                'success': False,
                'error': 'Results data not available'
            }), 404
        
        results = results_df.to_dict('records')
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/visualizations', methods=['GET'])
def get_visualizations():
    """Get list of available visualization files"""
    try:
        plots = get_visualization_files()
        return jsonify({
            'success': True,
            'count': len(plots),
            'visualizations': plots
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/visualization/<filename>', methods=['GET'])
def serve_visualization(filename):
    """Serve a visualization image file"""
    try:
        file_path = os.path.join(BASE_DIR, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'Visualization "{filename}" not found'
            }), 404
        
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict activity from uploaded sensor data
    
    Expected JSON format:
    {
        "sensor_data": [[...], [...], ...],  # 128x6 array
        "method": "Time-Domain"  # Feature extraction method
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Get method (default to Time-Domain)
        method = data.get('method', 'Time-Domain')
        
        # Get sensor data
        if 'sensor_data' in data:
            sensor_data = np.array(data['sensor_data'])
        else:
            # Generate sample data if none provided
            sensor_data = generate_sample_data()
        
        # Validate shape
        if sensor_data.shape != (128, 6):
            return jsonify({
                'success': False,
                'error': f'Invalid sensor data shape. Expected (128, 6), got {sensor_data.shape}'
            }), 400
        
        # Process prediction
        result = process_prediction(sensor_data, method)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    """Generate sample sensor data for testing"""
    try:
        sensor_data = generate_sample_data()
        
        return jsonify({
            'success': True,
            'sensor_data': sensor_data.tolist(),
            'shape': list(sensor_data.shape),
            'description': 'Sample walking activity data (128 samples, 6 axes)'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/comparison', methods=['POST'])
def compare_methods():
    """
    Compare multiple methods on the same sensor data
    
    Expected JSON format:
    {
        "sensor_data": [[...], [...], ...],  # Optional, 128x6 array
        "methods": ["Time-Domain", "FFT", "DCT-4x"]  # Methods to compare
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Get methods to compare
        methods = data.get('methods', ['Time-Domain', 'FFT', 'Hybrid'])
        
        # Get sensor data
        if 'sensor_data' in data:
            sensor_data = np.array(data['sensor_data'])
        else:
            sensor_data = generate_sample_data()
        
        # Validate shape
        if sensor_data.shape != (128, 6):
            return jsonify({
                'success': False,
                'error': f'Invalid sensor data shape. Expected (128, 6), got {sensor_data.shape}'
            }), 400
        
        # Process each method
        comparisons = []
        for method in methods:
            try:
                result = process_prediction(sensor_data, method)
                comparisons.append(result)
            except Exception as e:
                comparisons.append({
                    'method': method,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'count': len(comparisons),
            'comparisons': comparisons
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/activities', methods=['GET'])
def get_activities():
    """Get list of activity classes"""
    return jsonify({
        'success': True,
        'count': len(CLASS_NAMES),
        'activities': CLASS_NAMES
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found'
        }), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
    return render_template('500.html'), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Energy-Accuracy Tradeoff Web Application")
    print("=" * 80)
    print("\nStarting Flask development server...")
    print("\nAccess the application at:")
    print("  • Home:        http://localhost:5000")
    print("  • Results:     http://localhost:5000/results")
    print("  • Predict:     http://localhost:5000/predict")
    print("  • Compare:     http://localhost:5000/compare")
    print("\nAPI Documentation:")
    print("  • Health:      GET  /api/health")
    print("  • Methods:     GET  /api/methods")
    print("  • Results:     GET  /api/results")
    print("  • Predict:     POST /api/predict")
    print("  • Compare:     POST /api/comparison")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    # Get port from environment variable (for Render) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
