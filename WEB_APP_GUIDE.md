# Flask Web Application Guide

## Overview

This Flask web application provides an interactive interface for exploring the Energy-Accuracy Tradeoff research results. It includes a REST API and user-friendly frontend for testing different feature extraction methods for IoT activity recognition.

## Features

- **Interactive Dashboard**: View experimental results and visualizations
- **Real-time Prediction**: Test different methods with sensor data
- **Method Comparison**: Compare multiple methods side-by-side
- **REST API**: Programmatic access to all functionality
- **Dark Mode**: Toggle between light and dark themes
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Navigate to the project directory**:

   ```bash
   cd d:\energy-accuracy-tradeoff-iot-activity-recognition
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - Flask (web framework)
   - Flask-CORS (API access)
   - NumPy, Pandas (data processing)
   - Matplotlib, Seaborn (visualizations)
   - SciPy, scikit-learn (machine learning)

## Running the Application

### Method 1: Using the startup script (Recommended)

```bash
python run_app.py
```

### Method 2: Direct Flask execution

```bash
python app.py
```

### Method 3: Using Flask CLI

```bash
set FLASK_APP=app.py
flask run
```

The application will start on `http://localhost:5000`

## Application Structure

```
energy-accuracy-tradeoff-iot-activity-recognition/
├── app.py                      # Main Flask application
├── api_utils.py                # API utility functions
├── run_app.py                  # Startup script
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Home page
│   ├── results.html           # Results dashboard
│   ├── predict.html           # Prediction interface
│   └── compare.html           # Method comparison
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css          # Main stylesheet
│   └── js/
│       └── main.js            # JavaScript utilities
└── requirements.txt            # Python dependencies
```

## Using the Web Interface

### Home Page (`/`)

- Overview of the research project
- Key findings and statistics
- Quick navigation to other sections

### Results Dashboard (`/results`)

- Comprehensive results table
- All 6 research visualizations
- Sortable performance metrics

### Prediction Interface (`/predict`)

1. Select a feature extraction method
2. Choose data source:
   - Use sample data (pre-generated walking activity)
   - Upload CSV file (128 rows × 6 columns)
3. Click "Run Prediction"
4. View predicted activity and energy metrics

**CSV Format**:

```csv
accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
0.5,0.2,9.8,0.1,0.0,0.0
...
(128 rows total)
```

### Method Comparison (`/compare`)

1. Select 2-4 methods to compare
2. Click "Compare Selected Methods"
3. View side-by-side comparison:
   - Comparison table
   - Bar charts for accuracy, energy, features
   - Radar chart for multi-metric analysis

## API Documentation

### Base URL

```
http://localhost:5000/api
```

### Endpoints

#### Health Check

```http
GET /api/health
```

Response:

```json
{
  "status": "healthy",
  "message": "Energy-Accuracy Tradeoff API is running"
}
```

#### Get All Methods

```http
GET /api/methods
```

Response:

```json
{
  "success": true,
  "count": 8,
  "methods": [
    {
      "name": "Time-Domain",
      "description": "Statistical features...",
      "features": 36,
      "type": "statistical"
    },
    ...
  ]
}
```

#### Get Method Details

```http
GET /api/method/<method_name>
```

Example: `GET /api/method/Time-Domain`

#### Get Results

```http
GET /api/results
```

Returns experimental results for all methods.

#### Predict Activity

```http
POST /api/predict
Content-Type: application/json

{
  "sensor_data": [[...], [...], ...],  // 128x6 array
  "method": "Time-Domain"
}
```

Response:

```json
{
  "success": true,
  "prediction": {
    "predicted_activity": "Walking",
    "confidence": 0.92,
    "method": "Time-Domain",
    "features_extracted": 36,
    "energy": {
      "total_uj": 433.15,
      "computation_uj": 1.15,
      "transmission_uj": 432.00
    },
    ...
  }
}
```

#### Get Sample Data

```http
GET /api/sample-data
```

Returns sample sensor data for testing.

#### Compare Methods

```http
POST /api/comparison
Content-Type: application/json

{
  "sensor_data": [[...], [...], ...],  // Optional
  "methods": ["Time-Domain", "FFT", "DCT-4x"]
}
```

Returns predictions for all specified methods.

#### Get Visualizations

```http
GET /api/visualizations
```

Returns list of available visualization files.

#### Serve Visualization

```http
GET /api/visualization/<filename>
```

Example: `GET /api/visualization/plot1_pareto_frontier.png`

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:5000/api/health

# Get methods
curl http://localhost:5000/api/methods

# Get sample data
curl http://localhost:5000/api/sample-data

# Predict (with sample data)
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"method": "Time-Domain"}'
```

### Using Python

```python
import requests

# Get methods
response = requests.get('http://localhost:5000/api/methods')
print(response.json())

# Predict
data = {
    "method": "Time-Domain"
}
response = requests.post('http://localhost:5000/api/predict', json=data)
print(response.json())
```

## Troubleshooting

### Port Already in Use

If port 5000 is already in use, modify `app.py` or `run_app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### Visualizations Not Loading

Ensure the plot PNG files exist in the project directory:

- `plot1_pareto_frontier.png`
- `plot2_accuracy_vs_bytes.png`
- `plot3_accuracy_vs_flops.png`
- `plot4_f1_heatmap.png`
- `plot5_energy_breakdown.png`
- `plot6_confusion_matrices.png`

Run the research script to generate them:

```bash
python energy_accuracy_research.py
```

### CORS Errors

The application includes Flask-CORS for API access. If you encounter CORS issues, ensure Flask-CORS is installed:

```bash
pip install Flask-CORS
```

## Development

### Debug Mode

The application runs in debug mode by default, which:

- Auto-reloads on code changes
- Provides detailed error messages
- Enables interactive debugger

For production deployment, set `debug=False` in `app.py`.

### Adding New Endpoints

1. Add route function in `app.py`
2. Add utility functions in `api_utils.py` if needed
3. Update this documentation

### Customizing Styles

Edit `static/css/style.css` to customize:

- Colors (CSS variables in `:root`)
- Layout and spacing
- Dark mode styles

## Security Notes

⚠️ **This application is designed for local development and demonstration purposes.**

For production deployment, consider:

- Authentication and authorization
- Rate limiting
- Input validation and sanitization
- HTTPS/SSL
- Environment-based configuration
- Production WSGI server (Gunicorn, uWSGI)

## Support

For issues or questions:

- Check this documentation
- Review the implementation plan
- Create an issue on GitHub

## License

MIT License - See LICENSE file for details
