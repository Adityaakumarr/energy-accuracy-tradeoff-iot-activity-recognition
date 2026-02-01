"""
Simple startup script for Energy-Accuracy Tradeoff Web Application
Run this script to start the Flask development server
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    try:
        from app import app
        
        print("\n" + "=" * 80)
        print("ENERGY-ACCURACY TRADEOFF WEB APPLICATION")
        print("=" * 80)
        print("\nStarting Flask development server...")
        print("\n📊 Access the application at:")
        print("   • Home:        http://localhost:5000")
        print("   • Results:     http://localhost:5000/results")
        print("   • Predict:     http://localhost:5000/predict")
        print("   • Compare:     http://localhost:5000/compare")
        print("\n🔌 API Endpoints:")
        print("   • Health:      GET  http://localhost:5000/api/health")
        print("   • Methods:     GET  http://localhost:5000/api/methods")
        print("   • Results:     GET  http://localhost:5000/api/results")
        print("   • Predict:     POST http://localhost:5000/api/predict")
        print("   • Compare:     POST http://localhost:5000/api/comparison")
        print("\n💡 Press Ctrl+C to stop the server")
        print("=" * 80)
        print()
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print("\n❌ Error: Missing dependencies!")
        print(f"   {str(e)}")
        print("\n💡 Please install required packages:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)
