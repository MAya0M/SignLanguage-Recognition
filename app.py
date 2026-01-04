"""
Simple Flask Web App for Sign Language Recognition
Upload a video and get predictions from the trained model
"""

import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tempfile
import shutil

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from scripts.predict import SignLanguagePredictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'temp/uploads'
app.config['MODEL_DIR'] = 'models'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'webm'}

# Global predictor instance (singleton pattern for performance)
_predictor_instance = None
_predictor_model_path = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_latest_model():
    """Find the latest trained model"""
    models_dir = Path(app.config['MODEL_DIR'])
    if not models_dir.exists():
        return None
    
    # Find all run directories
    run_dirs = sorted(models_dir.glob('run_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for run_dir in run_dirs:
        model_path = run_dir / 'best_model.keras'
        if model_path.exists():
            return str(model_path)
    
    return None

def get_predictor():
    """Get or create predictor instance (singleton pattern)"""
    global _predictor_instance, _predictor_model_path
    
    model_path = find_latest_model()
    if not model_path:
        return None
    
    # If model path changed or predictor doesn't exist, create new one
    if _predictor_instance is None or _predictor_model_path != model_path:
        print(f"Loading model (first time or model changed): {model_path}")
        _predictor_instance = SignLanguagePredictor(model_path)
        _predictor_model_path = model_path
    else:
        print(f"Reusing existing model instance: {model_path}")
    
    return _predictor_instance

@app.route('/')
def index():
    """Main page"""
    model_path = find_latest_model()
    has_model = model_path is not None
    return render_template('index.html', has_model=has_model, model_path=model_path)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle video upload and prediction"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get predictor instance (reused for performance)
        predictor = get_predictor()
        if not predictor:
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 404
        
        # Make prediction with multiple words detection
        result = predictor.predict_from_video(filepath, detect_multiple_words=True)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Format response
        all_predictions = []
        if 'all_predictions' in result:
            for word, conf in result['all_predictions'].items():
                all_predictions.append({'word': word, 'confidence': float(conf)})
        
        # Format words list
        words_list = []
        if 'words' in result and result.get('multiple_words_detected', False):
            for word_data in result['words']:
                words_list.append({
                    'word': word_data['word'],
                    'confidence': float(word_data['confidence']),
                    'start_frame': word_data.get('start_frame', 0),
                    'end_frame': word_data.get('end_frame', 0)
                })
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': float(result.get('confidence', 0.0)),
            'all_predictions': all_predictions,
            'words': words_list,
            'multiple_words_detected': result.get('multiple_words_detected', False),
            'word_count': result.get('word_count', 1)
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict-live', methods=['POST'])
def predict_live():
    """Handle live video chunk and prediction"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video chunk provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded chunk temporarily (accept any video format for live)
    import time
    filename = f"live_{int(time.time() * 1000)}.webm"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get predictor instance (reused for performance)
        predictor = get_predictor()
        if not predictor:
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 404
        
        # Make prediction with multiple words detection (for live, we still detect multiple words in each chunk)
        result = predictor.predict_from_video(filepath, detect_multiple_words=True)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Format response
        all_predictions = []
        if 'all_predictions' in result:
            for word, conf in result['all_predictions'].items():
                all_predictions.append({'word': word, 'confidence': float(conf)})
        
        # Format words list (for live, usually one word per chunk, but could be multiple)
        words_list = []
        if 'words' in result and result.get('multiple_words_detected', False):
            for word_data in result['words']:
                words_list.append({
                    'word': word_data['word'],
                    'confidence': float(word_data['confidence']),
                    'start_frame': word_data.get('start_frame', 0),
                    'end_frame': word_data.get('end_frame', 0)
                })
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': float(result.get('confidence', 0.0)),
            'all_predictions': all_predictions,
            'words': words_list,
            'multiple_words_detected': result.get('multiple_words_detected', False),
            'word_count': result.get('word_count', 1)
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-status')
def model_status():
    """Check if model is available"""
    model_path = find_latest_model()
    return jsonify({
        'has_model': model_path is not None,
        'model_path': model_path
    })

if __name__ == '__main__':
    # Find model on startup
    model_path = find_latest_model()
    if model_path:
        print(f"Found model: {model_path}")
    else:
        print("No trained model found. Please train a model first.")
    
    # Development mode: use debug=True for auto-reload and better error messages
    # Production: set FLASK_ENV=production or use debug=False
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    port = int(os.environ.get('PORT', 5000))
    
    # Get local IP address for mobile access
    import socket
    try:
        # Connect to a remote server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"
    
    print("\n" + "="*60)
    print("Sign Language Recognition App is running!")
    print("="*60)
    print(f"Access from your phone:")
    print(f"   http://{local_ip}:{port}")
    print(f"\nAccess from this computer:")
    print(f"   http://localhost:{port}")
    print(f"   http://127.0.0.1:{port}")
    print("="*60)
    print("\nMake sure:")
    print("   1. Your phone and computer are on the same WiFi network")
    print("   2. Windows Firewall allows connections on port", port)
    print("   3. If it doesn't work, check your firewall settings")
    print("\n")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

