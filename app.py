"""
Simple Flask Web App for Sign Language Recognition
Upload a video and get predictions from the trained model
"""

import os
import sys
from pathlib import Path
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
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

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
        # Find model
        model_path = find_latest_model()
        if not model_path:
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 404
        
        # Initialize predictor (reuse instance if possible, but for simplicity create new)
        predictor = SignLanguagePredictor(model_path)
        
        # Make prediction
        result = predictor.predict_from_video(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Format response
        all_predictions = []
        if 'all_predictions' in result:
            for word, conf in result['all_predictions'].items():
                all_predictions.append({'word': word, 'confidence': float(conf)})
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': float(result.get('confidence', 0.0)),
            'all_predictions': all_predictions
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
        print(f"✅ Found model: {model_path}")
    else:
        print("⚠️  No trained model found. Please train a model first.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

