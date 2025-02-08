from flask import Flask, render_template, request, jsonify, session, url_for
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from gpt_agent import analyze_health_records  # Import your existing analysis function

app = Flask(__name__)
app.secret_key = 'dev-secret-key-123'  # Required for session management

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    file_paths = []
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                file.save(filepath)
                file_paths.append(filepath)
        
        if file_paths:
            try:
                analysis_result = analyze_health_records(file_paths)
                
                # Ensure we have all required fields with fallbacks
                if analysis_result.get('success'):
                    result = analysis_result.get('result', {})
                    session['last_analysis'] = {
                        'timestamp': datetime.now().isoformat(),
                        'synopsis': result.get('synopsis', 'No synopsis available'),
                        'insights_anomalies': result.get('insights_anomalies', 'No insights available'),
                        'citations': result.get('citations', 'No citations available')
                    }
                    return jsonify({
                        'success': True,
                        'result': {
                            'synopsis': result.get('synopsis', 'No synopsis available'),
                            'insights_anomalies': result.get('insights_anomalies', 'No insights available'),
                            'citations': result.get('citations', 'No citations available')
                        }
                    })
                else:
                    raise Exception(analysis_result.get('error', 'Analysis failed'))

            except Exception as e:
                print(f"Analysis error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        else:
            return jsonify({'success': False, 'error': 'No valid files uploaded'}), 400
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        # Clean up uploaded files
        for filepath in file_paths:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"Error cleaning up file {filepath}: {str(e)}"), 400

@app.route('/profile')
def profile():
    # Mock data for demonstration
    user_data = {
        'name': 'Michelle G',
        'vitals': {
            'blood_pressure': '120/80',
            'pulse_rate': '72',
            'height': "5'10\"",
            'weight': '150',
            'bmi': '21.5',
            'respiratory_rate': '16'
        }
    }
    return render_template('profile.html', user=user_data)

if __name__ == '__main__':
    app.run(debug=True)
