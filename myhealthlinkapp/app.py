from flask import Flask, render_template, request, jsonify, session, url_for, send_file
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from gpt_agent import analyze_health_records  # Import your existing analysis function
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from config import default_config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HealthAnalysisApp')

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
    # Get user name from session, default to "Guest" if not found
    user_name = session.get('user_name', 'Guest')
    return render_template('home.html', user_name=user_name)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    target_language = request.form.get('target_language')  # Get requested language
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
                # Pass target language to analysis function
                analysis_result = analyze_health_records(
                    file_paths,
                    target_language=target_language
                )
                
                if analysis_result.get('success'):
                    result = analysis_result.get('result', {})
                    session['last_analysis'] = {
                        'timestamp': datetime.now().isoformat(),
                        'synopsis': result.get('synopsis', 'No synopsis available'),
                        'insights_anomalies': result.get('insights_anomalies', 'No insights available'),
                        'citations': result.get('citations', 'No citations available'),
                        'target_language': target_language
                    }
                    return jsonify(analysis_result)
                else:
                    raise Exception(analysis_result.get('error', 'Analysis failed'))

            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        else:
            return jsonify({'success': False, 'error': 'No valid files uploaded'}), 400
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        # Clean up uploaded files
        for filepath in file_paths:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {str(e)}")

@app.route('/profile')
def profile():
    # Get user name from session or use default
    user_name = session.get('user_name', 'Michelle G')
    first_name, last_name = user_name.split(' ', 1) if ' ' in user_name else (user_name, '')
    
    # Mock data for demonstration
    user_data = {
        'name': {'first': first_name, 'last': last_name},
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

@app.route('/update_profile', methods=['POST'])
def update_profile():
    try:
        data = request.json
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        
        # Combine names and store in session
        full_name = f"{first_name} {last_name}".strip()
        session['user_name'] = full_name
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Profile update error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.json
        
        # Create a BytesIO buffer to receive PDF data
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create the document content
        content = []
        
        # Add title
        content.append(Paragraph("Health Record Analysis", title_style))
        content.append(Spacer(1, 20))
        
        # Add date
        date_str = datetime.now().strftime("%B %d, %Y")
        content.append(Paragraph(f"Generated on {date_str}", normal_style))
        content.append(Spacer(1, 20))
        
        # Add synopsis
        content.append(Paragraph("Synopsis", heading_style))
        content.append(Paragraph(data.get('synopsis', ''), normal_style))
        content.append(Spacer(1, 20))
        
        # Add insights and anomalies
        content.append(Paragraph("Insights and Anomalies", heading_style))
        content.append(Paragraph(data.get('insights_anomalies', ''), normal_style))
        content.append(Spacer(1, 20))
        
        # Add citations if available
        if data.get('citations'):
            content.append(Paragraph("Citations", heading_style))
            content.append(Paragraph(data.get('citations', ''), normal_style))
        
        # Build the PDF document
        doc.build(content)
        
        # Move to the beginning of the buffer
        buffer.seek(0)
        
        return send_file(
            buffer,
            download_name='health-analysis.pdf',
            as_attachment=True,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        return jsonify({'error': 'Failed to generate PDF'}), 500

if __name__ == '__main__':
    app.run(debug=True)
