"""
Flask application for PDF Merger Bot with web UI.
"""
import os
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pdf_processor import PDFProcessor


# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize PDF processor
pdf_processor = PDFProcessor()

# In-memory storage for sessions
sessions = {}


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_data(session_id):
    """Retrieve session data."""
    return sessions.get(session_id, {
        'files': [],
        'created': datetime.now().isoformat()
    })


def save_session_data(session_id, data):
    """Save session data."""
    sessions[session_id] = data


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/session/new', methods=['POST'])
def create_session():
    """Create a new session."""
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        'files': [],
        'created': datetime.now().isoformat()
    }
    return jsonify({
        'success': True,
        'session_id': session_id
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        session_id = request.form.get('session_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 400
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Only PDF files are allowed'
            }), 400
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(file_path)
        
        # Get PDF info
        pdf_info = pdf_processor.get_pdf_info(file_path)
        
        if not pdf_info['success']:
            os.remove(file_path)
            return jsonify({
                'success': False,
                'error': 'Invalid PDF file'
            }), 400
        
        # Add to session
        session_data = get_session_data(session_id)
        file_record = {
            'id': str(uuid.uuid4())[:8],
            'original_name': file.filename,
            'filename': unique_filename,
            'path': file_path,
            'pages': pdf_info['page_count'],
            'uploaded': datetime.now().isoformat()
        }
        session_data['files'].append(file_record)
        save_session_data(session_id, session_data)
        
        return jsonify({
            'success': True,
            'file': {
                'id': file_record['id'],
                'name': file.filename,
                'pages': pdf_info['page_count']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Upload error: {str(e)}'
        }), 500


@app.route('/api/files', methods=['GET'])
def get_files():
    """Get uploaded files for session."""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 400
        
        session_data = get_session_data(session_id)
        files = []
        
        for file_record in session_data['files']:
            files.append({
                'id': file_record['id'],
                'name': file_record['original_name'],
                'pages': file_record['pages']
            })
        
        return jsonify({
            'success': True,
            'files': files
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/remove-file', methods=['POST'])
def remove_file():
    """Remove a file from session."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        file_id = data.get('file_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 400
        
        session_data = get_session_data(session_id)
        
        for file_record in session_data['files']:
            if file_record['id'] == file_id:
                try:
                    os.remove(file_record['path'])
                except:
                    pass
                session_data['files'].remove(file_record)
                save_session_data(session_id, session_data)
                return jsonify({'success': True})
        
        return jsonify({
            'success': False,
            'error': 'File not found'
        }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/merge', methods=['POST'])
def merge_pdfs():
    """Merge PDFs."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        merge_config = data.get('config', [])
        output_filename = data.get('output_filename', 'merged_document')
        
        if not session_id or session_id not in sessions:
            return jsonify({
                'success': False,
                'error': 'Invalid session'
            }), 400
        
        if not merge_config:
            return jsonify({
                'success': False,
                'error': 'No files selected for merging'
            }), 400
        
        session_data = get_session_data(session_id)
        
        # Sanitize output filename
        output_filename = secure_filename(output_filename.replace('.pdf', ''))
        if not output_filename:
            output_filename = 'merged_document'
        
        # Build merge configuration with file paths
        merge_list = []
        for item in merge_config:
            file_id = item.get('file_id')
            pages = item.get('pages', 'all')
            
            # Find file path
            file_path = None
            for file_record in session_data['files']:
                if file_record['id'] == file_id:
                    file_path = file_record['path']
                    break
            
            if not file_path:
                return jsonify({
                    'success': False,
                    'error': f'File not found: {file_id}'
                }), 400
            
            # Convert page format if needed
            if pages != 'all':
                pages = [int(p) for p in pages]
            
            merge_list.append({
                'path': file_path,
                'pages': pages
            })
        
        # Generate output filename with user-provided name
        unique_suffix = uuid.uuid4().hex[:8]
        final_filename = f"{output_filename}_{unique_suffix}.pdf"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        
        # Merge PDFs
        result = pdf_processor.merge_pdfs(merge_list, output_path)
        
        if not result['success']:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'PDF merged successfully',
            'file': final_filename,
            'size': result['file_size']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Merge error: {str(e)}'
        }), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download merged PDF."""
    try:
        # Security: validate filename
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({
                'success': False,
                'error': 'Invalid filename'
            }), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Extract the user-provided filename from the stored filename
        # Format: "filename_XXXXXXXX.pdf"
        base_name = filename.rsplit('_', 1)[0] if '_' in filename else 'merged_document'
        download_name = f"{base_name}.pdf"
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup_session():
    """Clean up session files."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_id in sessions:
            session_data = get_session_data(session_id)
            for file_record in session_data['files']:
                try:
                    os.remove(file_record['path'])
                except:
                    pass
            del sessions[session_id]
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
