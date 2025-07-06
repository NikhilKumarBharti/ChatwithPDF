import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
from chat_with_pdf import ChatWithPDF

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatWithPDFApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
        
        # Configuration
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        
        # Ensure upload directory exists
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Initialize the ChatWithPDF system
        try:
            self.chat_system = ChatWithPDF()
            logger.info("ChatWithPDF system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChatWithPDF system: {e}")
            self.chat_system = None
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            if not self.chat_system:
                return jsonify({'error': 'Chat system not initialized'}), 500
            
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and self._allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    # Process the PDF
                    success = self.chat_system.process_pdf(file_path)
                    
                    if success:
                        return jsonify({
                            'message': 'PDF uploaded and processed successfully',
                            'filename': filename
                        })
                    else:
                        return jsonify({'error': 'Failed to process PDF'}), 500
                        
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    return jsonify({'error': 'Error processing file'}), 500
            
            return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            if not self.chat_system:
                return jsonify({'error': 'Chat system not initialized'}), 500
            
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({'error': 'No message provided'}), 400
            
            try:
                response = self.chat_system.chat(data['message'])
                return jsonify({'response': response})
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                return jsonify({'error': 'Error processing chat message'}), 500
        
        @self.app.route('/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'chat_system_ready': self.chat_system is not None
            })
    
    def _allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'
    
    def run(self, debug=False, host='0.0.0.0', port=5000, use_reloader=True):
        """Run the Flask application"""
        self.app.run(debug=debug, host=host, port=port)

if __name__ == '__main__':
    app = ChatWithPDFApp()
    app.run(debug=False, use_reloader=False)