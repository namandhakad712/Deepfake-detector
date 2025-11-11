#!/usr/bin/env python3
"""
Deepfake detection backend using Hugging Face model: Naman712/Deep-fake-detection
"""

import sys
import os
from pathlib import Path
import logging
import time
import json

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, request, jsonify, g
    from flask_cors import CORS
    from dotenv import load_dotenv
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import cv2
    import numpy as np
    import io
    import base64
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    
    # Load environment variables
    load_dotenv()
    
    # Create Flask app
    app = Flask(__name__)
    
    # Configure app
    app.config.update(
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max file size
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
        DEBUG=os.getenv('FLASK_DEBUG', '1') == '1'
    )
    
    # Enable CORS for development
    CORS(app, origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001"
    ])
    
    # Global variables for ML components
    ml_model = None
    ml_processor = None
    model_loaded = False
    
    def load_huggingface_model():
        """Load the custom Hugging Face deepfake detection model"""
        global ml_model, ml_processor, model_loaded
        
        if model_loaded:
            return True
            
        try:
            logger.info("Loading custom Hugging Face deepfake detection model...")
            
            model_name = os.getenv('MODEL_NAME', 'Naman712/Deep-fake-detection')
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            logger.info(f"Model: {model_name}")
            
            if not hf_token or hf_token == 'your_token_here_replace_this':
                logger.error("❌ No Hugging Face token provided!")
                logger.error("   Please set HUGGINGFACE_TOKEN in .env file")
                logger.error("   Get your token from: https://huggingface.co/settings/tokens")
                return False
            
            logger.info("🔑 Using authentication token")
            
            # Download custom model files (cached)
            from huggingface_hub import hf_hub_download
            
            logger.info("Downloading/Loading cached model files...")
            
            # Create persistent cache directory
            cache_dir = os.path.join(current_dir, 'model_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            try:
                # Download required files (will use cache if already downloaded)
                files_to_download = [
                    'modeling_deepfake.py',
                    'processor_deepfake.py', 
                    'model_87_acc_20_frames_final_data.pt',
                    'config.json'
                ]
                
                downloaded_files = {}
                for filename in files_to_download:
                    local_path = os.path.join(cache_dir, filename)
                    
                    if os.path.exists(local_path):
                        logger.info(f"✅ Using cached {filename}")
                        downloaded_files[filename] = local_path
                    else:
                        logger.info(f"📥 Downloading {filename}...")
                        file_path = hf_hub_download(
                            repo_id=model_name,
                            filename=filename,
                            token=hf_token,
                            local_dir=cache_dir,
                            local_dir_use_symlinks=False
                        )
                        downloaded_files[filename] = file_path
                        logger.info(f"✅ Downloaded {filename}")
                
                # Add cache directory to Python path
                sys.path.insert(0, cache_dir)
                
                logger.info(f"📁 Model cached in: {cache_dir}")
                
                # Try to use the original model, fallback to simple one
                logger.info("Loading model...")
                model_path = downloaded_files['model_87_acc_20_frames_final_data.pt']
                
                try:
                    # Try original complex model
                    from modeling_deepfake import DeepFakeDetectorModel
                    ml_model = DeepFakeDetectorModel.from_pretrained(model_path)
                    logger.info("✅ Using original complex model")
                except Exception as model_error:
                    logger.warning(f"Original model failed: {model_error}")
                    logger.info("Using correct model architecture...")
                    from correct_model import DeepFakeDetector
                    ml_model = DeepFakeDetector.load_from_file(model_path)
                    logger.info("✅ Using correct model architecture")
                
                ml_model.eval()
                
                # Try to use the original processor, fallback to simple one
                try:
                    from processor_deepfake import DeepFakeProcessor
                    ml_processor = DeepFakeProcessor()
                    logger.info("✅ Using original processor with face detection")
                except ImportError as e:
                    logger.warning(f"Original processor failed: {e}")
                    logger.info("Using simplified processor without face detection")
                    from simple_processor import SimpleDeepFakeProcessor
                    ml_processor = SimpleDeepFakeProcessor()
                    logger.info("✅ Using simplified processor")
                
                model_loaded = True
                logger.info("🎉 Custom Hugging Face model loaded successfully!")
                logger.info(f"💾 Model files cached for future use")
                return True
                
            except Exception as e:
                logger.error(f"Error loading custom model: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Failed to load Hugging Face model: {e}")
            if "authentication" in str(e).lower() or "token" in str(e).lower():
                logger.error("   This is likely an authentication issue")
                logger.error("   Please check your HUGGINGFACE_TOKEN in the .env file")
            elif "gated" in str(e).lower():
                logger.error("   This model requires special access")
                logger.error("   Please request access at: https://huggingface.co/Naman712/Deep-fake-detection")
            return False
    
    def extract_frames_from_video(video_file, max_frames=20):
        """Extract frames from uploaded video"""
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_video_{int(time.time())}.mp4"
            video_file.save(temp_path)
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(temp_path)
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error("No frames found in video")
                return []
            
            # Extract frames evenly distributed across the video
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    frame_pil = Image.fromarray(frame)
                    frames.append(frame_pil)
                
                if len(frames) >= max_frames:
                    break
            
            cap.release()
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def analyze_video_frames(frames):
        """Analyze video frames for deepfake detection using custom model"""
        try:
            if not ml_model or not ml_processor:
                logger.error("Model or processor not loaded")
                return None
            
            logger.info(f"Analyzing {len(frames)} frames...")
            
            # Process frames using custom processor
            processed_frames = []
            for frame in frames:
                processed_frame = ml_processor.preprocess_frame(frame)
                processed_frames.append(processed_frame)
            
            # Ensure we have exactly 20 frames (model requirement from README)
            sequence_length = 20
            if len(processed_frames) >= sequence_length:
                # Take evenly distributed frames across the video
                indices = np.linspace(0, len(processed_frames) - 1, sequence_length, dtype=int)
                batch_frames = [processed_frames[i] for i in indices]
            else:
                # Pad with repeated frames if less than 20
                batch_frames = processed_frames.copy()
                while len(batch_frames) < sequence_length:
                    batch_frames.append(batch_frames[-1] if batch_frames else torch.zeros((3, 112, 112)))
                batch_frames = batch_frames[:sequence_length]
            
            # Stack frames into tensor with correct shape
            frames_tensor = torch.stack(batch_frames)  # Shape: (20, 3, 112, 112)
            frames_tensor = frames_tensor.unsqueeze(0)  # Shape: (1, 20, 3, 112, 112)
            
            logger.info(f"Input tensor shape: {frames_tensor.shape}")
            
            # Run inference (handle different model output formats)
            with torch.no_grad():
                outputs = ml_model(frames_tensor)
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    # Complex model with SequenceClassifierOutput
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    # Original model returns (fmap, logits)
                    fmap, logits = outputs
                else:
                    # Simple model returns logits directly
                    logits = outputs
                
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Get prediction (0: real, 1: fake)
                prediction = int(np.argmax(probs))
                confidence = float(probs[prediction])
                
                logger.info(f"Model output - Prediction: {prediction}, Confidence: {confidence:.3f}")
                logger.info(f"Raw probabilities: Real={probs[0]:.3f}, Fake={probs[1]:.3f}")
            
            # Format results
            is_deepfake = prediction == 1  # 1 = fake
            
            result = {
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "prediction_summary": {
                    "total_frames": len(frames),
                    "processed_frames": len(batch_frames),
                    "prediction": "FAKE" if is_deepfake else "REAL"
                },
                "detailed_scores": {
                    "real_score": float(probs[0]),
                    "fake_score": float(probs[1]),
                    "confidence": confidence
                }
            }
            
            logger.info(f"Analysis complete: {'FAKE' if is_deepfake else 'REAL'} (confidence: {confidence:.3f})")
            return result
                
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    @app.before_request
    def before_request():
        """Set up request context"""
        g.start_time = time.time()
        g.request_id = f"{int(time.time())}-{id(request)}"
    
    @app.after_request
    def after_request(response):
        """Log request completion"""
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            logger.info(f"Request {g.request_id} completed in {duration:.3f}s")
        return response
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint - responds immediately without loading model"""
        try:
            # Don't load model here - just report current status
            # Model will be loaded on first /analyze request
            
            return jsonify({
                'success': True,
                'data': {
                    'status': 'healthy',
                    'timestamp': time.time(),
                    'version': '1.0.0-huggingface',
                    'services': {
                        'flask': True,
                        'cors': True,
                        'file_handler': True,
                        'video_processor': True,
                        'deepfake_detector': model_loaded,
                        'model_loaded': model_loaded,
                        'model_type': 'huggingface',
                        'model_name': os.getenv('MODEL_NAME', 'Naman712/Deep-fake-detection')
                    },
                    'endpoints': ['/health', '/upload', '/analyze']
                }
            })
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'success': False,
                'error': 'Health check failed',
                'message': str(e)
            }), 500
    
    @app.route('/upload', methods=['POST'])
    def upload_video():
        """Upload video endpoint"""
        try:
            if 'video' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No video file provided'
                }), 400
            
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No video file selected'
                }), 400
            
            # Store file info for analysis
            file_info = {
                'filename': video_file.filename,
                'size': len(video_file.read()),
                'upload_id': g.request_id
            }
            
            # Reset file pointer
            video_file.seek(0)
            
            return jsonify({
                'success': True,
                'data': {
                    'message': 'Video uploaded successfully',
                    **file_info,
                    'status': 'uploaded'
                }
            })
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return jsonify({
                'success': False,
                'error': 'Upload failed',
                'message': str(e)
            }), 500
    
    @app.route('/analyze', methods=['POST'])
    def analyze_video():
        """Analyze video for deepfakes using Hugging Face model"""
        try:
            logger.info("Starting video analysis...")
            logger.info(f"Model loaded status: {model_loaded}")
            
            # Load model if not already loaded
            if not load_huggingface_model():
                logger.error("Failed to load Hugging Face model for analysis")
                return jsonify({
                    'success': False,
                    'error': 'ML model not available',
                    'details': 'Could not load the Hugging Face deepfake detection model'
                }), 503
            
            # Check if video file is in request
            if 'video' in request.files:
                video_file = request.files['video']
                
                # Extract frames from video
                frames = extract_frames_from_video(video_file)
                
                if not frames:
                    return jsonify({
                        'success': False,
                        'error': 'Could not extract frames from video'
                    }), 400
                
                # Analyze frames
                analysis_result = analyze_video_frames(frames)
                
                if analysis_result is None:
                    return jsonify({
                        'success': False,
                        'error': 'Analysis failed'
                    }), 500
                
                return jsonify({
                    'success': True,
                    'data': {
                        'analysis_id': g.request_id,
                        'status': 'completed',
                        'result': analysis_result,
                        'frames_analyzed': len(frames),
                        'model_info': {
                            'name': os.getenv('MODEL_NAME', 'Naman712/Deep-fake-detection'),
                            'type': 'huggingface'
                        },
                        'timestamp': time.time()
                    }
                })
            
            else:
                # Return error if no video provided
                return jsonify({
                    'success': False,
                    'error': 'No video file provided for analysis'
                }), 400
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return jsonify({
                'success': False,
                'error': 'Analysis failed',
                'message': str(e)
            }), 500
    
    @app.route('/debug-model', methods=['POST'])
    def debug_model():
        """Run debug tests on the Hugging Face model"""
        try:
            logger.info("Running model debug tests...")
            
            # Import and run the debug function
            from debug_hf_model import debug_hf_model
            
            # Capture the debug output
            import io
            import contextlib
            
            # Redirect stdout to capture print statements
            output_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                success = debug_hf_model()
            
            debug_output = output_buffer.getvalue()
            
            return jsonify({
                'success': success,
                'data': {
                    'test_passed': success,
                    'debug_output': debug_output,
                    'timestamp': time.time(),
                    'steps_completed': 5 if success else 0,
                    'total_steps': 5
                }
            })
            
        except Exception as e:
            logger.error(f"Debug model failed: {e}")
            return jsonify({
                'success': False,
                'error': 'Debug test failed',
                'message': str(e)
            }), 500

    @app.route('/', methods=['GET'])
    def home():
        """Home endpoint"""
        cache_dir = os.path.join(current_dir, 'model_cache')
        cache_exists = os.path.exists(cache_dir)
        cache_size = 0
        
        if cache_exists:
            try:
                cache_size = sum(os.path.getsize(os.path.join(cache_dir, f)) 
                               for f in os.listdir(cache_dir) 
                               if os.path.isfile(os.path.join(cache_dir, f)))
                cache_size_mb = cache_size / (1024 * 1024)
            except:
                cache_size_mb = 0
        else:
            cache_size_mb = 0
        
        return jsonify({
            'message': 'Deepfake Detection API - Hugging Face Model',
            'version': '1.0.0-huggingface',
            'status': 'running',
            'model_info': {
                'name': os.getenv('MODEL_NAME', 'Naman712/Deep-fake-detection'),
                'type': 'huggingface',
                'loaded': model_loaded,
                'cache_location': cache_dir,
                'cache_exists': cache_exists,
                'cache_size_mb': round(cache_size_mb, 2)
            },
            'endpoints': {
                'health': '/health',
                'upload': '/upload (POST)',
                'analyze': '/analyze (POST)',
                'debug-model': '/debug-model (POST)'
            }
        })
    
    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({
            'success': False,
            'error': 'File too large',
            'message': 'Maximum file size is 100MB'
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Something went wrong on our end'
        }), 500
    
    if __name__ == '__main__':
        print("🚀 Starting Deepfake Detection Backend (Hugging Face Model)...")
        print(f"📁 Working directory: {current_dir}")
        print(f"🤖 Model: {os.getenv('MODEL_NAME', 'Naman712/Deep-fake-detection')}")
        print("🌐 Server: http://localhost:5000")
        print("📊 Health check: http://localhost:5000/health")
        print("📤 Upload endpoint: http://localhost:5000/upload")
        print("🔍 Analysis endpoint: http://localhost:5000/analyze")
        print("-" * 50)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n🔧 Missing dependency. Install with:")
    print("pip install Flask Flask-CORS python-dotenv torch torchvision transformers opencv-python Pillow")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)