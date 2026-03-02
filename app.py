import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, send_file
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import tensorflow as tf
import urllib.request
from werkzeug.wsgi import wrap_file
import base64
from datetime import datetime

# -------------------------------
# Flask App Initialization
# -------------------------------
app = Flask(__name__)

# Add MIME type for AVI/MJPG videos
@app.after_request
def add_mime_types(response):
    if response.headers.get('Content-Type', '').startswith('video/'):
        response.headers['Content-Type'] = 'video/x-msvideo'
    return response

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")    # for images & raw videos
MAP_FOLDER = os.path.join(BASE_DIR, "static", "maps")          # for density maps (images)
VIDEO_OUT_FOLDER = os.path.join(BASE_DIR, "static", "videos")  # for processed videos
MODEL_PATH = os.path.join(BASE_DIR, "model", "revuu.keras")

# -------------------------------
# Safe Folder Creation
# -------------------------------
def ensure_folder(path):
    """Ensure folder exists; if a file with same name exists, remove it."""
    if os.path.exists(path):
        if not os.path.isdir(path):
            os.remove(path)
    else:
        os.makedirs(path, exist_ok=True)

ensure_folder(UPLOAD_FOLDER)
ensure_folder(MAP_FOLDER)
ensure_folder(VIDEO_OUT_FOLDER)

# Flask config
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAP_FOLDER"] = MAP_FOLDER
app.config["VIDEO_OUT_FOLDER"] = VIDEO_OUT_FOLDER

# -------------------------------
# Load Model
# -------------------------------
print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# -------------------------------
# Helper Functions (Image)
# -------------------------------
def preprocess_image(image_path):
    """Preprocess input image for model prediction."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_and_save_map(image_path, map_path):
    """Generate density map and crowd count for a single image."""
    img = preprocess_image(image_path)
    prediction = model.predict(img, verbose=0)
    density_map = prediction[0, :, :, 0]
    predicted_count = np.sum(density_map)

    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(density_map, cmap='jet')
    plt.savefig(map_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return predicted_count

# -------------------------------
# Helper Functions (Video)
# -------------------------------
def preprocess_frame(frame):
    """Preprocess a single video frame for model prediction."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_resized = frame_resized.astype(np.float32) / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)
    return frame_resized

def process_video(video_path, output_path,
                  threshold=100, frame_skip=3, target_width=640):
    """
    Process video:
    - Run model only every `frame_skip` frames (for speed)
    - Reuse last prediction for skipped frames
    - Resize frames to `target_width` for smaller, faster video
    - Overlay count text
    Returns: (average_count, max_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize to smaller width for output (keep aspect ratio)
    if target_width is not None and target_width < orig_width:
        scale = target_width / float(orig_width)
        width = target_width
        height = int(orig_height * scale)
    else:
        width = orig_width
        height = orig_height

    # Use MP4V codec (MPEG-4 Part 2) for better compatibility and quality
    # Ensure output has .avi extension for MJPG codec compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Ensure the output path has correct extension
    if not output_path.lower().endswith('.avi'):
        output_path = os.path.splitext(output_path)[0] + '.avi'
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Verify video writer was initialized properly
    if not out.isOpened():
        raise RuntimeError(f"Failed to initialize video writer. Check codec support and output path: {output_path}")

    counts = []
    frame_idx = 0
    last_count = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Resize frame for output
        frame = cv2.resize(frame, (width, height))

        # Only run model every `frame_skip`-th frame
        if frame_idx % frame_skip == 1:
            inp = preprocess_frame(frame)
            prediction = model.predict(inp, verbose=0)
            density_map = prediction[0, :, :, 0]
            last_count = float(np.sum(density_map))
            counts.append(last_count)

        # Overlay count text using latest prediction
        text = f"Count: {int(last_count)}"
        color = (0, 255, 0) if last_count >= threshold else (0, 0, 255)
        cv2.putText(
            frame, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
        )

        out.write(frame)

    cap.release()
    out.release()

    if len(counts) == 0:
        avg_count = 0.0
        max_count = 0.0
    else:
        avg_count = float(np.mean(counts))
        max_count = float(np.max(counts))

    return avg_count, max_count

# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def upload_file():
    """
    Main page - handle image upload or image URL.
    (Video has separate /video route to keep things clean.)
    """
    if request.method == "POST":
        image_url = request.form.get("image_url", "").strip()
        file = request.files.get("file")

        if not file and not image_url:
            return render_template("index.html", message="⚠️ Please upload an image or enter a URL!")

        # Determine file path
        if image_url:
            # If the user selected a local sample (e.g. "/sample/IMG_2.jpg"), copy it
            try:
                # Normalize and extract filename
                filename = os.path.basename(image_url.split("?")[0])
                if not filename:
                    filename = "downloaded_image.jpg"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                # Handle local sample path: starts with '/sample/' or 'sample/'
                sample_prefix = image_url.lstrip('/').split('/')[0]
                if sample_prefix == 'sample':
                    # build local path and copy the file into uploads for processing
                    local_sample_path = os.path.join(BASE_DIR, image_url.lstrip('/'))
                    if os.path.exists(local_sample_path):
                        import shutil
                        shutil.copy(local_sample_path, filepath)
                    else:
                        return render_template("index.html", message=f"❌ Sample image not found: {local_sample_path}")
                else:
                    # Treat as a remote URL and download
                    urllib.request.urlretrieve(image_url, filepath)
            except Exception as e:
                return render_template("index.html", message=f"❌ Error handling image URL: {e}")
        else:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

        map_name = f"{os.path.splitext(filename)[0]}_map.png"
        map_path = os.path.join(app.config["MAP_FOLDER"], map_name)

        predicted_count = predict_and_save_map(filepath, map_path)
        threshold = 100

        # If below threshold, mark as no crowd and set displayed count to 0
        if predicted_count < threshold:
            message = "NO CROWD DETECTED"
            map_name = None
            display_count = 0
        else:
            message = f"Crowd Count: {int(predicted_count)}"
            display_count = int(predicted_count)

        return render_template(
            "result.html",
            filename=filename,
            mapname=map_name,
            message=message,
            count=display_count
        )

    return render_template("index.html")


@app.route("/video", methods=["GET", "POST"])
def upload_video():
    """Handle video upload and processing."""
    if request.method == "POST":
        # Check if it's a sample video request
        sample_video = request.form.get("sample_video")
        file = request.files.get("video_file")

        if sample_video:
            # Handle sample video from uploads directory
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], sample_video)
            if not os.path.exists(video_path):
                return render_template("index.html", vmessage="⚠️ Sample video file not found!")
            filename = sample_video
        elif file and file.filename != "":
            # Handle uploaded video file
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(video_path)
        else:
            return render_template("index.html", vmessage="⚠️ Please upload a video file or select a sample!")

        try:
            processed_name = f"{os.path.splitext(filename)[0]}_processed.avi"
            processed_path = os.path.join(app.config["VIDEO_OUT_FOLDER"], processed_name)

            # Process video
            avg_count, max_count = process_video(video_path, processed_path, threshold=100)
        except Exception as e:
            return render_template("index.html", vmessage=f"⚠️ Error processing video: {str(e)}")

        if avg_count < 100:
            message = "NO CROWD"
            crowd_detected = False
        else:
            message = f"CROWD DETECTED - Avg Count: {int(avg_count)}, Max Count: {int(max_count)}"
            crowd_detected = True

        return render_template(
            "video_result.html",
            original_filename=filename,
            processed_filename=processed_name,
            avg_count=int(avg_count),
            max_count=int(max_count),
            message=message,
            crowd_detected=crowd_detected
        )

    return render_template("index.html")


@app.route("/display/<filename>")
def display_image(filename):
    """Display uploaded image."""
    return redirect(url_for("static", filename=f"uploads/{filename}"), code=301)

@app.route("/maps/<mapname>")
def display_map(mapname):
    """Display density map."""
    return redirect(url_for("static", filename=f"maps/{mapname}"), code=301)

@app.route("/videos/<videoname>")
def display_video(videoname):
    """Stream processed video with proper headers and codec detection."""
    try:
        video_path = os.path.join(VIDEO_OUT_FOLDER, videoname)
        if not os.path.exists(video_path):
            return f"Video file not found: {videoname}", 404
        
        # Determine MIME type based on file extension
        file_ext = os.path.splitext(videoname)[1].lower()
        if file_ext == '.avi':
            mime_type = 'video/x-msvideo'
        elif file_ext == '.mp4':
            mime_type = 'video/mp4'
        elif file_ext == '.mov':
            mime_type = 'video/quicktime'
        else:
            mime_type = 'video/x-msvideo'  # default
        
        # Use send_file for proper streaming
        return send_file(
            video_path,
            mimetype=mime_type,
            as_attachment=False,
            download_name=videoname
        )
    except Exception as e:
        return f"Error serving video: {str(e)}", 500


@app.route('/download/video/<videoname>')
def download_video(videoname):
    """Download processed video file directly."""
    try:
        videoname = secure_filename(videoname)
        video_path = os.path.join(VIDEO_OUT_FOLDER, videoname)
        
        if not os.path.exists(video_path):
            return "Video not found", 404
        
        # Determine MIME type
        file_ext = os.path.splitext(videoname)[1].lower()
        if file_ext == '.avi':
            mime_type = 'video/x-msvideo'
        elif file_ext == '.mp4':
            mime_type = 'video/mp4'
        else:
            mime_type = 'video/x-msvideo'
        
        return send_file(
            video_path,
            mimetype=mime_type,
            as_attachment=True,
            download_name=videoname
        )
    except Exception as e:
        return f"Error downloading video: {str(e)}", 500


@app.route('/sample/<path:filename>')
def sample_file(filename):
    """Serve files from the project's sample/ directory.

    This lets template references like `/sample/IMG_2.jpg` work without moving files.
    """
    sample_dir = os.path.join(BASE_DIR, 'sample')
    return send_from_directory(sample_dir, filename)


@app.route('/download/image_report/<filename>')
def download_image_report(filename):
    """Download complete image analysis report as HTML."""
    try:
        # Sanitize filename
        filename = secure_filename(filename)
        
        # Get the uploaded image path
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(image_path):
            return "Image not found", 404
        
        # Get the density map name
        map_name = f"{os.path.splitext(filename)[0]}_map.png"
        map_path = os.path.join(MAP_FOLDER, map_name)
        
        # Convert images to base64
        def image_to_base64(img_path):
            with open(img_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        
        image_base64 = image_to_base64(image_path)
        map_base64 = image_to_base64(map_path) if os.path.exists(map_path) else None
        
        # Get crowd count from the analysis
        # We'll need to re-run prediction to get the count
        img = preprocess_image(image_path)
        prediction = model.predict(img, verbose=0)
        density_map = prediction[0, :, :, 0]
        predicted_count = int(np.sum(density_map))
        threshold = 100
        
        if predicted_count < threshold:
            message = "NO CROWD DETECTED"
            display_count = 0
            crowd_status = "low-crowd"
        else:
            message = f"Crowd Count: {predicted_count}"
            display_count = predicted_count
            crowd_status = "high-crowd"
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crowd Analysis Report - {filename}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 20px;
        }}

        .header h1 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2rem;
        }}

        .header p {{
            color: #999;
            font-size: 0.95rem;
        }}

        .timestamp {{
            color: #999;
            font-size: 0.85rem;
            margin-top: 10px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-box {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            text-align: center;
        }}

        .stat-label {{
            color: #999;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .stat-value {{
            color: #667eea;
            font-size: 1.8rem;
            font-weight: 700;
        }}

        .result-message {{
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 1rem;
            font-weight: 600;
        }}

        .result-message.high-crowd {{
            background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 107, 107, 0.05));
            color: #e74c3c;
            border-left: 4px solid #e74c3c;
        }}

        .result-message.low-crowd {{
            background: linear-gradient(135deg, rgba(46, 213, 115, 0.1), rgba(46, 213, 115, 0.05));
            color: #27ae60;
            border-left: 4px solid #27ae60;
        }}

        .images-section {{
            margin-bottom: 30px;
        }}

        .images-section h3 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3rem;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .image-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}

        .image-box h4 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }}

        .image-box img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}

        .image-description {{
            color: #999;
            font-size: 0.85rem;
            margin-top: 10px;
        }}

        .detail-info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            border-left: 4px solid #667eea;
        }}

        .detail-info h4 {{
            color: #333;
            margin-bottom: 15px;
        }}

        .detail-info p {{
            color: #666;
            margin-bottom: 10px;
            line-height: 1.6;
        }}

        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #999;
            font-size: 0.85rem;
        }}

        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Crowd Counting Analysis Report</h1>
        <p>Image Analysis - {filename}</p>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="result-message {crowd_status}">
        <span style="font-size: 1.5rem;">{'⚠️' if predicted_count >= threshold else '✓'}</span>
        <span>{message}</span>
    </div>

    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-label">Estimated Count</div>
            <div class="stat-value">{display_count}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Data Type</div>
            <div class="stat-value" style="font-size: 1.1rem; color: #764ba2;">Image</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Model</div>
            <div class="stat-value" style="font-size: 1.1rem; color: #764ba2;">CNN</div>
        </div>
    </div>

    <div class="images-section">
        <h3>Analysis Images</h3>
        <div class="two-column">
            <div class="image-box">
                <h4>Uploaded Image</h4>
                <img src="data:image/png;base64,{image_base64}" alt="Uploaded Image">
                <div class="image-description">Original image submitted for analysis</div>
            </div>
            <div class="image-box">
                <h4>Density Heatmap</h4>
                {'<img src="data:image/png;base64,' + map_base64 + '" alt="Density Map">' if map_base64 else '<p style="color: #999;">No density map generated</p>'}
                <div class="image-description">Heat intensity represents crowd density (Red = High, Blue = Low)</div>
            </div>
        </div>
    </div>

    <div class="detail-info">
        <h4>Analysis Details</h4>
        <p><strong>Model Type:</strong> Convolutional Neural Network (CNN)</p>
        <p><strong>Analysis Type:</strong> Crowd Density Estimation</p>
        <p><strong>Estimated Crowd Count:</strong> {display_count} people</p>
        <p><strong>Status:</strong> {message}</p>
        <p style="margin-top: 15px; color: #667eea;"><strong>Analysis Method:</strong> The deep learning model analyzes the image and generates a density map showing crowd concentration areas. Warmer colors (red/yellow) indicate higher density zones, while cooler colors (blue) indicate lower density areas.</p>
    </div>

    <div class="footer">
        <p>This report was automatically generated by the Crowd Counting Analysis System</p>
        <p>For questions or support, please visit the dashboard</p>
    </div>
</div>

</body>
</html>
"""
        
        # Create a BytesIO object for the file
        from io import BytesIO
        html_bytes = BytesIO(html_content.encode('utf-8'))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            html_bytes,
            mimetype='text/html',
            as_attachment=True,
            download_name=f'crowd_analysis_report_{timestamp}.html'
        )
    except Exception as e:
        return f"Error creating report: {str(e)}", 500


@app.route('/download/video_report/<processed_filename>')
def download_video_report(processed_filename):
    """Download complete video analysis report as HTML."""
    try:
        # Sanitize filename
        processed_filename = secure_filename(processed_filename)
        
        # Get the processed video path
        video_path = os.path.join(VIDEO_OUT_FOLDER, processed_filename)
        if not os.path.exists(video_path):
            return "Video not found", 404
        
        # Convert video to base64
        def video_to_base64(video_file_path):
            with open(video_file_path, 'rb') as video_file:
                return base64.b64encode(video_file.read()).decode('utf-8')
        
        video_base64 = video_to_base64(video_path)
        
        # Create HTML report with embedded video
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Crowd Analysis Report</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 20px;
        }}

        .header h1 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2rem;
        }}

        .header p {{
            color: #999;
            font-size: 0.95rem;
        }}

        .timestamp {{
            color: #999;
            font-size: 0.85rem;
            margin-top: 10px;
        }}

        .video-section {{
            margin-bottom: 30px;
        }}

        .video-section h3 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3rem;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}

        .video-wrapper {{
            background: #000;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}

        .video-wrapper video {{
            width: 100%;
            max-width: 100%;
            border-radius: 8px;
        }}

        .detail-info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            border-left: 4px solid #667eea;
        }}

        .detail-info h4 {{
            color: #333;
            margin-bottom: 15px;
        }}

        .detail-info p {{
            color: #666;
            margin-bottom: 10px;
            line-height: 1.6;
        }}

        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #999;
            font-size: 0.85rem;
        }}

        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Video Crowd Counting Analysis Report</h1>
        <p>Processed Video with Crowd Count Overlay</p>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="video-section">
        <h3>Analyzed Video</h3>
        <div class="video-wrapper">
            <video controls width="100%" style="max-width: 900px;">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        <p style="color: #999; font-size: 0.9rem; margin-top: 15px; text-align: center;">
            This video shows real-time crowd count estimation overlaid on each frame. 
            Green text indicates normal crowd levels, Red text indicates high crowd density.
        </p>
    </div>

    <div class="detail-info">
        <h4>Analysis Details</h4>
        <p><strong>Model Type:</strong> Convolutional Neural Network (CNN)</p>
        <p><strong>Analysis Type:</strong> Frame-by-Frame Crowd Density Estimation</p>
        <p><strong>Processing Method:</strong> Every 3rd frame analyzed, predictions interpolated for skipped frames</p>
        <p style="margin-top: 15px; color: #667eea;"><strong>How to Interpret:</strong> The system analyzes each frame of the video and overlays the estimated crowd count. The color of the count text changes based on crowd density - green for normal levels and red for high density areas.</p>
    </div>

    <div class="footer">
        <p>This report was automatically generated by the Crowd Counting Analysis System</p>
        <p>For questions or support, please visit the dashboard</p>
    </div>
</div>

</body>
</html>
"""
        
        # Create a BytesIO object for the file
        from io import BytesIO
        html_bytes = BytesIO(html_content.encode('utf-8'))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            html_bytes,
            mimetype='text/html',
            as_attachment=True,
            download_name=f'video_analysis_report_{timestamp}.html'
        )
    except Exception as e:
        return f"Error creating report: {str(e)}", 500


# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
