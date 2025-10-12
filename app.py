from flask import Flask, render_template, Response
from camera import VideoCamera
import download_model
import threading
import time
import atexit
import signal
import sys

app = Flask(__name__)

# Global camera instance to avoid recreating
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = VideoCamera()
        return camera

def cleanup_camera():
    """Cleanup camera resources on exit"""
    global camera
    if camera is not None:
        camera.release_camera()
        camera = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Stopping application...")
    cleanup_camera()
    sys.exit(0)

# ...existing code...

camera_stopped = False  # Add this global flag

def get_camera():
    global camera, camera_stopped
    with camera_lock:
        if camera_stopped:
            return None
        if camera is None:
            camera = VideoCamera()
        return camera



# ...existing code...

# Register cleanup functions
atexit.register(cleanup_camera)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    global camera
    try:
        camera = get_camera()
        while True:
            if camera is None:
                break  
            frame = camera.get_frame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # Small delay to reduce CPU usage
            time.sleep(0.03)  # ~30 FPS
    except Exception as e:
        print(f"Camera error: {e}")
        cleanup_camera()

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop_camera():
    """Endpoint to stop camera"""
    cleanup_camera()
    return "Camera stopped"

if __name__ == '__main__':
    try:
        print("🚀 Starting Face Emotion Detection...")
        print("📹 Camera will be released when you stop the app")
        print("🛑 Press Ctrl+C to stop gracefully")
        app.run(debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopping application...")
        cleanup_camera()
    finally:
        cleanup_camera()
