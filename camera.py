import os
import gdown
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        # Try to load lightweight model first, fallback to original
        lightweight_model_path = "lightweight_model.h5"
        lightweight_json_path = "lightweight_model.json"
        original_model_path = "static/model.h5"
        original_json_path = "static/model.json"

        if os.path.exists(lightweight_model_path) and os.path.exists(lightweight_json_path):
            print("Loading lightweight model for faster inference...")
            self.model = model_from_json(open(lightweight_json_path, "r").read())
            self.model.load_weights(lightweight_model_path)
        else:
            print("Loading original model...")
            # Download model.h5 from Google Drive if it doesn't exist
            if not os.path.exists(original_model_path):
                print("Downloading model.h5 from Google Drive...")
                url = "https://drive.google.com/uc?id=1MPql4BPEmMBw9y0XJP66kk8SFxaTxG7j"
                gdown.download(url, original_model_path, quiet=False)

            # Load model architecture and weights
            self.model = model_from_json(open(original_json_path, "r").read())
            self.model.load_weights(original_model_path)

        # Load Haar Cascade
        self.face_cascade = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")

        # Define emotions
        self.emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()
            cv2.destroyAllWindows()
    
    def release_camera(self):
        """Manually release camera resources"""
        if hasattr(self, 'video'):
            self.video.release()
            cv2.destroyAllWindows()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Fast face detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=4, 
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            # Scale coordinates back to original frame
            x, y, w, h = x*2, y*2, w*2, h*2
            
            # Extract face region
            roi_gray = gray[y//2:(y+h)//2, x//2:(x+w)//2]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Minimal preprocessing for speed
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255.0

            # Fast prediction
            prediction = self.model.predict(img_pixels, verbose=0)
            max_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            
            # Show emotion if confident enough
            if confidence > 0.4:
                predicted_emotion = self.emotions[max_index]
            else:
                predicted_emotion = "UNKNOWN"

            # Simple color coding
            if predicted_emotion == 'happy':
                color = (0, 255, 0)
            elif predicted_emotion == 'sad':
                color = (255, 0, 255)
            elif predicted_emotion == 'angry':
                color = (0, 0, 255)
            elif predicted_emotion == 'surprise':
                color = (0, 255, 255)
            elif predicted_emotion == 'fear':
                color = (128, 0, 128)
            elif predicted_emotion == 'disgust':
                color = (0, 128, 128)
            else:
                color = (128, 128, 128)

            # Display emotion
            text = f"{predicted_emotion.upper()}"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Fast encoding
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()
