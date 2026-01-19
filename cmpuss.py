import os
import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from twilio.rest import Client
from werkzeug.utils import secure_filename
import tempfile
import time
from datetime import datetime
import threading
import json
from ultralytics import YOLO
# --- Configuration ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "AC0516e9048ace2710872755b514322d6b")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "5e9ac391304179051f68c345dcc0fdf7")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "+12678022742")
TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER", "+14155238886")

# Check if credentials are set
if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, TWILIO_WHATSAPP_NUMBER]):
    print("WARNING: One or more Twilio environment variables are not set.")
    print("Set them using: export TWILIO_ACCOUNT_SID='your_sid'")

# --- Emergency Contact List ---
EMERGENCY_CONTACTS = [
    {"name": "Campus Security", "number": "+919052195502"},
    {"name": "Head of Safety", "number": "+918074939204"},
    {"name": "Test User", "number": "+918074939204"}  # Replace with your WhatsApp number
]


# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# --- Crowd Detection Configuration ---
CROWD_THRESHOLD = 10  # Minimum people count to trigger emergency alert
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class CrowdDetector:
    def __init__(self):
        """Initialize the crowd detector with YOLO model"""
        self.net = None
        self.output_layers = None
        self.classes = None
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.load_yolo_model()
    
    def load_yolo_model(self):
        """Load YOLO model for people detection"""
        try:
            # For simplicity, using OpenCV's built-in HOG detector
            # In production, you would use a proper YOLO model
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("HOG People Detector loaded successfully")
        except Exception as e:
            print(f"Error loading detection model: {e}")
            self.hog = None
    
    from ultralytics import YOLO  # Add at the top if not already

class CrowdDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # you can use yolov8s.pt for better accuracy

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        max_people_count = 0
        frame_skip = 10
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                results = self.model(frame)
                count = 0
                for r in results:
                    for cls in r.boxes.cls:
                        if int(cls) == 0:  # class 0 is person
                            count += 1
                max_people_count = max(max_people_count, count)

            frame_index += 1

        cap.release()

        crowd_detected = max_people_count >= CROWD_THRESHOLD
        return {
            "people_count": max_people_count,
            "confidence": 90,
            "crowd_detected": crowd_detected,
        }
        cap = cv2.VideoCapture(video_path)
        max_people_count = 0
        frame_skip = 10
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                results = self.model(frame)
                count = 0
                for r in results:
                    for cls in r.boxes.cls:
                        if int(cls) == 0:  # class 0 is person
                            count += 1
                max_people_count = max(max_people_count, count)

            frame_index += 1

        cap.release()

        crowd_detected = max_people_count >= CROWD_THRESHOLD
        return {
            "people_count": max_people_count,
            "confidence": 90,
            "crowd_detected": crowd_detected,
        }

# Initialize crowd detector
crowd_detector = CrowdDetector()

# --- Utility Functions ---
def send_emergency_alert_async(latitude, longitude):
    """Send emergency alert in background thread"""
    def send_alert():
        try:
            if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER, TWILIO_PHONE_NUMBER]):
                print("Twilio credentials not configured, skipping alert")
                return
            
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            location_link = f"https://www.google.com/maps?q={latitude},{longitude}"

# 🔍 Debug print to confirm location is being passed correctly
            print("📍 Sending location to WhatsApp:", latitude, longitude)

            
            whatsapp_message = f"🚨 CROWD EMERGENCY ALERT 🚨\n\nLarge crowd detected automatically by surveillance system!\n\n📍 Location:\n{location_link}\n\nImmediate response required."
            
            voice_message_twiml = (
                '<Response><Say voice="alice" language="en-US">'
                'This is an automated crowd emergency alert from the campus safety system. '
                'A large crowd has been detected and requires immediate attention. '
                'Location details have been sent via text message. Please respond immediately.'
                '</Say></Response>'
            )
            
            successful_messages = 0
            successful_calls = 0
            
            for contact in EMERGENCY_CONTACTS:
                try:
                    # Send WhatsApp
                    message = client.messages.create(
                        body=whatsapp_message,
                        from_=f'whatsapp:{"+14155238886"}',
                        to=f'whatsapp:{contact["number"]}'

                    )
                    successful_messages += 1
                    print(f"WhatsApp sent to {contact['name']}: {message.sid}")
                except Exception as e:
                    print(f"WhatsApp failed for {contact['name']}: {e}")
                
                try:
                    # Make voice call
                    call = client.calls.create(
                        twiml=voice_message_twiml,
                        to=contact['number'],
                        from_=TWILIO_PHONE_NUMBER
                    )
                    successful_calls += 1
                    print(f"Call initiated to {contact['name']}: {call.sid}")
                except Exception as e:
                    print(f"Call failed for {contact['name']}: {e}")
            
            print(f"Emergency alert completed: {successful_messages} messages, {successful_calls} calls")
            
        except Exception as e:
            print(f"Emergency alert failed: {e}")
    
    # Run in background thread
    thread = threading.Thread(target=send_alert)
    thread.daemon = True
    thread.start()

# --- API Endpoints ---
@app.route("/")
def index():
    return "<h1>Campus Safety Backend with Crowd Detection is Running</h1>"

@app.route("/contacts", methods=['GET'])
def get_contacts():
    """Provides the contact list to the frontend for display."""
    contacts_for_frontend = [
        {"name": contact["name"], "number": contact["number"]} 
        for contact in EMERGENCY_CONTACTS
    ]
    return jsonify(contacts_for_frontend)

@app.route("/combined-alert", methods=['POST'])
def send_combined_alert():
    """Manual emergency alert endpoint"""
    data = request.get_json()
    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({"error": "Missing location data in request."}), 400
    
    lat = data['latitude']
    lon = data['longitude']
    
    # Send alert in background
    send_emergency_alert_async(lat, lon)
    
    return jsonify({"message": "Manual emergency alert initiated successfully."})

@app.route("/analyze-crowd", methods=['POST'])
def analyze_crowd():
    """Analyze uploaded video for crowd detection"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a video file."}), 400
    
    try:
        # Save uploaded file to temporary location
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"crowd_analysis_{int(time.time())}_{filename}")
        file.save(temp_path)
        
        # Analyze video
        start_time = time.time()
        analysis_result = crowd_detector.analyze_video(temp_path)
        analysis_time = f"{time.time() - start_time:.2f}s"
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        if "error" in analysis_result:
            return jsonify(analysis_result), 500
        
        # Prepare response
        response = {
            "people_count": analysis_result["people_count"],
            "confidence": analysis_result["confidence"],
            "crowd_detected": analysis_result["crowd_detected"],
            "analysis_time": analysis_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send emergency alert if crowd detected
        if analysis_result["crowd_detected"]:
            # Get location from request or use default
            latitude = request.form.get('latitude', 13.0827)  # Chennai coordinates
            longitude = request.form.get('longitude', 80.2707)  # Chennai coordinates
            
            send_emergency_alert_async(latitude, longitude)
            response["emergency_sent"] = True
        else:
            response["emergency_sent"] = False
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

@app.route("/test-detection", methods=['POST'])
def test_detection():
    """Test endpoint for frontend demo"""
    try:
        data = request.get_json()
        people_count = data.get('people_count', 0)
        test_mode = data.get('test_mode', True)
        
        # Simulate analysis result
        confidence = min(85 + (people_count * 2), 95)
        crowd_detected = people_count >= CROWD_THRESHOLD
        
        response = {
            "people_count": people_count,
            "confidence": confidence,
            "crowd_detected": crowd_detected,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send emergency alert if crowd detected and not in test mode
        if crowd_detected and not test_mode:
            latitude = data.get('latitude', 13.0827)
            longitude = data.get('longitude', 80.2707)
            send_emergency_alert_async(latitude, longitude)
            response["emergency_sent"] = True
        else:
            response["emergency_sent"] = False
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Test failed: {str(e)}"}), 500

@app.route("/health", methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)