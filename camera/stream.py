from flask import Flask, Response
from detection.person import detect_weapons
import cv2
from ultralytics import YOLO # type: ignore


app = Flask(__name__)

#Load YOLO model once

def generate_frames():
    while True:
        #Gets frame from camera
        frame = app.camera.get_frame()
        
        #Run motion Detection
        motion_detected, boxes = app.motion_detector.detect(frame)
        
        #Draw boxes on motion area
        for (x,y,w,h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        #YOLO Detection
        detections = detect_weapons(frame, conf_threshold=0.25)
                
        #Draw YOLO boxes (red) and labels
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Weapon {float(det['conf']):.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        #Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        #Send frame to browser
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes+ b'\r\n')

@app.route('/')




def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

