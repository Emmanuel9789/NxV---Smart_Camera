from flask import Flask, Response
import cv2

app = Flask(__name__)



def generate_frames():
    while True:
        #Gets frame from camera
        frame = app.camera.get_frame()
        
        #Run motion Detection
        motion_detected, boxes = app.motion_detector.detect(frame)
        
        #Draw boxes on motion area
        for (x,y,w,h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
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

