#import camera system
from camera.input import Camera

#import motion detection
from detection.motion import MotionDetector

#import Flask stream app
from camera.stream import app, generate_frames

#Initialize Everything

#Create camera obj
camera = Camera()

#Create motion detector object
motion_detector = MotionDetector()

#Pass camera and motion_detector into the stream
app.camera = camera
app.motion_detector = motion_detector

if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 5000, debug = True, use_reloader = False) 