from picamera2 import Picamera2
from libcamera import controls
import cv2

class Camera:
    def __init__(self, resolution=(480, 360)):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.set_controls({"AwbMode": controls.AwbModeEnum.Daylight})
        self.picam2.start()

    def get_frame(self):
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame.copy()

