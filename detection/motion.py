import cv2 # OpenCV library for image processing

#This class will handle All motion detection logic
class MotionDetector:
    def __init__(self, min_area = 1500):
        
        """
        min_area:
         
         This is the minimum size of movement to consider as "real motion"
         Helps ignore tiny noise 
         
        """
        self.reference_frame = None #This stores the base image
        self.min_area = min_area #Min movement size threshold
        self.motion_counter = 0
        
    def detect(self, frame):
        """
        This function takes an image and check if there is motion
        
        Parameters:
        frame: current image from camera

        Return:
        motion detection(t/f)
        boxes
        
        """
        
        motion_detected = False #Default: no motion
        boxes = [] #Bounding areas
        
        #Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.resize(gray, (320, 240))
        
        #Blur the image
        gray = cv2.GaussianBlur(gray, (21,21), 0)
        
        #Set reference frame
        if self.reference_frame is None:
            self.reference_frame = gray #Save first frame
            return motion_detected, boxes #No motion for now
        
        #Find differences (Current frame - reference frame)
        frame_delta = cv2.absdiff(self.reference_frame, gray)
        
        #Convert difference into black and white
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        #Remove noise
        thresh = cv2.erode(thresh, None, iterations = 2)
        
        #Expand the image to make detection easier
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        #Do the contours
        contours, _ = cv2.findContours(thresh.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        #Filter small movements
        valid_contours = []
        
        for contour in contours:
            #if detected area is small, ignore it
            if cv2.contourArea(contour) < self.min_area:
                continue
            valid_contours.append(contour)
        
        #Update motion counter to reduce flickering
        if len(valid_contours) > 0:
            self.motion_counter += 1
        else:
            self.motion_counter = 0
        
        #Only confirm motion if it persists across frames
        if self.motion_counter > 2 and len(valid_contours) > 0:
            motion_detected = True
            
            #Merge all contours into ONE big region
            x_min, y_min = 9999, 9999
            x_max, y_max = 0, 0
            
            #Get rect around motion area
            for contour in valid_contours:
                (x, y, w, h) = cv2.boundingRect(contour)

                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            #Padding
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max += padding
            y_max += padding
            
            boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        try:
            self.reference_frame = cv2.addWeighted(
                self.reference_frame, 0.9, gray, 0.1, 0
            )
        except:
            #If anything goes wrong, reset safely
            self.reference_frame = gray
            return motion_detected, boxes

        return motion_detected, boxes