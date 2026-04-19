from ultralytics import YOLO # type: ignore

weaopon_model = YOLO("/home/emmanuel/camera_project/runs/detect/train3/weights/best.pt") #custome trained

def detect_weapons(frame, conf_threshold=0.5):
    results = weaopon_model(frame)
    detections = []
    
    for r in results:
        for box in r.boxes:
            if box.conf >= conf_threshold:
                detections.append({'bbox' : box.xyxy[0], 'conf': box.conf})
        
        return detections