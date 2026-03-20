import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Camera {i}:", "Works" if ret else "No frame")
    else:
        print(f"Camera {i}: Not opened")
    cap.release()

