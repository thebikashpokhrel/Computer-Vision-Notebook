from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # propid=3 width
cap.set(4, 720)  # propid=4 height

model = YOLO("yolo-weights/yolov8n.pt")

while True:
    scuess, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            cvzone.putTextRect(img, f"{conf}", (x1, y1 - 20))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
