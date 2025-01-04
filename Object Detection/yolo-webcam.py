from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)  # We can replace camera ID 0 with video path
cap.set(3, 1280)  # propid=3 width
cap.set(4, 720)  # propid=4 height

model = YOLO("yolo-weights/yolov8n.pt")

while True:
    scuess, img = cap.read()
    results = model(img, stream=True)
    names = model.names

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            class_id = int(box.cls[0])  # Class index
            class_name = names[class_id]  # Get the class name

            cvzone.putTextRect(
                img, f"{class_name} {conf}", (max(0, x1), max(0, y1 - 20))
            )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
