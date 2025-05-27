from ultralytics import YOLO
import cv2

model = YOLO(r"best.pt")

cap = cv2.VideoCapture(r"video_file")  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    annotated_frame = results[0].plot()  # Draw boxes and labels

    cv2.imshow("PPE Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
