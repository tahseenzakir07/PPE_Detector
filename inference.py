from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO(r"C:\Users\Tahseen Zakir\Downloads\best.pt")

# Open webcam (0) or replace with video file
cap = cv2.VideoCapture(r"C:\Users\Tahseen Zakir\Downloads\PPE_Part1.mp4")  # or use 0 for webcam

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
