import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile

model = YOLO(r"best.pt")

def process_video(video_file):
    if video_file is None:
        return None

    cap = cv2.VideoCapture(video_file)
    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        output_frames.append(annotated)

    cap.release()

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    h, w, _ = output_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    for frame in output_frames:
        out.write(frame)
    out.release()

    return output_path

gr.Interface(
    fn=process_video,
    inputs=gr.Video(),
    outputs=gr.Video()
).launch()
