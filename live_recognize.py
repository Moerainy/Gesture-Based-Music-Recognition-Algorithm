import time
import os
import cv2
import torch
import numpy as np
from collections import deque
from train_3dcnn import build_model

# ---- config ----
CHECKPOINT = 'models/best.pt'
BUFFER_SECONDS = 1
SAMPLE_FRAMES = 16
INFERENCE_HZ = 1          # max inferences per second
ROTATE_CW90 = False       # rotate camera frame 90° clockwise
RESIZE = (112, 112)
MEAN = (0.43216, 0.39467, 0.37645)
STD = (0.22803, 0.22145, 0.21699)
# -----------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model, _ = build_model(num_classes=10)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

class_names = {
    0: "Doing other things",
    1: "No gesture",
    2: "Sliding Two Fingers Left",
    3: "Sliding Two Fingers Right",
    4: "Stop Sign",
    5: "Swiping Down",
    6: "Swiping Left",
    7: "Swiping Right",
    8: "Zooming Out With Two Fingers",
    9: "Zooming In With Two Fingers"
}
print(f"Classes: {class_names}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")
if fps <= 0:
    fps = 36  # fallback

buffer_maxlen = int(fps * BUFFER_SECONDS)
frame_buffer = deque(maxlen=buffer_maxlen)

print(f"Camera FPS: {fps:.0f}, buffer: {buffer_maxlen} frames ({BUFFER_SECONDS}s)")
print(f"Inference: up to {INFERENCE_HZ} Hz")
print(f"Device: {device}")
# print("Press 'q' to quit.")

last_inference_time = 0
inference_interval = 1.0 / INFERENCE_HZ
pred_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if ROTATE_CW90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # BGR -> RGB, resize for model
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, RESIZE)
    frame_buffer.append(resized)

    # run inference at controlled rate, once buffer is full
    now = time.time()
    if len(frame_buffer) == buffer_maxlen and now - last_inference_time >= inference_interval:
        last_inference_time = now

        # uniformly sample SAMPLE_FRAMES frames from the buffer
        indices = torch.linspace(0, len(frame_buffer) - 1, SAMPLE_FRAMES).long()
        sampled = [frame_buffer[i.item()] for i in indices]

        # stack: (T, H, W, C) -> tensor (T, C, H, W) for normalize
        clip = np.stack(sampled, axis=0)  # (T, H, W, C)
        clip = torch.from_numpy(clip).float() / 255.0
        clip = clip.permute(0, 3, 1, 2)  # (T, C, H, W)

        # normalize (matches dataloader)
        mean_t = clip.new_tensor(MEAN).view(1, 3, 1, 1)
        std_t = clip.new_tensor(STD).view(1, 3, 1, 1)
        clip = (clip - mean_t) / std_t

        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
        clip = clip.unsqueeze(0).to(device)  # (1, C, T, H, W)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = model(clip)
            pred = output.argmax(dim=1).item()

        pred_label = class_names[pred]


    # draw prediction on frame (persists between inferences)
    display = frame.copy()
    if pred_label:
        cv2.putText(display, pred_label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        cv2.putText(display, "buffering...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Gesture Recognition", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
