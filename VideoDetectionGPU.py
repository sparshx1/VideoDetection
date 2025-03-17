from ultralytics import YOLO
import cv2
import cvzone
import math
import torch  # Import torch for GPU usage

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Should print 'cuda' if GPU is available

# Open Video File or Webcam
#cap = cv2.VideoCapture("video/party.mp4")  # Change to 0 for webcam
cap=cv2.VideoCapture(0) #Webcam
cap.set(3, 1280)
cap.set(4, 720)

# Load YOLO Model on GPU
model = YOLO('YoloWeights/yolov8l.pt').to(device)  

className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
             "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
             "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
             "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
             "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
             "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
             "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
             "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    if not success:
        break  # Exit if video ends

    # Run YOLO detection on GPU
    result = model(img, stream=True)  

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{className[cls]} {conf}', (x1, y1), scale=1, thickness=1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()