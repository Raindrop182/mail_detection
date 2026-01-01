from ultralytics import YOLO
import cv2 as cv
import os

model = YOLO("yolov5s.pt")

video_name = "20101103.mp4"
output_folder = "candidates"
os.makedirs(output_folder, exist_ok=True)

def create_data():
    """
    Extract frames containing trucks from a video and save them as images.
    """
    cap = cv.VideoCapture(video_name)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    frame_num = 0
    
    # cough=False
    while frame_num < total_frames:
        print(frame_num)
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        
        results=model(frame)
        
        # results[0].boxes contains detected bounding boxes
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            
            # YOLOv5 COCO classes: 7=truck, 5=bus, 2=car
            if cls_id == 7:  # only keep trucks
                cv.imwrite(os.path.join(output_folder, str(frame_num)+".jpg"), frame)
                frame_num-=100
                frame_num+=1
        frame_num += 100
        
    cap.release()


   