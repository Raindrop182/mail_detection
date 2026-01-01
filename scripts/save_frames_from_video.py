import cv2 as cv
import os
import torch
from pathlib import Path
import glob

from read_num import read_num, segment_digits
from tinyvgg import TinyVGG

def save_frames_from_video(video_path):
    """Extracts and saves selected frames from a video, labeling each frame with a
    timestamp inferred by a pretrained digit-recognition model.
    """
    model = TinyVGG(input_shape=1,hidden_units=10,output_shape=10)
    model.load_state_dict(torch.load("finetuned_number_recognition_model.pth"))
     
    output_dir = Path(__file__).parent.parent / "frames"
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    frame_num = 0
    
    while frame_num < total_frames:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        date = f"{output_dir}/{video_path[:-4]}"
        
        
        # digits=segment_digits(frame)
        # digits[0].shape
        # cv.imwrite(f"data/finetuning_data/train/{frame_num}1.png",cv.resize(digits[2],(28,28)))
        # print(f"{frame_num}1")
        
        time = read_num(frame,model)
        
        save_path=f"{date}-{time}.jpg"
        frame=frame[0:600,0:700] #only save region around mailbox
        cv.imwrite(save_path,frame)
    
        frame_num += 100
        
    cap.release()
    
