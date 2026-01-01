import torch
from torchvision import datasets, transforms, models
from torch import nn

import numpy as np
import cv2 as cv
from tqdm import tqdm
import glob
import pandas as pd
import os
from pathlib import Path

from model_train import train

from tinyvgg import TinyVGG
from read_num import read_num


def train_mail_model():
    """    
    Train a ResNet18 model to classify images as containing mail or not. 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir=Path(__file__).parent.parent / "data" / "mail_data" / "train"
    test_dir=Path(__file__).parent.parent / "data" / "mail_data" / "test"
    model_save_path=Path(__file__).parent.parent / "models"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.Resize((224, 224))
    ])
    
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds   = datasets.ImageFolder(test_dir, transform=transform)
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds, batch_size=32)
    
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2) #adjust the final layer to have 2 outputs, representing mail and no mail
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train(model, train_dl, val_dl, loss_fn, optimizer, 5,device)
    torch.save(obj=model.state_dict(),f=model_save_path / "mail_model.pth")

def mail_in_frame(image: np.ndarray, model):
    """
    Determine whether a given image contains mail using a trained classification model. 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         ),
         transforms.Resize((224, 224))
     ])
    
    image=transform(image).unsqueeze(dim=0)
    
    model.eval()
    with torch.inference_mode():
        outputs=model(image)
        
    result=torch.argmax(outputs,dim=1).item()
    return result==0

def identify_mail_in_video(video_path: str, model,save_image_path):
    """
    Parse video footage to identify when the mail arrives.
    Saves a snapshot of the mail truck and returns the time stamp of when the mail arrived
    If mail doesn't arrive in the video, return None
    """
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")

    frame_num=100
    while frame_num<total_frames:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = frame[0:600,0:700] #only look at subsection of frame with mailbox
        
        model.eval()
        with torch.inference_mode():
            is_mail = mail_in_frame(frame,model)
            
        if is_mail:
            cap.release()
            pbar.close()
            
            date=os.path.splitext(os.path.basename(video_path))[0]
            cv.imwrite(save_image_path+date+".png",frame)
            
            model = TinyVGG(input_shape=1,hidden_units=10,output_shape=10)
            model.load_state_dict(torch.load("finetuned_number_recognition_model.pth"))
            return read_num(frame,model)
        
        frame_num+=100
        pbar.update(100)
        
    cap.release()
    pbar.close()
    return None
 
    
# # train_mail_model()
# model = models.resnet18(weights="IMAGENET1K_V1")
# model.fc = nn.Linear(model.fc.in_features, 2)
# model.load_state_dict(torch.load("mail_model.pth"))

# save_image_path="mail_images/"
# os.makedirs(save_image_path, exist_ok=True)

# folder_path = r"//nas/securityVideo/New folder/C"
# files = glob.glob(f"{folder_path}/*.mp4")
# files.sort()
# print(files)
# data={"date":[],"time":[]}
# for video_name in files:
#     time=identify_mail_in_video(video_name,model,save_image_path)
#     date=os.path.splitext(os.path.basename(video_name))[0]
#     print(date)
#     print(time)
#     data["date"].append(date)
#     data["time"].append(time) 
#     df = pd.DataFrame(data)
#     df.to_excel("output2.xlsx", index=False)


