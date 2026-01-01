import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from tinyvgg import TinyVGG
from model_train import train

def train_readnum_model():
    """
    Train a TinyVGG model on the MNIST dataset for handwritten digit recognition.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
        ])
    
    train_data = datasets.MNIST(
        root="data", 
        train=True, 
        download=True, 
        transform=transform,
        target_transform=None 
    )
    
    test_data = datasets.MNIST(
        root="data",
        train=False, 
        download=True,
        transform=transform
    )
    
    BATCH_SIZE = 32
    train_dataloader = DataLoader(train_data, 
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    test_dataloader = DataLoader(test_data,
        batch_size=BATCH_SIZE,
        shuffle=False 
    )
    
    model=TinyVGG(input_shape=1,hidden_units=10,output_shape=len(train_data.classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    
    train(model,train_dataloader,test_dataloader,loss_fn,optimizer,3,device)
    
    output_path=Path(__file__).parent.parent / "models" / "number_recognition_model.pth"
    torch.save(obj=model.state_dict(),f=output_path)

def finetune():
    """
    Fine-tune a pre-trained TinyVGG model on a custom dataset for digit recognition.

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path=Path(__file__).parent.parent / "data" / "finetuning_data"
    train_dir = data_path+"/train"
    test_dir = data_path+"/test"
    model_save_path=Path(__file__).parent.parent / "models"
    
    data_transform=transforms.Compose([transforms.ToTensor(),
                                       transforms.Grayscale(num_output_channels=1),  # ensures 1 channel
                                      transforms.Normalize((0.5,), (0.5,))
                                   ])
    
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)
    
    BATCH_SIZE = 32
    train_dataloader = DataLoader(train_data, 
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    test_dataloader = DataLoader(test_data,
        batch_size=BATCH_SIZE,
        shuffle=False 
    )
    
    model=TinyVGG(input_shape=1,hidden_units=10,output_shape=10)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)    
    
    model.load_state_dict(torch.load(model_save_path / "number_recognition_model.pth"))
    
    train(model,train_dataloader,test_dataloader,loss_fn,optimizer,10,device)
    
    torch.save(obj=model.state_dict(),f=model_save_path / "finetuned_number_recognition_model.pth")

def segment_digits(frame):
    """
    Extract individual digit images from a video frame.

    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    thresh_val=240 if gray.mean()>100 else 220
    
    _, bw = cv.threshold(gray, thresh_val, 255, cv.THRESH_BINARY_INV)
    
    digits=[]
    digits.append(bw[5:50,260:290])
    digits.append(bw[5:50,290:315])
    digits.append(bw[5:50,335:360])
    digits.append(bw[5:50,360:385])
    digits.append(bw[5:50,405:430])
    digits.append(bw[5:50,430:455])
    
    digits=[255-digit for digit in digits]

    return digits

def read_num(image: np.ndarray, model):
    """
    Recognize a multi-digit number from an image using a trained neural network model.

    """
    digits=segment_digits(image)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform=transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                   ])
    
    digits=[transform(cv.resize(digit,(28,28))) for digit in digits]
    digits = torch.stack(digits).to(device)
    # for digit in digits:
    #     plt.imshow(digit.squeeze(), cmap='gray')
    #     plt.show()
    
    model.eval()
    with torch.inference_mode():
        outputs=model(digits)
        
    nums=[]
    for output in outputs:
        num=torch.argmax(torch.softmax(output,dim=0),dim=0).item()
        nums.append(num)

    return f"{nums[0]}{nums[1]}-{nums[2]}{nums[3]}-{nums[4]}{nums[5]}"