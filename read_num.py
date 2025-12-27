import cv2 as cv
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\raine\miniconda3\envs\cv_env\Library\bin\tesseract.exe"

def read_num(image: np.ndarray):
    image = image[0:50,200:380]
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite("yay.png",image)
    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist=0123456789:"
    )
    return pytesseract.image_to_string(image, lang='eng', config=config)

# image = cv.imread("object_frame.jpg")
# read_num(image)
# image = image[0:50,200:380,0]
# cv.imwrite("yay.png",image)
# # print(read_num(image))