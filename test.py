# from PIL import Image
# import numpy as np
# import pytesseract

# filename = 'ss.png'
# config = ('-l kor+eng')
# # config = ('-l kor+eng --oem 3 --psm 11')
# img1 = np.array(Image.open(filename))
# text = pytesseract.image_to_string(img1, config=config)
# print(text)

import tensorflow as tf
import cv2
import sys
import time
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import platform
import pytesseract as pt

def cv2_draw_label(image, text, x, y):
    x, y = int(x), int(y)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = 'malgun.ttf'
    if platform.system() == 'Darwin': #맥
        font = 'AppleGothic.ttf'
    try:
        imageFont = ImageFont.truetype(font, 28)
    except:
        imageFont = ImageFont.load_default()
    draw.text((x, y), text, font=imageFont, fill=(255, 255, 255))
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    sys.exit()

##########모델 로드

pt.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.4.1/share/tessdata' #TesseractNotFoundError: tesseract is not installed or it's not in your PATH. See README file for more information. 에러 메시지 나올 경우

##########모델 예측

start = False
result = ""

while True:
    ret, image = video_capture.read()
    #print(type(image)) #<class 'numpy.ndarray'>
    #print(image.shape) #(720, 1280, 3)

    if not ret:
        break

    if start:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = pt.image_to_string(rgb_image, lang='kor')
        # result = pt.image_to_string(image, config="-l kor")
        print(result)

        start = False

    image = cv2_draw_label(image, result, 30, 30)
    cv2.imshow('image', image)

    key = cv2.waitKey(1) #키 입력이 있을때 까지 1밀리 세컨드 동안 대기
    if key == ord('s'):
        start = True

    key = cv2.waitKey(1) #키 입력이 있을때 까지 1밀리 세컨드 동안 대기
    if key == ord('q'): 
        break

video_capture.release()
cv2.destroyAllWindows()
if platform.system() == 'Darwin': #맥
    cv2.waitKey(1)