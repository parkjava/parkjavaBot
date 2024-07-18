import os
from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, font_path, font_size, text_color=(0, 0, 0), bg_color=(0, 0, 0), image_path='text_image.png', padding=10):
    # 폰트 설정
    font = ImageFont.truetype(font_path, font_size)

    # 텍스트 크기 계산
    dummy_image = Image.new('RGBA', (1, 1), bg_color)
    draw = ImageDraw.Draw(dummy_image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # 여백을 포함한 이미지 크기 계산
    image_width = text_width + 2 * padding
    image_height = text_height + 2 * padding

    # 이미지 생성 (RGBA 모드)
    image = Image.new('RGBA', (image_width, image_height), bg_color)
    draw = ImageDraw.Draw(image)

    # 텍스트를 이미지에 그리기
    draw.text((padding, padding), text, fill=text_color, font=font)

    # 이미지 저장
    image.save(image_path)
    print(f'Text image saved as {image_path}')

# 예제 사용법

text_list = ['가','나','다','라','마','거','너','더','러','머','버','서','어','저','고','노','도','로','모','보','소','오','조','구','누','두','루','무','1','2','3','4','5','6','7','8','9','0','아','바','사','자','하','허','호','배','부','수','우','주']

print(len(text_list))
# font_path_list =['/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/parkjavaBot/youngwon/ttf/HanyangHeadLine.ttf',
#                  '/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/parkjavaBot/youngwon/ttf/NanumBarunGothicYetHangul.ttf',
#                  '/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/parkjavaBot/youngwon/ttf/NanumGothic.ttf',
#                  '/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/parkjavaBot/youngwon/ttf/NanumGothicD2Coding.ttf',
#                  '/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/parkjavaBot/youngwon/ttf/NanumGothicEco.ttf',
#                  '/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/parkjavaBot/youngwon/ttf/NanumSquareNeo-bRg.ttf',
#                  '/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/parkjavaBot/youngwon/ttf/NanumSquareR.ttf',
#                  '/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/parkjavaBot/youngwon/ttf/NanumSquareRoundR.ttf']


# font_path = font_path_list[5]  # 한글 폰트 파일 경로

# font_size = 20
# text_color = (0, 0, 0, 255)  # 검정색 (불투명)
# bg_color = (0, 0, 0, 0)  # 투명색
# padding = 10  # 여백 크기

# green = (153, 255, 153)
# sky =(153, 204, 255)
# yellow = (255, 212, 0)
# white = ( 255, 255, 255)

# black =(0, 0, 0)



# # 최상위 폴더 경로
# base_folder = '/Users/youngwonchoi/Desktop/20240307/01.FinalProject/parkJavaBot/dataset'

# # 최상위 폴더 생성
# if not os.path.exists(base_folder):
#     os.makedirs(base_folder)

# # 각 문자 및 숫자별 폴더와 이미지 생성 및 저장
# for text in text_list:
#     folder_path = os.path.join(base_folder, text)
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     image_path = os.path.join(folder_path, 'NanumSquareRoundR' + 'black' + text + '.png')
#     text_to_image(text, font_path, font_size, white, black, image_path, padding)
