import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import time
from urllib.request import urlopen
from firebase_admin import credentials, initialize_app, storage
import threading
import re

plt.ioff()  # 상호작용 모드 끄기

# 스트림 URL 설정
url = "http://192.168.137.6:8080/stream?topic=/csi_cam_1/image_raw"

cred = credentials.Certificate("./parkjavastorage-firebase-adminsdk-rttma-90b24d3ed3.json")
initialize_app(cred, {'storageBucket': 'parkjavastorage.appspot.com'})

class VideoStream:
    def __init__(self, url):
        # URL로부터 스트림을 열고 초기 설정
        self.stream = urlopen(url)
        self.buffer = b''  # 버퍼 초기화
        self.frame = None  # 현재 프레임
        self.running = True  # 스트림이 실행 중인지 여부
        self.lock = threading.Lock()  # 스레드 동기화를 위한 락

    def update(self):
        while self.running:
            # 스트림에서 데이터를 읽어 버퍼에 추가
            self.buffer += self.stream.read(4096)
            
            # JPEG 이미지의 시작과 끝을 찾음
            head = self.buffer.find(b'\xff\xd8')
            end = self.buffer.find(b'\xff\xd9')

            if head > -1 and end > -1:
                # JPEG 이미지를 추출하여 디코딩
                jpg = self.buffer[head:end+2]
                self.buffer = self.buffer[end+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                
                # 락을 사용하여 프레임을 안전하게 업데이트
                with self.lock:
                    self.frame = img

    def get_frame(self):
        # 락을 사용하여 현재 프레임을 반환
        with self.lock:
            return self.frame

    def stop(self):
        # 스트림 업데이트를 중지
        self.running = False

# 비디오 스트림 객체 생성
video_stream = VideoStream(url)

# 업데이트 스레드 시작
thread = threading.Thread(target=video_stream.update)
thread.start()

while True:
    # 현재 프레임을 가져옴
    frame = video_stream.get_frame()

    if frame is not None:
        # 프레임을 화면에 표시

        height, width, channel = frame.shape

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

        imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
        imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

        imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)

        gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

        #노이즈

        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

        img_thresh = cv2.adaptiveThreshold(
            img_blurred, 
            maxValue=255.0, 
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            thresholdType=cv2.THRESH_BINARY_INV, 
            blockSize=19, 
            C=9
        )
            #윤곽선
        contours, _ = cv2.findContours(
            img_thresh, 
            mode=cv2.RETR_LIST, 
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

        #사각형 범위찾기
    

        contours_dict = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
            
            # insert to dict
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })

        # 번호판 찾기
        MIN_AREA = 80
        MIN_WIDTH, MIN_HEIGHT = 2, 8
        MIN_RATIO, MAX_RATIO = 0.25, 1.0

        possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']
            
            if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)
                
        # visualize possible contours
        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        for d in possible_contours:
        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        contours_dict = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2)
            })
        # # plt.figure(figsize=(12,10))
        # # plt.imshow(temp_result, cmap='gray')

        MIN_AREA = 100
        MIN_WIDTH, MIN_HEIGHT = 3, 10
        MIN_RATIO, MAX_RATIO = 0.25, 1.0

        possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']

            if area > MIN_AREA \
                    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                    and MIN_RATIO < ratio < MAX_RATIO:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)

        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        for d in possible_contours:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                        thickness=2)

        # # plt.figure(figsize=(12, 10))
        # # plt.imshow(temp_result, cmap='gray')
        # # plt.show()

        MAX_DIAG_MULTIPLYER = 5
        MAX_ANGLE_DIFF = 12.0
        MAX_AREA_DIFF = 0.5
        MAX_WIDTH_DIFF = 0.8
        MAX_HEIGHT_DIFF = 0.2
        MIN_N_MATCHED = 6

        def find_chars(contour_list):
            matched_result_idx = []

            for d1 in contour_list:
                matched_contours_idx = []
                for d2 in contour_list:
                    if d1['idx'] == d2['idx']:
                        continue

                    dx = abs(d1['cx'] - d2['cx'])
                    dy = abs(d1['cy'] - d2['cy'])

                    diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                    distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                    if dx == 0:
                        angle_diff = 90
                    else:
                        angle_diff = np.degrees(np.arctan(dy / dx))
                    area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                    width_diff = abs(d1['w'] - d2['w']) / d1['w']
                    height_diff = abs(d1['h'] - d2['h']) / d1['h']

                    if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                        matched_contours_idx.append(d2['idx'])

                matched_contours_idx.append(d1['idx'])

                if len(matched_contours_idx) < MIN_N_MATCHED:
                    continue

                matched_result_idx.append(matched_contours_idx)

                unmatched_contour_idx = []
                for d4 in contour_list:
                    if d4['idx'] not in matched_contours_idx:
                        unmatched_contour_idx.append(d4['idx'])

                unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

                recursive_contour_list = find_chars(unmatched_contour)

                for idx in recursive_contour_list:
                    matched_result_idx.append(idx)

                break

            return matched_result_idx


        result_idx = find_chars(possible_contours)

        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(possible_contours, idx_list))

        temp_result = np.zeros((height, width, channel), dtype=np.uint8)

        for r in matched_result:
            for d in r:
                cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                            thickness=1)

        PLATE_WIDTH_PADDING = 1.0  # 1.3
        PLATE_HEIGHT_PADDING = 1.2  # 1.5
        MIN_PLATE_RATIO = 5
        MAX_PLATE_RATIO = 10

        plate_imgs = []
        plate_infos = []
        img_cropped =frame.shape
        img_rotated =frame.shape

        for i, matched_chars in enumerate(matched_result):
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )

            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

            rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=0.95)

    

            img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

            # print(img_rotated)

            img_cropped = cv2.getRectSubPix(
                img_rotated,
                patchSize=(int(plate_width), int(plate_height)),
                center=(int(plate_cx), int(plate_cy))
            )

            # if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            #     0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            #     continue

            plate_imgs.append(img_cropped)
            
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })
            
        
        longest_idx, longest_text = -1, 0
        plate_chars = []

        
        plate_img = frame.shape

        for i, plate_img in enumerate(plate_imgs):

            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
            _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # find contours again (same as above)
            contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
            plate_max_x, plate_max_y = 0, 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                area = w * h
                ratio = w / h

                if area > MIN_AREA \
                        and w > MIN_WIDTH and h > MIN_HEIGHT \
                        and MIN_RATIO < ratio < MAX_RATIO:
                    if x < plate_min_x:
                        plate_min_x = x
                    if y < plate_min_y:
                        plate_min_y = y
                    if x + w > plate_max_x:
                        plate_max_x = x + w
                    if y + h > plate_max_y:
                        plate_max_y = y + h

        for plate_info in plate_infos:
            
            x, y, w, h = plate_info['x'], plate_info['y'], plate_info['w'], plate_info['h']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h),(163, 100, 231), 3)

            img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        #tesseract를 사용한 번호판 문자열 검출
            img_result = cv2.GaussianBlur(img_result, ksize=(5, 5), sigmaX=0)
            _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
            
            pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.4.1/share/tessdata' #TesseractNotFoundError: tesseract is not installed or it's not in your PATH. See README file for more information. 에러 메시지 나올 경우
            chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 6 --oem 3')
            
            
            text_list = [ord('가'), ord('나'), ord('다'), ord('라'), ord('마'),ord('거'), ord('너'), ord('더'), ord('러'), 
                         ord('머'), ord('버'), ord('서'), ord('어'), ord('저'), ord('고'), ord('노'), ord('도'), ord('로'), 
                         ord('모'), ord('보'), ord('소'), ord('오'),ord('조'), ord('구'), ord('누'), ord('두'), ord('루'), 
                         ord('무'), '1','2','3','4','5','6','7','8','9','0',ord('아'), ord('바'), ord('사'), ord('자'), 
                         ord('하'), ord('허'), ord('호'), ord('배'), ord('부'), ord('수'), ord('우'), ord('주'),ord('육'),ord('해'),ord('공')]
            result_chars = ''
            
            has_digit = False

            numberPattern = re.compile(r'^\d{2,3}[가-힣]\d{4}$')
            
            

            
            for c in chars:
                if ord(c) in text_list or c.isdigit():
                    if c.isdigit():
                        has_digit = True
                    result_chars += c

            if numberPattern.match(result_chars): # 넘버 패턴 정규식에 번호판 문자열이 맞는지?                
                if len(result_chars) == 7 or len(result_chars) == 8: # 번호판 문자열이 7 or 8글자인지?
                    plate_chars.append(result_chars) # 번호판 문자열 리스트에 추가
                    bucket = storage.bucket()  # 파이어베이스 스토리지 불러오기
                    blobs = bucket.list_blobs( ) # 파이어베이스 스토리지의 리스트 
                    blobList = [blob.name for blob in blobs] # 스토리지의 사진 이름 반복문으로 blobList에 넣기
                    if result_chars in plate_chars[0]: # 검출된 번호가 번호판 문자열 첫번째에 있다면
                        if result_chars in blobList: # 파이어베이스 스토리지의 저장되어있는 사진리스트에 번호판문자열과 같은 이름이 있을 때
                            print(result_chars, '이미 추가되어 있는 데이터임') 
                        else:
                            try:
                                fileName = result_chars + ".jpg" # 파일이름 = 번호판.jpg
                                saveImg = cv2.imwrite(fileName, img_result) # 이미지 저장한다.
                                
                                if not saveImg:
                                    print(f"Error saving image {fileName}") # saveImg 없으면 
                                
                                else:
                                    saveName = result_chars # 파이어베이스에 저장할 이름 = 번호판문자열
                                    blob = bucket.blob(saveName) 
                                    blob.upload_from_filename(fileName)
                                    blob.make_public()
                                    print(saveName, blob.public_url)
                            except Exception as e:
                                print(f"An error occurred: {e}")
                else:
                    pass

            

            if has_digit and len(result_chars) > longest_text:
                longest_idx = i

            # plt.subplot(len(plate_imgs), 1, i+1)
            plt.imshow(img_result, cmap='gray')
            info = plate_infos[longest_idx]
            # chars = plate_chars[longest_idx]

            img_out = frame.copy()

        # info = plate_infos[longest_idx]

        # cv2.rectangle(img_cropped, pt1=int((plate_infos['x'], plate_infos['y']), pt2=(plate_infos['x']+plate_infos['w'], plate_infos['y']+plate_infos['h'])), color=(255,0,0), thickness=2)
        cv2.imshow("FIND NUMBER", img_cropped)
        
        cv2.imshow("FRAME", frame)

        # 'q' 키를 누르면 루프 종료
        key = cv2.waitKey(1)
        if key == ord('q'):
            video_stream.stop()
            break
    
 # 모든 창 닫기
cv2.destroyAllWindows()
# 업데이트 스레드 종료 대기
thread.join()

# 한양 헤드라인 장평 90%로 하면 유사합니다

# 0: 자동 페이지 세그멘테이션, 오존 표식을 사용
# 1: 자동 페이지 세그멘테이션, 오존 표식을 사용
# 2: 자동 페이지 세그멘테이션, 최대 2개의 열을 지원
# 3: 자동 페이지 세그멘테이션, 표식 없음
# 4: 고정 너비의 단일 열 텍스트
# 5: 단일 열 텍스트
# 6: 단일 균형 텍스트 블록
# 7: 수직 및 수평의 모든 블록을 탐색
# 8: 단일 열의 모든 텍스트를 인식 (위치 정보를 저장하지 않음)
# 9: 단일 열 텍스트, 페이지 내 위치를 포함한 인식
# 10: 텍스트 블록으로 페이지를 인식하고, 전역 레이아웃 분석
# 11: 리스트 및 일련의 항목
# 12: 텍스트 영역으로 페이지를 분석
# 13: 페이지에서 각 블록을 추출
