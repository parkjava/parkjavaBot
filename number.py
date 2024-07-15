import cv2
import numpy as np
from urllib.request import urlopen
import threading

# 스트림 URL 설정
url = "http://192.168.0.211:8080/stream?topic=/csi_cam_0/image_raw"

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
        cv2.imshow("stream", frame)

    # 'q' 키를 누르면 루프 종료
    key = cv2.waitKey(1)
    if key == ord('q'):
        video_stream.stop()
        break

# 모든 창 닫기
cv2.destroyAllWindows()
# 업데이트 스레드 종료 대기
thread.join()
