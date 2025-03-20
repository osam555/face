import cv2
import numpy as np
import os

# 이미지 디렉토리 생성
os.makedirs('images/eyebrow_types', exist_ok=True)

# 일자형 눈썹
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
cv2.line(img, (100, 150), (300, 150), (0, 0, 0), 3)
cv2.imwrite('images/eyebrow_types/straight.jpg', img)

# 아치형 눈썹
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
pts = np.array([[100, 150], [200, 100], [300, 150]], np.int32)
cv2.polylines(img, [pts], False, (0, 0, 0), 3)
cv2.imwrite('images/eyebrow_types/arched.jpg', img)

# 둥근형 눈썹
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
pts = np.array([[100, 150], [150, 120], [200, 100], [250, 120], [300, 150]], np.int32)
cv2.polylines(img, [pts], False, (0, 0, 0), 3)
cv2.imwrite('images/eyebrow_types/rounded.jpg', img)

# 각진형 눈썹
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
pts = np.array([[100, 150], [200, 100], [300, 150]], np.int32)
cv2.polylines(img, [pts], False, (0, 0, 0), 3)
cv2.line(img, (200, 100), (200, 150), (0, 0, 0), 2)
cv2.imwrite('images/eyebrow_types/angular.jpg', img)

# 기본형 눈썹
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
pts = np.array([[100, 150], [150, 130], [200, 120], [250, 130], [300, 150]], np.int32)
cv2.polylines(img, [pts], False, (0, 0, 0), 3)
cv2.imwrite('images/eyebrow_types/natural.jpg', img)

print("눈썹 유형별 이미지가 생성되었습니다.") 