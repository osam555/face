import cv2
import numpy as np
import os

# 이미지 디렉토리 생성
os.makedirs('images/eyebrow_types', exist_ok=True)

def draw_smooth_curve(img, points, color=(0, 0, 0), thickness=3):
    # 부드러운 곡선을 그리기 위해 points를 이용해 스플라인 곡선 생성
    points = np.array(points, dtype=np.float32)
    # 더 많은 중간점 생성
    t = np.linspace(0, 1, 100)
    points_interp = []
    for i in range(len(points) - 3):
        for t_i in t:
            x = (1-t_i)**3 * points[i][0] + 3*t_i*(1-t_i)**2 * points[i+1][0] + 3*t_i**2*(1-t_i) * points[i+2][0] + t_i**3 * points[i+3][0]
            y = (1-t_i)**3 * points[i][1] + 3*t_i*(1-t_i)**2 * points[i+1][1] + 3*t_i**2*(1-t_i) * points[i+2][1] + t_i**3 * points[i+3][1]
            points_interp.append([int(x), int(y)])
    points_interp = np.array(points_interp, dtype=np.int32)
    
    # 부드러운 곡선 그리기
    for i in range(len(points_interp)-1):
        cv2.line(img, tuple(points_interp[i]), tuple(points_interp[i+1]), color, thickness)
    return img

# 일자형 눈썹
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
cv2.line(img, (100, 150), (300, 150), (0, 0, 0), 3)
cv2.imwrite('images/eyebrow_types/straight.jpg', img)

# 아치형 눈썹 (더 부드러운 곡선)
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
points = np.array([
    [100, 150],  # 시작점
    [150, 140],  # 컨트롤 포인트 1
    [200, 120],  # 정점
    [250, 140],  # 컨트롤 포인트 2
    [300, 150]   # 끝점
], dtype=np.int32)
img = draw_smooth_curve(img, points)
cv2.imwrite('images/eyebrow_types/arched.jpg', img)

# 둥근형 눈썹 (더 부드러운 곡선)
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
points = np.array([
    [100, 150],  # 시작점
    [150, 135],  # 컨트롤 포인트 1
    [200, 130],  # 정점
    [250, 135],  # 컨트롤 포인트 2
    [300, 150]   # 끝점
], dtype=np.int32)
img = draw_smooth_curve(img, points)
cv2.imwrite('images/eyebrow_types/rounded.jpg', img)

# 각진형 눈썹
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
pts = np.array([[100, 150], [200, 100], [300, 150]], np.int32)
cv2.polylines(img, [pts], False, (0, 0, 0), 3)
cv2.line(img, (200, 100), (200, 150), (0, 0, 0), 2)
cv2.imwrite('images/eyebrow_types/angular.jpg', img)

# 기본형 눈썹
img = np.ones((300, 400, 3), dtype=np.uint8) * 255
points = np.array([
    [100, 150],  # 시작점
    [150, 140],  # 컨트롤 포인트 1
    [200, 135],  # 정점
    [250, 140],  # 컨트롤 포인트 2
    [300, 150]   # 끝점
], dtype=np.int32)
img = draw_smooth_curve(img, points)
cv2.imwrite('images/eyebrow_types/natural.jpg', img)

print("눈썹 유형별 이미지가 생성되었습니다.") 