import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
import time

# 전역 변수로 MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# 이미지 전처리 함수
def preprocess_image(image):
    try:
        # 이미지 크기 조정 (너무 큰 이미지 처리)
        max_dimension = 800
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        return image
    except Exception as e:
        st.error(f"이미지 전처리 중 오류가 발생했습니다: {str(e)}")
        return None

# 앱 제목 및 설명
st.set_page_config(
    page_title="얼굴 인상학 분석 및 눈썹 메이크업 추천",
    page_icon="👁️",
    layout="wide"
)

st.title('👁️ 얼굴 인상학 분석 및 눈썹 메이크업 추천')
st.write('사진을 업로드하면 눈썹을 분석하고 인상학적 해석과 메이크업 추천을 제공합니다.')

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    confidence_threshold = st.slider(
        "얼굴 인식 신뢰도 임계값",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )
    face_mesh.min_detection_confidence = confidence_threshold

# 얼굴 랜드마크 감지 함수
def detect_facial_landmarks(image):
    try:
        # RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 랜드마크 감지
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        return results.multi_face_landmarks[0]
    except Exception as e:
        st.error(f"얼굴 인식 중 오류가 발생했습니다: {str(e)}")
        return None

# 눈썹 랜드마크 추출 함수
def extract_eyebrow_landmarks(landmarks):
    # MediaPipe 눈썹 랜드마크 인덱스
    # 왼쪽 눈썹: 336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    # 오른쪽 눈썹: 70, 63, 105, 66, 107, 55, 65, 52, 53, 46
    
    left_eyebrow_indices = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    right_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    
    left_eyebrow = []
    right_eyebrow = []
    
    for idx in left_eyebrow_indices:
        left_eyebrow.append([landmarks.landmark[idx].x, landmarks.landmark[idx].y])
    
    for idx in right_eyebrow_indices:
        right_eyebrow.append([landmarks.landmark[idx].x, landmarks.landmark[idx].y])
    
    return np.array(left_eyebrow), np.array(right_eyebrow)

# 눈썹 모양 분석 함수
def analyze_eyebrow_shape(left_eyebrow, right_eyebrow, image_shape):
    # 눈썹 특성 계산
    
    # 1. 눈썹 기울기
    def calculate_slope(eyebrow):
        x1, y1 = eyebrow[0]
        x2, y2 = eyebrow[-1]
        if x2 == x1:
            return 100  # 거의 수직
        return (y2 - y1) / (x2 - x1)
    
    left_slope = calculate_slope(left_eyebrow)
    right_slope = calculate_slope(right_eyebrow)
    
    # 2. 눈썹 두께
    def calculate_thickness(eyebrow, image_shape):
        eyebrow_pixels = np.array(eyebrow) * np.array([image_shape[1], image_shape[0]])
        top_points = eyebrow_pixels[0:5]
        bottom_points = eyebrow_pixels[5:]
        distances = []
        for i in range(min(len(top_points), len(bottom_points))):
            dist = np.linalg.norm(top_points[i] - bottom_points[i])
            distances.append(dist)
        return np.mean(distances)
    
    left_thickness = calculate_thickness(left_eyebrow, image_shape)
    right_thickness = calculate_thickness(right_eyebrow, image_shape)
    avg_thickness = (left_thickness + right_thickness) / 2
    
    # 3. 눈썹 길이
    def calculate_length(eyebrow, image_shape):
        eyebrow_pixels = np.array(eyebrow) * np.array([image_shape[1], image_shape[0]])
        length = 0
        for i in range(len(eyebrow_pixels) - 1):
            length += np.linalg.norm(eyebrow_pixels[i+1] - eyebrow_pixels[i])
        return length
    
    left_length = calculate_length(left_eyebrow, image_shape)
    right_length = calculate_length(right_eyebrow, image_shape)
    avg_length = (left_length + right_length) / 2
    
    # 4. 눈썹 곡률
    def calculate_curvature(eyebrow):
        x_coords = [p[0] for p in eyebrow]
        y_coords = [p[1] for p in eyebrow]
        x_mid = np.mean(x_coords)
        y_mid = np.mean(y_coords)
        
        # 중앙 지점에서 가장 멀리 떨어진 지점의 거리
        max_dist = 0
        for x, y in zip(x_coords, y_coords):
            dist = np.sqrt((x - x_mid)**2 + (y - y_mid)**2)
            if dist > max_dist:
                max_dist = dist
        
        # 곡률 근사값
        curvature = max_dist / (max(x_coords) - min(x_coords))
        return curvature
    
    left_curvature = calculate_curvature(left_eyebrow)
    right_curvature = calculate_curvature(right_eyebrow)
    avg_curvature = (left_curvature + right_curvature) / 2
    
    # 눈썹 형태 결정
    eyebrow_shape = "기본형"
    if avg_curvature > 0.4:
        if left_slope < -0.1 and right_slope > 0.1:
            eyebrow_shape = "아치형"
        else:
            eyebrow_shape = "둥근형"
    elif avg_curvature < 0.2:
        eyebrow_shape = "일자형"
    else:
        if left_slope < -0.2 and right_slope > 0.2:
            eyebrow_shape = "각진형"
        else:
            eyebrow_shape = "기본형"
    
    # 눈썹 두께 결정
    thickness_category = "중간"
    if avg_thickness < 5:
        thickness_category = "얇음"
    elif avg_thickness > 10:
        thickness_category = "두꺼움"
    
    # 눈썹 길이 결정
    length_category = "중간"
    if avg_length < 50:
        length_category = "짧음"
    elif avg_length > 100:
        length_category = "긺"
    
    return {
        "shape": eyebrow_shape,
        "thickness": thickness_category,
        "length": length_category,
        "curvature": avg_curvature,
        "left_slope": left_slope,
        "right_slope": right_slope
    }

# 인상학적 특성 분석 함수
def physiognomy_analysis(eyebrow_features):
    shape = eyebrow_features["shape"]
    thickness = eyebrow_features["thickness"]
    length = eyebrow_features["length"]
    
    personality_traits = []
    
    # 눈썹 모양에 따른 해석
    shape_analysis = {
        "일자형": "논리적이고 실용적인 성격입니다. 결단력이 강하고 분명한 주관을 가지고 있으며, 목표를 향해 꾸준히 나아가는 성격입니다. 직관적이고 빠른 판단력이 특징입니다.",
        "아치형": "카리스마가 있고 표현력이 풍부합니다. 사교적이고 감성적인 면이 있으며, 창의력이 뛰어납니다. 타인과의 소통이 원활하고 리더십이 있습니다.",
        "둥근형": "부드럽고 친근한 성격입니다. 타인을 배려하고 협력적인 면이 있으며, 평화를 추구합니다. 안정적인 성격으로 신뢰할 수 있는 사람입니다.",
        "각진형": "리더십이 있고 추진력이 강합니다. 목표 지향적이고 성취욕이 높으며, 도전을 즐기는 성격입니다. 강인한 의지와 결단력을 가지고 있습니다.",
        "기본형": "균형 잡힌 성격의 소유자입니다. 적응력이 좋고 다양한 상황에 대처할 수 있으며, 안정적인 성격입니다. 타인과의 조화를 잘 이루는 능력이 있습니다."
    }
    
    # 눈썹 두께에 따른 해석
    thickness_analysis = {
        "얇음": "섬세하고 예민한 감성을 가졌습니다. 디테일에 신경 쓰는 성격으로, 예술적 감각이 뛰어납니다. 깊이 있는 사고와 통찰력이 특징입니다.",
        "중간": "균형 잡힌 정서를 가지고 있으며 상황에 따라 유연하게 대처합니다. 안정적인 성격으로 신뢰할 수 있는 사람입니다. 적절한 판단력과 통찰력을 가지고 있습니다.",
        "두꺼움": "강인한 생명력과 활력이 넘칩니다. 자신감이 있고 추진력이 강하며, 적극적인 성격입니다. 강한 의지력과 책임감이 특징입니다."
    }
    
    # 눈썹 길이에 따른 해석
    length_analysis = {
        "짧음": "직관적이고 결단력이 빠릅니다. 간결하고 핵심을 중시하는 경향이 있으며, 실용적인 성격입니다. 빠른 판단력과 행동력이 특징입니다.",
        "중간": "균형 잡힌 사고방식을 가지고 있으며 대인관계에서 원만함을 보입니다. 안정적인 성격으로 신뢰할 수 있는 사람입니다. 적절한 판단력과 통찰력을 가지고 있습니다.",
        "긺": "사려 깊고 신중한 성격입니다. 계획적이고 장기적인 안목을 가지고 있으며, 깊이 있는 통찰력이 있습니다. 신중한 판단과 안정적인 성격이 특징입니다."
    }
    
    personality_traits.append(shape_analysis.get(shape, ""))
    personality_traits.append(thickness_analysis.get(thickness, ""))
    personality_traits.append(length_analysis.get(length, ""))
    
    # 종합적인 인상학적 해석 추가
    personality_traits.append("\n💡 종합적인 인상학적 해석:")
    personality_traits.append(f"당신의 눈썹은 {shape}이며 {thickness}고 {length}습니다. 이는 {shape_analysis[shape].split('.')[0]}과 {thickness_analysis[thickness].split('.')[0]}을 나타내며, {length_analysis[length].split('.')[0]}을 보여줍니다.")
    
    return personality_traits

# 눈썹 메이크업 추천 함수
def recommend_eyebrow_makeup(eyebrow_features, face_shape="타원형"):
    shape = eyebrow_features["shape"]
    thickness = eyebrow_features["thickness"]
    length = eyebrow_features["length"]
    
    recommendations = []
    
    # 기본 추천
    base_recommendation = {
        "title": "자연스러운 눈썹 메이크업",
        "description": "자연스러운 눈썹 메이크업은 대부분의 얼굴형에 어울리며, 부드러운 아치형을 만들어 얼굴에 친근한 인상을 줍니다. 특히 직장이나 일상생활에서 활용하기 좋은 스타일입니다.",
        "steps": [
            "눈썹 브러시로 눈썹을 빗어 정돈합니다.",
            "아이브로우 펜슬로 눈썹의 윤곽을 따라 그립니다.",
            "눈썹 파우더로 자연스럽게 채웁니다.",
            "눈썹 마스카라로 마무리하여 자연스러운 모양을 유지합니다."
        ],
        "추천 제품": "자연스러운 브라운 계열 아이브로우 펜슬과 파우더"
    }
    recommendations.append(base_recommendation)
    
    # 얼굴형에 따른 추천
    if shape == "일자형":
        if face_shape in ["둥근형", "타원형"]:
            recommendations.append({
                "title": "부드러운 아치형 눈썹",
                "description": "일자형 눈썹에 부드러운 아치를 더해 얼굴의 곡선을 강조합니다. 친근하고 부드러운 인상을 주어 대인관계에 도움이 되는 스타일입니다.",
                "steps": [
                    "눈썹 중간에서 약간 위로 올라가는 아치를 만듭니다.",
                    "눈썹 꼬리는 자연스럽게 내려오게 그립니다.",
                    "눈썹 파우더로 자연스럽게 채웁니다.",
                    "눈썹 마스카라로 고정시킵니다."
                ],
                "추천 제품": "부드러운 브라운 계열 아이브로우 펜슬"
            })
        else:
            recommendations.append({
                "title": "각진 눈썹 메이크업",
                "description": "일자형 눈썹에 각진 느낌을 더해 강한 인상을 줍니다. 카리스마 있는 이미지를 연출하고 싶을 때 추천하는 스타일입니다.",
                "steps": [
                    "눈썹 시작점은 자연스럽게 유지합니다.",
                    "눈썹 중간에서 각진 형태로 올라갑니다.",
                    "눈썹 꼬리는 뚜렷하게 내려오게 그립니다.",
                    "눈썹 젤로 고정시킵니다."
                ],
                "추천 제품": "진한 브라운 계열 아이브로우 펜슬과 젤"
            })
    
    elif shape == "아치형":
        recommendations.append({
            "title": "부드러운 아치형 강조 메이크업",
            "description": "자연스러운 아치형을 더욱 강조하여 세련된 인상을 줍니다. 우아하고 세련된 이미지를 연출하고 싶을 때 추천하는 스타일입니다.",
            "steps": [
                "눈썹 시작점은 부드럽게 그립니다.",
                "눈썹 중간의 아치를 살짝 더 강조합니다.",
                "눈썹 꼬리는 자연스럽게 가늘어지게 마무리합니다.",
                "하이라이터를 눈썹 아래에 살짝 바릅니다."
            ],
            "추천 제품": "미디엄 브라운 계열 아이브로우 펜슬과 하이라이터"
        })
    
    elif shape == "둥근형":
        recommendations.append({
            "title": "아치형 변형 메이크업",
            "description": "둥근 눈썹에 약간의 각을 주어 세련된 인상을 줍니다. 부드러운 인상에 세련미를 더하고 싶을 때 추천하는 스타일입니다.",
            "steps": [
                "눈썹 시작점은 그대로 유지합니다.",
                "눈썹 중간을 약간 더 올려 아치를 만듭니다.",
                "눈썹 꼬리는 살짝 날카롭게 마무리합니다.",
                "눈썹 브러시로 잘 블렌딩하여 자연스럽게 만듭니다."
            ],
            "추천 제품": "미디엄 브라운 계열 아이브로우 펜슬과 브러시"
        })
    
    elif shape == "각진형":
        recommendations.append({
            "title": "부드러운 각진형 메이크업",
            "description": "각진 눈썹의 날카로운 느낌을 부드럽게 조절하여 균형 잡힌 인상을 줍니다. 강한 인상을 부드럽게 만들고 싶을 때 추천하는 스타일입니다.",
            "steps": [
                "눈썹 시작점은 자연스럽게 그립니다.",
                "각진 부분을 약간 부드럽게 조절합니다.",
                "눈썹 꼬리는 살짝 내려오게 그립니다.",
                "눈썹 브러시로 잘 블렌딩하여 자연스럽게 만듭니다."
            ],
            "추천 제품": "미디엄 브라운 계열 아이브로우 펜슬과 브러시"
        })
    
    # 눈썹 두께에 따른 추천
    if thickness == "얇음":
        recommendations.append({
            "title": "풍성한 눈썹 메이크업",
            "description": "얇은 눈썹에 볼륨을 더해 입체적인 인상을 줍니다. 자연스러운 풍성함을 연출하고 싶을 때 추천하는 스타일입니다.",
            "steps": [
                "눈썹 파우더를 사용해 자연스럽게 채웁니다.",
                "눈썹 마스카라를 사용해 볼륨을 더합니다.",
                "눈썹 결을 살려 자연스럽게 그립니다.",
                "눈썹 젤로 고정시킵니다."
            ],
            "추천 제품": "아이브로우 파우더와 마스카라"
        })
    
    elif thickness == "두꺼움":
        recommendations.append({
            "title": "세련된 정돈 메이크업",
            "description": "두꺼운 눈썹을 정돈하여 세련된 인상을 줍니다. 자연스러운 정돈된 느낌을 연출하고 싶을 때 추천하는 스타일입니다.",
            "steps": [
                "눈썹 정리기로 불필요한 부분을 다듬습니다.",
                "눈썹 펜슬로 윤곽을 정돈합니다.",
                "눈썹 마스카라로 결을 살립니다.",
                "눈썹 젤로 고정시킵니다."
            ],
            "추천 제품": "아이브로우 펜슬과 마스카라"
        })
    
    return recommendations[:3]  # 최대 3개의 추천사항 반환

# 눈썹 메이크업 이미지 생성 함수
def generate_eyebrow_makeup_images(image, landmarks, recommendations):
    makeup_images = []
    
    # 원본 이미지 복사
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    for idx, recommendation in enumerate(recommendations):
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)
        
        # 눈썹 랜드마크 추출
        left_eyebrow_indices = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        right_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        
        left_eyebrow = []
        right_eyebrow = []
        
        for idx_l in left_eyebrow_indices:
            pt = (int(landmarks.landmark[idx_l].x * w), int(landmarks.landmark[idx_l].y * h))
            left_eyebrow.append(pt)
        
        for idx_r in right_eyebrow_indices:
            pt = (int(landmarks.landmark[idx_r].x * w), int(landmarks.landmark[idx_r].y * h))
            right_eyebrow.append(pt)
        
        # 다양한 메이크업 스타일 적용
        if idx == 0:  # 첫 번째 추천 - 자연스러운 스타일
            color = (139, 69, 19)  # 브라운
            width = 2
        elif idx == 1:  # 두 번째 추천 - 아치형 강조
            color = (0, 0, 0)  # 블랙
            width = 3
        else:  # 세 번째 추천 - 각진형 스타일
            color = (105, 105, 105)  # 다크 그레이
            width = 2
        
        # 눈썹 그리기
        for i in range(len(left_eyebrow) - 1):
            draw.line([left_eyebrow[i], left_eyebrow[i+1]], fill=color, width=width)
        
        for i in range(len(right_eyebrow) - 1):
            draw.line([right_eyebrow[i], right_eyebrow[i+1]], fill=color, width=width)
        
        # 추천 제목 추가
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), recommendation["title"], fill=(255, 255, 255), font=font)
        
        # 이미지 저장
        makeup_img = np.array(img_pil)
        makeup_images.append(makeup_img)
    
    return makeup_images

# 메인 앱 로직
uploaded_image = st.file_uploader("얼굴 사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # 로딩 상태 표시
    with st.spinner("이미지를 처리하는 중..."):
        # 이미지 불러오기
        image = Image.open(uploaded_image)
        image = np.array(image)
        
        # 이미지 전처리
        image = preprocess_image(image)
        
        if image is not None:
            # 이미지 표시
            st.image(image, caption="업로드된 이미지", use_column_width=True)
            
            # 얼굴 랜드마크 감지
            landmarks = detect_facial_landmarks(image)
            
            if landmarks:
                # 진행 상태 표시바
                progress_bar = st.progress(0)
                
                # 눈썹 랜드마크 추출
                left_eyebrow, right_eyebrow = extract_eyebrow_landmarks(landmarks)
                progress_bar.progress(25)
                
                # 눈썹 분석
                eyebrow_features = analyze_eyebrow_shape(left_eyebrow, right_eyebrow, image.shape)
                progress_bar.progress(50)
                
                # 분석 결과 표시
                st.subheader("눈썹 분석 결과")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("눈썹 모양", eyebrow_features['shape'])
                with col2:
                    st.metric("눈썹 두께", eyebrow_features['thickness'])
                with col3:
                    st.metric("눈썹 길이", eyebrow_features['length'])
                
                progress_bar.progress(75)
                
                # 인상학적 특성 분석
                st.subheader("인상학적 특성 분석")
                personality_traits = physiognomy_analysis(eyebrow_features)
                for trait in personality_traits:
                    st.info(trait)
                
                # 눈썹 메이크업 추천
                st.subheader("눈썹 메이크업 추천")
                recommendations = recommend_eyebrow_makeup(eyebrow_features)
                
                # 메이크업 이미지 생성
                makeup_images = generate_eyebrow_makeup_images(image, landmarks, recommendations)
                
                # 탭 생성하여 각 메이크업 추천 표시
                tabs = st.tabs([rec["title"] for rec in recommendations])
                
                for i, tab in enumerate(tabs):
                    with tab:
                        st.image(makeup_images[i], caption=recommendations[i]["title"], use_column_width=True)
                        st.write(recommendations[i]["description"])
                        st.subheader("메이크업 방법")
                        for step_idx, step in enumerate(recommendations[i]["steps"]):
                            st.write(f"{step_idx + 1}. {step}")
                
                progress_bar.progress(100)
                
                # 결과 저장 버튼
                if st.button("분석 결과 저장하기"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"eyebrow_analysis_{timestamp}.txt"
                    
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write("=== 눈썹 분석 결과 ===\n\n")
                        f.write(f"눈썹 모양: {eyebrow_features['shape']}\n")
                        f.write(f"눈썹 두께: {eyebrow_features['thickness']}\n")
                        f.write(f"눈썹 길이: {eyebrow_features['length']}\n\n")
                        f.write("=== 인상학적 특성 ===\n")
                        for trait in personality_traits:
                            f.write(f"- {trait}\n")
                        f.write("\n=== 메이크업 추천 ===\n")
                        for rec in recommendations:
                            f.write(f"\n{rec['title']}\n")
                            f.write(f"{rec['description']}\n")
                            f.write("메이크업 방법:\n")
                            for step in rec['steps']:
                                f.write(f"- {step}\n")
                    
                    st.success(f"분석 결과가 {filename}에 저장되었습니다.")
            else:
                st.error("얼굴을 감지할 수 없습니다. 다른 사진을 업로드해 주세요.")
        else:
            st.error("이미지 처리 중 오류가 발생했습니다. 다시 시도해 주세요.")
else:
    st.info("얼굴 사진을 업로드하면 눈썹을 분석하고 메이크업을 추천해 드립니다.")
    
    # 예시 이미지 표시
    st.subheader("눈썹 모양 예시")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("일자형 눈썹")
        st.image("https://via.placeholder.com/150x100?text=일자형+눈썹", use_column_width=True)
    
    with col2:
        st.write("아치형 눈썹")
        st.image("https://via.placeholder.com/150x100?text=아치형+눈썹", use_column_width=True)
    
    with col3:
        st.write("둥근형 눈썹")
        st.image("https://via.placeholder.com/150x100?text=둥근형+눈썹", use_column_width=True)
    
    with col4:
        st.write("각진형 눈썹")
        st.image("https://via.placeholder.com/150x100?text=각진형+눈썹", use_column_width=True)