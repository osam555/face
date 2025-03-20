# FaceTalk - 얼굴 분석 애플리케이션

FaceTalk은 얼굴 분석을 통해 관상과 메이크업 추천을 제공하는 웹 애플리케이션입니다.

## 주요 기능

- 얼굴 랜드마크 감지
- 눈썹 분석
- 관상 해석
- 메이크업 추천

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone https://github.com/yourusername/facetalk.git
cd facetalk
```

2. 가상환경을 생성하고 활성화합니다:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

3. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run facetalk.py
```

## 사용된 기술

- Python
- Streamlit
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- Pillow

## 라이선스

MIT License 