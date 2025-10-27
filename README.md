📈 자사 이익 최적화 시뮬레이터

이 프로젝트는 2단계 머신러닝 모델을 사용하여 경쟁사 및 자사 마케팅 조건에 따른 최적의 이익을 도출하는 Streamlit 웹 애플리케이션입니다.

모델 1 (랭킹 예측): 현재 시장 상황과 자사 가격을 기반으로 예상 쿠팡 랭킹을 예측합니다.

모델 2 (판매량 예측): 모델 1의 예상 랭킹을 포함한 모든 변수를 사용하여 최종 일일 판매량을 예측합니다.

이익 계산: 예측된 판매량과 입력된 원가를 기반으로 가격대별 예상 이익을 계산하고, 이익이 극대화되는 최적 가격을 제시합니다.

📂 프로젝트 구조

이 시뮬레이터가 정상적으로 작동하려면 모델 및 스케일러 파일이 다음 폴더 구조로 배치되어야 합니다.

.
├── models/
│   ├── model_rank.joblib
│   └── model_sales.joblib
├── scalers/
│   ├── scaler_rank.joblib
│   └── scaler_sales.joblib
├── app.py           (제공해주신 시뮬레이터 .py 파일)
└── README.md        (현재 이 파일)


중요: models 폴더와 scalers 폴더를 직접 생성하고, 전달받은 4개의 .joblib 파일을 각각 올바른 위치에 넣어주세요.

🛠️ 설치 및 실행

1. 필수 패키지 설치

실행에 필요한 파이썬 패키지 목록입니다.

pip install streamlit pandas numpy joblib matplotlib scikit-learn


또는, 아래 내용으로 requirements.txt 파일을 생성한 뒤 다음 명령어를 실행해도 됩니다.

requirements.txt

streamlit
pandas
numpy
joblib
matplotlib
scikit-learn


pip install -r requirements.txt


2. 시뮬레이터 실행

터미널에서 다음 명령어를 입력하여 Streamlit 앱을 실행합니다. (app.py는 실제 파이썬 스크립트 파일명으로 변경하세요.)

streamlit run app.py


🚀 사용 방법

streamlit run 명령어를 실행하면 웹 브라우저가 자동으로 열립니다.

왼쪽 사이드바의 Step 1에서 현재 경쟁사 및 자사 홈쇼핑 조건을 입력합니다.

Step 2에서 자사 제품의 1개당 원가를 입력합니다.

Step 3에서 분석하고자 하는 자사 쿠팡 가격의 최소/최대 범위를 설정합니다.

최적 이익 가격 분석 실행 버튼을 클릭하면 메인 화면에 결과가 나타납니다.