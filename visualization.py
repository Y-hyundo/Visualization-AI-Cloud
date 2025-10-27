import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import platform

# Matplotlib이 MacOS에서 한글 폰트를 제대로 처리하도록 설정
if platform.system() == "Darwin":  # MacOS
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":  # Windows
    plt.rcParams["font.family"] = "Malgun Gothic"
else:  # Linux
    # Linux에서는 나눔폰트 등 적절한 한글 폰트가 설치되어 있어야 합니다.
    plt.rcParams["font.family"] = "NanumGothic"

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# --- 설정 (train_models.py와 일치시킴) ---
MODEL_DIR = "models"
SCALER_DIR = "scalers"

# 모델 파일 경로
MODEL_RANK_FILE = os.path.join(MODEL_DIR, "model_rank.joblib")
MODEL_SALES_FILE = os.path.join(MODEL_DIR, "model_sales.joblib")

# 스케일러 파일 경로
SCALER_RANK_FILE = os.path.join(SCALER_DIR, "scaler_rank.joblib")
SCALER_SALES_FILE = os.path.join(SCALER_DIR, "scaler_sales.joblib")


# --- [수정] 모델 및 스케일러 로드 ---
@st.cache_resource
def load_assets():
    try:
        model_rank = joblib.load(MODEL_RANK_FILE)
        model_sales = joblib.load(MODEL_SALES_FILE)
        scaler_rank = joblib.load(SCALER_RANK_FILE)
        scaler_sales = joblib.load(SCALER_SALES_FILE)
        return model_rank, model_sales, scaler_rank, scaler_sales
    except FileNotFoundError:
        st.error(
            f"‼️ 모델 또는 스케일러 파일을 찾을 수 없습니다. 2_train_models.py를 먼저 실행하세요."
        )
        st.error(
            f"필요한 파일: {MODEL_RANK_FILE}, {MODEL_SALES_FILE}, {SCALER_RANK_FILE}, {SCALER_SALES_FILE}"
        )
        return None, None, None, None


model_rank, model_sales, scaler_rank, scaler_sales = load_assets()

if model_rank is None:
    st.stop()

# --- [수정] 학습에 사용된 변수 리스트 (train_models.py 출력 결과 기준) ---
# 순서가 매우 중요합니다!
FEATURES_RANK = [
    "price_cp_BNR17",
    "rank_cp_Lactofit",
    "price_cp_Lactofit",
    "sales_cp_Lactofit",
    "rank_cp_Denps",
    "price_cp_Denps",
    "sales_cp_Denps",
    "price_gs_BNR17",
    "event_flag_gs_BNR17",
    "price_gs_Lactofit",
    "event_flag_gs_Lactofit",
    "price_gs_Denps",
    "event_flag_gs_Denps",
]

FEATURES_SALES = [
    "rank_cp_BNR17",
    "price_cp_BNR17",
    "rank_cp_Lactofit",
    "price_cp_Lactofit",
    "sales_cp_Lactofit",
    "rank_cp_Denps",
    "price_cp_Denps",
    "sales_cp_Denps",
    "price_gs_BNR17",
    "event_flag_gs_BNR17",
    "price_gs_Lactofit",
    "event_flag_gs_Lactofit",
    "price_gs_Denps",
    "event_flag_gs_Denps",
]


# --- 1. Streamlit UI (사이드바) ---
st.title("💡 자사 이익(Profit) 최적화 시뮬레이터")
st.write(
    "모델 1(랭킹 예측)과 모델 2(판매량 예측)를 결합하여 '최적 이익' 가격을 도출합니다."
)

st.sidebar.header("Step 1: 현재 시장 조건 입력")

# --- [수정] 학습에 사용된 모든 변수를 입력받도록 UI 변경 ---
st.sidebar.subheader("자사(BNR17) 홈쇼핑 조건")
price_gs_BNR17 = st.sidebar.number_input(
    "자사(BNR17) 홈쇼핑 가격 (원)", 30000, 50000, 39900, step=500
)
event_flag_gs_BNR17 = st.sidebar.selectbox(
    "자사(BNR17) 홈쇼핑 방송 여부", ["없음", "있음"], key="bnr_event"
)

st.sidebar.subheader("경쟁사1(락토핏) 조건")
rank_cp_Lactofit = st.sidebar.slider("락토핏 쿠팡 랭킹 (위)", 1, 50, 3)
price_cp_Lactofit = st.sidebar.number_input(
    "락토핏 쿠팡 가격 (원)", 20000, 40000, 25000, step=500
)
sales_cp_Lactofit = st.sidebar.slider("락토핏 쿠팡 일 판매량 (추정)", 10, 500, 150)
price_gs_Lactofit = st.sidebar.number_input(
    "락토핏 홈쇼핑 가격 (원)", 20000, 40000, 29900, step=500
)
event_flag_gs_Lactofit = st.sidebar.selectbox(
    "락토핏 홈쇼핑 방송 여부", ["없음", "있음"], key="lacto_event"
)

st.sidebar.subheader("경쟁사2(덴마크) 조건")
rank_cp_Denps = st.sidebar.slider("덴마크 쿠팡 랭킹 (위)", 1, 50, 5)
price_cp_Denps = st.sidebar.number_input(
    "덴마크 쿠팡 가격 (원)", 30000, 50000, 35000, step=500
)
sales_cp_Denps = st.sidebar.slider("덴마크 쿠팡 일 판매량 (추정)", 10, 500, 100)
price_gs_Denps = st.sidebar.number_input(
    "덴마크 홈쇼핑 가격 (원)", 30000, 50000, 39900, step=500
)
event_flag_gs_Denps = st.sidebar.selectbox(
    "덴마크 홈쇼핑 방송 여부", ["없음", "있음"], key="denps_event"
)


st.sidebar.header("Step 2: 자사 원가 입력 (이익 계산용)")
our_cogs = st.sidebar.number_input(
    "자사(BNR17) 1개당 원가 (원)", 10000, 30000, 15000, step=1000
)


# --- [수정] 2. 시뮬레이션 함수 (스케일러 적용) ---
def run_simulation(our_price_list, cogs, current_conditions):
    """
    입력된 가격 범위에 대해 랭킹, 판매량, 이익을 동시 예측합니다.
    """
    results = []

    for price in our_price_list:

        # --- 모델 1: 랭킹 예측 ---
        # 1-1. 입력 데이터 준비 (X1)
        x1_data = current_conditions.copy()
        x1_data["price_cp_BNR17"] = price  # 시뮬레이션 가격 적용

        # DataFrame 생성 (학습 때 사용한 순서와 동일하게)
        x1 = pd.DataFrame([x1_data], columns=FEATURES_RANK)

        # 1-2. [수정] 데이터 스케일링
        x1_scaled = scaler_rank.transform(x1)

        # 1-3. 랭킹 예측
        pred_rank = model_rank.predict(x1_scaled)[0]
        pred_rank = max(1, round(pred_rank))  # 랭킹은 1 이상

        # --- 모델 2: 판매량 예측 ---
        # 2-1. 입력 데이터 준비 (X2)
        x2_data = current_conditions.copy()
        x2_data["price_cp_BNR17"] = price  # 시뮬레이션 가격 적용
        x2_data["rank_cp_BNR17"] = pred_rank  # 모델1의 예측 결과를 입력으로 사용

        # DataFrame 생성 (학습 때 사용한 순서와 동일하게)
        x2 = pd.DataFrame([x2_data], columns=FEATURES_SALES)

        # 2-2. [수정] 데이터 스케일링
        x2_scaled = scaler_sales.transform(x2)

        # 2-3. 판매량 예측
        pred_sales = model_sales.predict(x2_scaled)[0]
        pred_sales = max(0, round(pred_sales))  # 판매량은 0 이상

        # --- 모델 3: 이익 계산 ---
        margin = price - cogs  # 1개당 이익
        pred_profit = margin * pred_sales

        results.append(
            {
                "우리가격": price,
                "예상랭킹": pred_rank,  # 이름 변경
                "예상판매량": pred_sales,
                "예상이익": pred_profit,
            }
        )
    return pd.DataFrame(results)


# --- 3. Streamlit UI (메인 화면) ---
st.header("Step 3: 자사 쿠팡 가격 조정 및 결과 확인")

price_min = st.number_input("최소 분석 가격 (원)", 28000, 50000, 30000, step=500)
price_max = st.number_input("최대 분석 가격 (원)", 28000, 50000, 45000, step=500)

if st.button("📈 최적 이익 가격 분석 실행"):

    # [수정] 시뮬레이션에 필요한 모든 현재 조건 값을 딕셔너리로 묶기
    current_market_conditions = {
        "rank_cp_Lactofit": rank_cp_Lactofit,
        "price_cp_Lactofit": price_cp_Lactofit,
        "sales_cp_Lactofit": sales_cp_Lactofit,
        "rank_cp_Denps": rank_cp_Denps,
        "price_cp_Denps": price_cp_Denps,
        "sales_cp_Denps": sales_cp_Denps,
        "price_gs_BNR17": price_gs_BNR17,
        "event_flag_gs_BNR17": 1 if event_flag_gs_BNR17 == "있음" else 0,
        "price_gs_Lactofit": price_gs_Lactofit,
        "event_flag_gs_Lactofit": 1 if event_flag_gs_Lactofit == "있음" else 0,
        "price_gs_Denps": price_gs_Denps,
        "event_flag_gs_Denps": 1 if event_flag_gs_Denps == "있음" else 0,
    }

    price_range = np.arange(price_min, price_max + 500, 500)

    # [수정] 시뮬레이션 함수 호출
    df_sim = run_simulation(price_range, our_cogs, current_market_conditions)

    if df_sim.empty:
        st.warning("분석 결과가 없습니다.")
    else:
        # 이익이 최대가 되는 지점 (최적 가격)
        optimal = df_sim.loc[df_sim["예상이익"].idxmax()]

        st.markdown("---")
        st.header(f"📊 분석 결과")
        st.write(
            f"(입력된 조건: 락토핏 쿠팡 {rank_cp_Lactofit}위 / 홈쇼핑 {price_gs_Lactofit:,}원)"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="👑 이익 극대화 가격 (Optimal Price)",
            value=f"{int(optimal['우리가격']):,} 원",
        )
        col2.metric(
            label="💰 이때의 예상 이익 (Profit)",
            value=f"{int(optimal['예상이익']):,} 원",
        )
        col3.metric(
            label="📦 이때의 예상 판매량 (Sales)",
            value=f"{int(optimal['예상판매량'])} 개",
        )

        # --- 시각화 (이익 vs 판매량) ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = "tab:green"
        ax1.set_xlabel("자사(BNR17) 쿠팡 가격 (원)")
        ax1.set_ylabel("예상 일일 이익 (Profit)", color=color1)
        ax1.plot(
            df_sim["우리가격"],
            df_sim["예상이익"],
            color=color1,
            marker="o",
            label="예상 이익",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.axvline(
            optimal["우리가격"],
            color=color1,
            linestyle="--",
            label=f"최적 이익 가격: {int(optimal['우리가격']):,}원",
        )

        ax2 = ax1.twinx()
        color2 = "tab:blue"
        ax2.set_ylabel("예상 일일 판매량 (Sales)", color=color2)
        ax2.plot(
            df_sim["우리가격"],
            df_sim["예상판매량"],
            color=color2,
            marker="x",
            linestyle=":",
            label="예상 판매량",
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.suptitle("자사 가격 변화에 따른 이익 및 판매량 시뮬레이션", fontsize=16)

        # [수정] 범례 위치 조정
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines + lines2,
            labels + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
        )

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # suptitle과 겹치지 않게
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("가격대별 상세 예측 데이터")
        st.dataframe(
            df_sim.style.format(
                {
                    "우리가격": "{:,.0f}원",
                    "예상랭킹": "{:.0f}위",  # 수정
                    "예상판매량": "{:.0f}개",
                    "예상이익": "{:,.0f}원",
                }
            ).background_gradient(subset=["예상이익"], cmap="Greens")
        )
