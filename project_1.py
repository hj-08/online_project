import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import os

# --- 한글 폰트 설정 수정: 시스템 환경에 구애받지 않도록 파일 경로 지정 ---
# 1. NanumGothicBold.ttf 파일을 프로젝트 폴더에 넣어주세요.
FONT_FILE_PATH = "NanumGothicBold.ttf"

# 폰트 파일이 존재하는지 확인하고 설정
try:
    if os.path.exists(FONT_FILE_PATH):
        # 폰트 속성 생성 (파일 경로 지정)
        font_prop = fm.FontProperties(fname=FONT_FILE_PATH)
        font_name = font_prop.get_name()

        # Matplotlib에 폰트 등록 및 설정
        fm.fontManager.addfont(FONT_FILE_PATH)
        plt.rcParams['font.family'] = font_name
        st.sidebar.success(f"한글 폰트 설정 완료: {font_name}")

    else:
        # 파일이 없을 경우 기본 폰트로 설정하고 경고
        font_name = "DejaVu Sans"
        plt.rcParams['font.family'] = font_name
        st.sidebar.warning(f"폰트 파일 '{FONT_FILE_PATH}'을 찾을 수 없습니다. 기본 폰트({font_name}) 사용.")
        # 기본 폰트일 경우 font_prop은 None 또는 기본 속성으로 설정
        font_prop = None 

    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

except Exception as e:
    st.sidebar.error(f"폰트 설정 중 오류 발생: {e}")
    font_name = "DejaVu Sans"
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False
    font_prop = None

# --- API KEY (공개 API 키이므로 그대로 사용) ---
API_KEY = "aea45d5692f9dc0fb20ff49e2cf104f6614d3a17df9e92420974a5defb3cd75e"

def fetch_air_data(station_name, num_rows=48):
    """실시간 측정소별 미세먼지 데이터를 가져옵니다."""
    URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {
        'serviceKey': API_KEY,
        'returnType': 'json',
        'numOfRows': num_rows,
        'pageNo': 1,
        'stationName': station_name,
        'dataTerm': 'DAILY',
        'ver': '1.3'
    }
    r = requests.get(URL, params=params, timeout=10)
    r.raise_for_status() # HTTP 오류 발생 시 예외 발생
    data = r.json()
    items = data['response']['body']['items']
    return items

def parse_pm(items, key='pm10Value'):
    """데이터 항목 리스트에서 시간과 PM 값을 파싱합니다."""
    times = []
    values = []
    for it in items:
        t = it.get('dataTime')
        val = it.get(key)
        try:
            v = float(val)
        except:
            continue
        
        dt = None
        # 다양한 시간 형식 처리 시도
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d%H%M"):
            try:
                dt = datetime.strptime(t, fmt)
                break
            except:
                continue
        
        if dt is None:
            continue
        
        times.append(dt)
        values.append(v)
        
    return times[::-1], values[::-1] # 데이터를 시간순으로 반전

def linear_regression_predict(values):
    """선형 회귀를 사용하여 다음 값을 예측합니다."""
    if len(values) < 3:
        return None
        
    X = np.arange(len(values)).reshape(-1,1)
    y = np.array(values)
    model = LinearRegression().fit(X, y)
    
    # 다음 시점 (인덱스 len(values))의 값을 예측
    pred = model.predict([[len(values)]])[0] 
    return pred

def recommend_by_value(val):
    """PM10 값에 따른 추천 등급과 메시지를 반환합니다."""
    if val is None:
        return "예측값을 계산할 수 없습니다."
        
    if val > 150:
        return "🔥 매우 나쁨: 외출 자제, 실내 활동 권장"
    if val > 80:
        return "⚠️ 나쁨: 장시간 외출 피하고 마스크 착용"
    if val > 30:
        return "🙂 보통: 민감군은 주의, 가벼운 외출 가능"
        
    return "🌿 좋음: 외부 활동 안전"

# --- Streamlit UI 구성 ---

st.title("🌫️ 실시간 미세먼지 분석 + 예측")
st.markdown("정부 공공데이터 포털의 실시간 미세먼지 데이터를 기반으로 합니다.")

city = st.text_input("시/도 입력", "서울")
gu = st.text_input("구/군 입력", "강남구")
station = gu # 측정소 이름으로 사용

if st.button("분석 시작", key="analyze_button"):
    try:
        with st.spinner('데이터 불러오는 중...'):
            items = fetch_air_data(station, num_rows=50)
        st.success("데이터 불러오기 성공!")
    except requests.HTTPError:
        st.error("데이터 요청 중 HTTP 오류가 발생했습니다. 지역명 또는 API 키를 확인하세요.")
        st.stop()
    except Exception as e:
        st.error(f"데이터 요청 중 오류 발생: {e}")
        st.stop()

    times, values = parse_pm(items)

    if not values:
        st.warning(f"측정소 '{station}'에 대한 유효한 PM10 데이터가 없습니다. 지역명을 다시 확인해주세요.")
        st.stop()
        
    predict = linear_regression_predict(values)

    # --- Matplotlib 시각화 ---
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.set_facecolor('#f9f9f9')
    ax.grid(True, color='#e1e1e1', linestyle='-', linewidth=1)

    # 실측 데이터 플롯
    ax.plot(times, values, color='#2a4d8f', marker='o', linewidth=2, label='실측 PM10')
    
    # 데이터 포인트 위에 값 표시
    for x, y in zip(times, values):
        ax.text(x, y + 1, f"{y:.0f}", color='#2a4d8f', fontsize=8, ha='center')

    # 예측값 플롯
    if predict is not None:
        next_time = times[-1] + timedelta(hours=1)
        ax.plot([times[-1], next_time],
                [values[-1], predict],
                color='#f28500', marker='o', linestyle='--', linewidth=2, 
                label=f'예측값: {predict:.1f}')
        ax.text(next_time, predict + 1, f"{predict:.0f}", color='#f28500', fontsize=8, ha='center')

    # X축 눈금 설정 (6시간 간격)
    ax.set_xticks(times[::6])
    ax.set_xticklabels([t.strftime("%m-%d %H:%M") for t in times[::6]], rotation=45)

    # Y축 레이블 설정
    ax.set_ylabel("PM10 (㎍/m³)")
    
    # 범례에 폰트 속성 적용 (폰트 파일이 없을 경우 대비)
    if font_prop:
        ax.legend(frameon=False, prop=font_prop)
    else:
        ax.legend(frameon=False) 

    plt.tight_layout()

    st.pyplot(fig)
    
    # --- 예측 결과 표시 ---
    st.subheader("📌 예측 결과")
    if predict is not None:
        st.write(f"다음 PM10 예측값: **{predict:.1f} ㎍/m³**")
        st.info(recommend_by_value(predict))
    else:
        st.warning("데이터 부족으로 예측값을 계산할 수 없습니다.")
