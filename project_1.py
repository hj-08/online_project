import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import os

# --- 한글 폰트 설정 수정: packages.txt를 통해 NanumGothic을 설치하도록 가정 ---
def set_korean_font():
    """시스템에 설치된 한글 폰트를 찾아 Matplotlib에 설정합니다."""
    
    # 폰트 검색 로직을 별도 함수로 분리
    def find_font_name():
        font_list = [f.name for f in fm.fontManager.ttflist]
        
        # NanumGothic 계열, Noto Sans, Malgun Gothic 순으로 검색
        for name in ["NanumGothic", "NanumGothic Bold", "NanumBarunGothic", "NanumSquare", "Noto Sans CJK KR", "Malgun Gothic"]:
            if name in font_list:
                return name
        return None

    font_name = find_font_name()
    
    # 폰트가 발견되지 않았을 경우, 캐시를 지우고 다시 시도 (Streamlit 환경에서 필수)
    if not font_name:
        # 이 부분은 Canvas 환경에서는 경고가 계속 나올 수 있지만, Streamlit Cloud 배포 시 해결을 위한 코드입니다.
        try:
            cache_dir = fm.get_cachedir()
            for filename in os.listdir(cache_dir):
                if filename.startswith('fontlist-'):
                    os.remove(os.path.join(cache_dir, filename))
            
            fm.fontManager._rebuild()
            font_name = find_font_name() # 다시 폰트 이름 찾기
        except Exception:
            pass # 권한 오류 등 무시

    # 4. 최종 기본 폰트 설정
    if not font_name:
        font_name = "DejaVu Sans"
        st.sidebar.warning(f"적절한 한글 폰트를 찾을 수 없습니다. 기본 폰트({font_name}) 사용. (Streamlit Cloud 사용 시 'packages.txt'에 'fonts-nanum' 추가 및 **재배포** 필요)")
        font_prop = None
    else:
        # 찾은 폰트로 Matplotlib 설정
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        st.sidebar.success(f"한글 폰트 설정 완료: {font_name}")
        # font_prop 생성
        font_prop = fm.FontProperties(family=font_name)

    plt.rcParams['axes.unicode_minus'] = False
    return font_prop

# 폰트 설정 실행 및 font_prop 변수에 저장
font_prop = set_korean_font()


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

# 주요 도시/측정소 매핑
AIR_STATION_MAP = {
    "서울": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"],
    "경기": ["수원", "성남", "안양", "안산", "용인", "평택", "고양", "남양주", "의정부", "광명", "화성", "파주", "시흥"]
}

# 1. 시/도 선택 (드롭다운)
default_city = "서울"
city = st.selectbox("시/도 선택", list(AIR_STATION_MAP.keys()), 
                    index=list(AIR_STATION_MAP.keys()).index(default_city) if default_city in AIR_STATION_MAP else 0)

# 2. 선택된 시/도에 따라 구/군 목록 업데이트
district_options = AIR_STATION_MAP.get(city, [])

# 3. 구/군 (측정소) 선택 (드롭다운)
if district_options:
    gu = st.selectbox("구/군 (측정소) 선택", district_options, 
                      index=district_options.index("강남구") if "강남구" in district_options else 0)
else:
    gu = st.text_input("구/군 (측정소) 입력 (목록 없음)", "")
    st.warning("선택된 시/도에 대한 측정소 목록이 없습니다. 직접 입력해주세요.")
    
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
    
    # 범례에 폰트 속성 적용 (font_prop이 None이 아닐 경우)
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
