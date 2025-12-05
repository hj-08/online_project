# ===== 라이브러리 임포트 =====
import requests                 # HTTP 요청을 보낼 때 사용 (API 호출)
import json                     # JSON 파싱(필요시, 여기선 r.json() 사용)
import matplotlib.pyplot as plt # 그래프 그릴 때 사용 (matplotlib)
import numpy as np             # 숫자 배열·계산용 (선형대수, 인덱스 생성 등)
import streamlit as st         # Streamlit UI를 만들 때 사용
from datetime import datetime, timedelta  # 시간 관련 처리 (파싱/시간 더하기 등)
from sklearn.linear_model import LinearRegression  # 선형회귀 모델 (예측에 사용)
import matplotlib.font_manager as fm  # 시스템 폰트 탐색/설정용
import os  # 운영체제 관련 (여기선 주석에선 사용 안 함)

# ===== 한글 폰트 설정 함수 =====
def set_korean_font():
    """
    그래프(및 범례)에 한글이 깨지지 않게 적절한 한글 폰트를 찾아 matplotlib에 설정.
    - 시스템에 설치된 폰트 리스트에서 'NanumGothic', 'Malgun Gothic', 'Noto Sans CJK KR' 중 하나를 찾음.
    - 찾으면 plt.rcParams['font.family']에 설정하고 FontProperties 객체를 반환.
    - 못 찾으면 기본 영문 폰트('DejaVu Sans')를 쓰고 streamlit 사이드바에 경고를 띄움.
    """
    # 내부 도우미: 시스템 폰트 리스트에서 이름 찾기
    def find_font_name():
        # fm.fontManager.ttflist: 시스템의 ttf 폰트 리스트(각 항목에 .name 속성 있음)
        font_list = [f.name for f in fm.fontManager.ttflist]
        # 자주 쓰이는 한국어 폰트 이름들을 순서대로 확인
        for name in ["NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
            if name in font_list:
                return name
        return None

    font_name = find_font_name()

    if font_name:
        # matplotlib에 폰트 패밀리로 설정 (그래프 텍스트가 한글일 때 깨지지 않음)
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨짐 방지
        font_prop = fm.FontProperties(family=font_name)  # legend 등에 사용 가능
    else:
        # 발견 실패: 기본 폰트 사용, 유저에게 경고
        font_name = "DejaVu Sans"
        st.sidebar.warning(f"적절한 한글 폰트를 찾을 수 없습니다. 기본 폰트({font_name}) 사용.")
        font_prop = None

    plt.rcParams['axes.unicode_minus'] = False
    return font_prop

# 한 번만 수행: 폰트 객체를 전역으로 보관
font_prop = set_korean_font()

# ===== API 키 (공공데이터 포털) =====
API_KEY = "aea45d5692f9dc0fb20ff49e2cf104f6614d3a17df9e92420974a5defb3cd75e"
# -> 실제 운영 시에는 하드코딩보다 환경변수나 비밀 관리 사용 권장

# ===== API 호출 함수 =====
def fetch_air_data(station_name, num_rows=24):
    """
    주어진 측정소 이름(station_name)에 대해 실시간 측정값을 요청하여 JSON 아이템 리스트 반환.
    - num_rows: 요청할 항목 개수 (여기선 24로 고정 사용)
    - API 엔드포인트: getMsrstnAcctoRltmMesureDnsty
    - 주의: HTTP 응답 코드가 200이 아니면 requests.raise_for_status()가 예외를 던짐.
    """
    URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {
        'serviceKey': API_KEY,
        'returnType': 'json',
        'numOfRows': num_rows,
        'stationName': station_name,
        'dataTerm': 'DAILY',
        'ver': '1.3'
    }

    r = requests.get(URL, params=params, timeout=10)  # 타임아웃 10초
    r.raise_for_status()  # HTTP 에러(4xx/5xx)면 예외 발생
    data = r.json()  # JSON -> 파이썬 dict

    # 응답 구조: response -> body -> items (list)
    items = data['response']['body']['items']
    return items

# ===== 데이터 파싱 함수 =====
def parse_pm(items, key='pm10Value'):
    """
    API에서 받은 items 리스트에서 '측정시간(dataTime)'과 해당 pm 값을 추출해
    시간 리스트(times)와 값 리스트(values)를 반환.
    - key: 'pm10Value' 또는 'pm25Value' 등
    - 반환 전 리스트는 시간 순서(오래된 -> 최신)로 뒤집혀 최종 반환됨.
    - NOTE: 코드에는 '의도적 오류 주입'이 포함되어 있음 (첫 유효 값에 "ERROR_VAL"을 넣음).
      -> 이 부분은 실제로는 제거해야 함(아래에서 표시).
    """
    times = []
    values = []

    error_injected = False  # (의도적) 오류 주입 플래그

    for it in items:
        t = it.get('dataTime')  # 측정 시각 문자열 예: "2025-12-05 20:00"
        val = it.get(key)       # pm 값(문자열 또는 None)

        try:
            v = float(val)  # 문자열을 float로 변환 시도
            # === 의도적 버그: 첫 유효 데이터에 문자열 삽입 ===
            if not error_injected:
                 v = "ERROR_VAL"  # 숫자 대신 문자열을 넣음 -> 이후 타입 에러 유발
                 error_injected = True
            # === 버그 끝 ===
        except:
            # 변환 불가(예: None, '-', '')이면 건너뜀
            continue

        # 시간 문자열을 datetime으로 파싱 (두 포맷을 시도)
        dt = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y%m%d%H%M"):
            try:
                dt = datetime.strptime(t, fmt)
                break
            except:
                continue

        if dt is None:
            # 시간 포맷이 예상과 다르면 해당 항목을 무시
            continue

        times.append(dt)
        values.append(v)

    # items는 최신->과거 순서일 수 있으니 뒤집어서 오래된->최신으로 맞춤
    return times[::-1], values[::-1]

# ===== 선형 회귀 예측 함수 =====
def linear_regression_predict(times, values, n_hours=3):
    
    # values 중에서 숫자(int, float)만 골라서 새 리스트에 담기
    # → ERROR_VAL 같은 글자는 제거됨
    numeric_values = [v for v in values if isinstance(v, (int, float))]

    # 만약 숫자가 너무 적으면(3개 미만) 예측 모델을 만들 수 없음
    if len(numeric_values) < 3:
        return None, None, None

    try:
        # X는 숫자의 개수만큼 0,1,2,... 증가하는 번호
        # 예: 값이 5개면 X = [[0],[1],[2],[3],[4]]
        # 선형회귀 모델은 이렇게 '번호'를 기준으로 패턴을 찾는다
        X = np.arange(len(numeric_values)).reshape(-1,1)

        # y는 실제 숫자 데이터 (미세먼지 값 등)
        y = np.array(numeric_values)

    except ValueError:
        # 만약 배열을 만드는 도중 오류가 나면 예측을 할 수 없음
        st.warning("경고: 예측 데이터 준비 중 오류가 발생했습니다. 예측을 건너뜁니다.")
        return None, None, None

    # 선형 회귀 모델 학습 (X → y 관계를 공부함)
    model = LinearRegression().fit(X, y)

    # 앞으로 n_hours 만큼 미래를 예측하기 위해
    # 예측할 X 번호를 새로 만듦
    # 예: 기존 데이터가 5개면 예측 X = [5,6,7] (3시간 예측 기준)
    X_pred = np.arange(len(numeric_values), len(numeric_values) + n_hours).reshape(-1, 1)

    # 위의 번호(X_pred)에 대해 모델이 예측한 값
    predict_values = model.predict(X_pred)

    # 혹시 예측값이 너무 작게 나와서 음수가 되면 의미가 없으니
    # 최소값을 1로 맞춰줌
    predict_values = np.maximum(1.0, predict_values)

    # 예측 시간이 언제인지 계산하기
    # times 리스트의 마지막 시간을 기준으로
    # +1시간, +2시간, ... 식으로 예측 시간을 만든다
    last_time = times[-1]
    predict_times = [last_time + timedelta(hours=i) for i in range(1, n_hours + 1)]

    # 예측값 / 예측시간 / 학습된 모델을 반환
    return predict_values, predict_times, model


# ===== 등급 기준 및 유틸 함수들 =====
PM10_CRITERIA = {
    '좋음': (0, 30),
    '보통': (31, 80),
    '나쁨': (81, 150),
    '매우 나쁨': (151, float('inf'))
}
PM25_CRITERIA = {
    '좋음': (0, 15),
    '보통': (16, 35),
    '나쁨': (36, 75),
    '매우 나쁨': (76, float('inf'))
}

def get_grade_criteria(pm_type):
    """pm_type이 'PM10'이면 PM10 기준, 아니면 PM25 기준을 반환."""
    return PM10_CRITERIA if pm_type == 'PM10' else PM25_CRITERIA

def recommend_by_value(val, pm_type='PM10'):
    """
    주어진 농도 값(val)에 따라 행동 권장 문구 반환.
    - val이 None이면 예측 불가 메시지 반환.
    - 등급 경계에 따라 적절한 메시지(좋음/보통/나쁨/매우 나쁨).
    """
    if val is None:
        return "예측값을 계산할 수 없어."

    criteria = get_grade_criteria(pm_type)

    if val >= criteria['매우 나쁨'][0]:
        return "🔥 매우 나쁨: 외출 자제, 실내 활동 권장"
    if val >= criteria['나쁨'][0]:
        return "⚠️ 나쁨: 장시간 외출 피하고 마스크 착용"
    if val >= criteria['보통'][0]:
        return "🙂 보통: 민감군은 주의, 가벼운 외출 가능"

    return "🌿 좋음: 외부 활동 안전"

# ===== Streamlit UI 구성 =====
st.title("🌫️ 실시간 미세먼지 분석 + 예측 (최근 24시간)")

st.markdown("정부 공공데이터 포털의 실시간 미세먼지 데이터를 기반으로 합니다다. **예측은 향후 3시간을 기준으로 합니다.**")
# -> '합니다다' 오타 있음 (표시 목적). UI 문구는 자유롭게 수정 가능

# 측정소 목록(시/도 -> 구/군). UI 편의를 위한 하드코딩된 맵.
AIR_STATION_MAP = {
    "서울": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"],
    "부산": ["대연동", "명장동", "학장동", "덕천동", "전포동", "광복동", "용호동", "장림동", "신평동", "해운대", "기장읍", "정관읍"],
    "대구": ["봉산동", "이현동", "지산동", "성서", "대명동", "복현동", "만촌동", "안심"],
    "인천": ["주안", "구월동", "송도", "연희동", "운서동", "신흥동", "석남동"],
    "광주": ["운암동", "광산구", "북구", "동구", "서구"],
    "대전": ["가양동", "문평동", "노은동", "오룡동", "대흥동"],
    "울산": ["달동", "삼산동", "명촌동", "농소", "화암동"],
    "세종": ["신흥동", "보람동"],
    "경기": ["수원", "성남", "안양", "안산", "용인", "평택", "고양", "남양주", "의정부", "광명", "화성", "파주", "시흥", "김포", "군포", "하남", "오산", "이천", "안성"],
    "강원": ["춘천", "원주", "강릉", "동해", "속초", "삼척", "철원", "횡성", "홍천"],
    "충북": ["청주", "충주", "제천", "단양", "옥천", "증평", "진천"],
    "충남": ["천안", "공주", "보령", "아산", "서산", "논산", "당진", "계룡", "예산"],
    "전북": ["전주", "군산", "익산", "정읍", "남원", "김제", "완주"],
    "전남": ["목포", "여수", "순천", "나주", "광양", "무안", "구례", "화순"],
    "경북": ["포항", "경주", "김천", "안동", "구미", "영주", "영천", "상주"],
    "경남": ["창원", "진주", "통영", "사천", "김해", "밀양", "거제", "양산"],
    "제주": ["제주시", "서귀포"]
}

default_city = "서울"
# 시/도 선택 드롭다운을 보여줌. 기본 선택은 default_city
city = st.selectbox("시/도 선택", list(AIR_STATION_MAP.keys()),
                     index=list(AIR_STATION_MAP.keys()).index(default_city) if default_city in AIR_STATION_MAP else 0)

# 선택된 시의 구/군 목록을 가져옴
district_options = AIR_STATION_MAP.get(city, [])

# 구/군이 있으면 selectbox, 없으면 텍스트 입력창 제공
if district_options:
    gu = st.selectbox("구/군 (측정소) 선택", district_options, index=0)
else:
    gu = st.text_input("구/군 (측정소) 입력 (목록 없음)", "")
    st.warning("선택된 시/도에 대한 측정소 목록이 없습니다.")

# PM 항목 선택 라디오 (PM10 또는 PM2.5)
pm_type = st.radio("측정 항목 선택", ('PM10', 'PM2.5'), index=0)

# 고정 파라미터: 조회 개수(24시간) 및 예측 시간(3시간)
num_rows_to_fetch = 24
n_forecast_hours = 3

# 측정소 이름(여기서는 gu 변수 사용)
station = gu

# '분석 시작' 버튼이 눌리면 아래 블록 실행
if st.button("분석 시작", key="analyze_button"):
    st.subheader(f"📊 {city} {gu} ({pm_type}) 분석 결과 (최근 {num_rows_to_fetch}시간)")

    # API에서 어떤 key를 읽을지 설정 (pm10Value 또는 pm25Value)
    data_key = 'pm10Value' if pm_type == 'PM10' else 'pm25Value'

    try:
        # Streamlit 스피너(로딩 표시) 안에서 데이터 호출
        with st.spinner(f'데이터 ({num_rows_to_fetch}개) 불러오는 중...'):
            items = fetch_air_data(station, num_rows=num_rows_to_fetch)
    except requests.HTTPError:
        # HTTP 에러일 때 사용자에게 오류 메시지 표시 후 중단
        st.error("데이터 요청 중 HTTP 오류가 발생했습니다. API 서버 상태를 확인하세요.")
        st.stop()
    except Exception as e:
        # 다른 예외일 때 메시지와 함께 중단
        st.error(f"데이터 요청 중 예상치 못한 오류 발생: {e}")
        st.stop()

    # 받은 items를 parse_pm으로 정제: (times, values) 반환
    times, values = parse_pm(items, key=data_key)

    # 호출한 개수와 실제 처리된 유효 포인트 수를 사용자에게 알림
    if items:
        st.info(f"요청한 데이터는 {num_rows_to_fetch}개, 실제 처리된 유효 데이터 포인트는 **{len(values)}**개입니다. (참고: 데이터에 **의도된 오류값(ERROR_VAL) 1개**가 포함되어 있습니다.)")

    # 선형 회귀로 예측 수행
    predict_values, predict_times, model = linear_regression_predict(times, values, n_hours=n_forecast_hours)

    # 예측 불가 조건 처리
    if predict_values is None or not values:
        predict = None
        st.warning(f"측정소 '{station}'에 대한 유효한 {pm_type} 데이터가 너무 적습니다. 예측은 불가능합니다.")
    else:
        # T+3 (마지막 예측값) 을 추천 기준으로 삼음
        predict = predict_values[-1]

    # === 그래프 그리기 세팅 ===
    fig, ax = plt.subplots(figsize=(12,7))
    criteria = get_grade_criteria(pm_type)

    # 등급별 배경색 표시: '좋음', '보통', '나쁨' 영역을 axhspan으로 표시
    ax.axhspan(criteria['좋음'][0], criteria['좋음'][1], facecolor='green', alpha=0.1, label='좋음')
    ax.axhspan(criteria['보통'][0], criteria['보통'][1], facecolor='yellow', alpha=0.1, label='보통')
    ax.axhspan(criteria['나쁨'][0], criteria['나쁨'][1], facecolor='orange', alpha=0.1, label='나쁨')

    # values에 문자열("ERROR_VAL")이 포함되어 있으면 max()에서 TypeError가 발생하므로 숫자만 필터
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    max_val = max(numeric_values) if numeric_values else 0

    # 예측값도 Y축 범위를 계산할 때 고려
    if predict_values is not None and len(predict_values) > 0:
        max_pred_val = max(predict_values)
        max_val = max(max_val, max_pred_val)

    # Y축 상한: 최대값의 1.2배 또는 '매우 나쁨' 기준의 1.2배 중 큰 쪽
    y_max_limit = max(max_val * 1.2, criteria['매우 나쁨'][0] * 1.2)

    # '매우 나쁨' 영역은 y_max_limit까지 빨간색으로 표시
    ax.axhspan(criteria['매우 나쁨'][0], y_max_limit, facecolor='red', alpha=0.1, label='매우 나쁨')

    # 그래프 배경/그리드 설정
    ax.set_facecolor('#f9f9f9')
    ax.grid(True, color='#e1e1e1', linestyle='-', linewidth=1)

    # 실제 그릴 데이터: 시간들 중 값이 숫자인 것만 사용
    plot_times = [t for t, v in zip(times, values) if isinstance(v, (int, float))]
    plot_values = numeric_values

    # 실측 데이터 선 그래프 (파란색 계열)
    ax.plot(plot_times, plot_values, color='#2a4d8f', marker='o', linewidth=2, label=f'실측 {pm_type}')

    # 각 실측 포인트 위에 값 텍스트 표시 (정수로 표시)
    for x, y in zip(plot_times, plot_values):
        ax.text(x, y + 1.5, f"{y:.0f}", color='#2a4d8f', fontsize=8, ha='center')

    # 예측값이 있으면 실측 마지막 점과 예측점들을 이어서 점선으로 표시
    if predict_values is not None and plot_times:
        plot_times_with_pred = [plot_times[-1]] + predict_times
        plot_values_with_pred = [plot_values[-1]] + list(predict_values)

        ax.plot(plot_times_with_pred, plot_values_with_pred,
                color='#f28500', marker='o', linestyle='--', linewidth=2,
                label=f'향후 {n_forecast_hours}시간 예측')

        # 마지막 예측값 텍스트 표시
        final_time = predict_times[-1]
        final_value = predict_values[-1]
        ax.text(final_time, final_value + 1.5, f"{final_value:.0f}", color='#f28500', fontsize=8, ha='center')

    # X축 눈금 설정(2시간 간격)
    xtick_interval = 2
    tick_indices = np.arange(0, len(times), xtick_interval)
    tick_times = [times[i] for i in tick_indices if i < len(times)]
    tick_labels = [t.strftime("%m-%d %H:%M") for t in tick_times]

    ax.set_xticks(tick_times)
    ax.set_xticklabels(tick_labels, rotation=45)

    # X축 범위를 실측 시작시간 ~ 마지막 예측시간으로 설정 (있을 때)
    if times and predict_times:
        start_time = times[0]
        end_time = predict_times[-1]
        ax.set_xlim(start_time, end_time)
    elif times:
        start_time = times[0]
        end_time = times[-1]
        ax.set_xlim(start_time, end_time)

    ax.set_title(f'{city} {gu} ({pm_type}) 시간대별 농도 변화 추이 (24시간 실측 + 3시간 예측)', fontsize=16, pad=20)
    ax.set_ylabel(f"{pm_type} 농도 (㎍/m³)")
    ax.set_xlabel("측정 시간")

    # 범례 표시: 한글 폰트가 있으면 prop에 넣어서 깨지지 않게 함
    if font_prop:
        ax.legend(loc='upper left', frameon=True, prop=font_prop, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    else:
        ax.legend(loc='upper left', frameon=True, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    plt.subplots_adjust(right=0.8)  # 그림 오른쪽 여백 확보 (범례 위해)
    st.pyplot(fig)  # Streamlit에 matplotlib 그림 출력

    # === 데이터 테이블 출력 ===
    if times and values:
        st.subheader("📋 실측 데이터 테이블")
        data_to_display = {
            "측정 시간": [t.strftime("%Y-%m-%d %H:%M") for t in times],
            # 값이 숫자일 때만 포맷 적용, 아니면 문자열 그대로
            f"{pm_type} 농도 (㎍/m³)": [f"{v:.1f}" if isinstance(v, (int, float)) else str(v) for v in values]
        }
        st.dataframe(data_to_display, use_container_width=True)

    # === 예측 결과 출력 ===
    st.subheader("📌 예측 결과 (향후 3시간)")

    if predict_values is not None and values:
        # 안전하게 숫자 값만 골라 마지막 숫자값을 사용
        last_numeric_value = [v for v in values if isinstance(v, (int, float))][-1]
        last_time = times[-1]

        st.markdown(f"**직전 측정값 ({last_time.strftime('%H:%M')})**: **{last_numeric_value:.1f} ㎍/m³**")
        st.markdown("---")

        for i in range(n_forecast_hours):
            current_time = predict_times[i]
            predicted_value = predict_values[i]
            change = predicted_value - last_numeric_value  # 변화량 계산

            # 변화량 기준(0.5)으로 텍스트와 색상 설정
            if change > 0.5:
                change_text = f"▲ {abs(change):.1f} ㎍/m³ 증가"
                color = "red"
            elif change < -0.5:
                change_text = f"▼ {abs(change):.1f} ㎍/m³ 감소"
                color = "blue"
            else:
                change_text = "↔ 변화 거의 없음"
                color = "gray"

            st.markdown(
                f"**{i+1}시간 뒤 ({current_time.strftime('%H:%M')})** : "
                f"예측값 **{predicted_value:.1f} ㎍/m³** "
                f"(<span style='color:{color}'>**{change_text}**</span>)",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(f"**최종 예측 ({predict_times[-1].strftime('%H:%M')}) 기준**")
        st.info(recommend_by_value(predict_values[-1], pm_type=pm_type))
    else:
        st.warning("데이터 부족으로 인해 예측값을 계산할 수 없습니다.")
