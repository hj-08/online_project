import requests # HTTP 요청 라이브러리
import json # JSON 파싱 라이브러리
import matplotlib.pyplot as plt # 그래프 시각화 모듈
import numpy as np # 숫자 배열 및 계산 모듈 (np로 통일)
import streamlit as st # 웹 앱 UI 구축 모듈
from datetime import datetime, timedelta # 날짜/시간 처리 모듈
from sklearn.linear_model import LinearRegression # 선형 회귀 예측 모델
import matplotlib.font_manager as fm # 폰트 관리 모듈
import os # 기본 OS 모듈 (여기서는 사용되지 않음)

# --- 한글 폰트 설정 함수 정의 ---
def set_korean_font(): # 한글 폰트 설정 메인 함수
    """그래프에서 한글 깨짐을 방지하고 폰트를 설정하는 메인 함수야."""
    
    # 폰트 이름을 찾는 함수
    def find_font_name(): # 폰트 이름 검색 도우미 함수
        """시스템에 설치된 한글 폰트 이름을 찾아서 돌려줘."""
        
        # 컴퓨터에 설치된 폰트 목록 확인
        font_list = [f.name for f in fm.fontManager.ttflist] # 폰트 이름 리스트 추출
        
        # 한글 폰트 이름을 검색
        for name in ["NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
            # 한글 폰트 이름을 검색
            if name in font_list:
                return name # 폰트 이름 반환
        return None # 폰트 찾기 실패

    font_name = find_font_name() 
    
    # 폰트를 찾았을때 실행되는 코드
    if font_name: # 폰트 검색 성공 시 설정
        plt.rcParams['font.family'] = font_name # Matplotlib 폰트 설정
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
        
        font_prop = fm.FontProperties(family=font_name) # 폰트 속성 객체 생성
    else: # 폰트 검색 실패 시
        font_name = "DejaVu Sans" # 기본 영문 폰트 사용
        st.sidebar.warning(f"적절한 한글 폰트를 찾을 수 없습니다. 기본 폰트({font_name}) 사용.") # 경고 메시지 출력
        font_prop = None # 폰트 속성 없음

    plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지 재확인
    return font_prop # 폰트 속성 반환

font_prop = set_korean_font() # 폰트 설정 함수 실행


# --- 미세먼지 공공 데이터 API 키 ---
API_KEY = "aea45d5692f9dc0fb20ff49e2cf104f6614d3a17df9e92420974a5defb3cd75e" # API 인증 키

def fetch_air_data(station_name, num_rows=24): # API 데이터 요청 함수 (기본값 24시간)
    """주어진 '측정소 이름'의 미세먼지 데이터를 API로 요청하고 받아오는 함수."""
    URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty" # API 엔드포인트 URL
    params = { # API 요청에 필요한 파라미터 설정
        'serviceKey': API_KEY, 
        'returnType': 'json', 
        'numOfRows': num_rows, # 요청 데이터 개수 (24개로 고정)
        'stationName': station_name, 
        'dataTerm': 'DAILY',
        'ver': '1.3'
    }
    
    r = requests.get(URL, params=params, timeout=10) # API 요청 및 응답 받기
    r.raise_for_status() # HTTP 오류 발생 시 예외 처리
    
    data = r.json() # JSON 응답을 딕셔너리로 변환
    
    items = data['response']['body']['items'] 
    
    return items # 데이터 목록 반환

def parse_pm(items, key='pm10Value'): # 데이터 파싱 및 정제 함수
    """API 데이터에서 '시간'과 '농도 값'만 골라내어 정리하는 함수."""
    times = [] # 시간 정보를 저장할 리스트
    values = [] # 농도 값을 저장할 리스트
    
    for it in items: # 데이터 항목 반복 처리
        t = it.get('dataTime') # 측정 시간 추출
        val = it.get(key) # 농도 값 추출
        
        try: # 값 변환 시도
            v = float(val) # 농도 값을 실수형으로 변환
        except: # 변환 실패 시
            continue # 다음 항목으로 건너뛰기
        
        dt = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y%m%d%H%M"): # 시간 형식 반복 시도
            try:
                dt = datetime.strptime(t, fmt) # datetime 객체로 변환
                break # 성공 시 반복 중단
            except:
                continue # 실패 시 다음 형식 시도
        
        if dt is None: # 시간 변환 최종 실패 시
            continue # 다음 항목으로 건너뛰기
        
        times.append(dt) # 유효한 시간 추가
        values.append(v) # 유효한 값 추가
        
    return times[::-1], values[::-1] # 시간 순서대로 뒤집어 반환

def linear_regression_predict(times, values, n_hours=3): # 선형 회귀 다중 예측 함수
    """선형 회귀 모델로 다음 n_hours 시간 뒤의 농도 값들을 예측하고, 해당 시간대 리스트와 함께 반환하는 함수."""
    if len(values) < 3: # 데이터 부족 시 예측 불가
        return None, None, None
        
    X = np.arange(len(values)).reshape(-1,1) # X축(시간 인덱스) 데이터 준비
    y = np.array(values) # Y축(농도 값) 데이터 준비
    
    model = LinearRegression().fit(X, y) # 선형 회귀 모델 학습
    
    # Predict n_hours points (T+1, T+2, ..., T+n)
    X_pred = np.arange(len(values), len(values) + n_hours).reshape(-1, 1)
    predict_values = model.predict(X_pred)
    
    # 예측값이 음수가 되지 않도록 최소값을 1.0으로 설정 (사용자 요청 반영)
    predict_values = np.maximum(1.0, predict_values)
    
    # Calculate the future times
    last_time = times[-1]
    predict_times = [last_time + timedelta(hours=i) for i in range(1, n_hours + 1)]
    
    return predict_values, predict_times, model # 예측값 배열, 예측 시간 배열, 모델 객체 반환

# --- 미세먼지 등급 기준 정의 ---
PM10_CRITERIA = { # PM10 기준 정의
    '좋음': (0, 30),
    '보통': (31, 80),
    '나쁨': (81, 150),
    '매우 나쁨': (151, float('inf')) 
}
PM25_CRITERIA = { # PM2.5 기준 정의
    '좋음': (0, 15),
    '보통': (16, 35),
    '나쁨': (36, 75),
    '매우 나쁨': (76, float('inf')) 
}

def get_grade_criteria(pm_type): # 등급 기준 반환 함수
    """'PM10'인지 'PM2.5'인지에 따라 알맞은 등급 기준 딕셔너리를 돌려줘."""
    return PM10_CRITERIA if pm_type == 'PM10' else PM25_CRITERIA # 기준 딕셔너리 반환

def recommend_by_value(val, pm_type='PM10'): # 행동 추천 메시지 함수
    """농도 값에 따라 행동 추천 메시지를 돌려주는 함수."""
    if val is None:
        return "예측값을 계산할 수 없어." # 예측 불가 시 메시지
    
    criteria = get_grade_criteria(pm_type) # 해당 PM 타입의 기준 가져오기
        
    # 등급별 조건 확인 및 메시지 반환 (매우 나쁨부터 시작)
    if val >= criteria['매우 나쁨'][0]:
        return "🔥 매우 나쁨: 외출 자제, 실내 활동 권장"
    if val >= criteria['나쁨'][0]:
        return "⚠️ 나쁨: 장시간 외출 피하고 마스크 착용"
    if val >= criteria['보통'][0]:
        return "🙂 보통: 민감군은 주의, 가벼운 외출 가능"
        
    return "🌿 좋음: 외부 활동 안전" # 좋음 등급 메시지

# --- Streamlit 웹 화면(UI) 구성 시작 ---

st.title("🌫️ 실시간 미세먼지 분석 + 예측 (최근 24시간)") # 웹 앱 제목 수정
st.markdown("정부 공공데이터 포털의 실시간 미세먼지 데이터를 기반으로 합니다다. **예측은 향후 3시간을 기준으로 합니다.**") # 설명 텍스트

AIR_STATION_MAP = { # 시/도별 측정소 목록 정의
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
city = st.selectbox("시/도 선택", list(AIR_STATION_MAP.keys()), # 시/도 선택 드롭다운
                     index=list(AIR_STATION_MAP.keys()).index(default_city) if default_city in AIR_STATION_MAP else 0)

district_options = AIR_STATION_MAP.get(city, []) # 선택된 시/도의 구/군 목록 가져오기

if district_options: # 구/군 목록이 있을 경우
    gu = st.selectbox("구/군 (측정소) 선택", district_options, index=0) # 구/군 선택 드롭다운
else: # 구/군 목록이 없을 경우
    gu = st.text_input("구/군 (측정소) 입력 (목록 없음)", "") # 수동 입력창
    st.warning("선택된 시/도에 대한 측정소 목록이 없습니다.") # 경고 메시지

pm_type = st.radio("측정 항목 선택", ('PM10', 'PM2.5'), index=0) # 측정 항목 라디오 버튼

# 데이터 조회 기간은 '최근 24시간'으로 고정
num_rows_to_fetch = 24
n_forecast_hours = 3 # 예측 시간: 3시간으로 확장

station = gu # 측정소 이름 설정

if st.button("분석 시작", key="analyze_button"): # '분석 시작' 버튼 클릭 시
    st.subheader(f"📊 {city} {gu} ({pm_type}) 분석 결과 (최근 {num_rows_to_fetch}시간)") # 분석 결과 부제목 출력
    
    data_key = 'pm10Value' if pm_type == 'PM10' else 'pm25Value' # API 요청을 위한 데이터 키 설정
    
    try: # 데이터 요청 및 오류 처리
        with st.spinner(f'데이터 ({num_rows_to_fetch}개) 불러오는 중...'): # 로딩 스피너 표시
            items = fetch_air_data(station, num_rows=num_rows_to_fetch) # 데이터 가져오기
        st.success("데이터 불러오기 성공!") # 성공 메시지
    except requests.HTTPError: # HTTP 오류 처리
        st.error("데이터 요청 중 HTTP 오류가 발생했습니다. API 서버 상태를 확인하세요.")
        st.stop()
    except Exception as e: # 기타 오류 처리
        st.error(f"데이터 요청 중 예상치 못한 오류 발생: {e}")
        st.stop() 

    times, values = parse_pm(items, key=data_key) # 데이터 파싱

    # 데이터 처리 개수 확인 메시지
    if items:
        st.info(f"요청한 데이터는 {num_rows_to_fetch}개, 실제 처리된 유효 데이터 포인트는 **{len(values)}**개입니다.")
    
    # 예측 실행
    predict_values, predict_times, model = linear_regression_predict(times, values, n_hours=n_forecast_hours)

    if predict_values is None or not values:
        predict = None
        st.warning(f"측정소 '{station}'에 대한 유효한 {pm_type} 데이터가 너무 적습니다. 예측은 불가능합니다.")
    else:
        # 최종 예측값 (T+3)을 추천 기준으로 사용
        predict = predict_values[-1]


    fig, ax = plt.subplots(figsize=(12,7)) # 그래프 영역 설정
    criteria = get_grade_criteria(pm_type) # 등급 기준 가져오기
    
    # 등급별 배경색 영역 표시 (좋음, 보통, 나쁨)
    ax.axhspan(criteria['좋음'][0], criteria['좋음'][1], facecolor='green', alpha=0.1, label='좋음')
    ax.axhspan(criteria['보통'][0], criteria['보통'][1], facecolor='yellow', alpha=0.1, label='보통')
    ax.axhspan(criteria['나쁨'][0], criteria['나쁨'][1], facecolor='orange', alpha=0.1, label='나쁨')
    
    max_val = max(values) if values else 0 # 데이터 최대값
    # 예측값 중 최대값도 포함하여 Y축 최대 범위를 계산
    if predict_values is not None and len(predict_values) > 0:
        max_pred_val = max(predict_values)
        max_val = max(max_val, max_pred_val)

    y_max_limit = max(max_val * 1.2, criteria['매우 나쁨'][0] * 1.2) # Y축 최대 범위 설정 (넉넉하게)
    
    # Y축 최소값도 0 대신 1로 시작하는 것을 고려할 수 있지만, 그래프의 시각적 연속성을 위해 0부터 시작하도록 유지
    ax.set_ylim(0, y_max_limit) # Y축 범위 적용
    
    ax.axhspan(criteria['매우 나쁨'][0], y_max_limit, facecolor='red', alpha=0.1, label='매우 나쁨') # 매우 나쁨 영역 표시

    ax.set_facecolor('#f9f9f9') # 그래프 배경색 설정
    ax.grid(True, color='#e1e1e1', linestyle='-', linewidth=1) # 그리드 선 추가
    
    ax.plot(times, values, color='#2a4d8f', marker='o', linewidth=2, label=f'실측 {pm_type}') # 실측 데이터 선 그래프
    
    # 24시간 데이터에 대해 값 텍스트 표시
    for x, y in zip(times, values):
        try:
            # 숫자일 때만 레이블 표시 시도 
            if isinstance(y, (int, float)):
                 ax.text(x, y + 1.5, f"{y:.0f}", color='#2a4d8f', fontsize=8, ha='center') 
        except:
            pass

    if predict_values is not None: # 예측값이 있을 경우
        # Combine the last real point with the predicted points for plotting
        plot_times = [times[-1]] + predict_times
        plot_values = [values[-1]] + list(predict_values)
        
        ax.plot(plot_times, plot_values, 
                color='#f28500', marker='o', linestyle='--', linewidth=2, 
                label=f'향후 {n_forecast_hours}시간 예측') 

        # Display the final predicted value text (T+3)
        final_time = predict_times[-1]
        final_value = predict_values[-1]
        ax.text(final_time, final_value + 1.5, f"{final_value:.0f}", color='#f28500', fontsize=8, ha='center')

    # X축 눈금 간격 설정 (24시간 데이터에 대해 2시간 간격으로 고정)
    xtick_interval = 2 # 2시간 간격
        
    tick_indices = np.arange(0, len(times), xtick_interval) # 눈금 인덱스 계산
    tick_times = [times[i] for i in tick_indices if i < len(times)] # 눈금 시간 객체 추출
    
    # X축 눈금 레이블 형식 설정 (월-일 시:분)
    tick_labels = [t.strftime("%m-%d %H:%M") for t in tick_times] 

    ax.set_xticks(tick_times) # X축 눈금 위치 설정
    ax.set_xticklabels(tick_labels, rotation=45) # X축 레이블 표시 및 45도 회전
    
    # === X축 범위 강제 설정 ===
    if times and predict_times:
        start_time = times[0] # 첫 측정 시간
        end_time = predict_times[-1] # 마지막 예측 시간 (T+3)
        
        # X축 범위를 명시적으로 설정하여 실측+예측 기간 전체를 표시합니다.
        ax.set_xlim(start_time, end_time) 
    elif times:
         start_time = times[0]
         end_time = times[-1]
         ax.set_xlim(start_time, end_time)
    # ========================

    ax.set_title(f'{city} {gu} ({pm_type}) 시간대별 농도 변화 추이 (24시간 실측 + 3시간 예측)', fontsize=16, pad=20) # 그래프 제목
    ax.set_ylabel(f"{pm_type} 농도 (㎍/m³)") # Y축 레이블
    ax.set_xlabel("측정 시간") # X축 레이블
    
    if font_prop: # 폰트 속성이 있으면
        ax.legend(loc='upper left', frameon=True, prop=font_prop, bbox_to_anchor=(1.01, 1), borderaxespad=0.) # 범례 표시 (한글 폰트 적용)
    else:
        ax.legend(loc='upper left', frameon=True, bbox_to_anchor=(1.01, 1), borderaxespad=0.) # 범례 표시 (기본 폰트)
        
    plt.subplots_adjust(right=0.8) # 그래프 오른쪽 여백 조정

    st.pyplot(fig) # 그래프를 Streamlit에 출력
    
    if times and values: # 실측 데이터가 있을 경우
        st.subheader("📋 실측 데이터 테이블") # 테이블 부제목
        data_to_display = { # 데이터 프레임용 딕셔너리
            "측정 시간": [t.strftime("%Y-%m-%d %H:%M") for t in times],
            f"{pm_type} 농도 (㎍/m³)": [f"{v:.1f}" for v in values]
        }
        st.dataframe(data_to_display, use_container_width=True) # 데이터 프레임 출력


    st.subheader("📌 예측 결과 (향후 3시간)") # 예측 결과 부제목
    
    if predict_values is not None and values: # 예측값과 실측값이 모두 있을 경우
        last_value = values[-1] # 직전 측정값
        st.markdown(f"**직전 측정값 ({times[-1].strftime('%H:%M')})**: **{last_value:.1f} ㎍/m³**")
        st.markdown("---")
        
        for i in range(n_forecast_hours):
            current_time = predict_times[i]
            predicted_value = predict_values[i]
            change = predicted_value - last_value
            
            # 변화량에 따른 아이콘과 색상 설정
            if change > 0.5: # 0.5 초과 시 증가
                change_text = f"▲ {abs(change):.1f} ㎍/m³ 증가"
                color = "red"
            elif change < -0.5: # -0.5 미만 시 감소
                change_text = f"▼ {abs(change):.1f} ㎍/m³ 감소"
                color = "blue"
            else: # 그 외 (거의 변화 없음)
                change_text = "↔ 변화 거의 없음"
                color = "gray"
            
            st.markdown(
                f"**{i+1}시간 뒤 ({current_time.strftime('%H:%M')})** : "
                f"예측값 **{predicted_value:.1f} ㎍/m³** "
                f"(<span style='color:{color}'>**{change_text}**</span>)",
                unsafe_allow_html=True
            )

        st.markdown("---")
        # 최종 (3시간 뒤) 예측값을 기준으로 한 행동 추천
        st.markdown(f"**최종 예측 ({predict_times[-1].strftime('%H:%M')}) 기준**")
        st.info(recommend_by_value(predict_values[-1], pm_type=pm_type))
    else:
        st.warning("데이터 부족으로 인해 예측값을 계산할 수 없습니다.") # 경고 메시지
