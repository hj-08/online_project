import requests # HTTP 요청 라이브러리
import json # JSON 파싱 라이브러리
import matplotlib.pyplot as plt # 그래프 시각화 모듈
import numpy as np # 숫자 배열 및 계산 모듈
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
        st.sidebar.success(f"한글 폰트 설정 완료: {font_name}") # 성공 메시지 출력
        
        font_prop = fm.FontProperties(family=font_name) # 폰트 속성 객체 생성
    else: # 폰트 검색 실패 시
        font_name = "DejaVu Sans" # 기본 영문 폰트 사용
        st.sidebar.warning(f"적절한 한글 폰트를 찾을 수 없어. 기본 폰트({font_name}) 사용.") # 경고 메시지 출력
        font_prop = None # 폰트 속성 없음

    plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지 재확인
    return font_prop # 폰트 속성 반환

font_prop = set_korean_font() # 폰트 설정 함수 실행


# --- 미세먼지 공공 데이터 API 키 ---
API_KEY = "aea45d5692f9dc0fb20ff49e2cf104f6614d3a17df9e92420974a5defb3cd75e" # API 인증 키

def fetch_air_data(station_name, num_rows=48): # API 데이터 요청 함수
    """주어진 '측정소 이름'의 미세먼지 데이터를 API로 요청하고 받아오는 함수."""
    URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty" # API 엔드포인트 URL
    params = { # API 요청에 필요한 파라미터 설정
        'serviceKey': API_KEY, 
        'returnType': 'json', 
        'numOfRows': num_rows, 
        'stationName': station_name, 
        'dataTerm': 'DAILY',
        'ver': '1.3'
    }
    
    r = requests.get(URL, params=params, timeout=10) # API 요청 및 응답 받기
    r.raise_for_status() # HTTP 오류 발생 시 예외 처리
    
    data = r.json() # JSON 응답을 딕셔너리로 변환
    items = data['response']['body']['items'] # 실제 측정 데이터 목록 추출
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

def linear_regression_predict(values): # 선형 회귀 예측 함수
    """선형 회귀 모델로 다음 1시간 뒤의 농도 값을 예측하는 함수."""
    if len(values) < 3: # 데이터 부족 시 예측 불가
        return None
        
    X = np.arange(len(values)).reshape(-1,1) # X축(시간 인덱스) 데이터 준비
    y = np.array(values) # Y축(농도 값) 데이터 준비
    
    model = LinearRegression().fit(X, y) # 선형 회귀 모델 학습
    
    pred = model.predict([[len(values)]])[0] # 다음 시점 값 예측
    return pred # 예측값 반환

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

# 사이드바에 메뉴를 만들어서 페이지를 전환할 수 있게 함
st.sidebar.title("메인 메뉴") # 사이드바 제목
page = st.sidebar.radio("원하는 페이지를 골라봐", # 페이지 선택 라디오 버튼
                        ["분석 및 예측", "미세먼지 정보"], 
                        index=0)

if page == "분석 및 예측":
    # --- 기존 분석 페이지 UI ---
    st.title("🌫️ 실시간 미세먼지 분석 + 예측") # 웹 앱 제목
    st.markdown("정부 공공데이터 포털의 실시간 미세먼지 데이터를 기반으로 해.") # 설명 텍스트

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
        st.warning("선택된 시/도에 대한 측정소 목록이 없어. 직접 입력해.") # 경고 메시지

    pm_type = st.radio("측정 항목 선택", ('PM10', 'PM2.5'), index=0) # 측정 항목 라디오 버튼

    data_range = st.selectbox("데이터 조회 기간", # 데이터 조회 기간 선택
                            ['최근 48시간', '지난 7일 (168시간)', '지난 30일 (720시간)'],
                            index=0)
        
    station = gu # 측정소 이름 설정

    if st.button("분석 시작", key="analyze_button"): # '분석 시작' 버튼 클릭 시
        st.subheader(f"📊 {city} {gu} ({pm_type}) 분석 결과") # 분석 결과 부제목 출력
        
        data_key = 'pm10Value' if pm_type == 'PM10' else 'pm25Value' # API 요청을 위한 데이터 키 설정
        
        num_rows_to_fetch = 48 # 기본 요청 데이터 개수 설정
        if data_range == '지난 7일 (168시간)':
            num_rows_to_fetch = 168
        elif data_range == '지난 30일 (720시간)':
            num_rows_to_fetch = 720 
        
        try: # 데이터 요청 및 오류 처리
            with st.spinner(f'데이터 ({num_rows_to_fetch}개) 불러오는 중...'): # 로딩 스피너 표시
                items = fetch_air_data(station, num_rows=num_rows_to_fetch) # 데이터 가져오기
            st.success("데이터 불러오기 성공!") # 성공 메시지
        except requests.HTTPError: # HTTP 오류 처리
            st.error("데이터 요청 중 HTTP 오류가 발생했어. 지역명 또는 API 키를 확인해.")
            st.stop() # 프로그램 중지
        except Exception as e: # 기타 오류 처리
            st.error(f"데이터 요청 중 예상치 못한 오류 발생: {e}")
            st.stop() 

        times, values = parse_pm(items, key=data_key) # 데이터 파싱

        if not values: # 유효한 데이터가 없을 경우
            st.warning(f"측정소 '{station}'에 대한 유효한 {pm_type} 데이터가 없어. 지역명을 다시 확인해.")
            st.stop() # 프로그램 중지
            
        if num_rows_to_fetch <= 48: # 단기 조회 시 예측 실행
            predict = linear_regression_predict(values) # 예측값 계산
        else: # 장기 조회 시 예측 비활성화
            predict = None
            st.warning("장기 데이터 조회 시에는 예측 기능이 비활성화돼.")

        fig, ax = plt.subplots(figsize=(14, 7)) # 그래프 영역 설정
        criteria = get_grade_criteria(pm_type) # 등급 기준 가져오기
        
        # 등급별 배경색 영역 표시 (좋음, 보통, 나쁨)
        ax.axhspan(criteria['좋음'][0], criteria['좋음'][1], facecolor='green', alpha=0.1, label='좋음')
        ax.axhspan(criteria['보통'][0], criteria['보통'][1], facecolor='yellow', alpha=0.1, label='보통')
        ax.axhspan(criteria['나쁨'][0], criteria['나쁨'][1], facecolor='orange', alpha=0.1, label='나쁨')
        
        max_val = max(values) if values else 0 # 데이터 최대값
        y_max_limit = max(max_val, criteria['매우 나쁨'][0]) * 1.5 # Y축 최대 범위 설정
        
        ax.set_ylim(0, y_max_limit) # Y축 범위 적용
        
        ax.axhspan(criteria['매우 나쁨'][0], y_max_limit, facecolor='red', alpha=0.1, label='매우 나쁨') # 매우 나쁨 영역 표시

        ax.set_facecolor('#f9f9f9') # 그래프 배경색 설정
        ax.grid(True, color='#e1e1e1', linestyle='-', linewidth=1) # 그리드 선 추가
        
        ax.plot(times, values, color='#2a4d8f', marker='o', linewidth=2, label=f'실측 {pm_type}') # 실측 데이터 선 그래프
        
        if num_rows_to_fetch <= 48: # 단기 조회 시 값 텍스트 표시
            for x, y in zip(times, values):
                ax.text(x, y + 1.5, f"{y:.0f}", color='#2a4d8f', fontsize=8, ha='center') # 각 점 위에 농도 값 표시

        if predict is not None: # 예측값이 있을 경우
            next_time = times[-1] + timedelta(hours=1) # 예측 시간 (마지막 시간 + 1시간)
            ax.plot([times[-1], next_time], 
                    [values[-1], predict], 
                    color='#f28500', marker='o', linestyle='--', linewidth=2, 
                    label=f'예측값: {predict:.1f}') # 예측값 점선으로 표시
            ax.text(next_time, predict + 1.5, f"{predict:.0f}", color='#f28500', fontsize=8, ha='center') # 예측값 텍스트 표시

        # X축 눈금 간격 설정
        if num_rows_to_fetch <= 48:
            xtick_interval = 2 # 2시간 간격
        elif num_rows_to_fetch <= 168:
            xtick_interval = 12 # 12시간 간격
        else:
            xtick_interval = 24 # 24시간 간격

        tick_indices = np.arange(0, len(times), xtick_interval) # 눈금 인덱스 계산
        tick_times = [times[i] for i in tick_indices if i < len(times)] # 눈금 시간 객체 추출
        
        # X축 눈금 레이블 형식 설정
        if num_rows_to_fetch <= 48:
            tick_labels = [t.strftime("%m-%d %H:%M") for t in tick_times] # 월-일 시:분
        else:
            tick_labels = [t.strftime("%Y-%m-%d") for t in tick_times] # 년-월-일

        ax.set_xticks(tick_times) # X축 눈금 위치 설정
        ax.set_xticklabels(tick_labels, rotation=45) # X축 레이블 표시 및 45도 회전

        ax.set_title(f'{city} {gu} ({pm_type}) 시간대별 농도 변화 추이', fontsize=16, pad=20) # 그래프 제목
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


        st.subheader("📌 예측 결과") # 예측 결과 부제목
        if predict is not None: # 예측값이 있을 경우
            st.markdown(f"다음 {pm_type} 예측값: **{predict:.1f} ㎍/m³**") # 예측 농도 값 출력
            st.info(recommend_by_value(predict, pm_type=pm_type)) # 행동 추천 메시지 출력
        else: # 예측값이 없을 경우
            st.warning("데이터 부족 또는 장기 조회로 인해 예측값을 계산할 수 없어.") # 경고 메시지

elif page == "미세먼지 정보":
    # --- 미세먼지 정보 페이지 UI: PM2.5/PM10 차이, 등급 기준, 대처 방안 ---
    st.title("💡 미세먼지 상식과 대처법")
    st.markdown("미세먼지 (PM10)와 초미세먼지 (PM2.5)가 뭔지, 그리고 어떻게 대처해야 하는지 알려줄게.")

    st.subheader("1. PM10과 PM2.5, 뭐가 다를까? (입자 크기 차이)")
    st.markdown("미세먼지는 입자의 크기로 구별돼. 작을수록 우리 몸속 깊숙이 들어와서 더 위험해.")
    
    # PM2.5와 PM10 크기 비교를 위한 이미지 태그 추가 (시각적 이해 도움)
    st.markdown("[미세먼지, 초미세먼지, 머리카락 크기 비교 이미지]") 
    
    col1, col2 = st.columns(2)
    with col1:
        st.error("🔴 PM2.5 (초미세먼지)")
        st.markdown(f"""
        * **크기:** 2.5µm(마이크로미터) 이하. 머리카락 지름의 약 1/30 수준이야. 
        * **위험도:** **매우 높음.** 폐포를 통과해 혈관까지 침투할 수 있어.
        * **주요 성분:** 주로 화석연료 연소나 공장에서 배출된 화학 물질이 많아.
        """)
        
    with col2:
        st.warning("🟠 PM10 (미세먼지)")
        st.markdown(f"""
        * **크기:** 10µm 이하.
        * **위험도:** **높음.** 기관지에서 걸러지기도 하지만, 일부는 폐까지 들어갈 수 있어.
        * **주요 성분:** 황사, 비산먼지(바람에 날리는 흙먼지) 등 자연 발생 물질도 섞여 있어.
        """)
        
    st.subheader("2. 미세먼지 등급 기준은? (보통, 나쁨 등 차이)")
    st.markdown("나라에서 정한 기준으로, 농도에 따라 **좋음**, **보통**, **나쁨**, **매우 나쁨** 4단계로 나눠.")
    
    # PM10 기준 테이블 (보통/나쁨 등 차이 명확히 보여줌)
    st.markdown("#### PM10 기준 (μg/m³)")
    pm10_data = []
    for grade, (start, end) in PM10_CRITERIA.items():
        pm10_data.append([grade, f"{start}~{end}"])
    
    st.table(pm10_data)

    # PM2.5 기준 테이블
    st.markdown("#### PM2.5 기준 (μg/m³)")
    pm25_data = []
    for grade, (start, end) in PM25_CRITERIA.items():
        pm25_data.append([grade, f"{start}~{end}"])
    
    st.table(pm25_data)
    
    st.subheader("3. 미세먼지 줄이기 위한 우리들의 방안")
    st.markdown("미세먼지는 우리 모두의 노력으로 줄일 수 있어. 생활 속에서 실천할 수 있는 방법을 알아보자!")
    
    st.markdown("""
    * **🚗 대중교통 이용:** 승용차 사용을 줄이고 대중교통이나 자전거를 타는 건 어때?
    * **🔌 불필요한 전기 끄기:** 전기를 만들 때도 미세먼지가 나오니까, 안 쓰는 플러그는 뽑고 절약하자.
    * **🌿 나무 심기:** 나무는 미세먼지를 흡수하고 공기를 깨끗하게 해주는 자연 필터야.
    * **💨 환기 습관:** 실내 공기 질도 중요해! 짧게라도 창문을 열어 환기시키고, 물걸레 청소를 자주 하자.
    * **😷 마스크 착용:** 예보가 '나쁨' 이상이면 외출할 때 KF94 같은 보건용 마스크를 꼭 써야 해.
    """)
