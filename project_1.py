import requests # 웹에서 데이터를 가져오기 위한 라이브러리
import json # JSON 형식 데이터를 다루기 위한 라이브러리
import matplotlib.pyplot as plt # 그래프를 그리기 위한 라이브러리
import numpy as np # 수학적 계산(배열 처리)을 위한 라이브러리
import streamlit as st # 웹 애플리케이션 UI를 만들기 위한 라이브러리
from datetime import datetime, timedelta # 날짜와 시간 계산을 위한 모듈
from sklearn.linear_model import LinearRegression # 선형 회귀 모델(예측)을 위한 모듈
import matplotlib.font_manager as fm # Matplotlib의 폰트 설정을 위한 모듈
import os # 운영체제 기능(파일 경로 등)을 다루기 위한 모듈

# --- 한글 폰트 설정 수정: packages.txt를 통해 NanumGothic을 설치하도록 가정 ---
def set_korean_font():
    """시스템에 설치된 한글 폰트를 찾아 Matplotlib에 설정합니다."""
    
    # 폰트 검색 로직을 별도 함수로 분리
    def find_font_name():
        font_list = [f.name for f in fm.fontManager.ttflist]
        
        # NanumGothic 계열, Noto Sans, Malgun Gothic 순으로 검색하여 사용 가능한 폰트를 찾습니다.
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
            # 폰트 캐시 파일 삭제 후
            for filename in os.listdir(cache_dir):
                if filename.startswith('fontlist-'):
                    os.remove(os.path.join(cache_dir, filename))
            
            fm.fontManager._rebuild() # 폰트 매니저를 재구축하고
            font_name = find_font_name() # 다시 폰트 이름 찾기 시도
        except Exception:
            pass # 권한 오류 등 무시

    # 4. 최종 기본 폰트 설정
    if not font_name:
        font_name = "DejaVu Sans"
        # 한글 폰트가 없을 경우 경고 메시지와 함께 기본 폰트(영어 전용)를 사용합니다.
        st.sidebar.warning(f"적절한 한글 폰트를 찾을 수 없습니다. 기본 폰트({font_name}) 사용. (Streamlit Cloud 사용 시 'packages.txt'에 'fonts-nanum' 추가 및 **재배포** 필요)")
        font_prop = None
    else:
        # 찾은 폰트로 Matplotlib 설정
        plt.rcParams['font.family'] = font_name # 폰트 패밀리 설정
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
        st.sidebar.success(f"한글 폰트 설정 완료: {font_name}") # 성공 메시지 출력
        # font_prop 생성 (범례 등 특정 요소에 폰트를 적용하기 위해 필요)
        font_prop = fm.FontProperties(family=font_name)

    plt.rcParams['axes.unicode_minus'] = False # 최종 설정 확인 (마이너스 깨짐 방지)
    return font_prop

# 폰트 설정 실행 및 font_prop 변수에 저장
font_prop = set_korean_font()


# --- API KEY (공개 API 키이므로 그대로 사용) ---
API_KEY = "aea45d5692f9dc0fb20ff49e2cf104f6614d3a17df9e92420974a5defb3cd75e" # 미세먼지 공공 데이터 포털 API 키

def fetch_air_data(station_name, num_rows=48):
    """실시간 측정소별 미세먼지 데이터를 가져옵니다."""
    URL = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty" # API 엔드포인트
    params = {
        'serviceKey': API_KEY, # 인증키
        'returnType': 'json', # 응답 데이터 형식
        'numOfRows': num_rows, # 조회할 데이터 개수 (시간당 1개)
        'pageNo': 1, # 페이지 번호
        'stationName': station_name, # 측정소 이름
        'dataTerm': 'DAILY', # 데이터 기간 (하루 단위)
        'ver': '1.3' # API 버전
    }
    
    # API 제약사항 처리: 최대 1000개까지만 데이터를 조회할 수 있습니다.
    if num_rows > 1000:
        st.error("API의 제약으로 인해 최대 1000개까지만 데이터를 조회할 수 있습니다.")
        params['numOfRows'] = 1000 # 최대치로 제한
        
    # API 요청을 보내고 응답을 받습니다.
    r = requests.get(URL, params=params, timeout=10)
    r.raise_for_status() # HTTP 오류(4xx, 5xx) 발생 시 예외 발생
    
    # JSON 응답을 파싱합니다.
    data = r.json()
    items = data['response']['body']['items'] # 실제 측정 데이터 목록
    return items

def parse_pm(items, key='pm10Value'):
    """데이터 항목 리스트에서 시간과 PM 값을 파싱하여 시간 순서대로 반환합니다."""
    times = [] # 시간 리스트
    values = [] # 미세먼지 값 리스트
    for it in items:
        t = it.get('dataTime') # 측정 시간 문자열
        val = it.get(key) # 미세먼지 값
        try:
            v = float(val) # 값을 실수형으로 변환 시도
        except:
            continue # 변환 실패 시 (예: 값이 '-'인 경우) 건너뜁니다.
        
        dt = None
        # 다양한 시간 형식 처리 시도 (API 데이터 형식 변화에 대비)
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d%H%M"):
            try:
                dt = datetime.strptime(t, fmt) # 시간 문자열을 datetime 객체로 변환
                break
            except:
                continue
        
        if dt is None:
            continue # 시간 파싱 실패 시 건너뜁니다.
        
        times.append(dt)
        values.append(v)
        
    return times[::-1], values[::-1] # API는 최신순으로 데이터를 주므로, 시간순서(오래된 순)로 반전하여 반환합니다.

def linear_regression_predict(values):
    """선형 회귀를 사용하여 다음 시점의 값을 예측합니다."""
    if len(values) < 3: # 최소 3개 이상의 데이터가 있어야 예측 가능
        return None
        
    # X: 독립 변수 (데이터의 순서, 0, 1, 2, ...)
    X = np.arange(len(values)).reshape(-1,1)
    # y: 종속 변수 (미세먼지 농도 값)
    y = np.array(values)
    
    # 선형 회귀 모델 훈련
    model = LinearRegression().fit(X, y)
    
    # 다음 시점 (인덱스 len(values))의 값을 예측
    pred = model.predict([[len(values)]])[0] 
    return pred

# PM10 등급 기준 (한국 환경부 기준)
PM10_CRITERIA = {
    '좋음': (0, 30),
    '보통': (31, 80),
    '나쁨': (81, 150),
    '매우 나쁨': (151, float('inf')) # 무한대
}

# PM2.5 등급 기준 (한국 환경부 기준)
PM25_CRITERIA = {
    '좋음': (0, 15),
    '보통': (16, 35),
    '나쁨': (36, 75),
    '매우 나쁨': (76, float('inf'))
}

def get_grade_criteria(pm_type):
    """PM 타입에 맞는 기준(PM10 또는 PM2.5)을 반환합니다."""
    return PM10_CRITERIA if pm_type == 'PM10' else PM25_CRITERIA

def recommend_by_value(val, pm_type='PM10'):
    """PM 값과 타입에 따라 행동 추천 등급과 메시지를 반환합니다."""
    if val is None:
        return "예측값을 계산할 수 없습니다."
    
    criteria = get_grade_criteria(pm_type)
        
    # 매우 나쁨 기준
    if val >= criteria['매우 나쁨'][0]:
        return "🔥 매우 나쁨: 외출 자제, 실내 활동 권장"
    # 나쁨 기준
    if val >= criteria['나쁨'][0]:
        return "⚠️ 나쁨: 장시간 외출 피하고 마스크 착용"
    # 보통 기준
    if val >= criteria['보통'][0]:
        return "🙂 보통: 민감군은 주의, 가벼운 외출 가능"
        
    # 좋음 기준
    return "🌿 좋음: 외부 활동 안전"

# --- Streamlit UI 구성 ---

st.title("🌫️ 실시간 미세먼지 분석 + 예측")
st.markdown("정부 공공데이터 포털의 실시간 미세먼지 데이터를 기반으로 합니다.")

# 주요 도시/측정소 매핑 (전국 주요 지역 확장)
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

# 1. 시/도 선택 (드롭다운)
default_city = "서울"
city = st.selectbox("시/도 선택", list(AIR_STATION_MAP.keys()), 
                    index=list(AIR_STATION_MAP.keys()).index(default_city) if default_city in AIR_STATION_MAP else 0)

# 2. 선택된 시/도에 따라 구/군 목록 업데이트
district_options = AIR_STATION_MAP.get(city, [])

# 3. 구/군 (측정소) 선택 (드롭다운)
if district_options:
    # 선택된 시/도의 첫 번째 항목을 기본값으로 설정
    default_district_index = 0
    gu = st.selectbox("구/군 (측정소) 선택", district_options, 
                      index=default_district_index)
else:
    gu = st.text_input("구/군 (측정소) 입력 (목록 없음)", "")
    st.warning("선택된 시/도에 대한 측정소 목록이 없습니다. 직접 입력해주세요.")

# 4. PM10/PM2.5 선택
pm_type = st.radio("측정 항목 선택", ('PM10', 'PM2.5'), index=0)

# 5. 데이터 조회 기간 선택
data_range = st.selectbox("데이터 조회 기간", 
                          ['최근 48시간', '지난 7일 (168시간)', '지난 30일 (720시간)'],
                          index=0)
    
station = gu # 측정소 이름으로 사용

# [분석 시작] 버튼 클릭 시 실행
if st.button("분석 시작", key="analyze_button"):
    st.subheader(f"📊 {city} {gu} ({pm_type}) 분석 결과") # 현재 위치 정보 표시
    
    # PM 타입에 따라 API 요청 시 사용할 데이터 필드 이름 설정
    data_key = 'pm10Value' if pm_type == 'PM10' else 'pm25Value'
    
    # 선택된 기간에 따라 API로 가져올 데이터 개수(num_rows) 설정
    num_rows_to_fetch = 48 # 기본값
    if data_range == '지난 7일 (168시간)':
        num_rows_to_fetch = 168
    elif data_range == '지난 30일 (720시간)':
        num_rows_to_fetch = 720 # API 제약(1000개) 미만으로 안전
    
    try:
        # 데이터 로딩 스피너 표시
        with st.spinner(f'데이터 ({num_rows_to_fetch}개) 불러오는 중...'):
            items = fetch_air_data(station, num_rows=num_rows_to_fetch)
        st.success("데이터 불러오기 성공!")
    except requests.HTTPError:
        st.error("데이터 요청 중 HTTP 오류가 발생했습니다. 지역명 또는 API 키를 확인하세요.")
        st.stop() # 오류 발생 시 코드 실행 중단
    except Exception as e:
        st.error(f"데이터 요청 중 오류 발생: {e}")
        st.stop() # 오류 발생 시 코드 실행 중단

    # 불러온 데이터 파싱 (시간과 값 분리)
    times, values = parse_pm(items, key=data_key)

    if not values:
        st.warning(f"측정소 '{station}'에 대한 유효한 {pm_type} 데이터가 없습니다. 지역명을 다시 확인해주세요.")
        st.stop()
        
    # 선형 회귀 예측 실행 조건 (단기 조회 시에만)
    if num_rows_to_fetch <= 48:
        predict = linear_regression_predict(values)
    else:
        predict = None
        st.warning("장기 데이터 조회 시에는 예측 기능이 비활성화됩니다. (선형 회귀는 단기 예측에 더 적합합니다)")


    # --- Matplotlib 시각화 ---
    # 그래프 크기를 (14, 7)로 설정하여 가시성 확보
    fig, ax = plt.subplots(figsize=(14, 7))
    criteria = get_grade_criteria(pm_type)
    
    # 1. 미세먼지 등급별 배경 색상 (기준선) 추가
    # '좋음' 영역 (초록)
    ax.axhspan(criteria['좋음'][0], criteria['좋음'][1], facecolor='green', alpha=0.1, label='좋음')
    # '보통' 영역 (노랑)
    ax.axhspan(criteria['보통'][0], criteria['보통'][1], facecolor='yellow', alpha=0.1, label='보통')
    # '나쁨' 영역 (주황)
    ax.axhspan(criteria['나쁨'][0], criteria['나쁨'][1], facecolor='orange', alpha=0.1, label='나쁨')
    
    # Y축 최대값 계산 및 설정
    max_val = max(values) if values else 0
    # 최대값 또는 '매우 나쁨' 기준 중 큰 값의 1.5배를 최대 Y축으로 설정하여 여유 공간 확보
    y_max_limit = max(max_val, criteria['매우 나쁨'][0]) * 1.5
    
    ax.set_ylim(0, y_max_limit) # Y축 범위 설정
    
    # '매우 나쁨' 영역 (빨강)
    ax.axhspan(criteria['매우 나쁨'][0], y_max_limit, facecolor='red', alpha=0.1, label='매우 나쁨')


    ax.set_facecolor('#f9f9f9') # 그래프 배경색 설정
    ax.grid(True, color='#e1e1e1', linestyle='-', linewidth=1) # 그리드 라인 설정
    
    # 실측 데이터 플롯 (파란색 선과 원형 마커)
    ax.plot(times, values, color='#2a4d8f', marker='o', linewidth=2, label=f'실측 {pm_type}')
    
    # 데이터 포인트 위에 값 표시 (48시간 조회 시에만 표시하여 그래프 혼잡도 줄임)
    if num_rows_to_fetch <= 48:
        for x, y in zip(times, values):
            # 텍스트 위치를 값보다 약간 위 (y + 1.5)에 표시
            ax.text(x, y + 1.5, f"{y:.0f}", color='#2a4d8f', fontsize=8, ha='center')

    # 예측값 플롯 (주황색 점선과 원형 마커)
    if predict is not None:
        next_time = times[-1] + timedelta(hours=1) # 다음 1시간 후 시간 계산
        ax.plot([times[-1], next_time], # 현재 마지막 시간과 예측 시간
                [values[-1], predict], # 현재 마지막 값과 예측값
                color='#f28500', marker='o', linestyle='--', linewidth=2, 
                label=f'예측값: {predict:.1f}')
        # 예측값 위에 텍스트 표시
        ax.text(next_time, predict + 1.5, f"{predict:.0f}", color='#f28500', fontsize=8, ha='center')

    # X축 눈금 설정 (기간에 따라 간격 조정)
    if num_rows_to_fetch <= 48:
        xtick_interval = 2 # 48시간: 2시간 간격
    elif num_rows_to_fetch <= 168:
        xtick_interval = 12 # 7일: 12시간 간격
    else:
        xtick_interval = 24 # 30일: 24시간 간격 (일 단위)

    # 설정된 간격에 따라 눈금 시간 및 레이블 생성
    tick_indices = np.arange(0, len(times), xtick_interval)
    tick_times = [times[i] for i in tick_indices if i < len(times)]
    
    if num_rows_to_fetch <= 48:
        tick_labels = [t.strftime("%m-%d %H:%M") for t in tick_times] # 48시간: 월-일 시:분
    else:
        tick_labels = [t.strftime("%Y-%m-%d") for t in tick_times] # 장기: 년-월-일

    ax.set_xticks(tick_times) # X축 눈금 위치 설정
    ax.set_xticklabels(tick_labels, rotation=45) # X축 레이블과 회전 각도 설정

    # 그래프 제목 추가 (위치, 타입 정보 포함)
    ax.set_title(f'{city} {gu} ({pm_type}) 시간대별 농도 변화 추이', fontsize=16, pad=20)
    
    # Y축 레이블 설정 (PM 타입에 따라 변경)
    ax.set_ylabel(f"{pm_type} 농도 (㎍/m³)")
    ax.set_xlabel("측정 시간") # X축 레이블 추가
    
    # 범례 설정 (그래프 오른쪽 상단 바깥에 위치하도록 조정)
    if font_prop:
        # loc='upper left'와 bbox_to_anchor=(1.01, 1)로 오른쪽 상단 바깥에 위치시킴
        ax.legend(loc='upper left', frameon=True, prop=font_prop, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    else:
        ax.legend(loc='upper left', frameon=True, bbox_to_anchor=(1.01, 1), borderaxespad=0.) 
        
    # 범례가 잘리지 않도록 그래프의 오른쪽 여백을 수동으로 확보 (0.8은 오른쪽에서 20% 여백을 남긴다는 의미)
    plt.subplots_adjust(right=0.8)

    st.pyplot(fig) # Streamlit에 그래프 표시
    
    # --- 실측 데이터 테이블 표시 ---
    if times and values:
        st.subheader("📋 실측 데이터 테이블")
        
        # Streamlit의 st.dataframe을 사용해 데이터를 표로 깔끔하게 표시
        data_to_display = {
            "측정 시간": [t.strftime("%Y-%m-%d %H:%M") for t in times],
            f"{pm_type} 농도 (㎍/m³)": [f"{v:.1f}" for v in values] # 소수점 첫째 자리까지 표시
        }
        
        # 표를 화면 너비에 맞게 표시
        st.dataframe(data_to_display, use_container_width=True)


    # --- 예측 결과 표시 ---
    st.subheader("📌 예측 결과")
    if predict is not None:
        # 예측값과 PM 타입을 함께 출력
        st.markdown(f"다음 {pm_type} 예측값: **{predict:.1f} ㎍/m³**")
        # 예측값에 따른 행동 추천 메시지 출력
        st.info(recommend_by_value(predict, pm_type=pm_type))
    else:
        st.warning("데이터 부족 또는 장기 조회로 인해 예측값을 계산할 수 없습니다.")
