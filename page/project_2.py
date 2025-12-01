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
    
    def find_font_name(): # 폰트 이름 검색 도우미 함수
        """시스템에 설치된 한글 폰트 이름을 찾아서 돌려줘."""
        font_list = [f.name for f in fm.fontManager.ttflist] # 폰트 이름 리스트 추출
        for name in ["NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
            if name in font_list:
                return name # 폰트 이름 반환
        return None # 폰트 찾기 실패

    font_name = find_font_name() 
    
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
        return "🔥 매우 나쁨: 외출 자제, 실
