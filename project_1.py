# pm_predict.py
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression  # 선택적

API_KEY = "aea45d5692f9dc0fb20ff49e2cf104f6614d3a17df9e92420974a5defb3cd75e"  # <- 반드시 바꿔

def fetch_air_data(station_name, num_rows=48):
    """
    station_name: 측정소(구/시) 이름
    num_rows: 불러올 데이터 수 (최대)
    반환: items 리스트(시간순, 최신 먼저)
    """
    URL = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
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
    r.raise_for_status()
    data = r.json()
    # 응답 구조가 다르면 KeyError 발생
    items = data['response']['body']['items']
    return items

def parse_pm(items, key='pm10Value'):
    """
    items: API items
    key: 'pm10Value' 또는 'pm25Value'
    반환: times(list of str), values(list of float)
    """
    times = []
    values = []
    for it in items:
        t = it.get('dataTime')
        val = it.get(key)
        # 값이 '-' 이거나 '' 인 경우 처리
        try:
            v = float(val)
        except:
            continue
        times.append(t)
        values.append(v)
    # API는 최신순 반환 -> 시간순 정렬(오래된->최신)
    times = times[::-1]
    values = values[::-1]
    return times, values

def moving_average_predict(values, window=3):
    if len(values) < window:
        return None, []
    ma = []
    for i in range(len(values)-window+1):
        ma.append(sum(values[i:i+window]) / window)
    return ma[-1], ma

def linear_regression_predict(values):
    # 간단한 시간 인덱스 기반 선형회귀
    if len(values) < 3:
        return None, None
    X = np.arange(len(values)).reshape(-1,1)
    y = np.array(values)
    model = LinearRegression().fit(X, y)
    next_x = np.array([[len(values)]])
    pred = model.predict(next_x)[0]
    return pred, model

def plot_result(times, values, ma_values, predict_value, city, gu, filename="pm_graph.png"):
    plt.figure(figsize=(10,5))
    plt.plot(times, values, marker='o', label='실측 PM10')
    if ma_values:
        plt.plot(times[len(times)-len(ma_values):], ma_values, marker='x', label='MA')
    # 예측값 시각화(마지막 다음칸)
    plt.axhline(predict_value, linestyle='--', label=f'다음 예측값: {predict_value:.1f}')
    plt.xticks(rotation=45)
    plt.title(f"{city} {gu} PM10 변화 & 예측")
    plt.xlabel("시간")
    plt.ylabel("PM10 (㎍/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def recommend_by_value(val):
    if val is None:
        return "예측값을 계산할 수 없습니다."
    if val > 80:
        return "⚠️ 매우 나쁨: 외출 자제, KF94 마스크 권장"
    if val > 30:
        return "🙂 보통: 가벼운 외출 가능"
    return "🌿 좋음: 외부 활동에 적합"

if __name__ == "__main__":
    city = input("시/도 입력 (예: 서울): ").strip()
    gu = input("구/군 입력 (예: 강남구): ").strip()
    full_station = gu  # 보통 구 이름으로 station 검색 가능

    try:
        items = fetch_air_data(full_station, num_rows=50)
    except Exception as e:
        print("데이터 요청/파싱 중 오류 발생:", e)
        print("디버깅: 입력한 지역명이 정확한지, API키 유효성(공백/인코딩) 확인하세요.")
        exit()

    times, values = parse_pm(items, key='pm10Value')
    if not values:
        print("유효한 PM10 값이 없습니다. 다른 측정소명/지역을 시도하세요.")
        exit()

    # 선택: 이동평균 예측
    ma_predict, ma_values = moving_average_predict(values, window=3)

    # 선택: 선형회귀 예측 (주석 해제하면 사용)
    lr_predict, lr_model = linear_regression_predict(values)

    # 여기서는 MA 우선 사용, 없으면 LR 사용
    predict = ma_predict if ma_predict is not None else lr_predict

    plot_result(times, values, ma_values, predict, city, gu, filename="pm10_graph.png")
    print("\n지역:", city, gu)
    print("다음 시간대 예측 PM10:", round(predict,2) if predict is not None else "계산불가")
    print("추천:", recommend_by_value(predict))
