# pm_predict_app.py

import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression

API_KEY = "여기에_네_API키_입력"

def fetch_air_data(station_name, num_rows=48):
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
    data = r.json()
    items = data['response']['body']['items']
    return items

def parse_pm(items, key='pm10Value'):
    times = []
    values = []
    for it in items:
        t = it.get('dataTime')
        val = it.get(key)
        try:
            v = float(val)
        except:
            continue
        times.append(t)
        values.append(v)

    return times[::-1], values[::-1]

def moving_average_predict(values, window=3):
    if len(values) < window:
        return None, []
    ma = []
    for i in range(len(values)-window+1):
        ma.append(sum(values[i:i+window]) / window)
    return ma[-1], ma

def linear_regression_predict(values):
    if len(values) < 3:
        return None, None
    X = np.arange(len(values)).reshape(-1,1)
    y = np.array(values)
    model = LinearRegression().fit(X, y)
    pred = model.predict([[len(values)]])[0]
    return pred, model

def recommend_by_value(val):
    if val is None:
        return "예측값을 계산할 수 없습니다."
    if val > 80:
        return "⚠️ 매우 나쁨: 외출 자제, KF94 마스크 필수"
    if val > 30:
        return "🙂 보통: 가벼운 외출 가능"
    return "🌿 좋음: 외부 활동에 적합"

# ------------------ Streamlit UI -------------------------

st.title("🌫️ 실시간 미세먼지 분석 + 예측")

city = st.text_input("시/도 입력", "서울")
gu = st.text_input("구/군 입력", "강남구")
station = gu  # API에서 측정소 이름은 대부분 '구' 이름

if st.button("분석 시작"):
    try:
        items = fetch_air_data(station, num_rows=50)
        st.success("데이터 불러오기 성공!")
    except Exception as e:
        st.error("데이터 요청 중 오류 발생. 지역명 또는 API 키 확인하세요.")
        st.stop()

    times, values = parse_pm(items)

    if not values:
        st.warning("유효한 PM10 데이터가 없습니다.")
        st.stop()

    ma_pred, ma_values = moving_average_predict(values, window=3)
    lr_pred, _ = linear_regression_predict(values)

    predict = ma_pred if ma_pred is not None else lr_pred

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, values, marker='o', label='실측 PM10')

    if ma_values:
        ax.plot(times[-len(ma_values):], ma_values, marker='x', label='이동평균')

    if predict is not None:
        ax.axhline(predict, linestyle='--', label=f'예측값: {predict:.1f}')

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("📌 예측 결과")
    if predict:
        st.write(f"다음 PM10 예측값: **{predict:.1f} ㎍/m³**")
        st.info(recommend_by_value(predict))
