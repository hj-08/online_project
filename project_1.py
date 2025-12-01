# pm_predict_app.py

import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

API_KEY = "aea45d5692f9dc0fb20ff49e2cf104f6614d3a17df9e92420974a5defb3cd75e"

# ------------------ 데이터 가져오기 -------------------------
def fetch_air_data(station_name, num_rows=48):
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
        # datetime 변환 시도
        dt = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d%H%M"):
            try:
                dt = datetime.strptime(t, fmt)
                break
            except:
                continue
        if dt is None:
            continue  # 변환 불가 시 스킵
        times.append(dt)
        values.append(v)
    return times[::-1], values[::-1]

def linear_regression_predict(values):
    if len(values) < 3:
        return None
    X = np.arange(len(values)).reshape(-1,1)
    y = np.array(values)
    model = LinearRegression().fit(X, y)
    pred = model.predict([[len(values)]])[0]
    return pred

def recommend_by_value(val):
    if val is None:
        return "예측값을 계산할 수 없습니다."
    if val > 150:
        return "🔥 매우 나쁨: 외출 자제, 실내 활동 권장"
    if val > 80:
        return "⚠️ 나쁨: 장시간 외출 피하고 마스크 착용"
    if val > 30:
        return "🙂 보통: 민감군은 주의, 가벼운 외출 가능"
    return "🌿 좋음: 외부 활동 안전"

# ------------------ Streamlit UI -------------------------

st.title("🌫️ 실시간 미세먼지 분석 + 예측")

city = st.text_input("시/도 입력", "서울")
gu = st.text_input("구/군 입력", "강남구")
station = gu  # 대부분 구 이름으로 측정소 지정

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

    predict = linear_regression_predict(values)

    # ------------------ 그래프 생성 -------------------------
    fig, ax = plt.subplots(figsize=(10, 4))

    # 실측치
    ax.plot(times, values, color='blue', marker='o', label='실측 PM10')

    # 예측선
    if predict is not None:
        next_time = times[-1] + timedelta(hours=1)
        ax.plot([times[-1], next_time],
                [values[-1], predict],
                color='orange', linestyle='-', marker='o', label=f'예측값: {predict:.1f}')

    # x축 라벨: 6시간 단위
    ax.set_xticks(times[::6])
    ax.set_xticklabels([t.strftime("%m-%d %H:%M") for t in times[::6]], rotation=45)

    ax.set_ylabel("PM10 (㎍/m³)")
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

    # ------------------ 위험도 표시 -------------------------
    st.subheader("📌 예측 결과")
    if predict is not None:
        st.write(f"다음 PM10 예측값: **{predict:.1f} ㎍/m³**")
        st.info(recommend_by_value(predict))
