# pm_predict_app.py

import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

import matplotlib.font_manager as fm

# 위에서 설정한 한글 폰트 이름 가져오기
font_list = [f.name for f in fm.fontManager.ttflist]
if "Malgun Gothic" in font_list:
    font_name = "Malgun Gothic"
elif "NanumGothic" in font_list:
    font_name = "NanumGothic"
else:
    font_name = "DejaVu Sans"

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# font_prop 생성
font_prop = fm.FontProperties(fname=None, family=font_name)

# ... 그래프 그릴 때

ax.legend(frameon=False, prop=font_prop)




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

    # 배경색과 그리드 설정
    ax.set_facecolor('#f9f9f9')  # 연한 회색 배경
    ax.grid(True, color='#e1e1e1', linestyle='-', linewidth=1)

    # 실측 데이터 라인 + 점 + 값 텍스트
    ax.plot(times, values, color='#2a4d8f', marker='o', linewidth=2, label='실측 PM10')
    for x, y in zip(times, values):
        ax.text(x, y + 1, f"{y:.0f}", color='#2a4d8f', fontsize=8, ha='center')

    # 예측선 (주황) + 점 + 값 텍스트
    if predict is not None:
        next_time = times[-1] + timedelta(hours=1)
        ax.plot([times[-1], next_time],
                [values[-1], predict],
                color='#f28500', marker='o', linewidth=2, label=f'예측값: {predict:.1f}')
        ax.text(next_time, predict + 1, f"{predict:.0f}", color='#f28500', fontsize=8, ha='center')

    # x축 레이블 6시간 간격, 회전 표시
    ax.set_xticks(times[::6])
    ax.set_xticklabels([t.strftime("%m-%d %H:%M") for t in times[::6]], rotation=45)

    ax.set_ylabel("PM10 (㎍/m³)")
    ax.legend(frameon=False)
    plt.tight_layout()

    st.pyplot(fig)

    # ------------------ 위험도 표시 -------------------------
    st.subheader("📌 예측 결과")
    if predict is not None:
        st.write(f"다음 PM10 예측값: **{predict:.1f} ㎍/m³**")
        st.info(recommend_by_value(predict))
