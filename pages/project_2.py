import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="미세먼지 부가 정보",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 메인 헤더 ---
st.title("✨ 미세먼지 부가 정보 및 대응 가이드")
st.markdown("""
이 페이지에서 **PM10**과 **PM2.5**의 차이점을 이해하고, 
각 미세먼지 농도별 건강 행동 수칙을 확인하세요.
""")
st.divider()

# --- PM10 vs PM2.5 선택 ---
st.header("🔍 미세먼지 종류 선택")

dust_type = st.radio(
    "알아보고 싶은 미세먼지 종류를 선택하세요:",
    options=["PM10 (미세먼지)", "PM2.5 (초미세먼지)"],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# --- PM10과 PM2.5 차이점 설명 ---
st.header("📊 PM10 vs PM2.5 무엇이 다를까요?")

col_pm10, col_pm25 = st.columns(2)

with col_pm10:
    st.subheader("🌫️ PM10 (미세먼지)")
    st.markdown("""
    **크기:** 지름 10㎛ 이하  
    **비유:** 머리카락 굵기의 약 1/7  
    **특징:** 코와 목에서 걸러질 수 있음  
    **영향:** 호흡기 질환 유발 가능
    """)

with col_pm25:
    st.subheader("💨 PM2.5 (초미세먼지)")
    st.markdown("""
    **크기:** 지름 2.5㎛ 이하  
    **비유:** 머리카락 굵기의 약 1/30  
    **특징:** 폐 깊숙이 침투 가능  
    **영향:** 심혈관 질환까지 유발 가능 (더 위험)
    """)

st.info("💡 **핵심:** PM2.5가 더 작고 위험합니다! 폐 속 깊이 들어가 심장과 혈관에도 영향을 줄 수 있어요.")
st.divider()

# --- 선택된 미세먼지 정보 표시 ---
if "PM10" in dust_type:
    st.header("🌫️ PM10 (미세먼지) 등급 기준 및 행동 수칙")
    
    # PM10 등급 기준
    st.subheader("📏 PM10 농도별 등급")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("**✅ 좋음**")
        st.markdown("**0 ~ 30 ㎍/m³**")
    
    with col2:
        st.warning("**🟡 보통**")
        st.markdown("**31 ~ 80 ㎍/m³**")
    
    with col3:
        st.error("**🟠 나쁨**")
        st.markdown("**81 ~ 150 ㎍/m³**")
    
    with col4:
        st.error("**🔴 매우 나쁨**")
        st.markdown("**151 ㎍/m³ 이상**")
    
    st.divider()
    
    # PM10 위험도 및 행동 수칙
    st.subheader("⚠️ PM10 농도별 위험도와 행동 수칙")
    
    with st.expander("✅ 좋음 (0~30) - 안전", expanded=True):
        st.markdown("""
        **위험도:** 🟢 매우 낮음  
        **야외 활동:** 자유롭게 가능  
        **환기:** 자주 환기하세요  
        **마스크:** 필요 없음
        """)
    
    with st.expander("🟡 보통 (31~80) - 주의"):
        st.markdown("""
        **위험도:** 🟡 낮음  
        **야외 활동:** 일반인은 정상 활동, 민감군은 장시간 활동 자제  
        **환기:** 오전 10시~오후 4시 권장  
        **마스크:** 민감군은 착용 권장
        """)
    
    with st.expander("🟠 나쁨 (81~150) - 위험"):
        st.markdown("""
        **위험도:** 🟠 보통  
        **야외 활동:** 장시간 또는 격렬한 활동 제한  
        **환기:** 최소화  
        **마스크:** KF80 이상 필수 착용
        """)
    
    with st.expander("🔴 매우 나쁨 (151 이상) - 심각"):
        st.markdown("""
        **위험도:** 🔴 높음  
        **야외 활동:** 전면 금지  
        **환기:** 창문 닫고 공기청정기 사용  
        **마스크:** 외출 시 KF94 이상 필수, 실외 활동 자제
        """)

else:  # PM2.5 선택
    st.header("💨 PM2.5 (초미세먼지) 등급 기준 및 행동 수칙")
    
    # PM2.5 등급 기준
    st.subheader("📏 PM2.5 농도별 등급")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("**✅ 좋음**")
        st.markdown("**0 ~ 15 ㎍/m³**")
    
    with col2:
        st.warning("**🟡 보통**")
        st.markdown("**16 ~ 35 ㎍/m³**")
    
    with col3:
        st.error("**🟠 나쁨**")
        st.markdown("**36 ~ 75 ㎍/m³**")
    
    with col4:
        st.error("**🔴 매우 나쁘**")
        st.markdown("**76 ㎍/m³ 이상**")
    
    st.divider()
    
    # PM2.5 위험도 및 행동 수칙
    st.subheader("⚠️ PM2.5 농도별 위험도와 행동 수칙")
    
    with st.expander("✅ 좋음 (0~15) - 안전", expanded=True):
        st.markdown("""
        **위험도:** 🟢 매우 낮음  
        **야외 활동:** 자유롭게 가능  
        **환기:** 자주 환기하세요  
        **마스크:** 필요 없음
        """)
    
    with st.expander("🟡 보통 (16~35) - 주의"):
        st.markdown("""
        **위험도:** 🟡 낮음  
        **야외 활동:** 일반인은 정상 활동, 민감군은 장시간 활동 자제  
        **환기:** 적절한 시간에 환기  
        **마스크:** 민감군(노약자, 어린이, 호흡기 질환자)은 착용 권장
        """)
    
    with st.expander("🟠 나쁨 (36~75) - 위험"):
        st.markdown("""
        **위험도:** 🟠 보통~높음  
        **야외 활동:** 장시간 또는 격렬한 활동 제한  
        **환기:** 최소화하고 공기청정기 사용  
        **마스크:** KF80 이상 필수 착용 (특히 어린이, 노약자)
        """)
    
    with st.expander("🔴 매우 나쁨 (76 이상) - 심각"):
        st.markdown("""
        **위험도:** 🔴 매우 높음 (심혈관 위험)  
        **야외 활동:** 전면 금지  
        **환기:** 절대 금지, 공기청정기 필수  
        **마스크:** 외출 시 KF94 이상 필수, 가급적 외출 자제
        """)

st.divider()

# --- 추가 정보 ---
st.subheader("💡 마스크 선택 가이드")
with st.expander("📌 KF 등급별 차단 효과"):
    st.markdown("""
    * **KF80:** 미세먼지 80% 차단 - 일상용
    * **KF94:** 미세먼지 94% 차단 - 나쁨 단계 권장
    * **KF99:** 미세먼지 99% 차단 - 매우 나쁨 단계 권장
    
    ⚠️ **주의:** 일반 마스크(면 마스크, 덴탈 마스크)는 미세먼지 차단 효과가 거의 없습니다!
    """)

st.caption("🚨 이 정보는 환경부 기준을 따릅니다. 건강 문제가 있으신 분은 전문의와 상담하세요.")
