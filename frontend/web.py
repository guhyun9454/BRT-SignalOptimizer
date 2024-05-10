import streamlit as st

def page1():
    st.title('2024 KHUTHON')
    st.subheader('- 지능형 교통 시스템을 통한 미래 계획 도시 구현')


def page2():
    st.title('교통혼잡도')
    st.subheader("- 교통혼잡도 확인")

def page3():
    st.title('교통상황')
    st.subheader('- 교통상황 simulation')

# 페이지의 기본 설정 변경
st.set_page_config(page_title="멀티페이지 애플리케이션", page_icon=":smiley:", layout="wide", initial_sidebar_state="auto")

# 배경색을 흰색으로 설정
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 사이드바에 페이지 선택 옵션 추가
page_selection = st.sidebar.radio("페이지 선택", ["기본페이지","교통혼잡도", "교통상황"])

# 선택된 페이지에 따라 해당 페이지를 표시


if page_selection =="기본페이지":
    page1()
elif page_selection == "교통혼잡도":
    page2()
elif page_selection == "교통상황":
    page3()

