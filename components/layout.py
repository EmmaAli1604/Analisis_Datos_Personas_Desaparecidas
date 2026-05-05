# streamlit/components/layout.py
import streamlit as st
import os

def load_css():
    css_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "styles.css"))
    with open(css_path, encoding="utf-8") as f:  # ← agrega encoding="utf-8"
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def page_layout(title: str, icon: str = ""):
    load_css()
    if icon:
        st.title(f"{icon} {title}")
    else:
        st.title(title)