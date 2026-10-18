import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Islamic Game Prompt Studio",
    page_icon="🕌",
    layout="wide",
)

base = Path(__file__).parent

html = (base / "index.html").read_text(encoding="utf-8")
css = (base / "style.css").read_text(encoding="utf-8")
js = (base / "app.js").read_text(encoding="utf-8")

html = html.replace(
    '<link rel="stylesheet" href="style.css">',
    f"<style>{css}</style>",
)
html = html.replace(
    '<script src="app.js"></script>',
    f"<script>{js}</script>",
)

st.components.v1.html(html, height=900, scrolling=True)
