import streamlit as st
from homepage import app as hp_app
from fdetectimg import app as fdi_app
from fdetectlive import app as fdl_app
from frecog import app as frl_app
from orecog import app as or_app
from handwrittendigit import app as hwd_app
from calihouseprice import app as chp_app
from nextword import app as nw_app
from speech2text import app as s2t_app

st.set_page_config(page_title="ML_App", page_icon=("🇻🇳"))
st.sidebar.title("FEATURES 🔍")
sidebar_col, main_col = st.columns([1, 4])
app_mapping = {
    "Detection": {"Image": fdi_app, "Webcam": fdl_app},
    "Recognition": {"Face": frl_app, "Objective": or_app, "Handwritten digit": hwd_app},
    "Prediction": {"Cali-house price": chp_app, "Next word": nw_app},
    "Speech-to-text": s2t_app
}

selected_app_mode = st.sidebar.selectbox("Machine Learning Apps [8]", list(app_mapping.keys()), index=0)

if sidebar_col.button("🏠"):
    hp_app()
else:
    if selected_app_mode != "Speech-to-text":
        selected_submode = st.sidebar.radio("Choose one of them:", list(app_mapping[selected_app_mode].keys()))
        app_mapping[selected_app_mode][selected_submode]()
    else:
        app_mapping[selected_app_mode]()
