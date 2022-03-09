import sys

from multiapp import MultiApp
sys.path.append('icons')

import streamlit as st
from st_parts import stuval, sthfm, stmeas

EMOJI_ICON = "icons/HFMpred.ico"
EMOJI_PNG = "icons/HFMpred.png"

st.set_page_config(page_title="HFMpred", page_icon=EMOJI_ICON,
                   layout='wide')

col1, col2 = st.columns(2)
with col2:
    st.image(EMOJI_PNG, width=80)
col1, col2 = st.columns([1.5,6])
with col2:
    st.title('HFMpred - a tool for HFM results analysis and prediction')

app = MultiApp()
app.add_app("HFM results prediction", sthfm.app)
app.add_app("U-value calculation", stuval.app)
app.add_app("Realtime Measurement", stmeas.app)
# Main HFMpred app
app.run()
