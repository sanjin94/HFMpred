import pandas as pd
import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import uval as U
import streamlit as st
from datetime import datetime
import sys
sys.path.append('data')
sys.path.append('icons')


def app():
    col1, col2, col3, col4, col5 = st.columns([[1, 1, 1, 1, 1]])
    with col1:
        start_date = []
        end_date = []
        col1_i = st.number_input('Number of measurements', min_value=0, max_value=10)
        for i in range(col1_i):
            #nešto
            date = date
            time = time
    with col2:
        nlayers = 0
