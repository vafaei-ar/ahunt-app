#! env python

import os
import time
import numpy as np
import streamlit as st
# import cv2
from zipfile import ZipFile
# from skimage.transform import resize
#from tf_keras_vis.utils import normalize
#from tf_keras_vis.saliency import Saliency
#from tf_keras_vis.scorecam import ScoreCAM
#from tf_keras_vis.gradcam import Gradcam,GradcamPlusPlus
# from scipy.ndimage import gaussian_filter as gauss
from ahunt import *
# import models
#import easygui


import ahunt
dirname = path = os.path.dirname(ahunt.__file__)

#st.set_page_config(layout="wide")

session_state = st.session_state #get(dirname=None,reset_model=True)


st.sidebar.image(os.path.join(dirname,'media/logo.png'), use_column_width=True)

# Using object notation
# task = st.sidebar.selectbox(
#     "What do you like to do?",
#     ('Labeler',) #, 'Classification'
# )


if not hasattr(session_state,'labeler_config') or \
    not session_state.labeler_config.get('labels'):
    imageshow_setstate(session_state)
elif session_state.success:
    tab1, tab2 = st.tabs(["Labeler", "Database"])
    with tab1:
        imageshow_label(session_state)
    with tab2:
        st.write(session_state.df)
else:
    del session_state.labeler_config
    st.write('Something went wrong!')


# if not hasattr(session_state,'labeler_config') or \
#     not session_state.labeler_config.get('labels'):
#     imageshow_setstate(session_state)
# elif session_state.success: 
#     imageshow_label(session_state)
# else:
#     del session_state.labeler_config
#     st.write('Something went wrong!')















