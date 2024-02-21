import streamlit as st
from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
import pandas as pd

st.set_page_config(layout='wide')
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)


library = "seaborn"

st.subheader("Summarization of your Data")
file_uploader = st.file_uploader("Upload your CSV", type="csv")    
if file_uploader is not None:
    path_to_save = "filename.csv"
    with open(path_to_save, "wb") as f:
        f.write(file_uploader.getvalue())
    st.write("") 
    st.write("")
    st.info("Dataset Preview")
    df = pd.read_csv(file_uploader)
    st.dataframe(df, use_container_width=True)

    st.write("") 
    st.write("")

    st.info("JSON View")
    summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)
    st.write(summary)

    
sidebar_styles = """
    <style>
        .sidebar-content {
            background-color: #f0f2f6; /* Set background color */
            padding: 20px; /* Add padding */
            border-radius: 10px; /* Add border radius */
        }
        .sidebar .sidebar-content a {
            color: #0366d6; /* Change link color */
        }
    </style>
"""
st.markdown(sidebar_styles, unsafe_allow_html=True)