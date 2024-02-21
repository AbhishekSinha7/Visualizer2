import streamlit as st
from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import time


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))
    # Use BytesIO to convert the byte data to image

lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)
st.set_page_config(layout='wide')

library = "seaborn"
file_uploader = st.file_uploader("Upload your CSV", type="csv")
        
st.subheader("Analysis of the Data")
if file_uploader is not None:
    path_to_save = "filename1.csv"
    with open(path_to_save, "wb") as f:
        f.write(file_uploader.getvalue())   
    st.write("") 
    st.write("")

    summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)

    st.write("") 
    st.write("")
    
    st.info("Summary of the Dataset")
    goals = lida.goals(summary, n=25, textgen_config=textgen_config)

    def show_Analytics():
        count =0
        for goal in goals:
            st.write(goal.question)
            st.write("Goal: ",goal.rationale)
            textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
            charts = lida.visualize(summary=summary, goal=goal.visualization, textgen_config=textgen_config, library=library)  
            img_base64_string = charts[0].raster
            img = base64_to_image(img_base64_string)
            st.image(img)
            count=count+1
            st.info("-------------------------------------------------------------------------------------------x-------------------------------------------------------------------------------------------")
            if count % 3==0:
                time.sleep(22)

    while True:
        show_Analytics()
        


    