import streamlit as st
from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
import pandas as pd

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

st.subheader("Query your Data to Generate Graph")
file_uploader = st.file_uploader("Upload your CSV", type="csv")
if file_uploader is not None:
    path_to_save = "filename1.csv"
    with open(path_to_save, "wb") as f:
        f.write(file_uploader.getvalue())
    text_area = st.text_area("Query your Data to Generate Graph", height=200)
    code=0
    if st.button("Generate Graph"):
        if len(text_area) > 0:
            st.info("Your Query: " + text_area)
            lida = Manager(text_gen = llm("openai")) 
            textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
            summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
            user_query = text_area
                
            charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
            code = charts[0].code
            st.code(code)
                
            image_base64 = charts[0].raster
            img = base64_to_image(image_base64)
            st.image(img)