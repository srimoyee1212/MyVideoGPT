import os
from apikey import mykey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['OPENAI_API_KEY'] = mykey
st.title('Video-Link GPT')
prompt = st.text_input('Enter your prompt')

title_template = PromptTemplate(input_variables=['topic'], template='write me a youtube video title about {topic}')

llm = OpenAI(temperature=0.9)

if prompt:
    response = llm(prompt)
    st.write(response)
   