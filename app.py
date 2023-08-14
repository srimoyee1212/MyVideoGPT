import os
from apikey import mykey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = mykey
st.title('Video-Link GPT')
prompt = st.text_input('Enter your prompt')

memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

title_template = PromptTemplate(input_variables=['topic'], template='write me a youtube video title about {topic}')
script_template = PromptTemplate(input_variables=['title'], template='write me a youtube video script on this title: {title}')

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)



if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Message History'):
        st.info(memory.buffer)
   