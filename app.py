import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## function to get response from LLama2 model

def getLLamaResponse(input_text,no_words,blog_style):

    ## llama2 model 
    llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type="llama",
                        config={'max_new_tokens':512,
                              'temperature':0.1})
    
    ## Prompt Template
    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt = PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the response from the llama2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response
    


st.set_page_config(
    page_title="Generate Blogs",
    page_icon="✍",
    layout="centered",
    initial_sidebar_state="collapsed")

st.header("Generate Blogs ")

input_text = st.text_input("Enter the Blog Topic")

col1,col2=st.columns([5,5])

with col1:
    no_words = st.text_input("Number of Words")
with col2:
    blog_style = st.selectbox("writing this blog for",("Web Developer","Data Scientist","AI Engineer"),index=0)

submit = st.button("Generate")

## Final Response

if submit:
    st.write(getLLamaResponse(input_text,no_words,blog_style))