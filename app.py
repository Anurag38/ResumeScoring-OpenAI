import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai.chat_models import ChatOpenAI
import PyPDF2 as pdf


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

load_dotenv() ## load all our environment variables

client = OpenAI()
def get_openai_repsonse(input_prompt, jd, text):
    model=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3, model_name="gpt-4-turbo", max_tokens="2000")
    input_data = {
    "resume": text,
    "job_desc": jd
    }   
    # chain = LLMChain(llm=model, prompt=input_prompt)
    parser = JsonOutputParser()
    chain = input_prompt | model | parser
    response = chain.invoke(input_data)
    
    return response

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

#Prompt Template

input_prompt = PromptTemplate(template="""
    Act as a highly experienced ATS specializing in tech fields like software engineering, data science, data analysis, and big data engineering. 

    **Your task:**

    1. **Evaluate the provided resume:** Analyze the resume based on the given job description, considering the competitive job market. 
    2. **Score the resume:** Assign a score from 1 (poor fit) to 10 (perfect fit) based on how well the resume aligns with the job requirements.
    3. **Explain the score:** Provide a brief explanation (2-4 sentences) justifying the assigned score, highlighting key strengths and areas for improvement.

    resume:{resume}
    job_description:{job_desc}

    I want the response to be in the following format
    {{
        "Score" : "%",
        "Explaination" : ""
    }}
    """, input_variables=["resume", "job_desc"])
## streamlit app
st.title("Resume Scoring (ATS)")
jd=st.text_area("Paste the Job Description")
uploaded_file=st.file_uploader("Upload Your Resume",type="pdf",help="Please uplaod the pdf")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        text=input_pdf_text(uploaded_file)
        response=get_openai_repsonse(input_prompt, jd, text)
        # Display the results
        st.write("**Score:**", response["Score"])
        st.write("**Explanation:**", response["Explanation"])
