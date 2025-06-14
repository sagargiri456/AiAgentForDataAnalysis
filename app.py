# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
import together
import io
import os
from dotenv import load_dotenv
load_dotenv()


# Setup API
model_name = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# File reading utility
def detectingAndReadingFile(file):
    ext = file.name.split('.')[-1]
    if ext == 'csv':
        return pd.read_csv(file)
    elif ext == 'xlsx':
        return pd.read_excel(file)
    elif ext == 'txt':
        return file.read().decode()
    elif ext == 'pdf':
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif ext == 'docx':
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext in ['png', 'jpg', 'jpeg']:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    else:
        return "Unsupported file format"

# LLM Query Function
def generatingResponseFromLLM(question, context=""):
    prompt = f"""
You are a data analyst agent. Analyze the data/text below and answer the user question.

Data/Content:
{context[:3000]}  # Limit to 3000 characters if needed

Question: {question}
Answer:"""

    response = together.Complete.create(
        prompt=prompt,
        model=model_name,
        max_tokens=500,
        temperature=0.7,
        top_k=1,
        top_p=0.9,
    )
    return response['choices'][0]['text']

# Plot Prompt Generator
def generatingCodeForTheGraph(question, dataframe_variable_name='df'):
    if not isinstance(parsed_data, pd.DataFrame):
        return "Error: The uploaded content is not a pandas DataFrame. Cannot generate plot code."

    column_info = parsed_data.dtypes.to_string()
    column_names = list(parsed_data.columns)

    prompt = f"""
You are a Python data visualization expert. The user wants to visualize data from a pandas DataFrame named '{dataframe_variable_name}'.

The DataFrame has the following columns and data types:
{column_info}

The column names are: {column_names}

Generate only clean and executable Python code — without markdown formatting, backticks, or extra explanation.

- Do not wrap the code in triple backticks.
- Do not return any comments or text outside the code.
- Only return valid, plain Python code using matplotlib.
- Assume the DataFrame is already loaded into a variable named '{dataframe_variable_name}'.

User's request: {question}

Python code only:
"""
    return prompt

# Streamlit Interface
st.title("🔍 LLM-Powered Data Analyzer & Plot Generator")

uploaded_file = st.file_uploader("Upload a file (CSV, Excel, PDF, DOCX, TXT, or Image)", type=["csv", "xlsx", "txt", "pdf", "docx", "png", "jpg", "jpeg"])

parsed_data = None

if uploaded_file:
    parsed_data = detectingAndReadingFile(uploaded_file)
    st.success(f"Uploaded: {uploaded_file.name}")

    if isinstance(parsed_data, pd.DataFrame):
        st.subheader("📄 Preview of DataFrame")
        st.dataframe(parsed_data.head())
    else:
        st.subheader("📄 Extracted Text Content")
        st.text(parsed_data[:1000])

    st.divider()

    st.subheader("🧠 Ask a Question about the Data/Text")
    question = st.text_input("Enter your question:")
    if question:
        response = generatingResponseFromLLM(question, context=parsed_data if isinstance(parsed_data, str) else parsed_data.to_string())
        st.markdown("**LLM Answer:**")
        st.write(response)

    if isinstance(parsed_data, pd.DataFrame):
        st.divider()
        st.subheader("📊 Generate a Plot")
        plot_question = st.text_input("Describe the plot (e.g., scatter plot of column A vs column B):")

        if plot_question:
            plot_prompt = generatingCodeForTheGraph(plot_question, dataframe_variable_name='parsed_data')
            plot_code = generatingResponseFromLLM(plot_prompt)

            st.markdown("**Generated Python Code:**")
            st.code(plot_code, language='python')

            if st.button("Run Plot Code"):
                try:
                    exec(plot_code, {'parsed_data': parsed_data, 'plt': plt, 'sns': sns, 'pd': pd})
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error executing code: {e}")
