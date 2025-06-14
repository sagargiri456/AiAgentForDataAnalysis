# AiAgentForDataAnalysis
# 🧠 LLM-Powered Data Analyst Agent

This project builds a backend-first intelligent agent using the Together.ai `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` model that can:

- 🗂️ Accept various file formats (`.csv`, `.xlsx`, `.txt`, `.doc`, `.pdf`, images)
- 📊 Analyze and answer questions about the uploaded data
- 📉 Generate visualizations (`matplotlib`) based on user prompts
- 🧠 Handle follow-up queries intelligently

> Focus is on backend logic, with optional support for Streamlit UI.

---

## 📌 Features

- ✅ File parsing for CSV, Excel, text, PDF, and images
- ✅ Text-based context passed to the LLM
- ✅ Matplotlib chart generation based on user questions
- ✅ Seamless integration with Together.ai’s Llama 4 Maverick model
- ✅ Built and tested on Google Colab (can run end-to-end in `.ipynb`)

---

## 🚀 How It Works

1. **User uploads a file**
2. **File is parsed into raw tabular `context` text**
3. **User asks a question like:**  
   _"Plot a bar chart of total sales per Item_Type"_
4. **LLM generates valid Python `matplotlib` code**
5. **Code is executed safely in Colab and output is shown**

---

## 🏗️ Architecture

## 📄 Requirements

- Python 3.x
- Google Colab (or Jupyter)
- [Together.ai API key](https://platform.together.xyz/)
- `pandas`, `matplotlib`, `together`, `pdf2image`, `pytesseract`, etc.

Install libraries:
```bash
pip install together pandas matplotlib seaborn pdf2image pytesseract openpyxl
Let me know if you'd like this to be split into sections within your `.ipynb` or if you're ready for the Streamlit UI scaffold too.







