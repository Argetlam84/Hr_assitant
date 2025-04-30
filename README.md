# 🤖 AI-Powered HR Assistant – Resume Evaluator with LangChain + Gemini

<a href="https://hr-assitant.onrender.com/">You can find in here!</a>

This project is a lightweight **Streamlit app** that assists HR professionals by evaluating candidates' resumes against job descriptions using **Google Gemini + LangChain**. It leverages **RAG (Retrieval-Augmented Generation)** and an **agent tool** to deliver structured, context-aware feedback.

---

## 🧠 What It Does

- 📄 **Parses PDF/DOCX resumes and job descriptions**
- 🔍 **Evaluates candidate fit** based on content similarity
- 🧾 **Generates structured feedback**
- 📬 **Sends emails** (selection or rejection) automatically
- ⚡ Powered by:
  - 🧠 **LangChain RAG** for resume-job context analysis
  - 🧰 **LangChain agent + tool** for automated email dispatch
  - 🔎 **FAISS** for vector similarity search
  - 🤖 **Google Gemini (via LangChain)** for reasoning and output

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Argetlam84/Hr_assistant.git
cd Hr_assistant

python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt

SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASS=your_email_password
GOOGLE_API_KEY=your_google_gemini_api_key

streamlit run streamlit/app.py
```

---

📁 How to Use

- Upload Candidate Resume (.pdf or .docx)

- Upload Job Description (.pdf or .docx)

- Enter Candidate Email

- Click Evaluate Resume

The app will:

- Parse the documents

- Run a contextual evaluation with LangChain

- Generate feedback

- Compose and send a professional email automatically

⚙️ Behind the Scenes

- ✅ Resume & JD Text Extraction: via pdfplumber and docx2txt

- 🔍 Vector Search: FAISS + Google Text Embedding (via text-embedding-004)

- 🧠 LLM Evaluation: Gemini 2.0 Flash model

- 📡 RAG Architecture: LangChain retrieval + combine documents chain

- 📬 Email Tool: Structured tool (LangChain agent) with smtplib

- 🎛️ Streamlit UI: For resume/jobdesc uploads and trigger

📸 UI Preview

- Clean and minimalistic Streamlit interface

- Drag-and-drop resume/job description

- Email sent instantly with success message and balloons 🎈

📊 Datasets Used

This project utilizes publicly available datasets to test and demonstrate the resume evaluation capabilities:
1. <a href="https://huggingface.co/datasets/brackozi/Resume#:~:text=Skills%20,Ernst">brackozi/Resume – Hugging Face</a> 

    Description: A collection of 962 resumes from various industries and roles.

    Format: Available in JSON and Parquet formats.

    Usage: Employed to test the resume parsing and evaluation functionalities of the application.

2. <a href="https://github.com/r7mvee/Web-Scraping-Job-Postings/blob/master/%5Bscrape2%5D.csv">Web-Scraping-Job-Postings – GitHub</a>

    Description: A dataset containing job postings scraped from various sources, focusing on data science roles.

    Format: CSV file with over 3,000 entries.

    Usage: Used to test the job description parsing and matching features of the application.

These datasets are instrumental in validating the application's ability to accurately assess resume and job description compatibility.

📜 License

This project is open source under the MIT License.
🤝 Contribution

Feel free to fork the project and submit pull requests for improvements or feature ideas.
👋 Contact


Developed by Mehmet Arslan
📧 mr.arslan84@icloud.com
