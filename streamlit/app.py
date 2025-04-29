import streamlit as st
import docx2txt
import pdfplumber
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
import os
from pydantic import BaseModel

# Load environment variables
load_dotenv()
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# Set up embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=1000
)

# System prompt with context placeholder
system_prompt = (
    """You are an AI-powered HR assistant specializing in evaluating job applicants' resumes against specific job descriptions. Your primary responsibilities include:
    1. **Resume Evaluation**: Analyze the applicant's resume in detail, comparing it with the provided job description to assess alignment in terms of skills, experience, and qualifications.
    2. **Decision Making**: Based on the evaluation, determine whether the candidate should proceed to the next stage of the hiring process.
    3. **Feedback Generation**: If the candidate is not selected, generate a clear and empathetic explanation highlighting the primary reasons for the decision, focusing on areas where the resume did not meet the job requirements.
    4. **Email Composition**":
       - Draft a professional rejection email when the candidate is not selected.
       - Draft a professional “next stage” invitation email when the candidate is selected, outlining the next steps (e.g., scheduling an interview).
    **Important Guidelines**:
    - **Scope Limitation**: You are strictly limited to tasks related to resume evaluation and candidate communication. Do not engage in topics unrelated to HR functions.
    - **Confidentiality**: Handle all applicant information with the utmost confidentiality and professionalism.
    - **Tone and Language**: Maintain a formal, respectful, and empathetic tone.
    - **Feedback Specificity**: Provide specific feedback related to the job description criteria.

    Use the following context to assist with evaluations:

    {context}"""
)

# Prompt for retrieval chain
retrieval_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=retrieval_prompt
)
retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=docs_chain
)

# Send email implementation
def send_email(to_email: str, subject: str, body: str):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = to_email
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)

# StructuredTool schema for email tool
class EmailInput(BaseModel):
    to_email: str
    subject: str
    body: str

send_email_tool = StructuredTool(
    name="send_email",
    description="Sends an email to the candidate using to_email, subject, and body fields.",
    args_schema=EmailInput,
    func=send_email
)

# Agent prompt for email composition
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an HR assistant. Compose and send emails."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

# Create agent and executor with structured tool
agent = create_openai_functions_agent(
    llm=llm,
    tools=[send_email_tool],
    prompt=agent_prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[send_email_tool],
    verbose=True
)

# Core evaluation & notification function
def evaluate_and_notify(candidate_email: str, combined_input: str):
    response = retrieval_chain.invoke({
        "input": combined_input,
        "agent_scratchpad": []
    })
    feedback = response["answer"].strip()

    base_subject = "Application Status - Your Recent Application"
    email_args = {
        "to_email": candidate_email,
        "subject": None,
        "body": None
    }

    if "not selected" in feedback.lower():
        email_args["subject"] = base_subject
        email_args["body"] = (
            f"Dear Candidate,\n\n"
            f"Thank you for your application. After careful review, we have decided not to move forward with your candidacy.\n\n"
            f"Below is our feedback based on your resume and the job description:\n\n"
            f"{feedback}\n\n"
            "We encourage you to apply for future openings that match your profile.\n\n"
            "Best regards,\nHR Team"
        )
    else:
        email_args["subject"] = "Application Status – Next Steps"
        email_args["body"] = (
            f"Dear Candidate,\n\n {feedback}\n\n"
            "We will reach out shortly to schedule an interview.\n\n"
            "Best regards,\nHR Team"
        )
    # Invoke structured email tool
    agent_executor.invoke({"input": email_args, "agent_scratchpad": []})

# Helper functions for file parsing
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

# Streamlit UI
st.title("HR Assistant - Resume Evaluation")
st.write("This application focuses on assisting the human resources process. " \
"It aims to speed up interactions for both job seekers and HR professionals.")

resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
jobdesc_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])

resume_text = ""
jobdesc_text = ""

if resume_file:
    resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)

if jobdesc_file:
    jobdesc_text = extract_text_from_pdf(jobdesc_file) if jobdesc_file.name.endswith(".pdf") else extract_text_from_docx(jobdesc_file)

if resume_text and jobdesc_text:
    st.success("Both Resume and Job Description uploaded successfully.")
    candidate_email = st.text_input("Candidate Email Address")
    if st.button("Evaluate Resume"):
        combined_input = (
            f"Here is the candidate's resume:\n{resume_text}\n\n"
            f"Here is the job description:\n{jobdesc_text}\n\n"
            "Evaluate whether the candidate fits the job description. "
            "If the candidate is **not selected**, provide **detailed feedback** explaining exactly which skills, experiences or qualifications did not align with the job requirements (do NOT draft the email itself). "
            "If the candidate **is selected**, simply confirm selection."
        )
        evaluate_and_notify(candidate_email, combined_input)
        st.success("Email has been successfully sent.")
        st.balloons()
