# resumebotai.py
"""
AI Resume Analyzer Module

Goal:
This module implements the AI Resume Analyzer using Groq integration. It provides functionality to:
  - Upload resume and job description PDFs.
  - Extract text, skills, years of experience, and email addresses from the documents.
  - Use the Groq API to process and analyze the text.
  - Compare resume details with job description requirements and produce a detailed match analysis.

Note: Do not include st.set_page_config() here because that must be called only once as the very first command in the main file.
"""

import os
from dotenv import load_dotenv
import streamlit as st
import nltk
import spacy
import base64
import time
import io
from pdfminer.high_level import extract_text
from streamlit_tags import st_tags
import ssl
from azure.storage.blob import BlobServiceClient
import re
from datetime import datetime
from dateutil import parser

# ------------------------- Groq Setup -------------------------
# Load the Groq API key from environment variables
groq_secret_key = os.getenv("GROQ_SECRET_KEY")
if not groq_secret_key:
    raise ValueError("GROQ_SECRET_KEY not found in environment variables.")
os.environ["GROQ_API_KEY"] = groq_secret_key  # Set the API key for Groq

# Import and initialize the Groq client
from groq import Groq

client = Groq()


def groq_call(prompt):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    output = completion.choices[0].message.content
    return output.strip()


# ------------------------- Azure Setup -------------------------
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connection_string:
    raise ValueError("Azure storage connection string not found in environment variables.")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client("resumebot")

# ------------------------- NLP & Model Loading -------------------------
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')


# ------------------------- Utility Functions -------------------------
def upload_file_to_blob(file, folder):
    blob_name = f"{folder}/{file.name}"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(file.getvalue(), overwrite=True)
    return blob_name


def download_blob_to_stream(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    stream_downloader = blob_client.download_blob()
    file_bytes = stream_downloader.readall()
    return io.BytesIO(file_bytes)


def show_file(file_stream, file_extension):
    if file_stream and file_extension.lower() == "pdf":
        file_stream.seek(0)
        base64_pdf = base64.b64encode(file_stream.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        file_stream.seek(0)
    else:
        st.error("Please upload a PDF file.")


def pdf_reader(file_stream):
    file_stream.seek(0)
    return extract_text(file_stream)


def process_resume_file(resume_file):
    if not resume_file.name.lower().endswith("pdf"):
        st.error(f"{resume_file.name} is not a valid PDF file.")
        return None
    blob_name = upload_file_to_blob(resume_file, "resumes")
    st.info(f"Uploaded resume: **{blob_name}**")
    file_stream = download_blob_to_stream(blob_name)
    show_file(file_stream, "pdf")
    resume_text = pdf_reader(file_stream)
    st.success(f"Text extracted from {resume_file.name}")
    emails = extract_emails(resume_text)
    if emails:
        st.write(f"Extracted Email(s): {', '.join(emails)}")
    else:
        st.write("No email addresses found.")
    resume_years = extract_years_experience(resume_text)
    if resume_years is not None:
        st.write(f"Extracted Years of Experience: {resume_years}")
    else:
        st.write("No explicit years of experience found.")
        resume_years = 0.0
    manual_years = st.number_input(
        f"Manually enter your years of experience for {resume_file.name}",
        min_value=0.0, value=resume_years, step=0.1, format="%.1f",
        key=f"manual_resume_years_{resume_file.name}"
    )
    resume_skills = extract_skills_from_text(resume_text)
    if resume_skills:
        st.write(f"Extracted Resume Skills: {', '.join(resume_skills)}")
    else:
        st.write("No skills section found.")
    return {
        "file_name": resume_file.name,
        "blob_name": blob_name,
        "resume_text": resume_text,
        "manual_years": manual_years,
        "emails": emails,
        "skills": resume_skills
    }


def process_jd_file(jd_file):
    if not jd_file.name.lower().endswith("pdf"):
        st.error(f"{jd_file.name} is not a valid PDF file.")
        return None
    blob_name = upload_file_to_blob(jd_file, "JobDescriptions")
    st.info(f"JD uploaded as: **{blob_name}**")
    file_stream = download_blob_to_stream(blob_name)
    show_file(file_stream, "pdf")
    jd_text = pdf_reader(file_stream)
    st.success(f"JD text extracted from {jd_file.name}")
    extracted_jd_key_skills = extract_skills_from_text(jd_text)
    extracted_jd_years = extract_years_experience(jd_text)
    if extracted_jd_key_skills:
        st.write("Extracted JD Key Skills: " + ", ".join(extracted_jd_key_skills))
    else:
        st.write(f"No skills found in {jd_file.name}.")
    if extracted_jd_years is not None:
        st.write(f"Extracted JD Years: {extracted_jd_years}")
    else:
        st.write(f"No years specified in {jd_file.name}.")
    processed_jd_text = ", ".join(extracted_jd_key_skills)
    st.write("Processed JD (Skills Only):", processed_jd_text)
    return {
        "file_name": jd_file.name,
        "blob_name": blob_name,
        "jd_text": processed_jd_text,
        "jd_key_skills": extracted_jd_key_skills,
        "jd_years": int(extracted_jd_years) if extracted_jd_years is not None else 0
    }


# ------------------------- Groq Text Processing Functions -------------------------
def normalize_document_text(text):
    prompt = f"Clean and normalize the following text: {text}"
    return groq_call(prompt)


def extract_years_experience(text):
    regex_explicit = re.compile(
        r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of\s*experience|exp)?',
        re.IGNORECASE
    )
    explicit_matches = regex_explicit.findall(text)
    if explicit_matches:
        try:
            years_list = [float(match) for match in explicit_matches]
            return round(max(years_list), 1)
        except ValueError:
            pass
    regex_date = re.compile(
        r'([A-Za-z]{3,}\.?\s+\d{4})\s*(?:–|-)\s*(Present|[A-Za-z]{3,}\.?\s+\d{4})',
        re.IGNORECASE
    )
    date_matches = regex_date.findall(text)
    intervals = []
    for start_str, end_str in date_matches:
        try:
            start_date = parser.parse(start_str)
            end_date = datetime.now() if 'present' in end_str.lower() else parser.parse(end_str)
            intervals.append((start_date, end_date))
        except Exception:
            continue
    if intervals:
        intervals.sort(key=lambda interval: interval[0])
        merged_intervals = [intervals[0]]
        for current in intervals[1:]:
            last = merged_intervals[-1]
            if current[0] <= last[1]:
                merged_intervals[-1] = (last[0], max(last[1], current[1]))
            else:
                merged_intervals.append(current)
        total_days = sum((end - start).days for start, end in merged_intervals)
        total_years = total_days / 365.25
        return round(total_years, 1)
    return None


def extract_skills_from_text(text):
    prompt = f"Extract all skills from the following resume text. Return a comma-separated list of one or two word skills after removing adjectives, adverbs, and generic recruiter terms. Text: {text}"
    result = groq_call(prompt)
    return [s.strip() for s in result.split(",") if s.strip()]


def extract_emails(text):
    prompt = f"Extract email addresses from the following text: {text}. Return them as a comma-separated list. If none found, return an empty list."
    result = groq_call(prompt)
    return [s.strip() for s in result.split(",") if s.strip()]


# ------------------------- Analysis Functions -------------------------
def analyze_similarity(resume_text, jd_text, jd_key_skills, resume_years, jd_years):
    skills_str = ", ".join(jd_key_skills)
    prompt = (
        f"You are a resume matching expert. Evaluate the following resume and job description in detail.\n"
        f"Resume text: {resume_text}\n"
        f"Job description text: {jd_text}\n"
        f"Job description key skills: {skills_str}\n"
        f"Candidate's years of experience: {resume_years}\n"
        f"Required years of experience: {jd_years}\n"
        f"Provide a summarized reasoning covering key skills overlap, experience matching, text similarity, strengths and weaknesses, and suggestions for improvement.\n"
        f"Then, on a new line, output the final summary in the following format:\n"
        f"<score>|<category>|<resonant_skill>\n"
        f"where <score> is a number between 0 and 100 representing the match percentage, "
        f"<category> is one of Worst, Bad, Average, Good, Best, and <resonant_skill> is the most relevant key skill."
    )
    result = groq_call(prompt)
    lines = [line.strip() for line in result.strip().splitlines() if line.strip()]
    summary_line = ""
    for line in reversed(lines):
        if re.match(r'^\d+(\.\d+)?\|.+\|.*$', line):
            summary_line = line
            break
    explanation = "\n".join(lines[:-1]) if summary_line else "\n".join(lines)
    parts = summary_line.split('|') if summary_line else []
    if len(parts) >= 3:
        try:
            score = float(parts[0].strip())
        except:
            score = 0.0
        category = parts[1].strip()
        resonant_skill = parts[2].strip() if parts[2].strip() else None
    else:
        score, category, resonant_skill = 0.0, "Worst", None
    return score, category, resonant_skill, explanation


def ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ------------------------- Chat & Session Helpers -------------------------
def add_message(sender, message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"sender": sender, "message": message})


def display_chat():
    if "chat_history" in st.session_state:
        for chat in st.session_state.chat_history:
            st.markdown(f"**{chat['sender'].capitalize()}:** {chat['message']}")


def process_chat_input(user_message):
    current_step = st.session_state.get("step", "init")
    user_message_clean = user_message.strip().lower()
    add_message("user", user_message)
    if current_step == "init":
        if user_message_clean == "ready":
            add_message("bot",
                        "Great! (Step 1) Upload your **Resume** (PDF only) using the widget below. Then type 'continue' in the chat.")
            st.session_state.step = "resume_upload"
        else:
            add_message("bot", "Please type 'ready' to begin.")
    elif current_step == "resume_upload":
        if user_message_clean == "continue":
            add_message("bot",
                        "Proceeding to Job Description input. (Step 2) Now, choose your method:\nType 'upload' to upload a JD PDF or 'manual' to enter JD details manually.")
            st.session_state.step = "jd_choice"
    elif current_step == "jd_choice":
        if user_message_clean == "upload":
            add_message("bot",
                        "You chose JD upload. (Step 2A) Please upload your Job Description (PDF only) using the widget below. Then type 'continue' when done.")
            st.session_state.step = "jd_upload"
        elif user_message_clean == "manual":
            add_message("bot",
                        "You chose manual JD input. (Step 2B) Enter the JD details using the fields below. Then type 'continue' when ready.")
            st.session_state.step = "jd_manual"
        else:
            add_message("bot", "Invalid input. Please type 'upload' or 'manual'.")
    elif current_step in ["jd_upload", "jd_manual"]:
        if user_message_clean == "continue":
            add_message("bot", "Proceeding to Score Calculation. (Step 3) Your scores will be computed shortly.")
            st.session_state.step = "score"
        else:
            add_message("bot", "Please type 'continue' when you have completed the JD input.")
    elif current_step == "score":
        add_message("bot", "To recalculate or restart, please use the buttons provided below.")


def init_session_state():
    if "matcher_type" not in st.session_state:
        st.session_state.matcher_type = st.sidebar.radio("Select Matcher Type",
                                                         ["Multi-Resume Matcher", "Multi-JD Matcher"])
    else:
        st.session_state.matcher_type = st.sidebar.radio("Select Matcher Type",
                                                         ["Multi-Resume Matcher", "Multi-JD Matcher"],
                                                         index=0 if st.session_state.matcher_type == "Multi-Resume Matcher" else 1)
    if "step" not in st.session_state:
        st.session_state.step = "init"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        add_message("bot", "Hi, I am the Resume-JD Analyser. (Type 'ready' to proceed.)")
    if st.session_state.matcher_type == "Multi-Resume Matcher":
        if "resumes" not in st.session_state:
            st.session_state.resumes = []
    else:
        if "resume" not in st.session_state:
            st.session_state.resume = None
        if "jd_list" not in st.session_state:
            st.session_state.jd_list = []


def get_instructions():
    step = st.session_state.get("step", "init")
    mode = st.session_state.get("matcher_type", "Multi-Resume Matcher")
    if step == "init":
        return "Step 0: Start\nType 'ready' in the chat box to begin."
    elif step == "resume_upload":
        if mode == "Multi-Resume Matcher":
            return "Step 1: Resume Upload\nUpload your Resume(s) (PDF only) using the widget below.\nThen type 'continue' in the chat."
        else:
            return "Step 1: Resume Upload\nUpload your Resume (PDF only) using the widget below.\nThen type 'continue' in the chat."
    elif step == "jd_choice":
        return "Step 2: Job Description Input\nChoose your method:\nType 'upload' to upload a JD PDF or 'manual' for manual JD input."
    elif step == "jd_upload":
        if mode == "Multi-Resume Matcher":
            return "Step 2A: JD Upload\nUpload your Job Description (PDF only) using the widget below.\nThen type 'continue' in the chat."
        else:
            return "Step 2A: JD Upload\nUpload your Job Description(s) (PDF only) using the widget below.\nThen type 'continue' in the chat."
    elif step == "jd_manual":
        if mode == "Multi-Resume Matcher":
            return "Step 2B: Manual JD Input\nEnter JD key skills and required years using the fields below.\nClick 'Save Manual JD Input' and type 'continue'."
        else:
            return "Step 2B: Manual JD Input\nEnter your Job Description details below. You can add multiple JDs.\nClick 'Add JD' after each entry, then type 'continue'."
    elif step == "score":
        if mode == "Multi-Resume Matcher":
            return "Step 3: Score Calculation\nYour Resume(s) and JD details will be compared to compute similarity scores.\nThe ranking (Best Fit to Worst Fit) will be displayed below."
        else:
            return "Step 3: Score Calculation\nYour Resume and Job Description details will be compared to compute similarity scores.\nThe ranking (Best Fit to Worst Fit) will be displayed below."
    else:
        return "Follow the instructions in the chat."


# ------------------------- Main Function for AI Resume Analyzer -------------------------
def run_ai_resume_analyzer():
    # Initialize session state
    init_session_state()

    # Display sidebar instructions and step toggle
    st.sidebar.markdown("## Instructions")
    st.sidebar.markdown(get_instructions())

    toggle_options = ["Resume Upload", "Job Description Input", "JD Upload", "JD Manual Input", "Score Calculation"]
    step_mapping = {
        "Resume Upload": "resume_upload",
        "Job Description Input": "jd_choice",
        "JD Upload": "jd_upload",
        "JD Manual Input": "jd_manual",
        "Score Calculation": "score"
    }
    toggle_value = st.sidebar.radio("Toggle Steps", toggle_options, key="toggle_steps")
    if st.sidebar.button("Go to Step"):
        st.session_state.step = step_mapping[toggle_value]

    # Chat & Conversation Container
    with st.container():
        st.markdown("## Conversation & Chat Input")
        display_chat()
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Enter your message (see sidebar for instructions):")
            submitted = st.form_submit_button("Send")
            if submitted and user_input:
                process_chat_input(user_input)

    # Mode-Specific UI
    if st.session_state.matcher_type == "Multi-Resume Matcher":
        # Multi-Resume Matcher: Multiple Resumes, Single JD
        if st.session_state.step == "resume_upload":
            st.markdown("### Step 1: Resume Upload")
            st.info("Upload your Resume(s) (PDF only) below. After uploading, type 'continue' in the chat.")
            resume_files = st.file_uploader("Select your Resume(s) (PDF only)", type=["pdf"], key="resume_files",
                                            accept_multiple_files=True)
            if resume_files:
                st.session_state.resumes = []
                for resume_file in resume_files:
                    result = process_resume_file(resume_file)
                    if result:
                        st.session_state.resumes.append(result)
                add_message("bot",
                            "All resumes processed. Now type 'continue' in the chat to proceed to Job Description input.")
        elif st.session_state.step == "jd_choice":
            st.markdown("### Step 2: Job Description Input")
            st.info(
                "Choose the input method:\n- Type 'upload' to upload a JD PDF.\n- Type 'manual' to enter JD key skills manually.")
        elif st.session_state.step == "jd_upload":
            st.markdown("### Step 2A: JD Upload")
            st.info("Upload your Job Description (PDF only) below. After uploading, type 'continue' in the chat.")
            jd_file = st.file_uploader("Select your Job Description (PDF only)", type=["pdf"], key="jd_file")
            if jd_file:
                jd_result = process_jd_file(jd_file)
                if jd_result:
                    st.session_state.jd_text = jd_result["jd_text"]
                    st.session_state.jd_key_skills = jd_result["jd_key_skills"]
                    st.session_state.jd_years = jd_result["jd_years"]
                    add_message("bot", "JD processed. Now type 'continue' in the chat to proceed to Score Calculation.")
        elif st.session_state.step == "jd_manual":
            st.markdown("### Step 2B: Manual JD Input")
            st.info(
                "Enter your JD key skills and required years below. Then click 'Save Manual JD Input' and type 'continue' in the chat.")
            manual_keywords_input = st_tags(
                label="JD Key Skills (manual)",
                text="Enter key skills separated by commas",
                value=[],
                key="manual_jd_keywords"
            )
            manual_years = st.number_input("Enter required years of experience", min_value=0, value=0, step=1,
                                           key="manual_jd_years")
            if st.button("Save Manual JD Input"):
                processed_keywords = []
                for keyword in manual_keywords_input:
                    processed_keywords.extend([skill.strip() for skill in keyword.split(",") if skill.strip()])
                jd_text = ", ".join(processed_keywords) if processed_keywords else ""
                st.session_state.jd_text = jd_text
                st.session_state.jd_key_skills = processed_keywords
                st.session_state.jd_years = manual_years
                add_message("bot",
                            "Manual JD input recorded. Now type 'continue' in the chat to proceed to Score Calculation.")
        elif st.session_state.step == "score":
            st.markdown("### Step 3: Score Calculation & Override")
            st.info(
                "Your Resume(s) and JD details will be compared to compute similarity scores. You may override JD parameters if desired.\nThe final ranking (Best Fit to Worst Fit) will be displayed below along with detailed reasoning.")
            if not st.session_state.resumes or "jd_text" not in st.session_state:
                st.warning("Please complete both Resume and JD steps first.")
            else:
                jd_text = st.session_state.jd_text
                current_jd_keywords = st.session_state.get("jd_key_skills", [])
                current_jd_years = st.session_state.get("jd_years", 0)
                st.write("**Current JD Key Skills:**", ", ".join(current_jd_keywords))
                st.write("**Current JD Required Years:**", current_jd_years)
                st.subheader("Override JD Parameters (Optional)")
                override_jd_keywords_input = st_tags(
                    label="Override JD Key Skills",
                    text="Enter new key skills separated by commas",
                    value=current_jd_keywords,
                    key="override_jd_keywords"
                )
                override_processed_keywords = []
                for keyword in override_jd_keywords_input:
                    override_processed_keywords.extend([skill.strip() for skill in keyword.split(",") if skill.strip()])
                override_jd_years = st.number_input("Override required years", min_value=0, value=int(current_jd_years),
                                                    step=1, key="override_jd_years")
                used_jd_skills = override_processed_keywords if override_processed_keywords else current_jd_keywords
                if st.button("Calculate Score"):
                    resume_results = []
                    for resume in st.session_state.resumes:
                        score, category, resonant_skill, explanation = analyze_similarity(
                            resume_text=resume["resume_text"],
                            jd_text=jd_text,
                            jd_key_skills=used_jd_skills,
                            resume_years=resume["manual_years"],
                            jd_years=override_jd_years
                        )
                        resume_results.append({
                            "file_name": resume["file_name"],
                            "score": score,
                            "category": category,
                            "emails": resume["emails"],
                            "resonant_skill": resonant_skill,
                            "explanation": explanation
                        })
                    resume_results = sorted(resume_results, key=lambda x: x["score"], reverse=True)
                    st.markdown("### Resume Ranking (Best Fit to Worst Fit)")
                    for i, result in enumerate(resume_results, start=1):
                        st.write(f"**{ordinal(i)}: {result['file_name']}**")
                        st.markdown("**Detailed Reasoning:**")
                        explanation_lines = result['explanation'].splitlines()
                        filtered_explanation_lines = [line for line in explanation_lines if
                                                      not line.startswith("Final Summary:")]
                        filtered_explanation = "\n".join(filtered_explanation_lines)
                        st.write(filtered_explanation)
                        final_summary = f"Final Summary: {result['score']}|{result['category']}|{result['resonant_skill'] if result['resonant_skill'] else 'None'}"
                        st.write(final_summary)
                    progress_bar = st.progress(0)
                    for percent in range(101):
                        time.sleep(0.005)
                        progress_bar.progress(percent)
                    st.success("Score Calculation Complete.")
                    add_message("bot", "Final resume ranking displayed above.")
    else:
        # Multi-JD Matcher: Single Resume, Multiple JDs
        if st.session_state.step == "resume_upload":
            st.markdown("### Step 1: Resume Upload")
            st.info("Upload your **Resume** (PDF only) below. After uploading, type 'continue' in the chat.")
            resume_file = st.file_uploader("Select your Resume (PDF only)", type=["pdf"], key="resume_file")
            if resume_file:
                result = process_resume_file(resume_file)
                if result:
                    st.session_state.resume = result
                    add_message("bot",
                                "Resume processed. Now type 'continue' in the chat to proceed to Job Description input.")
        elif st.session_state.step == "jd_upload":
            st.session_state.jd_list = []
            st.markdown("### Step 2A: JD Upload")
            st.info("Upload your Job Description(s) (PDF only) below. After uploading, type 'continue' in the chat.")
            jd_files = st.file_uploader("Select your Job Description(s) (PDF only)", type=["pdf"], key="jd_files",
                                        accept_multiple_files=True)
            if jd_files:
                for jd_file in jd_files:
                    jd_result = process_jd_file(jd_file)
                    if jd_result:
                        st.session_state.jd_list.append(jd_result)
                add_message("bot",
                            "All Job Descriptions processed. Now type 'continue' in the chat to proceed to Score Calculation.")
        elif st.session_state.step == "jd_manual":
            st.markdown("### Step 2B: Manual JD Input")
            st.info(
                "Enter your Job Description details below. You can add multiple JDs.\nClick 'Add JD' after each entry, then type 'continue' in the chat when done.")
            manual_keywords_input = st_tags(
                label="JD Key Skills (manual)",
                text="Enter key skills separated by commas",
                value=[],
                key="manual_jd_keywords"
            )
            manual_years = st.number_input("Enter required years of experience", min_value=0, value=0, step=1,
                                           key="manual_jd_years")
            if st.button("Add JD"):
                processed_keywords = []
                for keyword in manual_keywords_input:
                    processed_keywords.extend([skill.strip() for skill in keyword.split(",") if skill.strip()])
                jd_text = ", ".join(processed_keywords) if processed_keywords else ""
                st.session_state.jd_list.append({
                    "jd_text": jd_text,
                    "jd_key_skills": processed_keywords,
                    "jd_years": manual_years,
                    "file_name": "Manual JD"
                })
                add_message("bot",
                            "Manual JD input recorded. You can add more or type 'continue' in the chat to proceed to Score Calculation.")
        elif st.session_state.step == "score":
            st.markdown("### Step 3: Score Calculation & Override")
            st.info(
                "Your Resume and Job Description details will be compared to compute similarity scores.\nYou may override JD parameters if desired. The final ranking (Best Fit to Worst Fit) will be displayed below along with detailed reasoning.")
            if not st.session_state.resume or not st.session_state.jd_list:
                st.warning("Please complete both Resume and JD steps first.")
            else:
                resume_text = st.session_state.resume["resume_text"]
                resume_years = st.session_state.resume["manual_years"]
                st.write("**Current Job Descriptions:**")
                for idx, jd in enumerate(st.session_state.jd_list, start=1):
                    st.write(
                        f"JD {idx}: Key Skills: {', '.join(jd['jd_key_skills'])} | Required Years: {jd['jd_years']}")
                st.subheader("Override JD Parameters (Optional)")
                override_jd_list = []
                for idx, jd in enumerate(st.session_state.jd_list, start=1):
                    st.markdown(f"#### Override for JD {idx}")
                    override_keywords = st_tags(
                        label=f"Override JD {idx} Key Skills",
                        text="Enter new key skills separated by commas",
                        value=jd['jd_key_skills'],
                        key=f"override_jd_keywords_{idx}"
                    )
                    override_years = st.number_input(f"Override required years for JD {idx}", min_value=0,
                                                     value=int(jd['jd_years']), step=1, key=f"override_jd_years_{idx}")
                    override_processed_keywords = [skill.strip() for skill in override_keywords if skill.strip()]
                    override_jd_list.append({
                        "jd_key_skills": override_processed_keywords if override_processed_keywords else jd[
                            'jd_key_skills'],
                        "jd_years": override_years,
                        "jd_text": jd["jd_text"],
                        "file_name": jd["file_name"]
                    })
                if st.button("Calculate Score"):
                    jd_results = []
                    for jd in override_jd_list:
                        score, category, resonant_skill, explanation = analyze_similarity(
                            resume_text=resume_text,
                            jd_text=jd["jd_text"],
                            jd_key_skills=jd["jd_key_skills"],
                            resume_years=resume_years,
                            jd_years=jd["jd_years"]
                        )
                        jd_results.append({
                            "file_name": jd["file_name"],
                            "score": score,
                            "category": category,
                            "resonant_skill": resonant_skill,
                            "explanation": explanation
                        })
                    jd_results = sorted(jd_results, key=lambda x: x["score"], reverse=True)
                    st.markdown("### Job Description Ranking (Best Fit to Worst Fit)")
                    for i, result in enumerate(jd_results, start=1):
                        st.write(f"**{ordinal(i)}: {result['file_name']}**")
                        st.markdown("**Detailed Reasoning:**")
                        explanation_lines = result['explanation'].splitlines()
                        filtered_explanation_lines = [line for line in explanation_lines if
                                                      not line.startswith("Final Summary:")]
                        filtered_explanation = "\n".join(filtered_explanation_lines)
                        st.write(filtered_explanation)
                        final_summary = f"Final Summary: {result['score']}|{result['category']}|{result['resonant_skill'] if result['resonant_skill'] else 'None'}"
                        st.write(final_summary)
                    progress_bar = st.progress(0)
                    for percent in range(101):
                        time.sleep(0.005)
                        progress_bar.progress(percent)
                    st.success("Score Calculation Complete.")
                    add_message("bot", "Final job description ranking displayed above.")

    # ------------------------- Restart Option -------------------------
    if st.button("Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.write("Session restarted. Please refresh the page.")
        st.stop()
