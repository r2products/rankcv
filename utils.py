# utils.py
"""
Utility Functions Module

Goal:
This module contains common helper functions used by both analyzer modules:
  - File upload/download (via Azure Blob Storage)
  - PDF reading and display
  - Extraction of emails, years of experience, and skills
  - Similarity score calculation and category determination
  - Chat history and session state management
  - Step instructions for the multi-step workflow
"""

import streamlit as st
import os
import base64
import io
import re
from pdfminer.high_level import extract_text
import time
from datetime import datetime
from dateutil import parser
import nltk
import spacy
from streamlit_tags import st_tags
from azure.storage.blob import BlobServiceClient

# Download NLTK stopwords and load spaCy model
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

# Azure Blob Storage setup
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connection_string:
    raise ValueError("Azure storage connection string not found in environment variables.")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client("resumebot")

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

def extract_emails(text):
    email_pattern = r'(?:[a-zA-Z0-9._%+-]\s*)+@\s*(?:(?:[a-zA-Z0-9.-]\s*)+\.\s*[a-zA-Z]{2,})'
    matches = re.findall(email_pattern, text)
    return [re.sub(r'\s+', '', match) for match in matches]

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
    # A basic extraction: look for common skill headers and split by delimiters.
    header_keywords = [
        "skills", "skill set", "technical skills", "areas of expertise",
        "competencies", "proficiencies", "abilities", "tools", "technologies"
    ]
    skills = []
    normalized_text = " ".join(text.split())
    pattern = r"(?:" + "|".join(re.escape(k) for k in header_keywords) + r")\s*[:\-]\s*(.+)"
    matches = re.findall(pattern, normalized_text, re.IGNORECASE)
    for match in matches:
        parts = re.split(r'[,;\n]', match)
        for part in parts:
            skill = part.strip()
            if skill:
                skills.append(skill)
    return list(set(skills))

def calculate_similarity_score(resume_text, jd_text, jd_key_skills, resume_years, jd_years):
    try:
        candidate_years = int(re.search(r"\d+", str(resume_years)).group(0))
    except Exception:
        candidate_years = 0
    try:
        required_years = int(re.search(r"\d+", str(jd_years)).group(0))
    except Exception:
        required_years = 0
    jd_skills = [skill.lower() for skill in jd_key_skills]
    skill_appearances = sum(1 for s in jd_skills if resume_text.lower().count(s) > 0)
    if len(jd_skills) > 0 and skill_appearances == 0:
        return 0
    exp_score = 100 if required_years == 0 or candidate_years >= required_years else (candidate_years / required_years * 100)
    total_occurrences = sum(min(3, resume_text.lower().count(s)) for s in jd_skills)
    max_possible = 3 * len(jd_skills)
    skills_score = (total_occurrences / max_possible) * 100 if max_possible else 0
    overall_score = 0.4 * exp_score + 0.6 * skills_score
    return min(overall_score, 100)

def get_similarity_category(score):
    if score >= 90:
        return "Best"
    elif score >= 70:
        return "Better"
    elif score >= 50:
        return "Good"
    elif score >= 30:
        return "Average"
    elif score >= 10:
        return "Bad"
    else:
        return "Worst"

def get_resonant_skill(resume_text, jd_key_skills):
    resume_text_lower = resume_text.lower()
    filtered_skills = [skill.lower() for skill in jd_key_skills]
    max_count = 0
    resonant_skill = None
    for skill in filtered_skills:
        count = resume_text_lower.count(skill)
        if count > max_count:
            max_count = count
            resonant_skill = skill
    return resonant_skill

def ordinal(n):
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

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
            add_message("bot", "Great! (Step 1) Upload your **Resume** (PDF only) using the widget below. Then type 'continue' in the chat.")
            st.session_state.step = "resume_upload"
        else:
            add_message("bot", "Please type 'ready' to begin.")
    elif current_step == "resume_upload":
        if user_message_clean == "continue":
            add_message("bot", "Proceeding to Job Description input. (Step 2) Now, choose your method:\nType 'upload' to upload a JD PDF or 'manual' to enter JD details manually.")
            st.session_state.step = "jd_choice"
    elif current_step == "jd_choice":
        if user_message_clean == "upload":
            add_message("bot", "You chose JD upload. (Step 2A) Please upload your Job Description (PDF only) using the widget below. Then type 'continue' when done.")
            st.session_state.step = "jd_upload"
        elif user_message_clean == "manual":
            add_message("bot", "You chose manual JD input. (Step 2B) Enter the JD details using the fields below. Then type 'continue' when ready.")
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
        st.session_state.matcher_type = st.sidebar.radio("Select Matcher Type", ["Multi-Resume Matcher", "Multi-JD Matcher"])
    else:
        st.session_state.matcher_type = st.sidebar.radio("Select Matcher Type", ["Multi-Resume Matcher", "Multi-JD Matcher"],
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
