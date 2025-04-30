# resumebot.py
"""
Smart Resume Analyzer Module

Goal:
This module implements the Smart Resume Analyzer. It handles:
  - Resume (and Job Description) file uploads.
  - Text extraction and parsing (using common utilities).
  - Similarity scoring between resumes and job descriptions.
  - A multi-step interactive chat-based workflow.

Note: This module does not call st.set_page_config() since that is done in the main file.
"""

import streamlit as st
import time
from streamlit_tags import st_tags
from utils import (
    upload_file_to_blob,
    download_blob_to_stream,
    show_file,
    pdf_reader,
    extract_emails,
    extract_years_experience,
    extract_skills_from_text,
    calculate_similarity_score,
    get_similarity_category,
    get_resonant_skill,
    ordinal,
    add_message,
    display_chat,
    process_chat_input,
    init_session_state,
    get_instructions
)


def run_smart_resume_analyzer():
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
            # Clear previous resumes
            st.session_state.resumes = []
            st.markdown("### Step 1: Resume Upload")
            st.info("Upload your Resume(s) (PDF only) below. After uploading, type 'continue' in the chat.")
            resume_files = st.file_uploader("Select your Resume(s) (PDF only)", type=["pdf"], key="resume_files",
                                            accept_multiple_files=True)
            if resume_files:
                for resume_file in resume_files:
                    if not resume_file.name.lower().endswith("pdf"):
                        st.error(f"{resume_file.name} is not a valid PDF file.")
                        continue
                    blob_name = upload_file_to_blob(resume_file, "resumes")
                    st.info(f"Uploaded resume: **{blob_name}**")
                    file_stream = download_blob_to_stream(blob_name)
                    show_file(file_stream, "pdf")
                    resume_text = pdf_reader(file_stream)
                    st.success(f"Text extracted from {resume_file.name}")
                    emails = extract_emails(resume_text)
                    if emails:
                        st.write(f"Extracted Email(s) from {resume_file.name}: {', '.join(emails)}")
                    else:
                        st.write(f"No email addresses found in {resume_file.name}.")
                    resume_years_extracted = extract_years_experience(resume_text)
                    if resume_years_extracted is not None:
                        st.write(f"Extracted Years of Experience from {resume_file.name}: {resume_years_extracted}")
                    else:
                        st.write(f"No explicit years of experience found in {resume_file.name}.")
                        resume_years_extracted = 0.0
                    default_years = resume_years_extracted
                    manual_resume_years = st.number_input(
                        f"Manually enter your years of experience for {resume_file.name}",
                        min_value=0.0, value=default_years, step=0.1, format="%.1f",
                        key=f"manual_resume_years_{resume_file.name}"
                    )
                    resume_skills = extract_skills_from_text(resume_text)
                    if resume_skills:
                        st.write(f"Extracted Resume Skills from {resume_file.name}: " + ", ".join(resume_skills))
                    else:
                        st.write(f"No skills section found in {resume_file.name}.")
                    if "resumes" not in st.session_state:
                        st.session_state.resumes = []
                    st.session_state.resumes.append({
                        "file_name": resume_file.name,
                        "blob_name": blob_name,
                        "resume_text": resume_text,
                        "manual_years": manual_resume_years,
                        "emails": emails,
                        "skills": resume_skills
                    })
                add_message("bot",
                            "All resumes processed. Now type 'continue' in the chat to proceed to Job Description input.")
        if st.session_state.step == "jd_choice":
            st.markdown("### Step 2: Job Description Input")
            st.info(
                "Choose the input method:\n- Type 'upload' to upload a JD PDF.\n- Type 'manual' to enter JD key skills manually.")
        if st.session_state.step == "jd_upload":
            st.markdown("### Step 2A: JD Upload")
            st.info("Upload your Job Description (PDF only) below. After uploading, type 'continue' in the chat.")
            jd_file = st.file_uploader("Select your Job Description (PDF only)", type=["pdf"], key="jd_file")
            if jd_file:
                if not jd_file.name.lower().endswith("pdf"):
                    st.error("Please upload a valid PDF file.")
                else:
                    blob_name = upload_file_to_blob(jd_file, "JobDescriptions")
                    st.info(f"JD uploaded as: **{blob_name}**")
                    file_stream = download_blob_to_stream(blob_name)
                    show_file(file_stream, "pdf")
                    jd_text = pdf_reader(file_stream)
                    st.success("JD text extracted successfully.")
                    extracted_jd_key_skills = extract_skills_from_text(jd_text)
                    extracted_jd_years = extract_years_experience(jd_text)
                    if extracted_jd_key_skills:
                        add_message("bot", "Extracted JD Key Skills: " + ", ".join(extracted_jd_key_skills))
                    else:
                        add_message("bot", "No skills found in the JD.")
                    if extracted_jd_years is not None:
                        add_message("bot", f"Extracted JD Years: {extracted_jd_years}")
                    else:
                        add_message("bot", "No years specified in the JD.")
                    processed_jd_text = ", ".join(extracted_jd_key_skills)
                    st.write("Processed JD (Skills Only):", processed_jd_text)
                    st.session_state.jd_text = processed_jd_text
                    st.session_state.jd_key_skills = extracted_jd_key_skills
                    try:
                        st.session_state.jd_years = int(extracted_jd_years) if extracted_jd_years is not None else 0
                    except:
                        st.session_state.jd_years = 0
                    add_message("bot", "JD processed. Now type 'continue' in the chat to proceed to Score Calculation.")
        if st.session_state.step == "jd_manual":
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
        if st.session_state.step == "score":
            st.markdown("### Step 3: Score Calculation & Override")
            st.info(
                "Your Resume(s) and JD details will be compared to compute similarity scores. You may override JD parameters if desired.\nThe final ranking (Best Fit to Worst Fit) will be displayed below.")
            if "resumes" not in st.session_state or "jd_text" not in st.session_state:
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
                        score = calculate_similarity_score(
                            resume_text=resume["resume_text"],
                            jd_text=jd_text,
                            jd_key_skills=used_jd_skills,
                            resume_years=resume["manual_years"],
                            jd_years=override_jd_years
                        )
                        category = get_similarity_category(score)
                        resonant_skill = get_resonant_skill(resume["resume_text"], used_jd_skills)
                        resume_results.append({
                            "file_name": resume["file_name"],
                            "score": score,
                            "category": category,
                            "emails": resume["emails"],
                            "resonant_skill": resonant_skill
                        })
                    resume_results = sorted(resume_results, key=lambda x: x["score"], reverse=True)
                    st.markdown("### Resume Ranking (Best Fit to Worst Fit)")
                    for i, result in enumerate(resume_results, start=1):
                        st.write(
                            f"**{ordinal(i)}:** {result['file_name']} - {result['score']:.2f}% ({result['category']}) - Email(s): {', '.join(result['emails']) if result['emails'] else 'None'} - Resonant Skill: {result['resonant_skill'] if result['resonant_skill'] else 'None'}")
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
                if not resume_file.name.lower().endswith("pdf"):
                    st.error(f"{resume_file.name} is not a valid PDF file.")
                else:
                    blob_name = upload_file_to_blob(resume_file, "resumes")
                    st.info(f"Uploaded resume: **{blob_name}**")
                    file_stream = download_blob_to_stream(blob_name)
                    show_file(file_stream, "pdf")
                    resume_text = pdf_reader(file_stream)
                    st.success(f"Text extracted from {resume_file.name}")
                    emails = extract_emails(resume_text)
                    if emails:
                        st.write(f"Extracted Email(s) from {resume_file.name}: {', '.join(emails)}")
                    else:
                        st.write(f"No email addresses found in {resume_file.name}.")
                    resume_years_extracted = extract_years_experience(resume_text)
                    if resume_years_extracted is not None:
                        st.write(f"Extracted Years of Experience from {resume_file.name}: {resume_years_extracted}")
                    else:
                        st.write(f"No explicit years of experience found in {resume_file.name}.")
                        resume_years_extracted = 0.0
                    default_years = resume_years_extracted
                    manual_resume_years = st.number_input(
                        "Manually enter your years of experience",
                        min_value=0.0, value=default_years, step=0.1, format="%.1f",
                        key="manual_resume_years"
                    )
                    resume_skills = extract_skills_from_text(resume_text)
                    if resume_skills:
                        st.write("Extracted Resume Skills: " + ", ".join(resume_skills))
                    else:
                        st.write("No skills section found in the resume.")
                    st.session_state.resume = {
                        "file_name": resume_file.name,
                        "blob_name": blob_name,
                        "resume_text": resume_text,
                        "manual_years": manual_resume_years,
                        "emails": emails,
                        "skills": resume_skills
                    }
                    add_message("bot",
                                "Resume processed. Now type 'continue' in the chat to proceed to Job Description input.")
        if st.session_state.step == "jd_upload":
            st.session_state.jd_list = []
            st.markdown("### Step 2A: JD Upload")
            st.info("Upload your Job Description(s) (PDF only) below. After uploading, type 'continue' in the chat.")
            jd_files = st.file_uploader("Select your Job Description(s) (PDF only)", type=["pdf"], key="jd_files",
                                        accept_multiple_files=True)
            if jd_files:
                for jd_file in jd_files:
                    if not jd_file.name.lower().endswith("pdf"):
                        st.error(f"{jd_file.name} is not a valid PDF file.")
                        continue
                    blob_name = upload_file_to_blob(jd_file, "JobDescriptions")
                    st.info(f"JD uploaded as: **{blob_name}**")
                    file_stream = download_blob_to_stream(blob_name)
                    show_file(file_stream, "pdf")
                    jd_text = pdf_reader(file_stream)
                    st.success(f"JD text extracted from {jd_file.name}")
                    extracted_jd_key_skills = extract_skills_from_text(jd_text)
                    extracted_jd_years = extract_years_experience(jd_text)
                    if extracted_jd_key_skills:
                        add_message("bot", f"Extracted JD Key Skills from {jd_file.name}: " + ", ".join(
                            extracted_jd_key_skills))
                    else:
                        add_message("bot", f"No skills found in {jd_file.name}.")
                    if extracted_jd_years is not None:
                        add_message("bot", f"Extracted JD Years from {jd_file.name}: {extracted_jd_years}")
                    else:
                        add_message("bot", f"No years specified in {jd_file.name}.")
                    processed_jd_text = ", ".join(extracted_jd_key_skills)
                    st.write(f"Processed JD (Skills Only) from {jd_file.name}:", processed_jd_text)
                    st.session_state.jd_list.append({
                        "file_name": jd_file.name,
                        "blob_name": blob_name,
                        "jd_text": processed_jd_text,
                        "jd_key_skills": extracted_jd_key_skills,
                        "jd_years": int(extracted_jd_years) if extracted_jd_years is not None else 0
                    })
                add_message("bot",
                            "All Job Descriptions processed. Now type 'continue' in the chat to proceed to Score Calculation.")
        if st.session_state.step == "jd_manual":
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
        if st.session_state.step == "score":
            st.markdown("### Step 3: Score Calculation & Override")
            st.info(
                "Your Resume and Job Description details will be compared to compute similarity scores.\nYou may override JD parameters if desired. The final ranking (Best Fit to Worst Fit) will be displayed below.")
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
                        score = calculate_similarity_score(
                            resume_text=resume_text,
                            jd_text=jd["jd_text"],
                            jd_key_skills=jd["jd_key_skills"],
                            resume_years=resume_years,
                            jd_years=jd["jd_years"]
                        )
                        category = get_similarity_category(score)
                        resonant_skill = get_resonant_skill(resume_text, jd["jd_key_skills"])
                        jd_results.append({
                            "file_name": jd["file_name"],
                            "score": score,
                            "category": category,
                            "resonant_skill": resonant_skill
                        })
                    jd_results = sorted(jd_results, key=lambda x: x["score"], reverse=True)
                    st.markdown("### Job Description Ranking (Best Fit to Worst Fit)")
                    for i, result in enumerate(jd_results, start=1):
                        st.write(
                            f"**{ordinal(i)}:** {result['file_name']} - {result['score']:.2f}% ({result['category']}) - Resonant Skill: {result['resonant_skill'] if result['resonant_skill'] else 'None'}")
                    progress_bar = st.progress(0)
                    for percent in range(101):
                        time.sleep(0.005)
                        progress_bar.progress(percent)
                    st.success("Score Calculation Complete.")
                    add_message("bot", "Final job description ranking displayed above.")

    # Restart Option
    if st.button("Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.write("Session restarted. Please refresh the page.")
        st.stop()
