#

# resumeanalyzer.py
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (e.g., for Azure in other modules)
load_dotenv()

# Require bcrypt for secure password hashing
try:
    import bcrypt
except ImportError:
    st.error("Missing dependency: please run 'pip install bcrypt'")
    st.stop()

# Configure Streamlit page
st.set_page_config(page_title="Smart Resume & JD Analyzer Chatbot", layout="wide")

# Single-user credentials
USERNAME = "Px9H3sK_Tq4Vw2Z8"
# Bcrypt hash for 'G7!vRm#4xZ8kLp2D'
PASSWORD_HASH = b"$2b$12$kjay0q1GOFZ4UncMF7pACe9BEHpaVSDsLW0ToTTaB.SoIgeIHIEDG"

# Initialize login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------
# Login Form
# -------------------------
def login_form():
    st.sidebar.subheader("🔒 Login")
    with st.sidebar.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
        if submitted:
            if user_id != USERNAME or not bcrypt.checkpw(password.encode(), PASSWORD_HASH):
                st.sidebar.error("Invalid username or password.")
            else:
                st.session_state.logged_in = True

# If not logged in, show login form and exit
if not st.session_state.logged_in:
    login_form()
    st.stop()

# -------------------------
# Main Resume Analyzer Interface
# -------------------------
from resumebot import run_smart_resume_analyzer
from resumebotai import run_ai_resume_analyzer

st.sidebar.title("Resume Analyzer Options")
model_type = st.sidebar.radio(
    "Select your model:",
    ["Smart Resume Analyzer", "AI Resume Analyzer"],
    key="model_type_radio"
)
st.sidebar.markdown(f"**Selected Model:** {model_type}")

if model_type == "Smart Resume Analyzer":
    run_smart_resume_analyzer()
else:
    run_ai_resume_analyzer()



