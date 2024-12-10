import os
import openai
import streamlit as st
from io import BytesIO
from docx import Document
import spacy
import asyncio
import time
from mods import kmns
# from mods import km

# Set page configuration
st.set_page_config(page_title="Interview Assistant", layout="wide")

# Initialize OpenAI client with Groq's API
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set or not available in secrets.")

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    st.error("The spaCy model 'en_core_web_sm' is not installed. Install it with: `python -m spacy download en_core_web_sm`")

# Asynchronous function for Groq API
async def get_groq_chat_response_async(question, primer, model="mixtral-8x7b-32768"):
    try:
        messages = [
            {"role": "system", "content": primer},
            {"role": "user", "content": question},
        ]
        chat_completion = await asyncio.to_thread(
            client.chat.completions.create,
            messages=messages,
            model=model
        )
        response = chat_completion.choices[0].message.content
        return response
    except openai.OpenAIError as e:
        st.error(f"Error calling Groq API: {e}")
        return None

# File reading functions
def read_file(file):
    return file.read().decode("utf-8")

def read_word_document(file):
    try:
        doc = Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return ""

def extract_skills(text):
    if not nlp:
        return []
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ORG"]]
    return set(skills)

# Main function
async def main():
    

    st.sidebar.subheader("⚙️ Settings")
    response_length = st.sidebar.slider("Response Length (words)", 50, 500, 100)
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["llama-3.2-3b-preview", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "llama-2-13b", "gemma2-9b-it"],
    )

    uploaded_resume = st.sidebar.file_uploader("Upload Resume", type=["txt", "docx"], key="resume_uploader")
    text = read_file(uploaded_resume) if uploaded_resume else ""

    uploaded_job_desc = st.sidebar.file_uploader("Upload Job Description", type=["txt", "docx"], key="job_desc_uploader")
    jobd = read_file(uploaded_job_desc) if uploaded_job_desc else ""

    if "transcriptions" not in st.session_state:
        st.session_state.transcriptions = []

    model_name = st.sidebar.selectbox("Whisper Model", ["base", "small", "medium", "large"], index=0)
    start_button = st.sidebar.button("Start Interview")
    stop_button = st.sidebar.button("End Interview")

    left_col, right_col = st.columns([3, 1])

    with left_col:
        st.title("🎤 RealTime Interview Assistant")
        st.markdown(
            "Prepare for interviews with AI-powered insights. Upload your resume and job description, record your question, and get expert-level responses in real-time."
        )

        ########primer
        primer = f"""
                    You are a professional interviewee, skilled at answering questions with clarity, confidence, and relevance.
                    - Your role is to provide thoughtful, well-structured, and natural answers based solely on the provided information.
                    - You are interviewing for this job ({jobd[:1000]}).
                    - Craft your responses to be professional yet conversational, showcasing my experience, skills, and enthusiasm for the role.
                    - Tailor your answers to align with the job requirements, strictly using specific examples from my experience in the provided text: {text[:2500]}.
                    - Do not add any external information or speculation. Base your responses only on the provided job description and personal experience.
                    - Keep your answers concise and engaging, within the specified word limit: {response_length}.
                    - Focus on demonstrating a deep understanding of the role, industry, and how my background uniquely qualifies me for the position.
                    - If you're unsure of how to respond, refer back to the provided experience for guidance.
                """
        ########## End of system prompt
        #start kmns 
        if start_button:
            st.session_state.transcriptions.clear()
            kmns.stop_signal.clear()
            kmns.transcription_results.clear()
            # kmns.audio_queue.queue.clear()
            
            kmns.threading.Thread(target=kmns.capture_audio, args=(16000, 5), daemon=True).start()
            kmns.threading.Thread(target=kmns.process_audio, args=(16000, 15, 2, model_name), daemon=True).start()

        st.markdown("### Answers:")
        transcription_area = st.empty()

        async def update_transcriptions():
            while not kmns.stop_signal.is_set():
                if kmns.transcription_results:
                    interviewer_questions = [
                        result.split(": ", 1)[1]
                        for result in kmns.transcription_results
                        if (result.startswith("interviewer") or result.startswith("others:")) and ": " in result
                    ]
                    kmns.transcription_results.clear()

                    #Extract interviewee answers
                    # interviewee_answers = [
                    #     result.split("interviewee: ", 1)[1]
                    #     for result in kmns.transcription_results
                    #     if result.startswith("interviewee:")
                    # ]

                    for question in interviewer_questions:
                        response = await get_groq_chat_response_async(question, primer, model_choice)
                        formatted_output = f"**Question:** {question} \n\n**Response:** {response}\n"
                        st.session_state.transcriptions.append(formatted_output)

                    transcription_area.markdown("\n".join(st.session_state.transcriptions))

                await asyncio.sleep(0.25)

        # Start the transcription update task
        if start_button:
            await update_transcriptions()

    with right_col:
        # st.subheader("🔑 Skills from job description")
        st.markdown("### Skills Needed:")
        if jobd:
            skills = extract_skills(jobd)
            if skills:
                
                st.markdown("\n".join([f"- {skill}" for skill in skills]))
            else:
                st.write("No skills found.")
            

    st.markdown("### 🙋 Was this response helpful?")
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
    with feedback_col1:
        if st.button("👍 Yes"):
            st.success("Thanks for your feedback!")
    with feedback_col2:
        if st.button("👎 No"):
            st.warning("We'll strive to improve.")
    with feedback_col3:
        if st.button("🤔 Neutral"):
            st.info("Thanks for your feedback!")

if __name__ == "__main__":
    asyncio.run(main())
