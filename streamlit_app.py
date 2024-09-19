import streamlit as st
import docx
import openai
import anthropic
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API keys
def get_openai_api_key():
    if 'openai' in st.secrets:
        return st.secrets['openai']['api_key']
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY as an environment variable or in Streamlit secrets.")
    return api_key

def get_anthropic_api_key():
    if 'anthropic' in st.secrets:
        return st.secrets['anthropic']['api_key']
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key is None:
        raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY as an environment variable or in Streamlit secrets.")
    return api_key

# Set up clients
openai.api_key = get_openai_api_key()
anthropic_client = anthropic.Anthropic(api_key=get_anthropic_api_key())

def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def analyze_thread(thread, model, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    prompt = f"""
    You are an experienced reliability engineer. Analyze the following email thread related to machinery defects, incidents, or troubles, and format the data under these headings:
    - Failure Mode
    - Failure Symptom
    - Failure Effect
    - Failure Cause

    Use these FMEA (Failure Mode and Effects Analysis) definitions:

    Failure Mode: A specific combination of a component and a verb that describes how the component fails to perform its intended function. It is the precise way in which an item or system's ability to perform its required function is lost or degraded. (Example: "Piston ring fractures")

    Failure Symptom: An observable indicator that a failure mode is occurring or has occurred. (Example: "Increased vibration" or "Oil leakage")

    Failure Effect: The resulting impact or consequence of a failure mode on the system's performance or operation. (Example: "Reduced engine power" or "Loss of hydraulic pressure")

    Failure Cause: The underlying reason or mechanism that leads to the occurrence of a failure mode. (Example: "Wear and tear" or "Contaminated fuel")

    Create a detailed incident case study out of the email thread. Also include a timeline of events if it is available in the email thread. Extract as much meaningful data as possible from the email thread and include it in your analysis.

    Email thread to analyze:
    {thread}
    """

    if model == "OpenAI":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response.choices[0].message.content
    elif model == "Claude":
        response = anthropic_client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            model="claude-2",
            max_tokens_to_sample=1000,
            temperature=temperature,
            top_p=top_p,
        )
        return response.completion

def main():
    st.title("Single Email Thread FMEA Analyzer")

    # Add model selection
    model = st.sidebar.selectbox("Select Model", ["OpenAI", "Claude"])

    uploaded_file = st.file_uploader("Choose a DOCX file containing a single email thread", type="docx")

    if uploaded_file is not None:
        content = read_docx(uploaded_file)
        st.write("Email thread content:")
        st.write(content)

        # Add sliders for LLM parameters
        st.sidebar.header("LLM Parameters")
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0, 0.1)
        
        # Note: Claude API doesn't use frequency_penalty and presence_penalty
        if model == "OpenAI":
            frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
            presence_penalty = st.sidebar.slider("Presence Penalty", 0.0, 2.0, 0.0, 0.1)
        else:
            frequency_penalty = 0.0
            presence_penalty = 0.0

        if st.button("Analyze"):
            with st.spinner("Analyzing email thread..."):
                analysis = analyze_thread(content, model, temperature, top_p, frequency_penalty, presence_penalty)
                st.subheader("FMEA Analysis:")
                st.write(analysis)

if __name__ == "__main__":
    main()
