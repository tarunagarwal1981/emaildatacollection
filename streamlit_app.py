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
    # Prompt for FMECA and Incident Case Study Generation

You are an experienced reliability engineer tasked with generating a comprehensive FMECA (Failure Modes, Effects, and Criticality Analysis) and an accompanying incident case study for a marine vessel that experienced a specific failure mode. Your analysis should be based solely on the facts provided in the given data, without introducing any external information or assumptions.

## Context
A marine vessel has experienced a Main Engine (ME) failure to start in astern during critical maneuvers. Your task is to analyze this incident in detail, providing both a FMECA and a case study.

## FMEA (Failure Mode and Effects Analysis) Definitions

Use the following definitions to identify and categorize failure modes, symptoms, effects, and causes:

1. Failure Mode:
   A specific combination of a component and a verb that describes how the component fails to perform its intended function. It is the precise way in which an item or system's ability to perform its required function is lost or degraded.
   Example: "Piston ring fractures"

2. Failure Symptom:
   An observable indicator that a failure mode is occurring or has occurred.
   Example: "Increased vibration" or "Oil leakage"

3. Failure Effect:
   The resulting impact or consequence of a failure mode on the system's performance or operation.
   Example: "Reduced engine power" or "Loss of hydraulic pressure"

4. Failure Cause:
   The underlying reason or mechanism that leads to the occurrence of a failure mode.
   Example: "Wear and tear" or "Contaminated fuel"

## Requirements

### 1. FMECA Analysis
Provide a detailed FMECA with the following components:

a) Failure Mode:
   - Clearly identify the failure mode as "Main Engine failure to start in astern."
   - Describe this failure mode using a specific component-verb combination.

b) Failure Symptom:
   - List observable indicators of the failure mode.

c) Failure Effects:
   - Describe the system-wide effects of the failure. These may include, but are not limited to:
     - Delayed maneuvers
     - Risk of collision
     - Propulsion failure
   - Include any specific effects mentioned in the provided data

d) Failure Causes:
   - Analyze all potential causes mentioned in the data. These may include, but are not limited to:
     - Low starting air pressure
     - Valve leakage
     - Control system malfunction

e) Recommended Actions:
   - Provide corrective and preventive actions to reduce the risk of recurrence
   - Base these recommendations on the specific actions and lessons learned mentioned in the data

### 2. Incident Case Study
Create a detailed case study of the incident, including:

a) Incident Overview:
   - Provide a chronological summary of events
   - Include key actions and observations during the failure
   - Use specific dates, times, and locations mentioned in the data

b) Root Cause Analysis:
   - Detail the investigation process that uncovered the failure's root cause
   - Focus on the specific findings mentioned in the data

c) Corrective Actions:
   - Outline the actions taken to rectify the issue and return the vessel to normal operations
   - Include any temporary fixes and planned future repairs mentioned in the data

d) Lessons Learned:
   - Summarize recommendations for future preventive measures
   - Focus on improving system reliability and crew preparedness based on the specific insights provided in the data

## Instructions
1. Carefully analyze the provided email thread or incident report.
2. Extract all relevant data possible from the given information.
3. Ensure your analysis is as detailed as possible, but base it solely on the facts presented in the data.
4. Do not introduce any external information or make assumptions beyond what is explicitly stated in the provided data.
5. Present your findings in a clear, well-structured format, using the following headings:
   - Failure Mode
   - Failure Symptom
   - Failure Effect
   - Failure Cause
   - Recommended Actions
   - Incident Overview
   - Root Cause Analysis
   - Corrective Actions
   - Lessons Learned
6. If any required information is missing from the provided data, indicate this in your analysis rather than making assumptions.

Remember, as an experienced reliability engineer, your goal is to present a comprehensive understanding of the incident, its causes, effects, and the actions taken to address it, based strictly on the available information. Use your expertise to accurately categorize and analyze the failure according to the FMEA definitions provided.
    Email thread to analyze:
    {thread}
    """

    if model == "OpenAI":
        response = openai.ChatCompletion.create(
            model="gpt-4",
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
