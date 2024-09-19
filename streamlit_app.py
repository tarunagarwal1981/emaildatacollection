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
    Prompt for FMECA and Incident Case Study Generation
Context: You are tasked with generating a detailed FMECA (Failure Modes, Effects, and Criticality Analysis) for a marine vessel experiencing a specific failure mode, accompanied by an incident case study detailing the failure, its root causes, and corrective actions taken. The case involves the vessel's Main Engine (ME) failure to start in astern, observed during critical maneuvers. The FMECA should focus on identifying the root causes, effects, and criticality of the failure, along with recommendations for preventive actions. The case study should present a chronological overview of the incident and the actions taken to resolve it.

Requirements:
FMECA Analysis:

Failure Mode: Clearly identify the failure mode, in this case, "Main Engine failure to start in astern."
Root Causes: Analyze potential causes, including low starting air pressure, valve leakage, and control system malfunction.
Failure Effects: Describe the system-wide effects of the failure, such as delayed maneuvers, risk of collision, or propulsion failure.
Criticality (Risk Priority Number - RPN): Assign severity, occurrence, and detection ratings to evaluate the criticality of each failure cause.
Recommended Actions: Provide corrective and preventive actions to reduce the risk of recurrence (e.g., regular maintenance, system upgrades, or crew training).
Incident Case Study:

Incident Overview: Summarize the timeline of events, including key actions and observations during the failure.
Root Cause Analysis: Detail the investigation process that uncovered the failure’s root cause (e.g., Valve 25 leakage, low starting air pressure).
Corrective Actions: Outline the actions taken to rectify the issue and return the vessel to normal operations.
Lessons Learned: Include recommendations for future preventive measures, focusing on improving system reliability and crew preparedness.
Sample Output:
Incident Overview: On September 17th, 2024, the UACC Marah experienced a Main Engine failure to start in astern while preparing for berthing at Port Yeosu, South Korea. This failure reoccurred on September 18th, 2024 during anchoring operations post-cargo discharge, requiring emergency anchoring procedures. The vessel was unable to respond to astern orders, leading to delayed operations and pilot cancellation.

Root Cause Analysis:

Low starting air pressure (below 15 bar) was identified as the primary cause preventing proper engine operation in astern mode.
Valve 25, responsible for controlling air supply to puncture valves, exhibited leakage, further contributing to the engine’s inability to start astern.
Corrective Actions:

Valve 25 was inspected and temporarily blanked to allow proper air supply to the puncture valves.
All starting air valves were checked for leaks, and air connections were tested.
The vessel’s main engine was restored to normal functionality, with plans to replace 4 new plunger and barrel assemblies at the next available opportunity.
Lessons Learned:

Regular maintenance of critical pneumatic components like Valve 25 should be prioritized to avoid future propulsion failures.
Ensuring starting air pressure is maintained at optimal levels is critical for reliable engine operation in both ahead and astern modes.

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
