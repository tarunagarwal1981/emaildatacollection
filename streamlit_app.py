import streamlit as st
import docx
import openai
import anthropic
import replicate
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

def get_replicate_api_key():
    if 'replicate' in st.secrets:
        return st.secrets['replicate']['api_key']
    api_key = os.getenv('REPLICATE_API_KEY')
    if api_key is None:
        raise ValueError("Replicate API key not found. Set REPLICATE_API_KEY as an environment variable or in Streamlit secrets.")
    return api_key

# Set up clients
openai.api_key = get_openai_api_key()
anthropic_client = anthropic.Anthropic(api_key=get_anthropic_api_key())
os.environ['REPLICATE_API_TOKEN'] = get_replicate_api_key()

def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def separate_threads(content, model, user_input, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    max_tokens = 4000
    if len(content) > max_tokens * 4:
        content = content[:max_tokens * 4]
        st.warning("Content was truncated due to length. Analysis may be incomplete.")

    prompt = f"""
    The following content may contain multiple email threads related to machinery defects, incidents, or troubles. 
    Please separate these threads if required and return them as a numbered list. Each item in the list should be a complete thread.
    Use your intelligence to identify where one thread ends and another begins.

    Additional instructions from the user:
    {user_input}

    Content to separate:
    {content}
    """

    try:
        if model == "OpenAI":
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            return response.choices[0].message.content.split("\n")
        elif model == "Claude":
            response = anthropic_client.completions.create(
                prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                model="claude-2",
                max_tokens_to_sample=1500,
                temperature=temperature,
                top_p=top_p,
            )
            return response.completion.split("\n")
        elif model == "LLaMA":
            output = replicate.run(
                "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                input={"prompt": prompt,
                       "temperature": temperature,
                       "top_p": top_p,
                       "max_length": 1500,
                       "repetition_penalty": 1.0}
            )
            return ''.join(output).split('\n')
    except Exception as e:
        st.error(f"Error in API request: {str(e)}")
        return []

def analyze_thread(thread, model, user_input, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
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

    Create a sort of detailed incident case study out of each thread. Also include timeline of events if it is available in mail threads. Extract as much meaningful data as possible from the email threads and put it in the case study.

    Additional instructions from the user:
    {user_input}

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
    elif model == "LLaMA":
        output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": prompt,
                   "temperature": temperature,
                   "top_p": top_p,
                   "max_length": 1000,
                   "repetition_penalty": 1.0}
        )
        return ''.join(output)

def main():
    st.title("Multi-Thread FMEA Analyzer")

    # Add model selection
    model = st.sidebar.selectbox("Select Model", ["OpenAI", "Claude", "LLaMA"])

    # Add user input text area
    user_input = st.text_area("Additional Instructions or Context", 
                              "Please provide any additional instructions or context for the analysis here.")

    uploaded_file = st.file_uploader("Choose a DOCX file", type="docx")

    if uploaded_file is not None:
        content = read_docx(uploaded_file)
        st.write("File contents:")
        st.write(content)

        # Add sliders for LLM parameters
        st.sidebar.header("LLM Parameters")
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0, 0.1)
        
        # Note: Claude and LLaMA API don't use frequency_penalty and presence_penalty
        if model == "OpenAI":
            frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
            presence_penalty = st.sidebar.slider("Presence Penalty", 0.0, 2.0, 0.0, 0.1)
        else:
            frequency_penalty = 0.0
            presence_penalty = 0.0

        if st.button("Analyze"):
            with st.spinner("Separating threads..."):
                threads = separate_threads(content, model, user_input, temperature, top_p, frequency_penalty, presence_penalty)
                
            st.write(f"Found {len(threads)} threads.")
            
            for i, thread in enumerate(threads, 1):
                st.subheader(f"Thread {i}")
                st.write(thread)
                
                with st.spinner(f"Analyzing thread {i}..."):
                    analysis = analyze_thread(thread, model, user_input, temperature, top_p, frequency_penalty, presence_penalty)
                    st.write("FMEA Analysis:")
                    st.write(analysis)
                
                st.markdown("---")

if __name__ == "__main__":
    main()
