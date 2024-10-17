import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Union

# Constants
MAX_TOKENS_PER_CHUNK = 1000

# Load environment variables from .env file
load_dotenv()

# Proxy settings from .env file
http_proxy = os.getenv('HTTP_PROXY')
https_proxy = os.getenv('HTTPS_PROXY')
all_proxy = os.getenv('ALL_PROXY')

if http_proxy:
    os.environ['http_proxy'] = http_proxy
if https_proxy:
    os.environ['https_proxy'] = https_proxy
if all_proxy:
    os.environ['all_proxy'] = all_proxy

# Initialize Groq client
client = Groq()

# Function definitions
def transcribe_audio(filename):
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            prompt="转录成文本",
            response_format="verbose_json",
            temperature=0,
            language="zh",
        )
    return transcription.text, transcription.segments

def format_time(seconds):
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{milliseconds:03}"

def generate_subtitles(data):
    subtitles = []
    for item in data:
        start_time = format_time(item['start'])
        end_time = format_time(item['end'])
        subtitle = f"{item['id']}\n{start_time} --> {end_time}\n{item['text']}\n"
        subtitles.append(subtitle)
    return "\n".join(subtitles)

def get_completion(prompt: str, system_message: str = "You are a helpful assistant.", model: str = "gemini-1.5-pro-latest", temperature: float = 0.3, json_mode: bool = False) -> Union[str, dict]:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'], transport='rest')
    model = genai.GenerativeModel(model_name=model)
    
    chat = model.start_chat(history=[
        {"role": "user", "parts": [system_message]},
        {"role": "model", "parts": ["Understood. I'll act as a helpful assistant with the context you provided."]}
    ])
    
    response = chat.send_message(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature))
    
    if json_mode:
        json_prompt = f"{prompt}\nPlease format your response as a valid JSON object."
        json_response = chat.send_message(json_prompt, generation_config=genai.types.GenerationConfig(temperature=temperature))
        return json_response.text
    else:
        return response.text

def one_chunk_summary(source_text: str) -> str:
    system_message = "You are a professional writer, especially good at writing Chinese"
    summary_prompt = f"""我有一段从视频转录而来的文字稿,需要你帮助我进行整理和结构化。请按照以下步骤和要求完成这项任务:
    1.阅读全文:
      - 仔细阅读整个文字稿,理解其主要内容和结构。
    2.识别主题和关键点:
      - 找出文字稿中的主要主题。
      - 列出每个主题下的关键点。
    3.创建结构化大纲:
      -基于识别出的主题和关键点,创建一个清晰的层级大纲。
      -使用标题和子标题来组织内容。
    4.段落重组:
      -将相关内容分组到适当的段落中。
      -确保每个段落都有一个明确的主题句。
    5.添加过渡语:
      -在主要部分之间添加过渡句,使文稿更连贯。
    6.格式化:
      -使用适当的标点符号和段落间距。
      -如果有特定的格式要求(如MLA, APA等),请遵循这些要求。
    7.清理和润色:
      -删除任何重复、无关或冗余的内容。
      -修正语法和拼写错误。
      -确保用词准确、表达清晰。
    8.总结:
      -为整理后的文稿添加一个简短的总结或摘要。
    9.检查:
      -通读整理后的文稿,确保内容完整、逻辑清晰。
    {source_text}
    """
    return get_completion(summary_prompt, system_message=system_message)

def num_tokens_in_string(input_str: str) -> int:
    return len(input_str.split())

def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    if token_count <= token_limit:
        return token_count
    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks
    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks
    return chunk_size

def multichunk_summary(source_text_chunks: List[str]) -> List[str]:
    system_message = "You are a professional writer, especially good at writing Chinese."
    summary_prompt = """Your task is provide a structured and precise summary of PART of a text.

    The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Summarize only the part within the source text
    delimited by <SUMMARIZE_THIS> and </SUMMARIZE_THIS>. You can use the rest of the source text as context, but do not summarize any
    of the other text. Do not output anything other than the summary of the indicated part of the text.

    <SOURCE_TEXT>
    {tagged_text}
    </SOURCE_TEXT>

    To reiterate, you should summarize only this part of the text, shown here again between <SUMMARIZE_THIS> and </SUMMARIZE_THIS>:
    <SUMMARIZE_THIS>
    {chunk_to_summarize}
    </SUMMARIZE_THIS>

    Output only the summary of the portion you are asked to summarize, and nothing else.
    """

    summary_chunks = []
    for i in range(len(source_text_chunks)):
        tagged_text = "".join(source_text_chunks[0:i]) + "<SUMMARIZE_THIS>" + source_text_chunks[i] + "</SUMMARIZE_THIS>" + "".join(source_text_chunks[i + 1 :])
        prompt = summary_prompt.format(tagged_text=tagged_text, chunk_to_summarize=source_text_chunks[i])
        summary = get_completion(prompt, system_message=system_message)
        summary_chunks.append(summary)
    return summary_chunks

def summarize(source_text, max_tokens=MAX_TOKENS_PER_CHUNK):
    num_tokens_in_text = num_tokens_in_string(source_text)

    if num_tokens_in_text < max_tokens:
        return one_chunk_summary(source_text)
    else:
        token_size = calculate_chunk_size(token_count=num_tokens_in_text, token_limit=max_tokens)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=token_size, chunk_overlap=0)
        source_text_chunks = text_splitter.split_text(source_text)
        summary_chunks = multichunk_summary(source_text_chunks)
        return "".join(summary_chunks)

# Streamlit app
def main():
    st.title("Audio Transcription and Summarization App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "mp4"])

    if uploaded_file is not None:
        st.audio(uploaded_file)

        # Process button
        if st.button("Process Audio"):
            with st.spinner("Processing..."):
                # Save uploaded file temporarily
                temp_filename = "temp_audio_file"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Transcribe audio
                text, data = transcribe_audio(temp_filename)

                # Remove temporary file
                os.remove(temp_filename)

                # Display options
                st.subheader("Output Options")
                show_transcript = st.checkbox("Show Transcript")
                show_subtitles = st.checkbox("Show Subtitles")
                show_summary = st.checkbox("Show Summary")

                if show_transcript:
                    st.subheader("Transcript")
                    st.text_area("Full Transcript", text, height=200)
                    st.download_button("Download Transcript", text, "transcript.txt")

                if show_subtitles:
                    st.subheader("Subtitles")
                    subtitles = generate_subtitles(data)
                    st.text_area("Subtitles", subtitles, height=200)
                    st.download_button("Download Subtitles", subtitles, "subtitles.txt")

                if show_summary:
                    st.subheader("Summary")
                    summary = summarize(text)
                    st.markdown(summary)
                    st.download_button("Download Summary", summary, "summary.md")

if __name__ == "__main__":
    main()