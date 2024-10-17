import os
# from groq import Groq
import json
import google.generativeai as genai
import os
from IPython.display import Markdown
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from typing import Union

os.environ['http_proxy'] = 'http://127.0.0.1:10809'
os.environ['https_proxy'] = 'http://127.0.0.1:10809'
os.environ['all_proxy'] = 'socks5://127.0.0.1:10809'


# 读取文件
    
file_path = r"D:\Backup\LLM\groqnotes\source_content.md"
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 重写视频文稿

# 从系统变量配置 API_KEY
gemini_api_key = os.environ['GEMINI_API_KEY']
MAX_TOKENS_PER_CHUNK = 1000
genai.configure(api_key=gemini_api_key,transport='rest')

# 配置模型参数
def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "gemini-1.5-pro-latest",
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
    """
    Generate a completion using the Gemini API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
        model (str, optional): The name of the Gemini model to use for generating the completion.
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
        json_mode (bool, optional): Whether to return the response in JSON format.

    Returns:
        Union[str, dict]: The generated completion.
    """
    model = genai.GenerativeModel(model_name=model)
    
    chat = model.start_chat(history=[
        {
            "role": "user",
            "parts": [system_message]
        },
        {
            "role": "model",
            "parts": ["Understood. I'll act as a helpful assistant with the context you provided."]
        }
    ])
    
    response = chat.send_message(prompt, generation_config=genai.types.GenerationConfig(
        temperature=temperature
    ))
    
    if json_mode:
        # Note: Gemini doesn't have a built-in JSON mode, so we instruct it to return JSON
        json_prompt = f"{prompt}\nPlease format your response as a valid JSON object."
        json_response = chat.send_message(json_prompt, generation_config=genai.types.GenerationConfig(
            temperature=temperature
        ))
        return json_response.text  # This will be a JSON string
    else:
        return response.text



# 生成文稿摘要
def one_chunk_paraphrasing(source_text: str) -> str:
    """
    Rephrase the entire text as one chunk using Gemini.
    """
    system_message = f"You are an professional writer, especially good at writing chinese"

    paraphrasing_prompt = f"""我有一段视频转录的文稿，请帮我对其进行整理，保持原意和细节的同时，纠正语音识别中出现的错误，修正语法错误，确保用词准确。输出修改完善后的原文。
    {source_text}
"""

    prompt = paraphrasing_prompt.format(source_text=source_text)

    paraphrasing = get_completion(prompt, system_message=system_message)

    return paraphrasing


def num_tokens_in_string(input_str: str) -> int:
    """
    Estimate the number of tokens in a given string.
    
    Note: This is a rough estimate as Gemini doesn't provide a token counting method.
    We're using words as a proxy for tokens, which isn't perfect but should work for most cases.
    """
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # 计算tokens
    token_count = model.count_tokens(input_str)
    return token_count

def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    Calculate the chunk size based on the token count and token limit.
    """
    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size



def multichunk_paraphrasing(source_text_chunks: List[str]
) -> List[str]:
    """
    Rephrase a text in multiple chunks from the source language to the target language.
    """
    system_message = f"You are an professional writer, especially good at writing chinese."

    paraphrasing_prompt = """Your task is provide a structured and precise paraphrasing of PART of a text.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Rephrase only the part within the source text
delimited by <REPHRASE_THIS> and </REPFRASE_THIS>. You can use the rest of the source text as context, but do not REPHRASE any
of the other text. Do not output anything other than the paraphrasing of the indicated part of the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, you should rephrase only this part of the text, shown here again between <REPHRASE_THIS> and </REPFRASE_THIS>:
<REPHRASE_THIS>
{chunk_to_rephrase}
</REPFRASE_THIS>

Output only the paraphrasing of the portion you are asked to rephrase, and nothing else.
"""

    paraphrasing_chunks = []
    for i in range(len(source_text_chunks)):
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<REPHRASE_THIS>"
            + source_text_chunks[i]
            + "</REPFRASE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = paraphrasing_prompt.format(            
            tagged_text=tagged_text,
            chunk_to_rephrase=source_text_chunks[i],
        )

        paraphrasing = get_completion(prompt, system_message=system_message)
        paraphrasing_chunks.append(paraphrasing)

    return paraphrasing_chunks


def rephrase(source_text,max_tokens=MAX_TOKENS_PER_CHUNK,):
    """Rephrase the source_text."""

    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Rephrasing text as single chunk")

        final_paraphrasing = one_chunk_paraphrasing(source_text)

        return final_paraphrasing

    else:
        ic("Rephrasing text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        paraphrasing_chunks = multichunk_paraphrasing(source_text_chunks)

        return "".join(paraphrasing_chunks)
    




# 保存为 md 文件。
paraphrasing = rephrase(text)
with open("paraphrasing.md", "w", encoding='utf-8') as file:
    file.write(paraphrasing)

print("Paraphrasing saved to rephrase.md")



