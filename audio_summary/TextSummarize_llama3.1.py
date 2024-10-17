import os
import json
from openai import OpenAI
import os
from IPython.display import Markdown
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from typing import Union

# os.environ['http_proxy'] = 'http://127.0.0.1:10809'
# os.environ['https_proxy'] = 'http://127.0.0.1:10809'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:10809'

# 从系统变量配置 API_KEY

MAX_TOKENS_PER_CHUNK = 1000
# genai.configure(api_key=gemini_api_key,transport='rest')

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  nvidia_api_key = os.environ['NVIDIA_API_KEY']
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-405b-instruct",
  messages=[{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")


# 读取文件
    
file_path = r"D:\KGnotes\@210-Project\中级审计\未命名\01.审计师-审计相关基础知识-备考指导-吕丹_原文.md"
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 生成视频摘要



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
def one_chunk_summary(source_text: str) -> str:
    """
    Summarize the entire text as one chunk using Gemini.
    """
    system_message = f"You are an professional writer, especially good at writing chinese"

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

    prompt = summary_prompt.format(source_text=source_text)

    summary = get_completion(prompt, system_message=system_message)

    return summary


def num_tokens_in_string(input_str: str) -> int:
    """
    Estimate the number of tokens in a given string.
    
    Note: This is a rough estimate as Gemini doesn't provide a token counting method.
    We're using words as a proxy for tokens, which isn't perfect but should work for most cases.
    """
    return len(input_str.split())

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



def multichunk_summary(source_text_chunks: List[str]
) -> List[str]:
    """
    Summrize a text in multiple chunks from the source language to the target language.
    """
    system_message = f"You are an professional writer, especially good at writing chinese."

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
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<SUMMARIZE_THIS>"
            + source_text_chunks[i]
            + "</SUMMARIZE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = summary_prompt.format(            
            tagged_text=tagged_text,
            chunk_to_summarize=source_text_chunks[i],
        )

        summary = get_completion(prompt, system_message=system_message)
        summary_chunks.append(summary)

    return summary_chunks


def summarize(source_text,max_tokens=MAX_TOKENS_PER_CHUNK,):
    """Summarize the source_text."""

    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Summarizing text as single chunk")

        final_summary = one_chunk_summary(source_text)

        return final_summary

    else:
        ic("Summarizing text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        summary_chunks = multichunk_summary(source_text_chunks)

        return "".join(summary_chunks)
    




# 保存为 md 文件。
summary = summarize(text)
with open("summary.md", "w", encoding='utf-8') as file:
    file.write(summary)

print("Summary saved to summary.md")



