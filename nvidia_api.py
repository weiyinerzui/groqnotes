# import os
from openai import OpenAI

# os.environ['http_proxy'] = 'http://127.0.0.1:10809'
# os.environ['https_proxy'] = 'http://127.0.0.1:10809'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:10809'


client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
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


