import os
# import sys
import google.generativeai as genai

# Proxy settings
os.environ['http_proxy'] = 'http://127.0.0.1:10809'
os.environ['https_proxy'] = 'http://127.0.0.1:10809'
os.environ['all_proxy'] = 'socks5://127.0.0.1:10809'

# 配置API密钥
gemini_api_key = os.environ['GEMINI_API_KEY']
genai.configure(api_key=gemini_api_key,transport='rest')

model = genai.GenerativeModel("gemini-1.5-flash")

prompt = "The quick brown fox jumps over the lazy dog."

# Call `count_tokens` to get the input token count (`total_tokens`).
print("total_tokens: ", model.count_tokens(prompt))
# ( total_tokens: 10 )

response = model.generate_content(prompt)

# On the response for `generate_content`, use `usage_metadata`
# to get separate input and output token counts
# (`prompt_token_count` and `candidates_token_count`, respectively),
# as well as the combined token count (`total_token_count`).
print(response.usage_metadata)
# ( prompt_token_count: 11, candidates_token_count: 73, total_token_count: 84 )