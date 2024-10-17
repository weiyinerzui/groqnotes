import google.generativeai as genai
import os
os.environ['http_proxy'] = 'http://127.0.0.1:10809'
os.environ['https_proxy'] = 'http://127.0.0.1:10809'
os.environ['all_proxy'] = 'socks5://127.0.0.1:10809'


# 配置API密钥
gemini_api_key = os.environ['GEMINI_API_KEY']
genai.configure(api_key=gemini_api_key,transport='rest')

def count_tokens(text):
    # 使用gemini-pro模型
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 计算tokens
    response = model.count_tokens(text)

    # 从响应对象中获取token数量
    token_count = response.total_tokens
    
    return token_count

# 测试
text = "这是一段用来测试的文字。This is a test sentence."
token_count = count_tokens(text)
print(token_count)
print(type(token_count))
print(f"Gemini模型计算的token数量: {token_count}")