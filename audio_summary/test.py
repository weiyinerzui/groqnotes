import tiktoken

# 初始化编码器，选择适合的模型
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# 示例文本
text = "你好，今天是星期几？"

# 将文本编码为tokens
tokens = enc.encode(text)

# 打印tokens数量
print(f"文本中的token数量为: {len(tokens)}")
