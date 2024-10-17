import nltk
from nltk.tokenize import word_tokenize

# 下载必要的NLTK数据
nltk.download('punkt', quiet=True)

def count_tokens(text):
    # 使用NLTK的word_tokenize函数来分词
    tokens = word_tokenize(text)
    return len(tokens)

# 测试
text = "这是一段用来测试的文字。This is a test sentence."
token_count = count_tokens(text)
print(f"文本的token数量: {token_count}")