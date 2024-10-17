import streamlit as st
import os
import google.generativeai as genai
import srt
import io
import zipfile

# 设置页面标题
st.set_page_config(page_title="SRT 字幕翻译器")

# 设置 API 密钥
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def translate_subtitle(subtitle, model, generation_config):
    prompt = f"请将以下字幕翻译成中文，保持原始的时间戳和序号不变：\n\n{subtitle}"
    response = model.generate_content(prompt, generation_config=generation_config)
    return response.text

def process_srt_file(file, model, generation_config):
    content = file.getvalue().decode("utf-8")
    subtitles = list(srt.parse(content))
    
    translated_subtitles = []
    for subtitle in subtitles:
        translated_content = translate_subtitle(str(subtitle), model, generation_config)
        translated_subtitle = srt.Subtitle(
            index=subtitle.index,
            start=subtitle.start,
            end=subtitle.end,
            content=translated_content
        )
        translated_subtitles.append(translated_subtitle)
    
    return srt.compose(translated_subtitles)

st.title("SRT 字幕翻译器")

# 模型选择
model_name = st.selectbox("选择模型", ["gemini-1.5-pro-002", "gemini-1.5-flash"])

# 模型参数设置
st.subheader("模型参数设置")
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
top_p = st.slider("Top P", 0.0, 1.0, 0.95)
top_k = st.slider("Top K", 1, 100, 64)
max_output_tokens = st.slider("Max Output Tokens", 1000, 8192, 8192)

# 创建模型和配置
generation_config = {
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "max_output_tokens": max_output_tokens,
}

model = genai.GenerativeModel(model_name=model_name)

# 文件上传
uploaded_files = st.file_uploader("上传 SRT 文件", type="srt", accept_multiple_files=True)

if uploaded_files:
    if st.button("开始翻译"):
        translated_files = []
        
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            translated_content = process_srt_file(file, model, generation_config)
            translated_files.append((file.name, translated_content))
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # 创建 ZIP 文件
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for filename, content in translated_files:
                zip_file.writestr(f"translated_{filename}", content)
        
        # 提供下载链接
        st.download_button(
            label="下载翻译后的文件",
            data=zip_buffer.getvalue(),
            file_name="translated_subtitles.zip",
            mime="application/zip"
        )

st.markdown("---")
st.markdown("Made with ❤️ by Your Name")
