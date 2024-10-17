import streamlit as st
import os
import google.generativeai as genai
import srt
import io
from split_subtitle import split_subtitles
import time
from google.api_core import exceptions, retry

os.environ['http_proxy'] = 'http://127.0.0.1:10809'
os.environ['https_proxy'] = 'http://127.0.0.1:10809'
os.environ['all_proxy'] = 'socks5://127.0.0.1:10809'

# 设置页面标题
st.set_page_config(page_title="SRT 字幕处理工具", layout="wide")

# 设置 API 密钥
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

@retry.Retry(predicate=retry.if_exception_type(exceptions.ResourceExhausted))
def translate_subtitle_with_retry(subtitle, source_lang, target_lang, model, generation_config):
    prompt = f"请将以下{source_lang}字幕翻译成{target_lang}，保持原始的时间戳和序号不变：\n\n{subtitle}"
    response = model.generate_content(prompt, generation_config=generation_config)
    
    if not response.text:
        raise ValueError("翻译结果为空")
    
    return response.text

def process_srt_file(content, source_lang, target_lang, model, generation_config):
    subtitles = list(srt.parse(content))
    
    translated_subtitles = []
    for subtitle in subtitles:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                translated_content = translate_subtitle_with_retry(str(subtitle), source_lang, target_lang, model, generation_config)
                translated_subtitle = srt.Subtitle(
                    index=subtitle.index,
                    start=subtitle.start,
                    end=subtitle.end,
                    content=translated_content
                )
                translated_subtitles.append(translated_subtitle)
                break
            except exceptions.ResourceExhausted:
                if attempt < max_retries - 1:
                    st.warning(f"翻译字幕 {subtitle.index} 时遇到配额限制。等待 60 秒后重试...")
                    time.sleep(60)
                else:
                    st.error(f"翻译字幕 {subtitle.index} 失败，已达到最大重试次数。")
                    translated_subtitles.append(subtitle)  # 保留原字幕
            except ValueError as e:
                st.warning(f"翻译字幕 {subtitle.index} 时出现问题：{str(e)}。保留原字幕。")
                translated_subtitles.append(subtitle)  # 保留原字幕
                break
    
    return srt.compose(translated_subtitles)

st.title("SRT 字幕处理工具")

# 文件上传
uploaded_file = st.file_uploader("上传 SRT 文件", type="srt")

if uploaded_file:
    # 读取文件内容
    content = uploaded_file.getvalue().decode("utf-8")
    
    # 字幕内容预览
    st.subheader("字幕内容预览")
    st.text_area("前 10 行内容", "\n".join(content.split("\n")[:10]), height=200)

    # 自定义参数
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("源语言", ["英语", "中文", "日语"])
        target_lang = st.selectbox("目标语言", ["中文", "英语"])
        model_name = st.selectbox("翻译模型", ["gemini-1.5-pro-002", "gemini-1.5-flash"])
    with col2:
        temperature = st.slider("模型温度", 0.0, 1.0, 0.7)
        num_segments = st.number_input("分割段数", min_value=1, value=3, step=1)

    # 创建模型和配置
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(model_name=model_name)

    # 开始翻译按钮
    if st.button("开始翻译"):
        try:
            with st.spinner("正在处理字幕..."):
                # 1. 分割字幕
                split_results = split_subtitles(content, num_segments)

                # 2. 翻译每个分段
                translated_segments = []
                progress_bar = st.progress(0)
                for i, segment in enumerate(split_results):
                    translated_content = process_srt_file(segment, source_lang, target_lang, model, generation_config)
                    translated_segments.append(translated_content)
                    progress_bar.progress((i + 1) / len(split_results))

                # 3. 合并翻译后的字幕
                merged_content = "\n\n".join(translated_segments)

                # 显示翻译后的内容预览
                st.subheader("翻译后内容预览")
                st.text_area("前 10 行内容", "\n".join(merged_content.split("\n")[:10]), height=200)

                # 保存翻译结果到会话状态
                st.session_state.translated_content = merged_content

            st.success("翻译完成！")
        except Exception as e:
            st.error(f"翻译过程中发生错误: {str(e)}")

    # 导出按钮
    if 'translated_content' in st.session_state:
        output_filename = f"translated_{uploaded_file.name}"
        st.download_button(
            label="导出翻译后的字幕文件",
            data=st.session_state.translated_content,
            file_name=output_filename,
            mime="text/plain"
        )

st.markdown("---")
st.markdown("Made with ❤️ by Your Name")