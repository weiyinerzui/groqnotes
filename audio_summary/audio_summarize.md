```

**主要修改：**

*   `num_tokens_in_string` 函数现在使用 `gemini_model.count_tokens(input_str)` 来计算 token 数量，这是与 Gemini 1.5 Pro 模型兼容的。
*   其余代码保持不变，因为逻辑和功能与原始代码相同。

**如何使用：**

1.  **安装依赖：** 确保您已安装所有必要的库，包括 `google-generativeai`、`whisper`、`tiktoken` 、`langchain-text-splitters`、 `python-dotenv` 和 `icecream`。
    ```bash
    pip install google-generativeai openai-whisper tiktoken langchain-text-splitters python-dotenv icecream
    ```
2.  **设置环境变量：** 将 `GEMINI_API_KEY` 设置为您的 Gemini API 密钥。您可以使用 `.env` 文件或者直接在环境变量中设置。
3.  **替换音频文件路径：** 将 `audio_file_path` 变量的值替换为您要处理的音频文件的路径。
4.  **运行代码：**
    ```bash
    python your_script_name.py
    ```
**注意：**

*   请确保您已经正确安装了 `google-generativeai` 库，并且 API 密钥已正确设置。
*   您可以使用 `python-dotenv` 库将环境变量从 `.env` 文件加载。
*   确保您的 Gemini API 密钥是有效的，并且您具有足够的配额。

此代码应该可以正确处理音频文件，使用 Whisper 转录，然后使用 Gemini 1.5 Pro 模型进行文本优化和总结。 `num_tokens_in_string` 函数现在使用 Gemini 模型的内置方法，从而提供准确的 token 计数。
