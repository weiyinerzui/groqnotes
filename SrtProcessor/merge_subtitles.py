import os

def merge_subtitles(input_directory, output_file):
    # 获取目录下所有的 SRT 文件
    srt_files = [f for f in os.listdir(input_directory) if f.endswith('.srt')]
    
    # 用于存储合并后的内容
    merged_content = []

    for srt_file in srt_files:
        file_path = os.path.join(input_directory, srt_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取文件内容
            content = file.readlines()
            # 添加文件名作为分隔符
            merged_content.append(f"\n\n# File: {srt_file}\n")
            merged_content.extend(content)

    # 写入合并后的内容到输出文件
    with open(output_file, 'w', encoding='utf-8') as output:
        output.writelines(merged_content)

    return f"合并完成，输出文件: {output_file}"

# 删除 if __name__ == "__main__": 部分
