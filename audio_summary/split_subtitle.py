import tkinter as tk
from tkinter import filedialog, messagebox
import os

class SubtitleSplitter:
    def __init__(self, root):
        self.root = root
        self.root.title("字幕分割工具")
        self.root.geometry("400x300")
        
        self.file_path = ""
        self.export_path = ""
        self.num_segments = tk.IntVar(value=3)

        # 上传文件按钮
        self.upload_button = tk.Button(root, text="上传字幕文件", command=self.upload_file)
        self.upload_button.pack(pady=10)

        # 显示上传的文件路径
        self.file_label = tk.Label(root, text="未选择文件", wraplength=300)
        self.file_label.pack(pady=5)

        # 自定义分割段数
        self.segments_label = tk.Label(root, text="自定义分割段数:")
        self.segments_label.pack(pady=5)
        self.segments_entry = tk.Entry(root, textvariable=self.num_segments)
        self.segments_entry.pack(pady=5)

        # 导出位置
        self.export_label = tk.Label(root, text="导出位置:")
        self.export_label.pack(pady=5)
        self.export_entry = tk.Entry(root)
        self.export_entry.pack(pady=5)
        self.export_button = tk.Button(root, text="选择目录", command=self.select_export_directory)
        self.export_button.pack(pady=5)

        # 开始分割按钮
        self.split_button = tk.Button(root, text="开始分割", command=self.split_subtitles)
        self.split_button.pack(pady=20)

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("SRT files", "*.srt")])
        if self.file_path:
            self.file_label.config(text=self.file_path)
            self.export_entry.delete(0, tk.END)
            self.export_entry.insert(0, os.path.dirname(self.file_path))

    def select_export_directory(self):
        self.export_path = filedialog.askdirectory()
        if self.export_path:
            self.export_entry.delete(0, tk.END)
            self.export_entry.insert(0, self.export_path)

    def split_subtitles(self):
        if not self.file_path:
            messagebox.showerror("错误", "请先上传字幕文件")
            return

        try:
            num_segments = self.num_segments.get()
            if num_segments <= 0:
                raise ValueError("分割段数必须大于0")

            with open(self.file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 处理字幕分割
            segment_length = len(lines) // num_segments
            for i in range(num_segments):
                segment_lines = lines[i * segment_length:(i + 1) * segment_length]
                if i == num_segments - 1:  # 确保最后一段包含所有剩余行
                    segment_lines = lines[i * segment_length:]

                segment_file_path = os.path.join(self.export_entry.get(), f"segment_{i + 1}.srt")
                with open(segment_file_path, 'w', encoding='utf-8') as segment_file:
                    segment_file.writelines(segment_lines)

            messagebox.showinfo("成功", f"成功分割为 {num_segments} 段，文件已保存至 {self.export_entry.get()}")

        except Exception as e:
            messagebox.showerror("错误", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleSplitter(root)
    root.mainloop()
