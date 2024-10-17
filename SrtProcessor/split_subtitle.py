import srt

def split_subtitles(content, num_segments):
    subtitles = list(srt.parse(content))
    
    segment_length = len(subtitles) // num_segments
    split_results = []

    for i in range(num_segments):
        start_index = i * segment_length
        end_index = (i + 1) * segment_length if i < num_segments - 1 else len(subtitles)
        segment_subtitles = subtitles[start_index:end_index]
        
        # 重新编号字幕
        for j, subtitle in enumerate(segment_subtitles, start=1):
            subtitle.index = j
        
        segment_content = srt.compose(segment_subtitles)
        split_results.append(segment_content)

    return split_results

# 删除整个 SubtitleSplitter 类和 if __name__ == "__main__": 部分
