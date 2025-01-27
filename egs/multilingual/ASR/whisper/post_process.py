import re
import zhconv

def process_text(input_text):
    # 去掉所有标点符号
    text_without_punctuation = re.sub(r"[^\w\s']", "", input_text)
    # 转成小写
    processed_text = text_without_punctuation.lower()
    return processed_text

def process_asru_text(input_text):
    # 去掉所有标点符号
    text_without_punctuation = re.sub(r"[^\w\s']", "", input_text)
    # 转成大写
    processed_text = text_without_punctuation.upper()
    # 将繁体字转成简体字
    simplified_chinese_str = zhconv.convert(processed_text, 'zh-hans')
    return simplified_chinese_str

def portuguese_process_text(input_text):
    return input_text.replace("-", " ", 5)

def portuguese_process_text_large(input_text):
    return input_text.replace("-", "", 5)