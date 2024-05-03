import os
import re
from whisperx.utils import (LANGUAGES, TO_LANGUAGE_CODE, get_writer, optional_float,
                    optional_int, str2bool)

def split_srb(target_file, debug=False)->list:
    with open(target_file, "r") as text_file:
        subtitle = text_file.read()
        # print(subtitle)
    temp = subtitle.split("\n\n")

    sentences = []
    for i in temp:
        sentences.append(i.split("\n"))
    if debug == True:
        print(sentences)

    # 去除末尾的多余空行，否则会报错
    while(sentences[1] == [""] or sentences[1] ==""):
        sentences = sentences[2:]
    while(sentences[-1] == [""] or sentences[-1] == ""):
        sentences.pop()
    return sentences

"""
[['1', '00:00:00,189 --> 00:00:01,970', '大家好，我叫凯尔西。'], ['2', '00:00:02,510 --> 00:00:03,530', '这是我的同事尼莎拉。']]
"""
def dump_sub(translation_result:list):
    for i in range(len(translation_result)):
        translation_result[i] = "\n".join(translation_result[i])
    return "\n\n".join(translation_result)


def convert_vector_to_Sub(transcribe_result:dict, audio_path, output_format:str, output_dir:str, align_language:str):
    writer = get_writer(output_format, output_dir)

    writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}

    transcribe_result["language"] = align_language
    writer(transcribe_result, audio_path, writer_args)


