import os
import re
from whisperx.utils import (get_writer)

def srt_reader(srt_file, debug=False) -> []:
    with open(srt_file, "r") as text_file:
        subtitle = text_file.read()
        # print(subtitle)
    temp = subtitle.split("\n\n")

    sentences = []
    for i in temp:
        sentences.append(i.split("\n"))
    if debug == True:
        print(sentences)

    # 去除末尾的多余空行，否则会报错
    while (sentences[1] == [""] or sentences[1] == ""):
        sentences = sentences[2:]
    while (sentences[-1] == [""] or sentences[-1] == ""):
        sentences.pop()
    return sentences


def srt_writer(translation_result: list, output_path: str):
    for i in range(len(translation_result)):
        translation_result[i] = "\n".join(translation_result[i])
    srt_content = "\n\n".join(translation_result)

    try:
        with open(output_path, "w") as srt_file:
            srt_file.write(srt_content)
    except IOError as e:
        print(e)


def convert_vector_to_Sub(transcribe_result: dict, audio_path, output_format: str, output_dir: str,
                          align_language: str):
    writer = get_writer(output_format, output_dir)

    writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}

    transcribe_result["language"] = align_language
    writer(transcribe_result, audio_path, writer_args)


SRT_STANDARD_NAME = {
    "cn": ".zh-CN"
}
