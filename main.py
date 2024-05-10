from voice2sub import transcribe
from srt_util import convert_vector_to_Sub, srt_reader, srt_writer, SRT_STANDARD_NAME, split_long
from LLM_api import Ollama
from pathlib import Path
import os

# task default for transcribe and translation
task = None
if not task:
    task = "all"

# 需要你配置的一些变量
# 音频或者视频文件路径
audio_file = "/home/chase/Documents/chatglm/whisperx_Sub/openai_function_calling_unlimited_potential.mp3"

# faster-whisper模型文件文件夹的路径
model_dir = "/home/chase/Documents/chatglm/whisperX"
if not model_dir:
    model_dir = "./"

# output format 字幕文件输出可以是以下任意一种，默认是all
# "all", "srt", "vtt", "txt", "tsv", "json", "aud"
output_format = "all"

if not output_format:
    output_format = "all"

# output_dir 存放输出的字幕文件的路径
output_dir = "./output"
if not output_dir:
    output_dir = "."

# according to the whisperx, the align language should be same as transcribe language
transcribe_language = "en"
if not transcribe_language:
    align_language = "en"

translation_target_lang = "cn"
if not translation_target_lang:
    translation_target_lang = "cn"

split_en_long_sentence = True
if not split_en_long_sentence:
    split_en_long_sentence = False

print(f"Your initial config:\n\ttask={task}, audio_file={audio_file}, model_dir={model_dir}, output_dir={output_dir}")
print(f"\talign_language={transcribe_language}, translation_target_lang={translation_target_lang}")
print("start processing")



def whisperx_sub(output_format=output_format,
                 output_dir=output_dir,
                 task=task
                 ):
    # check the output directory existence
    os.makedirs(output_dir, exist_ok=True)

    # transcibe the audio file to vector
    transcribe_res = transcribe(audio_file, model_dir=model_dir, language="en")

    # store the vector for many format: srt json et al.
    convert_vector_to_Sub(transcribe_res,
                          audio_path=audio_file,
                          output_format=output_format,
                          output_dir=output_dir,
                          align_language=transcribe_language)


    # here you can decide whether to split a long sentence into several short sentences
    # we use a deep copy empty [] to deal with new short sentences

    if split_en_long_sentence:
        segments_copy  = []
        for index in range(len(transcribe_res['segments'])):
                if len(transcribe_res['segments'][index]["words"]) < 23:
                    segments_copy.append(transcribe_res['segments'][index])
                else:
                    splited_sentences = split_long(transcribe_res['segments'][index])
                    segments_copy.extend(splited_sentences)

        transcribe_res['segments'] = segments_copy


    os.makedirs(output_dir+"/cut", exist_ok=True)

    # store the vector for many format: srt json et al.
    convert_vector_to_Sub(transcribe_res,
                          audio_path=audio_file,
                          output_format=output_format,
                          output_dir=output_dir+"/cut",
                          align_language=transcribe_language)

    # your large model base url

    base_url = "http://localhost:11434/api/chat"
    translation_model_name = "qwen:32b"
    translation_prompt = ""

    # initial your LLM
    ollama = Ollama(model_name=translation_model_name,
                    api_key="",
                    base_url=base_url,
                    mode="chat",
                    translate_prompt="把这段英文字幕翻译成中文")

    # create subtitle output path and translation output path
    print("\n")
    # read subtitle file: xxx.srt
    output_path = Path(output_dir)
    print(f'output_path: {output_path}')

    audio_path = Path(audio_file)
    print(f'audio_path: {audio_path}')

    if not output_path.is_dir():
        output_path = input("current output path is empty.  please enter the output path: ")
    full_output_path = output_path / ( audio_path.with_suffix(".srt").name )
    print(f'full_output_path: {full_output_path}')
    print("\n")

    """
    srt_content = srt_reader(str(full_output_path))
    ollama.chat_translate(srt_content)

    print("\n")
    print(srt_content)

    # generate a new name with target translation language naming standard
    full_output_path = output_path / (audio_path.stem + SRT_STANDARD_NAME[f'{translation_target_lang}'] + ".srt")
    srt_writer(srt_content, str(full_output_path))

    """

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    whisperx_sub()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
