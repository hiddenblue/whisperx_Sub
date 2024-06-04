from voice2sub import sub_transcribe, sub_align
from srt_util import srt_reader, srt_writer, SRT_STANDARD_NAME, split_long, convert_vector_to_Sub
from LLM_api import Ollama
from pathlib import Path
from config import audio_file, task, output_dir, temp_dir, output_format
from config import WORDS_NUM_LIMITS
from config import base_url, translation_model_name, translation_prompt
import os

# task default for transcribe and translation
if not task:
    task = "all"

# 需要你配置的一些变量

# faster-whisper模型文件文件夹的路径
transcribe_model_dir = "/home/chase/Documents/chatglm/whisperX"
if not transcribe_model_dir:
    transcribe_model_dir = "./"

# output format 字幕文件输出可以是以下任意一种，默认是all
# "all", "srt", "vtt", "txt", "tsv", "json", "aud"
if not output_format:
    output_format = "all"

# output_dir 存放输出的字幕文件的路径, 默认是"./output"
if not output_dir:
    output_dir = "./output"

# temp_dir 存放中间文件的路径
if not temp_dir:
    temp_dir = "./temp"

# according to the whisperx, the align language should be same as transcribe language
transcribe_language = "en"
if not transcribe_language:
    align_language = "en"

translation_target_lang = "cn"
if not translation_target_lang:
    translation_target_lang = "cn"

Is_split_en_long_sentence = True
if not Is_split_en_long_sentence:
    split_en_long_sentence = False

if "audio_file" not in locals():
    print("Please set the audio_file variable in the config.py file")
else:
    if not audio_file:
        print("audio_file is not set")
        print("please refer to the config.py")

print(f"Your initial config:\n\ttask={task}, audio_file={audio_file}, transcribe_model_dir={transcribe_model_dir}, output_dir={output_dir}")
print(f"\talign_language={transcribe_language}, translation_target_lang={translation_target_lang}")
print("start processing")

debug = False

# turnoff running model
# please set the model_name to the LLM model that you used before
Ollama.close_model(model_name=translation_model_name, baseurl=base_url)
def whisperx_sub(output_format=output_format,
                 output_dir=output_dir,
                 task=task
                 ):
    
    # check the output directory existence
    os.makedirs(output_dir, exist_ok=True)

    # check the temp directory, which is used to store the intermediate product for debug

    os.makedirs(temp_dir, exist_ok=True)

    # transcribe the audio file to vector
    transcribe_res = sub_transcribe(audio_file, model_dir=transcribe_model_dir, language="en")

    # align the subtitle to audio

    align_result = sub_align(transcribe_res, audio_file, device="cuda")

    # store the vector for many format: srt json et al.
    convert_vector_to_Sub(align_result,
                          audio_path=audio_file,
                          output_format=output_format,
                          output_dir=output_dir,
                          align_language=transcribe_language)
    
    # here you can decide whether to split a long sentence into several short sentences
    # we use a deep copy empty [] to deal with new short sentences

    if Is_split_en_long_sentence:
        # WORDS_NUM_LIMITS is defined in config.py
        # if the number of words in a sentence is larger than WORDS_NUM_LIMITS, we will split it into several short sentences
        segments_copy  = []
        for index in range(len(align_result['segments'])):
                print(index)
                if len(align_result['segments'][index]["words"]) < WORDS_NUM_LIMITS:
                    segments_copy.append(align_result['segments'][index])
                else:
                    splited_sentences = split_long(align_result['segments'][index])
                    segments_copy.extend(splited_sentences)

        align_result['segments'] = segments_copy


    os.makedirs(output_dir+"/cut", exist_ok=True)

    # store the vector for many format: srt json et al.
    convert_vector_to_Sub(align_result,
                          audio_path=audio_file,
                          output_format=output_format,
                          output_dir=output_dir+"/cut",
                          align_language=transcribe_language)

    # these variables are defined in config.py

    if task == "all":

        # initial your LLM
        ollama = Ollama(model_name=translation_model_name,
                        api_key="",
                        base_url=base_url,
                        mode="chat",
                        translate_prompt="Translate this English sentence into Chinese. Keep the puncutaion if possible.")

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

        # if you want to execute the translation manually, please comment the following line

    
        srt_content = srt_reader(str(full_output_path))
        ollama.chat_translate(srt_content)

        print("\n")
        print(srt_content)

        # generate a new name with target translation language naming standard
        full_output_path = output_path / (audio_path.stem + SRT_STANDARD_NAME[f'{translation_target_lang}'] + ".srt")
        srt_writer(srt_content, str(full_output_path))
    

if __name__ == '__main__':
    whisperx_sub()
