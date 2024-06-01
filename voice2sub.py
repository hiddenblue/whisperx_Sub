from typing import List, Dict, Any

import whisperx
import gc
import torch

import copy
import numpy as np
import re
from SegmentType import TransSegment, SegmentMerge
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import nltk
from abc import ABC, abstractmethod
from config import temp_dir

import logging

# Assume the TransSegment class and other required functions/classes are defined

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def sub_transcribe(audio_file: str,  # auduio file
               device="cuda",
               batch_size=16,
               language=None,
               compute_type="float16",
               model_dir="./",
               output_dir="./") -> dict:
    # some VAD args
    vad_onset = 0.500
    vad_offset = 0.363
    chunk_size = 30

    # some asr options, we use whisperx default cli args
    temperature_increment_on_fallback = 0.2
    asr_options = {
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "temperatures": 0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "initial_prompt": None,
        "suppress_tokens": "-1",
        "suppress_numerals": False
    }

    temperature = asr_options["temperatures"]
    if (increment := temperature_increment_on_fallback) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 16
    if faster_whisper_threads > 0:
        torch.set_num_threads(faster_whisper_threads)

    asr_options["suppress_tokens"] = [int(x) for x in "-1".split(",")]

    # 1. Transcribe with original whisper (batched)
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root="./")

    # save model to local path (optional)
    gc.collect()
    torch.cuda.empty_cache()

    model = whisperx.load_model("large-v2",
                                device,
                                language=language,
                                compute_type=compute_type,
                                download_root=model_dir,
                                asr_options=asr_options,
                                vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
                                threads=faster_whisper_threads)

    transcribe_result = model.transcribe(audio_file, batch_size=batch_size, chunk_size=chunk_size, print_progress=True,
                              combined_progress=True)
    print(transcribe_result["segments"])  # before alignment

    print(len(transcribe_result["segments"]))

    # saved the intermediate product in temp_dir
    with open(temp_dir+"/transcribe_result.py", "w") as file:
        file.write(str(transcribe_result))

    # 保留一个transribe的结果，方便调试
    transcribe_result = copy.deepcopy(transcribe_result)

    # merge some segments in transcribe result that has continous end.
    transcribe_result["segments"] = SegmentMerge.merge_continue_segment(transcribe_result["segments"])

    with open(temp_dir+"/merge_result.py", "w") as file:
        file.write(str(transcribe_result))


    # repunctuate the result 
    rePunctuationer = RePunctuationer(FullStopModel())
    transcribe_result = rePunctuationer.re_punctuation(transcribe_result)


    # save intemiate product in temp_dir
    with open(temp_dir+"/after_re_punctuation.py", "w") as file:
        file.write(str(transcribe_result))

    # delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model

    print("**************" * 20)
    return transcribe_result


def sub_align(transcribe_result:dict,  audio_file: str, device="cuda"):
    # 2. Align whisper output
    # you can specify model_name here
    model_a, metadata = whisperx.load_align_model(language_code=transcribe_result["language"], device=device)

    print(">>Performing alignment...")
    align_result = whisperx.align(transcribe_result["segments"],
                            model_a,
                            metadata,
                            audio_file,
                            device,
                            return_char_alignments=False,
                            print_progress=True)

    print("\nlength of align_result: ",len(align_result["segments"]))
    print("\n")

    with open(temp_dir+"/align_result.py", "w") as file:
        file.write(str(align_result))

    # delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    return align_result

class PunctuationGenerator(ABC):

    @abstractmethod
    def generatePunctuation(self, text: str):
        pass


class FullStopModel(PunctuationGenerator):

    def __init__(self) -> None:
        from deepmultilingualpunctuation import PunctuationModel

        self.model = PunctuationModel()

    def generatePunctuation(self, text: str) -> str:
        return self.model.restore_punctuation(text)


class CTPuncModel(PunctuationGenerator):

    def __init__(self) -> None:
        super().__init__()
        from funasr import AutoModel
        self.model = AutoModel(model="ct-punc")

    def generatePunctuation(self, text: str):
        return self.model.generate(input=text)[0].get("text", "")


class RePunctuationer:
    """
    This class can remove all punctuation incluing ",", "." , "!", "?", ":", ";" and "..."
    the init()  accepts two kind of punctuation model:
    1. FullStopModel
    2. CTPuncModel

    the FullStopModel is the default model
    """

    def __init__(self, PunctuationModel: PunctuationGenerator):
        self.punctuation_list = ['.', ',', '!', '?', ':', ';', '...']
        train_text = state_union.raw("2005-GWBush.txt")
        self.custom_tokenizer = PunktSentenceTokenizer(train_text)
        print("PunctuationModel init. it will take 10s to load the model, please wait...")
        self.PunctuationModel = PunctuationModel

    def re_punctuation(self, transcribe_result: dict, ) -> dict:

        segments = transcribe_result["segments"]
        for i in range(len(segments)):
            segments[i]["text"] = self.remove_punctuation(segments[i]["text"])
            segments[i]["text"] = self.PunctuationModel.generatePunctuation(segments[i]["text"])
            print(segments[i]["text"])
        return transcribe_result

    def remove_punctuation(self, text: str) -> str:
        """
        This function can remove all punctuation incluing ",", "." , "!", "?", ":", ";" and "..."
        """

        tokenized_sentences = self.custom_tokenizer.tokenize(text)
        # print(tokenized_sentences)

        for i in range(len(tokenized_sentences)):
            tokenized_words = nltk.word_tokenize(tokenized_sentences[i])
            tagged_words = nltk.pos_tag(tokenized_words)
            # print("Before remove punctuation: ", tagged_words, end="\n\n")

            for j in range(len(tagged_words) - 1, 0, -1):
                if tagged_words[j][0] in self.punctuation_list:
                    tagged_words.remove(tagged_words[j])
            if tagged_words[0][0] in self.punctuation_list:
                tagged_words.remove(tagged_words[0])

            # print("After remove punctuation: ", tagged_words, end="\n\n")
            target = ""
            for word, tag in tagged_words:
                # need more speical situation
                if word in ["'s", "'d", "'ll", "'re", "'ve", "n't", "na", "ta", "'m"]:
                    target += word
                else:
                    target += " " + word
            tokenized_sentences[i] = target.strip()

        # print(tokenized_sentences)

        return " ".join(tokenized_sentences).strip()
