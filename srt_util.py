import os
import re
from whisperx.utils import (get_writer)
from typing import Union
import math

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

### split long sentence mode:

import nltk

def cal_preference(index_list:list, tokenized_sentences:str)->list:
    """
    this function is used to help calculate the preference of each choice
    different cut point has different type_value and distance_value
    :param index_list:
    :param tokenized_sentences:
    :return:
    """

    ## but 10 and 8 or 6
    # ,if 5 ,because 5  ,whether 4
    # distance to the center x , len -x

    all_choice_list = index_list

    # calculator preference
    sentences_length = len(tokenized_sentences)

    # we give a parameter to reduce the influence of raw distance_value to preferrence_value
    distance_control_parameter = 1

    #  [('and', 'CC'), ['have, and this is', 28]]
    for choice in all_choice_list:
        type_value = 0
        distance_value = 0
        # preference = type_value * disable
        preference_value = 0

        # Coorinary Conjunction
        # the token before that coord-conj must be ","
        if choice[0] == ('but', 'CC'): type_value = 10
        if choice[0] == ('and', 'CC'): type_value = 7
        if choice[0] == ('or', 'CC'): type_value = 6

        # Subordinary Conjunction
        # the token before that coord-conj must be ","
        if choice[0] == ('if', 'IN'): type_value = 5
        if choice[0] == ('because', 'IN'): type_value = 5
        if choice[0] == ('so', 'IN'): type_value = 5
        if choice[0] == ('whether', 'IN'): type_value = 4

        distance_mapp_10 = (choice[1][1]+1) / sentences_length * 10
        distance_value = math.fabs(distance_mapp_10 - 5)
        preference_value = type_value * distance_value * distance_control_parameter
        choice[1].append(round(preference_value, 3))

    print(all_choice_list, "\n\n")
    return all_choice_list

def regex_find(reg_expresssion:str, tagged_words:list)->list:

    chunkParser = nltk.RegexpParser(reg_expresssion)
    chunked_words = chunkParser.parse(tagged_words)

    print("Chunked_words: ")
    print("\t", [i for i in chunked_words], "\n\n")

    prefix_length_cut_point = 5
    suffix_length_cut_point = 5
    index_list = []
    for subtree in chunked_words.subtrees():
        # if the subtree is the whole tree, bypass it
        if subtree.label() == "S":
            continue

        if subtree.label() == "Conjunction":
            print(subtree)
            for i in subtree:
                if i in [("and", "CC") ,("or", "CC"), ("but", "CC")]:
                    if tagged_words.index(i) >= prefix_length_cut_point or tagged_words.index(i) <= len(chunked_words) - suffix_length_cut_point:
                        index_list.append([i, subtree])
                    continue

        # for subtree in chunked_words
        if subtree.label() == "Suborinary_conj":
            for subconj in subtree:
                if subconj  in [("if", "IN"), ("because", "IN"), ("whether", "IN"), ("so", "IN")]:
                    if tagged_words.index(subconj) >= prefix_length_cut_point or tagged_words.index(subconj) <= len(tagged_words) - suffix_length_cut_point:
                        index_list.append([subconj,subtree])
                    continue
    print(index_list, "\n\n")
    return index_list

def find_cut_pos(tokenized_sentences:str)->list:
    """
    This function could help you find the split pointer of a long sentence
    which you expected to be cut into two part
    :param tokenized_sentences: the sentence that has be tokenized with "? . ! " symobol

    """
    # the input sentence has alreay be tokenized, so we don't segment it.
    print("The long sentences: ")
    print("\t", tokenized_sentences, end="\n\n")

    # text_part = long_sentence["text"]

    # Tokenized the sentences
    tokenized_words = nltk.word_tokenize(tokenized_sentences)
    print("Tokenized_words: ")
    print("\t", tokenized_words, end="\n\n")

    # parts of speech tagging
    tagged_words = nltk.pos_tag(tokenized_words)
    print("Tagged_words: ")
    print("\t", tagged_words, end="\n\n")

    # this regular expression could be better. I write two rule to extract  Conjunction and subor_conj, repectively
    split_regex = r"""Conjunction: {<RB.*?|JJ.*?|VB.*?|IN?|MD?|NN.?>?<,><CC><DT>?<RB.?|VBZ?|JJ.|IN>*<PRP>?}
                      Suborinary_conj:  {<IN>?<JJ.>*?<IN>?<NN|NNS|NNP|NNPS|RB|RP|DT>*?<,><IN><DT|IN>?<RB.?|VBZ?|JJ.>*<PRP>?<VB.|MD>}
                   """

    # calling the regxr funcction
    index_list = regex_find(split_regex, tagged_words)

    """
    so you get a conj and subordinary conj positon dict like this:
    next we need to find the cut pointer and each conj and subordinary conj
    [[('and', 'CC'), Tree('Conjunction', [('have', 'VBP'), (',', ','), ('and', 'CC'), ('this', 'DT'), ('is', 'VBZ')])],
    [('and', 'CC'), Tree('Conjunction', [('used', 'VBD'), (',', ','), ('and', 'CC'), ('this', 'DT'), ('is', 'VBZ'), ('like', 'IN')])],
    [('because', 'IN'), Tree('Suborinary_conj', [('bummer', 'NN'), (',', ','), ('because', 'IN'), ('like', 'IN'), ('I', 'PRP'), ('said', 'VBD')])]] 
    """

    if  index_list:
        for single_ret in index_list:
            target = ""
            for index in range(len(single_ret[1])):
                # because in transcribe_res wors, the ',' and "'s" is part of a word, not a individual unit
                if (word := single_ret[1][index][0]) == ',' or word == "'s":
                    target += word
                else:
                    target += " " + word
            single_ret[1] = [target.strip(), tokenized_sentences.find(target)+target.find(single_ret[0][0])]

    return cal_preference(index_list, tokenized_sentences)


def rearrance_long_sentence(long_sentence:dict, choice:list)->Union[tuple, None]:
    """
     but > and > or  if > because > whether    prefer the cut point that near the center of the long sentences.
    :param long_sentence:
    :return:
    """

    # [('and', 'CC'), ['have, and this is', 28, 5.5019762845849804]]

    match_list = choice[1][0].split(" ")
    word = []
    start_time = long_sentence["start"]
    end_time = 0
    words = long_sentence["words"]

    front_sentence_part = long_sentence["text"][0:choice[1][1]]
    behind_sentence_part = long_sentence["text"][choice[1][1]:]

    for time_word_index in range(len(words)):
        for match_word_index in range(len(match_list)):
            if (word_out:= words[time_word_index]["word"]) == (word_inner := match_list[match_word_index]) or \
                word_out == word_inner + ",":
                time_word_index  += 1
                continue
            else:
                time_word_index -= match_word_index
                break
        if match_word_index == len(match_list)-1:
            time_word_index -= match_word_index + 1
            break

    if (match_word_index != len(match_list)-1):
        return None

    # find it
    inner_position = match_list.index(choice[0][0])
    actual_position = time_word_index + inner_position

    end_time = words[actual_position-1]["end"]
    new_start_time = words[actual_position]["start"]
    new_end_time = words[len(words)-1]["end"]

    front_words_part = words[:actual_position]
    behind_words_part= words[actual_position:]
    front_sentence = {"start": start_time, "end": end_time, "text": front_sentence_part, "words": front_words_part}
    behind_sentence = {"start": new_start_time, "end": new_end_time, "text":behind_sentence_part, "words": behind_words_part}

    return front_sentence, behind_sentence

def split_long(long_sentence:dict)->list:
    """
    :param long_sentence: a sentence that would be splited.
    :return:
    """

    # 这个句子是253 个characters， 大约25秒？
    if (s_lenght := len(long_sentence["text"])) < 140:
        print("This sentence doesn't  need to be segmentated\n\n")
        return [long_sentence]

    print(f"This length of this sentence is: {s_lenght} > 140 characters, So it will be splited\n\n")


    tokenized_sentences = long_sentence["text"]

    # we preserve the context of the cut position in order to failitite manually judge whether excute it
    all_choice_list = find_cut_pos(tokenized_sentences)
    print('All_choice_list: ')
    print("\t",all_choice_list, end="\n\n")

    # sort by preference_value
    all_choice_list.sort(key=lambda x: x[1][2], reverse=True)

    print("All_choice_list.sorted: ")
    print("\t",all_choice_list, end="\n\n")

    choice = []
    if all_choice_list:
        choice = all_choice_list[0]
    else:
        ## failed to found cut point, So return itself directly
        return [long_sentence]


    front_sentence , behind_sentence = rearrance_long_sentence(long_sentence, choice)
    # 这个句子是253 个characters， 大约25秒？

    ret_sentence = []

    # after cut ,we could iteralyy execute this function to find if any sentences is too long and any potential cut position
    for i in split_long(front_sentence):
        ret_sentence.append(i)
    for j in split_long(behind_sentence):
        ret_sentence.append(j)

    return ret_sentence



# def merge_sentence(splited_sentences:list, transcribe_res:list, index):
#     """
#     This function can merge some short sentence generated by split_long() back into transcibe_res
#     """
#
#     for i in range(len(splited_sentences)-1, -1, -1):
#         transcribe_res.insert(0, splited_sentences[i])
#
#     return transcribe_res


"""
the code below is used to test that split_long() function which could split long English sentence into several short English sentence
facilitate displaying them on screen
"""

if __name__ == "__main__":
    long_sentence1 = {'start': 129.04, 'end': 144.25,
                     'text': "So in this case, imagine you have, and this is the example that OpenAI used, and this is like the only one, which is kind of a bummer, because like I said, this is so powerful that I think we're going to see some crazy stuff coming from this capability.",
                     'words': [{'word': 'So', 'start': 129.04, 'end': 129.28, 'score': 0.838}, {'word': 'in', 'start': 130.081, 'end': 130.121, 'score': 1.0}, {'word': 'this', 'start': 130.161, 'end': 130.301, 'score': 0.71}, {'word': 'case,', 'start': 130.321, 'end': 130.621, 'score': 0.804}, {'word': 'imagine', 'start': 131.522, 'end': 132.002, 'score': 0.736}, {'word': 'you', 'start': 132.182, 'end': 132.322, 'score': 0.923}, {'word': 'have,', 'start': 132.362, 'end': 132.582, 'score': 0.864}, {'word': 'and', 'start': 132.683, 'end': 132.763, 'score': 0.861}, {'word': 'this', 'start': 132.803, 'end': 132.923, 'score': 0.96}, {'word': 'is', 'start': 132.983, 'end': 133.083, 'score': 0.56}, {'word': 'the', 'start': 133.223, 'end': 133.483, 'score': 0.693}, {'word': 'example', 'start': 133.703, 'end': 134.124, 'score': 0.9}, {'word': 'that', 'start': 134.164, 'end': 134.284, 'score': 0.9}, {'word': 'OpenAI', 'start': 134.384, 'end': 134.884, 'score': 0.826}, {'word': 'used,', 'start': 135.024, 'end': 135.244, 'score': 0.801}, {'word': 'and', 'start': 135.625, 'end': 135.705, 'score': 0.86}, {'word': 'this', 'start': 135.745, 'end': 135.845, 'score': 0.894}, {'word': 'is', 'start': 135.885, 'end': 135.945, 'score': 0.803}, {'word': 'like', 'start': 135.965, 'end': 136.085, 'score': 0.778}, {'word': 'the', 'start': 136.105, 'end': 136.205, 'score': 0.708}, {'word': 'only', 'start': 136.305, 'end': 136.485, 'score': 0.698}, {'word': 'one,', 'start': 136.585, 'end': 136.705, 'score': 0.622}, {'word': 'which', 'start': 137.386, 'end': 137.526, 'score': 0.91}, {'word': 'is', 'start': 137.586, 'end': 137.646, 'score': 0.802}, {'word': 'kind', 'start': 137.686, 'end': 137.786, 'score': 0.921}, {'word': 'of', 'start': 137.806, 'end': 137.846, 'score': 0.993}, {'word': 'a', 'start': 137.866, 'end': 137.886, 'score': 0.001}, {'word': 'bummer,', 'start': 137.966, 'end': 138.206, 'score': 0.855}, {'word': 'because', 'start': 138.246, 'end': 138.406, 'score': 0.999}, {'word': 'like', 'start': 138.446, 'end': 138.566, 'score': 0.76}, {'word': 'I', 'start': 138.586, 'end': 138.667, 'score': 0.608}, {'word': 'said,', 'start': 138.687, 'end': 138.807, 'score': 0.927}, {'word': 'this', 'start': 138.847, 'end': 138.967, 'score': 0.974}, {'word': 'is', 'start': 139.027, 'end': 139.087, 'score': 0.9}, {'word': 'so', 'start': 139.167, 'end': 139.347, 'score': 0.866}, {'word': 'powerful', 'start': 139.387, 'end': 139.807, 'score': 0.844}, {'word': 'that', 'start': 140.528, 'end': 140.648, 'score': 0.902}, {'word': 'I', 'start': 140.668, 'end': 140.708, 'score': 0.519}, {'word': 'think', 'start': 140.748, 'end': 140.948, 'score': 0.838}, {'word': "we're", 'start': 141.348, 'end': 141.468, 'score': 0.768}, {'word': 'going', 'start': 141.488, 'end': 141.588, 'score': 0.442}, {'word': 'to', 'start': 141.609, 'end': 141.669, 'score': 0.751}, {'word': 'see', 'start': 141.689, 'end': 141.809, 'score': 0.843}, {'word': 'some', 'start': 141.829, 'end': 141.969, 'score': 0.854}, {'word': 'crazy', 'start': 142.089, 'end': 142.429, 'score': 0.941}, {'word': 'stuff', 'start': 142.469, 'end': 142.729, 'score': 0.906}, {'word': 'coming', 'start': 142.849, 'end': 143.11, 'score': 0.862}, {'word': 'from', 'start': 143.19, 'end': 143.41, 'score': 0.825}, {'word': 'this', 'start': 143.55, 'end': 143.71, 'score': 0.78}, {'word': 'capability.', 'start': 143.73, 'end': 144.25, 'score': 0.826}]
                     }
    long_sentence2 = {'start': 87.553, 'end': 99.144,
                     'text': "So historically, if you wanted to do something like this, like for example, if you had a user and that was just chatting with GPT-4, they might ask, you know, hey, what's the weather like in Boston?",
                     'words': [{'word': 'So', 'start': 87.553, 'end': 87.693, 'score': 0.902}, {'word': 'historically,', 'start': 87.833, 'end': 88.554, 'score': 0.719}, {'word': 'if', 'start': 88.934, 'end': 88.974, 'score': 0.998}, {'word': 'you', 'start': 88.994, 'end': 89.114, 'score': 0.722}, {'word': 'wanted', 'start': 89.134, 'end': 89.355, 'score': 0.753}, {'word': 'to', 'start': 89.375, 'end': 89.435, 'score': 0.751}, {'word': 'do', 'start': 89.495, 'end': 89.635, 'score': 0.846}, {'word': 'something', 'start': 89.695, 'end': 89.995, 'score': 0.738}, {'word': 'like', 'start': 90.035, 'end': 90.195, 'score': 0.783}, {'word': 'this,', 'start': 90.235, 'end': 90.436, 'score': 0.805}, {'word': 'like', 'start': 90.476, 'end': 90.636, 'score': 0.744}, {'word': 'for', 'start': 90.676, 'end': 90.796, 'score': 0.833}, {'word': 'example,', 'start': 90.856, 'end': 91.417, 'score': 0.789}, {'word': 'if', 'start': 92.397, 'end': 92.658, 'score': 0.847}, {'word': 'you', 'start': 92.698, 'end': 92.838, 'score': 0.884}, {'word': 'had', 'start': 92.858, 'end': 92.978, 'score': 0.759}, {'word': 'a', 'start': 93.018, 'end': 93.058, 'score': 0.497}, {'word': 'user', 'start': 93.178, 'end': 93.498, 'score': 0.745}, {'word': 'and', 'start': 93.819, 'end': 93.899, 'score': 0.769}, {'word': 'that', 'start': 93.939, 'end': 94.099, 'score': 0.566}, {'word': 'was', 'start': 94.139, 'end': 94.239, 'score': 0.808}, {'word': 'just', 'start': 94.279, 'end': 94.439, 'score': 0.872}, {'word': 'chatting', 'start': 94.519, 'end': 94.94, 'score': 0.864}, {'word': 'with', 'start': 95.0, 'end': 95.26, 'score': 0.79}, {'word': 'GPT-4,', 'start': 95.46, 'end': 96.521, 'score': 0.82}, {'word': 'they', 'start': 96.561, 'end': 96.701, 'score': 0.684}, {'word': 'might', 'start': 96.721, 'end': 96.901, 'score': 0.694}, {'word': 'ask,', 'start': 97.142, 'end': 97.322, 'score': 0.886}, {'word': 'you', 'start': 97.342, 'end': 97.442, 'score': 0.392}, {'word': 'know,', 'start': 97.462, 'end': 97.542, 'score': 0.0}, {'word': 'hey,', 'start': 97.582, 'end': 97.742, 'score': 0.847}, {'word': "what's", 'start': 97.802, 'end': 98.002, 'score': 0.78}, {'word': 'the', 'start': 98.043, 'end': 98.123, 'score': 0.831}, {'word': 'weather', 'start': 98.163, 'end': 98.383, 'score': 0.93}, {'word': 'like', 'start': 98.423, 'end': 98.623, 'score': 0.814}, {'word': 'in', 'start': 98.663, 'end': 98.723, 'score': 0.996}, {'word': 'Boston?', 'start': 98.763, 'end': 99.144, 'score': 0.922}]
                      }
    long_sentence3 = {'start': 129.04, 'end': 144.25,
                      'text': "So in this case, imagine you have, and this is the example that OpenAI used, and this is like the only one, which is kind of a bummer, because like I said, this is so powerful that I think we're going to see some crazy stuff coming from this capability.",
                      'words': [{'word': 'So', 'start': 129.04, 'end': 129.28, 'score': 0.838}, {'word': 'in', 'start': 130.081, 'end': 130.121, 'score': 1.0}, {'word': 'this', 'start': 130.161, 'end': 130.301, 'score': 0.71}, {'word': 'case,', 'start': 130.321, 'end': 130.621, 'score': 0.804}, {'word': 'imagine', 'start': 131.522, 'end': 132.002, 'score': 0.736}, {'word': 'you', 'start': 132.182, 'end': 132.322, 'score': 0.923}, {'word': 'have,', 'start': 132.362, 'end': 132.582, 'score': 0.864}, {'word': 'and', 'start': 132.683, 'end': 132.763, 'score': 0.861}, {'word': 'this', 'start': 132.803, 'end': 132.923, 'score': 0.96}, {'word': 'is', 'start': 132.983, 'end': 133.083, 'score': 0.56}, {'word': 'the', 'start': 133.223, 'end': 133.483, 'score': 0.693}, {'word': 'example', 'start': 133.703, 'end': 134.124, 'score': 0.9}, {'word': 'that', 'start': 134.164, 'end': 134.284, 'score': 0.9}, {'word': 'OpenAI', 'start': 134.384, 'end': 134.884, 'score': 0.826}, {'word': 'used,', 'start': 135.024, 'end': 135.244, 'score': 0.801}, {'word': 'and', 'start': 135.625, 'end': 135.705, 'score': 0.86}, {'word': 'this', 'start': 135.745, 'end': 135.845, 'score': 0.894}, {'word': 'is', 'start': 135.885, 'end': 135.945, 'score': 0.803}, {'word': 'like', 'start': 135.965, 'end': 136.085, 'score': 0.778}, {'word': 'the', 'start': 136.105, 'end': 136.205, 'score': 0.708}, {'word': 'only', 'start': 136.305, 'end': 136.485, 'score': 0.698}, {'word': 'one,', 'start': 136.585, 'end': 136.705, 'score': 0.622}, {'word': 'which', 'start': 137.386, 'end': 137.526, 'score': 0.91}, {'word': 'is', 'start': 137.586, 'end': 137.646, 'score': 0.802}, {'word': 'kind', 'start': 137.686, 'end': 137.786, 'score': 0.921}, {'word': 'of', 'start': 137.806, 'end': 137.846, 'score': 0.993}, {'word': 'a', 'start': 137.866, 'end': 137.886, 'score': 0.001}, {'word': 'bummer,', 'start': 137.966, 'end': 138.206, 'score': 0.855}, {'word': 'because', 'start': 138.246, 'end': 138.406, 'score': 0.999}, {'word': 'like', 'start': 138.446, 'end': 138.566, 'score': 0.76}, {'word': 'I', 'start': 138.586, 'end': 138.667, 'score': 0.608}, {'word': 'said,', 'start': 138.687, 'end': 138.807, 'score': 0.927}, {'word': 'this', 'start': 138.847, 'end': 138.967, 'score': 0.974}, {'word': 'is', 'start': 139.027, 'end': 139.087, 'score': 0.9}, {'word': 'so', 'start': 139.167, 'end': 139.347, 'score': 0.866}, {'word': 'powerful', 'start': 139.387, 'end': 139.807, 'score': 0.844}, {'word': 'that', 'start': 140.528, 'end': 140.648, 'score': 0.902}, {'word': 'I', 'start': 140.668, 'end': 140.708, 'score': 0.519}, {'word': 'think', 'start': 140.748, 'end': 140.948, 'score': 0.838}, {'word': "we're", 'start': 141.348, 'end': 141.468, 'score': 0.768}, {'word': 'going', 'start': 141.488, 'end': 141.588, 'score': 0.442}, {'word': 'to', 'start': 141.609, 'end': 141.669, 'score': 0.751}, {'word': 'see', 'start': 141.689, 'end': 141.809, 'score': 0.843}, {'word': 'some', 'start': 141.829, 'end': 141.969, 'score': 0.854}, {'word': 'crazy', 'start': 142.089, 'end': 142.429, 'score': 0.941}, {'word': 'stuff', 'start': 142.469, 'end': 142.729, 'score': 0.906}, {'word': 'coming', 'start': 142.849, 'end': 143.11, 'score': 0.862}, {'word': 'from', 'start': 143.19, 'end': 143.41, 'score': 0.825}, {'word': 'this', 'start': 143.55, 'end': 143.71, 'score': 0.78}, {'word': 'capability.', 'start': 143.73, 'end': 144.25, 'score': 0.826}]
                     }
    long_sentence = 	{'start': 38.522, 'end': 52.613,
                     'text': " I don't think that their documentation is good enough, but I think it leaves a little bit to be desired, I suppose, in thinking about what are all the things that we really can do with this, because this is unbelievably powerful.",
                     'words': [{'word': 'I', 'start': 38.522, 'end': 38.582, 'score': 0.716}, {'word': "don't", 'start': 38.642, 'end': 38.842, 'score': 0.69}, {'word': 'think', 'start': 38.902, 'end': 39.142, 'score': 0.698}, {'word': 'that', 'start': 39.222, 'end': 39.382, 'score': 0.826}, {'word': 'their', 'start': 39.442, 'end': 39.763, 'score': 0.771}, {'word': 'documentation', 'start': 40.083, 'end': 40.743, 'score': 0.858}, {'word': 'is', 'start': 40.803, 'end': 40.904, 'score': 0.568}, {'word': 'good', 'start': 41.064, 'end': 41.204, 'score': 0.868}, {'word': 'enough,', 'start': 41.224, 'end': 41.444, 'score': 0.76}, {'word': 'but', 'start': 42.585, 'end': 42.785, 'score': 0.869}, {'word': 'I', 'start': 43.125, 'end': 43.186, 'score': 0.931}, {'word': 'think', 'start': 43.226, 'end': 43.386, 'score': 0.893}, {'word': 'it', 'start': 43.726, 'end': 43.786, 'score': 0.882}, {'word': 'leaves', 'start': 43.866, 'end': 44.066, 'score': 0.767}, {'word': 'a', 'start': 44.086, 'end': 44.126, 'score': 0.499}, {'word': 'little', 'start': 44.146, 'end': 44.326, 'score': 0.846}, {'word': 'bit', 'start': 44.346, 'end': 44.507, 'score': 0.794}, {'word': 'to', 'start': 44.547, 'end': 44.647, 'score': 0.62}, {'word': 'be', 'start': 44.707, 'end': 45.027, 'score': 0.837}, {'word': 'desired,', 'start': 46.288, 'end': 46.769, 'score': 0.866}, {'word': 'I', 'start': 46.829, 'end': 46.869, 'score': 0.937}, {'word': 'suppose,', 'start': 46.929, 'end': 47.389, 'score': 0.853}, {'word': 'in', 'start': 48.39, 'end': 48.47, 'score': 0.976}, {'word': 'thinking', 'start': 48.51, 'end': 48.77, 'score': 0.778}, {'word': 'about', 'start': 48.81, 'end': 49.09, 'score': 0.826}, {'word': 'what', 'start': 49.211, 'end': 49.331, 'score': 0.973}, {'word': 'are', 'start': 49.371, 'end': 49.451, 'score': 0.808}, {'word': 'all', 'start': 49.471, 'end': 49.611, 'score': 0.742}, {'word': 'the', 'start': 49.631, 'end': 49.711, 'score': 0.797}, {'word': 'things', 'start': 49.751, 'end': 49.951, 'score': 0.772}, {'word': 'that', 'start': 49.971, 'end': 50.051, 'score': 0.893}, {'word': 'we', 'start': 50.091, 'end': 50.191, 'score': 0.882}, {'word': 'really', 'start': 50.251, 'end': 50.472, 'score': 0.806}, {'word': 'can', 'start': 50.492, 'end': 50.612, 'score': 0.998}, {'word': 'do', 'start': 50.672, 'end': 50.772, 'score': 0.998}, {'word': 'with', 'start': 50.812, 'end': 50.932, 'score': 0.794}, {'word': 'this,', 'start': 50.952, 'end': 51.092, 'score': 0.951}, {'word': 'because', 'start': 51.112, 'end': 51.312, 'score': 0.494}, {'word': 'this', 'start': 51.332, 'end': 51.452, 'score': 0.423}, {'word': 'is', 'start': 51.472, 'end': 51.553, 'score': 0.395}, {'word': 'unbelievably', 'start': 51.713, 'end': 52.193, 'score': 0.893}, {'word': 'powerful.', 'start': 52.233, 'end': 52.613, 'score': 0.836}]
                    }

    ret_sentences = split_long(long_sentence2)

    for i in ret_sentences:
        print(i['text'], end="\n\n")



