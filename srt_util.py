import os
import re
from whisperx.utils import (get_writer)
import nltk
from typing import Union
import math
from collections import namedtuple
from typing import NamedTuple, Union, List, Dict, Tuple

# we create a datetype for every token in sentences
TokenTuple = namedtuple("Token", "word, tag")

UNCOMPLETE_STRUCTURE = ["'s", "'m", "'re", "'d", "'ll", "'re", "n't"]

# 这部分代码看不懂也没关系，太复杂了


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



def cal_preference(index_list: list, tokenized_sentences: str) -> list:
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
        if choice["conj"] in [('But', 'CC'), ('but', 'CC')]: type_value = 10
        if choice["conj"] in [('And', 'CC'), ('and', 'CC')]: type_value = 7
        if choice["conj"] in [('Or', 'CC'), ('or', 'CC')]: type_value = 5

        # Subordinary Conjunction
        # the token before that coord-conj must be ","
        if choice["conj"] in [('although', 'IN'), ('Although', 'IN')]: type_value = 6
        if choice["conj"] in [('If', 'IN'), ('if', 'IN')]: type_value = 5
        if choice["conj"] in [('Because', 'IN'), ('because', 'IN')]: type_value = 5
        if choice["conj"] in [('So', 'IN'), ('so', 'IN')]: type_value = 5
        if choice["conj"] in [('Whether', 'IN'), ('whether', 'IN')]: type_value = 4
        # if choice["conj"] in [('Like', 'IN'),('like', 'IN')]: type_value = 3

        # which dwt
        if choice["conj"] in [("which", "WDT")]: type_value = 4
        if choice["conj"] in [(":", ":")]: type_value = 3

        distance_mapp_10 = (choice["match_words"][1] + 1) / sentences_length * 10
        distance_value = 5 - math.fabs(distance_mapp_10 - 5)
        preference_value = type_value * distance_value * distance_control_parameter
        choice["match_words"].append(round(preference_value, 3))

    print(all_choice_list, "\n\n")
    return all_choice_list


def regex_find(reg_expresssion: str, tagged_words: list) -> list:
    chunkParser = nltk.RegexpParser(reg_expresssion)
    chunked_words = chunkParser.parse(tagged_words)

    print("Chunked_words: ")
    print("\t", [elem for elem in chunked_words], "\n\n")

    prefix_length_cut_point = 5
    suffix_length_cut_point = 5
    index_list = []
    pos_dict = {}
    for subtree in chunked_words.subtrees():

        # if the subtree is the whole tree, bypass it
        if subtree.label() == "S":
            continue

        if subtree.label() == "CoordinaryConj":
            print(subtree)
            # you may get a subree like this
            """Tree('CoordinaryConj', [('good', 'JJ'), ('enough', 'RB'), (',', ','), ('but', 'CC'), ('I', 'PRP')])"""
            for cordconj in subtree:
                if cordconj in [("and", "CC"), ("or", "CC"), ("but", "CC"), ("And", "CC"), ("Or", "CC"),
                                ("But", "CC")] and subtree[subtree.index(cordconj) - 1] == (',', ','):
                    try:
                        pos = tagged_words.index(cordconj, max((pos_dict.get(cordconj, 0) + 1), prefix_length_cut_point))
                    except ValueError:
                        continue
                    # I am not sure whether to use "or" or "and". It all depends. I make a strict to the position of the subtree.
                    if prefix_length_cut_point <= pos <= len(tagged_words) - suffix_length_cut_point:
                        index_list.append({"conj": cordconj, "match_words": subtree})
                    pos_dict[cordconj] = pos
                    continue

        # for subtree in chunked_words
        if subtree.label() == "SuborinaryConj":
            for subconj in subtree:
                if subconj in [("if", "IN"), ("because", "IN"), ("whether", "IN"), ("so", "IN"),("although", 'IN'),
                               ("If", "IN"),
                               ("Because", "IN"), ("Whether", "IN"), ("So", "IN"), ("Although", "IN")] and subtree[
                    subtree.index(subconj) - 1] == (',', ','):
                    # 这里存在一个bug，如果conjunction在句子的开头，那么就会不符合条件 index只返回第一个符合的。
                    # 这里还有一个复杂的问题，如果在subtree之外还有conj，就会错误检索到subtree之外，tagged_words以内的词导致出错。
                    # 用prefix_length来替代试试, 反正这部分匹配不到
                    # 如果有多个就会出错
                    try:
                        pos = tagged_words.index(subconj, max((pos_dict.get(subconj, 0) + 1), prefix_length_cut_point))
                    except ValueError:
                        continue
                    # I am not sure whether to use "or" or "and". It all depends. I make a strict to the position of the subtree.
                    if prefix_length_cut_point <= pos <= len(tagged_words) - suffix_length_cut_point:
                        index_list.append({"conj": subconj, "match_words": subtree})
                    pos_dict[subconj] = pos
                    continue

        # matching which
        if subtree.label() == "Which_group":
            for which_wdt in subtree:
                if which_wdt in [("which", "WDT")] and subtree[subtree.index(which_wdt) - 1] == (',', ','):
                    # 这里存在一个bug，如果conjunction在句子的开头，那么就会不符合条件 index只返回第一个符合的。
                    # 如果有多个就会出错
                    try:
                        pos = tagged_words.index(which_wdt, max((pos_dict.get(which_wdt, 0) + 1), prefix_length_cut_point))
                    except ValueError:
                        continue

                    # I am not sure whether to use "or" or "and". It all depends. I make a strict to the position of the subtree.
                    if prefix_length_cut_point <= pos <= len(tagged_words) - suffix_length_cut_point:
                        index_list.append({"conj": which_wdt, "match_words": subtree})
                    pos_dict[which_wdt] = pos
                    continue



    # add logic to match xxxx And xxxx

    Speical_token = ("And", "CC")
    Semicolon_token = (":", ":")

    if not index_list:
        pos_dict = {}
        for index, tagged_word in enumerate(tagged_words):
            if tagged_word != ("And", "CC"):
                continue
            else:
                # some new case, The "And" is placed at the begin the of sentence, So should start at -1+1
                try:
                    pos = tagged_words.index(Speical_token, max((pos_dict.get(Speical_token, 0) + 1), prefix_length_cut_point))
                except ValueError:
                    continue
                # I am not sure whether to use "or" or "and". It all depends. I make a strict to the position of the subtree.
                if prefix_length_cut_point <= pos <= len(tagged_words) - suffix_length_cut_point:
                    if tagged_words[index - 1] != (",", ",") and tagged_words[index + 1] != (",", ","):
                        if pos == index:
                            index_list.append({"conj": Speical_token, "match_words": tagged_words[index - 3:index + 2]})
                pos_dict[Speical_token] = pos
        for index, tagged_word in enumerate(tagged_words):
            if tagged_word != (":", ":"):
                continue
            else:
                # some new case, The "And" is placed at the begin the of sentence, So should start at -1+1
                try:
                    pos = tagged_words.index(Semicolon_token, max((pos_dict.get(Semicolon_token, 0) + 1), prefix_length_cut_point))
                except ValueError:
                    continue
                # I am not sure whether to use "or" or "and". It all depends. I make a strict to the position of the subtree.
                if prefix_length_cut_point <= pos <= len(tagged_words) - suffix_length_cut_point:
                    # if tagged_words[index + 1][0][0].isupper():
                    index_list.append({"conj": Semicolon_token, "match_words": tagged_words[index - 3:index + 2]})
                pos_dict[Semicolon_token] = pos


    """
    [
    [('but', 'CC'), Tree('CoordinaryConj', [('good', 'JJ'), ('enough', 'RB'), (',', ','), ('but', 'CC'), ('I', 'PRP')])],
    [('because', 'IN'), Tree('SuborinaryConj', [('with', 'IN'), ('this', 'DT'), (',', ','), ('because', 'IN'), ('this', 'DT')])]]
    """

    # besides, we need to deal with some minor cases.  "'s" "n't" and so on
    for i, chunk in enumerate(index_list):
        conj = chunk.get("conj")
        match_words = chunk.get("match_words")
        print(conj)
        print(match_words)
        chunk["match_words"] = tupleTreeToTokenlist(match_words)
        # tackle with case: match_words start with 's et at.
        if (first_word := chunk["match_words"][0][0]) in UNCOMPLETE_STRUCTURE or first_word == ',' or first_word == ':':
            # chunk["match_words"].insert(0, tagged_words.)
            match_words = chunk["match_words"][1:]

        index_list[i]["match_words"] = tupleTreeToTokenlist(match_words)

    print(index_list, "\n\n")
    return index_list


def tupleToToken(tagged_word: tuple) -> Union[TokenTuple, tuple]:
    if isinstance(tagged_word, TokenTuple):
        return tagged_word
    return TokenTuple(tagged_word[0], tagged_word[1])


def tupleTreeToTokenlist(tree: nltk.Tree) -> list:
    """
    this function is used to convert the tree to a list of token
    :param tree:
    :return:
    """
    token_list = []
    for elem in tree:
        if isinstance(elem, nltk.Tree):
            token_list.extend(tupleTreeToTokenlist(elem))
        else:
            token_list.append(tupleToToken(elem))
    return token_list


def find_cut_pos(tokenized_sentences: str) -> list:
    """
    This function could help you find the split pointer of a long sentence
    which you expected to be cut into two part
    :param tokenized_sentences: the sentence that has be tokenized with "? . ! " symobol
    """
    # the input sentence has alreay be tokenized, so we don't segment it.
    print("The tokenized_sentences: ")
    print("\t", tokenized_sentences, end="\n\n")

    # Tokenized the sentences
    tokenized_words = nltk.word_tokenize(tokenized_sentences)
    print("Tokenized_words: ")
    print("\t", tokenized_words, end="\n\n")

    # parts of speech tagging
    tagged_words = nltk.pos_tag(tokenized_words)
    # print("Tagged_words: ")
    # print("\t", tagged_words, end="\n\n")

    # this regular expression could be better. I write two rule to extract  Coordinary Conjunction  and Subordinary Conjunction, repectively
    # revise the previous regular expression to match more unit before the ','
    split_regex = r"""CoordinaryConj: {<.*>{2,3}<,><CC><.*>{2,3}}
                      SuborinaryConj: {<.*>{2,3}<,><IN><.*>{2,3}}
                      Which_group:{<.*>{2,3}<,><WDT><VB.|MD><..+>{4}}
                   """

    # calling the regxr funcction
    index_list = regex_find(split_regex, tagged_words)

    """
    so you get a conj and subordinary conj positon dict like this:
    next we need to find the cut pointer and each conj and subordinary conj
    [
    {'conj': ('but', 'CC'), 'match_words': [Token(word='good', tag='JJ'), Token(word='enough', tag='RB'), Token(word=',', tag=','), Token(word='but', tag='CC'), Token(word='I', tag='PRP')]},
    {'conj': ('because', 'IN'), 'match_words': [Token(word='with', tag='IN'), Token(word='this', tag='DT'), Token(word=',', tag=','), Token(word='because', tag='IN'), Token(word='this', tag='DT')]}
    ]
    """

    # convert token to a single sentence
    if index_list:
        for single_ret in index_list:
            target = ""
            for index in range(len(single_ret["match_words"])):
                position = None
                # because in transcribe_res wors, the ',' and "'s" is part of a word, not a individual unit
                if (word := single_ret["match_words"][index].word) in UNCOMPLETE_STRUCTURE or word == ',' or word == ":":
                    target += word
                else:
                    target += " " + word
            # 这里面即便有's 这些不完整的成分，还是能够找到正确的位置。
            outer_position = tokenized_sentences.find(target)

            if outer_position == -1:
                print(f"Not found position for \" {target} \"")
            else:
                print(f"Found position for \" {target} \"", outer_position)
            if single_ret["conj"] != (":", ":"):
                single_ret["match_words"] = [target.strip(), outer_position + target.find(single_ret["conj"][0])]
            else:
                # deal with semincolon   rearrance code
                single_ret["match_words"] = [target.strip(), outer_position + target.find(single_ret["conj"][0])+2]

    return cal_preference(index_list, tokenized_sentences)


def rearrance_long_sentence(long_sentence: dict, choice: list) -> Union[tuple, None]:
    """
     but > and > or  if > because > whether    prefer the cut point that near the center of the long sentences.
    :param long_sentence:
    :return:
    """

    # [('and', 'CC'), ['have, and this is', 28, 5.5019762845849804]]

    match_list = choice["match_words"][0].split(" ")
    word = []
    start_time = long_sentence["start"]
    end_time = 0
    words = long_sentence["words"]

    front_sentence_part = long_sentence["text"][0:choice["match_words"][1]]
    behind_sentence_part = long_sentence["text"][choice["match_words"][1]:]

    for outer_word_index in range(len(words)):
        for match_word_index in range(len(match_list)):
            if (word_out := words[outer_word_index]["word"]) == (word_inner := match_list[match_word_index]) or \
                    word_out == word_inner + ",":
                outer_word_index += 1
                continue
            # deal with some case that the last word is "it" and the next word is "'s" and so on
            # this can lead to some problems
            elif match_word_index == len(match_list) - 1 and word_out.removeprefix(word_inner) in UNCOMPLETE_STRUCTURE:
                outer_word_index += 1
                continue
            else:
                outer_word_index -= match_word_index
                break
        if match_word_index == len(match_list) - 1:
            outer_word_index -= match_word_index + 1
            break

    if (match_word_index != len(match_list) - 1):
        return None

    # find it
    if choice["conj"] != (":", ":"):
        inner_position = match_list.index(choice["conj"][0])
    else:
    # deal with semicolon
        for i in match_list:
            if i.find(":") != -1:
                inner_position = match_list.index(i)+1
                break

    actual_position = outer_word_index + inner_position

    # deal with some case that only a number in the word without start_time and end_time
    if not (end_time:=words[actual_position-1].get("end")):
        end_time = words[actual_position].get("start")-0.1

    if not (new_start_time:=words[actual_position].get("start")):
        new_start_time = words[actual_position-1].get("end")+0.1

    if not (new_end_time := words[-1].get("end")):
        new_end_time = words[len(words)-2].get("end")

    front_words_part = words[:actual_position]
    behind_words_part = words[actual_position:]
    front_sentence = {"start": start_time, "end": end_time, "text": front_sentence_part, "words": front_words_part}
    behind_sentence = {"start": new_start_time, "end": new_end_time, "text": behind_sentence_part,
                       "words": behind_words_part}

    return front_sentence, behind_sentence


def split_long(long_sentence: dict) -> list:
    """
    :param long_sentence: a sentence that would be splited.
    :return:
    """

    LENTH_LIMIT = 100

    # 这个句子是253 个characters， 大约25秒？
    if (s_lenght := len(long_sentence["text"])) < LENTH_LIMIT:
        print(f"This length of this sentence is: {s_lenght}. It doesn't  need to be segmentated")
        print(long_sentence["text"], "\n\n")
        return [long_sentence]

    print(f"This length of this sentence is: {s_lenght} > {LENTH_LIMIT} characters, So it will be splited")
    print(long_sentence["text"], "\n\n")

    # The input sentence has alreay be tokenized, so we don't segment it.
    tokenized_sentences = long_sentence["text"]

    # we preserve the context of the cut position in order to failitite manually judge whether excute it
    all_choice_list = find_cut_pos(tokenized_sentences)
    print('All_choice_list: ')
    print("\t", all_choice_list, end="\n\n")

    # sort by preference_value
    all_choice_list.sort(key=lambda x: x["match_words"][2], reverse=True)

    print("All_choice_list.sorted: ")
    print("\t", all_choice_list, end="\n\n")

    choice = []
    if all_choice_list:
        choice = all_choice_list[0]
    else:
        ## failed to found cut point, So return itself directly
        return [long_sentence]


    # add an second choice candidate to improve the probability of success. 
    # But pay attention to the failed match case.
    try:
        front_sentence, behind_sentence = rearrance_long_sentence(long_sentence, choice)
    # 这个句子是253 个characters， 大约25秒？
    except ValueError as e:
        print("\033[91m" + "first choince failed" + "\033[0m")
        print(e)
        if len(all_choice_list) >= 2:
            choice = all_choice_list[1]
        else:
            return [long_sentence]
        try:
            print("\033[91m" + "try second choinces" + "\033[0m")
            front_sentence, behind_sentence = rearrance_long_sentence(long_sentence, choice)
        except Exception as e:
            print(e)
            print("\033[91m" + "Second choices failed too" + "\033[0m")
            return [long_sentence]
        

    ret_sentence = []

    # after cut ,we could iteralyy execute this function to find if any sentences is too long and any potential cut position
    for i in split_long(front_sentence):
        ret_sentence.append(i)
    for j in split_long(behind_sentence):
        ret_sentence.append(j)

    return ret_sentence


class Benchmark:
    """
    The method of this benchmark is used to determine the split function performance
    """

    WORDS_LIMIT = 15

    @staticmethod
    def cal_length(segments: List)->int:
        return len(segments)
    @staticmethod
    def cal_avg_length(segments: List)->Tuple[float, List]:
        length_list = []
        for index,value in enumerate(segments):
            length_list.append(len(segments[index].get("words")))
        avg = math.fsum(length_list)/len(length_list)
        return (avg.__round__(3), length_list)

    # calculate the variance of the length of the segments
    @classmethod
    def cal_variance(cls, segments: List)->float:
        avg, length_list  = cls.cal_avg_length(segments)
        variance = math.fsum([(i - avg) ** 2 for i in length_list])
        return variance.__round__(3)

    @classmethod
    def cal_length_over(cls, segments: List):

        limit_list = []

        for i in Benchmark.cal_avg_length(segments)[1]:
            if i > cls.WORDS_LIMIT:
                limit_list.append((i - cls.WORDS_LIMIT)**2)

        return sum(limit_list).__round__(3)

    @staticmethod
    def run_bench(segments: List):
        avg = Benchmark.cal_avg_length(align_result["segments"])
        variance = Benchmark.cal_variance(align_result["segments"])
        length = Benchmark.cal_length(align_result["segments"])
        length_over = Benchmark.cal_length_over(align_result["segments"])
        print("avg_length: ", avg)
        print("variance: ", variance)
        print("segments length", length)
        print("length_over", length_over)


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

    with open("./align_result.py", "r") as file:
        align_result = eval(file.read())

    Benchmark.run_bench(align_result["segments"])




    new_segments = []
    for i,j in enumerate(align_result["segments"]):
        split_ret =  split_long(j)
        for i in split_ret:
            new_segments.append(i)

    align_result["segments"] = new_segments


    Benchmark.run_bench(align_result["segments"])


    c = ("and", "CC")

    print(tupleToToken(c))

    and_token = TokenTuple("and", "CC")

    list = []

    list.append(TokenTuple("or", "CC"))
    list.append(TokenTuple("but", "CC"))

    print(and_token)
    print(and_token.word, and_token.tag)

    # split_ret = split_long(long_sentence)