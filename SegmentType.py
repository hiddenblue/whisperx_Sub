from builtins import KeyError
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from typing import List, Dict, Any
import re

class TransSegment:
    """
    self-designed data type, which has some property responding to value in segment
    start: int
    end: int
    text: str
    """
    def __init__(self, segment:dict):
        self.text = segment.get('text')
        self.start = segment.get('start')
        self.end = segment.get('end')

    def __del__(self):
        # print("I am garbage. I will be collected!")
        pass

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return str(self.toDict())

    def toDict(self):
        return {'text': self.text, 'start': self.start, 'end': self.end}

    def __add__(self, other):

        if not isinstance(other, (TransSegment, dict)):
            raise TypeError
            # + other is TransSegment type
        if isinstance(other, TransSegment):
            if other.start < self.start:
                start = other.start
                end = self.end
                text = other.text + self.text
            else:
                start = self.start
                end = other.end
                text = self.text + other.text
            return TransSegment({'text': text, 'start': start, 'end': end}).toDict()

            # + other is a raw segment dict
        elif isinstance(other, dict) and other.get('text') and other.get('start') and other.get('end'):
            if other.get('start') < self.start:
                start = other.get('start')
                end = self.end
                text = other.get('text') + self.text
            else:
                start = self.start
                end = other.get('end')
                text = self.text + other.get('text')
            return TransSegment({'text': text, 'start': start, 'end': end}).toDict()
        else:
            raise TypeError("Unsupported type for addition. Expected TransSegment or dict with text, start, end keys")

    @staticmethod
    def add(seg1: Dict, seg2: Dict) -> Dict:
        """this method is used to merge two segment"""
        if not isinstance(seg1, Dict) and not isinstance(seg2, Dict):
            raise TypeError

        start1 = seg1.get('start')
        end1 = seg1.get('end')
        text1 = seg1.get('text')

        start2 = seg2.get('start')
        end2 = seg2.get('end')
        text2 = seg2.get('text')

        # if any of vaue above is None, return KeyError
        if start1 is None or end1 is None or text1 is None or start2 is None or end2 is None or text2 is None:
            raise KeyError

        if end1 <= start2:
            return {'text': text1 + text2, 'start': start1, 'end': end2, 'words': text1 + text2.rstrip()}
        elif end2 <= start1:
            return {'text': text2 + text1, 'start': start2, 'end': end1, 'words': text2 + text1.rstrip()}
        else:
            raise TypeError("Unsupported type for addition. Expected TransSegment or dict with text, start, end keys")

class AlignSegment(TransSegment):
    """

    data type for alignment segment
    """

    def __init__(self, alignSegment):
        super().__init__(alignSegment)
        self.words = alignSegment.get('words')

    def toDict(self):
        return {'text': self.text, 'start': self.start, 'end': self.end, 'words': self.words}

    def __add__(self, other):
        if not isinstance(other, (AlignSegment, dict)):
            raise TypeError
        if isinstance(other, AlignSegment):
            if other.start < self.start:
                raise TypeError
            else:
                start = self.start
                end = other.end
                text = self.text + other.text
                words = self.words + other.words
            return AlignSegment({'text': text, 'start': start, 'end': end, 'words': words})

        elif isinstance(other, dict):
            if other.get("start") < self.start:
                raise TypeError
            else:
                start = self.start
                end = other.get("end")
                text = self.text + other.get("text")
                words = self.words + other.get("words")
            return AlignSegment({'text': text, 'start': start, 'end': end, 'words': words})
        else:
            raise TypeError("Unsupported type for addition. Expected TransSegment or dict with 'text', 'start', 'end', 'words' keys.")

class SegmentMerge:
    @classmethod
    def is_mergeable(cls, seg1: Dict, seg2: Dict)->bool:
        """
        Check if two segments can be merged
        """
        continuing_end = re.findall("""[\w\d](?!\.|\?|\!)[,]*?$""", seg1.get('text', ''), flags=re.M | re.S)

        lower_case_start = re.findall("""^[a-z0-9]""", seg2.get('text', '').strip(), flags=re.M | re.S)
        if continuing_end and lower_case_start:
            return True
        else:
            return False
    @classmethod
    def merge_seg_list(cls, segList: List[Dict[str, Any]])->Dict[str, Any]:
        """
        you could input two seg [seg1, seg2]

        :param segList:
        :return:
        """
        if len(segList) == 2:
            return TransSegment.add(segList[0], segList[1])
        else:
            tmp = segList[0]
            for i in range(1, len(segList)):
                tmp = TransSegment.add(tmp, segList[i])
            return tmp

    @classmethod
    def merge_continue_segment(cls, segments: List[Dict[str, Any]])-> List[Dict[str,Any]]:
        """
        This function merges sentences in transcribe_result that don't end with ".", "?", or "!"
        To improve sentence tokenization and translation, these sentences are merged with their following ones.
        """
        if not segments:
            logging.info("Input segments list is empty.")
            return []

        new_segments = []
        curr = 0
        tmp = []
        while(curr < len(segments)):

            if cls.is_mergeable(segments[curr], segments[curr+1]):
                tmp.append(segments[curr])
                curr += 1

                if curr == len(segments)-1:
                    tmp.append(segments[curr])
                    new_segments.append(cls.merge_seg_list(tmp))
                    break
            else:
                if tmp:
                    tmp.append(segments[curr])
                    new_segments.append(cls.merge_seg_list(tmp))
                    tmp = []
                else:
                    new_segments.append(segments[curr])
                curr += 1
                if (curr == len(segments)-1):
                    new_segments.append(segments[curr])
                    break
        return new_segments



if __name__ == '__main__':

    a = {
        'text': ' What is going on everybody and welcome to a bit of an update and exhibition of this new function calling capability as well as like some of these other API updates but mostly things are just faster and cheaper now but the big thing here is function calling and I think',
        'start': 0.009, 'end': 16.374}

    b = {
        'text': " this is going to be a massive change in programming just in general, but also the kind of intersection of programming and AI and adding sort of intelligence to your programming. So what I want to do is just show some quick examples of kind of how I've played with it so far, as well as kind of run through their example.",
        'start': 16.374, 'end': 37.841}


    result = TransSegment.add(a, b)
    print(result)
