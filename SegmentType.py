from builtins import KeyError

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
        print("I am garbage. I will be collected!")

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return str(self.toDict())

    def toDict(self):
        return {'text': self.text, 'start': self.start, 'end': self.end}

    def __add__(self, other):
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
        else:
            raise KeyError
        return TransSegment({'text': text, 'start': start, 'end': end}).toDict()

class AlignSegment(TransSegment):
    """

    data type for alignment segment
    """

    def __init__(self, alignSegment):
        super().__init__(alignSegment)
        self.words = alignSegment.get('words')

    def __toDict(self):
        return {'text': self.text, 'start': self.start, 'end': self.end, 'words': self.words}

    def __add__(self, other):

        if isinstance(other, AlignSegment):
            if other.start < self.start:
                raise KeyError
            else:
                start = self.start
                end = other.end
                text = self.text + other.text
                words = self.words + other.words
        if isinstance(other, dict):
            if other.get("start") < self.start:
                raise KeyError
            else:
                start = self.start
                end = other.get("end")
                text = self.text + other.get("text")
                words = self.words + other.get("words")
        return AlignSegment({'text': text, 'start': start, 'end': end, 'words': words}).toDict()


if __name__ == '__main__':

    a = {
        'text': ' What is going on everybody and welcome to a bit of an update and exhibition of this new function calling capability as well as like some of these other API updates but mostly things are just faster and cheaper now but the big thing here is function calling and I think',
        'start': 0.009, 'end': 16.374}

    b = {
        'text': " this is going to be a massive change in programming just in general, but also the kind of intersection of programming and AI and adding sort of intelligence to your programming. So what I want to do is just show some quick examples of kind of how I've played with it so far, as well as kind of run through their example.",
        'start': 16.374, 'end': 37.841}
    transSegment = TransSegment(a)
    transSegmentb = TransSegment(b)

    print(transSegment.text)
    print(transSegment+b)

    print(transSegment.text)