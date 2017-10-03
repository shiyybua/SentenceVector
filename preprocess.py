# -*- coding: utf-8 -*-
import json
import re
import jieba
DATA_PATH = '/Users/mac/Desktop/law_raw_data_mini/criminal_0215.json'


def process(line, file_source, file_source_before, file_source_after):
    def to_unicode(line):
        line = line.strip()
        if type(line) is str:
            line = line.decode('utf8')
        return line

    split_token = re.compile(u"[。？！']")
    court_find = to_unicode(line.get("court_find"))
    judge_result = to_unicode(line.get("judge_result"))
    court_find = split_token.split(court_find)
    judge_result = split_token.split(judge_result)

    court_find = map(lambda x: ' '.join([e.encode('utf8') for e in jieba.cut(x.encode('utf8'))]) + '\n', court_find)
    judge_result = map(lambda x: ' '.join([e.encode('utf8') for e in jieba.cut(x.encode('utf8'))]) + '\n', judge_result)

    if len(court_find) >= 3:
        for i in range(1, len(court_find) - 1):
            file_source.write(court_find[i])
            file_source_before.write(court_find[i-1])
            file_source_after.write(court_find[i+1])

    if len(judge_result) >= 3:
        for i in range(1, len(judge_result) - 1):
            file_source.write(judge_result[i])
            file_source_before.write(judge_result[i-1])
            file_source_after.write(judge_result[i+1])


def start():
    with open(DATA_PATH, 'r') as file:
        position = 0
        file_source = open("resource/source.txt", "w")
        file_source_before = open("resource/source_before.txt", "w")
        file_source_after = open("resource/source_after.txt", "w")
        try:
            while True:
                try:
                    line = file.readline()
                    line = json.loads(line)
                    process(line, file_source, file_source_before, file_source_after)
                    position += 1
                except ValueError:
                    print position
                    pass

                # if position > 0: break
        except EOFError:
            file_source.close()
            file_source_before.close()
            file_source_after.close()


if __name__ == '__main__':
    start()

