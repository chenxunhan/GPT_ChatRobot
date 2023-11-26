import jieba
import json
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

def rmSpaceTabData(srcPath, optPath):
    with open(optPath, 'w', encoding='utf-8') as fo:
        with open(srcPath, 'r', encoding='utf-8') as fi:
            for line in fi:
                line = line.replace(" ",'')
                fo.write(line.strip()+'\n')

def tokTabData(srcPath, optPath):
    # 读取srcPath文件
    # 遍历每一行
        # 将每一行行里的所有空格消除
        # 将每一行暗\t拆分成s1 s2
        # 将s1 s2分词
        # 将s1 s2 的list 按" " join
        # 将s1 s2 用\t合并
        # 把每一行保存在optPath
    with open(optPath, 'w', encoding='utf-8') as fo:
        with open(srcPath, 'r', encoding='utf-8') as fi:
            for line in fi:
                line = line.replace(" ",'')
                l_line = line.split('\t')
                l_line = [" ".join(jieba.lcut(l)) for l in l_line]
                line = "\t".join(l_line)
                fo.write(line.strip()+'\n')

def combineFile(list_filesPath, optPath):
    with open(optPath, 'w', encoding='utf-8') as fo:
        for file in list_filesPath:
            with open(file, 'r', encoding='utf-8') as fi:
                for line in fi:
                    fo.write(line)

def getTabData2DictJson(srcPath, optPath):
    with open(srcPath, 'r', encoding='utf-8') as f:
        datas = f.readlines()

    word_count = {}
    for data in datas:
        l_data = data.strip().replace('\t', ' ').split(' ')
        for word in l_data:
            word = word.strip()
            word_count.setdefault(word, 0)
            word_count[word] += 1
    word2id = {"<pad>": 0, "<unk>": 1, "<sep>": 2}

    temp = {word: i + len(word2id) for i, word in enumerate(word_count.keys())}
    word2id.update(temp)
    id2word = list(word2id.keys())

    dict_datas = {"word2id": word2id, "id2word": id2word}

    json.dump(dict_datas, open(optPath, 'w', encoding='utf-8'))

def getCharTabData2DictJson(srcPath, optPath):
    with open(srcPath, 'r', encoding='utf-8') as f:
        datas = f.readlines()

    word_count = {}
    for data in datas:
        data = data.strip().replace('\t', '')
        for word in data:
            word_count.setdefault(word, 0)
            word_count[word] += 1
    word2id = {"<pad>": 0, "<unk>": 1, "<sep>": 2}

    temp = {word: i + len(word2id) for i, word in enumerate(word_count.keys())}
    word2id.update(temp)
    id2word = list(word2id.keys())

    dict_datas = {"word2id": word2id, "id2word": id2word}

    json.dump(dict_datas, open(optPath, 'w', encoding='utf-8'))

if __name__ == '__main__':
    # give up token
    # tokTabData('dataset/LCCC-base_test.txt', 'dataset/LCCC-base_test_tok.txt')
    # tokTabData('dataset/LCCC-base_valid.txt', 'dataset/LCCC-base_valid_tok.txt')
    # combineFile(['dataset/LCCC-base_train_tok.txt', 'dataset/LCCC-base_test_tok.txt', 'dataset/LCCC-base_valid_tok.txt'], 'dataset/LCCC-base_train_test_valid_tok.txt')
    # allLineQuestPath = 'dataset/LCCC-base_train_test_valid_tok.txt'
    # savDictJsonPath = 'dataset/dict_datas.json'
    # getTabData2DictJson(allLineQuestPath, savDictJsonPath)

    # char preprocess
    # rmSpaceTabData('dataset/LCCC-base_train.txt', 'dataset/train.txt')
    # rmSpaceTabData('dataset/LCCC-base_test.txt', 'dataset/test.txt')
    # rmSpaceTabData('dataset/LCCC-base_valid.txt', 'dataset/valid.txt')
    # combineFile(
    #     ['dataset/train.txt', 'dataset/test.txt', 'dataset/valid.txt'],
    #     'dataset/train_test_valid.txt')
    # savDictJsonPath = 'dataset/dict_datas.json'
    # getCharTabData2DictJson('dataset/train_test_valid.txt', savDictJsonPath)

    print(wordpunct_tokenize("你 你 你 你 你你你你你"))