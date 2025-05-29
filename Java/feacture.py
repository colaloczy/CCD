import javalang
import Levenshtein
import pandas as pd
import multiprocessing as mp
import tqdm
import math
from functools import partial
import time
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# The dataset is under the master branch, decompress id2sourcecode into the dataset folder.
def lda_similarity(group1, group2, num_topics=5):
    """计算两个字符串列表的LDA主题相似性"""
    
    # 确保输入为文本列表，并且都是字符串
    if not isinstance(group1, list) or not isinstance(group2, list):
        raise ValueError("group1 和 group2 必须是字符串列表。")
    if not all(isinstance(item, str) for item in group1):
        raise ValueError("group1 中的所有元素必须是字符串。")
    if not all(isinstance(item, str) for item in group2):
        raise ValueError("group2 中的所有元素必须是字符串。")
    
    # 将两个组拼接为一份文档
    documents = [' '.join(group1), ' '.join(group2)]
    
    # 使用CountVectorizer转换文档为词袋模型，去掉停用词
    stop_words = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words=stop_words)
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    # 训练LDA模型
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda_model.fit(doc_term_matrix)
    
    # 计算两个文档的主题分布
    topic_dist_1 = lda_model.transform(doc_term_matrix[0:1])[0]
    topic_dist_2 = lda_model.transform(doc_term_matrix[1:2])[0]
    
    # 计算余弦相似度
    similarity = cosine_similarity([topic_dist_1], [topic_dist_2])[0][0]
    
    return similarity



global logfile  # 声明logfile为全局变量
# 全局变量
logfile = None

def get_sim(tool, dataframe):
    inputpath = 'dataset/id2sourcecode/'

    sim = []
    for _, pair in dataframe.iterrows():
        id1, id2 = pair.FunID1, pair.FunID2

        sourcefile1 = inputpath + str(id1) + '.java'
        sourcefile2 = inputpath + str(id2) + '.java'
        try:
            similarity = runner(tool, sourcefile1, sourcefile2)
        except Exception as e:
            similarity = repr(e).split('(')[0]
            log = "\n" + time.asctime() + "\t" + tool + "\t" + str(id1) + "\t" + str(id2) + "\t" + similarity
            similarity = 'False'
#         print(similarity)
        sim.append(similarity)

    return sim


def getCodeBlock(file_path):     
    block = []
    # print(file_path)
    with open(file_path, 'r', encoding='utf-8') as temp_file:
        lines = temp_file.readlines()
        for line in lines:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                if type(token) == javalang.tokenizer.Identifier:
                    block.append("id")
                else:
                    block.append(token.value)
    return block

def getCodeBlock_type(file_path):     # 类型
    block = []
    # print(file_path)
    with open(file_path, 'r', encoding='utf-8') as temp_file:
        lines = temp_file.readlines()
        for line in lines:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                token_type = str(type(token))[:-2].split(".")[-1]
                block.append(token_type)
    return block

def getCodeBlock_token_and_type(file_path):     # 类型加token
    block = []
    # print(file_path)
    with open(file_path, 'r', encoding='utf-8') as temp_file:
        lines = temp_file.readlines()
        for line in lines:
            tokens = list(javalang.tokenizer.tokenize(line))
            for token in tokens:
                if type(token) == javalang.tokenizer.Identifier:
                    block.append("id")
                else:
                    block.append(token.value)
                token_type = str(type(token))[:-2].split(".")[-1]
                block.append(token_type)
    return block

def runner(tool, sourcefile1, sourcefile2):
    block1 = getCodeBlock(sourcefile1)
    block2 = getCodeBlock(sourcefile2)
    if tool == 't1':
        return Jaccard_sim(block1, block2)
    elif tool == 't3':
        return Jaro_sim(block1, block2)
    elif tool == 't6':
        return Levenshtein_ratio(block1, block2)
    elif tool == 't11':
        return lda_similarity(block1, block2)  # 通过LDA计算相似性


def intersection_and_union(group1, group2):
    intersection = 0
    union = 0
    triads_num1 = {}
    triads_num2 = {}
    for triad1 in group1:
        triads_num1[triad1] = triads_num1.get(triad1, 0) + 1
    for triad2 in group2:
        triads_num2[triad2] = triads_num2.get(triad2, 0) + 1

    for triad in list(set(group1).union(set(group2))):
        intersection += min(triads_num1.get(triad, 0), triads_num2.get(triad, 0))
        union += max(triads_num1.get(triad, 0), triads_num2.get(triad, 0))
    return intersection, union


def Jaccard_sim(group1, group2):
    # Jaccard 系数

    intersection, union = intersection_and_union(group1, group2)
    # 除零处理
    sim = float(intersection) / union if union != 0 else 0
    return sim



def Jaro_sim(group1, group2):
    # Jaro相似性

    sim = Levenshtein.jaro(group1, group2)
    return sim


def Levenshtein_ratio(group1, group2):
    # Levenshtein比
    sim = Levenshtein.ratio(group1, group2)
    return sim

def cut_df(df, n):
    df_num = len(df)
    every_epoch_num = math.floor((df_num/n))
    df_split = []
    for index in range(n):
        if index < n-1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        df_split.append(df_tem)
    return df_split


def main():
    inputcsv = "dataset/nonclone.csv"
    #inputcsv = "dataset/clone.csv"
#     inputcsv = "dataset/type-1.csv"

    Clonetype = inputcsv.split('/')[-1].split('.')[0]
    
    if 'nonclone' in inputcsv:
        Clonetype = 'nonclone'
    else:
        Clonetype = 'clone'
    
    
#     Clonetype = 'type-1'

    pairs = pd.read_csv(inputcsv, header=None)
    pairs = pairs.drop(labels=0)
    pairs.columns = ['FunID1', 'FunID2']
    
    df_split = cut_df(pairs, 60)
    
    func1 = partial(get_sim, 't1')
    '''
    偏函数允许你固定一个或多个参数的值，从而生成一个新的可调用对象。
    '''
    pool = mp.Pool(processes=60)
    sim_t1 = []
    it_sim_t1 = tqdm.tqdm(pool.imap(func1, df_split))
    '''
    tqdm.tqdm()：创建一个进度条对象。
    pool.imap()：从multiprocessing.Pool中调用，用于并行执行func1函数。imap会返回一个迭代器
    func1：要并行执行的函数。
    df_split：传递给func1的参数。
    '''
    for item in it_sim_t1:
        sim_t1 = sim_t1 + item
    pool.close()
    pool.join()

    func3 = partial(get_sim, 't3')
    pool = mp.Pool(processes=60)
    sim_t3 = []
    it_sim_t3 = tqdm.tqdm(pool.imap(func3, df_split))
    for item in it_sim_t3:
        sim_t3 = sim_t3 + item
    pool.close()
    pool.join()

    func6 = partial(get_sim, 't6')
    pool = mp.Pool(processes=60)
    sim_t6 = []
    it_sim_t6 = tqdm.tqdm(pool.imap(func6, df_split))
    for item in it_sim_t6:
        sim_t6 = sim_t6 + item
    pool.close()
    pool.join()
    
    func11 = partial(get_sim, 't11')

    pool = mp.Pool(processes=60)
    sim_t11 = []
    it_sim_t11 = tqdm.tqdm(pool.imap(func11, df_split))

    for item in it_sim_t11:
        sim_t11 = sim_t11 + item

    pool.close()
    pool.join()


    result = pd.DataFrame({
        'FunID1': pairs['FunID1'].to_list(),
        'FunID2': pairs['FunID2'].to_list(),
        't1_sim': sim_t1,
        't3_sim': sim_t3,
        't6_sim': sim_t6,
        't9_sim': sim_t11
    })

    result.to_csv(Clonetype + '_token_JaccardJaroLeven-RaLDA.csv', index=False)
    
    

if __name__ == '__main__':
    parse_er_file = open('parser_error.txt', 'r', encoding='utf-8')  # 没有则创建
    wrongfile = parse_er_file.read().split(' ')
    #logfile = open('', 'a', encoding='utf-8')
    start = time.time()
    main()

    end = time.time()
    t = end - start
    print(t)
