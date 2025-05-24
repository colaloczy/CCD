import os
import random
import pandas as pd
import re
import multiprocessing as mp
import tqdm
import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import time

from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib

# 下载必要的NLTK数据
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def get_file_pairs(program_dir, samples_per_folder=50):
    """
    生成克隆对和非克隆对
    
    参数:
    program_dir: 数据集目录路径
    samples_per_folder: 每个文件夹采样的文件数量
    """
    clone_pairs = []
    non_clone_pairs = []
    folders = os.listdir(program_dir)
    
    # 生成克隆对
    for folder in folders:
        folder_path = os.path.join(program_dir, folder)
        if os.path.isdir(folder_path):
            # 获取所有.txt文件
            all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            # 随机采样指定数量的文件
            if len(all_files) > samples_per_folder:
                files = random.sample(all_files, samples_per_folder)
            else:
                files = all_files
                
            # 在采样的文件中生成所有可能的文件对
            for i in range(len(files)):
                for j in range(i+1, len(files)):
                    clone_pairs.append((
                        os.path.join(folder_path, files[i]),
                        os.path.join(folder_path, files[j])
                    ))
    
    # 生成非克隆对
    folder_files = {}
    for folder in folders:
        folder_path = os.path.join(program_dir, folder)
        if os.path.isdir(folder_path):
            all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            # 对每个文件夹进行采样
            if len(all_files) > samples_per_folder:
                folder_files[folder] = [
                    os.path.join(folder_path, f) 
                    for f in random.sample(all_files, samples_per_folder)
                ]
            else:
                folder_files[folder] = [
                    os.path.join(folder_path, f) 
                    for f in all_files
                ]
    
    folder_list = list(folder_files.keys())
    # 在不同文件夹间随机选择文件对
    for i in range(len(folder_list)):
        for j in range(i+1, len(folder_list)):
            folder1 = folder_list[i]
            folder2 = folder_list[j]
            files1 = folder_files[folder1]
            files2 = folder_files[folder2]
            
            # 从每对文件夹中随机选择一些文件对
            pairs_needed = min(len(files1), len(files2))
            selected_files1 = random.sample(files1, pairs_needed)
            selected_files2 = random.sample(files2, pairs_needed)
            
            for f1, f2 in zip(selected_files1, selected_files2):
                non_clone_pairs.append((f1, f2))
    
    # 平衡数据集
    if len(clone_pairs) > len(non_clone_pairs):
        clone_pairs = random.sample(clone_pairs, len(non_clone_pairs))
    else:
        non_clone_pairs = random.sample(non_clone_pairs, len(clone_pairs))
    
    print(f"Generated {len(clone_pairs)} clone pairs and {len(non_clone_pairs)} non-clone pairs")
    return clone_pairs, non_clone_pairs


def get_code_block(file_path):
    """读取源代码文件并提取标记"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 移除注释
        content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # 定义C++关键字和运算符
        cpp_keywords = set(['auto', 'break', 'case', 'char', 'const', 'continue', 
                          'default', 'do', 'double', 'else', 'enum', 'extern', 
                          'float', 'for', 'goto', 'if', 'int', 'long', 'register', 
                          'return', 'short', 'signed', 'sizeof', 'static', 'struct', 
                          'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 
                          'while', 'class', 'new', 'delete', 'private', 'protected', 
                          'public', 'this', 'operator', 'template', 'try', 'catch', 
                          'throw', 'using', 'namespace', 'virtual'])
        
        # 标记化代码
        # 分割标识符、运算符和其他标记
        tokens = re.findall(r'[a-zA-Z_]\w*|[+\-*/=<>!]=?|[{}()\[\];,.]|"(?:\\.|[^"\\])*"|\d+(?:\.\d+)?', content)
        
        # 处理标记
        processed_tokens = []
        for token in tokens:
            if token in cpp_keywords:
                processed_tokens.append(('keyword', token))
            elif re.match(r'^[0-9]+(?:\.[0-9]+)?$', token):
                processed_tokens.append(('number', token))
            elif re.match(r'^".*"$', token):
                processed_tokens.append(('string', token))
            elif re.match(r'^[a-zA-Z_]\w*$', token):
                processed_tokens.append(('identifier', token))
            elif token in ['+', '-', '*', '/', '=', '==', '!=', '>', '<', '>=', '<=', '++', '--']:
                processed_tokens.append(('operator', token))
            else:
                processed_tokens.append(('punctuation', token))
                
        return processed_tokens
    except Exception as e:
        #print(f"Error processing file {file_path}: {str(e)}")
        return []
        
        
def get_code_block_token(file_path):
    """读取源代码文件并提取标记"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 移除注释
        content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # 定义C++关键字和运算符
        cpp_keywords = set(['auto', 'break', 'case', 'char', 'const', 'continue', 
                          'default', 'do', 'double', 'else', 'enum', 'extern', 
                          'float', 'for', 'goto', 'if', 'int', 'long', 'register', 
                          'return', 'short', 'signed', 'sizeof', 'static', 'struct', 
                          'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 
                          'while', 'class', 'new', 'delete', 'private', 'protected', 
                          'public', 'this', 'operator', 'template', 'try', 'catch', 
                          'throw', 'using', 'namespace', 'virtual'])
        
        # 标记化代码
        # 分割标识符、运算符和其他标记
        tokens = re.findall(r'[a-zA-Z_]\w*|[+\-*/=<>!]=?|[{}()\[\];,.]|"(?:\\.|[^"\\])*"|\d+(?:\.\d+)?', content)
        
        # 处理标记
        processed_tokens = []
        for token in tokens:
                processed_tokens.append(token)
                
        return processed_tokens
    except Exception as e:
        #print(f"Error processing file {file_path}: {str(e)}")
        return []


def get_code_block_type(file_path):
    """读取源代码文件并提取标记"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 移除注释
        content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # 定义C++关键字和运算符
        cpp_keywords = set(['auto', 'break', 'case', 'char', 'const', 'continue', 
                          'default', 'do', 'double', 'else', 'enum', 'extern', 
                          'float', 'for', 'goto', 'if', 'int', 'long', 'register', 
                          'return', 'short', 'signed', 'sizeof', 'static', 'struct', 
                          'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 
                          'while', 'class', 'new', 'delete', 'private', 'protected', 
                          'public', 'this', 'operator', 'template', 'try', 'catch', 
                          'throw', 'using', 'namespace', 'virtual'])
        
        # 标记化代码
        # 分割标识符、运算符和其他标记
        tokens = re.findall(r'[a-zA-Z_]\w*|[+\-*/=<>!]=?|[{}()\[\];,.]|"(?:\\.|[^"\\])*"|\d+(?:\.\d+)?', content)
        
        # 处理标记
        processed_tokens = []
        for token in tokens:
            if token in cpp_keywords:
                processed_tokens.append(token)
            elif re.match(r'^[0-9]+(?:\.[0-9]+)?$', token):
                processed_tokens.append('number')
            elif re.match(r'^".*"$', token):
                processed_tokens.append('string')
            elif re.match(r'^[a-zA-Z_]\w*$', token):
                processed_tokens.append('id')
            else:
                processed_tokens.append(token)
                
        return processed_tokens
    except Exception as e:
        #print(f"Error processing file {file_path}: {str(e)}")
        return []
        
def intersection_and_union(group1, group2):
    """计算两组标记的交集和并集"""
    intersection = 0
    union = 0
    tokens1 = {}
    tokens2 = {}
    
    for token in group1:
        tokens1[token] = tokens1.get(token, 0) + 1
    for token in group2:
        tokens2[token] = tokens2.get(token, 0) + 1
        
    for token in set(group1).union(set(group2)):
        intersection += min(tokens1.get(token, 0), tokens2.get(token, 0))
        union += max(tokens1.get(token, 0), tokens2.get(token, 0))
        
    return intersection, union

def Jaccard_sim(group1, group2):
    intersection, union = intersection_and_union(group1, group2)
    return float(intersection) / union if union != 0 else 0

def lda_similarity(group1, group2, num_topics=5):
    """计算两个字符串列表的LDA主题相似性"""
    try:
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
        vectorizer = CountVectorizer(stop_words=stop_words, min_df=1)
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
    except Exception as e:
        print(f"Error in LDA similarity calculation: {str(e)}")
        return 0.0

def calculate_similarity(file_pair):
    file1, file2 = file_pair
    try:
        block1 = get_code_block_token(file1)
        block2 = get_code_block_token(file2)
        
        if not block1 or not block2:
            return None
            
        # 计算各种相似度
        jaccard = Jaccard_sim(block1, block2)
        jaro = Levenshtein.jaro(" ".join(block1), " ".join(block2))
        levenshtein_ratio = Levenshtein.ratio(" ".join(block1), " ".join(block2))
        lda_sim = lda_similarity(block1, block2)
        
        return [file1, file2, jaccard, jaro,  levenshtein_ratio, lda_sim]
    except Exception as e:
        print(f"Error calculating similarity for {file1} and {file2}: {str(e)}")
        return None

def main():
    program_dir = "dataset/POJ104"  # 数据集路径
    #output_dir = ""
    samples_per_folder = 50  # 每个文件夹采样50个文件
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # 获取文件对
    print(f"Generating file pairs (sampling {samples_per_folder} files per folder)...")
    clone_pairs, non_clone_pairs = get_file_pairs(program_dir, samples_per_folder)
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用多进程计算相似度
    print("Calculating similarities...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # 处理克隆对
        clone_features = list(tqdm.tqdm(
            pool.imap(calculate_similarity, clone_pairs),
            total=len(clone_pairs)
        ))
        
        # 处理非克隆对
        non_clone_features = list(tqdm.tqdm(
            pool.imap(calculate_similarity, non_clone_pairs),
            total=len(non_clone_pairs)
        ))
    
    # 记录结束时间
    end_time = time.time()
    
    # 打印耗时
    total_time = end_time - start_time
    print(f"Total time taken for similarity calculation: {total_time:.2f} seconds")
    
    
    # 移除None值
    clone_features = [f for f in clone_features if f is not None]
    non_clone_features = [f for f in non_clone_features if f is not None]
    
    # 创建数据框
    columns = [
    'file1', 'file2', 
    'Jaccard',  'Jaro', 'Levenshtein_ratio',
    'LDA_similarity'
    ]

    clone_df = pd.DataFrame(clone_features, columns=columns)
    clone_df['label'] = 1

    non_clone_df = pd.DataFrame(non_clone_features, columns=columns)
    non_clone_df['label'] = 0

    
    # 合并并保存结果
    final_df = pd.concat([clone_df, non_clone_df], ignore_index=True)
    final_df.to_csv('poj104_token.csv', index=False)
    print(f"Saved features to poj104_token.csv")
    
if __name__ == '__main__':
    main() 
