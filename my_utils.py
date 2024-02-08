# Utils

import sqlite3
import argparse
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os
from numpy.linalg import norm
import sys
from itertools import combinations
import random
import re
import string

table_names = [
    "spark",
    "eclipse",
    # "eclipse_initial",
    "eclipse_old",
    "hadoop",
    "hadoop_1day",
    "hadoop_old",
    "kibana",
    "mozilla",
    # "mozilla_initial",
    "mozilla_old",
    "spark_1day",
    "vscode",]



class UnionFind:
    def __init__(self):
        self.parent = {}  # Dictionary to store parent nodes
        self.ranks = {}    # Dictionary to store rank (or size) of each set
        self.processed = False
        self.project_name = None

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.ranks[x] = 1
            return x

        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.ranks[root_x] < self.ranks[root_y]:
                self.parent[root_x] = root_y
                self.ranks[root_y] += self.ranks[root_x]
            else:
                self.parent[root_y] = root_x
                self.ranks[root_x] += self.ranks[root_y]
            
    def process_project(self, conn, project_name, min_desc_length=50):
        cursor = conn.cursor()
        column_names = get_column_names(conn, project_name)
        self.project_name = project_name
        
        cursor.execute(f"SELECT * FROM {project_name}")
        print("Processing", project_name)
        for row in tqdm(cursor.fetchall()):
            dup_id = int(row[column_names.index("dup_id")])
            if dup_id == -1: continue
            bug_id = int(row[column_names.index("bug_id")])
            if (dup_id == bug_id): continue
            assert(dup_id != bug_id)
            desc1 = get_description(conn, project_name, bug_id)
            desc2 = get_description(conn, project_name, dup_id)
            if (len(desc1.split(" ")) < 50 or len(desc2.split(" ")) < 50): continue
            self.union(bug_id, dup_id)
        self.processed = True
    
    def process_json_data(self, dataset, project_name):
        for point in dataset:
            dup_id = point["dup_id"]
            bug_id = point["bug_id"]
            if dup_id is None or bug_id is None: continue
            assert(bug_id != dup_id)
            self.union(bug_id, dup_id)
        self.processed = True
            
    def get_roots(self,):
        assert(self.processed)
        return list(set(self.parent.values()))
    
    def get_children(self, parent):
        assert(self.processed)
        parent = self.find(parent)
        if (parent is None): return None
        children = [key for key, value in self.parent.items() if value == parent]
        return children
    
    def get_all_children(self, ):
        return [key for key, value in self.parent.items()]
    
    def are_dups(this, bug_id1, bug_id2):
        if (bug_id1 not in this.parent.keys() or bug_id2 not in this.parent.keys()):
            return False
        return this.parent[bug_id1] == this.parent[bug_id2]
    
    def get_duplicated_pairs(self):
        roots = self.get_roots()
        pairs = []
        for root in roots:
            group = self.get_children(root)
            pairs += list(combinations(group, 2))
        return pairs
    
    def num_dups(self, ):
        roots = self.get_roots()
        count = 0
        for root in tqdm(roots):
            group = self.get_children(root)
            n = len(group)
            count += n*(n-1)/2
        print(self.project_name, "has", count, "duplicated_pairs")
        return count
    
    def get_dup(self,):
        assert(self.processed)
        
        
            
            
def get_bug_ids(conn, table_name, filter=True):
    cursor = conn.cursor()
    column_name = "bug_id"

    # Fetch table names using SQL query
    cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE length(description) ORDER BY {column_name};")
    distinct_values_sorted = cursor.fetchall()

    # Extract table names from the result
    return [value[0] for value in distinct_values_sorted]

def preprocess_text(text, remove_punc=True, lowercase=True, remove_nums=True, remove_stopwords=True, ):
    # Remove punctuations:
    text = cleaned_text = ''.join(char for char in text if char not in string.punctuation)
    # Lower casing:

def get_column_names(conn, table_name):
    cursor = conn.cursor()

    # Execute a query to get information about the columns in the specified table
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns_info = cursor.fetchall()

    # Extract and return the column names
    column_names = [column[1] for column in columns_info]
    return column_names


def get_description(conn, project_name, bug_id):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    query = f"SELECT * FROM {project_name} WHERE bug_id = {bug_id};"
    column_names = get_column_names(conn, project_name)
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()[0]
    desc = result[column_names.index("description")]
    # short_desc = result[column_names.index("short_desc")]

    # Extract table names from the result
    # return (desc + " \n ").replace("\\'", "'")
    return desc.replace("\\'", "'")

def get_short_desc(conn, project_name, bug_id):
    cursor = conn.cursor()
    # Fetch table names using SQL query
    query = f"SELECT * FROM {project_name} WHERE bug_id = {bug_id};"
    column_names = get_column_names(conn, project_name)
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()[0]
    # desc = result[column_names.index("description")]
    short_desc = result[column_names.index("short_desc")]

    # Extract table names from the result
    return short_desc.replace("\\'", "'")


def get_code_feature(conn, project_name, bug_id):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    query = f"SELECT * FROM {project_name} WHERE bug_id = {bug_id};"
    column_names = get_column_names(conn, project_name)
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()[0]
    return result[column_names.index("code_feature")]

def get_desc_wo_stacktrace(conn, project_name, bug_id):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    query = f"SELECT * FROM {project_name} WHERE bug_id = {bug_id};"
    column_names = get_column_names(conn, project_name)
    assert("desc_wo_stacktrace" in column_names)
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()[0]
    return result[column_names.index("short_desc")] + " " + result[column_names.index("desc_wo_stacktrace")]

def get_stacktrace(conn, project_name, bug_id):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    query = f"SELECT * FROM {project_name} WHERE bug_id = {bug_id};"
    column_names = get_column_names(conn, project_name)
    assert("stacktrace" in column_names)
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()[0]
    return result[column_names.index("stacktrace")]
    

def vectorize(description, stride_len, chunk_size):
    tokens = tokenizer.tokenize(description)
    # if len og token array is < 32, we do nothing as there is not enough information
    if (len(tokens) < chunk_size // 2): return None

    # remember to add cls and sep token at each chunk
    token_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token]+tokens+[tokenizer.sep_token])

    # divide token ids into batche of chunks
    chunk_list=[]
    for i in range(0, len(token_ids), stride_len):
        chunk = token_ids[i:min(i+chunk_size, len(token_ids))]
        assert(len(chunk) <= chunk_size)
        if len(chunk) < chunk_size:
            # keep going
            continue
            # if (len(chunk) < chunk_size // 2): continue
            # pad_length = chunk_size - len(chunk)
            # chunk += [tokenizer.pad_token_id]*pad_length
        assert(len(chunk) == chunk_size)
        # print(chunk)
        chunk_list.append(chunk)

    if(len(chunk_list) == 0): return None
    chunk_arr = np.array(chunk_list)
    # print("Chunk arr size{}".format(chunk_arr.shape))
    # context_embedding = model(torch.tensor(token_ids[:512])[None, :])[0]
    context_embedding = model(torch.tensor(chunk_arr)[:, :])[0]
    return context_embedding.detach().numpy()


def get_duplicated_pairs(union_find):
    roots = union_find.get_roots()
    pairs = []
    for root in roots:
        group = union_find.get_children(root)
        pairs += list(combinations(group, 2))
    return pairs


def get_non_duplicated_pairs(union_find, conn, size):
    from_dup = union_find.get_all_children()
    #sample in some other single reports
    assert(union_find.processed)
    samples = random.sample(get_bug_ids(conn, union_find.project_name), len(from_dup))
    
    pairs = []
    count = 0
    while (count < size):
        pair = random.sample(samples, 2)
        if pair[0] == pair[1] or union_find.are_dups(pair[0], pair[1]):
            continue
        pairs += [(pair[0], pair[1]),]
        count += 1
    return pairs


def get_mislabels(union_find, bug_ids, anchor_bug_id, threshold):
    assert(threshold >= 0 and threshold <= 1)
    ret = []
    for bug_id in tqdm(bug_ids):
        if not union_find.are_dups(anchor_bug_id, bug_id):
            sim_score = get_similarity_of_pair((anchor_bug_id, bug_id),)
            if sim_score > threshold:
                ret += [bug_id]
    return ret

def similarity_score_1d(vector1, vector2):
    assert(len(vector1.shape) == 1)
    assert(len(vector2.shape) == 1)
    assert(vector1.shape[0] == vector2.shape[0])
    return np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))

def contains_only_letters_and_dots(word):
    word = word.replace("<", "")
    word = word.replace(">", "")
    pattern = re.compile(r'^[a-zA-Z0-9.$:]+$')
    return bool(pattern.match(word))

def is_java_path(word):
    word = word.strip(string.punctuation)
    is_long = len(word) >= 10
    has_dots = word.count('.') >= 1
    separeted_dots = word.count("..") == 0
    return is_long and has_dots and separeted_dots and contains_only_letters_and_dots(word)

def contains_java_path(line):
    for word in line.split(" "):
        if is_java_path(word): return True
    return False

def java_path_is_majority(line):
    total_java_path_length = 0
    for word in line.split(" "):
        if is_java_path(word): total_java_path_length += len(word)
    return total_java_path_length / len(line) > 0.7

def starts_with_java_path(line):
    first_word = line.split(" ")[0]
    return is_java_path(first_word)

def startswith_datetime(line):
    first_word = line.split(" ")[0]
    match1 = re.match(r"\d+-\d+-\d+", first_word) is not None
    match2 = re.match(r"\d+:\d+:\d+.*", first_word) is not None
    match3 = re.match(r"\d+/\d+/\d+.*", first_word) is not None
    return match1 or match2 or match3

def startswith_allcaps(line):
    # first_word = line.split(" ")[0]
    # return first_word.isalpha() and len(first_word) > 2 and first_word.isupper()
    return line.startswith("INFO ")\
        or line.startswith("ERROR ")\
        or line.startswith("ERRORS ")\
        or line.startswith("WARN ")\
        or line.startswith("WARNING ")\
        or line.startswith("WARNINGS ")\
            
def startswith_label(line):
    first_word = line.split(" ")[0]
    pattern = re.compile(r'^\[[a-zA-Z]+\]$')

    # Check if the input string matches the pattern
    match = pattern.match(first_word)

    return bool(match)
    
def get_tag_name(line):
    # tag is something like this <a> <\a>
    pattern = re.compile(r'^\s*<([a-zA-Z][^\s>]*)\s*[^>]*>')
    match = pattern.match(line)
    return match.group(1) if match else None

def is_stacktrace_more(line):
    match = re.match(r"... \d+ more", line.strip()) is not None
    return match

def is_stacktrace(line):
    line = line.strip()
    return len(line) != 0\
            and (line.startswith("at ")\
            or line.startswith("Caused by: ")\
            or line.startswith("Exception in thread ")\
            or java_path_is_majority(line)\
            or startswith_datetime(line)\
            or startswith_allcaps(line)\
            or startswith_label(line)\
            or is_stacktrace_more(line))

def segregate_log_and_stacktrace(text):
    eng = ""
    log_and_stacktrace = ""
    
    for line in text.split("\n"):
        if get_tag_name(line) is not None:
            eng += line + "\n"
            continue
        if is_stacktrace(line):
            log_and_stacktrace += line + "\n"
        else:
            eng += line + "\n"
    return eng.strip(), log_and_stacktrace.strip()


def has_log_or_stacktrace(text):
    for line in text.split("\n"):
        if get_tag_name(line) is not None:
            continue
        if is_stacktrace(line):
            return True
    return False

def tokenize_stuctured_text(text):
    pattern = r'\s+|[-|.,;!?()"\[\]{}:]+'
    words = re.split(pattern, text)
    words = [word for word in words if word]
    return words

def filter_numeric(words):
    return [word for word in words if word.isdigit()]
        
       
def write_string_to_file(file_path, content):
    try:
        # Open the file in write mode ('w')
        with open(file_path, 'w') as file:
            # Write the content to the file
            file.write(content)
        # print(f"String has been written to {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
def read_string_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def create_folder(folder_path):
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            # If it doesn't exist, create the folder
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")
        else:
            print(f"Folder '{folder_path}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_file(file_path):
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            # If it exists, delete the file
            os.remove(file_path)
            print(f"File '{file_path}' deleted.")
        else:
            print(f"File '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")