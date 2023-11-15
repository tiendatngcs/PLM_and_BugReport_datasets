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
            
    def process_project(self, conn, project_name, column_names):
        cursor = conn.cursor()
        self.project_name = project_name
        
        cursor.execute(f"SELECT * FROM {project_name}")
        print("Processing", project_name)
        for row in tqdm(cursor.fetchall()):
            dup_id = int(row[column_names.index("dup_id")])
            if dup_id == -1: continue
            bug_id = int(row[column_names.index("bug_id")])
            if (dup_id == bug_id): continue
            assert(dup_id != bug_id)
            self.union(bug_id, dup_id)
        self.processed = True
            
    def get_roots(self,):
        assert(self.processed)
        return list(set(self.parent.values()))
    
    def get_children(self, parent):
        assert(self.processed)
        parent = self.find(parent)
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
        
            
            
def get_bug_ids(conn, table_name):
    cursor = conn.cursor()
    column_name = "bug_id"

    # Fetch table names using SQL query
    cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name};")
    distinct_values_sorted = cursor.fetchall()

    # Extract table names from the result
    return [value[0] for value in distinct_values_sorted]


def get_column_names(conn, table_name):
    cursor = conn.cursor()

    # Execute a query to get information about the columns in the specified table
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns_info = cursor.fetchall()

    # Extract and return the column names
    column_names = [column[1] for column in columns_info]
    return column_names


def get_code_feature(conn, project_name, bug_id):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    query = f"SELECT * FROM {project_name} WHERE bug_id = {bug_id};"
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()[0]
    return result[column_names.index("code_feature")]


def get_descriptions(conn, project_name, bug_id):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    query = f"SELECT * FROM {project_name} WHERE bug_id = {bug_id};"
    column_names = get_column_names(conn, project_name)
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()[0]
    desc = result[column_names.index("description")]
    short_desc = result[column_names.index("short_desc")]

    # Extract table names from the result
    return (desc + " \n " + short_desc).replace("\\'", "'")

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

