import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

import sqlite3
import argparse
from tqdm import tqdm
import csv
import json

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configuration = RobertaConfig(vocab_size=50265)
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
# model = RobertaModel(config=configuration).from_pretrained("microsoft/codebert-base")
# model.to(device)

column_names = ['bug_id',
                'key',
                'creation_ts',
                'short_desc',
                'product',
                'component',
                'version',
                'bug_status',
                'resolution',
                'priority',
                'bug_severity',
                'description',
                'dup_id']

column_enum = enumerate(column_names)

frequency_map = {}

def get_table_names(conn):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Extract table names from the result
    table_names = [table[0] for table in tables]
    return table_names

def description_iterator(conn, project_name, f, limit=None):
    cursor = conn.cursor()
    if limit is None:
        cursor.execute(f"SELECT * FROM {project_name};")
    else:
        cursor.execute(f"SELECT * FROM {project_name} LIMIT {limit};")
        
    rows = cursor.fetchall()
    
    for row in tqdm(rows):
        f(cursor, row)
        
def get_descriptions(cursor, row):
    desc = row[column_names.index("description")]
    short_desc = row[column_names.index("short_desc")]
    return desc + "\n" + short_desc
    
def vectorize(cursor, row):
    description = get_descriptions(cursor, row)
    tokens = tokenizer.tokenize(description)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Number of tokens", len(token_ids))
    context_embedding = model(torch.tensor(token_ids[:512])[None, :])[0]
    # print(context_embedding)
    return context_embedding

def update_frequency(cursor, row):
    words = get_descriptions(cursor, row).lower().split(" ")
    for word in words:
        if word in frequency_map.keys():
            frequency_map[word] += 1
        else:
            frequency_map[word] = 1
    


def main():
    parser = argparse.ArgumentParser(description="Example of handling command-line arguments.")
    
    parser.add_argument('database', type=str, help='path to database')
    parser.add_argument('project_name', type=str, help='project name')
    
    args = parser.parse_args()
    
    # connect to database
    conn = sqlite3.connect(args.database)

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    
    table_names = get_table_names(conn)
    if (args.project_name not in table_names):
        print("Project {} is not in the list of projects: \n{}".format(args.project_name, table_names))
        exit(1)
        
    
    description_iterator(conn, args.project_name, update_frequency, None)
    
    deleting_keys = []
    for key in frequency_map.keys():
        if frequency_map[key] < 5:
            deleting_keys.append(key)
    
    for key in deleting_keys:
        del frequency_map[key]
        
    sorted_frequency_map = dict(sorted(frequency_map.items(), key=lambda item: item[1]))
        
    with open("{}.json".format(args.project_name), 'w') as file:
        json.dump(sorted_frequency_map, file)
    
    
    # ending program
    conn.close()

if __name__ == "__main__":
    main()