import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import sqlite3
import argparse
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configuration = RobertaConfig(vocab_size=50265)
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = RobertaModel(config=configuration).from_pretrained("microsoft/codebert-base")
model = GPT2LMHeadModel.from_pretrained('gpt2')
word_embeddings = model.transformer.wte.weight

model.to(device)

chunk_size = 8
stride = 4

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
                'code_feature',
                'dup_id']

selected_bug_ids = [13129687, 13130758, 13131323, 13131443, 13131871, 13132016,
       13132089, 13132434, 13132436, 13132563, 13132882, 13133126,
       13134870, 13134871, 13135081, 13135082, 13137759, 13138115,
       13138216, 13138646, 13139260, 13140553, 13141684, 13142238,
       13142715, 13143053, 13145024, 13145952, 13146640, 13148782,
       13150721, 13150938, 13154038, 13155149, 13155360, 13155447,
       13156462, 13156549, 13156550, 13158097, 13158838, 13159154,
       13159445, 13160548, 13161249, 13161729, 13162690, 13162978,
       13164578, 13164594, 13165673, 13165716, 13165725, 13165821,
       13168172, 13169315, 13169448, 13171485, 13173832, 13174312,
       13174413, 13175312, 13176306, 13176924, 13176987, 13177881,
       13178438, 13178960, 13179431, 13179511, 13179580, 13179699,
       13180148, 13180150, 13181141, 13181392, 13181673, 13181674,
       13182573, 13183205, 13185089, 13186110, 13186111, 13186322,
       13186457, 13186527, 13186582, 13186636, 13187134, 13187243,
       13187733, 13188289, 13188839, 13190381, 13190592, 13190887,
       13191728, 13192427, 13192953, 13194148, 13194295, 13195681,
       13196338, 13196587, 13197919, 13198089, 13199062, 13199165,
       13200031, 13200176, 13200928, 13201105, 13201168, 13201616,
       13202115, 13202574, 13202790, 13203190, 13203526, 13203876,
       13205035, 13205219, 13206281, 13207747, 13207778, 13208759,
       13208798, 13208883, 13208907, 13209812, 13210226, 13210227,
       13211240, 13211249, 13211667, 13211901, 13212214, 13212402,
       13212576, 13212742, 13212902, 13216379, 13217285, 13217321,
       13217547, 13218022, 13218119, 13218898, 13218915, 13219044,
       13219232, 13219569, 13220487, 13220600, 13221477, 13221922,
       13222836, 13223257, 13223505, 13223507, 13223523, 13225603,
       13227033, 13227226, 13229219, 13229612, 13231643, 13232289,
       13232348, 13232451, 13233199, 13233216, 13234738, 13235056,
       13238901, 13238987, 13239398, 13239649, 13239899, 13239970,
       13240527, 13240535, 13241408, 13242507, 13242617, 13243105,
       13243106, 13243660, 13244218, 13244246, 13244522, 13245055,
       13245937, 13246226, 13246720, 13248760, 13250660, 13251257,
       13252148, 13252230, 13252269, 13253179, 13253192, 13253314,
       13253410, 13253656, 13253802, 13253844, 13253932, 13254305,
       13254309, 13254706, 13254880, 13254983, 13256278, 13256940,
       13257285, 13258051, 13259794, 13259948, 13260937, 13261082,
       13261483, 13261902, 13261912, 13261919, 13262206, 13262521,
       13264046, 13264162, 13264594, 13265500, 13265662, 13267009,
       13267164, 13267592, 13268090, 13268489, 13268632, 13269720,
       13270076, 13272259, 13272652, 13273274, 13273307, 13273825,
       13274220, 13274297, 13274353, 13275259, 13276210, 13278427,
       13285808, 13285890, 13286104, 13287805, 13288309, 13288450,
       13290159, 13291030, 13293049, 13295946, 13296229, 13297297,
       13298863, 13299388, 13300231, 13301431, 13301439, 13303349,
       13307904, 13307909, 13307933, 13307934, 13309781, 13309875,
       13311700, 13311794, 13312407, 13312547, 13312548, 13312600,
       13312653, 13313797, 13314319, 13314731, 13315082, 13315769,
       13318244, 13318249, 13320267, 13320353, 13322032, 13322388,
       13323725, 13324339, 13324538, 13325444, 13325445, 13326231,
       13326338, 13326903, 13328229, 13329042, 13330656, 13334539,
       13337329, 13337686, 13339536, 13344267, 13344728, 13346504,
       13346505]

column_enum = enumerate(column_names)

def get_table_names(conn):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Extract table names from the result
    table_names = [table[0] for table in tables]
    return table_names

def description_iterator(conn, project_name, limit=None):
    cursor = conn.cursor()
    if limit is None:
        cursor.execute(f"SELECT * FROM {project_name};")
    else:
        cursor.execute(f"SELECT * FROM {project_name} LIMIT {limit};")
        
    rows = cursor.fetchall()
    folder_name = os.path.join("./vectorize_with_start_token", project_name)
    

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
    for row in tqdm(rows):
        bug_id = row[column_names.index("bug_id")]
        # description = get_descriptions(cursor, row)
        # tokens = tokenizer.tokenize(description)
        code_feature = get_code_feature(cursor, row)
        # print(code_feature)
        tokens = tokenizer.tokenize(code_feature)
        # if len og token array is < 32, we do nothing as there is not enough information
        if (len(tokens) < chunk_size // 2):
            continue
        
        # remember to add cls and sep token at each chunk
        token_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] +[tokenizer.sep_token]+tokens+[tokenizer.eos_token])
        # token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # divide token ids into batche of chunks
        chunk_list=[]
        for i in range(0, len(token_ids), stride):
            chunk = token_ids[i:min(i+chunk_size, len(token_ids))]
            assert(len(chunk) <= chunk_size)
            if len(chunk) < chunk_size:
                continue
                # if (len(chunk) < chunk_size // 2): continue
                # pad_length = chunk_size - len(chunk)
                # chunk += [tokenizer.pad_token_id]*pad_length
            assert(len(chunk) == chunk_size)
            # print(chunk)
            chunk_list.append(chunk)

        if(len(chunk_list) == 0): continue
        chunk_arr = np.array(chunk_list)
        # print("Chunk arr size{}".format(chunk_arr.shape))
        # context_embedding = model(torch.tensor(token_ids[:512])[None, :])[0]
        context_embedding = model(torch.tensor(chunk_arr)[:, :])[0]
        # print(context_embedding.size())
        
        save_file_name = "{}_{}.npz".format(project_name, bug_id)
        save_path = os.path.join(folder_name, save_file_name)
        np.savez_compressed(save_path, context_embedding.detach().numpy())
        
        
# def process_selected_bug_ids(conn, project_name, chunk_size, stride, with_padding):
#     cursor = conn.cursor()
#     # if limit is None:
#     #     cursor.execute(f"SELECT * FROM {project_name};")
#     # else:
#     #     cursor.execute(f"SELECT * FROM {project_name} LIMIT {limit};")
        
#     # rows = cursor.fetchall()
#     folder_name = os.path.join(f"./vectorize_with_start_token_{chunk_size}_{stride}_{int(with_padding)}", project_name)
    

#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#         print(f"Folder '{folder_name}' created.")
#     else:
#         print(f"Folder '{folder_name}' already exists.")
    
#     for bug_id in tqdm(selected_bug_ids):
#         # bug_id = row[column_names.index("bug_id")]
#         # description = get_descriptions(cursor, row)
#         # tokens = tokenizer.tokenize(description)
#         code_feature = get_code_feature(conn, project_name, bug_id)
        
#         # print(code_feature)
#         tokens = tokenizer.tokenize(code_feature.lower())
#         tokens_file_name = f'tokens_{project_name}_{bug_id}.txt'
#         tokens_file_path = os.path.join(folder_name, tokens_file_name)
#         with open(tokens_file_path, 'w') as file:
#             # Write the string to the file
#             for token in tokens:
#                 file.write(token + "\n")
#         # if len og token array is < 32, we do nothing as there is not enough information
#         if (len(tokens) < chunk_size // 2):
#             continue
        
#         # remember to add cls and sep token at each chunk
#         token_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token]+[tokenizer.sep_token]+tokens+[tokenizer.sep_token])
#         # token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
#         # divide token ids into batche of chunks
#         chunk_list=[]
#         for i in range(0, len(token_ids), stride):
#             chunk = token_ids[i:min(i+chunk_size, len(token_ids))]
#             assert(len(chunk) <= chunk_size)
#             if not with_padding and len(chunk) < chunk_size:
#                 continue
#             else:
#                 pad_length = chunk_size - len(chunk)
#                 chunk += [tokenizer.pad_token_id]*pad_length
#             assert(len(chunk) == chunk_size)
#             # print(chunk)
#             chunk_list.append(chunk)

#         if(len(chunk_list) == 0): continue
#         chunk_arr = np.array(chunk_list)
#         # print("Chunk arr size{}".format(chunk_arr.shape))
#         # context_embedding = model(torch.tensor(token_ids[:512])[None, :])[0]
#         context_embedding = model(torch.tensor(chunk_arr)[:, :])[0]
#         # print(context_embedding.size())
        
#         save_file_name = "{}_{}.npz".format(project_name, bug_id)
#         save_path = os.path.join(folder_name, save_file_name)
#         np.savez_compressed(save_path, context_embedding.detach().numpy())

def process_selected_bug_ids_gpt2(conn, project_name, chunk_size, stride, with_padding):
    cursor = conn.cursor()
    # if limit is None:
    #     cursor.execute(f"SELECT * FROM {project_name};")
    # else:
    #     cursor.execute(f"SELECT * FROM {project_name} LIMIT {limit};")
        
    # rows = cursor.fetchall()
    folder_name = os.path.join(f"./vectorize_with_start_token_{chunk_size}_{stride}_{int(with_padding)}", project_name)
    

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
    for bug_id in tqdm(selected_bug_ids):
        # bug_id = row[column_names.index("bug_id")]
        # description = get_descriptions(cursor, row)
        # tokens = tokenizer.tokenize(description)
        code_feature = get_code_feature(conn, project_name, bug_id)
        
        # print(code_feature)
        # tokens = tokenizer.tokenize(code_feature.lower())
        text_index = tokenizer.encode(code_feature, add_prefix_space=True)
        vector = model.transformer.wte.weight[text_index,:]
        print(f"vector dim {np.array(vector).shape}")
        tokens_file_name = f'tokens_{project_name}_{bug_id}.txt'
        tokens_file_path = os.path.join(folder_name, tokens_file_name)
        with open(tokens_file_path, 'w') as file:
            # Write the string to the file
            for token in tokens:
                file.write(token + "\n")
        # if len og token array is < 32, we do nothing as there is not enough information
        if (len(tokens) < chunk_size // 2):
            continue
        
        # remember to add cls and sep token at each chunk
        # token_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token]+[tokenizer.sep_token]+tokens+[tokenizer.sep_token])
        # token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # divide token ids into batche of chunks
        chunk_list=[]
        for i in range(0, len(token_ids), stride):
            chunk = token_ids[i:min(i+chunk_size, len(token_ids))]
            assert(len(chunk) <= chunk_size)
            if not with_padding and len(chunk) < chunk_size:
                continue
            else:
                pad_length = chunk_size - len(chunk)
                chunk += [tokenizer.pad_token_id]*pad_length
            assert(len(chunk) == chunk_size)
            # print(chunk)
            chunk_list.append(chunk)

        if(len(chunk_list) == 0): continue
        chunk_arr = np.array(chunk_list)
        # print("Chunk arr size{}".format(chunk_arr.shape))
        # context_embedding = model(torch.tensor(token_ids[:512])[None, :])[0]
        context_embedding = model(torch.tensor(chunk_arr)[:, :])[0]
        # print(context_embedding.size())
        
        save_file_name = "{}_{}.npz".format(project_name, bug_id)
        save_path = os.path.join(folder_name, save_file_name)
        np.savez_compressed(save_path, context_embedding.detach().numpy())
        
def get_token_length_stats(conn, project_name, limit=None):
    
    cursor = conn.cursor()
    if limit is None:
        cursor.execute(f"SELECT * FROM {project_name};")
    else:
        cursor.execute(f"SELECT * FROM {project_name} LIMIT {limit};")
        
    rows = cursor.fetchall()
    
    max_token_length = 0
    total_token_length = 0
    total_rows = len(rows)
    token_length_histogram = np.zeros(300, dtype=int)
    for row in tqdm(rows):
        desc = get_descriptions(cursor, row)
        tokens = tokenizer.tokenize(desc)
        max_token_length = max(max_token_length, len(tokens))
        total_token_length += len(tokens)
        histogram_index = min(len(tokens)//10, 299)
        assert(histogram_index < len(token_length_histogram))
        token_length_histogram[histogram_index] += 1
        
    print("Max desc length in project {} is {}".format(project_name, max_token_length))
    print("Average desc length in project {} is {}".format(project_name, total_token_length/total_rows))
    # plt.bar(range(0, len(token_length_histogram)*100, 100), token_length_histogram)
    # plt.show()
    print("length histogram {}".format(token_length_histogram))
    
        
def get_descriptions(cursor, row):
    desc = row[column_names.index("description")]
    short_desc = row[column_names.index("short_desc")]
    return (desc + "\n" + short_desc).replace("\\'", "'")

def get_code_feature(cursor, row):
    code_feature = row[column_names.index("code_feature")]
    return code_feature

def get_code_feature(conn, project_name, bug_id):
    cursor = conn.cursor()

    # Fetch table names using SQL query
    query = f"SELECT * FROM {project_name} WHERE bug_id = {bug_id};"
    # print(query)
    cursor.execute(query)
    result = cursor.fetchall()[0]
    code_feature = result[column_names.index("code_feature")]

    # Extract table names from the result
    return code_feature
    
    
# def vectorize(cursor, row):
#     description = get_descriptions(cursor, row)
#     tokens = tokenizer.tokenize(description)
#     # if len og token array is < 32, we do nothing as there is not enough information
#     if (len(tokens) < chunk_size // 4):
#         return
    
#     # remember to add cls and sep token at each chunk
#     token_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token]+tokens+[tokenizer.sep_token])
    
#     # divide token ids into batche of chunks
#     chunk_list=[]
#     for i in range(0, len(token_ids), stride):
#         chunk = token_ids[i:min(i+chunk_size, len(token_ids))]
#         assert(len(chunk) <= chunk_size)
#         if len(chunk) < chunk_size:
#             if (len(chunk) < chunk_size // 2): continue
#             pad_length = chunk_size - len(chunk)
#             chunk += [tokenizer.pad_token_id]*pad_length
#         assert(len(chunk) == chunk_size)
#         # print(chunk)
#         chunk_list.append(chunk)
    
#     chunk_arr = np.array(chunk_list)
#     print("Chunk arr size{}".format(chunk_arr.shape))
#     # context_embedding = model(torch.tensor(token_ids[:512])[None, :])[0]
#     context_embedding = model(torch.tensor(chunk_arr)[:, :])[0]
#     print(context_embedding.size())
#     context_embedding.numpy()
#     np.savez("{}_{}.npz".format(project_name, bug_id))
#     return context_embedding



def main():
    parser = argparse.ArgumentParser(description="Example of handling command-line arguments.")
    
    parser.add_argument('database', type=str, help='path to database')
    parser.add_argument('project_name', type=str, help='project name')
    parser.add_argument('chunk_size', type=int, help='chunk size')
    parser.add_argument('stride', type=int, help='stride')
    parser.add_argument('with_padding', type=bool, nargs='?', const=False, help='include tail and add padding')
    
    args = parser.parse_args()
    
    # connect to database
    conn = sqlite3.connect(args.database)

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    project_name = args.project_name
    
    chunk_size = args.chunk_size
    stride = args.stride
    with_padding = args.with_padding
    
    table_names = get_table_names(conn)
    if (project_name not in table_names or project_name == "all"):
        print("Project {} is not in the list of projects: \n{}".format(project_name, table_names))
        exit(1)
        
    
    # description_iterator(conn, project_name, get_max_token_length, None)
    if project_name == "all":
        for p_name in table_names:
            # get_token_length_stats(conn, p_name, None)
            description_iterator(conn, p_name, None)
    else:
        # get_token_length_stats(conn, project_name, None)
        # description_iterator(conn, project_name, None)
        process_selected_bug_ids(conn, project_name, chunk_size, stride, with_padding)
    
    # bert for longer text
    # https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html
    # eventually, we want to achieve Meta NCS https://dl.acm.org/doi/pdf/10.1145/3211346.3211353 or similar
    # analysis of NCS approaches https://dl.acm.org/doi/pdf/10.1145/3338906.3340458
    # meta post about NCS https://ai.meta.com/blog/neural-code-search-ml-based-code-search-using-natural-language-queries/
    
    # after tokenizing the text and converted to ids, id arrays are divided into chunks of size 128.
    # Tail chunk that is smaller than 32 are cut off.
    # id arrays that has less then 32 tokens are not used for comparison for they are too small
    
    
    
    
    
    # ending program
    conn.close()

if __name__ == "__main__":
    main()