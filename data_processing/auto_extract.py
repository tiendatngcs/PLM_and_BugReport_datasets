import os
import my_utils
from tqdm import tqdm
import sqlite3
from ComponentAutoExtractor import ComponentAutoExtractor

home_path = "/home/grads/t/tiendat.ng.cs/github_repos/PLM_and_BugReport_datasets"
data_path = os.path.join(home_path, "datasets", "hand-gen-datasets")

# connect to db
database_path = os.path.join(home_path, "dbrd_processed.db")
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# process db table, create save folder
table = "spark"
save_path = os.path.join(data_path, table)
my_utils.create_folder(save_path)

union_find = my_utils.UnionFind()
union_find.process_project(conn, table, min_desc_length=10)
bug_ids = my_utils.get_bug_ids(conn, table)
bug_ids_w_duplicates = union_find.get_all_children()

# loop through each desc extract components, and save to file
print("Number of bug_ids before filter: ", len(bug_ids))
# remove bug_ids that are of very short desc and those that does not have log
to_remove_ids = []
for bug_id in bug_ids:
    desc = my_utils.get_description(conn, table, bug_id)
    short_desc = my_utils.get_short_desc(conn, table, bug_id)
    auto_extractor = ComponentAutoExtractor(desc)
    if len(desc) < 50 or not auto_extractor.has_log() or bug_id not in bug_ids_w_duplicates:
        to_remove_ids.append(bug_id)

for to_remove_id in to_remove_ids:
    bug_ids.remove(to_remove_id)

print("Number of bug_ids after filter: ", len(bug_ids))
print("Do you wish to continue? (y/n)", end=": ")
ans = input()

if ans.lower() == "n":
    exit(0)

desc_path = os.path.join(save_path, "desc")
eng_path = os.path.join(save_path, "eng")
log_path = os.path.join(save_path, "log")

my_utils.create_folder(desc_path)
my_utils.create_folder(eng_path)
my_utils.create_folder(log_path)

search_space = bug_ids.copy()
for bug_id in tqdm(bug_ids):
    dups = union_find.get_children(bug_id)
    for dup in dups:
        if dup != bug_id and dup not in search_space:
            search_space.append(dup)
            
for bug_id in tqdm(search_space):
    file_name = f"{str(bug_id)}.txt"

    desc_file_path = os.path.join(desc_path, file_name)
    eng_file_path = os.path.join(eng_path, file_name)
    log_file_path = os.path.join(log_path, file_name)

    desc = my_utils.get_description(conn, table, bug_id)
    # short_desc = my_utils.get_short_desc(conn, table, bug_id)
    # we sparate into eng, code, and log
    autoExtractor = ComponentAutoExtractor(desc)
    log, remain = autoExtractor.extract_log()

    if not os.path.exists(desc_file_path): my_utils.write_string_to_file(desc_file_path, desc)
    if not os.path.exists(eng_file_path): my_utils.write_string_to_file(eng_file_path, remain)
    if not os.path.exists(log_file_path): my_utils.write_string_to_file(log_file_path, log)

