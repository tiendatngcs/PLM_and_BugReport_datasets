from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoConfig
# from transformers import *
import torch
import sqlite3
from tqdm import tqdm


tech_term_labels = [
    "B-Algorithm",
    "B-Application",
    "B-Class",
    "B-Code_Block",
    "B-Data_Structure",
    "B-Data_Type",
    "B-Device",
    "B-Error_Name",
    "B-File_Name",
    "B-File_Type",
    "B-Function",
    "B-HTML_XML_Tag",
    "B-Keyboard_IP",
    "B-Language",
    "B-Library",
    "B-Licence",
    "B-Operating_System",
    "B-Organization",
    "B-Output_Block",
    "B-User_Interface_Element",
    "B-User_Name",
    "B-Value",
    "B-Variable",
    "B-Version",
    "B-Website",
    "I-Algorithm",
    "I-Application",
    "I-Class",
    "I-Code_Block",
    "I-Data_Structure",
    "I-Data_Type",
    "I-Device",
    "I-Error_Name",
    "I-File_Name",
    "I-File_Type",
    "I-Function",
    "I-HTML_XML_Tag",
    "I-Keyboard_IP",
    "I-Language",
    "I-Library",
    "I-Licence",
    "I-Operating_System",
    "I-Output_Block",
    "I-User_Interface_Element",
    "I-User_Name",
    "I-Value",
    "I-Variable",
    "I-Version",
    "I-Website",
]

def is_table_exist(cursor, table_name):
    # Check if the table exists in the database
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    result = cursor.fetchone()

    # Return True if the table exists, False otherwise
    return result is not None

def get_row_count(cursor, table_name):
    # Execute a query to get the number of rows in the specified table
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    row_count = cursor.fetchone()[0]
    return row_count

def delete_table(cursor, table_name):
    # Execute a query to drop the specified table
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

def get_tech_terms(nerpipeline, text):
    ret = ""
    last_end = 0
    prediction = nerpipeline(text)
    for entry in prediction:
        if entry["entity"] in tech_term_labels:
            if entry["start"] != last_end and last_end != 0:
                ret += " "
            ret += text[entry["start"]:entry["end"]]
            last_end = entry["end"]
    return ret

def create_table_in_new_db(db1_cursor, db2_cursor, table_name):
    db1_cursor.execute(f"PRAGMA table_info({table_name});")
    columns_info = db1_cursor.fetchall()
    # print(columns_info)
    columns_info += [(11, 'code_feature', 'TEXT', 0, None, 0)]
    
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{column[1]} {column[2]}' for column in columns_info])});"
    db2_cursor.execute(query)

def main():
    nerpipeline = pipeline(model="mrm8488/codebert-base-finetuned-stackoverflow-ner")
    
    db1_connection = sqlite3.connect("dbrd.db")

    # Create a cursor object to interact with the database
    db1_cursor = db1_connection.cursor()
    
    
    db2_connection = sqlite3.connect("dbrd_w_tech_terms.db")

    # Create a cursor object to interact with the database
    db2_cursor = db2_connection.cursor()
    
    # db1_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # tables = db1_cursor.fetchall()
    
    tables = [
    "spark",
    "eclipse",
    "eclipse_initial",
    "eclipse_old",
    "hadoop",
    "hadoop_1day",
    "hadoop_old",
    "kibana",
    "mozilla",
    "mozilla_initial",
    "mozilla_old",
    "spark_1day",
    "vscode",]
    
    for table in tables:
        # table_name = table[0]
        table_name = table
        print(table_name)
        
        if (is_table_exist(db2_cursor, table_name)):
            if (get_row_count(db1_cursor, table_name) != get_row_count(db2_cursor, table_name)):
                delete_table(db2_cursor, table_name)
            else:
                # table is there, continue to differente table
                continue
            
            
        
        db1_cursor.execute(f"PRAGMA table_info({table_name});")
        columns = db1_cursor.fetchall()
        # Extract and print the column names
        column_names = [column[1] for column in columns]

        # Fetch all rows from the current table in Database 1
        db1_cursor.execute(f"SELECT * FROM {table_name};")
        rows = db1_cursor.fetchall()
        
        # create new table if not exist
        create_table_in_new_db(db1_cursor, db2_cursor, table_name)
        
        for row in tqdm(rows):
            text = row[column_names.index("short_desc")] + "\n" + row[column_names.index("description")]
            tech_term = get_tech_terms(nerpipeline, text)
            # print(tech_term)
            new_row = row + (tech_term,)
            query = f"INSERT INTO {table_name} VALUES ({', '.join(['?']*len(new_row))});"
            db2_cursor.execute(query, new_row)  # Assuming the first column is an auto-incremented ID
        
        db2_connection.commit()
            

        # Create the corresponding table in Database 2
        # Note: You may need to adjust the table schema based on your requirements

        # # Insert each row into the corresponding table in Database 2
        # for row in rows:
        #     db2_cursor.execute(f"INSERT INTO {table_name} VALUES (NULL, ?, ?);", row[1:])  # Assuming the first column is an auto-incremented ID
            
    db1_connection.commit()
    db1_connection.close()

    db2_connection.commit()
    db2_connection.close()
    
    
    

if __name__ == "__main__":
    # This block will be executed only if the script is run as the main program
    main()
