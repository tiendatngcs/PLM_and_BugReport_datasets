import os
import tkinter as tk
from tkinter import scrolledtext, StringVar, ttk

class FileViewerApp:
    def __init__(self, master, folder_paths):
        self.master = master
        self.desc_folder_path           = folder_paths[0]
        self.eng_folder_path            = folder_paths[1]
        self.logStackTrace_folder_path  = folder_paths[2]
        
        self.files_list = self.get_files_list()
        self.current_index = 0
        
        self.selected_file_var = StringVar()

        self.create_widgets()

    def create_widgets(self):
        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_rowconfigure(2, weight=0)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)
        
        
        # Create the first scrolled text area
        self.text_area1 = scrolledtext.ScrolledText(self.master, wrap=tk.WORD)
        self.text_area1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Create the second scrolled text area
        self.text_area2 = scrolledtext.ScrolledText(self.master, wrap=tk.WORD)
        self.text_area2.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Create the third scrolled text area
        self.text_area3 = scrolledtext.ScrolledText(self.master, wrap=tk.WORD)
        self.text_area3.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

        # Set column weights for text areas
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

        self.prev_button = tk.Button(self.master, text="Previous", command=self.show_previous_file)
        self.prev_button.grid(row=2, column=0, padx=5, pady=5)

        self.next_button = tk.Button(self.master, text="Next", command=self.show_next_file)
        self.next_button.grid(row=2, column=1, padx=5, pady=5)
        
        self.file_dropdown = ttk.Combobox(self.master, textvariable=self.selected_file_var, values=self.files_list)
        self.file_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.file_dropdown.bind("<<ComboboxSelected>>", self.load_selected_file)

        self.show_current_file()

    def get_files_list(self):
        desc_files = [f for f in os.listdir(self.desc_folder_path) if os.path.isfile(os.path.join(self.desc_folder_path, f))]
        # eng_files               = [f for f in os.listdir(self.eng_folder_path)              if os.path.isfile(os.path.join(self.eng_folder_path, f))]
        # logStackTrace_files     = [f for f in os.listdir(self.logStackTrace_folder_path)    if os.path.isfile(os.path.join(self.logStackTrace_folder_path, f))]
        return desc_files

    def show_current_file(self):
        assert(self.files_list)

        current_file = self.files_list[self.current_index]

        desc_file_path = os.path.join(self.desc_folder_path, current_file)
        eng_file_path = os.path.join(self.eng_folder_path, current_file)
        logStackTrace_file_path = os.path.join(self.logStackTrace_folder_path, current_file)

        with open(desc_file_path, 'r') as file:
            desc_file_content = file.read()
            
        with open(eng_file_path, 'r') as file:
            eng_file_content = file.read()

        with open(logStackTrace_file_path, 'r') as file:
            logStackTrace_file_content = file.read()

        self.text_area1.delete(1.0, tk.END)
        self.text_area1.insert(tk.INSERT, desc_file_content)

        self.text_area2.delete(1.0, tk.END)
        self.text_area2.insert(tk.INSERT, eng_file_content)

        self.text_area3.delete(1.0, tk.END)
        self.text_area3.insert(tk.INSERT, logStackTrace_file_content)
        
        self.file_dropdown['values'] = self.files_list
        self.selected_file_var.set(current_file)

    def show_previous_file(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_file()

    def show_next_file(self):
        if self.current_index < len(self.files_list) - 1:
            self.current_index += 1
            self.show_current_file()
        
    def load_selected_file(self, event):
        selected_file = self.selected_file_var.get()

        if selected_file in self.files_list:
            self.current_index = self.files_list.index(selected_file)
            self.show_current_file()

if __name__ == "__main__":
    
    project_name = "./hadoop_old"
    
    desc_folder_name = "desc"
    eng_folder_name = "eng"
    logStackTrace_folder_name = "logStackTrace"
    
    desc_folder_path = os.path.join(project_name, desc_folder_name)
    eng_folder_path = os.path.join(project_name, eng_folder_name)
    logStackTrace_folder_path = os.path.join(project_name, logStackTrace_folder_name)
    
    # folder_path = "your_folder_path_here"  # Replace with the path to your folder

    root = tk.Tk()
    root.title("File Viewer")

    app = FileViewerApp(root, (desc_folder_path, eng_folder_path, logStackTrace_folder_path))

    root.mainloop()
