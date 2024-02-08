import os
import tkinter as tk
from tkinter import scrolledtext, StringVar, ttk

class FileViewerApp:
    def __init__(self, master, folder_paths):
        self.master = master
        self.desc_folder_path           = folder_paths[0]
        self.eng_folder_path            = folder_paths[1]
        self.log_folder_path  = folder_paths[2]
        self.progress_file_path = "./log_extractor_progress.txt"
        
        self.files_list = self.get_files_list()
        self.current_index = self.load_current_index()
        
        self.selected_file_var = StringVar()

        self.create_widgets()

    def create_widgets(self):
        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_rowconfigure(1, weight=0)
        self.master.grid_rowconfigure(2, weight=1)
        self.master.grid_rowconfigure(3, weight=0)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)
        
        
        # Create the first scrolled text area
        self.text_area1 = scrolledtext.ScrolledText(self.master, wrap=tk.WORD)
        self.text_area1.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Create the second scrolled text area
        self.text_area2 = scrolledtext.ScrolledText(self.master, wrap=tk.WORD)
        self.text_area2.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        # Create the third scrolled text area
        self.text_area3 = scrolledtext.ScrolledText(self.master, wrap=tk.WORD)
        self.text_area3.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")

        # Set column weights for text areas
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

        self.prev_button = tk.Button(self.master, text="Previous", command=self.show_previous_file)
        self.prev_button.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        self.next_button = tk.Button(self.master, text="Next", command=self.show_next_file)
        self.next_button.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        
        self.reset_button = tk.Button(self.master, text="Reset", command=self.reset)
        self.reset_button.grid(row=3, column=0, padx=5, pady=5)
        
        self.save_button = tk.Button(self.master, text="Save", command=self.save_file)
        self.save_button.grid(row=3, column=1, padx=5, pady=5)
        
        self.file_dropdown = ttk.Combobox(self.master, textvariable=self.selected_file_var, values=self.files_list)
        self.file_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.file_dropdown.bind("<<ComboboxSelected>>", self.load_selected_file)
        
        # Create progress bar
        self.progress_bar_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.master, variable=self.progress_bar_var, mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=3, pady=5, sticky="ew")

        self.show_current_file()

         # Bind keys
        self.master.bind("n", lambda event: self.show_next_file())
        self.master.bind("p", lambda event: self.show_previous_file())
        self.master.bind("s", lambda event: self.save_file())
        self.master.bind("r", lambda event: self.reset())

    def get_files_list(self):
        desc_files = [f for f in os.listdir(self.desc_folder_path) if os.path.isfile(os.path.join(self.desc_folder_path, f))]
        # eng_files               = [f for f in os.listdir(self.eng_folder_path)              if os.path.isfile(os.path.join(self.eng_folder_path, f))]
        # log_files     = [f for f in os.listdir(self.log_folder_path)    if os.path.isfile(os.path.join(self.log_folder_path, f))]
        return desc_files

    def show_current_file(self):
        assert(self.files_list)

        current_file = self.files_list[self.current_index]

        desc_file_path = os.path.join(self.desc_folder_path, current_file)
        eng_file_path = os.path.join(self.eng_folder_path, current_file)
        log_file_path = os.path.join(self.log_folder_path, current_file)

        with open(desc_file_path, 'r') as file:
            desc_file_content = file.read()
            
        with open(eng_file_path, 'r') as file:
            eng_file_content = file.read()

        with open(log_file_path, 'r') as file:
            log_file_content = file.read()

        self.text_area1.delete(1.0, tk.END)
        self.text_area1.insert(tk.INSERT, desc_file_content)

        self.text_area2.delete(1.0, tk.END)
        self.text_area2.insert(tk.INSERT, eng_file_content)

        self.text_area3.delete(1.0, tk.END)
        self.text_area3.insert(tk.INSERT, log_file_content)
        
        self.file_dropdown['values'] = self.files_list
        self.selected_file_var.set(current_file)
        
        total_files = len(self.files_list)
        progress_value = (self.current_index + 1) / total_files * 100
        self.progress_bar_var.set(progress_value)

    def update_progress(self,):
        # if not os.path.exists(self.progress_file_path):
        with open(self.progress_file_path, 'w') as file:
            # Write the content to the file
            file.write(self.current_index)
    
    def load_current_index(self,):
        # load the number saved in progress file 
        progress = 0
        if not os.path.exists(self.progress_file_path):
            return progress
        with open(self.progress_file_path, 'r') as file:
            line = file.readline()
            if line == "":
                return 0
            progress = int(line.split()[0])
        return progress
        
    def show_previous_file(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_file()
            self.update_progress()

    def show_next_file(self):
        if self.current_index < len(self.files_list) - 1:
            self.current_index += 1
            self.show_current_file()
            self.update_progress()
            
    def reset(self):
        assert(self.files_list)

        current_file = self.files_list[self.current_index]

        desc_file_path = os.path.join(self.desc_folder_path, current_file)
        eng_file_path = os.path.join(self.eng_folder_path, current_file)
        log_file_path = os.path.join(self.log_folder_path, current_file)

        with open(desc_file_path, 'r') as file:
            desc_file_content = file.read()

        self.text_area1.delete(1.0, tk.END)
        self.text_area1.insert(tk.INSERT, desc_file_content)

        self.text_area2.delete(1.0, tk.END)
        self.text_area2.insert(tk.INSERT, desc_file_content)

        self.text_area3.delete(1.0, tk.END)
        # self.text_area3.insert(tk.INSERT, log_file_content)
        
        self.file_dropdown['values'] = self.files_list
        self.selected_file_var.set(current_file)
    
    def save_file(self):
        # save the current content of the eng and log text area to files
        eng_content = self.text_area2.get("1.0", tk.END)
        log_content = self.text_area3.get("1.0", tk.END)
        
        current_file = self.files_list[self.current_index]
        
        eng_file_path = os.path.join(self.eng_folder_path, current_file)
        log_file_path = os.path.join(self.log_folder_path, current_file)
        
        with open(eng_file_path, 'w') as file:
            file.write(eng_content)

        with open(log_file_path, 'w') as file:
            file.write(log_content)

        
    def load_selected_file(self, event):
        selected_file = self.selected_file_var.get()

        if selected_file in self.files_list:
            self.current_index = self.files_list.index(selected_file)
            self.show_current_file()

if __name__ == "__main__":
    folder = "/home/grads/t/tiendat.ng.cs/github_repos/PLM_and_BugReport_datasets/datasets/hand-gen-datasets"
    project_name = "spark"

    project_path = os.path.join(folder, project_name)
    
    desc_folder_name = "desc"
    eng_folder_name = "eng"
    log_folder_name = "log"
    
    desc_folder_path = os.path.join(project_path, desc_folder_name)
    eng_folder_path = os.path.join(project_path, eng_folder_name)
    log_folder_path = os.path.join(project_path, log_folder_name)
    
    # folder_path = "your_folder_path_here"  # Replace with the path to your folder

    root = tk.Tk()
    root.title("File Viewer")

    app = FileViewerApp(root, (desc_folder_path, eng_folder_path, log_folder_path))

    root.mainloop()
