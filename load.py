#Init


#Sample Data: manuell

#### Load Data ####


import os


directory = "data" # directory where .txt files are stored
text_files = [] # list to store the contents of the text files

# loop through files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # open the file, read its contents, and append to the list
        with open(os.path.join(directory, filename), "r") as f:
            text_files.append(f.read())

# print the contents of the list
print(text_files)

#### Tokenize ####


#### TF-IDF ####


#### Cluster ####
