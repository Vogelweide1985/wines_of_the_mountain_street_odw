#Init


#Sample Data: manuell

#### Load Data ####


import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # Bag of Words + Tokenize
from sklearn.feature_extraction.text import TfidfTransformer # TF-IDF


directory = "data" # directory where .txt files are stored
corpus = [] # list to store the contents of the text files

# loop through files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        # open the file, read its contents, and append to the list
        with open(os.path.join(directory, filename), "r") as f:
            corpus.append(f.read())

# print the contents of the list
print(corpus)

#### Tokenize ####

# create a CountVectorizer object to tokenize the words
vectorizer = CountVectorizer(analyzer  = "word")

# fit the vectorizer to the text files and transform the text files into a tokenized matrix
tokenized_matrix = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()

#create dataframe
cv_dataframe=pd.DataFrame(tokenized_matrix.toarray(),columns=vectorizer.get_feature_names_out())
print(cv_dataframe)

#### TF-IDF ####

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(tokenized_matrix)
X_train_tfidf.shape
cv_dataframe=pd.DataFrame(X_train_tfidf.toarray(),columns=tfidf_transformer.get_feature_names_out())


#### Cluster ####
