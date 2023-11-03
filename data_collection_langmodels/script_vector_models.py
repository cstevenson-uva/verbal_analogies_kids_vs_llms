# Gensim
from gensim.models import KeyedVectors, Word2Vec, FastText
import pandas as pd
import numpy as np

# Download csv-file
df = pd.read_csv('../data/prowise_verbal_analogy_data.csv')

#Everything lowercase, remove - and '
df = df.applymap(lambda s: s.lower() if type(s) == str else s)
df = df.replace('-','', regex=True)
df = df.replace("'", '', regex=True)
#df = df.head()
print(df)

#list of words in df per column
A = list(df["A"])
B = list(df["B"])
C = list(df["C"])
D = list(df["D"])
IncD1 = list(df["Ans1"])
IncD2 = list(df["Ans2"])
IncD3 = list(df["Ans3"])
IncD4 = list(df["Ans4"])
#lists = [A, B, C, D, IncD1, IncD2, IncD3, IncD4]

#Word2Vec - Clips
model=KeyedVectors.load_word2vec_format("~/Documents/PyCharm/combined-320.txt")
#Fasttext
#model=KeyedVectors.load_word2vec_format("~/Documents/PyCharm/cc.nl.300.vec")


#List of words in model
words_in_model = list(model.index_to_key)
print(words_in_model)

#Words that are not in the model, but exists in the dataframe
difference1 = list(set(A) - set(words_in_model))
difference2 = list(set(B) - set(words_in_model))
difference3 = list(set(C) - set(words_in_model))
difference4 = list(set(D) - set(words_in_model))
difference5 = list(set(IncD1) - set(words_in_model))
difference6 = list(set(IncD2) - set(words_in_model))
difference7 = list(set(IncD3) - set(words_in_model))
difference8 = list(set(IncD4) - set(words_in_model))

#one list of differences
differences = difference1 + difference2 + difference3 + difference4 + difference5 + difference6 + difference7 + difference8
print(differences)

b = 0
df['A'] = df['A'].replace(differences,b,regex=True)
df['B'] = df['B'].replace(differences,b,regex=True)
df['C'] = df['C'].replace(differences,b,regex=True)
df['D'] = df['D'].replace(differences,b,regex=True)
df['Ans1'] = df['Ans1'].replace(differences,b,regex=True)
df['Ans2'] = df['Ans2'].replace(differences,b,regex=True)
df['Ans3'] = df['Ans3'].replace(differences,b,regex=True)
df['Ans4'] = df['Ans4'].replace(differences,b,regex=True)
print(df)


#Get word vectors Clips
df['VA'] = df['A'].apply(model.get_vector)
df['VB'] = df['B'].apply(model.get_vector)
df['VC'] = df['C'].apply(model.get_vector)
df['VD'] = df['D'].apply(model.get_vector)
df['VAns1'] = df['Ans1'].apply(model.get_vector)
df['VAns2'] = df['Ans2'].apply(model.get_vector)
df['VAns3'] = df['Ans3'].apply(model.get_vector)
df['VAns4'] = df['Ans4'].apply(model.get_vector)

#Calculate cosine similarity per answeroption
def sim_ans1 (row):
    question = row['VA'] - row['VC'] + row['VB']
    answer = row['VAns1']
    return np.dot(answer, question) / (np.linalg.norm(answer) * np.linalg.norm(question))

df['w2v_Ans1'] = df.apply(sim_ans1, axis=1)

def sim_ans2 (row):
    question = row['VA'] - row['VC'] + row['VB']
    answer = row['VAns2']
    return np.dot(answer, question) / (np.linalg.norm(answer) * np.linalg.norm(question))

df['w2v_Ans2'] = df.apply(sim_ans2, axis=1)

def sim_ans3 (row):
    question = row['VA'] - row['VC'] + row['VB']
    answer = row['VAns3']
    return np.dot(answer, question) / (np.linalg.norm(answer) * np.linalg.norm(question))

df['w2v_Ans3'] = df.apply(sim_ans3, axis=1)

def sim_ans4 (row):
    question = row['VA'] - row['VC'] + row['VB']
    answer = row['VAns4']
    return np.dot(answer, question) / (np.linalg.norm(answer) * np.linalg.norm(question))

df['w2v_Ans4'] = df.apply(sim_ans4, axis=1)

def correct_ans (row):
    question = row['VA'] - row['VC'] + row['VB']
    answer = row['VD']
    return np.dot(answer, question) / (np.linalg.norm(answer) * np.linalg.norm(question))

df['w2v_Ans5'] = df.apply(sim_ans1, axis=1)

#Drop the wordvectors, because they take up to much space
df = df.drop('VA', 1)
df = df.drop('VB', 1)
df = df.drop('VC', 1)
df = df.drop('VD', 1)
df = df.drop('VAns1', 1)
df = df.drop('VAns2', 1)
df = df.drop('VAns3', 1)
df = df.drop('VAns4', 1)

#Calculate minimal value over answeroptions
minValues = df[["w2v_Ans1", "w2v_Ans2", "w2v_Ans3", "w2v_Ans4", "w2v_Ans5"]].min(axis=1)
df['minValues'] = minValues
#Check if answeroption 5 (correct answer) is the given answer.
df['w2v_result'] = np.where(df["minValues"] == df["w2v_Ans5"], True, False)
#Drop columns not needed in endresult
df = df.drop("minValues", 1)

#Change value to N/A when word is not found in the dictionary, because value is not reliable
df['w2v_result'] = np.where((df["A"]==0) | (df["B"]==0) | (df["C"]==0) | (df["D"]==0) | (df["Ans1"]==0) | (df["Ans2"]==0) | (df["Ans3"]==0) | (df["Ans4"]==0), "N/A", df['w2v_result'])
df['w2v_Ans1'] = np.where((df["A"]==0) | (df["B"]==0) | (df["C"]==0) | (df["Ans1"]==0), "N/A", df['w2v_Ans1'])
df['w2v_Ans2'] = np.where((df["A"]==0) | (df["B"]==0) | (df["C"]==0) | (df["Ans2"]==0), "N/A", df['w2v_Ans2'])
df['w2v_Ans3'] = np.where((df["A"]==0) | (df["B"]==0) | (df["C"]==0) | (df["Ans3"]==0), "N/A", df['w2v_Ans3'])
df['w2v_Ans4'] = np.where((df["A"]==0) | (df["B"]==0) | (df["C"]==0) | (df["Ans4"]==0), "N/A", df['w2v_Ans4'])
df['w2v_Ans5'] = np.where((df["A"]==0) | (df["B"]==0) | (df["C"]==0) | (df["D"]==0), "N/A", df['w2v_Ans5'])

df.to_csv('data_ABC_prompt/word2vec_prompt_ABC_results.csv', index=False)

