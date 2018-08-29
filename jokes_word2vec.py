import pandas as pd
import nltk
import gensim 
from gensim import corpora,models,similarities
    
dataset = pd.read_csv('jokes.csv')
    
x = dataset['Question'].values.tolist()
y = dataset['Answer'].values.tolist()
    
# Here we are adding the words in both x and y as we want to create a corpus first for all the sentences
# And then we will create vectors for each word in the sentence
    
corpus = x + y
    
# Tokenize all the words in the corpus
tok_corp = [nltk.word_tokenize(sent) for sent in corpus]
    
model = gensim.models.Word2Vec(tok_corp,min_count=1,size=32)
    
#min_count is the minimum time a word should appear in the dataset
#size is the size of the vector for each word
    
# Saving the model
#model.save('testmodel')
# Loading a saved model
#model = gensim.models.Word2Vec.load('testmodel')
    
model.most_similar('dad')
