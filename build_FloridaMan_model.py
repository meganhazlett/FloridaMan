import numpy
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
import sys
import nltk 
from nltk.tokenize import word_tokenize
import pandas as pd 
import numpy as np
import logging as logger
logger.basicConfig(level=logger.INFO)

def clean_data(data): 
	try: 
		# load ascii text and covert to lowercase and remove back slashes 
		text_lower = data['article'].str.lower()
		logger.info("Text to lower case.")
		text_list = text_lower.to_list()
		mystring = " ".join(text_list)
		mystring = mystring.replace("\\","")
		logger.info("Removed back slashes")

		# tokenize
		tokenized_words = word_tokenize(mystring)
		logger.info("Tokenization complete")
		# remove all hyperlinks as tokens 
		pattern = ("http", "//t.co")
		tokenized_words = [token for token in tokenized_words if not token.startswith(pattern)]
		word_vocab = list(set(tokenized_words))
		word_vocab_vals = np.arange(0, len(word_vocab)+1, 1).tolist() 
		myword_dict = {word_vocab[i]: word_vocab_vals[i] for i in range(len(word_vocab))} 
		indx_char_word = np.array(word_vocab)
		logger.info("Removed hyperlink tokens")

		# vectorizing the text 
		text_indx_word = np.array([myword_dict[w] for w in tokenized_words])
	except: 
		logger.error("Problem with data cleaning")

	return tokenized_words, word_vocab, myword_dict

def define_model_word(hidden_layers, dropout_prob, activation, loss, optimizer, X_transformed_word):
    '''Builds LSTM model -- helper function'''
    # X information 
    time_steps = X_transformed_word.shape[1]
    features = X_transformed_word.shape[2]
    
    # Model 
    mod = Sequential()
    mod.add(LSTM(hidden_layers, input_shape = (time_steps, features)))
    mod.add(Dropout(dropout_prob))
    mod.add(Dense(len(word_vocab), activation = activation))
    mod.compile(loss = loss, optimizer = optimizer)
    return mod 


def build_model(seq_length = 10, hidden_layers = 256, dropout_prob = 0.2, activation = 'softmax', 
	loss = 'categorical_crossentropy', optimizer = 'adam', epochs = 50, batch_size = 128):
	seq_length = seq_length
	X_word = []
	Y_word= []

	for i in range(0, len(tokenized_words) - seq_length, 1):
	    seq_in = tokenized_words[i:i + seq_length]
	    seq_out = tokenized_words[i + seq_length]
	    char_in = [myword_dict[word] for word in seq_in]
	    char_out = myword_dict[seq_out]
	    X_word.append(char_in)
	    Y_word.append(char_out)

	n_patterns_word = len(X_word)
	print("Total number of patterns:", n_patterns_word)

	# Create features 
	X_transformed_word = numpy.reshape(X_word, (n_patterns_word, seq_length, 1))  
	X_transformed_word = X_transformed_word/float(len(word_vocab))
	Y_transformed_word = np_utils.to_categorical(Y_word)

	# Create model 
	mymod_word = define_model_word(hidden_layers, dropout_prob, activation, loss, optimizer, X_transformed_word)
	logger.info("Nodel read to be built.")

	# model check points 
	filepath_word = "weights-improvement-word-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint_word = ModelCheckpoint(filepath_word, monitor = 'loss', verbose = 1, save_best_only = True, mode = "min")
	early_stopping_word = EarlyStopping(patience = 2)
	callback_info_word = [checkpoint_word, early_stopping_word]

	# build model 
	mymod_word.fit(X_transformed_word, Y_transformed_word, epochs=epochs, batch_size=batch_size, callbacks = callback_info_word)
	logger.info("Model built. Check folder.")
	return mymod_word


if __name__ == '__main__':
	# Load data 
	try: 
		data = pd.read_csv("florida_articles_df.csv")
		data = data.drop(columns = ['Unnamed: 0'])
		logger.info("Data is loaded.")
	except: 
		logger.error("Data was not able to be loaded.")
	# Clean data 
	tokenized_words, word_vocab, myword_dict = clean_data(data = data)
	model = build_model(seq_length = 10, hidden_layers = 256, dropout_prob = 0.2, activation = 'softmax', 
	loss = 'categorical_crossentropy', optimizer = 'adam', epochs = 100, batch_size = 128)

