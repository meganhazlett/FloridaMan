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
import json
import pickle
from csv import reader
logger.basicConfig(level=logger.INFO)


def generate_text(word_vocab , X_word): 
	'''Generates new florida man articles'''
	# set seed 
	seed = numpy.random.randint(0, len(X_word)-1)
	pattern_word = X_word[seed]
    
    # generate characters
	new_pattern = [] 
	for i in range(20):
		temp_pattern = pattern_word[i:len(pattern_word)]
		x = numpy.reshape(temp_pattern, (1,len(temp_pattern), 1))
		x = x / float(len(word_vocab))
		prediction = mymod_word.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = indx_char_word[index]
		pattern_word.append(index)
		new_pattern.append(result)
	
	new_pattern_string = ' '.join([indx_char_word[val] for val in pattern_word])
	logger.info("Results generated")
	return new_pattern_string


def clean_results(florida_man_results): 
	'''Cleans up results generated from generate_text function; removes punctuation ''' 
	tokenizer = nltk.RegexpTokenizer(r"\w+")
	cleaned_list = list(map(lambda x: tokenizer.tokenize(x), florida_man_results))
	cleaned_sentences = [ ' '.join(item) for item in cleaned_list] 
	logger.info("Removed punctuation from results ")
	return cleaned_sentences 



if __name__ == '__main__':
	# Load model 
	try: 
		path = open('finalized_model.pkl', 'rb')
		mymod_word = pickle.load(path)
		logger.info("Model loaded in from file")
	except: 
		logger.error()

	# Load model weights  
	try:  
		filename_word = "weights-improvement-word-100-1.8516.hdf5"
		mymod_word.load_weights(filename_word)
		loss = "categorical_crossentropy"
		optimizer = "adam"
		mymod_word.compile(loss=loss, optimizer= optimizer)
		logger.info("Model weights loaded.")
	except: 
		logger.error("Model weights not found.")



	# Reverse dictionary
	try: 
		with open('myword_dict.json', 'r') as fp:
			myword_dict = json.load(fp)
		logger.info("Dictionary loaded")
		indx_char_word = dict((i, c) for i, c in enumerate(myword_dict))
		logger.info("Dictionary reversed for predictions")
	except: 
		logger.error("Dictionary unable to be loadeed")

	# Load in word_vocab 
	try: 
		word_vocab = []
		with open('word_vocab.txt', 'r') as filehandle:
			filecontents = filehandle.readlines()
			for line in filecontents:
				current_word = line[:-1]
				word_vocab.append(current_word)

		logger.info("Loaded in word_vocab.txt as list")
	except: 
		logger.error("Unable to load word vocab file")


	# Load in X_word 
	try: 
		with open('X_word.csv', 'r') as read_obj:
		    csv_reader = reader(read_obj)
		    X_word = list(csv_reader)
		# Convert to integeres 
		X_word = [[int(s) for s in xs] for xs in X_word]
		logger.info("Loaded in X_word")
	except:
		logger.error("Unable to load X_word")


	#Get results 
	florida_man_results = [] 
	for i in range(0,100): 
		new_pattern_string = generate_text(word_vocab = word_vocab, X_word = X_word)
		florida_man_results.append(new_pattern_string)

	# Clean results 
	cleaned_sentences = clean_results(florida_man_results)
	cleaned_sentences_df = pd.DataFrame(cleaned_sentences)
	cleaned_sentences_df.to_csv("cleaned_florida_man_results.csv")
	logger.info("Cleaned results saved to file")


	

