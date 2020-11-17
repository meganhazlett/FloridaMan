import numpy
import nltk
from nltk.tokenize import word_tokenize
import logging as logger
import json
import pickle
from csv import reader
logger.basicConfig(level=logger.ERROR)


def generate_user_text(user_sentence, word_vocab , X_word):
	'''Generates new florida man articles based on user input of 10 words'''
	# Load model
	try:
		path = open('finalized_model.pkl', 'rb')
		mymod_word = pickle.load(path)
		logger.info("Model loaded in from file")
	except:
		logger.error()

	# Load model weights
	try:
		filename_word = "weights-improvement-word-100-3.1510.hdf5"
		mymod_word.load_weights(filename_word)
		loss = "categorical_crossentropy"
		optimizer = "adam"
		mymod_word.compile(loss=loss, foptimizer= optimizer)
		logger.info("Model weights loaded.")
	except:
		logger.error("Model weights not found.")

	# Load dictionary
	try:
		with open('myword_dict.json', 'r') as fp:
			myword_dict = json.load(fp)
		logger.info("Dictionary loaded")
		indx_char_word = dict((i, c) for i, c in enumerate(myword_dict))
		logger.info("Dictionary reversed for predictions")
	except:
		logger.error("Dictionary unable to be loadeed")

	my_sentence_token = word_tokenize(user_sentence)

	pattern_word = [myword_dict[w] for w in my_sentence_token]

    # generate characters
	new_pattern = [] 
	for i in range(7):
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


def clean_user_results(user_results): 
	'''Cleans up results generated from generate_text function; removes punctuation ''' 
	tokenizer = nltk.RegexpTokenizer(r"\w+")
	cleaned_list = tokenizer.tokenize(user_results)
	cleaned_sentences = ' '.join(cleaned_list)
	logger.info("Removed punctuation from results ")
	return cleaned_sentences 



def generateText(user_sentence):
	'''Full model run '''
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
	try: 
		#user_sentence = input("Enter a 5 word phrase: ")
		# print("Your phrase is", user_sentence)
		# Get results 
		new_user_article = generate_user_text(user_sentence = user_sentence, word_vocab = word_vocab, X_word = X_word)
		cleaned_user_article = clean_user_results(new_user_article)
		print(cleaned_user_article)
		return cleaned_user_article
	except: 
		logger.error("A word is not in the dictionary or phrase is not long enough... try again")




