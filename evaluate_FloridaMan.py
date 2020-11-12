import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import statistics 
import logging as logger
logger.basicConfig(level=logger.INFO)

def generate_similarity(article1, article2): 
	'''Generates cosine similarity between article titles'''
	corpus = [article1, article2]
	vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
	tfidf = vect.fit_transform(corpus)                                                                                                                                                                                                                       
	pairwise_similarity = tfidf * tfidf.T 
	myvalue = pairwise_similarity.toarray()[0][1]
	return myvalue  

def original_article_avg_similarity(original_df): 
	'''Loops through 1000, randomly selecting 2 articles to calcualte the cosine similarity between'''
	# Evaluate similarity betweeen original and original
	random.seed(400)
	original_df_list = list(range(0,original_df.shape[0]))
	original_similarity_list = [] 

	# Do this 1000 times 
	for i in range(0,1000): 
	    # Generate 2 random articles
	    random_gen = random.sample(original_df_list, 2)
	    article1 = original_df.iloc[random_gen[0],0]
	    article2 = original_df.iloc[random_gen[1],0]
	    # Get the similarity statistic 
	    myvalue = generate_similarity(article1, article2)
	    original_similarity_list.append(myvalue)

	# Calculate the average similarity 
	original_similarity = statistics.mean(original_similarity_list)
	return original_similarity 


def generated_article_avg_similairty(orginial_df, generated_df): 
	'''Loops through 1000, randomly selecting 2 articles to calcualte the cosine similarity between --
	 comparing genrated and original articles'''
	 # Evaluate similarity betweeen original and generated
	random.seed(400) 
	original_df_list = list(range(0,original_df.shape[0]))
	generated_df_list = list(range(0, generated_df.shape[0]))
	new_similarity_list = [] 

	# Do this 1000 times 
	for i in range(0,1000): 
	    # Generate 2 random articles
	    random_gen_og = random.sample(original_df_list, 1)
	    random_gen_gen = random.sample(generated_df_list, 1)
	    article1 = original_df.iloc[random_gen_og[0],0]
	    article2 = generated_df.iloc[random_gen_gen[0],0]
	    # Get the similarity statistic 
	    myvalue = generate_similarity(article1, article2)
	    new_similarity_list.append(myvalue)

	# Calculate the average similarity 
	generated_similarity = statistics.mean(new_similarity_list)
	return generated_similarity 



if __name__ == '__main__':
	# Load data
	try: 
		original_df = pd.read_csv("florida_articles_df.csv")
		original_df = original_df.drop(columns = ['Unnamed: 0'])
		generated_df = pd.read_csv("cleaned_florida_man_results.csv")
		generated_df = generated_df.drop(columns = ['Unnamed: 0'])
		generated_df = generated_df.rename(columns={"0": "article"})
		logger.info("All data loaded")
	except: 
		logger.error("Issue loading data")


	# Generate statistics
	try: 
		original_similarity = original_article_avg_similarity(original_df)
		logger.info("Computation of original similarity complete")
		generated_similarity = generated_article_avg_similairty(original_df, generated_df)
		logger.info("Computation of generated similarity complete")
	except: 
		logger.error("There was a calculation issue")


	# Save to file 
	try: 
		results = {"Comparison": ["Original to Original Cosine Similarity", "Original to Generated Cosine Similarity"], 
	            "Statistic": [original_similarity, generated_similarity]}
		results_df = pd.DataFrame(results)
		results_df.to_csv("cosine_similarity_results.csv")
		logger.info("Results saved to file")
	except:
		logger.error("There was an issue saving the results to file")







