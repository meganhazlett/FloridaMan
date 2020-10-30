import pandas as pd
import numpy as np
import tweepy 
import praw 
import os 
import logging as logger
logger.basicConfig(level=logger.INFO)


def get_reddit_data(reddit_client_id, reddit_client_secret, reddit_user_agent): 
	''' Collects reddit posts from FloridaMan page''' 

	try: 
		reddit = praw.Reddit(client_id=reddit_client_id, client_secret=reddit_client_secret, user_agent=reddit_user_agent)
		logger.info("Reddit credientials valid")
	except: 
		logger.error("Please retry Reddit credentials")

	posts = []
	try: 
		ml_subreddit = reddit.subreddit('FloridaMan')
		for post in ml_subreddit.new(limit = 100000):
		    posts.append(post.title)
		for post in ml_subreddit.top(limit = 100000): 
		    posts.append(post.title)
		for post in ml_subreddit.hot(limit = 100000): 
		    posts.append(post.title)
		for post in ml_subreddit.controversial(limit = 100000): 
		    posts.append(post.title)
		logger.info("FloridaMan subreddit read in")
	except: 
		logger.error("Something went wrong with the FloridaMan subreddit")
	# Keep only unique 
	unique_posts = list(set(posts))

	# Data frame 
	reddit_df = pd.DataFrame(unique_posts,columns=['article'])
	logger.info("FloridaMan subreddit in data frame")
	return reddit_df


def get_twitter_data(twitter_consumer_key, twitter_consumer_secret, twitter_access_token, twitter_access_token_secret): 
	'''Collects tweets from @FloridaMan__ handle''' 

	try: 
		auth = tweepy.OAuthHandler(consumer_key = twitter_consumer_key, consumer_secret = twitter_consumer_secret)
		auth.set_access_token(access_token = twitter_access_token, access_token_secret = twitter_access_token_secret)
		api = tweepy.API(auth)
		logger.info("Twitter credientials valid")
	except:
		logger.error("Please retry Twitter credientials")


	username = 'FloridaMan__'
	count = 16000
	try:
		tweets = tweepy.Cursor(api.user_timeline,id=username).items(count)
		tweets_list = [[tweet.text] for tweet in tweets]
		tweets_df = pd.DataFrame(tweets_list)
		logger.info("@FloridaMan__ tweers read and placed in data frame")
	except: 
		logger.error("Failed to find @FloridaMan__ tweets. Try again")

	return tweets_df


if __name__ == '__main__':
	# Get access keys
    reddit_client_id = os.environ.get("REDDIT_CLIENT_ID")
    reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    reddit_user_agent = os.environ.get("REDDIT_USER_AGENT")

    twitter_consumer_key = os.environ.get("TWITTER_CONSUMER_KEY")
    twitter_consumer_secret = os.environ.get("TWITTER_CONSUMER_SECRET")
    twitter_access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    twitter_access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

    # Query data 
    logger.info("hello world")
    reddit_df = get_reddit_data(reddit_client_id, reddit_client_secret, reddit_user_agent)
    tweets_df = get_twitter_data(twitter_consumer_key, twitter_consumer_secret, twitter_access_token, twitter_access_token_secret) 

    try:
        florida_articles_df = pd.concat([reddit_df, tweets_df])
        florida_articles_df.to_csv("florida_articles_df.csv")
        logger.info("Florida man data read in file")
    except: 
        logger.error("Something went wrong when creating whole data set")


