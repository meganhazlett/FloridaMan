# Florida Man Returns
Generating New “Florida Man” Article Titles Using LSTM <br/>
An NPL Project by Megan Hazlett <br/>
November 2020

## Directory Structure 
```
├── README.md                                    <- You are here
├── FloridaMan_WriteUp.pdf                                  <- Full write up of process
│
#### Acquires data for project
├──get_florida_data.py                           <- Scrapes Twitter and Reddit for data (example of data can be found on FloridaMan subreddit and Twitter page @FloridaMan__)
│
#### Builds LSTM model
├── build_FloridaMan_model.py              <- Builds LSTM model
├── generate_Florida_Man_text.py          <- Generates 100 new florida man articles from random seeds. Results in cleaned_florida_man_results.csv
├── cleaned_florida_man_results.csv    <- Results of 100 new articles from random seeds
│
#### Cosine similarity evaluation and comparison (Model evaluation)
├── evaluate_FloridaMan.py                <- Creates cosine_similarity_results.csv
├── cosine_similarity_results.css             <- Results of cosine similarity tests
│
#### API (User interactive article generator)
├── user_generated_articles.py               <- Script that works with app.py to prompt user input
├── app.py                                                <- Launches REST instance of user generated articles 	
├── templates/					   <- Contains HTML for making REST instance
├── word_vocab.txt                                  <- Txt file of all word in vocab 	
├── myword_dict.json                               <- Dictionary in form of json file	
├── X_word.csv                                        <- CSV of vectorized dictionary 	
├── finalized_model.pkl                      <- Pkl file generated from bui;d_FlordaMan_model.py
├── weights-improvement-word-100.3.1510.hdf5   <- HDF5 files of weights for final model


```

## Instructions for running 
Clone the repo <br/>
Run the following line in your command line <br/>
```python app.py``` <br/>
Follow the address to http://127.0.0.1:5000 <br/>
Enter 5 words that could be part of a Florida Man article (all lower case, no punctuation) <br/>
Examples include: <br/>
- "florida man parks in car" <br/>
- "while going to store florida" <br/>
- "in a fit of rage" 
Note that errors are usually attributed to a word not being in the dictionary ... if this happens to you please try again. 
