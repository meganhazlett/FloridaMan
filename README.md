# Florida Man Returns
Generating New “Florida Man” Article Titles Using LSTM /n
An NPL Project by Megan Hazlett 

## Directory Structure 
```
├── README.md                         <- You are here
│
├──get_florida_data.py                        <- Scrapes Twitter and Reddit for data (example of data can be found on FloridaMan subreddit and Twitter page @FloridaMan__)\
│
├── build_FloridaMan_model.py                        <- Builds LSTM model
├── generate_Florida_Man_text.py                        <- Generates 100 new florida man articles from random seeds. Results in cleaned_florida_man_results.csv
├── cleaned_florida_man_results.csv                        <- Results of 100 new articles from random seeds
 | 
├── evaluate_FloridaMan.py                        <- Creates cosine_similarity_results.csv
├── cosine_similarity_results.css                        <- Results of cosine similarity tests
│
├── user_generated_articles.py                        <- Script that works with app.py to prompt user input
├── app.py                        <- Launches REST instance of user generated articles 	
├── templates/					<- Contains HTML for making REST instance

```

## Instructions for running 
```python app.py```
Follow the address to http://127.0.0.1:5000

