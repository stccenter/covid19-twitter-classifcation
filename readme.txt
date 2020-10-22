1. Download "GoogleNews-vectors-negative300.bin" online as word-embedding. https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
2. Run "total_classification.py" with the csv data with the input column "tweet_text" and "category_id", where "tweet_text" is the tweet content will be trained and "category_id" is the category each tweets is labeled.
3. line 93 "model.add(Dense(8))" is how many categories does the labeling have.
4. Run the "total_classification.py" with the data will output a directory called "model_save", which will have three components of the model.
5. Run "model_prediction.py" to use the model saved in "model_save" directory, which will output the predication of tweeter content.