<!--
 * @Author: your name
 * @Date: 2020-10-22 14:34:53
 * @LastEditTime: 2020-10-22 14:58:11
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \covid19-twitter-classifcation\README.md
-->
## What are the software requirement?   
    1. Python IDE 
    2. Numpy
    3. Pandas
    4. gensim
    5. keras
    6. nltk 
    7. sklearn

## How to set up the python environment and install packages?
    1. Clone code by clinking on the download button 
    ![clone pic]()
    2. Open the project folder with your choice of python IDE
    3. Execute below comment to install required python packages.

```BibTeX
    pip install -r requirements.txt
```
## Where to download the data ?
    1. "GoogleNews-vectors-negative300.bin" online as word-embedding. https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

## How to get the results? 
    1. Run "total_classification.py" with the csv data with the input column "tweet_text" and "category_id", where "tweet_text" is the tweet content will be trained and "category_id" is the category each tweets is labeled.
    2. Line 93 "model.add(Dense(8))" is how many categories does the labeling have, change the input value accordingly.
    3. Run the "total_classification.py" with the data will output a directory called "model_save", which will have three components of the model.
    4. Run "model_prediction.py" to use the model saved in "model_save" directory, which will output the predication of tweeter content.

## Tutorial Video