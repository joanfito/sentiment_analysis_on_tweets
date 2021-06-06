[![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joanfito/sentiment_analysis_on_tweets/blob/main/sentiment_analysis_training.ipynb)

# Sentiment analysis on tweets
### Usage
To run the script, use `python3 main.py`. By default, it will predict the sentiment of the tweets in _data/tweets.csv_ with a trained neural network stored in _model.h5_.

To change into training mode, use the modifier _-t_ (`python3 main.py -t`). The training data is in _data/clean_training.csv_ and can be changed with the _--train_data_ modifier (`python3 main.py -t --train_data <new_path>`).

The script can also get tweets with a given keyword to predict their sentiment: `python3 main.py -g -k <keyword>`.

Finally, it can be executed in Google Colab using the notebook. To successfully run it, the main.py file and the data folder have to be uploaded to your Google Drive inside a folder named _pepsico_.


### Neural network architecture
The neural network architecture comes from [Bhargava et al. 2019](https://doi.org/10.1515/jisys-2017-0398). It consists of 6 layers:
- Embedding (input dimension = 5000 and output dimension = 20)
- LSTM (units = 100)
- Dropout (dropout rate = 0.7)
- Dropout (dropout rate = 0.3)
- Dense (units = 20)
- Dense (units = 2)

### Training data
The neural network has been trained using [Sentiment140](http://help.sentiment140.com/for-students) dataset.

### Training results
The model has a 77% accuracy at its first epoch for the training subset, and it increases to 80%, while the validation accuracy is 78.5%. Therefore, it can be appreciated a slight overfitting. The model has a 79% accuracy (80% for positive and 77% for negative) with the test subset.

![Accuracy/lost plot](https://github.com/joanfito/sentiment_analysis_on_tweets/blob/main/plot.png?raw=true)
