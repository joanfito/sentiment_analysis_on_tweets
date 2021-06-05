# Libraries
import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Disable Tensorflow messages
import tweepy
import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model

# Constants
NUM_EPOCHS = 8
INIT_LR = 0.001
BATCH_SIZE = 32

# Methods and functions
def get_arguments(parser):
    """Configure the program arguments and return the received ones.

    Args:
        parser: argument parser.

    Returns:
        The program arguments input by the user.

    """
    parser.add_argument("-g", "--generate", action = "store_true", dest = "get_tweets", default = False, help = "Get tweets using the Twitter API")
    parser.add_argument("-k", "--keyword", action = "store", dest = "tweet_keyword", help = "Keyword to search in tweets")
    parser.add_argument("-t", "--train", action = "store_true", dest = "train_model", default = False, help = "Train the neural network")
    parser.add_argument("--train_data", action = "store", dest = "train_path", default = "./data/clean_training.csv", help = "Path to the training dataset")
    return parser.parse_args()

def clean_emails(text):
    """Remove the emails from the text.

    Args:
        text: text to preprocess.

    Returns:
        The text without the emails.

    """
    return re.sub("[^\s]+@[^\s]+", "", str(text))

def clean_usernames(text):
    """Remove the Twitter usernames from the text.

    Args:
        text: text to preprocess.

    Returns:
        The text without the usernames.

    """
    return re.sub("@[^\s]+", "", str(text))

def clean_urls(text):
    """Remove URLs from the text.

    Args:
        text: text to preprocess.

    Returns:
        The text without the URLs.

    """
    return re.sub("(www\.[^\s]+)|(https?://[^\s]+)", "", str(text))

def clean_punctuation(text):
    """Remove punctuation signs from the text.

    Args:
        text: text to preprocess.

    Returns:
        The text without the punctuation signs.

    """
    map = str.maketrans("", "", string.punctuation)
    return str(text).translate(map)

def clean_numbers(text):
    """Remove numbers from the text.

    Args:
        text: text to preprocess.

    Returns:
        The text without numbers.

    """
    return re.sub("[0-9]+", "", str(text))

def clean_stopwords(text):
    """Remove stopwords from the text.

    Args:
        text: text to preprocess.

    Returns:
        The text without the stopwords.

    """
    stopwords_ = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stopwords_])

def preprocess_tweets(tweets_df):
    """Preprocess the tweets.

    Args:
        tweets_df: data frame with the tweets to preprocess.

    """
    # Tranform the text to lowercase
    tweets_df["text"] = tweets_df["text"].str.lower()

    # Remove emails, usernames and URLs
    tweets_df["text"] = tweets_df["text"].apply(lambda text: clean_emails(text))
    tweets_df["text"] = tweets_df["text"].apply(lambda text: clean_usernames(text))
    tweets_df["text"] = tweets_df["text"].apply(lambda text: clean_urls(text))

    # Remove punctuation signs
    tweets_df["text"] = tweets_df["text"].apply(lambda text: clean_punctuation(text))

    # Remove numbers
    tweets_df["text"] = tweets_df["text"].apply(lambda text: clean_numbers(text))

    # Remove stepwords
    tweets_df["text"] = tweets_df["text"].apply(lambda text: clean_stopwords(text))

def _create_plot(history, path):
    """Create and save the loss/accuracy plot.

    Args:
        history: neural network history.
        path: path to save the plot.

    """
    plt.figure()
    plt.style.use("ggplot")

    epochs = np.arange(0, NUM_EPOCHS)

    # Plot the losses and the accuracies
    plt.plot(epochs, history.history["loss"], label = "train_loss")
    plt.plot(epochs, history.history["val_loss"], label = "val_loss")
    plt.plot(epochs, history.history["accuracy"], label = "train_acc")
    plt.plot(epochs, history.history["val_accuracy"], label = "val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("# Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(path)

def _save_nn(model, lb, model_path, labels_path):
    """Save the model and the labels.

    Args:
        model: neural network model.
        lb: label binarizer
        model_path: path to save the model.
        labels_path: path to save the labels.

    """
    model.save(model_path, save_format = "h5")
    f = open(labels_path, "wb")
    f.write(pickle.dumps(lb))
    f.close()

def train_model(tweets_df):
    """Create and train the neural network.

    Args:
        tweets_df: data frame with the tweets to preprocess.

    """
    tokenizer = Tokenizer(num_words = 5000)
    tokenizer.fit_on_texts(tweets_df["text"].values)
    sequences = tokenizer.texts_to_sequences(tweets_df["text"].values)
    data = pad_sequences(sequences, maxlen = 200)

    # Split the data into train and test subsets
    X, X_test, Y, Y_test = train_test_split(data, tweets_df["class"], test_size = 0.2, train_size = 0.8, random_state = 0)

    # Split the train subsets into train and validation subsets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.25, train_size = 0.75, random_state = 0)

    # Transform the label represented as a string to a one-hot encoding array
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_val = lb.transform(Y_val)
    Y_test = lb.transform(Y_test)

    # Create the model
    model = Sequential()
    model.add(InputLayer(input_shape = 200))
    model.add(Embedding(5000, 20))
    model.add(LSTM(100))
    model.add(Dropout(0.7))
    model.add(Dropout(0.3))
    model.add(Dense(20, activation = "relu"))
    model.add(Dense(len(lb.classes_),activation='softmax'))
    print(model.summary())

    opt = Adam(learning_rate = INIT_LR)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

    # Train the neural network
    history = model.fit(x = X_train, y = Y_train, validation_data = (X_val, Y_val), epochs = NUM_EPOCHS, batch_size = BATCH_SIZE)

    # Test the neural network
    Y_pred = model.predict(X_test, batch_size = BATCH_SIZE)
    print(classification_report(Y_test.argmax(axis = 1), Y_pred.argmax(axis = 1), target_names = lb.classes_))

    _create_plot(history, "plot.png")
    _save_nn(model, lb, "model.h5", "model_lb.pickle")


def predict_tweet():
    # TODO
    print("-")
    
def get_tweets(keyword, consumer_key, consumer_secret):
    """Get tweets from a given topic.

    Args:
        keyword: keyword to search in Twitter.
        consumer_key: consumer_key credential to access Twitter's API.
        consumer_secret: consumer_secret credential to access Twitter's API.

    """
    # API authentication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth)

    # Petition
    search_query = keyword + " -filter:retweets"
    tweets = tweepy.Cursor(api.search, q = search_query, lang = "en", tweet_mode = "extended")

    tweets_data = [[tweet.user.id, tweet.user.screen_name, tweet.full_text, tweet.retweet_count, tweet.favorite_count, "neutral"] for tweet in tweets.items(100)]
    tweets_df = pd.DataFrame(data = tweets_data, columns=["user_id", "username", "text", "retweets", "likes", "class"])
    tweets_df.to_csv("./data/tweets.csv", index = False)

def new_training_data():
    """Read Sentiment140 files, preprocess the data and save it in a CSV.

    Returns:
        The data frame with the data.

    """
    # Load the data from the files
    tweets_df = pd.read_csv("./data/training_1.csv", header = None, names = ["polarity", "id", "date", "query", "user", "text"], encoding = "latin")
    df2 = pd.read_csv("./data/training_2.csv", header = None, names = ["polarity", "id", "date", "query", "user", "text"], encoding = "latin")
    tweets_df = tweets_df.append(df2)

    # Assign the class label
    conditions = [tweets_df["polarity"] == 0, tweets_df["polarity"] == 2, tweets_df["polarity"] == 4]
    values = ["negative", "neutral", "positive"]
    tweets_df["class"] = np.select(conditions, values)

    # Drop the useless columns
    tweets_df = tweets_df.drop(["polarity", "id", "date", "query", "user"], 1)
    preprocess_tweets(tweets_df)
    tweets_df.to_csv("./data/clean_training.csv", index = False)

    return tweets_df

def main():
    parser = argparse.ArgumentParser(description = "Real-time sentiment analysis of twitter")
    options = get_arguments(parser)

    if options.get_tweets:
        if options.tweet_keyword is None:
            parser.error("You must input a keyword to search.")
        else:
            with open("./consumer.json") as file:
                json_data = json.load(file)

            consumer_key = json_data["consumer_key"]
            consumer_secret = json_data["consumer_secret"]

            get_tweets(options.tweet_keyword, consumer_key, consumer_secret)

    if options.train_model:
        #tweets_df = new_training_data()

        tweets_df = pd.read_csv(options.train_path)
        tweets_df["text"] = tweets_df["text"].astype(str)
        train_model(tweets_df)
    else:
        tweets_df = pd.read_csv("./data/tweets.csv")
        preprocess_tweets(tweets_df)
        predict_tweet()

if __name__ == "__main__":
    main()
