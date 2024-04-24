import re, string
import nltk
from nltk.corpus import stopwords
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import Model
from keras.layers import Concatenate, Input, Embedding, Dense
from keras.models import load_model
import numpy as np


class DataPreprocessor:
    stop = ''

    message = ''

    def __init__(self, message):
        nltk.download('stopwords')
        self.stop = stopwords.words('english')

        self.message = message

    def remove_emoji(self):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002500-\U00002BEF"
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        self.message = emoji_pattern.sub(r'', self.message)

    def remove_non_ascii(self):
        self.message = re.sub(r'[^\x00-\x7F]', '', self.message)

    def remove_handles(self):
        self.message = re.sub('@[^\s]+', '', self.message)

    def remove_hashtags(self):
        self.message = re.sub('#[^\s]+', '', self.message)

    def remove_stopwords(self):
        self.message = ' '.join([word for word in self.message.split() if word not in self.stop])

    def remove_punctuation(self):
        self.message = self.message.translate(str.maketrans('', '', string.punctuation))

    def remove_url(self):
        self.message = re.sub(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            '', self.message)

    def remove_amp(self):
        self.message = re.sub('amp', '', self.message)

    def remove_banned_words(self):
        banned = ['I', 'Emoji']
        self.message = ' '.join(w for w in self.message.split() if not w in banned)

    def clean_data(self):
        self.remove_url()
        self.remove_emoji()
        self.remove_non_ascii()
        self.remove_handles()
        self.remove_hashtags()
        self.remove_stopwords()
        self.remove_punctuation()
        self.remove_amp()
        self.remove_banned_words()

    def get_message(self):
        return self.message

    def get_sentiment(self):
        df = pd.DataFrame([self.message], columns=['text'])

        # Sentiment Analysis perfomed using VaderSentiment (Hutto & Gilbert, 2014)
        # Hutto, C. J., \& Gilbert, E. E. (2014, June). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Paper presented at the Eighth International Conference on Weblogs and Social Media (ICWSM-14), Ann Arbor, MI.
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(str(self.message))


        # AAdditional Emotion Scores provided by NRCLex (Bailey, n.d)
        # Bailey, M. (n.d). NRCLex. GitHub. \\ https://github.com/metalcorebear/NRCLex
        df['fear'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('fear', 0))
        df['anger'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('anger', 0))
        df['anticipation'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('anticipation', 0))
        df['trust'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('trust', 0))
        df['surprise'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('surprise', 0))
        df['positive'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('positive', 0))
        df['negative'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('negative', 0))
        df['sadness'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('sadness', 0))
        df['disgust'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('disgust', 0))
        df['joy'] = df['text'].apply(lambda x: NRCLex(str(x)).raw_emotion_scores.get('joy', 0))
        df['Message Size'] = df['text'].apply(lambda x: len(str(x)))
        df['negative'] = sentiment['neg']
        df['neutral'] = sentiment['neu']
        df['positive'] = sentiment['pos']
        df['compound'] = sentiment['compound']

        return df


class Predict:
    model = None
    tokenized_text = None
    attribute_data = None
    dataframe = None

    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.model = self.initialise_model()
        self.tokenized_text = self.tokenize()
        self.attribute_data = self.get_attr_from_dataframe()

    def tokenize(self):
        tokenizer = Tokenizer(num_words=80000, split=' ')
        tokenizer.fit_on_texts(self.dataframe['text'].values)
        X_text = tokenizer.texts_to_sequences(self.dataframe['text'].values)
        X_text = pad_sequences(X_text, 300)
        return X_text

    def get_attr_from_dataframe(self):
        X_attr = self.dataframe.loc[:, self.dataframe.columns != 'text']
        return X_attr

    def make_prediction(self):
        return np.rint(self.model.predict(x=[self.tokenized_text, self.attribute_data], verbose=0))

    def embedding(self):
        text_input = Input(shape=(300,))
        embedding = Embedding(80000 + 1, 46, input_length=128, trainable=False)(text_input)

        return Model(text_input, embedding)

    def initialise_model(self):
        embe_nn = self.embedding()
        textual_model = load_model(r'depression_detection_api/models/textual_model_0.9734146595001221.keras')
        embedded_model = Model(embe_nn.input, textual_model(embe_nn.output))

        attribute_model = load_model(r'depression_detection_api/models/attribute_model_0.9734146595001221.keras')

        output = Concatenate()([embedded_model.output, attribute_model.input])

        output = Dense(1, activation="sigmoid")(output)

        full_model = Model(inputs=[embedded_model.input, attribute_model.input], outputs=output)

        return full_model
