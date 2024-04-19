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

    def __init__(self):
        nltk.download('stopwords')
        self.stop = stopwords.words('english')

    def remove_emoji(self, s):
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
        return emoji_pattern.sub(r'', s)

    def remove_non_ascii(self, s):
        return re.sub(r'[^\x00-\x7F]', '', s)

    def remove_handles(self, s):
        return re.sub('@[^\s]+', '', s)

    def remove_hashtags(self, s):
        return re.sub('#[^\s]+', '', s)

    def remove_stopwords(self, s):
        return ' '.join([word for word in s.split() if word not in self.stop])

    def remove_punctuation(self, s):
        return s.translate(str.maketrans('', '', string.punctuation))

    def remove_url(self, s):
        return re.sub(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
            '', s)

    def remove_amp(self, s):
        return re.sub('amp', '', s)

    def remove_words_depression(self, s):
        banned = ['I', 'Emoji']
        return ' '.join(w for w in s.split() if not w in banned)

    def clean_data(self, string):
        string = self.remove_url(string)
        string = self.remove_emoji(string)
        string = self.remove_non_ascii(string)
        string = self.remove_handles(string)
        string = self.remove_hashtags(string)
        string = self.remove_stopwords(string)
        string = self.remove_punctuation(string)
        string = self.remove_amp(string)
        string = self.remove_words_depression(string)

        return string

    def get_sentiment(self, s):
        df = pd.DataFrame([s], columns=['text'])

        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(str(s))

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

    def __init__(self, dataframe):
        self.tokenized_text = self.tokenize(dataframe)
        self.attribute_data = self.get_attr_from_dataframe(dataframe)
        self.model = self.initialise_model()

    def tokenize(self, dataframe):
        tokenizer = Tokenizer(num_words=80000, split=' ')
        tokenizer.fit_on_texts(dataframe['text'].values)
        X_text = tokenizer.texts_to_sequences(dataframe['text'].values)
        X_text = pad_sequences(X_text, 300)
        return X_text

    def get_attr_from_dataframe(self, dataframe):
        X_attr = dataframe.loc[:, dataframe.columns != 'text']
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
