import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras import layers
import re




class AutoRegressiveModel:
    """
    Packs training a model on tweets into a usable class
    """

    def __init__(self, csv_filepath, type='char', embedding_size=256, rnn_size=256):
        """
        Initialize the model and perform preprocessing.

        We expect the data to be loaded from a csv that contains a 'text' column

        :param filepath: str, filepath to retrieve the csv of tweets to be used.
        :param type: str, expects 'char' or 'word'. Whether to generate one word at a time or one character at a time.
        :param embedding_size: int, size of the embedded representation of our vocabulary.
        :param rnn_size: int, size of the rnn state representation.

        """


        df = pd.read_csv(csv_filepath)
        text = df['text']

        self.type = type
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size

        if type == 'char':
            text = text.apply(preprocess_to_characters)
        elif type == 'word':
            text = text.apply(preprocess_tweet)

        else:
            raise ValueError("Invalid 'type' argument: expected 'char' or 'word'. ")

        self.tokenizer = Tokenizer(filters='', lower=False)
        self.tokenizer.fit_on_texts(text)
        self.sequences = self.tokenizer.texts_to_sequences(text)

        self.max_len = max([len(seq) for seq in self.sequences])
        self.vocab_size = len(self.tokenizer.word_counts) + 1




    def build_model(self, saved_model_path=None, training=False):
        """
        Builds the model, according to whether we are going to use for training or generation. For training, the rnn
        internal state must be reset every run in order to train properly, but should be left in its state for inference.
        Can be used to load a model that has been trained already

        :param saved_model_path: str, path to h5 file where the model is expected to be stored.
        :param training: bool, whether the model is being built for training or being loaded from a saved file
        :return:
        """

        assert saved_model_path or training

        if saved_model_path:
            training = False

        if training:
            stateful = False
            batch_size = None
        else:
            stateful = True
            batch_size = 1


        self.model = build_rnn_model(vocab_size=self.vocab_size, embedding_size=self.embedding_size,
                                     rnn_size=self.rnn_size, stateful=stateful, batch_size=batch_size)

        if not training:
            self.model.load_weights(saved_model_path)



    def train_model(self, batch_size=128, epochs=5, tensorboard_logdir=None, model_save_path=None, overwrite=False):
        """
        Train a model, using data loaded previously. Model will be stored at given location, and tensorboard logs
        will be stored if optional argument is passed.
        :param batch_size: int, size of batches to be used for training
        :param epochs: int, number of epochs to run
        :param tensorboard_logdir: str, optional - directory to store tensorboard training logs
        :param model_save_path: str, optional (but recommended) - path to store model at after training
        :param overwrite: bool, if model_save_path given and model already exists whether to overwrite
        :return:
        """


        if tensorboard_logdir:
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir)
            callbacks = [tensorboard]
        else:
            callbacks = None

        if isinstance(model_save_path, None):
            print('NO SAVE PATH GIVEN. MODEL WILL NOT BE SAVED')

        self.model.fit_generator(self.batch_generator(batch_size), steps_per_epoch=len(self.sequences)//batch_size + 1,
                                 epochs=epochs, workers=-1, use_multiprocessing=True, callbacks=callbacks)

        if model_save_path:
            self.model.save_weights(model_save_path, overwrite=overwrite)




    def sample_sentence(self, seed=''):
        """ Sample a sentence. Takes a seed string as input and runs through predictions by the model """

        self.model.reset_states()

        if isinstance(seed, str):
            if not re.match('<BEGIN_TWEET>', seed):
                seed = '<BEGIN_TWEET> ' + seed

        elif isinstance(seed, list):
            if seed[0] != '<BEGIN_TWEET>':
                seed = ['<BEGIN_TWEET>'] + seed

        sequence = self.tokenizer.texts_to_sequences([seed])
        predictions = sequence[0]

        for i in range(len(predictions) - 1):
            self.model.predict(np.expand_dims(predictions[i], 0))

        last_word_idx = predictions[-1]
        last_word = self.tokenizer.index_word[last_word_idx]
        while last_word != '<END_TWEET>':

            next_word_p = self.model.predict(np.expand_dims([last_word_idx], 0))[0, 0, :]
            next_word_p = next_word_p / sum(next_word_p)
            #next_word_idx = np.argmax(next_word_p)
            next_word_idx = np.random.choice(len(next_word_p), p=next_word_p)
            predictions.append(next_word_idx)

            last_word_idx = predictions[-1]
            last_word = self.tokenizer.index_word[last_word_idx]
            #print(last_word)


        join_token = '' if self.type == 'char' else ' '
        return join_token.join([self.tokenizer.index_word[k] for k in predictions])

    def batch_generator(self, batch_size):
        """
        Build a generator for returning batches. Used for keras training of variable length sentences.
        :param X: pd.Series, each element contains a np.array of shape (sentence_length, emb_dim).
        :param batch_size: int, size of each batch to be returned by the generator
        :return: a batch of embedded sentences, as a sequence of variable/target pairs: (x1, x2), (x2, x3), ... (xn-1, xn)
        """

        shuffled_index = np.arange(len(self.sequences))

        while True:
            np.random.shuffle(shuffled_index)
            n_batches = len(shuffled_index) // batch_size + 1
            for k in range(n_batches):
                X_batch, y_batch = select_batch(self.sequences, shuffled_index[k*batch_size: (k+1)*batch_size])
                X_batch = tf.keras.preprocessing.sequence.pad_sequences(X_batch, padding='post')
                y_batch = tf.keras.preprocessing.sequence.pad_sequences(y_batch, padding='post')
                yield X_batch, np.expand_dims(y_batch, -1)



def preprocess_to_characters(tweet_text):
    """
    Remove urls and turn tweet into a sequence of characters rather than words.
    :param tweet_text: str, text to preprocess
    :return: list of str, characters that compose the tweet
    """

    url_expression = re.compile(r'/^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/')
    tweet_text = re.sub(url_expression, '', tweet_text)
    tweet_text = list(tweet_text)
    tweet_text = ['<BEGIN_TWEET>'] + tweet_text + ['<END_TWEET>']

    return tweet_text



def preprocess_tweet(tweet_text):
    """
    Remove urls and turn @ and # symbols into seperate characters
    :param tweet_text: text to preprocess
    :return: preprocessed text
    """
    url_expression = re.compile(r'/^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/')
    tweet_text = re.sub(url_expression, '<URL>', tweet_text)
    tweet_text = ' '.join(re.split(r'(\W\s?)', tweet_text))
    tweet_text = re.sub(r'\s+', ' ', tweet_text)
    tweet_text = '<BEGIN_TWEET> ' + tweet_text + ' <END_TWEET>'

    return tweet_text



def sentence_to_target(words):
    """
    Converts a sentence into a list of sequence/target pairs : "My dog Rover" becomes [(["My"], "dog"), (["My", "dog"], "Rover)
    :param sentence: str, sentence to convert
    :return: list of sequence/target pairs
    """
    to_return = []
    for k in range(len(words)):
        init_seq = [w for w in words[:k]]
        target = words[k]
        to_return.append((init_seq, target))

    return to_return



def select_batch(sequences, idx):
    """ Select sequences at indexes idx and return [(sequence, offset sequence)] for training """

    sequences = np.array(sequences)[idx]
    training = [seq[:-1] for seq in sequences]
    target = [seq[1:] for seq in sequences]

    return (training, target)



def pad_batch(X_batch, max_seq_length=55):
    """ Pad a batch in a series with 0s to all be same sequence length"""
    if not max_seq_length:
        max_seq_length = max([len(x) for x in X_batch])
    needs_padding = np.where([len(x) < max_seq_length for x in X_batch], True, False)



    X_batch.loc[needs_padding] = X_batch.loc[needs_padding].apply(
        lambda x: np.vstack((x, np.zeros(shape=(max_seq_length - x.shape[0], emb_size))))
    )

    return X_batch


def build_rnn_model(vocab_size, embedding_size, rnn_size, stateful, batch_size=None):
    """
    Build an rnn model.
    :param vocab_size: int, size of vocabulary. Will give input size for embedding layer
    :param embedding_size: int, number of dimensions to embed model to
    :param rnn_size: int, size of rnn output representation
    :param stateful: bool, whether the rnn should save its internal state. False for training, True for inference
    :param batch_size: int or None, gives dimension size for tensorflow placeholder
    :return: keras.Model
    """

    source = tf.keras.layers.Input(shape=(None,), batch_size=batch_size, dtype=tf.int32)
    input = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)(source)
    lstm = tf.keras.layers.LSTM(units=rnn_size, stateful=stateful, return_sequences=True)(input)
    lstm = tf.keras.layers.LSTM(units=rnn_size, stateful=stateful, return_sequences=True)(lstm)

    predicted_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=vocab_size, activation='softmax'))(lstm)

    model = tf.keras.Model(source, predicted_char)
    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01), loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_crossentropy'])
    return model