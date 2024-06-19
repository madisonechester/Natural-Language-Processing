import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tensorflow import keras
import matplotlib.pyplot as plt
from skseq import sequence_list_c
from sklearn.metrics import f1_score
from keras.utils import pad_sequences
from tensorflow.keras import Model, Input
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from skseq.sequence_list import SequenceList
from skseq.label_dictionary import LabelDictionary
from tensorflow.keras.layers import InputLayer, Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, TimeDistributed



def get_data_target_sets(data):
    """
    Extracts sentences and tags from the provided data object based on sentence IDs.

    Args:
        data: A data object containing sentences and corresponding tags.

    Returns:
        A tuple (X, y) containing lists of sentences (X) and tags (y).
    """
    
    X = []  # the sentences
    y = []  # the tags

    # get unique sentence IDs from the data
    ids = data.sentence_id.unique()

    # use tqdm to create a progress bar
    progress_bar = tqdm(ids, desc="Processing", unit="sentence")

    # iterate over each unique sentence ID
    for sentence_id in progress_bar:
        # filter rows for the current sentence ID
        sentence_data = data[data["sentence_id"] == sentence_id]
        
        # retrieve words and tags for the current sentence ID
        words = list(sentence_data["words"].values)
        tags = list(sentence_data["tags"].values)
        
        # append the processed words and tags to X and y
        X.append(words)
        y.append(tags)

    return X, y  # return the lists of sentences and tags



def create_corpus(sentences, tags):
    """
    Creates a corpus by generating dictionaries for words and tags in the given sentences and tags.

    Args:
        sentences: A list of sentences.
        tags: A list of corresponding tags for the sentences.

    Returns:
        A tuple (word_dict, tag_dict, tag_dict_rev) containing dictionaries for words, tags,
        and a reversed tag dictionary.

    Example:
        sentences = [['I', 'love', 'Python'], ['Python', 'is', 'great']]
        tags = ['O', 'O', 'B']
        word_dict, tag_dict, tag_dict_rev = create_corpus(sentences, tags)
        # word_dict: {'I': 0, 'love': 1, 'Python': 2, 'is': 3, 'great': 4}
        # tag_dict: {'O': 0, 'B': 1}
        # tag_dict_rev: {0: 'O', 1: 'B'}
    """
    
    word_dict = {}  # unique words with corresponding indices
    tag_dict = {}  # unique tags with corresponding indices

    # generate word dictionary
    for sentence in sentences:
        for word in sentence:
            if word not in word_dict:
                word_dict[word] = len(word_dict)

    # generate tag dictionary
    for tag_list in tags:
        for tag in tag_list:
            if tag not in tag_dict:
                tag_dict[tag] = len(tag_dict)

    # reverse tag dictionary
    tag_dict_rev = {v: k for k, v in tag_dict.items()} 

    return word_dict, tag_dict, tag_dict_rev



def create_sequence_list(word_dict, tag_dict, X, y):
    """
    Creates a sequence list object by adding sequences from X and y to it without cython.

    Args:
        word_dict: A dictionary mapping words to their corresponding indices.
        tag_dict: A dictionary mapping tags to their corresponding indices.
        X: A list of input sequences (sentences).
        y: A list of corresponding target sequences (tags).

    Returns:
        A sequence list object populated with sequences from X and y.
    """
    
    seq = SequenceList(LabelDictionary(word_dict), LabelDictionary(tag_dict))

    # use tqdm to create a progress bar
    progress_bar = tqdm(range(len(X)), desc="Adding sequences", unit="sequence")

    for i in progress_bar:
        # add the sequence (X[i], y[i]) to the sequence list
        seq.add_sequence(X[i], y[i], LabelDictionary(word_dict), LabelDictionary(tag_dict))

    return seq



def create_sequence_listC(word_dict, tag_dict, X, y):
    """
    Creates a sequence list object by adding sequences from X and y to it using cython.

    Args:
        word_dict: A dictionary mapping words to their corresponding indices.
        tag_dict: A dictionary mapping tags to their corresponding indices.
        X: A list of input sequences (sentences).
        y: A list of corresponding target sequences (tags).

    Returns:
        A sequence list object populated with sequences from X and y.
    """
    
    seq = sequence_list_c.SequenceListC(LabelDictionary(word_dict), LabelDictionary(tag_dict))

    # use tqdm to create a progress bar
    progress_bar = tqdm(range(len(X)), desc="Adding sequences", unit="sequence")

    for i in progress_bar:
        # add the sequence (X[i], y[i]) to the sequence list
        seq.add_sequence(X[i], y[i], LabelDictionary(word_dict), LabelDictionary(tag_dict))

    return seq
    


def show_features(feature_mapper, seq, feature_type=["Initial features", "Transition features", "Final features", "Emission features"]):
    """
    Displays the features extracted from a sequence using a feature mapper.

    Args:
        feature_mapper: An object responsible for mapping feature IDs to feature names.
        seq: A sequence object containing the input sequence.
        feature_type: Optional. A list of feature types to display. Default is ["Initial features", "Transition features", "Final features", "Emission features"].

    Returns:
        None
    """
    
    inv_feature_dict = {word: pos for pos, word in feature_mapper.feature_dict.items()}

    for feat, feat_ids in enumerate(feature_mapper.get_sequence_features(seq)):
        # print the current feature type
        print(feature_type[feat]) 

        for id_list in feat_ids:
            for k, id_val in enumerate(id_list):
                # print the feature IDs and their corresponding names
                print(id_list, inv_feature_dict[id_val]) 

        print("\n")  # add a newline after printing all features of a certain type



def get_tiny_test():
    """
    Creates a tiny test dataset.

    Args:
        None

    Returns:
        A tuple (X, y) containing lists of sentences (X) and tags (y).
    """
    
    X = [['The programmers from Barcelona might write a sentence without a spell checker . '],
         ['The programmers from Barchelona cannot write a sentence without a spell checker . '],
         ['Jack London went to Parris . '],
         ['Jack London went to Paris . '],
         ['Bill gates and Steve jobs never though Microsoft would become such a big company . '],
         ['Bill Gates and Steve Jobs never though Microsof would become such a big company . '],
         ['The president of U.S.A though they could win the war . '],
         ['The president of the United States of America though they could win the war . '],
         ['The king of Saudi Arabia wanted total control . '],
         ['Robin does not want to go to Saudi Arabia . '],
         ['Apple is a great company . '],
         ['I really love apples and oranges . '],
         ['Alice and Henry went to the Microsoft store to buy a new computer during their trip to New York . ']]

    y = [['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'I-geo', 'I-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'I-geo', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O'],
            ['B-org', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'B-per', 'O', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo',
             'I-geo', 'O']]

    return [i[0].split() for i in X], y



def check_for_nan(sequence):
    """Check if a sequence contains NaN values."""
    if any(pd.isna(sequence)):
        raise ValueError(f"Input sequence contains NaN values: {sequence}")



def validate_input(sequence):
    """Validate the input sequence to ensure all elements are non-NaN and of valid types."""
    if not all(isinstance(x, (int, float, str)) and not pd.isna(x) for x in sequence):
        raise ValueError(f"Input sequence contains invalid or NaN values: {sequence}")



def handle_nan(sequence, replacement='None'):
    """Replace NaN values in a sequence with a specified replacement."""
    return [replacement if pd.isna(x) else x for x in sequence]



def predict_SP(model, X):
    """
    Predicts the tags for the input sequences using a StructuredPerceptron model.

    Args:
        model: A trained StructuredPerceptron model.
        X: A list of input sequences (sentences).

    Returns:
        A list of predicted tags for the input sequences.
    """
    
    y_pred = []
    
    # use tqdm to create a progress bar
    for i in tqdm(range(len(X)), desc="Predicting tags", unit="sequence"):
        sequence = X[i]
        # clean and validate the sequence
        sequence_cleaned = handle_nan(sequence)
        validate_input(sequence_cleaned)
        
        # predict the tags for the current input sequence
        predicted_tag = model.predict_tags_given_words(sequence_cleaned)
        y_pred.append(predicted_tag)

    y_pred = np.concatenate(y_pred).ravel().tolist()

    return y_pred



def accuracy(true, pred):
    """
    Computes the accuracy of predicted tags compared to true tags, excluding instances where true[i] == 'O'.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The accuracy score, which measures the proportion of correct predictions.
    """
    
    # get indexes of those that are not 'O'
    idx = [i for i, x in enumerate(true) if x != 'O']

    # get the true and predicted tags for those indexes
    true = [true[i] for i in idx]
    pred = [pred[i] for i in idx]

    # use sklearn's accuracy_score to compute the accuracy
    return accuracy_score(true, pred)

    

def plot_confusion_matrix(true, pred, tag_dict_rev):
    """
    Plots a confusion matrix using a different style.

    Args:
        true: A nested list or array of true labels (e.g., list of lists).
        pred: A nested list or array of predicted labels (e.g., list of lists).
        tag_dict_rev: A dictionary mapping tag values to their corresponding labels.

    Returns:
        None
    """

    # get all unique tag values from true and pred lists
    unique_tags = np.unique(np.concatenate((true, pred)))

    # create a tick label list with all unique tags
    tick_labels = [tag_dict_rev.get(tag, tag) for tag in unique_tags]

    # map tags to indices for confusion matrix consistency
    tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    true_indices = [tag2idx[tag] for tag in true]
    pred_indices = [tag2idx[tag] for tag in pred]

    # get the confusion matrix
    cm = confusion_matrix(true_indices, pred_indices, labels=range(len(unique_tags)))

    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(cm, cmap='coolwarm')
    fig.colorbar(cax)

    # annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # set axis labels
    ax.set_xticks(range(len(unique_tags)))
    ax.set_yticks(range(len(unique_tags)))
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    # set axis titles
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16)
    plt.show()



def f1_score_weighted(true, pred):
    """
    Computes the weighted F1 score based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The weighted F1 score.
    """
    
    # get the weighted F1 score using sklearn's f1_score function
    return f1_score(true, pred, average='weighted')


    
def evaluate(true, pred, tag_dict_rev):
    """
    Computes and prints evaluation metrics and displays a confusion matrix based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    
    # compute the accuracy and F1 score using predefined functions
    acc = accuracy(true, pred)
    f1 = f1_score_weighted(true, pred)

    # print the evaluation results
    print('Accuracy: {:.4f}'.format(acc))
    print('F1 Score: {:.4f}'.format(f1))

    # plot the confusion matrix
    plot_confusion_matrix(true, pred, tag_dict_rev)



def print_tiny_test_prediction(X, model, tag_dict_rev):
    """
    Prints the predicted tags for each input sequence.

    Args:
        X: A list of input sequences.
        model: The trained model used for prediction.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    
    y_pred = []
    for i in range(len(X)):
        # predict the tags for the current input sequence
        predicted_tag = model.predict_tags_given_words(X[i])
        y_pred.append(predicted_tag)

    for i in range(len(X)):
        sentence = X[i]
        tag_list = y_pred[i]
        prediction = ''
        for j in range(len(sentence)):
            prediction += sentence[j] + "/" + tag_dict_rev[tag_list[j]] + " "

        print(prediction + "\n")



#----------------------------------------------------------------------------------------------------------------------
#--------------------------------------------- DL Approach ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------



def preprocess_BiLSTM_train_data(df, max_len=128):
    """
    Preprocesses training data for a BiLSTM model.

    Args:
        df: DataFrame containing the training data.
        max_len: Maximum length of sequences (default: 128).

    Returns:
        X: Preprocessed input data (padded sequences of word indices).
        y: Preprocessed target data (padded sequences of tag indices).
        num_words: Total number of unique words in the training data.
        num_tags: Total number of unique tags in the training data.
        word2idx: Dictionary mapping words to their corresponding indices.
        tag2idx: Dictionary mapping tags to their corresponding indices.
    """

    # fill missing values in the "sentence_id" column with the previous non-null value
    df.loc[:, "sentence_id"] = df["sentence_id"].ffill()

    # define a lambda function to aggregate words and tags into tuples
    agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(), s["tags"].values.tolist())]

    # group the dataframe by "sentence_id" and apply the aggregation function
    sentences = df.groupby('sentence_id').apply(agg_func).tolist()

    # create a list of unique words in the training data and add an "ENDPAD" token
    words = list(dict.fromkeys(df["words"].values))
    words.append("ENDPAD")
    num_words = len(words)

    # create a list of unique tags in the training data
    tags = list(dict.fromkeys(df["tags"].values))
    num_tags = len(tags)

    # create a word-to-index dictionary
    word2idx = {w: i + 1 for i, w in enumerate(words)}  # Index starts from 1

    # create a tag-to-index dictionary
    tag2idx = {t: i for i, t in enumerate(tags)}

    # convert words to their corresponding indices using word2idx dictionary
    X = [[word2idx[w[0]] for w in s] for s in sentences]

    # pad the sequences of word indices to a fixed length
    # use the value num_words-1 for padding
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words - 1)

    # convert tags to their corresponding indices using tag2idx dictionary
    y = [[tag2idx[w[1]] for w in s] for s in sentences]

    # pad the sequences of tag indices to a fixed length
    # use the value of tag2idx["O"] for padding
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

    # return the processed data: X, y, num_words, num_tags, word2idx, tag2idx
    return X, y, num_words, num_tags, word2idx, tag2idx


    
def preprocess_BiLSTM_test_data(df, word2idx, tag2idx, num_words, max_len=128):
    """
    Preprocesses test data for a BiLSTM model.

    Args:
        df: DataFrame containing the test data.
        word2idx: Dictionary mapping words to their corresponding indices.
        tag2idx: Dictionary mapping tags to their corresponding indices.
        max_len: Maximum length of sequences (default: 128).

    Returns:
        X: Preprocessed input data (padded sequences of word indices).
        y: Preprocessed target data (padded sequences of tag indices).
    """

    # fill missing values in the "sentence_id" column with the previous non-null value
    df.loc[:, "sentence_id"] = df["sentence_id"].ffill()

    # define a lambda function to aggregate words and tags into tuples
    agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(), s["tags"].values.tolist())]

    # group the dataframe by "sentence_id" and apply the aggregation function
    sentences = df.groupby('sentence_id').apply(agg_func).tolist()

    # convert words to their corresponding indices using word2idx dictionary
    # ignore words not found in the dictionary (use 0 as the index)
    X = [[word2idx.get(w[0], 0) for w in s if w[0] in word2idx.keys()] for s in sentences]

    # pad the sequences of word indices to a fixed length
    # use the value num_words-1 for padding
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words - 1)

    # convert tags to their corresponding indices using tag2idx dictionary
    y = [[tag2idx[w[1]] for w in s] for s in sentences]

    # pad the sequences of tag indices to a fixed length
    # use the value of tag2idx["O"] for padding
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

    # return the processed data: X, y
    return X, y


    
def create_BiLSTM_model(num_words, max_len=128):
    """
    Creates a BiLSTM model.

    Args:
        num_words: Total number of unique words in the input data.
        max_len: Maximum length of sequences (default: 128).

    Returns:
        model: BiLSTM model.
    """

    # create a sequential model
    model = keras.Sequential()

    # add an input layer with the specified input shape
    model.add(InputLayer((max_len)))

    # add an embedding layer with the specified input dimension, output dimension, and input length
    model.add(Embedding(input_dim=num_words, output_dim=max_len, input_length=max_len))

    # add a spatial dropout layer with a dropout rate of 0.1
    model.add(SpatialDropout1D(0.1))

    # add a bidirectional LSTM layer with 100 units, returning sequences, and a recurrent dropout of 0.1
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))

    # compile the model with an optimizer, loss function, and metrics
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model



def accuracy_lstm(true, pred, tag2idx):
    """
    Computes the accuracy of predicted tags compared to true tags, excluding instances where true[i] == 'O'.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.
        tag2idx: Dictionary mapping tags to their corresponding indices.

    Returns:
        The accuracy score, which measures the proportion of correct predictions.
    """
    
    # flatten true and pred
    true_flat = [item for sublist in true for item in sublist]
    pred_flat = [item for sublist in pred for item in sublist]

    # get indexes of those that are not 'O' (or 0 in this case)
    idx = [i for i, x in enumerate(true_flat) if x != tag2idx['O']]

    # get the true and predicted tags for those indexes
    true_filtered = [true_flat[i] for i in idx]
    pred_filtered = [pred_flat[i] for i in idx]

    # use sklearn's accuracy_score to compute the accuracy
    return accuracy_score(true_filtered, pred_filtered)



def evaluate_lstm(true, pred, tag_dict_rev, tag2idx):
    """
    Computes and prints evaluation metrics and displays a confusion matrix based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.
        tag2idx: Dictionary mapping tags to their corresponding indices.

    Returns:
        None
    """
    
    # compute the accuracy
    acc = accuracy_lstm(true, pred, tag2idx)

    # flatten true and pred
    true_flat = [item for sublist in true for item in sublist]
    pred_flat = [item for sublist in pred for item in sublist]
    
    f1 = f1_score_weighted(true_flat, pred_flat)

    # print the evaluation results
    print('Accuracy: {:.4f}'.format(acc))
    print('F1 Score: {:.4f}'.format(f1))

    # plot the confusion matrix
    plot_confusion_matrix(true_flat, pred_flat, tag_dict_rev)



def predict_lstm(model, X):
    """
    Predicts labels using an LSTM model.

    Arguments:
    - model: The LSTM model used for prediction.
    - X: The input data to be used for prediction.

    Returns:
    - y_train_pred: The predicted labels based on the input data.
    """

    # predict using the LSTM model
    y_train_pred = model.predict(X)

    # get the index with the maximum probability as the predicted label
    y_train_pred = np.argmax(y_train_pred, axis=-1)

    # limit the predicted labels to a maximum of 16
    y_train_pred[y_train_pred > 16] = 16

    # return the predicted labels
    return y_train_pred



def get_tiny_test_lstm():
    """
    Creates a dataframe with the data from X_tiny and y_tiny.

    Returns:
    - df: A dataframe containing the sentence IDs, words, and tags.
    """

    # initialize empty lists for each column
    sentence_ids = []
    words = []
    tags = []

    # get X_tiny and y_tiny
    X_tiny, y_tiny = get_tiny_test()

    # iterate over each sentence and its corresponding tags
    for i, (sentence_tokens, tag_tokens) in enumerate(zip(X_tiny, y_tiny)):
        # extract words and tags for the current sentence
        sentence = sentence_tokens
        tags_list = tag_tokens

        # append words and tags to the respective lists
        words.extend(sentence)
        tags.extend(tags_list)
        # assign sentence ID to each word
        sentence_ids.extend([i + 1] * len(sentence))  

    # create a dictionary with the data
    data = {
        'sentence_id': sentence_ids,
        'words': words,
        'tags': tags
    }

    # create the dataframe
    df = pd.DataFrame(data)
    
    return df



def print_tiny_test_prediction_lstm(X_tiny, y_tiny_pred, word2idx, tag_dict_rev):
    """
    Prints the predictions for the X_tiny dataset using the LSTM model.

    Arguments:
    - X_tiny: The input data for prediction.
    - y_tiny_pred: The predicted tags for the input data.
    - word2idx: A dictionary mapping words to their corresponding indices.
    - tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    """

    # create a reversed vocabulary dictionary for mapping indices back to words
    reversed_vocabulary = {value: key for key, value in word2idx.items()}

    # iterate over the sentences and their predicted tags
    for i in range(0, 12):
        sentence = X_tiny[i]
        tags = y_tiny_pred[i]
        sentence_short = []

        # create a shortened sentence by stopping at the first occurrence of a point (value 22)
        for i in range(0, len(sentence)):
            sentence_short.append(sentence[i])
            if sentence_short[i] == 22:
                break

        # convert the sentence indices to word vectors using the reversed vocabulary
        word_vector = [reversed_vocabulary[position] for position in sentence_short]
        tags = tags[0:len(word_vector)]

        # convert the tag indices to tag vectors using the tag dictionary
        tags_vector = [tag_dict_rev[position] for position in tags]

        # print the token and its corresponding tag
        for token, tag in zip(word_vector, tags_vector):
            print(f'{token}/{tag}', end=' ')
            
        print('\n')
