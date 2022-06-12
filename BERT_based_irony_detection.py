import glob
from xml.dom.minidom import parse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, TFDistilBertModel, DistilBertConfig

from Transformer import Transformer

AUTOTUNE = tf.data.experimental.AUTOTUNE

DISTILBERT_DROPOUT = 0.2
DISTILBERT_ATT_DROPOUT = 0.2
LAYER_DROPOUT = 0.2
RANDOM_STATE = 42
MAX_LENGTH = 64


def make_dataset(file_name='pan22-author-profiling-training-2022-03-29'):
    xml_paths = glob.glob(file_name + '/*/*.xml')
    truth_path = glob.glob(file_name + '/*/*.txt')
    truth = open(truth_path[0], 'r').readlines()
    truth = [line.strip().split(':::') for line in truth]
    truth_dict = {line[0]: int(line[1] == 'I') for line in truth}
    sents, labels = [], []
    for idx, xml in enumerate(xml_paths):
        user = xml.split('\\')[-1].split('.')[0]
        DOMTree = parse(xml)
        collection = DOMTree.documentElement
        documents = collection.getElementsByTagName('document')
        labels += [truth_dict[user]]
        sents.append([])
        for document in documents:
            sent = document.childNodes[0].data
            sents[idx].append(sent)

    X_train, X_val, y_train, y_val = train_test_split(
        sents, labels, test_size=0.2, random_state=43)

    X_train = [line for id_ in X_train for line in id_]
    y_train = [label for label in y_train for i in range(200)]

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=43)

    return X_train, X_val, X_test, y_train, y_val, y_test


def batch_encode(tokenizer, texts, batch_size=256, max_length=MAX_LENGTH):
    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='longest',
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

X_train, X_val, X_test, y_train, y_val, y_test = make_dataset(
    file_name='pan22-author-profiling-training-2022-03-29'
)

X_train_ids, X_train_attention = batch_encode(tokenizer, X_train)
X_test_ids, X_test_attention = batch_encode(tokenizer, X_test)
flat_X_val = [line for id_ in X_val for line in id_]
val_ids, val_attention = batch_encode(tokenizer, flat_X_val)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)


def build_model(transformer, lr, max_length=MAX_LENGTH, use_lstm=False):
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=RANDOM_STATE)

    input_ids_layer = tf.keras.layers.Input(shape=(max_length,),
                                            name='input_ids',
                                            dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,),
                                                  name='input_attention',
                                                  dtype='int32')

    # (batch_size, sequence_length, hidden_size=768).
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]

    hidden_state = last_hidden_state[:, 0, :]

    if use_lstm:
        hidden_state = last_hidden_state

        hidden_state = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(1024, return_sequences=True)
        )(hidden_state)
        hidden_state = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(1024)
        )(hidden_state)

    hidden_state = tf.keras.layers.Dense(512,
                                         activation='relu',
                                         kernel_initializer=weight_initializer,
                                         kernel_constraint=None,
                                         bias_initializer='zeros'
                                         )(hidden_state)
    hidden_state = tf.keras.layers.Dropout(0.2)(hidden_state)
    hidden_state = tf.keras.layers.Dense(32,
                                         activation='relu',
                                         kernel_initializer=weight_initializer,
                                         kernel_constraint=None,
                                         bias_initializer='zeros'
                                         )(hidden_state)
    hidden_state = tf.keras.layers.Dropout(0.2)(hidden_state)

    output = tf.keras.layers.Dense(1,
                                   activation='sigmoid',
                                   kernel_initializer=weight_initializer,
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(hidden_state)

    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

    model.compile(tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(),
                  metrics=['accuracy'])

    return model


def compute_accuracy(pred, y):
    pred = np.mean(tf.reshape(pred, (-1, 200)), axis=-1)
    return ((pred >= 0.5).astype(int) == np.array(y)).sum() / len(pred)


def train(distilBERT, mode, use_lstm, lr, epochs, batch_size):
    num_steps = len(y_train) // batch_size

    for layer in distilBERT.layers:
        layer.trainable = (mode == 'fine-tuning')

    model = build_model(distilBERT, lr=lr, use_lstm=use_lstm)

    train_history1 = model.fit(
        x=[X_train_ids, X_train_attention],
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=num_steps,
        validation_data=([X_test_ids, X_test_attention], y_test)
    )

    preds = []
    for i in range(0, len(val_ids), 32):
        pred = model.predict([val_ids[i:i + 32], val_attention[i:i + 32]])
        preds.append(pred)

    return preds, compute_accuracy(preds, y_val)


NUM_LAYERS = 8
D_MODEL = 768
NUM_HEADS = 8
DFF = 2048

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 5e-5

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_ids, X_train_attention, y_train[:, tf.newaxis])).cache()
train_dataset = train_dataset.shuffle(len(y_train)).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_ids, X_test_attention, y_test[:, tf.newaxis])).cache()
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

vocab_size = tokenizer.vocab_size


def train_bert(use_lstm):
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=vocab_size,
        pe_input=vocab_size,
        use_lstm=use_lstm
    )

    transformer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    transformer.fit(
        x=[X_train_ids, X_train_attention],
        y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=len(y_train) // BATCH_SIZE,
        validation_data=([X_test_ids, X_test_attention], y_test)
    )

    preds = []
    for i in range(0, len(val_ids), 32):
        pred = transformer.predict([val_ids[i:i + 32], val_attention[i:i + 32]])
        preds.append(pred)

    return preds, compute_accuracy(preds, y_val)


if __name__ == '__main__':
    config = DistilBertConfig(dropout=DISTILBERT_DROPOUT,
                              attention_dropout=DISTILBERT_ATT_DROPOUT,
                              output_hidden_states=True)

    distilBERT = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

    # Model 1: Fine-Tuning distilBERT (LSTM classifier)
    preds1, acc1 = train(
        distilBERT,
        mode='fine-tuning',
        use_lstm=True,
        lr=5e-5,
        epochs=1,
        batch_size=64
    )
    print('Person-wise classification accuracy of Model 1: ', acc1)

    # Model 2: Fine-Tuning distilBERT (Dense classifier)
    preds2, acc2 = train(
        distilBERT,
        mode='fine-tuning',
        use_lstm=False,
        lr=5e-5,
        epochs=1,
        batch_size=64
    )
    print('Person-wise classification accuracy of Model 2: ', acc2)

    # Model 3: Transfer learning distilBERT (LSTM classifier)
    preds3, acc3 = train(
        distilBERT,
        mode='transfer learning',
        use_lstm=True,
        lr=5e-4,
        epochs=1,
        batch_size=64
    )
    print('Person-wise classification accuracy of Model 3: ', acc3)

    # Model 4: Transfer learning distilBERT (Dense classifier)
    preds4, acc4 = train(
        distilBERT,
        mode='transfer learning',
        use_lstm=False,
        lr=5e-3,
        epochs=1,
        batch_size=64
    )
    print('Person-wise classification accuracy of Model 4: ', acc4)

    # Model 5: BERT with LSTM Classifier
    preds5, acc5 = train_bert(use_lstm=True)
    print('Person-wise classification accuracy of Model 5: ', acc5)

    # Model 6: BERT with Dense Classifier
    preds6, acc6 = train_bert(use_lstm=False)
    print('Person-wise classification accuracy of Model 6: ', acc6)
