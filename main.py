import os
import csv
import numpy as np
import tensorflow as tf
import pickle

import yaml
from tensorflow.keras import layers, activations, models, preprocessing, utils
from gensim.models import Word2Vec
import re
from matplotlib import pyplot

print("tensor version {}".format(tf.version.VERSION))

dir_path = "data"
files_list = os.listdir(dir_path + os.sep)

questions = list()
resp = list()
answers = list()
vocab = []

print("Archivos a entrenar: {}".format(files_list))
for filepath in files_list:
    if filepath.endswith(".yaml"):
        stream = open(dir_path + os.sep + filepath, "rb")
        docs = yaml.safe_load(stream)
        conversations = docs["conversaciones"]
        for con in conversations:
            if len(con) > 2:
                pregunta = con[0]
                pregunta = pregunta.lower()
                pregunta = re.sub("[^a-zA-Z]", " ", pregunta)
                questions.append(pregunta)
                res = con[1:]

                ans = ""
                for rep in res:
                    r = str(rep).lower()
                    r = re.sub("[^a-zA-Z]", " ", rep)
                    ans += " " + rep
                resp.append(ans)
            elif len(con) > 1:
                questions.append(con[0])
                resp.append(con[1])


answers_with_tags = list()
for i in range(len(resp)):
    if type(resp[i]) == str:
        print("snd: {}".format(resp[i]))
        answers_with_tags.append(resp[i])
    else:
        questions.pop(i)

for i in range(len(answers_with_tags)):
    answers.append("<START> " + answers_with_tags[i] + " <END>")

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1
print("vocab size : {}".format(VOCAB_SIZE))

for word in tokenizer.word_index:
    if "mediante" in word:
        print("aqui vocab")
    vocab.append(word)


def tokenize(sentences):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)

    for i, tok in enumerate(tokens_list):
        if tok == "mediante":
            print("aqui")
    return tokens_list, vocabulary


p = tokenize(questions + answers)
model = Word2Vec(p[0])

embedding_matrix = np.zeros((VOCAB_SIZE, 100))
for i in range(len(tokenizer.word_index)):
    try:
        embedding_matrix[i] = model[vocab[i]]
    except KeyError as identifier:
        print("palabra no en vocab: {}".format(identifier))

# encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])
padded_questions = preprocessing.sequence.pad_sequences(
    tokenized_questions, maxlen=maxlen_questions, padding="post"
)
encoder_input_data = np.array(padded_questions)
print(encoder_input_data.shape, maxlen_questions)

# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])
padded_answers = preprocessing.sequence.pad_sequences(
    tokenized_answers, maxlen=maxlen_answers, padding="post"
)
decoder_input_data = np.array(padded_answers)
print(decoder_input_data.shape, maxlen_answers)

# decoder_output_data
tokenized_answers = tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=maxlen_answers, padding="post"
    )
    onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
    decoder_output_data = np.array(onehot_answers)
    print(decoder_output_data.shape)

    # Saving all the arrays to storage
    np.save("enc_in_data.npy", encoder_input_data)
    np.save("dec_in_data.npy", decoder_input_data)
    np.save("dec_tar_data.npy", decoder_output_data)

encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(
    encoder_inputs
)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True)(
    encoder_embedding
)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(
    decoder_inputs
)
decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(
    VOCAB_SIZE, activation=tf.keras.activations.softmax
)
output = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data,
    batch_size=32,
    epochs=125,
)

# print(history.history['loss'])
pyplot.plot(history.history["accuracy"])
pyplot.show()
model.save("model.h5")


def make_inference_models():
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model


def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(tokenizer.word_index[word])
    return preprocessing.sequence.pad_sequences(
        [tokens_list], maxlen=maxlen_questions, padding="post"
    )


enc_model, dec_model = make_inference_models()

for _ in range(10):
    states_values = enc_model.predict(str_to_tokens(input("Enter question : ")))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index["start"]
    stop_condition = False
    decoded_translation = ""
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += " {}".format(word)
                sampled_word = word

        if sampled_word == "end" or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation)
