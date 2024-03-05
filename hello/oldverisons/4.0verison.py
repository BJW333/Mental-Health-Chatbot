import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import random
import requests
import zipfile
import io
import string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import pipeline
import pickle
import logging
from datetime import datetime

os.system('clear')
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
os.system('clear')

#logging fuction
logging.basicConfig(level=logging.INFO, filename='chatbot_metrics.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

def log_metrics(user_input, bot_response, response_time, reward):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"{timestamp}, User Input: {user_input}, Bot Response: {bot_response}, Response Time: {response_time}, Reward: {reward}")

def collect_human_feedback(conversation_history):
    feedback_data = []
    for user_input, bot_response in conversation_history:
        print("User:", user_input)
        print("Bot:", bot_response)
        corrected_response = input("correct the bot's response if needed or press Enter to keep it: ")
        feedback_data.append((user_input, corrected_response if corrected_response else bot_response))
    return feedback_data

def train_with_feedback(chatbot, original_dataset, feedback_data, epochs):
    #combine the original dataset with the feedback data
    combined_dataset = original_dataset + feedback_data

    #preprocess the combined dataset as you did with the original dataset
    input_texts, target_texts = zip(*combined_dataset)
    input_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in input_texts]
    target_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in target_texts]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=chatbot.max_length, padding='post')
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=chatbot.max_length, padding='post')
    train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)

    #train the model
    chatbot.modeltrain(train_dataset, epochs)
#new methods above delte if issue

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
class DynamicRewardSystem:
    def __init__(self):
        self.reward_score = 0

    def evaluate_response(self, user_input, bot_response):
        # Get sentiment scores
        user_sentiment = sentiment_analyzer(user_input)[0]
        bot_sentiment = sentiment_analyzer(bot_response)[0]

        # Determine if the bot's sentiment aligns with the user's sentiment
        if user_sentiment['label'] == 'POSITIVE' and bot_sentiment['label'] == 'POSITIVE':
            self.reward_score += 1  # Positive reward
            return 1
        elif user_sentiment['label'] == 'NEGATIVE' and bot_sentiment['label'] == 'NEGATIVE':
            self.reward_score += 1  # Positive reward
            return 1
        else:
            # Penalize the bot if the sentiments do not align
            if self.reward_score > 0:
                self.reward_score -= 1  # Negative reward
                return -1
        return 0 # Neutral, no reward

    def get_total_reward(self):
        return self.reward_score
        
class Seq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Seq2SeqModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.encoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True, dropout=0.5)
        self.attention = BahdanauAttention(hidden_units)
        self.decoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True, dropout=0.5)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_embeddings = self.embedding(encoder_inputs)
        encoder_outputs, state_h, state_c = self.encoder(encoder_embeddings)
        context_vector, _ = self.attention(state_h, encoder_outputs)
        decoder_embeddings = self.embedding(decoder_inputs)

        #repeat the context vector across the sequence length
        repeated_context_vector = tf.repeat(tf.expand_dims(context_vector, 1), repeats=decoder_inputs.shape[1], axis=1)

        #decoder embeddings with the repeated context vector
        decoder_input_with_context = tf.concat([decoder_embeddings, repeated_context_vector], axis=-1)

        #pass the concatenated input to the decoder
        decoder_outputs, _, _ = self.decoder(decoder_input_with_context, initial_state=[state_h, state_c])
        logits = self.fc(decoder_outputs)
        return logits


class Chatbot:
    def __init__(self, vocab_size, embedding_dim, hidden_units, tokenizer, start_token, end_token, max_length):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.vocab_size = vocab_size
        self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)
        self.tokenizer = tokenizer
        self.start_token = start_token
        self.end_token = end_token
        self.max_length = max_length
        self.optimizer = tf.keras.optimizers.Adam()
    
    def preprocess_sentence(self, sentence):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        encoded_sentence = [self.tokenizer.encode(self.start_token)[0]] + self.tokenizer.encode(sentence) + [self.tokenizer.encode(self.end_token)[0]]
        #ensure the sentence does not exceed max_length
        encoded_sentence = encoded_sentence[:self.max_length]
        #pad the sentence to max_length
        encoded_sentence = encoded_sentence + [0] * (self.max_length - len(encoded_sentence))
        return encoded_sentence

#deletes start and end tokens new mothod delte if not working
    def postprocess_sentence(self, sentence):
        if isinstance(sentence, int):
            sentence = [sentence]
        elif isinstance(sentence, list) and all(isinstance(i, int) for i in sentence):
            pass
        elif isinstance(sentence, list) and all(isinstance(i, list) for i in sentence):
            sentence = [item for sublist in sentence for item in sublist]
        else:
            sentence = sentence

        decoded_sentence = self.tokenizer.decode(sentence)
        #remove <start> and <end> tokens
        decoded_sentence = decoded_sentence.replace('start', '').replace('end', '').strip()
        return decoded_sentence


    def generate_seq2seqresponse(self, input_sentence, num_candidates=5, temperature=0.6): #was at temp 0.9
        input_sequence = self.preprocess_sentence(input_sentence)
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences([input_sequence], maxlen=self.max_length, padding='post')
        input_tensor = tf.convert_to_tensor(input_sequence)
        start_token = self.tokenizer.encode(self.start_token)[0]
        end_token = self.tokenizer.encode(self.end_token)[0]

        candidates = []
        for _ in range(num_candidates):
            decoder_input = tf.expand_dims([start_token], 0)
            response = []
            for _ in range(self.max_length):
                predictions = self.model([input_tensor, decoder_input])

                # use a softmax function to sample with temperature
                predictions = predictions / temperature
                predicted_probabilities = tf.nn.softmax(predictions[:, -1, :]).numpy()[0]
                predicted_id = np.random.choice(range(self.vocab_size), p=predicted_probabilities)

                if predicted_id == end_token:
                    break
                response.append(predicted_id)
                decoder_input = tf.concat([decoder_input, tf.expand_dims([predicted_id], 0)], axis=-1)
            candidates.append(response)

        #rerank candidates with language model
        reranked, confidence = self.rerank_candidates(input_sentence, candidates)

        return self.postprocess_sentence(reranked), confidence


    def rerank_candidates(self, input_sentence, candidates):
        scores = []
        for candidate in candidates:
            candidate_sentence = self.postprocess_sentence(candidate)
            input_context = self.gpt2_tokenizer.encode(input_sentence + ' ' + candidate_sentence, return_tensors='pt')
            with torch.no_grad():  #save memory
                output = self.gpt2_model(input_context, labels=input_context)
                loss, logits = output[:2]
            scores.append(loss.item())
    
    #lower loss is better
        best_candidate_idx = np.argmin(scores)

        confidence = 1 / scores[best_candidate_idx]
        return candidates[best_candidate_idx], confidence  #return the best candidate sentence, not its index

    
    def modeltrain(self, dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for (batch, (encoder_inputs, decoder_inputs)) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    logits = self.model([encoder_inputs, decoder_inputs])
                    loss = self.compute_loss(decoder_inputs[:, 1:], logits[:, :-1, :])
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                total_loss += loss
            print('Epoch {}, Loss {:.4f}'.format(epoch + 1, total_loss))
    
    def compute_loss(self, labels, logits):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        mask = tf.cast(mask, dtype=tf.float32)
        loss_value = loss(labels, logits, sample_weight=mask)
        return loss_value
    
    def save_model(self):
        self.model.save_weights('/Users/blakeweiss/Desktop/hello/model_weights.h5')
        #if issue use the save weights line
        #self.model.save('/Users/blakeweiss/Desktop/hello/model_weights', save_format="tf")

         
    def load_model(self):
        if os.path.exists('/Users/blakeweiss/Desktop/hello/model_weights.h5'):
            # Recreate the model with the known vocab size
            self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)

            # Dummy call to the model to initialize the variables
            dummy_input = [tf.zeros((1, 1)), tf.zeros((1, 1))]
            self.model(dummy_input)

            # Load the weights
            self.model.load_weights('/Users/blakeweiss/Desktop/hello/model_weights.h5')

        
#if the paths exist then its not redownloaded
if os.path.exists("/Users/blakeweiss/Desktop/hello/inputtexts.txt") and os.path.exists("/Users/blakeweiss/Desktop/hello/outputtexts.txt"):
    print("Files already exist. Skipping download.")
else:
    print("Downloading dataset...")
    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    response = requests.get(url)
    zipfile.ZipFile(io.BytesIO(response.content)).extractall()

    #download and extract the dataset
    dataset_dir = 'cornell movie-dialogs corpus'
    dialogues_file = os.path.join(dataset_dir, 'movie_lines.txt')
    conversations_file = os.path.join(dataset_dir, 'movie_conversations.txt')

#load the dialogues into a dictionary
    dialogues = {}
    with open(dialogues_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            dialogues[parts[0]] = parts[-1].strip()

#extract the conversations and save to files
    with open(conversations_file, 'r', encoding='iso-8859-1') as f, \
         open('/Users/blakeweiss/Desktop/hello/inputtexts.txt', 'w', encoding='utf-8') as conv_file, \
         open('/Users/blakeweiss/Desktop/hello/outputtexts.txt', 'w', encoding='utf-8') as ans_file:
        for line in f:
            parts = line.split(' +++$+++ ')
            conversation = eval(parts[-1].strip())
            for i in range(len(conversation) - 1):
                input_dialogue = dialogues[conversation[i]]
                target_dialogue = dialogues[conversation[i + 1]]
                conv_file.write(input_dialogue + '\n')
                ans_file.write(target_dialogue + '\n')

with open('/Users/blakeweiss/Desktop/hello/inputtexts.txt', 'r', encoding='utf-8') as conv_file:
    input_texts = conv_file.readlines()
    

with open('/Users/blakeweiss/Desktop/hello/outputtexts.txt', 'r', encoding='utf-8') as ans_file:
    target_texts = ans_file.readlines()
    

dataset = list(zip(input_texts, target_texts))

#sample a subset of the dataset
#sample_size = 200
sampled_dataset = (dataset)
# Load tokenizer and get vocab size

if os.path.exists('/Users/blakeweiss/Desktop/hello/tokenizer.pickle'):
    with open('/Users/blakeweiss/Desktop/hello/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        vocab_size = tokenizer.vocab_size
else:
    input_texts, target_texts = zip(*sampled_dataset)  # use sampled dataset here
    all_texts = input_texts + target_texts
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (text for text in all_texts), target_vocab_size=2**13)
    vocab_size = tokenizer.vocab_size
    with open('/Users/blakeweiss/Desktop/hello/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

reward_system = DynamicRewardSystem()

if __name__ == '__main__':
    #define your dataset and other parameters
    input_texts, target_texts = zip(*sampled_dataset)  #use sampled dataset here
    all_texts = input_texts + target_texts

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
(text for text in all_texts), target_vocab_size=2**13)

    start_token = '<start>'
    end_token = '<end>'
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256 #256 is default
    hidden_units = 512 #default 512
    max_length = 20 #default 20 new 100
    epochs = 20 #default 10
    epochsstart = 35
    #start and train the chatbot
    chatbot = Chatbot(vocab_size, embedding_dim, hidden_units, tokenizer, start_token, end_token, max_length)

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    chatbot.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    input_sequences = [chatbot.preprocess_sentence(start_token + ' ' + sentence + ' ' + end_token) for sentence in input_texts]
    target_sequences = [chatbot.preprocess_sentence(start_token + ' ' + sentence + ' ' + end_token) for sentence in target_texts]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_length, padding='post')
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_length, padding='post')
    train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))



    if os.path.exists('/Users/blakeweiss/Desktop/hello/model_weights.h5'):
        chatbot.load_model()
        print("Vocabulary Size:", vocab_size)
        #print vocab size for debug
    else:
        chatbot.modeltrain(train_dataset, epochsstart)
        print("Vocabulary Size:", vocab_size)
        #print vocab size for debug
        chatbot.save_model()

    
    #generate responses
    #new
    conversation_history = []
    #delete if issue

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Vocabulary Size:", vocab_size)
            #print vocab size for debug
            chatbot.save_model()
            break
        #new
        elif user_input.lower() == 'feedback':
            feedback_data = collect_human_feedback(conversation_history)
            epochs3 = 10
            train_with_feedback(chatbot, sampled_dataset, feedback_data, epochs3)
            conversation_history = []  #clear the history after using it for training
        elif user_input.lower() == 'train':
            #train the model
            chatbot.modeltrain(train_dataset, epochs)
        elif user_input.lower() == 'train2':
            #train the model
            epochs2 = 25
            chatbot.modeltrain(train_dataset, epochs2)    
        elif user_input.lower() == 'save':
            chatbot.save_model()
            with open('/Users/blakeweiss/Desktop/hello/tokenizer.pickle', 'wb') as handle:
                pickle.dump(chatbot.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
        else:
            start_time = datetime.now()
            response, confidence = chatbot.generate_seq2seqresponse(user_input, temperature=0.5)
            reward = reward_system.evaluate_response(user_input, response)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            #log information
            log_metrics(user_input, response, response_time, reward)
            
            print("Reward for this response:", reward)
            print("Total reward:", reward_system.get_total_reward())        
            print(confidence)
            print("Bot:", response.replace(start_token, '').replace(end_token, ''))
            #new below delte if issue
            conversation_history.append((user_input, response))
        
       
        
        
        
