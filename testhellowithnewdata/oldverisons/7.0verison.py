import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import random
import requests
import zipfile
import io
import wikipedia
import json
import string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import pipeline
import pickle
import logging
from datetime import datetime
import openai
import time
from bs4 import BeautifulSoup

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.system('clear')
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
os.system('clear')

openai.api_key = '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'

class DataStore:
    def __init__(self, filepath):
        self.filepath = filepath

    def save_data(self, data):
        try:
            with open(self.filepath, 'r') as file:
                existing_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        existing_data.append(data)

        with open(self.filepath, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def load_data(self):
        try:
            with open(self.filepath, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        
data_store = DataStore('conversation_history.json')
conversation_history = data_store.load_data()


#takes the json data and processes it into txt file
def json_to_text(data):
    text_lines = []
    for entry in data:
        user_input = entry["user_input"]
        bot_response = entry["bot_response"]
        text_lines.append(f"User: {user_input}\nBot: {bot_response}\n")
    return "\n".join(text_lines)


#logging fuction
logging.basicConfig(level=logging.INFO, filename='chatbot_metrics.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

def log_metrics(user_input, bot_response, response_time, reward):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"{timestamp}, User Input: {user_input}, Bot Response: {bot_response}, Response Time: {response_time}, Reward: {reward}")

def collect_human_feedback(conversation_history):
    feedback_data = []
    for conversation in conversation_history:
        if isinstance(conversation, dict):
            user_input = conversation.get('user_input', '')
            bot_response = conversation.get('bot_response', '')
            print("User:", user_input)
            print("Bot:", bot_response)
            corrected_response = input("Correct the bot's response if needed or press Enter to keep it: ")
            feedback_data.append((user_input, corrected_response if corrected_response else bot_response))
        elif isinstance(conversation, tuple):
            user_input, bot_response = conversation
            print("User:", user_input)
            print("Bot:", bot_response)
            corrected_response = input("Correct the bot's response if needed or press Enter to keep it: ")
            feedback_data.append((user_input, corrected_response if corrected_response else bot_response))
        else:
            print("Skipping invalid conversation:", conversation)
    return feedback_data


def train_with_feedback(chatbot, original_dataset, feedback_data, conversation_history, epochs):
    # Combine the original dataset, feedback data, and conversation history
    combined_dataset = original_dataset + feedback_data + conversation_history

    # Preprocess the combined dataset as you did with the original dataset
    input_texts, target_texts = zip(*combined_dataset)
    input_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in input_texts]
    target_sequences = [chatbot.preprocess_sentence(chatbot.start_token + ' ' + sentence + ' ' + chatbot.end_token) for sentence in target_texts]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=chatbot.max_length, padding='post')
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=chatbot.max_length, padding='post')
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)
    train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(32)

    # Train the model
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
        
    def check_response_relevance(self, user_input, bot_response):
        messages = [
            {"role": "system", "content": "You are a response classifying assistant."}, #"You are a helpful assistant."
            {"role": "user", "content": f"""
            Consider a simple conversational interaction between a user and a chatbot. The user says: "{user_input}". The chatbot responds: "{bot_response}".

            Based on this interaction, evaluate the chatbot's response according to the following criteria:
            1. Does it make sense within the context of a simple conversational interaction?
            2. Is it something a simple chatbot would likely say in response to the user input?
            3. Does the response directly address the user's input in a meaningful way?

            Please provide a simple "yes" if the response meets all three criteria, or "no" if it fails to meet any of the criteria. Additionally, if the response is somewhat relevant but lacks detail or clarity, please indicate this by saying "was not detailed enough" and briefly explain why.
            """}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Using the chat model
                messages=messages,
                max_tokens=50
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            return exit()

        # Extracting the text from the response  answer = response.choices[0].message['content'].strip()
        answer = response.choices[0].message['content'].strip()

        # Analyzing the response for relevance
        if "yes" in answer.lower():
            print(answer)
            print('')
            return True
        #else:
            #one line below is new
        
        if "no" in answer.lower(): # or "was not detailed enough" in answer.lower():
            print(answer)
            print('')
            return False
        
        #three lines below are new, moved the else statment from above if issue uncoment else and indent the 4 lines below it also changed the if to elif in the above line where old else was
        else:
            print(f"An error occurred")
            return exit()
              
    def evaluate_response(self, user_input, bot_response):
        # Get sentiment scores
        user_sentiment = sentiment_analyzer(user_input)[0]
        bot_sentiment = sentiment_analyzer(bot_response)[0]

        #this line is new
        is_response_relevant = self.check_response_relevance(user_input, bot_response)  # This function needs to be defined

        # Determine if the bot's sentiment aligns with the user's sentiment
        if user_sentiment['label'] == 'POSITIVE' and bot_sentiment['label'] == 'POSITIVE':
            if is_response_relevant == True: #new line delte if problem
            #if is_response_relevant:
            #if user_sentiment['label'] == 'POSITIVE' and bot_sentiment['label'] == 'POSITIVE':
                self.reward_score += 1
                print('The statement is relevent postive')
                return 1
            #this needs to work doffrenlently it needs to make sure both relv and confi are correct and then reward or if is_response_relevant or confidence >= 0.185:

        #if
        elif user_sentiment['label'] == 'NEGATIVE' and bot_sentiment['label'] == 'NEGATIVE':
            if is_response_relevant == True: #new line delte if problem
            #if is_response_relevant:
            #if user_sentiment['label'] == 'NEGATIVE' and bot_sentiment['label'] == 'NEGATIVE':
                self.reward_score += 1
                print('The statement is relevent negative')
                return 1
        
        #this down below is new delte if issue 
        elif is_response_relevant == True:
                self.reward_score += 1
                print('The statement is relevent and makes sense')
                return 1    

        #else:
        elif is_response_relevant == False: 
            # Penalize the bot if the sentiments do not align
            #if is_response_relevant == False: 
            if self.reward_score >= 0 or self.reward_score < 0:
                #if self.reward_score >= 0:     
                self.reward_score -= 1 # Negative reward   
                print("The statement is not good return false")
                return -1
        #return 0
    
    

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

    def generate_seq2seqresponse(self, input_sentence, num_candidates=5, temperature=0.6, top_k=30):   #orginal top k = 50 #orginal num_canadiates = 10
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

            # Apply temperature scaling
                predictions = predictions / temperature

            # Convert logits to probabilities
                predicted_probabilities = tf.nn.softmax(predictions[:, -1, :], axis=-1).numpy()[0]

            # Select top-k tokens
                top_k_indices = np.argsort(predicted_probabilities)[-top_k:]
                top_k_probs = predicted_probabilities[top_k_indices]

            # Normalize probabilities
                top_k_probs /= np.sum(top_k_probs)

            # Sample from the top k tokens
                predicted_id = np.random.choice(top_k_indices, p=top_k_probs)

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
        self.model.save_weights('/Users/blakeweiss/Desktop/testhellowithnewdata/model_weights.h5')
        #if issue use the save weights line
        #self.model.save('/Users/blakeweiss/Desktop/hello/model_weights', save_format="tf")

         
    def load_model(self):
        if os.path.exists('/Users/blakeweiss/Desktop/testhellowithnewdata/model_weights.h5'):
            # Recreate the model with the known vocab size
            self.model = Seq2SeqModel(self.vocab_size, self.embedding_dim, self.hidden_units)

            # Dummy call to the model to initialize the variables
            dummy_input = [tf.zeros((1, 1)), tf.zeros((1, 1))]
            self.model(dummy_input)

            # Load the weights
            self.model.load_weights('/Users/blakeweiss/Desktop/testhellowithnewdata/model_weights.h5')

#actions that can be taken
def action_time():
    current_time = time.strftime("%H:%M")
    return f"Bot: The current time is {current_time}"

def get_the_news():
    url = 'https://www.bbc.com/news'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all 'a' tags, which might contain headlines
    headlines_links = soup.find('body').find_all('a')
    unwanted_bs = set(['BBC World News TV', 'BBC World Service Radio',
                       'News daily newsletter', 'Mobile app', 'Get in touch'])

    # Use dictionaries to store headlines by category
    headlines_by_category = {}

    for link in headlines_links:
        headline_text = link.text.strip()
        if headline_text and headline_text not in unwanted_bs:
            # Extract the category from the link URL if available
            href = link.get('href')
            if href and '/' in href:
                url_parts = href.split('/')
                if len(url_parts) > 2:
                    category = url_parts[2].replace('-', ' ').title()
                    headlines_by_category.setdefault(category, []).append(headline_text)
    
    return headlines_by_category

def getPerson(user_input):

    wordList = user_input.lower().split()

    for i in range(0, len(wordList)):
        if i + 3 <= len(wordList) - 1 and wordList[i].lower() == 'who' and wordList[i+1].lower() == 'is':
            return wordList[i+2] + ' ' + wordList[i+3]    
                     
#if the paths exist then its not redownloaded
if os.path.exists("/Users/blakeweiss/Desktop/testhellowithnewdata/data/inputtexts.txt") and os.path.exists("/Users/blakeweiss/Desktop/testhellowithnewdata/data/outputtexts.txt"):
    print("Files already exist.")
else:
    #print("Missing data files.")
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
         open('/Users/blakeweiss/Desktop/testhellowithnewdata/data/inputtexts.txt', 'w', encoding='utf-8') as conv_file, \
         open('/Users/blakeweiss/Desktop/testhellowithnewdata/data/outputtexts.txt', 'w', encoding='utf-8') as ans_file:
        for line in f:
            parts = line.split(' +++$+++ ')
            conversation = eval(parts[-1].strip())
            for i in range(len(conversation) - 1):
                input_dialogue = dialogues[conversation[i]]
                target_dialogue = dialogues[conversation[i + 1]]
                conv_file.write(input_dialogue + '\n')
                ans_file.write(target_dialogue + '\n')

with open('/Users/blakeweiss/Desktop/testhellowithnewdata/data/inputtexts.txt', 'r', encoding='utf-8') as conv_file:
    input_texts = conv_file.readlines()
    

with open('/Users/blakeweiss/Desktop/testhellowithnewdata/data/outputtexts.txt', 'r', encoding='utf-8') as ans_file:
    target_texts = ans_file.readlines()
    

dataset = list(zip(input_texts, target_texts))

#sample a subset of the dataset
#sample_size = 200
sampled_dataset = (dataset)
# Load tokenizer and get vocab size

if os.path.exists('/Users/blakeweiss/Desktop/testhellowithnewdata/tokenizer.pickle'):
    with open('/Users/blakeweiss/Desktop/testhellowithnewdata/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        vocab_size = tokenizer.vocab_size
else:
    input_texts, target_texts = zip(*sampled_dataset)  # use sampled dataset here
    all_texts = input_texts + target_texts
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (text for text in all_texts), target_vocab_size=2**13)
    vocab_size = tokenizer.vocab_size
    with open('/Users/blakeweiss/Desktop/testhellowithnewdata/tokenizer.pickle', 'wb') as handle:
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
    embedding_dim = 200 #256 is default
    hidden_units = 256 #default 512
    max_length = 25 #default 20 new 30 #25 is average sentence length of data
    epochs = 20 #default 10
    epochsstart = 20 #old value was 35
    #start and train the chatbot
    chatbot = Chatbot(vocab_size, embedding_dim, hidden_units, tokenizer, start_token, end_token, max_length)

    initial_learning_rate = 0.004 #orginal value of initial learning rate was 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=634, decay_rate=0.998, staircase=True) #orginal decay steps 100000
    chatbot.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    input_sequences = [chatbot.preprocess_sentence(start_token + ' ' + sentence + ' ' + end_token) for sentence in input_texts]
    target_sequences = [chatbot.preprocess_sentence(start_token + ' ' + sentence + ' ' + end_token) for sentence in target_texts]
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_length, padding='post')
    target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_length, padding='post')
    #uncomment out below line with suflle method if porblem arrise that makes training more random with sequences
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).shuffle(len(input_sequences)).batch(32)
    #suffle method not included in below line reduces randomness of training of sequences
    train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(32)
    
    #old line is above new line below delte if issue all it does is change batch size from 32 to 8
    #train_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(8)




    if os.path.exists('/Users/blakeweiss/Desktop/testhellowithnewdata/model_weights.h5'):
        chatbot.load_model()
        print("Vocabulary Size:", vocab_size)
        #print vocab size for debug
    else:
        chatbot.modeltrain(train_dataset, epochsstart)
        print("Vocabulary Size:", vocab_size)
        #print vocab size for debug
        chatbot.save_model()

    

    #conversation_history = []
    data_store = DataStore('conversation_history.json')
    
    time.sleep(1)
    #os.system('clear')
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Vocabulary Size:", vocab_size)
            #print vocab size for debug
            chatbot.save_model()
            break
        #new
        # Example usage of training with feedback and historical data
        elif user_input.lower() == 'feedback':
            feedback_data = collect_human_feedback(conversation_history)
            epochs3 = 10
            train_with_feedback(chatbot, sampled_dataset, feedback_data, conversation_history, epochs3)
        
  #clear the history after using it for training
        elif user_input.lower() == 'train':
            epochtotrain = input("Number of epochs to train by: ")
            epochtotrain = int(epochtotrain)
            #train the model
            chatbot.modeltrain(train_dataset, epochtotrain)   
            
        elif user_input.lower() == 'save':
            chatbot.save_model()
            with open('/Users/blakeweiss/Desktop/testhellowithnewdata/tokenizer.pickle', 'wb') as handle:
                pickle.dump(chatbot.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                

        elif user_input.lower() in ["news", "new", "tell me today's news", "tell me the news", "whats the news", "whats the news today", "whats happening in the world"]:
            print("Bot: Here's what's happening in the news:")
            news_today_recent = get_the_news()
            for category, headlines in news_today_recent.items():
                print(f"\n{category} News:")
                for i, headline in enumerate(headlines, 1):
                    print(f"{i}. {headline}")
                    
        elif "time" in user_input.lower():
            timeofday = action_time()    
            print(timeofday)
            
        elif "who is" in user_input.lower():
                        person = getPerson(user_input)
                        wiki = wikipedia.summary(person, sentences = 2)
                        responsewhois = ("Bot: This is") + ' ' + wiki
                        response = responsewhois
                        print(responsewhois)    
                        
        elif user_input.lower() == 'save json to text':
            #This gets the input and response from json file and puts into a txt so its possible to go into input txt and output txt
            conversation_jsontotxt = data_store.load_data()
            text_data = json_to_text(conversation_jsontotxt)
            file_pathforjsontotxt = '/Users/blakeweiss/Desktop/testhellowithnewdata/data/conversation_data.txt'

            with open(file_pathforjsontotxt, 'w') as file:
                file.write(text_data)
        
        else:
            start_time = datetime.now()
            response, confidence = chatbot.generate_seq2seqresponse(user_input, temperature=0.6) # temp was 0.5
            reward = reward_system.evaluate_response(user_input, response)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            #log information
            log_metrics(user_input, response, response_time, reward)
                        
            
            print("Bot:", response.replace(start_token, '').replace(end_token, ''))
            print('')
            print("Reward for this response:", reward)
            print("Total reward:", reward_system.get_total_reward())   
            print("Confidence:", confidence)
            print("Response time:", response_time) #new line delete if issue amount of time it takes for bot to respond
            print('')
            #print("Bot:", response.replace(start_token, '').replace(end_token, ''))
            #new below delte if issue
            
            conversation_history.append((user_input, response))
            
            conversation_data = {
            'user_input': user_input,
            'bot_response': response,
            'reward': reward  # Ensure you have a reward mechanism in place
            }            

            data_store.save_data(conversation_data)

            
            

        
        
        
