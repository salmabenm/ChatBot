import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data: #training file
    intents = json.load(json_data)

FILE = "data.pth" #save file
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) # loads the state dictionary model_state into my model. The state dictionary contains the learned parameters (weights and biases) of my model.
model.eval() #sets my model to evaluation mode. In evaluation mode, the model behaves differently from training mode.

bot_name = "Salma" #my bot name
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ") #the text enter by an user
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0]) #shape[0] nbr of colums
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()] #class label

    probs = torch.softmax(output, dim=1) #for the prob of each outputs
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']: #loop sur le file intents
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")