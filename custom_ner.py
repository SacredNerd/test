import spacy
import random

# Load a pre-trained model like 'en_core_web_md' as a base model
nlp = spacy.load('en_core_web_md')

# Add a new NER label for car model names
nlp.entity.add_label('CAR_MODEL')


import spacy
import pandas as pd

# Load a pre-trained SpaCy model
nlp = spacy.load('en_core_web_md')

# Define your training data as a list of tuples
your_training_data = []

# Assuming you have a dataframe 'news_df' with articles and 'car_df' with car models
for article in news_df['articles']:
    doc = nlp(article)
    entities = []
    for car_model in car_df['models']:
        if car_model in doc.text:
            start = doc.text.index(car_model)
            end = start + len(car_model)
            entities.append((start, end, 'CAR_MODEL'))
    your_training_data.append((doc.text, {'entities': entities}))

# Now, 'your_training_data' contains the training data with text and car model annotations



# Prepare your training data
train_data = [(text, {"entities": entities}) for text, entities in your_training_data]

# Configure training settings
n_iter = 100
dropout = 0.5
learning_rate = 0.001
batch_size = 32

# Train the NER model
for i in range(n_iter):
    random.shuffle(train_data)
    losses = {}
    
    for batch in spacy.util.minibatch(train_data, size=batch_size):
        texts, annotations = zip(*batch)
        nlp.update(texts, annotations, drop=dropout, losses=losses)
    
    print("Losses after iteration", i, ":", losses)

# Save the trained model
nlp.to_disk('custom_car_model_ner')

# Test your trained model
test_text = "I saw a Ford Mustang and a Toyota Camry on the street."
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.label_)

# Evaluate the model's accuracy on a test dataset
# (You'll need a separate test dataset with annotated car model names)
