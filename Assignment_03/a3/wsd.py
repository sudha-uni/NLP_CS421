# CS421: Natural Language Processing
# University of Illinois at Chicago
# Spring 2025
# Assignment 3
#
# Do not rename/delete any functions or global variables provided in this template. Write your implementation
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that test code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment, you will need to obtain approval from the course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor

#Function to load and return sentences from the semcor library
#num: the number of sentences to load
#Returns: first num sentences from semcor
def load_sentences(num):
    #YOUR CODE HERE
    sents = semcor.sents()[:num]
    return sents

#Function to load and return tagged sentences from the semcor library
#num: the number of sentences to load
#Returns: first num tagged sentences from semcor
def load_tagged_sents(num):
    #YOUR CODE HERE
    labels = semcor.tagged_sents(tag='sem')[:num]
    return labels


#DO NOT MODIFY THIS FUNCTION
#Function to process the labels and get the wordnet synset for words
#sentences: the tagged sentences to be processed
#Returns: first num tagged sentences from semcor
def process_labels(sentences):
    sents = []
    labels = []
    for sent in sentences:  # Iterate through the first 5 sentences for demonstration
        curr_sent = []
        curr_labels = []
        sense_words = []
        for word in sent:
            if isinstance(word, nltk.Tree):  # Check if it is a tree (has sense)
                lemma = word.label()  # Get the sense label
                text = "_".join(word.leaves())  # Get the word(s) corresponding to this sense
                try:
                    if 'group.n.' not in lemma.synset().name(): # Do not add if it is a group of proper nouns
                        curr_sent.append(text)
                        curr_labels.append(lemma.synset().name())
                except:
                    curr_sent.append(text)
                    curr_labels.append(lemma)
        sents.append(curr_sent)
        labels.append(curr_labels)
    return sents, labels


#Function to get the word sense for a given word using the most frequent word sense in wordnet
#word: the word for which the sense is to be calculated
#Returns: the name of the synset for the calculated sense
def most_freq_sense_model(word):
    sense = None
    #YOUR CODE HERE
    synsets = wn.synsets(word)
    if synsets:
        sense = synsets[0].name()
        return sense
    return None

#Function to run the most_freq_sense model on all sentences
#sentences: List of list of strings containing words which make up the sentences to get predictions on
#Return: list of list of predicted senses
def get_most_freq_predictions(sentences):
    #YOUR CODE HERE
    preds = [[most_freq_sense_model(word) for word in sentence] for sentence in sentences]
    return preds



#Function to get the word sense for a given word using the most frequent word sense in wordnet
#word: the word for which the sense is to be calculated
#sentence: the sentence in which the word is used
#Returns: the name of the synset for the calculated sense
def lesk_model(word, sentence):
    best_sense = None
    #YOUR CODE HERE
    max_overlap = 0
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(nltk.corpus.stopwords.words('english'))
    context = set(word.lower() for word in sentence if word.lower() not in stop_words)
    synsets = wn.synsets(word)
    if synsets:
        best_sense = synsets[0].name()

    for synset in synsets:
        signature = set(synset.definition().lower().split())
        for example in synset.examples():
            signature.update(example.lower().split())
        signature -= stop_words

        overlap = len(context & signature)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = synset.name()

    return best_sense


#Function to run the lesk on all sentences
#sentences: List of list of strings containing words which make up the sentences to get predictions on
#Return: list of list of predicted senses
def get_lesk_predictions(sentences):
    preds = []
    #YOUR CODE HERE
    for sentence in sentences:
        sentence_preds = [lesk_model(word, sentence) for word in sentence]
        preds.append(sentence_preds)

    return preds

#Function to evaluate the predictions
#labels: List of list of strings containing the actual senses
#predicted: List of list of strings containing the predicted senses
#Return: precision, recall and f1 score
def evaluate(labels, predicted):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_actual = 0
    total_predicted = 0
    p = None
    r = None
    f1 = None
    #YOUR CODE HERE
    for true_labels, predicted_labels in zip(labels, predicted):
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            if true_label == predicted_label:
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1

            total_actual += 1
            total_predicted += 1

    if total_predicted != 0:
        p = true_positive / total_predicted
    else:
        p = 0.0

    if total_actual != 0:
        r = true_positive / total_actual
    else:
        r = 0.0

    if (p + r) != 0:
        f1 = 2 * (p * r) / (p + r)
    else:
        f1 = 0.0
    
    return p, r, f1

# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. If you want, you may run the code from the terminal as:
# python wsd.py

def main():
    # Download WordNet and SemCor data if not already downloaded
    nltk.download('wordnet')
    nltk.download('semcor')
    #Load the sentences and tagged sentences
    sents = load_sentences(50)
    tagged_sents = load_tagged_sents(50)
    #Process the tagged sentences to get the labels
    processed_sentences, labels = process_labels(tagged_sents)
    #Get the predictions using most frequent sense model
    preds_mfs = get_most_freq_predictions(processed_sentences)
    #Evaluate the predictions on the most frequent sense model
    print(evaluate(labels, preds_mfs))
    #Get the predictions using lesk model
    preds_lesk = get_lesk_predictions(processed_sentences)
    #Evaluate the predictions on the lesk model
    print(evaluate(labels, preds_lesk))
    
if __name__ == '__main__':
    main()