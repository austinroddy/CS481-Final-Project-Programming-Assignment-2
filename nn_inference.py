
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import json

if __name__ == "__main__":

    sentence = input('predict a sentence, positive or negative review: ')
    sentence = sentence.lower()
    new_sentence = []
    new_sentence.append(''.join([ch for ch in sentence if ch.isalpha() or ch == ' ']))
    new_sentence = new_sentence[0].split()

    sentence = [' '.join(new_sentence)]


    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    sentence = tokenizer.texts_to_sequences(sentence)
    sentence = pad_sequences(sentence, maxlen=2000)


    model = load_model('steam_reviews.h5')
    predict = model.predict(sentence)
    print('Is this a positive or negative review?')
    if predict[0, 0] >= 0.5:
        print('This is a positive review')
    else:
        print('This is a negative review')
    confidence = abs((2*predict[0, 0]) - 1)
    print('Confidence: ' + str(confidence))
    print("Raw score: " + str(predict[0, 0]))