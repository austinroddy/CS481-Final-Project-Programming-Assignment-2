from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as sw
import numpy as np

def normalize_sentence(sentence):
    ps = PorterStemmer()
    sentence = sentence.lower()
    new_sentence = []
    new_sentence.append(''.join([ch for ch in sentence if ch.isalpha() or ch == ' ']))
    new_sentence = new_sentence[0].split()

    stop_words = set(sw.words('english'))
    sentence_no_stop_words = []
    for i in new_sentence:
        if i not in stop_words:
            sentence_no_stop_words.append(i)

    sentence_stemmed = []

    for i in range(len(new_sentence)):
        sentence_stemmed.append(ps.stem(new_sentence[i]))

    return sentence_stemmed

def inference(sentence_precoessed,prob_pos, pos_denom, total_vocab_size, pos_nom,prob_neg, neg_denom, neg_nom, p_print=True):
    pos_value = 1
    neg_value = 1
    for word in sentence_precoessed:
        if word in pos_nom:
            pos_value = pos_value + np.log( (pos_nom[word]+1) / (pos_denom+total_vocab_size) )
        elif(word in neg_nom):
            pos_value = pos_value + np.log( 1 / (pos_denom+total_vocab_size) )
        if word in neg_nom:  
            neg_value = neg_value + np.log( ( neg_nom[word]+1) / (neg_denom+total_vocab_size) )
        elif(word in pos_nom):
            neg_value = neg_value + np.log( 1 / (neg_denom+total_vocab_size) )

    pos_value = pos_value + np.log(prob_pos)
    neg_value = neg_value + np.log(prob_neg)

    if p_print:
        print('Positive review prediction: ' + str(pos_value))
        print('Negative review prediction: ' + str(neg_value))

    if pos_value >= neg_value:
        return 1
    else:
        return -1

def load_model(file):
    model = open(file, 'r', encoding="utf8")
    Lines = model.readlines()
    count = 0
    wourd_count_dict = {}
    for line in Lines:
        if count == 0:
            class_probability = line
        elif count == 1:
            denominator = line
        elif count == 2:
            vocab_size = line
        else:
            line_split = line.split()
            if len(line_split) == 2:
                wourd_count_dict[line_split[0]] = int(line_split[1])
        count = count + 1

    return float(class_probability), int(denominator), int(vocab_size), wourd_count_dict


if __name__ == "__main__":
    #From a sentence to input
    sentence = input('predict a sentence, positive or negative review: ')
    sentence_normalized = normalize_sentence(sentence)
    #print(sentence_normalized)
    class_prob_pos, denom_pos, total_vocab_size, wourd_dict_pos = load_model('pos_prob_stemmed_stopwords.txt')
    class_prob_neg, denom_neg, total_vocab_size, wourd_dict_neg = load_model('neg_prob_stemmed_stopwords.txt')
    prediction = inference(sentence_normalized, class_prob_pos, denom_pos, total_vocab_size, wourd_dict_pos, class_prob_neg, denom_neg, wourd_dict_neg)
    if prediction == 1:
        print('This is a positive review')
    else:
        print('This is a negative review')

