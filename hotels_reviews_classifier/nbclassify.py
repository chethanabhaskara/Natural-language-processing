import os
import json
import glob
import sys
import re
import math
import string
import fnmatch
import operator

def map_to_label(max_prob_class):
    class1 = ""
    class2 = ""
    if max_prob_class.find("deceptive")!= -1:
        class1 = "deceptive"
    else:
        class1 = "truthful"
    if max_prob_class.find("positive")!= -1:
        class2 = "positive"
    else:
        class2 = "negative"

    return class1+" "+class2

def calculate_prob(text, class_vocab, vocab_size, class_file_count, total_file_count):
    class_vocab_size = sum(class_vocab.itervalues())
    total_file_count *= 1.0
    class_file_count *= 1.0
    probability = 0
    for word in text:
        freq = 0
        if word in class_vocab:
            freq = class_vocab[word]
            # print freq, "  vocab size : ", vocab_size, " class_vocab_size: ", class_vocab_size
        probability += math.log(((freq + 1.0)/(vocab_size + class_vocab_size)))
    probability += math.log10((class_file_count/total_file_count))
    return probability

def preprocess(review_text):
    global stop_words

    review_text = review_text.lower()
    review_text = re.sub(r'\d+', '', review_text)
    review_text = review_text.translate(string.maketrans("", ""), string.punctuation).strip()
    review_text = review_text.strip()
    review_words_list = review_text.split()
    filtered_review_word_list = [word for word in review_words_list if not word in stop_words]
    return filtered_review_word_list

def process_file(file_path):
    with open(file_path) as review_file:
        review_data = review_file.read()
        review_data = preprocess(review_data)

    return review_data

if __name__ == '__main__':

    stop_words = ['all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'fifty', 'four', 'not', 'own', 'through',
         'yourselves', 'go', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere',
         'with', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'under', 'ours', 'has',
         'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very', 'de', 'none',
         'cannot', 'every', 'whether', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'several',
         'hereafter', 'always', 'who', 'cry', 'whither', 'this', 'someone', 'either', 'each', 'become', 'thereupon',
         'sometime', 'side', 'two', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up',
         'namely', 'towards', 'are', 'further', 'beyond', 'ourselves', 'yet', 'out', 'even', 'will', 'what', 'still',
         'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its', 'everything', 'behind', 'un', 'above',
         'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she', 'somehow', 'be', 'we', 'full', 'never', 'sixty',
         'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere', 'although', 'found', 'alone', 're', 'along',
         'fifteen', 'by', 'both', 'about', 'last', 'would', 'anything', 'via', 'many', 'could', 'thence', 'put',
         'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence', 'onto', 'or', 'con', 'among', 'already', 'co',
         'afterwards', 'formerly', 'within', 'seems', 'into', 'others', 'while', 'whatever', 'except', 'down', 'hers',
         'everyone', 'done', 'least', 'another', 'whoever', 'moreover', 'couldnt', 'throughout', 'anyhow', 'yourself',
         'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due', 'been', 'next', 'anyone', 'eleven', 'much',
         'call', 'therefore', 'interest', 'then', 'thru', 'themselves', 'hundred', 'was', 'sincere', 'empty', 'more',
         'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming', 'hereby', 'amongst', 'else', 'part',
         'everywhere', 'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made', 'twenty', 'these', 'bill',
         'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere', 'nine', 'can', 'of', 'toward', 'my',
         'something', 'and', 'whereafter', 'whenever', 'give', 'almost', 'wherever', 'is', 'describe', 'beforehand',
         'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt',
         'inc', 'thereby', 'thin', 'no', 'perhaps', 'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein',
         'beside', 'also', 'that', 'other', 'take', 'which', 'becomes', 'you', 'if', 'nobody', 'see', 'though', 'may',
         'after', 'upon', 'most', 'hereupon', 'eight', 'but', 'serious', 'nothing', 'such', 'your', 'why', 'a', 'off',
         'whereby', 'third', 'i', 'whole', 'noone', 'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather',
         'without', 'so', 'five', 'the', 'first', 'whereas', 'once']

    with open("nbmodel.txt", "r") as nbmodel_file:
        model_data = json.load(nbmodel_file)

    # Load all the learned data into data structs
    total_file_count = model_data["total_file_count"]
    vocab = model_data["vocab"]
    class_dicts = model_data["class_dicts"]
    class_file_count = model_data["class_file_count"]
    vocab_size = len(vocab)
    y_pred = []
    # class_enum = ["negative deceptive","negative truthful","positive truthful","positive truthful"]
    output_file = open("nboutput.txt","w")
    # read each file in the test set
    test_files_set = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

    for f in test_files_set:
        with open(f, 'r') as test_file:
            # read file
            test_review = test_file.read()
            test_word_list = preprocess(test_review)
            # calculateprob for each class
            MLEs = {}
            for cat, cat_vocab in class_dicts.items():
                cat_mle = calculate_prob(test_word_list, \
                                         cat_vocab, vocab_size, class_file_count[cat], total_file_count)
                MLEs[cat] = cat_mle
            max_prob_class = max(MLEs.iteritems(), key=operator.itemgetter(1))[0]
            label = map_to_label(max_prob_class)
            file_string = label + " " + f + "\n"
            output_file.write(file_string)

    output_file.close()
    print vocab_size
    for k, v in class_dicts.items():
        print k," : ", sum(v.values())
    for k, v in class_dicts.items():
        print k," : ", len(v.keys())


