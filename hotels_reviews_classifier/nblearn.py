import sys
import glob
import os
import collections
import re
import string
import json

def extract_vocab(words_list, class_vocab, gen_vocab):
    # print review_data
    for word in words_list:
        if word not in class_vocab:
            class_vocab[word] = 1
        else:
            class_vocab[word] += 1

        if word not in gen_vocab:
            gen_vocab[word] = 1
        else:
            gen_vocab[word] += 1

    return class_vocab, gen_vocab

def preprocess(review_text):
    global stop_words

    review_text = review_text.lower()
    review_text = re.sub(r'\d+', '', review_text)
    review_text = review_text.translate(string.maketrans("", ""), string.punctuation).strip()
    review_text = review_text.strip()
    review_words_list = review_text.split()
    filtered_review_word_list = [word for word in review_words_list if not word in stop_words]
    return filtered_review_word_list
    # return review_words_list

def process_file(file_path):
    with open(file_path) as review_file:
        review_data = review_file.read()
        words_list = preprocess(review_data)

    return words_list

def naive_bayes_learn(dir_name):
    total_file_count = 0
    all_files = glob.glob(os.path.join(dir_name, '*/*/*/*.txt'))
    files_by_class = collections.defaultdict(list)
    for f in all_files:
        class1, class2, fold, fname = f.split('\\')[-4:]
        files_by_class[class1 + class2].append(f)

    category = files_by_class.keys()

    class_file_count = {}
    for k, file_list in files_by_class.items():
        temp_count = len(file_list)
        class_file_count[k] =  temp_count
        total_file_count += temp_count

    class_dicts = {}
    vocab = {}
    total_word_count = 0
    for c, files in files_by_class.items():
        # class c1
        class_vocab = {}
        for f in files:
            words_list = process_file(f)
            # add the words into their classes along with their counts
            class_vocab, vocab = extract_vocab(words_list, class_vocab, vocab)

        class_dicts[c] = class_vocab
        total_word_count += sum(class_vocab.values())

    vocab_size = len(vocab.keys())

    model_data = {}
    model_data["total_file_count"] = total_file_count
    model_data["vocab"] = vocab
    model_data["class_dicts"] = class_dicts
    model_data["class_file_count"] = class_file_count
    #write the vocab and dicts to a file.
    with open("nbmodel.txt",'w') as nbmodel_file:
        nbmodel_file.write(json.dumps(model_data))


if __name__ == '__main__':
    ## List all files, given the root of training data.
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
    naive_bayes_learn(sys.argv[1])
    # naive_bayes_learn("op_spam_training_data")