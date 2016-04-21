import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`\t]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    # Load data from files
    train_examples = list(open("./data/msr_paraphrase_train.txt").readlines())
    train_examples = [s.strip() for s in train_examples]
    test_examples = list(open("./data/msr_paraphrase_test.txt").readlines())
    test_examples = [s.strip() for s in test_examples]

    # process train_text
    train_text = [clean_str(sent) for sent in train_examples]
    train_text_split = [s.split("\t") for s in train_text]

    train_text_removeid = []
    train_labels = []
    for train_element in train_text_split:
        tem = []
        if int(train_element[0]) == 0:
            train_labels.append([0])
        else:
            train_labels.append([0])
        tem.append(train_element[3])
        tem.append(train_element[4])
        train_text_removeid.append(tem)
    # spilt the text
    train_text_removeid_split = []
    for ele in train_text_removeid:
        temp_ls = []
        for ele_ele in ele:
            tpt = ele_ele.split(" ")
            for i in range(len(tpt)):
                ind = tpt[i].rfind('.')
                if ind > -1:
                    tpt[i] = tpt[i][:ind]
            temp_ls.append(tpt)
        train_text_removeid_split.append(temp_ls)

    # process test_text
    test_text = [clean_str(sent) for sent in test_examples]
    test_text = [s.split("\t") for s in test_text]
    test_text_removeid = []
    test_labels = []
    for test_element in test_text:
        tem = []
        # test_labels.append([int(test_element[0])])
        if int(test_element[0]) == 0:
            test_labels.append([0])
        else:
            test_labels.append([0])
        tem.append(test_element[3])
        tem.append(test_element[4])
        test_text_removeid.append(tem)
    # spilt the text
    test_text_removeid_split = []
    for ele in test_text_removeid:
        temp_ls = []
        for ele_ele in ele:
            tpmt = ele_ele.split(" ")
            for i in range(len(tpmt)):
                inde = tpmt[i].rfind('.')
                if inde > -1:
                    tpmt[i] = tpmt[i][:inde]
            temp_ls.append(tpmt)
            # temp_ls.append(ele_ele.split(" "))
        test_text_removeid_split.append(temp_ls)

    return [train_text_removeid_split, np.array(train_labels), test_text_removeid_split, np.array(test_labels)]


def pad_sentences(train_text_removeid_split, test_text_removeid_split, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    # the max length of the sentence in the trainning set and testing set
    sequence_length = 0
    for ele_a in train_text_removeid_split:
        for ele_a_a in ele_a:
            if len(ele_a_a) > sequence_length:
                sequence_length = len(ele_a_a)
    for ele_ab in test_text_removeid_split:
        for ele_ab_a in ele_ab:
            if len(ele_ab_a) > sequence_length:
                sequence_length = len(ele_ab_a)
    # print 'sequence_length:', sequence_length
    # pad training dataset
    train_trs_padded = []
    for ele_b in train_text_removeid_split:
        tem_sentence = []
        for ele_b_a in ele_b:
            num_padding = sequence_length - len(ele_b_a)
            new_sentence = ele_b_a + [padding_word] * num_padding
            tem_sentence.append(new_sentence)
        train_trs_padded.append(tem_sentence)

    # pad testing dataset
    test_trs_padded = []
    for ele_c in test_text_removeid_split:
        temp_sentence = []
        for ele_c_a in ele_c:
            num_padding = sequence_length - len(ele_c_a)
            new_sentence = ele_c_a + [padding_word] * num_padding
            temp_sentence.append(new_sentence)
        test_trs_padded.append(temp_sentence)
    return [sequence_length, train_trs_padded, test_trs_padded]


def build_vocab(train_trs_padded, test_trs_padded):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # build vocabulary_dict from pre-trained word2vec
    lines = list(open("./data/dict.txt").readlines())
    lines_split = [line.split("\n") for line in lines]
    lines_processed = [line[0] for line in lines_split]
    # for line in lines:
    #     print 'line:', line
    vocabulary_dict = {x: i for i, x in enumerate(lines_processed)}

    # train dataset
    trs_padded = train_trs_padded + test_trs_padded
    trsp_resu = itertools.chain(*trs_padded)
    trsp_result = itertools.chain(*trsp_resu)
    trsp_word_counts = Counter(trsp_result)
    # Mapping from index to word, after the operation bellow: vocabulary_inv=['a','cat',...,'wordn']
    trsp_vocabulary_inv = [x[0] for x in trsp_word_counts.most_common()]

    return [vocabulary_dict, trsp_vocabulary_inv]


def build_input_data(train_trs_padded, train_labels, test_trs_padded, test_labels, vocabulary_dict):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """

    # training dataset
    train_trsp_voca = []
    for ele_c in train_trs_padded:
        temp = []
        for ele_c_a in ele_c:
            tem = []
            for word in ele_c_a:
                try:
                    # print 'word---', word
                    # print 'vocabulary_dict[word]---', vocabulary_dict[word]
                    tem.append(vocabulary_dict[word])
                except KeyError:
                    tem.append(vocabulary_dict['<PAD/>'])
            temp.append(tem)
        train_trsp_voca.append(temp)
    train_x = np.array(train_trsp_voca)
    train_y = np.array(train_labels)

    # testing dataset
    test_trsp_voca = []
    for ele_d in test_trs_padded:
        temp = []
        for ele_d_a in ele_d:
            tem = []
            for word in ele_d_a:
                try:
                    tem.append(vocabulary_dict[word])
                except KeyError:
                    tem.append(vocabulary_dict['<PAD/>'])
            temp.append(tem)
        test_trsp_voca.append(temp)
    test_x = np.array(test_trsp_voca)
    test_y = np.array(test_labels)

    return [train_x, train_y, test_x, test_y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    train_text_removeid_split, train_labels, test_text_removeid_split, test_labels = load_data_and_labels()
    sequence_length, train_trs_padded, test_trs_padded = pad_sentences(train_text_removeid_split, test_text_removeid_split)
    vocabulary_dict, trsp_vocabulary_inv = build_vocab(train_trs_padded, test_trs_padded)
    train_x, train_y, test_x, test_y = build_input_data(train_trs_padded, train_labels, test_trs_padded, test_labels, vocabulary_dict)
    return [train_x, train_y, test_x, test_y, vocabulary_dict, trsp_vocabulary_inv, sequence_length]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
