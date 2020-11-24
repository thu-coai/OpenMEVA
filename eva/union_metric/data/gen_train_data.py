import numpy as np
np.random.seed(520)
import re
import copy
from nltk.corpus import stopwords
import nltk
pos_tag = nltk.pos_tag
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer().lemmatize
from eva.utils import tokenize
import sys
import json
import string
import os

def get_avail_phrases(special_token_list=[]):
    path = os.path.split(os.path.realpath(__file__))[0]
    punc_list = [punc for punc in string.punctuation]
    special_token_list += punc_list
    special_token_list += ["male", "female", "neutral", "[male]", "[female]", "[neutral]", "MALE", "FEMALE", "NEUTRAL", "[MALE]", "[FEMALE]", "[NEUTRAL]"]

    avail_phrases = {"all": set()}
    with open("%s/conceptnet_entity.csv"%path, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            avail_phrases["all"].add(' '.join(line.strip().split("|||")[:-1]))
        avail_phrases["all"] = avail_phrases["all"] - set(stopwords.words('english'))

    avail_phrases["negation"] = {}
    with open("%s/negation.txt"%path, 'r') as fin:
        negation_word = []
        for i, line in enumerate(fin):
            tmp_line = line.strip().split("|||")
            avail_phrases["all"].add(tmp_line[0].strip())
            avail_phrases["negation"][tmp_line[0].strip()] = tmp_line[1].strip()

    avail_phrases["special"] = set()
    if special_token_list is not None:
        for w in special_token_list:
            avail_phrases["all"].add(w)
            avail_phrases["special"].add(w)

    for key in ["all", "special"]:
        avail_phrases_ = {}
        for word in avail_phrases[key]:
            avail_phrases_[word] = 1
        avail_phrases[key] = avail_phrases_

    anotomy_word = {}
    with open("%s/conceptnet_antonym.txt"%path, "r", encoding='utf-8') as fin:
        for line in fin:
            tmp = line.strip().split("|||")
            if len(tmp) == 3:
                h, t = tmp[0], tmp[2].split()
                if h in anotomy_word:
                    anotomy_word[h] += t
                else:
                    anotomy_word[h] = t[:]

    return avail_phrases, anotomy_word

def build_vocab(data, tokenizer, output_dir, special_token_list=[]):
    avail_phrases, _ = get_avail_phrases(special_token_list=special_token_list)
    all_text = []
    for tmp_data in data:
        all_text.append(tmp_data['context'])
        origin_reference = tmp_data['reference']
        if isinstance(origin_reference, list):
            all_text += origin_reference
        else:
            all_text.append(origin_reference)

    vocab = {}
    print("number of all texts: ", len(all_text))
    for i, text in enumerate(all_text):
        if i % 10000 == 0:
            print("processing data, %d lines"%i)
        pos = pos_tag(tokenize(text.strip(), tokenizer))
        for word_pos in pos:
            if word_pos[0] in vocab:
                vocab[word_pos[0]]["number"] += 1
                if word_pos[1] in vocab[word_pos[0]]:
                    vocab[word_pos[0]][word_pos[1]] += 1
                else:
                    vocab[word_pos[0]][word_pos[1]] = 1
            else:
                vocab[word_pos[0]] = {word_pos[1]:1, "number":1}
    print("number of all vocab: ", len(vocab))
    new_vocab = copy.deepcopy(vocab)
    for i, v in enumerate(vocab):
        if i % 1000 == 0:
            print("building vocab, processing %d lines"%i)
        for p in vocab[v]:
            if lemma(v, 'v' if p[0] == 'V' else 'n') not in avail_phrases["all"]:
                del new_vocab[v]
                break

    with open("%s/entity_vocab.json"%output_dir, "w") as fout:
        json.dump(new_vocab, fout, indent=4)

    return new_vocab

def output(st, fout, tokenizer, chaotic_list=None):
    for sen in st:
        sen = tokenizer.convert_tokens_to_sentence(tokenize(sen, tokenizer))
        fout.write(sen+"\n")
    if chaotic_list is not None:
        fout.write(str(chaotic_list)+"\n")
    else:
        fout.write("none\n")

    fout.write("-"*5+"\n")

def repeat_sentence(st):
    # repeat one sentence and delete the original sentence
    idx = np.random.choice(np.arange(len(st))[1:], 1 + int(len(st)/2), replace=False).tolist()
    s = min(idx)
    tmp_st = copy.deepcopy(st)
    for l in idx:
        tmp_st[l] = copy.deepcopy(tmp_st[s])
    return tmp_st

def repeat_ngram(st, tokenizer):
    # repeat ngram in one sentence 1~4
    def repeat_sen_gram(st):
        flag = True
        for _ in range(10):
            try:
                idx = np.random.choice(np.arange(len(st))[1:])
                gram_num = np.random.choice(np.arange(5)[1:])
                split_sen = tokenize(st[idx].strip(), tokenizer)
                pointer_st = np.random.choice(np.arange(len(split_sen)))
                pointer_ed = pointer_st + gram_num
                if pointer_ed > len(split_sen):
                    pointer_ed = pointer_st
                    pointer_st = pointer_ed - gram_num
                    if pointer_st < 0:
                        continue
                    else:
                        flag = False
                        break
            except:
                continue
        if flag:
            return copy.deepcopy(st)
        tmp_st = copy.deepcopy(st)
        tmp_st[idx] = tokenizer.convert_tokens_to_sentence(split_sen[:pointer_st] + split_sen[pointer_st:pointer_ed] + split_sen[pointer_st:pointer_ed] + split_sen[pointer_ed:]).strip()
        return tmp_st
    for i in range(int(len(st)/2)):
        st = repeat_sen_gram(st)
    return st

def replace_sentence(st, stories):
    flag = True
    for _ in range(10):
        try:
            tmp_st = copy.deepcopy(st)
            idxs = np.random.choice(np.arange(len(st))[1:], np.random.choice(np.arange(1, len(st))), replace=False)
            replace_st_id = np.random.choice(np.arange(len(stories)))
            for idx in idxs:
                tmp_st[idx] = np.random.choice(stories[replace_st_id])
            flag = False
            break
        except:
            continue
    if flag:
        return copy.deepcopy(st)
    return tmp_st

def replace_word(st, tokenizer, pos_vocab_entity, avail_phrases, anotomy_word):
    def replace_one_word(st):
        flag = True
        for _ in range(100):
            tmp_st = copy.deepcopy(st)
            idx = np.random.choice(np.arange(len(st))[1:])
            split_sen = tokenize(tmp_st[idx].strip(), tokenizer)
            pos_split_sen = pos_tag(split_sen)
            avail_w_id = []
            for w_id, w in enumerate(split_sen):
                if (w in avail_phrases["all"] and w not in avail_phrases["special"] and "[" not in w):
                    avail_w_id.append(w_id)
            if len(avail_w_id) == 0: continue
            word_id = np.random.choice(avail_w_id)
            if pos_split_sen[word_id][1] not in pos_vocab_entity: continue
            lemma_word = lemma(pos_split_sen[word_id][0], 'v' if pos_split_sen[word_id][1][0] == 'V' else 'n')
            if lemma_word in anotomy_word:
                replace_word = np.random.choice(anotomy_word[lemma_word])
            else:
                word_freq = pos_vocab_entity[pos_split_sen[word_id][1]]
                replace_word = ""
                flag_in = True
                for _ in range(10):
                    replace_word = np.random.choice(word_freq["word"], p=word_freq["freq"]/np.sum(word_freq["freq"]))
                    if len(word_freq["word"]) == 1 or replace_word != pos_split_sen[word_id][0]:
                        flag_in = False
                        break
                if flag_in:
                    replace_word = pos_split_sen[word_id][0]
            tmp_split_sen = copy.deepcopy(split_sen)
            tmp_split_sen[word_id] = replace_word
            tmp_st[idx] = tokenizer.convert_tokens_to_sentence(tmp_split_sen)

            flag = False
            break
        if flag:
            return copy.deepcopy(st)
        return tmp_st
    num = 0
    for idx in np.arange(len(st))[1:]:
        for word in tokenize(st[idx].strip(), tokenizer):
            if word in avail_phrases["all"]:
                num += 1
    try:
        final_num = np.random.choice(np.arange(1, int(num*0.15+1)))
    except:
        final_num = 1
    for _ in range(final_num):
        st = replace_one_word(st)
    return st

def change_neg_helper(sen, tokenizer, negation_word):
    def pro(s):
        final_sen = " ".join(sen).replace("  ", " ")
        return final_sen

    sen = sen.split()
    for i, n in enumerate(sen):
        if n in negation_word:
            sen[i] = negation_word[n]
            return pro(sen)

    neg_list = ["not", "n't"]
    for i, n in enumerate(sen):
        if n in ["would", "'d", "'ve", "will", "'ll", "could", "may", "might", "shall", "should", "do", "does", "did", "am", "is", "are", "was", "were", "be", "been"]:
            sen.insert(i+1, np.random.choice(neg_list))
            return pro(sen)
        elif n in ["can"]:
            sen.insert(i+1, np.random.choice(["not", "'t"]))
            return pro(sen)

    pos_sen = pos_tag(sen)
    for i, n in enumerate(pos_sen):
        if n[1] == "VB":
            sen.insert(i, "do " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBD":
            sen[i] = lemma(sen[i], "v")
            sen.insert(i, "did " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBG":
            sen.insert(i, np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBN":
            sen.insert(i, np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBP":
            sen.insert(i, "do " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBZ":
            sen[i] = lemma(sen[i], "v")
            sen.insert(i, "does " + np.random.choice(neg_list))
            return pro(sen)
    # print("No verb is found in this sentence.")
    return None

def change_neg_sentence(st, tokenizer, negation_word):
    flag = True
    for _ in range(10):
        try:
            tmp_st = copy.deepcopy(st)
            idxs = np.random.choice(np.arange(len(st))[1:], np.random.choice(np.arange(1, len(st))), replace=False)
            for idx in idxs:
                tmp_st_idx = change_neg_helper(st[idx], tokenizer, negation_word)
                if tmp_st_idx is not None: 
                    tmp_st[idx] = tmp_st_idx
                    flag = False
            if flag == False:
                break
        except:
            continue
    if flag:
        return copy.deepcopy(st)
    return tmp_st

def shuffle_sentence(st):
    def exchange(l, ids, target_ids):
        tmp_l = copy.deepcopy(l)
        for o_id, t_id in zip(ids, target_ids):
            tmp_l[o_id] = copy.deepcopy(l[t_id])
        return tmp_l
    # exchange n sentences
    flag = True
    for _ in range(10):
        sen_ids = np.random.choice(np.arange(len(st))[1:], int(np.random.choice(np.arange(1,len(st)-1)+1)), replace=False)
        target_ids = np.random.permutation(sen_ids)
        tmp_st = exchange(st, sen_ids, target_ids)
        if st != tmp_st:
            flag = False
            break
    if flag:
        return copy.deepcopy(st)
    return tmp_st

def get_pos_vocab(vocab):
    pos_vocab_entity = {}
    for word in vocab:
        for pos in vocab[word]:
            if pos != "number":
                if pos in pos_vocab_entity:
                    pos_vocab_entity[pos]["word"].append(word)
                    pos_vocab_entity[pos]["freq"].append(float(vocab[word][pos]))
                else:
                    pos_vocab_entity[pos] = {"word": [word], "freq":[float(vocab[word][pos])]}
    return pos_vocab_entity
# ========================================================================================

def build_data(data, output_dir, name, tokenizer,
            vocab=None,
            only_positive=False,
            sen_split_token=".",
            error_type_prob_dict = {"repeat":0.1, "replace":0.3, "shuffle":0.4, "negation":0.2},
            error_time_prob_dict = {1:0.5, 2:0.2, 3:0.2, 4:0.1},
            special_token_list = []):
    # error_type_prob_dict = {"repeat":0.6, "replace":0.15, "shuffle":0.15, "negation":0.1}
    # time_prob_list = [0.2,0.4,0.3,0.1]
    stories = []
    for tmp_data in data:
        original_context = tmp_data['context']
        origin_reference = tmp_data['reference']
        if not isinstance(origin_reference, list):
            origin_reference = [origin_reference]
        for r in origin_reference:
            story = [original_context.strip()] + [tmpr.strip() for tmpr in r.strip().split(sen_split_token)]
            rmblank_story = []
            for st in story:
                if st.strip() != "":
                    if not st.strip()[-1].isalpha():
                        rmblank_story.append(st.strip())
                    else:
                        rmblank_story.append(st.strip()+" "+sen_split_token)
            if only_positive or (len(rmblank_story) >= 5):
                stories.append(rmblank_story)
    with open("%s/%s_positive.txt"%(output_dir, name), "w") as fout:
        for story in stories:
            output(story, fout, tokenizer)
    if only_positive:
        return

    avail_phrases, anotomy_word = get_avail_phrases(special_token_list=special_token_list)
    pos_vocab_entity = get_pos_vocab(vocab)
    error_type_list = list(error_type_prob_dict.keys())
    error_type_prob_list = [error_type_prob_dict[t] for t in error_type_list]
    error_type_prob_list = list(np.array(error_type_prob_list) / np.sum(error_type_prob_list))

    error_time_list = list(error_time_prob_dict.keys())
    error_time_prob_list = [error_time_prob_dict[t] for t in error_time_list]
    error_time_prob_list = list(np.array(error_time_prob_list) / np.sum(error_time_prob_list))

    with open("%s/%s_negative.txt"%(output_dir, name), "w") as fout:
        for story in stories:
            chaotic_list = np.random.choice(error_type_list, np.random.choice(error_time_list, p=error_time_prob_list), replace=False, p=error_type_prob_list).tolist()
            for c in chaotic_list:
                if c == "repeat":
                    st = repeat_sentence(story) if np.random.random() < 0.7 else repeat_ngram(story, tokenizer)
                if c == "replace":
                    st = replace_sentence(story, stories) if np.random.random() < 0.5 else replace_word(story, tokenizer=tokenizer, pos_vocab_entity=pos_vocab_entity, avail_phrases=avail_phrases, anotomy_word=anotomy_word)
                if c == "shuffle":
                    st = shuffle_sentence(story)
                if c == "negation":
                    st = change_neg_sentence(story, tokenizer=tokenizer, negation_word=avail_phrases["negation"])
            output(st, fout, tokenizer, chaotic_list)