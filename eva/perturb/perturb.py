import numpy as np
np.random.seed(520)
import collections
import re
import os
import json
import nltk
import copy
import pattern
from pathlib import Path
from nltk.corpus import wordnet as wn
from pattern.en import tenses, wordnet, lexeme, pluralize, singularize, comparative, superlative
from .utils import nlp, get_data, get_negative_data, get_vocab, build_kg, get_sw
from ..tokenizer import SimpleTokenizer

module_path = Path(__file__).parent

tokenizer = SimpleTokenizer(method="space")
def swap(l, i, j):
    ll = copy.deepcopy(l)
    ll[j] = l[i]
    ll[i] = l[j]
    return ll

def transform(tok, cand_tok):
    tag = tok.tag_
    if tag in ["NN", "NNP"]:
        # cand = singularize(cand)
        cand = cand_tok.lemma_
    elif tag in ["NNS", "NNPS"]:
        cand = pluralize(cand_tok.text)
    elif tag == "JJ":
        cand = cand_tok.lemma_
    elif tag == "JJR":
        cand = comparative(cand_tok.text)
    elif tag == "JJS":
        cand = superlative(cand_tok.text)
    elif tag.startswith("VB"):
        cand_lex_list = lexeme(cand_tok.text)
        try:
            tense = collections.Counter([x[0] for x in pattern.en.tenses(tok.text)]).most_common(1)[0][0]
            p = pattern.en.tenses(tok.text)
            params = [tense, 3]
            if p:
                params = list(p[0])
                params[0] = tense
            cand = pattern.en.conjugate(cand_tok.text, *params)
            if cand is None:
                cand = cand_tok.lemma_
        except:
            cand = cand_tok.lemma_
    else:
        print(tok.text, tok.pos_, tok.tag_, cand_tok.text)
        cand = cand_tok.text
    return cand

def get_all_sen(str_tokens, punct=[".", ",", "!", ";", "?"]):
    tmp_sen, all_sen = [], []
    for tok in str_tokens:
        if tok.text in punct:
            tmp_sen.append(tok)
            if len(tmp_sen):
                all_sen.append(tmp_sen)
                tmp_sen = []
        else:
            tmp_sen.append(tok)
    return all_sen

class perturb_class():
    def __init__(self, data_name, name="base"):
        self.data_name = data_name
        self.name = name
        self.id_list, self.id_dict = [], {}

    def load_list(self):
        with open(module_path/f"{self.data_name}_{self.name}.txt", "r") as fin:
            id_list = [int(line.strip().split("|||")[0]) for line in fin]
        return id_list

    def output(self, origin_data):
        if not (module_path/self.data_name).exists():
            (module_path/self.data_name).mkdir()
        with open(module_path/f"{self.data_name}/{self.data_name}_{self.name}_perturb.txt", "w") as fout:
            for id_ in self.id_list:
                fout.write("%d|||%s|||%s|||" %(id_, origin_data[id_]["ipt"], self.id_dict[id_]["text"]))
                fout.write(self.id_dict[id_]["type"]+"\n")

class synonym_substitute(perturb_class):
    def __init__(self, data_name, nlp, sub_num=2, name="synonym_substitute"):
        super(synonym_substitute, self).__init__(data_name, name)
        self.nlp = nlp
        self.sub_num = sub_num
        self.sw = get_sw()

    def to_string(self, doc):
        return " ".join([t.text for t in doc])        

    def construct(self, origin_data):
        for k, d in enumerate(origin_data):
            if k % 100 == 0:
                print("constructing %d stories"%k)
            string = d["truth"]
            str_tokens = nlp(string)

            text, words = self.substitute_synonym(str_tokens)
            if text != self.to_string(str_tokens):
                self.id_dict[k] = {"text":text, "type":"type_substitute_synonym:%s"%",".join([" ".join(w) for w in words])}
        self.id_list = sorted(self.id_dict, key=lambda x: int(x))
        self.output(origin_data)

    def substitute_synonym(self, doc):
        text_word_list, subs = [], []
        for tok in doc:
            p = tok.pos_
            if (len(subs) < self.sub_num) and (np.random.random() < 0.5) and (p in ["NOUN", "VERB", "ADJ", "ADV"]) and (tok.lemma_ not in self.sw):
                cand = self.all_possible_synonym(tok.lemma_)
                cand = [c for c in cand if (nlp(c)[0].pos_ == p)]
                if len(cand):
                    cand = str(np.random.choice(cand))
                    cand_tok = nlp(cand)[0]
                    cand = transform(tok, cand_tok)
                    text_word_list.append(cand)
                    subs.append([tok.text, cand])
                else:
                    text_word_list.append(tok.text)
            else:
                text_word_list.append(tok.text)
        return ' '.join(text_word_list), subs

    def all_possible_synonym(self, word, min_acceptable_reputation=1):
        synonyms = []
        word_synsets = wn.synsets(word)

        for syn in word_synsets:            
            for lemma in syn.lemmas():
                if lemma.count() < min_acceptable_reputation:
                    continue
                if lemma.name() != word:
                    lem = lemma.name()
                    if "_" in lem: continue
                    synonyms.append(lem)
        synonyms = list(set(synonyms))
        # print(word, synonyms)
        return synonyms

class consistency(perturb_class):
    def __init__(self, data_name, nlp, add_neg_ratio=0.2, name="consistency"):
        super(consistency, self).__init__(data_name, name)
        self.nlp = nlp
        self.add_neg_ration = add_neg_ratio
        # print(self.all_possible_antonyms("disadvantages"))
        # for sen in ["I have many disadvantages.", "I want to fly high.", "I am very happy.", "I am very upset."]:
        #     toks = nlp(sen)
        #     for t in toks:
        #         print(t.text, t.pos_, t.lemma_)
        #     print(self.substitute_antonym(toks))
        #     print(self.remove_negation(toks))
        #     print(self.add_negation(toks))
        #     print("="*10)
        self.map_neg = {}
        with open(module_path/"negation_prefix_vocab.txt", "r") as fin:
            for pww in [line.strip().split() for line in fin]:
                for pw in [[pww[0], pww[1], pww[2]], [pww[1], pww[0], pww[2]]]:
                    w1, w2, p = pw[0], pw[1], pw[2][0].lower()
                    p = p if p in ["n", "v"] else "o"
                    if w1 in self.map_neg:
                        if p in self.map_neg[w1]:
                            self.map_neg[w1][p].append(w2)
                        else:
                            self.map_neg[w1][p] = [w2]
                    else:
                        self.map_neg[w1] = {p: [w2]}
        # for k in self.map_neg:
        #     print(k, "\t\t")
        # exit()

    def to_string(self, doc):
        return " ".join([t.text for t in doc])

    def construct(self, origin_data):
        for k, d in enumerate(origin_data):
            if k % 1000 == 0:
                print("constructing %d stories"%k)            
            string = d["truth"]
            str_tokens = nlp(string)

            text, words = self.substitute_antonym(str_tokens)
            if len(words) and np.random.random() < 0.25:
                self.id_dict[d["id"]] = {"text":text, "type":"type_substitute_antonym:%s"%",".join([" ".join(w) for w in words])}
            else:
                text, words = "", []
                for i, tok in enumerate(str_tokens):
                    if tok.text in self.map_neg:
                        p = tok.pos_[0].lower()[0]
                        p = p if p in ["n", "v"] else "o"
                        if p in self.map_neg[tok.text]:
                            text += np.random.choice(self.map_neg[tok.text][p]) + " "
                            words.append(tok.text)
                        else:
                            text += tok.text + " "
                    else:
                        text += tok.text + " "
                text = text.strip()
                if len(words):
                    self.id_dict[d["id"]] = {"text":text, "type":"type_substitute_neg:%s"%" ".join(words)}
                else:
                    text = self.remove_negation(str_tokens)
                    if text is not None:
                        self.id_dict[d["id"]] = {"text":text, "type":"type_rm_neg"}
                    else:
                        all_sen = get_all_sen(str_tokens)
                        if len(all_sen) == 0:
                            continue
                        neg_num = max([1, int(len(all_sen)*self.add_neg_ration)])
                        neg_id = np.random.choice(range(len(all_sen)), neg_num, replace=False)
                        new_st = ""
                        for i, sen in enumerate(all_sen):
                            if i in neg_id:
                                text = self.add_negation(self.nlp(sen))
                                if text is not None:
                                    new_st += " " + text
                                    continue
                            new_st += " " + sen
                        new_st = new_st.strip()
                        if new_st != self.to_string(str_tokens):
                            self.id_dict[d["id"]] = {"text": new_st, "type":"type_add_neg"}
        self.id_list = sorted(self.id_dict, key=lambda x: int(x))
        self.output(origin_data)

    def substitute_antonym(self, doc):
        text_word_list, subs = [], []
        for tok in doc:
            p = tok.pos_
            cand = self.all_possible_antonyms(tok, pos=p)
            cand = [c for c in cand if nlp(c)[0].pos_ == p]
            if len(cand) and np.random.random() < 0.5:
                cand = str(np.random.choice(cand))
                cand_tok = nlp(cand)[0]
                cand = transform(tok, cand_tok)
                text_word_list.append(cand)
                subs.append([tok.text, cand])
            else:
                text_word_list.append(tok.text)
        return ' '.join(text_word_list), subs

    def all_possible_antonyms(self, word, pos=None):
        map = {
            'NOUN': wordnet.NOUN,
            'VERB': wordnet.VERB,
            'ADJ': wordnet.ADJECTIVE,
            'ADV': wordnet.ADVERB
            }
        if (pos is None) or (pos not in map):
            pos_list = [wordnet.VERB, wordnet.ADJECTIVE, wordnet.NOUN, wordnet.ADVERB]
        else:
            pos_list = [map[pos]]
        synset = []
        for pos in pos_list:
            synset.extend(wordnet.synsets(word, pos=pos))

        ret = []
        for syn in synset:
            if not syn.antonym:
                continue
            for s in syn.antonym:
                ret.extend(s.senses)
        return [x for x in set(ret) if '_' not in x]

    def remove_negation(self, doc):
        """Removes negation from doc.
        This is experimental, may or may not work.

        Parameters
        ----------
        doc : spacy.token.Doc
            input

        Returns
        -------
        string
            With all negations removed

        """
        # This removes all negations in the doc. I should maybe add an option to remove just some.
        notzs = [i for i, z in enumerate(doc) if z.lemma_ == 'not' or z.dep_ == 'neg']
        new = []
        for notz in notzs:
            before = doc[notz - 1] if notz != 0 else None
            after = doc[notz + 1] if len(doc) > notz + 1 else None
            if (after and after.pos_ == 'PUNCT') or (before and before.text in ['or']):
                continue
            new.append(notz)
        notzs = new
        if not notzs:
            return None
        ret = ''
        start = 0
        for i, notz in enumerate(notzs):
            id_start = notz
            to_add = ' '
            id_end = notz + 1
            before = doc[notz - 1] if notz != 0 else None
            after = doc[notz + 1] if len(doc) > notz + 1 else None
            if before and before.lemma_ in ['will', 'can', 'do']:
                id_start = notz - 1
                try:
                    tense = collections.Counter([x[0] for x in pattern.en.tenses(before.text)]).most_common(1)[0][0]
                except:
                    tense = collections.Counter([x[0] for x in pattern.en.tenses(before.lemma_)]).most_common(1)[0][0]
                    print(before, before.lemma_)
                p = pattern.en.tenses(before.text)
                params = [tense, 3]
                if p:
                    params = list(p[0])
                    params[0] = tense
                to_add_ = pattern.en.conjugate(before.lemma_, *params)
                if to_add_ is None:
                    to_add = " "
                else:
                    to_add = " "+ to_add_  + " "
            if before and after and before.lemma_ == 'do' and after.pos_ == 'VERB':
                id_start = notz - 1
                tense = collections.Counter([x[0] for x in pattern.en.tenses(before.text)]).most_common(1)[0][0]
                p = pattern.en.tenses(before.text)
                params = [tense, 3]
                if p:
                    params = list(p[0])
                    params[0] = tense
                to_add_ = pattern.en.conjugate(after.text, *params)
                if to_add_ is None:
                    to_add = " "
                else:
                    to_add = " " + to_add_ + " "
                id_end = notz + 2
            ret += doc[start:id_start].text + to_add
            start = id_end
        ret += doc[id_end:].text
        return ret

    def add_negation(self, doc):
        """Adds negation to doc
        This is experimental, may or may not work. It also only works for specific parses.

        Parameters
        ----------
        doc : spacy.token.Doc
            input

        Returns
        -------
        string
            With negations added

        """
        for sentence in doc.sents:
            if len(sentence) < 3:
                continue
            root_id = [x.i for x in sentence if x.dep_ == 'ROOT'][0]
            root = doc[root_id]
            if '?' in sentence.text and sentence[0].text.lower() == 'how':
                continue
            if root.lemma_.lower() in ['thank', 'use']:
                continue
            if root.pos_ not in ['VERB', 'AUX']:
                continue
            neg = [True for x in sentence if x.dep_ == 'neg' and x.head.i == root_id]
            if neg:
                continue
            if root.lemma_ == 'be':
                if '?' in sentence.text:
                    continue
                if root.text.lower() in ['is', 'was', 'were', 'am', 'are', '\'s', '\'re', '\'m']:
                    return doc[:root_id + 1].text + ' not ' + doc[root_id + 1:].text
                else:
                    return doc[:root_id].text + ' not ' + doc[root_id:].text
            else:
                aux = [x for x in sentence if x.dep_ in ['aux', 'auxpass'] and x.head.i == root_id]
                if aux:
                    aux = aux[0]
                    if aux.lemma_.lower() in ['can', 'do', 'could', 'would', 'will', 'have', 'should']:
                        lemma = doc[aux.i].lemma_.lower()
                        if lemma == 'will':
                            fixed = 'won\'t'
                        elif lemma == 'have' and doc[aux.i].text in ['\'ve', '\'d']:
                            fixed = 'haven\'t' if doc[aux.i].text == '\'ve' else 'hadn\'t'
                        elif lemma == 'would' and doc[aux.i].text in ['\'d']:
                            fixed = 'wouldn\'t'
                        else:
                            fixed = doc[aux.i].text.rstrip('n') + 'n\'t' if lemma != 'will' else 'won\'t'
                        fixed = ' %s ' % fixed
                        return doc[:aux.i].text + fixed + doc[aux.i + 1:].text
                    return doc[:root_id].text + ' not ' + doc[root_id:].text
                else:
                    # TODO: does, do, etc. Remover return None de cima
                    subj = [x for x in sentence if x.dep_ in ['csubj', 'nsubj']]
                    params = pattern.en.tenses(root.text)
                    try:
                        tense = collections.Counter([x[0] for x in pattern.en.tenses(root.text)]).most_common(1)[0][0]
                    except:
                        tense = "infinitive"
                    params = [tense, 3] if not params else list(params[0])
                    params[0] = tense
                    # params = [tense, 3]
                    if root.tag_ not in ['VBG']:
                        do_ = pattern.en.conjugate('do', *params)
                        if do_ is None:
                            do = 'do n\'t'
                        else:
                            do = do_ + ' n\'t'
                        new_root = pattern.en.conjugate(root.text, tense='infinitive')
                    else:
                        do = 'not'
                        new_root = root.text
                    return '%s %s %s %s' % (doc[:root_id].text, do, new_root,  doc[root_id + 1:].text)

class lexical_rept(perturb_class):
    def __init__(self, data_name, nlp, ngram=4, name="lexical_rept"):
        super(lexical_rept, self).__init__(data_name, name)
        self.nlp = nlp
        self.ngram = ngram
    
    def construct(self, origin_data):
        from nltk import ngrams
        for k, d in enumerate(origin_data):
            if k % 1000 == 0:
                print("constructing %d stories"%k)
            if np.random.random() < 0.5:
                tmp_word_list = tokenizer.tokenize(d["truth"])
                gram = list(ngrams(tmp_word_list, self.ngram))
                avail_idx = []
                for i, g in enumerate(gram):
                    if "".join(g).isalpha():
                        avail_idx.append(i)
                if len(avail_idx) < 1:
                    continue
                insert_idx = np.random.choice(avail_idx)
                insert_gram = list(gram[insert_idx])
                if insert_gram[0] != "and":
                    insert_gram.insert(0, "and")
                text_word_list = copy.deepcopy(tmp_word_list)
                text_word_list.insert(insert_idx+self.ngram, " ".join(insert_gram))
                text = " ".join(text_word_list)
                if text == " ".join(tmp_word_list):
                    continue
                self.id_dict[d["id"]] = {"text":text, "type":"type_lx_rept_ngram:%s"%" ".join(insert_gram)}
            else:
                str_tokens = self.nlp(d["truth"])
                sen_list = get_all_sen(str_tokens, punct=[".", "!", ";", "?"])
                if len(sen_list) < 2:
                    continue
                rept_idx = sorted(np.random.choice(range(len(sen_list)), 2, replace=False))
                sen_list[rept_idx[1]] = sen_list[rept_idx[0]]
                text = " ".join([" ".join([t.text for t in s]) for s in sen_list])
                if text == " ".join([t.text for t in str_tokens]):
                    continue
                self.id_dict[d["id"]] = {"text":text, "type":"type_lx_rept_sen:%d %d"%(rept_idx[0], rept_idx[1])}
        self.id_list = sorted(self.id_dict.keys())
        self.output(origin_data)

class semantic_rept(perturb_class):
    def __init__(self, data_name, nlp, ngram=4, name="semantic_rept"):
        super(semantic_rept, self).__init__(data_name, name)
        self.nlp = nlp
        with open(module_path/f"back_trans_data/{data_name}_bt.json", "r") as fin:
            self.map_bt = json.load(fin)
    def construct(self, origin_data):
        assert len(self.map_bt) == len(origin_data)
        for k, (m, d) in enumerate(zip(self.map_bt, origin_data)):
            if k % 1000 == 0:
                print("constructing %d stories"%k)            
            sen_list = np.random.permutation(list(m.keys()))
            # text = copy.deepcopy(d["truth"])
            # sub = []
            # for i, s in enumerate(sen_list):
            #     if (float(m[s]["score"]) >= 0.4) and (float(m[s]["score"]) <= 0.4) and (len(s.strip().split()) > 5) and (s in d["truth"]):
            #         for j, s2 in enumerate(sen_list):
            #             if (j != i) and (s2 in d["truth"]):
            #                 text.replace(s2, m[s]["trans"])
            #                 sub = [s, m[s]["trans"]]
            #                 break
            #     if len(sub):
            #         break

            str_tokens = self.nlp(d["truth"])
            sen_list = get_all_sen(str_tokens, punct=[".", "!", ";", "?"])
            if len(sen_list) == 0:
                continue
            avail_idx = []
            for i, s in enumerate(sen_list[:-1]):
                sen = " ".join([t.text for t in s])
                if (sen in m) and (float(m[sen]["score"]) >= 0.4) and (float(m[sen]["score"]) <= 0.7) and (len(s) >= 5):
                    avail_idx.append(i)
            if len(avail_idx) == 0:
                continue
            rept_idx = np.random.choice(avail_idx)
            rept_idx2 = np.random.choice(range(rept_idx+1, len(sen_list)))

            sen_list = [" ".join([t.text for t in s]) for s in sen_list]
            sen_list[rept_idx2] = m[sen_list[rept_idx]]["trans"]
            text = " ".join(sen_list)

            if text == " ".join([t.text for t in str_tokens]):
                continue
            self.id_dict[d["id"]] = {"text":text, "type":"type_sm_rept_sen:%d %d"%(rept_idx, rept_idx2)}
        self.id_list = sorted(self.id_dict.keys())
        self.output(origin_data)

class semantic_substitute(perturb_class):
    def __init__(self, data_name, nlp, ngram=4, name="semantic_substitute"):
        super(semantic_substitute, self).__init__(data_name, name)
        self.nlp = nlp
        with open(module_path/f"back_trans_data/{data_name}_bt.json", "r") as fin:
            self.map_bt = json.load(fin)
    def construct(self, origin_data):
        assert len(self.map_bt) == len(origin_data)
        for k, (m, d) in enumerate(zip(self.map_bt, origin_data)):
            if k % 1000 == 0:
                print("constructing %d stories"%k)            
            str_tokens = self.nlp(d["truth"])
            sen_list = get_all_sen(str_tokens, punct=[".", "!", ";", "?"])

            avail_idx = []
            for i, s in enumerate(sen_list[:-1]):
                sen = " ".join([t.text for t in s])
                if (sen in m) and (float(m[sen]["score"]) >= 0.4) and (float(m[sen]["score"]) <= 0.7) and (len(s) >= 5):
                    avail_idx.append(i)
            if len(avail_idx) == 0:
                continue
            rept_idx = np.random.choice(avail_idx)

            sen_list = [" ".join([t.text for t in s]) for s in sen_list]
            sen_list[rept_idx] = m[sen_list[rept_idx]]["trans"]
            text = " ".join(sen_list)

            if text == " ".join([t.text for t in str_tokens]):
                continue
            self.id_dict[k] = {"text":text, "type":"type_syn_sen:%d"%(rept_idx)}
        self.id_list = sorted(self.id_dict.keys())
        self.output(origin_data)

class character(perturb_class):
    def __init__(self, data_name, nlp, pronoun_num=3, name="character"):
        super(character, self).__init__(data_name, name)
        self.id_list, self.id_dict = [], {}
        self.lemma = nltk.stem.WordNetLemmatizer().lemmatize
        self.token = nltk.tokenize.TreebankWordTokenizer().tokenize
        self.pos_tag = nltk.pos_tag
        self.nlp = nlp
        self.pronoun_num = pronoun_num

        self.pronoun_vocab, self.mapchar = [], {}
        with open(module_path/"pronoun_vocab.txt", "r") as fin:
            for line in fin:
                char = line.strip().split()
                self.pronoun_vocab.append(char)
                for i, c in enumerate(char):
                    if c in self.mapchar:
                        self.mapchar[c].add(i)
                    else:
                        self.mapchar[c] = set([i])
            for k in self.mapchar:
                self.mapchar[k] = list(self.mapchar[k])
        # for t in self.nlp("i want my book. it is mine. let us go to our room for ours."):
        #     print(t.text, t.pos_, t.dep_)
        self.subj_id, self.obj_id, self.det_id = 0, 1, 2 # 'you': [0, 1], 'his': [2, 4], 'her': [1, 2], 'it': [0, 1], 'its': [2, 4]
        # print(self.mapchar)

    def substitute(self, l, pronoun_list, avail_char):
        p = [c for pl in pronoun_list for c in pl]
        lt = self.nlp(" ".join(l))
        ll = [w.text for w in lt]

        for i, w in enumerate(lt):
            if w.text in p:
                src = w.text
                if len(self.mapchar[src]) == 1:
                    idx = self.mapchar[src][0]
                else:
                    if self.det_id in self.mapchar[src] and w.pos_ == "DET":
                        idx = self.det_id
                    elif self.obj_id in self.mapchar[src] and ("root" not in w.dep_) and ("subj" not in w.dep_):
                        idx = self.obj_id
                    else:
                        idx = np.random.choice(self.mapchar[src])
                tgt_list = [self.pronoun_vocab[char_id][idx] for char_id in avail_char]
                tgt_list = [t for t in tgt_list if t != src]
                ll[i] = np.random.choice(tgt_list)
        return ll


    def reoder_sub_obj(self, wl, dl):
        sub_id_list, obj_id_list = [], []
        flag = False
        for i, (w, d) in enumerate(zip(wl, dl)):
            if "subj" in d:
                sub_id_list.append(i)
                if w in self.mapchar:
                    for p in self.pronoun_vocab:
                        if w in p:
                            wl[i] = p[self.obj_id]
                            break
            elif "obj" in d:
                obj_id_list.append(i)
                if w in self.mapchar:
                    for p in self.pronoun_vocab:
                        if w in p:
                            wl[i] = p[self.subj_id]
                            break
        if len(sub_id_list) and len(obj_id_list):
            sub_id, obj_id = np.random.choice(sub_id_list), np.random.choice(obj_id_list)
            nwl = swap(wl, obj_id, sub_id)
            return nwl, [wl[sub_id], wl[obj_id]]
        return wl, []


    def construct(self, origin_data):
        self.id_dict = {}
        for k, d in enumerate(origin_data):
            if k % 1000 == 0:
                print("constructing %d stories"%k)
            doc = self.nlp(d["truth"])
            word_list = [w.text for w in doc]
            dep_list = [w.dep_ for w in doc]

            pronoun_list, avail_char = [], []
            for i, ws in enumerate(self.pronoun_vocab):
                tmp_pronoun_list = []
                for w in ws:
                    if w in word_list:
                        tmp_pronoun_list.append(w)
                if len(tmp_pronoun_list):
                    pronoun_list.append(list(set(tmp_pronoun_list)))
                    avail_char.append(i)
            if len(pronoun_list) >= self.pronoun_num:
                text_word_list = self.substitute(word_list, pronoun_list, avail_char)
                text = " ".join(text_word_list)
                if text == " ".join(word_list):
                    continue
                self.id_dict[d["id"]] = {"text": text, "type":"type_pronoun:%s"%",".join([" ".join(w) for w in pronoun_list])}
            else:
                seg_id = [0] + [i+1 for i, d in enumerate(dep_list) if d == "punct"]
                self.reoder_sub_obj(word_list, dep_list)
                text_word_list = []
                flag, sub_obj_list = False, []
                for i in range(len(seg_id)-1):
                    tmp_word_list, sub_obj = self.reoder_sub_obj(word_list[seg_id[i]:seg_id[i+1]], dep_list[seg_id[i]:seg_id[i+1]])
                    text_word_list += tmp_word_list
                    if len(sub_obj):
                        sub_obj_list.append(sub_obj)
                        flag = True
                if flag:
                    text = " ".join(text_word_list)
                    self.id_dict[d["id"]] = {"text": text, "type":"type_subobj:%s"%",".join([" ".join(so) for so in sub_obj_list])}

                # passive_list = []
                # for w, dep in zip(word_list, dep_list):
                #     if dep == "nsubjpass":
                #         passive_list.append("%s %s"%(w, dep))
                # if len(passive_list) >= self.passive_num:
                #     if d["id"] in self.id_dict:
                #         self.id_dict[d["id"]]["passive"] = passive_list
                #     else:
                #         self.id_dict[d["id"]] = {"pronoun": [], "passive": passive_list}

        self.id_list = sorted(self.id_dict)
        self.output(origin_data)

class coherence(perturb_class):
    def __init__(self, data_name, nlp, name="coherence"):
        super(coherence, self).__init__(data_name, name)
        self.nlp = nlp
    
    def construct(self, origin_data):
        from nltk import ngrams
        for d in origin_data:
            toks = self.nlp(d["truth"])
            rep_toks = self.nlp(np.random.choice(origin_data)["truth"])
            text_word_list = []
            pos_map = {}
            for rt in rep_toks:
                if rt.tag_ in ["NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                    if rt.tag_ in pos_map:
                        pos_map[rt.tag_].append(rt)
                    else:
                        pos_map[rt.tag_] = [rt]
            sub = []
            for t in toks:
                if t.tag_ in pos_map and np.random.random() < 0.25:
                    rand_cand_tok = np.random.choice(pos_map[t.tag_])
                    cand = transform(t, rand_cand_tok)
                    text_word_list.append(cand)
                    sub.append([t.text, cand])
                else:
                    text_word_list.append(t.text)

            if len(sub) >= 3:
                text = " ".join(text_word_list)
                self.id_dict[d["id"]] = {"text":text, "type":"type_substitute_token:%s"%",".join([" ".join(s) for s in sub])}
            else:
                tmp_sen_list = [sen.strip() for sen in (d["truth"]).split(".")]
                sen_list = []
                for sen in tmp_sen_list:
                    if len(sen) > 1:
                        sen_list.append(sen)
                if len(sen_list) == 0:
                    continue
                while True:
                    cand_st = [sen.strip() for sen in np.random.choice(origin_data)["truth"].split(".")]
                    if len(cand_st):
                        cand_sen = np.random.choice(cand_st)
                        if cand_sen != "":
                            break
                idx = np.random.choice(range(len(sen_list)))
                sen_list[idx] = cand_sen
                text = " . ".join(sen_list) + " ."
                if text == " . ".join(tmp_sen_list):
                    continue
                self.id_dict[d["id"]] = {"text":text, "type":"type_substitute_sen:%d"%(idx)}
        self.id_list = sorted(self.id_dict.keys())
        self.output(origin_data)

class commonsense(perturb_class):
    def __init__(self, data_name, nlp, sub_num=3, stopwords=None, vocab_dict=None, name="commonsense"):
        super(commonsense, self).__init__(data_name, name)
        if stopwords is None:
            # from nltk.corpus import stopwords
            # sw = set(stopwords.words('english'))
            stopwords = set(get_sw())
        self.sub_num = sub_num
        self.sw = stopwords
        self.kg, self.word2kg = build_kg(sw=self.sw)
        self.vocab_dict = vocab_dict
        self.avail_pos_tag = ["NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    def allin(self, lkg, lword):
        for w in lkg:
            if w not in lword:
                return False
        return True
    def substitute(self, w_tok, kg_id_list, sen_lemma):
        cand_list, sub = set(), []
        for kg_id in kg_id_list:
            for k in self.kg[kg_id][0]:
                if self.vocab_dict is not None: # not assigned vocab_dict
                    if k in self.vocab_dict and self.vocab_dict[k] > 10 and k not in self.sw:
                        cand_list.add(k)
                else:
                    cand_list.add(k)
        cand_list = list(cand_list)
        if len(cand_list):
            for c in np.random.permutation(cand_list):
                cand_tok = nlp(str(c))[0]
                if cand_tok.pos_ == w_tok.pos_ and cand_tok.lemma_ not in sen_lemma:
                    break
            cand = transform(w_tok, cand_tok)
            return cand
        return None

    def construct(self, origin_data):
        self.id_dict = {}
        for i, d in enumerate(origin_data):
            if i % 1000 == 0:
                print("constructing %d stories"%i)
            tok = nlp(d["truth"])
            tok_lemma = [t.lemma_ for t in tok]
            text_word_list, sub = [], []
            for w in tok:
                flag = True
                if (w.lemma_ not in self.sw) and (w.tag_ in self.avail_pos_tag) and (w.lemma_ in self.word2kg) and (np.random.random()<0.75):
                    for kg_id in np.random.permutation(self.word2kg[w.lemma_]):
                        if self.allin(self.kg[kg_id][0], tok_lemma):
                            cand = self.substitute(w, self.word2kg[w.lemma_], tok_lemma)
                            if cand is not None:
                                flag = False
                                text_word_list.append(cand)
                                sub.append([w.text, cand])
                                break
                if flag:
                    text_word_list.append(w.text)
            if len(sub) >= self.sub_num:
                text = " ".join(text_word_list)
                self.id_dict[d["id"]] = {"text":text, "type":"type_substitute_cs:%s"%",".join([" ".join(s) for s in sub])}
        self.id_list = sorted(self.id_dict.keys())
        self.output(origin_data)

class cause(perturb_class):
    def __init__(self, data_name, nlp, name="cause"):
        super(cause, self).__init__(data_name, name)
        self.nlp = nlp
        self.cause_vocab = {}

        self.reorder_word = ["so", "because"]
        because_noun = ["reason", "cause", "sake"]
        because_nouns = ["reasons", "causes"]
        so_noun = ["effect", "result", "consequence"]
        so_nouns = ["effects", "results", "consequences"]
        self.map_phrase = {
            "so that": {"o": ["because", "since"]},
            "because of": {"o": ["resulting in", "leading to"]},
        }
        self.map_word = {
            "reason": {"n": so_noun},
            "reasons": {"n": so_nouns},
            "cause": {"n": so_noun, "v": ["result in", "lead to"]},
            "causes": {"n": so_nouns, "v": ["results in", "leads to"]},
            "caused": {"v": ["resulted in", "led to"]},
            "causing": {"v": ["resulting in", "leading to"]},
            "effect": {"n": because_noun},
            "result": {"n": because_noun},
            "consequence": {"n": because_noun},
            "effects": {"n": because_nouns},
            "results": {"n": because_nouns},
            "consequences": {"n": because_nouns},
        }
    def construct(self, origin_data):
        self.id_dict = {}
        for d in origin_data:
            str_tokens = self.nlp(d["truth"])
            all_sen = get_all_sen(str_tokens)
            if len(all_sen) == 0:
                continue
            text = None
            for i, sen in enumerate(all_sen):
                idx, cw = None, None
                for j, w in enumerate(sen):
                    if (w.text in self.reorder_word):# and (w.tag_ in self.cause_vocab[w.text]):
                        idx, cw = j, w
                if (idx is None) or ((len(sen) > idx+1) and ((sen[idx+1].text in ["of", "that"]) or (sen[idx+1].tag_[:2] in ["JJ", "RB"]))) or ((idx>0) and (sen[idx-1].text in ["is", "was", "be", "been", "being", "'s"])):
                    continue
                r1, r2 = sen[:idx], sen[idx+1:-1]
                if len(r1) and len(r2):
                    text_word_list = [t.text for s in all_sen[:i] for t in s]
                    text_word_list += [t.text for t in r2] + [cw.text] + [t.text for t in r1] + [sen[-1].text]
                    text_word_list += [t.text for s in all_sen[i+1:] for t in s]
                    type_id = 1
                elif len(r1)==0 and len(r2):
                    if cw.text == "because":
                        r1 = r2
                        if i+1 == len(all_sen): continue
                        r2 = all_sen[i+1][:-1]
                        if len(r2) == 0: continue
                        text_word_list = [t.text for s in all_sen[:i] for t in s]
                        text_word_list += [cw.text] + [t.text for t in r2] + [sen[-1].text] + [t.text for t in r1] + [all_sen[i+1][-1].text]
                        text_word_list += [t.text for s in all_sen[i+2:] for t in s]
                        type_id = 2
                    else:
                        if i == 0: continue
                        r1 = all_sen[i-1][:-1]
                        if len(r1) == 0: continue
                        text_word_list = [t.text for s in all_sen[:i-1] for t in s]
                        text_word_list += [t.text for t in r2] + [all_sen[i-1][-1].text] + [cw.text] + [t.text for t in r1] + [sen[-1].text]
                        text_word_list += [t.text for s in all_sen[i+1:] for t in s]
                        text = " ".join(text_word_list)
                        type_id = 3
                # elif len(r1) and len(r2)==0:
                #     if i+1 == len(all_sen): continue
                #     r2 = all_sen[i+1][:-1]
                #     if len(r2) == 0: continue
                #     text_word_list = [t.text for s in all_sen[:i] for t in s]
                #     text_word_list += [t.text for t in r2] + [cw.text] + [sen[-1].text] + [t.text for t in r1] + [all_sen[i+1][-1].text]
                #     text_word_list += [t.text for s in all_sen[i+2:] for t in s]
                #     type_id = 4
                else:
                    if i == 0: continue
                    r1 = all_sen[i-1][:-1]
                    if len(r1) == 0: continue
                    if i+1 == len(all_sen): continue
                    r2 = all_sen[i+1][:-1]
                    if len(r2) == 0: continue
                    text_word_list = [t.text for s in all_sen[:i-1] for t in s]
                    text_word_list += [t.text for t in r2] + [all_sen[i-1][-1].text] + [cw.text] + [sen[-1].text] + [t.text for t in r1] + [all_sen[i+1][-1].text]
                    text_word_list += [t.text for s in all_sen[i+1:] for t in s]
                    type_id = 4

                text = " ".join(text_word_list)
                self.id_dict[d["id"]] = {"text":text, "type":"type_reorder_%d:%s"%(type_id, cw.text)}
                break

            if text is None:
                sub = []
                for i, tok in enumerate(str_tokens):
                    if len(str_tokens) > i+1:
                        tmps = "%s %s" % (str_tokens[i].text, str_tokens[i+1].text)
                        if tmps in self.map_phrase:
                            rand_word = np.random.choice(self.map_phrase[tmps]["o"])
                            text_word_list = [t.text for t in str_tokens[:i]] + [rand_word] + [t.text for t in str_tokens[i+2:]]
                            sub.append([tmps, rand_word])
                            i += 1
                if len(sub) == 0:
                    text_word_list = []
                    for i, tok in enumerate(str_tokens):
                        flag = True
                        if tok.text in self.map_word:
                            p = tok.pos_.lower()[0]
                            if p in self.map_word[tok.text]:
                                rand_word = np.random.choice(self.map_word[tok.text][p])
                                text_word_list.append(rand_word)
                                sub.append([tok.text, rand_word])
                                flag = False
                        if flag:
                            text_word_list.append(tok.text)
                if len(sub) > 0:
                    text = " ".join(text_word_list)
                    self.id_dict[d["id"]] = {"text":text, "type":"type_substitute:%s"%",".join([" ".join(s) for s in sub])}

        self.id_list = sorted(self.id_dict.keys())
        self.output(origin_data)

class time(perturb_class):
    def __init__(self, data_name, nlp, name="time"):
        super(time, self).__init__(data_name, name)
        map_list = [[["second", "minute", "hour"], ["year", "month"], ["century"]],\
            [["seconds", "minutes", "hours"], ["years", "months"], ["centuries"]],\
            [["often", "sometimes", "ususally", "always", "ever", "once"], ["never"], ["forever"]],\
            [["after"], ["before"], ["when", "whenever"]],\
            [["perviously", "formerly"], ["simultaneously"], ["subsequently"]],\
            [["past"], ["future"]],\
            [["early"], ["later"]],\
            [["beginning"], ["ending"]]]
        self.map_word = {}
        for m in map_list:
            for mm in m:
                for mmm in mm:
                    self.map_word[mmm] = list(set([www for ww in m for www in ww]) - set(mm))
        self.reorder_word = ["then"]
        self.nlp = nlp
    def construct(self, origin_data):
        for d in origin_data:
            str_tokens = self.nlp(d["truth"])
            all_sen = get_all_sen(str_tokens, punct=[".", "!", ";", "?"])
            if len(all_sen) == 0:
                continue

            text, sub_text = None, None
            for i, sen in enumerate(all_sen):
                idx, cw = None, None
                for j, w in enumerate(sen):
                    if (w.text in self.reorder_word):# and (w.tag_ in self.cause_vocab[w.text]):
                        idx, cw = j, w
                if (idx is None):
                    continue
                if i == 0: continue
                else: 
                    idx_lh, sub_text = [idx, idx], [cw.text]
                    text_word_list = [t.text for s in all_sen[:i-1] for t in s]
                    if (idx>0) and (sen[idx-1].text in ["by", "since", "until", "till", "only"]):
                        idx_lh[0] = idx - 1
                        sub_text = [sen[idx-1].text] + sub_text
                    if (len(sen) > idx+1) and (sen[idx+1].pos_ == 'PUNCT'):
                        idx_lh[1] = idx + 1
                        sub_text = sub_text + [sen[idx+1].text]
                    text_word_list += [t.text for t in sen[:idx_lh[0]] + sen[(idx_lh[1]+1):]]
                    text_word_list += sub_text
                    text_word_list += [t.text for t in all_sen[i-1]] + [t.text for s in all_sen[(i+1):] for t in s]
                text = " ".join(text_word_list)
                self.id_dict[d["id"]] = {"text":text, "type":"type_reorder:%s"%" ".join(sub_text)}
                break

            if text is None:
                text_word_list, sub = [], []
                for i, tok in enumerate(str_tokens):
                    flag = True
                    if tok.text in self.map_word:
                        rand_word = np.random.choice(self.map_word[tok.text])
                        text_word_list.append(rand_word)
                        sub.append([tok.text, rand_word])
                        flag = False
                    if flag:
                        text_word_list.append(tok.text)  

                if len(sub) > 0:
                    text = " ".join(text_word_list)
                    self.id_dict[d["id"]] = {"text":text, "type":"type_substitute:%s"%",".join([" ".join(s) for s in sub])}

        self.id_list = sorted(self.id_dict.keys())
        self.output(origin_data)

class add_typos(perturb_class):
    def __init__(self, data_name, typos_ratio=0.02, name="typos"):
        super(add_typos, self).__init__(data_name, name)
        self.typos_ratio = typos_ratio
    
    def construct(self, origin_data):
        for k, d in enumerate(origin_data):
            if k % 1000 == 0:
                print("constructing %d stories"%k)            
            string = list(d["truth"].strip())
            if len(string) <= 2:
                continue
            typos_num = np.random.choice(range(1, max([int(len(string)*self.typos_ratio), 2])))
            position = sorted(np.random.choice(len(string) - 1, typos_num, replace=False), reverse=True)
            for pos in position:
                policy_rand = np.random.random()
                if policy_rand <= 0.5:
                    tmp = string[pos]
                    string[pos] = string[pos + 1]
                    string[pos + 1] = tmp
                elif policy_rand <= 0.75:
                    string.insert(pos, string[pos])
                else:
                    string.pop(pos)
            string = ''.join(string)
            if string != d["truth"]:
                typo_list = []
                for w1, w2 in zip(string.strip().split(), d["truth"].strip().split()):
                    if w1 != w2:
                        typo_list.append([w1, w2])
                if len(typo_list):
                    self.id_dict[k] = {"text":string, "type":"type_typo:%s"%",".join([" ".join(t) for t in typo_list])}

        self.id_list = sorted(self.id_dict)
        self.output(origin_data)

class contraction(perturb_class):
    def __init__(self, data_name, name="contraction"):
        """Perturbation functions, contracts and expands contractions if present
        """        
        super(contraction, self).__init__(data_name, name)
    def construct(self, origin_data):
        for k, d in enumerate(origin_data):
            if k % 1000 == 0:
                print("constructing %d stories"%k)
            text = self.expand_contractions(d["truth"])
            type_ = "expand_contraction"
            if text == d["truth"]:
                text = self.contract(d["truth"])
                type_ = "contract"
            if text == d["truth"]:
                continue
            self.id_dict[k] = {"text":text, "type":"type_%s"%type_}

        self.id_list = sorted(self.id_dict)
        self.output(origin_data)

    def expand_contractions(self, sentence):
        """Expands contractions in a sentence (if any)

        Parameters
        ----------
        sentence : str
            input string

        Returns
        -------
        string
            String with contractions expanded (if any)

        """
        contraction_map = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have", "couldn't":
            "could not", "didn't": "did not", "doesn't": "does not", "don't":
            "do not", "hadn't": "had not", "hasn't": "has not", "haven't":
            "have not", "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y":
            "how do you", "how'll": "how will", "how's": "how is",
            "I'd": "I would", "I'll": "I will", "I'm": "I am",
            "I've": "I have", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is", "ma'am":
            "madam", "might've": "might have", "mightn't": "might not",
            "must've": "must have", "mustn't": "must not", "needn't":
            "need not", "oughtn't": "ought not", "shan't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is",
            "should've": "should have", "shouldn't": "should not", "that'd":
            "that would", "that's": "that is", "there'd": "there would",
            "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are",
            "they've": "they have", "wasn't": "was not", "we'd": "we would",
            "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what're": "what are", "what's": "what is",
            "when's": "when is", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who's": "who is",
            "who've": "who have", "why's": "why is", "won't": "will not",
            "would've": "would have", "wouldn't": "would not",
            "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you're": "you are", "you've": "you have",
            "ai n't": "is not", "are n't": "are not", "ca n't": "cannot",
            "ca n't 've": "cannot have", "could 've": "could have", "could n't":
            "could not", "did n't": "did not", "does n't": "does not", "do n't":
            "do not", "had n't": "had not", "has n't": "has not", "have n't":
            "have not", "he 'd": "he would", "he 'd 've": "he would have",
            "he 'll": "he will", "he 's": "he is", "how 'd": "how did", "how 'd 'y":
            "how do you", "how 'll": "how will", "how 's": "how is",
            "I 'd": "I would", "I 'll": "I will", "I 'm": "I am",
            "I 've": "I have", "i 'd": "i would", "i 'll": "i will",
            "i 'm": "i am", "i 've": "i have", "is n't": "is not",
            "it 'd": "it would", "it 'll": "it will", "it 's": "it is", "ma 'am":
            "madam", "might 've": "might have", "might n't": "might not",
            "must 've": "must have", "must n't": "must not", "need n't":
            "need not", "ought n't": "ought not", "sha n't": "shall not",
            "she 'd": "she would", "she 'll": "she will", "she 's": "she is",
            "should 've": "should have", "should n't": "should not", "that 'd":
            "that would", "that 's": "that is", "there 'd": "there would",
            "there 's": "there is", "they 'd": "they would",
            "they 'll": "they will", "they 're": "they are",
            "they 've": "they have", "was n't": "was not", "we 'd": "we would",
            "we 'll": "we will", "we 're": "we are", "we 've": "we have",
            "were n't": "were not", "what 're": "what are", "what 's": "what is",
            "when 's": "when is", "where 'd": "where did", "where 's": "where is",
            "where 've": "where have", "who 'll": "who will", "who 's": "who is",
            "who 've": "who have", "why 's": "why is", "wo n't": "will not",
            "would 've": "would have", "would n't": "would not",
            "you 'd": "you would", "you 'd've": "you would have",
            "you 'll": "you will", "you 're": "you are", "you 've": "you have",            
            }
        # self.reverse_contraction_map = dict([(y, x) for x, y in self.contraction_map.items()])
        contraction_pattern = re.compile(r'\b({})\b'.format('|'.join(contraction_map.keys())),
            flags=re.IGNORECASE|re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_map.get(match, contraction_map.get(match.lower()))
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction
        return contraction_pattern.sub(expand_match, sentence)

    def contract(self, sentence):
        """Contract expanded contractions in a sentence (if any)

        Parameters
        ----------
        sentence : str
            input string

        Returns
        -------
        string
            String with contractions contracted (if any)

        """
        reverse_contraction_map = {
            'is not': "is n't", 'are not': "are n't", 'cannot': "ca n't",
            'could not': "could n't", 'did not': "did n't", 'does not':
            "does n't", 'do not': "do n't", 'had not': "had n't", 'has not':
            "has n't", 'have not': "have n't", 'he is': "he is", 'how did':
            "how 'd", 'how is': "how 's", 'I would': "I 'd", 'I will': "I 'll",
            'I am': "I 'm", 'i would': "i 'd", 'i will': "i 'll", 'i am': "i 'm",
            'it would': "it 'd", 'it will': "it 'll", 'it is': "it 's",
            'might not': "might n't", 'must not': "must n't", 'need not': "need n't",
            'ought not': "ought n't", 'shall not': "sha n't", 'she would': "she 'd",
            'she will': "she 'll", 'she is': "she 's", 'should not': "should n't",
            'that would': "that 'd", 'that is': "that 's", 'there would':
            "there 'd", 'there is': "there 's", 'they would': "they 'd",
            'they will': "they 'll", 'they are': "they 're", 'was not': "was n't",
            'we would': "we 'd", 'we will': "we 'll", 'we are': "we 're", 'were not':
            "were n't", 'what are': "what 're", 'what is': "what 's", 'when is':
            "when 's", 'where did': "where 'd", 'where is': "where 's",
            'who will': "who 'll", 'who is': "who 's", 'who have': "who 've", 'why is':
            "why 's", 'will not': "wo n't", 'would not': "would n't", 'you would':
            "you 'd", 'you will': "you 'll", 'you are': "you 're",

        }

        reverse_contraction_pattern = re.compile(r'\b({})\b '.format('|'.join(reverse_contraction_map.keys())),
            flags=re.IGNORECASE|re.DOTALL)
        def cont(possible):
            match = possible.group(1)
            first_char = match[0]
            expanded_contraction = reverse_contraction_map.get(match, reverse_contraction_map.get(match.lower()))
            expanded_contraction = first_char + expanded_contraction[1:] + ' '
            return expanded_contraction
        return reverse_contraction_pattern.sub(cont, sentence)

class delete_punct(perturb_class):
    def __init__(self, data_name, nlp, punct_num, name="delete_punct"):
        super(delete_punct, self).__init__(data_name, name)
        # from nltk.corpus import stopwords
        # sw = set(stopwords.words('english'))
        self.nlp = nlp
        self.punct_num = punct_num

    def construct(self, origin_data):
        for k, d in enumerate(origin_data):
            if k % 1000 == 0:
                print("constructing %d stories"%k)
            str_tokens = self.nlp(d["truth"])
            text_word_list, sub = [], []
            for t in str_tokens:
                if (t.pos_ == "PUNCT") and (t.text not in [".", "!", ";", "?"]):
                    sub.append(t.text)
                    continue
                text_word_list.append(t.text)
            text = " ".join(text_word_list)
            if len(sub) <= self.punct_num:
                continue
            self.id_dict[k] = {"text":text, "type":"type_del_punct:%s"%(" ".join(sub))}
        self.id_list = sorted(self.id_dict.keys(), key=lambda x: len(self.id_dict[x]["type"]), reverse=True)
        self.output(origin_data)

# name = "wp_all" # "wp_all"
# data = get_data(name)
# vocab_dict = get_vocab(name)

if __name__ == "__main__":
    name = "wp_all"
    data = get_negative_data(name)
    vocab_dict = get_vocab(name)
    name = "%s_negative_sample"%name

    method_list = [
        # consistency(name, nlp=nlp),
        # lexical_rept(name, nlp=nlp),
        # semantic_rept(name, nlp=nlp),
        # character(name, nlp=nlp, pronoun_num=6 if "wp" in name else 3),
        # coherence(name, nlp=nlp),
        # commonsense(name, nlp=nlp, sub_num=15 if "wp" in name else 3, vocab_dict=vocab_dict),
        # cause(name, nlp=nlp),
        # time(name, nlp=nlp),

        # synonym_substitute(name, nlp=nlp, sub_num=5 if "wp" in name else 2),
        semantic_substitute(name, nlp=nlp),
        # add_typos(name),
        # contraction(name),
        # delete_punct(name, nlp, punct_num=5 if "wp" in name else 2),
    ]

    for method in method_list:
        method.construct(data)