import numpy as np
np.random.seed(520)

def tokenize(sentence, tokenizer, max_sequence_length=None):
    token_sent = list(map(lambda x: str(x), list(tokenizer.tokenize(sentence))))
    if max_sequence_length is None:
        return token_sent
    else:
        return token_sent[:max_sequence_length]


def normalize_score(score_list):
    smin = min(score_list)
    smax = max(score_list)
    return [(s - smin) / (smax - smin + 1e-42) for s in score_list]

def pro_emb(f, unk_name="<unk>"):
    '''
    :param path: a path of english word embbeding file 'glove.840B.300d.txt'
    :return: a list, element is a 301 dimension word embbeding, it's form like this
            ['- 0.12332 ... -0.34542\n', ', 0.23421 ... -0.456733/n', ..., 'you 0.34521 0.78905 ... -0.23123/n']
    '''
    embed = {}
    with open(f, 'r', encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            if i % 1e4 == 0:
                print("processing %d line"%i)
            line = line.strip().split()
            assert line[0] not in embed
            embed[line[0]] = [float(n) for n in line[1:]]
            embed_dim = len(embed[line[0]])
    embed[unk_name] = np.random.normal(size=[embed_dim])
    return embed, embed_dim
 
def word2vec(word, embed, unk_name="<unk>"):
    '''
    :param sentence: s string
    :return: a list, the form like [[word_vector1],...,[word_vectorn]], save per word embbeding of a sentence.
    '''
    if word in embed:
        return embed[word]
    else:
        return embed[unk_name]


def sent2vec(word_list, embed, unk_name="<unk>"):
    word_list_embed = []
    for word in word_list:
        word_list_embed.append(word2vec(word=word, embed=embed, unk_name=unk_name))
    return word_list_embed
 
 
def cosine_similarity(x, y, norm=False):
    x, y = np.array(x), np.array(y)
    cos = np.dot(x,y)/(np.linalg.norm(x)*(np.linalg.norm(y)))
    return 0.5 * cos + 0.5 if norm else cos
