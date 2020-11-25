# name = "wp_all" # "wp_all"

def get_data(name):
    data = []
    with open("../%s_ipt.txt"%name) as fin1:
        with open("../%s_truth.txt"%name) as fin2:
            for i, (line1, line2) in enumerate(zip(fin1, fin2)):
                if "wp" in name:
                    data.append({"id":i, "ipt":line1.strip().lower(), "truth":line2.strip().lower()})
                else:
                    data.append({"id":i, "ipt":line1.strip(), "truth":line2.strip()})
    return data

def get_negative_data(name):
    data = []
    with open("./%s/%s_negative_sample.txt"%(name, name)) as fin:
        for line in fin:
            tmp = line.strip().split("|||")
            ipt, opt = tmp[1].strip(), tmp[2].strip()
            if "wp" in name:
                data.append({"id":int(tmp[0]), "ipt":ipt.strip().lower(), "truth":opt.strip().lower()})
            else:
                data.append({"id":int(tmp[0]), "ipt":ipt.strip(), "truth":opt.strip()})
    return data

def get_vocab(name):
    vocab = {}
    with open("./%s_truth_vocab.txt"%name) as fin:
        for line in fin:
            try:
                tmp = line.strip().split("|||")[0].split()
                vocab[tmp[0]] = int(tmp[1])
            except:
                continue
    return vocab
            
import spacy
from spacy.symbols import ORTH
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer.add_special_case("[MALE]", [{ORTH:"[MALE]"}])
nlp.tokenizer.add_special_case("[FEMALE]", [{ORTH:"[FEMALE]"}])
nlp.tokenizer.add_special_case("[NEUTRAL]", [{ORTH:"[NEUTRAL]"}])
for w in [" WP ", " TT ", " IP ", " PM ", " EU ", " CW ", " FF ", " PI ", " OT ", " CS ", " RF ", " FF "]:
    nlp.tokenizer.add_special_case("[%s]"%w, [{ORTH:""}])
    nlp.tokenizer.add_special_case("[%s]"%w.lower(), [{ORTH:""}])

def get_sw():
    return ["a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","just","keep","keeps","kept","know","known","knows","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","que","quite","qv","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","value","various","very","via","viz","vs","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","zero"]

import os
def build_kg(sw=None):
    if sw is None:
        sw = get_sw()
    kg, word2kg = [], {}
    if os.path.exists('kg.txt') and os.path.exists('word2kg.txt'):
        with open("kg.txt", "r") as fin:
            print("building kg")
            for line in fin:
                tmp = line.strip().split(">>>")
                kg.append([tmp[0].strip().split(), tmp[1].strip()])
        with open("word2kg.txt", "r") as fin:
            print("building word2kg")
            for line in fin:
                tmp = line.strip().split(">>>")
                word = tmp[0].strip()
                if word not in sw:
                    avail_kg_id, kw = [], set()
                    for kg_id in list(map(int, tmp[1].strip().split())):
                        length = len(kw)
                        kw.update(kg[kg_id][0])
                        if length == len(kw) - 1:
                            avail_kg_id.append(kg_id)
                    word2kg[word] = avail_kg_id
    else:
        with open("conceptnet_triple.csv", "r") as fin:
            with open("kg.txt", "w") as fout_kg:
                head_tail = set()
                for i, line in enumerate(fin):
                    if i % 10000 == 0:
                        print("reading %d knowledge triples"%i)
                    tmp = line.strip().split("|||")
                    head_tok, tail_tok = nlp(" ".join(tmp[0].strip().split("_"))), nlp(" ".join(tmp[2].strip().split("_")))
                    head, tail = [t.lemma_ for t in head_tok], [t.lemma_ for t in tail_tok]
                    if (" ".join(head) == " ".join(tail)) or (" ".join(head) in sw) or (" ".join(tail) in sw) or (" ".join(head+tail) in head_tail):
                        continue
                    head_tail.update([" ".join(head+tail), " ".join(tail+head)])
                    for word in head + tail:
                        if word in word2kg:
                            word2kg[word].append(len(kg))
                        else:
                            word2kg[word] = [len(kg)]
                    kg.append([list(set(head + tail)), line.strip()])
                    fout_kg.write(" ".join(kg[-1][0])+">>>"+kg[-1][1]+"\n")
            with open("word2kg.txt", "w") as fout_word2kg:
                for word in word2kg:
                    fout_word2kg.write("%s>>>%s\n"%(word, " ".join(list(map(str, word2kg[word])))))    
    print("finish building kg and word2kg")
    return kg, word2kg