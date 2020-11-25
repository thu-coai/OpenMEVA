from eva.perturb.perturb import *
name = "story"
data = [
        {
            'id': 1,
            'ipt': "his parents were overweight .",
            'truth': "he was overweight as well . so the doctors told his parents it was unhealthy . his parents understood and decided to make a change . they got themselves and him on a diet .",
        },
        {
            'id': 2,
            'ipt': "he was out jogging one morning .",
            'truth': "the weather was crisp and cool . then , he felt good and energetic . he decided to keep running longer than normal . he went several more miles out of his way ."
        },
    ]

method_list = [
    lexical_rept(name, nlp=nlp),
    semantic_rept(name, nlp=nlp),
    character(name, nlp=nlp, pronoun_num=2),
    consistency(name, nlp=nlp),
    coherence(name, nlp=nlp),
    commonsense(name, nlp=nlp, sub_num=1),
    cause(name, nlp=nlp),
    time(name, nlp=nlp),

    synonym_substitute(name, nlp=nlp, sub_num=2),
    semantic_substitute(name, nlp=nlp),
    add_typos(name),
    contraction(name),
    delete_punct(name, nlp, punct_num=1),
]

for method in method_list:
    method.construct(data)