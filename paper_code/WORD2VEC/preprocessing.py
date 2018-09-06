def preprocess(path):
    if path == 'brown':
        from nltk.corpus import brown
        datas = [[tkn.lower() for tkn in sent if tkn not in ["``", "''"]] for sent in brown.sents()]

        return datas
    else:
        with open(path, 'r', encoding='utf-8') as file:
            datas = file.read().split()
        return [datas]
