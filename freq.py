import numpy as np

def read_kw():
    f = open("./freq_keywords", "rt")
    w = f.read()
    f.close()
    return w.split(",")

def word_list2freq_dict(kw, words):
    d = dict()
    for w in words:
        if w not in d:
            d[w] = 0
        d[w] += 1
    v = []
    for kword in kw:
        if kword in d:
            v.append(float(d[kword]))
        else:
            v.append(0.0)
    return np.array(v)
