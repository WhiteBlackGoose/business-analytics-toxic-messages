from rnn import Encoder
from preprocess import proc
import tensorflow as tf
from tqdm import tqdm

encoder = Encoder(300, 300, 100, 1)
encoder.load_weights("encoder.weights")

w2v_dict = dict()
f = open("./glove.840B.300d.txt", "rt")
for i in tqdm(range(100000)):
    line = f.readline()
    els = line.split()
    token = els[0]
    value = tf.convert_to_tensor(list(map(float, els[1:])), dtype=tf.float32)
    w2v_dict[token] = value
f.close()

def comment_text_to_vec(comment_text):
    p = proc(comment_text, [1, 2])
    tweet = []
    for word in p:
        if word in w2v_dict:
            tweet.append(w2v_dict[word])
    if len(tweet) == 0:
        return None
    tweet = tf.stack(tweet)
    tweet = tf.expand_dims(tweet, axis=1)
    tweet = tf.expand_dims(tweet, axis=3)
    enc_hidden = encoder.initialize_hidden_state()
    for word in tweet:
        out_t, enc_hidden = encoder(word, enc_hidden)
    return enc_hidden[0].numpy()


train = pd.read_csv("./train.csv")[['target', 'comment_text']][:10000]
train['toxic'] = 1 * (train['target'] > 0.5)

X, y = [], []
for Xrow, yrow in tqdm(zip(train['comment_text'], train["toxic"])):
    v = comment_text_to_vec(Xrow)
    if v is not None:
        X.append(v)
        y.append(yrow)
