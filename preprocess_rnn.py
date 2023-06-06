import pandas as pd
from preprocess import proc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# READ TWEETS
pd.set_option('display.max_columns', None)
train = pd.read_csv("./train.csv")[['target', 'comment_text']][:10000]
train['toxic'] = 1 * (train['target'] > 0.5)

# READ WORDS DATABASE
# https://nlp.stanford.edu/projects/glove/
w2v_dict = dict()
f = open("./glove.840B.300d.txt", "rt")
for i in tqdm(range(100000)):
    line = f.readline()
    els = line.split()
    token = els[0]
    value = np.array(list(map(float, els[1:])))
    w2v_dict[token] = value
f.close()

# TOKENIZE
  
## To execute the training process
  
EPOCHS = 100

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
