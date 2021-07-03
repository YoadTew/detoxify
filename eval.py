from detoxify import Detoxify
import matplotlib.pyplot as plt
import numpy as np
import pickle

model = Detoxify('original', device='cuda')

with open('../reddit_text.txt') as f:
    lines = f.readlines()[:1200]

batch_size = 500

for batch in range(len(lines) // batch_size):
    curr_batch = []

    for i in range(batch_size * batch, (batch_size * batch) + batch_size):
        line = lines[i]

        prediction = model.predict(line)

        curr_batch.append((i, prediction))

    with open(f'batch_{batch}.pkl', 'wb') as handle:
        pickle.dump(curr_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)