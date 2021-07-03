import argparse
from detoxify import Detoxify
import pickle
from itertools import islice

parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", "-b", type=int, default=10000, help="Batch size")
parser.add_argument('--txt_file', default='../reddit_text.txt', type=str, help='path to text')

args = parser.parse_args()

model = Detoxify('original', device='cuda')
batch_size = args.batch_size

with open(args.txt_file) as f:
    for batch_idx, batch_lines in enumerate(iter(lambda: tuple(islice(f, batch_size)), ())):
        curr_batch = []

        for i, line in enumerate(batch_lines):
            index = (batch_idx * batch_size) + i

            if index % 5000:
                print(f'Curr line: {index}', flush=True)

            prediction = model.predict(line)
            curr_batch.append((index, prediction, len(line)))

        with open(f'batch_{batch_idx}.pkl', 'wb') as handle:
            pickle.dump(curr_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)