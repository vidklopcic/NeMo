import argparse
import json
import os

import editdistance
import librosa
import numpy as np
import tqdm
from scipy.io import wavfile

parser = argparse.ArgumentParser(description='Data Tail Checker')
parser.add_argument('--manifest', help='path to JSON manifest file', required=True)
parser.add_argument('--sr', type=int, help='sample rate of the audio', default=16000)


def filter(manifest, sr_target):
    """
    Filters out examples with high absolute mean value at the end of the segment,
    applies trims the audio,
    removes low wer segments
    """
    if not os.path.exists(manifest):
        raise ValueError(f'{manifest} not found')

    updated_manifest = manifest.replace('.json', '_tail.json')
    NUM_TAIL_CHARS = 5
    NUM_TAIL_SAMPLES = int(0.05 * args.sr)

    print(f'Processing {manifest}')
    with open(args.manifest, 'r') as f:
        with open(updated_manifest, 'w') as fo:
            for line in tqdm.tqdm(f):
                item = json.loads(line)
                item['td_1'] = editdistance.eval(item['text'][:NUM_TAIL_CHARS], item['pred_text'][:NUM_TAIL_CHARS])
                item['td_2'] = editdistance.eval(item['text'][-NUM_TAIL_CHARS:], item['pred_text'][-NUM_TAIL_CHARS:])
                sr, signal = wavfile.read(item['audio_filepath'])

                assert len(signal.shape) == 1
                assert sr == sr_target
                item['ta_1'] = np.mean(np.abs(signal[:NUM_TAIL_SAMPLES]))
                item['ta_2'] = np.mean(np.abs(signal[-NUM_TAIL_SAMPLES:]))

                fo.write(json.dumps(item) + '\n')
    print(f'Updated manifest saved at {updated_manifest}')


if __name__ == '__main__':
    args = parser.parse_args()
    filter(args.manifest, args.sr)
