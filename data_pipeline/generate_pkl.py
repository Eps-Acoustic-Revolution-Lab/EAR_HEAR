import glob
import os
import pickle
import argparse
import json
from pathlib import Path


def get_idx_to_scores(input_jsonl):
    idx_to_scores = {}
    with open(input_jsonl, 'r') as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            id = Path(data['file_name']).stem
            if len(data['annotation']) == 0:
                continue
            scores = {}
            for key, value in data['annotation'][0].items():
                if not isinstance(value, float):
                    continue
                scores[key] = sum([x[key] for x in data['annotation']]) / len(data['annotation'])
            idx_to_scores[id] = scores
    return idx_to_scores

def generate_pkl(idx_to_scores, embedding_txt_path, pkl_path):
    with open(embedding_txt_path, 'r') as f, open(pkl_path, 'wb') as fw:
        res = []
        for line in f.readlines():
            line = line.strip()
            audio_path, embedding_path = line.split('\t')
            id = Path(audio_path).stem.split('_')[0]
            if id not in idx_to_scores:
                continue
            scores = idx_to_scores[id]
            scores['npy_path'] = embedding_path
            res.append(scores)
        pickle.dump(res, fw)
    print(f"Saved {len(res)} to {pkl_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_embedding_txt_path', type=str, required=True)
    parser.add_argument('--save_pkl_path', type=str, required=True)
    parser.add_argument('--song_eval_metadata_jsonl', type=str, default='origin_song_eval_dataset/metadata.jsonl')
    args = parser.parse_args()
    song_eval_metadata_jsonl = args.song_eval_metadata_jsonl
    input_embedding_txt_path = args.input_embedding_txt_path
    save_pkl_path = args.save_pkl_path
    idx_to_scores = get_idx_to_scores(song_eval_metadata_jsonl)
    generate_pkl(idx_to_scores, input_embedding_txt_path, save_pkl_path)

        
if __name__ == '__main__':
    main()
