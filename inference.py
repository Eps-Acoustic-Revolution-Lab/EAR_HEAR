import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_pipeline'))
from muq import MuQ
from musicfm.model.musicfm_25hz import MusicFM25Hz

import numpy as np
import torch
import librosa
from models.HEAR import HEAR
import yaml
import json
import argparse

WIN_SIZE = 30  # seconds
HOP_SIZE = 30  # seconds

# Track 1: Single label (Musicality)
LABEL_NAMES_TRACK_1 = ["Musicality"]

# Track 2: Multiple labels (5 dimensions)
LABEL_NAMES_TRACK_2 = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]

@torch.no_grad()
def get_input_feature(audio_path: str, muq: MuQ, musicfm: MusicFM25Hz):
    device = next(muq.parameters()).device
    wav, sr = librosa.load(audio_path, sr=24000)
    audio = torch.tensor(wav).unsqueeze(0).to(device)
    output = muq(audio, output_hidden_states=True)
    muq_total_output = output["hidden_states"][10].detach().cpu().float().numpy()
    muq_segment_output = []
    _, hidden_states = musicfm.get_predictions(audio)
    mf_total_output = hidden_states[10].detach().cpu().float().numpy()
    mf_segment_output = []
    for i in range(0, audio.shape[-1], HOP_SIZE * sr):
        start_idx = i
        end_idx = min(
            (i + WIN_SIZE*sr), audio.shape[-1]
        )
        audio_seg = audio[:, start_idx:end_idx]
        if audio_seg.numel() == 0:
            break
        if audio_seg.shape[-1] < 1025:
            break

        output = muq(audio_seg, output_hidden_states=True)
        muq_segment_output.append(
                output["hidden_states"][10]
                .detach()
                .cpu()
                .float()
                .numpy()
            )

        _, hidden_states = musicfm.get_predictions(audio_seg)
        mf_segment_output.append(
            hidden_states[10].detach().cpu().float().numpy()
        )

    # [1, T, D=1024]
    muq_segment_output = np.concatenate(muq_segment_output, axis=1)
    mf_segment_output = np.concatenate(mf_segment_output, axis=1)
    all_embds = [
        mf_segment_output,
        muq_segment_output,
        mf_total_output,
        muq_total_output,
    ]
    embd_lens = [x.shape[1] for x in all_embds]
    max_embd_len = max(embd_lens)
    min_embd_len = min(embd_lens)
    if abs(max_embd_len - min_embd_len) > 4:
        raise ValueError(
            f"Embedding shapes differ too much: {max_embd_len} vs {min_embd_len}"
        )

    for idx in range(len(all_embds)):
        all_embds[idx] = all_embds[idx][:, :min_embd_len, :]
    # [1, T, D=1024*4]
    embd = np.concatenate(all_embds, axis=-1)
    return embd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio_path', type=str, required=True)
    parser.add_argument('--output_json_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    input_audio_path = args.input_audio_path
    model_path = args.model_path
    output_json_path = args.output_json_path
    model_config_path = args.model_config_path
    device = args.device
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter", cache_dir="data_pipeline/hub").to(device).eval()
    musicfm = MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join("data_pipeline", "MusicFM", "msd_stats.json"),
            model_path=os.path.join("data_pipeline", "MusicFM", "pretrained_msd.pt"),
        ).to(device).eval()
    with open(model_config_path, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_config = config['model_config']
    model = HEAR(model_config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device).eval()
    feature = get_input_feature(input_audio_path, muq, musicfm)
    feature = torch.from_numpy(feature).to(device)
    attn_mask = torch.ones([1, feature.shape[1]], dtype=torch.long, device=device)
    scores = model(
        feature,
        labels_1=None,
        attention_mask_1=attn_mask,
        mode='test'
        )
    res = {'input_audio_path': input_audio_path}
    if model_config['num_classes'] == 1:
        METRICS = LABEL_NAMES_TRACK_1
    else:
        METRICS = LABEL_NAMES_TRACK_2
    for i, metric in enumerate(METRICS):
        res[metric] = scores[0, i].detach().cpu().item()
    with open(output_json_path, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=True)
        print(json.dumps(res, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
