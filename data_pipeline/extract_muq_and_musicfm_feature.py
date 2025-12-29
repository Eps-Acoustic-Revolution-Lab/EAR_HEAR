import os
os.environ['HF_HOME'] = os.path.dirname(os.path.abspath(__file__))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import librosa
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from muq import MuQ, MuQConfig
from collections import defaultdict
from musicfm.model.musicfm_25hz import MusicFM25Hz
import gpustat
from threading import Lock
from pathlib import Path
import queue
import traceback

WIN_SIZE = 30  # seconds
HOP_SIZE = 30  # seconds

class BoundThreadPoolExecutor(ThreadPoolExecutor):
    """
    对ThreadPoolExecutor 进行重写，给队列设置边界
    """
    def __init__(self, qsize: int = None, *args, **kwargs):
        self._qsize = qsize
        super(BoundThreadPoolExecutor, self).__init__(*args, **kwargs)
        if qsize is not None:
            self._work_queue = queue.Queue(maxsize=qsize)

@torch.no_grad()
def inference(audio_path: str, save_path: str, muq: MuQ, musicfm: MusicFM25Hz, lock: Lock):
    try:
        device = next(muq.parameters()).device
        wav, sr = librosa.load(audio_path, sr=24000)
        audio = torch.tensor(wav).unsqueeze(0).to(device)
        lock.acquire()
        try:
            output = muq(audio, output_hidden_states=True)
        finally:
            lock.release()
        muq_total_output = output["hidden_states"][10].detach().cpu().float().numpy()
        muq_segment_output = []
        lock.acquire()
        try:
            _, hidden_states = musicfm.get_predictions(audio)
        finally:
            lock.release()
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
            lock.acquire()
            try:
                output = muq(audio_seg, output_hidden_states=True)
            finally:
                lock.release()
            muq_segment_output.append(
                    output["hidden_states"][10]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
            lock.acquire()
            try:
                _, hidden_states = musicfm.get_predictions(audio_seg)
            finally:
                lock.release()
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
        np.save(save_path, embd)
        return audio_path, os.path.abspath(save_path)
    except Exception as e:
        traceback.print_exc()
        raise e


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, required=True)
    parser.add_argument('--output_txt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    input_txt_path = args.input_txt
    output_txt_path = args.output_txt
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    gpu_num = gpustat.gpu_count()
    muq_models = []
    musicfm_models = []
    lock_list = []
    for i in range(gpu_num):
        device = f'cuda:{i}'
        muq_models.append(MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval())
        musicfm_models.append(MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "MusicFM", "msd_stats.json"),
            model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "MusicFM", "pretrained_msd.pt"),
        ).to(device).eval())
        lock_list.append(Lock())
    with ThreadPoolExecutor(max_workers=16) as executor, \
        open(input_txt_path) as f, \
        open(output_txt_path, 'w') as fw:
        futures = []
        task_idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            save_path = os.path.join(output_path, f"{Path(line).stem}.npy")
            future = executor.submit(inference, line, save_path, muq_models[task_idx % gpu_num], musicfm_models[task_idx % gpu_num], lock_list[task_idx % gpu_num])
            futures.append(future)
            task_idx += 1
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                audio_path, save_path = future.result()
                fw.write(f"{audio_path}\t{save_path}\n")
            except Exception as e:
                print(f"Error processing: {e}")

if __name__ == '__main__':
    main()
