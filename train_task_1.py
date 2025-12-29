import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from music_dataset import train_data_loader, test_data_loader, music_collate_fn, LABEL_NAMES_TRACK_1
from models.HEAR import HEAR
from models.listMLE import listMLE
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
import numpy as np
from utils import calculate_results_track_1, save_networks, load_networks
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser("Training")
parser.add_argument('--train-data', type=str, default=None, help='Path to training data pkl file')
parser.add_argument('--test-data', type=str, default=None, help='Path to test data pkl file')
parser.add_argument('--experiment_name', type=str, default='track_1')
parser.add_argument('--max-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=8) 
parser.add_argument('--load-checkpoint', action='store_true', default=False)   
parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for model") 
parser.add_argument('--weight_decay', type=float, default=1e-3, help="learning rate for model") 
parser.add_argument('--accum_steps', type=int, default=4)  
parser.add_argument('--lambda', type=float, default=0.15) 
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--workers', type=int, default=8, help='workers')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--log-dir', type=str, default=None, help='Path to tensorboard log directory')

def unwrap_model(model: nn.Module) -> nn.Module:
    """If model is DataParallel wrapper, return the underlying module."""
    return model.module if hasattr(model, "module") else model

def main(options):
    torch.manual_seed(options['seed'])
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    options['device'] = device
    if use_gpu:
        print(f"Detected {torch.cuda.device_count()} GPUs.")
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Using CPU")

    options['scaler'] = GradScaler('cuda')
    
    train_data = train_data_loader(pkl_path=options.get('train_data'), label_names=LABEL_NAMES_TRACK_1)
    test_data = test_data_loader(pkl_path=options.get('test_data'), label_names=LABEL_NAMES_TRACK_1)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=options['batch_size'],
                                               shuffle=True,
                                               num_workers=options['workers'],
                                               pin_memory=True,
                                               prefetch_factor=4,
                                               persistent_workers=True, 
                                               collate_fn=music_collate_fn)  
    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=options['batch_size'],
                                               shuffle=False,
                                               num_workers=options['workers'],
                                               pin_memory=True,
                                               prefetch_factor=4,
                                               persistent_workers=True, 
                                               collate_fn=music_collate_fn)
    
    with open("config_track_1.yaml", 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_config = config['model_config']
    model = HEAR(model_config)

    if options['load_checkpoint'] == True:
        model = unwrap_model(model)
        model = load_networks(model, "log/models/test_track_1")
    model = model.to(device)
    if use_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    loss_smooth = nn.SmoothL1Loss(beta=0.5, reduction='mean')

    # 模型存储路径 'log/models/experiment_name'
    model_path = os.path.join(options['outf'], 'models', options['experiment_name'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 仅测试
    if options['eval']:
        networks = unwrap_model(model)
        networks = load_networks(networks, model_path)
        results = test(networks, test_loader, epoch=0, step=0, **options)
        print(results)
        return
    
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_params_list, lr = options['lr'], weight_decay=options['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=400)

    best_results = {}
    best_results['Musicality'] = {
        "Pearson":0, 
        "Spearman":0,
        "Kendall":0,
        "Top_Tier_accuracy":0
    }

    for epoch in range(options['max_epoch']):
        print(f"\n==> Epoch {epoch+1}/{options['max_epoch']}")
        train_loss, best_results = train(
            epoch, model, train_loader, loss_smooth, optimizer, scheduler,
            test_loader=test_loader,
            best_results=best_results,
            model_path=model_path,
            **options
        )
        print(f"Train loss: {train_loss:.4f}")
        print(f"epoch {epoch}/{options['max_epoch']}\n Current best results: {best_results}\n")
    options['writer'].add_text("best_results", str(best_results))

def train(epoch, model, train_loader, loss_smooth, optimizer, scheduler,
          test_loader, best_results, model_path, **options):
    model.train()
    torch.cuda.empty_cache()
    losses = AverageMeter()
    scaler = options.get('scaler', None)
    total_steps = len(train_loader)
    eval_interval = 4
    print(f"Run evaluation every {eval_interval} training step")
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (audios, labels, audio_masks, partner_audios, partner_labels, partner_masks) in enumerate(train_loader):
        audios = audios.to(options['device'], non_blocking=True)
        labels = labels.to(options['device'], non_blocking=True)
        audio_masks = audio_masks.to(options['device'], non_blocking=True)
        partner_audios = partner_audios.to(options['device'], non_blocking=True)
        partner_labels = partner_labels.to(options['device'], non_blocking=True)
        partner_masks = partner_masks.to(options['device'], non_blocking=True)

        with autocast('cuda', dtype=torch.bfloat16):
            preds, real_labels = model(audio_tensor_1=audios, labels_1=labels, attention_mask_1=audio_masks, audio_tensor_2=partner_audios, labels_2=partner_labels, attention_mask_2=partner_masks, return_dict=False, mode="train")
            loss_reg = loss_smooth(preds, real_labels)
            loss_rank = listMLE(preds.transpose(0, 1), real_labels.transpose(0, 1))
            total_loss = (loss_reg + loss_rank * options['lambda']) / options['accum_steps']

        step = epoch * total_steps + batch_idx
        scaler.scale(total_loss).backward()

        # 梯度更新
        if (batch_idx + 1) % options['accum_steps'] == 0 or (batch_idx + 1) == total_steps:
            scaler.unscale_(optimizer)
            max_grad_norm = 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            options['writer'].add_scalar('loss/train_grad_norm', grad_norm.item(), step)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            scheduler.step(total_loss.item())
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], 1e-8)
        losses.update(total_loss.item() * options['accum_steps'], labels.size(0))
        options['writer'].add_scalar('loss/lr', optimizer.param_groups[0]['lr'], step)
        options['writer'].add_scalar('loss/train_regression', loss_reg.item(), step)
        options['writer'].add_scalar('loss/train_rank', loss_rank.item(), step)
        options['writer'].add_scalar('loss/train_total', total_loss.item(), step)

        # 测试
        if (batch_idx + 1) % eval_interval == 0 or (batch_idx + 1) == total_steps:
            print(f"\n Step {batch_idx + 1}/{total_steps} ...")
            model.eval()
            with torch.no_grad():
                results = test(model, test_loader, epoch, step, **options)
            model.train()

            total_score = sum(results['Musicality'].values())
            best_score = sum(best_results['Musicality'].values())

            for metric_name, metric_values in results.items():
                for sub_name, value in metric_values.items():
                    options['writer'].add_scalar(f'{metric_name}/{sub_name}', value, step)
            # 判断是否保存模型
            if total_score > best_score:
                best_results = results
                print('Current best results：', results)
                print(f"Current best model，Step={step}，saving...")
                to_save = unwrap_model(model)
                save_networks(to_save, model_path)
                options['writer'].add_scalar(f'best_results_sum/sum', round(total_score / 4, 2), step)
                options['writer'].add_text("best_results", str(best_results))
        torch.cuda.empty_cache()
    return losses.avg, best_results

def test(model, test_loader, epoch=0, step=0, **options):
    model.eval()
    torch.cuda.empty_cache()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for audios, labels, audio_masks in test_loader:
            audios = audios.to(options['device'], non_blocking=True)
            labels = labels.to(options['device'], non_blocking=True)
            audio_masks = audio_masks.to(options['device'], non_blocking=True)
            preds = model(audio_tensor_1=audios, labels_1=None, attention_mask_1=audio_masks, audio_tensor_2=None, labels_2=None, attention_mask_2=None, return_dict=False, mode='test')
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            torch.cuda.empty_cache()

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()
    
    results = calculate_results_track_1(y_true, y_pred, **options)
    return results

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    options = vars(args)
    if options.get('log_dir') is None:
        log_dir = f"./log/tensorboard_records/{options['experiment_name']}"
    else:
        log_dir = options['log_dir']
    os.makedirs(log_dir, exist_ok=True)  
    options['writer'] = SummaryWriter(log_dir)
    main(options)