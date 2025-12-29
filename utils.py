import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import torch

def safe_corr(a, b, method="pearson"):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    if method == "pearson":
        r, _ = pearsonr(a, b)
    elif method == "spearman":
        r, _ = spearmanr(a, b)
    elif method == "kendall":
        r, _ = kendalltau(a, b)
    else:
        raise ValueError(f"Unknown method: {method}")
    return 0.0 if np.isnan(r) else r


def calculate_results_task_1(y_true: np.ndarray, y_pred: np.ndarray, **options):
    """
    计算多维指标的相关系数与 Top_Tier Accuracy。
    输入:
        y_true: np.ndarray, shape (N, 1)
        y_pred: np.ndarray, shape (N, 1)
    输出:
        results: dict
    """

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    metrics = [ "Musicality"]
    dimension_thresholds = {
        "Musicality": 4.0,
    }
    results = {}

    sum_pearson, sum_spearman, sum_kendall = 0.0, 0.0, 0.0
    sum_top_tier = 0.0

    for i, metric in enumerate(metrics):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        pearson_corr = safe_corr(yt, yp, "pearson")
        spearman_corr = safe_corr(yt, yp, "spearman")
        kendall_corr = safe_corr(yt, yp, "kendall")

        # Top_Tier accuracy
        threshold_dim = dimension_thresholds[metric]
        true_top_dim = yt >= threshold_dim
        pred_top_dim = yp >= threshold_dim

        TP_dim = np.sum((pred_top_dim == True) & (true_top_dim == True))
        FP_dim = np.sum((pred_top_dim == True) & (true_top_dim == False))
        FN_dim = np.sum((pred_top_dim == False) & (true_top_dim == True))

        precision_dim = TP_dim / (TP_dim + FP_dim + 1e-8)
        recall_dim = TP_dim / (TP_dim + FN_dim + 1e-8)
        f1_dim = 2 * precision_dim * recall_dim / (precision_dim + recall_dim + 1e-8)

        metric_result = {
            "Pearson": round(float(pearson_corr), 3),
            "Spearman": round(float(spearman_corr), 3),
            "Kendall": round(float(kendall_corr), 3),
            "Top_Tier_accuracy": round(float(f1_dim), 3)
        }

        results[metric] = metric_result

    return results

def calculate_results_task_2(y_true: np.ndarray, y_pred: np.ndarray, **options):
    """
    计算多维指标的相关系数与 Top_Tier Accuracy。
    输入:
        y_true: np.ndarray, shape (N, 5)
        y_pred: np.ndarray, shape (N, 5)
    输出:
        results: dict
    """

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    metrics = ["Coherence", "Musicality", "Memorability", "Clarity", "Naturalness"]
    dimension_thresholds = {
        "Coherence": 4.0,
        "Musicality": 4.0,
        "Memorability": 3.75,
        "Clarity": 3.75,
        "Naturalness": 4.0,
    }
    results = {}

    sum_pearson, sum_spearman, sum_kendall = 0.0, 0.0, 0.0
    sum_top_tier = 0.0

    for i, metric in enumerate(metrics):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        pearson_corr = safe_corr(yt, yp, "pearson")
        spearman_corr = safe_corr(yt, yp, "spearman")
        kendall_corr = safe_corr(yt, yp, "kendall")

        # Top_Tier accuracy
        threshold_dim = dimension_thresholds[metric]
        true_top_dim = yt >= threshold_dim
        pred_top_dim = yp >= threshold_dim

        TP_dim = np.sum((pred_top_dim == True) & (true_top_dim == True))
        FP_dim = np.sum((pred_top_dim == True) & (true_top_dim == False))
        FN_dim = np.sum((pred_top_dim == False) & (true_top_dim == True))

        precision_dim = TP_dim / (TP_dim + FP_dim + 1e-8)
        recall_dim = TP_dim / (TP_dim + FN_dim + 1e-8)
        f1_dim = 2 * precision_dim * recall_dim / (precision_dim + recall_dim + 1e-8)

        metric_result = {
            "Pearson": round(float(pearson_corr), 3),
            "Spearman": round(float(spearman_corr), 3),
            "Kendall": round(float(kendall_corr), 3),
            "Top_Tier_accuracy": round(float(f1_dim), 3)
        }

        results[metric] = metric_result

        sum_pearson += pearson_corr
        sum_spearman += spearman_corr
        sum_kendall += kendall_corr
        sum_top_tier += f1_dim

    mean_result = {
        "Pearson": round(float(sum_pearson / len(metrics)), 3),
        "Spearman": round(float(sum_spearman / len(metrics)), 3),
        "Kendall": round(float(sum_kendall / len(metrics)), 3),
        "Top_Tier_accuracy": round(float(sum_top_tier / len(metrics)), 3),
    }

    results["mean"] = mean_result
    return results

def save_networks(networks, model_path):
    weights = networks.state_dict()
    filename = '{}/{}.pth'.format(model_path, "model")    # file_name = './log/models/exp_1/model.pth
    torch.save(weights, filename)


def load_networks(networks, model_path):
    filename = '{}/{}.pth'.format(model_path, "model")    # file_name = './log/models/exp_1/model.pth
    networks.load_state_dict(torch.load(filename), strict=False)
    return networks