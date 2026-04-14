import torch
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def calculate_ece_adaptive_bins(
    token_probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins=10,
    verbose=False,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
    token_probs = token_probs.to(device)
    labels = labels.to(device)
    
    sorted_indices = torch.argsort(token_probs)
    sorted_probs = token_probs[sorted_indices]
    sorted_labels = labels[sorted_indices]

    n_samples = len(sorted_probs)
    bin_size = n_samples // n_bins

    ece = torch.zeros(1).to(device)

    if verbose:
        bin_avg_confidences = []
        bin_accuracies_list = []
        bin_conf_min = []
        bin_conf_max = []

    for i in range(n_bins):
        start_idx = i * bin_size
        if i == n_bins - 1:
            end_idx = n_samples
        else:
            end_idx = (i + 1) * bin_size

        if end_idx > start_idx:
            bin_probs = sorted_probs[start_idx:end_idx]
<<<<<<< Updated upstream
            bin_accuracies = sorted_accuracies[start_idx:end_idx]
=======
            bin_accuracies = sorted_labels[start_idx:end_idx]
>>>>>>> Stashed changes

            prop_in_bin = (end_idx - start_idx) / n_samples

            if prop_in_bin > 0:
                accuracy_in_bin = bin_accuracies.float().mean()
                avg_confidence_in_bin = bin_probs.mean()
                ece += (
                    torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                    * prop_in_bin
                )

            if verbose:
                bin_avg_confidences.append(
                    avg_confidence_in_bin.detach().cpu()
                )
                bin_accuracies_list.append(accuracy_in_bin.detach().cpu())
                bin_conf_min.append(bin_probs.min().detach().cpu())
                bin_conf_max.append(bin_probs.max().detach().cpu())

    if verbose:
        bin_avg_confidences = torch.stack(bin_avg_confidences).numpy()
        bin_accuracies_list = torch.stack(bin_accuracies_list).numpy()
        bin_conf_min = torch.stack(bin_conf_min).numpy()
        bin_conf_max = torch.stack(bin_conf_max).numpy()

<<<<<<< Updated upstream
        plt.figure(figsize=(6, 6))
=======
        plt.figure(figsize=(4, 4))
>>>>>>> Stashed changes

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Perfect calibration",
        )

        plt.plot(
            bin_avg_confidences,
            bin_accuracies_list,
            marker="o",
            linewidth=2,
            label="Model (adaptive bins)",
        )

        for i in range(len(bin_accuracies_list)):
            plt.fill_between(
                [
                    bin_conf_min[i] if i > 0 else 0,
                    bin_conf_max[i] if i < len(bin_accuracies_list) - 1 else 1,
                ],
                0,
                bin_accuracies_list[i] + 0.005,
                alpha=0.4,
            )

        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Reliability Diagram with Adaptive Bin Coverage")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

<<<<<<< Updated upstream
    return ece.item()
=======
    return ece.item()

def calculate_ece_fixed_bins(
    token_probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins=10,
    verbose=False,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
    token_probs = token_probs.to(device)
    labels = labels.to(device)
    
    n_samples = len(token_probs)
    bin_size = n_samples // n_bins

    ece = torch.zeros(1).to(device)

    if verbose:
        bin_avg_confidences = []
        bin_accuracies_list = []
        bin_conf_min = []
        bin_conf_max = []

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_mask = (token_probs >= bin_boundaries[i]) & (token_probs < bin_boundaries[i + 1])

        if i == n_bins - 1:
            bin_mask = (token_probs >= bin_boundaries[i]) & (token_probs <= bin_boundaries[i + 1])
        
        if bin_size > 0:
            bin_probs = token_probs[bin_mask]
            bin_accuracies = labels[bin_mask]
            
            prop_in_bin = bin_size / n_samples
            
            accuracy_in_bin = bin_accuracies.float().mean()
            avg_confidence_in_bin = bin_probs.mean()
            ece += (
                torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                * prop_in_bin
            )

            if verbose:
                bin_avg_confidences.append(
                    avg_confidence_in_bin.detach().cpu()
                )
                bin_accuracies_list.append(accuracy_in_bin.detach().cpu())
                bin_conf_min.append(bin_probs.min(-1).detach().cpu())
                bin_conf_max.append(bin_probs.max(-1).detach().cpu())

    if verbose:
        bin_avg_confidences = torch.stack(bin_avg_confidences).numpy()
        bin_accuracies_list = torch.stack(bin_accuracies_list).numpy()
        bin_conf_min = torch.stack(bin_conf_min).numpy()
        bin_conf_max = torch.stack(bin_conf_max).numpy()

        plt.figure(figsize=(6, 6))

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Perfect calibration",
        )

        plt.plot(
            bin_avg_confidences,
            bin_accuracies_list,
            marker="o",
            linewidth=2,
            label="Model (adaptive bins)",
        )

        for i in range(len(bin_accuracies_list)):
            plt.fill_between(
                [
                    bin_conf_min[i] if i > 0 else 0,
                    bin_conf_max[i] if i < len(bin_accuracies_list) - 1 else 1,
                ],
                0,
                bin_accuracies_list[i] + 0.005,
                alpha=0.4,
            )

        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Reliability Diagram with Fixed Bin Coverage")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ece.item()

def calculate_roc_auc(probs: torch.Tensor, labels: torch.Tensor, verbose=False):
    fpr, tpr, _ = roc_curve(labels.cpu().numpy(), probs.cpu().numpy())
    roc_auc = auc(fpr, tpr)

    if verbose:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
    return roc_auc

def calculate_entropy(prob_scores: torch.Tensor, verbose=False):
    assert torch.all((prob_scores > 0) & (prob_scores <= 1)), \
           f"prob_scores must be in (0, 1] range, but: min={prob_scores.min():.4f}, " \
           f"max={prob_scores.max():.4f}, shape={prob_scores.shape}, " \
           f"negative values: {(prob_scores < 0).sum().item()}, " \
           f">1 values: {(prob_scores > 1).sum().item()}"
    logprobs = torch.log(prob_scores)
    entropy = -(logprobs * prob_scores).sum(dim=-1)

    if verbose:
        plt.figure()
        plt.hist(prob_scores.cpu().numpy(), bins=20, density=True, alpha=0.6, color='b')
        plt.title('Probability  Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.bar(
            x=[1],
            height=entropy.item(),
            width=0.5,
            color='r',
            alpha=0.6,
            align='center'
        )
        plt.show()

    return entropy

def calculate_norm_entropy(token_scores: torch.Tensor):
    entropy = calculate_entropy(token_scores)
    norm_attn_entropy = entropy / torch.log(torch.tensor(token_scores.shape[-1]))
    return norm_attn_entropy
>>>>>>> Stashed changes
