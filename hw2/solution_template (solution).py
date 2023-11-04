from math import log2
from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    # допишите ваш код здесь
    pass


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'exp2':
        return 2**y_value - 1.0
    return y_value


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    order = ys_pred.argsort(descending=True)
    index = torch.arange(len(order), dtype=torch.float64) + 1
    return (compute_gain(ys_true[order], gain_scheme) / torch.log2(index + 1)).sum().item()


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    dcg_val = dcg(ys_true, ys_pred, gain_scheme)
    dcg_best_val = dcg(ys_true, ys_true, gain_scheme)
    return dcg_val / dcg_best_val


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # допишите ваш код здесь
    pass


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    order = ys_pred.argsort(descending=True)
    return 1 / (ys_true[order].argsort(descending=True)[0] + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    # допишите ваш код здесь
    pass


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    pass
