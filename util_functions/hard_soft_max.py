import torch


def DiffSoftmax(logits: torch.Tensor, tau=1.0, hard=False, dim=-1):
    """
    使用 detach() 防止梯度回传到 y_soft，从而实现梯度直通。
    """
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        # detach() is needed to prevent backpropagation to y_soft.
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
