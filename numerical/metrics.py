
import torch

def topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    with torch.no_grad():
        topk = logits.topk(k, dim=1).indices
        correct = topk.eq(y.view(-1, 1)).any(dim=1).float()
        return float(correct.mean().item())

def worstq_accuracy(logits: torch.Tensor, y: torch.Tensor, q: float = 0.10) -> float:
    with torch.no_grad():
        probs = logits.softmax(dim=1)
        conf, pred = probs.max(dim=1)
        correct = pred.eq(y).float()
        n = y.numel()
        k = max(1, int(n * q))
        hardest = torch.topk(-conf, k=k).indices
        return float(correct[hardest].mean().item())
