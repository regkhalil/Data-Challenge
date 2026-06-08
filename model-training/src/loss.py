import torch


def _weighted_err(pred, gt):
    if gt.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    w = 1.0 / 30.0 + gt
    return (w * (pred - gt) ** 2).sum() / w.sum()


def challenge_loss(pred: torch.Tensor, gt: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
    """
    Loss = métrique officielle exacte : (ErrF + ErrM)/2 + |ErrF - ErrM|
    Optimiser cette loss = optimiser directement ce qu'on évalue.
    gender : 0.0 = Femme, 1.0 = Homme
    """
    pred = pred.squeeze()
    gt = gt.squeeze()
    gender = gender.squeeze()

    mask_f = gender == 0.0
    mask_m = gender == 1.0

    # Fallback si batch mono-genre (rare avec weighted sampler)
    if mask_f.sum() == 0 or mask_m.sum() == 0:
        w = 1.0 / 30.0 + gt
        return (w * (pred - gt) ** 2).sum() / w.sum()

    err_f = _weighted_err(pred[mask_f], gt[mask_f])
    err_m = _weighted_err(pred[mask_m], gt[mask_m])
    return (err_f + err_m) / 2.0 + torch.abs(err_f - err_m)


def compute_score(pred: torch.Tensor, gt: torch.Tensor, gender: torch.Tensor):
    """Retourne (ErrF, ErrM, Score) pour le monitoring — identique à la loss."""
    mask_f = gender == 0.0
    mask_m = gender == 1.0
    err_f = _weighted_err(pred[mask_f], gt[mask_f])
    err_m = _weighted_err(pred[mask_m], gt[mask_m])
    score = (err_f + err_m) / 2.0 + torch.abs(err_f - err_m)
    return err_f.item(), err_m.item(), score.item()
