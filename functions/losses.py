import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

def cond_noise_estimation_loss(
                          model,
                          trg_model,
                          encoder,
                          trg_step,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e1: torch.Tensor,
                          e2: torch.Tensor,
                          b: torch.Tensor, keepdim=False):

    trg_t = torch.clip(t-trg_step, min=0, max=999)
    t = torch.clip(t, min=0, max=999)

    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e1 * (1.0 - a).sqrt()

    a_trg = (1-b).cumprod(dim=0).index_select(0, trg_t).view(-1, 1, 1, 1)
    x_trg = x0 * a_trg.sqrt() + e2 * (1.0 - a_trg).sqrt()
    eps_trg_pred = trg_model(x_trg, trg_t.float())
    trg_x0 = (x_trg - (1.0 - a_trg).sqrt() * eps_trg_pred) / a_trg.sqrt()
    cond = encoder(trg_x0, trg_t)

    eps_pred = model(x, t.float(), cond)
    x0_pred = (x - (1.0 - a).sqrt() * eps_pred) / a.sqrt()

    if keepdim:
        return ((trg_x0 - x0_pred) * 0.5).square().sum(dim=(1, 2, 3))
    else:
        return ((trg_x0 - x0_pred) * 0.5).square().sum(dim=(1, 2, 3)).mean(dim=0)

loss_registry = {
    'simple': noise_estimation_loss,
    'cond_simple': cond_noise_estimation_loss
}
