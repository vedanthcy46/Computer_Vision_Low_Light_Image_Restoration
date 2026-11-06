import numpy as np
import torch

from lowlight.dl.losses import SupervisedLoss, ZeroDCELoss
from lowlight.dl.zero_dce import DCENet, apply_curves, count_params
from lowlight.dl.infer import DLEnhancer, load_model


def test_dcenet_shapes_and_param_count():
    m = DCENet().eval()
    assert count_params(m) == 79416                              # matches the Zero-DCE paper
    x = torch.rand(2, 3, 64, 48)
    y, maps = m(x)
    assert y.shape == x.shape and maps.shape == (2, 24, 64, 48)


def test_curves_keep_range_and_fewer_iterations_are_milder():
    x = torch.rand(1, 3, 16, 16) * 0.2
    maps = torch.full((1, 24, 16, 16), -0.9)   # a < 0 brightens: x + a*(x^2 - x)
    full, half = apply_curves(x, maps), apply_curves(x, maps, 3)
    assert full.min() >= 0 and full.max() <= 1 and (full >= half - 1e-6).all() and half.mean() > x.mean()


def test_losses_are_finite_and_trainable():
    m = DCENet()
    x = torch.rand(2, 3, 64, 64) * 0.2
    y, maps = m(x)
    l1, parts = ZeroDCELoss()(x, y, maps)
    l2, _ = SupervisedLoss()(x, y, maps, torch.rand_like(x))
    assert torch.isfinite(l1) and torch.isfinite(l2) and set(parts) == {"spa", "exp", "col", "tv"}
    (l1 + l2).backward()
    assert all(p.grad is not None for p in m.parameters())


def test_training_step_reduces_exposure_loss():
    torch.manual_seed(0)
    m, crit = DCENet(), ZeroDCELoss()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    x = torch.rand(4, 3, 64, 64) * 0.15
    first = None
    for _ in range(25):
        y, maps = m(x)
        loss, _ = crit(x, y, maps)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first


def test_quality_aware_leaves_bright_images_alone(pairs, dl):
    gt = pairs[0][1]
    out = dl.enhance(gt).image
    assert abs(out.mean() - gt.mean()) < 0.1


def test_checkpoint_roundtrip(tmp_path):
    m = DCENet()
    torch.save({"model": m.state_dict()}, tmp_path / "w.pt")
    loaded, ok = load_model(tmp_path / "w.pt")
    assert ok and all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), loaded.state_dict().values()))
    _, ok2 = load_model(tmp_path / "missing.pt")
    assert not ok2
