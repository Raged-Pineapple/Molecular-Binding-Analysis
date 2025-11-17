from __future__ import annotations
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


def _features_from_smiles(smiles: str) -> Dict[str, float]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("Invalid SMILES string")
    feats = {
        "logP": float(Crippen.MolLogP(m)),
        "MW": float(Descriptors.MolWt(m)),
        "HBD": float(Lipinski.NHOHCount(m)),
        "HBA": float(Lipinski.NOCount(m)),
        "RotBonds": float(Lipinski.NumRotatableBonds(m)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(m)),
    }
    return feats


class _MLP(nn.Module):  # type: ignore[misc]
    def __init__(self, in_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


_model: Optional[_MLP] = None


def _ensure_model():
    global _model
    if torch is None:
        return None
    if _model is None:
        _model = _MLP()
        # Random weights are fine; we keep determinism by setting fixed seed
        torch.manual_seed(0)
    return _model


def score_smiles(smiles: str, protein_id: str | None = None) -> Dict[str, Any]:
    """Return an ML-style score and the features used. Placeholder MLP if torch is available.
    Deterministic: fixed manual seed; otherwise returns a simple heuristic projection.
    """
    feats = _features_from_smiles(smiles)
    order = ["logP", "MW", "HBD", "HBA", "RotBonds", "TPSA"]
    x = [feats[k] for k in order]

    if torch is None:
        # Deterministic linear projection fallback (no randomness)
        w = [10.0, -0.05, -2.0, -1.0, -3.0, -0.2]
        b = 70.0
        y = b + sum(wi * xi for wi, xi in zip(w, x))
        score = float(max(1.0, min(100.0, y)))
        return {"score": score, "features": feats}

    model = _ensure_model()
    with torch.no_grad():
        import torch as _t
        inp = _t.tensor(x, dtype=_t.float32).view(1, -1)
        y = model(inp).item() if model is not None else 0.0
    # Map to [1,100]
    score = float(max(1.0, min(100.0, 50.0 + y)))
    return {"score": score, "features": feats}
