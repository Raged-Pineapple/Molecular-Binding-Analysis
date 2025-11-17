from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import random

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from . import explain
from .mutation import add_methyl_group, halogenate, amide_lock


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES: could not parse molecule")
    mol = Chem.AddHs(mol)
    return mol


def compute_raw_affinity_proxy(mol: Chem.Mol) -> float:
    """Heuristic placeholder affinity proxy.
    Combines size, lipophilicity, polarity, and flexibility into a mock score.
    """
    try:
        base = 0.0
        heavy = mol.GetNumHeavyAtoms()
        rings = rdMolDescriptors.CalcNumRings(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = Lipinski.NHOHCount(mol)
        hba = Lipinski.NOCount(mol)
        rot = Lipinski.NumRotatableBonds(mol)

        base += min(heavy, 50) * 1.2
        base += max(0.0, 3.0 - abs(logp - 2.5)) * 5.0
        base += max(0.0, 2 - rings) * 1.5  # too many rings penalized lightly
        base -= max(0.0, (tpsa - 90.0)) * 0.2
        base -= max(0, rot - 8) * 1.0
        base -= max(0, (hbd + hba) - 10) * 0.5

        # small random jitter to avoid ties
        base += random.uniform(-1.0, 1.0)
        return float(base)
    except Exception as e:
        raise RuntimeError(f"Failed to compute affinity proxy: {e}")


def calibrate_to_0_100(raw: float) -> int:
    """Map raw score to [0, 100] using a smooth logistic scaling and clamp."""
    # Shift-scale raw roughly into a reasonable range before logistic
    x = (raw - 20.0) / 15.0
    prob = 1.0 / (1.0 + math.exp(-x))
    score = int(round(prob * 100))
    if score < 0:
        score = 0
    if score > 100:
        score = 100
    return score


def fake_interactions() -> List[Dict[str, Any]]:
    return [
        {"type": "H-bond", "residue": "ASP25", "description": "Donor-acceptor contact", "score": 0.7},
        {"type": "Hydrophobic", "residue": "LEU84", "description": "Aromatic face contact", "score": 0.5},
        {"type": "Pi-stacking", "residue": "PHE52", "description": "Parallel displaced stacking", "score": 0.6},
    ]


def _molblock_3d_from_smiles(smiles: str) -> Tuple[Chem.Mol, str]:
    mol = _mol_from_smiles(smiles)
    # Generate 3D conformer
    params = AllChem.ETKDGv3()
    params.randomSeed = 13
    if AllChem.EmbedMolecule(mol, params) != 0:
        # fallback: try without v3
        if AllChem.EmbedMolecule(mol) != 0:
            raise RuntimeError("Failed to embed 3D conformer for the ligand")
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    block = Chem.MolToMolBlock(mol)
    return mol, block


def viewer_payload_from_mol(protein_id: str, smiles: str) -> Dict[str, Any]:
    mol, block = _molblock_3d_from_smiles(smiles)
    return {
        "protein_id": protein_id,
        "ligand_sdf": block,
    }


def score_smiles(smiles: str) -> Dict[str, Any]:
    mol = _mol_from_smiles(smiles)
    raw = compute_raw_affinity_proxy(mol)
    calibrated = calibrate_to_0_100(raw)
    vp = viewer_payload_from_mol("unknown", smiles)
    inters = fake_interactions()
    return {
        "raw_score": raw,
        "calibrated_score": calibrated,
        "viewer_payload": vp,
        "interactions": inters,
    }


def predict(smiles: str, protein_id: str = "unknown") -> Dict[str, Any]:
    """Deterministic heuristic predictor returning a normalized [1–100] score.
    Uses simple RDKit descriptors: logP, HBD/HBA, rotatable bonds, MW.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Compute descriptors
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NHOHCount(mol)
    hba = Lipinski.NOCount(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    # Heuristic scoring (deterministic and fast)
    score = 100.0

    # logP optimal ~2.5 (parabolic penalty)
    score -= min(40.0, (abs(logp - 2.5) * 10.0))

    # Rotatable bonds: soft cap at 6
    if rot > 6:
        score -= (rot - 6) * 3.0

    # MW window 200-500, penalize outside
    if mw < 200:
        score -= min(20.0, (200 - mw) * 0.05)
    elif mw > 500:
        score -= min(30.0, (mw - 500) * 0.05)

    # HBD/HBA upper bounds
    if hbd > 3:
        score -= (hbd - 3) * 2.0
    if hba > 10:
        score -= (hba - 10) * 1.0

    # Normalize and clamp to [1, 100]
    score = max(1.0, min(100.0, score))

    # Lacking factors
    lacking: List[str] = []
    if logp > 4.5:
        lacking.append("too hydrophobic (logP > 4.5)")
    if rot > 10:
        lacking.append("too flexible (rotatable bonds > 10)")
    if mw > 600:
        lacking.append("too large (MW > 600)")

    explanations = [
        f"logP={logp:.2f}",
        f"HBD/HBA={hbd}/{hba}",
        f"RotBonds={rot}",
        f"MW={mw:.1f}",
        f"TPSA={tpsa:.1f}",
    ]
    if lacking:
        explanations.append("Warnings: " + "; ".join(lacking))

    # Optional research recommendations
    try:
        props = {"logP": logp, "MW": mw, "HBD": hbd, "HBA": hba, "RotBonds": rot, "TPSA": tpsa}
        recs = explain.research_recommendations(int(round(score)), props)
        for r in recs[:5]:  # cap to avoid verbosity
            explanations.append(f"Rec: {r}")
    except Exception:
        pass

    # Generate simple 3D conformer SDF for frontend viewer
    try:
        vp = viewer_payload_from_mol(protein_id, smiles)
    except Exception:
        vp = {"protein_id": protein_id, "ligand_sdf": ""}

    return {
        "score": float(round(score, 1)),
        "calibration_info": "heuristic-v1",
        "explanations": explanations,
        "viewer_payload": vp,
    }


def stub_improve(smiles: str, target_score: int) -> List[Dict[str, Any]]:
    """Generate simple deterministic mutants and keep those scoring >= base.
    Returns up to 3 best unique SMILES with their scores.
    """
    # Base score using current heuristic predictor
    base_res = predict(smiles)
    base_score = float(base_res.get("score", 0.0))

    # Generate candidates via deterministic mutations
    candidates: List[str] = []
    for fn in (add_methyl_group, amide_lock):
        try:
            out = fn(smiles)
        except Exception:
            out = None
        if out:
            candidates.append(out)
    # Include two halogens deterministically
    for hal in ("F", "Cl"):
        try:
            out = halogenate(smiles, hal)
        except Exception:
            out = None
        if out:
            candidates.append(out)

    # Deduplicate by canonical SMILES
    uniq = []
    seen = set()
    for s in candidates:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        can = Chem.MolToSmiles(m, isomericSmiles=True)
        if can in seen:
            continue
        seen.add(can)
        uniq.append(can)

    # Score and filter >= base
    scored: List[Dict[str, Any]] = []
    for s in uniq:
        try:
            res = predict(s)
            sc = float(res.get("score", 0.0))
            if sc >= base_score:
                scored.append({"smiles": s, "score": sc})
        except Exception:
            continue

    # Sort by score desc, keep up to 3
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:3]
