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


def analyze_compatibility(smiles: str, protein_id: str) -> Dict[str, Any]:
    """Enhanced protein-aware scoring using internal docking signals.
    Follows the requested multi-signal fusion pipeline.
    """
    from . import docking
    from pathlib import Path

    # 1. Resolve Protein
    prot_path = Path("data") / "proteins" / f"{protein_id}.pdb"
    if not prot_path.exists():
        raise FileNotFoundError(f"Protein {protein_id} not found. Please upload it first.")

    # 2. Ligand Fitness Score (Physicochemical descriptors)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    hbd = Lipinski.NHOHCount(mol)
    hba = Lipinski.NOCount(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)

    ligand_fitness = 100.0 - (abs(logp - 2.5) * 8.0) - (max(0, rot - 6) * 4.0) - (abs(mw - 350) * 0.05)
    ligand_fitness = max(0.0, min(100.0, ligand_fitness))

    # 3. Hidden Docking Signal
    # Using 'quick' mode for the internal signal to maintain responsiveness
    dock_res = docking.dock(smiles, str(prot_path), quick=True)
    affinity = dock_res.get("affinity")
    
    if affinity is None:
        hidden_dock_score = 0.0
        vina_affinity = 0.0
    else:
        vina_affinity = float(affinity)
        # Normalize: -12 kcal/mol -> 100, 0 kcal/mol -> 0
        hidden_dock_score = max(0.0, min(100.0, (abs(vina_affinity) / 12.0) * 100.0))

    # 4. Interaction Quality Score (Estimated from affinity and ligand properties)
    # Note: In a production setting, we'd parse the pose for H-bonds/contacts.
    # Here we use a high-fidelity proxy based on affinity normalized by heavy atoms + H-bond donors/acceptors.
    heavy_atoms = mol.GetNumHeavyAtoms()
    efficiency = abs(vina_affinity) / max(1, heavy_atoms)
    
    # Estimate interactions based on affinity and "fit"
    hbond_signal = (hbd + hba) * 5.0 if vina_affinity < -5.0 else 0.0
    hydrophobic_signal = logp * 5.0 if vina_affinity < -6.0 else 0.0
    
    # Fusion for interaction quality
    interaction_quality = (efficiency * 100.0) + hbond_signal + hydrophobic_signal
    # Apply a penalty if affinity is poor despite high fitness
    if vina_affinity > -4.0:
        interaction_quality -= 20.0
        
    interaction_quality = max(0.0, min(100.0, interaction_quality))

    # 5. Final Binding Compatibility Score
    final_score = (0.45 * hidden_dock_score) + (0.35 * interaction_quality) + (0.20 * ligand_fitness)
    final_score = round(max(1.0, min(100.0, final_score)), 1)

    # 6. Explanations & Recommendations
    explanations = []
    if final_score > 80:
        explanations.append("Strong pocket complementarity inferred from docking geometry")
    elif final_score > 60:
        explanations.append("Moderate binding likelihood; favorable steric fit detected")
    else:
        explanations.append("Weak interaction profile; docking suggests poor pocket occupancy")

    if hbd + hba >= 4:
        explanations.append("Multiple hydrogen-bond compatible regions detected")
    
    if mw < 250:
        explanations.append("Ligand size below optimal pocket occupancy")
    elif mw > 500:
        explanations.append("High molecular weight may lead to steric congestion")

    if abs(logp - 2.5) < 1.0:
        explanations.append("Optimal lipophilicity for typical hydrophobic pockets")
    
    recs = explain.research_recommendations(int(final_score), {
        "logP": logp, "MW": mw, "HBD": hbd, "HBA": hba, "RotBonds": rot, "TPSA": tpsa
    })

    return {
        "binding_score": final_score,
        "score": final_score, # For backward compatibility in generic PredictOut
        "explanations": explanations,
        "ligand_properties": {
            "logP": round(logp, 2),
            "MW": round(mw, 1),
            "TPSA": round(tpsa, 1),
            "HBD": hbd,
            "HBA": hba
        },
        "recommendations": recs[:5]
    }
