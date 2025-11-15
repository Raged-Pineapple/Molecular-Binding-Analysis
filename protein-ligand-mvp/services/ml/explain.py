from __future__ import annotations
from typing import List, Dict

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Lipinski, Crippen


def lacking_factors(smiles: str) -> List[Dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for lacking_factors")

    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    hbd = Lipinski.NHOHCount(mol)
    hba = Lipinski.NOCount(mol)
    logp = Crippen.MolLogP(mol)

    suggestions: List[Dict] = []

    if tpsa < 40:
        suggestions.append({
            "issue": "Low polarity (TPSA)",
            "detail": f"TPSA={tpsa:.1f} is low; consider adding H-bond donors/acceptors or polar groups.",
            "severity": "medium",
        })
    elif tpsa > 120:
        suggestions.append({
            "issue": "High polarity (TPSA)",
            "detail": f"TPSA={tpsa:.1f} is high; may hinder permeability; consider reducing polar surface.",
            "severity": "medium",
        })

    if logp < 1.0:
        suggestions.append({
            "issue": "Low lipophilicity",
            "detail": f"logP={logp:.1f}; add hydrophobic moieties to improve binding in hydrophobic pockets.",
            "severity": "low",
        })
    elif logp > 4.0:
        suggestions.append({
            "issue": "High lipophilicity",
            "detail": f"logP={logp:.1f}; risk of promiscuity/solubility issues; consider polar substituents.",
            "severity": "medium",
        })

    if rot > 10:
        suggestions.append({
            "issue": "High flexibility",
            "detail": f"Rotatable bonds={rot}; consider ring closures or rigidifying linkers.",
            "severity": "high",
        })
    elif rot < 2:
        suggestions.append({
            "issue": "Low flexibility",
            "detail": f"Rotatable bonds={rot}; may limit conformational adaptation; consider flexible linkers.",
            "severity": "low",
        })

    if hbd < 1 and hba < 2:
        suggestions.append({
            "issue": "Few H-bonding groups",
            "detail": f"HBD={hbd}, HBA={hba}; consider donors/acceptors to form specific interactions.",
            "severity": "medium",
        })

    return suggestions
