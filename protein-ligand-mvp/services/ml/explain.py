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


def research_recommendations(score: int, properties: Dict) -> List[str]:
    """Return actionable recommendations based on score and simple physchem rules.
    Expected keys in properties: logP, MW, HBD, HBA, RotBonds (optional TPSA).
    """
    recs: List[str] = []

    logp = float(properties.get("logP", 0.0))
    mw = float(properties.get("MW", 0.0))
    hbd = int(properties.get("HBD", 0))
    hba = int(properties.get("HBA", 0))
    rot = int(properties.get("RotBonds", 0))
    tpsa = float(properties.get("TPSA", -1.0))

    # Size and hydrophobicity
    if mw > 500:
        recs.append("Consider reducing MW below 500 to improve permeability and oral exposure")
    elif mw < 200:
        recs.append("Consider increasing MW above 200 if potency is limited by size")

    if logp > 4.0:
        recs.append("Avoid excessive hydrophobicity; introduce polar groups to lower logP")
    elif logp < 1.0:
        recs.append("Add hydrophobic moieties to increase logP and strengthen hydrophobic contacts")

    # H-bonding balance
    if hba < 2:
        recs.append("Add H-bond acceptors to improve specific interactions and solubility")
    if hbd < 1:
        recs.append("Introduce an H-bond donor to enable directional binding")
    if (hbd + hba) > 12:
        recs.append("Reduce total H-bonding groups to mitigate high polarity and clearance risk")

    # Flexibility and shape
    if rot > 10:
        recs.append("Reduce flexibility (e.g., ring closures or rigid linkers) to improve binding entropy")
    elif rot < 2:
        recs.append("Introduce limited flexibility to allow conformational adaptation in the pocket")

    # TPSA-based notes (if provided)
    if tpsa >= 0:
        if tpsa > 120:
            recs.append("Reduce TPSA (<120 Å²) to favor permeability and oral bioavailability")
        elif tpsa < 40:
            recs.append("Increase TPSA (>40 Å²) to aid solubility and specific interactions")

    # Generic score-based guidance
    if score < 60:
        recs.append("Prioritize potency drivers: enforce key pharmacophores and reduce liabilities")
    elif score < 80:
        recs.append("Fine-tune properties: subtle lipophilicity and flexibility adjustments may help")

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out
