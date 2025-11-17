from __future__ import annotations
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromSmiles(smiles)
        return m
    except Exception:
        return None


def _to_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def add_methyl_group(smiles: str) -> Optional[str]:
    """Attach a methyl group to the first atom with available valence.
    Deterministic: scans atoms by index and returns the first valid mutation.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        em = Chem.EditableMol(Chem.AddHs(m))
        base = em.GetMol()
        for idx, atom in enumerate(base.GetAtoms()):
            # prefer carbon atoms with degree <= 3, else any atom with degree <= 2
            deg = atom.GetTotalDegree()
            sym = atom.GetSymbol()
            if (sym == "C" and deg <= 3) or (deg <= 2):
                c_idx = em.AddAtom(Chem.Atom("C"))
                em.AddBond(idx, c_idx, Chem.BondType.SINGLE)
                new = em.GetMol()
                Chem.SanitizeMol(new)
                return Chem.MolToSmiles(Chem.RemoveHs(new), isomericSmiles=True)
        return None
    except Exception:
        return None


def halogenate(smiles: str, element: str = "F") -> Optional[str]:
    """Attach a halogen (default F) to the first carbon with available valence.
    Deterministic selection by atom order.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    if element not in {"F", "Cl", "Br", "I"}:
        element = "F"
    try:
        em = Chem.EditableMol(Chem.AddHs(m))
        base = em.GetMol()
        for idx, atom in enumerate(base.GetAtoms()):
            if atom.GetSymbol() == "C" and atom.GetTotalDegree() <= 3:
                x_idx = em.AddAtom(Chem.Atom(element))
                em.AddBond(idx, x_idx, Chem.BondType.SINGLE)
                new = em.GetMol()
                Chem.SanitizeMol(new)
                return Chem.MolToSmiles(Chem.RemoveHs(new), isomericSmiles=True)
        return None
    except Exception:
        return None


def amide_lock(smiles: str) -> Optional[str]:
    """Convert a terminal carboxylic acid group to an amide (adds -CONH-).
    Uses a simple SMARTS reaction; returns the first product if any.
    Deterministic: returns the first enumerated product.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        rxn = AllChem.ReactionFromSmarts("[C:1](=O)[O:2][H]>>[C:1](=O)N")
        outcomes = rxn.RunReactants((m,))
        for prods in outcomes:
            pmol = prods[0]
            Chem.SanitizeMol(pmol)
            return Chem.MolToSmiles(pmol, isomericSmiles=True)
        return None
    except Exception:
        return None
