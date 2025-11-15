from __future__ import annotations
from typing import List, Dict, Optional, Callable
import itertools

from rdkit import Chem
from rdkit.Chem import AllChem

from .scoring import score_smiles


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("Invalid SMILES for improver")
    return m


def _to_smiles(mol: Chem.Mol) -> str:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def mutate_add_halogen(mol: Chem.Mol, halogen: str = "F") -> Optional[str]:
    em = Chem.EditableMol(Chem.AddHs(mol))
    # choose first carbon with available valence
    for idx, atom in enumerate(em.GetMol().GetAtoms()):
        if atom.GetSymbol() == "C" and atom.GetTotalDegree() <= 3:
            x = Chem.Atom(halogen)
            x_idx = em.AddAtom(x)
            em.AddBond(idx, x_idx, Chem.BondType.SINGLE)
            new = em.GetMol()
            Chem.SanitizeMol(new)
            return Chem.MolToSmiles(Chem.RemoveHs(new), isomericSmiles=True)
    return None


def mutate_methylate(mol: Chem.Mol) -> Optional[str]:
    em = Chem.EditableMol(Chem.AddHs(mol))
    for idx, atom in enumerate(em.GetMol().GetAtoms()):
        if atom.GetTotalDegree() <= 3:
            c_idx = em.AddAtom(Chem.Atom("C"))
            em.AddBond(idx, c_idx, Chem.BondType.SINGLE)
            # add 3 hydrogens to methyl carbon implicitly
            new = em.GetMol()
            Chem.SanitizeMol(new)
            return Chem.MolToSmiles(Chem.RemoveHs(new), isomericSmiles=True)
    return None


def mutate_amide_lock(mol: Chem.Mol) -> Optional[str]:
    # Convert carboxylic acid to amide (adds NH if acid present)
    rxn = AllChem.ReactionFromSmarts("[C:1](=O)[O:2][H]>>[C:1](=O)N")
    try:
        ps = rxn.RunReactants((mol,))
        for prods in ps:
            pmol = prods[0]
            Chem.SanitizeMol(pmol)
            return Chem.MolToSmiles(pmol, isomericSmiles=True)
    except Exception:
        return None
    return None


def _unique_valid(cands: List[Optional[str]]) -> List[str]:
    out = []
    seen = set()
    for s in cands:
        if not s:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        can = Chem.MolToSmiles(m, isomericSmiles=True)
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
    return out


def improve_until(smiles: str, target: int, max_iters: int) -> Dict:
    current = smiles
    best_score = -1
    best_smiles = current
    history: List[Dict] = []

    for i in range(1, max_iters + 1):
        # generate mutations
        mol = _mol_from_smiles(current)
        variants = [
            mutate_add_halogen(mol, "F"),
            mutate_add_halogen(mol, "Cl"),
            mutate_methylate(mol),
            mutate_amide_lock(mol),
        ]
        variants = _unique_valid(variants)
        if not variants:
            # if no variants generated, keep current
            variants = [current]

        # score variants
        scored = []
        for v in variants:
            sc = score_smiles(v)
            scored.append((v, sc["calibrated_score"], sc["raw_score"]))

        # pick best candidate
        scored.sort(key=lambda x: (x[1], x[2]))
        v_best, v_best_cal, v_best_raw = scored[-1]

        history.append({
            "iter": i,
            "smiles": v_best,
            "raw_score": float(v_best_raw),
            "calibrated_score": int(v_best_cal),
        })

        if v_best_cal > best_score:
            best_score = v_best_cal
            best_smiles = v_best

        current = v_best
        if v_best_cal >= target:
            break

    return {
        "achieved": best_score >= target,
        "best_smiles": best_smiles,
        "best_score": int(best_score),
        "history": history,
    }
