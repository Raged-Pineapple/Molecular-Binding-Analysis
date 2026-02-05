from __future__ import annotations
from typing import List, Dict, Optional, Callable
import itertools

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors, DataStructs

from .scoring import score_smiles, predict
from .mutation import add_methyl_group, halogenate, amide_lock
from .mutation import generate_mutants
from . import docking as _docking
from . import model as _mlmodel
import logging
import os
from pathlib import Path

log = logging.getLogger("improve")


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


# ------------------------------
# New advanced, additive API (does not alter existing functions)
# ------------------------------

def _inchikey_of(smiles: str) -> str | None:
    try:
        from rdkit.Chem.inchi import MolToInchiKey
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        return MolToInchiKey(m)
    except Exception:
        return None


def _ecfp4(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if not m:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def _tanimoto(a_fp, b_fp) -> float:
    if a_fp is None or b_fp is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(a_fp, b_fp))


def score_for_improvement(smiles: str, protein_path: str | None = None, mode: str = "heuristic", quick: bool = False) -> dict:
    """Unified scoring wrapper.
    mode: heuristic | docking | ml
    Returns {score: float, detail: dict}
    """
    mode = (mode or "heuristic").lower()
    # Heuristic fast path
    if mode == "heuristic" or (mode == "docking" and not protein_path):
        res = predict(smiles)
        return {"score": float(res.get("score", 0.0)), "detail": {"calibration": res.get("calibration_info")}}
    if mode == "ml":
        try:
            mres = _mlmodel.score_smiles(smiles)
            return {"score": float(mres.get("score", 0.0)), "detail": {"features": mres.get("features", {})}}
        except Exception as e:
            log.warning("ml scoring failed; fallback to heuristic: %s", e)
            res = predict(smiles)
            return {"score": float(res.get("score", 0.0)), "detail": {"fallback": "heuristic"}}
    # Docking path: return raw affinity as score; if no affinity, raise to let caller discard
    try:
        dres = _docking.dock(smiles, protein_path, quick=quick)
        aff = dres.get("affinity")
        if aff is None:
            raise ValueError("docking produced no affinity")
        return {"score": float(aff), "detail": {"affinity": aff}}
    except Exception as e:
        raise


def _select_diverse(scored: list[dict], pop_size: int, tanimoto_threshold: float = 0.95) -> list[dict]:
    """Keep top-k with diversity by Tanimoto on ECFP4."""
    kept: list[dict] = []
    fps: list = []
    for item in scored:
        fp = _ecfp4(item["smiles"])  # may be None
        ok = True
        for f2 in fps:
            if _tanimoto(fp, f2) >= tanimoto_threshold:
                ok = False
                break
        if ok:
            kept.append(item)
            fps.append(fp)
        if len(kept) >= pop_size:
            break
    # If not enough, relax threshold
    if len(kept) < pop_size:
        for item in scored:
            if item in kept:
                continue
            kept.append(item)
            if len(kept) >= pop_size:
                break
    return kept


def advanced_genetic_optimize(
    seed_smiles: str,
    protein_path: str | None = None,
    mode: str = "heuristic",
    n_iters: int = 8,
    pop_size: int = 8,
    mutate_rate: float = 0.5,
    target_score: float | None = None,
    persist_debug_dir: str | None = None,
    quick: bool = False,
) -> dict:
    """Advanced GA with diversity and optional docking/ML scoring. Additive API.
    Returns {base_score, final:[{smiles,score,step,op_name,parent_smiles}], trace:[...]}.
    """
    # Debug directory
    dbg_dir = None
    if persist_debug_dir:
        dbg_dir = Path(persist_debug_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)

    # Base scoring
    # Base score; if docking mode and no affinity, base_score omitted (use 0.0)
    try:
        base_res = score_for_improvement(seed_smiles, protein_path, mode, quick=quick)
        base_score = float(base_res["score"])
    except Exception:
        base_score = 0.0

    # Seed population
    pop = [(seed_smiles, "seed", None)]  # (smiles, op_name, parent)
    for s, op in generate_mutants(seed_smiles, n_per_operator=2):
        pop.append((s, op, seed_smiles))

    # Seen set by InChIKey
    seen_keys = set()
    def _key(s: str) -> str:
        return _inchikey_of(s) or s

    # Score and select initial
    scored: list[dict] = []
    for s, op, parent in pop:
        k = _key(s)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        try:
            sc = score_for_improvement(s, protein_path, mode, quick=quick)
        except Exception:
            continue
        scored.append({"smiles": s, "score": sc["score"], "op_name": op, "parent_smiles": parent, "step": 0})
    # Determine sort direction based on scoring mode
    dock_mode = (mode.lower() == "docking" and bool(protein_path))
    scored.sort(key=lambda d: d["score"], reverse=not dock_mode)
    current = _select_diverse(scored, pop_size)

    trace = [{"step": 0, "smiles": it["smiles"], "score": it["score"]} for it in current]

    # GA iterations
    import random as _r
    for it in range(1, n_iters + 1):
        # Mutate current with given rate
        new_candidates: list[tuple[str, str, str]] = []
        for item in current:
            s = item["smiles"]
            # Always keep parent in pool
            new_candidates.append((s, item.get("op_name") or "carry", item.get("parent_smiles") or None))
            if _r.random() <= mutate_rate:
                muts = generate_mutants(s, n_per_operator=1)
                for ms, op in muts:
                    new_candidates.append((ms, op, s))

        # Score all new candidates
        round_scored: list[dict] = []
        for s, op, parent in new_candidates:
            k = _key(s)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            try:
                sc = score_for_improvement(s, protein_path, mode, quick=quick)
            except Exception:
                continue
            round_scored.append({"smiles": s, "score": sc["score"], "op_name": op, "parent_smiles": parent, "step": it})

        # Merge with current and select
        merged = current + round_scored
        merged.sort(key=lambda d: d["score"], reverse=not dock_mode)
        current = _select_diverse(merged, pop_size)

        trace.extend({"step": it, "smiles": it2["smiles"], "score": it2["score"]} for it2 in current)

        # optional early stop check (collect near-hits in this iteration anyway)
        if target_score is not None:
            if dock_mode and any(x["score"] <= target_score for x in current):
                break
            if (not dock_mode) and any(x["score"] >= target_score for x in current):
                # we still finish this iteration as above
                break

    # persist debug if requested
    if dbg_dir is not None:
        try:
            import csv
            with open(dbg_dir / "trace.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["step", "smiles", "score"]) 
                w.writeheader(); w.writerows(trace)
        except Exception:
            pass

    # Final list sorted
    final = sorted(current, key=lambda d: d["score"], reverse=not dock_mode)
    return {"base_score": base_score, "final": final, "trace": trace}



# ------------------------------
# New GA-style improvement API
# ------------------------------

def mutate_smiles(smiles: str) -> List[str]:
    """Generate a small, deterministic set of mutants for a SMILES string.
    Uses basic operators from mutation.py and internal RDKit edits for diversity.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    cands: List[str] = []
    # Deterministic set of mutation operators
    for fn in (add_methyl_group, amide_lock):
        try:
            out = fn(smiles)
        except Exception:
            out = None
        if out:
            cands.append(out)
    for hal in ("F", "Cl"):
        try:
            out = halogenate(smiles, hal)
        except Exception:
            out = None
        if out:
            cands.append(out)

    # De-duplicate canonical SMILES
    uniq = []
    seen = set()
    for s in cands:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, isomericSmiles=True)
        if can in seen:
            continue
        seen.add(can)
        uniq.append(can)
    return uniq


def evaluate_candidates(smiles_list: List[str], protein_path: str | None = None, quick: bool = False) -> List[tuple[str, float]]:
    """Score candidates.
    - If protein_path is provided: use docking and take raw affinity as the score; discard if no affinity.
    - Else: fallback to heuristic predictor score.
    Returns list of (smiles, score).
    """
    out: List[tuple[str, float]] = []
    for s in smiles_list:
        try:
            if protein_path:
                dres = _docking.dock(s, protein_path, quick=quick)
                aff = dres.get("affinity")
                if aff is None:
                    continue
                out.append((s, float(aff)))
            else:
                res = predict(s)
                sc = float(res.get("score", 0.0))
                out.append((s, sc))
        except Exception:
            continue
    return out


def docking_driven_improve(seed_smiles: str, protein_id: str) -> Dict[str, Any]:
    """Generates improved ligands for a protein using docking affinity as the signal."""
    # 1. Resolve Protein
    protein_path = Path("data") / "proteins" / f"{protein_id}.pdb"
    if not protein_path.exists():
        raise FileNotFoundError(f"Protein {protein_id} not found.")

    # 2. Get Base Affinity
    base_res = _docking.dock(seed_smiles, str(protein_path))
    base_affinity = base_res.get("affinity")
    if base_affinity is None:
        raise ValueError("Could not establish base affinity for docking.")

    # 3. Generate Mutations (Limit to ~10 unique mutations)
    # We use a subset of operators to stay within limits and ensure scientific relevance
    mutants_with_ops = generate_mutants(seed_smiles, n_per_operator=1)
    # Filter to top 10 unique mutants if we have more
    mutants_with_ops = mutants_with_ops[:10]

    improvements = []
    seen_smiles = {seed_smiles}

    # 4. Dock each mutant against the SAME protein
    for mut_smiles, op_name in mutants_with_ops:
        if mut_smiles in seen_smiles:
            continue
        seen_smiles.add(mut_smiles)

        try:
            # Use quick=True for inner-loop docking to speed up improvement
            res = _docking.dock(mut_smiles, str(protein_path), quick=True)
            aff = res.get("affinity")
            if aff is None:
                continue
            
            # 5. Selection Logic: mutant_affinity < base_affinity (and >= 0.3 kcal/mol improvement)
            delta = float(aff) - float(base_affinity)
            if delta <= -0.3:
                improvements.append({
                    "smiles": mut_smiles,
                    "score": float(aff),
                    "affinity": float(aff),
                    "delta_affinity": round(delta, 2),
                    "mutation_description": op_name.replace("_", " ").title(),
                    "op_name": op_name,
                    "step": 1
                })
        except Exception as e:
            log.warning(f"Failed to dock mutant {mut_smiles}: {e}")
            continue

    # 6. Ranking: Sort by affinity (lowest first)
    improvements.sort(key=lambda x: x["affinity"])
    
    # Return top 5
    top_improvements = improvements[:5]

    return {
        "base_score": float(base_affinity),
        "base_affinity": float(base_affinity),
        "improvements": top_improvements,
        "trace": [{"step": 0, "smiles": seed_smiles, "score": float(base_affinity)}] + 
                 [{"step": 1, "smiles": x["smiles"], "score": x["score"]} for x in top_improvements]
    }
