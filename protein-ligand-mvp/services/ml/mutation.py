from __future__ import annotations
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    try:
        m = Chem.MolFromSmiles(smiles)
        return m
    except Exception:
        return None


# ------------------------------
# Additional requested operators
# ------------------------------

def add_ring_halogen(smiles: str, element: str = "F") -> Optional[str]:
    """Add a halogen (F/Cl) to the first eligible aromatic carbon.
    Wrapper around halogenate_aryl for clarity.
    """
    if element not in {"F", "Cl"}:
        element = "F"
    return halogenate_aryl(smiles, element)


def _extend_terminal_alkyl(smiles: str) -> Optional[str]:
    """Extend a terminal alkyl by one carbon (…-CH3 -> …-CH2-CH3)."""
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        em = Chem.EditableMol(Chem.AddHs(m))
        base = em.GetMol()
        for atom in base.GetAtoms():
            if atom.IsInRing():
                continue
            if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 1:
                cidx = atom.GetIdx()
                nidx = em.AddAtom(Chem.Atom("C"))
                em.AddBond(cidx, nidx, Chem.BondType.SINGLE)
                nm = em.GetMol()
                Chem.SanitizeMol(nm)
                return Chem.MolToSmiles(Chem.RemoveHs(nm), isomericSmiles=True)
        return None
    except Exception:
        return None


def _shorten_terminal_alkyl(smiles: str) -> Optional[str]:
    """Remove one terminal carbon when safe (…-CH2-CH3 -> …-CH3)."""
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        rw = Chem.RWMol(Chem.AddHs(m))
        for atom in list(rw.GetAtoms()):
            if atom.IsInRing():
                continue
            if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 1:
                nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
                if not nbrs:
                    continue
                aidx = atom.GetIdx()
                rw.RemoveAtom(aidx)
                nm = rw.GetMol()
                Chem.SanitizeMol(nm)
                return Chem.MolToSmiles(Chem.RemoveHs(nm), isomericSmiles=True)
        return None
    except Exception:
        return None


def replace_alkyl_chain_length(smiles: str) -> Optional[str]:
    """Adjust simple terminal alkyl length by ±1 carbon; prefer extension, else shortening."""
    out = _extend_terminal_alkyl(smiles)
    if out:
        return out
    return _shorten_terminal_alkyl(smiles)


def add_hydroxyl(smiles: str) -> Optional[str]:
    """Add an -OH to a terminal carbon (wrapper of add_small_polar)."""
    return add_small_polar(smiles)


def swap_amide_ester(smiles: str) -> Optional[str]:
    """Swap amide <-> ester functional groups using simple SMARTS.
    Tries amide->ester first, then ester->amide.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        # amide (C(=O)N) to ester (C(=O)O)
        rxn1 = AllChem.ReactionFromSmarts("[C:1](=O)N[!#1:2]>>[C:1](=O)O[\2]")
        prods = rxn1.RunReactants((m,))
        for ps in prods:
            pm = ps[0]
            Chem.SanitizeMol(pm)
            return Chem.MolToSmiles(pm, isomericSmiles=True)
        # ester (C(=O)O) to amide (C(=O)N)
        rxn2 = AllChem.ReactionFromSmarts("[C:1](=O)O[!#1:2]>>[C:1](=O)N")
        prods = rxn2.RunReactants((m,))
        for ps in prods:
            pm = ps[0]
            Chem.SanitizeMol(pm)
            return Chem.MolToSmiles(pm, isomericSmiles=True)
        return None
    except Exception:
        return None


def aromatize_ring(smiles: str) -> Optional[str]:
    """Attempt to aromatize existing aromatic systems; returns canonical SMILES or None if unchanged.
    Note: This does not synthesize new aromaticity from fully aliphatic rings.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
        Chem.SetAromaticity(m)
        can = Chem.MolToSmiles(m, isomericSmiles=True)
        return can
    except Exception:
        return None


def _insert_linker(smiles: str, linker_atom: str = "C") -> Optional[str]:
    """Insert a small linker atom between the first eligible non-ring single bond.
    linker_atom in {"C","O","N"}. For "C" this behaves like -CH2- insertion.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        rw = Chem.RWMol(m)
        for bond in m.GetBonds():
            if bond.IsInRing():
                continue
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            a = bond.GetBeginAtomIdx(); b = bond.GetEndAtomIdx()
            # break a-b, insert X between
            rw.RemoveBond(a, b)
            x = Chem.Atom(linker_atom)
            x_idx = rw.AddAtom(x)
            rw.AddBond(a, x_idx, Chem.BondType.SINGLE)
            rw.AddBond(x_idx, b, Chem.BondType.SINGLE)
            nm = rw.GetMol()
            Chem.SanitizeMol(nm)
            return Chem.MolToSmiles(nm, isomericSmiles=True)
        return None
    except Exception:
        return None


def simple_ring_expansion(smiles: str) -> Optional[str]:
    """Expand a 5-member ring to 6 by inserting a -CH2- into the first 5-ring bond found."""
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        ri = m.GetRingInfo()
        five_rings = [tuple(r) for r in ri.AtomRings() if len(r) == 5]
        if not five_rings:
            return None
        ring_atoms = set(five_rings[0])
        rw = Chem.RWMol(m)
        # find first bond within the 5-ring
        for bond in m.GetBonds():
            if bond.GetBeginAtomIdx() in ring_atoms and bond.GetEndAtomIdx() in ring_atoms and bond.IsInRing():
                a = bond.GetBeginAtomIdx(); b = bond.GetEndAtomIdx()
                rw.RemoveBond(a, b)
                c_idx = rw.AddAtom(Chem.Atom("C"))
                rw.AddBond(a, c_idx, Chem.BondType.SINGLE)
                rw.AddBond(c_idx, b, Chem.BondType.SINGLE)
                nm = rw.GetMol()
                Chem.SanitizeMol(nm)
                return Chem.MolToSmiles(nm, isomericSmiles=True)
        return None
    except Exception:
        return None


def insert_linker_ch2(smiles: str) -> Optional[str]:
    return _insert_linker(smiles, "C")


def insert_linker_o(smiles: str) -> Optional[str]:
    return _insert_linker(smiles, "O")


def insert_linker_nh(smiles: str) -> Optional[str]:
    return _insert_linker(smiles, "N")


# ------------------------------
# New, conservative mutation operators (additive only)
# ------------------------------

def add_methyl(smiles: str) -> Optional[str]:
    """Add a methyl to the first eligible heavy-atom with available valence.
    Conservative and deterministic.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        em = Chem.EditableMol(Chem.AddHs(m))
        base = em.GetMol()
        for atom in base.GetAtoms():
            if atom.GetAtomicNum() in (6, 7, 8) and atom.GetTotalDegree() <= 3:
                idx = atom.GetIdx()
                c_idx = em.AddAtom(Chem.Atom("C"))
                em.AddBond(idx, c_idx, Chem.BondType.SINGLE)
                new = em.GetMol()
                Chem.SanitizeMol(new)
                return Chem.MolToSmiles(Chem.RemoveHs(new), isomericSmiles=True)
        return None
    except Exception:
        return None


def aryloxy_ether_extend(smiles: str) -> Optional[str]:
    """Extend -O-alkyl by one carbon if present: -OCH3 -> -OCH2CH3.
    Uses a simple reaction on ether linkage.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        rxn = AllChem.ReactionFromSmarts("[O:1][CH3:2]>>[O:1][CH2:3][CH3]")
        prods = rxn.RunReactants((m,))
        for ps in prods:
            pm = ps[0]
            Chem.SanitizeMol(pm)
            return Chem.MolToSmiles(pm, isomericSmiles=True)
        return None
    except Exception:
        return None


def add_small_polar(smiles: str) -> Optional[str]:
    """Append an -OH to a terminal carbon if possible.
    Keeps donor/acceptor counts modest.
    """
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        em = Chem.EditableMol(Chem.AddHs(m))
        base = em.GetMol()
        for atom in base.GetAtoms():
            if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 1:
                cidx = atom.GetIdx()
                oidx = em.AddAtom(Chem.Atom("O"))
                em.AddBond(cidx, oidx, Chem.BondType.SINGLE)
                new = em.GetMol()
                Chem.SanitizeMol(new)
                return Chem.MolToSmiles(Chem.RemoveHs(new), isomericSmiles=True)
        return None
    except Exception:
        return None


def halogenate_aryl(smiles: str, element: str = "F") -> Optional[str]:
    """Add a halogen to an aromatic carbon at the first unsubstituted position."""
    if element not in {"F", "Cl"}:
        element = "F"
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        m2 = Chem.RWMol(m)
        for a in m2.GetAtoms():
            if a.GetIsAromatic() and a.GetDegree() <= 2:
                x = Chem.Atom(element)
                xidx = m2.AddAtom(x)
                m2.AddBond(a.GetIdx(), xidx, Chem.BondType.SINGLE)
                nm = m2.GetMol()
                Chem.SanitizeMol(nm)
                return Chem.MolToSmiles(nm, isomericSmiles=True)
        return None
    except Exception:
        return None


def add_small_alkyl_branch(smiles: str) -> Optional[str]:
    """Add a small methyl branch on a terminal carbon (no ring breaking)."""
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        em = Chem.EditableMol(Chem.AddHs(m))
        base = em.GetMol()
        for atom in base.GetAtoms():
            if atom.IsInRing():
                continue
            if atom.GetSymbol() == "C" and atom.GetTotalDegree() == 1:
                cidx = atom.GetIdx()
                c2 = em.AddAtom(Chem.Atom("C"))
                em.AddBond(cidx, c2, Chem.BondType.SINGLE)
                new = em.GetMol()
                Chem.SanitizeMol(new)
                return Chem.MolToSmiles(Chem.RemoveHs(new), isomericSmiles=True)
        return None
    except Exception:
        return None


def replace_halogen(smiles: str) -> Optional[str]:
    """Swap F<->Cl on the first match found."""
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        for atom in m.GetAtoms():
            if atom.GetSymbol() == "F":
                atom.SetAtomicNum(17)
                Chem.SanitizeMol(m)
                return Chem.MolToSmiles(m, isomericSmiles=True)
            if atom.GetSymbol() == "Cl":
                atom.SetAtomicNum(9)
                Chem.SanitizeMol(m)
                return Chem.MolToSmiles(m, isomericSmiles=True)
        return None
    except Exception:
        return None


def append_ethoxy(smiles: str) -> Optional[str]:
    """Append –OCH2CH3 to a terminal OH/ether oxygen using a reaction."""
    m = _mol_from_smiles(smiles)
    if m is None:
        return None
    try:
        rxn = AllChem.ReactionFromSmarts("[O:1][H]>>[O:1][CH2][CH3]")
        prods = rxn.RunReactants((m,))
        for ps in prods:
            pm = ps[0]
            Chem.SanitizeMol(pm)
            return Chem.MolToSmiles(pm, isomericSmiles=True)
        return None
    except Exception:
        return None


def mutate_random(smiles: str, n_variants: int = 3, rng: Optional[int] = None) -> list[str]:
    """Produce up to n_variants random mutations deterministically from a seed RNG."""
    import random as _r
    if rng is not None:
        _r.seed(int(rng))
    ops = [
        add_methyl, add_small_alkyl_branch, add_small_polar,
        aryloxy_ether_extend, append_ethoxy,
        lambda s: halogenate_aryl(s, "F"), lambda s: halogenate_aryl(s, "Cl"),
        replace_halogen,
    ]
    out: list[str] = []
    for _ in range(n_variants * 2):
        fn = _r.choice(ops)
        try:
            res = fn(smiles)
        except Exception:
            res = None
        if res:
            out.append(res)
        if len(out) >= n_variants:
            break
    # dedup canonical
    uniq = []
    seen = set()
    for s in out:
        m = Chem.MolFromSmiles(s)
        if not m:
            continue
        can = Chem.MolToSmiles(m, isomericSmiles=True)
        if can in seen:
            continue
        seen.add(can)
        uniq.append(can)
    return uniq


def generate_mutants(smiles: str, n_per_operator: int = 2, rng: Optional[int] = None) -> list[tuple[str, str]]:
    """Run each operator and collect up to n_per_operator unique variants.
    Returns list of (smiles, operator_name).
    """
    candidates: list[tuple[str, str]] = []
    def _push(name: str, val: Optional[str]):
        if val:
            candidates.append((val, name))

    _push("add_methyl", add_methyl(smiles))
    _push("add_small_alkyl_branch", add_small_alkyl_branch(smiles))
    _push("add_small_polar", add_small_polar(smiles))
    _push("add_hydroxyl", add_hydroxyl(smiles))
    _push("aryloxy_ether_extend", aryloxy_ether_extend(smiles))
    _push("append_ethoxy", append_ethoxy(smiles))
    _push("halogenate_aryl_F", halogenate_aryl(smiles, "F"))
    _push("halogenate_aryl_Cl", halogenate_aryl(smiles, "Cl"))
    _push("add_ring_halogen_F", add_ring_halogen(smiles, "F"))
    _push("add_ring_halogen_Cl", add_ring_halogen(smiles, "Cl"))
    _push("replace_halogen", replace_halogen(smiles))
    _push("replace_alkyl_chain_length", replace_alkyl_chain_length(smiles))
    _push("swap_amide_ester", swap_amide_ester(smiles))
    _push("aromatize_ring", aromatize_ring(smiles))
    _push("simple_ring_expansion", simple_ring_expansion(smiles))
    _push("insert_linker_CH2", insert_linker_ch2(smiles))
    _push("insert_linker_O", insert_linker_o(smiles))
    _push("insert_linker_NH", insert_linker_nh(smiles))

    # Stochastic extras if requested
    if rng is not None:
        for s in mutate_random(smiles, n_variants=n_per_operator, rng=rng):
            candidates.append((s, "mutate_random"))

    # Deduplicate by canonical and cap per-operator
    per_op = {}
    uniq: list[tuple[str, str]] = []
    seen = set()
    for s, name in candidates:
        m = Chem.MolFromSmiles(s)
        if not m:
            continue
        can = Chem.MolToSmiles(m, isomericSmiles=True)
        key = (can, name)
        if can in seen:
            continue
        cnt = per_op.get(name, 0)
        if cnt >= n_per_operator:
            continue
        per_op[name] = cnt + 1
        seen.add(can)
        uniq.append((can, name))
    return uniq


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
