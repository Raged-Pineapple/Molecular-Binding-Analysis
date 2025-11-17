from typing import Tuple, Iterable
from pathlib import Path



def smiles_from_input(ligand_type: str, value: str) -> Tuple[str, str]:
    """Return (representation_type, smiles). Currently only supports 'smiles'."""
    lt = (ligand_type or "").lower()
    if lt != "smiles":
        raise ValueError(f"Unsupported ligand.type '{ligand_type}'. Only 'smiles' is supported.")
    smiles = (value or "").strip()
    if not smiles:
        raise ValueError("Ligand value must be a non-empty SMILES string")
    return lt, smiles


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_pdb_preview(lines: Iterable[str]) -> Tuple[int, int]:
    """Parse basic PDB preview information: number of atoms and residues.
    Counts ATOM/HETATM lines and unique residue identifiers (chain, resseq, icode).
    """
    atom_count = 0
    residues = set()
    for ln in lines:
        if not ln:
            continue
        rec = ln[0:6].strip().upper()
        if rec in ("ATOM", "HETATM"):
            atom_count += 1
            chain = (ln[21:22] or "").strip()
            resseq = (ln[22:26] or "").strip()
            icode = (ln[26:27] or "").strip()
            resname = (ln[17:20] or "").strip()
            residues.add((chain, resseq, icode, resname))
    return atom_count, len(residues)


def write_bytes(path: Path, data: bytes) -> None:
    path.write_bytes(data)
