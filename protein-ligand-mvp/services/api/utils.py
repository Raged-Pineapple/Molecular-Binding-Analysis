from typing import Tuple


def smiles_from_input(ligand_type: str, value: str) -> Tuple[str, str]:
    """Return (representation_type, smiles). Currently only supports 'smiles'."""
    lt = (ligand_type or "").lower()
    if lt != "smiles":
        raise ValueError(f"Unsupported ligand.type '{ligand_type}'. Only 'smiles' is supported.")
    smiles = (value or "").strip()
    if not smiles:
        raise ValueError("Ligand value must be a non-empty SMILES string")
    return lt, smiles
