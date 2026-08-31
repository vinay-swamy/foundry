"""Preserve missing-density gaps when training/inferring from .pdb inputs.

atomworks' `AddWithinChainInstanceResIdx` sets `within_chain_res_idx` to a contiguous
`arange` over the residues actually present in the atom array. For CIF this still carries
gaps, because the parsed array includes placeholder tokens for residues that are in the
canonical sequence but unmodeled. A `.pdb` has no such placeholders (its parser assumes
every residue is resolved), so the index comes out gap-free ("welded"): two residues that
flank a missing-density stretch look sequence-adjacent, which corrupts the model's relative
positional encoding (`R_idx[i]-R_idx[j]`, clipped to +/-32) for neighbor pairs across gaps.

For polymer chains the deposited author numbering (`res_id`) survives the pipeline (only
non-polymer chains are renumbered), so it is the best available record of where residues
are missing. This transform recomputes `within_chain_res_idx` from `res_id` for polymer
chains: consecutive residues advance by their author-number delta (so a jump in numbering
reproduces the gap), while non-monotonic steps or insertion codes (delta <= 0) advance by 1.
The result matches the CIF `within_chain_res_idx` gap structure and, because only pairwise
differences feed the model, is also exactly how vanilla ProteinMPNN derives its residue
index (straight from PDB numbering). Non-polymer residues are left untouched.
"""
from typing import Any

import biotite.structure as struc
import numpy as np
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.transforms.base import Transform


class OverrideResIdxFromAuthorNumbering(Transform):
    """Rewrite `within_chain_res_idx` from author `res_id` on polymer chains (gap-aware)."""

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(
            data, ["chain_iid", "res_id", "is_polymer", "within_chain_res_idx"]
        )

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # collapse to one row per residue (first atom of each)
        res_starts = struc.get_residue_starts(atom_array)
        chain = atom_array.chain_iid[res_starts]
        res_id = atom_array.res_id[res_starts]
        is_poly = atom_array.is_polymer[res_starts]

        # start from the existing per-residue index; only polymer chains are rewritten
        new_res_idx = atom_array.within_chain_res_idx[res_starts].astype(np.int64).copy()

        for cid in np.unique(chain[is_poly]):
            idxs = np.where((chain == cid) & is_poly)[0]  # residue rows, in array order
            rid = res_id[idxs]
            out = np.zeros(len(idxs), dtype=np.int64)
            for k in range(1, len(idxs)):
                delta = int(rid[k] - rid[k - 1])
                out[k] = out[k - 1] + (delta if delta >= 1 else 1)
            new_res_idx[idxs] = out

        atom_array.set_annotation(
            "within_chain_res_idx",
            struc.spread_residue_wise(atom_array, new_res_idx).astype(np.int32),
        )
        data["atom_array"] = atom_array
        return data
