from pathlib import Path
from time import time
import random
import argparse
import warnings
import utils

import re
from tqdm import tqdm
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
from Bio.PDB import PDBIO, Select
from rdkit import Chem
from analysis.molecule_builder import build_molecule
from analysis.metrics import rdmol_to_smiles
# If openbabel is available in the environment, it can be used for Mol2 conversion
try:
    from openbabel import openbabel
except ImportError:
    openbabel = None

import constants
from constants import dataset_params, covalent_radii

from rdkit.Chem import ChemicalFeatures, rdMolDescriptors
from rdkit import RDConfig

fdef = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
)
# Use PDBBind dataset parameters (atom and amino acid encoders/decoders)
dataset_info = dataset_params['bindingmoad']
amino_acid_dict = dataset_info['aa_encoder']
atom_dict = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

class Model0(Select):
    """Select only model id 0 (first model) when saving PDB."""
    def accept_model(self, model):
        return model.id == 0

def get_physchem_flags(mol, atom_indices):
    """ Return lists of [donor, acceptor, aromatic, in_ring] per selected atom idx. """
    feats = fdef.GetFeaturesForMol(mol)
    donors    = {f.GetAtomIds()[0] for f in feats if f.GetFamily()=="Donor"}
    acceptors = {f.GetAtomIds()[0] for f in feats if f.GetFamily()=="Acceptor"}
    flags = []
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        aromatic = atom.GetIsAromatic()
        in_ring  = atom.IsInRing()
        flags.append([
            1 if idx in donors    else 0,
            1 if idx in acceptors else 0,
            1 if aromatic         else 0,
            1 if in_ring          else 0
        ])
    return np.array(flags, dtype=np.float32)

def process_ligand_from_file(ligand_path: Path):
    """
    Load ligand from file (SDF or MOL2) and return its coordinates and one-hot atom features.
    Prefers SDF format; falls back to MOL2 if needed.
    Returns:
        ligand_data: dict with 'lig_coords' and 'lig_one_hot' numpy arrays.
    Raises:
        KeyError if an atom type is not in the atom_dict.
        FileNotFoundError if no ligand file can be found.
        ValueError if RDKit cannot parse any ligand file.
    """
    lig_mol = None
    # Try SDF first
    if ligand_path.with_suffix('.sdf').exists():
        sdf_path = ligand_path.with_suffix('.sdf')
        lig_mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    # If SDF not found or failed, try MOL2
    if lig_mol is None:
        mol2_path = ligand_path.with_suffix('.mol2')
        if mol2_path.exists():
            try:
                lig_mol = Chem.MolFromMol2File(str(mol2_path), removeHs=False)
            except Exception as e:
                lig_mol = None
        else:
            # No ligand file found
            raise FileNotFoundError(f"No ligand file (.sdf or .mol2) found for {ligand_path.stem}")
    # If RDKit failed to load, try OpenBabel conversion as last resort
    if lig_mol is None and openbabel is not None:
        mol2_path = ligand_path.with_suffix('.mol2')
        if mol2_path.exists():
            # Convert MOL2 to SDF using OpenBabel
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("mol2", "sdf")
            obMol = openbabel.OBMol()
            success = obConversion.ReadFile(obMol, str(mol2_path))
            if success:
                tmp_sdf = ligand_path.with_name("temp_conv.sdf")
                obConversion.WriteFile(obMol, str(tmp_sdf))
                lig_mol = Chem.MolFromMolFile(str(tmp_sdf), removeHs=False)
                try:
                    tmp_sdf.unlink()  # remove the temporary file
                except Exception:
                    pass
    if lig_mol is None:
        # Unable to parse ligand file
        raise ValueError(f"Could not parse ligand file for {ligand_path.stem}")
    # Filter atoms (exclude hydrogens if not in dictionary)
    atoms = []
    for atom in lig_mol.GetAtoms():
        sym = atom.GetSymbol().capitalize()
        if sym in atom_dict or sym != 'H':
            atoms.append(atom)
    # Coordinates for selected atoms
    conf = lig_mol.GetConformer()
    lig_coords = np.array([list(conf.GetAtomPosition(atom.GetIdx())) for atom in atoms], dtype=np.float32)
    # One-hot encoding for atom types
    try:
        lig_one_hot = np.stack([
            np.eye(len(atom_dict))[atom_dict[atom.GetSymbol().capitalize()]]
            for atom in atoms
        ]).astype(np.float32)
    except KeyError as e:
        # If an atom symbol wasn't in atom_dict, raise KeyError to skip this complex
        raise KeyError(f'Ligand atom {e.args[0]} not in atom dict')

    atom_idxs = [atom.GetIdx() for atom in atoms]
    phys_flags = get_physchem_flags(lig_mol, atom_idxs)
    lig_one_hot = np.hstack([lig_one_hot, phys_flags])
    
    return {'lig_coords': lig_coords, 'lig_one_hot': lig_one_hot}

def process_pocket_from_file(pocket_path: Path, ca_only: bool = False):
    """
    Load the pocket PDB file and return pocket coordinates and one-hot features (either residue or atom-level).
    Returns:
        pocket_data: dict with 'pocket_coords', 'pocket_one_hot', and 'pocket_ids'.
    Raises:
        KeyError if an atom or residue type is not in the respective dictionary.
    """
    pdb_parser = PDBParser(QUIET=True)
    pocket_struct = pdb_parser.get_structure('', str(pocket_path))
    pocket_residues = [res for res in pocket_struct.get_residues() 
                       if is_aa(res, standard=True)]
    pocket_ids = [f"{res.parent.id}:{res.id[1]}" for res in pocket_residues]
    if ca_only:
        # Use only CA atom of each residue
        coords = []
        one_hot_list = []
        for res in pocket_residues:
            # Some residues (e.g., Gly) have CA as well; assume standard residues only
            if 'CA' not in res:
                continue  # skip residues without CA (non-standard residues, if any)
            coords.append(res['CA'].get_coord())
            aa_one = three_to_one(res.get_resname())
            if aa_one not in amino_acid_dict:
                raise KeyError(f"Residue {aa_one} not in amino acid dict")
            one_hot_list.append(np.eye(len(amino_acid_dict))[amino_acid_dict[aa_one]])
        pocket_coords = np.array(coords, dtype=np.float32)
        pocket_one_hot = np.stack(one_hot_list).astype(np.float32)
    else:
        # Use all heavy atoms of the pocket residues
        pocket_atoms = []
        for res in pocket_residues:
            for atom in res.get_atoms():
                elem = atom.element.capitalize()
                if elem in atom_dict or elem != 'H':
                    pocket_atoms.append(atom)
        pocket_coords = np.array([atom.get_coord() for atom in pocket_atoms], dtype=np.float32)
        try:
            pocket_one_hot = np.stack([
                np.eye(len(atom_dict))[atom_dict[atom.element.capitalize()]]
                for atom in pocket_atoms
            ]).astype(np.float32)
        except KeyError as e:
            raise KeyError(f"Pocket atom {e.args[0]} not in atom dict")
    return {
        'pocket_coords': pocket_coords,
        'pocket_one_hot': pocket_one_hot,
        'pocket_ids': pocket_ids
    }

def compute_smiles(positions, one_hot, mask):
    print("Computing SMILES ...")

    atom_types = np.argmax(one_hot, axis=-1)

    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []

    pbar = tqdm(enumerate(zip(positions, atom_types)),
                total=len(np.unique(mask)))
    for i, (pos, atom_type) in pbar:
        mol = build_molecule(pos, atom_type, dataset_info)

        # BasicMolecularMetrics() computes SMILES after sanitization
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            continue

        mol = rdmol_to_smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        pbar.set_description(f'{len(mols_smiles)}/{i + 1} successful')

    return mols_smiles


def get_lennard_jones_rm(atom_mapping):
    # Bond radii for the Lennard-Jones potential
    LJ_rm = np.zeros((len(atom_mapping), len(atom_mapping)))

    for a1 in atom_mapping.keys():
        for a2 in atom_mapping.keys():
            all_bond_lengths = []
            for btype in ['bonds1', 'bonds2', 'bonds3']:
                bond_dict = getattr(constants, btype)
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    all_bond_lengths.append(bond_dict[a1][a2])

            if len(all_bond_lengths) > 0:
                # take the shortest possible bond length because slightly larger
                # values aren't penalized as much
                bond_len = min(all_bond_lengths)
            else:
                # Replace missing values with sum of average covalent radii
                bond_len = covalent_radii[a1] + covalent_radii[a2]

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path, help="Base directory of the PDBBind dataset (contains index and subdirs)")
    parser.add_argument('--outdir', type=Path, default=None, help="Output directory for processed dataset")
    parser.add_argument('--num_val', type=int, default=300, help="Number of validation complexes")
    parser.add_argument('--num_test', type=int, default=300, help="Number of test complexes")
    parser.add_argument('--dist_cutoff', type=float, default=8.0, help="Distance cutoff for pocket definition (not used explicitly if pocket files provided)")
    parser.add_argument('--ca_only', action='store_true', default=False, help="Use CA atoms only for pocket representation")
    parser.add_argument('--random_seed', type=int, default=1, help="Random seed for data splitting")
    parser.add_argument('--gbs_training', action='store_true', default=True,help="Use a small subset (100 train, 15 val, 15 test) for quick training/benchmarking")
    args = parser.parse_args()

    # Determine output directory name if not provided
    if args.outdir is None:
        suffix = '' if 'H' in atom_dict else '_noH'
        suffix += '_ca_only' if args.ca_only else '_full'
        processed_dir = args.basedir / f'processed_pdbbind{suffix}'
    else:
        processed_dir = args.outdir
    processed_dir.mkdir(exist_ok=True, parents=True)

    # Parse the PDBBind index file
    index_file = args.basedir /'index'/ 'INDEX_demo_PL_data.2021'
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    all_data = []
    with open(index_file, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line.strip() or line.startswith('#'):
                continue  # skip empty lines or comments
            parts = line.split('//', 1)
            main = parts[0].strip()
            tail = parts[1] if len(parts) > 1 else ''

            tokens = main.split()
            if len(tokens) < 4:
                continue

            pdb_id = tokens[0].lower()

            # parse affinity like "Ki=0.068nM" into a float (in M)
            aff_tok = tokens[3]
            m = re.match(r".*?([-+]?[0-9]*\.?[0-9]+)\s*([npµm]*M)?", aff_tok, re.I)
            if not m:
                warnings.warn(f"Cannot parse affinity '{aff_tok}' — skipping")
                continue
            val = float(m.group(1))
            unit = (m.group(2) or '').lower()
            if unit == 'nm':
                affinity_val = val * 1e-9
            elif unit in ('um','µm'):
                affinity_val = val * 1e-6
            elif unit == 'mm':
                affinity_val = val * 1e-3
            else:
                affinity_val = val

            # extract ligand code from the parentheses in the comment
            lm = re.search(r"\((\w+)\)", tail)
            ligand_id = lm.group(1) if lm else 'UNK'

            all_data.append((pdb_id, ligand_id, affinity_val))

    total_complexes = len(all_data)
    print(f"Total complexes in index: {total_complexes}")

    # Shuffle and split data
    random.seed(args.random_seed)
    random.shuffle(all_data)
    if args.gbs_training:
        # Use only 130 examples (100 train, 15 val, 15 test) for GBS training
        subset = all_data[:130]
        # If fewer than 130, adjust splits proportionally (or at least ensure some test/val if possible)
        train_count, val_count, test_count = 100, 15, 15
        test_list = subset[:test_count]
        val_list = subset[test_count:test_count+val_count]
        train_list = subset[test_count+val_count:test_count+val_count+train_count]
    else:
        # Standard random split using args.num_val and args.num_test
        test_count = min(args.num_test, total_complexes)
        val_count = min(args.num_val, total_complexes - test_count)
        train_count = total_complexes - val_count - test_count
        test_list = all_data[:test_count]
        val_list = all_data[test_count:test_count+val_count]
        train_list = all_data[test_count+val_count:]
    data_split = {'train': train_list, 'val': val_list, 'test': test_list}
    n_train_before = len(train_list)
    n_val_before = len(val_list)
    n_test_before = len(test_list)
    print(f"Split sizes (before processing): train={n_train_before}, val={n_val_before}, test={n_test_before}")

    # Prepare to record affinity values for output
    affinity_dict = {}  # (split, complex_name) -> affinity

    # Process each split
    n_samples_after = {}
    for split, split_list in data_split.items():
        lig_coords_list = []
        lig_one_hot_list = []
        lig_mask_list = []
        pocket_coords_list = []
        pocket_one_hot_list = []
        pocket_mask_list = []
        complex_names = []
        receptor_files = []
        count = 0

        split_dir = processed_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)
        n_tot = len(split_list)
        # Group by PDB ID in case of multiple ligand entries for the same protein
        ligands_by_protein = {}
        for pdb_id, lig_id, aff in split_list:
            ligands_by_protein.setdefault(pdb_id, []).append((lig_id, aff))
        num_failed = 0
        tic = time()
        with tqdm(total=n_tot, desc=f"Processing {split} set") as pbar:
            for pdb_id, lig_info_list in ligands_by_protein.items():
                # Set up file paths
                pdb_dir = args.basedir / pdb_id
                protein_file = pdb_dir / f"{pdb_id}_protein.pdb"
                pocket_file = pdb_dir / f"{pdb_id}_pocket.pdb"
                if not protein_file.exists() or not pocket_file.exists():
                    warnings.warn(f"Missing protein or pocket file for {pdb_id}, skipping")
                    # Update progress for all ligands of this protein as failed
                    num_failed += len(lig_info_list)
                    pbar.update(len(lig_info_list))
                    continue
                # Parse the receptor structure (for potential output)
                pdb_struct = PDBParser(QUIET=True).get_structure('', str(protein_file))
                struct_copy = pdb_struct.copy()  # copy in case we needed to modify (not really needed if already no ligand)
                # Process each ligand for this protein
                pdb_successful = 0
                for lig_id, affinity_val in lig_info_list:
                    ligand_base = pdb_dir / f"{pdb_id}_ligand"  # base path for ligand (without extension)
                    try:
                        ligand_data = process_ligand_from_file(ligand_base)
                        pocket_data = process_pocket_from_file(pocket_file, ca_only=args.ca_only)
                    except Exception as e:
                        # If any error (KeyError, FileNotFoundError, ValueError), skip this ligand
                        num_failed += 1
                        continue
                    # Create a unique name for this complex: PDBID + ligand identifier
                    if lig_id is None or lig_id == 'UNK':
                        complex_name = pdb_id.upper()
                    else:
                        complex_name = f"{pdb_id.upper()}_{lig_id}"
                    complex_names.append(complex_name)
                    receptor_files.append(protein_file.name)
                    # Append ligand data
                    lig_coords_list.append(ligand_data['lig_coords'])
                    lig_one_hot_list.append(ligand_data['lig_one_hot'])
                    lig_mask_list.append(np.full(len(ligand_data['lig_coords']), fill_value=count, dtype=np.int32))
                    # Append pocket data
                    pocket_coords_list.append(pocket_data['pocket_coords'])
                    pocket_one_hot_list.append(pocket_data['pocket_one_hot'])
                    pocket_mask_list.append(np.full(len(pocket_data['pocket_coords']), fill_value=count, dtype=np.int32))
                    # Record affinity
                    affinity_dict[(split, complex_name)] = affinity_val
                    # Output ligand SDF and pocket text if val/test
                    if split in {'val', 'test'}:
                        # Directly copy the existing SDF (heavy-atom only) if present,
                        # otherwise fall back to converting MOL2→SDF via OpenBabel.
                        import shutil
                        sdf_in = ligand_base.with_suffix('.sdf')
                        sdf_out_name = f"{pdb_id.upper()}_{lig_id}.sdf" \
                                       if lig_id not in (None, 'UNK') \
                                       else f"{pdb_id.upper()}.sdf"
                        sdf_out_path = split_dir / sdf_out_name

                        if sdf_in.exists():
                            # Simply copy the original SDF
                            shutil.copy(str(sdf_in), str(sdf_out_path))
                        else:
                            # Fallback: convert MOL2 to SDF
                            mol2_in = ligand_base.with_suffix('.mol2')
                            if mol2_in.exists() and openbabel is not None:
                                obConv = openbabel.OBConversion()
                                obConv.SetInAndOutFormats("mol2", "sdf")
                                obMol = openbabel.OBMol()
                                if obConv.ReadFile(obMol, str(mol2_in)):
                                    obConv.WriteFile(obMol, str(sdf_out_path))
                    # Save pocket residue identifiers
                    if split in {'val', 'test'}:
                        pocket_residues_file = split_dir / f"{pdb_id.upper()}_{lig_id}.txt" if lig_id not in (None, 'UNK') else split_dir / f"{pdb_id.upper()}.txt"
                        with open(pocket_residues_file, 'w') as pf:
                            pf.write(' '.join(pocket_data['pocket_ids']))
                    count += 1
                    pdb_successful += 1
                # After processing all ligands for this PDB, save receptor once for val/test if any ligand succeeded
                if split in {'val', 'test'} and pdb_successful > 0:
                    receptor_out_name = f"{pdb_id.upper()}.pdb"
                    receptor_out_path = split_dir / receptor_out_name
                    io = PDBIO()
                    io.set_structure(struct_copy)
                    io.save(str(receptor_out_path), select=Model0())
                # Update progress
                pbar.update(len(lig_info_list))
                pbar.set_description(f"{split}: #failed {num_failed}")
        # Concatenate lists into arrays
        if lig_coords_list:
            lig_coords = np.concatenate(lig_coords_list, axis=0)
            lig_one_hot = np.concatenate(lig_one_hot_list, axis=0)
            lig_mask = np.concatenate(lig_mask_list, axis=0)
            pocket_coords = np.concatenate(pocket_coords_list, axis=0)
            pocket_one_hot = np.concatenate(pocket_one_hot_list, axis=0)
            pocket_mask = np.concatenate(pocket_mask_list, axis=0)
        else:
            # If no examples (edge case), create empty arrays
            lig_coords = np.zeros((0, 3), dtype=np.float32)
            lig_one_hot = np.zeros((0, len(atom_dict)), dtype=np.float32)
            lig_mask = np.zeros((0,), dtype=np.int32)
            pocket_coords = np.zeros((0, 3), dtype=np.float32)
            pocket_one_hot = np.zeros((0, pocket_one_hot_list[0].shape[1] if pocket_one_hot_list else len(atom_dict)), dtype=np.float32)
            pocket_mask = np.zeros((0,), dtype=np.int32)
        # Save NPZ for this split
        np.savez(processed_dir / f"{split}.npz",
                 names=np.array(complex_names), 
                 receptors=np.array(receptor_files),
                 lig_coords=lig_coords,
                 lig_one_hot=lig_one_hot,
                 lig_mask=lig_mask,
                 pocket_coords=pocket_coords,
                 pocket_one_hot=pocket_one_hot,
                 pocket_mask=pocket_mask)
        n_samples_after[split] = len(complex_names)
        elapsed = (time() - tic) / 60.0
        print(f"Processed {split} set: {n_samples_after[split]} complexes (took {elapsed:.2f} minutes, {num_failed} failures skipped)")
    # Save affinity values to file for reference
    affinity_file = processed_dir / "affinities.txt"
    with open(affinity_file, 'w') as af:
        af.write("split,name,affinity\n")
        for (split, name), aff in affinity_dict.items():
            af.write(f"{split},{name},{aff}\n")

    # Compute additional statistics on the training set for summary
    if n_samples_after.get('train', 0) > 0:
        with np.load(processed_dir / 'train.npz', allow_pickle=True) as data:
            lig_mask = data['lig_mask']
            pocket_mask = data['pocket_mask']
            lig_coords = data['lig_coords']
            lig_one_hot = data['lig_one_hot']
            pocket_one_hot = data['pocket_one_hot']
        # Joint histogram of ligand and pocket sizes
        idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
        idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
        joint_histogram = np.zeros((np.max(n_nodes_lig) + 1, np.max(n_nodes_pocket) + 1))
        for nl, npk in zip(n_nodes_lig, n_nodes_pocket):
            joint_histogram[nl, npk] += 1
        from scipy.ndimage import gaussian_filter

        train_smiles = compute_smiles(lig_coords, lig_one_hot, lig_mask)
        np.save(processed_dir / 'train_smiles.npy', train_smiles)
        n_nodes = gaussian_filter(joint_histogram, sigma=1.0)  # smooth the histogram
        np.save(processed_dir / 'size_distribution.npy', n_nodes)
        # Bond length arrays and Lennard-Jones parameters
        bonds1 = np.array(constants.bonds1)
        bonds2 = np.array(constants.bonds2)
        bonds3 = np.array(constants.bonds3)
        # Lennard-Jones r_m (using the same function from original if available)
        rm_LJ = get_lennard_jones_rm(atom_dict)
        # Histograms of atom and amino acid types
        atom_counts = lig_one_hot.sum(axis=0)
        atom_hist = {atom_decoder[i]: int(atom_counts[i]) for i in range(len(atom_decoder))}
        aa_counts = pocket_one_hot.sum(axis=0) if args.ca_only else None
        aa_hist = {}
        if args.ca_only and aa_counts is not None:
            # aa_counts length equals len(amino_acid_dict)
            inv_aa_dict = {v: k for k, v in amino_acid_dict.items()}
            for idx, count in enumerate(aa_counts):
                aa_hist[inv_aa_dict[idx]] = int(count)
        elif not args.ca_only:
            # Pocket one-hot is atom-level if not CA only
            aa_hist = None
        # Compile summary text
        summary_lines = []
        summary_lines.append("# SUMMARY")
        summary_lines.append("# Before processing")
        summary_lines.append(f"num_samples train: {n_train_before}")
        summary_lines.append(f"num_samples val: {n_val_before}")
        summary_lines.append(f"num_samples test: {n_test_before}")
        summary_lines.append("")
        summary_lines.append("# After processing")
        summary_lines.append(f"num_samples train: {n_samples_after.get('train', 0)}")
        summary_lines.append(f"num_samples val: {n_samples_after.get('val', 0)}")
        summary_lines.append(f"num_samples test: {n_samples_after.get('test', 0)}")
        summary_lines.append("")
        summary_lines.append("# Info")
        summary_lines.append(f"'atom_encoder': {atom_dict}")
        summary_lines.append(f"'atom_decoder': {list(atom_dict.keys())}")
        summary_lines.append(f"'aa_encoder': {amino_acid_dict}")
        summary_lines.append(f"'aa_decoder': {list(amino_acid_dict.keys())}")
        summary_lines.append(f"'bonds1': {bonds1.tolist()}")
        summary_lines.append(f"'bonds2': {bonds2.tolist()}")
        summary_lines.append(f"'bonds3': {bonds3.tolist()}")
        summary_lines.append(f"'lennard_jones_rm': {rm_LJ.tolist()}")
        summary_lines.append(f"'atom_hist': {atom_hist}")
        summary_lines.append(f"'aa_hist': {aa_hist}" if aa_hist is not None else "'aa_hist': None")
        # summary_lines.append(f"'n_nodes': {n_nodes.tolist()}")
        summary_text = "\n".join(summary_lines)
        with open(processed_dir / 'summary.txt', 'w') as sf:
            sf.write(summary_text)
        print("\n".join(summary_lines))
