import os

import numpy as np
import copy
import torch
#from openbabel import openbabel, pybel
from rdkit import Chem
#from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from src.utils.dist_to_coords_utils import get_mask_rotate, modify_conformer
from src.utils.docking_utils import set_coord

def set_coord(mol, coords, idx=0):
    _mol = copy.deepcopy(mol)
    if len(_mol.GetConformers()) == 0:
        conf = Chem.Conformer(len(_mol.GetAtoms()))
        for i in range(len(_mol.GetAtoms())):
            conf.SetAtomPosition(i, coords[i].tolist())
        _mol.AddConformer(conf)
    else:
        for i in range(coords.shape[0]):
            _mol.GetConformer(idx).SetAtomPosition(i, coords[i].tolist())
    return _mol


def get_torsions(m):
    m = Chem.RemoveHs(m)
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                        m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(
        conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale
    )




def single_conf_gen(tgt_mol, num_confs=1000, seed=42, mmff=False):
    mol = copy.deepcopy(tgt_mol)
    mol = Chem.AddHs(mol)
    allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=seed, clearConfs=True, numThreads=0)
    try:
        if mmff:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    except:
        pass
    # sz = len(allconformers)
    # for i in range(sz):
    #     try:
    #         AllChem.MMFFOptimizeMolecule(mol, confId=i)
    #     except:
    #         continue
    mol = Chem.RemoveHs(mol)
    return mol


def rdkit_gen(input_rdkit_mol, total_confs=10, num_classes=5):
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(input_rdkit_mol)
    rdkit_mol = single_conf_gen(input_rdkit_mol, num_confs=total_confs, mmff=True)
    rdkit_mol = Chem.RemoveAllHs(rdkit_mol)
    sz = len(rdkit_mol.GetConformers())
    if sz == 0:
        rdkit_mol = copy.deepcopy(input_rdkit_mol)
        rdkit_mol = Chem.RemoveAllHs(rdkit_mol)
    coord_list = [conf.GetPositions() for conf in rdkit_mol.GetConformers()]
    new_coords = clustering(coord_list, num_classes)
    new_mol_list = [set_coord(copy.deepcopy(rdkit_mol), coord, 0) for coord in new_coords]
    return new_mol_list


def clustering(coords, num_classes=5):
    tgt_coords = coords[0]
    tgt_coords = tgt_coords - np.mean(tgt_coords, axis=0)
    rdkit_coords_list = []
    for _coords in coords:
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    if len(rdkit_coords_list) > num_classes:
        rdkit_coords_flatten = np.array(rdkit_coords_list).reshape(len(coords), -1)
        cluster_size = num_classes
        ids = (
            KMeans(n_clusters=cluster_size, random_state=42, n_init=10)
            .fit_predict(rdkit_coords_flatten)
            .tolist()
        )
        # 部分小分子仅可聚出较少的类
        ids_set = set(ids)
        coords_list = [rdkit_coords_list[ids.index(i)] for i in range(cluster_size) if i in ids_set]
    else:
        coords_list = rdkit_coords_list[:num_classes]
    return coords_list


def gen_init_conf(mol, num_confs=5):
    if num_confs < 3:
        num_confs = 3
    times = 5
    mol_list = rdkit_gen(mol, total_confs=num_confs * times, num_classes=num_confs)
    return mol_list
