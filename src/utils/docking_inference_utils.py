import copy
import time
from multiprocessing import Pool
from typing import List

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.autograd import Variable
from torch.distributions import Normal
import math

from src.utils.conf_gen import gen_init_conf, get_torsions, single_conf_gen

from src.utils.dist_to_coords_utils import modify_conformer, get_mask_rotate
from src.utils.docking_utils import optimize_rotatable_bonds, prepare_log_data, add_coord, save_sdf, dock_with_gradient, \
    single_SF_loss, get_symmetry_rmsd, set_coord


def prepare_data_from_mol(mol_list, dictionary, prefix='mol', max_atoms=384, device='cuda'):
    atoms = [a.GetSymbol().upper() for a in mol_list[0].GetAtoms()]
    atoms = [a if '[' not in a else a[1] for a in atoms]
    indcov = np.array([False if a.HasProp("covatom") else True for a in mol_list[0].GetAtoms()])
    indices = (np.array(atoms) != 'H')
    # 限制口袋最大原子个数
    if (np.sum(indices) > max_atoms) and (prefix != 'mol'):
        _indices = np.random.choice(np.sum(indices), max_atoms, replace=False)
        drop_indices = np.zeros(np.sum(indices), dtype=bool)
        drop_indices[_indices] = True
        indices[indices] = drop_indices
    tokens = torch.from_numpy(dictionary.vec_index(atoms)[indices][indcov]).long()
    ori_coordinates = torch.from_numpy(np.array([mol.GetConformer().GetPositions()[indices] for mol in mol_list]))
    coordinates = ori_coordinates.clone()
    bsz, sz = coordinates.shape[:2]
    center = coordinates.mean(dim=1).unsqueeze(1)
    coordinates = torch.cat([center, coordinates[:,indcov,:], center], dim=1)
    distance = (coordinates.unsqueeze(2) - coordinates.unsqueeze(1)).norm(dim=-1)
    # sos & eos
    tokens = torch.cat([torch.full((1,), dictionary.bos()), tokens, torch.full((1,), dictionary.eos())], dim=0)
    edge_type = tokens.view(-1, 1) * len(dictionary) + tokens.view(1, -1)
    tokens = tokens.unsqueeze(0).repeat(bsz, 1)
    edge_type = edge_type.unsqueeze(0).repeat(bsz, 1, 1)
    return {
        f'{prefix}_src_tokens': tokens.to(device=device),
        f'{prefix}_src_distance': distance.to(device=device, dtype=torch.float32),
        f'{prefix}_src_edge_type': edge_type.to(device=device),
    }, ori_coordinates.to(device=device, dtype=torch.float32)


class MultiProcess:
    def __init__(self, mol, pocket_coords, pocket_center, pi, mu, sigma, iterations=20000, early_stoping=5, **unused):
        self.mol = Chem.RemoveHs(mol)
        self.indcov = np.array([False if a.HasProp("covatom") else True for a in self.mol.GetAtoms()])
        self.pocket_coords = pocket_coords
        self.pocket_center = pocket_center
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.iterations = iterations
        self.early_stoping = early_stoping
        torsions, masks = get_mask_rotate(mol)
        self.torsions = torsions
        self.masks = masks
        self.rotable_bonds = get_torsions(mol)

    def dist_to_coords_with_tor(self, init_coord, cov_pos, aidx, pred_cross_dist, pred_holo_dist):
        values = Variable(torch.zeros(3 + len(self.torsions), device=init_coord.device), requires_grad=True)
        optimizer = torch.optim.LBFGS([values], lr=0.1)
        best_loss, times, best_values, best_score, best_coords = 10000.0, 0, None, 0, None
        for i in range(self.iterations):
            def closure():
                optimizer.zero_grad()
                new_pos = modify_conformer(init_coord, values, cov_pos, aidx, self.torsions, self.masks)
                loss = single_SF_loss(new_pos[self.indcov], self.pocket_coords, pred_cross_dist, pred_holo_dist)
                loss.backward()
                return loss

            def fn(v):
                new_pos = modify_conformer(init_coord, torch.from_numpy(v).float(), cov_pos, aidx, self.torsions, self.masks)
                return single_SF_loss(new_pos[self.indcov], self.pocket_coords, pred_cross_dist, pred_holo_dist)

            def der(v):
                eps = 0.001
                y = fn(v)
                g = np.zeros_like(v, dtype=np.float32)
                for i in range(v.shape[0]):
                    save = v[i]
                    v[i] += eps
                    yd = fn(v)
                    g[i] = (yd.item() - y.item()) / eps
                    v[i] = save
                return y, g

            def closure_Finite_Difference():
                optimizer.zero_grad()
                # 计算梯度并将其存储在values.grad中
                loss, grad = der(values.detach().numpy())
                values.grad = torch.tensor(grad, dtype=torch.float32, device=values.device)
                return loss

            loss = optimizer.step(closure)
            # loss = optimizer.step(closure_Finite_Difference)
            if loss.item() < best_loss:
                best_loss = loss.item()
                times = 0
                best_values = copy.deepcopy(values).detach()
            else:
                times += 1
                if times > self.early_stoping:
                    break
        if best_loss < 100:
            with torch.no_grad():
                best_coords = modify_conformer(init_coord, best_values, cov_pos, aidx, self.torsions, self.masks)
                score = mdn_score(self.pi, self.mu, self.sigma, best_coords[self.indcov], self.pocket_coords).item()
                best_coords = best_coords.cpu().data.numpy()
            opt_mol = set_coord(self.mol, best_coords)
            #opt_mol = add_coord(opt_mol, self.pocket_center.cpu().data.numpy())
            opt_mol.SetProp('loss', f'{best_loss}')
            opt_mol.SetProp('score', f'{score}')
            opt_mol.SetProp('CarsiScore', "%s"%(5 * best_loss - score))
            return opt_mol, best_loss, score


def convert_dist2coord(infer_output, ligands: List):#, 
                      #convert_mode=2, lbfgsbsrv=None):
    mp = MultiProcess(mol=ligands[0], **infer_output)
    args = [(ic, infer_output['cov_pos'], ai, pcd, phd) for ic, ai in zip(infer_output['ligands_coords'], infer_output['aidx']) for pcd, phd in
            zip(infer_output['distance_predict_tta'], infer_output['holo_distance_predict_tta'])]        
    
    mol_list = [mp.dist_to_coords_with_tor(*arg) for arg in args]

    mol_list = sorted([m for m in mol_list if m is not None], key=lambda x: 5 * x[1] - 1 * x[2])
    mol_list = [x[0] for x in mol_list]
    return mol_list


def mdn_score(pi, mu, sigma, predict_coords=None, pocket_coords=None, dist=None, threshold=5, reduction='sum'):
    if dist is None:
        dist = torch.norm(predict_coords.unsqueeze(1) - pocket_coords.unsqueeze(0), dim=-1)
    dist_mask = dist < threshold
    normal = Normal(mu, sigma)
    # [BSZ, N, M, 10]
    loglik = normal.log_prob(dist.unsqueeze(-1))
    logprob = loglik + torch.log(pi)
    # [BSZ, N, M]
    prob = logprob.exp().sum(-1)
    if reduction == 'mean':
        score = (prob[dist_mask] / (dist[dist_mask] ** 2 + 1e-6)).mean() * 1000
    else:
        # score = torch.stack([p[m].sum() for p, m in zip(prob, dist_mask)])
        score = prob[dist_mask].sum() / dist.shape[0]
        # score = prob[dist_mask].sum()
    conf_mask = dist < 1.5
    if conf_mask.sum() > 0:
        score += torch.log(dist[conf_mask] / 1.5).sum() * 10
    return score


def mdn_score_list(pi, mu, sigma, dist=None, threshold=5, reduction='sum'):
    dist_mask = dist < threshold
    normal = Normal(mu, sigma)
    # [BSZ, N, M, 10]
    loglik = normal.log_prob(dist.unsqueeze(-1))
    logprob = loglik + torch.log(pi)
    # [BSZ, N, M]
    prob = logprob.exp().sum(-1)
    if reduction == 'mean':
        score = (prob[dist_mask] / (dist[dist_mask] ** 2 + 1e-6)).mean() * 1000
    else:
        # score = torch.stack([p[m].sum() / dist.shape[1] for p, m in zip(prob, dist_mask)])
        score = torch.stack([p[m].sum() for p, m in zip(prob, dist_mask)])
        # score = prob[dist_mask].sum() / dist.shape[0]
        # score = prob[dist_mask].sum()
    return score


def read_ligands(mol_list=None, smiles=None, num_gen_conf=100, num_use_conf=5):
    """
    read ligands, generate conformer if smiles is provided.
    """
    if mol_list is None:
        assert smiles is not None
        mol_list = [Chem.MolFromSmiles(smi) for smi in smiles]
        for mol in mol_list:
            mol.SetProp('_Name', Chem.MolToInchiKey(mol))
    mol_list = [Chem.RemoveAllHs(mol) for mol in mol_list if mol is not None]
    total_mol_list = [gen_init_conf(mol, num_confs=num_use_conf) for mol in mol_list]
    return total_mol_list

def get_aidx(m):
	for a in m.GetAtoms():
		if a.HasProp("covatom"):
			return a.GetIdx()
			

@torch.no_grad()
def model_inference(model, pocket, ligands: List, cov_pos, ligand_dict, pocket_dict, device='cuda', bsz=8):
    model.eval()
    ligand_nums = len(ligands)
    aidx = [get_aidx(m) for m in ligands]  
    indcov = np.array([False if a.HasProp("covatom") else True for a in ligands[0].GetAtoms()])
    # print('prepare data...')
    l_data, p_data = [], []
    for i in range(int(math.ceil(ligand_nums / bsz))):
        l_data.append(
            prepare_data_from_mol(ligands[i * bsz:(i + 1) * bsz], ligand_dict, device=device))
        length = len(ligands[i * bsz:(i + 1) * bsz])
        p_data.append(prepare_data_from_mol([pocket for _ in range(length)], pocket_dict, 'pocket', device=device))
    # print('inference distance matrix...')
    with torch.no_grad():
        outputs = [model(**pocket_data[0], **ligand_data[0]) for pocket_data, ligand_data in zip(p_data, l_data)]
        pocket_coords = p_data[0][1][0]
        ligands_coords = torch.cat([l[1] for l in l_data], dim=0)
        #ligands_indices = l_data[0][-1]
        
        pocket_center = pocket_coords.mean(dim=0)
        cov_pos = torch.from_numpy(cov_pos).float().to(ligands_coords.device)
        indcov = torch.from_numpy(indcov).to(ligands_coords.device) 
        distance_predict_tta = torch.cat([output.cross_distance_predict for output in outputs])[:, 1:-1, 1:-1]
        holo_distance_predict_tta = torch.cat([output.holo_distance_predict for output in outputs])[:, 1:-1, 1:-1]
        pi = torch.cat([output.mdn[0] for output in outputs], dim=0).mean(dim=0)[1:-1, 1:-1]
        mu = torch.cat([output.mdn[1] for output in outputs], dim=0).mean(dim=0)[1:-1, 1:-1]
        sigma = torch.cat([output.mdn[2] for output in outputs], dim=0).mean(dim=0)[1:-1, 1:-1]
        mean_cross_dist = torch.mean(distance_predict_tta, dim=0)
        score = mdn_score(pi, mu, sigma, dist=mean_cross_dist).item()
		
        distance_predict_tta = distance_predict_tta.cpu()
        holo_distance_predict_tta = holo_distance_predict_tta.cpu()
        pi, mu, sigma = pi.cpu(), mu.cpu(), sigma.cpu()
        ligands_coords = ligands_coords.cpu()
        pocket_coords = pocket_coords.cpu()
        pocket_center = pocket_center.cpu()
        cov_pos = cov_pos.cpu()
        
        inference_output = {
            'score': score,
            'distance_predict_tta': distance_predict_tta,
            'holo_distance_predict_tta': holo_distance_predict_tta,
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'ligands_coords': ligands_coords,  ##5
            "cov_pos": cov_pos,#torch.from_numpy(cov_pos).float(),
            "aidx": aidx,#torch.tensor(aidx).int(),
            'pocket_coords': pocket_coords,
            'pocket_center': pocket_center,
        }
        return inference_output

