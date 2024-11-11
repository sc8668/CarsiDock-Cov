import argparse
import os, re
import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem
import multiprocessing as mp
#from copy import deepcopy
import pytorch_lightning as pl
#import sys
#sys.path.append(".")

from src.utils.docking_inference_utils import model_inference, convert_dist2coord
from src.utils.utils import get_carsidock_model
from src.utils.conf_gen import gen_init_conf
from src.utils.docking_utils import set_coord, extract_pocket
from src.utils.cov_prepare import extract_carsidock_pocket, load_mol, load_mols, mol_edit, mol_prep
from src.utils.postprocess import obtain_pose_with_pocket, remove_dummyat, write_pdb, write_sdf
from RTMScore.utils import scoring, get_rtmscore_model



def UserInput():
	p = argparse.ArgumentParser()
	p.add_argument('-p', '--pdb_file', required=True, type=str, help='protein file (.pdb)')
	p.add_argument('-rl', '--ref_lig_file', required=True, type=str, help='the reference ligand to determine the pocket (.sdf)')
	p.add_argument('-l','--lig_file', type=str, help='ligand to be docked (.sdf|.smi)')
	p.add_argument('-smi','--smiles', default=None, type=str, help='ligand smiles (if the argument "lig_file" is not set, "smiles" will be used)')
	p.add_argument('-covres','--covres', required=True, type=str, help='Reactive Residue with the form of "ChainID:ResName:ResID", (e.g., C:CYS:539)')
	p.add_argument('-rectype','--rectype', default="Michael Addition", type=str, 
				choices = ["Michael Addition", "Nucleophilic Addition to a Double Bond",\
						   "Nucleophilic Addition to a Triple Bond", "Nucleophilic Substitution","Boronic Acid Addition",\
						   "Epoxide/Aziridine Opening","Gamma Lactone/Lactam Addition", "Imine Condensation","Beta Lactone/Lactam Addition",\
						   "Disulfide Formation","Others"], 
			    help='Reaction Type (default: "Michael Addition"). the "Others" class contains some ambiguous reaction types, which are observed in existing PDB structures.')    	
	p.add_argument('-rtms_rescoring','--rtms_rescoring', default=False, action='store_true', 
			    help='whether to use RTMScore for rescoring.')
	p.add_argument('-remove_dummyatom','--remove_dummyatom', default=False, action='store_true', 
			    help='whether to output the .sdf file with removed dummy atoms.')	
	p.add_argument('-remain_pocket','--remain_pocket', default=False, action='store_true', 
			    help='whether to output the .pdb file with the ligand bond with the pocket')	
	p.add_argument('-remain_protein','--remain_protein', default=False, action='store_true', 
			    help='whether to output the .pdb file with the ligand bond with the whole protein')
	p.add_argument('-multipdb','--multipdb', default=False, action='store_true', 
			    help='if set to True, one complex (ligand + protein/pocket) will be stored in its individual .pdb file; otherwise, those complexes are stored in a single file.')		    
	p.add_argument('-s','--seed', default=0, type=int)	
	p.add_argument('-o','--outprefix', default="./out", help='Prefix of the output file (default: "./out")')	
	p.add_argument('--ckpt_path', default='./checkpoints/carsidock_230731.ckpt')
	p.add_argument('--rtms_ckpt_path', default='./checkpoints/rtmscore_model1.pth')
	p.add_argument('--num_conformer', default=5, type=int,
					help='number of initial conformer, resulting in num_conformer * num_conformer docking conformations.')
	p.add_argument('--num_threads', default=1, type=int, help='recommend 1')
	p.add_argument('--device', default="cuda")
	p.add_argument('--cuda_device_index', default=0, type=int, help="gpu device index")
	args = p.parse_args()
	if args.lig_file is None and args.smiles is None:
		raise IOError('One of the argument "lig_file" and "smiles" should be provided!')
	if not re.search(r'.pdb$', args.pdb_file):
		raise IOError("only the protein file with .pdb is supported!")
	if not re.search(r'.sdf$', args.ref_lig_file):
		raise IOError("only the reference ligand file with .sdf is supported!")
	if args.lig_file is not None:
		if not re.search(r'.sdf$|.smi$', args.lig_file):
			raise IOError("only the ligand file with .sdf or .smi is supported!")		
	if len(args.covres.split(":")) != 3:
		raise IOError('Reactive Residue with the form of "ChainID:ResName:ResID", (e.g., C:CYS:539)')	
	return args

def get_heavy_atom_positions(ligand_file):
	ligand = load_mol(ligand_file)
	positions = ligand.GetConformer().GetPositions()
	atoms = np.array([a.GetSymbol() for a in ligand.GetAtoms()])
	positions = positions[atoms != 'H']
	return positions
	

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_index)
	model, ligand_dict, pocket_dict = get_carsidock_model(args.ckpt_path, args.device)
	
		
	if args.rtms_rescoring:
		rtms_model = get_rtmscore_model(args.rtms_ckpt_path)
		positions = get_heavy_atom_positions(args.ref_lig_file)
		rtms_pocket = extract_pocket(args.pdb_file, positions, distance=10, del_water=True)
		
	print('read data...')
	if args.remain_protein:
		prot, pocket, covatpos, attype = extract_carsidock_pocket(args.pdb_file, args.ref_lig_file, args.covres, cutoff=6, remain_prot=True)
	else:
		pocket, covatpos, attype = extract_carsidock_pocket(args.pdb_file, args.ref_lig_file, args.covres, cutoff=6)
	
	if args.lig_file:
		mols = load_mols(args.lig_file)
	else:
		mols = [Chem.MolFromSmiles(args.smiles)]
	mols = sum([mol_edit(mol, attype, args.covres, str(args.rectype)) for mol in mols], [])
	all_mol_list = [gen_init_conf(mol, num_confs=args.num_conformer) for mol in mols]
	allout_mol_list = []
	pp1_mol_list = [] 
	pp2_mol_list = [] 
	pp3_mol_list = [] 
	for i, mol_list in enumerate(all_mol_list):
		mol_list = [mol_prep(m, covatpos) for m in mol_list]
		print(f'docking...{i}')
		infer_output = model_inference(model, pocket, mol_list, covatpos, ligand_dict, pocket_dict, device=args.device, bsz=len(mol_list))
		omol_list = convert_dist2coord(infer_output, mol_list)		
		allout_mol_list.append(omol_list)
		if args.remove_dummyatom:
			omol_list2 = [remove_dummyat(m) for m in omol_list]
			pp1_mol_list.append(omol_list2)	
		if args.remain_pocket:
			omol_list2 = [obtain_pose_with_pocket(pocket, m, args.covres) for m in omol_list]
			pp2_mol_list.append(omol_list2)
		if args.remain_protein:
			omol_list2 = [obtain_pose_with_pocket(prot, m, args.covres) for m in omol_list]
			pp3_mol_list.append(omol_list2)
	
	outdir = os.path.dirname(args.outprefix)
	if outdir != "":
		if not os.path.exists(outdir):
			os.makedirs(outdir)
	write_sdf(sum(allout_mol_list,[]), "%s.sdf"%args.outprefix)
	
	if args.remove_dummyatom:
		write_sdf(sum(pp1_mol_list,[]), "%s_nodumat.sdf"%args.outprefix)
		
	if args.remain_pocket:
		write_pdb(sum(pp2_mol_list,[]), "%s_withpckt"%args.outprefix, multipdb=args.multipdb)
		
	if args.remain_protein:
		write_pdb(sum(pp3_mol_list,[]), "%s_withprot"%args.outprefix, multipdb=args.multipdb)
		
	if args.rtms_rescoring:
		if not args.remove_dummyatom:
			raise ValueError("if use RTMScore for rescoring, please set \"remove_dummyatom\" as True.")
		ids, scores = scoring(rtms_pocket, [x[0] for x in pp1_mol_list], rtms_model)
		df = pd.DataFrame(zip(ids, scores), columns=["ligid", "score"])
		df.to_csv("%s_rtms.csv"%args.outprefix, index=False, sep=",")

if __name__ == '__main__':
	args = UserInput()
	pl.seed_everything(args.seed)
	main(args)
