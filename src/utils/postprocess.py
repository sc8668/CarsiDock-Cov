from rdkit.Chem import AllChem as Chem
import numpy as np
from copy import deepcopy
#from src.utils.docking_utils import add_coord
from rdkit.Chem.rdMolTransforms import GetBondLength

def add_coord2(mol, xyz, at_dict):
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos[list(at_dict.keys())] = xyz[list(at_dict.values())]
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(
            i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
        )
    return mol

def remove_dummyat(m):
	mw = Chem.RWMol(m)
	aidx = [a.GetIdx() for a in mw.GetAtoms() if a.HasProp("covatom")][0]
	mw.RemoveAtom(aidx)	
	return mw.GetMol()
	
def obtain_pose_with_pocket(pocket, m, covres="C:CYS:539"):
	covat = obtain_covat_in_pocket(pocket, covres)
	mw = Chem.RWMol(pocket)
	at_dict = {}
	aid0 = None
	for a in deepcopy(m).GetAtoms():	
		if not a.HasProp("covatom"):
			aid1 = a.GetIdx()
			aid2 = mw.AddAtom(a)		
			at_dict.update({aid2:aid1})
	
	at_dict_r = {v:k for k,v in at_dict.items()}
	for b in m.GetBonds():	
		a1 = b.GetBeginAtom()
		a2 = b.GetEndAtom()
		if a1.HasProp("covatom"):
			aid0 = a2.GetIdx()
		elif a2.HasProp("covatom"):
			aid0 = a1.GetIdx()
		else:
			bid = mw.AddBond(at_dict_r[a1.GetIdx()], at_dict_r[a2.GetIdx()], order=b.GetBondType())
		
	bid0 = mw.AddBond(at_dict_r[aid0], covat.GetIdx(), Chem.BondType.SINGLE)
	mw = add_coord2(mw, m.GetConformer().GetPositions(), at_dict)
	mw.SetProp("_Name", m.GetProp("_Name"))
	return mw.GetMol()


def obtain_covat_in_pocket(pocket, covres="C:CYS:539"):
	ax=[]
	covreschain, covresname, covresnum = covres.split(":")
	for a in pocket.GetAtoms():
		res_info = a.GetPDBResidueInfo()
		if res_info.GetResidueNumber() == int(covresnum) and res_info.GetResidueName() == covresname and res_info.GetChainId() == covreschain:
			ax.append(a)
	if covresname in ["CYS", "SER", "TYR", "HIS", "LYS", "ASN"]:
		return ax[-1]
	elif covresname=="THR": 
		return ax[-2]
	elif covresname in ["ASP", "GLU", "ARG"]: 
		if ax[-1].GetTotalValence() == 1:
			return ax[-1]
		else:
			return ax[-2]
	else:
		raise ValueError("Reaction residue %s is not supported in current version!"%covresname)
	

def write_file(output_file, outline):
    buffer = open(output_file, 'w')
    buffer.write(outline)
    buffer.close()

def write_sdf(mols, outpath):
	with Chem.SDWriter(outpath) as w:
		for mx in mols:
			w.write(mx)


def write_pdb(mols, outprefex, multipdb=False):
	if not multipdb:
		buffer = open("%s.pdb"%outprefex, 'w')
		for i, m in enumerate(mols):
			mblock = "".join(["MODEL        %s\n"%(i+1)] + [Chem.MolToPDBBlock(m).replace("END\n","ENDMDL\n")])
			buffer.write(mblock)
		buffer.close()
	else:
		for i, m in enumerate(mols):
			write_file("%s_%s.pdb"%(outprefex, i+1), Chem.MolToPDBBlock(m))
	
	
