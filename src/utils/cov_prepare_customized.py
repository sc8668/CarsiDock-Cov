import prody as pr
#from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import PandasTools as rdpd
#from rdkit.Chem.MolStandardize import rdMolStandardize
from copy import deepcopy
import pandas as pd
import sys, re
import io



def mol_edit(mol, attype, pdbid):
	lig_new = []
	if pdbid in ["1cef","1ceg","1ghm","1vm1","1w8y",'2v35',"2vgj","3beb","3mze","3upp", "2h5s","2wke","3cg5","3d4f"]:
		query = Chem.MolFromSmarts("[O-0X1]=[C][C]-,=[C][N]")
		if pdbid == "3d4f":
			Chem.Kekulize(mol, clearAromaticFlags=True)
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[1], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())
	if pdbid in ["5lbg"]:
		query = Chem.MolFromSmarts("[O-0X1]=[C][C]-,=[C][N,O]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[1], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())
	if pdbid in ["1pw8"]:
		query = Chem.MolFromSmarts("[O;H1][C]1[C]-,=[C][O,N]1")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(Chem.AddHs(mol))
			aidx = [a.GetIdx() for a in mw.GetAtoms()[match[1]].GetNeighbors() if a.GetSymbol() == "H"][0]
			mw.ReplaceAtom(aidx, Chem.Atom(attype))
			mw.GetAtoms()[aidx].SetProp("covatom", "1")
			Chem.SanitizeMol(mw)
			lig_new.append(Chem.RemoveHs(mw))	
	if pdbid in ["1ewo","4i24"]:
		query = Chem.MolFromSmarts("[C,c;!R]-[C,c]-[C,c]=[O,o]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())
	if pdbid in ["5dgj"]:
		query = Chem.MolFromSmarts("[C,c]-[O,S,N;D1]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())
	if pdbid in ["4hav","4hax","4hay"]:
		query = Chem.MolFromSmarts("[C,c]-[C,c]-[C,c](=[O,o])O")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())

	if pdbid in ["1ewp","1khq","5rhe","1khp"]:
		query = Chem.MolFromSmarts("[C;H3][C;H0]=[O,S,N;D1]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())	
	if pdbid in ["1h8i"]:
		query = Chem.MolFromSmarts("[P;D3](=O)[O]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())	
	if pdbid in ["1u9w"]:
		query = Chem.MolFromSmarts("[C](a)[N]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())		
	if pdbid in ["3n3g"]:
		query = Chem.MolFromSmarts("[C]=[N]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())	
	if pdbid in ["3kjq"]:
		query = Chem.MolFromSmarts("[C;!H0][C;H0](=O)C")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(Chem.AddHs(mol))
			aidx = [a.GetIdx() for a in mw.GetAtoms()[match[0]].GetNeighbors() if a.GetSymbol() == "H"][0]
			mw.ReplaceAtom(aidx, Chem.Atom(attype))
			mw.GetAtoms()[aidx].SetProp("covatom", "1")
			Chem.SanitizeMol(mw)
			lig_new.append(Chem.RemoveHs(mw))
			
	if pdbid in ["2dcc"]:  ##intermediate state
		query = Chem.MolFromSmarts("[C;r3][O,N;r3][C;r3]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(Chem.AddHs(mol))
			aidx = [a.GetIdx() for a in mw.GetAtoms()[match[0]].GetNeighbors() if a.GetSymbol() == "H"][0]
			mw.ReplaceAtom(aidx, Chem.Atom(attype))
			mw.GetAtoms()[aidx].SetProp("covatom", "1")
			Chem.SanitizeMol(mw)
			lig_new.append(Chem.RemoveHs(mw))	
			mw = Chem.RWMol(Chem.AddHs(mol))
			aidx = [a.GetIdx() for a in mw.GetAtoms()[match[2]].GetNeighbors() if a.GetSymbol() == "H"][0]
			mw.ReplaceAtom(aidx, Chem.Atom(attype))
			mw.GetAtoms()[aidx].SetProp("covatom", "1")
			Chem.SanitizeMol(mw)
			lig_new.append(Chem.RemoveHs(mw))	
	if pdbid in ["4wsk"]:
		query = Chem.MolFromSmarts("C1C(O)C(O)C(O)C(C)C(N)1")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())		

	if pdbid in ["2jai"]:
		query = Chem.MolFromSmarts("OC(=N)N")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())		

	if pdbid in ["2tod"]:
		query = Chem.MolFromSmarts("CC(=N)C")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())		

	if pdbid in ["5dv6","5dv8"]:
		query = Chem.MolFromSmarts("c1c(C=O)cc(N(O)(O))cc1")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())	

	if pdbid in ["5w1y"]:
		query = Chem.MolFromSmarts("C=C(N)C(=O)")
		Chem.Kekulize(mol, clearAromaticFlags=True)
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())	

	if pdbid in ["1haz"]:
		query = Chem.MolFromSmarts("C(O)(O)C")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())	
	
	[set_mol_name(m, i) for i, m in enumerate(lig_new)]
	return lig_new	

def set_mol_name(m, i):
	if m.HasProp("_Name"):
		m.SetProp('_Name',"%s-%s"%(m.GetProp("_Name"), i))
	else:
		m.SetProp('_Name',str(i))

