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
	if pdbid in ['5rej', '5rek', '5rel', '5rem', '5ren', '5reo', '5rep', '5rer', '5res', '5ret',\
				 '5reu', '5rev', '5rew', '5rex', '5rey', '5rff', '5rfh', '5rfi', '5rfj', '5rfk',\
				 '5rfl', '5rfm', '5rfn', '5rfo', '5rfp', '5rfq', '5rfr', '5rfs', '5rft', '5rfu',\
				 '5rfv', '5rfw', '5rfx', '5rfy', '5rfz', '5rg0', '5rgl', '5rgm', '5rgn', '5rgo',\
				 '5rgp', '5rha', '5rhe', '5rhf']:
		query = Chem.MolFromSmarts("[C;H3][C;H0]=[O,S,N;D1]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())	
	if pdbid in ['5rgt','5rh5','5rh6','5rh7','5rh9']:
		query = Chem.MolFromSmarts("[O-0X1]=[C](N)[C]-,=[C]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[-1], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())
		
	if pdbid in ['6lu7','7bqy','7c8r']:
		query = Chem.MolFromSmarts("[O-0X1]=[C](O)[C]-,=[C]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[-1], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())
		
	if pdbid in ['6ynq']:
		query = Chem.MolFromSmarts("[O-0X1]=[C;R][C;R]-,=[C;R]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(mol)
			aidx = mw.AddAtom(Chem.Atom(attype))
			bidx2 = mw.AddBond(aidx, match[-1], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(mw.GetMol())
	
	if pdbid in ['6lze','6m0k','6wnp','6wtj','6wtk','6wtt','6xa4','6xbg','6xbh','6xbi',\
	             '6xfn','6xmk','6xqs','6xqt','6xqu','6xr3','6zrt','6zru','7c7p','7c8t','7com']:
		query = Chem.MolFromSmarts("C-[O;D1]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(Chem.AddHs(mol))			
			aidx = [x.GetIdx() for x in mw.GetAtoms()[match[0]].GetNeighbors() if x.GetSymbol()=="H"][0]
			#aidx = mw.AddAtom(Chem.Atom(attype))
			mw.ReplaceAtom(aidx, Chem.Atom(attype))
			#bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(Chem.RemoveHs(mw))
		
	if pdbid in ['6xhm']:
		query = Chem.MolFromSmarts("C(CN)-[O;D1]")
		matches = mol.GetSubstructMatches(query)
		for match in matches:
			mw = Chem.RWMol(Chem.AddHs(mol))			
			aidx = [x.GetIdx() for x in mw.GetAtoms()[match[0]].GetNeighbors() if x.GetSymbol()=="H"][0]
			mw.ReplaceAtom(aidx, Chem.Atom(attype))
			mw.GetAtoms()[aidx].SetProp("covatom", "1")		
			Chem.SanitizeMol(mw)
			lig_new.append(Chem.RemoveHs(mw))
		
	if pdbid in ["5rhb"]:
		query = Chem.MolFromSmarts("[C]=[N]")
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

