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
from src.utils.docking_utils import set_coord


def load_mol(ligpath):
	if re.search(r'.sdf$', ligpath):
		df = rdpd.LoadSDF(ligpath, molColName='MOL', includeFingerprints=False)	
		mol = df['MOL'][0]	
	elif re.search(r'.smi$', ligpath):
		df = pd.read_csv(ligpath, header=None, sep='\s+')
		df.columns = ['smiles', 'ID']
		df['MOL'] = df.smiles.apply(Chem.MolFromSmiles)
		mol = df['MOL'][0]
		mol.SetProp("_Name", df["ID"][0])
	else:
		raise IOError("only the molecule files with .sdf or .smi are supported!")	
	return mol 

def load_mols(ligpath):
	if re.search(r'.sdf$', ligpath):	
		df = rdpd.LoadSDF(ligpath, molColName='MOL', includeFingerprints=False)	
		mols = df['MOL'].tolist()
	elif re.search(r'.smi$', ligpath):
		df = pd.read_csv(ligpath, header=None, sep='\s+')
		df.columns = ['smiles', 'ID']
		df['MOL'] = df.smiles.apply(Chem.MolFromSmiles)	
		mols = df['MOL'].tolist()
		[mol.SetProp("_Name", i) for i, mol in zip(*(df["ID"], mols))]
	else:
		raise IOError("only the molecule files with .sdf or .smi are supported!")	
	return mols 


def extract_carsidock_pocket(pdb_file, ref_lig_file, covres, cutoff=6, remain_prot=False):#covres="C:CYS:539"
	lig = load_mol(ref_lig_file)
	ligpos = lig.GetConformer().GetPositions()
	xprot = pr.parsePDB(pdb_file).select("protein and not element H")
	selected = xprot.select(f'same residue as exwithin {cutoff} of ligand', ligand=ligpos)
	covreschain, covresname, covresnum = covres.split(":")
	cov_res = selected.select("(chain %s) and (resnum %s)"%(covreschain, covresnum))
	if cov_res is None:
		raise ValueError("Reaction residue %s is not included in the pocket!"%covres)
	else:
		if covresname=="CYS":
			cov_at = cov_res.select("name SG")
			attype=16
		elif covresname=="THR": 
			cov_at = cov_res.select("name OG1")
			attype=8
		elif covresname=="SER": 
			cov_at = cov_res.select("name OG") 	
			attype=8
		elif covresname=="TYR": 
			cov_at = cov_res.select("name OH") 
			attype=8
		elif covresname=="HIS": 
			cov_at = cov_res.select("name NE2") 
			attype=7
		elif covresname=="LYS": 
			cov_at = cov_res.select("name NZ") 
			attype=7
		elif covresname=="ASN": 
			cov_at = cov_res.select("name ND2") 
			attype=7
		elif covresname=="ASP": 
			fx = io.StringIO()
			pr.writePDBStream(fx, cov_res)
			cov_res_rd = Chem.MolFromPDBBlock(fx.getvalue(), sanitize=False, removeHs=True) 
			x1 = cov_res.getNames()
			x2 = [a.GetTotalValence() for a in cov_res_rd.GetAtoms()]
			xdict = dict(zip(*(x1,x2)))
			if xdict["OD1"] == 1:
				cov_at = cov_res.select("name OD1") 
			else:
				cov_at = cov_res.select("name OD2") 
			attype=8
		elif covresname=="GLU": 
			fx = io.StringIO()
			pr.writePDBStream(fx, cov_res)
			cov_res_rd = Chem.MolFromPDBBlock(fx.getvalue(), sanitize=False, removeHs=True) 
			x1 = cov_res.getNames()
			x2 = [a.GetTotalValence() for a in cov_res_rd.GetAtoms()]
			xdict = dict(zip(*(x1,x2)))
			if xdict["OE1"] == 1:
				cov_at = cov_res.select("name OE1") 
			else:
				cov_at = cov_res.select("name OE2")
			attype=8
		elif covresname=="ARG": 
			fx = io.StringIO()
			pr.writePDBStream(fx, cov_res)
			cov_res_rd = Chem.MolFromPDBBlock(fx.getvalue(), sanitize=False, removeHs=True) 
			x1 = cov_res.getNames()
			x2 = [a.GetTotalValence() for a in cov_res_rd.GetAtoms()]
			xdict = dict(zip(*(x1,x2)))
			if xdict["NH1"] == 1:
				cov_at = cov_res.select("name NH1") 
			else:
				cov_at = cov_res.select("name NH2")
			attype=8
		else:
			raise ValueError("Reaction residue %s is not supported in current version!"%covresname)
	
	covatpos = cov_at.getCoords()
	f = io.StringIO()
	pr.writePDBStream(f, selected)
	pocket = Chem.MolFromPDBBlock(f.getvalue(), sanitize=False, removeHs=True)
	if remain_prot:
		f = io.StringIO()
		pr.writePDBStream(f, xprot)
		prot = Chem.MolFromPDBBlock(f.getvalue(), sanitize=False, removeHs=True)		
		return prot, pocket, covatpos, attype
	else:
		return pocket, covatpos, attype



def mol_edit(mol, attype, covres, rectype="Michael Addition"):#, leaving_group=None):
	lig_new = []
	cov_type = covres.split(":")[1]
	if rectype=="Michael Addition":  #cys_ser_smarts
		if cov_type in ["SER", "CYS"]:
			query = Chem.MolFromSmarts("[C,c;!R]=[C,c]-[C,c,S,s]=[O,o]")
			query2 = Chem.MolFromSmarts("[C,c;!R]=[C,c]-[N+](=O)[O-]")
			queryq = Chem.MolFromSmarts("[C,c;R;!H0]=[C,c]-[C,c,S,s]=[O,o]")
			query2q = Chem.MolFromSmarts("[C,c;R;!H0]=[C,c]-[N+](=O)[O-]")
			queryp = Chem.MolFromSmarts("[C,c;!R]=[C,c]-[C,c,S,s](=[N,n])-[O,o;-1]")
			
			query3 = Chem.MolFromSmarts("c1cc([O;H1])c([O;H1])c[c;H0]1")
			query0 = Chem.MolFromSmarts("[C,c]=[C,c]-[C,c]=[C,c]-[C,c,S,s]=[O,o]")
			querx = Chem.MolFromSmarts("[C,c]#[C,c]-[C,c,S,s]=[O,o]")
			querx2 = Chem.MolFromSmarts("[C,c]#[C,c]-[c][s,o]")
			
			querz = Chem.MolFromSmarts("c1cc(=O)nc(=O)n1")
			matches = mol.GetSubstructMatches(query) + mol.GetSubstructMatches(query2) + mol.GetSubstructMatches(queryq) + mol.GetSubstructMatches(query2q) + mol.GetSubstructMatches(queryp)
			matches2 = mol.GetSubstructMatches(query3)
			matches0 = mol.GetSubstructMatches(query0)
			matchesx = mol.GetSubstructMatches(querx) + mol.GetSubstructMatches(querx2)
			matchesz = mol.GetSubstructMatches(querz)
								
			if len(matches0) > 0:
				for match in matches0:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[1], match[0])
					bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.SINGLE)
					mw.RemoveBond(match[1], match[2])
					bidx2 = mw.AddBond(match[1], match[2], Chem.BondType.DOUBLE)
					mw.RemoveBond(match[3], match[2])
					bidx3 = mw.AddBond(match[3], match[2], Chem.BondType.SINGLE)
					aidx = mw.AddAtom(Chem.Atom(attype))
					mw.GetAtoms()[aidx].SetProp("covatom", "1")						
					bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
			if len(matches) > 0:				
				for match in matches:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[1], match[0])
					bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.SINGLE)
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
			if len(matches2) > 0:	 #"c1cc([O;H1])c([O;H1])cc1"
				for match in matches2:
					Chem.Kekulize(mol, clearAromaticFlags=True)
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[1], match[0])
					bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.SINGLE)
					mw.RemoveBond(match[2], match[1])
					bidx2 = mw.AddBond(match[2], match[1], Chem.BondType.SINGLE)
					mw.RemoveBond(match[2], match[4])
					bidx3 = mw.AddBond(match[2], match[4], Chem.BondType.SINGLE)
					mw.RemoveBond(match[6], match[4])
					bidx4 = mw.AddBond(match[6], match[4], Chem.BondType.DOUBLE)
					mw.RemoveBond(match[6], match[7])
					bidx5 = mw.AddBond(match[6], match[7], Chem.BondType.SINGLE)				
					mw.RemoveBond(match[0], match[7])
					bidx6 = mw.AddBond(match[0], match[7], Chem.BondType.SINGLE)
					mw.RemoveBond(match[2], match[3])
					bidx7 = mw.AddBond(match[2], match[3], Chem.BondType.DOUBLE)
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx8 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
			
			if len(matchesx) > 0:  #[C,c]#[C,c]-[C,c,S,s]=[O]
				for match in matchesx:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[1], match[0])
					bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.DOUBLE)
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())			

			if len(matchesz) > 0:				
				for match in matchesz:
					mw = Chem.RWMol(mol)
					Chem.Kekulize(mw, clearAromaticFlags=True)
					mw.RemoveBond(match[1], match[0])
					bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.SINGLE)
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
				
		else:
			raise ValueError('Currently only "CYS" and "SER" support "Michael Addition".')
				
	elif rectype=="Nucleophilic Addition to a Double Bond": #cys_ser_smarts
		if cov_type in ["SER", "CYS"]:		
			query1 = Chem.MolFromSmarts("[C,c][C,c](=[O,S])NN")
			query2 = Chem.MolFromSmarts("[C,c][C,c](=[O,S])[C,c]")
			query2x = Chem.MolFromSmarts("N[C,c](=[O,S])N")
			query3 = Chem.MolFromSmarts("[C,c;H1]=[O,S,N;D1]")
			query4 = Chem.MolFromSmarts("[C,c](=[N;D1])N")
			matches = mol.GetSubstructMatches(query1) + mol.GetSubstructMatches(query2) + mol.GetSubstructMatches(query2x)
			matches2 = mol.GetSubstructMatches(query3) + mol.GetSubstructMatches(query4)
			if len(matches) > 0:
				for match in matches:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[2], match[1])
					bidx1 = mw.AddBond(match[2], match[1], Chem.BondType.SINGLE)
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[1], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
			if len(matches2) > 0:
				for match in matches2:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[1], match[0])
					bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.SINGLE)
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())	
		else:
			raise ValueError('Currently only "CYS" and "SER" support "Nucleophilic Addition to a Double Bond".')					
						
	elif rectype=="Nucleophilic Addition to a Triple Bond": #cys_ser_smarts
		if cov_type in ["SER", "CYS"]:	
			query = Chem.MolFromSmarts("[C]#[N]")
			matches = mol.GetSubstructMatches(query)
			for match in matches:
				mw = Chem.RWMol(mol)
				mw.RemoveBond(match[1], match[0])
				bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.DOUBLE)
				aidx = mw.AddAtom(Chem.Atom(attype))
				bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				Chem.SanitizeMol(mw)
				lig_new.append(mw.GetMol())
		else:
			raise ValueError('Currently only "CYS" and "SER" support "Nucleophilic Addition to a Triple Bond".')				
				
	elif rectype=="Nucleophilic Substitution": #cys_ser_smarts, lys_smarts
		if cov_type in ["SER", "CYS", "LYS"]:
			query1 = Chem.MolFromSmarts("[F,Cl,Br,I][C;H2][C;H0]=[O,S,N;D1]")
			query2 = Chem.MolFromSmarts("[F,Cl,Br,I][a]")
			query3 = Chem.MolFromSmarts("[F,Cl,Br,I][P;D4](=O)[O]")
			
			query1x = Chem.MolFromSmarts("[O,N;D2]([a])[C;H2][C;H0]=[O,S,N;D1]")
			query2x = Chem.MolFromSmarts("[O,N;D2]([a])[C;H0]=[O,S,N;D1]")
			query3x = Chem.MolFromSmarts("[O,N;D2]([a])[P;D4](=O)[O]")
			query4x = Chem.MolFromSmarts("[O,N;D2]([C;H0](=[O,S,N;D1])[a])[C;H2][C;H0]=[O,S,N;D1]")
			query5x = Chem.MolFromSmarts("[O,N;D2](C(C(F)(F)(F))(C(F)(F)(F)))[C;H0]=[O,S,N;D1]")
			query6x = Chem.MolFromSmarts("n1(nccn1)[C;H0]=[O,S,N;D1]")
			
			
			query4 = Chem.MolFromSmarts("[N-]=[N+]=[C;H1][C;H0]=[O,S,N;D1]")
			query5 = Chem.MolFromSmarts("S(=O)(=O)(O)[C,c][O,o]")
			query6 = Chem.MolFromSmarts("S(=O)(=O)([a])[C,c][O,o,N,n]")
			query6q = Chem.MolFromSmarts("S(=O)(=O)([C;H2][a])[C,c][O,o,N,n]")
			query7 = Chem.MolFromSmarts("[S;H0][C;H2][C;H2][O;H1]")
								
			matches = mol.GetSubstructMatches(query1) + mol.GetSubstructMatches(query2) + mol.GetSubstructMatches(query3)
			if len(matches) > 0:
				for match in matches:
					mw = Chem.RWMol(mol)
					mw.ReplaceAtom(match[0], Chem.Atom(attype))
					aidx = match[0]
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
			
			matches2 = mol.GetSubstructMatches(query1x) + mol.GetSubstructMatches(query2x) + mol.GetSubstructMatches(query3x)
			if len(matches2) > 0:
				for match in matches2:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[2], match[0])
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[2], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					mxs = Chem.GetMolFrags(mw, asMols=True)
					for mx in mxs:
						if len([a for a in mx.GetAtoms() if a.HasProp("covatom")]) != 0:
							Chem.SanitizeMol(mx)
							lig_new.append(mx)
				
			matches2x = mol.GetSubstructMatches(query4x)
			if len(matches2x) > 0:
				for match in matches2x:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[4], match[0])
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[4], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					mxs = Chem.GetMolFrags(mw, asMols=True)
					for mx in mxs:
						if len([a for a in mx.GetAtoms() if a.HasProp("covatom")]) != 0:
							Chem.SanitizeMol(mx)
							lig_new.append(mx)
			
			matches3x = mol.GetSubstructMatches(query5x)
			if len(matches3x) > 0: #"[O,N;D2](C(C(F)(F)(F))(C(F)(F)(F)))[C;H0]=[O,S,N;D1]"
				for match in matches3x:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[1], match[0])
					mw.ReplaceAtom(match[0], Chem.Atom(attype))
					mw.GetAtoms()[match[0]].SetProp("covatom", "1")
					mxs = Chem.GetMolFrags(mw, asMols=True)
					for mx in mxs:
						if len([a for a in mx.GetAtoms() if a.HasProp("covatom")]) != 0:
							Chem.SanitizeMol(mx)
							lig_new.append(mx)

					
			matches3 = mol.GetSubstructMatches(query4)
			if len(matches3) > 0:
				for match in matches3:
					mw = Chem.RWMol(mol)
					mw.RemoveAtom(match[0])
					mw.RemoveAtom(match[1])
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[2], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())

			matches4 = mol.GetSubstructMatches(query5) + mol.GetSubstructMatches(query6) + mol.GetSubstructMatches(query6q)
			if len(matches4) > 0:  #"S(=O)(=O)(O)CO"
				for match in matches4:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[-2], match[0])
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[-2], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					mxs = Chem.GetMolFrags(mw, asMols=True)
					for mx in mxs:	
						if len([a for a in mx.GetAtoms() if a.HasProp("covatom")]) !=0:
							Chem.SanitizeMol(mx)
							lig_new.append(mx)

			matches5 = mol.GetSubstructMatches(query7)
			if len(matches5) > 0:  #"SCCO"
				for match in matches5:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[0], match[1])
					mw.ReplaceAtom(match[0], Chem.Atom(attype))
					mw.GetAtoms()[match[0]].SetProp("covatom", "1")
					mxs = Chem.GetMolFrags(mw, asMols=True)
					for mx in mxs:
						if len([a for a in mx.GetAtoms() if a.HasProp("covatom")]) !=0:
							Chem.SanitizeMol(mx)
							lig_new.append(mx)
			
			matches6 = mol.GetSubstructMatches(query6x)
			if len(matches6) > 0:  #"n1(nccn1)[C;H0]=[O,S,N;D1]"
				for match in matches6:
					mw = Chem.RWMol(mol)
					Chem.Kekulize(mw, clearAromaticFlags=True)
					mw.RemoveBond(match[0], match[-2])
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[-2], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					mxs = Chem.GetMolFrags(mw, asMols=True)
					for mx in mxs:
						#if not mx.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)(O)")):	
						if len([a for a in mx.GetAtoms() if a.HasProp("covatom")]) !=0:
							Chem.SanitizeMol(mx)
							lig_new.append(mx)
			

		elif cov_type in ["ASP"]:
			query = Chem.MolFromSmarts("[F,Cl,Br,I]")
			matches = mol.GetSubstructMatches(query)
			if len(matches) > 0:
				for match in matches:
					mw = Chem.RWMol(mol)
					mw.ReplaceAtom(match[0], Chem.Atom(attype))
					aidx = match[0]
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())	
			
		else:
			raise ValueError('Currently only "CYS","SER","ASP" and "LYS" support "Nucleophilic Substitution".')				 
	
	elif rectype=="Boronic Acid Addition": #cys_ser_smarts
		if cov_type in ["SER", "CYS"]:
			query = Chem.MolFromSmarts("[B]([O])[O]")
			matches = mol.GetSubstructMatches(query)
			for match in matches:
				mw = Chem.RWMol(mol)
				mw.GetAtoms()[match[0]].SetFormalCharge(-1)
				aidx = mw.AddAtom(Chem.Atom(attype))
				bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				Chem.SanitizeMol(mw)
				lig_new.append(mw.GetMol())
		else:
			raise ValueError('Currently only "CYS" and "SER" support "Boronic Acid Addition".')				
						
	elif rectype=="Epoxide/Aziridine Opening": #cys_ser_smarts, his_smarts 
		#it's difficult to dertermine the chirality
		if cov_type in ["SER", "CYS", "HIS", "ASP", "GLU"]:
			query = Chem.MolFromSmarts("[C;r3][O,N;r3][C;r3]")
			query2 = Chem.MolFromSmarts("N1CC2CC23C1=CC(=O)c1ccccc13")			
			matches = mol.GetSubstructMatches(query)
			if len(matches) > 0:
				for match in matches:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[1], match[0])
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())	
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[1], match[2])
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[2], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
			else:
				matches = mol.GetSubstructMatches(query2)		
				for match in matches:
					mw = Chem.RWMol(mol)
					mw.RemoveBond(match[3], match[4])
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[3], Chem.BondType.SINGLE)
					mw.RemoveBond(match[4], match[5])
					bidx3 = mw.AddBond(match[4], match[5], Chem.BondType.DOUBLE)
					mw.RemoveBond(match[6], match[5])
					bidx4 = mw.AddBond(match[6], match[5], Chem.BondType.SINGLE)					
					mw.RemoveBond(match[6], match[7])
					bidx5 = mw.AddBond(match[6], match[7], Chem.BondType.DOUBLE)
					mw.RemoveBond(match[8], match[7])
					bidx6 = mw.AddBond(match[8], match[7], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())	
		else:
			raise ValueError('Currently only "CYS", "SER" "ASP", "GLU" and "HIS" support "Epoxide/Aziridine Opening".')					
					
	elif rectype=="Imine Condensation": #lys_smarts
		if cov_type in ["LYS"]:		
			#query = Chem.MolFromSmarts("[C](=[O])-[#6]")			
			query1 = Chem.MolFromSmarts("[#6]-[C](=[O])-[#6]")
			query2 = Chem.MolFromSmarts("[C;H1](=[O])-[#6]")
			
			matches = mol.GetSubstructMatches(query2)
			if len(matches) > 0:
				for match in matches:
					mw = Chem.RWMol(mol)
					mw.ReplaceAtom(match[1], Chem.Atom(attype))				
					mw.RemoveBond(match[1], match[0])
					bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.SINGLE)
					mw.GetAtoms()[match[1]].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
					
			matches = mol.GetSubstructMatches(query1)
			if len(matches) > 0:
				for match in matches:
					mw = Chem.RWMol(mol)
					mw.ReplaceAtom(match[2], Chem.Atom(attype))				
					mw.RemoveBond(match[1], match[2])
					bidx1 = mw.AddBond(match[1], match[2], Chem.BondType.SINGLE)
					mw.GetAtoms()[match[2]].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())				
		else:
			raise ValueError('Currently only "LYS" support "Imine Condensation".')					
					
	elif rectype=="Beta Lactone/Lactam Addition": #cys_ser_smarts
		if cov_type in ["SER", "CYS"]:
			query1 = Chem.MolFromSmarts("[O-0X1]=[C]1[C]-,=[C][O,N]1")
			query2 = Chem.MolFromSmarts("[O-0X1]=[S]1(=O)[C]-,=[C][O,N]1")
			matches = mol.GetSubstructMatches(query1) + mol.GetSubstructMatches(query2) 
			for match in matches:
				mw = Chem.RWMol(mol)
				mw.RemoveBond(match[1], match[-1])
				aidx = mw.AddAtom(Chem.Atom(attype))
				bidx2 = mw.AddBond(aidx, match[1], Chem.BondType.SINGLE)
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				Chem.SanitizeMol(mw)
				lig_new.append(mw.GetMol())
				
		else:
			raise ValueError('Currently only "CYS" and "SER" support "Beta Lactone/Lactam Additio".')								
				
	elif rectype=="Gamma Lactone/Lactam Addition": #cys_ser_smarts
		if cov_type in ["SER", "CYS"]:				
			query1 = Chem.MolFromSmarts("[O-0X1]=[C]1[C]-,=[C]-,=[C][O,N]1")
			query2 = Chem.MolFromSmarts("[O-0X1]=[S]1(=O)[C]-,=[C]-,=[C][O,N]1")
			matches = mol.GetSubstructMatches(query1) + mol.GetSubstructMatches(query2)
			for match in matches:
				mw = Chem.RWMol(mol)
				mw.RemoveBond(match[1], match[-1])
				aidx = mw.AddAtom(Chem.Atom(attype))
				bidx2 = mw.AddBond(aidx, match[1], Chem.BondType.SINGLE)
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				Chem.SanitizeMol(mw)
				lig_new.append(mw.GetMol())
		else:
			raise ValueError('Currently only "CYS" and "SER" support "Gamma Lactone/Lactam Addition".')
								
	elif rectype=="Disulfide Formation": #"[C]-[S;H1,-1]"
		if cov_type in ["CYS"]:
			query1 = Chem.MolFromSmarts("[S;X2;H1]")
			query2 = Chem.MolFromSmarts("[S]1[N][C](=O)[C]-,=[C]1")
			query3 = Chem.MolFromSmarts("[S][S]")
			matches = mol.GetSubstructMatches(query1)
			if len(matches) > 0:
				for match in matches:
					mw = Chem.RWMol(mol)
					aidx = mw.AddAtom(Chem.Atom(attype))
					bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
					mw.GetAtoms()[aidx].SetProp("covatom", "1")
					Chem.SanitizeMol(mw)
					lig_new.append(mw.GetMol())
			else:
				matches = mol.GetSubstructMatches(query3)
				if len(matches) > 0:
					for match in matches:
						for ax in match:
							ax2 = [x.GetIdx() for x in mol.GetAtoms()[ax].GetNeighbors() if x.GetIdx() not in match][0]
							mw = Chem.RWMol(mol)
							aidx = ax
							mw.RemoveBond(ax, ax2)
							mw.GetAtoms()[aidx].SetProp("covatom", "1")
							mxs = Chem.GetMolFrags(mw, asMols=True)
							for mx in mxs:
								if len([a for a in mx.GetAtoms() if a.HasProp("covatom")]) > 0:
									Chem.SanitizeMol(mx)
									lig_new.append(mx)
				else:			
					Chem.Kekulize(mol, clearAromaticFlags=True)
					matches = mol.GetSubstructMatches(query2)
					for match in matches:
						mw = Chem.RWMol(mol)
						mw.RemoveBond(match[0], match[1])
						aidx = mw.AddAtom(Chem.Atom(attype))
						bidx2 = mw.AddBond(aidx, match[0], Chem.BondType.SINGLE)
						mw.GetAtoms()[aidx].SetProp("covatom", "1")
						Chem.SanitizeMol(mw)
						lig_new.append(mw.GetMol())
		else:
			raise ValueError('Currently only "CYS" support "Disulfide Formation".')				
	
	elif rectype=="Others":
		query1 = Chem.MolFromSmarts("[N;+1]=[C]")
		mol_ = deepcopy(mol)
		Chem.Kekulize(mol_, clearAromaticFlags=True)	
		matches = mol_.GetSubstructMatches(query1)			
		if len(matches) > 0:
			for match in matches:
				mw = Chem.RWMol(mol_)
				mw.GetAtoms()[match[0]].SetFormalCharge(0)
				mw.RemoveBond(match[1], match[0])
				bidx1 = mw.AddBond(match[1], match[0], Chem.BondType.SINGLE)
				aidx = mw.AddAtom(Chem.Atom(attype))
				bidx2 = mw.AddBond(aidx, match[1], Chem.BondType.SINGLE)
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				Chem.SanitizeMol(mw)
				lig_new.append(mw.GetMol())	
		
		query2 = Chem.MolFromSmarts("[N][C;H0](=[N;D1])[N]1[C](O)CCCC1")
		matches = mol.GetSubstructMatches(query2)
		if len(matches) > 0:
			for match in matches:
				mw = Chem.RWMol(Chem.AddHs(mol))			
				mw.RemoveBond(match[4], match[3])
				mw.RemoveBond(match[4], match[5])
				aidx = [a.GetIdx() for a in mw.GetAtoms()[match[4]].GetNeighbors() if a.GetSymbol() == "H"][0]
				mw.ReplaceAtom(aidx, Chem.Atom(attype))
				aidxOH = [a.GetIdx() for a in mw.GetAtoms()[match[5]].GetNeighbors() if a.GetSymbol() == "H"][0]
				bidx1 = mw.AddBond(match[4], match[5], Chem.BondType.DOUBLE)	
				mw.RemoveBond(aidxOH, match[5])
				mw.RemoveAtom(aidxOH)	
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				Chem.SanitizeMol(mw)
				lig_new.append(Chem.RemoveHs(mw))
				
		query3 = Chem.MolFromSmarts("NCCc1ccc(O)cc1")
		matches = mol.GetSubstructMatches(query3)
		if len(matches) > 0:
			for match in matches:
				mw = Chem.RWMol(mol)			
				mw.RemoveBond(match[0], match[1])
				aidx = match[0]
				mw.ReplaceAtom(match[0], Chem.Atom(attype))
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				mxs = Chem.GetMolFrags(mw, asMols=True)
				for mx in mxs:
					if len([a for a in mx.GetAtoms() if a.HasProp("covatom")]) > 0:
						Chem.SanitizeMol(mx)
						lig_new.append(mx)	
				
		query4 = Chem.MolFromSmarts("o1c(=O)cccc1")
		matches = mol.GetSubstructMatches(query4)
		if len(matches) > 0:
			for match in matches:
				mw = Chem.RWMol(mol)
				Chem.Kekulize(mw, clearAromaticFlags=True)		
				mw.RemoveBond(match[0], match[1])
				aidx = mw.AddAtom(Chem.Atom(attype))
				bidx2 = mw.AddBond(aidx, match[1], Chem.BondType.SINGLE)
				mw.RemoveBond(match[5], match[6])
				bidx3 = mw.AddBond(match[5], match[6], Chem.BondType.SINGLE)
				mw.RemoveBond(match[0], match[6])
				bidx4 = mw.AddBond(match[0], match[6], Chem.BondType.DOUBLE)
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				Chem.SanitizeMol(mw)
				lig_new.append(Chem.RemoveHs(mw))
		
		query5 = Chem.MolFromSmarts("[O-0X1]=[P]1(O)[O][C](=O)-[C]=[C]O1")
		Chem.Kekulize(mol, clearAromaticFlags=True)	
		matches = mol.GetSubstructMatches(query5)
		if len(matches) > 0:
			for match in matches:
				mw = Chem.RWMol(mol)
				mw.RemoveBond(match[1], match[3])
				aidx = mw.AddAtom(Chem.Atom(attype))
				bidx2 = mw.AddBond(aidx, match[1], Chem.BondType.SINGLE)
				mw.RemoveBond(match[1], match[2])
				mw.GetAtoms()[match[2]].SetFormalCharge(0)
				bidx3 = mw.AddBond(match[1], match[2], Chem.BondType.DOUBLE)
				mw.GetAtoms()[aidx].SetProp("covatom", "1")
				Chem.SanitizeMol(mw)
				lig_new.append(mw.GetMol())
	
	else:
		raise ValueError('The currently supported reaction type include "Michael Addition", "Nucleophilic Addition to a Double Bond",\
							"Nucleophilic Addition to a Triple Bond", "Nucleophilic Substitution","Boronic Acid Addition",\
							"Epoxide/Aziridine Opening","Imine Condensation","Beta Lactone/Lactam Addition","Gamma Lactone/Lactam Addition",\
							"Disulfide Formation", "Others"!')
	
	#[m.SetProp('_Name',"%s-%s"%(m.GetProp("_Name"), i)) for i, (m,_) in enumerate(lig_new)]
	[set_mol_name(m, i) for i, m in enumerate(lig_new)]
	return lig_new
	
def set_mol_name(m, i):
	if m.HasProp("_Name"):
		m.SetProp('_Name',"%s-%s"%(m.GetProp("_Name"), i))
	else:
		m.SetProp('_Name',str(i))


def mol_prep(m, covatpos):
	coords = m.GetConformer().GetPositions()
	aidx = [a.GetIdx() for a in m.GetAtoms() if a.HasProp("covatom")][0]
	coords = coords - coords[aidx,:] + covatpos
	m_ = set_coord(m, coords, idx=0)
	m_.GetAtoms()[aidx].SetProp("covatom","1")
	return m_

