# CarsiDock-Cov

CarsiDock-Cov is a deep learning-guided protocol for automated covalent docking. It inherits the core components of [CarsiDock]((https://github.com/carbonsilicon-ai/CarsiDock/tree/main)), but makes a few changes  to facilitate the applicability of the whole protocol for covalent docking.
<div align=center>
<img src="https://github.com/sc8668/CarsiDock-Cov/blob/main/data/111.jpg" width="600px" height="600px">
</div> 

### Requirements
rdkit==2022.9.4    
ProDy==2.1.0    
pytorch-lightning==1.5.0    
python==3.8.19    
pytorch==1.12.1+cu116   
transformers==4.15.0   
spyrmsd==0.6.0   
joblib==1.4.2   
torchmetrics==1.4.1   
numpy==1.24.4   
scipy==1.10.1   
torch-scatter==2.1.2   
scikit-learn==1.3.2   
pandas==2.0.3   

### Datasets
The datasets employed for the validation of our approach have been shared at [Zenodo](https://zenodo.org/uploads/14064834).

### License
The code of this repository is licensed under [Aapache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0). CarsiDock-Cov directly ueses the model trained in [CarsiDock](https://github.com/carbonsilicon-ai/CarsiDock/tree/main) to predict the protein-ligand distance matrices, so the use of the CarsiDock model weights should follow the Model License. CarsiDock weights are completely open for academic research, please contact bd@carbonsilicon.ai for commercial use.

### Checkpoints
If you agree to the above license, please download checkpoints from the corresponding repository and put them in the checkpoints folder.

[CarsiDock](https://github.com/carbonsilicon-ai/CarsiDock/tree/main)  
[RTMScore](https://github.com/sc8668/RTMScore/tree/main)   

### Examples for using our protocol for covalent docking
___# Docking using the SMILES as input___
```
python run_carsidockcov.py -p ./example_data/2qlq_p.pdb -rl ./example_data/2qlq_l.sdf \
-smi "CN(C)C/C=C/C(=O)Nc1ccc2ncnc(Nc3cccc(Br)c3)c2c1" -covres "A:CYS:345" \
-rectype 'Michael Addition' -remove_dummyatom -remain_pocket -o './example_data/2qlq_out' 
```
___# Docking using the sdf as input___
```
python run_carsidockcov.py -p ./example_data/2qlq_p.pdb -rl ./example_data/2qlq_l.sdf \
-l ./example_data/2qlq_l_orig.sdf -covres "A:CYS:345" \
-rectype 'Michael Addition' -remove_dummyatom -remain_pocket -o './example_data/2qlq_out'  
```
___# Virtual screening___
```
python run_carsidockcov_screening.py -p ./example_data/2qlq_p.pdb -rl ./example_data/2qlq_l.sdf \
-l ./example_data/covvs.sdf -covres "A:CYS:345" \
-rectype 'Michael Addition' -remove_dummyatom -remain_pocket -o './example_data/2qlq_out'  
```


