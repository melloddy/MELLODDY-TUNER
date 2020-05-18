import importlib
import json
import time

import rdkit.Chem.MolStandardize.standardize as MolVS_standardizer
import rdkit.Chem.MolStandardize.tautomer as tautomer
from rdkit import rdBase
from rdkit.Chem.MolStandardize.tautomer import TautomerTransform

rdBase.DisableLog('rdApp.*')
import logging
from multiprocessing import Pool
import tqdm
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit.Chem.AllChem import GetHashedMorganFingerprint, GetMorganFingerprint


import numpy as np


from .helper import *
from .config import parameters


####################################

class ReadConfig(object):
    """
    Read parameters from config file.
    """

    def __init__(self, conf_dict=None):
        if conf_dict is None:
            conf_dict = {}
        self._conf_dict = conf_dict
        self._param = self._load_conf_dict()

    def get_conf_dict(self, conf_dict=None):
        if conf_dict is None:
            conf_dict = {}
        self._conf_dict = self._conf_dict if conf_dict is None else conf_dict
        return self._load_conf_dict()

    def _load_conf_dict(self):
        tmp_dict = self._conf_dict
        return tmp_dict


def update_tautomer_rules():
    """
    Update of the RDKit tautomerization rules, especially rules for amide tautomerization.
    :return: updated list of tautomer rules
    """
    newtfs = (
        #######################################################################
        # Updated tautomer rules
        TautomerTransform('1,3 (thio)keto/enol f', '[CX4!H0]-[C;!$([C]([CH1])(=[O,S,Se,Te;X1])-[N,O])]=[O,S,Se,Te;X1]'),
        TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C;!$(C(=[C])(-[O,S,Se,Te;X2!H0])-[N,O])]=[C]'),
        TautomerTransform('1,5 (thio)keto/enol f',
                          '[CX4,NX3;!H0]-[C]=[C]-[C;!$([C]([C])(=[O,S,Se,Te;X1])-[N,O])]=[O,S,Se,Te;X1]'),
        TautomerTransform('1,5 (thio)keto/enol r',
                          '[O,S,Se,Te;X2!H0]-[C;!$(C(=[C])(-[O,S,Se,Te;X2!H0])-[N,O])]=[C]-[C]=[C,N]'),
        TautomerTransform('aliphatic imine f', '[CX4!H0]-[C;$([CH1](C)=N),$(C(C)([#6])=N)]=[N;$([NX2][#6]),$([NX2H])]'),
        TautomerTransform('aliphatic imine r',
                          '[N!H0;$([NX3H1][#6]C),$([NX3H2])]-[C;$(C(N)([#6])=C),$([CH1](N)=C)]=[CX3]'),
        TautomerTransform('special imine f', '[N!H0]-[C]=[CX3R0]'),
        TautomerTransform('special imine r', '[CX4!H0]-[c]=[n]'),

        #######################################################################
        #######################################################################
        # OLD tautomer rules
        # TautomerTransform('1,3 (thio)keto/enol f', '[CX4!H0]-[C]=[O,S,Se,Te;X1]'),
        # TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]'),
        # TautomerTransform('1,5 (thio)keto/enol f', '[CX4,NX3;!H0]-[C]=[C][CH0]=[O,S,Se,Te;X1]'),
        # TautomerTransform('1,5 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[CH0]=[C]-[C]=[C,N]'),
        # TautomerTransform('aliphatic imine f', '[CX4!H0]-[C]=[NX2]'),
        # TautomerTransform('aliphatic imine r', '[NX3!H0]-[C]=[CX3]'),
        # TautomerTransform('special imine f', '[N!H0]-[C]=[CX3R0]'),
        # TautomerTransform('special imine r', '[CX4!H0]-[c]=[n]'),
        #######################################################################
        TautomerTransform('1,3 aromatic heteroatom H shift f', '[#7!H0]-[#6R1]=[O,#7X2]'),
        TautomerTransform('1,3 aromatic heteroatom H shift r', '[O,#7;!H0]-[#6R1]=[#7X2]'),
        TautomerTransform('1,3 heteroatom H shift', '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,5 aromatic heteroatom H shift', '[#7,#16,#8;!H0]-[#6,#7]=[#6]-[#6,#7]=[#7,#16,#8;H0]'),
        TautomerTransform('1,5 aromatic heteroatom H shift f',
                          '[#7,#16,#8,Se,Te;!H0]-[#6,nX2]=[#6,nX2]-[#6,#7X2]=[#7X2,S,O,Se,Te]'),
        TautomerTransform('1,5 aromatic heteroatom H shift r',
                          '[#7,S,O,Se,Te;!H0]-[#6,#7X2]=[#6,nX2]-[#6,nX2]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,7 aromatic heteroatom H shift f',
                          '[#7,#8,#16,Se,Te;!H0]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6]-[#6,#7X2]=[#7X2,S,O,Se,Te,CX3]'),
        TautomerTransform('1,7 aromatic heteroatom H shift r',
                          '[#7,S,O,Se,Te,CX4;!H0]-[#6,#7X2]=[#6]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[NX2,S,O,Se,Te]'),
        TautomerTransform('1,9 aromatic heteroatom H shift f',
                          '[#7,O;!H0]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#7,O]'),
        TautomerTransform('1,11 aromatic heteroatom H shift f',
                          '[#7,O;!H0]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#7X2,O]'),
        TautomerTransform('furanone f', '[O,S,N;!H0]-[#6r5]=[#6X3r5;$([#6]([#6r5])=[#6r5])]'),
        TautomerTransform('furanone r', '[#6r5!H0;$([#6]([#6r5])[#6r5])]-[#6r5]=[O,S,N]'),
        TautomerTransform('keten/ynol f', '[C!H0]=[C]=[O,S,Se,Te;X1]', bonds='#-'),
        TautomerTransform('keten/ynol r', '[O,S,Se,Te;!H0X2]-[C]#[C]', bonds='=='),
        TautomerTransform('ionic nitro/aci-nitro f', '[C!H0]-[N+;$([N][O-])]=[O]'),
        TautomerTransform('ionic nitro/aci-nitro r', '[O!H0]-[N+;$([N][O-])]=[C]'),
        TautomerTransform('oxim/nitroso f', '[O!H0]-[N]=[C]'),
        TautomerTransform('oxim/nitroso r', '[C!H0]-[N]=[O]'),
        TautomerTransform('oxim/nitroso via phenol f', '[O!H0]-[N]=[C]-[C]=[C]-[C]=[OH0]'),
        TautomerTransform('oxim/nitroso via phenol r', '[O!H0]-[c]=[c]-[c]=[c]-[N]=[OH0]'),
        TautomerTransform('cyano/iso-cyanic acid f', '[O!H0]-[C]#[N]', bonds='=='),
        TautomerTransform('cyano/iso-cyanic acid r', '[N!H0]=[C]=[O]', bonds='#-'),
        # TautomerTransform('formamidinesulfinic acid f', '[O,N;!H0]-[C]=[S,Se,Te]=[O]', bonds='=--'),
        # TautomerTransform('formamidinesulfinic acid r', '[O!H0]-[S,Se,Te]-[C]=[O,N]', bonds='=--'),
        TautomerTransform('isocyanide f', '[C-0!H0]#[N+0]', bonds='#', charges='-+'),
        TautomerTransform('isocyanide r', '[N+!H0]#[C-]', bonds='#', charges='-+'),
        TautomerTransform('phosphonic acid f', '[OH]-[PH0]', bonds='='),
        TautomerTransform('phosphonic acid r', '[PH]=[O]', bonds='-'),
    )
    return newtfs


####################################


def structure_standardization(smi):
    """
    Standardization function to clean up smiles with RDKit. First, the input smiles is converted into a mol object.
    Not-readable SMILES are written to the log file. The molecule size is checked by the number of atoms (non-hydrogen).
    If the molecule has more than 100 non-hydrogen atoms, the compound is discarded and written in the log file.
    Molecules with number of non-hydrogen atoms <= 100 are standardized with the MolVS toolkit
    (https://molvs.readthedocs.io/en/latest/index.html) relying on RDKit. Molecules which failed the standardization
    process are saved in the log file. The remaining standardized structures are converted back into their canonical
    SMILES format.
    :param smi: Input SMILES from the given structure data file T4
    :return: smi_clean: Cleaned and standardized canonical SMILES of the given input SMILES.
    """
    tautomer.TAUTOMER_TRANSFORMS = update_tautomer_rules()
    importlib.reload(MolVS_standardizer)
    param = ReadConfig()
    standardization_param = param.get_conf_dict(parameters.get_parameters())['standardization']

    max_num_atoms = standardization_param['max_num_atoms']
    max_num_tautomers = standardization_param['max_num_tautomers']
    include_stereoinfo = standardization_param['include_stereoinfo']
    my_standardizer = MolVS_standardizer.Standardizer(max_tautomers=max_num_tautomers)

    mol = MolFromSmiles(smi)  # Read SMILES and convert it to RDKit mol object.
    if mol is not None:  # Check, if the input SMILES has been converted into a mol object.
        if mol.GetNumAtoms() <= max_num_atoms:  # check size of the molecule based on the non-hydrogen atom count.
            try:

                mol = my_standardizer.charge_parent(mol)  # standardize molecules using MolVS and RDKit
                mol = my_standardizer.isotope_parent(mol)
                if include_stereoinfo is False:
                    mol = my_standardizer.stereo_parent(mol)
                    mol = my_standardizer.tautomer_parent(mol)
                    mol_clean = my_standardizer.standardize(mol)
                    smi_clean = MolToSmiles(mol_clean)  # convert mol object back to SMILES
                else:
                    mol = my_standardizer.tautomer_parent(mol)
                    mol_clean = my_standardizer.standardize(mol)
                    smi_clean = MolToSmiles(mol_clean)
            except (ValueError, AttributeError) as e:
                smi_clean = np.nan
                logging.error(
                    'Standardization error, ' + smi + ', Error Type: ' + str(
                        e))  # write failed molecules during standardization to log file

        else:
            smi_clean = np.nan
            logging.error('Molecule too large, ' + smi)

    else:
        smi_clean = np.nan
        logging.error('Reading Error, ' + smi)

    return smi_clean


def run_standardize(smi, num_cpu):
    """
    Main function to run the standardization protocol in multiprocessing fashion. Standardization of molecules is
    distributed onto the given number of CPUs. The standardization script returns a csv file with the IDs and
    the standardized SMILES ('canonical_smiles') and a log file of the non-processed structures which failed for
    certain reasons.
    :param smi: input smiles
    :param num_cpu: number of CPUs available to un the script. Default = 32
    :return: Standardized molecular structures as column in dataframe (canonical_smiles)
    """
    with Pool(processes=num_cpu) as pool:
        ####################################
        smi_standardized = list(tqdm.tqdm(pool.imap(multi_standardization, smi), total=len(smi)))
        ####################################
        pool.close()
        pool.join()
    return smi_standardized


def multi_standardization(x):
    """
    Wrapper for ultiprocessing RDKit standardization
    """
    return structure_standardization(x)

    ####################################



def output_processed_smiles(smi_standardized, df_structure):
    """
    formatting structure data to output dataframes.
    :param smi_standardized: standardized and cleaned SMILES
    :param df_structure: input structure data as dataframe
    :return: cleaned structures in a dataframe
    """
    df_clean_structures = df_structure
    df_clean_structures['canonical_smiles'] = smi_standardized

    df_clean_structures.dropna(subset=['canonical_smiles'], inplace=True)

    return df_clean_structures


def output_failed_smiles(smi_standardized, df_structure):
    """
    Output of structure failed during the standardization process.
    :param smi_standardized:
    :param df_structure:
    :return: dataframe of failed structures
    """
    df_clean_structures = df_structure
    df_clean_structures['canonical_smiles'] = smi_standardized
    df_failed_structures = df_clean_structures[pd.isna(df_structure['canonical_smiles'])]
    return df_failed_structures

    ####################################



def fp_calc(smi):
    """
    Calculation of Morgan fingerprints (ECFP equivalent) with a given radius (default: radius = 3 similar to ECFP6).
    Failed structures are saved in a log file.
    :param smi: standardized SMILES from previously processed structure data
    :return:    fingerprint non-zero elements (on bits) and the frequencies of each element
                as an array of lists stored in a tuple.
    """
    param = ReadConfig()
    fingerprint_param = param.get_conf_dict(parameters.get_parameters())['fingerprint']
    fp_radius = fingerprint_param['radius']
    fp_hashed = fingerprint_param['hashed']
    fp_nbits = fingerprint_param['fold_size']

    mol_fp = {}
    mol = MolFromSmiles(smi)

    if mol is not None:
        if fp_hashed is True:
            try:
                mol_fp = GetHashedMorganFingerprint(mol, fp_radius, fp_nbits).GetNonzeroElements()
            except:
                mol_fp = np.nan
                logging.error('Fingerprint calculation error, ' + smi)
        else:
            try:
                mol_fp = GetMorganFingerprint(mol, fp_radius).GetNonzeroElements()
            except:
                mol_fp = np.nan
                logging.error('Fingerprint calculation error, ' + smi)
    return np.array(list(mol_fp.keys())), np.array(list(mol_fp.values()))



def multi_fp_calc(x):
    """
    Wrapper for multiprocessing fingerprint calculations
    """
    return fp_calc(x)


def run_fingerprint(smi, num_cpu):
    """
    Calculation of Morgan fingerprints with a multiprocessing setup. The standardized SMILES strings are distributed onto
    the given number of CPUs.
    :param smi:Standardized SMILES structures from the previously processed file ending with the name _standardized.csv
    :param num_cpu: The given number of available CPUs.
    :return: Tuple of fingerprint features and values as lists in a NumPy array
    """
    with Pool(processes=num_cpu) as pool:
        ####################################
        start = time.time()
        ####################################
        mol_fp = list(tqdm.tqdm(pool.imap(multi_fp_calc, smi), total=len(smi)))
        print(f'Calculating fingerprints of {len(smi)} molecules took {time.time() - start:.08} seconds.')
        pool.close()
        pool.join()
    return mol_fp


def make_fp_lists(ecfp_feat, ecfp_val):
    """
    Converts fingerprint features and values into a scrambled lists of fingerprint features.
    :param ecfp_feat:   Fingerprint features
    :param ecfp_val:    Fingerprint values
    :return:
    """
    ecfp_feat_list = list(ecfp_feat)
    ecfp_val_list = list(ecfp_val)
    param = ReadConfig()
    fingerprint_param = param.get_conf_dict(parameters.get_parameters())['fingerprint']
    fp_binarized = fingerprint_param['binarized']
    fp_bitsize = fingerprint_param['fold_size']
    secret = param.get_conf_dict(parameters.get_parameters())['key']['key']
    ecfp_feat_list_scrambled = make_scrambled_lists(ecfp_feat_list, secret, fp_bitsize)
    if fp_binarized:
        for e in ecfp_val_list:
            e.fill(1)
    return ecfp_feat_list_scrambled, ecfp_val_list



def output_processed_descriptors(ecfp, df_structure):
    """
    Save the fingerprint information in the given structure data frame.
    :param ecfp: Dictionary of fingerprint features (keys) and values (values)
    :param df_structure: given structure dataframe
    :return: Dataframe with descriptor id and fingerprint information
    """
    # Convert list of tuples back to dictionary of fingerprint features and frequencies
    ecfp_feat, ecfp_val = make_desc_dict(ecfp)

    ecfp_feat_list_scrambled = make_fp_lists(ecfp_feat, ecfp_val)[0]

    ecfp_val_list = make_fp_lists(ecfp_feat, ecfp_val)[1]
    # adding fingerprint information to dataframe columns
    df_structure['ecfp_feat'] = ecfp_feat_list_scrambled
    df_structure['ecfp_val'] = ecfp_val_list
    # create fingerprint column in json format to identify duplicates
    df_structure['fp_json'] = [arr.tolist() for arr in ecfp_feat_list_scrambled]
    df_structure['fp_val_json'] = [arr.tolist() for arr in ecfp_val_list]
    df_structure['fp_json'] = df_structure['fp_json'].apply(lambda x: json.dumps(x))   
    df_structure['fp_val_json'] = df_structure['fp_val_json'].apply(lambda x: json.dumps(x))

    # identify duplicated fingerprint, create unique descriptor vector ID for them,
    # and sort them according to the new descriptor ID
    df_structure['descriptor_vector_id'] = df_structure.groupby(['fp_json', 'fp_val_json']).ngroup()
    return df_structure


def make_desc_dict(ecfp):
    """
    Format tuple of fingerprint information into dictionary
    :param ecfp: Tuple of fingerprint features and values
    :return: dictionary of fingerprint features and values
    """
    ecfp_feat, ecfp_val = zip(*ecfp)
    return ecfp_feat, ecfp_val


def output_descriptor_duplicates(df_structure):
    """
    Output duplicate entries based on descriptor id.
    :param df_structure: Given dataframe with fingerprint information and descriptor id
    :return: duplicate descriptor ids
    """
    df_structure_duplicates = pd.DataFrame(df_structure,
                                           columns=['input_compound_id', 'canonical_smiles', 'fp_json',
                                                    'descriptor_vector_id'])
    df_structure_duplicates = df_structure_duplicates[
        df_structure_duplicates.duplicated(['descriptor_vector_id'],
                                           keep=False)]
    df_structure_duplicates = df_structure_duplicates.sort_values('descriptor_vector_id')
    return df_structure_duplicates


def output_mapping_table_T5(df_structure):
    """
    Create mapping table between input compound ids and descriptor ids
    :param df_structure: structure dataframe including fingerprint information and descriptor ids
    :return: Mapping table T5 containing original input_compound_id and descriptor_vector_id
    """
    mapping_table_T5 = pd.DataFrame(df_structure,
                                    columns=['input_compound_id', 'descriptor_vector_id']) \
        .sort_values(['input_compound_id'])

    return mapping_table_T5
