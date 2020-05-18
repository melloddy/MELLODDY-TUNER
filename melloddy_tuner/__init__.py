# -*- coding: utf-8 -*-

"""
MELLODDY-TUNER: MELLODDY - Tool for Unifying aNd Encrypting of data Records
Structure and activity data standardization tool for the IMI project MELLODDY.
Authors:        Lukas Friedrich (Merck KGaA, Darmstadt), Jaak Simm (University of Leuven, Leuven)
Contributors:   Wouter Heyndrickx (Janssen), Noe Sturm (Novartis), Ansgar Schuffenhauer (Novartis),
                Lina Humbeck (Boehringer Ingelheim), Mervin Lewis (AstraZeneca), Adam Zalewski (Amgen),
                WP1 Team

"""
from .version import __version__
from .config import ConfigDict
from .helper import read_csv, bits_to_str, make_scrambled_lists, make_lsh, int_to_sha256, sha256, lsh_to_fold, \
    hashed_fold_lsh
from .chem_utils import run_standardize, output_processed_smiles, output_failed_smiles, run_fingerprint, output_processed_descriptors, output_descriptor_duplicates, output_mapping_table_T5, make_desc_dict
from .formatting import *
from .folding import LSHFolding

