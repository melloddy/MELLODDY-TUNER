import os
import unittest
from pathlib import Path

from melloddy_tuner.utils.config import ConfigDict, SecretDict
from melloddy_tuner.utils.df_transformer import DfTransformer
from melloddy_tuner.utils.helper import read_csv
from melloddy_tuner.utils.standardizer import Standardizer

# sys.path.insert(0, '../melloddy_tuner')
from rdkit import rdBase

rdBase.DisableLog("rdApp.*")
import filecmp

from pandas._testing import assert_frame_equal

curDir = Path(os.path.dirname(os.path.abspath(__file__)))
print(curDir)


class StandardizationTests(unittest.TestCase):
    referenceFilePath = curDir / "output/smiles_prepared.csv"

    ############################
    #### setup and teardown ####
    ############################

    # executed after each test
    def tearDown(self):
        pass

    def setUp(self):
        self.config = ConfigDict(
            config_path=Path(
                os.path.join(curDir, "reference_files", "example_parameters.json")
            )
        ).get_parameters()
        self.keys = SecretDict(
            key_path=Path(os.path.join(curDir, "reference_files", "example_key.json"))
        ).get_secrets()

    ###############
    #### tests ####
    ###############

    def test_standardizer_calculate_single(self):
        """Testing standardization of a single smiles from Chembl with reference smiles output obtained Feb 14th 2020"""
        # self.defineConfig()
        st = Standardizer.from_param_dict(
            method_param_dict=self.config["standardization"], verbosity=0
        )
        response = st.calculate_single(
            "Cc1ccc(cc1)S(=O)(=O)Nc2ccc(cc2)c3nc4ccc(NS(=O)(=O)c5ccc(C)cc5)cc4[nH]3"
        )[0]
        self.assertEqual(
            response,
            "Cc1ccc(S(=O)(=O)Nc2ccc(-c3nc4cc(NS(=O)(=O)c5ccc(C)cc5)ccc4[nH]3)cc2)cc1",
        )

    def test_standardizer_multiprocessing(self):
        """Testing standardization of smiles using threading"""

        df_smiles = read_csv(curDir / "input/chembl/chembl_23_example_T2.csv", nrows=10)
        st = Standardizer(
            max_num_atoms=self.config["standardization"]["max_num_atoms"],
            max_num_tautomers=self.config["standardization"]["max_num_tautomers"],
            include_stereoinfo=self.config["standardization"]["include_stereoinfo"],
            verbosity=0,
        )
        outcols = ["canonical_smiles", "success", "error_message"]
        out_types = ["object", "bool", "object"]
        dt_2 = DfTransformer(
            st,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=2,
            verbosity=0,
        )
        response2 = dt_2.process_dataframe(df_smiles)[0]
        dt_4 = DfTransformer(
            st,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=4,
            verbosity=0,
        )
        response4 = dt_4.process_dataframe(df_smiles)[0]
        assert_frame_equal(response2, response4)

    def test_standardizer_parameter_atom_count(self):
        """Testing standardization with different number of max atom count"""

        df_smiles = read_csv(curDir / "input/test_standardizer.csv")
        outcols = ["canonical_smiles", "success", "error_message"]
        out_types = ["object", "bool", "object"]

        ## Load ref standardizer
        st_ref = Standardizer(
            max_num_atoms=self.config["standardization"]["max_num_atoms"],
            max_num_tautomers=self.config["standardization"]["max_num_tautomers"],
            include_stereoinfo=self.config["standardization"]["include_stereoinfo"],
            verbosity=0,
        )
        dt_ref = DfTransformer(
            st_ref,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=4,
            verbosity=0,
        )
        response_ref = dt_ref.process_dataframe(df_smiles)[0]

        ## load test standardizer
        st_tmp = Standardizer(
            max_num_atoms=5,
            max_num_tautomers=self.config["standardization"]["max_num_tautomers"],
            include_stereoinfo=self.config["standardization"]["include_stereoinfo"],
            verbosity=0,
        )

        dt_tmp = DfTransformer(
            st_tmp,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=2,
            verbosity=0,
        )
        response_tmp = dt_tmp.process_dataframe(df_smiles)[0]

        try:
            assert_frame_equal(response_ref, response_tmp)
        except AssertionError:
            # frames are not equal
            pass
        else:
            # frames are equal
            raise AssertionError

    def test_standardizer_parameter_tautomer_count(self):
        """Testing standardization with different number of max tautomers"""

        df_smiles = read_csv(curDir / "input/test_standardizer.csv")
        outcols = ["canonical_smiles", "success", "error_message"]
        out_types = ["object", "bool", "object"]

        ## Load ref standardizer
        st_ref = Standardizer(
            max_num_atoms=self.config["standardization"]["max_num_atoms"],
            max_num_tautomers=self.config["standardization"]["max_num_tautomers"],
            include_stereoinfo=self.config["standardization"]["include_stereoinfo"],
            verbosity=0,
        )
        dt_ref = DfTransformer(
            st_ref,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=4,
            verbosity=0,
        )
        response_ref = dt_ref.process_dataframe(df_smiles)[0]

        ## load test standardizer
        st_tmp = Standardizer(
            max_num_atoms=self.config["standardization"]["max_num_atoms"],
            max_num_tautomers=1,
            include_stereoinfo=self.config["standardization"]["include_stereoinfo"],
            verbosity=0,
        )

        dt_tmp = DfTransformer(
            st_tmp,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=2,
            verbosity=0,
        )
        response_tmp = dt_tmp.process_dataframe(df_smiles)[0]

        try:
            assert_frame_equal(response_ref, response_tmp)
        except AssertionError:
            # frames are not equal
            pass
        else:
            # frames are equal
            raise AssertionError

    def test_standardizer_parameter_stereoinfo(self):
        """Testing standardization with including stereochemistry"""

        df_smiles = read_csv(curDir / "input/test_standardizer.csv")
        outcols = ["canonical_smiles", "success", "error_message"]
        out_types = ["object", "bool", "object"]

        ## Load ref standardizer
        st_ref = Standardizer(
            max_num_atoms=self.config["standardization"]["max_num_atoms"],
            max_num_tautomers=self.config["standardization"]["max_num_tautomers"],
            include_stereoinfo=self.config["standardization"]["include_stereoinfo"],
            verbosity=0,
        )
        dt_ref = DfTransformer(
            st_ref,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=4,
            verbosity=0,
        )
        response_ref = dt_ref.process_dataframe(df_smiles)[0]

        ## load test standardizer
        st_tmp = Standardizer(
            max_num_atoms=self.config["standardization"]["max_num_atoms"],
            max_num_tautomers=self.config["standardization"]["max_num_tautomers"],
            include_stereoinfo=True,
            verbosity=0,
        )

        dt_tmp = DfTransformer(
            st_tmp,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=2,
            verbosity=0,
        )
        response_tmp = dt_tmp.process_dataframe(df_smiles)[0]

        try:
            assert_frame_equal(response_ref, response_tmp)
        except AssertionError:
            # frames are not equal
            pass
        else:
            # frames are equal
            raise AssertionError

    def test_standardizer_different_configs(self):
        """Testing standardization of smiles using threading"""

        df_smiles = read_csv(curDir / "input/test_standardizer.csv")
        outcols = ["canonical_smiles", "success", "error_message"]
        out_types = ["object", "bool", "object"]

        ## Load ref standardizer
        st_ref = Standardizer(
            max_num_atoms=self.config["standardization"]["max_num_atoms"],
            max_num_tautomers=self.config["standardization"]["max_num_tautomers"],
            include_stereoinfo=self.config["standardization"]["include_stereoinfo"],
            verbosity=0,
        )
        dt_ref = DfTransformer(
            st_ref,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=4,
            verbosity=0,
        )
        response_ref = dt_ref.process_dataframe(df_smiles)[0]
        config_2 = ConfigDict(
            config_path=Path(
                os.path.join(curDir, "input/", "example_parameters_2.json")
            )
        ).get_parameters()
        ## load test standardizer
        st_tmp = Standardizer(
            max_num_atoms=config_2["standardization"]["max_num_atoms"],
            max_num_tautomers=config_2["standardization"]["max_num_tautomers"],
            include_stereoinfo=config_2["standardization"]["include_stereoinfo"],
            verbosity=0,
        )

        dt_tmp = DfTransformer(
            st_tmp,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=2,
            verbosity=0,
        )
        response_tmp = dt_tmp.process_dataframe(df_smiles)[0]

        try:
            assert_frame_equal(response_ref, response_tmp)
        except AssertionError:
            # frames are not equal
            pass
        else:
            # frames are equal
            raise AssertionError

    def test_standardizer_pipeline(self):
        """
        Testing standardization of a larger set of smiles from Chembl using serial execution
        Compared are resulting output files.
        """
        infile = os.path.join(curDir, "input", "test_standardizer.csv")
        outfile = os.path.join(curDir, "output", "sn_fold_output.OK.csv")
        errfile = os.path.join(curDir, "output", "sn_fold_output.failed.csv")
        outfile_tmp = os.path.join(curDir, "output", "tmp", "sn_fold_output.OK.csv")
        errfile_tmp = os.path.join(curDir, "output", "tmp", "sn_fold_output.failed.csv")
        st = Standardizer.from_param_dict(
            method_param_dict=self.config["standardization"], verbosity=0
        )
        outcols = ["canonical_smiles", "success", "error_message"]
        out_types = ["object", "bool", "object"]
        dt = DfTransformer(
            st,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=2,
            verbosity=0,
        )

        # build reference files, only run once
        # dt.process_file(infile, outfile, errfile)

        # run test with tmp files
        dt.process_file(infile, outfile_tmp, errfile_tmp)

        result = filecmp.cmp(outfile, outfile_tmp, shallow=False)
        error = filecmp.cmp(errfile, errfile_tmp, shallow=False)
        os.remove(outfile_tmp)
        os.remove(errfile_tmp)
        self.assertEqual(result, error, True)


# if __name__ == "__main__":
# unittest.main()
