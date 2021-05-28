import os
import unittest
import sys
from melloddy_tuner.utils.config import ConfigDict, SecretDict
from melloddy_tuner.utils.scaffold_folding import ScaffoldFoldAssign
from melloddy_tuner.utils.df_transformer import DfTransformer
import filecmp
from pathlib import Path

curDir = os.path.dirname(os.path.abspath(__file__))
print(curDir)


class SNFoldCalculationTests(unittest.TestCase):

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

    #    def defineConfig(self,fp=3):
    #        if(fp==3):
    #            tuner.config.parameters.get_parameters(path=curDir+"/../tests/structure_preparation_test/example_parameters.json")
    #        else:
    #            tuner.config.parameters.get_parameters(path=curDir+"/input/ecfp2_param.json")
    #
    #    def defineConfigNewSecret(self):
    #        tuner.config.parameters.get_parameters(path=curDir+"/input/new_secret_param.json")
    ###############
    #### tests ####
    ###############

    def test_calculate_snfold_single_hard(self):
        """test the single claculation based on hard coded parameters"""
        input_smiles = (
            "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"  # imatinib
        )
        sa = ScaffoldFoldAssign(nfolds=5, secret="melloddy")
        result_actual = sa.calculate_single(input_smiles)
        result_expected = (
            "O=C(Nc1cccc(Nc2nccc(-c3cccnc3)n2)c1)c1ccc(CN2CCNCC2)cc1",
            "c1ccc(Nc2nccc(-c3cccnc3)n2)cc1",
            2,
            True,
            None,
        )
        self.assertEqual(result_actual, result_expected)

    def test_calculate_snfold_single_config(self):
        """test the single claculation based on config file conent"""
        input_smiles = (
            "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"  # imatinib
        )
        sa = ScaffoldFoldAssign(
            nfolds=self.config["scaffold_folding"]["nfolds"], secret=self.keys["key"]
        )
        result_actual = sa.calculate_single(input_smiles)
        result_expected = (
            "O=C(Nc1cccc(Nc2nccc(-c3cccnc3)n2)c1)c1ccc(CN2CCNCC2)cc1",
            "c1ccc(Nc2nccc(-c3cccnc3)n2)cc1",
            2,
            True,
            None,
        )
        self.assertEqual(result_actual, result_expected)

    def test_calculate_sn_fold_multiple(self):
        infile = os.path.join(curDir, "input", "test_sn_fold_input.csv")
        outfile = os.path.join(curDir, "output", "tmp", "sn_fold_output.csv")
        output_columns = [
            "murcko_smiles",
            "sn_smiles",
            "fold_id",
            "success",
            "error_message",
        ]
        output_types = ["object", "object", "int", "bool", "object"]

        sa = ScaffoldFoldAssign(
            nfolds=self.config["scaffold_folding"]["nfolds"], secret=self.keys["key"]
        )

        dt = DfTransformer(
            sa,
            input_columns={"canonical_smiles": "smiles"},
            output_columns=output_columns,
            output_types=output_types,
        )
        dt.process_file(infile, outfile)
        result = filecmp.cmp(
            os.path.join(curDir, "output", "test_sn_fold_output.csv"),
            os.path.join(outfile),
            shallow=False,
        )
        self.assertEqual(result, True)

    def test_calculate_sn_fold_multiple_split(self):
        infile = os.path.join(curDir, "input", "test_sn_fold_input.csv")
        outfile = os.path.join(curDir, "output", "tmp", "sn_fold_output.OK.csv")
        errfile = os.path.join(curDir, "output", "tmp", "sn_fold_output.failed.csv")
        output_columns = [
            "murcko_smiles",
            "sn_smiles",
            "fold_id",
            "success",
            "error_message",
        ]
        output_types = ["object", "object", "int", "bool", "object"]

        sa = ScaffoldFoldAssign(
            nfolds=self.config["scaffold_folding"]["nfolds"], secret=self.keys["key"]
        )
        dt = DfTransformer(
            sa,
            input_columns={"canonical_smiles": "smiles"},
            output_columns=output_columns,
            output_types=output_types,
            success_column="success",
        )
        dt.process_file(infile, outfile, error_file=errfile)

        result_OK = filecmp.cmp(
            os.path.join(curDir, "output", "test_sn_fold_output.OK.csv"),
            os.path.join(outfile),
            shallow=False,
        )
        result_failed = filecmp.cmp(
            os.path.join(curDir, "output", "test_sn_fold_output.failed.csv"),
            os.path.join(errfile),
            shallow=False,
        )
        self.assertEqual(result_OK & result_failed, True)

    def test_calculate_sn_fold_multiple_split_par(self):
        infile = os.path.join(curDir, "input", "test_sn_fold_input.csv")
        outfile = os.path.join(
            curDir, "output", "tmp", "sn_fold_output_parallel.OK.csv"
        )
        errfile = os.path.join(
            curDir, "output", "tmp", "sn_fold_output.parallel.failed.csv"
        )
        output_columns = [
            "murcko_smiles",
            "sn_smiles",
            "fold_id",
            "success",
            "error_message",
        ]
        output_types = ["object", "object", "int", "bool", "object"]

        sa = ScaffoldFoldAssign(
            nfolds=self.config["scaffold_folding"]["nfolds"], secret=self.keys["key"]
        )
        dt = DfTransformer(
            sa,
            input_columns={"canonical_smiles": "smiles"},
            output_columns=output_columns,
            output_types=output_types,
            success_column="success",
            nproc=2,
        )
        dt.process_file(infile, outfile, error_file=errfile)

        result_OK = filecmp.cmp(
            os.path.join(curDir, "output", "test_sn_fold_output.OK.csv"),
            os.path.join(outfile),
            shallow=False,
        )
        result_failed = filecmp.cmp(
            os.path.join(curDir, "output", "test_sn_fold_output.failed.csv"),
            os.path.join(errfile),
            shallow=False,
        )
        self.assertEqual(result_OK & result_failed, True)


# if __name__ == "__main__":
#    unittest.main()
