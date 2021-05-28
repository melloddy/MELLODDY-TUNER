import pandas as pd
import dask.dataframe as daskdf
import os


class DfTransformer(object):
    def __init__(
        self,
        calculator_object,
        input_columns,
        output_columns,
        output_types=None,
        success_column=None,
        meta=None,
        nproc=1,
        verbosity=0,
    ):
        """Constructor
        Args:
            calculator_object(): an instantiated calculator object
            input_columns(dict): dictionary mapping function keyword arguments to dataframe input columns
            output_columns(list or dict): list of column names to map the function output to
            output_types
            success_column(str): Name of a boolean column indicating successfull computation, must be member of output_columns
            nproc(int): number of prcoessors to use
            verbosity(int): verbosity level

        Returns:
            DfTransformer object
        """
        self.calculator = calculator_object
        self.input_columns = input_columns
        self.output_columns = output_columns
        if output_types is not None:
            self.meta = {i: type for i, type in enumerate(output_types)}
        else:
            self.meta = None
        if success_column is not None:
            if success_column not in output_columns:
                raise ValueError(
                    "success_column {0} is not in the output columns {1}".format(
                        success_column, output_columns
                    )
                )
        self.success_column = success_column
        self.nproc = nproc
        self.verbosity = verbosity

    # simple non-paralell implementation
    def process_dataframe(self, df):
        """Function to add a fingerprint column to a datafarme containing the input smiles

        Args:
            df (DataFrame): the dataframe to operate on

        Returns:
            Tuple(DataFrame, DataFrame) : successfully processed dataframe with added result columns, dataframe with failed records (in case success column attribute is set)
        """
        if self.nproc == 1:
            df[self.output_columns] = (
                df[self.input_columns.keys()]
                .rename(columns=self.input_columns)
                .apply(
                    lambda row: self.calculator.calculate_single(**row),
                    axis=1,
                    result_type="expand",
                )
            )
        else:
            # paralleleized implementaion using dask, very basic, still all in memory
            ddf = daskdf.from_pandas(df, npartitions=self.nproc)
            # dask dataframe want an explicit list, rather then the key iterable for column indexing
            # if we have type meta information we provide it to dask
            if self.meta is None:
                ddf[self.output_columns] = (
                    ddf[list(self.input_columns.keys())]
                    .rename(columns=self.input_columns)
                    .apply(
                        lambda row: self.calculator.calculate_single(**row),
                        axis=1,
                        result_type="expand",
                    )
                )
            else:
                ddf[self.output_columns] = (
                    ddf[list(self.input_columns.keys())]
                    .rename(columns=self.input_columns)
                    .apply(
                        lambda row: self.calculator.calculate_single(**row),
                        axis=1,
                        meta=self.meta,
                        result_type="expand",
                    )
                )
            df = ddf.compute(scheduler="processes", num_workers=self.nproc)
        if self.success_column is not None:
            processed_df = df[df[self.success_column] == True]
            failed_df = df[df[self.success_column] == False]
        else:
            processed_df = df
            failed_df = None
        return processed_df, failed_df

    # simple implentation going over the dafarme in memory
    def process_file(self, input_file, output_file, error_file=None):
        """Function to calculate a fingeprint for a csv file with a smiles column, and to write out a copy of that csv file with the fingerprint column added

        Args:
            input_file (Path): name of the input csv file
            output_file (Path): name of the csv file with the output
            error_file (Path): File to which error records are written. Must be specified if error_split_column is used

        """
        if self.success_column is not None:
            if error_file is None:
                raise ValueError(
                    "error file must be specified if success_column is defined"
                )

        df = pd.read_csv(input_file)
        processed_df, failed_df = self.process_dataframe(df)
        processed_df.to_csv(output_file)
        if self.success_column is not None:
            failed_df.to_csv(error_file)
