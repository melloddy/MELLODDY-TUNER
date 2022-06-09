import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import scipy.sparse

# functionalities: 
# - from HTS %activity, calculate the r-score
# - creates a T0 for HTS assays
# - from r-score, call for frequent hitters

# Instructions: 
# This script can be used to prepare HTS auxiliary data, it can also be used to identify frequent hitters based on HTS assays and the binomical survival function
# No HTS data will be filtered, the output should contain the full HTS data with activity converted into r-score
# Since the HTS data requires a rather large amount of memory to store data frames and execute operations on them, this script should be allocated appropriate memory (potentially > 60Gb)


# Required argument: 
# --aux_file: T1 exclusive to hts auxiliary data (three columns required: input_compound_id, input_assay_id, activity)
#             This file should have been prepared according to the data preparation manual. 
#             The activity values are assumed to be numeric (float)
#             The script assumes aggregated replicate compound-assay pairs was performed by you before hand, according to your standards
#             Compound-assay pair replicates will be detected and the script will not complete, leaving a log file allowing you to identify the replicates
#
# Optional arguements
# --output          : file basename for outputted HTS data (T1) and assay metadata (T0)
# -l                : logfile, if compound-assay replicate or missing activity values are detected, this file will contain the identified replicates
# --frequent_hitters: is a switch to turn on the detection of frequent hitters based on present HTS assays and binomial survival function (default: turned off)
# -f                : frequent hitters logfile, will list the compounds detected to be frequent hitters (p-value < -12), you will have to remove them from your HTS data if desired
# -v                : prints out some counts (data volume, number of data points, number of compounds, number of HTS assays)

parser = argparse.ArgumentParser(description="Converts HTS activity values into r-scores (and can flag frequent hitters based on binomial survival function)")
parser.add_argument("-a", "--aux_file", type=str, help="Path to the HTS activity file (3 columns: input_compound_id, input_assay_id, activiy, where activity is %activity)", required=True)
parser.add_argument("-o", "--output", type=str, help="Path to the output file (T1_ and T0_ will be prefixed to file basename)", default="hts_data_rscores.csv")
parser.add_argument("-l", "--logfile", type=str, help="Path to the log file containing problematic data points (missing or replicates)", default="log.csv", required=False)
parser.add_argument("--frequent_hitters", action="store_true", help="Turn on frequent hitters detection", default=False, required=False)
parser.add_argument("-f", "--frequent_hitters_logfile", type=str , help="Path to the log file containing frequent hitters p-values", default="log_HTS_frequent_hitters.csv", required=False)
parser.add_argument("-v", "--verbose", action="store_true", help="verbosity", default=False)
args = parser.parse_args()




def check_dataset(df, logfilename):
    """From the HTS activity data frame, finds out if there are some replicates or missing values
#     :param pandas df: data frame containing single-dose data (3 columns: input_compound_id, input_assay_id, activiy)
#     :param str logfilename: filename in which the data of possible replicates will be saved. 
#     :return bool has_replicate, if True, there are some replicates, a log file is saved containing the data relative to replicates
    """
    has_issues = False
    
    #1/ check for replicates
    mask_dup = df.duplicated(subset=['input_compound_id', 'input_assay_id'], keep=False)
    df_dup = df[mask_dup]
    
    if df_dup.shape[0] > 0:    
        print(f'# ERROR : found replicated compound-assay pairs in provided HTS data. Please aggregate these before hand.')
        print(f'# Saving replicate compound-assay records in {logfilename}')
        has_issues = True
    
    #2/ check for missing values
    df_missing = df.loc[df['activity'].isna()]
    if df_missing.shape[0] > 0 : 
        print(f'# ERROR : found {df_missing.shape[0]} missing activity values in proivded HTS data. Please filter these before hand.')
        print(f'# Saving missing activity data records in {logfilename}')
    
    df_problems = pd.concat([df_dup, df_missing], ignore_index=True)
    df_problems.to_csv(logfilename, index=None)
    
    return has_issues


def calc_rscores(df, rscore_colname="standard_value", assay_colname="input_assay_id", activity_colname="activity"):
    """ from raw values of single-dose assays, calculate r-scores per assay.
#     :param pandas df: data frame containing single-dose data (3 columns: input_compound_id, input_assay_id, activity)
#     :param str rscore_colname: name of column containing rscore values (default=standard_value)
#     :param str assay_colname: name of column containing the assay identifier (default=input_assay_id)
#     :param str activity_colname: name of column containing the %activity values (default=activity)
#     :return pandas df_out --> input df with rscore column
    """
    # set the r-score column using %activity.

    # R-scores are defined based on each assay following the formula:
    # r-score = ( x - median(X) ) / median_absolute_deviation(X)

    assert rscore_colname not in df.columns, f"auxiliary data must not contain {rscore_colname} column before calculating rscore"
    assert assay_colname in df.columns, f"auxiliary data must contain {assay_colname}"
    assert activity_colname in df.columns, f"auxiliary data must contain {activity_colname}"

    df_out = df.copy()

    #df_out[rscore_colname] = df.groupby([assay_colname])[activity_colname].apply(lambda x: (x - np.median(x)) / stats.median_absolute_deviation(x)).round(4)
    df_out[rscore_colname] = df.groupby([assay_colname])[activity_colname].apply(lambda x: (x - np.median(x)) / stats.median_abs_deviation(x, scale="normal")).round(4)
    
    return df_out




def create_hts_matrix(df, rscore_cutoff=3, rscore_colname="standard_value", assay_colname="input_assay_id", compound_colname='input_compound_id'):
    """ from the calculated rscores, determine frequent hitter compounds.
#     :param pandas df: data frame containing single-dose rscores (requires 3 columns: input_compound_id, input_assay_id, standard_value), where standard_value is the calculated rscore
#     :param str rscore_colname: name of column containing rscore values (default=rscore)
#     :param str assay_colname: name of column containing the assay identifier (default=input_assay_id)
#     :param str compound_colname: name of column containing the compound identifier (default=input_compound_id)
#     :return1 pandas df_out --> input df with frequent hitters flag column
#     :return2 dict row_indices_reverse --> maps row matrix indices back to compounds identifiers
    """
    
    df_out = df.copy()

    # define column and row indices for assays and compounds respectively: 
    compound_list = df_out[compound_colname].unique()
    assay_list  = df_out[assay_colname].unique()

    col_indices = {a:idx for idx,a in enumerate(assay_list)}
    row_indices = {s:idx for idx,s in enumerate(compound_list)}

    # back-map to compounds
    row_indices_reverse = {v:k for (k,v) in row_indices.items()}

    df_out['row_indices'] = df_out[compound_colname].map(row_indices)
    df_out['col_indices'] = df_out[assay_colname].map(col_indices)


    # set positive/negative labels : positive = abs(rscore) > rscore_cutoff
    values = np.where(abs(df_out[rscore_colname].values)>rscore_cutoff, 1, -1)

    # put these into a scipy.csc matrix
    Y_hts = scipy.sparse.csr_matrix((values, (df_out['row_indices'], df_out['col_indices'])), shape=(len(row_indices), len(col_indices)))
    
    return Y_hts, row_indices_reverse





def calc_median_hit_rate(Y):
    """ from the HTS compound assay matrix, determine the median hit rate with an rscore cutoff = 3.
#     :param pandas df: data frame containing single-dose rscores (requires 3 columns: input_compound_id, input_assay_id, rscore)
#     :param Y scipy.sparse.csr_matrix: HTS compound (rows) assay (columns) matrix 
#     :return float median hit rate over all compounds and all HTS assays
    """

    n_pos = np.array((Y>0).sum(0))[0]
    n_neg = np.array((Y<0).sum(0))[0]

    median_hit_rate = np.median(n_pos/(n_neg+n_pos))

    return median_hit_rate




def calc_logsf(Y, median_hit_rate, cmpd_map, logsf_cutoff=-12.):
    """ from the HTS binary compound assay matrix, compute the binomial log survival function.
#     :param Y scipy.sparse.csr_matrix: HTS compound (rows) assay (columns) matrix 
#     :param median_hit_rate float: median hit rate over all compounds and all HTS assays
#     :param dict cmpd_map: dictionary mapping row indices in Y to input_compound_id
#     :param float logsf_cutoff: p-value cutoff below which a compound is called a frequent hitter
#     :return np.array logsf containing the logBSF of all compounds in order of rows in Y
    """
    # for each compound: 
    #  - count the number of assays screened
    #  - count the number of times it was a hit
    # then use these counts to compute the p-val of a compound to be a hit according to the binomial log survival function
    
    n_hits = np.array((Y>0).sum(1)).flatten()   # numbers of HTS assays in which compound is a hit (according to rscore cutoff used)
    n_misses = np.array((Y<0).sum(1)).flatten() # numbers of HTS assays in which compound is not a hit (according to rscore cutoff used)
    n_screens = n_hits + n_misses               # numbers of HTS assays in which the compounds is screnned 

    # compute the log survival function 
    # (n_hits - 1) => logsf for n_hits and more
    logsf = scipy.stats.binom.logsf(k=n_hits - 1,n=n_screens, p=median_hit_rate) 
    
    
    # then integrate the logsf p-val in the aux_data matrix
    logsf_df = pd.DataFrame({'logsf':logsf}).reset_index()
    logsf_df['input_compound_id'] = logsf_df['index'].map(cmpd_map)
    
    logsf_df['frequent_hitter'] = np.where(logsf_df.logsf<logsf_cutoff, True,False) # True => is FH

    return logsf_df.drop('index', axis=1) 



def create_hts_assay_metadata(df):
    """ From the auxiliary data, create a T0 file, HTS assay metadata
    """
    
    colnames = ["input_assay_id",
                "assay_type",
                "use_in_regression",
                "is_binary",
                "expert_threshold_1",
                "expert_threshold_2",
                "expert_threshold_3",
                "expert_threshold_4",
                "expert_threshold_5",
                "direction",
                "catalog_assay_id",
                "parent_assay_id"]
    
    list_assays = df['input_assay_id'].unique()
    
    t0 = pd.DataFrame({'input_assay_id':list_assays})
    t0['assay_type'] = 'AUX_HTS'
    t0['use_in_regression'] = False
    t0['is_binary'] = False
    t0['catalog_assay_id'] = np.nan
    t0['parent_assay_id'] = np.nan    
    
    for col in colnames[3:]: t0[col] = None
    
    return t0



def main():
    args = parser.parse_args()

    if args.verbose:
        for k in args.__dict__:
            print(f"{k:>30} : {args.__dict__[k]}")
        print("\n")

    assert os.path.isfile(args.aux_file),f"Cannot find auxiliary data file at {args.aux_file}"
    
    # load data
    aux_data = pd.read_csv(args.aux_file)
    assert 'input_assay_id' in aux_data.columns, f"Did not find 'input_assay_id' column in HTS auxiliary data file: {args.aux_file}"
    assert 'activity' in aux_data.columns, f"Did not find 'activity' column in auxiliary data file: {args.aux_file}"
    assert 'input_compound_id' in aux_data.columns, f"Did not find 'input_compound_id' column in auxiliary data file: {args.aux_file}"

    if args.verbose: 
        print(f"Number of records: {aux_data.shape[0]}")
        print(f"Number of assays : {aux_data.input_assay_id.unique().shape[0]}")

        
    # check for replicates and for missing activity values, stop if there are some replicate found
    has_issues = check_dataset(aux_data, args.logfile)
    
    if has_issues:
        print("# Exiting...")
        return
        
        
    # compute rscores
    aux_data_rscores = calc_rscores(aux_data, 
                                    rscore_colname="standard_value", 
                                    assay_colname="input_assay_id", 
                                    activity_colname="activity")
    del aux_data

    
    # filter infinite rscore values
    inf_records = aux_data_rscores.loc[(aux_data_rscores["standard_value"] > 1E308)|
                                       (aux_data_rscores["standard_value"] < -1E308)]
    
    if inf_records.shape[0]>0:
        print(f"warning, {inf_records.shape[0]} infinite r-score values found. Will be dropped")
        inf_records_filename = os.path.join(os.path.dirname(args.output), 'T1_inifite_rscores_'+os.path.basename(args.output)) 
        inf_records.to_csv(inf_records_filename, index=None)
        
        aux_data_rscores = aux_data_rscores.loc[(aux_data_rscores["standard_value"] < 1E308)&
                                                (aux_data_rscores["standard_value"] > -1E308)]
    
    
    # filter missing rscore values
    miss_records = aux_data_rscores.loc[aux_data_rscores["standard_value"].isna()]
    
    if miss_records.shape[0]>0:
        print(f"warning, {miss_records.shape[0]} missing r-scores values found. Will be dropped")
        miss_records_filename = os.path.join(os.path.dirname(args.output), 'T1_missing_rscores_'+os.path.basename(args.output)) 
        miss_records_filename.to_csv(miss_records_filename, index=None)
        
        aux_data_rscores = aux_data_rscores.loc[~aux_data_rscores["standard_value"].isna()]
    
    
    aux_data_rscores.drop('activity', axis=1, inplace=True)
    aux_data_rscores['standard_qualifier'] = '='
    
    # save HTS data expressed as rscore
    t1_filename = os.path.join(os.path.dirname(args.output), 'T1_'+os.path.basename(args.output)) 
    aux_data_rscores.to_csv(t1_filename, index=None)
    if args.verbose: 
        print(f"Saved: {t1_filename}")
    
    # create the HTS task T0
    t0_filename = os.path.join(os.path.dirname(args.output), 'T0_'+os.path.basename(args.output)) 
    t0_hts = create_hts_assay_metadata(aux_data_rscores)
    t0_hts.to_csv(t0_filename, index=None)
    if args.verbose: 
        print(f"Saved: {t0_filename}")
    
    
    # flag frequent hitters
    if args.frequent_hitters: 

        # 1/create a csr matrix with compounds as rows and HTS assays as columns
        if args.verbose:
            print("\nDetermine frequent hitter compounds")

        # POSSIBLE ISSUES TO TEST FOR: 
        # - duplicate compound identifiers might create an error

        # OPEN QUESTIONS: 
        # - should the rscore cutoff used for frequent hitters determination be an argument in command line ? 
        # - how should this handle duplicate (input_compound_id, inpu_assay_id) pairs ?
        # - the frequent hitters will be determined based on any HTS assay in the inpt datafram, should we drop HTS assays with less than 100,000 input_compound_ids ?
        Y_hts, cmpd_map = create_hts_matrix(aux_data_rscores, 
                                            rscore_cutoff=3, 
                                            rscore_colname="standard_value", 
                                            assay_colname="input_assay_id", 
                                            compound_colname='input_compound_id')



        # 2/ Determine the median hit rate over all the assays, this will then be the base probability for the binomial test. 
        median_hit_rate = calc_median_hit_rate(Y_hts)
    
        # 3/ for each compound compute log binomial survival p-values
        logsf_df = calc_logsf(Y_hts, 
                              median_hit_rate, 
                              cmpd_map,
                              logsf_cutoff=-12)


        # 4/ list the frequent hitters and save them in a log file
        frequent_hitters = logsf_df[logsf_df['frequent_hitter']]
        
        
        if frequent_hitters.shape[0] > 0: 
            frequent_hitters.to_csv(args.frequent_hitters_logfile, index=None)

            if args.verbose:
                print(f"Number of frequent hitters found: {frequent_hitters.shape[0]}")
                print(f"Frequent hitters saved in {args.frequent_hitters_logfile}")
    
        elif args.verbose: 
            print ("No frequent hitters found")
    return
    
if __name__ == "__main__":
    main()
