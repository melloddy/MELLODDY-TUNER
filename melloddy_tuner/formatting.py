from .config import parameters
from .helper import *

from scipy import stats


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


class ActivityDataFormatting(object):
    """
    Class of Functions to format activity data
    """
    def __init__(self, data, mapping_table_T5, mapping_table_T10):
        param = ReadConfig()

        self.min_data_points = param.get_conf_dict(parameters.get_parameters())['count_task']['count_data_points']
        self.data = data
        self.table_T5 = mapping_table_T5
        self.table_T10 = mapping_table_T10
        self.data_merged = pd.DataFrame()
        self.data_mapped = pd.DataFrame()
        self.less_freq_class = None
        self.data_duplicates = pd.DataFrame()
        self.data_dereplicated = pd.DataFrame()
        self.data_processed = pd.DataFrame()
        self.data_remapped = pd.DataFrame()

    def run_formatting(self):
        """
        Run formatting of the given activity data file
        :return: processed activity data as dataframe
        """
        self.data_merged = self.merge_with_table_T5()
        self.data_mapped = self.map_filtered_data()
        self.less_freq_class = self.val_class(self.data_mapped['class_label'].value_counts())
        self.data_duplicates = self.detect_duplicated_ids()
        data_grouped = self.groupby_ids()
        self.data_dereplicated = self.update_class_label(data_grouped)
        self.data_filtered = self.filter_count_labels_per_classification_task()
        data_selected = self.select_size_filtered_data(self.data_filtered, self.data_dereplicated)
        self.data_processed = self.merge_with_table_T10(data_selected)
        return self.data_processed

    def remapping_2_cont_ids(self):
        """
        Map IDs of a given file
        :return:
        """
        data_processed = self.data_processed
        data_remapped = self.map_2_cont_id(data_processed, 'classification_task_id')
        self.data_remapped = self.map_2_cont_id(data_remapped, 'descriptor_vector_id')
        return self.data_remapped

    def make_T11(self, table_T6):
        data_mapped_desc = self.data_remapped[['descriptor_vector_id', 'cont_descriptor_vector_id']]
        T6_filtered = table_T6[table_T6.descriptor_vector_id.isin(data_mapped_desc.descriptor_vector_id)]
        data_mapped_desc.set_index('descriptor_vector_id', inplace=True)
        T6_filtered.set_index('descriptor_vector_id', inplace=True)
        table_T11 = T6_filtered.join(data_mapped_desc, on = 'descriptor_vector_id').drop_duplicates()
        table_T11 = table_T11.reset_index()
        return table_T11

    def map_T3(self, table_T3):
        data_mapped_task = self.data_remapped[['classification_task_id', 'cont_classification_task_id']]
        data_mapped_task.set_index('classification_task_id', inplace=True)
        table_T3.set_index('classification_task_id', inplace=True)
        table_T3_mapped = table_T3.join(data_mapped_task, on='classification_task_id')
        table_T3_mapped = table_T3_mapped.drop_duplicates()
        table_T3_mapped = table_T3_mapped.reset_index()

        return table_T3_mapped

    @staticmethod
    def val_class(class_label_counts):
        """
        Count the overall number of actives and inactives to aggregate conflicting class (assign less populated class)
        :param class_label_counts: number of actives (class_label_counts[1]) and inactives (class_label_counts[0])
        :return: assigned value of the less populated class
        """
        if class_label_counts[0] < class_label_counts[1]:
            val = 0
        else:
            val = 1
        return val

    def merge_with_table_T5(self):
        """
        merge data with mapping table T5 to replace input compound id with descriptor vector id
        :param data: input activity data
        :param mapping_table_T5: contain mapping between input compound id and descriptor vector id
        :return: merged data.
        """
        data = self.data.set_index('input_compound_id')
        mapping_table_T5 = self.table_T5.set_index('input_compound_id')
        data_merged = pd.merge(data, mapping_table_T5, how='left', on='input_compound_id')
        return data_merged

    def filter_failed_structures(self):
        """
        identify entries which failed the structure standardization
        :param data_merged:  a result of the mapping with T5
        :return: failed data entries which did not pass the standardization
        """
        data_merged = self.data_merged
        failed_data = data_merged[pd.isna(data_merged.descriptor_vector_id)]
        failed_data = failed_data.reset_index()
        return failed_data

    def map_filtered_data(self):
        """
        transform dataframe to T7 containing descriptor_vector_id, classification_task_id and class_label and remove NaN entries.
        :param data_merged: data_merged as a result of the mapping with T5
        :return: data mapped: table T7 with descriptor_vector_id, classification_task_id and class labels
        """
        data_mapped = self.data_merged.dropna(subset=['descriptor_vector_id'])
        data_mapped = data_mapped.reset_index()
        data_mapped = data_mapped[['descriptor_vector_id', 'classification_task_id', 'class_label']].astype('int32')
        return data_mapped

    def detect_duplicated_ids(self):
        """
        identify duplicated id pairs
        :param data: table T7
        :return:
        """
        data_duplicates = self.data_mapped[
            self.data_mapped.duplicated(subset=['descriptor_vector_id', 'classification_task_id'], keep=False)]
        data_duplicates = data_duplicates.reset_index()
        return data_duplicates


    @staticmethod
    def modes(df, key_cols, value_col, count_col):
        '''
        Pandas does not provide a `mode` aggregation function
        for its `GroupBy` objects. This function is meant to fill
        that gap, though the semantics are not exactly the same.
        The input is a DataFrame with the columns `key_cols`
        that you would like to group on, and the column
        `value_col` for which you would like to obtain the modes.
        The output is a DataFrame with a record per group that has at least
        one mode (null values are not counted). The `key_cols` are included as
        columns, `value_col` contains lists indicating the modes for each group,
        and `count_col` indicates how many times each mode appeared in its group.
        '''
        return df.groupby(key_cols + [value_col]).size() \
            .to_frame(count_col).reset_index() \
            .groupby(key_cols + [count_col])[value_col].unique() \
            .to_frame().reset_index() \
            .sort_values(count_col, ascending=False) \
            .drop_duplicates(subset=key_cols)


    def groupby_ids(self):
        """
        group by classification_task_id and descriptor_vector_id and aggregate class labels as pandas Series.
        replace conflicting aggregations (equal amount of actives and inactives) by the less frequent class label.
        :param data: T7 table
        :param less_freq_class: overall less frequent class label
        :return: grouped data with aggregated  class labels
        """

        group_cols = ['classification_task_id', 'descriptor_vector_id']
        label_col = 'class_label'
        label_count = 'label_count'
        
        data_replicates_grouped = self.data_duplicates # if no duplicates --> is empty
        if self.data_duplicates.shape[0]>0:
            # modes breaks if self.data_duplicate is empty
            data_replicates_grouped = self.modes(self.data_duplicates, group_cols, label_col, label_count)
        data_replicates_grouped['count_unique'] = data_replicates_grouped.class_label.apply(lambda x: x.size)
        ind_mask = data_replicates_grouped[data_replicates_grouped.count_unique != 2].index
        data_replicates_grouped['class_label'] = np.where(data_replicates_grouped.index.isin(ind_mask), data_replicates_grouped['class_label'].apply(lambda x: x[0]), int(self.less_freq_class))
        return data_replicates_grouped[['classification_task_id', 'descriptor_vector_id', 'class_label']]



    def update_class_label(self, data_replicates):
        data_replicates.set_index(['classification_task_id', 'descriptor_vector_id'], inplace=True)
        self.data_mapped.set_index(['classification_task_id', 'descriptor_vector_id'], inplace=True)
        self.data_mapped.update(data_replicates)
        self.data_mapped = self.data_mapped.reset_index()
        self.data_dereplicated = self.data_mapped.drop_duplicates()
        self.data_dereplicated = self.data_dereplicated.astype('int64')
        self.data_dereplicated['class_label'] = self.data_dereplicated.class_label.replace(to_replace=0, value=-1)
        return self.data_dereplicated

    def filter_count_labels_per_classification_task(self):
        """
        count the class labels per classification_task_id and keep these ids with 25 or more labels
        :param data_grouped: Grouped data with aggregated class labels
        :return: data_size_filtered: table T8 with classification_task_ids with more than 25 actives and 25 inactives
        """
        count_threshold = self.min_data_points
        data_grouped_label_counts = self.data_dereplicated.class_label.groupby([self.data_dereplicated['classification_task_id'], self.data_dereplicated['class_label']]).size().to_frame(
            'label_counts').unstack()
        data_size_filtered = data_grouped_label_counts[data_grouped_label_counts['label_counts'] >= count_threshold]
        return data_size_filtered


    def select_excluded_data(self):
        """
        select and save excluded entries which do not fulfil the required amount of class labels
        :param data_size_filtered: entries with 25 or more labels
        :param data_grouped: initial grouped data table
        :return: data_excluded_out: excluded data entries
        """
        data_excluded = self.data_filtered[self.data_filtered.isna().any(axis=1)]
        data_excluded = data_excluded.reset_index()
        data_excluded_out = self.data_dereplicated[self.data_dereplicated.classification_task_id.isin(data_excluded.classification_task_id)]
        return data_excluded_out

    @staticmethod
    def select_size_filtered_data(data_size_filtered, data_grouped):
        """
        save filtered data as table T8
        :param data_size_filtered: data table with classfication task ids with more than 25 labels each.
        :param data_grouped: grouped data table T7
        :return: output data table
        """
        data_filtered = data_size_filtered.dropna().astype('int32')
        data_filtered_stacked = data_filtered.stack().reset_index()
        data_filtered_out = data_grouped[
            data_grouped.classification_task_id.isin(data_filtered_stacked.classification_task_id)]
        return data_filtered_out

    def merge_with_table_T10(self,data):
        """
        join activity data and structure data based on descriptor_vector_id and add fold_id to activity table
        :param data: data table T8
        :param mapping_table_T10: table T10
        :return: merged table T11
        """
        data = data.set_index('descriptor_vector_id')
        mapping_table_T10 = self.table_T10.set_index('descriptor_vector_id')
        data_merged = pd.merge(data, mapping_table_T10, how='left', on='descriptor_vector_id')
        data_merged = data_merged.reset_index()
        return data_merged

    @staticmethod
    def map_2_cont_id(data, colname):
        map_id = {val: ind for ind, val in enumerate(np.unique(data[colname]))}
        map_id_df = pd.DataFrame.from_dict(map_id, orient='index').reset_index()
        map_id_df = map_id_df.rename(columns={'index': colname, 0: 'cont_' + colname})
        data_remapped = pd.merge(data, map_id_df, how='inner', on=colname)
        return data_remapped

    @staticmethod
    def count_labels_per_fold(data_final):
        """
        counting labels per classification task and fold
        :param data_final: data table T11
        :return: T11 with counts
        """
        data_final = data_final[['cont_classification_task_id', 'classification_task_id', 'fold_id', 'class_label']]
        data_final_label_counts = data_final.groupby(
            [data_final['fold_id'],
             data_final['class_label'], data_final['cont_classification_task_id']]).size().to_frame(
            'label_counts').unstack()
        data_final_label_counts = data_final_label_counts.fillna(0).astype('int32').stack().reset_index()
        return data_final_label_counts
