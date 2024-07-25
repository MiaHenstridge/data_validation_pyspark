# monitoring class


import pyspark.sql.types as T
from datetime import timedelta, datetime
import numpy as np
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql import Window, Row
from pyspark.sql import DataFrame as PySparkDataFrame
import pyspark.sql.functions as F
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score
from pyspark.sql.window import Window
import scipy.stats as stats

from typing import Union, List, Optional, Any, Tuple, Dict

from functools import wraps



class ModelMonitoring:
    def __init__(self):
        pass

    # udf to assign value range
    @staticmethod
    def _get_range(value, cutoffs) -> str:
        if not value:
            return None
        for i in range(len(cutoffs)-1):
            if cutoffs[i] < value <= cutoffs[i+1]:
                return f'({cutoffs[i]}, {cutoffs[i+1]}]'
        return None


    def get_range(self, df: PySparkDataFrame, cutoffs: Union[List[int], List[float]], values: Optional[str]='score', output_col: Optional[str]='range') -> Tuple[str, PySparkDataFrame]:
        """
        Function for binning continuous values based on given cutoffs

        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        values (str, optional): column containing score
        cutoffs (Union[List[int], List[float]]): list of cutoffs sorted in ascending order
        """
        cutoffs = sorted(cutoffs)
        get_range_udf = F.udf(lambda value: self._get_range(value, cutoffs))
        df = df.withColumn(output_col, get_range_udf(F.col(values)))
        return output_col, df


    def _get_gb(self, df, gb_label='GB', include_intermediate=False, intermediate_as_good=True):
        if include_intermediate:
            if intermediate_as_good:
                df = df.withColumn(gb_label, F.when(F.col(gb_label) == 2, 0).otherwise(F.col(gb_label)))
            else:
                df = df.withColumn(gb_label, F.when(F.col(gb_label) == 2, 1).otherwise(F.col(gb_label)))

        # if intermediate not included
        df = df.where(F.col(gb_label) < 2)
        return df

    # udf to calculate gini
    @staticmethod
    @F.pandas_udf(T.DoubleType(), functionType=F.PandasUDFType.GROUPED_AGG)
    def _gini_udf(true, pred):
        try:
            return abs(2*roc_auc_score(true, pred)-1)
        except:
            return 0


    def get_gini(self, df: PySparkDataFrame, true_label: Optional[str]='GB', prediction: Optional[str]='score', output_col: Optional[str]='gini', 
        include_intermediate: Optional[bool]=False, intermediate_as_good: Optional[bool]=True,
        aggregate_by: Optional[Union[str, List[str]]]=None) -> PySparkDataFrame:
        """
        Function to calculate gini from true label and prediction

        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        true_label(str, optional): column containing true label (0: good, 1: bad)
        prediction(str, optional): column containing predictions (probability or score)
        output_col(str, optional): column name to store result
        include_intermediate(boolean, optional): whether to include intermediate (2) in calculations, default to False
        intermediate_as_good(boolean, optional): if True, consider intermediates as good. Otherwise, consider as bad. Defaults to True
        aggregate_by(str, List(str), optional): column(s) to aggregate results by
        """
        gini_data = self._get_gb(df=df, gb_label=true_label, include_intermediate=include_intermediate, intermediate_as_good=intermediate_as_good)

        gini = gini_data.groupBy(aggregate_by)\
                .agg(self._gini_udf(F.col(true_label), F.col(prediction)).alias(output_col))

        return gini


    def get_ks_label(self, df: PySparkDataFrame, score: Optional[str]=None, cutoffs: Optional[Union[List[int], List[float]]]=None, 
        score_range: Optional[str]=None, true_label: Optional[str]='GB', aggregate_by: Optional[Union[str, List[str]]]=None) -> PySparkDataFrame:
        """
        Function to calculate KS statistics between good and bad labels over score range

        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        true_label(str, optional): column containing true label (0: good, 1: bad)
        score (str, optional): column containing scores. Has to be specified if score_range is None
        cutoffs (Union[List[int], List[float]], optional): list of cutoffs to extract score range. Has to be specified if score_range is None.
        score_range (str, optional): column containing score range. Has to be specified if score and/or cutoffs is None
        aggregate_by(str, List(str), optional): column(s) to aggregate results by
        """
        # Check input
        if not score and not score_range:
            raise ValueError("Either 'score' or 'score_range' column has to be specified. If 'score' is specified, please also provide cutoffs.")
        if score and not cutoffs:
            raise ValueError("Please provide cutoffs")

        # Extract score range if score_range not provided as input
        if not score_range:
            score_range, df = self.get_range(df, score, cutoffs)

        
        if not isinstance(aggregate_by, list):
            aggregate_by = [aggregate_by]
        # aggregate_by  = aggregate_by + [score_range]

        w1 = Window.partitionBy(aggregate_by).orderBy(score_range)
        w2 = Window.partitionBy(aggregate_by)

        # get KS
        ks = df.groupBy(aggregate_by + [score_range])\
                .agg(F.count(F.when(F.col(true_label)==1, True)).alias('event_count'),
                    F.count(F.when(F.col(true_label)==0, True)).alias('non_event_count'))\
                .withColumn('event_cumsum', F.sum('event_count').over(w1))\
                .withColumn('non_event_cumsum', F.sum('non_event_count').over(w1))\
                .withColumn('event_total', F.sum('event_count').over(w2))\
                .withColumn('non_event_total', F.sum('non_event_count').over(w2))\
                .withColumn('event_cumperc', F.col('event_cumsum') / F.col('event_total'))\
                .withColumn('non_event_cumperc', F.col('non_event_cumsum') / F.col('non_event_total'))\
                .withColumn('ks', F.max(F.abs(F.col('event_cumperc') - F.col('non_event_cumperc'))).over(w2))\
                .select(*aggregate_by, score_range, 'event_cumperc', 'non_event_cumperc', 'ks')
        return ks


    def get_gain_lift(self, df: PySparkDataFrame, score: Optional[str]=None, cutoffs: Optional[Union[List[int], List[float]]]=None, 
        score_range: Optional[str]=None, true_label: Optional[str]='GB', aggregate_by: Optional[Union[str, List[str]]]=None) -> PySparkDataFrame:
        """
        Function to calculate gain and lift over score range

        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        true_label(str, optional): column containing true label (0: good, 1: bad)
        score (str, optional): column containing scores. Has to be specified if score_range is None
        cutoffs (Union[List[int], List[float]], optional): list of cutoffs to extract score range. Has to be specified if score_range is None.
        score_range (str, optional): column containing score range. Has to be specified if score and/or cutoffs is None
        aggregate_by(str, List(str), optional): column(s) to aggregate results by
        """
        # Check input
        if not score and not score_range:
            raise ValueError("Either 'score' or 'score_range' column has to be specified. If 'score' is specified, please also provide cutoffs.")
        if score and not cutoffs:
            raise ValueError("Please provide cutoffs")

        # Extract score range if score_range not provided as input
        if not score_range:
            score_range, df = self.get_range(df, score, cutoffs)

        if not isinstance(aggregate_by, list):
            aggregate_by = [aggregate_by]


        # create gain and lift table

        w1 = Window.partitionBy(aggregate_by).orderBy(score_range)
        w2 = Window.partitionBy(aggregate_by)

        gl = df.groupBy(aggregate_by + [score_range])\
                .agg(F.count(F.when(F.col(true_label)==1, True)).alias('event_count'),
                    F.count(score_range).alias('count'))\
                .withColumn('event_cumsum', F.sum('event_count').over(w1))\
                .withColumn('count_cumsum', F.sum('count').over(w1))\
                .withColumn('event_total', F.sum('event_count').over(w2))\
                .withColumn('count_total', F.sum('count').over(w2))\
                .withColumn('event_cumperc', F.col('event_cumsum') / F.col('event_total'))\
                .withColumn('count_cumperc', F.col('count_cumsum') / F.col('count_total'))\
                .withColumn('gain', F.col('event_cumperc')*F.lit(100))\
                .withColumn('lift', F.col('gain') / (F.col('count_cumperc') * 100))\
                .select(*aggregate_by, score_range, 'gain', 'lift')

        return gl


    @staticmethod
    def _create_drift_udf(base_dist, range_col, drift_measure):
        @F.pandas_udf(T.FloatType(), functionType=F.PandasUDFType.GROUPED_AGG)
        def _drift_udf(range_values):
            range_values = pd.DataFrame(range_values.tolist(), columns=[range_col])

            if drift_measure != 'chi_square':
                observed_df = range_values[range_col].value_counts(normalize=True).reset_index()
                observed_df.columns = [range_col, 'observed_dist']
            else:
                observed_df = range_values[range_col].value_counts(normalize=False).reset_index()
                observed_df.columns = [range_col, 'observed_freq']
                total_obs = observed_df['observed_freq'].sum()

            base_df = pd.DataFrame(list(base_dist.items()), columns=[range_col, 'base_dist'])

            if observed_df[range_col].nunique() != base_df[range_col].nunique():
                return None

            dist = observed_df.merge(base_df, on=range_col, how='left').sort_values(range_col)
            drift = 0.0

            if drift_measure == 'chi_square':
                dist['base_freq'] = dist['base_dist'] * total_obs
                dist = list(dist[['observed_freq', 'base_freq']].itertuples(index=False, name=None))
                drift = sum((obs - base)**2 / base for obs, base in dist)
            
            else:
                if drift_measure in ['psi', 'kullback_leibler', 'jensen_shannon']:
                    epsilon = 1e-10
                    dist['observed_dist'] += epsilon
                    dist['base_dist'] += epsilon

                if drift_measure == 'psi':
                    dist = list(dist[['observed_dist', 'base_dist']].itertuples(index=False, name=None))
                    drift = sum((obs - base) * np.log(obs / base) for obs, base in dist)

                elif drift_measure == 'kullback_leibler':
                    dist = list(dist[['observed_dist', 'base_dist']].itertuples(index=False, name=None))
                    drift = sum(obs * np.log(obs / base) for obs, base in dist)

                elif drift_measure == 'jensen_shannon':
                    dist['mean_dist'] = (dist['observed_dist'] + dist['base_dist']) / 2
                    dist = list(dist[['observed_dist', 'base_dist', 'mean_dist']].itertuples(index=False, name=None))
                    kl_base = sum(base * np.log(base / mean) for _, base, mean in dist)
                    kl_observed = sum(obs * np.log(obs / mean) for obs, _, mean in dist)
                    drift = (kl_observed + kl_base) / 2

                elif drift_measure == 'hellinger':
                    dist = list(dist[['observed_dist', 'base_dist']].itertuples(index=False, name=None))
                    drift = np.sqrt(sum((np.sqrt(obs) - np.sqrt(base))**2 for obs, base in dist)) / np.sqrt(2)

                elif drift_measure == 'wasserstein':
                    dist['observed_cdf'] = dist['observed_dist'].cumsum()
                    dist['base_cdf'] = dist['base_dist'].cumsum()
                    drift = abs(dist['observed_cdf'] - dist['base_cdf']).sum()

                elif drift_measure == 'kolmogorov_smirnov':
                    dist['observed_cdf'] = dist['observed_dist'].cumsum()
                    dist['base_cdf'] = dist['base_dist'].cumsum()
                    drift = abs(dist['observed_cdf'] - dist['base_cdf']).max()

            return drift

        return _drift_udf


    def get_drift_distance(self, df: PySparkDataFrame, base_distribution: Dict[str, float], column: Optional[str]=None, cutoffs: Optional[Union[List[int], List[float]]]=None,
        range_column: Optional[str]=None, drift_measure: Optional[str]='psi', output_col: Optional[str]='distance', aggregate_by: Optional[Union[str, List[str]]]=None) -> PySparkDataFrame:
        """
        Function to calculate drift distance based on expected distribution and drift measure

        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        column (str, optional): column containing values. Has to be specified if range_column is None
        cutoffs (Union[List[int], List[float]], optional): list of cutoffs to extract range. Has to be specified if range_column is None.
        range_column (str, optional): column containing range. Has to be specified if column and/or cutoffs is None
        base_distribution(Dict[str, float]): dictionary containing base distribution by range of format: {bin: distribution}
        aggregate_by(str, List(str), optional): column(s) to aggregate results by
        drift_measure (str, optional): drift measure to calculate drift. Can be among the following: ['psi', 'kullback_leibler', 'hellinger', 'kolmogorov_smirnov', 'wasserstein', 'jensen_shannon', 'chi_square']. Defaults to psi.

        """
        # Check input
        if not column and not range_column:
            raise ValueError("Either 'column' or 'range_column' column has to be specified. If 'column' is specified, please also provide cutoffs.")
        if column and not cutoffs:
            raise ValueError("Please provide cutoffs")

        if drift_measure not in ['psi', 'kullback_leibler', 'hellinger', 'kolmogorov_smirnov', 'wasserstein', 'jensen_shannon', 'chi_square']:
            raise ValueError("'drift_measure' must be in ['psi', 'kullback_leibler', 'hellinger', 'kolmogorov_smirnov', 'wasserstein', 'jensen_shannon', 'chi_square']")

        # Extract score range if score_range not provided as input
        if not range_column:
            range_column, df = self.get_range(df, column, range_column)

        drift_udf = self._create_drift_udf(base_distribution, range_column, drift_measure)

        drift = df.groupBy(aggregate_by)\
                .agg(drift_udf(F.col(range_column)).alias(output_col))

        if drift_measure == 'chi_square':
            dof = df.groupBy(aggregate_by)\
                .agg((F.countDistinct(range_column) - F.lit(1)).alias('dof'))

            drift = drift.join(dof, aggregate_by, 'left')

            # Define UDF to calculate p-value
            def calculate_p_value(chisq, dof):
                if chisq is None or dof is None:
                    return None
                return float(stats.chi2.sf(chisq, dof))

            # Register the UDF
            calculate_p_value_udf = F.udf(calculate_p_value, T.DoubleType())

            drift = drift.withColumn("p_value", calculate_p_value_udf(F.col(output_col), F.col("dof")))

        return drift


    def get_score_distribution(self, df: PySparkDataFrame, score: Optional[str]=None, cutoffs: Optional[Union[List[int], List[float]]]=None, 
        score_range: Optional[str]=None, aggregate_by: Optional[Union[str, List[str]]]=None) -> PySparkDataFrame:
        """
        Function to calculate score distribution of score ranges
        
        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        score (str, optional): column containing scores. Has to be specified if score_range is None
        cutoffs (Union[List[int], List[float]], optional): list of cutoffs to extract score range. Has to be specified if score_range is None.
        score_range (str, optional): column containing score range. Has to be specified if score and/or cutoffs is None
        aggregate_by(str, List(str), optional): column(s) to aggregate results by
        """
        # Check input
        if not score and not score_range:
            raise ValueError("Either 'score' or 'score_range' column has to be specified. If 'score' is specified, please also provide cutoffs.")
        if score and not cutoffs:
            raise ValueError("Please provide cutoffs")

        # Extract score range if score_range not provided as input
        if not score_range:
            score_range, df = self.get_range(df, score, cutoffs)

        if not isinstance(aggregate_by, list):
            aggregate_by = [aggregate_by]

        # get score distribution
        w = Window.partitionBy(aggregate_by)

        sd = df.groupBy(aggregate_by + [score_range])\
                .agg(F.count(score_range).alias('count'))\
                .withColumn('total', F.sum('count').over(w))\
                .withColumn('distribution', F.col('count') / F.col('total'))\
                .select(*aggregate_by, score_range, 'distribution')

        return sd


    def get_average_score(self, df: PySparkDataFrame, score: Optional[str]='score', aggregate_by: Optional[Union[str, List[str]]]=None, output_col: Optional[str]='average_score') -> PySparkDataFrame:
        """
        Function to get average score

        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        score (str, optional): column containing scores. Has to be specified if score_range is None
        aggregate_by(str, List(str), optional): column(s) to aggregate results by
        output_col (str, optional): name for output column
        """
        avg = df.groupBy(aggregate_by)\
                .agg(F.avg(score).alias(output_col))

        return avg


    def get_bad_rate(self, df: PySparkDataFrame, true_label: Optional[str]='GB', aggregate_by: Optional[Union[str, List[str]]]=None, output_col: Optional[str]='bad_rate') -> PySparkDataFrame:
        """
        Function to get bad rate

        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        true_label (str, optional): column containing good/bad labels
        aggregate_by(str, List(str), optional): column(s) to aggregate results by
        output_col (str, optional): name for output column
        """
        bad = df.groupBy(aggregate_by)\
                .agg(F.count(F.when(F.col(true_label)==1, True)).alias('bad_count'),
                    F.count('*').alias('total_count'))\
                .withColumn(output_col, F.col('bad_count') / F.col('total_count'))

        return bad


    def get_approval_rate(self, df: PySparkDataFrame, approval_label: Optional[str]='is_approved', aggregate_by: Optional[Union[str, List[str]]]=None, output_col: Optional[str]='approval_rate') -> PySparkDataFrame:
        """
        Function to get approval rate

        Parameters:
        df (PySparkDataFrame): dataframe containing true labels and predictions for calculating gini
        approval_label (str, optional): column containing approval labels (1: approved, 0: not approved)
        aggregate_by(str, List(str), optional): column(s) to aggregate results by
        output_col (str, optional): name for output column
        """
        approval = df.groupBy(aggregate_by)\
                .agg(F.count(F.when(F.col(approval_label)==1, True)).alias('approval_count'),
                    F.count('*').alias('total_count'))\
                .withColumn(output_col, F.col('approval_count') / F.col('total_count'))

        return approval

