from pyspark import SparkContext, SparkConf, SQLContext
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
import pandas as pd
import numpy as np

from typing import Union, List, Optional, Any
from pyspark.sql import DataFrame as PySparkDataFrame
from functools import wraps

import json
import os

from decimal import Decimal


############################################ CUSTOM EXCEPTION CLASSES ####################################
class ColumnInputNotExistException(Exception):
    def __init__(self, col, df_name):
        self.col = col
        self.df_name = df_name
        message = f"Column {col} does not exist in the dataframe {df_name}"
        super().__init__(message)

class ColumnInputNotOfExpectedDataTypeException(Exception):
    def __init__(self, col, expected_dtypes):
        self.col = col
        self.expected_dtypes = expected_dtypes
        message = f"Column {col} is not of type {expected_dtypes}"
        super().__init__(message)

class DataframeInputNotExistException(Exception):
    def __init__(self, df_name):
        self.df_name = df_name
        message = f"Dataframe {df_name} does not exist as attribute of this Validator instance"
        super().__init__(message)

class ColumnNotExistException(Exception):
    def __init__(self, nonexist_cols, input_cols, expectation):
        self.nonexist_cols = nonexist_cols
        self.input_cols = input_cols
        self.expectation = expectation
        self.message = f"Column(s) {nonexist_cols} does not exist in the dataframe"
        super().__init__(self.message)

class DuplicateException(Exception):
    def __init__(self, dup_cols, input_cols, expectation):
        self.dup_cols = dup_cols
        self.input_cols = input_cols
        self.expectation = expectation
        self.message = f"Duplicate values found in column(s): {', '.join(dup_cols)}"
        super().__init__(self.message)

class MissingValueException(Exception):
    def __init__(self, missing_cols, input_cols, expectation):
        self.missing_cols = missing_cols
        self.input_cols = input_cols
        self.expectation = expectation
        self.message = f"Missing values found in column(s): {missing_cols}"
        super().__init__(self.message)

class DataTypeException(Exception):
    def __init__(self, actual_dtype, input_cols, expectation):
        self.actual_dtype = actual_dtype
        self.input_cols = input_cols
        self.expectation = expectation
        self.message = f"Actual data type: {actual_dtype}"
        super().__init__(self.message)

class InvalidValueException(Exception):
    def __init__(self, invalid_vals, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.invalid_vals = invalid_vals
        self.message = f"Invalid values: {invalid_vals}"
        super().__init__(self.message)

class OutOfRangeValueException(Exception):
    def __init__(self, n_invalid, input_cols, expectation, perc_tol):
        self.n_invalid = n_invalid
        self.expectation = expectation
        self.input_cols = input_cols
        if not perc_tol:
            self.message = f"{n_invalid} values out of range"
        else:
            self.message = f"{n_invalid*100}% of values out of range"
        super().__init__(self.message)

class ColumnCountException(Exception):
    def __init__(self, actual_col_count, input_cols, expectation):
        self.actual_col_count = actual_col_count
        self.input_cols = input_cols
        self.expectation = expectation
        self.message = f"Column count: {actual_col_count}"
        super().__init__(self.message)

class PatternMismatchException(Exception):
    def __init__(self, mismatch_cnt, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.mismatch_cnt = mismatch_cnt
        self.message = f"{mismatch_cnt} values with mismatched patterns"
        super().__init__(self.message)

class NonMidnightTimestampException(Exception):
    def __init__(self, non_midnight_cnt, input_cols, expectation):
        self.non_midnight_cnt = non_midnight_cnt
        self.input_cols = input_cols
        self.expectation = expectation
        self.message = f'{non_midnight_cnt} non-midnight timestamp values'
        super().__init__(self.message)

class MissingValueWithReferenceColsException(Exception):
    def __init__(self, missing_cnt, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.missing_cnt = missing_cnt
        self.message = f"{missing_cnt} missing values"
        super().__init__(self.message)

class InvalidAgeException(Exception):
    def __init__(self, invalid_cnt, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.invalid_cnt = invalid_cnt
        self.message = f"{invalid_cnt} values of calculated age out of range"
        super().__init__(self.message)

class MonthlyMissingValueException(Exception):
    def __init__(self, missing_months, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.missing_months = missing_months
        self.message = f"Months with missing count in column {input_cols} higher than expected: {missing_months}"
        super().__init__(self.message)

class OutlierAboveThresholdException(Exception):
    def __init__(self, n_outliers, input_cols, expectation, perc_tol):
        self.input_cols = input_cols
        self.expectation = expectation
        self.n_outliers = n_outliers
        if not perc_tol:
            self.message = f"{n_outliers} outlier values found in column {input_cols}"
        else:
            self.message = f"{n_outliers*100}% of values in column {input_cols} are outliers"
        super().__init__(self.message)

class PSIAboveThresholdException(Exception):
    def __init__(self, psi, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.psi = psi 
        self.message = f"PSI = {psi}"
        super().__init__(self.message)

class InsufficientSaokeDateException(Exception):
    def __init__(self, actual_date_cnt, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.actual_date_cnt = actual_date_cnt
        self.message = f"Distinct saoke date count: {actual_date_cnt}"
        super().__init__(self.message)

class ContractCountException(Exception):
    def __init__(self, invalid_cnt, input_cols, expectation):
        self.input_cols = input_cols[0]
        self.expectation = expectation
        self.invalid_cnt = invalid_cnt
        self.message = f"{invalid_cnt} cases of {input_cols[0]} count grouped by {input_cols[1]} out of range."
        super().__init__(self.message)

class MultipleValueDateException(Exception):
    def __init__(self, mul_count, input_cols, expectation):
        self.input_cols = input_cols[1]
        self.expectation = expectation
        self.mul_count = mul_count
        self.message = f"{mul_count} cases of {input_cols[0]} with multiple {input_cols[1]} values."
        super().__init__(self.message)
        
class NewObservedValueException(Exception):
    def __init__(self, new_values, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.new_values = new_values
        self.message = f"New distinct values: {new_values}"
        super().__init__(self.message)

class ChangedContractCountBySaokeDateException(Exception):
    def __init__(self, changed_dates, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.changed_dates = changed_dates
        self.message = f"Contract count grouped by {input_cols} changed on dates: {changed_dates}"
        super().__init__(self.message) 

class ChangedMonthlyContractCountException(Exception):
    def __init__(self, changed_months, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.changed_months = changed_months
        self.message = f"Monthly contract count grouped by {input_cols} changed on months: {changed_months}"
        super().__init__(self.message) 

class ContractWithInsufficientSaokeDateException(Exception):
    def __init__(self, n_contracts, input_cols, expectation):
        self.input_cols = input_cols[0]
        self.expectation = expectation
        self.n_contracts = n_contracts
        self.message = f"{n_contracts} contracts with insufficient {input_cols[1]}"
        super().__init__(self.message) 

class ContractWithNullValueDateException(Exception):
    def __init__(self, n_contracts, input_cols, expectation):
        self.input_cols = input_cols[1]
        self.expectation = expectation
        self.n_contracts = n_contracts
        self.message = f"{n_contracts} contracts with null {input_cols[1]}"
        super().__init__(self.message)

class ContractWithInsufficientNextDueDateException(Exception):
    def __init__(self, n_contracts, input_cols, expectation):
        self.input_cols = input_cols[1]
        self.expectation = expectation
        self.n_contracts = n_contracts
        self.message = f"{n_contracts} contracts with insufficient {input_cols[1]}"
        super().__init__(self.message)

class ContractWithInconsistentDPDTrackRecordException(Exception):
    def __init__(self, n_contracts, input_cols, expectation):
        self.input_cols = input_cols[1]
        self.expectation = expectation
        self.n_contracts = n_contracts
        self.message = f"{n_contracts} contracts with inconsistent {input_cols[1]} track record"
        super().__init__(self.message)

class MissingLatestSaokeDateException(Exception):
    def __init__(self, actual_date_delay, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.actual_date_delay = actual_date_delay
        self.message = f"Saoke has only been updated to {actual_date_delay} days before today"
        super().__init__(self.message)

class CrossTableDifferentGroupedCountException(Exception):
    def __init__ (self, mismatch_months, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.mismatch_months = mismatch_months
        self.message = f"Months with mismatched counts: {mismatch_months}"
        super().__init__(self.message)

class CrossTableValueMismatchException(Exception):
    def __init__(self, n_mismatch, input_cols, expectation):
        self.input_cols = input_cols
        self.expectation = expectation
        self.n_mismatch1 = n_mismatch[0]
        self.n_mismatch2 = n_mismatch[1]
        self.message = f"{self.n_mismatch1} values in column {input_cols[0]} does not appear in column {input_cols[1]}. {self.n_mismatch2} values in column {input_cols[1]} does not appear in column {input_cols[0]}"
        super().__init__(self.message)

######################################## LIST OF CUSTOM EXCEPTIONS ####################################################


validation_exceptions =(
    ColumnNotExistException,
    DuplicateException,
    MissingValueException,
    DataTypeException,
    InvalidValueException,
    OutOfRangeValueException,
    ColumnCountException,
    PatternMismatchException,
    NonMidnightTimestampException,
    MissingValueWithReferenceColsException,
    InvalidAgeException,
    MonthlyMissingValueException,
    OutlierAboveThresholdException,
    PSIAboveThresholdException,
    InsufficientSaokeDateException,
    ContractCountException,
    MultipleValueDateException,
    NewObservedValueException,
    ChangedContractCountBySaokeDateException,
    ChangedMonthlyContractCountException,
    ContractWithInsufficientSaokeDateException,
    ContractWithNullValueDateException,
    ContractWithInsufficientNextDueDateException,
    ContractWithInconsistentDPDTrackRecordException,
    MissingLatestSaokeDateException,
    CrossTableDifferentGroupedCountException,
    CrossTableValueMismatchException
)



####################################### REPORT CONSTRUCTOR FUNCTION #####################################################
def report_constructor(attr_name):
    # exception handler function (decorator)
    def handle_exceptions(func):
        @wraps(func) # Use this wraps decorator to retain the docstrings of the wrapped class methods
        def wrapper(self, *args, **kwargs):
            # report dataframe
            rp = pd.DataFrame()
            res = []
            columns = []
            expects = []
            try:
                c, e, r = func(self, *args, **kwargs)
                columns.append(c)
                expects.append(e)
                res.append(r)
                
            except validation_exceptions as e:
                columns.append(f"{e.input_cols}")
                expects.append(f"{e.expectation}")
                res.append(f"{e.message}")

            except Exception as e:
                # Re-raise the exception for any non-custom exceptions
                raise
            
            # Construct report 
            dataframe_name = getattr(self, attr_name)
            rp['Columns'] = columns
            rp['Expectation'] = expects
            rp['Result'] = res 

            rp['Timestamp'] = datetime.now().date()
            rp['Dataframe'] = dataframe_name

            rp['Pass'] = rp['Result'].apply(lambda x: 1 if x == 'Pass' else 0)
            
            rp = rp[['Timestamp', 'Dataframe', 'Columns', 'Expectation', 'Result', 'Pass']]
            return rp    
        return wrapper
    return handle_exceptions


####################################### BASE VALIDATOR CLASS #######################################
class Validator:
    """
    Base Validator class
    """
    def __init__(self, df, df_name):
        self.df = df
        self.df_name = df_name

    def _check_col_exist(self, column_names: Union[str, List[str]]):
        """
        Private method to check if given column names exist in the dataframe. 
        This is used in other class methods below to validate input columns exist in the dataframe before running validation.
        If you want to call this method through any instance of the Validator family, please use the check_col_exist() method instead.
        
        Parameters:
        column_names (list, str): A list of column names
        
        Returns:
        ColumnInputNotExistException if any of the given column does not exist in the dataframe
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        for col in column_names:
            if col not in self.df.columns:
                raise ColumnInputNotExistException(col, self.df_name)

    def _check_dtype_col(self, column_names: Union[str, List[str]], expected_dtypes):
        """
        Private method to check if given column names are of specified dtypes. 
        This is used in other class methods below to validate input columns running validation.
        
        Parameters:
        column_names (list, str): A list of column names
        
        Returns:
        ColumnInputNotOfExpectedDataTypeException if any of the given columns is of non-numeric type
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        for col in column_names:
            if not isinstance(self.df.schema[col].dataType, expected_dtypes):
                raise ColumnInputNotOfExpectedDataTypeException(col, expected_dtypes)

    def _calculate_psi_continuous(self, values, cut_offs, expected_dist):
        vl = pd.DataFrame(values, columns=['Values'])
        vl['Range'] = pd.cut(vl['Values'], cut_offs)
        vcd = list(vl['Range'].value_counts(normalize=True).sort_index().values)

        if len(expected_dist) == len(vcd):
            psi = 0.0
            for i in range(len(expected_dist)):
                psi += (vcd[i] - expected_dist[i]) * np.log(vcd[i] / expected_dist[i])
            return psi 
        else:
            return 9999

    def _calculate_psi_categorical(self, values, categories, expected_dist):
        vl = pd.DataFrame(values, columns=['Values'])

        # Create a mapping dict
        category_map = {}
        for i, cat in enumerate(categories):
            if isinstance(cat, list):
                for val in cat:
                    category_map[val] = f'cat_{i}'
            else:
                category_map[cat] = f'cat_{i}'

        vl['Category'] = vl['Values'].apply(lambda x: category_map.get(x, 'other'))
        vcd = list(vl['Category'].value_counts(normalize=True).sort_index().values)

        if len(expected_dist) == len(vcd):
            psi = 0.0
            for i in range(len(expected_dist)):
                psi += (vcd[i] - expected_dist[i]) * np.log(vcd[i] / expected_dist[i])
            return psi 
        else:
            return 9999

    @report_constructor('df_name')
    def check_col_exist(self, column_names: Union[str, List[str]]):
        """
        Private method to check if given column names exist in the dataframe. 
        This can be called by the user as an independent check
        
        Parameters:
        column_names (list, optional): A list of column names
        
        Returns:
        ColumnNotExistException if any of the given column does not exist in the dataframe
        """
        if isinstance(column_names, str):
            column_names = [column_names]

        # Info about this validation to pass on to @report_constructor('df_name')
        input_cols = f"{', '.join(column_names)}"
        exp = f"Column(s) exists in dataframe"
        
        non_col = []
        for col in column_names:
            if col not in self.df.columns:
                non_col.append(col)

        if len(non_col) > 0:
            raise ColumnNotExistException(non_col, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_col_count(self, col_count: int):
        """
        Check if the dataframe contains a specified number of columns
        
        Parameters:
        col_count (int): The number of columns that the dataframe should have
        
        Returns:
        ColumnCountException if the number of columns in the dataframe does not match the specified number
        """
        # Info about this validation to pass on to @report_constructor('df_name')
        input_cols = f"*"
        exp = f"Dataset has {col_count} columns"

        if len(self.df.columns) != col_count:
            raise ColumnCountException(len(self.df.columns), input_cols, exp)
        else:
            # Return something here to pass on to @report_constructor('df_name'): Column(s), expectation, alert/result
            res = f"Pass"
            return input_cols, exp, res        

    @report_constructor('df_name')
    def check_duplicates(self, column_names: Optional[Union[str, List[str]]]=None):
        """
        Check for duplicates based on given column names
        
        Parameters:
        column_names (list, str, optional): A list of column names to check for duplicates.
        If None, the function will check for duplicates in all columns of the dataframe
        
        Returns: 
        ColumnNotExistException if any of the given column does not exist in the dataframe
        DuplicateException if duplicates is found in any of the given columns
        """
        # check input type. If not None, then column_names should be a string or a list of strings
        if column_names:
            if isinstance(column_names, str):
                column_names = [column_names]
            elif isinstance(column_names, list) and all(isinstance(d, str) for d in column_names):
                pass
            else:
                raise ValueError('Input column_names must be a string, a list of strings, or None')        
            # If column_names is given, check if column(s) exist in df
            self._check_col_exist(column_names)
            input_cols = f"{', '.join(column_names)}"   
        else:
            column_names = self.df.columns
            input_cols = f"*"

        # Info about this validation to pass on to @report_constructor('df_name')
        exp = f"No duplicate values"

        # If column_names is given, check if column(s) exist in df
        dup_cols = []
        tol = self.df.count()
        for col in column_names:
            if (self.df.select(col).distinct().count() / tol) < 1:
                dup_cols.append(col)

        if len(dup_cols) > 0:
            raise DuplicateException(dup_cols, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_missing(self, column_names: Optional[Union[str, List[str]]]=None, tol: int=0, filter_condition: str=None):
        """
        Check for missing values based on given column names
        
        Parameters:
        column_names (list, str, optional): A list of column names or a column_name to check for missing values.
        If None, the function will check for missing values in all columns of the dataframe
        tol (int): Minimum number of missing values allowed (if missing_cnt > tol, the validation will fail). Default is 0.
        filter_condition(str, optional): String expression of filter condition on the dataset before checking for missing values (e.g., DOI_TAC != MOMO)
        
        Returns: 
        ColumnNotExistException if any of the given column does not exist in the dataframe
        MissingValueException if missing values are found in any of the given columns
        """
        # check input type. If not None, then column_names should be a string or a list of strings
        if column_names:
            if isinstance(column_names, str):
                column_names = [column_names]
            elif isinstance(column_names, list) and all(isinstance(d, str) for d in column_names):
                pass
            else:
                raise ValueError('Input column_names must be a string, a list of strings, or None')        
            # If column_names is given, check if column(s) exist in df
            self._check_col_exist(column_names)
            input_cols = f"{', '.join(column_names)}"    
        else:
            column_names = self.df.columns
            input_cols = f"*"

        # Info about this validation to pass on to @report_constructor('df_name')
        exp = f"No missing values"
        if filter_condition:
            exp += f' if {filter_condition}'
        else:
            filter_condition = '1==1'
                    
        # Check missing in columns. 

        # If a column contains missing value, add to missing_col list   
        missing_cols = {}
        for col in column_names:
            missing_cnt = self.df.filter(filter_condition)\
                                .agg(F.count(F.when(F.col(col).isNull(), True))).collect()[0][0]
            if missing_cnt > tol: 
                missing_cols[col] = missing_cnt

        if len(missing_cols) > tol:
            raise MissingValueException(missing_cols, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_missing_if_ref_columns_not_missing(self, column_name: str, ref_columns: Union[str, List[str]], tol: int=0):
        """
        Check for missing values in a column based on another reference column(s).
        Values in the column to check cannot be null/missing if the reference column(s) is not null/missing
        
        Parameters:
        column_name(str): Column to check for missing values
        ref_columns(list, str): (List of) reference column(s)
        tol (int): Minimum number of missing values allowed (if missing_cnt > tol, the validation will fail). Default is 0.
        
        Returns:
        MissingValueWithReferenceColException if missing value is detected in the target column where reference column(s) are not missing
        """
        # Check input, ref_columns must be a string or a list of strings
        if isinstance(ref_columns, str):
            ref_columns = [ref_columns]
        elif isinstance(ref_columns, list) and all(isinstance(d, str) for d in ref_columns):
            pass
        else:
            raise ValueError('Input ref_columns must be a string or a list of strings')
        
        # Check target column exists in dataframe
        self._check_col_exist(column_name)
        # Check referece column(s) exist in dataframe
        self._check_col_exist(ref_columns)
        
        # Info about this validation to pass on to @report_constructor('df_name')
        input_cols = f"{column_name}"
        exp = f"No missing values in {column_name} where {ref_columns} is not null"
        
        # Check missing value based on list of ref_columns
        ## Filter data where ref_columns are not null/missing
        # Constructing the filter condition
        condition = F.col(ref_columns[0]).isNotNull()
        if len(ref_columns) > 1:
            for c in ref_columns[1:]:
                condition = condition & F.col(c).isNotNull()
        
        ## Check missing values in the target column
        missing_cnt = self.df.filter(condition)\
                            .agg(F.count(F.when(F.col(column_name).isNull(), True))).collect()[0][0]

        if missing_cnt > tol:
            raise MissingValueWithReferenceColsException(missing_cnt, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_data_type(self, column_name: str, expected_dtype: str):
        """
        Check if a column's values is of the expected dtype

        Parameters:
        column_name: Name of the column to check
        expected_dtype: Expected data type of the column

        Returns:
        DataTypeException if the column's dtype does not match the input
        """
        # check if given column exists in the df
        self._check_col_exist(column_name)

        # check input expected_dtype
        expected_dtype = expected_dtype.lower()

        valid_dtypes = ["byte", "short", "integer", "long", "float", "double", 
                        "decimal", "string", "boolean", "date", "timestamp", "binary"]

        if expected_dtype not in valid_dtypes:
            raise ValueError(f"Invalid data type: {expected_dtype}. Must be one of valid dtypes: {valid_dtypes}")

        # Info for report_constructor
        input_cols = f"{column_name}"
        exp = f"Values in column {column_name} are of type {expected_dtype}"

        # Check dtype
        actual_dtype = dict(self.df.dtypes).get(column_name)
        if actual_dtype != expected_dtype:
            raise DataTypeException(actual_dtype, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_invalid(self, column_name: str, valid_vals: List[Any], filter_condition: str=None):
        """
        Check if all values in the given column are in the value_set
        
        Parameters:
        column_name(str): Name of the column to check 
        valid_vals(str): A set of valid values for the given column
        filter_condition(str, optional): String expression of filter condition(s) to apply on the dataset before running the validation
        
        Returns:
        InvalidValueException if the column contains invalid values
        """
        # check if given column exists in the df
        self._check_col_exist(column_name)

        # Info about this validation to pass on to @report_constructor('df_name')
        input_cols = f"{column_name}"
        exp = f"Values in column are in valid set"
        
        # check input valid_vals
        if not isinstance(valid_vals, list):
            raise ValueError("Input valid_vals must be a list")
        
        # check filter_condition
        if filter_condition:
            exp += f" if{filter_condition}"
        else:
            filter_condition = '1==1'

        # check valid values
        invalid_vals = [i[0] for i in self.df.filter(filter_condition).select(column_name).distinct().collect() if i[0] not in valid_vals]

        if invalid_vals:
            raise InvalidValueException(invalid_vals, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res      

    @report_constructor('df_name')
    def check_pattern(self, column_name: str, pattern, filter_condition: str=None):
        """
        Check if all values in the given column match the given regex pattern
        
        Parameters:
        column_name: Name of the column to check
        pattern: regex pattern to match against column values
        filter_condition(str, optional): String expression of filter condition(s) to apply on the dataset before running the validation
        
        Returns:
        PatternMismatchException if the column contains values that do not match the pattern
        """
        # check if given column exists in the df
        self._check_col_exist(column_name)

        # Info about this validation to pass on to @report_constructor('df_name')
        input_cols = f"{column_name}"
        exp = f"Values in column match pattern {pattern}"

        # check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # check match pattern
        mismatch_cnt = self.df.filter(filter_condition).filter(~(F.col(column_name).rlike(pattern))).count()

        if mismatch_cnt > 0:
            raise PatternMismatchException(mismatch_cnt, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res 

    @report_constructor('df_name')
    def check_non_midnight_timestamps(self, column_name: str, tol:int=0, filter_condition:str=None):
        """
        Check if the specified timestamp column contains any non-midnight timestamps.

        Parameters:
        column_name(str): Name of timestamp column
        tol(int): Minimum number of non-midnight timestamp values accepted (if non_midnight_cnt > tol, the validation will fail)

        Returns:
        NonMidnightTimestampException
        """
        # check if input column exist in dataframe
        self._check_col_exist(column_name)

        # check if input column is of timestamp type
        self._check_dtype_col(column_name, (T.TimestampType))

        # Return something here to pass on to @report_constructor('df_name'): Column(s), expectation, alert/result
        input_cols = f"{column_name}"
        exp = f"Count of non-midnight timestamp values does not exceed {tol}"

        if filter_condition:  
            exp += f" if {filter_condition}" 
        else:
            filter_condition = '1==1'

        # count cases of non-midnight timestamps
        non_midnight_cnt = self.df.filter(filter_condition)\
                                .where((F.hour(column_name) != 0) | (F.minute(column_name) != 0) | (F.second(column_name) != 0))\
                                .count()

        if non_midnight_cnt > tol:
            raise NonMidnightTimestampException(non_midnight_cnt, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_range(self, column_name: str, min_value: Union[int, float]=-np.inf, max_value: Union[int, float]=np.inf, 
        left_inclusive: bool=True, right_inclusive: bool=True,
        tol: int=None, perc_tol: float=None,
        filter_condition: str=None):
        """
        Check if all values in the given column fall within a specified range

        Parameters:
        column_name(str): Name of the column to check
        min_value(int, float): minimum value (left boundary). If not specified, defaults to -np.inf
        max_value(int, float): maximum value (right boundary). If not specified, defaults to np.inf
        left_inclusive(bool, optional): If False, apply strict min (values must be greater than min_value). Defaults to True
        right_inclusive(bool, optional): If False, apply strict max (values must be greater than max_value). Defaults to True
        tol(int, optional): Minimum number of outlier values accepted. Cannot be specified at the same time with perc_tol.
        perc_tol(float, optional): Minimum percentage (between 0.0 and 1.0) of outlier values accepted. Cannot be specified at the same time with tol.
        filter_condition(str, optional): String expression of filter condition to apply 

        Returns: 
        OutOfRangeValueException if the column contains values that is out of specified range 
        """
        # check if given column exists in the df
        self._check_col_exist(column_name)
        # check if given column is numeric
        self._check_dtype_col(column_name, (T.NumericType, T.StringType))

        # Check min and max value
        if not isinstance(min_value, int) and not isinstance(min_value, float):
            raise ValueError('Input min_value must be an integer or float')
            
        if not isinstance(max_value, int) and not isinstance(max_value, float):
            raise ValueError('Input max_value must be an integer or float')
            
        # If both min_age and max_age input are specified, max_age must be greater than min_age        
        if min_value >= max_value:
            raise ValueError('Input max_value must be greater than min_value')

        # Check that tol and perc_tol are not specified at the same time
        if tol is not None and perc_tol is not None:
            raise ValueError("Only one of 'tol' or 'perc_tol' can be specified, not both.")
        elif tol is None and perc_tol is None:
            # Set perc_tol to 0.05 (5%) if none is specified
            perc_tol = 0.05

        if perc_tol is not None and not 0 <= perc_tol <=1 :
            raise ValueError("perc_tol must be within the range [0, 1].")

        # Constructing the filter condition
        if not left_inclusive:
            left_cond = F.col(column_name) > min_value
        left_cond = F.col(column_name) >= min_value

        if not right_inclusive:
            right_cond = F.col(column_name) < max_value
        right_cond = F.col(column_name) <= max_value

        cond = left_cond & right_cond

        # Info about this validation to pass on to @report_constructor('df_name')
        input_cols = f"{column_name}"
        exp = f"Values in column are between {min_value} and {max_value}"

        
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Count number of values in column that do not satisfy condition
        invalid_cnt = self.df.filter(filter_condition)\
                            .filter(~(cond)).count()

        # If perc_tol is specified, need to get the total count of the column
        if perc_tol is not None:
            invalid_perc = invalid_cnt / self.df.filter(filter_condition)\
                                                .select(F.count(column_name).alias('count'))\
                                                .collect()[0]['count']
            if invalid_perc > perc_tol:
                raise OutOfRangeValueException(invalid_perc, input_cols, exp, perc_tol=True)
            else: 
                res = f"Pass"
                return input_cols, exp, res        
        else:
            if invalid_cnt > tol:
                raise OutOfRangeValueException(invalid_cnt, input_cols, exp, perc_tol=False)
            else:
                res = f"Pass"
                return input_cols, exp, res

    @report_constructor('df_name')
    def check_valid_age_aggregated(self, dob_column: str, ref_column: str, min_age: int=18, max_age: int=70, 
        include_null:bool=False, tol:int=0, filter_condition: str=None):
        """
        Check if aggregated age (calculated by ref_column - dob_column) is within valid range
        
        Parameters:
        dob_column(str): Name of column containing dob
        ref_column(str): Name of reference column containing dates to subtract dob from. If not specified, the reference date will be today's date
        min_age(int, optional): Minimum valid age. If not specified, default to 18
        max_age(int, optional): Maximum valid age. If not specified, default to 70
        include_null(bool): Whether to count null cases, default to False
        tol(int): Minimum number of invalid calculated age accepted (if invalid_cnt > tol, the validation will fail)
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running the validation
        
        Returns:
        InvalidAgeException if any cases of invalid age is found
        """
        # Check if input columns exist in dataframe
        self._check_col_exist(dob_column)
        self._check_col_exist(ref_column)

        # check if input columns are of timestamp type or datetype
        self._check_dtype_col(dob_column, (T.DateType, T.TimestampType))
        self._check_dtype_col(ref_column, (T.DateType, T.TimestampType))
        
        # Check min_age and max_age input
        if not isinstance(min_age, int):
            raise ValueError('Input min_age must be an integer or None')
            
        if not isinstance(max_age, int):
            raise ValueError('Input max_age must be an integer or None')
            
        # If both min_age and max_age input are specified, max_age must be greater than min_age        
        if min_age >= max_age:
            raise ValueError('Input max_age must be greater than min_age')

        # Return something here to pass on to @report_constructor('df_name'): Column(s), expectation, alert/result
        input_cols = f"{dob_column}"
        exp = f"Calculated age values are between {min_age} and {max_age}"
        
              
        if filter_condition:  
            exp += f" if {filter_condition}" 
        else:
            filter_condition = '1==1'

        # Count cases of invalid age 
        invalid_cnt = self.df.filter(filter_condition)\
                            .withColumn(dob_column, F.to_date(dob_column))\
                            .withColumn(ref_column, F.to_date(ref_column))\
                            .filter(~(F.datediff(F.col(ref_column), F.col(dob_column)) / 365.25).between(min_age, max_age))\
                            .count()
        
        # If include_null = True, include cases where caluclated age is null (due to dob and/or ref missing)
        if include_null:
            null_cnt = self.df.filter(filter_condition)\
                        .withColumn(dob_column, F.to_date(dob_column))\
                        .withColumn(ref_column, F.to_date(ref_column))\
                        .withColumn('agg_age', F.datediff(F.col(ref_column), F.col(dob_column)) / 365.25)\
                        .filter(F.col('agg_age').isNull())\
                        .select('agg_age')\
                        .count()
            invalid_cnt += null_cnt
            tol += null_cnt

        if invalid_cnt > tol:
            raise InvalidAgeException(invalid_cnt, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_outlier_by_statistical_methods(self, column_name: str, method: str='iqr', n_threshold: Union[float, int]=1.5,
        tol: int=None, perc_tol: float=None, filter_condition: str=None):
        """
        Check for outliers using statistical methods (Z-score, Interquartile Range, Median Absolute Deviation)

        Parameters:
        column_name(str): Name of numerical data column to be checked
        method(str): Name of statistical method used to detect outlier. Either 'iqr' for Interquartile Range method, 'z_score' for Z-score method, or 'mad' for Median Absolute Deviation.
        n_threshold(float, int): Multiplier to set the upper and lower thresholds. Must be > 0
        tol(int, optional): Minimum number of outlier values accepted. Cannot be specified at the same time with perc_tol.
        perc_tol(float, optional): Minimum percentage (between 0.0 and 1.0) of outlier values accepted. Cannot be specified at the same time with tol.
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running the validation

        Returns:
        OutlierAboveThresholdException
        """
        # Check target column exists in dataframe
        self._check_col_exist(column_name)
        # Check numeric column
        self._check_dtype_col(column_name, T.NumericType)

        # Check that tol and perc_tol are not specified at the same time
        if tol is not None and perc_tol is not None:
            raise ValueError("Only one of 'tol' or 'perc_tol' can be specified, not both.")
        elif tol is None and perc_tol is None:
            # Set perc_tol to 0.05 (5%) if none is specified
            perc_tol = 0.05

        if perc_tol is not None and not 0 <= perc_tol <=1 :
            raise ValueError("perc_tol must be within the range [0, 1].")

        # info for report_constructor
        input_cols = column_name
        exp = f"Outliers in column {column_name} does not exceed "

        if perc_tol is not None:
            exp += f"{perc_tol*100}%"
        elif tol is not None:
            exp += f"{tol} values"

        # Check input for method 
        if method not in ['z_score', 'iqr', 'mad']:
            raise ValueError("Input for 'method' must be either 'iqr', 'z_score', or 'mad'.")

        # Check input for n_threshold
        if n_threshold <= 0:
            raise ValueError("Input for 'n_threshold' must be positive.")

        # Check input for filter_condition
        if filter_condition:
            exp += f' if {filter_condition}'
        else:
            filter_condition = '1==1'

        # iqr method
        if method == 'iqr':
            # calculate 25th and 75th quantiles
            quantiles = self.df.filter(filter_condition)\
                                .stat.approxQuantile(column_name, [0.25, 0.75], 0.0)
            # calculate iqr
            iqr = quantiles[1] - quantiles[0]
            # set up lower and upper bound
            lb, ub = (quantiles[0] - n_threshold*iqr, quantiles[1] + n_threshold*iqr)
            # count number of outliers
            outlier_cnt = self.df.filter(filter_condition)\
                                .select(column_name)\
                                .filter(~F.col(column_name).between(lb, ub))\
                                .count()
        # z_score method
        elif method == 'z_score':
            # calculate mean, standard deviation
            mean_val = self.df.filter(filter_condition)\
                            .select(F.mean(column_name)).collect()[0][0]
            std_val = self.df.filter(filter_condition)\
                            .select(F.stddev_pop(column_name)).collect()[0][0]

            # calculate z-score to detect outlier
            outlier_cnt = self.df.filter(filter_condition)\
                                .withColumn('z_score', (F.col(column_name) - mean_val)/std_val)\
                                .select(column_name, 'z_score')\
                                .filter(F.col('z_score') > n_threshold | (F.col('z_score') < -1*n_threshold))\
                                .count()
        # mad method
        elif method == 'mad':
            # calculate median
            median_val = self.df.filter(filter_condition)\
                                .stat.approxQuantile(column_name, [0.5], 0.0)[0]

            # Calculate the MAD (Median Absolute Deviation)
            mad_val = self.df.filter(filter_condition)\
                            .select(F.abs(F.col(column_name) - median_val).alias("abs_deviation")) \
                            .stat.approxQuantile("abs_deviation", [0.5], 0.0)[0]

            # Define a threshold for outliers by multiplying n_threshold with mad_val
            threshold = n_threshold * mad_val

            # count number of outliers
            outlier_cnt = self.df.filter(filter_condition)\
                                .select(column_name)\
                                .agg(F.count(F.when(F.abs(F.col(column_name) - median_val) > threshold, True)).alias('outlier_cnt'))\
                                .collect()[0]['outlier_cnt']              

        # If perc_tol is specified, need to get the total count of the column
        if perc_tol is not None:
            outlier_perc = outlier_cnt / self.df.filter(filter_condition)\
                                                .select(F.count(column_name).alias('count'))\
                                                .collect()[0]['count']
            if outlier_perc > perc_tol:
                raise OutlierAboveThresholdException(outlier_perc, input_cols, exp, perc_tol=True)
            else: 
                res = f"Pass"
                return input_cols, exp, res        
        else:
            if outlier_cnt > tol:
                raise OutlierAboveThresholdException(outlier_cnt, input_cols, exp, perc_tol=False)
            else:
                res = f"Pass"
                return input_cols, exp, res

    @report_constructor('df_name')
    def check_outlier_by_percentile(self, column_name: str, lower_percentile: float=0.01, upper_percentile: float=0.99, 
        tol: int=0, filter_condition: str=None):
        """
        Check for outliers using percentile

        Parameters:
        column_name(str): Name of column to check
        lower_percentile(float): Percentile (within [0, 1] range) used to set lower bound. Default is 0.01
        upper_percentile(float): Percentile (within [0, 1] range) used to set upper bound. Default is 0.99
        tol(int, optional): Minimum number of outlier values accepted
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running the validation

        Returns:
        OutlierAboveThresholdException
        """
        # Check target column exists in dataframe
        self._check_col_exist(column_name)
        self._check_dtype_col(column_name, T.NumericType)

        # Check input for lower and upper percentile
        if not 0 <= lower_percentile <= 1:
            raise ValueError("'lower_percentile' must be within range [0, 1].")

        if not 0 <= upper_percentile <= 1:
            raise ValueError("'upper_percentile' must be within range [0, 1].")

        if lower_percentile >= upper_percentile:
            raise ValueError("'upper_percentile' must be greater than 'lower_percentile'.")

        # info for report_constructor
        input_cols = column_name
        exp = f"Outliers in column {column_name} does not exceed {tol} values"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'
        
        # Calculate lb and ub using lower and upper percentile
        lb, ub = self.df.filter(filter_condition)\
                                .stat.approxQuantile(column_name, [float(lower_percentile), float(upper_percentile)], 0.0)

        # count number of outliers
        outlier_cnt = self.df.filter(filter_condition)\
                            .select(column_name)\
                            .filter(~F.col(column_name).between(lb, ub))\
                            .count()

        if outlier_cnt > tol:
            raise OutlierAboveThresholdException(outlier_cnt, input_cols, exp, perc_tol=False)
        else:
            res = f"Pass"
            return input_cols, exp, res    

    @report_constructor('df_name')
    def check_outlier_by_min_max(self, column_name: str, 
        min_value: Union[int, float]=None, max_value: Union[int, float]=None,
        min_multiplier: float=0.9, max_multiplier: float=1.1,
        tol: int=None, perc_tol: float=None,
        filter_condition: str=None):
        """
        Check if outlier count/percentage in the column (depending on the min max values) is above threshold

        Parameters:
        column_name(str): Name of column to check
        min_value(int, float): Min value to be multiplied with min_multiplier to set the lower bound
        max_value(int, float): Max value to be multiplied with max_multiplier to set the upper bound
        min_multiplier(float): Multiplier for min_value to set the lower bound
        max_multiplier(float): Multiplier for max_value to set the upper bound
        tol(int, optional): Minimum number of outlier values accepted. Cannot be specified at the same time with perc_tol.
        perc_tol(float, optional): Minimum percentage (between 0.0 and 1.0) of outlier values accepted. Cannot be specified at the same time with tol.
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running the validation

        Returns:
        OutlierAboveThresholdException
        """
        # Check target column exists in dataframe
        self._check_col_exist(column_name)
        self._check_dtype_col(column_name, T.NumericType)

        # Check that tol and perc_tol are not specified at the same time
        if tol is not None and perc_tol is not None:
            raise ValueError("Only one of 'tol' or 'perc_tol' can be specified, not both.")
        elif tol is None and perc_tol is None:
            # Set perc_tol to 0.05 (5%) if none is specified
            perc_tol = 0.05

        if perc_tol is not None and not 0 <= perc_tol <=1:
            raise ValueError("perc_tol must be within the range [0, 1].")

        # Check input for min and max values
        if min_value is None and max_value is None:
            raise ValueError("Either 'min_value' or 'max_value' must be specified, or both.")

        # info for report_constructor
        input_cols = column_name
        exp = f"Outliers in column {column_name} does not exceed "

        if perc_tol is not None:
            exp += f"{perc_tol*100}%"
        elif tol is not None:
            exp += f"{tol} values"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # calculate lower bound and upper bound
        if min_value is not None:
            lb = min_value * min_multiplier
        else:
            lb = -np.inf

        if max_value is not None:
            ub = max_value * max_multiplier
        else:
            ub = np.inf

        # Get count of outliers
        outlier_cnt = self.df.filter(filter_condition)\
                        .select(column_name)\
                        .filter(~F.col(column_name).between(lb, ub))\
                        .count()

        # If perc_tol is specified, need to get the total count of the column
        if perc_tol is not None:
            outlier_perc = outlier_cnt / self.df.filter(filter_condition)\
                                                .select(F.count(column_name).alias('count'))\
                                                .collect()[0]['count']
            if outlier_perc > perc_tol:
                raise OutlierAboveThresholdException(outlier_perc, input_cols, exp, perc_tol=True)
            else: 
                res = f"Pass"
                return input_cols, exp, res
        
        else:
            if outlier_cnt > tol:
                raise OutlierAboveThresholdException(outlier_cnt, input_cols, exp, perc_tol=False)
            else:
                res = f"Pass"
                return input_cols, exp, res

    @report_constructor('df_name')
    def check_psi_continuous_var(self, column_name: str, cut_offs: List[Union[int, float]], expected_dist: List[float],
        psi_threshold: float=0.25, filter_condition: str=None):
        """
        Check if psi of a continuous variable against predefined cutoffs and distributions

        Parameters:
        column_name(str): Name of (numeric) column to check
        cut_offs(List[int, float]): List of predefined cutoffs
        expected_dist(List[float]): List of corresponding distributions for each range. Must add up to 1.0 (or close to 1.0)
        psi_threshold(float): Maximum psi allowed. Default is 0.25
        filter_condition(str): String expression of filter condition(s) to be applied on the dataset before running validation

        Returns:
        PSIAboveThresholdException
        """
        # Check input column
        self._check_col_exist(column_name)
        self._check_dtype_col(column_name, T.NumericType)

        # Check input for expected_dist here.
        # Sum of distributions must add up to 1.0 (or close to 1.0)
        assert abs(sum(expected_dist) - 1.0) < 0.000001

        # info for report_constructor
        input_cols = column_name
        exp = f"PSI of column {column_name} does not exceed {psi_threshold}"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Get values of column into a list
        values = [row[column_name] for row in self.df.filter(filter_condition)\
                        .select(column_name)\
                        .collect()]

        psi = self._calculate_psi_continuous(values, cut_offs, expected_dist)

        if psi > psi_threshold:
            raise PSIAboveThresholdException(psi, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_psi_categorical_var(self, column_name: str, categories: List[Any], expected_dist: List[float],
        psi_threshold: float=0.25, filter_condition: str=None):
        """
        Check if psi of a categorical variable against predefined bins/groupings and distributions

        Parameters:
        column_name(str): Name of categorical variable to check 
        categories(List[Any]): List of predefined categories. If a category contains multiple values, wrap them in a list.
        E.g.,: ['category1', 'category2', ['category3', 'category4'], 'category5']
        expected_dist(List[float]): List of predefined distributions for each range. Must add up to 1.0 (or close to 1.0)
        psi_threshold(float): Maximum psi allowed. Default is 0.25
        filter_condition(str): String expression of filter condition(s) to be applied on the dataset before running validation

        Returns:
        PSIAboveThresholdException
        """
        # Check input column
        self._check_col_exist(column_name)

        # Check input for expected_dist here.
        # Sum of distributions must add up to 1.0 (or close to 1.0)
        assert abs(sum(expected_dist) - 1.0) < 0.000001

        # info for report_constructor
        input_cols = column_name
        exp = f"PSI of column {column_name} does not exceed {psi_threshold}"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Get values of column into a list
        values = [row[column_name] for row in self.df.filter(filter_condition)\
                        .select(column_name)\
                        .collect()]

        psi = self._calculate_psi_categorical(values, cut_offs, expected_dist)

        if psi > psi_threshold:
            raise PSIAboveThresholdException(psi, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res


    @report_constructor('df_name')
    def check_record_count_by_date(self, 
        column_name:str, date_col:str, 
        min_count:int=1000000, max_count:int=None,
        filter_condition: str=None):
        """
        Check if the count of records (grouped by another date column) is sufficient or within a specified range.
        
        Parameters:
        column_name(str): Name of the column for counting records (e.g., APPNUMBER)
        date_col(str): Name of the date column to group the record count by (e.g., created_date)
        min_count(int): Minimum number of contract count expected. Default is 1000000
        max_count(int, optional): Maximum number of contract count expected
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation
        
        Returns:
        ContractCountException if contract count after grouping is not sufficient or out of range 
        """
        # Check column exist dataframe
        self._check_col_exist(column_name)
        self._check_col_exist(date_col)

        # check if input columns are of timestamp type or datetype
        self._check_dtype_col(date_col, (T.DateType, T.TimestampType))
        
        # Check min_count and max_count input
        if not isinstance(min_count, int):
            raise ValueError('Input min_count must be an integer')

        # Return something here to pass on to @report_constructor('df_name'): Column(s), expectation, alert/result
        input_cols = [column_name, date_col]
        exp = f"{column_name} count grouped by {date_col} are at least {min_count}"
        
        if max_count:
            if not isinstance(max_count, int):
                raise ValueError('Input max_count must be an integer')
            if max_count <= min_count:
                raise ValueError('Input max_count must be greater than min_count or None')
            exp = f"{column_name} count grouped by {date_col} are between {min_count} and {max_count}"
                
        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Check contract count
        # Count number of days where contract count < min_count 
        invalid_cnt = self.df.filter(filter_condition)\
                        .filter(F.col(column_name).isNotNull())\
                        .select(column_name, date_col)\
                        .withColumn(date_col, F.col(date_col).cast(T.DateType()))\
                        .groupBy(date_col)\
                        .agg(F.count(F.col(column_name)).alias('count'))\
                        .filter(F.col('count') < min_count)\
                        .count()

        # if max_count is specified
        if max_count:
            invalid_cnt += self.df.filter(filter_condition)\
                                .filter(F.col(column_name).isNotNull())\
                                .select(column_name, date_col)\
                                .withColumn(date_col, F.col(date_col).cast(T.DateType()))\
                                .groupBy(date_col)\
                                .agg(F.count(F.col(column_name)).alias('count'))\
                                .filter(F.col('count') > max_count)\
                                .count()


        if invalid_cnt > 0:
            raise ContractCountException(invalid_cnt, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols[0], exp, res



#################################### SAOKE VALIDATOR CLASS ########################################
class SaokeValidator(Validator):
    """
    Validator class for saoke dataframes that inherits from base Validator class
    """
    def __init__(self, df, df_name):
        super().__init__(df, df_name)
        # List to track validation status and data for each validation 
        # this will help whether to update the observed values or not later on
        self.validation_data = []

    def _load_observed_values(self, file_name):
        """
        Load observed values from file
        """
        try:
            with open(file_name, 'r') as file:
                observed_vals = set(json.load(file))
        except FileNotFoundError:
            observed_vals = set()
        return observed_vals

    def _save_observed_values(self, file_name, values):
        """
        Save observed values into the file
        """
        with open(file_name, 'w') as file:
            json.dump(list(values), file)

    def finalize_updates(self):
        """
        Updating observed values files and cached values files if all validations run successfully
        """
        for data in self.validation_data:
            # update json file
            if '.json' in data['file_path']:
                self._save_observed_values(data['file_path'], data['current_values'])
            # update parquet file
            elif '.parquet' in data['file_path']:
                data['current_values'].to_parquet(data['file_path'])

    @report_constructor('df_name')
    def check_count_saoke_date_within_n_month(self, column_name: str, n_months:int = 1):
        """
        Check if number of distinct values in given column is equal to 
        the number of days in the most recent n month(s)
        
        Parameters:
        column_name(str): Name of the column to check
        n_months(int): Number of recent months to check. Default = 1
        
        Returns:
        InsufficientSaokeDateException if the number of distinct dates in given column 
        is less than the number of days in the previous n_months
        """
        # Check column exist in dataframe
        self._check_col_exist(column_name)
        self._check_dtype_col(column_name, (T.DateType, T.TimestampType))

        
        # Get upper bound and lower bound date based on n_months
        ub = datetime.today().date().replace(day=1) - timedelta(days=1)   # last day of the previous month
        lb = (ub - relativedelta(months=n_months-1)).replace(day=1)    # first day of the previous x month(s)
        
        # Check number of distinct values in column_name within n_months
        # and compare it to max(date) - min(date) + 1
        expected = self.df.withColumn(column_name, F.to_date(F.col(column_name)))\
                    .filter((F.col(column_name) >= lb) & (F.col(column_name) <= ub))\
                    .agg(F.datediff(F.max(column_name), F.min(column_name)) + 1)\
                    .collect()[0][0]

        actual = self.df.withColumn(column_name, F.to_date(F.col(column_name)))\
                        .filter((F.col(column_name) >= lb) & (F.col(column_name) <= ub))\
                        .select(column_name).distinct().count()

        # Return something here to pass on to @report_constructor('df_name'): Column(s), expectation, alert/result
        input_cols = f"{column_name}"
        exp = f"{expected} distinct values of {column_name} within recent {n_months} month(s)."

        if actual < expected:
            raise InsufficientSaokeDateException(actual, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res
      
    @report_constructor('df_name')
    def check_contract_count(self, 
        contract_ref:str, group_by:str, 
        min_count:int=1000000, max_count:int=None,
        filter_condition: str=None):
        """
        Check if the count of contracts (grouped by another column) is sufficient or within a specified range.
        
        Parameters:
        column_name(str): Name of the contract reference column (e.g., Y_CONTRACT_REF)
        group_by(str): Name of the column to group the contract count by (e.g., Y_SEL_DATE)
        min_count(int): Minimum number of contract count expected. Default is 1000000
        max_count(int, optional): Maximum number of contract count expected
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation
        
        Returns:
        ContractCountException if contract count after grouping is not sufficient or out of range 
        """
        # Check column exist dataframe
        self._check_col_exist(contract_ref)
        self._check_col_exist(group_by)
        
        # Check min_count and max_count input
        if not isinstance(min_count, int):
            raise ValueError('Input min_count must be an integer')

        # Return something here to pass on to @report_constructor('df_name'): Column(s), expectation, alert/result
        input_cols = [contract_ref, group_by]
        exp = f"{contract_ref} count grouped by {group_by} are at least {min_count}"
        
        if max_count:
            if not isinstance(max_count, int):
                raise ValueError('Input max_count must be an integer')
            if max_count <= min_count:
                raise ValueError('Input max_count must be greater than min_count or None')
            exp = f"{contract_ref} count grouped by {group_by} are between {min_count} and {max_count}"
                
        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Check contract count
        # Count number of days where contract count < min_count 
        invalid_cnt = self.df.filter(filter_condition)\
                        .filter(F.col(contract_ref).isNotNull())\
                        .select(contract_ref, group_by)\
                        .groupBy(group_by)\
                        .agg(F.count(F.col(contract_ref)).alias('count'))\
                        .filter(F.col('count') < min_count)\
                        .count()

        # if max_count is specified
        if max_count:
            invalid_cnt += self.df.filter(filter_condition)\
                                .filter(F.col(contract_ref).isNotNull())\
                                .select(contract_ref, group_by)\
                                .groupBy(group_by)\
                                .agg(F.count(F.col(contract_ref)).alias('count'))\
                                .filter(F.col('count') > max_count)\
                                .count()


        if invalid_cnt > 0:
            raise ContractCountException(invalid_cnt, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols[0], exp, res

    @report_constructor('df_name')
    def check_contract_unique_value_date(self, contract_ref: str, value_date_column: str, 
        observed_values_file:str='./contracts_with_multiple_value_dates.json',
        filter_condition: str=None):
        """
        Check if only one non-null value date exists for each contract ref
        
        Parameters:
        contract_ref(str): Name of contract reference column
        value_date_column(str): Name of value date column
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation
        
        Returns:
        MultipleValueDateException if there are any cases of contract with more than 1 value date
        """
        # check column exists in dataframe
        self._check_col_exist(contract_ref)
        self._check_col_exist(value_date_column)

        self._check_dtype_col(value_date_column, (T.DateType, T.TimestampType))

        # Return something here to pass on to @report_constructor('df_name'): Column(s), expectation, alert/result
        input_cols = [contract_ref, value_date_column]
        exp = f"Each {contract_ref} has only one value of {value_date_column}"
        
        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Check value date for contracts
        current_vals = self.df.filter(filter_condition)\
                        .select(contract_ref, value_date_column)\
                        .dropDuplicates()\
                        .groupBy(contract_ref)\
                        .agg(F.count(F.col(value_date_column)).alias('count'))\
                        .where(F.col('count')>1)\
                        .drop('count', value_date_column)\
                        .collect()

        # Get list of contract_ref with multiple value_date
        current_vals = {i[contract_ref] for i in current_vals}
        mul_count = len(current_vals)

        # Load observed values from the file
        observed_vals = self._load_observed_values(observed_values_file)

        # Identify new values
        new_values = current_vals - observed_vals

        if new_values:
            # self._save_observed_values(observed_values_file, current_vals)
            self.validation_data.append({'file_path': observed_values_file, 'current_values': current_vals})
            raise MultipleValueDateException(len(new_values), input_cols, exp)
        else:
            res = f"Pass"
            return input_cols[1], exp, res

    @report_constructor('df_name')
    def check_contract_null_value_date(self, contract_ref: str, value_date_column:str,
        observed_values_file:str='./contracts_with_null_value_date.json',
        filter_condition: str=None):
        """
        Check if any null value date exists for each contract

        Parameters:
        contract_ref(str): Name of contract ref column (e.g., Y_CONTRACT_REF)
        value_date_column(str): Name of value date column (e.g., VALUE_DATE)
        observed_values_file(str): File path to store observed values
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation

        Returns:
        ContractWithNullValueDateException
        """
        # Check column exist in dataframe
        self._check_col_exist(contract_ref)
        self._check_col_exist(value_date_column)

        self._check_dtype_col(value_date_column, (T.DateType, T.TimestampType))

        # Info for report_constructor 
        input_cols=[contract_ref, value_date_column]
        exp = f'No {contract_ref} has null {value_date_column}'

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Check current values of contract ref with null value date
        current_vals = {row[contract_ref] for row in self.df.filter(filter_condition)\
                                                            .select(contract_ref, value_date_column)\
                                                            .dropDuplicates()\
                                                            .filter(F.col(value_date_column).isNull())\
                                                            .select(contract_ref).distinct().collect()}

        # Load observed values from the file
        observed_vals = self._load_observed_values(observed_values_file)

        # Identify new values
        new_values = current_vals - observed_vals

        if new_values:
            # self._save_observed_values(observed_values_file, current_vals)
            self.validation_data.append({'file_path': observed_values_file, 'current_values': current_vals})
            raise ContractWithNullValueDateException(len(current_vals), input_cols, exp)
        else:
            res= f"Pass"
            return input_cols[1], exp, res

    @report_constructor('df_name')
    def check_contract_count_by_saoke_date_change(self, contract_ref: str, date_column: str, n_days: Optional[int]=None, 
        cached_values_file: str = './latest_contract_count_by_saoke_date.parquet',
        filter_condition: str=None):
        """
        Check if count of contracts grouped by saoke date remains consistent through each time the dataframe is refreshed

        Parameters:
        contract_ref(str): Name of contract ref column (e.g., Y_CONTRACT_REF)
        date_column(str): Name of saoke date column (e.g., Y_SEL_DATE)
        n_days(int): Number of saoke dates to check. Default is 20
        cached_values_file(str): file path storing most recent count of contracts grouped by saoke date
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation
        
        Returns:
        ChangedContractCountBySaokeDateException if detecting any saoke date with changed contract count
        """
        # Check column exist in dataframe
        self._check_col_exist(contract_ref)
        self._check_col_exist(date_column)

        self._check_dtype_col(date_column, (T.DateType, T.TimestampType))

        # Grab latest saoke date - 1 day from df as upper bound
        # then calculate lower bound
        if n_days:
            ub = self.df.select(F.max(date_column)).collect()[0][0] - relativedelta(days=1)
            lb = ub - relativedelta(days=n_days)

        # Info for report_constructor
        input_cols = [contract_ref, date_column]
        exp = f"{contract_ref} count of the same {date_column} remains unchanged when dataset is refreshed"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Check contract count grouped by saoke date
        # from saoke date = lb to saoke date = ub
        current_cnt = self.df.filter(filter_condition)

        if n_days:
            current_cnt = current_cnt.where(F.col(date_column).between(lb, ub))

        current_cnt = current_cnt.select(contract_ref, date_column)\
                                .groupBy(date_column)\
                                .agg(F.countDistinct(contract_ref).alias('current_count'))\
                                .orderBy(date_column)\
                                .toPandas()

        # Get most recent cached contract count grouped by saoke date
        # If the path does not exist, save the current_cnt and have the validation automatically pass
        if not os.path.exists(cached_values_file):
            current_cnt.to_parquet(cached_values_file)
            res = f"Pass"
            return input_cols[1], exp, res
        else:
            old_cnt = pd.read_parquet(cached_values_file)
            old_cnt.columns = [date_column, 'old_count']

            # check if there are any changes in current contract count by saoke date and result from the previous check
            cnt_joined = current_cnt.merge(old_cnt, on=[date_column], how='inner')
            cnt_joined = cnt_joined[cnt_joined['current_count'] != cnt_joined['old_count']]                        

            # save curent count
            # current_cnt.to_parquet(cached_values_file)
            self.validation_data.append({'file_path': cached_values_file, 'current_values': current_cnt})

        if len(cnt_joined) > 0:
            changed_dates = cnt_joined[date_column].to_list()
            raise ChangedContractCountBySaokeDateException(changed_dates, input_cols[1], exp)
        else:
            res = f"Pass"
            return input_cols[1], exp, res

    @report_constructor('df_name')
    def check_missing_within_n_month(self, column_name: str, date_column: str, n_months:int = 1, tol:int = 0, filter_condition: str=None):
        """
        Check if value in given is missing within recent n months

        Parameters:
        value_date_column(str): Name of column to check for missing values (e.g., VALUE_DATE)
        date_column(str): Name of date column to filter dataset by (e.g., Y_SEL_DATE)
        n_months(int): Number of months to filter data by (default = 1)
        tol(int): Minimum number of missing values allowed (default = 0)
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running valiation
        
        Returns:
        MissingValueException if missing value is detected
        """
        # Check column exist in dataframe
        self._check_col_exist(column_name)
        self._check_col_exist(date_column)

        self._check_dtype_col(date_column, (T.DateType, T.TimestampType))

        # Get upper bound and lower bound date based on n_months
        ub = self.df.select(F.max(date_column)).collect()[0][0]
        lb = ub - relativedelta(months=n_months)

        # Info for report_constructor
        input_cols = f"{column_name}"
        exp = f"No missing value in column {column_name} within {n_months} months"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # count missing values 
        missing_cols = {}  
        missing_cnt = self.df.filter(filter_condition)\
                        .filter(F.col(date_column).between(lb, ub))\
                        .agg(F.count(F.when(F.col(column_name).isNull(), True)))\
                        .collect()[0][0]

        if missing_cnt > tol:
            missing_cols[column_name] = missing_cnt

        if len(missing_cols) > 0:    
            raise MissingValueException(missing_cols, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_count_saoke_date_by_contract(self, contract_ref:str, value_date_column:str, date_column:str, observed_values_file: str, 
        filter_condition: str=None, 
        multiple_value_dates_contracts:Optional[str]=None):
        """
        Check if a contract has sufficient saoke dates.
        Note that this method does not take into account contracts with multiple non-null value dates.
        It does, however, still count contracts with 1 non-null value date and a null value date

        Parameters:
        contract_ref(str): Name of contract ref column (e.g., Y_CONTRACT_REF)
        value_date_column(str): Name of value date column (e.g., VALUE_DATE)
        date_column(str): Name of saoke date column (e.g., Y_SEL_DATE)
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation
        multiple_value_dates_contracts(str, optional): File path for list of contracts with multiple non-null value dates.
        If not specified or if the file does not exist, the method will perform functions to get contracts with multiple value dates first
        observed_values_file(str): File path to store observed invalid values 

        Returns:
        ContractWithInsufficientSaokeDateException
        """
        # Check column exist in dataframe
        self._check_col_exist(contract_ref)
        self._check_col_exist(value_date_column)
        self._check_col_exist(date_column)

        self._check_dtype_col(value_date_column, (T.DateType, T.TimestampType))
        self._check_dtype_col(date_column, (T.DateType, T.TimestampType))

        # Info for report_constructor 
        input_cols=[contract_ref, date_column]
        exp = f'Each {contract_ref} has sufficient {date_column}'

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'


        # Check multiple_value_dates_contracts
        hd_exists = False
        if multiple_value_dates_contracts:      # if the arg is specified 
            if os.path.exists(multiple_value_dates_contracts):       # if the path actually exists
                hd = self._load_observed_values(multiple_value_dates_contracts)
                hd_exists = True

        if not hd_exists:
            hd = {row[contract_ref] for row in self.df.filter(filter_condition)\
                                        .select(contract_ref, value_date_column)\
                                        .dropDuplicates()\
                                        .groupBy(contract_ref)\
                                        .agg(F.count(F.col(value_date_column)).alias('count'))\
                                        .where(F.col('count') > 1)\
                                        .drop('count', value_date_column)\
                                        .collect()}

        # Count number of contract with insufficient saoke datecount 
        # filter out contracts w multiple non-null value dates
        w = Window.partitionBy(contract_ref)
        current_vals = {row[contract_ref] for row in self.df.filter(filter_condition)\
                                                .select(contract_ref, date_column, value_date_column)\
                                                .filter(~F.col(contract_ref).isin(hd))\
                                                .withColumn('MIN_SAOKE_DATE', F.min(date_column).over(w))\
                                                .withColumn('MAX_SAOKE_DATE', F.max(date_column).over(w))\
                                                .withColumn('Datediff', F.datediff(F.col('MAX_SAOKE_DATE'), F.col('MIN_SAOKE_DATE')) + 1)\
                                                .withColumn('Saoke_datecount', F.count('*').over(w))\
                                                .filter(F.col('Datediff') != F.col('Saoke_datecount'))\
                                                .drop(date_column)\
                                                .dropDuplicates()\
                                                .select(contract_ref).distinct().collect()}

        # Load observed values from the file
        observed_vals = self._load_observed_values(observed_values_file)

        # Identify new values
        new_values = current_vals - observed_vals


        if new_values:
            # self._save_observed_values(observed_values_file, current_vals)
            self.validation_data.append({'file_path': observed_values_file, 'current_values': current_vals})
            raise ContractWithInsufficientSaokeDateException(len(current_vals), input_cols, exp)
        else:
            res = f"Pass"
            return input_cols[0], exp, res

    @report_constructor('df_name')
    def check_count_next_due_date_by_contract(self, contract_ref:str, next_due_date_column:str, value_date_column:str, 
        observed_values_file:str='./contracts_with_insufficient_next_due_date.json', filter_condition: str=None):
        """
        Check if any contract has insufficient date_next values

        Parameters:
        contract_ref(str): Name of contract ref column (e.g., Y_CONTRACT_REF)
        next_due_date_column(str): Name of next due date column (e.g., Y_DATE_NEXT)
        value_date_column(str): Name of value date column (e.g., VALUE_DATE)
        observed_values_file(str): File path storing observed values
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation

        Returns:
        ContractWithInsufficientNextDueDateException
        """
        # Check column exist in dataframe
        self._check_col_exist(contract_ref)
        self._check_col_exist(value_date_column)
        self._check_col_exist(next_due_date_column)

        self._check_dtype_col(value_date_column, (T.DateType, T.TimestampType))
        self._check_dtype_col(next_due_date_column, (T.DateType, T.TimestampType))

        # Info for report_constructor 
        input_cols=[contract_ref, next_due_date_column]
        exp = f'Each {contract_ref} has sufficient {next_due_date_column}'

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Get month diff between min_next_due_date and max_next_due_date for each contract
        # The month diff is the number of distinct values of next_due_date each contract is supposed to have
        w = Window.partitionBy(contract_ref)
        diff = self.df.filter(filter_condition)\
                        .select(contract_ref, value_date_column, next_due_date_column)\
                        .drop(value_date_column)\
                        .withColumn('MAX_NEXT_DUE_DATE', F.max(next_due_date_column).over(w))\
                        .withColumn('MIN_NEXT_DUE_DATE', F.min(next_due_date_column).over(w))\
                        .withColumn('Month_diff', F.months_between(F.col('MAX_NEXT_DUE_DATE'), F.col('MIN_NEXT_DUE_DATE'))+1)\
                        .filter(F.col('MAX_NEXT_DUE_DATE') <= F.current_date())\
                        .dropDuplicates([contract_ref])

        # Get distinct count of next_due_date values for each contract
        cnt = self.df.filter(filter_condition)\
                    .select(contract_ref, value_date_column, next_due_date_column)\
                    .drop(value_date_column)\
                    .groupBy(contract_ref)\
                    .agg(F.countDistinct(next_due_date_column).alias('Count'))


        # Filter for contracts with distinct count of next due date != month diff between max and min next due date
        current_vals = {row[contract_ref] for row in diff.join(cnt, on=[contract_ref], how='left')\
                            .filter(F.col('Month_diff') != F.col('Count'))\
                            .select(contract_ref).distinct().collect()}

        # Load observed values from the file
        observed_vals = self._load_observed_values(observed_values_file)

        # Identify new values
        new_values = current_vals - observed_vals

        if new_values:
            # self._save_observed_values(observed_values_file, current_vals)
            self.validation_data.append({'file_path': observed_values_file, 'current_values': current_vals})
            raise ContractWithInsufficientNextDueDateException(len(current_vals), input_cols, exp)
        else:
            res = f"Pass"
            return input_cols[1], exp, res

    @report_constructor('df_name')
    def check_inconsistent_dpd_track_record_by_contract(self, contract_ref:str, dpd_column:str, date_column:str, 
        filter_condition:str=None,
        observed_values_file:str='./contracts_with_inconsistent_dpd_track_record.json'):
        """
        Check if any contract has inconsistent dpd record. 
        Example cases of inconsistent dpd record:
        1. 0, 0, 0, 1, 2, 4, 5, 6, etc. (skipped values in a progression)
        2. 0, 1, 2, 3, 4, 0, 0, 4, etc. (skipped values after reset to 0)
        3. 0, 1, 2, 3, 4, 5, 6, 2, etc. (skipped the reset (& skipped values in progression afterwards))

        Parameters:
        contract_ref(str): Name of contract ref column (e.g., Y_CONTRACT_REF)
        dpd_column(str): Name of days past due column (e.g., Y_SONGAY_QH)
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation
        date_column(str): Name of saoke date column (e.g., Y_SEL_DATE)
        observed_values_file(str): File path to store observed values

        Returns:
        ContractWithInconsistentDPDTrackRecordException
        """
        # Check column exist in dataframe
        self._check_col_exist(contract_ref)
        self._check_col_exist(dpd_column)
        self._check_col_exist(date_column)

        self._check_dtype_col(dpd_column, (T.NumericType, T.StringType))
        self._check_dtype_col(date_column, (T.DateType, T.TimestampType))

        # Input for report_constructor
        input_cols = [contract_ref, dpd_column]
        exp = f"No {contract_ref} with inconsistent track record of {dpd_column} values"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Begin detecting inconsistent track record of dpd by contract
        w = Window.partitionBy(contract_ref).orderBy(date_column)
        inconsistencies = self.df.filter(filter_condition)\
                                .select(date_column, contract_ref, dpd_column)\
                                .withColumn('prev_dpd', F.lag(dpd_column).over(w))\
                                .withColumn('next_dpd', F.lead(dpd_column).over(w))\
                                .withColumn('days_diff_from_prev', F.col(dpd_column) - F.col('prev_dpd'))\
                                .withColumn('days_diff_from_next', F.col(dpd_column) - F.col('next_dpd'))\
                                .withColumn("reset_point", ((F.col(dpd_column) == 0) & (F.col(dpd_column) > 0)).cast("int"))\
                                .withColumn("inconsistency_flag",
                                            (((F.col("reset_point") == 0) & (F.col("days_diff_from_prev") > 1)) | 
                                            ((F.col("reset_point") == 0) & (F.col("days_diff_from_next") < -1) & (F.col('next_dpd') > 0)) |
                                            ((F.col("reset_point") == 1) & (F.col("next_dpd") > 1))).cast("int"))\
                                .filter(F.col('inconsistency_flag') == 1)

            
        # Get distinct values of contract ref with inconsistent dpd track record
        current_vals = {row[contract_ref] for row in inconsistencies.select(contract_ref).distinct().collect()}

        # Load observed values from the file
        observed_vals = self._load_observed_values(observed_values_file)

        # Identify new values
        new_values = current_vals - observed_vals

        if new_values:
            # self._save_observed_values(observed_values_file, current_vals)
            self.validation_data.append({'file_path': observed_values_file, 'current_values': current_vals})
            raise ContractWithInconsistentDPDTrackRecordException(len(current_vals), input_cols, exp)
        else:
            res = f"Pass"
            return input_cols[1], exp, res


    @report_constructor('df_name')
    def check_latest_saoke_date(self, date_column: str, n_days_delay: int=1, filter_condition: str=None):
        """
        Check if saoke is updated to the latest date (today - n_days_delay)

        Parameters:
        date_column(str): Name of saoke date column (e.g., Y_SEL_DATE)
        n_days_delay(int): Number of delay days from today (validate if saoke is updated to today's date - n_days_delay). Default is 1.
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation

        Returns:
        MissingLatestSaokeDateException
        """
        # Check column exists in dataframe
        self._check_col_exist(date_column)
        self._check_dtype_col(date_column, (T.DateType, T.TimestampType))

        # Input for report_constructor
        input_cols = date_column
        exp = f"Saoke is updated to {n_days_delay} day(s) before today"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        actual_date_delay = self.df.filter(filter_condition)\
                            .agg(F.datediff(F.current_date(), F.max(F.to_date(date_column))))\
                            .collect()[0][0]

        if actual_date_delay > n_days_delay:
            raise MissingLatestSaokeDateException(actual_date_delay, input_cols, exp)
        else:
            res = f'Pass'
            return input_cols, exp, res





################################## CACHED VALIDATOR CLASS ######################################
class CachedValidator(Validator):
    """
    Cached Validator class containing validations that requires cached data. Inherits from base Validator class
    """
    def __init__(self, df, df_name):
        super().__init__(df, df_name)
        # List to track validation status and data for each validation 
        # this will help whether to update the observed values or not later on
        self.validation_data = []

    def _load_observed_values(self, file_name):
        """
        Load observed values from file
        """
        try:
            with open(file_name, 'r') as file:
                observed_vals = set(json.load(file))
        except FileNotFoundError:
            observed_vals = set()
        return observed_vals

    def _save_observed_values(self, file_name, values):
        """
        Save observed values into the file
        """
        with open(file_name, 'w') as file:
            json.dump(list(values), file)

    def finalize_updates(self):
        """
        Updating observed values files and cached values files if all validations run successfully
        """
        for data in self.validation_data:
            # update json file
            if '.json' in data['file_path']:
                self._save_observed_values(data['file_path'], data['current_values'])
            # update parquet file
            elif '.parquet' in data['file_path']:
                data['current_values'].to_parquet(data['file_path'])

    @report_constructor('df_name')
    def check_new_observed_value(self, column_name: str, observed_values_file: str='./observed_values.json', filter_condition: str=None):
        """
        Check if new observed distinct values in the given column is detected

        Parameters:
        column_name(str): Name of column to check
        observed_values_file(str): File path to store observed values
        filter_condition(str): String expression of filter condition(s) applied on the dataset before running the validation

        Returns:
        NewObservedValueException if new distinct value in the given column is detected
        """
        # Check column exist in dataframe
        self._check_col_exist(column_name)

        # Info for @report_constructor
        input_cols = f"{column_name}"
        exp = f"No new observed values"

        # Load observed values from the file
        observed_vals = self._load_observed_values(observed_values_file)

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Get the currernt distinct values in the column
        current_vals = {row[column_name] for row in self.df.filter(filter_condition).select(column_name).distinct().collect()}
        
        # Identify new values
        new_values = current_vals - observed_vals

        # If new_values is detected, update the observed values set and raise exception
        if new_values:
            # self._save_observed_values(observed_values_file, current_vals)
            self.validation_data.append({'file_path': observed_values_file, 'current_values': current_vals})
            raise NewObservedValueException(new_values, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res
            
    @report_constructor('df_name')
    def check_monthly_count_change(self, column_name: str, date_column: str, 
        up_to_months_before: int=2, n_months: int=24, 
        cached_values_file: str = './latest_contract_count_monthly.parquet',
        filter_condition: str=None):
        """
        Check if contract count monthly has changed when dataset is refreshed.

        Parameters:
        column_name(str): Name of column to count (e.g., APPNUMBER)
        date_column(str): Name of date column to group contract count by (e.g., Ngay_khoi_tao, DC_start_time)
        up_to_months_before(int): Number of months before month of current date to filter data
        n_months(int): Number of months to check data (default = 12)
        cached_values_file(str): File path storing cached values
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running validation

        Returns:
        ChangedMonthlyContractCountException if detecting any month with changed contract count
        """ 
        # Check column exist in dataframe
        self._check_col_exist(column_name)
        self._check_col_exist(date_column)

        self._check_dtype_col(date_column, (T.DateType, T.TimestampType))

        # Get ub and lb for time
        ub = datetime.today().date() - relativedelta(months=up_to_months_before)       # ub is the day x months before today
        lb = ub - relativedelta(months=n_months)    # lb is n_months before ub
        # Convert ub and lb to month
        ub = ub.strftime('%Y-%m')
        lb = lb.strftime('%Y-%m')

        # Info for report_constructor
        input_cols = [column_name, date_column]
        exp = f"No change in monthly {column_name} count when dataset is refreshed"

        # Check filter_condition
        if filter_condition:
            exp += f" if{filter_condition}"
        else:
            filter_condition = '1==1'

        # Get monthly contract count   
        current_cnt = self.df.filter(filter_condition)\
                            .select(column_name, date_column)\
                            .withColumn(date_column, F.date_format(F.col(date_column), 'yyyy-MM'))\
                            .filter(F.col(date_column).between(lb, ub))\
                            .groupBy(date_column)\
                            .agg(F.countDistinct(F.col(column_name)).alias('current_count'))\
                            .orderBy(date_column)\
                            .toPandas()

        # Get most recent cached contract count monthly
        # If the path does not exist, save the current_cnt and have the validation automatically pass
        if not os.path.exists(cached_values_file):
            current_cnt.to_parquet(cached_values_file)
            res = f"Pass"
            return input_cols[1], exp, res
        else:
            old_cnt = pd.read_parquet(cached_values_file)
            old_cnt.columns = [date_column, 'old_count']
            # check if there are any changes in current contract count by saoke date and result from the previous check
            cnt_joined = current_cnt.merge(old_cnt, on=[date_column], how='inner')
            cnt_joined = cnt_joined[cnt_joined['current_count'] != cnt_joined['old_count']]                        
            # save curent count
            # current_cnt.to_parquet(cached_values_file)
            self.validation_data.append({'file_path': cached_values_file, 'current_values': current_cnt})


        if len(cnt_joined) > 0:
            changed_months = cnt_joined[date_column].to_list()
            raise ChangedMonthlyContractCountException(changed_months, input_cols[1], exp)
        else:
            res = f"Pass"
            return input_cols[1], exp, res

    @report_constructor('df_name')
    def check_monthly_missing(self, column_name: str, date_column: str, 
        tol: int=0, filter_condition: str=None):
        """
        Check number of missing values in a column grouped by another column(s) (monthly)

        Parameters:
        column_name(str): Column to check for missing value 
        date_column(str): Column to group count of missing value in target column by (e.g., Ngay_khoi_tao)
        ref_column(str, optional): If specified, the method will only look for missing values in target column where ref_column is not missing
        tol (int): Minimum number of missing values allowed (if any of the grouped missing count > tol, the validation will fail)
        filter_condition(str, optional): String expression of condition(s) applied to the dataset before running the validation

        Returns:
        MonthlyMissingValueException
        """
        # Check target column exists in dataframe
        self._check_col_exist(column_name)
        # Check referece column(s) exist in dataframe
        self._check_col_exist(date_column)

        self._check_dtype_col(date_column, (T.DateType, T.TimestampType))

        # Info about this validation to pass on to @report_constructor('df_name')
        input_cols = f"{column_name}"
        exp = f"Monthly missing count in {column_name} grouped by {date_column} does not exceed {tol}"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # Check monthly missing
        missing_cnt = self.df.filter(filter_condition)\
                            .withColumn(date_column, F.date_format(F.col(date_column), 'yyyy-MM'))\
                            .groupBy(date_column)\
                            .agg(F.count(F.when(F.col(column_name).isNull(), True)).alias('missing_count'))\
                            .orderBy(date_column)\
                            .filter(F.col('missing_count') > tol)

        if missing_cnt.count() > 0:
            missing_months = {row[date_column] for row in missing_cnt.select(date_column).collect()}
            raise MonthlyMissingValueException(missing_months, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_name')
    def check_outlier_by_latest_min_max(self, column_name: str, latest_min_max_values: str=None, 
        min_multiplier: float=0.9, max_multiplier: float=1.1,
        tol: int=None, perc_tol: float=None,
        filter_condition: str=None):
        """
        Check if outlier count/percentage in the column depending on the most recently observed min max values is above threshold

        Parameters:
        column_name(str): Name of column to check 
        latest_min_max_values(str, optional): File path storing latest observed min and max values of the column.
        If not specified, the method will calculate it before running the validation, but will NOT store it.
        min_multiplier(float): Multiplier for min value to set the lower bound
        max_multiplier(float): Multiplier for max value to set the upper bound
        tol(int, optional): Minimum number of outlier values accepted. Cannot be specified at the same time with perc_tol.
        perc_tol(float, optional): Minimum percentage (between 0.0 and 1.0) of outlier values accepted. Cannot be specified at the same time with tol.
        filter_condition(str, optional): String expression of filter condition(s) to be applied on the dataset before running the validation

        Returns:
        OutlierAboveThresholdException
        """
        # Check target column exists in dataframe
        self._check_col_exist(column_name)
        self._check_dtype_col(column_name, T.NumericType)

        # Check that tol and perc_tol are not specified at the same time
        if tol is not None and perc_tol is not None:
            raise ValueError("Only one of 'tol' or 'perc_tol' can be specified, not both.")
        elif tol is None and perc_tol is None:
            # Set perc_tol to 0.05 (5%) if none is specified
            perc_tol = 0.05

        if perc_tol is not None and not 0 <= perc_tol <=1 :
            raise ValueError("perc_tol must be within the range [0, 1].")

        # info for report_constructor
        input_cols = column_name
        exp = f"Outliers in column {column_name} does not exceed "

        if perc_tol is not None:
            exp += f"{perc_tol*100}%"
        elif tol is not None:
            exp += f"{tol} values"

        # Check filter_condition
        if filter_condition:
            exp += f" if {filter_condition}"
        else:
            filter_condition = '1==1'

        # calculate current min max values
        current_min = self.df.filter(filter_condition)\
                                .agg(F.min(column_name).alias('min'))\
                                .collect()[0]['min']
        current_max = self.df.filter(filter_condition)\
                                .agg(F.max(column_name).alias('max'))\
                                .collect()[0]['max']

        # Check if latest_min_max_values exists,
        # if yes, grab them from the file path
        observed_exists = False
        if latest_min_max_values:
            minmax = self._load_observed_values(latest_min_max_values)
            if minmax:
                minmax = list(minmax)
                observed_min = minmax[0]
                observed_max = minmax[1]
                observed_exists = True
                # compare current and observed values
                # if different, update latest observed values
                if current_min != observed_min or current_max != observed_max:
                    # self._save_observed_values(latest_min_max_values, {float(current_min), float(current_max)})
                    self.validation_data.append({'file_path': latest_min_max_values, 'current_values': {float(current_min), float(current_max)}})
                # Calculate lower bound and upper bound using observed min and max
                lb = float(observed_min) * min_multiplier
                ub = float(observed_max) * max_multiplier

        # if observed values do not exists
        if not observed_exists:
            # Calculate lower bound and upper bound using current min and max
            lb = float(current_min) * min_multiplier
            ub = float(current_max) * max_multiplier
            if latest_min_max_values:
                # self._save_observed_values(latest_min_max_values, {float(current_min), float(current_max)})
                self.validation_data.append({'file_path': latest_min_max_values, 'current_values': {float(current_min), float(current_max)}})

        # Get count of outliers
        outlier_cnt = self.df.filter(filter_condition)\
                        .select(column_name)\
                        .filter(~F.col(column_name).between(lb, ub))\
                        .count()

        # If perc_tol is specified, need to get the total count of the column
        if perc_tol is not None:
            outlier_perc = outlier_cnt / self.df.filter(filter_condition)\
                                                .select(F.count(column_name).alias('count'))\
                                                .collect()[0]['count']
            if outlier_perc > perc_tol:
                raise OutlierAboveThresholdException(outlier_perc, input_cols, exp, perc_tol=True)
            else: 
                res = f"Pass"
                return input_cols, exp, res
        
        else:
            if outlier_cnt > tol:
                raise OutlierAboveThresholdException(outlier_cnt, input_cols, exp, perc_tol=False)
            else:
                res = f"Pass"
                return input_cols, exp, res

        
################################# CROSS TABLE VALIDATOR CLASS ##################################
class CrossTableValidator:
    """Base Cross Table Validator class"""
    def __init__(self, df_dict: dict):
        # Store dataframes with their reference names in a dictionary
        # expected format: {'df1_name': df1, 'df2_name': df2}
        self.dfs = df_dict
        self.df_names = f"{', '.join(df_dict.keys())}"

    def _check_df_exist(self, df_names:Union[str, List[str]]):
        """
        Internal method to check if a df_name exists as attribute passed to this instance of CrossTableValidator

        Parameters:
        df_name(str): Name of the df to check

        Returns:
        DataframeInputNotExistException
        """
        if isinstance(df_names, str):
            df_names = [df_names]
        for name in df_names:
            if name not in self.dfs.keys():
                raise DataframeInputNotExistException(name)

    def _get_df(self, df_name:str):
        """
        Internal method to get df based on name. 
        If df_name is passed as attribute of this instance, the method will return the dataframe associated with this name
        """
        self._check_df_exist(df_name)
        return self.dfs[df_name], df_name

    def _check_col_exist(self, df_name:str, df:PySparkDataFrame, column_names:Union[str, List[str]]):
        """
        Private method to check if given column names exist in the dataframe. 
        This is used in other class methods below to validate input columns exist in the dataframe before running validation.
        If you want to call this method through any instance of the Validator family, please use the check_col_exist() method instead.
        
        Parameters:
        column_names (list, str): A list of column names
        
        Returns:
        ColumnInputNotExistException if any of the given column does not exist in the dataframe
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        for col in column_names:
            if col not in df.columns:
                raise ColumnInputNotExistException(col, df_name)

    def _check_dtype_col(self, df_name, column_names: Union[str, List[str]], expected_dtypes):
        """
        Private method to check if given column names are of specified dtypes. 
        This is used in other class methods below to validate input columns running validation.
        
        Parameters:
        column_names (list, str): A list of column names
        
        Returns:
        ColumnInputNotOfExpectedDataTypeException if any of the given columns is of non-numeric type
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        for col in column_names:
            if not isinstance(self.dfs[df_name].schema[col].dataType, expected_dtypes):
                raise ColumnInputNotOfExpectedDataTypeException(col, expected_dtypes)

    @report_constructor('df_names')
    def check_match_monthly_grouped_count(self,  
        df1_name:str, 
        df2_name:str,
        df1_column_name:str, 
        df2_column_name:str,
        df1_groupby:str, 
        df2_groupby:str,
        df1_filter_condition: str=None,
        df2_filter_condition: str=None):
        """
        Check if grouped count of a column in df1 matches that in df2

        Parameters:
        df1_name, df2_name(str): Names of df1 and df2
        df1_column_name, df2_column_name(str): Column names of df1 and df2 to be counted
        df1_groupby, df2_groupby(str): Column names of df1 and df2 to group the count by
        df1_filter_condition, df2_filter_condition(str, optional): String expression of filter conditions to be applied on df1 and df2 before running validation

        Returns:
        CrossTableDifferentGroupedCountException
        """
        # Check if df exists as attribute of the class instance
        # and grab it from the attributes
        df1, _ = self._get_df(df1_name)
        df2, _ = self._get_df(df2_name)

        # Check if column inputs exists for each df
        self._check_col_exist(df1_name, df1, [df1_column_name, df1_groupby])
        self._check_col_exist(df2_name, df2, [df2_column_name, df2_groupby])

        # Verify that df1_groupby and df2_groupby are of valid types
        self._check_dtype_col(df1_name, df1_groupby, (T.DateType, T.TimestampType))
        self._check_dtype_col(df2_name, df2_groupby, (T.DateType, T.TimestampType))

        # input for report_constructor later 
        input_cols = (df1_column_name, df2_column_name)
        exp = f"Monthly count of {df1_column_name} matches monthly count of {df2_column_name}"

        # Check filter_condition for df1 and df2
        if df1_filter_condition:
            exp += f" ({df1_name} condition: {df1_filter_condition})"
        else:
            df1_filter_condition = '1==1'

        if df2_filter_condition:
            exp += f" ({df2_name} condition: {df2_filter_condition})"
        else:
            df2_filter_condition = '1==1'

        # Get monthly grouped count of df1 & df2
        df1_cnt = df1.filter(df1_filter_condition)\
                    .select(df1_column_name, df1_groupby)\
                    .withColumn('month', F.date_format(F.col(df1_groupby), 'yyyy-MM'))\
                    .drop(df1_groupby)\
                    .groupBy('month')\
                    .agg(F.count(df1_column_name).alias('df1_count'))

        df2_cnt = df2.filter(df2_filter_condition)\
                    .select(df2_column_name, df2_groupby)\
                    .withColumn('month', F.date_format(F.col(df2_groupby), 'yyyy-MM'))\
                    .drop(df2_groupby)\
                    .groupBy('month')\
                    .agg(F.count(df2_column_name).alias('df2_count'))

        # inner join and filter for months with mismatch count
        mismatch_months = {row['month'] for row in df1_cnt.join(df2_cnt, on=['month'], how='inner')\
                            .filter(F.col('df1_count') != F.col('df2_count'))\
                            .select('month').collect()}

        mismatch_cnt = len(mismatch_months)

        if mismatch_cnt > 0:
            raise CrossTableDifferentGroupedCountException(mismatch_months, input_cols, exp)
        else:
            res = f"Pass"
            return input_cols, exp, res

    @report_constructor('df_names')
    def check_match_values(self,
        df1_name:str, 
        df2_name:str,
        df1_column_name:str,
        df2_column_name:str,
        df1_filter_condition: str=None,
        df2_filter_condition: str=None):
        """
        Check if all values in df1_column_name appears in df2_column_name and vice-versa

        Parameters:
        df1_name, df2_name(str): Names of df1 and df2
        df1_column_name, df2_column_name(str): Column names of df1 and df2 to be checked
        df1_filter_condition, df2_filter_condition(str, optional): String expression of filter conditions to be applied on df1 and df2 before running validation

        Returns: 
        CrossTableValueMismatchException
        """
        # Check if df exists as attribute of the class instance
        # and grab it from the attributes
        df1, _ = self._get_df(df1_name)
        df2, _ = self._get_df(df2_name)

        # Check if column inputs exists for each df
        self._check_col_exist(df1_name, df1, [df1_column_name])
        self._check_col_exist(df2_name, df2, [df2_column_name])

        # Info for report_constructor
        input_cols = (df1_column_name, df2_column_name)
        exp = f"All values in {df1_column_name} appear in {df2_column_name} and vice-versa"

        # Check filter_condition for df1 and df2
        if df1_filter_condition:
            exp += f" ({df1_name} condition: {df1_filter_condition})"
        else:
            df1_filter_condition = '1==1'

        if df2_filter_condition:
            exp += f" ({df2_name} condition: {df2_filter_condition})"
        else:
            df2_filter_condition = '1==1'

        # Check if any value in df1_column_name does not appear in df2_column_name
        mismatch_vals1 = {row['Values'] for row in df1.filter(df1_filter_condition)\
                            .select(df1_column_name).distinct()\
                            .withColumnRenamed(df1_column_name, 'Values')\
                            .join(df2.select(df2_column_name).distinct().withColumnRenamed(df2_column_name, 'Values'),
                                on=['Values'], how='left_anti')\
                            .collect()}

        # Check if any value in df2_column_name does not appear in df1_column_name
        mismatch_vals2 = {row['Values'] for row in df2.filter(df2_filter_condition)\
                            .select(df2_column_name).distinct()\
                            .withColumnRenamed(df2_column_name, 'Values')\
                            .join(df1.select(df1_column_name).distinct().withColumnRenamed(df1_column_name, 'Values'),
                                on=['Values'], how='left_anti')\
                            .collect()}

        # check mismatch count
        if (mismatch_vals1 and mismatch_vals2) or (mismatch_vals1 or mismatch_vals2):
            n_mismatch1 = len(mismatch_vals1)
            n_mismatch2 = len(mismatch_vals2)
            raise CrossTableValueMismatchException((n_mismatch1, n_mismatch2), (df1_column_name, df2_column_name), exp)
        else:
            res = f"Pass"
            return input_cols, exp, res










