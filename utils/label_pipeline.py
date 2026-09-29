import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.helper import generate_first_of_month_dates
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Build label engineering pipeline 
def build_label_pipeline(spark, start_date_str="2023-01-01", end_date_str="2024-12-01"):
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
    print("Date list: \n", dates_str_lst)

    # create BRONZE datalake
    bronze_loan_directory = "datamart/bronze/lms/"

    if not os.path.exists(bronze_loan_directory):
        os.makedirs(bronze_loan_directory)

    # run bronze backfill
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_label_bronze_table(date_str, bronze_loan_directory, spark)


    # create SILVER datalake
    silver_loan_directory = "datamart/silver/loan_daily/"

    if not os.path.exists(silver_loan_directory):
        os.makedirs(silver_loan_directory)

    # run silver backfill
    for date_str in dates_str_lst:
        utils.data_processing_silver_table.process_label_silver_table(date_str, bronze_loan_directory, silver_loan_directory, spark)



    # create GOLD datalake
    gold_label_store_directory = "datamart/gold/label_store/"

    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)

    # run gold backfill
    for date_str in dates_str_lst:
        utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

    print("Label pipeline completed")

"""
Tentative: Remove this section


    folder_path = gold_label_store_directory
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    print("row_count:",df.count())

    df.show()

"""
    
    