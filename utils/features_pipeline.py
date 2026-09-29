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

from utils.data_processing_silver_table import (
    process_silver_attributes,
    process_silver_financials,
    process_silver_clickstream
)


def build_feature_pipeline(spark, start_date_str, feature_end_date_str):
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
    print("Date list: \n", dates_str_lst)
    
    # =====================
    # BRONZE
    # =====================
    
    # Run bronze backfill
    feature_sources = {
        "attributes": "data/features_attributes.csv",
        "financials": "data/features_financials.csv",
        "clickstream": "data/feature_clickstream.csv"
    }

    for table_name, source_path in feature_sources.items():
        bronze_directory = f"datamart/bronze/{table_name}"
        for date in date_str_lst:
            utils.data_processing_bronze_table.process_feature_bronze_table(
                date,
                source_path,
                table_name,
                bronze_directory,
                spark
            )


    # =====================
    # SILVER
    # =====================
    bronze_directory = "datamart/bronze/attributes"
    silver_directory = "datamart/silver/attributes"

    for snapshot_date in date_str_lst:
        utils.data_processing_silver_table.process_feature_attributes_silver_table(
            snapshot_date,
            bronze_directory,
            silver_directory,
            spark
        )


