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

from utils.label_pipeline import build_label_pipeline



def main():
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("MLE Assignment 1") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # set up config
    snapshot_date_str = "2023-01-01"
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"    

    # start label pipeline
    print("Starting label pipeline...")
    build_label_pipeline(spark, start_date_str, end_date_str)
    print("Label pipeline completed.")


    # Start feature pipeline


if __name__ == "__main__":
    main()