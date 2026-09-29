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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


# FEATURE GOLD TABLE

CLICKSTREAM_FEATURE_COLS = [f"fe_{i}" for i in range(1, 21)]
ROLLING_3M_FEATURE_COLS = [
    f"{feature_name}_mean_3m"
    for feature_name in CLICKSTREAM_FEATURE_COLS
]


def _silver_feature_path(silver_directory, table_name, snapshot_date_str):
    partition_name = (
        f"silver_feature_{table_name}_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )
    return os.path.join(silver_directory, partition_name)


def _safe_ratio(numerator_col, denominator_col):
    numerator = F.col(numerator_col)
    denominator = F.col(denominator_col)

    return (
        F.when(
            numerator.isNotNull()
            & denominator.isNotNull()
            & (denominator > 0),
            numerator / denominator
        )
        .otherwise(F.lit(None).cast("double"))
    )


def _check_unique_customer_snapshot(df, table_name):
    duplicate_count = (
        df.groupBy("customer_id", "snapshot_date")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{table_name} contains {duplicate_count} duplicated "
            "(customer_id, snapshot_date) keys."
        )


def _build_gold_attributes(attributes_df):
    return (
        attributes_df
        .select(
            "customer_id",
            "snapshot_date",
            "age",
            "occupation"
        )
        .withColumn(
            "age_under_18_flag",
            F.when(
                F.col("age").isNull(),
                F.lit(None).cast(IntegerType())
            )
            .when(F.col("age") < 18, F.lit(1))
            .otherwise(F.lit(0))
            .cast(IntegerType())
        )
    )


def _build_gold_financials(financials_df):
    return (
        financials_df
        .withColumn(
            "emi_to_salary_ratio",
            _safe_ratio(
                "total_emi_per_month",
                "monthly_inhand_salary"
            )
        )
        .withColumn(
            "debt_to_annual_income_ratio",
            _safe_ratio(
                "outstanding_debt",
                "annual_income"
            )
        )
        .withColumn(
            "investment_to_salary_ratio",
            _safe_ratio(
                "amount_invested_monthly",
                "monthly_inhand_salary"
            )
        )
        .withColumn(
            "balance_to_salary_ratio",
            _safe_ratio(
                "monthly_balance",
                "monthly_inhand_salary"
            )
        )
        .withColumn(
            "income_after_emi",
            F.when(
                F.col("monthly_inhand_salary").isNotNull()
                & F.col("total_emi_per_month").isNotNull(),
                F.col("monthly_inhand_salary")
                - F.col("total_emi_per_month")
            )
            .otherwise(F.lit(None).cast("double"))
        )
        .withColumn(
            "pays_minimum_amount_flag",
            F.when(
                F.col("payment_of_min_amount") == "Yes",
                F.lit(1)
            )
            .when(
                F.col("payment_of_min_amount") == "No",
                F.lit(0)
            )
            .otherwise(F.lit(None).cast(IntegerType()))
        )
        .drop("payment_of_min_amount")
    )


def _build_gold_clickstream(
    snapshot_date_str,
    silver_clickstream_directory,
    spark
):
    """
    Keep current fe_1..fe_20 and add 3-calendar-month rolling means.

    fe_i_mean_3m is populated only when the customer has all three
    snapshots: current month, month - 1, and month - 2.
    No future snapshot is read.
    """
    snapshot_date = datetime.strptime(
        snapshot_date_str,
        "%Y-%m-%d"
    )

    history_dates = [
        (
            snapshot_date
            - relativedelta(months=months_back)
        ).strftime("%Y-%m-%d")
        for months_back in [2, 1, 0]
    ]

    current_path = _silver_feature_path(
        silver_clickstream_directory,
        "clickstream",
        snapshot_date_str
    )

    if not os.path.exists(current_path):
        # raise FileNotFoundError(
        #     f"Current clickstream Silver partition not found: {current_path}"
        # )
        print(
            f"No clickstream Silver partition for {snapshot_date_str}: "
            "clickstream features will be NULL, has_clickstream = 0"
        )
        return None

    current_clickstream_df = (
        spark.read.parquet(current_path)
        .select(
            "customer_id",
            "snapshot_date",
            *CLICKSTREAM_FEATURE_COLS
        )
    )

    _check_unique_customer_snapshot(
        current_clickstream_df,
        f"Silver clickstream {snapshot_date_str}"
    )

    history_paths = [
        _silver_feature_path(
            silver_clickstream_directory,
            "clickstream",
            history_date
        )
        for history_date in history_dates
    ]

    existing_history_paths = [
        path for path in history_paths
        if os.path.exists(path)
    ]

    history_df = (
        spark.read.parquet(*existing_history_paths)
        .filter(F.col("snapshot_date").isin(history_dates))
        .select(
            "customer_id",
            "snapshot_date",
            *CLICKSTREAM_FEATURE_COLS
        )
    )

    _check_unique_customer_snapshot(
        history_df,
        f"Silver clickstream history ending {snapshot_date_str}"
    )

    agg_exprs = [
        F.countDistinct("snapshot_date").alias("_history_months_3m")
    ] + [
        F.avg(F.col(feature_name)).alias(
            f"_{feature_name}_mean_3m_raw"
        )
        for feature_name in CLICKSTREAM_FEATURE_COLS
    ]

    rolling_df = (
        history_df
        .groupBy("customer_id")
        .agg(*agg_exprs)
    )

    for feature_name in CLICKSTREAM_FEATURE_COLS:
        rolling_df = rolling_df.withColumn(
            f"{feature_name}_mean_3m",
            F.when(
                F.col("_history_months_3m") == 3,
                F.col(f"_{feature_name}_mean_3m_raw")
            )
            .otherwise(F.lit(None).cast("double"))
        )

    rolling_df = rolling_df.drop(
        *[
            f"_{feature_name}_mean_3m_raw"
            for feature_name in CLICKSTREAM_FEATURE_COLS
        ]
    )

    return (
        current_clickstream_df
        .join(
            rolling_df,
            on="customer_id",
            how="left"
        )
        .withColumn(
            "has_clickstream",
            F.lit(1).cast(IntegerType())
        )
        .withColumn(
            "has_full_clickstream_3m_history",
            (
                F.col("_history_months_3m") == 3
            ).cast(IntegerType())
        )
        .select(
            "customer_id",
            "snapshot_date",
            "has_clickstream",
            "has_full_clickstream_3m_history",
            *CLICKSTREAM_FEATURE_COLS,
            *ROLLING_3M_FEATURE_COLS
        )
    )


def process_features_gold_table(
    snapshot_date_str,
    silver_attributes_directory,
    silver_financials_directory,
    silver_clickstream_directory,
    gold_feature_store_directory,
    spark
):
    """    
    Grain:
        one row per customer_id x snapshot_date

    Join:
        attributes
          LEFT JOIN financials
          LEFT JOIN clickstream

    """
    attributes_path = _silver_feature_path(
        silver_attributes_directory,
        "attributes",
        snapshot_date_str
    )
    financials_path = _silver_feature_path(
        silver_financials_directory,
        "financials",
        snapshot_date_str
    )

    if not os.path.exists(attributes_path):
        raise FileNotFoundError(
            f"Silver attributes partition not found: {attributes_path}"
        )

    if not os.path.exists(financials_path):
        raise FileNotFoundError(
            f"Silver financials partition not found: {financials_path}"
        )

    attributes_df = spark.read.parquet(attributes_path)
    financials_df = spark.read.parquet(financials_path)

    print(
        "loaded from:",
        attributes_path,
        "row count:",
        attributes_df.count()
    )
    print(
        "loaded from:",
        financials_path,
        "row count:",
        financials_df.count()
    )

    _check_unique_customer_snapshot(
        attributes_df,
        f"Silver attributes {snapshot_date_str}"
    )
    _check_unique_customer_snapshot(
        financials_df,
        f"Silver financials {snapshot_date_str}"
    )

    gold_attributes_df = _build_gold_attributes(attributes_df)
    gold_financials_df = _build_gold_financials(financials_df)
    gold_clickstream_df = _build_gold_clickstream(
        snapshot_date_str,
        silver_clickstream_directory,
        spark
    )

    attribute_row_count = gold_attributes_df.count()

    gold_df = (
        gold_attributes_df
        .join(
            gold_financials_df,
            ["customer_id", "snapshot_date"],
            "left"
        )
    )

    if gold_clickstream_df is not None:
        gold_df = (
            gold_df
            .join(
                gold_clickstream_df,
                ["customer_id", "snapshot_date"],
                "left"
            )
            .fillna({
                "has_clickstream": 0,
                "has_full_clickstream_3m_history": 0
            })
        )

    else:
        # No clickstream snapshot for this month
        gold_df = (
            gold_df
            .withColumn(
                "has_clickstream",
                F.lit(0).cast(IntegerType())
            )
            .withColumn(
                "has_full_clickstream_3m_history",
                F.lit(0).cast(IntegerType())
            )
        )

        for feature_name in CLICKSTREAM_FEATURE_COLS:
            gold_df = gold_df.withColumn(
                feature_name,
                F.lit(None).cast("double")
            )

            gold_df = gold_df.withColumn(
                f"{feature_name}_mean_3m",
                F.lit(None).cast("double")
            )

    _check_unique_customer_snapshot(
        gold_df,
        f"Gold feature store {snapshot_date_str}"
    )

    gold_row_count = gold_df.count()

    if gold_row_count != attribute_row_count:
        raise ValueError(
            "Gold row count does not match the attribute anchor cohort "
            f"for {snapshot_date_str}: "
            f"attributes={attribute_row_count}, gold={gold_row_count}"
        )

    coverage = (
        gold_df
        .agg(
            F.sum("has_clickstream").alias("rows_with_clickstream"),
            F.sum(
                "has_full_clickstream_3m_history"
            ).alias("rows_with_full_3m_history")
        )
        .first()
    )

    print(
        f"Gold {snapshot_date_str}: rows={gold_row_count}, "
        f"rows_with_clickstream={coverage['rows_with_clickstream']}, "
        "rows_with_full_3m_history="
        f"{coverage['rows_with_full_3m_history']}"
    )

    if not os.path.exists(gold_feature_store_directory):
        os.makedirs(gold_feature_store_directory)

    partition_name = (
        "gold_feature_store_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )
    filepath = os.path.join(
        gold_feature_store_directory,
        partition_name
    )

    gold_df.write.mode("overwrite").parquet(filepath)

    print(
        "saved to:",
        filepath,
        "row count:",
        gold_row_count
    )

    return gold_df
