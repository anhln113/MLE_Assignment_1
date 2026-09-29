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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType

# Process lms silver tables for label
def process_label_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


# Process features_attributes silver tables for feature pipeline
def process_feature_attributes_silver_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # create silver directory if not yet exist
    os.makedirs(silver_attributes_directory, exist_ok=True)

    # connect to bronze table
    partition_name = ("bronze_feature_attributes_" + snapshot_date_str.replace("-", "_") + ".csv")

    filepath = os.path.join(bronze_attributes_directory, partition_name)

    # Keep Bronze values as strings first; enforce schema in Silver
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print("loaded from:", filepath, "row count:", df.count())


    # ---------------------------------------------------------
    # clean Age column
    # ---------------------------------------------------------

    # remove trailing "_" and cast to integer
    df = df.withColumn(
        "age",
        F.regexp_replace(
            F.trim(F.col("Age")),
            r"_+$",
            ""
        ).cast(IntegerType())
    )

    # invalid ages (<0 or > 100) -> null
    df = df.withColumn(
        "age",
        F.when(
            (F.col("age") < 0) |
            (F.col("age") > 100),
            F.lit(None).cast(IntegerType())
        ).otherwise(F.col("age"))
    )


    # ---------------------------------------------------------
    # clean Occupation column
    # ---------------------------------------------------------

    df = df.withColumn(
        "occupation",
        F.when(
            F.trim(F.col("Occupation")) == "_______",
            F.lit(None).cast(StringType())
        ).otherwise(
            F.trim(F.col("Occupation"))
        )
    )

    # ---------------------------------------------------------
    # enforce schema
    # ---------------------------------------------------------

    df = (
        df
        .withColumn(
            "customer_id",
            F.col("Customer_ID").cast(StringType())
        )
        .withColumn(
            "snapshot_date",
            F.col("snapshot_date").cast(DateType())
        )
    )

    # ---------------------------------------------------------
    # keep columns required for downstream feature pipeline
    # Name and SSN are excluded as they are PII and are not required for feature generation
    # ---------------------------------------------------------

    df = df.select(
        "customer_id",
        "age",
        "occupation",
        "snapshot_date"
    )

    # ---------------------------------------------------------
    # save silver table
    # ---------------------------------------------------------

    partition_name = (
        "silver_feature_attributes_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )

    filepath = os.path.join(
        silver_attributes_directory,
        partition_name
    )

    df.write.mode("overwrite").parquet(filepath)

    print(
        "saved to:",
        filepath,
        "row count:",
        df.count()
    )

    return df


# Process features_financials silver tables for feature pipeline
def process_feature_financials_silver_table(
    snapshot_date_str,
    bronze_financials_directory,
    silver_financials_directory,
    spark,
):
    """Clean one Bronze financials snapshot and write one Silver parquet partition.
    Bronze is read as strings so malformed numeric values can be handled explicitly.
    """

    # Validate expected date format early
    datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    os.makedirs(silver_financials_directory, exist_ok=True)

    partition_name = (
        "bronze_feature_financials_"
        + snapshot_date_str.replace("-", "_")
        + ".csv"
    )
    filepath = os.path.join(bronze_financials_directory, partition_name)

    # Keep Bronze source values as strings; Silver owns parsing/schema enforcement
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print("loaded from:", filepath, "row count:", df.count())

    # ---------------------------------------------------------
    # 1. Parse numeric columns
    # ---------------------------------------------------------
    # Rules from EDA:
    # - trailing underscore: remove underscore, then parse
    # - lone underscore(s): NULL
    # - wrapped underscore sentinel (e.g. __10000__): NULL
    numeric_columns = {
        "Annual_Income": DoubleType(),
        "Monthly_Inhand_Salary": DoubleType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": DoubleType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Outstanding_Debt": DoubleType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "Total_EMI_per_month": DoubleType(),
        "Amount_invested_monthly": DoubleType(),
        "Monthly_Balance": DoubleType(),
    }

    for column_name, target_type in numeric_columns.items():
        raw = F.trim(F.col(column_name))
        is_sentinel = raw.rlike(r"^_+$") | raw.rlike(r"^__.*__$")
        normalized = F.regexp_replace(raw, r"_+$", "")

        # Cast via double first so values such as "3.0" can become integer cleanly
        parsed_value = normalized.cast("double").cast(target_type)

        df = df.withColumn(
            column_name,
            F.when(
                is_sentinel,
                F.lit(None).cast(target_type),
            ).otherwise(parsed_value),
        )

    # ---------------------------------------------------------
    # 2. Clean categorical placeholders
    # ---------------------------------------------------------
    categorical_placeholders = {
        "Credit_Mix": ["_"],
        "Payment_of_Min_Amount": ["NM"],
        "Payment_Behaviour": ["!@9#%8"],
    }

    for column_name, placeholders in categorical_placeholders.items():
        df = df.withColumn(
            column_name,
            F.when(
                F.trim(F.col(column_name)).isin(*placeholders),
                F.lit(None).cast(StringType()),
            ).otherwise(F.trim(F.col(column_name))),
        )

    # Trim Type_of_Loan without changing its meaning
    df = df.withColumn("Type_of_Loan", F.trim(F.col("Type_of_Loan")))

    # ---------------------------------------------------------
    # 3. Cross-check / clean Num_of_Loan using Type_of_Loan
    # ---------------------------------------------------------
    df = df.withColumn(
        "loan_type_count",
        F.when(
            F.col("Type_of_Loan").isNull(),
            F.lit(0),
        ).otherwise(
            F.size(
                F.split(
                    F.regexp_replace(
                        F.col("Type_of_Loan"),
                        r",\s*and\s+",
                        ", ",
                    ),
                    r",\s*",
                )
            )
        ),
    )

    num_of_loan = F.col("Num_of_Loan")
    loan_type_count = F.col("loan_type_count")
    type_of_loan = F.col("Type_of_Loan")

    df = df.withColumn(
        "Num_of_Loan_clean",
        # Type_of_Loan missing: retain a plausible Num_of_Loan (including 0)
        F.when(
            type_of_loan.isNull() & num_of_loan.between(0, 9),
            num_of_loan,
        )
        # Both sources available and consistent
        .when(
            type_of_loan.isNotNull()
            & num_of_loan.between(0, 9)
            & num_of_loan.eqNullSafe(loan_type_count),
            num_of_loan,
        )
        # Num_of_Loan missing/out of range but Type_of_Loan can recover the count
        .when(
            type_of_loan.isNotNull()
            & loan_type_count.isNotNull()
            & (num_of_loan.isNull() | ~num_of_loan.between(0, 9)),
            loan_type_count,
        )
        # Remaining inconsistent/unusable cases
        .otherwise(F.lit(None).cast(IntegerType())),
    )

    # ---------------------------------------------------------
    # 4. Clean Annual_Income
    # ---------------------------------------------------------
    # EDA identified a dataset-specific break around 180k together with strong inconsistency versus Monthly_Inhand_Salary above the break.
    df = df.withColumn(
        "Annual_Income_clean",
        F.when(
            F.col("Annual_Income") <= 180000,
            F.col("Annual_Income"),
        ).otherwise(F.lit(None).cast(DoubleType())),
    )

    # ---------------------------------------------------------
    # 5. Clean Total_EMI_per_month using cross-column consistency
    # ---------------------------------------------------------
    emi = F.col("Total_EMI_per_month")
    salary = F.col("Monthly_Inhand_Salary")
    num_loans = F.col("Num_of_Loan_clean")

    df = df.withColumn(
        "Total_EMI_per_month_clean",
        F.when(
            emi < 0,
            F.lit(None).cast(DoubleType()),
        )
        .when(
            (num_loans == 0) & (emi > 0),
            F.lit(None).cast(DoubleType()),
        )
        .when(
            (emi > 0)
            & (emi == F.floor(emi))
            & (emi > salary),
            F.lit(None).cast(DoubleType()),
        )
        .otherwise(emi),
    )

    # ---------------------------------------------------------
    # 6. Parse Credit_History_Age into total months
    # ---------------------------------------------------------
    history_col = F.col("Credit_History_Age")
    history_years = F.regexp_extract(
        history_col,
        r"(\d+)\s+Years?",
        1,
    ).cast(IntegerType())
    history_months = F.regexp_extract(
        history_col,
        r"(\d+)\s+Months?",
        1,
    ).cast(IntegerType())

    df = df.withColumn(
        "credit_history_age_months",
        F.when(
            history_col.rlike(
                r"^\s*\d+\s+Years?\s+and\s+\d+\s+Months?\s*$"
            ),
            history_years * 12 + history_months,
        ).otherwise(F.lit(None).cast(IntegerType())),
    )

    # ---------------------------------------------------------
    # 7. Apply EDA-derived valid ranges to remaining numeric fields
    # ---------------------------------------------------------
    # Delay_from_due_date and Changed_Credit_Limit are intentionally kept as observed because their negative values form plausible continuous ranges.
    range_rules = {
        "Num_Bank_Accounts": (0, 11),
        "Num_Credit_Card": (0, 11),
        "Interest_Rate": (0, 34),
        "Num_of_Delayed_Payment": (0, 28),
        "Num_Credit_Inquiries": (0, 17),
    }

    for column_name, (lower, upper) in range_rules.items():
        target_type = numeric_columns[column_name]
        df = df.withColumn(
            f"{column_name}_clean",
            F.when(
                F.col(column_name).between(lower, upper),
                F.col(column_name),
            ).otherwise(F.lit(None).cast(target_type)),
        )

    # ---------------------------------------------------------
    # 8. Build final Silver schema
    # ---------------------------------------------------------
    df = df.select(
        F.col("Customer_ID").cast(StringType()).alias("customer_id"),
        F.col("Annual_Income_clean").cast(DoubleType()).alias("annual_income"),
        F.col("Monthly_Inhand_Salary").cast(DoubleType()).alias("monthly_inhand_salary"),
        F.col("Num_Bank_Accounts_clean").cast(IntegerType()).alias("num_bank_accounts"),
        F.col("Num_Credit_Card_clean").cast(IntegerType()).alias("num_credit_card"),
        F.col("Interest_Rate_clean").cast(IntegerType()).alias("interest_rate"),
        F.col("Num_of_Loan_clean").cast(IntegerType()).alias("num_of_loan"),
        F.col("Type_of_Loan").cast(StringType()).alias("type_of_loan"),
        F.col("Delay_from_due_date").cast(IntegerType()).alias("delay_from_due_date"),
        F.col("Num_of_Delayed_Payment_clean").cast(IntegerType()).alias("num_of_delayed_payment"),
        F.col("Changed_Credit_Limit").cast(DoubleType()).alias("changed_credit_limit"),
        F.col("Num_Credit_Inquiries_clean").cast(IntegerType()).alias("num_credit_inquiries"),
        F.col("Credit_Mix").cast(StringType()).alias("credit_mix"),
        F.col("Outstanding_Debt").cast(DoubleType()).alias("outstanding_debt"),
        F.col("Credit_Utilization_Ratio").cast(DoubleType()).alias("credit_utilization_ratio"),
        F.col("credit_history_age_months").cast(IntegerType()).alias("credit_history_age_months"),
        F.col("Payment_of_Min_Amount").cast(StringType()).alias("payment_of_min_amount"),
        F.col("Total_EMI_per_month_clean").cast(DoubleType()).alias("total_emi_per_month"),
        F.col("Amount_invested_monthly").cast(DoubleType()).alias("amount_invested_monthly"),
        F.col("Payment_Behaviour").cast(StringType()).alias("payment_behaviour"),
        F.col("Monthly_Balance").cast(DoubleType()).alias("monthly_balance"),
        F.col("snapshot_date").cast(DateType()).alias("snapshot_date"),
    )

    # ---------------------------------------------------------
    # 9. Save Silver partition
    # ---------------------------------------------------------
    silver_partition_name = (
        "silver_feature_financials_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )
    silver_filepath = os.path.join(
        silver_financials_directory,
        silver_partition_name,
    )

    df.write.mode("overwrite").parquet(silver_filepath)

    print(
        "saved to:",
        silver_filepath,
        "row count:",
        df.count(),
    )

    return df


# Process features_clickstream silver tables for feature pipeline
def process_feature_clickstream_silver_table(
    snapshot_date_str,
    bronze_clickstream_directory,
    silver_clickstream_directory,
    spark,
):
    # Validate input date format early
    datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    os.makedirs(silver_clickstream_directory, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load Bronze snapshot as strings
    # ---------------------------------------------------------
    partition_name = (
        "bronze_feature_clickstream_"
        + snapshot_date_str.replace("-", "_")
        + ".csv"
    )

    filepath = os.path.join(bronze_clickstream_directory, partition_name)

    df = spark.read.csv(filepath, header=True, inferSchema=False)

    bronze_row_count = df.count()
    print("loaded from:", filepath, "row count:", bronze_row_count)

    if bronze_row_count == 0:
        print(
            f"Skipping {snapshot_date_str}: "
            "Bronze clickstream partition is empty."
        )
        return None
    # ---------------------------------------------------------
    # 2. Apply Silver cleaning / schema enforcement
    # ---------------------------------------------------------
    feature_cols = [f"fe_{i}" for i in range(1, 21)]

    # Parse anonymised clickstream features.
    # Negative values are intentionally retained.
    for column_name in feature_cols:
        df = df.withColumn(
            column_name,
            F.expr(
                f"try_cast(trim(`{column_name}`) as int)"
            )
        )

    # Enforce date type
    df = df.withColumn(
        "snapshot_date",
        F.to_date(F.trim(F.col("snapshot_date")), "yyyy-MM-dd")
    )

    df = df.withColumn(
        "customer_id",
        F.trim(
            F.col("Customer_ID")
            ).cast(StringType()))

    # Keep only columns required downstream
    df = df.select(
        "customer_id",
        *feature_cols,
        "snapshot_date"
    )
    
    # ---------------------------------------------------------
    # 3. Save Silver partition
    # ---------------------------------------------------------
    partition_name = (
        "silver_feature_clickstream_"
        + snapshot_date_str.replace("-", "_")
        + ".parquet"
    )

    output_path = os.path.join(
        silver_clickstream_directory,
        partition_name
    )

    df.write.mode("overwrite").parquet(output_path)

    print(
        "saved to:",
        output_path
    )
    return df