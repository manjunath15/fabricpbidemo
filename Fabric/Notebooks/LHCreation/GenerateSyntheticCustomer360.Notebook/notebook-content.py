# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "2b4692ce-465e-44d9-915b-d96710c1b0f0",
# META       "default_lakehouse_name": "Customer360",
# META       "default_lakehouse_workspace_id": "5fdce046-399f-40c6-81a5-829f33ee5e00",
# META       "known_lakehouses": [
# META         {
# META           "id": "2b4692ce-465e-44d9-915b-d96710c1b0f0"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import date

# For large ingestion, keep V-Order off for bronze/write-heavy tables.
# Turn it on later for Gold/Power BI read-heavy tables.
spark.conf.set("spark.sql.parquet.vorder.default", "false")

# Optimize write helps reduce small files.
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")

# Adaptive execution helps large joins/aggregations.
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Tune this based on capacity size.
spark.conf.set("spark.sql.shuffle.partitions", "800")

START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"
DAYS = 1096



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Scale Settings

# CELL ********************

# ===== SCALE SETTINGS =====
# For quick test
CUSTOMER_ROWS = 10_000_000
PRODUCT_ROWS  = 2_000_000

# Increase these for full 1 TB target.
# Actual compressed Delta size depends on cardinality, columns and compression.
TRANSACTION_ROWS_TOTAL = 1_500_000_000
ORDER_ROWS_TOTAL       =   600_000_000
CLICK_ROWS_TOTAL       = 1_000_000_000
CASE_ROWS_TOTAL        =    80_000_000
REVIEW_ROWS_TOTAL      =    80_000_000

# Batch size: keep manageable for your capacity.
BATCH_ROWS = 100_000_000

# Partition count per batch.
RANGE_PARTITIONS = 800

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lookup values

# CELL ********************

countries = [
    ("USA", "North America"),
    ("Canada", "North America"),
    ("UK", "Europe"),
    ("Germany", "Europe"),
    ("France", "Europe"),
    ("India", "Asia"),
    ("Japan", "Asia"),
    ("Singapore", "Asia"),
    ("Australia", "Oceania"),
    ("Brazil", "South America"),
    ("Mexico", "North America"),
    ("South Africa", "Africa")
]

channels = ["Web", "Mobile App", "Store", "Partner", "Call Center"]
segments = ["Bronze", "Silver", "Gold", "Platinum", "VIP"]
product_categories = [
    "Electronics", "Banking", "Insurance", "Retail",
    "Healthcare", "Travel", "Telecom", "Manufacturing"
]

df_country = spark.createDataFrame(countries, ["country", "region"])
df_country.write.mode("overwrite").format("delta").saveAsTable("dbo.dim_country")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Date dimension

# CELL ********************

from pyspark.sql import functions as F
df_date = (
    spark.range(0, DAYS)
    .withColumn("date", F.date_add(F.to_date(F.lit(START_DATE)), F.col("id").cast("int")))
    .withColumn("date_key", F.date_format("date", "yyyyMMdd").cast("int"))
    .withColumn("year", F.year("date"))
    .withColumn("quarter", F.quarter("date"))
    .withColumn("month", F.month("date"))
    .withColumn("month_name", F.date_format("date", "MMMM"))
    .withColumn("day_of_month", F.dayofmonth("date"))
    .withColumn("day_of_week", F.date_format("date", "EEEE"))
    .drop("id")
)

df_date.write.mode("overwrite").format("delta").saveAsTable("dbo.dim_date")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Customer Dimension

# CELL ********************

country_array = F.array(*[F.lit(c[0]) for c in countries])
segment_array = F.array(*[F.lit(s) for s in segments])

df_customer = (
    spark.range(1, CUSTOMER_ROWS + 1, 1, RANGE_PARTITIONS)
    .withColumnRenamed("id", "customer_id")
    .withColumn("customer_number", F.concat(F.lit("CUST-"), F.lpad(F.col("customer_id").cast("string"), 12, "0")))
    .withColumn("country", F.element_at(country_array, (F.pmod(F.col("customer_id"), F.lit(len(countries))) + 1).cast("int")))
    .withColumn("age", (F.pmod(F.col("customer_id") * 13, F.lit(58)) + 18).cast("int"))
    .withColumn(
        "age_group",
        F.when(F.col("age") <= 25, "18-25")
         .when(F.col("age") <= 35, "26-35")
         .when(F.col("age") <= 45, "36-45")
         .when(F.col("age") <= 55, "46-55")
         .when(F.col("age") <= 65, "56-65")
         .otherwise("65+")
    )
    .withColumn("gender", F.expr("CASE pmod(customer_id,3) WHEN 0 THEN 'Male' WHEN 1 THEN 'Female' ELSE 'Other' END"))
    .withColumn("segment", F.element_at(segment_array, (F.pmod(F.col("customer_id"), F.lit(len(segments))) + 1).cast("int")))
    .withColumn("signup_date", F.date_add(F.to_date(F.lit("2020-01-01")), F.pmod(F.col("customer_id"), F.lit(1460)).cast("int")))
    .withColumn("email_hash", F.sha2(F.concat(F.lit("customer"), F.col("customer_id")), 256))
)

df_customer.write.mode("overwrite").format("delta").partitionBy("country").saveAsTable("dbo.dim_customer")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Product Dimension

# CELL ********************

category_array = F.array(*[F.lit(c) for c in product_categories])

df_product = (
    spark.range(1, PRODUCT_ROWS + 1, 1, 400)
    .withColumnRenamed("id", "product_id")
    .withColumn("product_sku", F.concat(F.lit("SKU-"), F.lpad(F.col("product_id").cast("string"), 10, "0")))
    .withColumn("category", F.element_at(category_array, (F.pmod(F.col("product_id"), F.lit(len(product_categories))) + 1).cast("int")))
    .withColumn("brand", F.concat(F.lit("Brand-"), F.pmod(F.col("product_id"), F.lit(500)).cast("string")))
    .withColumn("base_price", (F.pmod(F.col("product_id") * 17, F.lit(50000)) / 10 + 100).cast("decimal(12,2)"))
)

df_product.write.mode("overwrite").format("delta").partitionBy("category").saveAsTable("dbo.dim_product")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Fact Customer Transactions

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Fact Customer Transaction

# CELL ********************

from pyspark.sql import functions as F

TRANSACTION_ROWS_TOTAL = 100_000_000
BATCH_ROWS = 10_000_000
RANGE_PARTITIONS = 400

countries = [
    "USA","Canada","UK","Germany","France",
    "India","Japan","Singapore","Australia",
    "Brazil","Mexico","South Africa"
]

channels = [
    "Web",
    "Mobile",
    "Store",
    "Partner",
    "CallCenter"
]

country_array = F.array(*[F.lit(x) for x in countries])
channel_array = F.array(*[F.lit(x) for x in channels])

def build_transactions(start_id, end_id):

    return (
        spark.range(start_id, end_id, 1, RANGE_PARTITIONS)

        .withColumnRenamed("id","transaction_id")

        .withColumn(
            "customer_id",
            (F.pmod(F.col("transaction_id") * 37,
                    F.lit(CUSTOMER_ROWS)) + 1).cast("long")
        )

        .withColumn(
            "product_id",
            (F.pmod(F.col("transaction_id") * 19,
                    F.lit(PRODUCT_ROWS)) + 1).cast("long")
        )

        .withColumn(
            "transaction_date",
            F.date_add(
                F.to_date(F.lit(START_DATE)),
                F.pmod(F.col("transaction_id"),F.lit(DAYS)).cast("int")
            )
        )

        .withColumn("year",F.year("transaction_date"))
        .withColumn("month",F.month("transaction_date"))

        .withColumn(
            "country",
            F.element_at(
                country_array,
                (F.pmod(F.col("customer_id"),
                        F.lit(len(countries))) + 1).cast("int")
            )
        )

        .withColumn(
            "channel",
            F.element_at(
                channel_array,
                (F.pmod(F.col("transaction_id"),
                        F.lit(len(channels))) + 1).cast("int")
            )
        )

        .withColumn(
            "quantity",
            (F.pmod(F.col("transaction_id"),5)+1).cast("int")
        )

        .withColumn(
            "unit_price",
            (
                (
                    F.pmod(
                        F.col("product_id") * 17,
                        10000
                    ) / 10
                ) + 50
            ).cast("decimal(18,2)")
        )

        .withColumn(
            "gross_amount",
            (
                F.col("quantity") *
                F.col("unit_price")
            ).cast("decimal(18,2)")
        )

        .withColumn(
            "discount_pct",
            (
                F.pmod(
                    F.col("transaction_id"),
                    20
                ) / 100
            ).cast("decimal(5,2)")
        )

        .withColumn(
            "net_amount",
            (
                F.col("gross_amount")
                *
                (1 - F.col("discount_pct"))
            ).cast("decimal(18,2)")
        )

        .withColumn(
            "tax_amount",
            (
                F.col("net_amount") * 0.18
            ).cast("decimal(18,2)")
        )

        .withColumn(
            "total_amount",
            (
                F.col("net_amount")
                +
                F.col("tax_amount")
            ).cast("decimal(18,2)")
        )

        .withColumn(
            "payment_type",
            F.expr("""
                CASE
                WHEN pmod(transaction_id,4)=0 THEN 'Credit Card'
                WHEN pmod(transaction_id,4)=1 THEN 'UPI'
                WHEN pmod(transaction_id,4)=2 THEN 'Wallet'
                ELSE 'Bank Transfer'
                END
            """)
        )
    )


rows_written = 0
batch_no = 0

while rows_written < TRANSACTION_ROWS_TOTAL:

    current_batch = min(
        BATCH_ROWS,
        TRANSACTION_ROWS_TOTAL - rows_written
    )

    start_id = rows_written + 1
    end_id = rows_written + current_batch + 1

    print(
        f"Batch {batch_no}, rows {current_batch:,}"
    )

    df=batch_df=build_transactions(
        start_id,
        end_id
    )

    if batch_no == 0:

        (
            df.write
            .format("delta")
            .mode("overwrite")
            .partitionBy(
                "year",
                "month",
                "country"
            )
            .saveAsTable(
                "dbo.fact_customer_transactions"
            )
        )

    else:

        (
            df.write
            .format("delta")
            .mode("append")
            .insertInto(
                "dbo.fact_customer_transactions"
            )
        )

    rows_written += current_batch
    batch_no += 1

print("Completed")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def build_transactions(start_id, end_id):
    return (
        spark.range(start_id, end_id, 1, RANGE_PARTITIONS)
        .withColumnRenamed("id", "transaction_id")
        .withColumn("customer_id", (F.pmod(F.col("transaction_id") * 37, F.lit(CUSTOMER_ROWS)) + 1).cast("long"))
        .withColumn("product_id", (F.pmod(F.col("transaction_id") * 19, F.lit(PRODUCT_ROWS)) + 1).cast("long"))
        .withColumn("transaction_date", F.date_add(F.to_date(F.lit(START_DATE)), F.pmod(F.col("transaction_id"), F.lit(DAYS)).cast("int")))
        .withColumn("year", F.year("transaction_date"))
        .withColumn("month", F.month("transaction_date"))
        .withColumn("country", F.element_at(country_array, (F.pmod(F.col("customer_id"), F.lit(len(countries))) + 1).cast("int")))
        .withColumn("channel", F.element_at(channel_array, (F.pmod(F.col("transaction_id"), F.lit(len(channels))) + 1).cast("int")))
        .withColumn("quantity", (F.pmod(F.col("transaction_id"), F.lit(5)) + 1).cast("int"))
        .withColumn("unit_price", ((F.pmod(F.col("product_id") * 23, F.lit(100000)) / 100) + 50).cast("decimal(12,2)"))
        .withColumn("discount_pct", (F.pmod(F.col("transaction_id"), F.lit(30)) / 100).cast("decimal(5,2)"))
        .withColumn("gross_amount", (F.col("quantity") * F.col("unit_price")).cast("decimal(18,2)"))
        .withColumn("net_amount", (F.col("gross_amount") * (1 - F.col("discount_pct"))).cast("decimal(18,2)"))
        .withColumn("tax_amount", (F.col("net_amount") * F.lit(0.18)).cast("decimal(18,2)"))
        .withColumn("total_amount", (F.col("net_amount") + F.col("tax_amount")).cast("decimal(18,2)"))
        .withColumn("payment_type", F.expr("CASE pmod(transaction_id,4) WHEN 0 THEN 'Credit Card' WHEN 1 THEN 'UPI' WHEN 2 THEN 'Wallet' ELSE 'Bank Transfer' END"))
        .withColumn("transaction_hash", F.sha2(F.concat_ws("|", F.col("transaction_id"), F.col("customer_id"), F.col("product_id"), F.col("total_amount")), 256))
    )

write_delta_in_batches(
    TRANSACTION_ROWS_TOTAL,
    BATCH_ROWS,
    build_transactions,
    "dbo.fact_customer_transactions",
    ["year", "month", "country"]
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
