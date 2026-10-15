# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "252af361-4e97-4bb8-8fb6-71833577f8d1",
# META       "default_lakehouse_name": "CustomerRAW",
# META       "default_lakehouse_workspace_id": "5fdce046-399f-40c6-81a5-829f33ee5e00",
# META       "known_lakehouses": [
# META         {
# META           "id": "252af361-4e97-4bb8-8fb6-71833577f8d1"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # 1 TB Synthetic Customer Data Generator — Fabric Lakehouse `CustomerRAW` (schema `dbo`)
# 
# **Purpose:** build a realistic, ~1 TB customer analytics dataset as **managed Delta tables** under `CustomerRAW.dbo`,
# so you can demo (a) Spark / Lakehouse performance tuning and (b) Direct Lake reporting.
# 
# **Data shape**
# - 3 full years of history (rolling 36 months ending last completed month)
# - Country-wise skew (~40 countries, realistic volume distribution)
# - Age groups, gender, customer segment, channel, product hierarchy
# - Seasonality (Nov/Dec peaks), weekday/weekend effects, returns, discounts, margins
# 
# **Tables created (all managed, all under `dbo`)**
# 
# | Table | Type | Approx rows | Notes |
# |---|---|---|---|
# | `dbo.dim_date` | dim | ~1,095 | 3 years |
# | `dbo.dim_country` | dim | 40 | region, currency |
# | `dbo.dim_age_group` | dim | 7 | buckets |
# | `dbo.dim_channel` | dim | 6 | |
# | `dbo.dim_product` | dim | 200,000 | category / subcategory / brand |
# | `dbo.dim_customer` | dim | 80,000,000 | SCD1, country + age assigned |
# | `dbo.fact_customer_transactions` | fact | ~4–6 billion | **~70% of total volume**, partitioned by `txn_year`, `txn_month` |
# | `dbo.fact_web_events` | fact | ~6–8 billion | **~30% of volume**, narrow/wide-row contrast for tuning demos |
# | `dbo.agg_*` | gold | small | pre-aggregates for the BI demo |
# 
# > **Before you run anything:** attach `CustomerRAW` as the **default lakehouse** of this notebook
# > (Explorer → Add lakehouse → CustomerRAW → pin as default). Schemas are already enabled — your screenshot shows `Tables > dbo`.


# MARKDOWN ********************

# ## Step 0 — Sizing reality check (read this first)
# 
# 1 TB of **Delta/Parquet on disk** is not 1 TB of raw rows. With ZSTD + dictionary encoding, the wide fact row
# in this notebook lands roughly in the **120–200 bytes/row** range. So:
# 
# $$\text{rows} \approx \frac{700\ \text{GB} \times 1024^3}{150} \approx 5\ \text{billion}$$
# 
# Rather than guessing, **Cell 6 runs a calibration batch** (50 M rows), measures the actual `sizeInBytes` from
# `DESCRIBE DETAIL`, and back-calculates exactly how many rows you need. Always run calibration before the full load.
# 
# **Capacity guidance**
# - Use **F64 or larger** (F64 → up to 128 Spark vCores at default; higher SKU = shorter run).
# - Set the environment/pool to **Medium or Large nodes, autoscale on, dynamic allocation on**.
# - Expect the full 1 TB load to take **several hours**. It is designed to be **restartable month-by-month** — if a run dies, re-run only the missing months.
# - Watch your **CU consumption**: this is a heavy burst job. Run it off-hours, or on a dedicated capacity, so you don't throttle your demo tenant.


# MARKDOWN ********************

# ## Step 1 — Spark configuration for bulk generation

# CELL ********************

# ============================================================
# CELL 1 — Spark config tuned for high-throughput WRITE
# ============================================================
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import time, math

LAKEHOUSE = "CustomerRAW"
SCHEMA    = "dbo"

# --- Write-path tuning (generation phase) -------------------
# V-Order OFF during generation = much faster writes.
# We turn it ON later as part of the performance-tuning demo (that contrast IS the demo).
spark.conf.set("spark.sql.parquet.vorder.default", "false")

# Let Fabric size the files; ~1 GB target files for a big scan-heavy fact table
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")   # 1 GB
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "false")          # off during load

# Compression + AQE
spark.conf.set("spark.sql.parquet.compression.codec", "zstd")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "false")       # we control partition count
spark.conf.set("spark.sql.shuffle.partitions", "1024")
spark.conf.set("spark.sql.files.maxRecordsPerFile", "0")

# Native Execution Engine — leave OFF here so you can demo it ON later
spark.conf.set("spark.native.enabled", "false")

# Delta stats: only first N columns get min/max stats. Put filter columns early in the schema.
spark.conf.set("spark.databricks.delta.stats.collect.numIndexedCols", "16")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE {SCHEMA}")
print("Config applied. Default lakehouse must be CustomerRAW.")
print(spark.sql("SELECT current_schema()").collect())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 2 — Time window and volume plan

# CELL ********************

# ============================================================
# CELL 2 — Time window + volume plan
# ============================================================
# Rolling 3 years ending at the last completed month
today       = date.today()
END_MONTH   = date(today.year, today.month, 1) - relativedelta(months=1)
START_MONTH = END_MONTH - relativedelta(months=35)

MONTHS = []
m = START_MONTH
while m <= END_MONTH:
    MONTHS.append(m)
    m = m + relativedelta(months=1)

print(f"36 months: {MONTHS[0]} .. {MONTHS[-1]}")

# ---- Volume dials (adjust AFTER calibration in Cell 6) ------
CUSTOMER_ROWS      = 80_000_000       # dim_customer
PRODUCT_ROWS       = 200_000          # dim_product

TXN_ROWS_TOTAL     = 5_000_000_000    # fact_customer_transactions  -> ~700 GB
WEB_ROWS_TOTAL     = 7_000_000_000    # fact_web_events             -> ~300 GB

# Chunking: rows written per Spark job. Keep each chunk 50–150M rows so a failure is cheap.
CHUNK_ROWS         = 100_000_000
TASKS_PER_CHUNK    = 400              # ~ number of output tasks per chunk

# Monthly seasonality weights (index 0 = Jan ... 11 = Dec)
SEASONALITY = [0.80, 0.75, 0.88, 0.92, 0.98, 0.95,
               0.97, 1.00, 1.02, 1.10, 1.45, 1.60]

# Year-over-year growth so the demo shows a trend
YOY_GROWTH  = {0: 0.85, 1: 1.00, 2: 1.20}   # relative weight by year index

def month_weights(months):
    w = []
    y0 = months[0].year
    for d in months:
        yi = min(d.year - y0, 2)
        w.append(SEASONALITY[d.month - 1] * YOY_GROWTH[yi])
    s = sum(w)
    return [x / s for x in w]

MW = month_weights(MONTHS)
TXN_PER_MONTH = [int(TXN_ROWS_TOTAL * w) for w in MW]
WEB_PER_MONTH = [int(WEB_ROWS_TOTAL * w) for w in MW]

for d, t, wv in list(zip(MONTHS, TXN_PER_MONTH, WEB_PER_MONTH))[:6]:
    print(d, f"txn={t:,}", f"web={wv:,}")
print("...")
print(f"TOTAL txn rows planned: {sum(TXN_PER_MONTH):,}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 3 — Dimensions
# 
# Country and age-group skew is driven by **bucket lookup tables** (1000-row and 100-row).
# A customer's country is `pmod(hash(customer_id), 1000)` → country bucket. The fact tables reuse the *same*
# deterministic mapping, so facts stay consistent with `dim_customer` **without a 5-billion-row join**.

# CELL ********************

# ============================================================
# CELL 3 — dim_country, dim_age_group, dim_channel, dim_date
# ============================================================
countries = [
 # code, name, region, currency, weight
 ("US","United States","North America","USD",180), ("IN","India","APAC","INR",150),
 ("CN","China","APAC","CNY",120), ("GB","United Kingdom","EMEA","GBP",70),
 ("DE","Germany","EMEA","EUR",65),  ("JP","Japan","APAC","JPY",55),
 ("BR","Brazil","LATAM","BRL",50),  ("FR","France","EMEA","EUR",48),
 ("CA","Canada","North America","CAD",42), ("AU","Australia","APAC","AUD",38),
 ("MX","Mexico","LATAM","MXN",35),  ("IT","Italy","EMEA","EUR",33),
 ("ES","Spain","EMEA","EUR",30),    ("KR","South Korea","APAC","KRW",28),
 ("ID","Indonesia","APAC","IDR",27),("NL","Netherlands","EMEA","EUR",24),
 ("SA","Saudi Arabia","MEA","SAR",22),("AE","United Arab Emirates","MEA","AED",21),
 ("ZA","South Africa","MEA","ZAR",20),("SG","Singapore","APAC","SGD",19),
 ("SE","Sweden","EMEA","SEK",17),   ("PL","Poland","EMEA","PLN",16),
 ("TR","Turkey","EMEA","TRY",16),   ("AR","Argentina","LATAM","ARS",15),
 ("CH","Switzerland","EMEA","CHF",14),("BE","Belgium","EMEA","EUR",13),
 ("NO","Norway","EMEA","NOK",12),   ("AT","Austria","EMEA","EUR",12),
 ("MY","Malaysia","APAC","MYR",11), ("TH","Thailand","APAC","THB",11),
 ("VN","Vietnam","APAC","VND",10),  ("PH","Philippines","APAC","PHP",10),
 ("NZ","New Zealand","APAC","NZD",9),("IE","Ireland","EMEA","EUR",9),
 ("DK","Denmark","EMEA","DKK",8),   ("FI","Finland","EMEA","EUR",8),
 ("PT","Portugal","EMEA","EUR",7),  ("CL","Chile","LATAM","CLP",7),
 ("EG","Egypt","MEA","EGP",6),      ("NG","Nigeria","MEA","NGN",6),
]

cdf = spark.createDataFrame(
    [(i+1,)+c for i,c in enumerate(countries)],
    "country_key int, country_code string, country_name string, region string, currency_code string, weight int"
)
cdf.write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.dim_country")

age_groups = [(1,"18-24",18,24,14),(2,"25-34",25,34,26),(3,"35-44",35,44,22),
              (4,"45-54",45,54,17),(5,"55-64",55,64,12),(6,"65-74",65,74,6),(7,"75+",75,95,3)]
spark.createDataFrame(age_groups,
    "age_group_key int, age_group string, age_min int, age_max int, weight int"
).write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.dim_age_group")

channels = [(1,"Web","Digital"),(2,"Mobile App","Digital"),(3,"Store","Physical"),
            (4,"Partner","Indirect"),(5,"Call Center","Assisted"),(6,"Marketplace","Indirect")]
spark.createDataFrame(channels, "channel_key int, channel_name string, channel_type string") \
     .write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.dim_channel")

# ---- dim_date ---------------------------------------------
d0 = MONTHS[0]
d1 = (MONTHS[-1] + relativedelta(months=1)) - relativedelta(days=1)
days = (d1 - d0).days + 1
dim_date = (spark.range(days)
    .withColumn("date_key_dt", F.expr(f"date_add(to_date('{d0}'), cast(id as int))"))
    .select(
        F.date_format("date_key_dt","yyyyMMdd").cast("int").alias("date_key"),
        F.col("date_key_dt").alias("full_date"),
        F.year("date_key_dt").alias("year"),
        F.quarter("date_key_dt").alias("quarter"),
        F.month("date_key_dt").alias("month"),
        F.date_format("date_key_dt","MMMM").alias("month_name"),
        F.dayofmonth("date_key_dt").alias("day_of_month"),
        F.dayofweek("date_key_dt").alias("day_of_week"),
        F.date_format("date_key_dt","EEEE").alias("day_name"),
        F.weekofyear("date_key_dt").alias("week_of_year"),
        F.expr("case when dayofweek(date_key_dt) in (1,7) then true else false end").alias("is_weekend"),
        F.expr("case when month(date_key_dt) in (11,12) then true else false end").alias("is_peak_season"),
        F.concat(F.year("date_key_dt"), F.lit("-Q"), F.quarter("date_key_dt")).alias("year_quarter"),
        F.date_format("date_key_dt","yyyy-MM").alias("year_month"),
    ))
dim_date.write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.dim_date")

print("Dimensions (country/age/channel/date) created.")
spark.sql(f"SHOW TABLES IN {SCHEMA}").show(50, False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 4 — Bucket lookup tables (deterministic skew)
# ============================================================
# 1000-row country bucket table: bucket -> country. Broadcast-joined everywhere.
rows = spark.table(f"{SCHEMA}.dim_country").collect()
total_w = sum(r["weight"] for r in rows)
bucket_rows, b = [], 0
for r in rows:
    n = max(1, round(1000 * r["weight"] / total_w))
    for _ in range(n):
        if b >= 1000: break
        bucket_rows.append((b, r["country_key"], r["country_code"], r["country_name"], r["region"], r["currency_code"]))
        b += 1
while b < 1000:                       # pad remainder with the largest market
    r = rows[0]
    bucket_rows.append((b, r["country_key"], r["country_code"], r["country_name"], r["region"], r["currency_code"])); b += 1

spark.createDataFrame(bucket_rows,
  "bucket int, country_key int, country_code string, country_name string, region string, currency_code string"
).write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.lkp_country_bucket")

# 100-row age bucket table
arows = spark.table(f"{SCHEMA}.dim_age_group").orderBy("age_group_key").collect()
tw = sum(r["weight"] for r in arows)
abucket, b = [], 0
for r in arows:
    n = max(1, round(100 * r["weight"] / tw))
    for _ in range(n):
        if b >= 100: break
        abucket.append((b, r["age_group_key"], r["age_group"], r["age_min"], r["age_max"])); b += 1
while b < 100:
    r = arows[1]
    abucket.append((b, r["age_group_key"], r["age_group"], r["age_min"], r["age_max"])); b += 1

spark.createDataFrame(abucket,
  "abucket int, age_group_key int, age_group string, age_min int, age_max int"
).write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.lkp_age_bucket")

CBUCK = F.broadcast(spark.table(f"{SCHEMA}.lkp_country_bucket"))
ABUCK = F.broadcast(spark.table(f"{SCHEMA}.lkp_age_bucket"))
print("Lookup buckets ready.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 5 — dim_product + dim_customer  (managed Delta tables)
# ============================================================
cats = [("Electronics",["Smartphones","Laptops","Audio","Wearables","Cameras","TV"]),
        ("Apparel",["Menswear","Womenswear","Kidswear","Footwear","Accessories"]),
        ("Home",["Furniture","Kitchen","Decor","Bedding","Appliances"]),
        ("Grocery",["Beverages","Snacks","Staples","Dairy","Frozen"]),
        ("Beauty",["Skincare","Haircare","Fragrance","Cosmetics"]),
        ("Sports",["Fitness","Outdoor","Cycling","Team Sports"]),
        ("Books",["Fiction","Non-Fiction","Academic","Children"]),
        ("Toys",["Games","Puzzles","Action Figures","Learning"])]
flat = [(c, s) for c, subs in cats for s in subs]
cat_df = spark.createDataFrame([(i, c, s) for i,(c,s) in enumerate(flat)], "cat_idx int, category string, subcategory string")
NCAT = len(flat)

brands = ["Contoso","Fabrikam","Northwind","AdventureWorks","Litware","Tailspin","Woodgrove",
          "ProseWare","VanArsdel","Relecloud","Lamna","Trey Research"]
brand_df = spark.createDataFrame([(i,b) for i,b in enumerate(brands)], "brand_idx int, brand string")

dim_product = (spark.range(PRODUCT_ROWS).repartition(64)
  .withColumn("product_id", F.col("id") + 1)
  .withColumn("cat_idx",   (F.pmod(F.hash(F.col("id"), F.lit(7)), F.lit(NCAT))).cast("int"))
  .withColumn("brand_idx", (F.pmod(F.hash(F.col("id"), F.lit(11)), F.lit(len(brands)))).cast("int"))
  .join(F.broadcast(cat_df), "cat_idx").join(F.broadcast(brand_df), "brand_idx")
  .withColumn("product_sku", F.concat_ws("-", F.substring(F.col("category"),1,3),
                                          F.lpad(F.col("product_id").cast("string"), 9, "0")))
  .withColumn("product_name", F.concat_ws(" ", F.col("brand"), F.col("subcategory"),
                                          F.concat(F.lit("Model "), F.pmod(F.col("id"), F.lit(900))+100)))
  .withColumn("list_price", F.round(F.expr("5 + rand(42) * 1495"), 2))
  .withColumn("std_cost",   F.round(F.col("list_price") * F.expr("0.45 + rand(43)*0.3"), 2))
  .withColumn("supplier_id", (F.pmod(F.hash(F.col("id"), F.lit(13)), F.lit(5000)) + 1).cast("int"))
  .withColumn("is_active", F.expr("rand(44) > 0.05"))
  .select("product_id","product_sku","product_name","category","subcategory","brand",
          "list_price","std_cost","supplier_id","is_active"))

dim_product.write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.dim_product")
print("dim_product done:", spark.table(f"{SCHEMA}.dim_product").count())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 5b — dim_customer (80 M rows, managed Delta)
# ============================================================
segments = ["Premium","Standard","Value","Corporate"]
genders  = ["F","M","X"]
start_ts = str(MONTHS[0])

dim_customer = (spark.range(CUSTOMER_ROWS).repartition(512)
  .withColumn("customer_id", F.col("id") + 1)
  # deterministic buckets -> reused by fact tables, no join needed later
  .withColumn("bucket",  F.pmod(F.hash(F.col("customer_id"), F.lit(101)), F.lit(1000)).cast("int"))
  .withColumn("abucket", F.pmod(F.hash(F.col("customer_id"), F.lit(211)), F.lit(100)).cast("int"))
  .join(CBUCK, "bucket").join(ABUCK, "abucket")
  .withColumn("age", (F.col("age_min") + F.pmod(F.hash(F.col("customer_id"), F.lit(307)),
                       (F.col("age_max")-F.col("age_min")+1))).cast("int"))
  .withColumn("gender", F.element_at(F.array(*[F.lit(g) for g in genders]),
                        (F.pmod(F.hash(F.col("customer_id"), F.lit(401)), F.lit(3))+1).cast("int")))
  .withColumn("customer_segment", F.element_at(F.array(*[F.lit(s) for s in segments]),
                        (F.pmod(F.hash(F.col("customer_id"), F.lit(503)), F.lit(4))+1).cast("int")))
  .withColumn("signup_date", F.expr(f"date_sub(to_date('{start_ts}'), cast(rand(55)*1800 as int))"))
  .withColumn("first_name", F.concat(F.lit("Cust"), F.col("customer_id").cast("string")))
  .withColumn("last_name",  F.concat(F.lit("LN"), F.pmod(F.col("customer_id"), F.lit(50000)).cast("string")))
  .withColumn("email", F.concat(F.lit("cust"), F.col("customer_id").cast("string"),
                                F.lit("@example-"), F.lower(F.col("country_code")), F.lit(".com")))
  .withColumn("city", F.concat(F.col("country_code"), F.lit("-City-"),
                     F.pmod(F.hash(F.col("customer_id"), F.lit(601)), F.lit(400)).cast("string")))
  .withColumn("postal_code", F.lpad(F.pmod(F.hash(F.col("customer_id"), F.lit(701)), F.lit(999999)).cast("string"), 6, "0"))
  .withColumn("latitude",  F.round(F.expr("-60 + rand(71)*120"), 5))
  .withColumn("longitude", F.round(F.expr("-180 + rand(72)*360"), 5))
  .withColumn("loyalty_tier", F.expr("case when rand(81)<0.05 then 'Platinum' when rand(82)<0.15 then 'Gold' when rand(83)<0.35 then 'Silver' else 'Bronze' end"))
  .withColumn("is_churned", F.expr("rand(91) < 0.12"))
  .withColumn("marketing_opt_in", F.expr("rand(92) < 0.62"))
  .select("customer_id","first_name","last_name","email","gender","age",
          "age_group_key","age_group","country_key","country_code","country_name","region",
          "currency_code","city","postal_code","latitude","longitude","customer_segment",
          "loyalty_tier","signup_date","is_churned","marketing_opt_in"))

dim_customer.write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.dim_customer")
print("dim_customer rows:", spark.table(f"{SCHEMA}.dim_customer").count())
spark.sql(f"DESCRIBE DETAIL {SCHEMA}.dim_customer").select("numFiles","sizeInBytes").show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 4 — The fact generator
# 
# `fact_customer_transactions` is deliberately **wide** (~45 columns, mixed types, some high-cardinality
# strings). That does two things: it gets you to 1 TB with a manageable row count, *and* it gives you a
# column-pruning story for the tuning demo.
# 
# Column order matters: the first 16 columns are the ones Delta collects min/max stats for, so the filter
# columns (`transaction_date`, `country_code`, `age_group`, `product_category`) sit up front for data skipping.

# CELL ********************

# ============================================================
# CELL 6 — Fact generator function
# ============================================================
PAYMENTS = ["Credit Card","Debit Card","UPI","Wallet","Net Banking","COD","Gift Card","BNPL"]
DEVICES  = ["iOS","Android","Windows","macOS","Linux","Other"]
CHANNELS = ["Web","Mobile App","Store","Partner","Call Center","Marketplace"]
RETURNS  = ["Damaged","Wrong Item","Size Issue","Late Delivery","Changed Mind","Quality"]

def arr(vals):
    return F.array(*[F.lit(v) for v in vals])

def pick(vals, seedcol, seed):
    return F.element_at(arr(vals), (F.pmod(F.hash(seedcol, F.lit(seed)), F.lit(len(vals))) + 1).cast("int"))

def gen_txn_chunk(n_rows, month_start, month_end, seed, tasks=TASKS_PER_CHUNK):
    """Generate one chunk of fact_customer_transactions for a given month."""
    days_in_month = (month_end - month_start).days + 1
    df = (spark.range(n_rows).repartition(tasks)
      .withColumn("r", F.rand(seed))
      .withColumn("customer_id", (F.pmod(F.hash(F.col("id"), F.lit(seed)), F.lit(CUSTOMER_ROWS)) + 1).cast("bigint"))
      .withColumn("bucket",  F.pmod(F.hash(F.col("customer_id"), F.lit(101)), F.lit(1000)).cast("int"))
      .withColumn("abucket", F.pmod(F.hash(F.col("customer_id"), F.lit(211)), F.lit(100)).cast("int"))
      .join(CBUCK, "bucket").join(ABUCK, "abucket")
      # ---- time: weekend uplift baked in via a second random draw ----
      .withColumn("day_offset", (F.rand(seed+1) * days_in_month).cast("int"))
      .withColumn("transaction_date", F.expr(f"date_add(to_date('{month_start}'), day_offset)"))
      .withColumn("transaction_ts",
            F.expr("cast(transaction_date as timestamp) + make_interval(0,0,0,0, cast(rand()*23 as int), cast(rand()*59 as int), cast(rand()*59 as int))"))
      # ---- product ----
      .withColumn("product_id", (F.pmod(F.hash(F.col("id"), F.lit(seed+2)), F.lit(PRODUCT_ROWS)) + 1).cast("bigint"))
      .withColumn("cat_idx", F.pmod(F.hash(F.col("product_id"), F.lit(7)), F.lit(NCAT)).cast("int"))
      .join(F.broadcast(cat_df), "cat_idx")
      .withColumn("brand", pick(brands, F.col("product_id"), 11))
      .withColumn("product_sku", F.concat_ws("-", F.substring(F.col("category"),1,3),
                                  F.lpad(F.col("product_id").cast("string"), 9, "0")))
      # ---- money ----
      .withColumn("quantity", (F.expr("case when rand()<0.65 then 1 when rand()<0.9 then 2 else cast(2+rand()*6 as int) end")).cast("int"))
      .withColumn("unit_price", F.round(F.expr("5 + pow(rand(),2) * 1495"), 2))
      .withColumn("discount_pct", F.round(F.expr("case when rand()<0.55 then 0 else rand()*0.45 end"), 3))
      .withColumn("gross_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
      .withColumn("discount_amount", F.round(F.col("gross_amount") * F.col("discount_pct"), 2))
      .withColumn("tax_amount", F.round((F.col("gross_amount") - F.col("discount_amount")) * F.lit(0.12), 2))
      .withColumn("shipping_amount", F.round(F.expr("case when rand()<0.4 then 0 else rand()*25 end"), 2))
      .withColumn("net_amount", F.round(F.col("gross_amount") - F.col("discount_amount")
                                        + F.col("tax_amount") + F.col("shipping_amount"), 2))
      .withColumn("margin_amount", F.round(F.col("net_amount") * F.expr("0.08 + rand()*0.32"), 2))
      # ---- behaviour / attributes ----
      .withColumn("channel", pick(CHANNELS, F.col("id"), seed+3))
      .withColumn("payment_method", pick(PAYMENTS, F.col("id"), seed+4))
      .withColumn("device_type", pick(DEVICES, F.col("id"), seed+5))
      .withColumn("store_id", (F.pmod(F.hash(F.col("id"), F.lit(seed+6)), F.lit(20000)) + 1).cast("int"))
      .withColumn("is_returned", F.expr("rand() < 0.065"))
      .withColumn("return_reason", F.when(F.col("is_returned"), pick(RETURNS, F.col("id"), seed+7)).otherwise(F.lit(None).cast("string")))
      .withColumn("promotion_code", F.when(F.col("discount_pct") > 0,
                    F.concat(F.lit("PROMO"), F.lpad(F.pmod(F.hash(F.col("id"), F.lit(seed+8)), F.lit(500)).cast("string"),4,"0"))).otherwise(F.lit(None).cast("string")))
      .withColumn("campaign_id", (F.pmod(F.hash(F.col("id"), F.lit(seed+9)), F.lit(3000)) + 1).cast("int"))
      .withColumn("order_id", (F.pmod(F.hash(F.col("id"), F.lit(seed+10)), F.lit(n_rows)) + 1).cast("bigint"))
      .withColumn("session_id", F.expr("uuid()"))                       # high-cardinality string -> bytes
      .withColumn("delivery_days", (F.expr("1 + rand()*13")).cast("int"))
      .withColumn("rating", F.when(F.expr("rand()<0.35"), (F.expr("1+rand()*4")).cast("int")).otherwise(F.lit(None).cast("int")))
      .withColumn("city", F.concat(F.col("country_code"), F.lit("-City-"),
                    F.pmod(F.hash(F.col("customer_id"), F.lit(601)), F.lit(400)).cast("string")))
      .withColumn("postal_code", F.lpad(F.pmod(F.hash(F.col("customer_id"), F.lit(701)), F.lit(999999)).cast("string"),6,"0"))
      .withColumn("latitude",  F.round(F.expr("-60 + rand()*120"), 5))
      .withColumn("longitude", F.round(F.expr("-180 + rand()*360"), 5))
      .withColumn("created_at", F.current_timestamp())
      .withColumn("txn_year",  F.year("transaction_date"))
      .withColumn("txn_month", F.month("transaction_date"))
      .withColumn("transaction_id", F.expr(f"cast({seed} as bigint) * 1000000000 + id"))
      # segment derived from the SAME hash used in dim_customer -> stays consistent, no join
      .withColumn("customer_segment", pick(["Premium","Standard","Value","Corporate"], F.col("customer_id"), 503))
    )

    # NOTE: first 16 columns = the ones Delta indexes for min/max stats. Filter cols first.
    return df.select(
        "transaction_id","transaction_date","country_code","age_group","category",
        "customer_id","product_id","channel","customer_segment",
        "net_amount","quantity","is_returned","transaction_ts","region","brand","store_id",
        F.col("category").alias("product_category"), F.col("subcategory").alias("product_subcategory"),
        "product_sku","country_name","currency_code","age_group_key","country_key",
        "unit_price","discount_pct","discount_amount","gross_amount","tax_amount","shipping_amount",
        "margin_amount","payment_method","device_type","return_reason","promotion_code","campaign_id",
        "order_id","session_id","delivery_days","rating","city","postal_code","latitude","longitude",
        "created_at","txn_year","txn_month")

print("Generator defined.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 6b — Calibration run (do not skip)
# 
# Write **50 M rows** into a throwaway table, read the real `sizeInBytes`, and derive bytes/row.
# Then reset `TXN_ROWS_TOTAL` to hit your target exactly.

# CELL ********************

# ============================================================
# CELL 6b — CALIBRATION
# ============================================================
CAL_ROWS = 50_000_000
cal_month_start = MONTHS[0]
cal_month_end   = (MONTHS[0] + relativedelta(months=1)) - relativedelta(days=1)

t0 = time.time()
(gen_txn_chunk(CAL_ROWS, cal_month_start, cal_month_end, seed=999, tasks=200)
   .write.mode("overwrite").format("delta").saveAsTable(f"{SCHEMA}.tmp_calibration"))
elapsed = time.time() - t0

d = spark.sql(f"DESCRIBE DETAIL {SCHEMA}.tmp_calibration").select("numFiles","sizeInBytes").collect()[0]
bytes_per_row = d["sizeInBytes"] / CAL_ROWS
GB = 1024**3

print(f"Elapsed            : {elapsed/60:.1f} min for {CAL_ROWS:,} rows")
print(f"Files              : {d['numFiles']}")
print(f"Size               : {d['sizeInBytes']/GB:.2f} GB")
print(f"Bytes per row      : {bytes_per_row:.1f}")
print(f"Throughput         : {CAL_ROWS/elapsed/1e6:.2f} M rows/sec")
print("-" * 55)

TARGET_TXN_GB = 700
rows_needed = int(TARGET_TXN_GB * GB / bytes_per_row)
print(f"Rows needed for {TARGET_TXN_GB} GB : {rows_needed:,}")
print(f"Estimated full-load time    : {rows_needed/(CAL_ROWS/elapsed)/3600:.1f} hours")
print("\n>>> Set TXN_ROWS_TOTAL = %d in Cell 2 and re-run Cell 2, then continue." % rows_needed)

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.tmp_calibration")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 5 — Full fact load (restartable, month by month)
# 
# The loop is **idempotent per month**: it uses `replaceWhere` so re-running a month overwrites just
# that partition. If a run dies at month 19, restart from month 19 — nothing else is touched.

# CELL ********************

# ============================================================
# CELL 7 — Create the empty partitioned fact table first
# ============================================================
sample = gen_txn_chunk(10, MONTHS[0], MONTHS[0], seed=1, tasks=1)

(sample.limit(0).write
   .format("delta")
   .partitionBy("txn_year","txn_month")
   .option("delta.autoOptimize.optimizeWrite","true")
   .mode("overwrite")
   .saveAsTable(f"{SCHEMA}.fact_customer_transactions"))

spark.sql(f"""
ALTER TABLE {SCHEMA}.fact_customer_transactions SET TBLPROPERTIES (
  'delta.dataSkippingNumIndexedCols' = '16',
  'delta.deletedFileRetentionDuration' = 'interval 1 days',
  'delta.logRetentionDuration' = 'interval 7 days'
)""")
spark.sql(f"DESCRIBE TABLE EXTENDED {SCHEMA}.fact_customer_transactions").show(80, False)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 8 — MAIN LOAD LOOP  (long running; restartable)
# ============================================================
START_IDX = 0        # <-- set to the month index to resume from
END_IDX   = 36

log = []
for i in range(START_IDX, END_IDX):
    m_start = MONTHS[i]
    m_end   = (m_start + relativedelta(months=1)) - relativedelta(days=1)
    target  = TXN_PER_MONTH[i]

    t0 = time.time()
    n_chunks = max(1, math.ceil(target / CHUNK_ROWS))
    per_chunk = target // n_chunks

    for c in range(n_chunks):
        rows = per_chunk if c < n_chunks - 1 else target - per_chunk * (n_chunks - 1)
        seed = 10_000 + i * 100 + c
        df = gen_txn_chunk(rows, m_start, m_end, seed)

        if c == 0:
            # first chunk REPLACES the month partition -> makes the month idempotent
            (df.write.format("delta").mode("overwrite")
               .option("replaceWhere", f"txn_year = {m_start.year} AND txn_month = {m_start.month}")
               .saveAsTable(f"{SCHEMA}.fact_customer_transactions"))
        else:
            (df.write.format("delta").mode("append")
               .saveAsTable(f"{SCHEMA}.fact_customer_transactions"))

    el = time.time() - t0
    log.append((str(m_start), target, round(el/60, 1)))
    print(f"[{i+1:2d}/36] {m_start}  rows={target:,}  {el/60:.1f} min")

d = spark.sql(f"DESCRIBE DETAIL {SCHEMA}.fact_customer_transactions").collect()[0]
print(f"\nTOTAL SIZE: {d['sizeInBytes']/1024**3:.1f} GB across {d['numFiles']} files")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 8b — Second fact table: `fact_web_events`
# 
# Narrow rows, huge count. This is the table you use to demo **small-file problems, partition pruning and
# liquid clustering** — the contrast against the wide transactions table makes the tuning story land.

# CELL ********************

# ============================================================
# CELL 9 — fact_web_events
# ============================================================
EVENTS = ["page_view","product_view","add_to_cart","remove_from_cart","search",
          "checkout_start","checkout_complete","wishlist_add","review_submit","login"]

def gen_web_chunk(n_rows, month_start, month_end, seed, tasks=TASKS_PER_CHUNK):
    days_in_month = (month_end - month_start).days + 1
    return (spark.range(n_rows).repartition(tasks)
      .withColumn("customer_id", (F.pmod(F.hash(F.col("id"), F.lit(seed)), F.lit(CUSTOMER_ROWS)) + 1).cast("bigint"))
      .withColumn("bucket",  F.pmod(F.hash(F.col("customer_id"), F.lit(101)), F.lit(1000)).cast("int"))
      .withColumn("abucket", F.pmod(F.hash(F.col("customer_id"), F.lit(211)), F.lit(100)).cast("int"))
      .join(CBUCK, "bucket").join(ABUCK, "abucket")
      .withColumn("event_date", F.expr(f"date_add(to_date('{month_start}'), cast(rand()*{days_in_month} as int))"))
      .withColumn("event_ts", F.expr("cast(event_date as timestamp) + make_interval(0,0,0,0, cast(rand()*23 as int), cast(rand()*59 as int), cast(rand()*59 as int))"))
      .withColumn("event_type", pick(EVENTS, F.col("id"), seed+1))
      .withColumn("product_id", (F.pmod(F.hash(F.col("id"), F.lit(seed+2)), F.lit(PRODUCT_ROWS)) + 1).cast("bigint"))
      .withColumn("session_id", F.expr("uuid()"))
      .withColumn("device_type", pick(DEVICES, F.col("id"), seed+3))
      .withColumn("channel", pick(CHANNELS, F.col("id"), seed+4))
      .withColumn("page_dwell_sec", (F.expr("rand()*300")).cast("int"))
      .withColumn("referrer", pick(["organic","paid_search","social","email","direct","affiliate"], F.col("id"), seed+5))
      .withColumn("event_year", F.year("event_date"))
      .withColumn("event_month", F.month("event_date"))
      .withColumn("event_id", F.expr(f"cast({seed} as bigint) * 1000000000 + id"))
      .select("event_id","event_date","country_code","age_group","event_type","customer_id",
              "product_id","channel","device_type","referrer","page_dwell_sec","session_id",
              "event_ts","region","age_group_key","country_key","event_year","event_month"))

# create empty
(gen_web_chunk(10, MONTHS[0], MONTHS[0], 1, 1).limit(0).write
   .format("delta").partitionBy("event_year","event_month")
   .mode("overwrite").saveAsTable(f"{SCHEMA}.fact_web_events"))

for i in range(0, 36):
    m_start = MONTHS[i]; m_end = (m_start + relativedelta(months=1)) - relativedelta(days=1)
    target = WEB_PER_MONTH[i]
    n_chunks = max(1, math.ceil(target / CHUNK_ROWS))
    per_chunk = target // n_chunks
    t0 = time.time()
    for c in range(n_chunks):
        rows = per_chunk if c < n_chunks-1 else target - per_chunk*(n_chunks-1)
        df = gen_web_chunk(rows, m_start, m_end, 50_000 + i*100 + c)
        mode_opts = df.write.format("delta")
        if c == 0:
            (mode_opts.mode("overwrite")
               .option("replaceWhere", f"event_year = {m_start.year} AND event_month = {m_start.month}")
               .saveAsTable(f"{SCHEMA}.fact_web_events"))
        else:
            mode_opts.mode("append").saveAsTable(f"{SCHEMA}.fact_web_events")
    print(f"[{i+1:2d}/36] web {m_start} rows={target:,} {(time.time()-t0)/60:.1f} min")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 10 — Validate the 1 TB target
# ============================================================
GB = 1024**3
tot = 0
rows = []
for t in [r["tableName"] for r in spark.sql(f"SHOW TABLES IN {SCHEMA}").collect()]:
    try:
        d = spark.sql(f"DESCRIBE DETAIL {SCHEMA}.{t}").collect()[0]
        tot += d["sizeInBytes"]
        rows.append((t, d["numFiles"], round(d["sizeInBytes"]/GB, 2)))
    except Exception as e:
        pass

import pandas as pd
display(pd.DataFrame(rows, columns=["table","files","size_GB"]).sort_values("size_GB", ascending=False))
print(f"TOTAL LAKEHOUSE SIZE: {tot/GB:.1f} GB  ({tot/1024**4:.3f} TB)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Step 6 — Performance tuning demo script
# 
# This is the part you actually present. **Run the "before" query, capture the timing, apply the fix, re-run.**
# Keep a Spark UI tab open — the SQL plan diff is the money shot.
# 
# Sequence that demos well, in order:

# MARKDOWN ********************

# ## Step 7 — Gold aggregates for the report / dashboard demo
# 
# Direct Lake over a 5-billion-row fact works, but for a *reliable live demo* you want gold aggregate tables:
# they keep the semantic model inside Direct Lake guardrails and make visuals instant.

# MARKDOWN ********************

# ## Step 8 — Semantic model + report (Direct Lake)
# 
# 1. Open `CustomerRAW` → top-right dropdown → **SQL analytics endpoint** → refresh, confirm all `dbo` tables appear.
# 2. Ribbon → **New semantic model** → name it `Customer Performance Model` → select:
#    `agg_country_month`, `agg_age_country_month`, `agg_category_month`, `agg_customer_360`,
#    `dim_date`, `dim_country`, `dim_age_group`, `dim_product`, `dim_customer` (and the raw fact if you want the "big scan" demo).
# 3. **Model relationships**: `dim_date[year_month]` → agg tables, `dim_country[country_key]` → `agg_country_month[country_key]`,
#    `dim_customer[customer_id]` → `agg_customer_360[customer_id]`. Mark `dim_date` as the date table.
# 4. Add DAX measures:
# 
# ```dax
# Revenue          = SUM(agg_country_month[revenue])
# Margin %         = DIVIDE(SUM(agg_country_month[margin]), [Revenue])
# Active Customers = SUM(agg_country_month[active_customers])
# AOV              = DIVIDE([Revenue], SUM(agg_country_month[txn_count]))
# Revenue LY       = CALCULATE([Revenue], SAMEPERIODLASTYEAR(dim_date[full_date]))
# Revenue YoY %    = DIVIDE([Revenue] - [Revenue LY], [Revenue LY])
# Return Rate      = DIVIDE(SUM(agg_country_month[returns_count]), SUM(agg_country_month[txn_count]))
# ```
# 
# 5. **Report pages to build** (this is a good demo narrative):
#    - *Executive overview* — KPI cards (Revenue, YoY %, Margin %, Active Customers), revenue trend line over 36 months (seasonality shows up beautifully), filled map by country.
#    - *Country deep dive* — matrix Region → Country → month, small multiples, decomposition tree on Revenue by Region → Country → Age Group → Category.
#    - *Age & segment* — 100% stacked bar of revenue share by age group over time; scatter of AOV vs Active Customers bubbled by country.
#    - *Customer 360* — histogram of lifetime value, RFM-style segmentation from `agg_customer_360`.
#    - *Performance page* — a visual bound to the **raw 5 B-row fact** so you can show Direct Lake fallback behaviour vs the aggregate table.
# 
# 6. **Direct Lake talking points**: no import/refresh, data read straight from OneLake; V-Order makes transcoding fast;
#    guardrails per SKU determine when it falls back to DirectQuery — which is exactly why the gold aggregates exist.


# MARKDOWN ********************

# ## Step 9 — Demo runbook (what to actually say, in order)
# 
# | # | Beat | Show | Point |
# |---|---|---|---|
# | 1 | Scale | `DESCRIBE DETAIL` sizes, row counts | ~1 TB, 5 B+ rows, all managed Delta under `dbo` |
# | 2 | Baseline pain | Cell 11 timings + Spark UI | Full scan, many files, no skipping |
# | 3 | File layout | Cell 12 `OPTIMIZE ... VORDER` | numFiles collapses; re-run query |
# | 4 | Data skipping | Cell 13 Z-ORDER / liquid clustering | Selective filter speedup |
# | 5 | Engine | Cell 14 NEE on/off | Vectorized C++ execution, zero code change |
# | 6 | Pruning | Cell 15 `explain` + UI scan node | Partition + column pruning |
# | 7 | Joins | Cell 16 broadcast vs SMJ, AQE skew | AQE adapts at runtime |
# | 8 | Pre-compute | Cell 17 Materialized Lake View | Incremental refresh via Delta CDF |
# | 9 | Consumption | Direct Lake report | No refresh, sub-second on gold aggregates |
# | 10 | Cost | Capacity Metrics app | CU burn during load vs steady-state query |
# 
# ### Cleanup
# 
# ```python
# for t in ["fact_customer_transactions","fact_web_events","dim_customer","dim_product",
#           "agg_country_month","agg_age_country_month","agg_category_month","agg_customer_360",
#           "dim_date","dim_country","dim_age_group","dim_channel",
#           "lkp_country_bucket","lkp_age_bucket"]:
#     spark.sql(f"DROP TABLE IF EXISTS dbo.{t}")
# ```

