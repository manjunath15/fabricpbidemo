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

# # Step 7 — Gold Aggregate Tables for the Direct Lake Demo
# ### Lakehouse `CustomerRAW` · schema `dbo`
# 
# Builds the serving layer that your semantic model and report sit on top of.
# 
# **Why aggregates rather than pointing Direct Lake at the raw fact:** Direct Lake *can* read a 5-billion-row
# table, but every visual then depends on transcoding a huge column segment into memory, and you're one
# guardrail away from a silent fallback to DirectQuery mid-demo. Gold aggregates collapse the fact by ~6 orders
# of magnitude, keeping visuals instant and the model comfortably inside guardrails.
# 
# ---
# 
# ## What gets built
# 
# | Table | Grain | Est. rows | Cost | Powers |
# |---|---|---|---|---|
# | `agg_country_month` | year × month × country | ~1,400 | 💚 cheap | KPI cards, map, trend |
# | `agg_age_country_month` | + age group | ~10,000 | 💚 cheap | age/segment page |
# | `agg_category_month` | year × month × category × subcat × brand × channel | ~500,000 | 💚 cheap | product page |
# | `agg_channel_day` | date × channel × country | ~250,000 | 🟡 moderate | daily trend, seasonality |
# | `agg_customer_360` | one row per customer | **80,000,000** | 🔴 **expensive** | LTV, RFM, histograms |
# 
# > **Run order matters.** Cells are sequenced cheap → expensive. The first four finish in minutes.
# > `agg_customer_360` groups 5 B rows into 80 M — it's the long pole and is deliberately last, so you can
# > demo everything else while it runs or skip it entirely for a first pass.
# 
# Every cell is **idempotent** and guarded — safe to re-run, and it skips work already done.


# MARKDOWN ********************

# ## Setup — config tuned for the serving layer

# CELL ********************

# ============================================================
# CELL 1 — Config for gold / serving tables
# ============================================================
from pyspark.sql import functions as F
import time, pandas as pd

SCHEMA = "dbo"
FACT   = f"{SCHEMA}.fact_customer_transactions"
WEB    = f"{SCHEMA}.fact_web_events"
GB     = 1024**3

spark.sql(f"USE {SCHEMA}")

# --- Serving-layer profile: V-Order ON, ~1 GB files ---------
# This mirrors what the readHeavyForPBI resource profile applies.
spark.conf.set("spark.sql.parquet.vorder.default", "true")
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.optimizeWrite.binSize", "1073741824")  # 1 GB

# --- Make the heavy aggregations survive -------------------
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(200 * 1024 * 1024))
spark.conf.set("spark.sql.shuffle.partitions", "1024")

# Native Execution Engine helps a lot on wide GROUP BY aggregations
spark.conf.set("spark.native.enabled", "true")

# --- Approximate vs exact distinct counts -------------------
# COUNT(DISTINCT customer_id) over billions of rows is the single most expensive
# operation in this notebook. approx_count_distinct uses HyperLogLog: ~2% error,
# but dramatically cheaper. For a performance DEMO, approximate is the right call.
USE_APPROX_DISTINCT = True

# Defined HERE so every build cell can be run independently / out of order.
dc = ("approx_count_distinct(customer_id)" if USE_APPROX_DISTINCT
      else "COUNT(DISTINCT customer_id)")
print(f"Distinct counts: {'APPROXIMATE (HyperLogLog, ~2% error)' if USE_APPROX_DISTINCT else 'EXACT (slow)'}")
print(f"  expression: {dc}")

# --- Preflight ---------------------------------------------
missing = [t for t in [FACT] if not spark.catalog.tableExists(t)]
if missing:
    raise Exception(f"Required source table(s) missing: {missing}. Run the generator notebook first.")

d = spark.sql(f"DESCRIBE DETAIL {FACT}").collect()[0]
print(f"\nSource: {FACT}")
print(f"  size  : {d['sizeInBytes']/GB:.1f} GB")
print(f"  files : {d['numFiles']:,}")
print(f"\nConfig ready. V-Order={spark.conf.get('spark.sql.parquet.vorder.default')}, "
      f"NEE={spark.conf.get('spark.native.enabled')}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 2 — Helper: build a table idempotently, with timing
# ============================================================
try:
    dc
except NameError:
    raise Exception("Run Cell 1 first — it defines SCHEMA, FACT and the `dc` distinct-count expression.")

BUILD_LOG = []

def build(name, sql, rebuild=False, vorder=True):
    """Create a gold table if absent (or if rebuild=True), then V-Order it."""
    full = f"{SCHEMA}.{name}"
    if spark.catalog.tableExists(full) and not rebuild:
        n = spark.table(full).count()
        det = spark.sql(f"DESCRIBE DETAIL {full}").collect()[0]
        print(f"⏭  {name:26s} exists — {n:,} rows, {det['sizeInBytes']/1024**2:.1f} MB (set rebuild=True to force)")
        BUILD_LOG.append({"table": name, "rows": n, "minutes": 0.0, "status": "skipped"})
        return

    print(f"▶  building {name} ...")
    t0 = time.time()
    spark.sql(f"CREATE OR REPLACE TABLE {full} AS {sql}")
    if vorder:
        spark.sql(f"ALTER TABLE {full} SET TBLPROPERTIES ('delta.parquet.vorder.enabled' = 'true')")
        spark.sql(f"OPTIMIZE {full} VORDER")
    el = (time.time() - t0) / 60

    n   = spark.table(full).count()
    det = spark.sql(f"DESCRIBE DETAIL {full}").collect()[0]
    print(f"✅ {name:26s} {n:,} rows · {det['sizeInBytes']/1024**2:.1f} MB · "
          f"{det['numFiles']} files · {el:.1f} min")
    BUILD_LOG.append({"table": name, "rows": n, "minutes": round(el, 1), "status": "built"})

print("Helper ready. Re-running any build cell is safe.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Aggregate 1 — `agg_country_month`
# 
# The workhorse. Powers KPI cards, the filled map, and the 36-month revenue trend.

# CELL ********************

# ============================================================
# CELL 3 — agg_country_month
# ============================================================
build("agg_country_month", f"""
SELECT
    txn_year,
    txn_month,
    CAST(CONCAT(txn_year, '-', LPAD(txn_month, 2, '0')) AS STRING) AS year_month,
    country_key,
    country_code,
    country_name,
    region,
    currency_code,
    SUM(net_amount)                                  AS revenue,
    SUM(margin_amount)                               AS margin,
    SUM(discount_amount)                             AS discount,
    SUM(gross_amount)                                AS gross_revenue,
    SUM(tax_amount)                                  AS tax,
    SUM(shipping_amount)                             AS shipping,
    SUM(quantity)                                    AS units,
    COUNT(*)                                         AS txn_count,
    {dc}                                             AS active_customers,
    SUM(CASE WHEN is_returned THEN 1 ELSE 0 END)     AS returns_count,
    SUM(CASE WHEN is_returned THEN net_amount ELSE 0 END) AS returns_value,
    AVG(net_amount)                                  AS avg_order_value,
    AVG(delivery_days)                               AS avg_delivery_days
FROM {FACT}
GROUP BY txn_year, txn_month, country_key, country_code, country_name, region, currency_code
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Aggregate 2 — `agg_age_country_month`
# 
# Country × age group over time. This is the one that shows off the demographic dimension you built in.

# CELL ********************

# ============================================================
# CELL 4 — agg_age_country_month
# ============================================================
build("agg_age_country_month", f"""
SELECT
    txn_year,
    txn_month,
    CAST(CONCAT(txn_year, '-', LPAD(txn_month, 2, '0')) AS STRING) AS year_month,
    country_code,
    region,
    age_group_key,
    age_group,
    customer_segment,
    SUM(net_amount)                              AS revenue,
    SUM(margin_amount)                           AS margin,
    SUM(quantity)                                AS units,
    COUNT(*)                                     AS txn_count,
    {dc}                                         AS active_customers,
    AVG(net_amount)                              AS avg_order_value,
    SUM(CASE WHEN is_returned THEN 1 ELSE 0 END) AS returns_count
FROM {FACT}
GROUP BY txn_year, txn_month, country_code, region,
         age_group_key, age_group, customer_segment
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Aggregate 3 — `agg_category_month`
# 
# Product hierarchy × channel. Feeds the decomposition tree and the category deep-dive page.

# CELL ********************

# ============================================================
# CELL 5 — agg_category_month
# ============================================================
build("agg_category_month", f"""
SELECT
    txn_year,
    txn_month,
    CAST(CONCAT(txn_year, '-', LPAD(txn_month, 2, '0')) AS STRING) AS year_month,
    product_category,
    product_subcategory,
    brand,
    channel,
    region,
    SUM(net_amount)                              AS revenue,
    SUM(margin_amount)                           AS margin,
    SUM(discount_amount)                         AS discount,
    SUM(quantity)                                AS units,
    COUNT(*)                                     AS txn_count,
    AVG(unit_price)                              AS avg_unit_price,
    AVG(discount_pct)                            AS avg_discount_pct,
    SUM(CASE WHEN is_returned THEN 1 ELSE 0 END) AS returns_count,
    AVG(rating)                                  AS avg_rating
FROM {FACT}
GROUP BY txn_year, txn_month, product_category, product_subcategory,
         brand, channel, region
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Aggregate 4 — `agg_channel_day`
# 
# Daily grain. Needed for the seasonality story — the Nov/Dec peaks and weekend effects you baked into the
# generator only become visible at day level.

# CELL ********************

# ============================================================
# CELL 6 — agg_channel_day
# ============================================================
build("agg_channel_day", f"""
SELECT
    transaction_date,
    CAST(DATE_FORMAT(transaction_date, 'yyyyMMdd') AS INT) AS date_key,
    txn_year,
    txn_month,
    channel,
    country_code,
    region,
    payment_method,
    device_type,
    SUM(net_amount)                              AS revenue,
    SUM(margin_amount)                           AS margin,
    SUM(quantity)                                AS units,
    COUNT(*)                                     AS txn_count,
    {dc}                                         AS active_customers,
    AVG(net_amount)                              AS avg_order_value,
    SUM(CASE WHEN is_returned THEN 1 ELSE 0 END) AS returns_count
FROM {FACT}
GROUP BY transaction_date, txn_year, txn_month, channel,
         country_code, region, payment_method, device_type
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Aggregate 5 — `agg_customer_360` ⚠️ the expensive one
# 
# **Read before running.** This groups ~5 billion fact rows into ~80 million customer rows. It is by far the
# heaviest operation in the notebook — expect it to dominate total runtime and produce a large shuffle.
# 
# Two things make it survivable:
# - **Approximate distinct counts** (set in Cell 1) instead of exact
# - **AQE skew join + coalesce**, already enabled
# 
# If you only need the demo running quickly, **skip this cell** — the first four aggregates cover the
# executive overview, country, age and product pages. Come back to it before the Customer 360 page.

# CELL ********************

# ============================================================
# CELL 7 — agg_customer_360  (LONG RUNNING — see note above)
# ============================================================
RUN_CUSTOMER_360 = True   # <-- set False to skip

if not RUN_CUSTOMER_360:
    print("⏭  Skipped by flag. Set RUN_CUSTOMER_360 = True when you need the Customer 360 page.")
else:
    cat_dc = ("approx_count_distinct(product_category)" if USE_APPROX_DISTINCT
              else "COUNT(DISTINCT product_category)")
    build("agg_customer_360", f"""
    SELECT
        f.customer_id,
        MIN(f.transaction_date)                          AS first_purchase_date,
        MAX(f.transaction_date)                          AS last_purchase_date,
        DATEDIFF(CURRENT_DATE(), MAX(f.transaction_date)) AS recency_days,
        COUNT(*)                                          AS lifetime_txns,
        SUM(f.net_amount)                                 AS lifetime_value,
        SUM(f.margin_amount)                              AS lifetime_margin,
        AVG(f.net_amount)                                 AS avg_order_value,
        MAX(f.net_amount)                                 AS max_order_value,
        SUM(f.quantity)                                   AS lifetime_units,
        SUM(CASE WHEN f.is_returned THEN 1 ELSE 0 END)    AS returns_count,
        {cat_dc}                                          AS categories_bought,
        MAX(f.country_code)                               AS country_code,
        MAX(f.region)                                     AS region,
        MAX(f.age_group)                                  AS age_group,
        MAX(f.customer_segment)                           AS customer_segment
    FROM {FACT} f
    GROUP BY f.customer_id
    """)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 8 — Enrich customer_360 with RFM segments (cheap, runs on 80M)
# ============================================================
C360 = f"{SCHEMA}.agg_customer_360"

if spark.catalog.tableExists(C360):
    spark.sql(f"""
    CREATE OR REPLACE TABLE {SCHEMA}.agg_customer_rfm AS
    WITH scored AS (
      SELECT
        customer_id, country_code, region, age_group, customer_segment,
        recency_days, lifetime_txns, lifetime_value, avg_order_value,
        returns_count, categories_bought,
        first_purchase_date, last_purchase_date,
        NTILE(5) OVER (ORDER BY recency_days DESC)   AS r_score,
        NTILE(5) OVER (ORDER BY lifetime_txns)       AS f_score,
        NTILE(5) OVER (ORDER BY lifetime_value)      AS m_score
      FROM {C360}
    )
    SELECT *,
      CONCAT(r_score, f_score, m_score) AS rfm_cell,
      CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3                  THEN 'Loyal'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'New / Promising'
        WHEN r_score <= 2 AND f_score >= 4                  THEN 'At Risk'
        WHEN r_score <= 2 AND f_score <= 2                  THEN 'Hibernating'
        ELSE 'Needs Attention'
      END AS rfm_segment,
      CASE
        WHEN lifetime_value >= 10000 THEN 'A: >10k'
        WHEN lifetime_value >= 5000  THEN 'B: 5-10k'
        WHEN lifetime_value >= 1000  THEN 'C: 1-5k'
        WHEN lifetime_value >= 250   THEN 'D: 250-1k'
        ELSE 'E: <250'
      END AS value_band
    FROM scored
    """)
    spark.sql(f"OPTIMIZE {SCHEMA}.agg_customer_rfm VORDER")
    print("✅ agg_customer_rfm built")
    display(spark.sql(f"""
      SELECT rfm_segment, COUNT(*) customers,
             ROUND(SUM(lifetime_value)/1e6, 1) AS ltv_millions
      FROM {SCHEMA}.agg_customer_rfm
      GROUP BY rfm_segment ORDER BY customers DESC
    """))
else:
    print("⏭  agg_customer_360 not present — run Cell 7 first.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Optional — web funnel aggregate
# 
# Only if you want the engagement page. Runs against `fact_web_events`.

# CELL ********************

# ============================================================
# CELL 9 — agg_web_funnel_month (optional)
# ============================================================
if spark.catalog.tableExists(WEB):
    build("agg_web_funnel_month", f"""
    SELECT
        event_year,
        event_month,
        CAST(CONCAT(event_year, '-', LPAD(event_month, 2, '0')) AS STRING) AS year_month,
        country_code,
        region,
        age_group,
        channel,
        device_type,
        referrer,
        event_type,
        COUNT(*)                                 AS event_count,
        {dc}                                     AS unique_customers,
        AVG(page_dwell_sec)                      AS avg_dwell_sec
    FROM {WEB}
    GROUP BY event_year, event_month, country_code, region,
             age_group, channel, device_type, referrer, event_type
    """)
else:
    print("⏭  fact_web_events not present — skipping.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Validation

# CELL ********************

# ============================================================
# CELL 10 — Validate the gold layer
# ============================================================
GOLD = ["agg_country_month", "agg_age_country_month", "agg_category_month",
        "agg_channel_day", "agg_customer_360", "agg_customer_rfm",
        "agg_web_funnel_month"]

rows = []
for t in GOLD:
    full = f"{SCHEMA}.{t}"
    if not spark.catalog.tableExists(full):
        rows.append({"table": t, "status": "❌ missing"})
        continue
    d = spark.sql(f"DESCRIBE DETAIL {full}").collect()[0]
    n = spark.table(full).count()
    vo = spark.sql(f"SHOW TBLPROPERTIES {full}").filter(
            "key = 'delta.parquet.vorder.enabled'").collect()
    rows.append({
        "table": t,
        "rows": f"{n:,}",
        "MB": round(d["sizeInBytes"]/1024**2, 1),
        "files": d["numFiles"],
        "vorder": "✅" if vo and vo[0]["value"] == "true" else "—",
        "status": "🟢 ready",
    })

display(pd.DataFrame(rows))

if BUILD_LOG:
    print("\nBuild timings:")
    display(pd.DataFrame(BUILD_LOG))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 11 — Reconciliation: do the aggregates match the fact?
# ============================================================
# Never demo numbers you haven't reconciled.
AGG = f"{SCHEMA}.agg_country_month"

fact_tot = spark.sql(f"SELECT SUM(net_amount) r, COUNT(*) c FROM {FACT}").collect()[0]
agg_tot  = spark.sql(f"SELECT SUM(revenue) r, SUM(txn_count) c FROM {AGG}").collect()[0]

dr = abs(fact_tot["r"] - agg_tot["r"]) / fact_tot["r"] * 100
dc_ = abs(fact_tot["c"] - agg_tot["c"]) / fact_tot["c"] * 100

print(f"Revenue  fact={fact_tot['r']:,.2f}  agg={agg_tot['r']:,.2f}  diff={dr:.6f}%")
print(f"Txn cnt  fact={fact_tot['c']:,}      agg={agg_tot['c']:,}      diff={dc_:.6f}%")
print("\n" + ("🟢 RECONCILED" if dr < 0.01 and dc_ < 0.01 else "🔴 MISMATCH — investigate before demoing"))
print("""
Note: additive measures (SUM, COUNT) reconcile exactly. Distinct counts will NOT,
because approx_count_distinct trades ~2% accuracy for speed. Say so if asked —
"active customers is approximate" is a fine answer; being caught unaware is not.
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 12 — Query performance: gold vs raw fact
# ============================================================
def timed(sql, label):
    spark.sql("CLEAR CACHE")
    t0 = time.time(); spark.sql(sql).count(); el = time.time() - t0
    print(f"  {label:38s} {el:7.2f}s")
    return el

raw = timed(f"""
SELECT country_code, txn_year, SUM(net_amount) FROM {FACT}
GROUP BY country_code, txn_year""", "same query on RAW fact")

gold = timed(f"""
SELECT country_code, txn_year, SUM(revenue) FROM {AGG}
GROUP BY country_code, txn_year""", "on GOLD aggregate")

print(f"\n>>> {raw/gold:.0f}x faster — this is the Direct Lake story in one number.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Next — wire it into the semantic model
# 
# 1. Lakehouse `CustomerRAW` → top-right dropdown → **SQL analytics endpoint** → confirm the `agg_*` tables appear
# 2. Ribbon → **New semantic model** → name it `Customer Performance Model`
# 3. Select: all `agg_*` tables plus `dim_date`, `dim_country`, `dim_age_group`, `dim_product`
# 4. **Relationships** — mark `dim_date` as the date table first:
#    - `dim_date[year_month]` → each agg's `year_month` (many-to-one, single direction)
#    - `dim_country[country_key]` → `agg_country_month[country_key]`
#    - `dim_date[date_key]` → `agg_channel_day[date_key]`
# 5. **Measures** (put them on `agg_country_month`):
# 
# ```dax
# Revenue          = SUM(agg_country_month[revenue])
# Margin           = SUM(agg_country_month[margin])
# Margin %         = DIVIDE([Margin], [Revenue])
# Txn Count        = SUM(agg_country_month[txn_count])
# AOV              = DIVIDE([Revenue], [Txn Count])
# Active Customers = SUM(agg_country_month[active_customers])
# Return Rate      = DIVIDE(SUM(agg_country_month[returns_count]), [Txn Count])
# Revenue LY       = CALCULATE([Revenue], SAMEPERIODLASTYEAR(dim_date[full_date]))
# Revenue YoY %    = DIVIDE([Revenue] - [Revenue LY], [Revenue LY])
# Revenue YTD      = TOTALYTD([Revenue], dim_date[full_date])
# ```
# 
# ### Why this stays in Direct Lake
# 
# | | Raw fact | Gold aggregate |
# |---|---|---|
# | Rows | ~5,000,000,000 | ~1,400 |
# | Transcoding | large column segments | trivial |
# | Guardrail risk | fallback to DirectQuery | none |
# | Visual response | seconds+ | instant |
# 
# Keep the raw fact in the model too, on its own page — then you can *deliberately* show the contrast, rather
# than discovering it live.
# 
# ### One honest caveat for the demo
# 
# `active_customers` uses HyperLogLog (~2% error) and is **not additive** — summing monthly active customers
# does not give you distinct annual actives. If someone drills from month to year, that number will be wrong
# in a way that's easy to defend but embarrassing to be surprised by. Either set `USE_APPROX_DISTINCT = False`
# in Cell 1 and accept a much longer build, or label the measure "Active Customers (est.)".

