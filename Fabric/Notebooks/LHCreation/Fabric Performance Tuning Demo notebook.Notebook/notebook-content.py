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

# # Fabric Lakehouse Performance Tuning — Demo Runbook
# ### Lakehouse `CustomerRAW` · schema `dbo` · ~1 TB customer dataset
# 
# This notebook is the **demo script**. Every lever follows the same shape:
# 
# > **measure → show the problem → apply the fix → re-measure → explain why**
# 
# Run it top-to-bottom in a fresh session with the **Spark UI open in a second tab**. The plan diffs and
# scan-node metrics are what make the story land — the timings alone are just numbers.
# 
# ---
# 
# ## Demo agenda (~45–60 min)
# 
# | # | Lever | What it fixes | Expected gain |
# |---|---|---|---|
# | 0 | Baseline + instrumentation | — | establishes the "before" |
# | 1 | Small files → `OPTIMIZE` | I/O overhead per file | 2–5× |
# | 2 | V-Order | read efficiency across all Fabric engines | ~10% Spark, much more for Direct Lake / SQL |
# | 3 | Data skipping → Z-Order / Liquid Clustering | scanning data you don't need | 3–10× on selective filters |
# | 4 | Delta stats + column order | file-level pruning | varies |
# | 5 | Partition + column pruning | reading whole table for 1 month | 10–50× |
# | 6 | Join strategy + AQE skew | shuffle blowups | 2–8× |
# | 7 | Native Execution Engine | JVM row-at-a-time overhead | up to ~4× |
# | 8 | Materialized Lake Views | recomputing the same aggregate | 10–100× |
# | 9 | Session/pool config | concurrency, startup latency | — |
# | 10 | Cost view | CU efficiency, not just wall clock | — |


# MARKDOWN ********************

# ## ⚠️ Read this before you demo — correction to the generator notebook
# 
# The generation notebook created `fact_customer_transactions` **partitioned by** `txn_year, txn_month`.
# That is fine and it demos partition pruning well — but be aware:
# 
# **Liquid Clustering is not compatible with partitioning or Z-Order.** A table is either Hive-partitioned,
# or `CLUSTER BY`, not both. So Lever 3 gives you two mutually exclusive paths:
# 
# - **Path A (Z-Order)** — works on the existing partitioned table. Use this if you want a single-table demo.
# - **Path B (Liquid Clustering)** — requires an **unpartitioned** copy. Cell in Lever 3B builds
#   `fact_txn_lc` for a genuine side-by-side. This is the more modern, more forward-looking story and it's
#   what current Fabric guidance favours over static partitioning.
# 
# Also note: **clustering only physically happens when `OPTIMIZE` runs.** Declaring `CLUSTER BY` and never
# running `OPTIMIZE` gives you zero file-skipping benefit — a very common misunderstanding worth calling out live.
# 
# And: `OPTIMIZE` / `ZORDER` / `VACUUM` are **Spark SQL** commands. Run them in this notebook or a Spark Job
# Definition — **not** in the SQL analytics endpoint.


# MARKDOWN ********************

# ## Lever 0 — Instrumentation harness

# CELL ********************

# ============================================================
# CELL 1 — Setup + measurement harness
# ============================================================
from pyspark.sql import functions as F
from datetime import date
from dateutil.relativedelta import relativedelta
import time, json, pandas as pd

SCHEMA = "dbo"
FACT   = f"{SCHEMA}.fact_customer_transactions"
WEB    = f"{SCHEMA}.fact_web_events"
GB     = 1024**3

spark.sql(f"USE {SCHEMA}")

RESULTS = []   # every timed run lands here for the final scorecard

def bench(sql, label, lever="", runs=1, warm=False):
    """Time a query with cold cache. Records into RESULTS."""
    if not warm:
        spark.sql("CLEAR CACHE")
    times = []
    n = None
    for _ in range(runs):
        t0 = time.time()
        n = spark.sql(sql).count()
        times.append(time.time() - t0)
    el = min(times)
    RESULTS.append({"lever": lever, "label": label, "seconds": round(el, 2), "rows": n})
    print(f"  {label:44s} {el:8.2f}s   rows={n:,}")
    return el

def table_profile(tbl):
    d = spark.sql(f"DESCRIBE DETAIL {tbl}").collect()[0]
    return {
        "table": tbl,
        "numFiles": d["numFiles"],
        "size_GB": round(d["sizeInBytes"]/GB, 2),
        "avg_file_MB": round(d["sizeInBytes"]/max(d["numFiles"],1)/1024**2, 1),
        "partitionCols": d["partitionColumns"],
        "clusterCols": d.asDict().get("clusteringColumns"),
    }

def show_profiles(*tbls):
    display(pd.DataFrame([table_profile(t) for t in tbls]))

def scorecard():
    df = pd.DataFrame(RESULTS)
    display(df)
    return df

# Demo time window
today     = date.today()
END_M     = date(today.year, today.month, 1) - relativedelta(months=1)
LAST_YEAR = END_M.year
LAST_MON  = END_M.month
print(f"Demo focus month: {LAST_YEAR}-{LAST_MON:02d}")

show_profiles(FACT, WEB)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 2 — Reset to a "cold, untuned" starting state
# ============================================================
# Turn OFF everything we intend to demo turning ON.
spark.conf.set("spark.sql.parquet.vorder.default", "false")
spark.conf.set("spark.native.enabled", "false")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")   # deliberately bad
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "false")
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "false")

for k in ["spark.native.enabled","spark.sql.parquet.vorder.default",
          "spark.sql.adaptive.enabled","spark.sql.autoBroadcastJoinThreshold"]:
    print(f"{k:48s} = {spark.conf.get(k)}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### The three demo queries
# 
# Deliberately chosen to stress different things: a **wide aggregate scan**, a **selective filter**,
# and a **multi-table join**. Different levers move different queries — that contrast is the point.

# CELL ********************

# ============================================================
# CELL 3 — Demo query definitions
# ============================================================
Q1_SCAN = f"""
SELECT country_code, age_group, SUM(net_amount) AS revenue, COUNT(*) AS txns
FROM {FACT}
WHERE txn_year = {LAST_YEAR}
GROUP BY country_code, age_group
"""

Q2_FILTER = f"""
SELECT product_category, SUM(net_amount) AS revenue, COUNT(*) AS txns
FROM {FACT}
WHERE country_code = 'IN'
  AND age_group    = '25-34'
  AND txn_year     = {LAST_YEAR}
GROUP BY product_category
"""

Q3_JOIN = f"""
SELECT c.customer_segment, p.category, SUM(f.net_amount) AS revenue
FROM {FACT} f
JOIN {SCHEMA}.dim_customer c ON f.customer_id = c.customer_id
JOIN {SCHEMA}.dim_product  p ON f.product_id  = p.product_id
WHERE f.txn_year = {LAST_YEAR} AND f.txn_month = {LAST_MON}
GROUP BY c.customer_segment, p.category
"""

Q4_WIDE = f"""
SELECT * FROM {FACT}
WHERE txn_year = {LAST_YEAR} AND txn_month = {LAST_MON} AND net_amount > 500
"""

print("=== BASELINE (everything off) ===")
bench(Q1_SCAN,   "Q1 aggregate scan",   "0-baseline")
bench(Q2_FILTER, "Q2 selective filter", "0-baseline")
bench(Q3_JOIN,   "Q3 three-table join", "0-baseline")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 1 — Small files
# 
# **The pitch:** "Spark isn't slow. Your file layout is." Every file costs a metadata lookup, an open,
# a footer read. Thousands of small files means most of your job is overhead, not compute.
# 
# `fact_web_events` was loaded with optimizeWrite off in this session, so it's the better victim here.
# Show the profile, run `OPTIMIZE`, show it again.

# CELL ********************

# ============================================================
# CELL 4 — LEVER 1: OPTIMIZE (bin-packing / compaction)
# ============================================================
print("BEFORE:"); show_profiles(WEB); 

Q_WEB = f"""
SELECT event_type, country_code, COUNT(*) AS events, AVG(page_dwell_sec) AS avg_dwell
FROM {WEB}
WHERE event_year = {LAST_YEAR} AND event_month = {LAST_MON}
GROUP BY event_type, country_code
"""
bench(Q_WEB, "web agg BEFORE optimize", "1-smallfiles")

t0 = time.time()
spark.sql(f"OPTIMIZE {WEB} WHERE event_year = {LAST_YEAR} AND event_month = {LAST_MON}")
print(f"OPTIMIZE took {(time.time()-t0)/60:.1f} min")

print("AFTER:"); show_profiles(WEB)
bench(Q_WEB, "web agg AFTER optimize", "1-smallfiles")

# TALKING POINTS
# - numFiles collapses, avg_file_MB jumps
# - OPTIMIZE rewrites layout only; results are byte-identical
# - Scope it with WHERE so you rewrite recent partitions, not 3 years of history

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 4b — Prevent the problem instead of curing it
# ============================================================
# Write-side settings so small files never accumulate in the first place.
spark.conf.set("spark.microsoft.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.microsoft.delta.optimizeWrite.binSize", "1073741824")  # 1 GB
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

# Or pin it on the table so every writer inherits it:
spark.sql(f"""
ALTER TABLE {WEB} SET TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
)""")
print("Optimize-write + auto-compact enabled.")

# TALKING POINT: for SQL analytics endpoint / Warehouse consumers, target files up to ~4 GB
# with ~2M rows per row group. Spark tolerates a wider range; the SQL engine is fussier.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 2 — V-Order
# 
# **The pitch:** V-Order is a *write-time* Parquet layout optimization — special sorting, row-group
# distribution, dictionary encoding and compression — that makes files read faster across **every** Fabric
# engine. Power BI and SQL benefit most (Verti-Scan gets near in-memory access times); Spark sees a smaller
# but real gain, around 10% on average.
# 
# **The tradeoff to state honestly:** writes get roughly 15% slower. That's why V-Order is **off by default
# in new Fabric workspaces** — Microsoft optimized the default for write-heavy engineering, not read-heavy
# serving. Your job is to decide per-table which side of that tradeoff you're on.
# 
# Rule of thumb: **gold / serving tables → V-Order on. Bronze staging → leave it off.**

# CELL ********************

# ============================================================
# CELL 5 — LEVER 2: V-Order
# ============================================================
print("V-Order session default:", spark.conf.get("spark.sql.parquet.vorder.default"))

AGG = f"{SCHEMA}.agg_country_month"
Q_AGG = f"SELECT country_code, SUM(revenue) r FROM {AGG} GROUP BY country_code"

bench(Q_AGG, "gold agg BEFORE V-Order", "2-vorder")

# Apply V-Order to the serving-layer tables
spark.conf.set("spark.sql.parquet.vorder.default", "true")
for t in ["agg_country_month","agg_age_country_month","agg_category_month","agg_customer_360"]:
    spark.sql(f"ALTER TABLE {SCHEMA}.{t} SET TBLPROPERTIES ('delta.parquet.vorder.enabled' = 'true')")
    spark.sql(f"OPTIMIZE {SCHEMA}.{t} VORDER")
    print("V-Ordered:", t)

bench(Q_AGG, "gold agg AFTER V-Order", "2-vorder")

# Three levels of control, worth showing explicitly:
#   session : spark.sql.parquet.vorder.default = true
#   table   : TBLPROPERTIES ('delta.parquet.vorder.enabled' = 'true')
#   write   : df.write.option("parquet.vorder.enabled","true")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 3 — Data skipping
# 
# This is usually the biggest win on a 5-billion-row table, and the most intuitive to explain:
# 
# > "The fastest data to read is the data you never read."
# 
# Delta stores min/max stats per file. If a file's `country_code` range can't contain `'IN'`, Spark skips
# the file entirely. But that only works if rows with the same `country_code` are **physically co-located**
# — which is exactly what Z-Order and Liquid Clustering do.
# 
# ### 3A — Z-Order (works on the partitioned table)

# CELL ********************

# ============================================================
# CELL 6 — LEVER 3A: Z-ORDER
# ============================================================
before = bench(Q2_FILTER, "Q2 filter BEFORE Z-Order", "3-zorder")

t0 = time.time()
spark.sql(f"""
OPTIMIZE {FACT}
WHERE txn_year = {LAST_YEAR}
ZORDER BY (country_code, age_group)
""")
print(f"Z-ORDER took {(time.time()-t0)/60:.1f} min")

after = bench(Q2_FILTER, "Q2 filter AFTER Z-Order", "3-zorder")
print(f"\n>>> Speedup: {before/after:.1f}x")

# TALKING POINTS
# - Pick 1-3 columns that dominate your interactive filters. Z-Ordering "everything" burns
#   compute for almost no benefit - the curve loses locality fast as dimensions increase.
# - Scoped with WHERE so we only reorganize the current year.
# - Z-Order must be re-run as new data lands; the definition is NOT stored on the table.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### 3B — Liquid Clustering (the modern alternative)
# 
# Liquid Clustering replaces both static partitioning **and** manual Z-Order. The key differences worth
# narrating:
# 
# | | Hive partitioning | Z-Order | Liquid Clustering |
# |---|---|---|---|
# | Physical layout | directories per value | min/max ranges in commit metadata | min/max ranges in commit metadata |
# | Change the columns | full rewrite | specify on every run | `ALTER TABLE ... CLUSTER BY` |
# | Definition stored on table | yes | **no** | **yes** |
# | Skew / high cardinality | poor | ok | good |
# 
# Under the hood: one clustering column uses a Z-order curve; two or more use a **Hilbert curve**, which
# preserves locality across dimensions better.
# 
# In **Fabric Runtime 2.0**, clustering became *incremental* — `OPTIMIZE` now only touches files that are
# unclustered, undersized, or heavy with deletion vectors, instead of rewriting entire ~100 GB file groups.
# That changed the economics: maintenance cost now scales with **new data**, not total table size. Worth
# mentioning if the customer previously evaluated liquid clustering and found it too expensive.


# CELL ********************

# ============================================================
# CELL 7 — LEVER 3B: Liquid Clustering (unpartitioned side-by-side)
# ============================================================
LC = f"{SCHEMA}.fact_txn_lc"

# Build an UNPARTITIONED clustered copy of one year (partitioning + CLUSTER BY are incompatible)
spark.sql(f"DROP TABLE IF EXISTS {LC}")
spark.sql(f"""
CREATE TABLE {LC}
CLUSTER BY (country_code, age_group)
AS SELECT * FROM {FACT} WHERE txn_year = {LAST_YEAR}
""")
print("Created clustered table. NOTE: not yet physically clustered!")
show_profiles(LC)

Q_LC_UNCLUSTERED = f"""
SELECT product_category, SUM(net_amount) r FROM {LC}
WHERE country_code='IN' AND age_group='25-34' GROUP BY product_category
"""
b1 = bench(Q_LC_UNCLUSTERED, "LC table BEFORE OPTIMIZE", "3-liquid")

# *** THE CRITICAL STEP *** clustering only materializes when OPTIMIZE runs
t0 = time.time()
spark.sql(f"OPTIMIZE {LC}")
print(f"Clustering OPTIMIZE took {(time.time()-t0)/60:.1f} min")

b2 = bench(Q_LC_UNCLUSTERED, "LC table AFTER OPTIMIZE", "3-liquid")
print(f"\n>>> Speedup: {b1/b2:.1f}x")
show_profiles(LC)

# Change clustering keys later WITHOUT rewriting - impossible with partitioning:
# spark.sql(f"ALTER TABLE {LC} CLUSTER BY (product_category, country_code)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 4 — Delta statistics and column ordering
# 
# A subtle one that most customers have never heard of, which makes it a good "you're in expert hands" moment.

# CELL ********************

# ============================================================
# CELL 8 — LEVER 4: stats-indexed columns
# ============================================================
# Delta only collects min/max stats for the FIRST N columns (default 32; we set 16 at load).
# A filter on a column beyond that boundary gets ZERO file skipping - it must read everything.
print("dataSkippingNumIndexedCols:",
      spark.sql(f"SHOW TBLPROPERTIES {FACT}").filter("key like '%dataSkipping%'").collect())

# Column ORDER in the schema is therefore a performance decision, not cosmetics.
spark.sql(f"DESCRIBE {FACT}").show(20, False)

# Filter on an EARLY column (indexed -> skipping works)
bench(f"SELECT COUNT(*) FROM {FACT} WHERE country_code='DE' AND txn_year={LAST_YEAR}",
      "filter on INDEXED column", "4-stats")

# Filter on a LATE column (not indexed -> full scan of the partition)
bench(f"SELECT COUNT(*) FROM {FACT} WHERE payment_method='UPI' AND txn_year={LAST_YEAR}",
      "filter on NON-indexed column", "4-stats")

# Fix: raise the boundary, or reorder columns so filter predicates sit early.
# spark.sql(f"ALTER TABLE {FACT} SET TBLPROPERTIES ('delta.dataSkippingNumIndexedCols'='32')")
# Then OPTIMIZE to rewrite files and regenerate stats.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 5 — Partition and column pruning
# 
# The cheapest wins in the whole deck, and the ones customers most often leave on the table.
# Show the **physical plan**, not just the clock.

# CELL ********************

# ============================================================
# CELL 9 — LEVER 5: pruning, proven in the plan
# ============================================================
print("=" * 70)
print("NO PRUNING — SELECT * across all 3 years")
print("=" * 70)
spark.sql(f"SELECT * FROM {FACT} WHERE net_amount > 500").explain("formatted")

print("\n" + "=" * 70)
print("PRUNED — partition filter + narrow projection")
print("=" * 70)
spark.sql(f"""
SELECT country_code, SUM(net_amount)
FROM {FACT}
WHERE txn_year = {LAST_YEAR} AND txn_month = {LAST_MON}
GROUP BY country_code
""").explain("formatted")

# In the plan, point at:
#   PartitionFilters  -> proves partition pruning
#   PushedFilters     -> proves predicate pushdown to Parquet
#   ReadSchema        -> proves column pruning (short list = good)
# In the Spark UI scan node: "number of files read" and "size of files read".

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 9b — Column pruning, measured
# ============================================================
bench(f"SELECT * FROM {FACT} WHERE txn_year={LAST_YEAR} AND txn_month={LAST_MON}",
      "SELECT * (all ~45 columns)", "5-pruning")

bench(f"SELECT country_code, net_amount FROM {FACT} WHERE txn_year={LAST_YEAR} AND txn_month={LAST_MON}",
      "SELECT 2 columns", "5-pruning")

# TALKING POINT: this is why the wide fact table exists in the demo. Columnar formats charge you
# only for the columns you touch - "SELECT *" in a notebook is a real, measurable cost.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 6 — Join strategy and skew
# 
# The classic Spark failure mode: one enormous shuffle, then a handful of stragglers holding up the stage.

# CELL ********************

# ============================================================
# CELL 10 — LEVER 6: broadcast, AQE, skew
# ============================================================
# --- BAD: broadcast disabled -> SortMergeJoin, full shuffle of both sides ---
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.adaptive.enabled", "false")
bad = bench(Q3_JOIN, "SortMergeJoin, AQE off", "6-joins")

spark.sql(Q3_JOIN).explain("formatted")   # show SortMergeJoin in the plan

# --- GOOD: AQE + broadcast + skew handling ---
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(100 * 1024 * 1024))
good = bench(Q3_JOIN, "AQE + broadcast + skew join", "6-joins")

print(f"\n>>> Speedup: {bad/good:.1f}x")

# TALKING POINTS
# - dim_product (200k rows) broadcasts easily; dim_customer (80M) does not -> AQE decides at RUNTIME
#   using actual shuffle statistics, which static planning cannot do.
# - Country skew is real in this dataset (US/IN/CN dominate) - a genuine skew-join demo, not synthetic.
# - AQE also coalesces the 1024 shuffle partitions down to what the data actually needs.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 10b — Skew, made visible
# ============================================================
display(spark.sql(f"""
SELECT country_code, COUNT(*) AS rows,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM {FACT} WHERE txn_year = {LAST_YEAR}
GROUP BY country_code ORDER BY rows DESC LIMIT 15
"""))
# Point at the top 3 countries. THAT is the skew AQE is compensating for.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 7 — Native Execution Engine
# 
# **The pitch, in one line:** *"Same code, same cluster, same cost — roughly 4× faster."*
# 
# NEE is a vectorized C++ execution layer built on **Velox** (Meta) and **Apache Gluten** (Intel). Supported
# operators are offloaded from JVM Spark to a columnar, SIMD-accelerated native path. It's **GA** in Fabric
# Runtime 1.3 (Spark 3.5) and supported on Runtime 2.0 (Spark 4.1), across Parquet, Delta and CSV.
# 
# Why JVM Spark leaves performance on the table:
# - Garbage collection introduces unpredictable pauses
# - Row-based processing is a poor fit for columnar Parquet/Delta
# - Minimal SIMD usage means modern CPUs sit underutilized
# 
# Microsoft's benchmark: **~4× faster on a 1 TB TPC-DS workload** vs vanilla Spark, with up to 6× end-to-end
# on some jobs. Your dataset is also 1 TB, so that's a fair comparison to reference.
# 
# Crucially, AQE, cost-based rewrites, column pruning and predicate pushdown all **stay active** when
# operators are offloaded — and NEE specifically accelerates tables organized with Z-Order and Liquid
# Clustering, so Levers 3 and 7 compound.


# CELL ********************

# ============================================================
# CELL 11 — LEVER 7: NEE off vs on
# ============================================================
spark.conf.set("spark.native.enabled", "false")
off1 = bench(Q1_SCAN,   "Q1 scan   — NEE OFF", "7-nee")
off2 = bench(Q2_FILTER, "Q2 filter — NEE OFF", "7-nee")
off3 = bench(Q3_JOIN,   "Q3 join   — NEE OFF", "7-nee")

spark.conf.set("spark.native.enabled", "true")
spark.conf.set("spark.shuffle.manager",
               "org.apache.spark.shuffle.sort.ColumnarShuffleManager")
print("\nspark.native.enabled =", spark.conf.get("spark.native.enabled"), "\n")

on1 = bench(Q1_SCAN,   "Q1 scan   — NEE ON", "7-nee")
on2 = bench(Q2_FILTER, "Q2 filter — NEE ON", "7-nee")
on3 = bench(Q3_JOIN,   "Q3 join   — NEE ON", "7-nee")

print(f"\n>>> Q1 {off1/on1:.1f}x   Q2 {off2/on2:.1f}x   Q3 {off3/on3:.1f}x")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ### Enabling NEE properly (three ways — show the UI path, it's what they'll actually use)
# 
# **1. Environment (recommended, inherits everywhere)**
# Fabric portal → **New item → Environment** → **Spark compute → Acceleration tab** →
# check **Enable native execution engine** → **Save + Publish**.
# All notebooks and Spark Job Definitions using that environment inherit it.
# 
# **2. Per-session** — put this in the **first cell** of a notebook:
# ```
# %%configure
# {
#     "conf": {
#         "spark.native.enabled": "true"
#     }
# }
# ```
# 
# **3. Workspace default** — Workspace settings → Data Engineering/Science → Environment → attach the
# NEE-enabled environment as default. Every new Spark workload picks it up with no per-job config.
# 
# Verify with `spark.conf.get("spark.native.enabled")`.
# 
# **Show which operators actually offloaded:** in the Spark UI SQL tab, native operators render distinctly
# (look for the columnar/native node names). Not every operator is supported — unsupported ones fall back
# to JVM Spark transparently, which is worth saying out loud so nobody is surprised by a mixed plan.


# CELL ********************

# ============================================================
# CELL 12 — Confirm NEE and inspect the plan for offloaded operators
# ============================================================
print("spark.native.enabled :", spark.conf.get("spark.native.enabled"))
print("shuffle manager      :", spark.conf.get("spark.shuffle.manager", "default"))
print()
spark.sql(Q1_SCAN).explain("formatted")
# Look for native/columnar operator names. Anything not offloaded ran on the JVM.

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 8 — Materialized Lake Views
# 
# Stop recomputing the same aggregate on 5 billion rows for every dashboard refresh.

# CELL ********************

# ============================================================
# CELL 13 — LEVER 8: MLV vs raw aggregation
# ============================================================
raw = bench(f"""
SELECT txn_year, txn_month, country_code, SUM(net_amount) r, COUNT(*) c
FROM {FACT} GROUP BY txn_year, txn_month, country_code
""", "aggregate from RAW fact", "8-mlv")

# CDF must be on for efficient incremental refresh (the banner in the Lakehouse UI)
spark.sql(f"ALTER TABLE {FACT} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

spark.sql(f"""
CREATE MATERIALIZED LAKE VIEW IF NOT EXISTS {SCHEMA}.mlv_country_month AS
SELECT txn_year, txn_month, country_code, region, age_group,
       SUM(net_amount)  AS revenue,
       SUM(margin_amount) AS margin,
       COUNT(*)         AS txn_count,
       COUNT(DISTINCT customer_id) AS active_customers
FROM {FACT}
GROUP BY txn_year, txn_month, country_code, region, age_group
""")

mlv = bench(f"SELECT txn_year, txn_month, country_code, SUM(revenue) FROM {SCHEMA}.mlv_country_month "
            f"GROUP BY txn_year, txn_month, country_code", "read from MLV", "8-mlv")

print(f"\n>>> Speedup: {raw/mlv:.0f}x")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 9 — Session, pool and concurrency configuration
# 
# Less visual, but this is where the customer's real-world complaints usually live.

# CELL ********************

# ============================================================
# CELL 14 — LEVER 9: session + pool settings that matter
# ============================================================
cfg = {
  "spark.sql.shuffle.partitions": "auto — let AQE coalesce; only pin it for very predictable jobs",
  "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128–256 MB target post-shuffle partition",
  "spark.sql.files.maxPartitionBytes": "default 128 MB; raise for large scans with few files",
  "spark.databricks.delta.optimizeWrite.enabled": "prevents small files at the source",
  "spark.sql.parquet.filterPushdown": "keep true — pushes predicates into the Parquet reader",
  "spark.dynamicAllocation.enabled": "scale executors with the workload",
}
for k, v in cfg.items():
    try: cur = spark.conf.get(k)
    except Exception: cur = "(unset)"
    print(f"{k:52s} {str(cur):10s}  {v}")

print("""
NOT set in code — configure these in the Fabric UI:
  • Starter pools      : ~5–10s session start (pre-warmed) vs minutes for a custom pool
  • Node size          : Medium/Large for big scans; more small nodes for high parallelism
  • High-concurrency   : session sharing across notebooks — big CU saver for a demo team
  • Autoscale + dynamic allocation : Workspace settings → Data Engineering/Science → Spark
  • Resource profiles  : readHeavyForSpark / writeHeavy presets instead of hand-tuning
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Lever 10 — Cost, not just speed
# 
# The question that always comes after "it's fast": *"what does it cost me?"*

# CELL ********************

# ============================================================
# CELL 15 — LEVER 10: efficiency framing + maintenance
# ============================================================
spark.sql(f"VACUUM {FACT} RETAIN 168 HOURS")
spark.sql(f"VACUUM {WEB}  RETAIN 168 HOURS")

display(spark.sql(f"DESCRIBE HISTORY {FACT}").select(
    "version","timestamp","operation","operationMetrics").limit(15))

print("""
COST TALKING POINTS
  • Faster query = fewer CU-seconds. Every lever here is a COST lever, not just a latency lever.
  • NEE is the rare free lunch: less runtime at no additional compute cost.
  • V-Order costs ~15% on write to buy read gains across Spark, SQL endpoint and Direct Lake.
    Apply it to gold/serving tables; skip it on bronze staging.
  • OPTIMIZE/VACUUM themselves burn CU. Schedule them off-peak; scope with WHERE.
  • VACUUM is a governance decision, not just cleanup — retention IS your rollback window.
  • Show the Fabric Capacity Metrics app: load burst vs steady-state query, smoothing, bursting.
""")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ============================================================
# CELL 16 — FINAL SCORECARD
# ============================================================
df = scorecard()

base = df[df.lever == "0-baseline"].set_index("label")["seconds"].to_dict()
print("\n" + "=" * 60)
print("CUMULATIVE IMPROVEMENT vs BASELINE")
print("=" * 60)
for lbl, sec in base.items():
    key = lbl.split("—")[0].strip()[:2]
    later = df[(df.label.str.startswith(key)) & (df.lever != "0-baseline")]
    if len(later):
        best = later.seconds.min()
        print(f"{lbl:28s} {sec:7.1f}s -> {best:6.1f}s   ({sec/best:.1f}x faster)")

# Persist the scorecard so you can screenshot / re-show it after the session ends
spark.createDataFrame(df).write.mode("overwrite").format("delta") \
     .saveAsTable(f"{SCHEMA}.demo_tuning_scorecard")
print("\nSaved to dbo.demo_tuning_scorecard")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## Closing narrative — the order that persuades
# 
# Don't present these as ten independent tricks. Present them as **four layers**, outermost first:
# 
# **1. Layout (Levers 1–4)** — the data on disk.
# *"Most Spark performance problems are storage problems wearing a compute costume."* Small files, no
# clustering, stats on the wrong columns. Cheapest to fix, biggest wins, and no code changes.
# 
# **2. Query (Levers 5–6)** — what you ask for.
# Pruning and join strategy. AQE means Spark now adapts at runtime using real statistics — which is why
# hand-tuning shuffle partitions is mostly obsolete advice.
# 
# **3. Engine (Lever 7)** — how it executes.
# NEE is the "flip a switch" moment. No rewrites, no lock-in, same cost.
# 
# **4. Pre-computation (Lever 8)** — don't compute it at all.
# MLVs and gold aggregates. The fastest query is the one that already ran.
# 
# Then close on **cost**: every one of these reduces CU consumption, so performance tuning and FinOps are
# the same conversation in Fabric.
# 
# ---
# 
# ### Common objections, and honest answers
# 
# | Objection | Answer |
# |---|---|
# | "Is V-Order proprietary lock-in?" | No — output is 100% open-source Parquet compliant. Any Parquet engine reads it. |
# | "Why is V-Order off by default then?" | Fabric optimizes new workspaces for write-heavy engineering. It's a deliberate default, not an oversight. Turn it on for serving tables. |
# | "Does NEE need code changes?" | None. Unsupported operators fall back to JVM Spark transparently. |
# | "Should we partition or cluster?" | Current guidance favours Liquid Clustering. Partitioning is rigid, needs a full rewrite to change, and conflicts with auto-compaction. |
# | "We tried liquid clustering, maintenance was too expensive." | That was pre-Runtime 2.0. Incremental clustering now scales with new data, not table size. Worth re-evaluating. |
# | "Can we run OPTIMIZE from the SQL endpoint?" | No — Spark notebook, Spark Job Definition, or the Lakehouse maintenance UI. |
# 
# ### Reset between demo runs
# 
# ```python
# spark.conf.set("spark.native.enabled", "false")
# spark.conf.set("spark.sql.parquet.vorder.default", "false")
# spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
# spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "false")
# spark.sql("DROP TABLE IF EXISTS dbo.fact_txn_lc")
# ```

