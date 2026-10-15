# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "e8f96c27-2c8d-4f51-9b6e-6ea7756b34a8",
# META       "default_lakehouse_name": "SalesLakehouse",
# META       "default_lakehouse_workspace_id": "5fdce046-399f-40c6-81a5-829f33ee5e00",
# META       "known_lakehouses": [
# META         {
# META           "id": "e8f96c27-2c8d-4f51-9b6e-6ea7756b34a8"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# We will create following four tables for MCP server demo and Fabric IQ demo
# 1. Customers
# 2. Products
# 3. Orders
# 4. OrderItems

# CELL ********************

from pyspark.sql import Row
from datetime import datetime

customers = [
Row(CustomerID=1, CustomerName="Contoso Retail", City="Bengaluru", Segment="Enterprise"),
Row(CustomerID=2, CustomerName="Fabrikam Stores", City="Mumbai", Segment="Enterprise"),
Row(CustomerID=3, CustomerName="Adventure Works", City="Delhi", Segment="Mid-Market"),
Row(CustomerID=4, CustomerName="Northwind Traders", City="Chennai", Segment="SMB"),
Row(CustomerID=5, CustomerName="Wide World Importers", City="Hyderabad", Segment="Enterprise")
]

products = [
Row(ProductID=101, ProductName="Surface Laptop", Category="Computers", UnitPrice=125000.0),
Row(ProductID=102, ProductName="Surface Pro", Category="Computers", UnitPrice=105000.0),
Row(ProductID=103, ProductName="Azure AI Subscription", Category="Cloud", UnitPrice=200000.0),
Row(ProductID=104, ProductName="Power BI Premium", Category="Analytics", UnitPrice=150000.0),
Row(ProductID=105, ProductName="Microsoft 365", Category="Productivity", UnitPrice=30000.0)
]

orders = [
Row(OrderID=1001, CustomerID=1, OrderDate=datetime(2026,8,5), Status="Completed"),
Row(OrderID=1002, CustomerID=2, OrderDate=datetime(2026,8,12), Status="Completed"),
Row(OrderID=1003, CustomerID=1, OrderDate=datetime(2026,9,3), Status="Completed"),
Row(OrderID=1004, CustomerID=3, OrderDate=datetime(2026,9,10), Status="Pending"),
Row(OrderID=1005, CustomerID=5, OrderDate=datetime(2026,9,18), Status="Completed")
]

order_items = [
Row(OrderID=1001, ProductID=101, Quantity=5),
Row(OrderID=1001, ProductID=105, Quantity=20),
Row(OrderID=1002, ProductID=104, Quantity=10),
Row(OrderID=1003, ProductID=103, Quantity=4),
Row(OrderID=1004, ProductID=102, Quantity=8),
Row(OrderID=1005, ProductID=103, Quantity=6),
Row(OrderID=1005, ProductID=104, Quantity=3)
]

spark.createDataFrame(customers).write.mode("overwrite").format("delta").saveAsTable("Customers")
spark.createDataFrame(products).write.mode("overwrite").format("delta").saveAsTable("Products")
spark.createDataFrame(orders).write.mode("overwrite").format("delta").saveAsTable("Orders")
spark.createDataFrame(order_items).write.mode("overwrite").format("delta").saveAsTable("OrderItems")

print("Fabric MCP demo tables created successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
