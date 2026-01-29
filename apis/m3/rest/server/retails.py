from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/retails/retails.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of line items based on order key and return flag
@app.get("/v1/retails/lineitem_count_by_orderkey_returnflag", operation_id="get_lineitem_count", summary="Retrieves the total number of line items associated with a particular order, filtered by a specified return flag. The order is identified by its unique key, and the return flag indicates the status of the order's return process.")
async def get_lineitem_count(orderkey: int = Query(..., description="Order key"), returnflag: str = Query(..., description="Return flag")):
    cursor.execute("SELECT COUNT(l_linenumber) FROM lineitem WHERE l_orderkey = ? AND l_returnflag = ?", (orderkey, returnflag))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum ship date for a specific order key
@app.get("/v1/retails/max_shipdate_by_orderkey", operation_id="get_max_shipdate", summary="Retrieves the latest ship date associated with a given order. The order is identified by its unique key, which is provided as an input parameter.")
async def get_max_shipdate(orderkey: int = Query(..., description="Order key")):
    cursor.execute("SELECT MAX(l_shipdate) FROM lineitem WHERE l_orderkey = ?", (orderkey,))
    result = cursor.fetchone()
    if not result:
        return {"max_shipdate": []}
    return {"max_shipdate": result[0]}

# Endpoint to get the order key with the latest ship date from a list of order keys
@app.get("/v1/retails/latest_orderkey_by_orderkeys", operation_id="get_latest_orderkey", summary="Retrieves the order key with the most recent ship date from a provided set of order keys. The input parameters specify the order keys to consider for this determination.")
async def get_latest_orderkey(orderkey1: int = Query(..., description="First order key"), orderkey2: int = Query(..., description="Second order key")):
    cursor.execute("SELECT l_orderkey FROM lineitem WHERE l_orderkey IN (?, ?) ORDER BY l_shipdate DESC LIMIT 1", (orderkey1, orderkey2))
    result = cursor.fetchone()
    if not result:
        return {"orderkey": []}
    return {"orderkey": result[0]}

# Endpoint to get the comment of the order with the highest total price
@app.get("/v1/retails/order_comment_by_max_totalprice", operation_id="get_order_comment_by_max_totalprice", summary="Retrieves the comment associated with the order that has the highest total price. This operation provides insight into the most expensive order placed, offering valuable context about the order's details.")
async def get_order_comment_by_max_totalprice():
    cursor.execute("SELECT o_comment FROM orders WHERE o_totalprice = ( SELECT MAX(o_totalprice) FROM orders )")
    result = cursor.fetchone()
    if not result:
        return {"comment": []}
    return {"comment": result[0]}

# Endpoint to get the phone number of a customer by name
@app.get("/v1/retails/customer_phone_by_name", operation_id="get_customer_phone", summary="Retrieves the phone number of a customer based on the provided name. The operation searches for a customer with the specified name and returns the associated phone number.")
async def get_customer_phone(name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT c_phone FROM customer WHERE c_name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the count of orders for a specific market segment
@app.get("/v1/retails/order_count_by_market_segment", operation_id="get_order_count_by_market_segment", summary="Retrieves the total number of orders associated with a particular market segment. The market segment is specified as an input parameter, allowing for a targeted count of orders.")
async def get_order_count_by_market_segment(market_segment: str = Query(..., description="Market segment")):
    cursor.execute("SELECT COUNT(T1.o_orderkey) FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_mktsegment = ?", (market_segment,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum total price of orders for a specific market segment
@app.get("/v1/retails/max_totalprice_by_market_segment", operation_id="get_max_totalprice_by_market_segment", summary="Retrieves the highest total price of orders for a given market segment. The market segment is specified as an input parameter.")
async def get_max_totalprice_by_market_segment(market_segment: str = Query(..., description="Market segment")):
    cursor.execute("SELECT MAX(T1.o_totalprice) FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_mktsegment = ?", (market_segment,))
    result = cursor.fetchone()
    if not result:
        return {"max_totalprice": []}
    return {"max_totalprice": result[0]}

# Endpoint to get the comments of orders for a specific market segment
@app.get("/v1/retails/order_comments_by_market_segment", operation_id="get_order_comments_by_market_segment", summary="Retrieves the comments of orders associated with a specific market segment. The market segment is used to filter the orders and return only the comments of those orders that belong to the specified market segment.")
async def get_order_comments_by_market_segment(market_segment: str = Query(..., description="Market segment")):
    cursor.execute("SELECT T1.o_comment FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_mktsegment = ?", (market_segment,))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": [row[0] for row in result]}

# Endpoint to get the customer name with the highest total order price
@app.get("/v1/retails/customer_name_by_max_totalprice", operation_id="get_customer_name_by_max_totalprice", summary="Retrieves the name of the customer who has made the highest total order price. This operation fetches the customer with the maximum total order price from the orders table and returns their name from the customer table.")
async def get_customer_name_by_max_totalprice():
    cursor.execute("SELECT T2.c_name FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey ORDER BY T1.o_totalprice DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the order keys for customers with account balance less than a specified value
@app.get("/v1/retails/orderkeys_by_account_balance", operation_id="get_orderkeys_by_account_balance", summary="Retrieves the order keys for customers whose account balance is less than the provided value. This operation allows you to filter orders based on the account balance of the customers who placed them.")
async def get_orderkeys_by_account_balance(account_balance: float = Query(..., description="Account balance")):
    cursor.execute("SELECT T1.o_orderkey FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_acctbal < ?", (account_balance,))
    result = cursor.fetchall()
    if not result:
        return {"orderkeys": []}
    return {"orderkeys": [row[0] for row in result]}

# Endpoint to get the count of orders based on market segment and order priority
@app.get("/v1/retails/order_count_by_segment_priority", operation_id="get_order_count", summary="Retrieves the total number of orders associated with a specific market segment and order priority. This operation provides a quantitative measure of order volume based on the given market segment and order priority, enabling users to analyze order trends and patterns.")
async def get_order_count(market_segment: str = Query(..., description="Market segment of the customer"), order_priority: str = Query(..., description="Order priority")):
    cursor.execute("SELECT COUNT(T1.o_orderpriority) FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_mktsegment = ? AND T1.o_orderpriority = ?", (market_segment, order_priority))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on nation name
@app.get("/v1/retails/customer_count_by_nation", operation_id="get_customer_count", summary="Retrieves the total number of customers associated with a specific nation. The nation is identified by its name, which is provided as an input parameter.")
async def get_customer_count(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT COUNT(T1.c_custkey) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T2.n_name = ?", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer phone numbers based on market segment and nation name
@app.get("/v1/retails/customer_phones_by_segment_nation", operation_id="get_customer_phones", summary="Retrieves the phone numbers of customers who belong to a specific market segment and reside in a particular nation. The operation requires the market segment and nation name as input parameters to filter the customer data.")
async def get_customer_phones(market_segment: str = Query(..., description="Market segment of the customer"), nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT T1.c_phone FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T1.c_mktsegment = ? AND T2.n_name = ?", (market_segment, nation_name))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": [row[0] for row in result]}

# Endpoint to get the count of customers based on nation name and account balance
@app.get("/v1/retails/customer_count_by_nation_balance", operation_id="get_customer_count_balance", summary="Retrieves the total number of customers from a specific nation who have an account balance greater than a specified minimum. The nation is identified by its name, and the minimum balance is provided as a numerical value.")
async def get_customer_count_balance(nation_name: str = Query(..., description="Name of the nation"), min_balance: float = Query(..., description="Minimum account balance")):
    cursor.execute("SELECT COUNT(T1.c_custkey) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T2.n_name = ? AND T1.c_acctbal > ?", (nation_name, min_balance))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers with orders based on nation name
@app.get("/v1/retails/customer_count_with_orders_by_nation", operation_id="get_customer_count_with_orders", summary="Retrieves the total number of customers who have placed orders in a specific nation. The nation is identified by its name.")
async def get_customer_count_with_orders(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT COUNT(T2.c_custkey) FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN orders AS T3 ON T2.c_custkey = T3.o_custkey WHERE T1.n_name = ?", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total order price based on nation name
@app.get("/v1/retails/total_order_price_by_nation", operation_id="get_total_order_price", summary="Retrieves the total order price for a specified nation. This operation calculates the sum of all order prices associated with a particular nation, providing a comprehensive view of the total sales volume. The nation is identified by its name, which is provided as an input parameter.")
async def get_total_order_price(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT SUM(T3.o_totalprice) FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN orders AS T3 ON T2.c_custkey = T3.o_custkey WHERE T1.n_name = ?", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_price": []}
    return {"total_price": result[0]}

# Endpoint to get the first order key based on nation name
@app.get("/v1/retails/first_order_key_by_nation", operation_id="get_first_order_key", summary="Retrieves the order key of the first order placed by a customer from a specified nation. The order key is determined based on the earliest order date.")
async def get_first_order_key(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT T3.o_orderkey FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN orders AS T3 ON T2.c_custkey = T3.o_custkey WHERE T1.n_name = ? ORDER BY T3.o_orderdate LIMIT 1", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"order_key": []}
    return {"order_key": result[0]}

# Endpoint to get the average order price based on nation name
@app.get("/v1/retails/average_order_price_by_nation", operation_id="get_average_order_price", summary="Retrieves the average total order price for a specified nation. This operation calculates the average order price by aggregating order totals for customers associated with the given nation. The result provides a statistical overview of purchasing trends within the specified nation.")
async def get_average_order_price(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT AVG(T3.o_totalprice) FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN orders AS T3 ON T2.c_custkey = T3.o_custkey WHERE T1.n_name = ?", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the percentage of customers from a specific nation
@app.get("/v1/retails/percentage_customers_by_nation", operation_id="get_percentage_customers", summary="Retrieves the percentage of customers from a specified nation. This operation calculates the proportion of customers from the given nation by comparing the total count of customers from that nation to the overall customer count. The input parameter determines the nation for which the customer percentage is computed.")
async def get_percentage_customers(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.n_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.c_custkey) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of nations based on region key
@app.get("/v1/retails/nation_count_by_region", operation_id="get_nation_count", summary="Retrieves the total number of nations associated with the specified region. The region is identified by its unique key.")
async def get_nation_count(region_key: int = Query(..., description="Region key")):
    cursor.execute("SELECT COUNT(n_nationkey) FROM nation WHERE n_regionkey = ?", (region_key,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the nation name based on supplier key
@app.get("/v1/retails/nation_name_by_supplier_key", operation_id="get_nation_name_by_supplier_key", summary="Retrieves the name of the nation associated with the specified supplier key. The supplier key is used to identify the supplier, and the nation name is obtained from the nation table using the supplier's nation key.")
async def get_nation_name_by_supplier_key(s_suppkey: int = Query(..., description="Supplier key")):
    cursor.execute("SELECT T2.n_name FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T1.s_suppkey = ?", (s_suppkey,))
    result = cursor.fetchall()
    if not result:
        return {"nation_names": []}
    return {"nation_names": [row[0] for row in result]}

# Endpoint to get the region name based on supplier name
@app.get("/v1/retails/region_name_by_supplier_name", operation_id="get_region_name_by_supplier_name", summary="Retrieves the name of the region associated with the specified supplier. The operation uses the provided supplier name to look up the corresponding region name in the database.")
async def get_region_name_by_supplier_name(s_name: str = Query(..., description="Supplier name")):
    cursor.execute("SELECT T3.r_name FROM nation AS T1 INNER JOIN supplier AS T2 ON T1.n_nationkey = T2.s_nationkey INNER JOIN region AS T3 ON T1.n_regionkey = T3.r_regionkey WHERE T2.s_name = ?", (s_name,))
    result = cursor.fetchall()
    if not result:
        return {"region_names": []}
    return {"region_names": [row[0] for row in result]}

# Endpoint to get the nation name based on customer name
@app.get("/v1/retails/nation_name_by_customer_name", operation_id="get_nation_name_by_customer_name", summary="Retrieves the nation name associated with the specified customer. This operation uses the provided customer name to look up the corresponding nation name in the database.")
async def get_nation_name_by_customer_name(c_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT T2.n_name FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_name = ?", (c_name,))
    result = cursor.fetchall()
    if not result:
        return {"nation_names": []}
    return {"nation_names": [row[0] for row in result]}

# Endpoint to get the region name based on customer key
@app.get("/v1/retails/region_name_by_customer_key", operation_id="get_region_name_by_customer_key", summary="Retrieves the name of the region associated with the provided customer key. This operation fetches the region name by joining the customer, nation, and region tables using the respective keys. The customer key is used to filter the results.")
async def get_region_name_by_customer_key(c_custkey: int = Query(..., description="Customer key")):
    cursor.execute("SELECT T3.r_name FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN region AS T3 ON T1.n_regionkey = T3.r_regionkey WHERE T2.c_custkey = ?", (c_custkey,))
    result = cursor.fetchall()
    if not result:
        return {"region_names": []}
    return {"region_names": [row[0] for row in result]}

# Endpoint to get the count of customers based on nation name and account balance
@app.get("/v1/retails/customer_count_by_nation_and_balance", operation_id="get_customer_count_by_nation_and_balance", summary="Retrieves the total number of customers from a specific nation with an account balance less than the provided amount. The nation is identified by its name, and the account balance is specified as a numerical value.")
async def get_customer_count_by_nation_and_balance(n_name: str = Query(..., description="Nation name"), c_acctbal: float = Query(..., description="Account balance")):
    cursor.execute("SELECT COUNT(T1.c_name) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T2.n_name = ? AND T1.c_acctbal < ?", (n_name, c_acctbal))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the discounted price based on supplier key and total price
@app.get("/v1/retails/discounted_price_by_supplier_and_total", operation_id="get_discounted_price_by_supplier_and_total", summary="Retrieves the discounted price for a specific supplier based on the total price of the order. This operation calculates the discounted price by multiplying the extended price of the line item by the discount percentage. The supplier is identified by the provided supplier key, and the total price of the order is used to filter the results. The discounted price is returned for the matching line items.")
async def get_discounted_price_by_supplier_and_total(l_suppkey: int = Query(..., description="Supplier key"), o_totalprice: float = Query(..., description="Total price")):
    cursor.execute("SELECT T1.l_extendedprice * (1 - T1.l_discount) AS DISCOUNTERPRICE FROM lineitem AS T1 INNER JOIN orders AS T2 ON T2.o_orderkey = T1.l_orderkey WHERE T1.l_suppkey = ? AND T2.o_totalprice = ?", (l_suppkey, o_totalprice))
    result = cursor.fetchall()
    if not result:
        return {"discounted_prices": []}
    return {"discounted_prices": [row[0] for row in result]}

# Endpoint to get the supplier key based on total price and return flag
@app.get("/v1/retails/supplier_key_by_total_and_return_flag", operation_id="get_supplier_key_by_total_and_return_flag", summary="Retrieves the supplier key associated with a specific total price and return flag. This operation filters orders based on the provided total price and return flag, and returns the corresponding supplier key. The total price and return flag are input parameters that determine the filtering criteria.")
async def get_supplier_key_by_total_and_return_flag(o_totalprice: float = Query(..., description="Total price"), l_returnflag: str = Query(..., description="Return flag")):
    cursor.execute("SELECT T2.l_suppkey FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_totalprice = ? AND T2.l_returnflag = ?", (o_totalprice, l_returnflag))
    result = cursor.fetchall()
    if not result:
        return {"supplier_keys": []}
    return {"supplier_keys": [row[0] for row in result]}

# Endpoint to get the calculated price based on clerk, ship mode, order status, and order date
@app.get("/v1/retails/calculated_price_by_clerk_shipmode_status_date", operation_id="get_calculated_price_by_clerk_shipmode_status_date", summary="Retrieves the calculated price for a specific clerk, ship mode, order status, and order date. The calculation is based on the extended price, discount, and tax of the order. The clerk, ship mode, and order status are used to filter the results, while the order date is used to specify the time period.")
async def get_calculated_price_by_clerk_shipmode_status_date(o_clerk: str = Query(..., description="Clerk"), l_shipmode: str = Query(..., description="Ship mode"), o_orderstatus: str = Query(..., description="Order status"), o_orderdate: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.l_extendedprice * (1 - T2.l_discount) * (1 + T2.l_tax) AS num FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_clerk = ? AND T2.l_shipmode = ? AND T1.o_orderstatus = ? AND T1.o_orderdate = ?", (o_clerk, l_shipmode, o_orderstatus, o_orderdate))
    result = cursor.fetchall()
    if not result:
        return {"calculated_prices": []}
    return {"calculated_prices": [row[0] for row in result]}

# Endpoint to get the difference in days between receipt date and commit date based on customer key and order date
@app.get("/v1/retails/days_difference_by_customer_and_order_date", operation_id="get_days_difference_by_customer_and_order_date", summary="Retrieves the time difference in days between the receipt date and the commit date for a specific customer's order on a given date. The calculation is based on the provided customer key and order date.")
async def get_days_difference_by_customer_and_order_date(o_custkey: int = Query(..., description="Customer key"), o_orderdate: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT JULIANDAY(T2.l_receiptdate) - JULIANDAY(T2.l_commitdate) FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_custkey = ? AND T1.o_orderdate = ?", (o_custkey, o_orderdate))
    result = cursor.fetchall()
    if not result:
        return {"days_difference": []}
    return {"days_difference": [row[0] for row in result]}

# Endpoint to get the customer name based on order date and clerk
@app.get("/v1/retails/customer_name_by_order_date_and_clerk", operation_id="get_customer_name_by_order_date_and_clerk", summary="Retrieves the name of the customer who placed an order on a specific date and was served by a particular clerk. The operation requires the order date in 'YYYY-MM-DD' format and the clerk's identifier as input parameters.")
async def get_customer_name_by_order_date_and_clerk(o_orderdate: str = Query(..., description="Order date in 'YYYY-MM-DD' format"), o_clerk: str = Query(..., description="Clerk")):
    cursor.execute("SELECT T2.c_name FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T1.o_orderdate = ? AND T1.o_clerk = ?", (o_orderdate, o_clerk))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the calculated value based on supplier key and order key
@app.get("/v1/retails/calculated_value_by_suppkey_orderkey", operation_id="get_calculated_value", summary="Get the calculated value based on supplier key and order key")
async def get_calculated_value(suppkey: int = Query(..., description="Supplier key"), orderkey: int = Query(..., description="Order key")):
    cursor.execute("SELECT T1.l_extendedprice * (1 - T1.l_discount) - T2.ps_supplycost * T1.l_quantity FROM lineitem AS T1 INNER JOIN partsupp AS T2 ON T1.l_suppkey = T2.ps_suppkey WHERE T1.l_suppkey = ? AND T1.l_orderkey = ?", (suppkey, orderkey))
    result = cursor.fetchone()
    if not result:
        return {"value": []}
    return {"value": result[0]}

# Endpoint to get the nation name with the highest negative account balance
@app.get("/v1/retails/nation_with_highest_negative_balance", operation_id="get_nation_with_highest_negative_balance", summary="Retrieves the name of the nation with the highest negative account balance. This operation calculates the sum of negative account balances for each nation and returns the nation with the highest total. The result provides insight into the nation with the most significant negative financial impact.")
async def get_nation_with_highest_negative_balance():
    cursor.execute("SELECT T.n_name FROM ( SELECT T2.n_name, SUM(T1.s_acctbal) AS num FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T1.s_acctbal < 0 GROUP BY T2.n_name ) AS T ORDER BY T.num LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"nation_name": []}
    return {"nation_name": result[0]}

# Endpoint to get the percentage of nations in a specific region
@app.get("/v1/retails/percentage_nations_in_region", operation_id="get_percentage_nations_in_region", summary="Retrieves the percentage of nations located in a specified region. The calculation is based on the total count of nations in the database and the count of nations in the given region. The region is identified by its name.")
async def get_percentage_nations_in_region(region_name: str = Query(..., description="Region name")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.r_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.n_name) FROM nation AS T1 INNER JOIN region AS T2 ON T1.n_regionkey = T2.r_regionkey", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of suppliers with negative account balance in a specific nation
@app.get("/v1/retails/percentage_suppliers_negative_balance_nation", operation_id="get_percentage_suppliers_negative_balance_nation", summary="Retrieves the percentage of suppliers with a negative account balance in a specified nation. The nation is identified by its name, which is used to filter the suppliers and calculate the percentage.")
async def get_percentage_suppliers_negative_balance_nation(nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.n_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.s_name) FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T1.s_acctbal < 0", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the customer with the minimum account balance
@app.get("/v1/retails/customer_with_minimum_balance", operation_id="get_customer_with_minimum_balance", summary="Retrieves the name of the customer who has the lowest account balance. This operation is useful for identifying the customer with the least funds in their account. The result is based on the current account balances of all customers.")
async def get_customer_with_minimum_balance():
    cursor.execute("SELECT c_name FROM customer WHERE c_acctbal = ( SELECT MIN(c_acctbal) FROM customer )")
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the count of orders shipped in a specific year
@app.get("/v1/retails/count_orders_shipped_year", operation_id="get_count_orders_shipped_year", summary="Retrieves the total number of orders that were shipped in a specified year. The year must be provided in the 'YYYY' format.")
async def get_count_orders_shipped_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(l_orderkey) FROM lineitem WHERE STRFTIME('%Y', l_shipdate) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers with negative account balance
@app.get("/v1/retails/count_customers_negative_balance", operation_id="get_count_customers_negative_balance", summary="Retrieves the total number of customers who currently have a negative account balance.")
async def get_count_customers_negative_balance():
    cursor.execute("SELECT COUNT(c_custkey) FROM customer WHERE c_acctbal < 0")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of line items based on return flag, ship mode, and ship year
@app.get("/v1/retails/count_lineitems_returnflag_shipmode_year", operation_id="get_count_lineitems_returnflag_shipmode_year", summary="Retrieves the total number of line items that match the specified return flag, ship mode, and ship year. The return flag indicates whether the item was returned, the ship mode specifies the shipping method, and the ship year is the year when the item was shipped.")
async def get_count_lineitems_returnflag_shipmode_year(return_flag: str = Query(..., description="Return flag"), ship_mode: str = Query(..., description="Ship mode"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(l_linenumber) FROM lineitem WHERE l_returnflag = ? AND l_shipmode = ? AND STRFTIME('%Y', l_shipdate) = ?", (return_flag, ship_mode, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers in a specific market segment
@app.get("/v1/retails/count_customers_market_segment", operation_id="get_count_customers_market_segment", summary="Retrieves the total number of customers belonging to a specified market segment.")
async def get_count_customers_market_segment(market_segment: str = Query(..., description="Market segment")):
    cursor.execute("SELECT COUNT(c_custkey) FROM customer WHERE c_mktsegment = ?", (market_segment,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top N order keys based on extended price
@app.get("/v1/retails/top_orderkeys_by_extendedprice", operation_id="get_top_orderkeys_by_extendedprice", summary="Retrieves the top N order keys, ranked by their respective extended prices in descending order. The 'limit' parameter determines the number of top order keys to return.")
async def get_top_orderkeys_by_extendedprice(limit: int = Query(..., description="Number of top order keys to retrieve")):
    cursor.execute("SELECT l_orderkey FROM lineitem ORDER BY l_extendedprice DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"order_keys": []}
    return {"order_keys": [row[0] for row in result]}

# Endpoint to get the ship date of the highest total price order
@app.get("/v1/retails/highest_total_price_order_ship_date", operation_id="get_highest_total_price_order_ship_date", summary="Retrieves the ship date of the order with the highest total price. This operation identifies the order with the highest total price and returns the corresponding ship date. The result provides valuable insights into the order with the most significant total price.")
async def get_highest_total_price_order_ship_date():
    cursor.execute("SELECT T2.l_shipdate FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey ORDER BY T1.o_totalprice DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ship_date": []}
    return {"ship_date": result[0]}

# Endpoint to get the nation with the highest number of customers
@app.get("/v1/retails/nation_with_most_customers", operation_id="get_nation_with_most_customers", summary="Retrieves the name of the nation with the highest number of customers. This operation calculates the total number of customers for each nation and returns the nation with the maximum count.")
async def get_nation_with_most_customers():
    cursor.execute("SELECT T.n_name FROM ( SELECT T2.n_name, COUNT(T1.c_custkey) AS num FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey GROUP BY T2.n_name ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"nation": []}
    return {"nation": result[0]}

# Endpoint to get the count of orders shipped exactly one day after the order date with a specific priority
@app.get("/v1/retails/count_orders_shipped_one_day_after", operation_id="get_count_orders_shipped_one_day_after", summary="Retrieves the total number of orders that were shipped exactly one day after their order date, filtered by a specified order priority. This operation is useful for analyzing order fulfillment efficiency based on priority levels.")
async def get_count_orders_shipped_one_day_after(order_priority: str = Query(..., description="Order priority (e.g., '1-URGENT')")):
    cursor.execute("SELECT COUNT(T2.o_orderkey) FROM lineitem AS T1 INNER JOIN orders AS T2 ON T2.o_orderkey = T1.l_orderkey WHERE JULIANDAY(T1.l_shipdate) - JULIANDAY(T2.o_orderdate) = 1 AND T2.o_orderpriority = ?", (order_priority,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of orders with a specific ship mode and order priority
@app.get("/v1/retails/count_orders_by_shipmode_priority", operation_id="get_count_orders_by_shipmode_priority", summary="Retrieves the total number of orders that have a specified shipping method and order priority. The response is based on a combination of data from the orders and lineitem tables, filtered by the provided ship mode and order priority.")
async def get_count_orders_by_shipmode_priority(shipmode: str = Query(..., description="Ship mode (e.g., 'SHIP')"), orderpriority: str = Query(..., description="Order priority (e.g., '3-MEDIUM')")):
    cursor.execute("SELECT COUNT(T1.o_orderkey) FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T2.l_shipmode = ? AND T1.o_orderpriority = ?", (shipmode, orderpriority))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the market segment with the highest number of customers in a specific nation
@app.get("/v1/retails/top_market_segment_by_nation", operation_id="get_top_market_segment_by_nation", summary="Retrieves the market segment with the most customers in a specified nation. The operation filters customers by nation and calculates the number of customers per market segment. The market segment with the highest customer count is then returned.")
async def get_top_market_segment_by_nation(nation: str = Query(..., description="Nation name (e.g., 'UNITED STATES')")):
    cursor.execute("SELECT T.c_mktsegment FROM ( SELECT T1.c_mktsegment, COUNT(T1.c_custkey) AS num FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T2.n_name = ? GROUP BY T1.c_mktsegment ) AS T ORDER BY T.num DESC LIMIT 1", (nation,))
    result = cursor.fetchone()
    if not result:
        return {"market_segment": []}
    return {"market_segment": result[0]}

# Endpoint to get the names of nations in a specific region
@app.get("/v1/retails/nations_by_region", operation_id="get_nations_by_region", summary="Retrieves the names of all nations located within a specified region. The region is identified by its name, which is provided as an input parameter.")
async def get_nations_by_region(region: str = Query(..., description="Region name (e.g., 'ASIA')")):
    cursor.execute("SELECT T1.n_name FROM nation AS T1 INNER JOIN region AS T2 ON T1.n_regionkey = T2.r_regionkey WHERE T2.r_name = ?", (region,))
    result = cursor.fetchall()
    if not result:
        return {"nations": []}
    return {"nations": [row[0] for row in result]}

# Endpoint to get the names of parts with a specific supply cost and manufacturer
@app.get("/v1/retails/parts_by_supplycost_manufacturer", operation_id="get_parts_by_supplycost_manufacturer", summary="Retrieves the names of parts that have a specified supply cost and are manufactured by a given manufacturer. The supply cost is a numeric value, and the manufacturer is identified by its name.")
async def get_parts_by_supplycost_manufacturer(supplycost: float = Query(..., description="Supply cost (e.g., 1000)"), manufacturer: str = Query(..., description="Manufacturer name (e.g., 'Manufacturer#3')")):
    cursor.execute("SELECT T2.p_name FROM partsupp AS T1 INNER JOIN part AS T2 ON T1.ps_partkey = T2.p_partkey WHERE T1.ps_supplycost = ? AND T2.p_mfgr = ?", (supplycost, manufacturer))
    result = cursor.fetchall()
    if not result:
        return {"parts": []}
    return {"parts": [row[0] for row in result]}

# Endpoint to get the count of nations in a region with a specific comment
@app.get("/v1/retails/count_nations_by_region_comment", operation_id="get_count_nations_by_region_comment", summary="Retrieves the total number of nations within a specific region, based on the provided region comment. The region comment is a descriptive text that characterizes the region.")
async def get_count_nations_by_region_comment(comment: str = Query(..., description="Region comment (e.g., 'asymptotes sublate after the r')")):
    cursor.execute("SELECT COUNT(T1.n_nationkey) FROM nation AS T1 INNER JOIN region AS T2 ON T1.n_regionkey = T2.r_regionkey WHERE T2.r_comment = ?", (comment,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of part keys based on manufacturer, retail price, and ship mode
@app.get("/v1/retails/count_part_keys_by_manufacturer_price_shipmode", operation_id="get_count_part_keys", summary="Retrieves the total count of part keys that meet the specified manufacturer, retail price, and ship mode criteria. The manufacturer parameter filters the results by manufacturer name, the retail price parameter limits the results to parts with a retail price below the specified value, and the ship mode parameter restricts the results to a specific ship mode.")
async def get_count_part_keys(manufacturer: str = Query(..., description="Manufacturer name"), retail_price: float = Query(..., description="Maximum retail price"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT COUNT(T1.ps_partkey) FROM partsupp AS T1 INNER JOIN lineitem AS T2 ON T1.ps_suppkey = T2.l_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey WHERE T3.p_mfgr = ? AND T3.p_retailprice < ? AND T2.l_shipmode = ?", (manufacturer, retail_price, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top 10 orders by total price and their average extended price
@app.get("/v1/retails/top_orders_by_total_price", operation_id="get_top_orders", summary="Retrieves the top 10 orders with the highest total price and calculates their average extended price. The extended price is determined by multiplying the quantity of each item in an order by its price.")
async def get_top_orders():
    cursor.execute("SELECT SUM(T2.l_extendedprice) / 10 FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey ORDER BY T1.o_totalprice DESC LIMIT 10")
    result = cursor.fetchone()
    if not result:
        return {"average_extended_price": []}
    return {"average_extended_price": result[0]}

# Endpoint to get the top 3 customers by order count and their average order total price
@app.get("/v1/retails/top_customers_by_order_count", operation_id="get_top_customers", summary="Retrieves the top three customers with the highest order count and their respective average order total price. The operation calculates the average order total price for each customer and ranks them based on the total number of orders placed. The result is a list of the top three customers with the highest order count and their corresponding average order total price.")
async def get_top_customers():
    cursor.execute("SELECT T.c_name, T.res FROM ( SELECT T2.c_name, SUM(T1.o_totalprice) / COUNT(T1.o_orderkey) AS res , COUNT(T1.o_orderkey) AS num FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey GROUP BY T1.o_custkey ) AS T ORDER BY T.num DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"name": row[0], "average_total_price": row[1]} for row in result]}

# Endpoint to get the count of line items based on ship date
@app.get("/v1/retails/count_line_items_by_ship_date", operation_id="get_count_line_items", summary="Retrieves the total number of line items that were shipped on a specific date. The date must be provided in the 'YYYY-MM-DD' format.")
async def get_count_line_items(ship_date: str = Query(..., description="Ship date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(l_linenumber) FROM lineitem WHERE l_shipdate = ?", (ship_date,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the order date of the order with the maximum total price
@app.get("/v1/retails/order_date_max_total_price", operation_id="get_order_date_max_total_price", summary="Retrieves the date of the order with the highest total price from the orders database.")
async def get_order_date_max_total_price():
    cursor.execute("SELECT o_orderdate FROM orders WHERE o_totalprice = ( SELECT MAX(o_totalprice) FROM orders )")
    result = cursor.fetchone()
    if not result:
        return {"order_date": []}
    return {"order_date": result[0]}

# Endpoint to get the percentage of customers with a negative account balance
@app.get("/v1/retails/percentage_negative_account_balance", operation_id="get_percentage_negative_account_balance", summary="Retrieves the percentage of customers who have a negative account balance. This operation calculates the proportion of customers with a negative balance by summing the number of customers with a negative balance and dividing it by the total number of customers. The result is then multiplied by 100 to obtain the percentage.")
async def get_percentage_negative_account_balance():
    cursor.execute("SELECT CAST(SUM(IIF(c_acctbal < 0, 1, 0)) AS REAL) * 100 / COUNT(c_custkey) FROM customer")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of supplier keys based on available quantity
@app.get("/v1/retails/count_supplier_keys_by_available_quantity", operation_id="get_count_supplier_keys", summary="Retrieves the total number of unique supplier keys associated with part supplies that have an available quantity less than the specified value. This operation is useful for understanding the distribution of part supplies across suppliers based on their available quantities.")
async def get_count_supplier_keys(available_quantity: int = Query(..., description="Available quantity")):
    cursor.execute("SELECT COUNT(ps_suppkey) FROM partsupp WHERE ps_availqty < ?", (available_quantity,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of parts from a specific manufacturer
@app.get("/v1/retails/percentage_parts_by_manufacturer", operation_id="get_percentage_parts_by_manufacturer", summary="Retrieves the percentage of parts produced by a specific manufacturer. The operation calculates the ratio of parts made by the given manufacturer to the total number of parts in the database, providing a quantitative measure of the manufacturer's contribution to the overall parts inventory.")
async def get_percentage_parts_by_manufacturer(manufacturer: str = Query(..., description="Manufacturer name")):
    cursor.execute("SELECT CAST(SUM(IIF(p_mfgr = ?, 1, 0)) AS REAL) * 100 / COUNT(p_partkey) FROM part", (manufacturer,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the names of parts based on part type
@app.get("/v1/retails/part_names_by_type", operation_id="get_part_names_by_type", summary="Retrieves the names of up to five parts that match the specified part type. This operation is useful for obtaining a concise list of part names based on a given part type.")
async def get_part_names_by_type(part_type: str = Query(..., description="Part type")):
    cursor.execute("SELECT p_name FROM part WHERE p_type = ? LIMIT 5", (part_type,))
    result = cursor.fetchall()
    if not result:
        return {"part_names": []}
    return {"part_names": [row[0] for row in result]}

# Endpoint to get the count of customers based on account balance and nation name
@app.get("/v1/retails/customer_count_by_balance_nation", operation_id="get_customer_count_by_balance", summary="Retrieves the total number of customers who have an account balance below the specified value and belong to the specified nation.")
async def get_customer_count_by_balance(account_balance: float = Query(..., description="Account balance"), nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT COUNT(T1.c_custkey) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T1.c_acctbal < ? AND T2.n_name = ?", (account_balance, nation_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get nation details based on region name
@app.get("/v1/retails/nation_details_by_region", operation_id="get_nation_details", summary="Retrieves the names and unique identifiers of nations located within the specified region. The region is identified by its name.")
async def get_nation_details(region_name: str = Query(..., description="Region name")):
    cursor.execute("SELECT T1.n_name, T1.n_nationkey FROM nation AS T1 INNER JOIN region AS T2 ON T1.n_regionkey = T2.r_regionkey WHERE T2.r_name = ?", (region_name,))
    result = cursor.fetchall()
    if not result:
        return {"nations": []}
    return {"nations": [{"name": row[0], "nationkey": row[1]} for row in result]}

# Endpoint to get order details for a specific customer
@app.get("/v1/retails/order_details_by_customer", operation_id="get_order_details", summary="Retrieves the total number of orders and the sum of the extended prices (after discount and tax) for a specific customer's orders. The customer is identified by their name.")
async def get_order_details(customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT COUNT(T2.o_orderkey), SUM(T3.l_extendedprice * (1 - T3.l_discount) * (1 + T3.l_tax)) FROM customer AS T1 INNER JOIN orders AS T2 ON T1.c_custkey = T2.o_custkey INNER JOIN lineitem AS T3 ON T2.o_orderkey = T3.l_orderkey WHERE T1.c_name = ? GROUP BY T3.l_linenumber", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"order_details": []}
    return {"order_details": [{"order_count": row[0], "total_price": row[1]} for row in result]}

# Endpoint to get the total profit for a specific part
@app.get("/v1/retails/total_profit_by_part", operation_id="get_total_profit", summary="Retrieves the total profit generated by a specific part. This operation calculates the total profit by summing the extended price (after discount) of all line items for the specified part, subtracting the total supply cost for the part. The part name is required as an input parameter.")
async def get_total_profit(part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT SUM(T3.l_extendedprice * (1 - T3.l_discount) - T2.ps_supplycost * T3.l_quantity) FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN lineitem AS T3 ON T2.ps_partkey = T3.l_partkey AND T2.ps_suppkey = T3.l_suppkey WHERE T1.p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_profit": []}
    return {"total_profit": result[0]}

# Endpoint to get the percentage of suppliers from a specific nation with negative account balance
@app.get("/v1/retails/supplier_percentage_by_nation", operation_id="get_supplier_percentage", summary="Retrieves the percentage of suppliers from a specified nation who have a negative account balance. The nation is identified by its name, and the account balance threshold is provided as a parameter. The result is calculated by summing the number of suppliers from the specified nation with a negative account balance and dividing it by the total number of suppliers in that nation.")
async def get_supplier_percentage(nation_name: str = Query(..., description="Nation name"), account_balance: float = Query(..., description="Account balance")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.n_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.s_suppkey) FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T1.s_acctbal < ?", (nation_name, account_balance))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get supplier names for a specific part
@app.get("/v1/retails/supplier_names_by_part", operation_id="get_supplier_names", summary="Retrieves the names of suppliers who provide a specific part. The part is identified by its name, which is provided as an input parameter.")
async def get_supplier_names(part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT T3.s_name FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey WHERE T1.p_name = ?", (part_name,))
    result = cursor.fetchall()
    if not result:
        return {"supplier_names": []}
    return {"supplier_names": [row[0] for row in result]}

# Endpoint to get the count of suppliers with negative account balance in a specific region
@app.get("/v1/retails/supplier_count_by_region", operation_id="get_supplier_count", summary="Retrieves the number of suppliers with a negative account balance in a specified region. The operation requires the account balance and region name as input parameters to filter the suppliers and region, respectively.")
async def get_supplier_count(account_balance: float = Query(..., description="Account balance"), region_name: str = Query(..., description="Region name")):
    cursor.execute("SELECT COUNT(T3.s_name) FROM region AS T1 INNER JOIN nation AS T2 ON T1.r_regionkey = T2.n_regionkey INNER JOIN supplier AS T3 ON T2.n_nationkey = T3.s_nationkey WHERE T3.s_acctbal < ? AND T1.r_name = ?", (account_balance, region_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get part names based on discount, ship date, and ship mode
@app.get("/v1/retails/part_names_by_discount_shipdate_shipmode", operation_id="get_part_names", summary="Retrieves the names of parts that meet the specified discount, ship date, and ship mode criteria. The discount value, ship date in 'YYYY-MM-DD' format, and ship mode are used to filter the results. This operation returns a list of part names that match the provided conditions.")
async def get_part_names(discount: float = Query(..., description="Discount value"), ship_date: str = Query(..., description="Ship date in 'YYYY-MM-DD' format"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT T2.p_name FROM partsupp AS T1 INNER JOIN part AS T2 ON T1.ps_partkey = T2.p_partkey INNER JOIN lineitem AS T3 ON T1.ps_partkey = T3.l_partkey WHERE T3.l_discount = ? AND T3.l_shipdate = ? AND T3.l_shipmode = ?", (discount, ship_date, ship_mode))
    result = cursor.fetchall()
    if not result:
        return {"part_names": []}
    return {"part_names": [row[0] for row in result]}

# Endpoint to get part names based on supply cost and supplier name
@app.get("/v1/retails/part_names_by_supplycost_suppliername", operation_id="get_part_names_by_supplycost_suppliername", summary="Retrieves the names of parts that have a supply cost greater than the provided value and are supplied by the specified supplier. The operation filters the parts based on the given supply cost and supplier name, and returns the names of the parts that meet the criteria.")
async def get_part_names_by_supplycost_suppliername(supply_cost: float = Query(..., description="Supply cost"), supplier_name: str = Query(..., description="Supplier name")):
    cursor.execute("SELECT T2.p_name FROM partsupp AS T1 INNER JOIN part AS T2 ON T1.ps_partkey = T2.p_partkey INNER JOIN supplier AS T3 ON T1.ps_suppkey = T3.s_suppkey WHERE T1.ps_supplycost > ? AND T3.s_name = ?", (supply_cost, supplier_name))
    result = cursor.fetchall()
    if not result:
        return {"part_names": []}
    return {"part_names": [row[0] for row in result]}

# Endpoint to get the count of line numbers based on quantity and ship mode
@app.get("/v1/retails/count_linenumbers_by_quantity_shipmode", operation_id="get_count_linenumbers_by_quantity_shipmode", summary="Retrieves the total count of line numbers where the quantity is less than the provided value and the ship mode matches the specified mode. This operation is useful for understanding the distribution of line numbers based on quantity and ship mode.")
async def get_count_linenumbers_by_quantity_shipmode(quantity: int = Query(..., description="Quantity"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT COUNT(l_linenumber) FROM lineitem WHERE l_quantity < ? AND l_shipmode = ?", (quantity, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customer keys based on market segment and nation key
@app.get("/v1/retails/count_customerkeys_by_mktsegment_nationkey", operation_id="get_count_customerkeys_by_mktsegment_nationkey", summary="Retrieves the total number of unique customers categorized by a specific market segment and nation. The market segment and nation are provided as input parameters.")
async def get_count_customerkeys_by_mktsegment_nationkey(mktsegment: str = Query(..., description="Market segment"), nationkey: int = Query(..., description="Nation key")):
    cursor.execute("SELECT COUNT(c_custkey) FROM customer WHERE c_mktsegment = ? AND c_nationkey = ?", (mktsegment, nationkey))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone number of the customer with the highest account balance
@app.get("/v1/retails/customer_phone_highest_balance", operation_id="get_customer_phone_highest_balance", summary="Retrieves the phone number of the customer who has the highest account balance. This operation does not require any input parameters and returns a single phone number.")
async def get_customer_phone_highest_balance():
    cursor.execute("SELECT c_phone FROM customer ORDER BY c_acctbal DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the order priority of the order with the highest total price
@app.get("/v1/retails/order_priority_highest_totalprice", operation_id="get_order_priority_highest_totalprice", summary="Retrieves the priority level of the order with the highest total price from the orders database.")
async def get_order_priority_highest_totalprice():
    cursor.execute("SELECT o_orderpriority FROM orders WHERE o_totalprice = ( SELECT MAX(o_totalprice) FROM orders )")
    result = cursor.fetchone()
    if not result:
        return {"order_priority": []}
    return {"order_priority": result[0]}

# Endpoint to get the count of order keys based on nation name
@app.get("/v1/retails/count_orderkeys_by_nationname", operation_id="get_count_orderkeys_by_nationname", summary="Retrieves the total number of orders associated with a specific nation. The nation is identified by its name, which is provided as an input parameter.")
async def get_count_orderkeys_by_nationname(nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT COUNT(T1.o_orderkey) FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey INNER JOIN nation AS T3 ON T2.c_nationkey = T3.n_nationkey WHERE T3.n_name = ?", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customer keys based on market segment and nation name
@app.get("/v1/retails/count_customerkeys_by_mktsegment_nationname", operation_id="get_count_customerkeys_by_mktsegment_nationname", summary="Retrieves the total number of unique customers categorized by a specific market segment and nation. The operation filters customers based on the provided market segment and nation name, then returns the count of distinct customer keys that match the criteria.")
async def get_count_customerkeys_by_mktsegment_nationname(mktsegment: str = Query(..., description="Market segment"), nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT COUNT(T1.c_custkey) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T1.c_mktsegment = ? AND T2.n_name = ?", (mktsegment, nation_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get order comments based on market segment
@app.get("/v1/retails/order_comments_by_mktsegment", operation_id="get_order_comments_by_mktsegment", summary="Retrieves the first five order comments associated with a specific market segment. The market segment is used to filter the order comments from the orders table, which is joined with the customer table to ensure the market segment is correctly associated with each order.")
async def get_order_comments_by_mktsegment(mktsegment: str = Query(..., description="Market segment")):
    cursor.execute("SELECT T1.o_comment FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_mktsegment = ? LIMIT 5", (mktsegment,))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": [row[0] for row in result]}

# Endpoint to get nation names based on region comment
@app.get("/v1/retails/nation_names_by_region_comment", operation_id="get_nation_names_by_region_comment", summary="Retrieves the names of all nations located within a region that matches the provided comment. The comment is used to identify the specific region.")
async def get_nation_names_by_region_comment(region_comment: str = Query(..., description="Comment of the region")):
    cursor.execute("SELECT T1.n_name FROM nation AS T1 INNER JOIN region AS T2 ON T1.n_regionkey = T2.r_regionkey WHERE T2.r_comment = ?", (region_comment,))
    result = cursor.fetchall()
    if not result:
        return {"nation_names": []}
    return {"nation_names": [row[0] for row in result]}

# Endpoint to get the count of suppliers based on nation name
@app.get("/v1/retails/supplier_count_by_nation", operation_id="get_supplier_count_by_nation", summary="Retrieves the total number of suppliers originating from a specified nation. The nation is identified by its name, which is provided as an input parameter.")
async def get_supplier_count_by_nation(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT COUNT(T1.s_suppkey) FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T2.n_name = ?", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of nations based on account balance and region name
@app.get("/v1/retails/nation_count_by_account_balance_region", operation_id="get_nation_count_by_account_balance_region", summary="Retrieves the number of nations that have customers with account balances below a specified threshold in a given region. This operation allows you to analyze the distribution of customers based on their account balances and regional locations.")
async def get_nation_count_by_account_balance_region(account_balance: float = Query(..., description="Account balance threshold"), region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT COUNT(T1.n_name) FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN region AS T3 ON T1.n_regionkey = T3.r_regionkey WHERE T2.c_acctbal < ? AND T3.r_name = ?", (account_balance, region_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer phone number with the highest order total price
@app.get("/v1/retails/customer_phone_highest_order", operation_id="get_customer_phone_highest_order", summary="Retrieves the phone number of the customer who has placed the highest-priced order. The operation considers all orders and identifies the customer with the single highest order total price.")
async def get_customer_phone_highest_order():
    cursor.execute("SELECT T2.c_phone FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey ORDER BY T1.o_totalprice DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the count of part suppliers based on retail price and ship mode
@app.get("/v1/retails/part_supplier_count_by_retail_price_ship_mode", operation_id="get_part_supplier_count_by_retail_price_ship_mode", summary="Retrieves the number of suppliers who provide parts with a retail price exceeding the specified threshold and are shipped via the specified mode. This operation considers the retail price of parts and the shipping mode used for orders.")
async def get_part_supplier_count_by_retail_price_ship_mode(retail_price: float = Query(..., description="Retail price threshold"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT COUNT(T1.ps_suppkey) FROM partsupp AS T1 INNER JOIN lineitem AS T2 ON T1.ps_suppkey = T2.l_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey WHERE T3.p_retailprice > ? AND T2.l_shipmode = ?", (retail_price, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer names and market segments based on order total price
@app.get("/v1/retails/customer_info_by_order_total_price", operation_id="get_customer_info_by_order_total_price", summary="Retrieves the names and market segments of customers who have placed an order with a specific total price. The order total price is provided as an input parameter.")
async def get_customer_info_by_order_total_price(order_total_price: float = Query(..., description="Order total price")):
    cursor.execute("SELECT T2.c_name, T2.c_mktsegment FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T1.o_totalprice = ?", (order_total_price,))
    result = cursor.fetchall()
    if not result:
        return {"customer_info": []}
    return {"customer_info": [{"name": row[0], "market_segment": row[1]} for row in result]}

# Endpoint to get nation and region names based on customer address
@app.get("/v1/retails/nation_region_by_customer_address", operation_id="get_nation_region_by_customer_address", summary="Retrieves the names of the nation and region associated with a given customer address. The operation uses the provided customer address to look up the corresponding nation and region names in the database.")
async def get_nation_region_by_customer_address(customer_address: str = Query(..., description="Customer address")):
    cursor.execute("SELECT T1.n_name, T3.r_name FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN region AS T3 ON T1.n_regionkey = T3.r_regionkey WHERE T2.c_address = ?", (customer_address,))
    result = cursor.fetchall()
    if not result:
        return {"nation_region": []}
    return {"nation_region": [{"nation": row[0], "region": row[1]} for row in result]}

# Endpoint to get the count of customers based on nation name and account balance
@app.get("/v1/retails/customer_count_by_nation_account_balance", operation_id="get_customer_count_by_nation_account_balance", summary="Retrieves the number of customers from a specified nation whose account balance is less than a given threshold. This operation is useful for understanding the distribution of customers based on their account balance within a particular nation.")
async def get_customer_count_by_nation_account_balance(nation_name: str = Query(..., description="Name of the nation"), account_balance: float = Query(..., description="Account balance threshold")):
    cursor.execute("SELECT COUNT(T1.c_custkey) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T2.n_name = ? AND T1.c_acctbal < ?", (nation_name, account_balance))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get nation names of customers with account balance above 80% of the average
@app.get("/v1/retails/nation_names_above_avg_account_balance", operation_id="get_nation_names_above_avg_account_balance", summary="Retrieves the names of nations whose customers have an account balance exceeding 80% of the average account balance. This operation compares individual customer account balances to a threshold derived from the average account balance, and returns the names of nations with qualifying customers.")
async def get_nation_names_above_avg_account_balance():
    cursor.execute("SELECT T2.n_name FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey INNER JOIN ( SELECT AVG(c_acctbal) * 0.8 AS avg_acctbal FROM customer ) AS T3 WHERE T1.c_acctbal > T3.avg_acctbal")
    result = cursor.fetchall()
    if not result:
        return {"nation_names": []}
    return {"nation_names": [row[0] for row in result]}

# Endpoint to get the percentage of customers from a specific nation with account balance below a threshold
@app.get("/v1/retails/percentage_customers_by_nation_account_balance", operation_id="get_percentage_customers_by_nation_account_balance", summary="Retrieves the percentage of customers from a specified nation who have an account balance less than a given threshold. This operation calculates the proportion of customers from the specified nation with account balances below the provided threshold, providing insights into the distribution of account balances within that nation.")
async def get_percentage_customers_by_nation_account_balance(nation_name: str = Query(..., description="Name of the nation"), account_balance: float = Query(..., description="Account balance threshold")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.n_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.c_custkey) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T1.c_acctbal < ?", (nation_name, account_balance))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get customer names and phone numbers based on account balance
@app.get("/v1/retails/customer_details_by_balance", operation_id="get_customer_details_by_balance", summary="Retrieves the names and phone numbers of customers who have an account balance exceeding the specified minimum amount.")
async def get_customer_details_by_balance(min_balance: float = Query(..., description="Minimum account balance")):
    cursor.execute("SELECT c_name, c_phone FROM customer WHERE c_acctbal > ?", (min_balance,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the average line number for items shipped within a date range
@app.get("/v1/retails/avg_line_number_by_ship_date", operation_id="get_avg_line_number_by_ship_date", summary="Retrieves the average line number for items shipped between the specified start and end dates. The start and end dates must be provided in 'YYYY-MM-DD' format.")
async def get_avg_line_number_by_ship_date(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT AVG(l_linenumber) FROM lineitem WHERE l_shipdate BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_line_number": []}
    return {"average_line_number": result[0]}

# Endpoint to get order keys based on total price range
@app.get("/v1/retails/order_keys_by_total_price", operation_id="get_order_keys_by_total_price", summary="Retrieves order keys for orders with a total price that falls within the specified range. The range is defined by a minimum and maximum total price, which are provided as input parameters.")
async def get_order_keys_by_total_price(min_price: float = Query(..., description="Minimum total price"), max_price: float = Query(..., description="Maximum total price")):
    cursor.execute("SELECT o_orderkey FROM orders WHERE o_totalprice BETWEEN ? AND ?", (min_price, max_price))
    result = cursor.fetchall()
    if not result:
        return {"order_keys": []}
    return {"order_keys": result}

# Endpoint to get part keys with retail price above the average
@app.get("/v1/retails/part_keys_above_avg_retail_price", operation_id="get_part_keys_above_avg_retail_price", summary="Retrieves a list of part keys that have a retail price higher than the average retail price of all parts. This operation provides a quick way to identify parts with above-average pricing.")
async def get_part_keys_above_avg_retail_price():
    cursor.execute("SELECT p_partkey FROM part WHERE p_retailprice > ( SELECT AVG(p_retailprice) FROM part )")
    result = cursor.fetchall()
    if not result:
        return {"part_keys": []}
    return {"part_keys": result}

# Endpoint to get the percentage of parts with supply cost above a specified value
@app.get("/v1/retails/percentage_parts_above_supply_cost", operation_id="get_percentage_parts_above_supply_cost", summary="Retrieves the percentage of parts whose supply cost exceeds a specified threshold. This operation calculates the proportion of parts with a supply cost greater than the provided threshold, relative to the total number of parts. The result is expressed as a percentage.")
async def get_percentage_parts_above_supply_cost(supply_cost: float = Query(..., description="Supply cost threshold")):
    cursor.execute("SELECT CAST(SUM(IIF(ps_supplycost > ?, 1, 0)) AS REAL) * 100 / COUNT(ps_suppkey) FROM partsupp", (supply_cost,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get top suppliers by account balance
@app.get("/v1/retails/top_suppliers_by_balance", operation_id="get_top_suppliers_by_balance", summary="Retrieves a list of top suppliers, ranked by their account balances in descending order. The number of suppliers returned is determined by the provided limit parameter.")
async def get_top_suppliers_by_balance(limit: int = Query(..., description="Number of top suppliers to return")):
    cursor.execute("SELECT s_suppkey, s_acctbal FROM supplier ORDER BY s_acctbal DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"suppliers": []}
    return {"suppliers": result}

# Endpoint to get the count of customers with specific order priority and account balance
@app.get("/v1/retails/customer_count_by_order_priority_and_balance", operation_id="get_customer_count_by_order_priority_and_balance", summary="Retrieves the number of customers who have an order priority matching the provided value and an account balance exceeding the specified minimum amount.")
async def get_customer_count_by_order_priority_and_balance(min_balance: float = Query(..., description="Minimum account balance"), order_priority: str = Query(..., description="Order priority")):
    cursor.execute("SELECT COUNT(T2.c_custkey) FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_acctbal > ? AND T1.o_orderpriority = ?", (min_balance, order_priority))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer names and phone numbers with account balance above the average
@app.get("/v1/retails/customer_details_above_avg_balance", operation_id="get_customer_details_above_avg_balance", summary="Retrieves the names and phone numbers of customers who have an account balance exceeding the average account balance. The data is sorted by customer name.")
async def get_customer_details_above_avg_balance():
    cursor.execute("SELECT T1.c_name, T1.c_phone FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T1.c_acctbal > ( SELECT AVG(c_acctbal) FROM customer ) ORDER BY T1.c_name")
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get part keys from a specific supplier ordered by supply cost
@app.get("/v1/retails/part_keys_by_supplier_and_supply_cost", operation_id="get_part_keys_by_supplier_and_supply_cost", summary="Retrieves a specified number of part keys from a particular supplier, sorted by supply cost in descending order.")
async def get_part_keys_by_supplier_and_supply_cost(supplier_name: str = Query(..., description="Supplier name"), limit: int = Query(..., description="Number of part keys to return")):
    cursor.execute("SELECT T2.ps_partkey FROM supplier AS T1 INNER JOIN partsupp AS T2 ON T1.s_suppkey = T2.ps_suppkey WHERE T1.s_name = ? ORDER BY T2.ps_supplycost DESC LIMIT ?", (supplier_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"part_keys": []}
    return {"part_keys": result}

# Endpoint to get the percentage of customers in a specific market segment and nation
@app.get("/v1/retails/percentage_customers_by_market_segment_and_nation", operation_id="get_percentage_customers_by_market_segment_and_nation", summary="Retrieves the percentage of customers belonging to a specific market segment within a given nation. The calculation is based on the total number of customers in the specified nation.")
async def get_percentage_customers_by_market_segment_and_nation(market_segment: str = Query(..., description="Market segment"), nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.c_mktsegment = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.c_name) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T2.n_name = ?", (market_segment, nation_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the part name with the highest calculated value
@app.get("/v1/retails/part_name_highest_value", operation_id="get_part_name_highest_value", summary="Retrieves the name of the part with the highest calculated value, based on extended price, discount, supply cost, and quantity. The calculation is performed by subtracting the product of supply cost and quantity from the product of extended price and discount. The operation returns the top result(s) as specified by the input limit parameter.")
async def get_part_name_highest_value(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T.p_name FROM ( SELECT T3.p_name , T2.l_extendedprice * (1 - T2.l_discount) - T1.ps_supplycost * T2.l_quantity AS num FROM partsupp AS T1 INNER JOIN lineitem AS T2 ON T1.ps_suppkey = T2.l_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey ) AS T ORDER BY T.num DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"part_name": []}
    return {"part_name": result[0]}

# Endpoint to get nation names with customer count above average
@app.get("/v1/retails/nation_names_above_avg_customer_count", operation_id="get_nation_names_above_avg_customer_count", summary="Retrieves the names of nations that have a customer count exceeding the average customer count per nation. The data is obtained by joining the customer and nation tables, grouping by nation name, and filtering for those with a customer count above the average. The results are ordered by customer count in descending order.")
async def get_nation_names_above_avg_customer_count():
    cursor.execute("SELECT T2.n_name FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey GROUP BY T2.n_name HAVING COUNT(T1.c_name) > ( SELECT COUNT(customer.c_name) / COUNT(DISTINCT nation.n_name) FROM customer INNER JOIN nation ON customer.c_nationkey = nation.n_nationkey ) ORDER BY COUNT(T1.c_name)")
    result = cursor.fetchall()
    if not result:
        return {"nation_names": []}
    return {"nation_names": [row[0] for row in result]}

# Endpoint to get the percentage of customers in a specific market segment and region
@app.get("/v1/retails/percentage_customers_market_segment_region", operation_id="get_percentage_customers_market_segment_region", summary="Retrieves the percentage of customers belonging to a specific market segment within a given region. The calculation is based on the total number of customers in the region.")
async def get_percentage_customers_market_segment_region(region_name: str = Query(..., description="Name of the region"), market_segment: str = Query(..., description="Market segment of the customer")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.r_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.n_nationkey) FROM nation AS T1 INNER JOIN region AS T2 ON T1.n_regionkey = T2.r_regionkey INNER JOIN customer AS T3 ON T1.n_nationkey = T3.c_nationkey WHERE T3.c_mktsegment = ?", (region_name, market_segment))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get part names ordered by available quantity
@app.get("/v1/retails/part_names_by_available_quantity", operation_id="get_part_names_by_available_quantity", summary="Retrieves a list of part names, ordered by their available quantity in descending order. The number of results returned can be limited by specifying the desired maximum count.")
async def get_part_names_by_available_quantity(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T1.p_name FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey ORDER BY T2.ps_availqty DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"part_names": []}
    return {"part_names": [row[0] for row in result]}

# Endpoint to get the difference in average retail price between two ship modes
@app.get("/v1/retails/avg_retail_price_difference_ship_modes", operation_id="get_avg_retail_price_difference_ship_modes", summary="Retrieves the difference in average retail price between two specified ship modes. This operation calculates the average retail price for each ship mode by summing the retail prices of parts and dividing by the total number of parts for each mode. The difference between these averages is then returned.")
async def get_avg_retail_price_difference_ship_modes(ship_mode_1: str = Query(..., description="First ship mode"), ship_mode_2: str = Query(..., description="Second ship mode")):
    cursor.execute("SELECT (CAST(SUM(IIF(T3.l_shipmode = ?, T1.p_retailprice, 0)) AS REAL) / SUM(IIF(T3.l_shipmode = ?, 1, 0))) - (CAST(SUM(IIF(T3.l_shipmode = ?, T1.p_retailprice, 0)) AS REAL) / SUM(IIF(T3.l_shipmode = ?, 1, 0))) FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN lineitem AS T3 ON T2.ps_suppkey = T3.l_suppkey", (ship_mode_1, ship_mode_1, ship_mode_2, ship_mode_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the average discount for parts from a specific manufacturer
@app.get("/v1/retails/avg_discount_by_manufacturer", operation_id="get_avg_discount_by_manufacturer", summary="Get the average discount for parts from a specific manufacturer")
async def get_avg_discount_by_manufacturer(manufacturer: str = Query(..., description="Name of the manufacturer")):
    cursor.execute("SELECT AVG(T3.l_discount) FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN lineitem AS T3 ON T2.ps_suppkey = T3.l_suppkey WHERE T1.p_mfgr = ?", (manufacturer,))
    result = cursor.fetchone()
    if not result:
        return {"average_discount": []}
    return {"average_discount": result[0]}

# Endpoint to get the count of parts for a specific ship mode and order priority
@app.get("/v1/retails/count_parts_ship_mode_order_priority", operation_id="get_count_parts_ship_mode_order_priority", summary="Retrieves the total number of parts associated with a particular ship mode and order priority. The ship mode and order priority are provided as input parameters, allowing for a targeted count of parts.")
async def get_count_parts_ship_mode_order_priority(ship_mode: str = Query(..., description="Ship mode"), order_priority: str = Query(..., description="Order priority")):
    cursor.execute("SELECT COUNT(T2.l_partkey) FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T2.l_shipmode = ? AND T1.o_orderpriority = ?", (ship_mode, order_priority))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of suppliers with account balance below average in a specific region
@app.get("/v1/retails/percentage_suppliers_below_avg_acctbal_region", operation_id="get_percentage_suppliers_below_avg_acctbal_region", summary="Retrieves the percentage of suppliers in a specified region whose account balance is below the average account balance. The calculation is based on the total count of suppliers in the region.")
async def get_percentage_suppliers_below_avg_acctbal_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.s_acctbal < ( SELECT AVG(supplier.s_acctbal) FROM supplier ), 1, 0)) AS REAL) * 100 / COUNT(T1.n_nationkey) FROM nation AS T1 INNER JOIN region AS T2 ON T1.n_regionkey = T2.r_regionkey INNER JOIN supplier AS T3 ON T1.n_nationkey = T3.s_nationkey WHERE T2.r_name = ?", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in average monthly shipments between two years for a specific order priority and ship mode
@app.get("/v1/retails/avg_monthly_shipments_difference_years", operation_id="get_avg_monthly_shipments_difference_years", summary="Retrieves the difference in average monthly shipments between two specified years for a given order priority and ship mode. This operation calculates the average monthly shipments for each year and returns the difference between them.")
async def get_avg_monthly_shipments_difference_years(year_1: int = Query(..., description="First year"), year_2: int = Query(..., description="Second year"), order_priority: str = Query(..., description="Order priority"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT (CAST(SUM(IIF(STRFTIME('%Y', T2.l_shipdate) = ?, 1, 0)) AS REAL) / 12) - (CAST(SUM(IIF(STRFTIME('%Y', T2.l_shipdate) = ?, 1, 0)) AS REAL) / 12) FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_orderpriority = ? AND T2.l_shipmode = ?", (year_1, year_2, order_priority, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get customer keys with account balance below a specified value
@app.get("/v1/retails/customer_keys_below_acctbal", operation_id="get_customer_keys_below_acctbal", summary="Retrieves the unique identifiers of customers whose account balance is below the specified threshold. This operation allows you to filter customers based on their account balance, providing a targeted list of customer keys for further processing or analysis.")
async def get_customer_keys_below_acctbal(account_balance: float = Query(..., description="Account balance threshold")):
    cursor.execute("SELECT c_custkey FROM customer WHERE c_acctbal < ?", (account_balance,))
    result = cursor.fetchall()
    if not result:
        return {"customer_keys": []}
    return {"customer_keys": [row[0] for row in result]}

# Endpoint to get order keys from lineitem ordered by extended price with discount and limited by a specified number
@app.get("/v1/retails/lineitem_orderkeys_by_extendedprice", operation_id="get_lineitem_orderkeys_by_extendedprice", summary="Retrieves a specified number of order keys from the lineitem table, sorted by the extended price after applying a discount. This operation allows you to obtain the most relevant order keys based on their adjusted prices.")
async def get_lineitem_orderkeys_by_extendedprice(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT l_orderkey FROM lineitem ORDER BY l_extendedprice * (1 - l_discount) LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"orderkeys": []}
    return {"orderkeys": [row[0] for row in result]}

# Endpoint to get the count of line numbers from lineitem based on quantity and return flag
@app.get("/v1/retails/lineitem_count_by_quantity_returnflag", operation_id="get_lineitem_count_by_quantity_returnflag", summary="Retrieves the total number of line items that meet the specified quantity and return flag criteria. The quantity parameter filters line items based on the minimum quantity, while the return flag parameter determines the type of return flag associated with the line items.")
async def get_lineitem_count_by_quantity_returnflag(quantity: int = Query(..., description="Quantity of items"), return_flag: str = Query(..., description="Return flag")):
    cursor.execute("SELECT COUNT(l_linenumber) FROM lineitem WHERE l_quantity > ? AND l_returnflag = ?", (quantity, return_flag))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total price from lineitem based on ship mode and ship instruction
@app.get("/v1/retails/lineitem_totalprice_by_shipmode_shipinstruct", operation_id="get_lineitem_totalprice_by_shipmode_shipinstruct", summary="Retrieves the total price of line items based on the specified shipping mode and shipping instructions. The calculation takes into account the extended price, discount, and tax of each line item.")
async def get_lineitem_totalprice_by_shipmode_shipinstruct(ship_mode: str = Query(..., description="Ship mode"), ship_instruct: str = Query(..., description="Ship instruction")):
    cursor.execute("SELECT l_extendedprice * (1 - l_discount) * (1 + l_tax) AS totalprice FROM lineitem WHERE l_shipmode = ? AND l_shipinstruct = ?", (ship_mode, ship_instruct))
    result = cursor.fetchall()
    if not result:
        return {"totalprices": []}
    return {"totalprices": [row[0] for row in result]}

# Endpoint to get the count of order keys from orders based on order priority and limited by a specified number
@app.get("/v1/retails/orders_count_by_orderpriority", operation_id="get_orders_count_by_orderpriority", summary="Retrieves the total number of orders with a specified priority, grouped by order date in descending order. The results are limited to a specified number.")
async def get_orders_count_by_orderpriority(order_priority: str = Query(..., description="Order priority"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT COUNT(o_orderkey) FROM orders WHERE o_orderpriority = ? GROUP BY o_orderdate ORDER BY o_orderdate DESC LIMIT ?", (order_priority, limit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of supplier keys based on account balance and nation name
@app.get("/v1/retails/supplier_count_by_accountbalance_nation", operation_id="get_supplier_count_by_accountbalance_nation", summary="Retrieves the total number of suppliers with an account balance below the specified threshold and belonging to the specified nation.")
async def get_supplier_count_by_accountbalance_nation(account_balance: float = Query(..., description="Account balance"), nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT COUNT(T1.s_suppkey) FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T1.s_acctbal < ? AND T2.n_name = ?", (account_balance, nation_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of order keys based on ship mode and order date
@app.get("/v1/retails/orders_count_by_shipmode_orderdate", operation_id="get_orders_count_by_shipmode_orderdate", summary="Retrieves the total number of orders that were shipped using a specific mode and on a particular date. The ship mode and order date are provided as input parameters.")
async def get_orders_count_by_shipmode_orderdate(ship_mode: str = Query(..., description="Ship mode"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.o_orderkey) FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T2.l_shipmode = ? AND T1.o_orderdate = ?", (ship_mode, order_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the market segment with the highest number of orders in a given month
@app.get("/v1/retails/market_segment_by_order_count", operation_id="get_market_segment_by_order_count", summary="Retrieves the market segment(s) with the highest number of orders in a given month, based on the provided order date pattern. The results can be limited to a specific number of segments. This operation is useful for identifying the most active market segments during a particular month.")
async def get_market_segment_by_order_count(order_date_pattern: str = Query(..., description="Order date pattern in 'YYYY-MM-%' format"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T.c_mktsegment FROM ( SELECT T2.c_mktsegment, COUNT(T1.o_orderkey) AS num FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T1.o_orderdate LIKE ? GROUP BY T1.o_custkey ) AS T ORDER BY T.num DESC LIMIT ?", (order_date_pattern, limit))
    result = cursor.fetchone()
    if not result:
        return {"market_segment": []}
    return {"market_segment": result[0]}

# Endpoint to get part names based on supplier name
@app.get("/v1/retails/part_names_by_supplier", operation_id="get_part_names_by_supplier", summary="Retrieves the names of parts supplied by a specific supplier. The operation requires the supplier's name as input and returns a list of corresponding part names.")
async def get_part_names_by_supplier(supplier_name: str = Query(..., description="Supplier name")):
    cursor.execute("SELECT T3.p_name FROM partsupp AS T1 INNER JOIN supplier AS T2 ON T1.ps_suppkey = T2.s_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey WHERE T2.s_name = ?", (supplier_name,))
    result = cursor.fetchall()
    if not result:
        return {"part_names": []}
    return {"part_names": [row[0] for row in result]}

# Endpoint to get supply costs based on part type
@app.get("/v1/retails/supply_costs_by_part_type", operation_id="get_supply_costs_by_part_type", summary="Retrieves the supply costs for parts of a specified type. The operation filters parts based on the provided type and returns the corresponding supply costs.")
async def get_supply_costs_by_part_type(part_type: str = Query(..., description="Part type")):
    cursor.execute("SELECT T2.ps_supplycost FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey WHERE T1.p_type = ?", (part_type,))
    result = cursor.fetchall()
    if not result:
        return {"supply_costs": []}
    return {"supply_costs": [row[0] for row in result]}

# Endpoint to get the part name with available quantity less than a specified value
@app.get("/v1/retails/part_name_avail_quantity", operation_id="get_part_name", summary="Retrieves the name of a part for which the available quantity is less than the specified value. The part with the lowest supply cost is returned when multiple parts meet the criteria. The input parameter specifies the maximum available quantity.")
async def get_part_name(avail_quantity: int = Query(..., description="Available quantity")):
    cursor.execute("SELECT T1.p_name FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey WHERE T2.ps_availqty < ? ORDER BY T2.ps_supplycost LIMIT 1", (avail_quantity,))
    result = cursor.fetchone()
    if not result:
        return {"part_name": []}
    return {"part_name": result[0]}

# Endpoint to get the count of order clerks based on customer address
@app.get("/v1/retails/order_clerk_count_address", operation_id="get_order_clerk_count", summary="Retrieves the total number of order clerks who have processed orders for customers residing at a specific address. The address is provided as an input parameter.")
async def get_order_clerk_count(address: str = Query(..., description="Customer address")):
    cursor.execute("SELECT COUNT(T1.o_clerk) FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_address = ?", (address,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the part name based on discount value
@app.get("/v1/retails/part_name_discount", operation_id="get_part_name_discount", summary="Get the part name based on discount value")
async def get_part_name_discount(discount: float = Query(..., description="Discount value")):
    cursor.execute("SELECT T3.p_name FROM partsupp AS T1 INNER JOIN lineitem AS T2 ON T1.ps_suppkey = T2.l_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey WHERE T2.l_discount = ?", (discount,))
    result = cursor.fetchone()
    if not result:
        return {"part_name": []}
    return {"part_name": result[0]}

# Endpoint to get the count of line items based on order date and ship date
@app.get("/v1/retails/line_item_count_order_ship_date", operation_id="get_line_item_count", summary="Retrieves the total count of line items associated with orders placed on a specific date and shipped on another specific date. The count is determined by joining the orders and line item tables based on the order key. The order date and ship date are provided as input parameters in 'YYYY-MM-DD' format.")
async def get_line_item_count(order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format"), ship_date: str = Query(..., description="Ship date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.l_partkey) FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_orderdate = ? AND T2.l_shipdate = ?", (order_date, ship_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average profit per part type
@app.get("/v1/retails/average_profit_part_type", operation_id="get_average_profit", summary="Retrieves the average profit per part type, calculated as the sum of extended prices (after discounts) minus the total supply cost, divided by the count of parts of the specified type. The calculation is based on data from the part, partsupp, and lineitem tables, filtered by the provided part type.")
async def get_average_profit(part_type: str = Query(..., description="Part type")):
    cursor.execute("SELECT SUM(T2.l_extendedprice * (1 - T2.l_discount) - T1.ps_supplycost * T2.l_quantity) / COUNT(T1.ps_partkey) FROM partsupp AS T1 INNER JOIN lineitem AS T2 ON T1.ps_suppkey = T2.l_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey WHERE T3.p_type = ?", (part_type,))
    result = cursor.fetchone()
    if not result:
        return {"average_profit": []}
    return {"average_profit": result[0]}

# Endpoint to get the percentage of customers from a specific nation in a market segment
@app.get("/v1/retails/customer_percentage_nation_segment", operation_id="get_customer_percentage", summary="Retrieves the percentage of customers from a specified nation within a given market segment. This operation calculates the proportion of customers from the provided nation in the specified market segment, providing insights into customer distribution across nations and segments.")
async def get_customer_percentage(nation_name: str = Query(..., description="Name of the nation"), market_segment: str = Query(..., description="Market segment")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.n_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T2.n_name) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T1.c_mktsegment = ?", (nation_name, market_segment))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get customer details based on customer key
@app.get("/v1/retails/customer_details", operation_id="get_customer_details", summary="Retrieves the market segment, name, address, and phone number of a specific customer using their unique key.")
async def get_customer_details(custkey: int = Query(..., description="Customer key")):
    cursor.execute("SELECT c_mktsegment, c_name, c_address, c_phone FROM customer WHERE c_custkey = ?", (custkey,))
    result = cursor.fetchone()
    if not result:
        return {"customer_details": []}
    return {"customer_details": {"c_mktsegment": result[0], "c_name": result[1], "c_address": result[2], "c_phone": result[3]}}

# Endpoint to get line numbers based on discount value
@app.get("/v1/retails/line_numbers_discount", operation_id="get_line_numbers", summary="Retrieves the first three line numbers that have a specified discount value. The discount value is provided as an input parameter.")
async def get_line_numbers(discount: float = Query(..., description="Discount value")):
    cursor.execute("SELECT l_linenumber FROM lineitem WHERE l_discount = ? LIMIT 3", (discount,))
    result = cursor.fetchall()
    if not result:
        return {"line_numbers": []}
    return {"line_numbers": [row[0] for row in result]}

# Endpoint to get the ship mode comparison result based on ship instruction
@app.get("/v1/retails/shipmode_comparison_result", operation_id="get_shipmode_comparison_result", summary="Retrieves a comparison result between two shipping modes based on a specific ship instruction. The comparison is determined by the count of line items associated with each shipping mode. The result will be 'RAIL' if the count of the first shipping mode is greater, and 'MAIL' otherwise.")
async def get_shipmode_comparison_result(shipmode1: str = Query(..., description="First ship mode to compare"), shipmode2: str = Query(..., description="Second ship mode to compare"), result1: str = Query(..., description="Result if the first ship mode count is greater"), result2: str = Query(..., description="Result if the second ship mode count is greater"), shipinstruct: str = Query(..., description="Ship instruction")):
    cursor.execute("SELECT IIF(SUM(IIF(l_shipmode = ?, 1, 0)) - SUM(IIF(l_shipmode = ?, 1, 0)), ?, ?) AS result FROM lineitem WHERE l_shipinstruct = ?", (shipmode1, shipmode2, result1, result2, shipinstruct))
    result = cursor.fetchone()
    if not result:
        return {"result": []}
    return {"result": result[0]}

# Endpoint to get customer names based on account balance and limit
@app.get("/v1/retails/customer_names_acctbal_limit", operation_id="get_customer_names", summary="Retrieves a specified number of customer names with account balances below a given threshold. This operation is useful for identifying customers with lower account balances, which can be helpful for various business purposes such as targeted promotions or account management.")
async def get_customer_names(acctbal: float = Query(..., description="Account balance"), limit: int = Query(..., description="Limit of results")):
    cursor.execute("SELECT c_name FROM customer WHERE c_acctbal < ? LIMIT ?", (acctbal, limit))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the discounted extended price of a line item based on line number
@app.get("/v1/retails/discounted_extended_price", operation_id="get_discounted_extended_price", summary="Get the discounted extended price of a line item based on the line number")
async def get_discounted_extended_price(linenumber: int = Query(..., description="Line number of the line item")):
    cursor.execute("SELECT l_extendedprice * (1 - l_discount) FROM lineitem WHERE l_linenumber = ?", (linenumber,))
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get the difference in return flags based on extended price
@app.get("/v1/retails/return_flag_difference", operation_id="get_return_flag_difference", summary="Retrieves the difference in the count of line items with two different return flags, where the extended price of the line item is less than a specified value.")
async def get_return_flag_difference(returnflag1: str = Query(..., description="First return flag"), returnflag2: str = Query(..., description="Second return flag"), extendedprice: float = Query(..., description="Extended price of the line item")):
    cursor.execute("SELECT SUM(IIF(l_returnflag = ?, 1, 0)) - SUM(IIF(l_returnflag = ?, 1, 0)) AS diff FROM lineitem WHERE l_extendedprice < ?", (returnflag1, returnflag2, extendedprice))
    result = cursor.fetchone()
    if not result:
        return {"diff": []}
    return {"diff": result[0]}

# Endpoint to get the nation names based on region name
@app.get("/v1/retails/nation_names_by_region_name", operation_id="get_nation_names_by_region_name", summary="Retrieves the names of all nations located within the specified region. The region is identified by its name, which is provided as an input parameter.")
async def get_nation_names_by_region_name(region_name: str = Query(..., description="Region name")):
    cursor.execute("SELECT T2.n_name FROM region AS T1 INNER JOIN nation AS T2 ON T1.r_regionkey = T2.n_regionkey WHERE T1.r_name = ?", (region_name,))
    result = cursor.fetchall()
    if not result:
        return {"nation_names": []}
    return {"nation_names": [row[0] for row in result]}

# Endpoint to get the region name with the highest number of nations
@app.get("/v1/retails/region_with_most_nations", operation_id="get_region_with_most_nations", summary="Retrieves the name of the region that has the most nations. This operation calculates the number of nations in each region and returns the region with the highest count.")
async def get_region_with_most_nations():
    cursor.execute("SELECT T.r_name FROM ( SELECT T1.r_name, COUNT(T2.n_name) AS num FROM region AS T1 INNER JOIN nation AS T2 ON T1.r_regionkey = T2.n_regionkey GROUP BY T1.r_name ) AS T ORDER BY T.num LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"region_name": []}
    return {"region_name": result[0]}

# Endpoint to get the customer name based on order total price and customer key
@app.get("/v1/retails/customer_name_by_order_total_price_and_customer_key", operation_id="get_customer_name_by_order_total_price_and_customer_key", summary="Retrieves the name of the customer associated with a specific order total price and customer key. The operation requires the total price of the order and the unique customer key as input parameters. It returns the customer's name, providing a direct link between the order's total price and the customer who placed it.")
async def get_customer_name_by_order_total_price_and_customer_key(total_price: float = Query(..., description="Total price of the order"), customer_key: int = Query(..., description="Customer key")):
    cursor.execute("SELECT T2.c_name FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T1.o_totalprice = ? AND T1.o_custkey = ?", (total_price, customer_key))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the nation and region names based on customer name
@app.get("/v1/retails/nation_and_region_by_customer_name", operation_id="get_nation_and_region_by_customer_name", summary="Retrieves the names of the nation and region associated with a specific customer. The customer is identified by their name, which is provided as an input parameter. The operation returns the nation and region names as separate fields.")
async def get_nation_and_region_by_customer_name(customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT T1.n_name, T3.r_name FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN region AS T3 ON T1.n_regionkey = T3.r_regionkey WHERE T2.c_name = ?", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"nation_and_region": []}
    return {"nation_and_region": [{"nation_name": row[0], "region_name": row[1]} for row in result]}

# Endpoint to get the difference in days between receipt and commit dates and the order clerk based on order key
@app.get("/v1/retails/days_difference_and_clerk_by_order_key", operation_id="get_days_difference_and_clerk_by_order_key", summary="Retrieves the number of days between the receipt and commit dates, as well as the clerk associated with a specific order. This operation uses the provided order key to fetch the relevant data from the orders and lineitem tables.")
async def get_days_difference_and_clerk_by_order_key(order_key: int = Query(..., description="Order key")):
    cursor.execute("SELECT JULIANDAY(T2.l_receiptdate) - JULIANDAY(T2.l_commitdate), T1.o_clerk FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_orderkey = ?", (order_key,))
    result = cursor.fetchall()
    if not result:
        return {"days_difference_and_clerk": []}
    return {"days_difference_and_clerk": [{"days_difference": row[0], "clerk": row[1]} for row in result]}

# Endpoint to get the order dates based on order priority
@app.get("/v1/retails/order_dates_by_order_priority", operation_id="get_order_dates_by_order_priority", summary="Retrieves the dates of orders with a specified priority. The operation filters orders based on the provided priority level and returns the corresponding order dates.")
async def get_order_dates_by_order_priority(order_priority: str = Query(..., description="Order priority")):
    cursor.execute("SELECT o_orderdate FROM orders WHERE o_orderpriority = ?", (order_priority,))
    result = cursor.fetchall()
    if not result:
        return {"order_dates": []}
    return {"order_dates": [row[0] for row in result]}

# Endpoint to get the count of line items based on ship instruction
@app.get("/v1/retails/line_item_count_by_ship_instruction", operation_id="get_line_item_count_by_ship_instruction", summary="Retrieves the total number of line items associated with a specific ship instruction. The ship instruction is provided as an input parameter, allowing for a targeted count of line items that adhere to the given instruction.")
async def get_line_item_count_by_ship_instruction(ship_instruction: str = Query(..., description="Ship instruction")):
    cursor.execute("SELECT COUNT(l_linenumber) FROM lineitem WHERE l_shipinstruct = ?", (ship_instruction,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum account balance from suppliers
@app.get("/v1/retails/max_supplier_account_balance", operation_id="get_max_supplier_account_balance", summary="Retrieves the highest account balance among all suppliers. This operation provides a quick overview of the supplier with the largest account balance, which can be useful for financial analysis and decision-making.")
async def get_max_supplier_account_balance():
    cursor.execute("SELECT MAX(s_acctbal) FROM supplier")
    result = cursor.fetchone()
    if not result:
        return {"max_account_balance": []}
    return {"max_account_balance": result[0]}

# Endpoint to get supply costs based on part name
@app.get("/v1/retails/supply_costs_by_part_name", operation_id="get_supply_costs_by_part_name", summary="Retrieves the supply costs associated with a specific part, identified by its name. The operation returns the supply cost for the specified part, obtained by joining the 'part' and 'partsupp' tables using the part key.")
async def get_supply_costs_by_part_name(part_name: str = Query(..., description="Name of the part")):
    cursor.execute("SELECT T2.ps_supplycost FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey WHERE T1.p_name = ?", (part_name,))
    result = cursor.fetchall()
    if not result:
        return {"supply_costs": []}
    return {"supply_costs": [row[0] for row in result]}

# Endpoint to get customer phone numbers based on nation name
@app.get("/v1/retails/customer_phones_by_nation", operation_id="get_customer_phones_by_nation", summary="Retrieves the phone numbers of customers who belong to a specified nation. The nation is identified by its name.")
async def get_customer_phones_by_nation(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT T1.c_phone FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T2.n_name = ?", (nation_name,))
    result = cursor.fetchall()
    if not result:
        return {"customer_phones": []}
    return {"customer_phones": [row[0] for row in result]}

# Endpoint to get the total order price based on customer phone number
@app.get("/v1/retails/total_order_price_by_customer_phone", operation_id="get_total_order_price_by_customer_phone", summary="Retrieves the total price of all orders placed by a customer identified by their phone number. The response includes the sum of the total price of all orders associated with the provided customer phone number.")
async def get_total_order_price_by_customer_phone(customer_phone: str = Query(..., description="Phone number of the customer")):
    cursor.execute("SELECT SUM(T1.o_totalprice) FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_phone = ?", (customer_phone,))
    result = cursor.fetchone()
    if not result:
        return {"total_order_price": []}
    return {"total_order_price": result[0]}

# Endpoint to get distinct ship modes based on order date
@app.get("/v1/retails/distinct_ship_modes_by_order_date", operation_id="get_distinct_ship_modes_by_order_date", summary="Retrieves a list of unique shipping methods used for orders placed on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_distinct_ship_modes_by_order_date(order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.l_shipmode FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_orderdate = ?", (order_date,))
    result = cursor.fetchall()
    if not result:
        return {"ship_modes": []}
    return {"ship_modes": [row[0] for row in result]}

# Endpoint to get the supplier account balance with the highest number of supplied parts
@app.get("/v1/retails/supplier_with_most_supplied_parts", operation_id="get_supplier_with_most_supplied_parts", summary="Retrieves the account balance of the supplier who has supplied the highest number of parts. This operation calculates the number of parts supplied by each supplier and returns the account balance of the supplier with the highest count.")
async def get_supplier_with_most_supplied_parts():
    cursor.execute("SELECT T.s_acctbal FROM ( SELECT T1.s_acctbal, COUNT(T2.ps_suppkey) AS num FROM supplier AS T1 INNER JOIN partsupp AS T2 ON T1.s_suppkey = T2.ps_suppkey GROUP BY T1.s_suppkey ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"supplier_account_balance": []}
    return {"supplier_account_balance": result[0]}

# Endpoint to get nation names based on supplier account balance
@app.get("/v1/retails/nation_names_by_supplier_account_balance", operation_id="get_nation_names_by_supplier_account_balance", summary="Retrieves the names of nations associated with suppliers who have a specific account balance. The account balance is provided as an input parameter.")
async def get_nation_names_by_supplier_account_balance(account_balance: float = Query(..., description="Supplier account balance")):
    cursor.execute("SELECT T2.n_name FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T1.s_acctbal = ?", (account_balance,))
    result = cursor.fetchall()
    if not result:
        return {"nation_names": []}
    return {"nation_names": [row[0] for row in result]}

# Endpoint to get the region name with the highest number of customers
@app.get("/v1/retails/region_with_most_customers", operation_id="get_region_with_most_customers", summary="Retrieves the name of the region that has the highest number of customers. This operation calculates the total number of customers for each region by aggregating customer data from the nation and region tables. The region with the maximum count is then returned.")
async def get_region_with_most_customers():
    cursor.execute("SELECT T.r_name FROM ( SELECT T3.r_name, COUNT(T2.c_custkey) AS num FROM nation AS T1 INNER JOIN customer AS T2 ON T1.n_nationkey = T2.c_nationkey INNER JOIN region AS T3 ON T1.n_regionkey = T3.r_regionkey GROUP BY T3.r_name ) AS T ORDER BY T.num DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"region_name": []}
    return {"region_name": result[0]}

# Endpoint to get customer phone numbers based on minimum order total price
@app.get("/v1/retails/customer_phones_by_min_order_total", operation_id="get_customer_phones_by_min_order_total", summary="Retrieves the phone numbers of customers who have placed orders with a total price greater than the specified minimum order total. The input parameter is the minimum order total price.")
async def get_customer_phones_by_min_order_total(min_order_total: float = Query(..., description="Minimum order total price")):
    cursor.execute("SELECT T2.c_phone FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T1.o_totalprice > ?", (min_order_total,))
    result = cursor.fetchall()
    if not result:
        return {"customer_phones": []}
    return {"customer_phones": [row[0] for row in result]}

# Endpoint to get the order clerk based on ship mode
@app.get("/v1/retails/order_clerk_by_ship_mode", operation_id="get_order_clerk_by_ship_mode", summary="Retrieves the order clerk associated with a specific ship mode. This operation fetches the order clerk's details by filtering orders based on the provided ship mode. The ship mode parameter is used to identify the relevant orders and their corresponding clerks.")
async def get_order_clerk_by_ship_mode(ship_mode: str = Query(..., description="Ship mode of the order")):
    cursor.execute("SELECT T1.o_clerk FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T2.l_shipmode = ?", (ship_mode,))
    result = cursor.fetchall()
    if not result:
        return {"order_clerks": []}
    return {"order_clerks": [row[0] for row in result]}

# Endpoint to get the nation name of the supplier with the highest account balance
@app.get("/v1/retails/nation_name_highest_account_balance", operation_id="get_nation_name_highest_account_balance", summary="Retrieves the name of the nation associated with the supplier who has the highest account balance. This operation returns the nation name as a single result, providing insight into the financial standing of suppliers across different nations.")
async def get_nation_name_highest_account_balance():
    cursor.execute("SELECT T2.n_name FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey ORDER BY T1.s_acctbal LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"nation_name": []}
    return {"nation_name": result[0]}

# Endpoint to get the supplier address of the largest part
@app.get("/v1/retails/supplier_address_largest_part", operation_id="get_supplier_address_largest_part", summary="Retrieves the address of the supplier who provides the largest part, based on size. This operation involves joining the partsupp, supplier, and part tables, and ordering the results by the size of the part in descending order. The address of the supplier associated with the largest part is then returned.")
async def get_supplier_address_largest_part():
    cursor.execute("SELECT T2.s_address FROM partsupp AS T1 INNER JOIN supplier AS T2 ON T1.ps_suppkey = T2.s_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey ORDER BY T3.p_size DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"supplier_address": []}
    return {"supplier_address": result[0]}

# Endpoint to get the part name and supplier name with the highest profit
@app.get("/v1/retails/part_supplier_highest_profit", operation_id="get_part_supplier_highest_profit", summary="Retrieves the name of the part and its supplier that yield the highest profit. The profit is calculated as the extended price of the line item minus the supply cost of the part multiplied by the quantity of the line item. The result is sorted in descending order and the top record is returned.")
async def get_part_supplier_highest_profit():
    cursor.execute("SELECT T3.p_name, T4.s_name FROM partsupp AS T1 INNER JOIN lineitem AS T2 ON T1.ps_suppkey = T2.l_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey INNER JOIN supplier AS T4 ON T1.ps_suppkey = T4.s_suppkey ORDER BY T2.l_extendedprice * (1 - T2.l_discount) - T1.ps_supplycost * T2.l_quantity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"part_name": [], "supplier_name": []}
    return {"part_name": result[0], "supplier_name": result[1]}

# Endpoint to get the percentage of suppliers in a specific region
@app.get("/v1/retails/percentage_suppliers_in_region", operation_id="get_percentage_suppliers_in_region", summary="Retrieves the percentage of suppliers located in a specific region. The operation calculates this percentage by counting the number of suppliers in the specified region and dividing it by the total number of suppliers across all regions. The region is identified by its name.")
async def get_percentage_suppliers_in_region(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.r_name = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.r_regionkey) FROM region AS T1 INNER JOIN nation AS T2 ON T1.r_regionkey = T2.n_regionkey INNER JOIN supplier AS T3 ON T2.n_nationkey = T3.s_nationkey", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total price of a specific order
@app.get("/v1/retails/order_total_price", operation_id="get_order_total_price", summary="Retrieves the total price of a specific order identified by the provided order key.")
async def get_order_total_price(order_key: int = Query(..., description="Order key")):
    cursor.execute("SELECT o_totalprice FROM orders WHERE o_orderkey = ?", (order_key,))
    result = cursor.fetchone()
    if not result:
        return {"total_price": []}
    return {"total_price": result[0]}

# Endpoint to get the count of line items with a specific discount
@app.get("/v1/retails/count_line_items_by_discount", operation_id="get_count_line_items_by_discount", summary="Retrieves the total number of line items that have a specified discount value. This operation provides a count of line items that match the given discount, offering insights into the prevalence of specific discounts in the line items.")
async def get_count_line_items_by_discount(discount: float = Query(..., description="Discount value")):
    cursor.execute("SELECT COUNT(l_orderkey) FROM lineitem WHERE l_discount = ?", (discount,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the line numbers of items shipped before a specific year and with a specific ship mode
@app.get("/v1/retails/line_numbers_by_year_ship_mode", operation_id="get_line_numbers_by_year_ship_mode", summary="Retrieves the line numbers of items shipped before a specified year and using a particular shipping mode. The year is provided in 'YYYY' format, and the shipping mode is specified as a string.")
async def get_line_numbers_by_year_ship_mode(year: int = Query(..., description="Year in 'YYYY' format"), ship_mode: str = Query(..., description="Ship mode of the line item")):
    cursor.execute("SELECT l_linenumber FROM lineitem WHERE STRFTIME('%Y', l_shipdate) < ? AND l_shipmode = ?", (year, ship_mode))
    result = cursor.fetchall()
    if not result:
        return {"line_numbers": []}
    return {"line_numbers": [row[0] for row in result]}

# Endpoint to get the line number of the item with the highest quantity for a specific ship mode
@app.get("/v1/retails/line_number_highest_quantity_by_ship_mode", operation_id="get_line_number_highest_quantity_by_ship_mode", summary="Get the line number of the item with the highest quantity for a specific ship mode")
async def get_line_number_highest_quantity_by_ship_mode(ship_mode: str = Query(..., description="Ship mode of the line item")):
    cursor.execute("SELECT l_linenumber FROM lineitem WHERE l_shipmode = ? ORDER BY l_quantity DESC LIMIT 1", (ship_mode,))
    result = cursor.fetchone()
    if not result:
        return {"line_number": []}
    return {"line_number": result[0]}

# Endpoint to get customer names with account balance less than a specified value
@app.get("/v1/retails/customer_names_by_acctbal", operation_id="get_customer_names_by_acctbal", summary="Get customer names with account balance less than a specified value")
async def get_customer_names_by_acctbal(max_acctbal: float = Query(..., description="Maximum account balance")):
    cursor.execute("SELECT c_name FROM customer WHERE c_acctbal < ?", (max_acctbal,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the count of customers in a specific market segment and nation
@app.get("/v1/retails/customer_count_by_segment_nation", operation_id="get_customer_count_by_segment_nation", summary="Get the count of customers in a specific market segment and nation")
async def get_customer_count_by_segment_nation(market_segment: str = Query(..., description="Market segment"), nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT COUNT(T1.c_name) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T1.c_mktsegment = ? AND T2.n_name = ?", (market_segment, nation_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer phone numbers based on order priority
@app.get("/v1/retails/customer_phones_by_order_priority", operation_id="get_customer_phones_by_order_priority", summary="Get customer phone numbers based on order priority")
async def get_customer_phones_by_order_priority(order_priority: str = Query(..., description="Order priority")):
    cursor.execute("SELECT T2.c_phone FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T1.o_orderpriority = ?", (order_priority,))
    result = cursor.fetchall()
    if not result:
        return {"customer_phones": []}
    return {"customer_phones": [row[0] for row in result]}

# Endpoint to get the customer name with the highest discount
@app.get("/v1/retails/customer_name_highest_discount", operation_id="get_customer_name_highest_discount", summary="Get the customer name with the highest discount")
async def get_customer_name_highest_discount():
    cursor.execute("SELECT T3.c_name FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey INNER JOIN customer AS T3 ON T1.o_custkey = T3.c_custkey ORDER BY T2.l_discount DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get customer names based on order total price
@app.get("/v1/retails/customer_names_by_order_total_price", operation_id="get_customer_names_by_order_total_price", summary="Get customer names based on order total price")
async def get_customer_names_by_order_total_price(min_total_price: float = Query(..., description="Minimum total price")):
    cursor.execute("SELECT T2.c_name FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T1.o_totalprice > ?", (min_total_price,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get customer names based on account balance and nation
@app.get("/v1/retails/customer_names_by_acctbal_nation", operation_id="get_customer_names_by_acctbal_nation", summary="Get customer names based on account balance and nation")
async def get_customer_names_by_acctbal_nation(min_acctbal: float = Query(..., description="Minimum account balance"), nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT T1.c_name FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T1.c_acctbal > ? AND T2.n_name = ?", (min_acctbal, nation_name))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get supplier phone numbers based on nation
@app.get("/v1/retails/supplier_phones_by_nation", operation_id="get_supplier_phones_by_nation", summary="Get supplier phone numbers based on nation")
async def get_supplier_phones_by_nation(nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT T1.s_phone FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T2.n_name = ?", (nation_name,))
    result = cursor.fetchall()
    if not result:
        return {"supplier_phones": []}
    return {"supplier_phones": [row[0] for row in result]}

# Endpoint to get supplier names based on account balance and nation
@app.get("/v1/retails/supplier_names_by_acctbal_nation", operation_id="get_supplier_names_by_acctbal_nation", summary="Get supplier names based on account balance and nation")
async def get_supplier_names_by_acctbal_nation(max_acctbal: float = Query(..., description="Maximum account balance"), nation_name: str = Query(..., description="Nation name")):
    cursor.execute("SELECT T1.s_name FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T1.s_acctbal < ? AND T2.n_name = ?", (max_acctbal, nation_name))
    result = cursor.fetchall()
    if not result:
        return {"supplier_names": []}
    return {"supplier_names": [row[0] for row in result]}

# Endpoint to get the count of regions based on nation name
@app.get("/v1/retails/count_regions_by_nation_name", operation_id="get_count_regions_by_nation_name", summary="Get the count of regions for a given nation name")
async def get_count_regions_by_nation_name(nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT COUNT(T1.r_name) FROM region AS T1 INNER JOIN nation AS T2 ON T1.r_regionkey = T2.n_regionkey WHERE T2.n_name = ?", (nation_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer names based on discount and order date range
@app.get("/v1/retails/customer_names_by_discount_and_order_date", operation_id="get_customer_names_by_discount_and_order_date", summary="Get customer names for orders with a specific discount and within a specified order date range")
async def get_customer_names_by_discount_and_order_date(discount: float = Query(..., description="Discount value"), start_year: int = Query(..., description="Start year in 'YYYY' format"), end_year: int = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT T3.c_name FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey INNER JOIN customer AS T3 ON T1.o_custkey = T3.c_custkey WHERE T2.l_discount = ? AND STRFTIME('%Y', T1.o_orderdate) BETWEEN ? AND ?", (discount, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the percentage of customers in a specific market segment and nation
@app.get("/v1/retails/percentage_customers_in_market_segment_and_nation", operation_id="get_percentage_customers_in_market_segment_and_nation", summary="Get the percentage of customers in a specific market segment and nation")
async def get_percentage_customers_in_market_segment_and_nation(market_segment: str = Query(..., description="Market segment"), nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.c_mktsegment = ?, 1, 0)) AS REAL) * 100 / COUNT(T1.c_mktsegment) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey WHERE T2.n_name = ?", (market_segment, nation_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get part comments based on part name
@app.get("/v1/retails/part_comments_by_name", operation_id="get_part_comments_by_name", summary="Get part comments for a specific part name")
async def get_part_comments_by_name(part_name: str = Query(..., description="Name of the part")):
    cursor.execute("SELECT p_comment FROM part WHERE p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"part_comment": []}
    return {"part_comment": result[0]}

# Endpoint to get the count of parts based on retail price
@app.get("/v1/retails/count_parts_by_retail_price", operation_id="get_count_parts_by_retail_price", summary="Get the count of parts with a retail price greater than a specified value")
async def get_count_parts_by_retail_price(retail_price: float = Query(..., description="Retail price threshold")):
    cursor.execute("SELECT COUNT(p_partkey) FROM part WHERE p_retailprice > ?", (retail_price,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of parts based on type and manufacturer
@app.get("/v1/retails/count_parts_by_type_and_manufacturer", operation_id="get_count_parts_by_type_and_manufacturer", summary="Get the count of parts with a specific type and manufacturer")
async def get_count_parts_by_type_and_manufacturer(part_type: str = Query(..., description="Type of the part"), manufacturer: str = Query(..., description="Manufacturer of the part")):
    cursor.execute("SELECT COUNT(p_partkey) FROM part WHERE p_type = ? AND p_mfgr = ?", (part_type, manufacturer))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get part brands based on part type
@app.get("/v1/retails/part_brands_by_type", operation_id="get_part_brands_by_type", summary="Get part brands for a specific part type")
async def get_part_brands_by_type(part_type: str = Query(..., description="Type of the part")):
    cursor.execute("SELECT p_brand FROM part WHERE p_type = ?", (part_type,))
    result = cursor.fetchall()
    if not result:
        return {"part_brands": []}
    return {"part_brands": [row[0] for row in result]}

# Endpoint to get part names with the highest retail price
@app.get("/v1/retails/part_names_with_highest_retail_price", operation_id="get_part_names_with_highest_retail_price", summary="Get part names with the highest retail price")
async def get_part_names_with_highest_retail_price():
    cursor.execute("SELECT p_name FROM part WHERE p_retailprice = ( SELECT MAX(p_retailprice) FROM part )")
    result = cursor.fetchall()
    if not result:
        return {"part_names": []}
    return {"part_names": [row[0] for row in result]}

# Endpoint to get the part name with the largest size from a given list of part names
@app.get("/v1/retails/largest_part_name", operation_id="get_largest_part_name", summary="Get the part name with the largest size from a given list of part names")
async def get_largest_part_name(part_name1: str = Query(..., description="First part name"), part_name2: str = Query(..., description="Second part name")):
    cursor.execute("SELECT T.p_name FROM ( SELECT p_name, p_size FROM part WHERE p_name IN (?, ?) ) AS T ORDER BY p_size DESC LIMIT 1", (part_name1, part_name2))
    result = cursor.fetchone()
    if not result:
        return {"part_name": []}
    return {"part_name": result[0]}

# Endpoint to get the count of parts with a specific container type
@app.get("/v1/retails/count_parts_by_container", operation_id="get_count_parts_by_container", summary="Get the count of parts with a specific container type")
async def get_count_parts_by_container(container_type: str = Query(..., description="Container type")):
    cursor.execute("SELECT COUNT(p_partkey) FROM part WHERE p_container = ?", (container_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the minimum size of parts with a specific container type
@app.get("/v1/retails/min_part_size_by_container", operation_id="get_min_part_size_by_container", summary="Get the minimum size of parts with a specific container type")
async def get_min_part_size_by_container(container_type: str = Query(..., description="Container type")):
    cursor.execute("SELECT MIN(p_size) FROM part WHERE p_container = ?", (container_type,))
    result = cursor.fetchone()
    if not result:
        return {"min_size": []}
    return {"min_size": result[0]}

# Endpoint to get the count of suppliers with account balance less than a specified value
@app.get("/v1/retails/count_suppliers_by_account_balance", operation_id="get_count_suppliers_by_account_balance", summary="Get the count of suppliers with account balance less than a specified value")
async def get_count_suppliers_by_account_balance(account_balance: float = Query(..., description="Account balance threshold")):
    cursor.execute("SELECT COUNT(s_suppkey) FROM supplier WHERE s_acctbal < ?", (account_balance,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get supplier names ordered by account balance in descending order with a limit
@app.get("/v1/retails/top_suppliers_by_account_balance", operation_id="get_top_suppliers_by_account_balance", summary="Get supplier names ordered by account balance in descending order with a limit")
async def get_top_suppliers_by_account_balance(limit: int = Query(..., description="Limit of suppliers to return")):
    cursor.execute("SELECT s_name FROM supplier ORDER BY s_acctbal DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"suppliers": []}
    return {"suppliers": [row[0] for row in result]}

# Endpoint to get the minimum supply cost for a specific part name
@app.get("/v1/retails/min_supply_cost_by_part", operation_id="get_min_supply_cost_by_part", summary="Get the minimum supply cost for a specific part name")
async def get_min_supply_cost_by_part(part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT MIN(T1.ps_supplycost) FROM partsupp AS T1 INNER JOIN part AS T2 ON T1.ps_partkey = T2.p_partkey WHERE T2.p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"min_supply_cost": []}
    return {"min_supply_cost": result[0]}

# Endpoint to get the supplier name with the lowest supply cost for a specific part name
@app.get("/v1/retails/supplier_with_lowest_supply_cost", operation_id="get_supplier_with_lowest_supply_cost", summary="Get the supplier name with the lowest supply cost for a specific part name")
async def get_supplier_with_lowest_supply_cost(part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT T2.s_name FROM partsupp AS T1 INNER JOIN supplier AS T2 ON T1.ps_suppkey = T2.s_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey WHERE T3.p_name = ? ORDER BY T1.ps_supplycost LIMIT 1", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"supplier_name": []}
    return {"supplier_name": result[0]}

# Endpoint to get the total available quantity for a specific part name
@app.get("/v1/retails/total_available_quantity_by_part", operation_id="get_total_available_quantity_by_part", summary="Get the total available quantity for a specific part name")
async def get_total_available_quantity_by_part(part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT SUM(T1.ps_availqty) FROM partsupp AS T1 INNER JOIN part AS T2 ON T1.ps_partkey = T2.p_partkey WHERE T2.p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the supplier phone number for a part with a specific name, ordered by available quantity
@app.get("/v1/retails/supplier_phone_by_part_name_availqty", operation_id="get_supplier_phone_by_part_name_availqty", summary="Get the supplier phone number for a part with a specific name, ordered by available quantity")
async def get_supplier_phone_by_part_name_availqty(part_name: str = Query(..., description="Name of the part")):
    cursor.execute("SELECT T3.s_phone FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey WHERE T1.p_name = ? ORDER BY T2.ps_availqty DESC LIMIT 1", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"s_phone": []}
    return {"s_phone": result[0]}

# Endpoint to get the supplier phone number for a part with a specific name, ordered by part size
@app.get("/v1/retails/supplier_phone_by_part_name_size", operation_id="get_supplier_phone_by_part_name_size", summary="Get the supplier phone number for a part with a specific name, ordered by part size")
async def get_supplier_phone_by_part_name_size(part_name: str = Query(..., description="Name of the part")):
    cursor.execute("SELECT T3.s_phone FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey WHERE T1.p_name = ? ORDER BY T1.p_size DESC LIMIT 1", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"s_phone": []}
    return {"s_phone": result[0]}

# Endpoint to get the count of suppliers for a part with a specific name and nation
@app.get("/v1/retails/count_suppliers_by_part_name_nation", operation_id="get_count_suppliers_by_part_name_nation", summary="Get the count of suppliers for a part with a specific name and nation")
async def get_count_suppliers_by_part_name_nation(part_name: str = Query(..., description="Name of the part"), nation_name: str = Query(..., description="Name of the nation")):
    cursor.execute("SELECT COUNT(T3.s_name) FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey INNER JOIN nation AS T4 ON T3.s_nationkey = T4.n_nationkey WHERE T1.p_name = ? AND T4.n_name = ?", (part_name, nation_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of suppliers with a negative account balance and a specific part type
@app.get("/v1/retails/count_suppliers_by_acctbal_part_type", operation_id="get_count_suppliers_by_acctbal_part_type", summary="Get the count of suppliers with a negative account balance and a specific part type")
async def get_count_suppliers_by_acctbal_part_type(acctbal: float = Query(..., description="Account balance"), part_type: str = Query(..., description="Type of the part")):
    cursor.execute("SELECT COUNT(T3.s_name) FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey WHERE T3.s_acctbal < ? AND T1.p_type = ?", (acctbal, part_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the supplier names for a specific part brand
@app.get("/v1/retails/supplier_names_by_part_brand", operation_id="get_supplier_names_by_part_brand", summary="Get the supplier names for a specific part brand")
async def get_supplier_names_by_part_brand(part_brand: str = Query(..., description="Brand of the part")):
    cursor.execute("SELECT T3.s_name FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey WHERE T1.p_brand = ?", (part_brand,))
    result = cursor.fetchall()
    if not result:
        return {"s_name": []}
    return {"s_name": [row[0] for row in result]}

# Endpoint to get the sum of supplier counts for parts with a specific type and available quantity less than a specified value
@app.get("/v1/retails/sum_supplier_counts_by_part_type_availqty", operation_id="get_sum_supplier_counts_by_part_type_availqty", summary="Get the sum of supplier counts for parts with a specific type and available quantity less than a specified value")
async def get_sum_supplier_counts_by_part_type_availqty(part_type: str = Query(..., description="Type of the part"), availqty: int = Query(..., description="Available quantity")):
    cursor.execute("SELECT SUM(num) FROM ( SELECT COUNT(T3.s_name) AS num FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey INNER JOIN supplier AS T3 ON T2.ps_suppkey = T3.s_suppkey WHERE T1.p_type = ? GROUP BY T2.ps_partkey HAVING SUM(T2.ps_availqty) < ? ) T", (part_type, availqty))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the count of part keys for a specific part name
@app.get("/v1/retails/count_part_keys_by_part_name", operation_id="get_count_part_keys_by_part_name", summary="Get the count of part keys for a specific part name")
async def get_count_part_keys_by_part_name(part_name: str = Query(..., description="Name of the part")):
    cursor.execute("SELECT COUNT(T1.p_partkey) FROM part AS T1 INNER JOIN lineitem AS T2 ON T1.p_partkey = T2.l_partkey WHERE T1.p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of part keys for a specific part name
@app.get("/v1/retails/sum_part_keys_by_part_name", operation_id="get_sum_part_keys_by_part_name", summary="Get the sum of part keys for a specific part name")
async def get_sum_part_keys_by_part_name(part_name: str = Query(..., description="Name of the part")):
    cursor.execute("SELECT SUM(T1.p_partkey) FROM part AS T1 INNER JOIN lineitem AS T2 ON T1.p_partkey = T2.l_partkey WHERE T1.p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the order keys for parts with a specific container and more than a specified number of part keys
@app.get("/v1/retails/order_keys_by_part_container_count", operation_id="get_order_keys_by_part_container_count", summary="Get the order keys for parts with a specific container and more than a specified number of part keys")
async def get_order_keys_by_part_container_count(part_container: str = Query(..., description="Container of the part"), count: int = Query(..., description="Number of part keys")):
    cursor.execute("SELECT T.l_orderkey FROM ( SELECT T2.l_orderkey, COUNT(T2.l_partkey) AS num FROM part AS T1 INNER JOIN lineitem AS T2 ON T1.p_partkey = T2.l_partkey WHERE T1.p_container = ? GROUP BY T2.l_orderkey ) AS T WHERE T.num > ?", (part_container, count))
    result = cursor.fetchall()
    if not result:
        return {"l_orderkey": []}
    return {"l_orderkey": [row[0] for row in result]}

# Endpoint to get the count of nation keys for a specific region and account balance
@app.get("/v1/retails/count_nation_keys_by_region_acctbal", operation_id="get_count_nation_keys_by_region_acctbal", summary="Get the count of nation keys for a specific region and account balance")
async def get_count_nation_keys_by_region_acctbal(region_name: str = Query(..., description="Name of the region"), acctbal: float = Query(..., description="Account balance")):
    cursor.execute("SELECT COUNT(T1.n_nationkey) FROM nation AS T1 INNER JOIN region AS T2 ON T1.n_regionkey = T2.r_regionkey INNER JOIN supplier AS T3 ON T1.n_nationkey = T3.s_nationkey WHERE T2.r_name = ? AND T3.s_acctbal < ?", (region_name, acctbal))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of regions based on region name
@app.get("/v1/retails/count_regions_by_name", operation_id="get_count_regions_by_name", summary="Get the count of regions based on the region name")
async def get_count_regions_by_name(region_name: str = Query(..., description="Name of the region")):
    cursor.execute("SELECT COUNT(T1.r_regionkey) FROM region AS T1 INNER JOIN nation AS T2 ON T1.r_regionkey = T2.n_regionkey INNER JOIN supplier AS T3 ON T2.n_nationkey = T3.s_nationkey WHERE T1.r_name = ?", (region_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get supplier phone number based on order key
@app.get("/v1/retails/supplier_phone_by_order_key", operation_id="get_supplier_phone_by_order_key", summary="Get the supplier phone number based on the order key")
async def get_supplier_phone_by_order_key(order_key: int = Query(..., description="Order key")):
    cursor.execute("SELECT T2.s_phone FROM lineitem AS T1 INNER JOIN supplier AS T2 ON T1.l_suppkey = T2.s_suppkey WHERE T1.l_orderkey = ?", (order_key,))
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the count of line items based on order key and account balance
@app.get("/v1/retails/count_line_items_by_order_key_and_account_balance", operation_id="get_count_line_items_by_order_key_and_account_balance", summary="Get the count of line items based on the order key and account balance")
async def get_count_line_items_by_order_key_and_account_balance(order_key: int = Query(..., description="Order key"), account_balance: float = Query(..., description="Account balance")):
    cursor.execute("SELECT COUNT(T1.l_linenumber) FROM lineitem AS T1 INNER JOIN supplier AS T2 ON T1.l_suppkey = T2.s_suppkey WHERE T1.l_orderkey = ? AND T2.s_acctbal < ?", (order_key, account_balance))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of line items based on return flag and account balance
@app.get("/v1/retails/count_line_items_by_return_flag_and_account_balance", operation_id="get_count_line_items_by_return_flag_and_account_balance", summary="Get the count of line items based on the return flag and account balance")
async def get_count_line_items_by_return_flag_and_account_balance(return_flag: str = Query(..., description="Return flag"), account_balance: float = Query(..., description="Account balance")):
    cursor.execute("SELECT COUNT(T1.l_partkey) FROM lineitem AS T1 INNER JOIN supplier AS T2 ON T1.l_suppkey = T2.s_suppkey WHERE T1.l_returnflag = ? AND T2.s_acctbal < ?", (return_flag, account_balance))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ship date based on order key and part name
@app.get("/v1/retails/ship_date_by_order_key_and_part_name", operation_id="get_ship_date_by_order_key_and_part_name", summary="Get the ship date based on the order key and part name")
async def get_ship_date_by_order_key_and_part_name(order_key: int = Query(..., description="Order key"), part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT T1.l_shipdate FROM lineitem AS T1 INNER JOIN part AS T2 ON T1.l_partkey = T2.p_partkey WHERE T1.l_orderkey = ? AND T2.p_name = ?", (order_key, part_name))
    result = cursor.fetchone()
    if not result:
        return {"ship_date": []}
    return {"ship_date": result[0]}

# Endpoint to get the quantity based on order key and part name
@app.get("/v1/retails/quantity_by_order_key_and_part_name", operation_id="get_quantity_by_order_key_and_part_name", summary="Get the quantity based on the order key and part name")
async def get_quantity_by_order_key_and_part_name(order_key: int = Query(..., description="Order key"), part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT T1.l_quantity FROM lineitem AS T1 INNER JOIN part AS T2 ON T1.l_partkey = T2.p_partkey WHERE T1.l_orderkey = ? AND T2.p_name = ?", (order_key, part_name))
    result = cursor.fetchone()
    if not result:
        return {"quantity": []}
    return {"quantity": result[0]}

# Endpoint to get the part name with the highest quantity based on a list of part names
@app.get("/v1/retails/part_name_with_highest_quantity", operation_id="get_part_name_with_highest_quantity", summary="Get the part name with the highest quantity based on a list of part names")
async def get_part_name_with_highest_quantity(part_name_1: str = Query(..., description="First part name"), part_name_2: str = Query(..., description="Second part name")):
    cursor.execute("SELECT T.p_name FROM ( SELECT T2.p_name, SUM(T1.l_quantity) AS num FROM lineitem AS T1 INNER JOIN part AS T2 ON T1.l_partkey = T2.p_partkey WHERE T2.p_name IN (?, ?) GROUP BY T1.l_partkey ) AS T ORDER BY T.num DESC LIMIT 1", (part_name_1, part_name_2))
    result = cursor.fetchone()
    if not result:
        return {"part_name": []}
    return {"part_name": result[0]}

# Endpoint to get the maximum discount based on part name
@app.get("/v1/retails/max_discount_by_part_name", operation_id="get_max_discount_by_part_name", summary="Get the maximum discount based on the part name")
async def get_max_discount_by_part_name(part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT MAX(T1.l_discount) FROM lineitem AS T1 INNER JOIN part AS T2 ON T1.l_partkey = T2.p_partkey WHERE T2.p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"max_discount": []}
    return {"max_discount": result[0]}

# Endpoint to get distinct ship modes based on part name
@app.get("/v1/retails/distinct_ship_modes_by_part_name", operation_id="get_distinct_ship_modes_by_part_name", summary="Get distinct ship modes based on the part name")
async def get_distinct_ship_modes_by_part_name(part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT DISTINCT T1.l_shipmode FROM lineitem AS T1 INNER JOIN part AS T2 ON T1.l_partkey = T2.p_partkey WHERE T2.p_name = ?", (part_name,))
    result = cursor.fetchall()
    if not result:
        return {"ship_modes": []}
    return {"ship_modes": [row[0] for row in result]}

# Endpoint to get the average supply cost based on part name
@app.get("/v1/retails/average_supply_cost_by_part_name", operation_id="get_average_supply_cost_by_part_name", summary="Get the average supply cost based on the part name")
async def get_average_supply_cost_by_part_name(part_name: str = Query(..., description="Part name")):
    cursor.execute("SELECT AVG(T1.ps_supplycost) FROM partsupp AS T1 INNER JOIN supplier AS T2 ON T1.ps_suppkey = T2.s_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey WHERE T3.p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_supply_cost": []}
    return {"average_supply_cost": result[0]}

# Endpoint to get the percentage difference between max and min supply cost for a specific part name
@app.get("/v1/retails/supply_cost_percentage_difference", operation_id="get_supply_cost_percentage_difference", summary="Get the percentage difference between max and min supply cost for a specific part name")
async def get_supply_cost_percentage_difference(part_name: str = Query(..., description="Name of the part")):
    cursor.execute("SELECT CAST((MAX(T1.ps_supplycost) - MIN(T1.ps_supplycost)) AS REAL) * 100 / MIN(T1.ps_supplycost) FROM partsupp AS T1 INNER JOIN supplier AS T2 ON T1.ps_suppkey = T2.s_suppkey INNER JOIN part AS T3 ON T1.ps_partkey = T3.p_partkey WHERE T3.p_name = ?", (part_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the profit for a specific order and part key
@app.get("/v1/retails/profit_by_order_and_part", operation_id="get_profit_by_order_and_part", summary="Get the profit for a specific order and part key")
async def get_profit_by_order_and_part(order_key: int = Query(..., description="Order key"), part_key: int = Query(..., description="Part key")):
    cursor.execute("SELECT T1.l_extendedprice * (1 - T1.l_discount) - T2.ps_supplycost * T1.l_quantity FROM lineitem AS T1 INNER JOIN partsupp AS T2 ON T1.l_suppkey = T2.ps_suppkey WHERE T1.l_orderkey = ? AND T1.l_partkey = ?", (order_key, part_key))
    result = cursor.fetchone()
    if not result:
        return {"profit": []}
    return {"profit": result[0]}

# Endpoint to get the market segment of the customer with the minimum account balance
@app.get("/v1/retails/min_account_balance_market_segment", operation_id="get_min_account_balance_market_segment", summary="Get the market segment of the customer with the minimum account balance")
async def get_min_account_balance_market_segment():
    cursor.execute("SELECT c_mktsegment FROM customer WHERE c_acctbal = ( SELECT MIN(c_acctbal) FROM customer )")
    result = cursor.fetchone()
    if not result:
        return {"market_segment": []}
    return {"market_segment": result[0]}

# Endpoint to get the count of line items shipped in a specific year and mode
@app.get("/v1/retails/count_line_items_shipped", operation_id="get_count_line_items_shipped", summary="Get the count of line items shipped in a specific year and mode")
async def get_count_line_items_shipped(year: str = Query(..., description="Year in 'YYYY' format"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT COUNT(l_orderkey) FROM lineitem WHERE STRFTIME('%Y', l_shipdate) = ? AND l_shipmode = ?", (year, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of line items shipped in a specific year, return flag, and ship mode
@app.get("/v1/retails/count_line_items_shipped_with_return_flag", operation_id="get_count_line_items_shipped_with_return_flag", summary="Get the count of line items shipped in a specific year, return flag, and ship mode")
async def get_count_line_items_shipped_with_return_flag(year: str = Query(..., description="Year in 'YYYY' format"), return_flag: str = Query(..., description="Return flag"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT COUNT(l_orderkey) FROM lineitem WHERE STRFTIME('%Y', l_shipdate) = ? AND l_returnflag = ? AND l_shipmode = ?", (year, return_flag, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers with account balance less than a specific value and in a specific market segment
@app.get("/v1/retails/count_customers_by_account_balance_and_market_segment", operation_id="get_count_customers_by_account_balance_and_market_segment", summary="Get the count of customers with account balance less than a specific value and in a specific market segment")
async def get_count_customers_by_account_balance_and_market_segment(account_balance: float = Query(..., description="Account balance"), market_segment: str = Query(..., description="Market segment")):
    cursor.execute("SELECT COUNT(c_custkey) FROM customer WHERE c_acctbal < ? AND c_mktsegment = ?", (account_balance, market_segment))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of orders in a specific year, by a specific clerk, and with a specific order priority
@app.get("/v1/retails/count_orders_by_year_clerk_priority", operation_id="get_count_orders_by_year_clerk_priority", summary="Get the count of orders in a specific year, by a specific clerk, and with a specific order priority")
async def get_count_orders_by_year_clerk_priority(year: str = Query(..., description="Year in 'YYYY' format"), clerk: str = Query(..., description="Clerk"), order_priority: str = Query(..., description="Order priority")):
    cursor.execute("SELECT COUNT(o_orderkey) FROM orders WHERE STRFTIME('%Y', o_orderdate) = ? AND o_clerk = ? AND o_orderpriority = ?", (year, clerk, order_priority))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer name with the longest delay between receipt and commit date
@app.get("/v1/retails/customer_with_longest_delay", operation_id="get_customer_with_longest_delay", summary="Get the customer name with the longest delay between receipt and commit date")
async def get_customer_with_longest_delay():
    cursor.execute("SELECT T3.c_name FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey INNER JOIN customer AS T3 ON T1.o_custkey = T3.c_custkey ORDER BY (JULIANDAY(T2.l_receiptdate) - JULIANDAY(T2.l_commitdate)) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the count of customers in a specific market segment with orders exceeding a certain total price
@app.get("/v1/retails/customer_count_by_market_segment_and_total_price", operation_id="get_customer_count_by_market_segment_and_total_price", summary="Get the count of customers in a specific market segment with orders exceeding a certain total price")
async def get_customer_count_by_market_segment_and_total_price(market_segment: str = Query(..., description="Market segment of the customer"), total_price: int = Query(..., description="Total price of the order")):
    cursor.execute("SELECT COUNT(T2.c_name) FROM orders AS T1 INNER JOIN customer AS T2 ON T1.o_custkey = T2.c_custkey WHERE T2.c_mktsegment = ? AND T1.o_totalprice > ?", (market_segment, total_price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the nation with the highest number of suppliers
@app.get("/v1/retails/nation_with_most_suppliers", operation_id="get_nation_with_most_suppliers", summary="Get the nation with the highest number of suppliers")
async def get_nation_with_most_suppliers():
    cursor.execute("SELECT T2.n_name FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey GROUP BY T1.s_nationkey ORDER BY COUNT(T1.s_name) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"nation": []}
    return {"nation": result[0]}

# Endpoint to get the top 2 nations with the highest negative account balances
@app.get("/v1/retails/top_nations_by_negative_account_balance", operation_id="get_top_nations_by_negative_account_balance", summary="Get the top 2 nations with the highest negative account balances")
async def get_top_nations_by_negative_account_balance():
    cursor.execute("SELECT T.n_name FROM ( SELECT T2.n_name, SUM(T1.s_acctbal) AS num FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey WHERE T1.s_acctbal < 0 GROUP BY T1.s_nationkey ) AS T ORDER BY T.num LIMIT 2")
    result = cursor.fetchall()
    if not result:
        return {"nations": []}
    return {"nations": [row[0] for row in result]}

# Endpoint to get the names of parts with a supply cost greater than a specified value
@app.get("/v1/retails/part_names_by_supply_cost", operation_id="get_part_names_by_supply_cost", summary="Get the names of parts with a supply cost greater than a specified value")
async def get_part_names_by_supply_cost(supply_cost: int = Query(..., description="Supply cost of the part")):
    cursor.execute("SELECT T1.p_name FROM part AS T1 INNER JOIN partsupp AS T2 ON T1.p_partkey = T2.ps_partkey WHERE T2.ps_supplycost > ?", (supply_cost,))
    result = cursor.fetchall()
    if not result:
        return {"part_names": []}
    return {"part_names": [row[0] for row in result]}

# Endpoint to get the nation of the supplier with the highest supplier key
@app.get("/v1/retails/nation_of_highest_supplier_key", operation_id="get_nation_of_highest_supplier_key", summary="Get the nation of the supplier with the highest supplier key")
async def get_nation_of_highest_supplier_key():
    cursor.execute("SELECT T2.n_name FROM supplier AS T1 INNER JOIN nation AS T2 ON T1.s_nationkey = T2.n_nationkey ORDER BY T1.s_suppkey DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"nation": []}
    return {"nation": result[0]}

# Endpoint to get the clerk of the order with the highest extended price
@app.get("/v1/retails/clerk_of_highest_extended_price_order", operation_id="get_clerk_of_highest_extended_price_order", summary="Get the clerk of the order with the highest extended price")
async def get_clerk_of_highest_extended_price_order():
    cursor.execute("SELECT T1.o_clerk FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey ORDER BY T2.l_extendedprice DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"clerk": []}
    return {"clerk": result[0]}

# Endpoint to get the total quantity of items ordered on a specific date by a specific customer
@app.get("/v1/retails/total_quantity_by_order_date_and_customer", operation_id="get_total_quantity_by_order_date_and_customer", summary="Get the total quantity of items ordered on a specific date by a specific customer")
async def get_total_quantity_by_order_date_and_customer(order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format"), cust_key: int = Query(..., description="Customer key")):
    cursor.execute("SELECT SUM(T2.l_quantity) FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_orderdate = ? AND T1.o_custkey = ?", (order_date, cust_key))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the total revenue from orders by a specific customer on a specific date
@app.get("/v1/retails/total_revenue_by_customer_and_order_date", operation_id="get_total_revenue_by_customer_and_order_date", summary="Get the total revenue from orders by a specific customer on a specific date")
async def get_total_revenue_by_customer_and_order_date(cust_key: int = Query(..., description="Customer key"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(T2.l_extendedprice * (1 - T2.l_discount) * (1 + T2.l_tax)) FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey WHERE T1.o_custkey = ? AND T1.o_orderdate = ?", (cust_key, order_date))
    result = cursor.fetchone()
    if not result:
        return {"total_revenue": []}
    return {"total_revenue": result[0]}

# Endpoint to get part names from orders based on customer key
@app.get("/v1/retails/part_names_by_customer_key", operation_id="get_part_names_by_customer_key", summary="Get part names from orders based on customer key")
async def get_part_names_by_customer_key(o_custkey: int = Query(..., description="Customer key")):
    cursor.execute("SELECT T3.p_name FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey INNER JOIN part AS T3 ON T2.l_partkey = T3.p_partkey WHERE T1.o_custkey = ?", (o_custkey,))
    result = cursor.fetchall()
    if not result:
        return {"part_names": []}
    return {"part_names": [row[0] for row in result]}

# Endpoint to get extended price and part names from orders based on customer key and order key
@app.get("/v1/retails/extended_price_part_names_by_customer_order_key", operation_id="get_extended_price_part_names_by_customer_order_key", summary="Get extended price and part names from orders based on customer key and order key")
async def get_extended_price_part_names_by_customer_order_key(o_custkey: int = Query(..., description="Customer key"), o_orderkey: int = Query(..., description="Order key")):
    cursor.execute("SELECT T2.l_extendedprice * (1 - T2.l_discount), T3.p_name FROM orders AS T1 INNER JOIN lineitem AS T2 ON T1.o_orderkey = T2.l_orderkey INNER JOIN part AS T3 ON T2.l_partkey = T3.p_partkey WHERE T1.o_custkey = ? AND T1.o_orderkey = ?", (o_custkey, o_orderkey))
    result = cursor.fetchall()
    if not result:
        return {"extended_price_part_names": []}
    return {"extended_price_part_names": [{"extended_price": row[0], "part_name": row[1]} for row in result]}

api_calls = [
    "/v1/retails/lineitem_count_by_orderkey_returnflag?orderkey=5&returnflag=R",
    "/v1/retails/max_shipdate_by_orderkey?orderkey=1",
    "/v1/retails/latest_orderkey_by_orderkeys?orderkey1=4&orderkey2=36",
    "/v1/retails/order_comment_by_max_totalprice",
    "/v1/retails/customer_phone_by_name?name=Customer%23000000001",
    "/v1/retails/order_count_by_market_segment?market_segment=HOUSEHOLD",
    "/v1/retails/max_totalprice_by_market_segment?market_segment=HOUSEHOLD",
    "/v1/retails/order_comments_by_market_segment?market_segment=HOUSEHOLD",
    "/v1/retails/customer_name_by_max_totalprice",
    "/v1/retails/orderkeys_by_account_balance?account_balance=0",
    "/v1/retails/order_count_by_segment_priority?market_segment=HOUSEHOLD&order_priority=1-URGENT",
    "/v1/retails/customer_count_by_nation?nation_name=BRAZIL",
    "/v1/retails/customer_phones_by_segment_nation?market_segment=HOUSEHOLD&nation_name=BRAZIL",
    "/v1/retails/customer_count_by_nation_balance?nation_name=GERMANY&min_balance=1000",
    "/v1/retails/customer_count_with_orders_by_nation?nation_name=GERMANY",
    "/v1/retails/total_order_price_by_nation?nation_name=GERMANY",
    "/v1/retails/first_order_key_by_nation?nation_name=GERMANY",
    "/v1/retails/average_order_price_by_nation?nation_name=GERMANY",
    "/v1/retails/percentage_customers_by_nation?nation_name=GERMANY",
    "/v1/retails/nation_count_by_region?region_key=2",
    "/v1/retails/nation_name_by_supplier_key?s_suppkey=34",
    "/v1/retails/region_name_by_supplier_name?s_name=Supplier%23000000129",
    "/v1/retails/nation_name_by_customer_name?c_name=Customer%23000000055",
    "/v1/retails/region_name_by_customer_key?c_custkey=106936",
    "/v1/retails/customer_count_by_nation_and_balance?n_name=MOROCCO&c_acctbal=0",
    "/v1/retails/discounted_price_by_supplier_and_total?l_suppkey=9397&o_totalprice=231499.38",
    "/v1/retails/supplier_key_by_total_and_return_flag?o_totalprice=218195.43&l_returnflag=R",
    "/v1/retails/calculated_price_by_clerk_shipmode_status_date?o_clerk=Clerk%23000000936&l_shipmode=TRUCK&o_orderstatus=4-NOT%20SPECIFIED&o_orderdate=1995-03-13",
    "/v1/retails/days_difference_by_customer_and_order_date?o_custkey=129301&o_orderdate=1996-07-27",
    "/v1/retails/customer_name_by_order_date_and_clerk?o_orderdate=1997-12-10&o_clerk=Clerk%23000000803",
    "/v1/retails/calculated_value_by_suppkey_orderkey?suppkey=7414&orderkey=817154",
    "/v1/retails/nation_with_highest_negative_balance",
    "/v1/retails/percentage_nations_in_region?region_name=EUROPE",
    "/v1/retails/percentage_suppliers_negative_balance_nation?nation_name=JAPAN",
    "/v1/retails/customer_with_minimum_balance",
    "/v1/retails/count_orders_shipped_year?year=1998",
    "/v1/retails/count_customers_negative_balance",
    "/v1/retails/count_lineitems_returnflag_shipmode_year?return_flag=R&ship_mode=AIR&year=1994",
    "/v1/retails/count_customers_market_segment?market_segment=AUTOMOBILE",
    "/v1/retails/top_orderkeys_by_extendedprice?limit=2",
    "/v1/retails/highest_total_price_order_ship_date",
    "/v1/retails/nation_with_most_customers",
    "/v1/retails/count_orders_shipped_one_day_after?order_priority=1-URGENT",
    "/v1/retails/count_orders_by_shipmode_priority?shipmode=SHIP&orderpriority=3-MEDIUM",
    "/v1/retails/top_market_segment_by_nation?nation=UNITED%20STATES",
    "/v1/retails/nations_by_region?region=ASIA",
    "/v1/retails/parts_by_supplycost_manufacturer?supplycost=1000&manufacturer=Manufacturer%233",
    "/v1/retails/count_nations_by_region_comment?comment=asymptotes%20sublate%20after%20the%20r",
    "/v1/retails/count_part_keys_by_manufacturer_price_shipmode?manufacturer=Manufacturer%235&retail_price=1000&ship_mode=RAIL",
    "/v1/retails/top_orders_by_total_price",
    "/v1/retails/top_customers_by_order_count",
    "/v1/retails/count_line_items_by_ship_date?ship_date=1993-12-04",
    "/v1/retails/order_date_max_total_price",
    "/v1/retails/percentage_negative_account_balance",
    "/v1/retails/count_supplier_keys_by_available_quantity?available_quantity=10",
    "/v1/retails/percentage_parts_by_manufacturer?manufacturer=Manufacturer%233",
    "/v1/retails/part_names_by_type?part_type=MEDIUM%20PLATED%20BRASS",
    "/v1/retails/customer_count_by_balance_nation?account_balance=0&nation_name=MOROCCO",
    "/v1/retails/nation_details_by_region?region_name=AFRICA",
    "/v1/retails/order_details_by_customer?customer_name=Customer%23000021159",
    "/v1/retails/total_profit_by_part?part_name=chocolate%20floral%20blue%20coral%20cyan",
    "/v1/retails/supplier_percentage_by_nation?nation_name=GERMANY&account_balance=0",
    "/v1/retails/supplier_names_by_part?part_name=smoke%20red%20pale%20saddle%20plum",
    "/v1/retails/supplier_count_by_region?account_balance=0&region_name=MIDDLE%20EAST",
    "/v1/retails/part_names_by_discount_shipdate_shipmode?discount=0.1&ship_date=1995-12-01&ship_mode=RAIL",
    "/v1/retails/part_names_by_supplycost_suppliername?supply_cost=900&supplier_name=Supplier%23000000018",
    "/v1/retails/count_linenumbers_by_quantity_shipmode?quantity=30&ship_mode=RAIL",
    "/v1/retails/count_customerkeys_by_mktsegment_nationkey?mktsegment=FURNITURE&nationkey=1",
    "/v1/retails/customer_phone_highest_balance",
    "/v1/retails/order_priority_highest_totalprice",
    "/v1/retails/count_orderkeys_by_nationname?nation_name=UNITED%20STATES",
    "/v1/retails/count_customerkeys_by_mktsegment_nationname?mktsegment=AUTOMOBILE&nation_name=BRAZIL",
    "/v1/retails/order_comments_by_mktsegment?mktsegment=Furniture",
    "/v1/retails/nation_names_by_region_comment?region_comment=furiously%20express%20accounts%20wake%20sly",
    "/v1/retails/supplier_count_by_nation?nation_name=GERMANY",
    "/v1/retails/nation_count_by_account_balance_region?account_balance=0&region_name=ASIA",
    "/v1/retails/customer_phone_highest_order",
    "/v1/retails/part_supplier_count_by_retail_price_ship_mode?retail_price=1000&ship_mode=SHIP",
    "/v1/retails/customer_info_by_order_total_price?order_total_price=199180.63",
    "/v1/retails/nation_region_by_customer_address?customer_address=wH55UnX7%20VI",
    "/v1/retails/customer_count_by_nation_account_balance?nation_name=BRAZIL&account_balance=1000",
    "/v1/retails/nation_names_above_avg_account_balance",
    "/v1/retails/percentage_customers_by_nation_account_balance?nation_name=United%20States&account_balance=4000",
    "/v1/retails/customer_details_by_balance?min_balance=9000",
    "/v1/retails/avg_line_number_by_ship_date?start_date=1994-01-01&end_date=1994-01-30",
    "/v1/retails/order_keys_by_total_price?min_price=200000&max_price=300000",
    "/v1/retails/part_keys_above_avg_retail_price",
    "/v1/retails/percentage_parts_above_supply_cost?supply_cost=500",
    "/v1/retails/top_suppliers_by_balance?limit=10",
    "/v1/retails/customer_count_by_order_priority_and_balance?min_balance=0&order_priority=1-URGENT",
    "/v1/retails/customer_details_above_avg_balance",
    "/v1/retails/part_keys_by_supplier_and_supply_cost?supplier_name=Supplier%23000000654&limit=5",
    "/v1/retails/percentage_customers_by_market_segment_and_nation?market_segment=AUTOMOBILE&nation_name=FRANCE",
    "/v1/retails/part_name_highest_value?limit=1",
    "/v1/retails/nation_names_above_avg_customer_count",
    "/v1/retails/percentage_customers_market_segment_region?region_name=AFRICA&market_segment=HOUSEHOLD",
    "/v1/retails/part_names_by_available_quantity?limit=10",
    "/v1/retails/avg_retail_price_difference_ship_modes?ship_mode_1=SHIP&ship_mode_2=AIR",
    "/v1/retails/avg_discount_by_manufacturer?manufacturer=Manufacturer#5",
    "/v1/retails/count_parts_ship_mode_order_priority?ship_mode=RAIL&order_priority=3-MEDIUM",
    "/v1/retails/percentage_suppliers_below_avg_acctbal_region?region_name=EUROPE",
    "/v1/retails/avg_monthly_shipments_difference_years?year_1=1995&year_2=1996&order_priority=5-LOW&ship_mode=TRUCK",
    "/v1/retails/customer_keys_below_acctbal?account_balance=0",
    "/v1/retails/lineitem_orderkeys_by_extendedprice?limit=3",
    "/v1/retails/lineitem_count_by_quantity_returnflag?quantity=10&return_flag=R",
    "/v1/retails/lineitem_totalprice_by_shipmode_shipinstruct?ship_mode=AIR&ship_instruct=NONE",
    "/v1/retails/orders_count_by_orderpriority?order_priority=1-URGENT&limit=1",
    "/v1/retails/supplier_count_by_accountbalance_nation?account_balance=0&nation_name=EGYPT",
    "/v1/retails/orders_count_by_shipmode_orderdate?ship_mode=REG%20AIR&order_date=1995-03-22",
    "/v1/retails/market_segment_by_order_count?order_date_pattern=1994-04-%&limit=1",
    "/v1/retails/part_names_by_supplier?supplier_name=Supplier%23000000034",
    "/v1/retails/supply_costs_by_part_type?part_type=LARGE%20BURNISHED%20COPPER",
    "/v1/retails/part_name_avail_quantity?avail_quantity=10",
    "/v1/retails/order_clerk_count_address?address=uFTe2u518et8Q8UC",
    "/v1/retails/part_name_discount?discount=0.0000",
    "/v1/retails/line_item_count_order_ship_date?order_date=1994-09-21&ship_date=1994-11-19",
    "/v1/retails/average_profit_part_type?part_type=PROMO%20BRUSHED%20STEEL",
    "/v1/retails/customer_percentage_nation_segment?nation_name=IRAN&market_segment=HOUSEHOLD",
    "/v1/retails/customer_details?custkey=3",
    "/v1/retails/line_numbers_discount?discount=0.1",
    "/v1/retails/shipmode_comparison_result?shipmode1=RAIL&shipmode2=MAIL&result1=RAIL&result2=MAIL&shipinstruct=DELIVER%20IN%20PERSON",
    "/v1/retails/customer_names_acctbal_limit?acctbal=0&limit=3",
    "/v1/retails/discounted_extended_price?linenumber=1",
    "/v1/retails/return_flag_difference?returnflag1=A&returnflag2=N&extendedprice=16947.7",
    "/v1/retails/nation_names_by_region_name?region_name=Africa",
    "/v1/retails/region_with_most_nations",
    "/v1/retails/customer_name_by_order_total_price_and_customer_key?total_price=191918.92&customer_key=93697",
    "/v1/retails/nation_and_region_by_customer_name?customer_name=Customer%23000000008",
    "/v1/retails/days_difference_and_clerk_by_order_key?order_key=6",
    "/v1/retails/order_dates_by_order_priority?order_priority=1-URGENT",
    "/v1/retails/line_item_count_by_ship_instruction?ship_instruction=DELIVER%20IN%20PERSON",
    "/v1/retails/max_supplier_account_balance",
    "/v1/retails/supply_costs_by_part_name?part_name=violet%20olive%20rose%20ivory%20sandy",
    "/v1/retails/customer_phones_by_nation?nation_name=Ethiopia",
    "/v1/retails/total_order_price_by_customer_phone?customer_phone=627-220-3983",
    "/v1/retails/distinct_ship_modes_by_order_date?order_date=1994-12-31",
    "/v1/retails/supplier_with_most_supplied_parts",
    "/v1/retails/nation_names_by_supplier_account_balance?account_balance=4393.04",
    "/v1/retails/region_with_most_customers",
    "/v1/retails/customer_phones_by_min_order_total?min_order_total=300000",
    "/v1/retails/order_clerk_by_ship_mode?ship_mode=MAIL",
    "/v1/retails/nation_name_highest_account_balance",
    "/v1/retails/supplier_address_largest_part",
    "/v1/retails/part_supplier_highest_profit",
    "/v1/retails/percentage_suppliers_in_region?region_name=ASIA",
    "/v1/retails/order_total_price?order_key=32",
    "/v1/retails/count_line_items_by_discount?discount=0",
    "/v1/retails/line_numbers_by_year_ship_mode?year=1997&ship_mode=truck",
    "/v1/retails/line_number_highest_quantity_by_ship_mode?ship_mode=AIR",
    "/v1/retails/customer_names_by_acctbal?max_acctbal=0",
    "/v1/retails/customer_count_by_segment_nation?market_segment=HOUSEHOLD&nation_name=GERMANY",
    "/v1/retails/customer_phones_by_order_priority?order_priority=1-URGENT",
    "/v1/retails/customer_name_highest_discount",
    "/v1/retails/customer_names_by_order_total_price?min_total_price=300000",
    "/v1/retails/customer_names_by_acctbal_nation?min_acctbal=5000&nation_name=INDIA",
    "/v1/retails/supplier_phones_by_nation?nation_name=JAPAN",
    "/v1/retails/supplier_names_by_acctbal_nation?max_acctbal=0&nation_name=ARGENTINA",
    "/v1/retails/count_regions_by_nation_name?nation_name=ALGERIA",
    "/v1/retails/customer_names_by_discount_and_order_date?discount=0.1&start_year=1994&end_year=1995",
    "/v1/retails/percentage_customers_in_market_segment_and_nation?market_segment=HOUSEHOLD&nation_name=INDONESIA",
    "/v1/retails/part_comments_by_name?part_name=burlywood%20plum%20powder%20puff%20mint",
    "/v1/retails/count_parts_by_retail_price?retail_price=1900",
    "/v1/retails/count_parts_by_type_and_manufacturer?part_type=PROMO%20BRUSHED%20STEEL&manufacturer=Manufacturer%235",
    "/v1/retails/part_brands_by_type?part_type=PROMO%20BRUSHED%20STEEL",
    "/v1/retails/part_names_with_highest_retail_price",
    "/v1/retails/largest_part_name?part_name1=pink%20powder%20drab%20lawn%20cyan&part_name2=cornflower%20sky%20burlywood%20green%20beige",
    "/v1/retails/count_parts_by_container?container_type=JUMBO%20CASE",
    "/v1/retails/min_part_size_by_container?container_type=JUMBO%20CASE",
    "/v1/retails/count_suppliers_by_account_balance?account_balance=0",
    "/v1/retails/top_suppliers_by_account_balance?limit=3",
    "/v1/retails/min_supply_cost_by_part?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/supplier_with_lowest_supply_cost?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/total_available_quantity_by_part?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/supplier_phone_by_part_name_availqty?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/supplier_phone_by_part_name_size?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/count_suppliers_by_part_name_nation?part_name=hot%20spring%20dodger%20dim%20light&nation_name=VIETNAM",
    "/v1/retails/count_suppliers_by_acctbal_part_type?acctbal=0&part_type=PROMO%20BRUSHED%20STEEL",
    "/v1/retails/supplier_names_by_part_brand?part_brand=Brand%2355",
    "/v1/retails/sum_supplier_counts_by_part_type_availqty?part_type=PROMO%20BRUSHED%20STEEL&availqty=5000",
    "/v1/retails/count_part_keys_by_part_name?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/sum_part_keys_by_part_name?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/order_keys_by_part_container_count?part_container=JUMBO%20CASE&count=2",
    "/v1/retails/count_nation_keys_by_region_acctbal?region_name=EUROPE&acctbal=0",
    "/v1/retails/count_regions_by_name?region_name=EUROPE",
    "/v1/retails/supplier_phone_by_order_key?order_key=1",
    "/v1/retails/count_line_items_by_order_key_and_account_balance?order_key=4&account_balance=0",
    "/v1/retails/count_line_items_by_return_flag_and_account_balance?return_flag=R&account_balance=0",
    "/v1/retails/ship_date_by_order_key_and_part_name?order_key=1&part_name=burnished%20seashell%20gainsboro%20navajo%20chocolate",
    "/v1/retails/quantity_by_order_key_and_part_name?order_key=1&part_name=burnished%20seashell%20gainsboro%20navajo%20chocolate",
    "/v1/retails/part_name_with_highest_quantity?part_name_1=salmon%20white%20grey%20tan%20navy&part_name_2=burnished%20seashell%20gainsboro%20navajo%20chocolate",
    "/v1/retails/max_discount_by_part_name?part_name=burnished%20seashell%20gainsboro%20navajo%20chocolate",
    "/v1/retails/distinct_ship_modes_by_part_name?part_name=burnished%20seashell%20gainsboro%20navajo%20chocolate",
    "/v1/retails/average_supply_cost_by_part_name?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/supply_cost_percentage_difference?part_name=hot%20spring%20dodger%20dim%20light",
    "/v1/retails/profit_by_order_and_part?order_key=1&part_key=98768",
    "/v1/retails/min_account_balance_market_segment",
    "/v1/retails/count_line_items_shipped?year=1997&ship_mode=MAIL",
    "/v1/retails/count_line_items_shipped_with_return_flag?year=1994&return_flag=R&ship_mode=TRUCK",
    "/v1/retails/count_customers_by_account_balance_and_market_segment?account_balance=0&market_segment=MACHINERY",
    "/v1/retails/count_orders_by_year_clerk_priority?year=1997&clerk=Clerk%23000000001&order_priority=1-URGENT",
    "/v1/retails/customer_with_longest_delay",
    "/v1/retails/customer_count_by_market_segment_and_total_price?market_segment=BUILDING&total_price=50000",
    "/v1/retails/nation_with_most_suppliers",
    "/v1/retails/top_nations_by_negative_account_balance",
    "/v1/retails/part_names_by_supply_cost?supply_cost=1000",
    "/v1/retails/nation_of_highest_supplier_key",
    "/v1/retails/clerk_of_highest_extended_price_order",
    "/v1/retails/total_quantity_by_order_date_and_customer?order_date=1995-10-05&cust_key=101660",
    "/v1/retails/total_revenue_by_customer_and_order_date?cust_key=88931&order_date=1994-07-13",
    "/v1/retails/part_names_by_customer_key?o_custkey=110942",
    "/v1/retails/extended_price_part_names_by_customer_order_key?o_custkey=111511&o_orderkey=53159"
]
