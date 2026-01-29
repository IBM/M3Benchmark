from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/regional_sales/regional_sales.sqlite')
cursor = conn.cursor()

# Endpoint to get the top region by the number of distinct sales teams
@app.get("/v1/regional_sales/top_region_by_sales_teams", operation_id="get_top_region_by_sales_teams", summary="Retrieves the top region with the highest number of distinct sales teams, based on the specified limit. This operation provides a ranked list of regions, with the region having the most unique sales teams at the top. The limit parameter allows you to control the number of regions returned in the response.")
async def get_top_region_by_sales_teams(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT Region FROM `Sales Team` GROUP BY Region ORDER BY COUNT(DISTINCT `Sales Team`) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"region": []}
    return {"region": result[0]}

# Endpoint to get customer names containing a specific substring
@app.get("/v1/regional_sales/customer_names_containing_substring", operation_id="get_customer_names_containing_substring", summary="Retrieves a list of customer names that contain a specified substring. The operation filters the 'Customers' table to find names that match the provided substring, returning only those that contain the substring.")
async def get_customer_names_containing_substring(substring: str = Query(..., description="Substring to search for in customer names")):
    cursor.execute("SELECT T FROM ( SELECT IIF(`Customer Names` LIKE ?, `Customer Names`, NULL) AS T FROM Customers ) WHERE T IS NOT NULL", ('%' + substring + '%',))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the average median income for a specific type of store location
@app.get("/v1/regional_sales/average_median_income_by_type", operation_id="get_average_median_income_by_type", summary="Retrieves the average median income for a specified type of store location. The operation calculates the average median income based on the provided store location type, offering insights into the financial demographics of the area.")
async def get_average_median_income_by_type(type: str = Query(..., description="Type of store location")):
    cursor.execute("SELECT AVG(`Median Income`) FROM `Store Locations` WHERE Type = ?", (type,))
    result = cursor.fetchone()
    if not result:
        return {"average_median_income": []}
    return {"average_median_income": result[0]}

# Endpoint to get the sales team and region for a specific order number
@app.get("/v1/regional_sales/sales_team_region_by_order_number", operation_id="get_sales_team_region_by_order_number", summary="Retrieves the sales team and region associated with a specific order number. The order number is used to look up the corresponding sales team and region in the sales orders and sales team tables.")
async def get_sales_team_region_by_order_number(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT T2.`Sales Team`, T2.Region FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T1.OrderNumber = ?", (order_number,))
    result = cursor.fetchone()
    if not result:
        return {"sales_team": [], "region": []}
    return {"sales_team": result[0], "region": result[1]}

# Endpoint to get distinct products for a specific sales team
@app.get("/v1/regional_sales/distinct_products_by_sales_team", operation_id="get_distinct_products_by_sales_team", summary="Retrieve a unique list of products associated with a specific sales team. This operation fetches product details from the Products table and matches them with corresponding sales orders in the Sales Orders table. It then filters the results based on the provided sales team, ensuring that only products linked to that team are returned.")
async def get_distinct_products_by_sales_team(sales_team: str = Query(..., description="Sales team")):
    cursor.execute("SELECT DISTINCT T1.ProductID, T1.`Product Name` FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID INNER JOIN `Sales Team` AS T3 ON T3.SalesTeamID = T2._SalesTeamID WHERE T3.`Sales Team` = ?", (sales_team,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [{"product_id": row[0], "product_name": row[1]} for row in result]}

# Endpoint to get distinct customer names for a specific product, order date suffix, and maximum discount applied
@app.get("/v1/regional_sales/distinct_customer_names_by_product_order_date_discount", operation_id="get_distinct_customer_names_by_product_order_date_discount", summary="Retrieves a unique list of customer names who have purchased a specific product, based on the last two digits of the order date and the maximum discount applied for that product in the given month. This operation is useful for identifying customers who have made purchases under specific conditions, such as during a promotional period or for a particular product.")
async def get_distinct_customer_names_by_product_order_date_discount(product_name: str = Query(..., description="Product name"), order_date_suffix: str = Query(..., description="Order date suffix (last two characters)"), discount_applied: float = Query(..., description="Maximum discount applied")):
    cursor.execute("SELECT DISTINCT T1.`Customer Names` FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID INNER JOIN Products AS T3 ON T3.ProductID = T2._ProductID WHERE T3.`Product Name` = ? AND SUBSTR(T2.OrderDate, -2) = ? AND T2.`Discount Applied` = ( SELECT T2.`Discount Applied` FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID INNER JOIN Products AS T3 ON T3.ProductID = T2._ProductID WHERE T3.`Product Name` = ? AND T2.OrderDate LIKE ? ORDER BY T2.`Discount Applied` DESC LIMIT 1 )", (product_name, order_date_suffix, product_name, '%/%/' + order_date_suffix))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get distinct order numbers and city names for a specific sales channel
@app.get("/v1/regional_sales/distinct_order_numbers_city_names_by_sales_channel", operation_id="get_distinct_order_numbers_city_names_by_sales_channel", summary="Retrieves unique order numbers and corresponding city names for a specified sales channel. This operation provides a distinct list of orders and their associated city locations, enabling analysis of sales distribution across different regions for a given sales channel.")
async def get_distinct_order_numbers_city_names_by_sales_channel(sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT DISTINCT T1.OrderNumber, T2.`City Name` FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T1.`Sales Channel` = ?", (sales_channel,))
    result = cursor.fetchall()
    if not result:
        return {"order_numbers_city_names": []}
    return {"order_numbers_city_names": [{"order_number": row[0], "city_name": row[1]} for row in result]}

# Endpoint to get order details with the highest unit cost
@app.get("/v1/regional_sales/order_details_highest_unit_cost", operation_id="get_order_details_highest_unit_cost", summary="Retrieves the order details with the highest unit cost, up to a specified limit. The operation returns the order number, customer name, and order date for the orders with the highest unit cost. The limit parameter allows you to control the number of results returned.")
async def get_order_details_highest_unit_cost(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T2.OrderNumber, T1.`Customer Names`, T2.OrderDate FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID INNER JOIN Products AS T3 ON T3.ProductID = T2._ProductID ORDER BY T2.`Unit Cost` DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"order_details": []}
    return {"order_details": [{"order_number": row[0], "customer_name": row[1], "order_date": row[2]} for row in result]}

# Endpoint to get distinct order numbers for a specific order date pattern and customer name
@app.get("/v1/regional_sales/distinct_order_numbers_by_date_pattern_customer", operation_id="get_distinct_order_numbers_by_date_pattern_customer", summary="Retrieves a list of unique order numbers for a given order date pattern and customer name. This operation filters sales orders based on the provided date pattern and customer name, and returns only the distinct order numbers that match the criteria.")
async def get_distinct_order_numbers_by_date_pattern_customer(order_date_pattern: str = Query(..., description="Order date pattern (e.g., '%/%/18')"), customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT DISTINCT T FROM ( SELECT CASE  WHEN T1.OrderDate LIKE ? AND T2.`Customer Names` = ? THEN T1.OrderNumber ELSE NULL END AS T FROM `Sales Orders` T1 INNER JOIN Customers T2 ON T2.CustomerID = T1._CustomerID ) WHERE T IS NOT NULL", (order_date_pattern, customer_name))
    result = cursor.fetchall()
    if not result:
        return {"order_numbers": []}
    return {"order_numbers": [row[0] for row in result]}

# Endpoint to get distinct product names and sales teams for a specific warehouse code
@app.get("/v1/regional_sales/distinct_product_names_sales_teams_by_warehouse_code", operation_id="get_distinct_product_names_sales_teams_by_warehouse_code", summary="Retrieve a unique list of product names and their associated sales teams for a specified warehouse. The warehouse is identified by its code.")
async def get_distinct_product_names_sales_teams_by_warehouse_code(warehouse_code: str = Query(..., description="Warehouse code")):
    cursor.execute("SELECT DISTINCT T1.`Product Name`, T3.`Sales Team` FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID INNER JOIN `Sales Team` AS T3 ON T3.SalesTeamID = T2._SalesTeamID WHERE T2.WarehouseCode = ?", (warehouse_code,))
    result = cursor.fetchall()
    if not result:
        return {"product_names_sales_teams": []}
    return {"product_names_sales_teams": [{"product_name": row[0], "sales_team": row[1]} for row in result]}

# Endpoint to get customer names based on sales channel
@app.get("/v1/regional_sales/customer_names_by_sales_channel", operation_id="get_customer_names_by_sales_channel", summary="Retrieves a list of customer names associated with the specified sales channel. The operation filters customers based on the provided sales channel and returns only those with a valid match. The sales channel parameter is a string that represents the sales channel category (e.g., 'Online').")
async def get_customer_names_by_sales_channel(sales_channel: str = Query(..., description="Sales channel (e.g., 'Online')")):
    cursor.execute("SELECT T FROM ( SELECT CASE  WHEN T2.`Sales Channel` = ? THEN T1.`Customer Names` ELSE NULL END AS T FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID ) WHERE T IS NOT NULL", (sales_channel,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the average profit margin for a specific product
@app.get("/v1/regional_sales/average_profit_margin_by_product", operation_id="get_average_profit_margin_by_product", summary="Retrieves the average profit margin for a specific product by calculating the difference between the average unit price and the average unit cost. The product is identified by its name, which is provided as an input parameter.")
async def get_average_profit_margin_by_product(product_name: str = Query(..., description="Product name (e.g., 'Bakeware')")):
    cursor.execute("SELECT AVG(REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', ''))  FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T2.`Product Name` = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_profit_margin": []}
    return {"average_profit_margin": result[0]}

# Endpoint to get the top sales team by profit margin for a specific year
@app.get("/v1/regional_sales/top_sales_team_by_profit_margin", operation_id="get_top_sales_team_by_profit_margin", summary="Retrieves the top-performing sales team based on the highest profit margin for a specified year. The profit margin is calculated as the difference between the unit price and unit cost of sales orders. The year is determined by the provided order date pattern. The result is limited to the top team.")
async def get_top_sales_team_by_profit_margin(order_date: str = Query(..., description="Order date pattern (e.g., '%/%/20' for the year 2020)")):
    cursor.execute("SELECT T2.`Sales Team` FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T1.OrderDate LIKE ? GROUP BY T2.`Sales Team` ORDER BY SUM(REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '')) DESC LIMIT 1", (order_date,))
    result = cursor.fetchone()
    if not result:
        return {"top_sales_team": []}
    return {"top_sales_team": result[0]}

# Endpoint to get order numbers and profit margins for a specific sales team
@app.get("/v1/regional_sales/order_profit_margins_by_sales_team", operation_id="get_order_profit_margins_by_sales_team", summary="Retrieves the order numbers and corresponding profit margins for a specific sales team. The sales team is identified by its name. The profit margin is calculated as the difference between the unit price and unit cost of each order.")
async def get_order_profit_margins_by_sales_team(sales_team: str = Query(..., description="Sales team name (e.g., 'Joshua Bennett')")):
    cursor.execute("SELECT T1.OrderNumber , REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '')  FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T2.`Sales Team` = ?", (sales_team,))
    result = cursor.fetchall()
    if not result:
        return {"order_profit_margins": []}
    return {"order_profit_margins": [{"order_number": row[0], "profit_margin": row[1]} for row in result]}

# Endpoint to get the percentage of orders for a specific product shipped in a specific month
@app.get("/v1/regional_sales/percentage_orders_by_product_shipped", operation_id="get_percentage_orders_by_product_shipped", summary="Retrieves the percentage of orders for a specific product that were shipped in a given month. The calculation is based on the total number of orders for that product in the specified month. The product is identified by its name, and the month is specified using a date pattern.")
async def get_percentage_orders_by_product_shipped(product_name: str = Query(..., description="Product name (e.g., 'Home Fragrances')"), ship_date: str = Query(..., description="Ship date pattern (e.g., '7/%/18' for July 2018)")):
    cursor.execute("SELECT SUM(CASE WHEN T2.`Product Name` = ? THEN 1 ELSE 0 END) * 100 / COUNT(T1.OrderNumber)  FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T1.ShipDate LIKE ?", (product_name, ship_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage_orders": []}
    return {"percentage_orders": result[0]}

# Endpoint to get distinct customer IDs and names based on a name pattern
@app.get("/v1/regional_sales/distinct_customers_by_name_pattern", operation_id="get_distinct_customers_by_name_pattern", summary="Retrieves a list of unique customers, identified by their IDs and names, that match a specified name pattern. The results are sorted in descending order by customer name.")
async def get_distinct_customers_by_name_pattern(name_pattern: str = Query(..., description="Name pattern (e.g., 'W%')")):
    cursor.execute("SELECT DISTINCT CustomerID, `Customer Names` FROM Customers WHERE `Customer Names` LIKE ? ORDER BY `Customer Names` DESC", (name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"customer_id": row[0], "customer_name": row[1]} for row in result]}

# Endpoint to get product IDs and names based on a product name pattern
@app.get("/v1/regional_sales/product_ids_by_name_pattern", operation_id="get_product_ids_by_name_pattern", summary="Retrieves product IDs and names that match a specified name pattern. The search is case-insensitive and supports wildcard characters. The results are ordered by name in descending order.")
async def get_product_ids_by_name_pattern(name_pattern: str = Query(..., description="Product name pattern (e.g., '%Outdoor%')")):
    cursor.execute("SELECT ProductID, T FROM ( SELECT ProductID , CASE  WHEN `Product Name` LIKE ? THEN `Product Name` ELSE NULL END AS T FROM Products ) WHERE T IS NOT NULL ORDER BY T DESC", (name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [{"product_id": row[0], "product_name": row[1]} for row in result]}

# Endpoint to get the count of distinct orders based on sales channel, warehouse code, and discount applied
@app.get("/v1/regional_sales/count_distinct_orders", operation_id="get_count_distinct_orders", summary="Retrieves the count of unique orders that match the specified sales channel, warehouse code, and discount applied. This operation filters the sales orders based on the provided parameters and returns the count of distinct orders that meet the criteria.")
async def get_count_distinct_orders(sales_channel: str = Query(..., description="Sales channel (e.g., 'In-Store')"), warehouse_code: str = Query(..., description="Warehouse code (e.g., 'WARE-NMK1003')"), discount_applied: str = Query(..., description="Discount applied (e.g., '0.4')")):
    cursor.execute("SELECT COUNT(DISTINCT T) FROM ( SELECT CASE  WHEN `Sales Channel` = ? AND WarehouseCode = ? AND `Discount Applied` = ? THEN OrderNumber ELSE NULL END AS T FROM `Sales Orders` ) WHERE T IS NOT NULL", (sales_channel, warehouse_code, discount_applied))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city with the highest population in a specific state
@app.get("/v1/regional_sales/highest_population_city_by_state", operation_id="get_highest_population_city_by_state", summary="Retrieves the city with the highest population and its median income from a specific state. The operation filters store locations by the provided state name and returns the city with the highest population along with its median income.")
async def get_highest_population_city_by_state(state: str = Query(..., description="State name (e.g., 'Florida')")):
    cursor.execute("SELECT `City Name`, `Median Income` FROM `Store Locations` WHERE State = ? ORDER BY Population DESC LIMIT 1", (state,))
    result = cursor.fetchone()
    if not result:
        return {"city": []}
    return {"city": {"city_name": result[0], "median_income": result[1]}}

# Endpoint to get distinct store IDs, city names, and regions based on county
@app.get("/v1/regional_sales/distinct_stores_by_county", operation_id="get_distinct_stores_by_county", summary="Retrieves unique store identifiers, city names, and corresponding regions for a given county. This operation allows you to obtain a list of distinct stores within a specific county, along with their respective city and regional information.")
async def get_distinct_stores_by_county(county: str = Query(..., description="County name (e.g., 'Allen County')")):
    cursor.execute("SELECT DISTINCT T2.StoreID, T2.`City Name`, T1.Region FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode WHERE T2.County = ?", (county,))
    result = cursor.fetchall()
    if not result:
        return {"stores": []}
    return {"stores": [{"store_id": row[0], "city_name": row[1], "region": row[2]} for row in result]}

# Endpoint to get distinct store locations based on type
@app.get("/v1/regional_sales/distinct_store_locations_by_type", operation_id="get_distinct_store_locations", summary="Retrieves a list of distinct store locations based on the provided store types. The operation filters store locations by the specified types and returns their unique identifiers, city names, and corresponding states.")
async def get_distinct_store_locations(type1: str = Query(..., description="First type of store location"), type2: str = Query(..., description="Second type of store location")):
    cursor.execute("SELECT DISTINCT T2.StoreID, T2.`City Name`, T1.State, T2.Type FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode WHERE T2.Type = ? OR T2.Type = ?", (type1, type2))
    result = cursor.fetchall()
    if not result:
        return {"store_locations": []}
    return {"store_locations": result}

# Endpoint to get sales team information based on sales team ID and sales channels
@app.get("/v1/regional_sales/sales_team_info_by_id_and_channels", operation_id="get_sales_team_info", summary="Get sales team information based on sales team ID and sales channels")
async def get_sales_team_info(sales_team_id: int = Query(..., description="Sales team ID"), sales_channel1: str = Query(..., description="First sales channel"), sales_channel2: str = Query(..., description="Second sales channel")):
    cursor.execute("SELECT T2.Region, T2.`Sales Team` FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T2.SalesTeamID = ? AND (T1.`Sales Channel` = ? OR T1.`Sales Channel` = ?)", (sales_team_id, sales_channel1, sales_channel2))
    result = cursor.fetchall()
    if not result:
        return {"sales_team_info": []}
    return {"sales_team_info": result}

# Endpoint to get the percentage of in-store sales for a specific customer
@app.get("/v1/regional_sales/percentage_in_store_sales_by_customer", operation_id="get_percentage_in_store_sales", summary="Get the percentage of in-store sales for a specific customer")
async def get_percentage_in_store_sales(sales_channel: str = Query(..., description="Sales channel"), customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.`Sales Channel` = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1._CustomerID) FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID WHERE T2.`Customer Names` = ?", (sales_channel, customer_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get customer names and delivery dates based on sales channel, product name, and order date pattern
@app.get("/v1/regional_sales/customer_delivery_dates_by_channel_product_order_date", operation_id="get_customer_delivery_dates", summary="Retrieves the names of customers and their respective delivery dates based on the specified sales channel, product name, and order date pattern. This operation filters sales orders by the provided sales channel and product name, and matches the order date to the given pattern. The result is a list of customer names and their corresponding delivery dates.")
async def get_customer_delivery_dates(sales_channel: str = Query(..., description="Sales channel"), product_name: str = Query(..., description="Product name"), order_date_pattern: str = Query(..., description="Order date pattern (e.g., '%/%/19')")):
    cursor.execute("SELECT T1.`Customer Names`, T2.DeliveryDate FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID INNER JOIN Products AS T3 ON T3.ProductID = T2._ProductID WHERE T2.`Sales Channel` = ? AND T3.`Product Name` = ? AND T2.OrderDate LIKE ?", (sales_channel, product_name, order_date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"customer_delivery_dates": []}
    return {"customer_delivery_dates": result}

# Endpoint to get distinct customer names and product names based on profit margin
@app.get("/v1/regional_sales/distinct_customer_product_by_profit_margin", operation_id="get_distinct_customer_product", summary="Retrieves a list of unique customer and product combinations that have a profit margin greater than the specified value. The profit margin is calculated as the difference between the unit price and the unit cost.")
async def get_distinct_customer_product(profit_margin: float = Query(..., description="Profit margin")):
    cursor.execute("SELECT DISTINCT `Customer Names`, `Product Name` FROM ( SELECT T1.`Customer Names`, T3.`Product Name` , REPLACE(T2.`Unit Price`, ',', '') - REPLACE(T2.`Unit Cost`, ',', '') AS T FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID INNER JOIN Products T3 ON T3.ProductID = T2._ProductID ) WHERE T > ?", (profit_margin,))
    result = cursor.fetchall()
    if not result:
        return {"customer_product": []}
    return {"customer_product": result}

# Endpoint to get distinct city names based on state and water area
@app.get("/v1/regional_sales/distinct_city_names_by_state_water_area", operation_id="get_distinct_city_names", summary="Retrieves a list of unique city names within a specified state that have a particular water area. The state and water area are provided as input parameters.")
async def get_distinct_city_names(state: str = Query(..., description="State"), water_area: str = Query(..., description="Water area")):
    cursor.execute("SELECT DISTINCT T2.`City Name` FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode WHERE T2.State = ? AND T2.`Water Area` = ?", (state, water_area))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": result}

# Endpoint to get the percentage of orders by a specific sales team
@app.get("/v1/regional_sales/percentage_orders_by_sales_team", operation_id="get_percentage_orders_by_sales_team", summary="Retrieves the percentage of total orders attributed to a specific sales team. The operation calculates this percentage by summing the orders associated with the specified sales team and dividing it by the total number of orders. The sales team is identified by its unique name.")
async def get_percentage_orders_by_sales_team(sales_team: str = Query(..., description="Sales team")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.`Sales Team` = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID", (sales_team,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of specific products in sales orders
@app.get("/v1/regional_sales/count_specific_products", operation_id="get_count_specific_products", summary="Retrieves the total count of two specific products in sales orders. The operation calculates the total number of occurrences for each product in the sales orders, providing a quantitative measure of their presence in the sales data.")
async def get_count_specific_products(product_name1: str = Query(..., description="First product name"), product_name2: str = Query(..., description="Second product name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.`Product Name` = ? THEN 1 ELSE 0 END) AS num1 , SUM(CASE WHEN T2.`Product Name` = ? THEN 1 ELSE 0 END) AS num2 FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID", (product_name1, product_name2))
    result = cursor.fetchone()
    if not result:
        return {"counts": []}
    return {"counts": {"num1": result[0], "num2": result[1]}}

# Endpoint to get the total profit margin for the store location with the highest median income
@app.get("/v1/regional_sales/total_profit_margin_highest_median_income", operation_id="get_total_profit_margin_highest_median_income", summary="Retrieves the total profit margin for the store location with the highest median income. This operation calculates the profit margin by subtracting the unit cost from the unit price for each sale, then summing these values for the store with the highest median income.")
async def get_total_profit_margin_highest_median_income():
    cursor.execute("SELECT SUM(REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '')) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID ORDER BY T2.`Median Income` DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"total_profit_margin": []}
    return {"total_profit_margin": result[0]}

# Endpoint to get the count of orders by sales team IDs in a specific region
@app.get("/v1/regional_sales/count_orders_by_sales_team_region", operation_id="get_count_orders_by_sales_team_region", summary="Retrieve the total number of orders for each sales team within a specified region, where the sales team ID falls within a given range. The operation filters sales teams by their region and IDs, then aggregates the order count for each team.")
async def get_count_orders_by_sales_team_region(region: str = Query(..., description="Region"), min_sales_team_id: int = Query(..., description="Minimum sales team ID"), max_sales_team_id: int = Query(..., description="Maximum sales team ID")):
    cursor.execute("SELECT COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T2.Region = ? AND T2.SalesTeamID BETWEEN ? AND ? GROUP BY T2.SalesTeamID HAVING COUNT(T1.OrderNumber)", (region, min_sales_team_id, max_sales_team_id))
    result = cursor.fetchall()
    if not result:
        return {"order_counts": []}
    return {"order_counts": result}

# Endpoint to get the sum of orders on a specific date
@app.get("/v1/regional_sales/sum_orders_by_date", operation_id="get_sum_orders_by_date", summary="Retrieves the total number of orders placed on a specific date. The date must be provided in 'MM/DD/YY' format.")
async def get_sum_orders_by_date(order_date: str = Query(..., description="Order date in 'MM/DD/YY' format")):
    cursor.execute("SELECT SUM(IIF(OrderDate = ?, 1, 0)) FROM `Sales Orders`", (order_date,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct order numbers based on delivery date
@app.get("/v1/regional_sales/distinct_order_numbers_by_delivery_date", operation_id="get_distinct_order_numbers_by_delivery_date", summary="Retrieve a unique set of order numbers associated with a given delivery date. The delivery date must be provided in the 'MM/DD/YY' format.")
async def get_distinct_order_numbers_by_delivery_date(delivery_date: str = Query(..., description="Delivery date in 'MM/DD/YY' format")):
    cursor.execute("SELECT DISTINCT T FROM ( SELECT IIF(DeliveryDate = ?, OrderNumber, NULL) AS T FROM `Sales Orders` ) WHERE T IS NOT NULL", (delivery_date,))
    result = cursor.fetchall()
    if not result:
        return {"order_numbers": []}
    return {"order_numbers": [row[0] for row in result]}

# Endpoint to get the sum of orders with quantity greater than a specified value
@app.get("/v1/regional_sales/sum_orders_by_quantity", operation_id="get_sum_orders_by_quantity", summary="Retrieves the total count of orders with a quantity greater than the specified value. The input parameter determines the minimum order quantity for inclusion in the count.")
async def get_sum_orders_by_quantity(order_quantity: int = Query(..., description="Order quantity")):
    cursor.execute("SELECT SUM(IIF(`Order Quantity` > ?, 1, 0)) FROM `Sales Orders`", (order_quantity,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct states based on state code
@app.get("/v1/regional_sales/distinct_states_by_state_code", operation_id="get_distinct_states_by_state_code", summary="Retrieve a list of distinct states that match the provided state code. The operation filters the 'Regions' table to find states with the specified state code and returns them as a list.")
async def get_distinct_states_by_state_code(state_code: str = Query(..., description="State code")):
    cursor.execute("SELECT T FROM ( SELECT IIF(StateCode = ?, State, NULL) AS T FROM Regions ) WHERE T IS NOT NULL", (state_code,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get the count of distinct state codes based on region
@app.get("/v1/regional_sales/count_distinct_state_codes_by_region", operation_id="get_count_distinct_state_codes_by_region", summary="Retrieves the count of unique state codes associated with a specified region. This operation allows you to determine the number of distinct states within a given region, providing insights into the regional distribution of sales data.")
async def get_count_distinct_state_codes_by_region(region: str = Query(..., description="Region")):
    cursor.execute("SELECT COUNT(DISTINCT T) FROM ( SELECT CASE  WHEN Region = ? THEN StateCode ELSE NULL END AS T FROM Regions ) WHERE T IS NOT NULL", (region,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct product names based on unit cost
@app.get("/v1/regional_sales/distinct_product_names_by_unit_cost", operation_id="get_distinct_product_names_by_unit_cost", summary="Retrieves a list of unique product names that have a specified unit cost. The operation filters the sales orders based on the provided unit cost and returns the distinct product names associated with that cost.")
async def get_distinct_product_names_by_unit_cost(unit_cost: float = Query(..., description="Unit cost")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT IIF(T1.`Unit Cost` = ?, T2.`Product Name`, NULL) AS T FROM `Sales Orders` T1 INNER JOIN Products T2 ON T2.ProductID = T1._ProductID ) WHERE T IS NOT NULL", (unit_cost,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get distinct delivery dates based on product name
@app.get("/v1/regional_sales/distinct_delivery_dates_by_product_name", operation_id="get_distinct_delivery_dates_by_product_name", summary="Retrieves a list of unique delivery dates associated with a specific product. The product is identified by its name, which is provided as an input parameter.")
async def get_distinct_delivery_dates_by_product_name(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT IIF(T2.`Product Name` = ?, T1.DeliveryDate, NULL) AS T FROM `Sales Orders` T1 INNER JOIN Products T2 ON T2.ProductID = T1._ProductID ) WHERE T IS NOT NULL", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"delivery_dates": []}
    return {"delivery_dates": [row[0] for row in result]}

# Endpoint to get the sum of orders based on order date pattern and product name
@app.get("/v1/regional_sales/sum_orders_by_date_pattern_and_product_name", operation_id="get_sum_orders_by_date_pattern_and_product_name", summary="Retrieves the total number of orders that match a specified date pattern and product name. The date pattern should be provided in the format 'MM/DD/YY'. The product name should be the exact name of the product as it appears in the system.")
async def get_sum_orders_by_date_pattern_and_product_name(order_date_pattern: str = Query(..., description="Order date pattern in '%/%/YY' format"), product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.OrderDate LIKE ? AND T2.`Product Name` = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID", (order_date_pattern, product_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct product names based on discount applied
@app.get("/v1/regional_sales/distinct_product_names_by_discount", operation_id="get_distinct_product_names_by_discount", summary="Retrieves a list of unique product names for which a specific discount has been applied. The operation filters sales orders based on the provided discount and extracts the corresponding product names from the Products table. Only non-null product names are returned.")
async def get_distinct_product_names_by_discount(discount_applied: float = Query(..., description="Discount applied")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT IIF(T1.`Discount Applied` = ?, T2.`Product Name`, NULL) AS T FROM `Sales Orders` T1 INNER JOIN Products T2 ON T2.ProductID = T1._ProductID ) WHERE T IS NOT NULL", (discount_applied,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the average profit margin for a specific product and sales channel
@app.get("/v1/regional_sales/average_profit_margin", operation_id="get_average_profit_margin", summary="Retrieves the average profit margin for a specific product sold through a particular sales channel. The calculation is based on the difference between the unit price and unit cost of each sale, divided by the total number of sales for the specified product and sales channel.")
async def get_average_profit_margin(product_name: str = Query(..., description="Product name"), sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT SUM(REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '')) / COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T2.`Product Name` = ? AND T1.`Sales Channel` = ?", (product_name, sales_channel))
    result = cursor.fetchone()
    if not result:
        return {"average_profit_margin": []}
    return {"average_profit_margin": result[0]}

# Endpoint to get the average profit per order for a specific product and minimum order quantity
@app.get("/v1/regional_sales/average_profit_per_order", operation_id="get_average_profit_per_order", summary="Retrieves the average profit per order for a specific product, considering only orders with a quantity greater than a specified minimum. This calculation is based on the difference between the unit price and unit cost of each order.")
async def get_average_profit_per_order(product_name: str = Query(..., description="Name of the product"), min_order_quantity: int = Query(..., description="Minimum order quantity")):
    cursor.execute("SELECT SUM(REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '')) / COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T2.`Product Name` = ? AND T1.`Order Quantity` > ?", (product_name, min_order_quantity))
    result = cursor.fetchone()
    if not result:
        return {"average_profit": []}
    return {"average_profit": result[0]}

# Endpoint to get distinct city names in a specific region
@app.get("/v1/regional_sales/distinct_city_names_by_region", operation_id="get_distinct_city_names_by_region", summary="Retrieve a unique list of city names within a specified region. The operation filters store locations based on the provided region and returns the distinct city names.")
async def get_distinct_city_names_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T1.Region = ? THEN T2.`City Name` END AS T FROM Regions T1 INNER JOIN `Store Locations` T2 ON T2.StateCode = T1.StateCode ) WHERE T IS NOT NULL", (region,))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get distinct regions for a specific store location type
@app.get("/v1/regional_sales/distinct_regions_by_store_type", operation_id="get_distinct_regions_by_store_type", summary="Retrieves a list of unique regions where a specific store type is present. The operation filters regions based on the provided store type, ensuring that only regions with at least one store of the specified type are included in the response.")
async def get_distinct_regions_by_store_type(store_type: str = Query(..., description="Store location type")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.Type = ? THEN T1.Region END AS T FROM Regions T1 INNER JOIN `Store Locations` T2 ON T2.StateCode = T1.StateCode ) WHERE T IS NOT NULL", (store_type,))
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": [row[0] for row in result]}

# Endpoint to get the count of sales orders for a specific customer
@app.get("/v1/regional_sales/sales_order_count_by_customer", operation_id="get_sales_order_count_by_customer", summary="Retrieves the total number of sales orders associated with a specific customer. The customer is identified by their name, which is provided as an input parameter.")
async def get_sales_order_count_by_customer(customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.`Customer Names` = ? THEN 1 ELSE 0 END) FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get distinct discounts applied for a specific customer
@app.get("/v1/regional_sales/distinct_discounts_by_customer", operation_id="get_distinct_discounts_by_customer", summary="Retrieves a list of unique discounts that have been applied to a specific customer's orders. The customer is identified by their name, which is provided as an input parameter.")
async def get_distinct_discounts_by_customer(customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T1.`Customer Names` = ? THEN T2.`Discount Applied` END AS T FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID ) WHERE T IS NOT NULL", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"discounts": []}
    return {"discounts": [row[0] for row in result]}

# Endpoint to get distinct customer names for a specific ship date
@app.get("/v1/regional_sales/distinct_customer_names_by_ship_date", operation_id="get_distinct_customer_names_by_ship_date", summary="Retrieves a list of unique customer names associated with a specific ship date. The ship date must be provided in 'M/D/YY' format. This operation returns a comprehensive set of distinct customer names, excluding any null or duplicate entries.")
async def get_distinct_customer_names_by_ship_date(ship_date: str = Query(..., description="Ship date in 'M/D/YY' format")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.ShipDate = ? THEN T1.`Customer Names` END AS T FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID ) WHERE T IS NOT NULL", (ship_date,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the count of orders with a specific minimum quantity and customer name
@app.get("/v1/regional_sales/order_count_by_quantity_and_customer", operation_id="get_order_count_by_quantity_and_customer", summary="Get the count of orders with a specific minimum quantity and customer name")
async def get_order_count_by_quantity_and_customer(min_order_quantity: int = Query(..., description="Minimum order quantity"), customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.`Order Quantity` > ? AND T2.`Customer Names` = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID", (min_order_quantity, customer_name))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get the count of orders with a specific discount and customer name
@app.get("/v1/regional_sales/order_count_by_discount_and_customer", operation_id="get_order_count_by_discount_and_customer", summary="Retrieves the total count of orders that have a specified discount and are associated with a particular customer. The operation uses the provided discount and customer name to filter the sales orders and calculate the total count.")
async def get_order_count_by_discount_and_customer(discount_applied: float = Query(..., description="Discount applied"), customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.`Discount Applied` = ? AND T2.`Customer Names` = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID", (discount_applied, customer_name))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get distinct customer names for orders with a unit cost above a specific value
@app.get("/v1/regional_sales/distinct_customer_names_by_unit_cost", operation_id="get_distinct_customer_names_by_unit_cost", summary="Retrieve a list of unique customer names who have placed orders with a unit cost exceeding the provided minimum value. This operation filters the sales orders database to identify distinct customers who meet the specified unit cost threshold.")
async def get_distinct_customer_names_by_unit_cost(min_unit_cost: float = Query(..., description="Minimum unit cost")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.`Unit Cost` > ? THEN T1.`Customer Names` END AS T FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID ) WHERE T IS NOT NULL", (min_unit_cost,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get store details for a specific city
@app.get("/v1/regional_sales/store_details_by_city", operation_id="get_store_details_by_city", summary="Retrieves the store details, including the store ID, latitude, and longitude, for a specific city. The city is identified by the provided city name.")
async def get_store_details_by_city(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT StoreID, Latitude, Longitude FROM `Store Locations` WHERE `City Name` = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"store_details": []}
    return {"store_details": [{"StoreID": row[0], "Latitude": row[1], "Longitude": row[2]} for row in result]}

# Endpoint to get the city name with the highest population
@app.get("/v1/regional_sales/city_with_highest_population", operation_id="get_city_with_highest_population", summary="Retrieves the name of the city with the highest population from the store locations data. The operation allows you to limit the number of results returned, if desired.")
async def get_city_with_highest_population(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT `City Name` FROM `Store Locations` ORDER BY Population DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the count of store locations in a specific state and type
@app.get("/v1/regional_sales/count_store_locations_state_type", operation_id="get_count_store_locations_state_type", summary="Retrieves the total number of store locations in a specified state and type. The state and type parameters are used to filter the store locations and calculate the count.")
async def get_count_store_locations_state_type(state: str = Query(..., description="State of the store location"), type: str = Query(..., description="Type of the store location")):
    cursor.execute("SELECT SUM(CASE WHEN State = ? AND Type = ? THEN 1 ELSE 0 END) FROM `Store Locations`", (state, type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the order number and product name with the lowest unit price
@app.get("/v1/regional_sales/order_product_lowest_unit_price", operation_id="get_order_product_lowest_unit_price", summary="Get the order number and product name with the lowest unit price")
async def get_order_product_lowest_unit_price(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T1.OrderNumber, T2.`Product Name` FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE REPLACE(T1.`Unit Price`, ',', '') = ( SELECT REPLACE(T1.`Unit Price`, ',', '') FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID ORDER BY REPLACE(T1.`Unit Price`, ',', '') LIMIT ? )", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"order_number": [], "product_name": []}
    return {"order_number": result[0], "product_name": result[1]}

# Endpoint to get the product name with the highest profit margin in a specific year
@app.get("/v1/regional_sales/product_highest_profit_margin", operation_id="get_product_highest_profit_margin", summary="Retrieve the name of the product with the highest profit margin for a specified year. The year is provided in 'YY' format. The operation also allows limiting the number of results returned. The profit margin is calculated by subtracting the unit cost from the unit price.")
async def get_product_highest_profit_margin(year: str = Query(..., description="Year in 'YY' format"), limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T2.`Product Name` FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T1.OrderDate LIKE ? ORDER BY REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '') DESC LIMIT ?", (f'%/%/{year}', limit))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the average unit price of a specific product
@app.get("/v1/regional_sales/average_unit_price_product", operation_id="get_average_unit_price_product", summary="Retrieves the average unit price of a specific product by calculating the mean of the unit prices for all sales orders associated with the product. The product is identified by its name.")
async def get_average_unit_price_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT AVG(REPLACE(T1.`Unit Price`, ',', '')) FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T2.`Product Name` = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_unit_price": []}
    return {"average_unit_price": result[0]}

# Endpoint to get the sales team for a specific order date
@app.get("/v1/regional_sales/sales_team_specific_order_date", operation_id="get_sales_team_specific_order_date", summary="Get the sales team for a specific order date")
async def get_sales_team_specific_order_date(order_date: str = Query(..., description="Order date in 'MM/DD/YY' format")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T1.OrderDate = ? THEN T2.`Sales Team` ELSE NULL END AS T FROM `Sales Orders` T1 INNER JOIN `Sales Team` T2 ON T2.SalesTeamID = T1._SalesTeamID ) WHERE T IS NOT NULL", (order_date,))
    result = cursor.fetchone()
    if not result:
        return {"sales_team": []}
    return {"sales_team": result[0]}

# Endpoint to get the sales team with the fewest orders in a specific year
@app.get("/v1/regional_sales/sales_team_fewest_orders_year", operation_id="get_sales_team_fewest_orders_year", summary="Retrieves the sales team that has handled the fewest orders in a specified year. The operation allows you to limit the number of results returned. The data is filtered by the year of the order date and sorted in ascending order based on the count of order numbers.")
async def get_sales_team_fewest_orders_year(year: str = Query(..., description="Year in 'YY' format"), limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T2.`Sales Team` FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T1.OrderDate LIKE ? GROUP BY T2.`Sales Team` ORDER BY COUNT(T1.OrderNumber) ASC LIMIT ?", (f'%/%/{year}', limit))
    result = cursor.fetchone()
    if not result:
        return {"sales_team": []}
    return {"sales_team": result[0]}

# Endpoint to get the year with the most orders for a specific sales team
@app.get("/v1/regional_sales/year_most_orders_sales_team", operation_id="get_year_most_orders_sales_team", summary="Retrieves the year with the highest number of orders for a specified sales team. The operation returns the year with the most orders, based on the count of order numbers. The number of results can be limited by providing a value for the limit parameter.")
async def get_year_most_orders_sales_team(sales_team: str = Query(..., description="Name of the sales team"), limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT SUBSTR(T1.OrderDate, -2, 2) FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T2.`Sales Team` = ? GROUP BY SUBSTR(T1.OrderDate, -2, 2) ORDER BY COUNT(T1.OrderNumber) DESC LIMIT ?", (sales_team, limit))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the percentage of orders from a specific county in a specific year
@app.get("/v1/regional_sales/percentage_orders_county_year", operation_id="get_percentage_orders_county_year", summary="Retrieves the percentage of total orders originating from a specified county in a given year. The calculation is based on the total number of orders placed in that year. The county is identified by its name, and the year is specified in 'YY' format.")
async def get_percentage_orders_county_year(county: str = Query(..., description="County name"), year: str = Query(..., description="Year in 'YY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.County = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T1.OrderDate LIKE ?", (county, f'%/%/{year}'))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the order number with the highest unit price
@app.get("/v1/regional_sales/order_highest_unit_price", operation_id="get_order_highest_unit_price", summary="Get the order number with the highest unit price")
async def get_order_highest_unit_price(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT OrderNumber FROM `Sales Orders` WHERE REPLACE(`Unit Price`, ',', '') = ( SELECT REPLACE(`Unit Price`, ',', '') FROM `Sales Orders` ORDER BY REPLACE(`Unit Price`, ',', '') DESC LIMIT ? )", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"order_number": []}
    return {"order_number": result[0]}

# Endpoint to get the top sales team ID based on order date pattern
@app.get("/v1/regional_sales/top_sales_team_by_order_date", operation_id="get_top_sales_team", summary="Retrieves the ID of the top-performing sales team based on the specified order date pattern. The operation filters sales orders by the given date pattern, groups them by sales team ID, and orders the results by the count of sales team IDs in descending order. The ID of the top sales team is then returned.")
async def get_top_sales_team(order_date_pattern: str = Query(..., description="Order date pattern in '%%/%/YY' format")):
    cursor.execute("SELECT _SalesTeamID FROM `Sales Orders` WHERE OrderDate LIKE ? GROUP BY _SalesTeamID ORDER BY COUNT(_SalesTeamID) DESC LIMIT 1", (order_date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"sales_team_id": []}
    return {"sales_team_id": result[0]}

# Endpoint to get distinct unit costs for a specific order number
@app.get("/v1/regional_sales/distinct_unit_costs_by_order_number", operation_id="get_distinct_unit_costs", summary="Retrieves a list of unique unit costs associated with a specific order number from the sales orders database.")
async def get_distinct_unit_costs(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT DISTINCT T FROM ( SELECT IIF(OrderNumber = ?, `Unit Cost`, NULL) AS T FROM `Sales Orders` ) WHERE T IS NOT NULL", (order_number,))
    result = cursor.fetchall()
    if not result:
        return {"unit_costs": []}
    return {"unit_costs": [row[0] for row in result]}

# Endpoint to get the count of orders in a specific county and order date pattern
@app.get("/v1/regional_sales/order_count_by_county_and_date", operation_id="get_order_count_by_county_and_date", summary="Retrieves the total number of orders placed in a specified county and matching a given date pattern. The operation calculates the sum of orders based on the provided county name and date pattern, using an inner join between the 'Sales Orders' and 'Store Locations' tables.")
async def get_order_count_by_county_and_date(county: str = Query(..., description="County name"), order_date_pattern: str = Query(..., description="Order date pattern in '%%/%/YY' format")):
    cursor.execute("SELECT SUM(CASE WHEN T2.County = ? AND OrderDate LIKE ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID", (county, order_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get the latitude and longitude of a store based on order number
@app.get("/v1/regional_sales/store_location_by_order_number", operation_id="get_store_location", summary="Retrieves the geographical coordinates (latitude and longitude) of the store associated with a specific order number. The order number is used to identify the store and its location.")
async def get_store_location(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT T2.Latitude, T2.Longitude FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T1.OrderNumber = ?", (order_number,))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": {"latitude": result[0], "longitude": result[1]}}

# Endpoint to get the count of orders by city name and order date pattern
@app.get("/v1/regional_sales/order_count_by_city_and_date", operation_id="get_order_count_by_city_and_date", summary="Retrieves the total number of orders for each city, filtered by a specified order date pattern. The date pattern should be provided in the 'MM/DD/YY' format. This operation is useful for analyzing sales trends across different cities and time periods.")
async def get_order_count_by_city_and_date(order_date_pattern: str = Query(..., description="Order date pattern in '%%/%/YY' format")):
    cursor.execute("SELECT COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T1.OrderDate LIKE ? GROUP BY T2.`City Name` HAVING COUNT(T1.OrderNumber)", (order_date_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"order_counts": []}
    return {"order_counts": [row[0] for row in result]}

# Endpoint to get distinct customer names with more than a specified number of orders in a given date pattern
@app.get("/v1/regional_sales/distinct_customer_names_by_order_count_and_date", operation_id="get_distinct_customer_names", summary="Retrieve a list of unique customer names that have placed more than a specified number of orders on a given date pattern. The input parameters include the minimum order count and the order date pattern in '%%/%/YY' format.")
async def get_distinct_customer_names(min_order_count: int = Query(..., description="Minimum order count"), order_date_pattern: str = Query(..., description="Order date pattern in '%%/%/YY' format")):
    cursor.execute("SELECT DISTINCT IIF(COUNT(T2.CustomerID) > ?, T2.`Customer Names`, NULL) FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID WHERE T1.OrderDate LIKE ? GROUP BY T1._CustomerID HAVING COUNT(T2.CustomerID)", (min_order_count, order_date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the count of orders for a specific customer in given years
@app.get("/v1/regional_sales/order_count_by_customer_and_years", operation_id="get_order_count_by_customer_and_years", summary="Retrieve the total number of orders placed by a specific customer during a specified range of years. The input parameters include the customer's name and three years in 'YY' format, which are used to filter the orders and calculate the total count.")
async def get_order_count_by_customer_and_years(year1: str = Query(..., description="Year in 'YY' format"), year2: str = Query(..., description="Year in 'YY' format"), year3: str = Query(..., description="Year in 'YY' format"), customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT SUM(CASE WHEN SUBSTR(T1.OrderDate, -2) IN (?, ?, ?) AND T2.`Customer Names` = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID", (year1, year2, year3, customer_name))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get distinct customer names with order quantity greater than a specified value on a specific date
@app.get("/v1/regional_sales/distinct_customer_names_by_order_quantity_and_date", operation_id="get_distinct_customer_names_by_order_quantity_and_date", summary="Retrieve a list of unique customer names who have placed orders with a total quantity exceeding a specified minimum on a given date. The input parameters include the minimum order quantity and the order date in 'MM/DD/YY' format.")
async def get_distinct_customer_names_by_order_quantity_and_date(min_order_quantity: int = Query(..., description="Minimum order quantity"), order_date: str = Query(..., description="Order date in 'MM/DD/YY' format")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN SUM(T1.`Order Quantity`) > ? THEN T2.`Customer Names` END AS T FROM `Sales Orders` T1 INNER JOIN Customers T2 ON T2.CustomerID = T1._CustomerID WHERE T1.OrderDate = ? GROUP BY T1._CustomerID ) WHERE T IS NOT NULL", (min_order_quantity, order_date))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the percentage of orders with a price difference greater than a specified value for a specific sales team
@app.get("/v1/regional_sales/percentage_orders_by_price_difference_and_sales_team", operation_id="get_percentage_orders_by_price_difference_and_sales_team", summary="Retrieves the percentage of orders with a price difference greater than a specified value for a given sales team. The price difference is calculated by subtracting the unit cost from the unit price, excluding any commas. The sales team is identified by its name.")
async def get_percentage_orders_by_price_difference_and_sales_team(price_difference: int = Query(..., description="Price difference"), sales_team: str = Query(..., description="Sales team name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '') > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T2.`Sales Team` = ?", (price_difference, sales_team))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of sales teams in a specific region
@app.get("/v1/regional_sales/sales_team_count_by_region", operation_id="get_sales_team_count_by_region", summary="Retrieves the total number of sales teams operating in a specified region. The region is identified by its name.")
async def get_sales_team_count_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT SUM(CASE WHEN Region = ? THEN 1 ELSE 0 END) FROM `Sales Team`", (region,))
    result = cursor.fetchone()
    if not result:
        return {"sales_team_count": []}
    return {"sales_team_count": result[0]}

# Endpoint to get distinct store locations based on county
@app.get("/v1/regional_sales/store_locations_by_county", operation_id="get_store_locations_by_county", summary="Retrieve unique store locations, including their city names, latitudes, and longitudes, for a specific county.")
async def get_store_locations_by_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT DISTINCT `City Name`, Latitude, Longitude FROM `Store Locations` WHERE County = ?", (county,))
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get order numbers with the highest unit cost
@app.get("/v1/regional_sales/order_numbers_highest_unit_cost", operation_id="get_order_numbers_highest_unit_cost", summary="Retrieves the order numbers associated with the highest unit cost in the sales orders database. The operation filters the sales orders based on the highest unit cost, excluding any commas in the cost values, and returns the corresponding order numbers.")
async def get_order_numbers_highest_unit_cost():
    cursor.execute("SELECT OrderNumber FROM `Sales Orders` WHERE REPLACE(`Unit Cost`, ',', '') = ( SELECT REPLACE(`Unit Cost`, ',', '') FROM `Sales Orders` ORDER BY REPLACE(`Unit Cost`, ',', '') DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"order_numbers": []}
    return {"order_numbers": result}

# Endpoint to get product names within a specified product ID range
@app.get("/v1/regional_sales/product_names_by_id_range", operation_id="get_product_names_by_id_range", summary="Retrieves the names of products that fall within the specified product ID range. The operation filters the product list based on the provided minimum and maximum product IDs, returning only the names of products that meet the criteria.")
async def get_product_names_by_id_range(min_product_id: int = Query(..., description="Minimum product ID"), max_product_id: int = Query(..., description="Maximum product ID")):
    cursor.execute("SELECT T FROM ( SELECT CASE  WHEN ProductID BETWEEN ? AND ? THEN `Product Name` ELSE NULL END AS T FROM Products ) WHERE T IS NOT NULL", (min_product_id, max_product_id))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the ratio of the highest to the lowest unit cost
@app.get("/v1/regional_sales/unit_cost_ratio", operation_id="get_unit_cost_ratio", summary="Retrieves the ratio of the highest to the lowest unit cost from the sales orders database. This operation considers the unit cost values after removing any commas and returns a single value representing the ratio.")
async def get_unit_cost_ratio():
    cursor.execute("SELECT ( SELECT REPLACE(`Unit Cost`, ',', '') FROM `Sales Orders` WHERE REPLACE(`Unit Cost`, ',', '') = ( SELECT REPLACE(`Unit Cost`, ',', '') FROM `Sales Orders` ORDER BY REPLACE(`Unit Cost`, ',', '') DESC LIMIT 1 ) ORDER BY REPLACE(`Unit Cost`, ',', '') DESC LIMIT 1 ) / ( SELECT REPLACE(`Unit Cost`, ',', '') FROM `Sales Orders` WHERE REPLACE(`Unit Cost`, ',', '') = ( SELECT REPLACE(`Unit Cost`, ',', '') FROM `Sales Orders` ORDER BY REPLACE(`Unit Cost`, ',', '') ASC LIMIT 1 ) ORDER BY REPLACE(`Unit Cost`, ',', '') ASC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the most ordered product name for a specific order date pattern
@app.get("/v1/regional_sales/most_ordered_product_by_date", operation_id="get_most_ordered_product_by_date", summary="Retrieves the name of the product that was ordered the most on a specific date pattern. The date pattern should be provided in the 'MM/DD/YY' format. The operation fetches the product name from the Sales Orders table, which is joined with the Products table using the ProductID. The results are grouped by ProductID and ordered by the count of ProductID in descending order, with the top result being returned.")
async def get_most_ordered_product_by_date(order_date_pattern: str = Query(..., description="Order date pattern in '%%/%%/YY' format")):
    cursor.execute("SELECT T2.`Product Name` FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T1.OrderDate LIKE ? GROUP BY T1._ProductID ORDER BY COUNT(T1._ProductID) DESC LIMIT 1", (order_date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the count of sales orders for a specific sales team member
@app.get("/v1/regional_sales/sales_order_count_by_team_member", operation_id="get_sales_order_count_by_team_member", summary="Retrieves the total number of sales orders associated with a specific sales team member. The operation calculates the sum of sales orders for the specified team member, providing a quantitative measure of their sales performance.")
async def get_sales_order_count_by_team_member(sales_team_member: str = Query(..., description="Sales team member name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.`Sales Team` = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID", (sales_team_member,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get order numbers for a specific city
@app.get("/v1/regional_sales/order_numbers_by_city", operation_id="get_order_numbers_by_city", summary="Retrieves a distinct list of order numbers associated with a specific city. The operation filters sales orders based on the provided city name and returns the corresponding order numbers.")
async def get_order_numbers_by_city(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.`City Name` = ? THEN T1.OrderNumber END AS T FROM `Sales Orders` T1 INNER JOIN `Store Locations` T2 ON T2.StoreID = T1._StoreID ) WHERE T IS NOT NULL", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"order_numbers": []}
    return {"order_numbers": result}

# Endpoint to get the order number with the highest order quantity for a specific customer
@app.get("/v1/regional_sales/highest_order_quantity_by_customer", operation_id="get_highest_order_quantity_by_customer", summary="Retrieves the order number with the highest order quantity for a specific customer. The operation requires the customer's name as input and returns the order number with the highest quantity associated with the provided customer.")
async def get_highest_order_quantity_by_customer(customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT T1.OrderNumber FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID WHERE T2.`Customer Names` = ? ORDER BY T1.`Order Quantity` DESC LIMIT 1", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"order_number": []}
    return {"order_number": result[0]}

# Endpoint to get distinct order numbers and product names for a specific sales channel
@app.get("/v1/regional_sales/order_product_by_sales_channel", operation_id="get_order_product_by_sales_channel", summary="Retrieve a unique list of order numbers and corresponding product names for a specified sales channel.")
async def get_order_product_by_sales_channel(sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT DISTINCT T1.OrderNumber, T2.`Product Name` FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T1.`Sales Channel` = ?", (sales_channel,))
    result = cursor.fetchall()
    if not result:
        return {"order_products": []}
    return {"order_products": result}

# Endpoint to get the sum of orders based on order date pattern, sales channel, and city name
@app.get("/v1/regional_sales/sum_orders_by_date_channel_city", operation_id="get_sum_orders_by_date_channel_city", summary="Retrieves the total number of orders that match a specific date pattern, sales channel, and city. The date pattern should be provided in 'MM/%/YY' format. The sales channel and city name are also required to filter the results.")
async def get_sum_orders_by_date_channel_city(order_date_pattern: str = Query(..., description="Order date pattern in 'MM/%/YY' format"), sales_channel: str = Query(..., description="Sales channel"), city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.OrderDate LIKE ? AND T1.`Sales Channel` = ? AND T2.`City Name` = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID", (order_date_pattern, sales_channel, city_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the product name with the lowest order quantity in a specific county
@app.get("/v1/regional_sales/product_with_lowest_order_quantity_in_county", operation_id="get_product_with_lowest_order_quantity_in_county", summary="Retrieves the name of the product with the lowest order quantity in a specified county. The operation filters sales orders by county and sorts them by order quantity in ascending order, returning the product name associated with the first order.")
async def get_product_with_lowest_order_quantity_in_county(county: str = Query(..., description="County name")):
    cursor.execute("SELECT T1.`Product Name` FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID INNER JOIN `Store Locations` AS T3 ON T3.StoreID = T2._StoreID WHERE T3.County = ? ORDER BY T2.`Order Quantity` ASC LIMIT 1", (county,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get distinct order numbers for a specific sales team
@app.get("/v1/regional_sales/distinct_order_numbers_by_sales_team", operation_id="get_distinct_order_numbers_by_sales_team", summary="Retrieve a unique set of order numbers associated with a particular sales team. The operation filters the sales orders based on the provided sales team name and returns only the distinct order numbers.")
async def get_distinct_order_numbers_by_sales_team(sales_team: str = Query(..., description="Sales team name")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.`Sales Team` = ? THEN T1.OrderNumber ELSE NULL END AS T FROM `Sales Orders` T1 INNER JOIN `Sales Team` T2 ON T2.SalesTeamID = T1._SalesTeamID ) WHERE T IS NOT NULL", (sales_team,))
    result = cursor.fetchall()
    if not result:
        return {"order_numbers": []}
    return {"order_numbers": [row[0] for row in result]}

# Endpoint to get the count of orders for a specific product and order date pattern
@app.get("/v1/regional_sales/count_orders_by_product_and_date", operation_id="get_count_orders_by_product_and_date", summary="Retrieves the total number of orders for a specific product and date pattern. The product is identified by its name, and the date pattern follows the 'MM/%/YY' format. This operation provides a quantitative overview of sales for a particular product within a specified date range.")
async def get_count_orders_by_product_and_date(product_name: str = Query(..., description="Product name"), order_date_pattern: str = Query(..., description="Order date pattern in 'MM/%/YY' format")):
    cursor.execute("SELECT COUNT(T2.OrderNumber) FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID WHERE T1.`Product Name` = ? AND T2.OrderDate LIKE ?", (product_name, order_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average monthly order quantity for a specific product and order date pattern
@app.get("/v1/regional_sales/average_monthly_order_quantity", operation_id="get_average_monthly_order_quantity", summary="Retrieves the average monthly order quantity for a specific product over a given year. The product is identified by its name, and the year is determined by the order date pattern provided in the '%m/%d/%Y' format.")
async def get_average_monthly_order_quantity(product_name: str = Query(..., description="Product name"), order_date_pattern: str = Query(..., description="Order date pattern in '%/%/YY' format")):
    cursor.execute("SELECT CAST(SUM(T2.`Order Quantity`) AS REAL) / 12 FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID WHERE T1.`Product Name` = ? AND T2.OrderDate LIKE ?", (product_name, order_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"average_quantity": []}
    return {"average_quantity": result[0]}

# Endpoint to get the percentage of order quantity for a specific city and order date pattern
@app.get("/v1/regional_sales/percentage_order_quantity_by_city_and_date", operation_id="get_percentage_order_quantity_by_city_and_date", summary="Retrieves the percentage of total order quantity for a specific city and date pattern. The calculation is based on the sum of order quantities for the specified city and date pattern, divided by the total order quantity across all cities and dates. The city name and date pattern are provided as input parameters.")
async def get_percentage_order_quantity_by_city_and_date(city_name: str = Query(..., description="City name"), order_date_pattern: str = Query(..., description="Order date pattern in '%/%/YY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.`City Name` = ? THEN T2.`Order Quantity` ELSE 0 END) AS REAL) * 100 / SUM(T2.`Order Quantity`) FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID INNER JOIN `Store Locations` AS T3 ON T3.StoreID = T2._StoreID WHERE T2.OrderDate LIKE ?", (city_name, order_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in order counts between two warehouse codes
@app.get("/v1/regional_sales/difference_in_order_counts_by_warehouse", operation_id="get_difference_in_order_counts_by_warehouse", summary="Retrieve the difference in order counts between two specified warehouses. This operation compares the total number of orders from the first warehouse code with the total number of orders from the second warehouse code, returning the difference.")
async def get_difference_in_order_counts_by_warehouse(warehouse_code_1: str = Query(..., description="First warehouse code"), warehouse_code_2: str = Query(..., description="Second warehouse code")):
    cursor.execute("SELECT SUM(IIF(WarehouseCode = ?, 1, 0)) - SUM(IIF(WarehouseCode = ?, 1, 0)) AS difference FROM `Sales Orders`", (warehouse_code_1, warehouse_code_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get distinct product names for a specific customer and delivery date pattern
@app.get("/v1/regional_sales/distinct_product_names_by_customer_and_delivery_date", operation_id="get_distinct_product_names_by_customer_and_delivery_date", summary="Get distinct product names for a specific customer and delivery date pattern")
async def get_distinct_product_names_by_customer_and_delivery_date(delivery_date_pattern: str = Query(..., description="Delivery date pattern in '%/%/YY' format"), customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.DeliveryDate LIKE ? AND T1.`Customer Names` = ? THEN T3.`Product Name` END AS T FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID INNER JOIN Products T3 ON T3.ProductID = T2._ProductID ) WHERE T IS NOT NULL", (delivery_date_pattern, customer_name))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get distinct store IDs and regions for a specific state
@app.get("/v1/regional_sales/distinct_store_ids_and_regions_by_state", operation_id="get_distinct_store_ids_and_regions_by_state", summary="Retrieve a unique list of store IDs and their corresponding regions for a given state. The state is specified as an input parameter.")
async def get_distinct_store_ids_and_regions_by_state(state: str = Query(..., description="State name")):
    cursor.execute("SELECT DISTINCT T2.StoreID, T1.Region FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode WHERE T2.State = ?", (state,))
    result = cursor.fetchall()
    if not result:
        return {"store_ids_and_regions": []}
    return {"store_ids_and_regions": [{"store_id": row[0], "region": row[1]} for row in result]}

# Endpoint to get the count of orders for two specific customers
@app.get("/v1/regional_sales/count_orders_by_two_customers", operation_id="get_count_orders_by_two_customers", summary="Retrieves the total number of orders for two specified customers. The operation computes the sum of orders for each customer by comparing the provided customer names with the customer names in the sales orders database. The result is a pair of order counts for the two customers.")
async def get_count_orders_by_two_customers(customer_name_1: str = Query(..., description="First customer name"), customer_name_2: str = Query(..., description="Second customer name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.`Customer Names` = ? THEN 1 ELSE 0 END), SUM(CASE WHEN T2.`Customer Names` = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID", (customer_name_1, customer_name_2))
    result = cursor.fetchone()
    if not result:
        return {"counts": []}
    return {"counts": {"customer_1": result[0], "customer_2": result[1]}}

# Endpoint to get the top store ID based on order count in specified cities
@app.get("/v1/regional_sales/top_store_by_order_count", operation_id="get_top_store_by_order_count", summary="Retrieves the ID of the store with the highest order count across the two specified cities. The operation considers sales orders from stores located in the provided cities and ranks them based on the number of orders. The store with the most orders is returned.")
async def get_top_store_by_order_count(city_name_1: str = Query(..., description="First city name"), city_name_2: str = Query(..., description="Second city name")):
    cursor.execute("SELECT T2.StoreID FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T2.`City Name` = ? OR T2.`City Name` = ? GROUP BY T2.StoreID ORDER BY COUNT(T1.OrderNumber) DESC LIMIT 1", (city_name_1, city_name_2))
    result = cursor.fetchone()
    if not result:
        return {"store_id": []}
    return {"store_id": result[0]}

# Endpoint to get distinct customer names and product names based on sales team and sales channel
@app.get("/v1/regional_sales/customer_product_by_sales_team_channel", operation_id="get_customer_product_by_sales_team_channel", summary="Retrieves unique combinations of customer names and product names associated with a specific sales team and sales channel. This operation filters sales orders based on the provided sales team and sales channel, ensuring that only relevant customer-product pairs are returned.")
async def get_customer_product_by_sales_team_channel(sales_team: str = Query(..., description="Sales team name"), sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT DISTINCT T1.`Customer Names`, T4.`Product Name` FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID INNER JOIN `Sales Team` AS T3 ON T3.SalesTeamID = T2._SalesTeamID INNER JOIN Products AS T4 ON T4.ProductID = T2._ProductID WHERE T3.`Sales Team` = ? AND T2.`Sales Channel` = ?", (sales_team, sales_channel))
    result = cursor.fetchall()
    if not result:
        return {"customer_product": []}
    return {"customer_product": result}

# Endpoint to get distinct customer names and IDs based on sales channel and total profit
@app.get("/v1/regional_sales/customer_by_sales_channel_profit", operation_id="get_customer_by_sales_channel_profit", summary="Retrieve a list of unique customers, identified by their names and IDs, who have generated a profit greater than the specified minimum threshold through a specific sales channel. The operation filters sales orders based on the provided sales channel and calculates the total profit for each customer. It then returns the customers who have exceeded the minimum profit requirement.")
async def get_customer_by_sales_channel_profit(sales_channel: str = Query(..., description="Sales channel"), min_profit: float = Query(..., description="Minimum profit")):
    cursor.execute("SELECT DISTINCT `Customer Names`, CustomerID FROM ( SELECT T2.`Customer Names`, T2.CustomerID , SUM(REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '')) AS T FROM `Sales Orders` T1 INNER JOIN Customers T2 ON T2.CustomerID = T1._CustomerID WHERE T1.`Sales Channel` = ? GROUP BY T2.CustomerID ) WHERE T > ?", (sales_channel, min_profit))
    result = cursor.fetchall()
    if not result:
        return {"customer_info": []}
    return {"customer_info": result}

# Endpoint to get the total profit for a specific product delivered in a specific year
@app.get("/v1/regional_sales/total_profit_by_product_year", operation_id="get_total_profit_by_product_year", summary="Retrieves the total profit for a specified product delivered in a given year. The calculation is based on the sum of the difference between the unit price and unit cost of the product. The delivery year is provided in the '%%/%/YY' format, and the product name is required to identify the specific product.")
async def get_total_profit_by_product_year(delivery_year: str = Query(..., description="Delivery year in '%%/%/YY' format"), product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT SUM(REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '')) FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T1.DeliveryDate LIKE ? AND T2.`Product Name` = ?", (f'%/{delivery_year}', product_name))
    result = cursor.fetchone()
    if not result:
        return {"total_profit": []}
    return {"total_profit": result[0]}

# Endpoint to get the count of orders based on store population range
@app.get("/v1/regional_sales/order_count_by_population_range", operation_id="get_order_count_by_population_range", summary="Retrieves the total number of orders associated with stores that have a population within the specified range. The range is defined by the minimum and maximum population values provided as input parameters.")
async def get_order_count_by_population_range(min_population: int = Query(..., description="Minimum population"), max_population: int = Query(..., description="Maximum population")):
    cursor.execute("SELECT COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T2.Population BETWEEN ? AND ?", (min_population, max_population))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get distinct product names based on time zone and sales channel
@app.get("/v1/regional_sales/product_names_by_time_zone_sales_channel", operation_id="get_product_names_by_time_zone_sales_channel", summary="Retrieves a list of unique product names that are sold in a specific time zone and through a particular sales channel. The operation filters products based on the provided time zone and sales channel, ensuring that only products meeting these criteria are returned.")
async def get_product_names_by_time_zone_sales_channel(time_zone: str = Query(..., description="Time zone"), sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T3.`Time Zone` = ? AND T2.`Sales Channel` = ? THEN T1.`Product Name` ELSE NULL END AS T FROM Products T1 INNER JOIN `Sales Orders` T2 ON T2._ProductID = T1.ProductID INNER JOIN `Store Locations` T3 ON T3.StoreID = T2._StoreID ) WHERE T IS NOT NULL", (time_zone, sales_channel))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get distinct order numbers and product names based on order date
@app.get("/v1/regional_sales/order_product_by_date", operation_id="get_order_product_by_date", summary="Retrieves unique order numbers and corresponding product names for a specified order date. The operation filters out null values from the results.")
async def get_order_product_by_date(order_date: str = Query(..., description="Order date in 'M/D/YY' format")):
    cursor.execute("SELECT DISTINCT OrderNumber, `Product Name` FROM ( SELECT IIF(T2.OrderDate = ?, T2.OrderNumber, NULL) AS \"OrderNumber\" , IIF(T2.OrderDate = ?, T1.`Product Name`, NULL) AS \"Product Name\" FROM Products T1 INNER JOIN `Sales Orders` T2 ON T2._ProductID = T1.ProductID ) WHERE OrderNumber IS NOT NULL AND `Product Name` IS NOT NULL", (order_date, order_date))
    result = cursor.fetchall()
    if not result:
        return {"order_product": []}
    return {"order_product": result}

# Endpoint to get the average order count per year for a specific customer
@app.get("/v1/regional_sales/average_order_count_per_year", operation_id="get_average_order_count_per_year", summary="Retrieves the average annual order count for a specific customer over three consecutive years. The calculation is based on the provided customer name and the years specified in the '%%/%/YY' format. This operation provides insights into the customer's purchasing trends and behavior.")
async def get_average_order_count_per_year(customer_name: str = Query(..., description="Customer name"), year_1: str = Query(..., description="First year in '%%/%/YY' format"), year_2: str = Query(..., description="Second year in '%%/%/YY' format"), year_3: str = Query(..., description="Third year in '%%/%/YY' format")):
    cursor.execute("SELECT COUNT(T1.OrderNumber) / 3 FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID WHERE (T1.OrderDate LIKE ? AND T2.`Customer Names` = ?) OR (T1.OrderDate LIKE ? AND T2.`Customer Names` = ?) OR (T1.OrderDate LIKE ? AND T2.`Customer Names` = ?)", (f'%/{year_1}', customer_name, f'%/{year_2}', customer_name, f'%/{year_3}', customer_name))
    result = cursor.fetchone()
    if not result:
        return {"average_order_count": []}
    return {"average_order_count": result[0]}

# Endpoint to get warehouse statistics based on warehouse code, product name, and order year
@app.get("/v1/regional_sales/warehouse_statistics", operation_id="get_warehouse_statistics", summary="Retrieves monthly average and percentage of total sales, as well as the total quantity sold for a specific product in a given year at a particular warehouse. The operation filters sales data by warehouse code, product name, and order year.")
async def get_warehouse_statistics(warehouse_code: str = Query(..., description="Warehouse code"), product_name: str = Query(..., description="Product name"), order_year: str = Query(..., description="Order year in '%%/%/YY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.WarehouseCode = ? THEN 1 ELSE 0 END) AS REAL) / 12 , CAST(SUM(CASE WHEN T2.WarehouseCode = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.OrderNumber), COUNT(CASE WHEN T1.`Product Name` = ? AND T2.WarehouseCode = ? THEN T2.`Order Quantity` ELSE NULL END) FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID WHERE T2.OrderDate LIKE ?", (warehouse_code, warehouse_code, product_name, warehouse_code, f'%/{order_year}'))
    result = cursor.fetchone()
    if not result:
        return {"warehouse_statistics": []}
    return {"warehouse_statistics": result}

# Endpoint to get distinct procurement dates based on customer ID
@app.get("/v1/regional_sales/procurement_dates_by_customer", operation_id="get_procurement_dates_by_customer", summary="Retrieves a list of unique procurement dates associated with a specific customer. The customer is identified by the provided customer_id.")
async def get_procurement_dates_by_customer(customer_id: int = Query(..., description="Customer ID")):
    cursor.execute("SELECT DISTINCT T FROM ( SELECT IIF(_CustomerID = ?, ProcuredDate, NULL) AS T FROM `Sales Orders` ) WHERE T IS NOT NULL", (customer_id,))
    result = cursor.fetchall()
    if not result:
        return {"procurement_dates": []}
    return {"procurement_dates": result}

# Endpoint to get the sum of orders with a specific quantity and sales channel
@app.get("/v1/regional_sales/sum_orders_quantity_channel", operation_id="get_sum_orders_quantity_channel", summary="Retrieves the total count of orders that match a specific quantity and sales channel from the sales orders database.")
async def get_sum_orders_quantity_channel(order_quantity: int = Query(..., description="Order quantity"), sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT SUM(CASE WHEN `Order Quantity` = ? AND `Sales Channel` = ? THEN 1 ELSE 0 END) FROM `Sales Orders`", (order_quantity, sales_channel))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct sales team IDs based on discount applied and sales channel
@app.get("/v1/regional_sales/distinct_sales_team_ids", operation_id="get_distinct_sales_team_ids", summary="Retrieves a unique set of sales team identifiers that meet the specified discount and sales channel criteria. This operation filters sales orders based on the provided discount and sales channel, and returns the distinct sales team IDs associated with the filtered orders.")
async def get_distinct_sales_team_ids(discount_applied: str = Query(..., description="Discount applied"), sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT DISTINCT T FROM ( SELECT CASE  WHEN `Discount Applied` = ? AND `Sales Channel` = ? THEN _SalesTeamID ELSE NULL END AS T FROM `Sales Orders` ) WHERE T IS NOT NULL", (discount_applied, sales_channel))
    result = cursor.fetchall()
    if not result:
        return {"sales_team_ids": []}
    return {"sales_team_ids": [row[0] for row in result]}

# Endpoint to get the sum of store locations based on population, type, and city name
@app.get("/v1/regional_sales/sum_store_locations", operation_id="get_sum_store_locations", summary="Retrieves the total count of store locations that meet the specified population, type, and city name criteria. The population parameter filters store locations based on the population of their respective cities. The type parameter filters store locations based on their type. The city_name parameter filters store locations based on the name of the city they are located in.")
async def get_sum_store_locations(population: int = Query(..., description="Population"), type: str = Query(..., description="Type"), city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT SUM(CASE WHEN Population < ? AND Type = ? AND `City Name` = ? THEN 1 ELSE 0 END) FROM `Store Locations`", (population, type, city_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the count of distinct states in a given region
@app.get("/v1/regional_sales/count_distinct_states", operation_id="get_count_distinct_states", summary="Retrieves the number of unique states within a specified region. The operation filters states based on the provided region and returns the count of distinct states.")
async def get_count_distinct_states(region: str = Query(..., description="Region")):
    cursor.execute("SELECT COUNT(DISTINCT T) FROM ( SELECT CASE  WHEN Region = ? THEN State ELSE NULL END AS T FROM Regions ) WHERE T IS NOT NULL", (region,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get top products by profit margin
@app.get("/v1/regional_sales/top_products_by_profit_margin", operation_id="get_top_products_by_profit_margin", summary="Retrieves a list of top-performing products based on their profit margins. The list is generated by calculating the difference between the total sales price and the total cost price for each product, then sorting the products in descending order based on this profit margin. The number of products returned can be limited by specifying the 'limit' parameter.")
async def get_top_products_by_profit_margin(limit: int = Query(..., description="Limit of top products to return")):
    cursor.execute("SELECT T2.`Product Name` FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID GROUP BY T1._ProductID ORDER BY SUM(REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '')) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get distinct sales teams for a specific customer
@app.get("/v1/regional_sales/distinct_sales_teams_customer", operation_id="get_distinct_sales_teams_customer", summary="Retrieves the unique sales teams that have served a specific customer. The operation filters the sales teams based on the provided customer name.")
async def get_distinct_sales_teams_customer(customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT DISTINCT T3.`Sales Team` FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID INNER JOIN `Sales Team` AS T3 ON T3.SalesTeamID = T2._SalesTeamID WHERE T1.`Customer Names` = ?", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"sales_teams": []}
    return {"sales_teams": [row[0] for row in result]}

# Endpoint to get distinct regions based on warehouse code
@app.get("/v1/regional_sales/distinct_regions_warehouse_code", operation_id="get_distinct_regions_warehouse_code", summary="Retrieves a list of distinct regions associated with the specified warehouse code. This operation filters regions based on the provided warehouse code and returns only those regions that have a corresponding store location and sales order linked to the warehouse.")
async def get_distinct_regions_warehouse_code(warehouse_code: str = Query(..., description="Warehouse code")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T3.WarehouseCode = ? THEN T1.Region END AS T FROM Regions T1 INNER JOIN `Store Locations` T2 ON T2.StateCode = T1.StateCode INNER JOIN `Sales Orders` T3 ON T3._StoreID = T2.StoreID ) WHERE T IS NOT NULL", (warehouse_code,))
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": [row[0] for row in result]}

# Endpoint to get product names and customer names based on order date and delivery date
@app.get("/v1/regional_sales/product_customer_names", operation_id="get_product_customer_names", summary="Retrieves the names of products and their respective customers based on a specified order date and delivery date. The operation uses the provided dates to filter sales orders and returns the corresponding product and customer names.")
async def get_product_customer_names(order_date: str = Query(..., description="Order date in 'M/D/YY' format"), delivery_date: str = Query(..., description="Delivery date in 'M/D/YY' format")):
    cursor.execute("SELECT T3.`Product Name`, T1.`Customer Names` FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID INNER JOIN Products AS T3 ON T3.ProductID = T2._ProductID WHERE T2.OrderDate = ? AND T2.DeliveryDate = ?", (order_date, delivery_date))
    result = cursor.fetchall()
    if not result:
        return {"product_customer_names": []}
    return {"product_customer_names": [{"product_name": row[0], "customer_name": row[1]} for row in result]}

# Endpoint to get the sum of procured orders on a specific date and city
@app.get("/v1/regional_sales/sum_procured_orders_date_city", operation_id="get_sum_procured_orders", summary="Retrieves the total number of procured orders for a specific city on a given date. The operation calculates the sum based on the provided date and city name, using data from the 'Sales Orders' and 'Store Locations' tables.")
async def get_sum_procured_orders(procured_date: str = Query(..., description="Procured date in 'MM/DD/YY' format"), city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT SUM(CASE WHEN T1.ProcuredDate = ? AND T2.`City Name` = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID", (procured_date, city_name))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the top sales channel based on median income
@app.get("/v1/regional_sales/top_sales_channel_median_income", operation_id="get_top_sales_channel", summary="Retrieves the sales channel with the highest frequency among the top records based on median income. The number of top records considered can be specified using the 'limit' input parameter.")
async def get_top_sales_channel(limit: int = Query(..., description="Limit of top median income records")):
    cursor.execute("SELECT `Sales Channel` FROM ( SELECT T1.`Sales Channel` FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID ORDER BY T2.`Median Income` DESC LIMIT ? ) GROUP BY `Sales Channel` ORDER BY COUNT(`Sales Channel`) DESC LIMIT 1", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"sales_channel": []}
    return {"sales_channel": result[0]}

# Endpoint to get the top sales teams based on profit margin
@app.get("/v1/regional_sales/top_sales_teams_profit_margin", operation_id="get_top_sales_teams", summary="Retrieves a list of the top sales teams, ranked by their profit margin. The profit margin is calculated as the difference between the unit price and unit cost of sales orders. The number of teams returned is determined by the provided limit parameter.")
async def get_top_sales_teams(limit: int = Query(..., description="Limit of top sales teams")):
    cursor.execute("SELECT T2.`Sales Team` FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID ORDER BY REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '') DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"sales_teams": []}
    return {"sales_teams": [row[0] for row in result]}

# Endpoint to get the maximum discount applied in a specific state and land area
@app.get("/v1/regional_sales/max_discount_state_land_area", operation_id="get_max_discount", summary="Retrieves the highest discount percentage that has been applied to sales orders in a specific state and land area. The state and land area are provided as input parameters to filter the sales orders and store locations data.")
async def get_max_discount(state: str = Query(..., description="State name"), land_area: int = Query(..., description="Land area")):
    cursor.execute("SELECT MAX(T1.`Discount Applied`) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T2.State = ? AND T2.`Land Area` = ?", (state, land_area))
    result = cursor.fetchone()
    if not result:
        return {"max_discount": []}
    return {"max_discount": result[0]}

# Endpoint to get the count of distinct time zones in a specific region
@app.get("/v1/regional_sales/count_distinct_time_zones_region", operation_id="get_count_distinct_time_zones", summary="Retrieves the number of unique time zones found in the specified region. The region is identified by its name, and the count is determined by considering the time zones of all store locations within that region.")
async def get_count_distinct_time_zones(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.`Time Zone`) FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode WHERE T1.Region = ?", (region,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the type of store location with the maximum population
@app.get("/v1/regional_sales/store_type_max_population", operation_id="get_store_type_max_population", summary="Retrieves the type of store location that has the highest population in each region. This operation uses the provided region data to determine the store type with the maximum population. The result is a distinct list of store types that have the highest population in their respective regions.")
async def get_store_type_max_population():
    cursor.execute("SELECT DISTINCT CASE WHEN MAX(T2.Population) THEN T2.Type END FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode")
    result = cursor.fetchone()
    if not result:
        return {"store_type": []}
    return {"store_type": result[0]}

# Endpoint to get the top region based on order count for a specific warehouse code
@app.get("/v1/regional_sales/top_region_order_count_warehouse", operation_id="get_top_region_order_count", summary="Retrieves the region with the highest order count for a given warehouse. The warehouse is identified by its unique code.")
async def get_top_region_order_count(warehouse_code: str = Query(..., description="Warehouse code")):
    cursor.execute("SELECT T2.Region FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T1.WarehouseCode = ? GROUP BY T2.Region ORDER BY COUNT(T1.OrderNumber) DESC LIMIT 1", (warehouse_code,))
    result = cursor.fetchone()
    if not result:
        return {"region": []}
    return {"region": result[0]}

# Endpoint to get the city with the highest unit price
@app.get("/v1/regional_sales/city_highest_unit_price", operation_id="get_city_highest_unit_price", summary="Retrieves the city with the highest unit price from the sales orders. The operation filters the sales orders based on the highest unit price and identifies the corresponding city from the store locations. The result is the city with the highest unit price.")
async def get_city_highest_unit_price():
    cursor.execute("SELECT T2.`City Name` FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE REPLACE(T1.`Unit Price`, ',', '') = ( SELECT REPLACE(T1.`Unit Price`, ',', '') FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID ORDER BY REPLACE(T1.`Unit Price`, ',', '') DESC LIMIT 1 ) ORDER BY REPLACE(T1.`Unit Price`, ',', '') DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the sum of orders based on sales channel, customer name, and order date pattern
@app.get("/v1/regional_sales/sum_orders_sales_channel_customer_date", operation_id="get_sum_orders", summary="Retrieves the total number of orders for a specific sales channel, customer, and date pattern. The operation calculates the sum of orders that match the provided sales channel, customer name, and order date pattern. The date pattern should be provided in the 'M/%/YY' format.")
async def get_sum_orders(sales_channel: str = Query(..., description="Sales channel"), customer_name: str = Query(..., description="Customer name"), order_date_pattern: str = Query(..., description="Order date pattern in 'M/%/YY' format")):
    cursor.execute("SELECT SUM(CASE WHEN T1.`Sales Channel` = ? AND T2.`Customer Names` = ? AND T1.OrderDate LIKE ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID", (sales_channel, customer_name, order_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the sum of orders based on order quantity, sales channel, and county
@app.get("/v1/regional_sales/sum_orders_quantity_channel_county", operation_id="get_sum_orders_quantity_channel_county", summary="Retrieves the total number of orders that match the specified order quantity, sales channel, and county. This operation calculates the sum based on the provided criteria, offering insights into sales performance across different channels and regions.")
async def get_sum_orders_quantity_channel_county(order_quantity: int = Query(..., description="Order quantity"), sales_channel: str = Query(..., description="Sales channel"), county: str = Query(..., description="County")):
    cursor.execute("SELECT SUM(CASE WHEN T1.`Order Quantity` = ? AND T1.`Sales Channel` = ? AND T2.County = ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID", (order_quantity, sales_channel, county))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the least popular product name in a given city
@app.get("/v1/regional_sales/least_popular_product_by_city", operation_id="get_least_popular_product_by_city", summary="Retrieves the name of the product with the least sales in a specified city. The operation identifies the product with the lowest sales count by joining the Products, Sales Orders, and Store Locations tables based on the provided city name. The result is the name of the product with the least sales in the given city.")
async def get_least_popular_product_by_city(city_name: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T1.`Product Name` FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID INNER JOIN `Store Locations` AS T3 ON T3.StoreID = T2._StoreID WHERE T3.`City Name` = ? GROUP BY T1.`Product Name` ORDER BY COUNT(T1.`Product Name`) ASC LIMIT 1", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the latitude and longitude of the store with the least orders for a given warehouse code
@app.get("/v1/regional_sales/store_location_by_warehouse_code", operation_id="get_store_location_by_warehouse_code", summary="Retrieves the geographical coordinates of the store with the fewest orders associated with the specified warehouse code. The operation identifies the store with the least order count for the given warehouse code and returns its latitude and longitude.")
async def get_store_location_by_warehouse_code(warehouse_code: str = Query(..., description="Warehouse code")):
    cursor.execute("SELECT T2.Latitude, T2.Longitude FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T1.WarehouseCode = ? GROUP BY T2.StoreID ORDER BY COUNT(T1.WarehouseCode) ASC LIMIT 1", (warehouse_code,))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": {"latitude": result[0], "longitude": result[1]}}

# Endpoint to get the percentage of orders from a specific state on a given date
@app.get("/v1/regional_sales/percentage_orders_by_state_and_date", operation_id="get_percentage_orders_by_state_and_date", summary="Retrieves the percentage of total orders placed in a specific state on a given date. The calculation is based on the total number of orders made on that date and the number of orders from the specified state. The state is identified by its name, and the date is provided in 'MM/DD/YY' format.")
async def get_percentage_orders_by_state_and_date(state: str = Query(..., description="State name"), order_date: str = Query(..., description="Order date in 'MM/DD/YY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.State = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.OrderNumber) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T1.OrderDate = ?", (state, order_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average land area of stores with a specific unit price
@app.get("/v1/regional_sales/average_land_area_by_unit_price", operation_id="get_average_land_area_by_unit_price", summary="Get the average land area of stores with a specific unit price")
async def get_average_land_area_by_unit_price(unit_price: float = Query(..., description="Unit price")):
    cursor.execute("SELECT AVG(T2.`Land Area`) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T1.`Unit Price` = ?", (unit_price,))
    result = cursor.fetchone()
    if not result:
        return {"average_land_area": []}
    return {"average_land_area": result[0]}

# Endpoint to get the average household income in a specific state and type
@app.get("/v1/regional_sales/average_household_income_by_state_and_type", operation_id="get_average_household_income_by_state_and_type", summary="Retrieves the average household income for a specific type of location within a given state. The operation calculates the average income based on the provided state and location type, using data from the Regions and Store Locations tables.")
async def get_average_household_income_by_state_and_type(state: str = Query(..., description="State name"), type: str = Query(..., description="Type of location (e.g., City)")):
    cursor.execute("SELECT AVG(T2.`Household Income`) FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode WHERE T2.State = ? AND T2.Type = ?", (state, type))
    result = cursor.fetchone()
    if not result:
        return {"average_household_income": []}
    return {"average_household_income": result[0]}

# Endpoint to get distinct order numbers after a specific date
@app.get("/v1/regional_sales/distinct_order_numbers_after_date", operation_id="get_distinct_order_numbers_after_date", summary="Retrieves a list of unique order numbers that were created after a specified date. The input parameter 'order_date' in 'MM/DD/YY' format is used to filter the orders. The operation returns only the distinct order numbers that meet the specified date criteria.")
async def get_distinct_order_numbers_after_date(order_date: str = Query(..., description="Order date in 'MM/DD/YY' format")):
    cursor.execute("SELECT DISTINCT T FROM ( SELECT CASE  WHEN OrderDate > ? THEN OrderNumber ELSE NULL END AS T FROM `Sales Orders` ) WHERE T IS NOT NULL", (order_date,))
    result = cursor.fetchall()
    if not result:
        return {"order_numbers": []}
    return {"order_numbers": [row[0] for row in result]}

# Endpoint to get the count of sales channels in a specific region
@app.get("/v1/regional_sales/count_sales_channels_by_region", operation_id="get_count_sales_channels_by_region", summary="Retrieves the total number of unique sales channels operating in the specified region. The region is identified by its name.")
async def get_count_sales_channels_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT COUNT(T1.`Sales Channel`) FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T2.Region = ?", (region,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sales team with the highest unit price
@app.get("/v1/regional_sales/sales_team_with_highest_unit_price", operation_id="get_sales_team_with_highest_unit_price", summary="Retrieves the sales team with the highest unit price from the sales orders. The unit price is determined by removing any commas from the price and comparing it with other sales orders. The sales team with the highest unit price is then returned.")
async def get_sales_team_with_highest_unit_price():
    cursor.execute("SELECT T2.`Sales Team` FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE REPLACE(T1.`Unit Price`, ',', '') = ( SELECT REPLACE(T1.`Unit Price`, ',', '') FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID ORDER BY REPLACE(T1.`Unit Price`, ',', '') DESC LIMIT 1 ) ORDER BY REPLACE(T1.`Unit Price`, ',', '') DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"sales_team": []}
    return {"sales_team": result[0]}

# Endpoint to get the region with the highest discount applied for a specific sales channel
@app.get("/v1/regional_sales/region_with_highest_discount_by_sales_channel", operation_id="get_region_with_highest_discount_by_sales_channel", summary="Retrieves the region with the highest discount applied for a given sales channel. The operation filters sales orders by the specified sales channel and identifies the region associated with the sales team that has the highest discount applied. The result is a single region that meets the criteria.")
async def get_region_with_highest_discount_by_sales_channel(sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT T2.Region FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T1.`Sales Channel` = ? ORDER BY T1.`Discount Applied` DESC LIMIT 1", (sales_channel,))
    result = cursor.fetchone()
    if not result:
        return {"region": []}
    return {"region": result[0]}

# Endpoint to get the order number and order date for a specific customer
@app.get("/v1/regional_sales/order_details_by_customer", operation_id="get_order_details_by_customer", summary="Retrieve the most recent order details for a specific customer, including the order number and date. The order details are sorted by unit price in descending order.")
async def get_order_details_by_customer(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT T1.OrderNumber, T1.OrderDate FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID WHERE T2.`Customer Names` = ? ORDER BY T1.`Unit Price` DESC LIMIT 1", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"order_details": []}
    return {"order_details": result}

# Endpoint to get distinct order numbers and warehouse codes for a specific customer
@app.get("/v1/regional_sales/distinct_order_warehouse_by_customer", operation_id="get_distinct_order_warehouse_by_customer", summary="Retrieves unique order numbers and associated warehouse codes for a specified customer. The customer is identified by the provided customer name.")
async def get_distinct_order_warehouse_by_customer(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT DISTINCT T1.OrderNumber, T1.WarehouseCode FROM `Sales Orders` AS T1 INNER JOIN Customers AS T2 ON T2.CustomerID = T1._CustomerID WHERE T2.`Customer Names` = ?", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"order_warehouse": []}
    return {"order_warehouse": result}

# Endpoint to get distinct customer names for a specific product and sales channel
@app.get("/v1/regional_sales/distinct_customer_names_by_product_sales_channel", operation_id="get_distinct_customer_names_by_product_sales_channel", summary="Retrieve a unique list of customer names associated with a specific product and sales channel. This operation filters customers based on the provided product name and sales channel, ensuring that only distinct customer names are returned.")
async def get_distinct_customer_names_by_product_sales_channel(product_name: str = Query(..., description="Name of the product"), sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T3.`Product Name` = ? AND T2.`Sales Channel` = ? THEN T1.`Customer Names` END AS T FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID INNER JOIN Products T3 ON T3.ProductID = T2._ProductID ) WHERE T IS NOT NULL", (product_name, sales_channel))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": result}

# Endpoint to get the store ID with the highest profit margin in a specific state
@app.get("/v1/regional_sales/store_id_highest_profit_margin_by_state", operation_id="get_store_id_highest_profit_margin_by_state", summary="Retrieves the store ID of the store with the highest profit margin in a specified state. The profit margin is calculated as the difference between the unit price and unit cost of sales orders. The state is provided as an input parameter.")
async def get_store_id_highest_profit_margin_by_state(state: str = Query(..., description="State")):
    cursor.execute("SELECT T2.StoreID FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID WHERE T2.State = ? ORDER BY T1.`Unit Price` - T1.`Unit Cost` DESC LIMIT 1", (state,))
    result = cursor.fetchone()
    if not result:
        return {"store_id": []}
    return {"store_id": result[0]}

# Endpoint to get the difference in total unit price between two states for a specific product
@app.get("/v1/regional_sales/unit_price_difference_by_states_product", operation_id="get_unit_price_difference_by_states_product", summary="Retrieves the difference in total unit price for a specific product between two states. The operation compares the total unit price of the product in the first state with the total unit price of the product in the second state. The result is the difference between these two values.")
async def get_unit_price_difference_by_states_product(state1: str = Query(..., description="First state"), state2: str = Query(..., description="Second state"), product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(CASE WHEN T3.State = ? THEN T2.`Unit Price` ELSE 0 END) - SUM(CASE WHEN T3.State = ? THEN T2.`Unit Price` ELSE 0 END) FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID INNER JOIN `Store Locations` AS T3 ON T3.StoreID = T2._StoreID WHERE T1.`Product Name` = ?", (state1, state2, product_name))
    result = cursor.fetchone()
    if not result:
        return {"price_difference": []}
    return {"price_difference": result[0]}

# Endpoint to get distinct sales teams in a specific region with order quantity greater than a specified value
@app.get("/v1/regional_sales/distinct_sales_teams_by_region_order_quantity", operation_id="get_distinct_sales_teams_by_region_order_quantity", summary="Retrieves a list of unique sales teams from a specific region that have processed orders exceeding a certain quantity. The operation requires the region and the minimum order quantity as input parameters.")
async def get_distinct_sales_teams_by_region_order_quantity(region: str = Query(..., description="Region"), order_quantity: int = Query(..., description="Order quantity")):
    cursor.execute("SELECT DISTINCT T2.`Sales Team` FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T2.Region = ? AND T1.`Order Quantity` > ?", (region, order_quantity))
    result = cursor.fetchall()
    if not result:
        return {"sales_teams": []}
    return {"sales_teams": result}

# Endpoint to get distinct store IDs based on state, sales channel, and discount applied
@app.get("/v1/regional_sales/distinct_store_ids_by_state_sales_channel_discount", operation_id="get_distinct_store_ids_by_state_sales_channel_discount", summary="Retrieve a unique list of store IDs that have sales in a specific state, through a particular sales channel, and with a certain discount applied. This operation filters sales data based on the provided state, sales channel, and discount, and returns the distinct store IDs that meet these criteria.")
async def get_distinct_store_ids_by_state_sales_channel_discount(state: str = Query(..., description="State"), sales_channel: str = Query(..., description="Sales channel"), discount_applied: float = Query(..., description="Discount applied")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.State = ? AND T1.`Sales Channel` = ? AND T1.`Discount Applied` = ? THEN T2.StoreID END AS T FROM `Sales Orders` T1 INNER JOIN `Store Locations` T2 ON T2.StoreID = T1._StoreID ) WHERE T IS NOT NULL", (state, sales_channel, discount_applied))
    result = cursor.fetchall()
    if not result:
        return {"store_ids": []}
    return {"store_ids": result}

# Endpoint to get the customer name with the highest order quantity in specific years
@app.get("/v1/regional_sales/customer_name_highest_order_quantity_by_years", operation_id="get_customer_name_highest_order_quantity_by_years", summary="Retrieves the name of the customer who placed the highest order quantity in the specified years. The operation filters sales orders by the provided years and identifies the customer with the maximum order quantity. The response includes the customer's name.")
async def get_customer_name_highest_order_quantity_by_years(year1: str = Query(..., description="First year in 'YY' format"), year2: str = Query(..., description="Second year in 'YY' format"), year3: str = Query(..., description="Third year in 'YY' format")):
    cursor.execute("SELECT T1.`Customer Names` FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID WHERE T2.OrderDate LIKE ? OR T2.OrderDate LIKE ? OR T2.OrderDate LIKE ? ORDER BY T2.`Order Quantity` DESC LIMIT 1", (f'%/{year1}', f'%/{year2}', f'%/{year3}'))
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the total order quantity and percentage for a specific product
@app.get("/v1/regional_sales/order_quantity_percentage_by_product", operation_id="get_order_quantity_percentage_by_product", summary="Retrieves the total order quantity and corresponding percentage for a specified product, calculated based on the sum of order quantities across all store locations.")
async def get_order_quantity_percentage_by_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(CASE WHEN T1.`Product Name` = ? THEN T2.`Order Quantity` ELSE 0 END), CAST(SUM(CASE WHEN T1.`Product Name` = ? THEN T2.`Order Quantity` ELSE 0 END) AS REAL) * 100 / SUM(T2.`Order Quantity`) FROM Products AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._ProductID = T1.ProductID INNER JOIN `Store Locations` AS T3 ON T3.StoreID = T2._StoreID", (product_name, product_name))
    result = cursor.fetchone()
    if not result:
        return {"order_quantity_percentage": []}
    return {"order_quantity": result[0], "percentage": result[1]}

# Endpoint to get distinct regions for a specific sales team
@app.get("/v1/regional_sales/distinct_regions_by_sales_team", operation_id="get_distinct_regions_by_sales_team", summary="Retrieves a list of unique regions where the specified sales team has made sales. The operation filters the sales data to identify distinct regions associated with the given sales team, excluding any null or empty values.")
async def get_distinct_regions_by_sales_team(sales_team: str = Query(..., description="Name of the sales team")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN `Sales Team` = ? THEN Region ELSE NULL END AS T FROM `Sales Team` ) WHERE T IS NOT NULL", (sales_team,))
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": result}

# Endpoint to get the StoreID with the maximum population
@app.get("/v1/regional_sales/store_with_max_population", operation_id="get_store_with_max_population", summary="Retrieves the unique identifier of the store location with the highest population count. This operation provides a quick way to identify the store with the most populous area, which can be useful for resource allocation or market analysis.")
async def get_store_with_max_population():
    cursor.execute("SELECT CASE WHEN MAX(Population) THEN StoreID END FROM `Store Locations`")
    result = cursor.fetchone()
    if not result:
        return {"StoreID": []}
    return {"StoreID": result[0]}

# Endpoint to get the Type of store with the maximum water area
@app.get("/v1/regional_sales/store_type_with_max_water_area", operation_id="get_store_type_with_max_water_area", summary="Retrieves the type of store that has the largest water area. The water area is a measure of the size of the water body near the store location. This operation does not require any input parameters and returns the store type with the maximum water area.")
async def get_store_type_with_max_water_area():
    cursor.execute("SELECT CASE WHEN MAX(`Water Area`) THEN Type END FROM `Store Locations`")
    result = cursor.fetchone()
    if not result:
        return {"Type": []}
    return {"Type": result[0]}

# Endpoint to get the count of sales orders with a specific ship date pattern and sales channel
@app.get("/v1/regional_sales/count_sales_orders_by_ship_date_and_channel", operation_id="get_count_sales_orders", summary="Retrieves the total number of sales orders that match a specified ship date pattern and sales channel. The ship date pattern should be provided in 'MM/DD/YY' format. This operation is useful for tracking sales volume based on shipping date and sales channel.")
async def get_count_sales_orders(ship_date_pattern: str = Query(..., description="Ship date pattern in 'MM/DD/YY' format"), sales_channel: str = Query(..., description="Sales channel")):
    cursor.execute("SELECT SUM(IIF(ShipDate LIKE ? AND `Sales Channel` = ?, 1, 0)) FROM `Sales Orders`", (ship_date_pattern, sales_channel))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the discount applied for the highest unit price sales order
@app.get("/v1/regional_sales/discount_applied_highest_unit_price", operation_id="get_discount_applied_highest_unit_price", summary="Retrieves the discount applied to the sales order with the highest unit price. The unit price is converted to a numerical format for accurate comparison and sorting. The result is the discount applied to the sales order with the highest unit price.")
async def get_discount_applied_highest_unit_price():
    cursor.execute("SELECT `Discount Applied` FROM `Sales Orders` WHERE REPLACE(`Unit Price`, ',', '') = ( SELECT REPLACE(`Unit Price`, ',', '') FROM `Sales Orders` ORDER BY REPLACE(`Unit Price`, ',', '') DESC LIMIT 1 ) ORDER BY REPLACE(`Unit Price`, ',', '') DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"Discount Applied": []}
    return {"Discount Applied": result[0]}

# Endpoint to get the product name with the highest profit margin
@app.get("/v1/regional_sales/product_with_highest_profit_margin", operation_id="get_product_with_highest_profit_margin", summary="Retrieves the name of the product with the highest profit margin from the sales orders and products tables. The profit margin is calculated by subtracting the unit cost from the unit price, with commas removed for accurate calculation. The result is sorted in descending order and the top product is returned.")
async def get_product_with_highest_profit_margin():
    cursor.execute("SELECT T2.`Product Name` FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID ORDER BY REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '') DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"Product Name": []}
    return {"Product Name": result[0]}

# Endpoint to get the average household income for the state with the most stores in a given region
@app.get("/v1/regional_sales/avg_household_income_by_region", operation_id="get_avg_household_income_by_region", summary="Retrieves the average household income for the state with the highest store count in a specified region. The operation calculates the average income based on the state with the most stores within the given region.")
async def get_avg_household_income_by_region(region: str = Query(..., description="Region name")):
    cursor.execute("SELECT AVG(T2.`Household Income`) FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode WHERE T1.Region = ? GROUP BY T2.State ORDER BY COUNT(T2.StoreID) DESC LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"Average Household Income": []}
    return {"Average Household Income": result[0]}

# Endpoint to get distinct regions with median income below a specified value
@app.get("/v1/regional_sales/regions_with_median_income_below", operation_id="get_regions_with_median_income_below", summary="Retrieves a list of unique regions where the median income is below a specified threshold. This operation filters regions based on the median income of their respective store locations, providing a focused view of areas with lower income levels.")
async def get_regions_with_median_income_below(median_income: int = Query(..., description="Median income threshold")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.`Median Income` < ? THEN T1.Region END AS T FROM Regions T1 INNER JOIN `Store Locations` T2 ON T2.StateCode = T1.StateCode ) WHERE T IS NOT NULL", (median_income,))
    result = cursor.fetchall()
    if not result:
        return {"Regions": []}
    return {"Regions": [row[0] for row in result]}

# Endpoint to get the count of stores in a specific region with land area below a specified value
@app.get("/v1/regional_sales/count_stores_by_region_and_land_area", operation_id="get_count_stores_by_region_and_land_area", summary="Retrieves the total number of stores in a specified region where the land area is less than a given threshold. This operation aggregates data from the Regions and Store Locations tables, considering the region name and land area values provided as input parameters.")
async def get_count_stores_by_region_and_land_area(region: str = Query(..., description="Region name"), land_area: int = Query(..., description="Land area threshold")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Region = ? AND T2.`Land Area` < ? THEN 1 ELSE 0 END) FROM Regions AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StateCode = T1.StateCode", (region, land_area))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer name with the highest net profit
@app.get("/v1/regional_sales/customer_with_highest_net_profit", operation_id="get_customer_with_highest_net_profit", summary="Retrieves the name of the customer who has generated the highest net profit. This operation calculates the net profit for each customer by subtracting the unit cost from the unit price of their sales orders. The customer with the highest calculated net profit is then returned.")
async def get_customer_with_highest_net_profit():
    cursor.execute("SELECT `Customer Names` FROM ( SELECT T1.`Customer Names`, T2.`Unit Price` - T2.`Unit Cost` AS \"net profit\" FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID ) ORDER BY `net profit` DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"Customer Names": []}
    return {"Customer Names": result[0]}

# Endpoint to get the sales team with the highest number of orders in a specific year
@app.get("/v1/regional_sales/top_sales_team_by_year", operation_id="get_top_sales_team_by_year", summary="Retrieves the sales team with the highest number of orders in a given year, based on the provided order and ship date patterns. The result is determined by counting the number of orders for each sales team and selecting the team with the highest count.")
async def get_top_sales_team_by_year(order_date_pattern: str = Query(..., description="Order date pattern in 'MM/DD/YY' format"), ship_date_pattern: str = Query(..., description="Ship date pattern in 'MM/DD/YY' format")):
    cursor.execute("SELECT COUNT(T1.OrderNumber), T2.`Sales Team` FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T1.OrderDate LIKE ? AND T1.ShipDate LIKE ? GROUP BY T2.`Sales Team` ORDER BY COUNT(T1.OrderNumber) DESC LIMIT 1", (order_date_pattern, ship_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"Sales Team": []}
    return {"Sales Team": result[1], "Order Count": result[0]}

# Endpoint to get the product name based on order quantity and ship date
@app.get("/v1/regional_sales/product_name_by_order_quantity_ship_date", operation_id="get_product_name", summary="Retrieve the name of the product with the highest profit margin for orders that exceed a specified quantity and have a ship date matching a given pattern. The profit margin is calculated by subtracting the unit cost from the unit price. The order quantity and ship date pattern are provided as input parameters.")
async def get_product_name(order_quantity: int = Query(..., description="Order quantity"), ship_date_pattern: str = Query(..., description="Ship date pattern in 'MM/DD/YY' format")):
    cursor.execute("SELECT T2.`Product Name` FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID WHERE T1.`Order Quantity` > ? AND ShipDate LIKE ? ORDER BY REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '') ASC LIMIT 1", (order_quantity, ship_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get store locations based on unit price and unit cost
@app.get("/v1/regional_sales/store_locations_by_unit_price_cost", operation_id="get_store_locations", summary="Retrieves the geographical coordinates of a store location, ranked by the difference between the unit price and unit cost of its sales orders. The result is limited to a single store, excluding the top three ranked stores.")
async def get_store_locations():
    cursor.execute("SELECT T2.Latitude, T2.Longitude FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID ORDER BY REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '') DESC LIMIT 3, 1")
    result = cursor.fetchall()
    if not result:
        return {"locations": []}
    return {"locations": result}

# Endpoint to get the count of orders by sales team and ship date pattern
@app.get("/v1/regional_sales/order_count_by_sales_team_ship_date", operation_id="get_order_count_by_sales_team", summary="Retrieves the number of orders for each sales team that have a ship date matching the provided pattern. The results are sorted in descending order based on the order count, with the sales team having the highest order count listed first. Only the top sales team is returned.")
async def get_order_count_by_sales_team(ship_date_pattern: str = Query(..., description="Ship date pattern in '%/%/YY' format")):
    cursor.execute("SELECT COUNT(T1.OrderNumber), T2.`Sales Team` FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE T1.ShipDate LIKE ? GROUP BY T2.`Sales Team` ORDER BY COUNT(T1.OrderNumber) DESC LIMIT 1", (ship_date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0], "sales_team": result[1]}

# Endpoint to get the average order count for a specific sales team over multiple years
@app.get("/v1/regional_sales/average_order_count_by_sales_team", operation_id="get_average_order_count", summary="Get the average order count for a specific sales team over multiple years")
async def get_average_order_count(sales_team: str = Query(..., description="Sales team name"), ship_date_pattern_1: str = Query(..., description="Ship date pattern in '%/%/YY' format"), ship_date_pattern_2: str = Query(..., description="Ship date pattern in '%/%/YY' format"), ship_date_pattern_3: str = Query(..., description="Ship date pattern in '%/%/YY' format")):
    cursor.execute("SELECT CAST(COUNT(T1.OrderNumber) AS REAL) / 3 FROM `Sales Orders` AS T1 INNER JOIN `Sales Team` AS T2 ON T2.SalesTeamID = T1._SalesTeamID WHERE (T2.`Sales Team` = ? AND ShipDate LIKE ?) OR (T2.`Sales Team` = ? AND ShipDate LIKE ?) OR (T2.`Sales Team` = ? AND ShipDate LIKE ?)", (sales_team, ship_date_pattern_1, sales_team, ship_date_pattern_2, sales_team, ship_date_pattern_3))
    result = cursor.fetchone()
    if not result:
        return {"average_order_count": []}
    return {"average_order_count": result[0]}

# Endpoint to get the discounted unit price and product name based on unit price and unit cost
@app.get("/v1/regional_sales/discounted_unit_price_product_name", operation_id="get_discounted_unit_price_product_name", summary="Retrieves the discounted unit price and corresponding product name from the sales orders, sorted by the difference between the unit price and unit cost in descending order. The result is limited to the top record.")
async def get_discounted_unit_price_product_name():
    cursor.execute("SELECT T1.`Unit Price` * T1.`Discount Applied`, T2.`Product Name` FROM `Sales Orders` AS T1 INNER JOIN Products AS T2 ON T2.ProductID = T1._ProductID ORDER BY REPLACE(T1.`Unit Price`, ',', '') - REPLACE(T1.`Unit Cost`, ',', '') DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"discounted_unit_price": [], "product_name": []}
    return {"discounted_unit_price": result[0], "product_name": result[1]}

# Endpoint to get customer names based on total order value
@app.get("/v1/regional_sales/customer_names_by_order_value", operation_id="get_customer_names_by_order_value", summary="Retrieves the top three customers with the highest total order value. The total order value is calculated by subtracting the discounted amount from the total order cost. The results are ordered in descending order based on the total order value.")
async def get_customer_names_by_order_value():
    cursor.execute("SELECT `Customer Names` FROM ( SELECT T1.`Customer Names` , REPLACE(T2.`Unit Price`, ',', '') * T2.`Order Quantity` - REPLACE(T2.`Unit Price`, ',', '') * T2.`Discount Applied` AS T FROM Customers T1 INNER JOIN `Sales Orders` T2 ON T2._CustomerID = T1.CustomerID ) ORDER BY T DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the product name for a specific order number
@app.get("/v1/regional_sales/product_name_by_order_number", operation_id="get_product_name_by_order_number", summary="Retrieves the name of the product associated with a specific order number. This operation filters through the product catalog and sales orders to identify the product linked to the provided order number. The result is a distinct product name, ensuring no duplicates are returned.")
async def get_product_name_by_order_number(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE  WHEN T2.OrderNumber = ? THEN T1.`Product Name` ELSE NULL END AS T FROM Products T1 INNER JOIN `Sales Orders` T2 ON T2._ProductID = T1.ProductID ) WHERE T IS NOT NULL", (order_number,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the sales team and city name for a specific order number
@app.get("/v1/regional_sales/sales_team_city_by_order_number", operation_id="get_sales_team_city_by_order_number", summary="Retrieves the sales team and city name associated with a specific order number. This operation fetches data from the 'Sales Orders' and 'Store Locations' tables, joining them based on the store ID. The 'Sales Team' table is also joined using the sales team ID. The order number is used as a filter to return the relevant sales team and city name.")
async def get_sales_team_city_by_order_number(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT T3.`Sales Team`, T1.`City Name` FROM `Store Locations` AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._StoreID = T1.StoreID INNER JOIN `Sales Team` AS T3 ON T3.SalesTeamID = T2._SalesTeamID WHERE T2.OrderNumber = ?", (order_number,))
    result = cursor.fetchone()
    if not result:
        return {"sales_team": [], "city_name": []}
    return {"sales_team": result[0], "city_name": result[1]}

# Endpoint to get customer names based on city name and procured date pattern
@app.get("/v1/regional_sales/customer_names_by_city_procured_date", operation_id="get_customer_names_by_city_procured_date", summary="Retrieves the name of the customer who procured the highest-margin product in the specified city, based on the provided procured date pattern. The margin is calculated by subtracting the unit cost from the unit price. The date pattern should be provided in the format 'MM/DD/YY'.")
async def get_customer_names_by_city_procured_date(city_name: str = Query(..., description="City name"), procured_date_pattern: str = Query(..., description="Procured date pattern in '%/%/YY' format")):
    cursor.execute("SELECT T1.`Customer Names` FROM Customers AS T1 INNER JOIN `Sales Orders` AS T2 ON T2._CustomerID = T1.CustomerID INNER JOIN `Store Locations` AS T3 ON T3.StoreID = T2._StoreID WHERE T3.`City Name` = ? AND T2.ProcuredDate LIKE ? ORDER BY REPLACE(T2.`Unit Price`, ',', '') - REPLACE(T2.`Unit Cost`, ',', '') DESC LIMIT 1", (city_name, procured_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"customer_names": []}
    return {"customer_names": result[0]}

# Endpoint to get the sum of orders based on city name and order date pattern
@app.get("/v1/regional_sales/sum_orders_city_date", operation_id="get_sum_orders_city_date", summary="Retrieves the total number of orders for a specific city and date pattern. The operation calculates the sum based on the provided city name and order date pattern, which should be in the format 'MM/DD/YY'. This endpoint is useful for analyzing sales trends in specific cities and time periods.")
async def get_sum_orders_city_date(city_name: str = Query(..., description="City name"), order_date_pattern: str = Query(..., description="Order date pattern in '%%/%%/YY' format")):
    cursor.execute("SELECT SUM(CASE WHEN T2.`City Name` = ? AND T1.OrderDate LIKE ? THEN 1 ELSE 0 END) FROM `Sales Orders` AS T1 INNER JOIN `Store Locations` AS T2 ON T2.StoreID = T1._StoreID", (city_name, order_date_pattern))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the average household income for a specific city
@app.get("/v1/regional_sales/avg_household_income_city", operation_id="get_avg_household_income_city", summary="Retrieves the average household income for a specified city. The operation calculates the mean value from the 'Store Locations' data, filtering results based on the provided city name.")
async def get_avg_household_income_city(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT AVG(`Household Income`) FROM `Store Locations` WHERE `City Name` = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the maximum discount applied for a specific order date pattern
@app.get("/v1/regional_sales/max_discount_applied_date", operation_id="get_max_discount_applied_date", summary="Retrieves the highest discount applied to any sales order that matches the specified order date pattern. The order date pattern should be provided in the 'MM/DD/YY' format.")
async def get_max_discount_applied_date(order_date_pattern: str = Query(..., description="Order date pattern in '%%/%%/YY' format")):
    cursor.execute("SELECT MAX(`Discount Applied`) FROM `Sales Orders` WHERE OrderDate LIKE ?", (order_date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"max_discount": []}
    return {"max_discount": result[0]}

api_calls = [
    "/v1/regional_sales/top_region_by_sales_teams?limit=1",
    "/v1/regional_sales/customer_names_containing_substring?substring=Group",
    "/v1/regional_sales/average_median_income_by_type?type=City",
    "/v1/regional_sales/sales_team_region_by_order_number?order_number=SO%20-%20000137",
    "/v1/regional_sales/distinct_products_by_sales_team?sales_team=Douglas%20Tucker",
    "/v1/regional_sales/distinct_customer_names_by_product_order_date_discount?product_name=Cocktail%20Glasses&order_date_suffix=20&discount_applied=0.1",
    "/v1/regional_sales/distinct_order_numbers_city_names_by_sales_channel?sales_channel=In-Store",
    "/v1/regional_sales/order_details_highest_unit_cost?limit=1",
    "/v1/regional_sales/distinct_order_numbers_by_date_pattern_customer?order_date_pattern=%25/%25/18&customer_name=Rochester%20Ltd",
    "/v1/regional_sales/distinct_product_names_sales_teams_by_warehouse_code?warehouse_code=WARE-NMK1003",
    "/v1/regional_sales/customer_names_by_sales_channel?sales_channel=Online",
    "/v1/regional_sales/average_profit_margin_by_product?product_name=Bakeware",
    "/v1/regional_sales/top_sales_team_by_profit_margin?order_date=%25/%25/20",
    "/v1/regional_sales/order_profit_margins_by_sales_team?sales_team=Joshua%20Bennett",
    "/v1/regional_sales/percentage_orders_by_product_shipped?product_name=Home%20Fragrances&ship_date=7/%25/18",
    "/v1/regional_sales/distinct_customers_by_name_pattern?name_pattern=W%25",
    "/v1/regional_sales/product_ids_by_name_pattern?name_pattern=%25Outdoor%25",
    "/v1/regional_sales/count_distinct_orders?sales_channel=In-Store&warehouse_code=WARE-NMK1003&discount_applied=0.4",
    "/v1/regional_sales/highest_population_city_by_state?state=Florida",
    "/v1/regional_sales/distinct_stores_by_county?county=Allen%20County",
    "/v1/regional_sales/distinct_store_locations_by_type?type1=Borough&type2=CDP",
    "/v1/regional_sales/sales_team_info_by_id_and_channels?sales_team_id=18&sales_channel1=In-Store&sales_channel2=Online",
    "/v1/regional_sales/percentage_in_store_sales_by_customer?sales_channel=In-Store&customer_name=Medline",
    "/v1/regional_sales/customer_delivery_dates_by_channel_product_order_date?sales_channel=Wholesale&product_name=Bedroom%20Furniture&order_date_pattern=%25/%25/19",
    "/v1/regional_sales/distinct_customer_product_by_profit_margin?profit_margin=3800",
    "/v1/regional_sales/distinct_city_names_by_state_water_area?state=California&water_area=0",
    "/v1/regional_sales/percentage_orders_by_sales_team?sales_team=Carlos%20Miller",
    "/v1/regional_sales/count_specific_products?product_name1=Platters&product_name2=Serveware",
    "/v1/regional_sales/total_profit_margin_highest_median_income",
    "/v1/regional_sales/count_orders_by_sales_team_region?region=South&min_sales_team_id=1&max_sales_team_id=9",
    "/v1/regional_sales/sum_orders_by_date?order_date=5%2F31%2F18",
    "/v1/regional_sales/distinct_order_numbers_by_delivery_date?delivery_date=6%2F13%2F18",
    "/v1/regional_sales/sum_orders_by_quantity?order_quantity=5",
    "/v1/regional_sales/distinct_states_by_state_code?state_code=GA",
    "/v1/regional_sales/count_distinct_state_codes_by_region?region=Midwest",
    "/v1/regional_sales/distinct_product_names_by_unit_cost?unit_cost=781.22",
    "/v1/regional_sales/distinct_delivery_dates_by_product_name?product_name=Cookware",
    "/v1/regional_sales/sum_orders_by_date_pattern_and_product_name?order_date_pattern=%25%2F%25%2F18&product_name=Furniture%20Cushions",
    "/v1/regional_sales/distinct_product_names_by_discount?discount_applied=0.1",
    "/v1/regional_sales/average_profit_margin?product_name=Phones&sales_channel=Distributor",
    "/v1/regional_sales/average_profit_per_order?product_name=Bar%20Tools&min_order_quantity=5",
    "/v1/regional_sales/distinct_city_names_by_region?region=South",
    "/v1/regional_sales/distinct_regions_by_store_type?store_type=Town",
    "/v1/regional_sales/sales_order_count_by_customer?customer_name=Medsep%20Group",
    "/v1/regional_sales/distinct_discounts_by_customer?customer_name=Ole%20Group",
    "/v1/regional_sales/distinct_customer_names_by_ship_date?ship_date=7%2F8%2F18",
    "/v1/regional_sales/order_count_by_quantity_and_customer?min_order_quantity=4&customer_name=Ei",
    "/v1/regional_sales/order_count_by_discount_and_customer?discount_applied=0.05&customer_name=Pacific%20Ltd",
    "/v1/regional_sales/distinct_customer_names_by_unit_cost?min_unit_cost=4000",
    "/v1/regional_sales/store_details_by_city?city_name=Birmingham",
    "/v1/regional_sales/city_with_highest_population?limit=1",
    "/v1/regional_sales/count_store_locations_state_type?state=California&type=CDP",
    "/v1/regional_sales/order_product_lowest_unit_price?limit=1",
    "/v1/regional_sales/product_highest_profit_margin?year=19&limit=1",
    "/v1/regional_sales/average_unit_price_product?product_name=Cookware",
    "/v1/regional_sales/sales_team_specific_order_date?order_date=5/31/18",
    "/v1/regional_sales/sales_team_fewest_orders_year?year=19&limit=1",
    "/v1/regional_sales/year_most_orders_sales_team?sales_team=George%20Lewis&limit=1",
    "/v1/regional_sales/percentage_orders_county_year?county=Orange%20County&year=18",
    "/v1/regional_sales/order_highest_unit_price?limit=1",
    "/v1/regional_sales/top_sales_team_by_order_date?order_date_pattern=%25/%25/18",
    "/v1/regional_sales/distinct_unit_costs_by_order_number?order_number=SO%20-%20000103",
    "/v1/regional_sales/order_count_by_county_and_date?county=Maricopa%20County&order_date_pattern=%25/%25/20",
    "/v1/regional_sales/store_location_by_order_number?order_number=SO%20-%20000115",
    "/v1/regional_sales/order_count_by_city_and_date?order_date_pattern=%25/%25/19",
    "/v1/regional_sales/distinct_customer_names_by_order_count_and_date?min_order_count=3&order_date_pattern=%25/%25/18",
    "/v1/regional_sales/order_count_by_customer_and_years?year1=18&year2=19&year3=20&customer_name=Medsep%20Group",
    "/v1/regional_sales/distinct_customer_names_by_order_quantity_and_date?min_order_quantity=5&order_date=6/1/18",
    "/v1/regional_sales/percentage_orders_by_price_difference_and_sales_team?price_difference=1000&sales_team=Stephen%20Payne",
    "/v1/regional_sales/sales_team_count_by_region?region=Northeast",
    "/v1/regional_sales/store_locations_by_county?county=Maricopa%20County",
    "/v1/regional_sales/order_numbers_highest_unit_cost",
    "/v1/regional_sales/product_names_by_id_range?min_product_id=30&max_product_id=40",
    "/v1/regional_sales/unit_cost_ratio",
    "/v1/regional_sales/most_ordered_product_by_date?order_date_pattern=%25/%25/18",
    "/v1/regional_sales/sales_order_count_by_team_member?sales_team_member=Adam%20Hernandez",
    "/v1/regional_sales/order_numbers_by_city?city_name=Daly%20City",
    "/v1/regional_sales/highest_order_quantity_by_customer?customer_name=Qualitest%20",
    "/v1/regional_sales/order_product_by_sales_channel?sales_channel=In-Store",
    "/v1/regional_sales/sum_orders_by_date_channel_city?order_date_pattern=5/%/18&sales_channel=Online&city_name=Norman",
    "/v1/regional_sales/product_with_lowest_order_quantity_in_county?county=Maricopa%20County",
    "/v1/regional_sales/distinct_order_numbers_by_sales_team?sales_team=Samuel%20Fowler",
    "/v1/regional_sales/count_orders_by_product_and_date?product_name=Baseball&order_date_pattern=12/%/18",
    "/v1/regional_sales/average_monthly_order_quantity?product_name=Ornaments&order_date_pattern=%25/%25/18",
    "/v1/regional_sales/percentage_order_quantity_by_city_and_date?city_name=Burbank&order_date_pattern=%25/%25/18",
    "/v1/regional_sales/difference_in_order_counts_by_warehouse?warehouse_code_1=WARE-MKL1006&warehouse_code_2=WARE-NBV1002",
    "/v1/regional_sales/distinct_product_names_by_customer_and_delivery_date?delivery_date_pattern=%25/%25/21&customer_name=Sundial",
    "/v1/regional_sales/distinct_store_ids_and_regions_by_state?state=Michigan",
    "/v1/regional_sales/count_orders_by_two_customers?customer_name_1=Apollo%20Ltd&customer_name_2=Pacific%20Ltd",
    "/v1/regional_sales/top_store_by_order_count?city_name_1=Aurora%20(Township)&city_name_2=Babylon%20(Town)",
    "/v1/regional_sales/customer_product_by_sales_team_channel?sales_team=Anthony%20Torres&sales_channel=Distributor",
    "/v1/regional_sales/customer_by_sales_channel_profit?sales_channel=Online&min_profit=5000",
    "/v1/regional_sales/total_profit_by_product_year?delivery_year=21&product_name=Floral",
    "/v1/regional_sales/order_count_by_population_range?min_population=3000000&max_population=4000000",
    "/v1/regional_sales/product_names_by_time_zone_sales_channel?time_zone=Pacific/Honolulu&sales_channel=Wholesale",
    "/v1/regional_sales/order_product_by_date?order_date=6/6/18",
    "/v1/regional_sales/average_order_count_per_year?customer_name=Weimei%20Corp&year_1=18&year_2=19&year_3=20",
    "/v1/regional_sales/warehouse_statistics?warehouse_code=WARE-NMK1003&product_name=Floor%20Lamps&order_year=19",
    "/v1/regional_sales/procurement_dates_by_customer?customer_id=11",
    "/v1/regional_sales/sum_orders_quantity_channel?order_quantity=1&sales_channel=Distributor",
    "/v1/regional_sales/distinct_sales_team_ids?discount_applied=0.1&sales_channel=In-Store",
    "/v1/regional_sales/sum_store_locations?population=3000000&type=Borough&city_name=Brooklyn",
    "/v1/regional_sales/count_distinct_states?region=Midwest",
    "/v1/regional_sales/top_products_by_profit_margin?limit=10",
    "/v1/regional_sales/distinct_sales_teams_customer?customer_name=Apotheca,%20Ltd",
    "/v1/regional_sales/distinct_regions_warehouse_code?warehouse_code=WARE-UHY1004",
    "/v1/regional_sales/product_customer_names?order_date=10/21/18&delivery_date=11/21/19",
    "/v1/regional_sales/sum_procured_orders_date_city?procured_date=10%2F27%2F18&city_name=Orlando",
    "/v1/regional_sales/top_sales_channel_median_income?limit=3",
    "/v1/regional_sales/top_sales_teams_profit_margin?limit=5",
    "/v1/regional_sales/max_discount_state_land_area?state=Colorado&land_area=111039036",
    "/v1/regional_sales/count_distinct_time_zones_region?region=Northeast",
    "/v1/regional_sales/store_type_max_population",
    "/v1/regional_sales/top_region_order_count_warehouse?warehouse_code=WARE-MKL1006",
    "/v1/regional_sales/city_highest_unit_price",
    "/v1/regional_sales/sum_orders_sales_channel_customer_date?sales_channel=Online&customer_name=Ole%20Group&order_date_pattern=5%2F%25%2F19",
    "/v1/regional_sales/sum_orders_quantity_channel_county?order_quantity=1&sales_channel=Distributor&county=Washtenaw%20County",
    "/v1/regional_sales/least_popular_product_by_city?city_name=Santa%20Clarita",
    "/v1/regional_sales/store_location_by_warehouse_code?warehouse_code=WARE-PUJ1005",
    "/v1/regional_sales/percentage_orders_by_state_and_date?state=New%20York&order_date=4/4/20",
    "/v1/regional_sales/average_land_area_by_unit_price?unit_price=998.30",
    "/v1/regional_sales/average_household_income_by_state_and_type?state=New%20Hampshire&type=City",
    "/v1/regional_sales/distinct_order_numbers_after_date?order_date=1/1/18",
    "/v1/regional_sales/count_sales_channels_by_region?region=Midwest",
    "/v1/regional_sales/sales_team_with_highest_unit_price",
    "/v1/regional_sales/region_with_highest_discount_by_sales_channel?sales_channel=Online",
    "/v1/regional_sales/order_details_by_customer?customer_name=Apollo%20Ltd",
    "/v1/regional_sales/distinct_order_warehouse_by_customer?customer_name=Elorac%2C%20Corp",
    "/v1/regional_sales/distinct_customer_names_by_product_sales_channel?product_name=Cocktail%20Glasses&sales_channel=Online",
    "/v1/regional_sales/store_id_highest_profit_margin_by_state?state=Arizona",
    "/v1/regional_sales/unit_price_difference_by_states_product?state1=Florida&state2=Texas&product_name=Computers",
    "/v1/regional_sales/distinct_sales_teams_by_region_order_quantity?region=Midwest&order_quantity=5",
    "/v1/regional_sales/distinct_store_ids_by_state_sales_channel_discount?state=California&sales_channel=In-Store&discount_applied=0.2",
    "/v1/regional_sales/customer_name_highest_order_quantity_by_years?year1=18&year2=19&year3=20",
    "/v1/regional_sales/order_quantity_percentage_by_product?product_name=Candles",
    "/v1/regional_sales/distinct_regions_by_sales_team?sales_team=Joshua%20Bennett",
    "/v1/regional_sales/store_with_max_population",
    "/v1/regional_sales/store_type_with_max_water_area",
    "/v1/regional_sales/count_sales_orders_by_ship_date_and_channel?ship_date_pattern=6/%/18&sales_channel=Online",
    "/v1/regional_sales/discount_applied_highest_unit_price",
    "/v1/regional_sales/product_with_highest_profit_margin",
    "/v1/regional_sales/avg_household_income_by_region?region=Northeast",
    "/v1/regional_sales/regions_with_median_income_below?median_income=30000",
    "/v1/regional_sales/count_stores_by_region_and_land_area?region=West&land_area=20000000",
    "/v1/regional_sales/customer_with_highest_net_profit",
    "/v1/regional_sales/top_sales_team_by_year?order_date_pattern=%/%/19&ship_date_pattern=%/%/19",
    "/v1/regional_sales/product_name_by_order_quantity_ship_date?order_quantity=5&ship_date_pattern=5/%/19",
    "/v1/regional_sales/store_locations_by_unit_price_cost",
    "/v1/regional_sales/order_count_by_sales_team_ship_date?ship_date_pattern=%/%/20",
    "/v1/regional_sales/average_order_count_by_sales_team?sales_team=Carl%20Nguyen&ship_date_pattern_1=%/%/18&ship_date_pattern_2=%/%/19&ship_date_pattern_3=%/%/20",
    "/v1/regional_sales/discounted_unit_price_product_name",
    "/v1/regional_sales/customer_names_by_order_value",
    "/v1/regional_sales/product_name_by_order_number?order_number=SO%20-%200005951",
    "/v1/regional_sales/sales_team_city_by_order_number?order_number=SO%20-%200001004",
    "/v1/regional_sales/customer_names_by_city_procured_date?city_name=Gilbert&procured_date_pattern=%/%/19",
    "/v1/regional_sales/sum_orders_city_date?city_name=Chandler&order_date_pattern=%25/%25/20",
    "/v1/regional_sales/avg_household_income_city?city_name=Glendale",
    "/v1/regional_sales/max_discount_applied_date?order_date_pattern=%25/%25/20"
]
