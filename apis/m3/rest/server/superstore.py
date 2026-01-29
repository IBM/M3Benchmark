from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/superstore/superstore.sqlite')
cursor = conn.cursor()

# Endpoint to get distinct product names for a given order ID
@app.get("/v1/superstore/distinct_product_names_by_order_id", operation_id="get_distinct_product_names", summary="Retrieves a unique list of product names associated with a specific order ID. This operation fetches the distinct product names from the central superstore data, based on the provided order ID.")
async def get_distinct_product_names(order_id: str = Query(..., description="Order ID")):
    cursor.execute("SELECT DISTINCT T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Order ID` = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the longest time in days between ship date and order date for a given customer name
@app.get("/v1/superstore/longest_time_days_by_customer_name", operation_id="get_longest_time_days", summary="Retrieves the maximum duration in days between the ship date and order date for a specific customer. The customer is identified by the provided customer name.")
async def get_longest_time_days(customer_name: str = Query(..., description="Customer Name")):
    cursor.execute("SELECT MAX(strftime('%J', `Ship Date`) - strftime('%J', `Order Date`)) AS longestTimeDays FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ?", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"longest_time_days": []}
    return {"longest_time_days": result[0]}

# Endpoint to get the count of distinct order IDs for a given customer name and ship mode
@app.get("/v1/superstore/count_distinct_order_ids_by_customer_name_and_ship_mode", operation_id="get_count_distinct_order_ids", summary="Retrieves the number of unique orders placed by a specific customer using a particular shipping method. The operation requires the customer's name and the chosen shipping method as input parameters.")
async def get_count_distinct_order_ids(customer_name: str = Query(..., description="Customer Name"), ship_mode: str = Query(..., description="Ship Mode")):
    cursor.execute("SELECT COUNT(DISTINCT T2.`Order ID`) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ? AND T2.`Ship Mode` = ?", (customer_name, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct order IDs for a given customer name
@app.get("/v1/superstore/count_distinct_order_ids_by_customer_name", operation_id="get_count_distinct_order_ids_by_customer_name", summary="Retrieve the number of unique orders associated with a specific customer. The operation uses the provided customer name to identify the customer and calculate the count of distinct order IDs linked to them.")
async def get_count_distinct_order_ids_by_customer_name(customer_name: str = Query(..., description="Customer Name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.`Order ID`) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ?", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct order IDs for a given customer name where the product ID count is greater than a specified value
@app.get("/v1/superstore/distinct_order_ids_by_customer_name_and_product_count", operation_id="get_distinct_order_ids_by_customer_name_and_product_count", summary="Retrieves unique order identifiers for a specific customer, considering only those orders that contain more than a specified number of distinct products.")
async def get_distinct_order_ids_by_customer_name_and_product_count(customer_name: str = Query(..., description="Customer Name"), product_count: int = Query(..., description="Product count threshold")):
    cursor.execute("SELECT DISTINCT T2.`Order ID` FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ? GROUP BY T2.`Product ID` HAVING COUNT(T2.`Product ID`) > ?", (customer_name, product_count))
    result = cursor.fetchall()
    if not result:
        return {"order_ids": []}
    return {"order_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct order IDs for a given customer name and product category
@app.get("/v1/superstore/count_distinct_order_ids_by_customer_name_and_category", operation_id="get_count_distinct_order_ids_by_customer_name_and_category", summary="Retrieves the number of unique orders placed by a specific customer for a particular product category. The operation requires the customer's name and the category of interest as input parameters.")
async def get_count_distinct_order_ids_by_customer_name_and_category(category: str = Query(..., description="Product category"), customer_name: str = Query(..., description="Customer Name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.`Order ID`) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T3.Category = ? AND T1.`Customer Name` = ?", (category, customer_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct product names for a given customer name and ship year
@app.get("/v1/superstore/distinct_product_names_by_customer_name_and_ship_year", operation_id="get_distinct_product_names_by_customer_name_and_ship_year", summary="Retrieve a unique list of product names associated with a specific customer and ship year. This operation filters products based on the provided customer name and ship year, ensuring that only distinct product names are returned.")
async def get_distinct_product_names_by_customer_name_and_ship_year(customer_name: str = Query(..., description="Customer Name"), ship_year: str = Query(..., description="Ship year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T3.`Product Name` FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ? AND STRFTIME('%Y', T2.`Ship Date`) = ?", (customer_name, ship_year))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the total quantity for a given product name
@app.get("/v1/superstore/total_quantity_by_product_name", operation_id="get_total_quantity_by_product_name", summary="Retrieves the total quantity of a specific product from the central superstore database. The product is identified by its name, which is provided as an input parameter.")
async def get_total_quantity_by_product_name(product_name: str = Query(..., description="Product Name")):
    cursor.execute("SELECT SUM(T1.Quantity) FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get distinct customer names for a given product name
@app.get("/v1/superstore/distinct_customer_names_by_product_name", operation_id="get_distinct_customer_names_by_product_name", summary="Retrieves a unique list of customer names associated with a specific product. The product is identified by its name, which is provided as an input parameter.")
async def get_distinct_customer_names_by_product_name(product_name: str = Query(..., description="Product Name")):
    cursor.execute("SELECT DISTINCT T1.`Customer Name` FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T3.`Product Name` = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the count of distinct customer names for a given product name and customer segment
@app.get("/v1/superstore/count_distinct_customer_names_by_product_name_and_segment", operation_id="get_count_distinct_customer_names_by_product_name_and_segment", summary="Retrieves the count of unique customers who have purchased a specific product and belong to a certain segment. The product is identified by its name, and the segment refers to the customer's category.")
async def get_count_distinct_customer_names_by_product_name_and_segment(product_name: str = Query(..., description="Product Name"), segment: str = Query(..., description="Customer Segment")):
    cursor.execute("SELECT COUNT(DISTINCT T1.`Customer Name`) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T3.`Product Name` = ? AND T1.Segment = ?", (product_name, segment))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total quantity of a specific product ordered by a specific customer on a specific date
@app.get("/v1/superstore/total_quantity_product_customer_date", operation_id="get_total_quantity", summary="Retrieves the total quantity of a specific product ordered by a particular customer on a given date. The operation requires the customer's name, product name, and order date in 'YYYY-MM-DD' format to accurately calculate the total quantity.")
async def get_total_quantity(customer_name: str = Query(..., description="Name of the customer"), product_name: str = Query(..., description="Name of the product"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(T2.Quantity) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ? AND T3.`Product Name` = ? AND T2.`Order Date` = ?", (customer_name, product_name, order_date))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the count of distinct orders for a specific product by a specific customer
@app.get("/v1/superstore/count_distinct_orders_product_customer", operation_id="get_count_distinct_orders", summary="Retrieve the number of unique orders placed by a specific customer for a particular product. This operation requires the customer's name and the product's name as input parameters.")
async def get_count_distinct_orders(customer_name: str = Query(..., description="Name of the customer"), product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT COUNT(DISTINCT T2.`Order ID`) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ? AND T3.`Product Name` = ?", (customer_name, product_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the distinct sales amount after discount for a specific product ordered by a specific customer on a specific date
@app.get("/v1/superstore/distinct_sales_after_discount_product_customer_date", operation_id="get_distinct_sales_after_discount", summary="Retrieve the unique sales amount after discount for a specific product purchased by a particular customer on a given date. The operation requires the customer's name, product name, and order date as input parameters.")
async def get_distinct_sales_after_discount(customer_name: str = Query(..., description="Name of the customer"), product_name: str = Query(..., description="Name of the product"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.Sales / (1 - T2.Discount) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ? AND T3.`Product Name` = ? AND T2.`Order Date` = ?", (customer_name, product_name, order_date))
    result = cursor.fetchall()
    if not result:
        return {"sales_after_discount": []}
    return {"sales_after_discount": [row[0] for row in result]}

# Endpoint to get the distinct profit for a specific product ordered by a specific customer on a specific date
@app.get("/v1/superstore/distinct_profit_product_customer_date", operation_id="get_distinct_profit", summary="Retrieve the unique profit value for a specific product ordered by a particular customer on a given date. This operation calculates the profit by subtracting the existing profit from the net sales, which is determined by multiplying the sales amount by the quantity and then adjusting for the discount. The customer, product, and order date are used to filter the data.")
async def get_distinct_profit(customer_name: str = Query(..., description="Name of the customer"), product_name: str = Query(..., description="Name of the product"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT (T2.Sales / (1 - T2.discount)) * T2.Quantity - Profit FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ? AND T3.`Product Name` = ? AND T2.`Order Date` = ?", (customer_name, product_name, order_date))
    result = cursor.fetchall()
    if not result:
        return {"profit": []}
    return {"profit": [row[0] for row in result]}

# Endpoint to get the count of distinct products in a specific sub-category, region, and year
@app.get("/v1/superstore/count_distinct_products_subcategory_region_year", operation_id="get_count_distinct_products", summary="Retrieve the number of unique products within a specified sub-category, region, and year. This operation provides a count of distinct products that meet the given criteria, offering insights into product diversity across different sub-categories, regions, and years.")
async def get_count_distinct_products(sub_category: str = Query(..., description="Sub-category of the product"), region: str = Query(..., description="Region"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.`Product ID`) FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Sub-Category` = ? AND T1.Region = ? AND STRFTIME('%Y', T1.`Order Date`) = ?", (sub_category, region, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer with the highest total profit from a single order
@app.get("/v1/superstore/top_customer_by_profit", operation_id="get_top_customer_by_profit", summary="Retrieves the customer who has generated the highest profit from a single order. The calculation considers the sales amount, discount, quantity, and profit of each order. The result is obtained by joining the 'east_superstore' and 'people' tables.")
async def get_top_customer_by_profit():
    cursor.execute("SELECT T2.`Customer Name` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` GROUP BY T1.`Order ID`, T2.`Customer Name` ORDER BY SUM((T1.Sales / (1 - T1.Discount)) * T1.Quantity - T1.Profit) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the product with the highest sales amount after discount
@app.get("/v1/superstore/top_product_by_sales", operation_id="get_top_product_by_sales", summary="Retrieves the product with the highest net sales amount, calculated as the total sales divided by one minus the discount rate. The data is sourced from the east_superstore table and joined with the product table to obtain the product name.")
async def get_top_product_by_sales():
    cursor.execute("SELECT T2.`Product Name` FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` ORDER BY (T1.Sales / (1 - T1.Discount)) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the most recent product ordered by a specific customer
@app.get("/v1/superstore/most_recent_product_customer", operation_id="get_most_recent_product", summary="Retrieves the name of the most recent product ordered by the specified customer. The operation uses the customer's name to identify their orders and returns the name of the product from the most recent order.")
async def get_most_recent_product(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT T3.`Product Name` FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ? ORDER BY T2.`Order Date` DESC LIMIT 1", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the order ID with the highest profit for a specific product
@app.get("/v1/superstore/top_order_by_profit_product", operation_id="get_top_order_by_profit", summary="Retrieves the order ID associated with the highest profit generated by a specific product. The order ID is determined by calculating the total profit for each order containing the specified product, considering sales, discounts, quantity, and profit. The order with the highest total profit is then returned.")
async def get_top_order_by_profit(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T1.`Order ID` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ? GROUP BY T1.`Order ID` ORDER BY SUM((T1.Sales / (1 - T1.Discount)) * T1.Quantity - T1.Profit) DESC LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"order_id": []}
    return {"order_id": result[0]}

# Endpoint to get distinct product names for a given customer name
@app.get("/v1/superstore/distinct_product_names_by_customer", operation_id="get_distinct_product_names_by_customer", summary="Retrieves a unique list of product names associated with a specific customer. The operation filters products based on the provided customer name and returns only distinct product names.")
async def get_distinct_product_names_by_customer(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT DISTINCT T3.`Product Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T2.`Customer Name` = ?", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the count of distinct customers for a given product name, city, and minimum quantity
@app.get("/v1/superstore/count_distinct_customers_by_product_city_quantity", operation_id="get_count_distinct_customers_by_product_city_quantity", summary="Retrieves the number of unique customers who have purchased a specified product in a given city with a quantity greater than a defined minimum. This operation provides insights into customer diversity based on product and location preferences.")
async def get_count_distinct_customers_by_product_city_quantity(product_name: str = Query(..., description="Name of the product"), city: str = Query(..., description="City of the customer"), min_quantity: int = Query(..., description="Minimum quantity purchased")):
    cursor.execute("SELECT COUNT(DISTINCT T1.`Customer ID`) FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T3.`Product Name` = ? AND T2.City = ? AND T1.Quantity > ?", (product_name, city, min_quantity))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct product names with profit greater than a specified value
@app.get("/v1/superstore/distinct_product_names_by_profit", operation_id="get_distinct_product_names_by_profit", summary="Retrieve a list of unique product names from the west_superstore dataset that have a profit greater than the specified minimum profit value.")
async def get_distinct_product_names_by_profit(min_profit: int = Query(..., description="Minimum profit value")):
    cursor.execute("SELECT DISTINCT T2.`Product Name` FROM west_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.Profit > ?", (min_profit,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get product names for a given ship mode and region, limited to 10 results
@app.get("/v1/superstore/product_names_by_ship_mode_region", operation_id="get_product_names_by_ship_mode_region", summary="Retrieves the first 10 product names associated with a specific ship mode and region. The ship mode and region are used to filter the results, ensuring that only relevant product names are returned.")
async def get_product_names_by_ship_mode_region(ship_mode: str = Query(..., description="Ship mode"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT T2.`Product Name` FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Ship Mode` = ? AND T2.Region = ? LIMIT 10", (ship_mode, region))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get distinct product names for a given customer name and region
@app.get("/v1/superstore/distinct_product_names_by_customer_region", operation_id="get_distinct_product_names_by_customer_region", summary="Retrieves a unique list of product names associated with a specific customer and region. The operation filters products based on the provided customer name and region, ensuring that only distinct product names are returned.")
async def get_distinct_product_names_by_customer_region(customer_name: str = Query(..., description="Name of the customer"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT DISTINCT T3.`Product Name` FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ? AND T3.Region = ?", (customer_name, region))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get distinct customer names for a given region and ship mode, limited to 5 results
@app.get("/v1/superstore/distinct_customer_names_by_region_ship_mode", operation_id="get_distinct_customer_names_by_region_ship_mode", summary="Retrieves a list of up to 5 unique customer names associated with a specific region and ship mode. The operation filters customers based on the provided region and ship mode, ensuring that only distinct customer names are returned.")
async def get_distinct_customer_names_by_region_ship_mode(region: str = Query(..., description="Region"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.Region = ? AND T1.`Ship Mode` = ? LIMIT 5", (region, ship_mode))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the total profit for a given customer name and region
@app.get("/v1/superstore/total_profit_by_customer_region", operation_id="get_total_profit_by_customer_region", summary="Retrieves the total profit accumulated by a specific customer in a given region. The operation calculates the sum of profits from all transactions associated with the customer in the specified region.")
async def get_total_profit_by_customer_region(customer_name: str = Query(..., description="Name of the customer"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT SUM(T2.Profit) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ? AND T1.Region = ?", (customer_name, region))
    result = cursor.fetchone()
    if not result:
        return {"total_profit": []}
    return {"total_profit": result[0]}

# Endpoint to get product names for a given ship date, order date, and region
@app.get("/v1/superstore/product_names_by_ship_order_date_region", operation_id="get_product_names_by_ship_order_date_region", summary="Retrieves the names of products that were shipped on a specific date, ordered on a specific date, and belong to a particular region. The input parameters include the ship date, order date, and region, which are used to filter the results.")
async def get_product_names_by_ship_order_date_region(ship_date: str = Query(..., description="Ship date in 'YYYY-MM-DD' format"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT T2.`Product Name` FROM south_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Ship Date` = ? AND T2.Region = ? AND T1.`Order Date` = ?", (ship_date, region, order_date))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the total sales for a given product name and region
@app.get("/v1/superstore/total_sales_by_product_region", operation_id="get_total_sales_by_product_region", summary="Retrieves the total sales for a specific product in a given region. The operation calculates the sum of sales for the product in the specified region, providing a comprehensive view of its performance.")
async def get_total_sales_by_product_region(product_name: str = Query(..., description="Name of the product"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT SUM(T1.Sales) FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ? AND T2.Region = ?", (product_name, region))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get the top product name by quantity for a given customer name and region
@app.get("/v1/superstore/top_product_by_customer_region", operation_id="get_top_product_by_customer_region", summary="Retrieves the product with the highest quantity sold to a specific customer in a given region. The operation requires the customer's name and their region as input parameters.")
async def get_top_product_by_customer_region(customer_name: str = Query(..., description="Name of the customer"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT T3.`Product Name` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T2.`Customer Name` = ? AND T2.Region = ? ORDER BY T1.Quantity DESC LIMIT 1", (customer_name, region))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the total quantity and product name for a specific ship date and region
@app.get("/v1/superstore/total_quantity_product_name", operation_id="get_total_quantity_product_name", summary="Retrieves the total quantity of products and their respective names for a given ship date and region. The operation aggregates the quantity of products shipped on the specified date and belonging to the provided region.")
async def get_total_quantity_product_name(ship_date: str = Query(..., description="Ship date in 'YYYY-MM-DD' format"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT SUM(T1.Quantity), T2.`Product Name` FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Ship Date` = ? AND T2.Region = ?", (ship_date, region))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": [], "product_name": []}
    return {"total_quantity": result[0], "product_name": result[1]}

# Endpoint to get distinct customer names for a specific product name, order date, and region
@app.get("/v1/superstore/distinct_customer_names", operation_id="get_distinct_customer_names", summary="Retrieve a unique list of customer names who have purchased a specific product on a given date in a particular region. The product name, order date, and region are required as input parameters to filter the results.")
async def get_distinct_customer_names(product_name: str = Query(..., description="Product name"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T3.`Product Name` = ? AND T1.`Order Date` = ? AND T1.Region = ?", (product_name, order_date, region))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get distinct product categories for a specific customer name, order date, and region
@app.get("/v1/superstore/distinct_product_categories", operation_id="get_distinct_product_categories", summary="Retrieves a unique list of product categories associated with a specific customer, order date, and region. The operation filters the data based on the provided customer name, order date, and region, and returns the distinct product categories that meet the criteria.")
async def get_distinct_product_categories(customer_name: str = Query(..., description="Customer name"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT DISTINCT T3.Category FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T2.`Customer Name` = ? AND T1.`Order Date` = ? AND T2.Region = ?", (customer_name, order_date, region))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get the percentage of a specific category in total quantity for a specific region and ship mode
@app.get("/v1/superstore/category_percentage_in_quantity", operation_id="get_category_percentage_in_quantity", summary="Retrieves the percentage of a specified category's contribution to the total quantity of products sold in a given region and through a specific shipping mode. The calculation is based on the sum of quantities from the west_superstore table, filtered by the provided category, region, and ship mode.")
async def get_category_percentage_in_quantity(category: str = Query(..., description="Category"), region: str = Query(..., description="Region"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN T2.Category = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(T1.Quantity) FROM west_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Region = ? AND T1.`Ship Mode` = ?", (category, region, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get ship dates for a specific customer name and region
@app.get("/v1/superstore/ship_dates_customer_region", operation_id="get_ship_dates_customer_region", summary="Retrieves the shipment dates for a specific customer in a given region. The operation requires the customer's name and region as input parameters to filter the results accordingly.")
async def get_ship_dates_customer_region(customer_name: str = Query(..., description="Customer name"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT T2.`Ship Date` FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ? AND T1.Region = ?", (customer_name, region))
    result = cursor.fetchall()
    if not result:
        return {"ship_dates": []}
    return {"ship_dates": [row[0] for row in result]}

# Endpoint to get distinct segments for a specific region and order ID
@app.get("/v1/superstore/distinct_segments_region_order", operation_id="get_distinct_segments_region_order", summary="Retrieve unique customer segments associated with a specific region and order. This operation fetches distinct segments from the west_superstore table, joining it with the people table based on the customer ID. The result is filtered by the provided region and order ID.")
async def get_distinct_segments_region_order(region: str = Query(..., description="Region"), order_id: str = Query(..., description="Order ID")):
    cursor.execute("SELECT DISTINCT T2.Segment FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.Region = ? AND T1.`Order ID` = ?", (region, order_id))
    result = cursor.fetchall()
    if not result:
        return {"segments": []}
    return {"segments": [row[0] for row in result]}

# Endpoint to get the total sales for a specific product name and region
@app.get("/v1/superstore/total_sales_product_region", operation_id="get_total_sales_product_region", summary="Retrieves the total sales for a specific product in a given region. The operation requires the product name and region as input parameters to filter the sales data and calculate the sum.")
async def get_total_sales_product_region(product_name: str = Query(..., description="Product name"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT SUM(T1.Sales) FROM west_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ? AND T1.Region = ?", (product_name, region))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get order IDs for a specific region and customer name
@app.get("/v1/superstore/order_ids_region_customer", operation_id="get_order_ids_region_customer", summary="Retrieves the order IDs associated with a specific customer in a given region. The operation filters orders based on the provided region and customer name, and returns the corresponding order IDs.")
async def get_order_ids_region_customer(region: str = Query(..., description="Region"), customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT T1.`Order ID` FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T2.Region = ? AND T2.`Customer Name` = ?", (region, customer_name))
    result = cursor.fetchall()
    if not result:
        return {"order_ids": []}
    return {"order_ids": [row[0] for row in result]}

# Endpoint to get product names for a specific order date, ship date, and region
@app.get("/v1/superstore/product_names_order_ship_date", operation_id="get_product_names_order_ship_date", summary="Retrieve the names of products associated with a specific order date, ship date, and region. This operation filters the central superstore data based on the provided order and ship dates, as well as the selected region, to return a list of corresponding product names.")
async def get_product_names_order_ship_date(order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format"), ship_date: str = Query(..., description="Ship date in 'YYYY-MM-DD' format"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Order Date` = ? AND T1.`Ship Date` = ? AND T2.Region = ?", (order_date, ship_date, region))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the city and state of the customer with the highest sales
@app.get("/v1/superstore/top_customer_city_state", operation_id="get_top_customer_city_state", summary="Retrieves the city and state of the customer who has made the highest total sales across all regions. The data is aggregated from the sales records of the west, east, central, and south superstores, and then matched with the corresponding customer information.")
async def get_top_customer_city_state():
    cursor.execute("SELECT T5.City, T5.State FROM west_superstore AS T1 INNER JOIN east_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN central_superstore AS T3 ON T3.`Customer ID` = T2.`Customer ID` INNER JOIN south_superstore AS T4 ON T4.`Customer ID` = T3.`Customer ID` INNER JOIN people AS T5 ON T5.`Customer ID` = T4.`Customer ID` ORDER BY T2.Sales DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city": [], "state": []}
    return {"city": result[0], "state": result[1]}

# Endpoint to get the customer name with the highest profit in a specific region
@app.get("/v1/superstore/customer_highest_profit_by_region", operation_id="get_customer_highest_profit_by_region", summary="Retrieves the name of the customer who has generated the highest profit in a specified region. The operation filters data from the east_superstore table based on the provided region and joins it with the people table using the 'Customer ID'. The result is ordered by profit in descending order and limited to the top entry.")
async def get_customer_highest_profit_by_region(region: str = Query(..., description="Region of the customer")):
    cursor.execute("SELECT T2.`Customer Name` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.Region = ? ORDER BY T1.Profit DESC LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the highest quantity ordered by a customer in a specific city and state
@app.get("/v1/superstore/highest_quantity_by_city_state", operation_id="get_highest_quantity_by_city_state", summary="Retrieve the maximum quantity ordered by a customer in a given city and state. This operation combines data from multiple superstore databases and the people database to determine the highest quantity. The city and state of the customer are used as input parameters to filter the results.")
async def get_highest_quantity_by_city_state(city: str = Query(..., description="City of the customer"), state: str = Query(..., description="State of the customer")):
    cursor.execute("SELECT T1.Quantity FROM west_superstore AS T1 INNER JOIN east_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN central_superstore AS T3 ON T3.`Customer ID` = T2.`Customer ID` INNER JOIN south_superstore AS T4 ON T4.`Customer ID` = T3.`Customer ID` INNER JOIN people AS T5 ON T5.`Customer ID` = T4.`Customer ID` WHERE T5.City = ? AND T5.State = ? ORDER BY T1.Quantity DESC LIMIT 1", (city, state))
    result = cursor.fetchone()
    if not result:
        return {"quantity": []}
    return {"quantity": result[0]}

# Endpoint to get the order date and product name for a specific order ID and region
@app.get("/v1/superstore/order_date_product_name_by_order_id_region", operation_id="get_order_date_product_name_by_order_id_region", summary="Retrieves the order date and product name associated with a specific order ID and region. The order ID and region are used to filter the results, providing a focused view of the requested data.")
async def get_order_date_product_name_by_order_id_region(order_id: str = Query(..., description="Order ID"), region: str = Query(..., description="Region of the product")):
    cursor.execute("SELECT T1.`Order Date`, T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Order ID` = ? AND T2.Region = ?", (order_id, region))
    result = cursor.fetchall()
    if not result:
        return {"order_date_product_name": []}
    return {"order_date_product_name": result}

# Endpoint to get distinct customer names for a specific product in a specific region
@app.get("/v1/superstore/distinct_customer_names_by_product_region", operation_id="get_distinct_customer_names_by_product_region", summary="Retrieves a unique list of customer names who have purchased a specific product in a given region. The operation filters customers based on the provided region and product name, ensuring that only distinct customer names are returned.")
async def get_distinct_customer_names_by_product_region(region: str = Query(..., description="Region of the customer"), product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T1.Region = ? AND T3.`Product Name` = ?", (region, product_name))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the product name with the highest sales in a specific category and region
@app.get("/v1/superstore/highest_sales_product_by_category_region", operation_id="get_highest_sales_product_by_category_region", summary="Retrieves the name of the product with the highest sales in a specified category and region. The operation requires the category and region as input parameters to filter the data and identify the top-selling product.")
async def get_highest_sales_product_by_category_region(category: str = Query(..., description="Category of the product"), region: str = Query(..., description="Region of the product")):
    cursor.execute("SELECT T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Category = ? AND T2.Region = ? ORDER BY T1.Sales DESC LIMIT 1", (category, region))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the customer name with the highest discount in a specific region
@app.get("/v1/superstore/customer_highest_discount_by_region", operation_id="get_customer_highest_discount_by_region", summary="Retrieves the name of the customer with the highest discount in a specified region. The operation filters data from the west_superstore table based on the provided region and joins it with the people table using the customer ID. The result is ordered by the discount in descending order and limited to the top entry.")
async def get_customer_highest_discount_by_region(region: str = Query(..., description="Region of the customer")):
    cursor.execute("SELECT T2.`Customer Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.Region = ? ORDER BY T1.Discount DESC LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get distinct product names with profit above a certain percentage of the average profit in a specific region
@app.get("/v1/superstore/distinct_product_names_above_avg_profit_by_region", operation_id="get_distinct_product_names_above_avg_profit_by_region", summary="Retrieves a list of unique product names from a specific region that have a profit margin exceeding a given percentage of the average profit. The operation filters products based on the provided region and profit percentage, ensuring that only those with a profit margin above the specified threshold are returned.")
async def get_distinct_product_names_above_avg_profit_by_region(region: str = Query(..., description="Region of the product"), profit_percentage: float = Query(..., description="Percentage of the average profit")):
    cursor.execute("SELECT DISTINCT T2.`Product Name` FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Region = ? AND T1.Profit > ( SELECT AVG(Profit) * ? FROM east_superstore )", (region, profit_percentage))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get distinct customer names with a specific sales condition in a specific region
@app.get("/v1/superstore/distinct_customer_names_by_sales_condition_region", operation_id="get_distinct_customer_names_by_sales_condition_region", summary="Retrieves a list of unique customer names that meet a specified sales condition within a given region. The sales condition is determined by comparing the total sales (calculated as sales divided by one minus discount, multiplied by quantity, and then subtracting profit) to a provided value. This operation is useful for identifying distinct customers who meet specific sales criteria in a particular region.")
async def get_distinct_customer_names_by_sales_condition_region(region: str = Query(..., description="Region of the customer"), sales_condition: float = Query(..., description="Sales condition value")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.Region = ? AND T1.Sales / (1 - T1.Discount) * T1.Quantity - T1.Profit > ?", (region, sales_condition))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the count of distinct order IDs for a specific customer in a specific year
@app.get("/v1/superstore/count_distinct_order_ids_by_customer_year", operation_id="get_count_distinct_order_ids_by_customer_year", summary="Retrieve the number of unique orders placed by a specific customer during a given year. The operation requires the customer's name and the year (in YYYY format) as input parameters.")
async def get_count_distinct_order_ids_by_customer_year(customer_name: str = Query(..., description="Customer name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.`Order ID`) FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T2.`Customer Name` = ? AND STRFTIME('%Y', T1.`Order Date`) = ?", (customer_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total profit for a specific product across all superstores
@app.get("/v1/superstore/total_profit_by_product", operation_id="get_total_profit_by_product", summary="Retrieves the total profit generated by a specific product across all superstores. The operation calculates the cumulative profit by aggregating the profit data from all superstores for the given product.")
async def get_total_profit_by_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(T1.Profit) + SUM(T2.Profit) + SUM(T3.Profit) + SUM(T4.Profit) AS totalProfit FROM west_superstore AS T1 INNER JOIN east_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN central_superstore AS T3 ON T3.`Customer ID` = T2.`Customer ID` INNER JOIN south_superstore AS T4 ON T4.`Customer ID` = T3.`Customer ID` INNER JOIN product AS T5 ON T5.`Product ID` = T4.`Product ID` WHERE T5.`Product Name` = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"totalProfit": []}
    return {"totalProfit": result[0]}

# Endpoint to get distinct product names purchased by customers in a specific city
@app.get("/v1/superstore/distinct_products_by_city", operation_id="get_distinct_products_by_city", summary="Retrieve a unique list of product names that have been purchased by customers residing in the specified city. This operation provides insights into the variety of products bought by customers in a particular location.")
async def get_distinct_products_by_city(city: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT DISTINCT T3.`Product Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T2.City = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get the count of orders in east and west superstores for a specific year
@app.get("/v1/superstore/order_count_east_west_by_year", operation_id="get_order_count_east_west_by_year", summary="Retrieves the total number of orders placed in the east and west superstores during a specified year. The year should be provided in the 'YYYY' format.")
async def get_order_count_east_west_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT east, west FROM ( SELECT COUNT(`Order ID`) AS east , ( SELECT COUNT(`Order ID`) FROM west_superstore WHERE `Order Date` LIKE ? ) AS west FROM east_superstore WHERE `Order Date` LIKE ? )", (year + '%', year + '%'))
    result = cursor.fetchone()
    if not result:
        return {"east": [], "west": []}
    return {"east": result[0], "west": result[1]}

# Endpoint to get distinct product names purchased by a specific customer in a specific year
@app.get("/v1/superstore/distinct_products_by_customer_year", operation_id="get_distinct_products_by_customer_year", summary="Retrieve a unique list of product names that a particular customer has purchased within a specified year. The operation filters the products based on the provided customer name and year, ensuring that only distinct product names are returned.")
async def get_distinct_products_by_customer_year(customer_name: str = Query(..., description="Name of the customer"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T3.`Product Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T2.`Customer Name` = ? AND STRFTIME('%Y', T1.`Order Date`) = ?", (customer_name, year))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get the total cost for a specific customer in a specific region and year
@app.get("/v1/superstore/total_cost_by_customer_region_year", operation_id="get_total_cost_by_customer_region_year", summary="Retrieve the total cost incurred by a specific customer in a given region and year. This operation calculates the total cost by summing the sales, adjusted for discounts and quantities, and subtracting the profit. The customer's name, region, and year are required as input parameters to filter the results.")
async def get_total_cost_by_customer_region_year(region: str = Query(..., description="Region"), customer_name: str = Query(..., description="Name of the customer"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM((T1.Sales / (1 - T1.Discount)) * T1.Quantity - T1.Profit) AS cost FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T1.`Product ID` = T3.`Product ID` AND T1.Region = T3.Region WHERE T1.Region = ? AND T2.`Customer Name` = ? AND strftime('%Y', T1.`Order Date`) = ?", (region, customer_name, year))
    result = cursor.fetchone()
    if not result:
        return {"cost": []}
    return {"cost": result[0]}

# Endpoint to get distinct customer details who purchased a specific product
@app.get("/v1/superstore/distinct_customers_by_product", operation_id="get_distinct_customers_by_product", summary="Retrieves unique customer details, including their name, city, and country, for a specific product. This operation identifies distinct customers who have purchased the specified product by cross-referencing the west_superstore, people, and product tables.")
async def get_distinct_customers_by_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name`, T2.City, T2.Country FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T3.`Product Name` = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"name": row[0], "city": row[1], "country": row[2]} for row in result]}

# Endpoint to get distinct ship dates and product names for a specific customer
@app.get("/v1/superstore/distinct_ship_dates_products_by_customer", operation_id="get_distinct_ship_dates_products_by_customer", summary="Retrieves a unique list of shipment dates and corresponding product names associated with a specific customer. The customer is identified by their name.")
async def get_distinct_ship_dates_products_by_customer(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT DISTINCT T2.`Ship Date`, T3.`Product Name` FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ?", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"ship_dates_products": []}
    return {"ship_dates_products": [{"ship_date": row[0], "product_name": row[1]} for row in result]}

# Endpoint to get the count of orders with a specific ship mode and product category
@app.get("/v1/superstore/order_count_by_ship_mode_category", operation_id="get_order_count_by_ship_mode_category", summary="Retrieves the total number of orders that were shipped using a specific mode and belong to a certain product category. The operation requires the ship mode and product category as input parameters.")
async def get_order_count_by_ship_mode_category(ship_mode: str = Query(..., description="Ship mode"), category: str = Query(..., description="Product category")):
    cursor.execute("SELECT COUNT(T1.`Order ID`) FROM south_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Ship Mode` = ? AND T2.Category = ?", (ship_mode, category))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of orders for a specific product category and customer name
@app.get("/v1/superstore/order_count_by_category_customer", operation_id="get_order_count_by_category_customer", summary="Retrieves the total number of orders for a specific product category and customer. The operation requires the product category and customer name as input parameters to filter the results accordingly.")
async def get_order_count_by_category_customer(category: str = Query(..., description="Product category"), customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT COUNT(*) FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T3.Category = ? AND T2.`Customer Name` = ?", (category, customer_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the category of the most profitable product in the south superstore
@app.get("/v1/superstore/south/most_profitable_product_category", operation_id="get_most_profitable_product_category", summary="Retrieves the category of the product with the highest profit from the south superstore. The operation calculates the profit for each product in the south superstore and returns the category of the product with the highest profit.")
async def get_most_profitable_product_category():
    cursor.execute("SELECT T2.Category FROM south_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` ORDER BY T1.Profit DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category": []}
    return {"category": result[0]}

# Endpoint to get product names purchased by a specific customer in the south superstore
@app.get("/v1/superstore/south/products_by_customer", operation_id="get_products_by_customer", summary="Retrieves the names of products purchased by a specified customer in the south superstore. The operation requires the customer's name as input and returns a list of product names associated with the customer's purchases.")
async def get_products_by_customer(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT T3.`Product Name` FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T2.`Customer Name` = ?", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get product names shipped on a specific date with a specific ship mode in the west superstore
@app.get("/v1/superstore/west/products_by_ship_mode_and_date", operation_id="get_products_by_ship_mode_and_date", summary="Retrieves the names of products that were shipped using a specified shipping method on a given date in the west superstore. The shipping method and date are provided as input parameters.")
async def get_products_by_ship_mode_and_date(ship_mode: str = Query(..., description="Ship mode"), ship_date: str = Query(..., description="Ship date in 'YYYY%' format")):
    cursor.execute("SELECT T2.`Product Name` FROM west_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Ship Mode` = ? AND T1.`Ship Date` LIKE ?", (ship_mode, ship_date))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get distinct product categories purchased by a specific customer in the east superstore
@app.get("/v1/superstore/east/distinct_categories_by_customer", operation_id="get_distinct_categories_by_customer", summary="Retrieve a unique list of product categories that a particular customer has purchased from the east superstore. The operation requires the customer's name as input to filter the results.")
async def get_distinct_categories_by_customer(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT DISTINCT T3.Category FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` LEFT JOIN central_superstore AS T4 ON T3.`Product ID` = T4.`Product ID` WHERE T2.`Customer Name` = ?", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"categories": []}
    return {"categories": [row[0] for row in result]}

# Endpoint to get the total quantity of a specific product purchased by a specific customer in the south superstore
@app.get("/v1/superstore/south/total_quantity_by_customer_and_product", operation_id="get_total_quantity_by_customer_and_product", summary="Retrieves the total quantity of a specific product that a particular customer has purchased in the south superstore. The operation requires the customer's name and the product's name as input parameters to filter the data accordingly.")
async def get_total_quantity_by_customer_and_product(customer_name: str = Query(..., description="Name of the customer"), product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(T1.Quantity) FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T2.`Customer Name` = ? AND T3.`Product Name` = ?", (customer_name, product_name))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get distinct product names shipped with a specific ship mode and quantity in the central superstore
@app.get("/v1/superstore/central/distinct_products_by_ship_mode_and_quantity", operation_id="get_distinct_products_by_ship_mode_and_quantity", summary="Retrieve a list of unique product names that have been shipped using a specified shipping method and meet a minimum quantity threshold in the central superstore. The operation filters products based on the provided shipping method and quantity, ensuring that only distinct product names are returned.")
async def get_distinct_products_by_ship_mode_and_quantity(ship_mode: str = Query(..., description="Ship mode"), min_quantity: int = Query(..., description="Minimum quantity")):
    cursor.execute("SELECT DISTINCT T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Ship Mode` = ? AND T1.Quantity >= ?", (ship_mode, min_quantity))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the category of the product with the highest sales in the west superstore
@app.get("/v1/superstore/west/highest_sales_product_category", operation_id="get_highest_sales_product_category", summary="Retrieves the product category with the highest sales from the west superstore. This operation identifies the product with the maximum sales in the west superstore and returns its corresponding category. The result provides insights into the most popular product category based on sales performance.")
async def get_highest_sales_product_category():
    cursor.execute("SELECT T2.Category FROM west_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` ORDER BY T1.Sales LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"category": []}
    return {"category": result[0]}

# Endpoint to get the total sales for a specific category in a specific year in the east superstore
@app.get("/v1/superstore/east/total_sales_by_category_and_year", operation_id="get_total_sales_by_category_and_year", summary="Retrieves the total sales for a specified product category in a given year for the East Superstore. The operation calculates the sum of sales from the East Superstore database, filtered by the provided year and product category.")
async def get_total_sales_by_category_and_year(year: str = Query(..., description="Year in 'YYYY' format"), category: str = Query(..., description="Category of the product")):
    cursor.execute("SELECT SUM(T1.Sales) FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE STRFTIME('%Y', T1.`Order Date`) = ? AND T2.Category = ?", (year, category))
    result = cursor.fetchone()
    if not result:
        return {"total_sales": []}
    return {"total_sales": result[0]}

# Endpoint to get the average sales for a specific product in the west superstore
@app.get("/v1/superstore/west/average_sales_by_product", operation_id="get_average_sales_by_product", summary="Retrieves the average sales figure for a specific product in the west superstore. This operation calculates the average sales by joining data from the west_superstore, people, and product tables using the provided product name.")
async def get_average_sales_by_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT AVG(T1.Sales) FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T3.`Product Name` = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_sales": []}
    return {"average_sales": result[0]}

# Endpoint to get the percentage of a specific category in the product table
@app.get("/v1/superstore/category_percentage", operation_id="get_category_percentage", summary="Retrieves the percentage of a specific product category in the product table. This operation calculates the proportion of products belonging to the specified category, based on the total number of products in the table. The category is provided as an input parameter.")
async def get_category_percentage(category: str = Query(..., description="Category of the product")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN T3.Category = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.Category) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID`", (category,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of people in one state compared to another
@app.get("/v1/superstore/state_percentage_comparison", operation_id="get_state_percentage_comparison", summary="Retrieves the percentage of people in the first state compared to the second state. This operation compares the population of two specified states and returns the percentage of people in the first state relative to the second state. The input parameters are used to select the states for comparison.")
async def get_state_percentage_comparison(state1: str = Query(..., description="First state for comparison"), state2: str = Query(..., description="Second state for comparison")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN State = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE  WHEN State = ? THEN 1 ELSE 0 END) FROM people", (state1, state2))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the most popular product name in a specific sub-category
@app.get("/v1/superstore/most_popular_product_in_subcategory", operation_id="get_most_popular_product_in_subcategory", summary="Retrieves the name of the most frequently purchased product within a specified sub-category. The sub-category is used to filter the products and the product with the highest sales count is returned.")
async def get_most_popular_product_in_subcategory(sub_category: str = Query(..., description="Sub-category of the product")):
    cursor.execute("SELECT T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Sub-Category` = ? GROUP BY T2.`Product Name` ORDER BY COUNT(T2.`Product ID`) DESC LIMIT 1", (sub_category,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get distinct customer names with more than a specified number of orders in a given year
@app.get("/v1/superstore/distinct_customers_by_year_and_order_count", operation_id="get_distinct_customers_by_year_and_order_count", summary="Retrieve a list of unique customer names who have placed more than a specified number of orders in a given year. This operation filters the orders database by the provided year and groups the results by customer name. It then returns only those customers who have exceeded the specified minimum order count.")
async def get_distinct_customers_by_year_and_order_count(year: str = Query(..., description="Year in 'YYYY' format"), min_order_count: int = Query(..., description="Minimum number of orders")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE STRFTIME('%Y', T1.`Order Date`) = ? GROUP BY T2.`Customer Name` HAVING COUNT(T2.`Customer Name`) > ?", (year, min_order_count))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the maximum profit for a specific customer
@app.get("/v1/superstore/max_profit_by_customer", operation_id="get_max_profit_by_customer", summary="Retrieves the highest profit value associated with a specific customer from the central superstore data. The customer is identified by their name.")
async def get_max_profit_by_customer(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT MAX(T2.Profit) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ?", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"max_profit": []}
    return {"max_profit": result[0]}

# Endpoint to get the count of customer IDs for a specific customer in a given year
@app.get("/v1/superstore/customer_id_count_by_year", operation_id="get_customer_id_count_by_year", summary="Retrieves the total number of unique customer IDs associated with a specific customer in a given year. The operation requires the customer's name and the year as input parameters. The year should be provided in the 'YYYY' format.")
async def get_customer_id_count_by_year(customer_name: str = Query(..., description="Name of the customer"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.`Customer ID`) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ? AND STRFTIME('%Y', T2.`Ship Date`) = ?", (customer_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in sales between east and west superstores
@app.get("/v1/superstore/sales_difference_east_west", operation_id="get_sales_difference_east_west", summary="Retrieves the total sales difference between the east and west superstores, calculated by summing the sales of each store and subtracting the total sales of the west superstore from the east superstore. The comparison is based on matching customer IDs.")
async def get_sales_difference_east_west():
    cursor.execute("SELECT SUM(T1.Sales) - SUM(T2.Sales) AS difference FROM east_superstore AS T1 INNER JOIN west_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID`")
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get distinct product names in a specific region with negative profit
@app.get("/v1/superstore/distinct_products_by_region_negative_profit", operation_id="get_distinct_products_by_region_negative_profit", summary="Retrieve a unique list of product names from a specified region that have incurred a loss. The operation filters products based on the provided region and identifies those with a negative profit.")
async def get_distinct_products_by_region_negative_profit(region: str = Query(..., description="Region of the product")):
    cursor.execute("SELECT DISTINCT T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Region = ? AND T1.Profit < 0", (region,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the product name and ship mode for the fastest delivery in a specific region
@app.get("/v1/superstore/fastest_delivery_product_by_region", operation_id="get_fastest_delivery_product_by_region", summary="Retrieves the name of the product and its associated ship mode that has the fastest delivery time in the specified region. The region is a required input parameter.")
async def get_fastest_delivery_product_by_region(region: str = Query(..., description="Region of the product")):
    cursor.execute("SELECT DISTINCT T2.`Product Name`, T1.`Ship Mode` FROM west_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Region = ? ORDER BY T1.`Ship Date` - T1.`Order Date` LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": [], "ship_mode": []}
    return {"product_name": result[0], "ship_mode": result[1]}

# Endpoint to get the count of distinct order IDs for a specific product in a specific region and ship mode
@app.get("/v1/superstore/order_count_by_product_region_ship_mode", operation_id="get_order_count_by_product_region_ship_mode", summary="Retrieves the total number of unique orders for a specific product in a given region and ship mode. The operation considers the product name, region, and ship mode as input parameters to filter the data and provide an accurate count of distinct order IDs.")
async def get_order_count_by_product_region_ship_mode(product_name: str = Query(..., description="Name of the product"), region: str = Query(..., description="Region of the product"), ship_mode: str = Query(..., description="Ship mode")):
    cursor.execute("SELECT COUNT(DISTINCT T1.`Order ID`) FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ? AND T2.Region = ? AND T1.`Ship Mode` = ?", (product_name, region, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top customer name based on segment, state, region, and order year
@app.get("/v1/superstore/top_customer_name", operation_id="get_top_customer_name", summary="Retrieves the name of the top customer from the east superstore based on the provided segment, state, region, and order year. The customer with the highest number of orders matching the specified criteria is returned.")
async def get_top_customer_name(segment: str = Query(..., description="Segment of the customer"), state: str = Query(..., description="State of the customer"), region: str = Query(..., description="Region of the customer"), order_year: str = Query(..., description="Year of the order in 'YYYY' format")):
    cursor.execute("SELECT T2.`Customer Name` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T2.Segment = ? AND T2.State = ? AND T2.Region = ? AND STRFTIME('%Y', T1.`Order Date`) = ? GROUP BY T2.`Customer Name` ORDER BY COUNT(T2.`Customer Name`) DESC LIMIT 1", (segment, state, region, order_year))
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the top segment based on region and sales
@app.get("/v1/superstore/top_segment", operation_id="get_top_segment", summary="Retrieves the top-performing segment from the east superstore based on the provided region and sales data. The segment is determined by analyzing sales figures, adjusted for discounts, and is derived from the customer and product data. The operation returns the name of the top segment.")
async def get_top_segment(region: str = Query(..., description="Region of the product")):
    cursor.execute("SELECT T2.Segment FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T1.Region = ? ORDER BY (T1.Sales / (1 - T1.Discount)) DESC LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"segment": []}
    return {"segment": result[0]}

# Endpoint to get the duration between ship date and order date based on order ID
@app.get("/v1/superstore/order_duration", operation_id="get_order_duration", summary="Retrieve the time elapsed between the order date and ship date for a specific order in the central superstore. The operation requires the order ID as input to identify the relevant order.")
async def get_order_duration(order_id: str = Query(..., description="Order ID")):
    cursor.execute("SELECT DISTINCT strftime('%J', `Ship Date`) - strftime('%J', `Order Date`) AS duration FROM central_superstore WHERE `Order ID` = ?", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"duration": []}
    return {"duration": result[0]}

# Endpoint to get the count of distinct order IDs based on quantity and ship mode
@app.get("/v1/superstore/count_distinct_order_ids_quantity_ship_mode", operation_id="get_count_distinct_order_ids_quantity_ship_mode", summary="Retrieves the total number of unique orders from the central superstore that meet the specified quantity and ship mode criteria. The quantity parameter determines the minimum order quantity required, while the ship mode parameter specifies the desired shipping method.")
async def get_count_distinct_order_ids_quantity_ship_mode(quantity: int = Query(..., description="Quantity of the order"), ship_mode: str = Query(..., description="Ship mode of the order")):
    cursor.execute("SELECT COUNT(DISTINCT `Order ID`) FROM central_superstore WHERE Quantity > ? AND `Ship Mode` = ?", (quantity, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get order IDs with profit less than a specified value
@app.get("/v1/superstore/order_ids_profit_less_than", operation_id="get_order_ids_profit_less_than", summary="Retrieves the first three order IDs from the central superstore where the profit is less than the provided value. This operation is useful for identifying orders that have not met a certain profit threshold.")
async def get_order_ids_profit_less_than(profit: float = Query(..., description="Profit value")):
    cursor.execute("SELECT `Order ID` FROM central_superstore WHERE Profit < ? LIMIT 3", (profit,))
    result = cursor.fetchall()
    if not result:
        return {"order_ids": []}
    return {"order_ids": [row[0] for row in result]}

# Endpoint to get the top customer name based on product name
@app.get("/v1/superstore/top_customer_name_product", operation_id="get_top_customer_name_product", summary="Retrieves the name of the customer who has made the most purchases of a specific product from the south superstore. The product is identified by the provided product name.")
async def get_top_customer_name_product(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT T2.`Customer Name` FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T1.`Product ID` WHERE T3.`Product Name` = ? GROUP BY T2.`Customer Name` ORDER BY COUNT(T2.`Customer Name`) DESC LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get distinct profit values based on product name
@app.get("/v1/superstore/distinct_profit_values", operation_id="get_distinct_profit_values", summary="Retrieves a unique set of profit values associated with a specific product from the central superstore. The product is identified by its name.")
async def get_distinct_profit_values(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT DISTINCT T1.Profit FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"profit_values": []}
    return {"profit_values": [row[0] for row in result]}

# Endpoint to get the sum of quantities based on product name
@app.get("/v1/superstore/sum_quantities", operation_id="get_sum_quantities", summary="Retrieves the total quantity of a specific product from the west superstore. The product is identified by its name, which is provided as an input parameter.")
async def get_sum_quantities(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT SUM(T1.Quantity) FROM west_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum_quantities": []}
    return {"sum_quantities": result[0]}

# Endpoint to get the count of orders for a specific customer name and ship mode
@app.get("/v1/superstore/order_count_customer_ship_mode", operation_id="get_order_count", summary="Retrieves the total number of orders associated with a given customer and ship mode. The operation requires the customer's name and the desired ship mode as input parameters to filter the results accordingly.")
async def get_order_count(customer_name: str = Query(..., description="Customer Name"), ship_mode: str = Query(..., description="Ship Mode")):
    cursor.execute("SELECT COUNT(*) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ? AND T2.`Ship Mode` = ?", (customer_name, ship_mode))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the adjusted sales for a specific product name
@app.get("/v1/superstore/adjusted_sales_product_name", operation_id="get_adjusted_sales", summary="Retrieves the adjusted sales for a specific product. The adjusted sales are calculated by dividing the total sales by one minus the discount rate. The product is identified by its name, which is provided as an input parameter.")
async def get_adjusted_sales(product_name: str = Query(..., description="Product Name")):
    cursor.execute("SELECT T1.Sales / (1 - T1.Discount) FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"adjusted_sales": []}
    return {"adjusted_sales": [row[0] for row in result]}

# Endpoint to get distinct profits for a specific product name
@app.get("/v1/superstore/distinct_profits_product_name", operation_id="get_distinct_profits", summary="Retrieves the unique profit values associated with a specific product. The product is identified by its name, which is provided as an input parameter.")
async def get_distinct_profits(product_name: str = Query(..., description="Product Name")):
    cursor.execute("SELECT DISTINCT T1.Profit FROM south_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"profits": []}
    return {"profits": [row[0] for row in result]}

# Endpoint to get the count of product categories for a specific ship mode
@app.get("/v1/superstore/count_categories_ship_mode", operation_id="get_count_categories", summary="Retrieves the total number of unique product categories associated with a given ship mode. The ship mode is specified as an input parameter.")
async def get_count_categories(ship_mode: str = Query(..., description="Ship Mode")):
    cursor.execute("SELECT COUNT(T2.Category) FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Ship Mode` = ?", (ship_mode,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the order ID with the highest profit for a specific city and state
@app.get("/v1/superstore/highest_profit_order_city_state", operation_id="get_highest_profit_order", summary="Retrieves the order ID associated with the highest profit for a given city and state. The operation filters orders based on the specified city and state, then sorts them by profit in descending order. The order ID of the top-ranking order is returned.")
async def get_highest_profit_order(city: str = Query(..., description="City"), state: str = Query(..., description="State")):
    cursor.execute("SELECT T1.`Order ID` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T2.City = ? AND T2.State = ? ORDER BY T1.Profit DESC LIMIT 1", (city, state))
    result = cursor.fetchone()
    if not result:
        return {"order_id": []}
    return {"order_id": result[0]}

# Endpoint to get the count of orders for a specific product category
@app.get("/v1/superstore/order_count_product_category", operation_id="get_order_count_category", summary="Retrieves the total number of orders for a specified product category. The category is provided as an input parameter.")
async def get_order_count_category(category: str = Query(..., description="Product Category")):
    cursor.execute("SELECT COUNT(*) FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Category = ?", (category,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct product names for a specific ship date (YYYY-MM format)
@app.get("/v1/superstore/distinct_product_names_ship_date", operation_id="get_distinct_product_names_ship_date", summary="Retrieves a list of unique product names associated with a specific ship date (in YYYY-MM format). This operation filters the central superstore data based on the provided ship date and returns the distinct product names from the product table.")
async def get_distinct_product_names_ship_date(ship_date: str = Query(..., description="Ship Date in YYYY-MM format")):
    cursor.execute("SELECT DISTINCT T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE strftime('%Y-%m', T1.`Ship Date`) = ?", (ship_date,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the count of distinct order IDs for a specific customer name and order year
@app.get("/v1/superstore/count_distinct_order_ids_customer_year", operation_id="get_count_distinct_order_ids_customer_year", summary="Retrieves the number of unique orders placed by a specific customer during a given year. The operation requires the customer's name and the year of the orders as input parameters.")
async def get_count_distinct_order_ids_customer_year(customer_name: str = Query(..., description="Customer Name"), order_year: str = Query(..., description="Order Year in YYYY format")):
    cursor.execute("SELECT COUNT(DISTINCT T2.`Order ID`) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Customer Name` = ? AND STRFTIME('%Y', T2.`Order Date`) = ?", (customer_name, order_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct product names from east_superstore based on order ID
@app.get("/v1/superstore/east/distinct_product_names_by_order_id", operation_id="get_distinct_product_names_by_order_id", summary="Retrieves a list of unique product names associated with the specified order ID from the east_superstore database. This operation helps identify the distinct products purchased in a particular order.")
async def get_distinct_product_names_by_order_id(order_id: str = Query(..., description="Order ID")):
    cursor.execute("SELECT DISTINCT T2.`Product Name` FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Order ID` = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the percentage of orders with a specific discount in a given state
@app.get("/v1/superstore/central/discount_percentage_by_state", operation_id="get_discount_percentage_by_state", summary="Retrieves the percentage of orders with a specified discount in a particular state. The operation calculates this percentage by summing the number of orders with the given discount in the specified state and dividing it by the total number of orders in that state.")
async def get_discount_percentage_by_state(discount: float = Query(..., description="Discount value"), state: str = Query(..., description="State")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN T2.Discount = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.State = ?", (discount, state))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of orders shipped by a specific mode for a given product category and year
@app.get("/v1/superstore/central/ship_mode_percentage_by_category_year", operation_id="get_ship_mode_percentage_by_category_year", summary="Retrieve the proportion of orders shipped via a specific mode for a selected product category and year. This operation calculates the percentage by comparing the count of orders shipped using the specified mode to the total number of orders for the given category and year.")
async def get_ship_mode_percentage_by_category_year(ship_mode: str = Query(..., description="Ship mode"), category: str = Query(..., description="Product category"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN T1.`Ship Mode` = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Category = ? AND STRFTIME('%Y', T1.`Ship Date`) = ?", (ship_mode, category, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct customer names from west_superstore based on order date, discount, and region
@app.get("/v1/superstore/west/distinct_customer_names_by_order_date_discount_region", operation_id="get_distinct_customer_names_by_order_date_discount_region", summary="Retrieve a list of unique customer names from the west_superstore dataset, filtered by a specific order date, discount, and region. This operation allows you to identify distinct customers who placed orders on a given date, with a particular discount, and in a certain region.")
async def get_distinct_customer_names_by_order_date_discount_region(order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format"), discount: float = Query(..., description="Discount value"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Order Date` = ? AND T1.Discount = ? AND T1.Region = ?", (order_date, discount, region))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get distinct order IDs from central_superstore based on product name and order date
@app.get("/v1/superstore/central/distinct_order_ids_by_product_name_order_date", operation_id="get_distinct_order_ids_by_product_name_order_date", summary="Retrieves a list of unique order identifiers from the central superstore, filtered by a specific product name and order date. The product name and order date are provided as input parameters.")
async def get_distinct_order_ids_by_product_name_order_date(product_name: str = Query(..., description="Product name"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T1.`Order ID` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.`Product Name` = ? AND T1.`Order Date` = ?", (product_name, order_date))
    result = cursor.fetchall()
    if not result:
        return {"order_ids": []}
    return {"order_ids": [row[0] for row in result]}

# Endpoint to get distinct product names from central_superstore based on customer name and region
@app.get("/v1/superstore/central/distinct_product_names_by_customer_name_region", operation_id="get_distinct_product_names_by_customer_name_region", summary="Retrieves a unique list of product names associated with a specific customer and region from the central superstore database. The operation filters products based on the provided customer name and region, ensuring that only distinct product names are returned.")
async def get_distinct_product_names_by_customer_name_region(customer_name: str = Query(..., description="Customer name"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT DISTINCT T3.`Product Name` FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.`Customer Name` = ? AND T2.Region = ?", (customer_name, region))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get distinct customer names from west_superstore based on quantity and region
@app.get("/v1/superstore/west/distinct_customer_names_by_quantity_region", operation_id="get_distinct_customer_names_by_quantity_region", summary="Retrieves a list of unique customer names from the west_superstore dataset, filtered by a specified quantity and region. This operation is useful for identifying distinct customers based on their purchase quantity and geographical location.")
async def get_distinct_customer_names_by_quantity_region(quantity: int = Query(..., description="Quantity"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM west_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.Quantity = ? AND T1.Region = ?", (quantity, region))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the total profit from central_superstore based on city, state, and region
@app.get("/v1/superstore/central/total_profit_by_city_state_region", operation_id="get_total_profit_by_city_state_region", summary="Retrieves the total profit from the central superstore, filtered by a specific city, state, and region. The operation calculates the sum of profits from sales made to customers in the specified city and state, and belonging to the given region.")
async def get_total_profit_by_city_state_region(city: str = Query(..., description="City"), state: str = Query(..., description="State"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT SUM(T2.Profit) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` INNER JOIN product AS T3 ON T3.`Product ID` = T2.`Product ID` WHERE T1.City = ? AND T1.State = ? AND T2.Region = ?", (city, state, region))
    result = cursor.fetchone()
    if not result:
        return {"total_profit": []}
    return {"total_profit": result[0]}

# Endpoint to get distinct customer names from east_superstore based on ship date
@app.get("/v1/superstore/east/distinct_customer_names_by_ship_date", operation_id="get_distinct_customer_names_by_ship_date", summary="Retrieves a list of unique customer names from the east_superstore dataset, filtered by a specific ship date. The ship date is provided in 'YYYY-MM-DD' format.")
async def get_distinct_customer_names_by_ship_date(ship_date: str = Query(..., description="Ship date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM east_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T1.`Ship Date` = ?", (ship_date,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the product name with the lowest profit in a given year and region
@app.get("/v1/superstore/central/lowest_profit_product_by_year_region", operation_id="get_lowest_profit_product_by_year_region", summary="Retrieves the name of the product with the lowest profit for a specific year and region. The operation filters data based on the provided region and year, and returns the product name with the lowest profit value.")
async def get_lowest_profit_product_by_year_region(region: str = Query(..., description="Region"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.`Product Name` FROM central_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Region = ? AND STRFTIME('%Y', T1.`Order Date`) = ? ORDER BY T1.Profit ASC LIMIT 1", (region, year))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get sales, profit, and sub-category based on order ID and region
@app.get("/v1/superstore/sales_profit_subcategory", operation_id="get_sales_profit_subcategory", summary="Retrieves the sales, profit, and associated sub-category for a specific order in a given region from the east superstore. The order is identified by its unique ID, and the region is specified to filter the results.")
async def get_sales_profit_subcategory(order_id: str = Query(..., description="Order ID"), region: str = Query(..., description="Region")):
    cursor.execute("SELECT T1.Sales, T1.Profit, T2.`Sub-Category` FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T1.`Order ID` = ? AND T2.Region = ?", (order_id, region))
    result = cursor.fetchall()
    if not result:
        return {"sales_profit_subcategory": []}
    return {"sales_profit_subcategory": [{"sales": row[0], "profit": row[1], "sub_category": row[2]} for row in result]}

# Endpoint to get the top product name by quantity in a specific region
@app.get("/v1/superstore/top_product_by_quantity", operation_id="get_top_product_by_quantity", summary="Retrieves the name of the product with the highest quantity in the specified region of the east superstore.")
async def get_top_product_by_quantity(region: str = Query(..., description="Region")):
    cursor.execute("SELECT T2.`Product Name` FROM east_superstore AS T1 INNER JOIN product AS T2 ON T1.`Product ID` = T2.`Product ID` WHERE T2.Region = ? ORDER BY T1.Quantity DESC LIMIT 1", (region,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get distinct customer names based on region, ship mode, and sales threshold
@app.get("/v1/superstore/distinct_customer_names_sales_threshold", operation_id="get_distinct_customer_names_sales_threshold", summary="Retrieve a unique list of customer names from the south superstore, filtered by a specific region, ship mode, and sales threshold. The sales threshold is calculated as a percentage of the average sales in the south superstore. Only customers who meet the specified sales threshold are included in the result.")
async def get_distinct_customer_names_sales_threshold(region: str = Query(..., description="Region"), ship_mode: str = Query(..., description="Ship Mode"), sales_threshold: float = Query(..., description="Sales threshold percentage")):
    cursor.execute("SELECT DISTINCT T2.`Customer Name` FROM south_superstore AS T1 INNER JOIN people AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T2.Region = ? AND T1.`Ship Mode` = ? AND 100 * T1.Sales / ( SELECT AVG(Sales) FROM south_superstore ) > ?", (region, ship_mode, sales_threshold))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the percentage of orders with no discount in a specific region and state
@app.get("/v1/superstore/percentage_no_discount", operation_id="get_percentage_no_discount", summary="Retrieves the percentage of orders with no discount in the central superstore for a specific region and state. The calculation is based on the total number of orders in the given region and state, and the number of orders with no discount. The result is returned as a real number representing the percentage.")
async def get_percentage_no_discount(region: str = Query(..., description="Region"), state: str = Query(..., description="State")):
    cursor.execute("SELECT CAST(SUM(CASE  WHEN T2.Discount = 0 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM people AS T1 INNER JOIN central_superstore AS T2 ON T1.`Customer ID` = T2.`Customer ID` WHERE T2.Region = ? AND T1.State = ?", (region, state))
    result = cursor.fetchone()
    if not result:
        return {"percentage_no_discount": []}
    return {"percentage_no_discount": result[0]}

api_calls = [
    "/v1/superstore/distinct_product_names_by_order_id?order_id=CA-2011-112326",
    "/v1/superstore/longest_time_days_by_customer_name?customer_name=Aimee%20Bixby",
    "/v1/superstore/count_distinct_order_ids_by_customer_name_and_ship_mode?customer_name=Aimee%20Bixby&ship_mode=Standard%20Class",
    "/v1/superstore/count_distinct_order_ids_by_customer_name?customer_name=Aimee%20Bixby",
    "/v1/superstore/distinct_order_ids_by_customer_name_and_product_count?customer_name=Aimee%20Bixby&product_count=3",
    "/v1/superstore/count_distinct_order_ids_by_customer_name_and_category?category=Furniture&customer_name=Aimee%20Bixby",
    "/v1/superstore/distinct_product_names_by_customer_name_and_ship_year?customer_name=Aimee%20Bixby&ship_year=2016",
    "/v1/superstore/total_quantity_by_product_name?product_name=Telescoping%20Adjustable%20Floor%20Lamp",
    "/v1/superstore/distinct_customer_names_by_product_name?product_name=Telescoping%20Adjustable%20Floor%20Lamp",
    "/v1/superstore/count_distinct_customer_names_by_product_name_and_segment?product_name=Telescoping%20Adjustable%20Floor%20Lamp&segment=Consumer",
    "/v1/superstore/total_quantity_product_customer_date?customer_name=Aimee%20Bixby&product_name=Xerox%201952&order_date=2014-09-10",
    "/v1/superstore/count_distinct_orders_product_customer?customer_name=Aimee%20Bixby&product_name=Xerox%201952",
    "/v1/superstore/distinct_sales_after_discount_product_customer_date?customer_name=Aimee%20Bixby&product_name=Xerox%201952&order_date=2014-09-10",
    "/v1/superstore/distinct_profit_product_customer_date?customer_name=Aimee%20Bixby&product_name=Xerox%201952&order_date=2014-09-10",
    "/v1/superstore/count_distinct_products_subcategory_region_year?sub_category=Art&region=East&year=2013",
    "/v1/superstore/top_customer_by_profit",
    "/v1/superstore/top_product_by_sales",
    "/v1/superstore/most_recent_product_customer?customer_name=Darren%20Powers",
    "/v1/superstore/top_order_by_profit_product?product_name=Logitech%20G600%20MMO%20Gaming%20Mouse",
    "/v1/superstore/distinct_product_names_by_customer?customer_name=Alejandro%20Grove",
    "/v1/superstore/count_distinct_customers_by_product_city_quantity?product_name=Cardinal%20EasyOpen%20D-Ring%20Binders&city=Chicago&min_quantity=10",
    "/v1/superstore/distinct_product_names_by_profit?min_profit=1000",
    "/v1/superstore/product_names_by_ship_mode_region?ship_mode=First%20Class&region=East",
    "/v1/superstore/distinct_product_names_by_customer_region?customer_name=Becky%20Martin&region=Central",
    "/v1/superstore/distinct_customer_names_by_region_ship_mode?region=West&ship_mode=Second%20Class",
    "/v1/superstore/total_profit_by_customer_region?customer_name=Patrick%20Gardner&region=Central",
    "/v1/superstore/product_names_by_ship_order_date_region?ship_date=2013-03-04&order_date=2013-03-04&region=South",
    "/v1/superstore/total_sales_by_product_region?product_name=Avery%20Hi-Liter%20EverBold%20Pen%20Style%20Fluorescent%20Highlighters%2C%204%2FPack&region=Central",
    "/v1/superstore/top_product_by_customer_region?customer_name=Jonathan%20Doherty&region=East",
    "/v1/superstore/total_quantity_product_name?ship_date=2015-03-25&region=East",
    "/v1/superstore/distinct_customer_names?product_name=Global%20High-Back%20Leather%20Tilter%2C%20Burgundy&order_date=2013-10-13&region=East",
    "/v1/superstore/distinct_product_categories?customer_name=Katherine%20Murray&order_date=2018-11-04&region=South",
    "/v1/superstore/category_percentage_in_quantity?category=Furniture&region=West&ship_mode=Standard%20Class",
    "/v1/superstore/ship_dates_customer_region?customer_name=Ann%20Chong&region=Central",
    "/v1/superstore/distinct_segments_region_order?region=West&order_id=CA-2011-108189",
    "/v1/superstore/total_sales_product_region?product_name=Hon%20Valutask%20Swivel%20Chairs&region=West",
    "/v1/superstore/order_ids_region_customer?region=South&customer_name=Frank%20Olsen",
    "/v1/superstore/product_names_order_ship_date?order_date=2018-04-26&ship_date=2018-04-27&region=Central",
    "/v1/superstore/top_customer_city_state",
    "/v1/superstore/customer_highest_profit_by_region?region=East",
    "/v1/superstore/highest_quantity_by_city_state?city=Chicago&state=Illinois",
    "/v1/superstore/order_date_product_name_by_order_id_region?order_id=CA-2011-137274&region=Central",
    "/v1/superstore/distinct_customer_names_by_product_region?region=South&product_name=Xerox%2023",
    "/v1/superstore/highest_sales_product_by_category_region?category=Office%20Supplies&region=Central",
    "/v1/superstore/customer_highest_discount_by_region?region=West",
    "/v1/superstore/distinct_product_names_above_avg_profit_by_region?region=East&profit_percentage=0.98",
    "/v1/superstore/distinct_customer_names_by_sales_condition_region?region=East&sales_condition=80000",
    "/v1/superstore/count_distinct_order_ids_by_customer_year?customer_name=Maxwell%20Schwartz&year=2015",
    "/v1/superstore/total_profit_by_product?product_name=Cisco%20SPA301",
    "/v1/superstore/distinct_products_by_city?city=Coachella",
    "/v1/superstore/order_count_east_west_by_year?year=2015",
    "/v1/superstore/distinct_products_by_customer_year?customer_name=Matt%20Abelman&year=2013",
    "/v1/superstore/total_cost_by_customer_region_year?region=East&customer_name=Brad%20Thomas&year=2016",
    "/v1/superstore/distinct_customers_by_product?product_name=Plantronics%20Single%20Ear%20Headset",
    "/v1/superstore/distinct_ship_dates_products_by_customer?customer_name=Gene%20Hale",
    "/v1/superstore/order_count_by_ship_mode_category?ship_mode=First%20Class&category=Furniture",
    "/v1/superstore/order_count_by_category_customer?category=Office%20Supplies&customer_name=Cindy%20Stewart",
    "/v1/superstore/south/most_profitable_product_category",
    "/v1/superstore/south/products_by_customer?customer_name=Cindy%20Stewart",
    "/v1/superstore/west/products_by_ship_mode_and_date?ship_mode=Same%20Day&ship_date=2013%25",
    "/v1/superstore/east/distinct_categories_by_customer?customer_name=Sam%20Craven",
    "/v1/superstore/south/total_quantity_by_customer_and_product?customer_name=Cindy%20Stewart&product_name=Lexmark%20X%209575%20Professional%20All-in-One%20Color%20Printer",
    "/v1/superstore/central/distinct_products_by_ship_mode_and_quantity?ship_mode=Standard%20Class&min_quantity=10",
    "/v1/superstore/west/highest_sales_product_category",
    "/v1/superstore/east/total_sales_by_category_and_year?year=2016&category=Furniture",
    "/v1/superstore/west/average_sales_by_product?product_name=Sharp%20AL-1530CS%20Digital%20Copier",
    "/v1/superstore/category_percentage?category=Office%20Supplies",
    "/v1/superstore/state_percentage_comparison?state1=Texas&state2=Indiana",
    "/v1/superstore/most_popular_product_in_subcategory?sub_category=Art",
    "/v1/superstore/distinct_customers_by_year_and_order_count?year=2015&min_order_count=3",
    "/v1/superstore/max_profit_by_customer?customer_name=Anna%20Chung",
    "/v1/superstore/customer_id_count_by_year?customer_name=Corey%20Roper&year=2015",
    "/v1/superstore/sales_difference_east_west",
    "/v1/superstore/distinct_products_by_region_negative_profit?region=Central",
    "/v1/superstore/fastest_delivery_product_by_region?region=West",
    "/v1/superstore/order_count_by_product_region_ship_mode?product_name=O'Sullivan%20Plantations%202-Door%20Library%20in%20Landvery%20Oak&region=Central&ship_mode=First%20Class",
    "/v1/superstore/top_customer_name?segment=Corporate&state=Rhode%20Island&region=East&order_year=2016",
    "/v1/superstore/top_segment?region=East",
    "/v1/superstore/order_duration?order_id=CA-2011-134103",
    "/v1/superstore/count_distinct_order_ids_quantity_ship_mode?quantity=5&ship_mode=First%20Class",
    "/v1/superstore/order_ids_profit_less_than?profit=0",
    "/v1/superstore/top_customer_name_product?product_name=Hon%20Multipurpose%20Stacking%20Arm%20Chairs",
    "/v1/superstore/distinct_profit_values?product_name=O'Sullivan%20Living%20Dimensions%202-Shelf%20Bookcases",
    "/v1/superstore/sum_quantities?product_name=Hon%20Pagoda%20Stacking%20Chairs",
    "/v1/superstore/order_count_customer_ship_mode?customer_name=Aaron%20Bergman&ship_mode=Standard%20Class",
    "/v1/superstore/adjusted_sales_product_name?product_name=Blackstonian%20Pencils",
    "/v1/superstore/distinct_profits_product_name?product_name=Sauder%20Camden%20County%20Barrister%20Bookcase%2C%20Planked%20Cherry%20Finish",
    "/v1/superstore/count_categories_ship_mode?ship_mode=Standard%20Class",
    "/v1/superstore/highest_profit_order_city_state?city=Houston&state=Texas",
    "/v1/superstore/order_count_product_category?category=Furniture",
    "/v1/superstore/distinct_product_names_ship_date?ship_date=2013-03",
    "/v1/superstore/count_distinct_order_ids_customer_year?customer_name=Alan%20Barnes&order_year=2015",
    "/v1/superstore/east/distinct_product_names_by_order_id?order_id=CA-2011-141817",
    "/v1/superstore/central/discount_percentage_by_state?discount=0.2&state=Texas",
    "/v1/superstore/central/ship_mode_percentage_by_category_year?ship_mode=First%20Class&category=Furniture&year=2013",
    "/v1/superstore/west/distinct_customer_names_by_order_date_discount_region?order_date=2013-08-12&discount=0.2&region=West",
    "/v1/superstore/central/distinct_order_ids_by_product_name_order_date?product_name=Security-Tint%20Envelopes&order_date=2013-06-03",
    "/v1/superstore/central/distinct_product_names_by_customer_name_region?customer_name=Bill%20Shonely&region=Central",
    "/v1/superstore/west/distinct_customer_names_by_quantity_region?quantity=8&region=West",
    "/v1/superstore/central/total_profit_by_city_state_region?city=Houston&state=Texas&region=Central",
    "/v1/superstore/east/distinct_customer_names_by_ship_date?ship_date=2013-03-05",
    "/v1/superstore/central/lowest_profit_product_by_year_region?region=Central&year=2016",
    "/v1/superstore/sales_profit_subcategory?order_id=US-2011-126571&region=East",
    "/v1/superstore/top_product_by_quantity?region=East",
    "/v1/superstore/distinct_customer_names_sales_threshold?region=South&ship_mode=Standard%20Class&sales_threshold=88",
    "/v1/superstore/percentage_no_discount?region=Central&state=Indiana"
]
