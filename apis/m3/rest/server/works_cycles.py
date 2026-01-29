from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/works_cycles/works_cycles.sqlite')
cursor = conn.cursor()

# Endpoint to get the average standard cost of a product based on its product number
@app.get("/v1/works_cycles/avg_standard_cost_by_product_number", operation_id="get_avg_standard_cost", summary="Retrieves the average standard cost of a product by utilizing its unique product number. This operation calculates the average cost by aggregating the standard cost history of the product, providing a comprehensive overview of its cost over time.")
async def get_avg_standard_cost(product_number: str = Query(..., description="Product number of the product")):
    cursor.execute("SELECT AVG(T2.StandardCost) FROM Product AS T1 INNER JOIN ProductCostHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductNumber = ?", (product_number,))
    result = cursor.fetchone()
    if not result:
        return {"average_standard_cost": []}
    return {"average_standard_cost": result[0]}

# Endpoint to get the names and start dates of products with no end date in their cost history
@app.get("/v1/works_cycles/product_names_start_dates_no_end_date", operation_id="get_product_names_start_dates", summary="Retrieves the names and start dates of products that currently have an active cost history. This operation returns a list of products that are actively being tracked for cost changes, as indicated by the absence of an end date in their cost history.")
async def get_product_names_start_dates():
    cursor.execute("SELECT T1.Name, T2.StartDate FROM Product AS T1 INNER JOIN ProductCostHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T2.EndDate IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"product_names_start_dates": []}
    return {"product_names_start_dates": result}

# Endpoint to get the names of products where the difference in standard cost exceeds a specified value
@app.get("/v1/works_cycles/product_names_by_cost_difference", operation_id="get_product_names_by_cost_difference", summary="Retrieve the names of products for which the difference between their current standard cost and a historical standard cost surpasses a specified value. This operation allows you to identify products with significant cost changes over time.")
async def get_product_names_by_cost_difference(cost_difference: float = Query(..., description="Difference in standard cost")):
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN ProductCostHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.StandardCost - T2.StandardCost > ? GROUP BY T1.Name", (cost_difference,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the names and quantities of products in a specific shopping cart
@app.get("/v1/works_cycles/product_names_quantities_by_shopping_cart", operation_id="get_product_names_quantities", summary="Retrieves the names and quantities of products associated with a specific shopping cart. The operation requires the unique identifier of the shopping cart as input and returns a list of product names and their respective quantities.")
async def get_product_names_quantities(shopping_cart_id: int = Query(..., description="Shopping cart ID")):
    cursor.execute("SELECT T1.Name, T2.Quantity FROM Product AS T1 INNER JOIN ShoppingCartItem AS T2 ON T1.ProductID = T2.ProductID WHERE T2.ShoppingCartID = ?", (shopping_cart_id,))
    result = cursor.fetchall()
    if not result:
        return {"product_names_quantities": []}
    return {"product_names_quantities": result}

# Endpoint to get the names of products with quantities greater than a specified value in shopping carts
@app.get("/v1/works_cycles/product_names_by_quantity", operation_id="get_product_names_by_quantity", summary="Retrieve the names of products that have a quantity greater than the specified value in shopping carts. This operation allows you to filter products based on their quantity in shopping carts, providing a targeted list of product names that meet the specified quantity threshold.")
async def get_product_names_by_quantity(quantity: int = Query(..., description="Quantity of the product")):
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN ShoppingCartItem AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Quantity > ?", (quantity,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get distinct product names based on class and transaction type
@app.get("/v1/works_cycles/distinct_product_names_by_class_transaction_type", operation_id="get_distinct_product_names", summary="Retrieves a list of unique product names that belong to a specified product class and are associated with a given transaction type. The products are sorted alphabetically by name.")
async def get_distinct_product_names(product_class: str = Query(..., description="Class of the product"), transaction_type: str = Query(..., description="Transaction type")):
    cursor.execute("SELECT DISTINCT T1.Name FROM Product AS T1 INNER JOIN TransactionHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Class = ? AND T2.TransactionType = ? ORDER BY T1.Name", (product_class, transaction_type))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get distinct product names and list prices based on transaction quantity
@app.get("/v1/works_cycles/distinct_product_names_list_prices_by_quantity", operation_id="get_distinct_product_names_list_prices", summary="Retrieves a list of unique product names and their respective list prices for transactions involving quantities greater than the specified value. This operation is useful for identifying products and their prices based on transaction volume.")
async def get_distinct_product_names_list_prices(quantity: int = Query(..., description="Quantity of the product")):
    cursor.execute("SELECT DISTINCT T1.Name, T1.listPrice FROM Product AS T1 INNER JOIN TransactionHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Quantity > ?", (quantity,))
    result = cursor.fetchall()
    if not result:
        return {"product_names_list_prices": []}
    return {"product_names_list_prices": result}

# Endpoint to get the product name with the lowest transaction quantity for a specific class
@app.get("/v1/works_cycles/product_name_lowest_quantity_by_class", operation_id="get_product_name_lowest_quantity", summary="Retrieves the name of the product with the lowest transaction quantity within a specified class. The class is determined by the provided input parameter.")
async def get_product_name_lowest_quantity(product_class: str = Query(..., description="Class of the product")):
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN TransactionHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Class = ? ORDER BY T2.Quantity ASC LIMIT 1", (product_class,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the count of transactions for a specific product line
@app.get("/v1/works_cycles/transaction_count_by_product_line", operation_id="get_transaction_count", summary="Retrieves the total number of transactions associated with a specific product line. The product line is identified by the provided input parameter.")
async def get_transaction_count(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT COUNT(T2.TransactionID) FROM Product AS T1 INNER JOIN TransactionHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductLine = ?", (product_line,))
    result = cursor.fetchone()
    if not result:
        return {"transaction_count": []}
    return {"transaction_count": result[0]}

# Endpoint to get the total profit from a specific shopping cart
@app.get("/v1/works_cycles/total_profit_by_shopping_cart", operation_id="get_total_profit", summary="Retrieves the total profit generated by a specific shopping cart. This is calculated by summing the profit from each item in the cart, which is determined by subtracting the standard cost from the list price and then multiplying by the quantity of each item.")
async def get_total_profit(shopping_cart_id: int = Query(..., description="Shopping cart ID")):
    cursor.execute("SELECT SUM((T1.listPrice - T1.StandardCost) * T2.Quantity) FROM Product AS T1 INNER JOIN ShoppingCartItem AS T2 ON T1.ProductID = T2.ProductID WHERE T2.ShoppingCartID = ?", (shopping_cart_id,))
    result = cursor.fetchone()
    if not result:
        return {"total_profit": []}
    return {"total_profit": result[0]}

# Endpoint to get product names and list prices based on class
@app.get("/v1/works_cycles/product_names_list_prices_by_class", operation_id="get_product_names_list_prices_by_class", summary="Retrieves a list of product names and their respective list prices for a specified product class. The class is provided as an input parameter, allowing the operation to filter the results accordingly.")
async def get_product_names_list_prices_by_class(class_name: str = Query(..., description="Class of the product")):
    cursor.execute("SELECT Name, listPrice FROM Product WHERE Class = ?", (class_name,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the most common product line for finished goods
@app.get("/v1/works_cycles/most_common_product_line_finished_goods", operation_id="get_most_common_product_line_finished_goods", summary="Retrieves the product line that is most frequently associated with finished goods. The operation filters products based on the provided finished goods flag and returns the product line that appears most often among the filtered products.")
async def get_most_common_product_line_finished_goods(finished_goods_flag: int = Query(..., description="Finished goods flag (1 for finished goods)")):
    cursor.execute("SELECT ProductLine FROM Product WHERE FinishedGoodsFlag = ? GROUP BY ProductLine ORDER BY COUNT(FinishedGoodsFlag) DESC LIMIT 1", (finished_goods_flag,))
    result = cursor.fetchone()
    if not result:
        return {"product_line": []}
    return {"product_line": result[0]}

# Endpoint to get product reviews based on reviewer name pattern
@app.get("/v1/works_cycles/product_reviews_by_reviewer_name", operation_id="get_product_reviews_by_reviewer_name", summary="Retrieves product reviews based on a specified pattern for the reviewer's name. The operation filters the reviews by matching the reviewer name against the provided pattern, and returns the product ID, rating, and comments for each matching review.")
async def get_product_reviews_by_reviewer_name(reviewer_name_pattern: str = Query(..., description="Pattern for the reviewer name (e.g., 'J%')")):
    cursor.execute("SELECT ProductID, Rating, Comments FROM ProductReview WHERE ReviewerName LIKE ?", (reviewer_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get the product with the lowest rating
@app.get("/v1/works_cycles/product_with_lowest_rating", operation_id="get_product_with_lowest_rating", summary="Retrieves the product with the lowest rating, along with its product line and list price. The product is identified by its unique ID, which is matched with the corresponding product review to determine the rating. The result is sorted in ascending order by rating, and only the product with the lowest rating is returned.")
async def get_product_with_lowest_rating():
    cursor.execute("SELECT T1.Name, T1.ProductLine, T2.Rating, T1.listPrice FROM Product AS T1 INNER JOIN ProductReview AS T2 ON T1.ProductID = T2.ProductID ORDER BY T2.Rating ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result}

# Endpoint to get distinct product names based on price difference
@app.get("/v1/works_cycles/distinct_product_names_by_price_difference", operation_id="get_distinct_product_names_by_price_difference", summary="Retrieves unique product names for items with a price difference greater than the specified value. The price difference is calculated as the difference between the list price and the standard cost.")
async def get_distinct_product_names_by_price_difference(price_difference: float = Query(..., description="Difference between list price and standard cost")):
    cursor.execute("SELECT DISTINCT Name FROM Product WHERE listPrice - StandardCost > ?", (price_difference,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get product reviews based on product line
@app.get("/v1/works_cycles/product_reviews_by_product_line", operation_id="get_product_reviews_by_product_line", summary="Retrieves product reviews for a specified product line. The operation returns a list of reviews, each containing the product name, reviewer name, rating, and comments. The product line is used to filter the results.")
async def get_product_reviews_by_product_line(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT T1.Name, T2.ReviewerName, T2.Rating, T2.Comments FROM Product AS T1 INNER JOIN ProductReview AS T2 USING (productID) WHERE T1.ProductLine = ?", (product_line,))
    result = cursor.fetchall()
    if not result:
        return {"reviews": []}
    return {"reviews": result}

# Endpoint to get the count and average rating of a product
@app.get("/v1/works_cycles/product_count_avg_rating", operation_id="get_product_count_avg_rating", summary="Retrieves the total count of a specific product and its average rating from the reviews. The product is identified by its name.")
async def get_product_count_avg_rating(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT COUNT(T1.ProductID), AVG(T2.Rating) FROM Product AS T1 INNER JOIN ProductReview AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "avg_rating": []}
    return {"count": result[0], "avg_rating": result[1]}

# Endpoint to get product names where rejected quantity equals received quantity
@app.get("/v1/works_cycles/product_names_rejected_equals_received", operation_id="get_product_names_rejected_equals_received", summary="Retrieves the names of products for which the quantity rejected upon receipt matches the quantity received and is not zero. This operation identifies products with a complete rejection of the received quantity, indicating potential quality or delivery issues.")
async def get_product_names_rejected_equals_received():
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN PurchaseOrderDetail AS T2 ON T1.ProductID = T2.ProductID WHERE T2.RejectedQty = T2.ReceivedQty AND T2.RejectedQty <> 0")
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the product with the highest line total and zero rejected quantity
@app.get("/v1/works_cycles/product_highest_line_total_zero_rejected", operation_id="get_product_highest_line_total_zero_rejected", summary="Get the product with the highest line total and zero rejected quantity")
async def get_product_highest_line_total_zero_rejected(rejected_qty: int = Query(..., description="Rejected quantity (0 for no rejected quantity)")):
    cursor.execute("SELECT T1.Name, T2.UnitPrice FROM Product AS T1 INNER JOIN PurchaseOrderDetail AS T2 ON T1.ProductID = T2.ProductID WHERE T2.RejectedQty = ? ORDER BY T2.LineTotal DESC LIMIT 1", (rejected_qty,))
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result}

# Endpoint to get product names and product lines based on order quantity
@app.get("/v1/works_cycles/product_names_lines_by_order_quantity", operation_id="get_product_names_lines_by_order_quantity", summary="Retrieve the names and product lines of items with an order quantity exceeding the provided threshold. This operation filters products based on their order quantity, returning only those that surpass the specified value.")
async def get_product_names_lines_by_order_quantity(order_qty: int = Query(..., description="Order quantity")):
    cursor.execute("SELECT T1.Name, T1.ProductLine FROM Product AS T1 INNER JOIN PurchaseOrderDetail AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderQty > ?", (order_qty,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the total order quantity for a specific product line
@app.get("/v1/works_cycles/total_order_quantity_product_line", operation_id="get_total_order_quantity", summary="Retrieves the total quantity of orders for a specific product line. The product line is specified as an input parameter, and the operation calculates the sum of order quantities for all products within that line.")
async def get_total_order_quantity(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT SUM(T2.OrderQty) FROM Product AS T1 INNER JOIN PurchaseOrderDetail AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductLine = ?", (product_line,))
    result = cursor.fetchone()
    if not result:
        return {"total_order_quantity": []}
    return {"total_order_quantity": result[0]}

# Endpoint to get the product name and line total for a specific class with the highest order quantity times unit price
@app.get("/v1/works_cycles/product_line_total_highest_order_quantity_unit_price", operation_id="get_product_line_total", summary="Retrieves the product name and corresponding line total for the product line with the highest order quantity times unit price within a specified class type. The class type is provided as an input parameter.")
async def get_product_line_total(class_type: str = Query(..., description="Class type")):
    cursor.execute("SELECT T1.Name, T2.LineTotal FROM Product AS T1 INNER JOIN PurchaseOrderDetail AS T2 ON T1.ProductID = T2.ProductID WHERE Class = ? ORDER BY OrderQty * UnitPrice DESC LIMIT 1", (class_type,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": [], "line_total": []}
    return {"product_name": result[0], "line_total": result[1]}

# Endpoint to get the product name with the highest difference between last receipt cost and standard price
@app.get("/v1/works_cycles/product_highest_cost_difference", operation_id="get_product_highest_cost_difference", summary="Retrieves the name of the product with the largest discrepancy between its most recent purchase cost and its standard price. The operation compares the last recorded cost of each product with its standard price, and returns the product with the highest difference.")
async def get_product_highest_cost_difference():
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN ProductVendor AS T2 ON T1.ProductID = T2.ProductID ORDER BY T2.LastReceiptCost - T2.StandardPrice DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the product name and profit margin for the product with the highest rating
@app.get("/v1/works_cycles/product_highest_rating_profit_margin", operation_id="get_product_highest_rating_profit_margin", summary="Retrieves the name and profit margin of the product with the highest rating. The profit margin is calculated as the difference between the list price and the standard cost. The product rating is determined by reviews.")
async def get_product_highest_rating_profit_margin():
    cursor.execute("SELECT T1.Name, T1.listPrice - T1.StandardCost FROM Product AS T1 INNER JOIN ProductReview AS T2 ON T1.ProductID = T2.ProductID ORDER BY T2.Rating DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product_name": [], "profit_margin": []}
    return {"product_name": result[0], "profit_margin": result[1]}

# Endpoint to get the currency pair with the highest average rate
@app.get("/v1/works_cycles/currency_pair_highest_average_rate", operation_id="get_currency_pair_highest_average_rate", summary="Retrieves the currency pair with the highest average exchange rate. The operation returns the currency codes for the pair with the highest average rate, as determined by the historical data in the CurrencyRate table.")
async def get_currency_pair_highest_average_rate():
    cursor.execute("SELECT FromCurrencyCode, ToCurrencyCode FROM CurrencyRate ORDER BY AverageRate DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"from_currency_code": [], "to_currency_code": []}
    return {"from_currency_code": result[0], "to_currency_code": result[1]}

# Endpoint to get the order quantity for the highest unit price
@app.get("/v1/works_cycles/order_quantity_highest_unit_price", operation_id="get_order_quantity_highest_unit_price", summary="Retrieves the order quantity associated with the highest unit price from the purchase order details.")
async def get_order_quantity_highest_unit_price():
    cursor.execute("SELECT OrderQty FROM PurchaseOrderDetail ORDER BY UnitPrice DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"order_quantity": []}
    return {"order_quantity": result[0]}

# Endpoint to get the sales territory name with the highest sales last year for a specific country region code and territory names
@app.get("/v1/works_cycles/sales_territory_highest_sales_last_year", operation_id="get_sales_territory_highest_sales_last_year", summary="Retrieves the name of the sales territory with the highest sales last year for a given country region code and two specified territory names. The operation filters the sales territories based on the provided country region code and territory names, then sorts them by last year's sales in descending order. The name of the territory with the highest sales is returned.")
async def get_sales_territory_highest_sales_last_year(country_region_code: str = Query(..., description="Country region code"), name1: str = Query(..., description="First territory name"), name2: str = Query(..., description="Second territory name")):
    cursor.execute("SELECT Name FROM SalesTerritory WHERE CountryRegionCode = ? AND (Name = ? OR Name = ?) ORDER BY SalesLastYear DESC LIMIT 1", (country_region_code, name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"territory_name": []}
    return {"territory_name": result[0]}

# Endpoint to get the names of employees with a specific job title, document level, and status
@app.get("/v1/works_cycles/employee_names_job_title_document_level_status", operation_id="get_employee_names", summary="Retrieve the full names of employees who hold a specific job title, have a certain document level, and possess a particular status. The results are grouped by first, middle, and last names.")
async def get_employee_names(job_title: str = Query(..., description="Job title"), document_level: int = Query(..., description="Document level"), status: int = Query(..., description="Status")):
    cursor.execute("SELECT T1.FirstName, T1.MiddleName, T1.LastName FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Document AS T3 ON T3.Owner = T2.BusinessEntityID WHERE T2.JobTitle = ? AND T3.DocumentLevel = ? AND T3.Status = ? GROUP BY T1.FirstName, T1.MiddleName, T1.LastName", (job_title, document_level, status))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"first_name": row[0], "middle_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get the customer ID with the highest subtotal
@app.get("/v1/works_cycles/customer_with_highest_subtotal", operation_id="get_customer_with_highest_subtotal", summary="Retrieves the unique identifier of the customer who has the highest total sales amount. This operation considers all sales orders and joins them with the corresponding salesperson data to ensure accurate calculation. The result is sorted in descending order based on the total sales amount, and only the top record is returned.")
async def get_customer_with_highest_subtotal():
    cursor.execute("SELECT T1.CustomerID FROM SalesOrderHeader AS T1 INNER JOIN SalesPerson AS T2 ON T1.SalesPersonID = T2.BusinessEntityID ORDER BY T1.SubTotal DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer_id": []}
    return {"customer_id": result[0]}

# Endpoint to get the total price for a specific special offer, product, and sales order
@app.get("/v1/works_cycles/total_price_special_offer", operation_id="get_total_price_special_offer", summary="Retrieves the total price for a specific product and sales order associated with a given special offer. The calculation is based on the unit price and order quantity of the product in the sales order detail. The special offer is identified by its description and ID, while the product and sales order are identified by their respective IDs.")
async def get_total_price_special_offer(description: str = Query(..., description="Description of the special offer"), special_offer_id: int = Query(..., description="Special offer ID"), product_id: int = Query(..., description="Product ID"), sales_order_id: int = Query(..., description="Sales order ID")):
    cursor.execute("SELECT T2.UnitPrice * T2.OrderQty FROM SpecialOffer AS T1 INNER JOIN SalesOrderDetail AS T2 ON T1.SpecialOfferID = T2.SpecialOfferID WHERE T1.Description = ? AND T1.SpecialOfferID = ? AND T2.ProductID = ? AND T2.SalesOrderID = ?", (description, special_offer_id, product_id, sales_order_id))
    result = cursor.fetchone()
    if not result:
        return {"total_price": []}
    return {"total_price": result[0]}

# Endpoint to get the count of products based on manufacturing criteria
@app.get("/v1/works_cycles/count_products_manufacturing_criteria", operation_id="get_count_products_manufacturing_criteria", summary="Retrieves the total count of products that meet specific manufacturing criteria. The criteria include the make flag, days to manufacture, bill of materials level, and reorder point. This operation provides a quantitative overview of products that match these conditions.")
async def get_count_products_manufacturing_criteria(make_flag: int = Query(..., description="Make flag"), days_to_manufacture: int = Query(..., description="Days to manufacture"), bom_level: int = Query(..., description="BOM level"), reorder_point: int = Query(..., description="Reorder point")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM Product AS T1 INNER JOIN BillOfMaterials AS T2 ON T1.ProductID = T2.ProductAssemblyID WHERE T1.MakeFlag = ? AND T1.DaysToManufacture = ? AND T2.BOMLevel = ? AND T1.ReorderPoint <= ?", (make_flag, days_to_manufacture, bom_level, reorder_point))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest bonus for a specific country region code
@app.get("/v1/works_cycles/highest_bonus_country_region", operation_id="get_highest_bonus_country_region", summary="Retrieves the highest bonus awarded to a salesperson in a specific country region. The salesperson with the highest sales quota in the given country region is considered for this calculation. The country region code is required as an input parameter.")
async def get_highest_bonus_country_region(country_region_code: str = Query(..., description="Country region code")):
    cursor.execute("SELECT T2.Bonus FROM SalesTerritory AS T1 INNER JOIN SalesPerson AS T2 ON T1.TerritoryID = T2.TerritoryID WHERE T1.CountryRegionCode = ? ORDER BY T2.SalesQuota DESC LIMIT 1", (country_region_code,))
    result = cursor.fetchone()
    if not result:
        return {"bonus": []}
    return {"bonus": result[0]}

# Endpoint to get the product name with the lowest rating
@app.get("/v1/works_cycles/product_name_lowest_rating", operation_id="get_product_name_lowest_rating", summary="Retrieves the name of the product with the lowest rating from the database. This operation identifies the product with the lowest rating based on user reviews and returns its name. The result provides insight into the product that has received the least favorable feedback.")
async def get_product_name_lowest_rating():
    cursor.execute("SELECT T2.Name FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Rating = ( SELECT Rating FROM ProductReview ORDER BY Rating ASC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the count of employees in a department within a date range
@app.get("/v1/works_cycles/count_employees_department_date_range", operation_id="get_count_employees_department_date_range", summary="Retrieves the total number of employees who started working in a specific department within a given date range. The date range is defined by a start date minimum and a start date maximum, both in 'YYYY-MM-DD' format. The department is identified by its name.")
async def get_count_employees_department_date_range(start_date_min: str = Query(..., description="Start date minimum in 'YYYY-MM-DD' format"), start_date_max: str = Query(..., description="Start date maximum in 'YYYY-MM-DD' format"), department_name: str = Query(..., description="Department name")):
    cursor.execute("SELECT COUNT(T2.BusinessEntityID) FROM Department AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.DepartmentID = T2.DepartmentID WHERE T2.StartDate >= ? AND T2.StartDate < ? AND T1.Name = ?", (start_date_min, start_date_max, department_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest paid employee with specific marital status and gender
@app.get("/v1/works_cycles/highest_paid_employee_marital_gender", operation_id="get_highest_paid_employee_marital_gender", summary="Retrieves the details of the highest paid employee who has the specified marital status and gender. The response includes the employee's first name, middle name, last name, and job title.")
async def get_highest_paid_employee_marital_gender(marital_status: str = Query(..., description="Marital status"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT T3.FirstName, T3.MiddleName, T3.LastName, T1.JobTitle FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Person AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T1.MaritalStatus = ? AND T1.Gender = ? ORDER BY T2.Rate DESC LIMIT 1", (marital_status, gender))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "middle_name": result[1], "last_name": result[2], "job_title": result[3]}}

# Endpoint to get employee details based on job title
@app.get("/v1/works_cycles/employee_details_job_title", operation_id="get_employee_details_job_title", summary="Retrieves the first name, middle name, last name, and hire date of employees who hold a specific job title. The job title is provided as an input parameter.")
async def get_employee_details_job_title(job_title: str = Query(..., description="Job title")):
    cursor.execute("SELECT T2.FirstName, T2.MiddleName, T2.LastName, T1.HireDate FROM Employee AS T1 INNER JOIN Person AS T2 USING (BusinessEntityID) WHERE T1.JobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    employees = [{"first_name": row[0], "middle_name": row[1], "last_name": row[2], "hire_date": row[3]} for row in result]
    return {"employees": employees}

# Endpoint to get the count of current employees with a pay rate below a specified amount
@app.get("/v1/works_cycles/count_current_employees_pay_rate", operation_id="get_count_current_employees_pay_rate", summary="Retrieves the number of active employees who are currently paid at or below a specified rate. The operation filters employees based on their current status and pay rate, providing a count of those who meet the criteria.")
async def get_count_current_employees_pay_rate(current_flag: int = Query(..., description="Current flag"), rate: float = Query(..., description="Pay rate")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.CurrentFlag = ? AND T2.Rate <= ?", (current_flag, rate))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most recently started department
@app.get("/v1/works_cycles/most_recent_department", operation_id="get_most_recent_department", summary="Retrieves the name of the department that started most recently. This operation fetches the department name from the Department table, which is joined with the EmployeeDepartmentHistory table based on the DepartmentID. The department with the most recent start date is selected.")
async def get_most_recent_department():
    cursor.execute("SELECT T1.Name FROM Department AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.DepartmentID = T2.DepartmentID ORDER BY T2.StartDate DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"department_name": []}
    return {"department_name": result[0]}

# Endpoint to get employee details based on person type
@app.get("/v1/works_cycles/employee_details_by_person_type", operation_id="get_employee_details", summary="Retrieves the first name, middle name, last name, and gender of employees based on the specified person type. The person type is a categorical identifier that distinguishes between different types of employees.")
async def get_employee_details(person_type: str = Query(..., description="Person type (e.g., 'SP')")):
    cursor.execute("SELECT T2.FirstName, T2.MiddleName, T2.LastName, T1.Gender FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.PersonType = ?", (person_type,))
    result = cursor.fetchall()
    if not result:
        return {"employee_details": []}
    return {"employee_details": result}

# Endpoint to get the pay frequency of the employee with the least sick leave hours
@app.get("/v1/works_cycles/pay_frequency_least_sick_leave", operation_id="get_pay_frequency_least_sick_leave", summary="Retrieves the pay frequency of the employee who has taken the least amount of sick leave hours. This operation returns the pay frequency of the employee with the lowest number of sick leave hours, as recorded in the employee and employee pay history tables.")
async def get_pay_frequency_least_sick_leave():
    cursor.execute("SELECT T2.PayFrequency FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID ORDER BY T1.SickLeaveHours ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"pay_frequency": []}
    return {"pay_frequency": result[0]}

# Endpoint to get the job title of the employee with the lowest pay rate
@app.get("/v1/works_cycles/job_title_lowest_pay_rate", operation_id="get_job_title_lowest_pay_rate", summary="Retrieves the job title of the employee who has the lowest pay rate. This operation fetches the job title from the Employee table and matches it with the corresponding pay rate from the EmployeePayHistory table. The data is then sorted by pay rate in ascending order, and the job title of the employee with the lowest pay rate is returned.")
async def get_job_title_lowest_pay_rate():
    cursor.execute("SELECT T1.JobTitle FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID ORDER BY T2.Rate ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"job_title": []}
    return {"job_title": result[0]}

# Endpoint to get the count of employees in a specific department
@app.get("/v1/works_cycles/employee_count_by_department", operation_id="get_employee_count_by_department", summary="Retrieves the total number of employees who have been part of a specified department at any point in time. The department is identified by its name.")
async def get_employee_count_by_department(department_name: str = Query(..., description="Name of the department")):
    cursor.execute("SELECT COUNT(T2.BusinessEntityID) FROM Department AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 USING (DepartmentID) WHERE T1.Name = ?", (department_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest and lowest profit margins of products
@app.get("/v1/works_cycles/highest_lowest_profit_margins", operation_id="get_highest_lowest_profit_margins", summary="Retrieves the highest and lowest profit margins of products by comparing the list price and standard cost of each product. The profit margin is calculated as the difference between the list price and standard cost. The product with the highest list price difference is considered to have the highest profit margin, while the product with the lowest list price difference is considered to have the lowest profit margin.")
async def get_highest_lowest_profit_margins():
    cursor.execute("SELECT ( SELECT listPrice - StandardCost FROM Product WHERE listPrice != 0 ORDER BY listPrice DESC LIMIT 1 ) , ( SELECT listPrice - StandardCost FROM Product WHERE listPrice != 0 ORDER BY listPrice LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"profit_margins": []}
    return {"highest_profit_margin": result[0], "lowest_profit_margin": result[1]}

# Endpoint to get vendor details based on credit rating
@app.get("/v1/works_cycles/vendor_details_by_credit_rating", operation_id="get_vendor_details_by_credit_rating", summary="Retrieves vendor details, including the vendor's name and the difference between the last receipt cost and the standard price, based on the provided credit rating.")
async def get_vendor_details_by_credit_rating(credit_rating: int = Query(..., description="Credit rating of the vendor")):
    cursor.execute("SELECT T2.Name, T1.LastReceiptCost - T1.StandardPrice FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.CreditRating = ?", (credit_rating,))
    result = cursor.fetchall()
    if not result:
        return {"vendor_details": []}
    return {"vendor_details": result}

# Endpoint to get the count of addresses with a non-empty second address line
@app.get("/v1/works_cycles/count_addresses_with_second_line", operation_id="get_count_addresses_with_second_line", summary="Retrieves the total number of addresses that have a non-empty second address line. This operation provides a count of addresses with additional location details, which can be useful for understanding address distribution and data completeness.")
async def get_count_addresses_with_second_line():
    cursor.execute("SELECT COUNT(*) FROM Address WHERE AddressLine2 <> ''")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the postal code of the most recently modified address
@app.get("/v1/works_cycles/postal_code_most_recent_address", operation_id="get_postal_code_most_recent_address", summary="Retrieves the postal code of the most recently updated address. This operation returns the postal code of the address that was last modified, providing the latest postal code information.")
async def get_postal_code_most_recent_address():
    cursor.execute("SELECT PostalCode FROM Address ORDER BY ModifiedDate DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"postal_code": []}
    return {"postal_code": result[0]}

# Endpoint to get the longest duration between start and end dates in bill of materials
@app.get("/v1/works_cycles/longest_duration_bill_of_materials", operation_id="get_longest_duration_bill_of_materials", summary="Retrieves the longest duration between the start and end dates in the bill of materials. This operation returns the maximum time difference between the start and end dates of a bill of materials, providing insight into the longest production cycle.")
async def get_longest_duration_bill_of_materials():
    cursor.execute("SELECT JULIANDAY(EndDate) - JULIANDAY(StartDate) FROM BillOfMaterials ORDER BY JULIANDAY(EndDate) - JULIANDAY(StartDate) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"duration": []}
    return {"duration": result[0]}

# Endpoint to get the count of bill of materials with no end date
@app.get("/v1/works_cycles/count_bill_of_materials_no_end_date", operation_id="get_count_bill_of_materials_no_end_date", summary="Retrieves the total number of bill of materials records that do not have an end date specified.")
async def get_count_bill_of_materials_no_end_date():
    cursor.execute("SELECT COUNT(BillOfMaterialsID) FROM BillOfMaterials WHERE EndDate IS NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the UnitMeasureCode of the BillOfMaterials with the highest PerAssemblyQty
@app.get("/v1/works_cycles/unit_measure_code_highest_per_assembly_qty", operation_id="get_unit_measure_code_highest_per_assembly_qty", summary="Retrieves the unit of measure code associated with the bill of materials that has the highest per-assembly quantity, up to a specified limit. This operation is useful for identifying the most frequently used unit of measure in the assembly process.")
async def get_unit_measure_code_highest_per_assembly_qty(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT UnitMeasureCode FROM BillOfMaterials ORDER BY PerAssemblyQty DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"UnitMeasureCode": []}
    return {"UnitMeasureCode": result[0]}

# Endpoint to get the count of documents with a null DocumentSummary
@app.get("/v1/works_cycles/count_documents_null_summary", operation_id="get_count_documents_null_summary", summary="Retrieves the total number of documents that do not have a summary. This operation is useful for tracking the number of documents that are incomplete or require further attention.")
async def get_count_documents_null_summary():
    cursor.execute("SELECT COUNT(DocumentNode) FROM Document WHERE DocumentSummary IS NULL")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the titles of documents with a specific status
@app.get("/v1/works_cycles/document_titles_by_status", operation_id="get_document_titles_by_status", summary="Retrieves the titles of documents that match the specified status. The operation filters documents based on their status and returns a list of corresponding titles.")
async def get_document_titles_by_status(status: int = Query(..., description="Status of the document")):
    cursor.execute("SELECT Title FROM Document WHERE Status = ?", (status,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get distinct BusinessEntityID and JobTitle of employees owning documents with a specific status
@app.get("/v1/works_cycles/employee_details_by_document_status", operation_id="get_employee_details_by_document_status", summary="Retrieves unique employee identifiers and job titles for individuals who possess documents with a specified status. The status of the document is a required input parameter.")
async def get_employee_details_by_document_status(status: int = Query(..., description="Status of the document")):
    cursor.execute("SELECT DISTINCT T2.BusinessEntityID, T2.JobTitle FROM Document AS T1 INNER JOIN Employee AS T2 ON T1.Owner = T2.BusinessEntityID WHERE T1.Status = ?", (status,))
    result = cursor.fetchall()
    if not result:
        return {"employee_details": []}
    return {"employee_details": [{"BusinessEntityID": row[0], "JobTitle": row[1]} for row in result]}

# Endpoint to get the PayFrequency of the employee with the earliest BirthDate
@app.get("/v1/works_cycles/pay_frequency_earliest_birthdate", operation_id="get_pay_frequency_earliest_birthdate", summary="Retrieves the pay frequency of the employee with the earliest birth date, limiting the results to the specified number.")
async def get_pay_frequency_earliest_birthdate(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T1.PayFrequency FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID ORDER BY T2.BirthDate ASC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"PayFrequency": []}
    return {"PayFrequency": result[0]}

# Endpoint to get the count of employees with a specific marital status and the highest PayFrequency
@app.get("/v1/works_cycles/count_employees_marital_status_highest_pay_frequency", operation_id="get_count_employees_marital_status_highest_pay_frequency", summary="Retrieves the total number of employees who share a particular marital status and receive the highest pay frequency. The marital status is specified as an input parameter. The number of results returned can be limited by another input parameter.")
async def get_count_employees_marital_status_highest_pay_frequency(marital_status: str = Query(..., description="Marital status of the employee"), limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.MaritalStatus = ? AND T1.PayFrequency = ( SELECT PayFrequency FROM EmployeePayHistory ORDER BY PayFrequency DESC LIMIT ? )", (marital_status, limit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the Rate of the employee with the most recent HireDate
@app.get("/v1/works_cycles/rate_most_recent_hire_date", operation_id="get_rate_most_recent_hire_date", summary="Retrieves the pay rate of the employee who was most recently hired, optionally limiting the number of results returned.")
async def get_rate_most_recent_hire_date(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T1.Rate FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID ORDER BY T2.HireDate DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"Rate": []}
    return {"Rate": result[0]}

# Endpoint to get the count of employees with a specific gender and rate greater than a specified value
@app.get("/v1/works_cycles/count_employees_gender_rate", operation_id="get_count_employees_gender_rate", summary="Retrieves the total number of employees of a specified gender who have a rate greater than a given threshold. This operation calculates the sum of employees that meet the gender and rate criteria, providing a count of qualifying employees.")
async def get_count_employees_gender_rate(gender: str = Query(..., description="Gender of the employee"), rate: float = Query(..., description="Rate threshold")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Gender = ? THEN 1 ELSE 0 END) FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.Rate > ?", (gender, rate))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest Rate of salaried employees
@app.get("/v1/works_cycles/highest_rate_salaried_employees", operation_id="get_highest_rate_salaried_employees", summary="Retrieves the highest rate of pay for salaried employees, based on the provided salaried flag. The number of results returned can be limited by specifying the desired limit. The data is ordered in descending order based on the rate of pay.")
async def get_highest_rate_salaried_employees(salaried_flag: int = Query(..., description="Salaried flag of the employee"), limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T1.Rate FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.SalariedFlag = ? ORDER BY T1.Rate DESC LIMIT ?", (salaried_flag, limit))
    result = cursor.fetchone()
    if not result:
        return {"Rate": []}
    return {"Rate": result[0]}

# Endpoint to get the VacationHours of the employee with the highest Rate
@app.get("/v1/works_cycles/vacation_hours_highest_rate_employee", operation_id="get_vacation_hours_highest_rate_employee", summary="Retrieves the vacation hours of the employee with the highest pay rate, as determined by the provided limit. The limit parameter is used to specify the number of top-paying employees to consider when determining the highest rate.")
async def get_vacation_hours_highest_rate_employee(limit: int = Query(1, description="Limit the number of results returned")):
    cursor.execute("SELECT T2.VacationHours FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.BusinessEntityID = ( SELECT BusinessEntityID FROM EmployeePayHistory ORDER BY Rate DESC LIMIT ? )", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"VacationHours": []}
    return {"VacationHours": result[0]}

# Endpoint to get the highest rate of an employee based on vacation hours
@app.get("/v1/works_cycles/highest_rate_by_vacation_hours", operation_id="get_highest_rate_by_vacation_hours", summary="Retrieves the highest pay rate among employees, considering those with the most vacation hours. The operation returns the top pay rate from the EmployeePayHistory table, which is joined with the Employee table to consider vacation hours. The result is ordered by vacation hours in descending order, with the highest rate being returned.")
async def get_highest_rate_by_vacation_hours():
    cursor.execute("SELECT T1.Rate FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID ORDER BY T2.VacationHours DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"rate": []}
    return {"rate": result[0]}

# Endpoint to get the count of employees with sick leave hours greater than a specified value and rate greater than a specified value
@app.get("/v1/works_cycles/count_employees_sick_leave_rate", operation_id="get_count_employees_sick_leave_rate", summary="Retrieves the number of employees who have exceeded the specified sick leave hours and rate. This operation considers the sick leave hours and rate of each employee to determine the count.")
async def get_count_employees_sick_leave_rate(sick_leave_hours: int = Query(..., description="Sick leave hours"), rate: float = Query(..., description="Rate")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.SickLeaveHours > ? AND T1.Rate > ?", (sick_leave_hours, rate))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of employees with specific current flag, gender, and pay frequency
@app.get("/v1/works_cycles/count_employees_current_flag_gender_pay_frequency", operation_id="get_count_employees_current_flag_gender_pay_frequency", summary="Retrieves the total number of employees who meet the specified current flag, gender, and pay frequency criteria. This operation is useful for obtaining a count of employees based on their current employment status, gender, and pay frequency. The input parameters determine the conditions for the count.")
async def get_count_employees_current_flag_gender_pay_frequency(current_flag: int = Query(..., description="Current flag"), gender: str = Query(..., description="Gender"), pay_frequency: int = Query(..., description="Pay frequency")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.CurrentFlag = ? AND T2.Gender = ? AND T1.PayFrequency = ?", (current_flag, gender, pay_frequency))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of employees with specific gender and person type
@app.get("/v1/works_cycles/count_employees_gender_person_type", operation_id="get_count_employees_gender_person_type", summary="Retrieves the total number of employees who match the specified gender and person type. The gender and person type are provided as input parameters.")
async def get_count_employees_gender_person_type(gender: str = Query(..., description="Gender"), person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.Gender = ? AND T2.PersonType = ?", (gender, person_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the person type of the youngest employee
@app.get("/v1/works_cycles/youngest_employee_person_type", operation_id="get_youngest_employee_person_type", summary="Retrieves the person type of the youngest employee in the database. This operation identifies the employee with the most recent birth date and returns their corresponding person type.")
async def get_youngest_employee_person_type():
    cursor.execute("SELECT T2.PersonType FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID ORDER BY T1.BirthDate ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"person_type": []}
    return {"person_type": result[0]}

# Endpoint to get the name style of the employee with the lowest rate
@app.get("/v1/works_cycles/lowest_rate_employee_name_style", operation_id="get_lowest_rate_employee_name_style", summary="Retrieves the name style of the employee who has the lowest non-null pay rate. The name style is obtained from the associated person record.")
async def get_lowest_rate_employee_name_style():
    cursor.execute("SELECT T2.NameStyle FROM EmployeePayHistory AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.Rate IS NOT NULL ORDER BY T1.Rate ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name_style": []}
    return {"name_style": result[0]}

# Endpoint to get the count of employees with specific name style and marital status
@app.get("/v1/works_cycles/count_employees_name_style_marital_status", operation_id="get_count_employees_name_style_marital_status", summary="Retrieves the total number of employees who have a specific name style and marital status. The name style and marital status are provided as input parameters, allowing for a targeted count of employees that meet these criteria.")
async def get_count_employees_name_style_marital_status(name_style: int = Query(..., description="Name style"), marital_status: str = Query(..., description="Marital status")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.NameStyle = ? AND T1.MaritalStatus = ?", (name_style, marital_status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of employees with specific email promotion and sick leave hours
@app.get("/v1/works_cycles/count_employees_email_promotion_sick_leave", operation_id="get_count_employees_email_promotion_sick_leave", summary="Retrieves the total number of employees who have a specific email promotion and more than a certain number of sick leave hours. The response is based on the provided email promotion and minimum sick leave hours.")
async def get_count_employees_email_promotion_sick_leave(email_promotion: int = Query(..., description="Email promotion"), sick_leave_hours: int = Query(..., description="Sick leave hours")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.EmailPromotion = ? AND T1.SickLeaveHours > ?", (email_promotion, sick_leave_hours))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the business entity IDs of employees with specific email promotion and vacation hours
@app.get("/v1/works_cycles/employee_ids_email_promotion_vacation_hours", operation_id="get_employee_ids_email_promotion_vacation_hours", summary="Retrieve the unique identifiers of employees who have a specific email promotion status and more than a certain number of vacation hours. This operation allows you to filter employees based on their email promotion status and available vacation hours, providing a targeted list of employee IDs for further processing or analysis.")
async def get_employee_ids_email_promotion_vacation_hours(email_promotion: int = Query(..., description="Email promotion"), vacation_hours: int = Query(..., description="Vacation hours")):
    cursor.execute("SELECT T1.BusinessEntityID FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.EmailPromotion = ? AND T1.VacationHours > ?", (email_promotion, vacation_hours))
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": [row[0] for row in result]}

# Endpoint to get the additional contact info of the youngest employee with a specific person type
@app.get("/v1/works_cycles/youngest_employee_additional_contact_info", operation_id="get_youngest_employee_additional_contact_info", summary="Retrieves the additional contact information of the youngest employee with the specified person type. The operation filters employees based on the provided person type and sorts them by birth date in ascending order to identify the youngest employee. The additional contact information associated with the identified employee is then returned.")
async def get_youngest_employee_additional_contact_info(person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT T2.AdditionalContactInfo FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE PersonType = ? ORDER BY T1.BirthDate ASC LIMIT 1", (person_type,))
    result = cursor.fetchone()
    if not result:
        return {"additional_contact_info": []}
    return {"additional_contact_info": result[0]}

# Endpoint to get the first names of employees based on name style and gender
@app.get("/v1/works_cycles/employee_first_names", operation_id="get_employee_first_names", summary="Retrieves the first names of employees based on the specified name style and gender. The name style and gender are used to filter the results, ensuring that only relevant employee first names are returned.")
async def get_employee_first_names(name_style: int = Query(..., description="Name style of the person"), gender: str = Query(..., description="Gender of the employee")):
    cursor.execute("SELECT T2.FirstName FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.NameStyle = ? AND T1.Gender = ?", (name_style, gender))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the count of current employees based on title
@app.get("/v1/works_cycles/current_employee_count", operation_id="get_current_employee_count", summary="Retrieves the total number of current employees with a specific job title. The operation filters employees based on their current status and job title, then returns the count.")
async def get_current_employee_count(current_flag: int = Query(..., description="Current flag of the employee"), title: str = Query(..., description="Title of the person")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.CurrentFlag = ? AND T2.Title = ?", (current_flag, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the demographics of the highest paid employee based on marital status
@app.get("/v1/works_cycles/highest_paid_employee_demographics", operation_id="get_highest_paid_employee_demographics", summary="Retrieves the demographic information of the highest paid employee with the specified marital status. The operation filters employees based on their marital status and identifies the one with the highest pay rate.")
async def get_highest_paid_employee_demographics(marital_status: str = Query(..., description="Marital status of the employee")):
    cursor.execute("SELECT T2.Demographics FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN EmployeePayHistory AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T1.MaritalStatus = ? ORDER BY T3.Rate DESC LIMIT 1", (marital_status,))
    result = cursor.fetchone()
    if not result:
        return {"demographics": []}
    return {"demographics": result[0]}

# Endpoint to get the suffix of the employee with the most sick leave hours based on person type
@app.get("/v1/works_cycles/employee_suffix_most_sick_leave", operation_id="get_employee_suffix_most_sick_leave", summary="Retrieves the suffix of the employee with the highest number of sick leave hours for a given person type. The person type is used to filter the results.")
async def get_employee_suffix_most_sick_leave(person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT T2.Suffix FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.PersonType = ? ORDER BY T1.SickLeaveHours DESC LIMIT 1", (person_type,))
    result = cursor.fetchone()
    if not result:
        return {"suffix": []}
    return {"suffix": result[0]}

# Endpoint to get the count of employees based on marital status, name style, and highest pay rate
@app.get("/v1/works_cycles/employee_count_marital_name_style_highest_rate", operation_id="get_employee_count_marital_name_style_highest_rate", summary="Retrieves the total number of employees who share a specific marital status, name style, and the highest pay rate. The marital status and name style are provided as input parameters.")
async def get_employee_count_marital_name_style_highest_rate(marital_status: str = Query(..., description="Marital status of the employee"), name_style: int = Query(..., description="Name style of the person")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN EmployeePayHistory AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T1.MaritalStatus = ? AND T2.NameStyle = ? AND T3.Rate = ( SELECT Rate FROM EmployeePayHistory ORDER BY Rate DESC LIMIT 1 )", (marital_status, name_style))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of current employees based on email promotion
@app.get("/v1/works_cycles/current_employee_count_email_promotion", operation_id="get_current_employee_count_email_promotion", summary="Retrieves the total number of current employees based on their email promotion status. The operation filters employees by their current employment status and email promotion flag, providing a count of those who meet the specified criteria.")
async def get_current_employee_count_email_promotion(current_flag: int = Query(..., description="Current flag of the employee"), email_promotion: int = Query(..., description="Email promotion flag of the person")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.CurrentFlag = ? AND T2.EmailPromotion = ?", (current_flag, email_promotion))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the credit card IDs of persons based on person type
@app.get("/v1/works_cycles/person_credit_card_ids", operation_id="get_person_credit_card_ids", summary="Retrieves the unique identifiers of credit cards associated with individuals of a specified type. The operation filters individuals based on their type and returns the corresponding credit card IDs.")
async def get_person_credit_card_ids(person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT T2.CreditCardID FROM Person AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.PersonType = ?", (person_type,))
    result = cursor.fetchall()
    if not result:
        return {"credit_card_ids": []}
    return {"credit_card_ids": [row[0] for row in result]}

# Endpoint to get the average vacation hours of employees based on gender and person type
@app.get("/v1/works_cycles/average_vacation_hours", operation_id="get_average_vacation_hours", summary="Retrieves the average number of vacation hours taken by employees, filtered by gender and person type. The calculation is based on the total vacation hours and the count of employees that match the specified gender and person type.")
async def get_average_vacation_hours(gender: str = Query(..., description="Gender of the employee"), person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT CAST(SUM(T1.VacationHours) AS REAL) / COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.Gender = ? AND T2.PersonType = ?", (gender, person_type))
    result = cursor.fetchone()
    if not result:
        return {"average_vacation_hours": []}
    return {"average_vacation_hours": result[0]}

# Endpoint to get the difference between the maximum and average pay rate based on email promotion and marital status
@app.get("/v1/works_cycles/pay_rate_difference", operation_id="get_pay_rate_difference", summary="Retrieves the difference between the highest and average pay rate for employees who share a specific email promotion flag and marital status.")
async def get_pay_rate_difference(email_promotion: int = Query(..., description="Email promotion flag of the person"), marital_status: str = Query(..., description="Marital status of the employee")):
    cursor.execute("SELECT MAX(T1.Rate) - SUM(T1.Rate) / COUNT(T1.BusinessEntityID) FROM EmployeePayHistory AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Employee AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T2.EmailPromotion = ? AND T3.MaritalStatus = ?", (email_promotion, marital_status))
    result = cursor.fetchone()
    if not result:
        return {"pay_rate_difference": []}
    return {"pay_rate_difference": result[0]}

# Endpoint to get the proportion of persons with a specific person type, name style, and marital status
@app.get("/v1/works_cycles/proportion_person_type", operation_id="get_proportion_person_type", summary="Retrieves the proportion of individuals with a specified person type, name style, and marital status. This operation calculates the ratio of individuals with the given attributes to the total number of individuals in the database.")
async def get_proportion_person_type(person_type: str = Query(..., description="Person type"), name_style: int = Query(..., description="Name style of the person"), marital_status: str = Query(..., description="Marital status of the employee")):
    cursor.execute("SELECT CAST(COUNT(IIF(T1.PersonType = ?, T1.PersonType, NULL)) AS REAL) / COUNT(T1.PersonType) FROM Person AS T1 INNER JOIN Employee AS T2 WHERE T1.PersonType = ? AND T1.NameStyle = ? AND T2.MaritalStatus = ?", (person_type, person_type, name_style, marital_status))
    result = cursor.fetchone()
    if not result:
        return {"proportion": []}
    return {"proportion": result[0]}

# Endpoint to get the percentage of employees with vacation hours above a certain threshold
@app.get("/v1/works_cycles/employee_vacation_hours_percentage", operation_id="get_employee_vacation_hours_percentage", summary="Retrieves the percentage of current employees who have vacation hours exceeding a specified threshold and sick leave hours above a certain threshold. This operation calculates the ratio of employees meeting these criteria to the total number of current employees.")
async def get_employee_vacation_hours_percentage(vacation_hours: int = Query(..., description="Vacation hours threshold"), current_flag: int = Query(..., description="Current flag status"), sick_leave_hours: int = Query(..., description="Sick leave hours threshold")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.VacationHours > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.BusinessEntityID) FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.CurrentFlag = ? AND T2.SickLeaveHours > ?", (vacation_hours, current_flag, sick_leave_hours))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average last receipt cost for products with a specific average lead time
@app.get("/v1/works_cycles/average_last_receipt_cost", operation_id="get_average_last_receipt_cost", summary="Retrieves the average cost of the most recent receipt for products that have a specified average lead time. The calculation is based on the sum of the last receipt costs divided by the count of unique product IDs that meet the specified average lead time.")
async def get_average_last_receipt_cost(average_lead_time: int = Query(..., description="Average lead time")):
    cursor.execute("SELECT SUM(LastReceiptCost) / COUNT(ProductID) FROM ProductVendor WHERE AverageLeadTime = ?", (average_lead_time,))
    result = cursor.fetchone()
    if not result:
        return {"average_cost": []}
    return {"average_cost": result[0]}

# Endpoint to get the average actual cost for transactions of a specific type within a date range
@app.get("/v1/works_cycles/average_actual_cost", operation_id="get_average_actual_cost", summary="Retrieves the average actual cost of transactions of a specified type that occurred within a given date range. The calculation is based on the sum of actual costs divided by the total number of transactions. The transaction type and date range are provided as input parameters.")
async def get_average_actual_cost(transaction_type: str = Query(..., description="Transaction type"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(ActualCost) AS REAL) / COUNT(TransactionID) FROM TransactionHistoryArchive WHERE TransactionType = ? AND TransactionDate >= ? AND TransactionDate < ?", (transaction_type, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_cost": []}
    return {"average_cost": result[0]}

# Endpoint to get the percentage of employees with a specific marital status hired in a specific year and gender
@app.get("/v1/works_cycles/employee_marital_status_percentage", operation_id="get_employee_marital_status_percentage", summary="Retrieves the percentage of employees with a specified marital status, hired in a specific year, and of a certain gender. The calculation is based on the total number of employees who meet the specified criteria.")
async def get_employee_marital_status_percentage(marital_status: str = Query(..., description="Marital status"), hire_year: str = Query(..., description="Hire year in 'YYYY' format"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN MaritalStatus = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(BusinessEntityID) FROM Employee WHERE SUBSTR(HireDate, 1, 4) = ? AND Gender = ?", (marital_status, hire_year, gender))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of people with a specific email promotion type and person type given a first name
@app.get("/v1/works_cycles/email_promotion_percentage", operation_id="get_email_promotion_percentage", summary="Retrieves the percentage of individuals with a specific email promotion type and person type, filtered by a given first name. The operation calculates the proportion of individuals with the specified email promotion type out of all individuals with the specified person type who share the same first name.")
async def get_email_promotion_percentage(email_promotion: int = Query(..., description="Email promotion type"), person_type: str = Query(..., description="Person type"), first_name: str = Query(..., description="First name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN EmailPromotion = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN PersonType = ? THEN 1 ELSE 0 END) FROM Person WHERE FirstName = ?", (email_promotion, person_type, first_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct product IDs where the standard price minus the last receipt cost is less than a specified value
@app.get("/v1/works_cycles/distinct_product_ids", operation_id="get_distinct_product_ids", summary="Retrieves a unique set of product identifiers for items where the difference between the standard price and the last receipt cost is less than the specified price difference threshold.")
async def get_distinct_product_ids(price_difference: float = Query(..., description="Price difference threshold")):
    cursor.execute("SELECT DISTINCT ProductID FROM ProductVendor WHERE StandardPrice - LastReceiptCost < ?", (price_difference,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get the average total due for purchase orders with a specific status
@app.get("/v1/works_cycles/average_total_due", operation_id="get_average_total_due", summary="Retrieves the average total due for purchase orders with a specified status. The operation calculates the sum of all total due amounts and divides it by the count of total due amounts for purchase orders that match the provided status.")
async def get_average_total_due(status: int = Query(..., description="Status of the purchase order")):
    cursor.execute("SELECT SUM(TotalDue) / COUNT(TotalDue) FROM PurchaseOrderHeader WHERE Status = ?", (status,))
    result = cursor.fetchone()
    if not result:
        return {"average_due": []}
    return {"average_due": result[0]}

# Endpoint to get the percentage of sales orders with order quantity below a threshold and a specific unit price discount
@app.get("/v1/works_cycles/sales_order_percentage", operation_id="get_sales_order_percentage", summary="Retrieves the proportion of sales orders that have an order quantity less than a specified threshold and a particular unit price discount. The calculation is based on the total number of sales orders.")
async def get_sales_order_percentage(order_qty: int = Query(..., description="Order quantity threshold"), unit_price_discount: float = Query(..., description="Unit price discount")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN OrderQty < ? AND UnitPriceDiscount = ? THEN 1 ELSE 0 END) AS REAL) / COUNT(SalesOrderID) FROM SalesOrderDetail", (order_qty, unit_price_discount))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get business entity IDs of salespersons with sales YTD and bonus above specified thresholds
@app.get("/v1/works_cycles/salesperson_business_entity_ids", operation_id="get_salesperson_business_entity_ids", summary="Get business entity IDs of salespersons with sales YTD and bonus above specified thresholds")
async def get_salesperson_business_entity_ids(sales_ytd: float = Query(..., description="Sales YTD threshold"), bonus: float = Query(..., description="Bonus threshold")):
    cursor.execute("SELECT BusinessEntityID FROM SalesPerson WHERE SalesYTD > ? AND Bonus > ?", (sales_ytd, bonus))
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": [row[0] for row in result]}

# Endpoint to get the count of business entity addresses by address type names
@app.get("/v1/works_cycles/address_type_counts", operation_id="get_address_type_counts", summary="Retrieves the total count of business entity addresses categorized by two specified address types. The response provides a breakdown of the count for each address type, allowing users to compare their prevalence.")
async def get_address_type_counts(address_type_1: str = Query(..., description="First address type name"), address_type_2: str = Query(..., description="Second address type name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Name = ? THEN 1 ELSE 0 END), SUM(CASE WHEN T2.Name = ? THEN 1 ELSE 0 END) FROM BusinessEntityAddress AS T1 INNER JOIN AddressType AS T2 ON T1.AddressTypeID = T2.AddressTypeID", (address_type_1, address_type_2))
    result = cursor.fetchone()
    if not result:
        return {"counts": []}
    return {"counts": {"address_type_1": result[0], "address_type_2": result[1]}}

# Endpoint to get customer IDs based on unit price and order quantity
@app.get("/v1/works_cycles/customer_ids_by_unit_price_order_qty", operation_id="get_customer_ids", summary="Retrieves the unique identifiers of customers who have placed orders with a specific unit price and quantity. The operation filters the sales order details based on the provided unit price and order quantity, and returns the corresponding customer IDs.")
async def get_customer_ids(unit_price: float = Query(..., description="Unit price of the order"), order_qty: int = Query(..., description="Order quantity")):
    cursor.execute("SELECT T2.CustomerID FROM SalesOrderDetail AS T1 INNER JOIN Customer AS T2 WHERE T1.UnitPrice = ? AND T1.OrderQty = ?", (unit_price, order_qty))
    result = cursor.fetchall()
    if not result:
        return {"customer_ids": []}
    return {"customer_ids": [row[0] for row in result]}

# Endpoint to get business entity IDs based on credit card details
@app.get("/v1/works_cycles/business_entity_ids_by_credit_card_details", operation_id="get_business_entity_ids", summary="Retrieves the business entity IDs associated with credit cards of a specific type, expiration month, and year. This operation filters the credit card details and returns the corresponding business entity identifiers.")
async def get_business_entity_ids(card_type: str = Query(..., description="Credit card type"), exp_month: int = Query(..., description="Expiration month"), exp_year: int = Query(..., description="Expiration year")):
    cursor.execute("SELECT T2.BusinessEntityID FROM CreditCard AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.CreditCardID = T2.CreditCardID WHERE T1.CardType = ? AND T1.ExpMonth = ? AND T1.ExpYear = ?", (card_type, exp_month, exp_year))
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": [row[0] for row in result]}

# Endpoint to get credit ratings based on product vendor details
@app.get("/v1/works_cycles/credit_ratings_by_product_vendor_details", operation_id="get_credit_ratings", summary="Retrieves credit ratings for vendors based on the provided standard price, average lead time, and last receipt date. The operation filters the ProductVendor table using the input parameters and returns the corresponding credit ratings from the Vendor table.")
async def get_credit_ratings(standard_price: float = Query(..., description="Standard price"), average_lead_time: int = Query(..., description="Average lead time"), last_receipt_date: str = Query(..., description="Last receipt date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.CreditRating FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.StandardPrice = ? AND T1.AverageLeadTime = ? AND STRFTIME('%Y-%m-%d', T1.LastReceiptDate) = ?", (standard_price, average_lead_time, last_receipt_date))
    result = cursor.fetchall()
    if not result:
        return {"credit_ratings": []}
    return {"credit_ratings": [row[0] for row in result]}

# Endpoint to get the count of products based on name patterns
@app.get("/v1/works_cycles/product_count_by_name_patterns", operation_id="get_product_count", summary="Retrieves the total count of products that match either of the provided name patterns. The operation uses the specified patterns to filter the product names and returns the count of matching products.")
async def get_product_count(name_pattern1: str = Query(..., description="First name pattern"), name_pattern2: str = Query(..., description="Second name pattern")):
    cursor.execute("SELECT COUNT(ProductID) FROM Product WHERE Name LIKE ? OR Name LIKE ?", (name_pattern1, name_pattern2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest job title based on department ID
@app.get("/v1/works_cycles/latest_job_title_by_department_id", operation_id="get_latest_job_title", summary="Retrieves the most recent job title associated with the specified department. The operation uses the provided department ID to search for the corresponding job title in the employee and department history records. The result is determined by the latest start date of the job title within the department.")
async def get_latest_job_title(department_id: int = Query(..., description="Department ID")):
    cursor.execute("SELECT T1.JobTitle FROM Employee AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.DepartmentID = ? ORDER BY T2.StartDate DESC LIMIT 1", (department_id,))
    result = cursor.fetchone()
    if not result:
        return {"job_title": []}
    return {"job_title": result[0]}

# Endpoint to get the count of locations based on location name
@app.get("/v1/works_cycles/location_count_by_name", operation_id="get_location_count", summary="Retrieves the total number of locations that match the specified location name. This operation considers the inventory of products associated with each location to determine the count.")
async def get_location_count(location_name: str = Query(..., description="Location name")):
    cursor.execute("SELECT COUNT(T1.LocationID) FROM Location AS T1 INNER JOIN ProductInventory AS T2 USING (LocationID) WHERE T1.Name = ?", (location_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total scrapped quantity based on scrap reason name
@app.get("/v1/works_cycles/total_scrapped_qty_by_scrap_reason", operation_id="get_total_scrapped_qty", summary="Retrieves the total quantity of items that have been scrapped due to a specific reason. The operation calculates the sum of scrapped quantities associated with the provided scrap reason name.")
async def get_total_scrapped_qty(scrap_reason_name: str = Query(..., description="Scrap reason name")):
    cursor.execute("SELECT SUM(T2.ScrappedQty) FROM ScrapReason AS T1 INNER JOIN WorkOrder AS T2 ON T1.ScrapReasonID = T2.ScrapReasonID WHERE T1.Name = ?", (scrap_reason_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_scrapped_qty": []}
    return {"total_scrapped_qty": result[0]}

# Endpoint to get the total order quantity based on purchase order status
@app.get("/v1/works_cycles/total_order_qty_by_status", operation_id="get_total_order_qty", summary="Retrieves the total quantity of orders associated with a specific purchase order status. The status is used to filter the orders and calculate the sum of their quantities.")
async def get_total_order_qty(status: int = Query(..., description="Purchase order status")):
    cursor.execute("SELECT SUM(T2.OrderQty) FROM PurchaseOrderHeader AS T1 INNER JOIN PurchaseOrderDetail AS T2 ON T1.PurchaseOrderID = T2.PurchaseOrderID WHERE T1.Status = ?", (status,))
    result = cursor.fetchone()
    if not result:
        return {"total_order_qty": []}
    return {"total_order_qty": result[0]}

# Endpoint to get the count of distinct products based on order quantity and unit price discount
@app.get("/v1/works_cycles/distinct_product_count_by_order_qty_discount", operation_id="get_distinct_product_count", summary="Retrieves the count of unique products that meet the specified order quantity and unit price discount criteria. This operation considers the order quantity and unit price discount of each product in the sales order details, as well as the associated special offer and unit price discount information.")
async def get_distinct_product_count(order_qty: int = Query(..., description="Order quantity"), unit_price_discount: float = Query(..., description="Unit price discount")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ProductID) FROM SalesOrderDetail AS T1 INNER JOIN SpecialOfferProduct AS T2 ON T1.SpecialOfferID = T2.SpecialOfferID INNER JOIN SpecialOffer AS T3 ON T2.SpecialOfferID = T3.SpecialOfferID WHERE T1.OrderQty > ? AND T1.UnitPriceDiscount = ?", (order_qty, unit_price_discount))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct transaction types based on product size, color, and safety stock level
@app.get("/v1/works_cycles/distinct_transaction_types", operation_id="get_distinct_transaction_types", summary="Retrieve a unique set of transaction types associated with products that match the specified size, color, and safety stock level. This operation filters products based on the provided parameters and returns the distinct transaction types linked to them.")
async def get_distinct_transaction_types(size: int = Query(..., description="Size of the product"), color: str = Query(..., description="Color of the product"), safety_stock_level: int = Query(..., description="Safety stock level of the product")):
    cursor.execute("SELECT DISTINCT T2.TransactionType FROM Product AS T1 INNER JOIN TransactionHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Size = ? AND T1.Color = ? AND T1.SafetyStockLevel = ?", (size, color, safety_stock_level))
    result = cursor.fetchall()
    if not result:
        return {"transaction_types": []}
    return {"transaction_types": [row[0] for row in result]}

# Endpoint to get product subcategory names based on product color
@app.get("/v1/works_cycles/product_subcategory_names", operation_id="get_product_subcategory_names", summary="Retrieves a list of distinct product subcategory names associated with the specified product color. The operation filters products based on the provided color and returns the names of their respective subcategories.")
async def get_product_subcategory_names(color: str = Query(..., description="Color of the product")):
    cursor.execute("SELECT T1.Name FROM ProductSubcategory AS T1 INNER JOIN Product AS T2 USING (ProductSubcategoryID) WHERE T2.Color = ? GROUP BY T1.Name", (color,))
    result = cursor.fetchall()
    if not result:
        return {"subcategory_names": []}
    return {"subcategory_names": [row[0] for row in result]}

# Endpoint to get the end date of the heaviest product based on weight unit measure code
@app.get("/v1/works_cycles/heaviest_product_end_date", operation_id="get_heaviest_product_end_date", summary="Retrieves the end date of the product with the highest weight, as determined by the provided weight unit measure code. The weight unit measure code is used to filter the products and identify the heaviest one based on its weight. The end date of the heaviest product is then returned.")
async def get_heaviest_product_end_date(weight_unit_measure_code: str = Query(..., description="Weight unit measure code of the product")):
    cursor.execute("SELECT T2.EndDate FROM Product AS T1 INNER JOIN ProductCostHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.WeightUnitMeasureCode = ? ORDER BY T1.Weight DESC LIMIT 1", (weight_unit_measure_code,))
    result = cursor.fetchone()
    if not result:
        return {"end_date": []}
    return {"end_date": result[0]}

# Endpoint to get the average order quantity for a specific ship method ID
@app.get("/v1/works_cycles/average_order_quantity", operation_id="get_average_order_quantity", summary="Retrieves the average quantity of products ordered for a specific shipping method. This operation calculates the total quantity of products ordered using the provided shipping method and divides it by the total number of unique products ordered. The result is a single value representing the average order quantity for the specified shipping method.")
async def get_average_order_quantity(ship_method_id: int = Query(..., description="Ship method ID")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.ShipMethodID = ?, T3.OrderQty, 0)) AS REAL) / COUNT(T3.ProductID) FROM ShipMethod AS T1 INNER JOIN PurchaseOrderHeader AS T2 ON T1.ShipMethodID = T2.ShipMethodID INNER JOIN PurchaseOrderDetail AS T3 ON T2.PurchaseOrderID = T3.PurchaseOrderID", (ship_method_id,))
    result = cursor.fetchone()
    if not result:
        return {"average_order_quantity": []}
    return {"average_order_quantity": result[0]}

# Endpoint to get the state province name with the highest sales growth
@app.get("/v1/works_cycles/highest_sales_growth_state", operation_id="get_highest_sales_growth_state", summary="Retrieves the name of the state or province with the highest sales growth. This operation calculates the sales growth by comparing the current year's sales to the previous year's sales. The data is obtained by joining the SalesTerritory, StateProvince, and SalesTaxRate tables. The result is ordered in descending order based on the sales growth percentage, and the top state or province is returned.")
async def get_highest_sales_growth_state():
    cursor.execute("SELECT T2.Name FROM SalesTerritory AS T1 INNER JOIN StateProvince AS T2 ON T1.CountryRegionCode = T2.CountryRegionCode INNER JOIN SalesTaxRate AS T3 ON T2.StateProvinceID = T3.StateProvinceID ORDER BY (T1.SalesYTD - T1.SalesLastYear) / T1.SalesLastYear DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"state_name": []}
    return {"state_name": result[0]}

# Endpoint to get the count of employees with a pay rate above a certain threshold
@app.get("/v1/works_cycles/employee_count_above_pay_rate", operation_id="get_employee_count_above_pay_rate", summary="Retrieves the total number of employees who have a pay rate that exceeds the specified threshold. The threshold is defined by the input parameter, which represents the minimum pay rate required for an employee to be included in the count.")
async def get_employee_count_above_pay_rate(pay_rate_threshold: float = Query(..., description="Pay rate threshold")):
    cursor.execute("SELECT COUNT(BusinessEntityID) FROM EmployeePayHistory WHERE rate * PayFrequency > ?", (pay_rate_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of a person based on business entity ID and person type
@app.get("/v1/works_cycles/person_name", operation_id="get_person_name", summary="Retrieves the full name of a person associated with a specific business entity and person type. The operation requires the business entity ID and the person type as input parameters to accurately identify the person.")
async def get_person_name(business_entity_id: int = Query(..., description="Business entity ID"), person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT FirstName, MiddleName, LastName FROM Person WHERE BusinessEntityID = ? AND PersonType = ?", (business_entity_id, person_type))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": {"first_name": result[0], "middle_name": result[1], "last_name": result[2]}}

# Endpoint to get the vendor name based on business entity ID
@app.get("/v1/works_cycles/vendor_name", operation_id="get_vendor_name", summary="Retrieves the name of the vendor associated with the provided business entity ID.")
async def get_vendor_name(business_entity_id: int = Query(..., description="Business entity ID")):
    cursor.execute("SELECT NAME FROM Vendor WHERE BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchone()
    if not result:
        return {"vendor_name": []}
    return {"vendor_name": result[0]}

# Endpoint to get the count of product vendors based on minimum and maximum order quantities
@app.get("/v1/works_cycles/product_vendor_count", operation_id="get_product_vendor_count", summary="Retrieves the total number of product vendors that have a minimum order quantity greater than the provided value and a maximum order quantity less than the provided value.")
async def get_product_vendor_count(min_order_qty: int = Query(..., description="Minimum order quantity"), max_order_qty: int = Query(..., description="Maximum order quantity")):
    cursor.execute("SELECT COUNT(*) FROM ProductVendor WHERE MinOrderQty > ? AND MaxOrderQty < ?", (min_order_qty, max_order_qty))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the department name based on group name
@app.get("/v1/works_cycles/department_name", operation_id="get_department_name", summary="Get the department name based on group name")
async def get_department_name(group_name: str = Query(..., description="Group name")):
    cursor.execute("SELECT Name FROM Department WHERE GroupName = ?", (group_name,))
    result = cursor.fetchone()
    if not result:
        return {"department_name": []}
    return {"department_name": result[0]}

# Endpoint to get last names of persons based on person type and middle name prefix
@app.get("/v1/works_cycles/persons/last_names_by_type_and_middle_name", operation_id="get_last_names", summary="Retrieves the last names of individuals who match the specified person type and have a middle name that starts with the provided prefix. This operation is useful for filtering and searching for individuals based on their type and middle name.")
async def get_last_names(person_type: str = Query(..., description="Type of person"), middle_name_prefix: str = Query(..., description="Prefix of the middle name")):
    cursor.execute("SELECT LastName FROM Person WHERE PersonType = ? AND MiddleName LIKE ?", (person_type, middle_name_prefix + '%'))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the count of distinct business entities based on average lead time
@app.get("/v1/works_cycles/product_vendors/count_distinct_business_entities", operation_id="get_count_distinct_business_entities", summary="Retrieves the total number of unique business entities that have a specified average lead time. The average lead time is provided as an input parameter.")
async def get_count_distinct_business_entities(average_lead_time: int = Query(..., description="Average lead time")):
    cursor.execute("SELECT COUNT(DISTINCT BusinessEntityID) FROM ProductVendor WHERE AverageLeadTime = ?", (average_lead_time,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get product IDs ordered by standard cost in ascending order with a limit
@app.get("/v1/works_cycles/product_cost_history/product_ids_by_standard_cost", operation_id="get_product_ids_by_standard_cost", summary="Retrieves a specified number of product IDs, sorted by their standard cost in ascending order. This operation is useful for obtaining a list of products ranked from the least to the most expensive based on their standard cost.")
async def get_product_ids_by_standard_cost(limit: int = Query(..., description="Limit of product IDs to return")):
    cursor.execute("SELECT ProductID FROM ProductCostHistory ORDER BY StandardCost ASC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get the count of products based on finished goods flag and color
@app.get("/v1/works_cycles/products/count_by_finished_goods_flag_and_color", operation_id="get_count_by_finished_goods_flag_and_color", summary="Retrieves the total count of products that match the specified finished goods flag and color. The finished goods flag indicates whether the product is a finished good (1) or not (0), and the color parameter filters the products by their color.")
async def get_count_by_finished_goods_flag_and_color(finished_goods_flag: int = Query(..., description="Finished goods flag (0 or 1)"), color: str = Query(..., description="Color of the product")):
    cursor.execute("SELECT COUNT(ProductID) FROM Product WHERE FinishedGoodsFlag = ? AND Color = ?", (finished_goods_flag, color))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get job titles ordered by sick leave hours in descending order with a limit
@app.get("/v1/works_cycles/employees/job_titles_by_sick_leave_hours", operation_id="get_job_titles_by_sick_leave_hours", summary="Retrieves a specified number of job titles, ordered by the highest number of sick leave hours taken by employees. This operation provides insight into job titles with the most sick leave hours, which can be useful for workforce management and planning.")
async def get_job_titles_by_sick_leave_hours(limit: int = Query(..., description="Limit of job titles to return")):
    cursor.execute("SELECT JobTitle FROM Employee ORDER BY SickLeaveHours DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"job_titles": []}
    return {"job_titles": [row[0] for row in result]}

# Endpoint to get address lines based on address ID
@app.get("/v1/works_cycles/addresses/address_lines_by_id", operation_id="get_address_lines_by_id", summary="Retrieves the first and second lines of an address associated with the provided address ID.")
async def get_address_lines_by_id(address_id: int = Query(..., description="Address ID")):
    cursor.execute("SELECT AddressLine1, AddressLine2 FROM Address WHERE AddressID = ?", (address_id,))
    result = cursor.fetchone()
    if not result:
        return {"address_lines": []}
    return {"address_lines": {"AddressLine1": result[0], "AddressLine2": result[1]}}

# Endpoint to get the difference between last receipt cost and standard price based on business entity ID
@app.get("/v1/works_cycles/product_vendors/cost_difference_by_business_entity", operation_id="get_cost_difference_by_business_entity", summary="Retrieves the cost difference between the last receipt cost and the standard price for a specific business entity. The operation requires the business entity ID as input to determine the relevant product vendor data.")
async def get_cost_difference_by_business_entity(business_entity_id: int = Query(..., description="Business entity ID")):
    cursor.execute("SELECT LastReceiptCost - StandardPrice FROM ProductVendor WHERE BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchone()
    if not result:
        return {"cost_difference": []}
    return {"cost_difference": result[0]}

# Endpoint to get the difference between list price and standard cost based on product ID
@app.get("/v1/works_cycles/products/price_difference_by_product_id", operation_id="get_price_difference_by_product_id", summary="Retrieves the price difference between the list price and the standard cost for a specific product, identified by its unique product ID.")
async def get_price_difference_by_product_id(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT listPrice - StandardCost FROM Product WHERE ProductID = ?", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"price_difference": []}
    return {"price_difference": result[0]}

# Endpoint to get reviewer names based on rating
@app.get("/v1/works_cycles/product_reviews/reviewer_names_by_rating", operation_id="get_reviewer_names_by_rating", summary="Retrieves the names of reviewers who have given a specific rating to a product. The rating is provided as an input parameter.")
async def get_reviewer_names_by_rating(rating: int = Query(..., description="Rating of the product review")):
    cursor.execute("SELECT ReviewerName FROM ProductReview WHERE Rating = ?", (rating,))
    result = cursor.fetchall()
    if not result:
        return {"reviewer_names": []}
    return {"reviewer_names": [row[0] for row in result]}

# Endpoint to get business entity IDs based on the highest credit rating
@app.get("/v1/works_cycles/vendors/business_entity_ids_by_highest_credit_rating", operation_id="get_business_entity_ids_by_highest_credit_rating", summary="Retrieves the business entity IDs of vendors with the highest credit rating. This operation identifies the top credit rating from the vendor database and returns the corresponding business entity IDs.")
async def get_business_entity_ids_by_highest_credit_rating():
    cursor.execute("SELECT BusinessEntityID FROM Vendor WHERE CreditRating = ( SELECT CreditRating FROM Vendor ORDER BY CreditRating DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": [row[0] for row in result]}

# Endpoint to get the contact type name based on BusinessEntityID
@app.get("/v1/works_cycles/contact_type_name", operation_id="get_contact_type_name", summary="Retrieves the name of the contact type associated with the specified BusinessEntityID. This operation fetches the contact type name from the ContactType table, which is linked to the BusinessEntityContact table via the ContactTypeID. The BusinessEntityID is used to filter the results and return the corresponding contact type name.")
async def get_contact_type_name(business_entity_id: int = Query(..., description="BusinessEntityID of the contact")):
    cursor.execute("SELECT T1.Name FROM ContactType AS T1 INNER JOIN BusinessEntityContact AS T2 ON T1.ContactTypeID = T2.ContactTypeID WHERE T2.BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchall()
    if not result:
        return {"contact_type_names": []}
    return {"contact_type_names": result}

# Endpoint to get BusinessEntityIDs based on phone number type and limit
@app.get("/v1/works_cycles/business_entity_ids_by_phone_type", operation_id="get_business_entity_ids_by_phone_type", summary="Get BusinessEntityIDs based on phone number type and limit")
async def get_business_entity_ids_by_phone_type(phone_number_type: str = Query(..., description="Name of the phone number type"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.BusinessEntityID FROM PhoneNumberType AS T1 INNER JOIN PersonPhone AS T2 ON T1.PhoneNumberTypeID = T2.PhoneNumberTypeID WHERE T1.Name = ? LIMIT ?", (phone_number_type, limit))
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": result}

# Endpoint to get currency names based on country region name
@app.get("/v1/works_cycles/currency_names_by_country", operation_id="get_currency_names_by_country", summary="Retrieves the names of currencies used in the specified country region. The operation filters the currencies based on the provided country region name and returns the corresponding currency names.")
async def get_currency_names_by_country(country_name: str = Query(..., description="Name of the country region")):
    cursor.execute("SELECT T1.Name FROM Currency AS T1 INNER JOIN CountryRegionCurrency AS T2 ON T1.CurrencyCode = T2.CurrencyCode INNER JOIN CountryRegion AS T3 ON T2.CountryRegionCode = T3.CountryRegionCode WHERE T3.Name = ?", (country_name,))
    result = cursor.fetchall()
    if not result:
        return {"currency_names": []}
    return {"currency_names": result}

# Endpoint to get average lead time and vendor name based on BusinessEntityID
@app.get("/v1/works_cycles/average_lead_time_vendor_name", operation_id="get_average_lead_time_vendor_name", summary="Retrieves the average lead time and the corresponding vendor name for a given vendor, identified by its BusinessEntityID.")
async def get_average_lead_time_vendor_name(business_entity_id: int = Query(..., description="BusinessEntityID of the vendor")):
    cursor.execute("SELECT T1.AverageLeadTime, T2.Name FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 USING (businessentityid) WHERE T2.BusinessEntityID = ? GROUP BY T1.AverageLeadTime, T2.Name", (business_entity_id,))
    result = cursor.fetchall()
    if not result:
        return {"average_lead_time_vendor_name": []}
    return {"average_lead_time_vendor_name": result}

# Endpoint to get the difference in counts of addresses between two cities grouped by state province code
@app.get("/v1/works_cycles/address_count_difference_by_cities", operation_id="get_address_count_difference_by_cities", summary="Retrieve the difference in the number of addresses between two specified cities, grouped by the state or province code. This operation compares the total count of addresses in the first city with that of the second city, providing a state-wise comparison of address distribution.")
async def get_address_count_difference_by_cities(city1: str = Query(..., description="First city name"), city2: str = Query(..., description="Second city name")):
    cursor.execute("SELECT SUM(IIF(T1.city = ?, 1, 0)) - SUM(IIF(T1.city = ?, 1, 0)) , stateprovincecode FROM Address AS T1 INNER JOIN StateProvince AS T2 ON T1.stateprovinceid = T2.stateprovinceid GROUP BY stateprovincecode", (city1, city2))
    result = cursor.fetchall()
    if not result:
        return {"address_count_difference": []}
    return {"address_count_difference": result}

# Endpoint to get ProductModelIDs based on description and culture name
@app.get("/v1/works_cycles/product_model_ids_by_description_culture", operation_id="get_product_model_ids_by_description_culture", summary="Retrieves the ProductModelIDs associated with a specific product description and culture name. The operation filters the ProductModelProductDescriptionCulture table based on the provided description and culture name, and returns the corresponding ProductModelIDs.")
async def get_product_model_ids_by_description_culture(description: str = Query(..., description="Product description"), culture_name: str = Query(..., description="Culture name")):
    cursor.execute("SELECT T1.ProductModelID FROM ProductModelProductDescriptionCulture AS T1 INNER JOIN Culture AS T2 USING (cultureid) INNER JOIN ProductDescription AS T3 USING (productdescriptionid) WHERE T3.Description LIKE ? AND T2.Name = ?", (description, culture_name))
    result = cursor.fetchall()
    if not result:
        return {"product_model_ids": []}
    return {"product_model_ids": result}

# Endpoint to get BusinessEntityIDs and sales territory groups based on commission percentage
@app.get("/v1/works_cycles/sales_person_territory_by_commission", operation_id="get_sales_person_territory_by_commission", summary="Retrieves the unique identifiers of business entities and their corresponding sales territory groups for salespersons with a commission percentage equal to or greater than the provided value. This operation enables the identification of high-commission salespersons and their associated territories.")
async def get_sales_person_territory_by_commission(commission_pct: float = Query(..., description="Commission percentage")):
    cursor.execute("SELECT T1.BusinessEntityID, T2.'Group' FROM SalesPerson AS T1 INNER JOIN SalesTerritory AS T2 USING (territoryid) WHERE T1.CommissionPct >= ?", (commission_pct,))
    result = cursor.fetchall()
    if not result:
        return {"sales_person_territory": []}
    return {"sales_person_territory": result}

# Endpoint to get phone number types ordered by their count
@app.get("/v1/works_cycles/phone_number_types_ordered", operation_id="get_phone_number_types_ordered", summary="Get phone number types ordered by their count")
async def get_phone_number_types_ordered():
    cursor.execute("SELECT T2.Name FROM PersonPhone AS T1 INNER JOIN PhoneNumberType AS T2 ON T1.PhoneNumberTypeID = T2.PhoneNumberTypeID GROUP BY T2.Name ORDER BY COUNT(T2.Name) DESC")
    result = cursor.fetchall()
    if not result:
        return {"phone_number_types": []}
    return {"phone_number_types": result}

# Endpoint to get the most common contact type name
@app.get("/v1/works_cycles/most_common_contact_type", operation_id="get_most_common_contact_type", summary="Retrieves the name of the contact type that is most frequently associated with business entities. The operation returns the name of the contact type that appears most often in the business entity contact records.")
async def get_most_common_contact_type():
    cursor.execute("SELECT T1.Name FROM ContactType AS T1 INNER JOIN BusinessEntityContact AS T2 ON T1.ContactTypeID = T2.ContactTypeID GROUP BY T1.Name ORDER BY COUNT(T1.Name) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"contact_type": []}
    return {"contact_type": result[0]}

# Endpoint to get email addresses based on person type
@app.get("/v1/works_cycles/email_addresses_by_person_type", operation_id="get_email_addresses_by_person_type", summary="Retrieves the email addresses associated with a specific type of person. The person type is provided as an input parameter, which is used to filter the results. The operation returns a list of email addresses that match the specified person type.")
async def get_email_addresses_by_person_type(person_type: str = Query(..., description="Type of person (e.g., 'SP')")):
    cursor.execute("SELECT T2.EmailAddress FROM Person AS T1 INNER JOIN EmailAddress AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.PersonType = ?", (person_type,))
    result = cursor.fetchall()
    if not result:
        return {"email_addresses": []}
    return {"email_addresses": [row[0] for row in result]}

# Endpoint to get job titles based on first name
@app.get("/v1/works_cycles/job_titles_by_first_name", operation_id="get_job_titles_by_first_name", summary="Retrieves the job titles of employees who share a specific first name. The operation filters the Person and Employee tables based on the provided first name and returns the corresponding job titles.")
async def get_job_titles_by_first_name(first_name: str = Query(..., description="First name of the person")):
    cursor.execute("SELECT T2.JobTitle FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.FirstName = ?", (first_name,))
    result = cursor.fetchall()
    if not result:
        return {"job_titles": []}
    return {"job_titles": [row[0] for row in result]}

# Endpoint to get the count of employees based on person type and marital status
@app.get("/v1/works_cycles/employee_count_by_person_type_marital_status", operation_id="get_employee_count_by_person_type_marital_status", summary="Retrieves the total number of employees who match a specified person type and marital status. The response is based on a query that filters employees by their person type and marital status.")
async def get_employee_count_by_person_type_marital_status(person_type: str = Query(..., description="Type of person (e.g., 'EM')"), marital_status: str = Query(..., description="Marital status (e.g., 'S')")):
    cursor.execute("SELECT COUNT(T1.BusinessentityID) FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.PersonType = ? AND T2.MaritalStatus = ?", (person_type, marital_status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in counts between two cultures
@app.get("/v1/works_cycles/culture_count_difference", operation_id="get_culture_count_difference", summary="Retrieves the difference in the total number of records associated with two specified cultures. The operation compares the count of records for the first culture with the count for the second culture, returning the difference. The input parameters specify the names of the two cultures to be compared.")
async def get_culture_count_difference(culture1: str = Query(..., description="First culture name (e.g., 'English')"), culture2: str = Query(..., description="Second culture name (e.g., 'Arabic')")):
    cursor.execute("SELECT SUM(CASE WHEN T1.Name = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.Name = ? THEN 1 ELSE 0 END) FROM Culture AS T1 INNER JOIN ProductModelProductDescriptionCulture AS T2 ON T1.CultureID = T2.CultureID WHERE T1.Name = ? OR T1.Name = ?", (culture1, culture2, culture1, culture2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get address line based on business entity ID
@app.get("/v1/works_cycles/address_line_by_business_entity_id", operation_id="get_address_line_by_business_entity_id", summary="Retrieves the first line of the address associated with the provided business entity ID. This operation fetches the address line from the Address table, which is linked to the BusinessEntityAddress table using the AddressID. The business entity ID is used to filter the results and return the corresponding address line.")
async def get_address_line_by_business_entity_id(business_entity_id: int = Query(..., description="Business entity ID")):
    cursor.execute("SELECT T1.AddressLine1 FROM Address AS T1 INNER JOIN BusinessEntityAddress AS T2 USING (AddressID) WHERE T2.BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchone()
    if not result:
        return {"address_line": []}
    return {"address_line": result[0]}

# Endpoint to get business entity IDs based on city
@app.get("/v1/works_cycles/business_entity_ids_by_city", operation_id="get_business_entity_ids_by_city", summary="Retrieves the unique identifiers of business entities located in a specified city. The operation filters the business entities based on the provided city name and returns their corresponding identifiers.")
async def get_business_entity_ids_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.BusinessEntityID FROM Address AS T1 INNER JOIN BusinessEntityAddress AS T2 ON T1.AddressID = T2.AddressID WHERE T1.City = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": [row[0] for row in result]}

# Endpoint to get the percentage of a specific culture
@app.get("/v1/works_cycles/culture_percentage", operation_id="get_culture_percentage", summary="Retrieves the percentage of a specific culture from the total number of cultures. The culture is identified by its name. The result is calculated by summing the instances of the specified culture and dividing it by the total count of cultures.")
async def get_culture_percentage(culture_name: str = Query(..., description="Culture name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.CultureID) FROM Culture AS T1 INNER JOIN ProductModelProductDescriptionCulture AS T2 ON T1.CultureID = T2.CultureID", (culture_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of employees of a specific gender and person type
@app.get("/v1/works_cycles/employee_gender_percentage", operation_id="get_employee_gender_percentage", summary="Retrieves the percentage of employees of a specific gender and person type from the database. The operation calculates the ratio of employees with the given gender to the total number of employees of the specified person type.")
async def get_employee_gender_percentage(gender: str = Query(..., description="Gender (e.g., 'M')"), person_type: str = Query(..., description="Type of person (e.g., 'EM')")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Gender = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.BusinessentityID) FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessentityID = T2.BusinessentityID WHERE T1.PersonType = ?", (gender, person_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get state province details based on address ID
@app.get("/v1/works_cycles/state_province_details_by_address_id", operation_id="get_state_province_details_by_address_id", summary="Retrieves the city, name, and state/province flag for a specific address ID. This operation fetches the corresponding state/province details from the database using the provided address ID.")
async def get_state_province_details_by_address_id(address_id: int = Query(..., description="Address ID")):
    cursor.execute("SELECT T2.City, T1.Name, T1.IsOnlyStateProvinceFlag FROM StateProvince AS T1 INNER JOIN Address AS T2 ON T1.StateProvinceID = T2.StateProvinceID WHERE T2.AddressID = ?", (address_id,))
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"details": {"city": result[0], "name": result[1], "is_only_state_province_flag": result[2]}}

# Endpoint to get address details based on BusinessEntityID
@app.get("/v1/works_cycles/address_details_by_business_entity_id", operation_id="get_address_details", summary="Retrieves the first and second lines of an address associated with a specific business entity. The business entity is identified by the provided BusinessEntityID.")
async def get_address_details(business_entity_id: int = Query(..., description="BusinessEntityID to filter the address details")):
    cursor.execute("SELECT T1.AddressLine1, T1.AddressLine2 FROM Address AS T1 INNER JOIN BusinessEntityAddress AS T2 ON T1.AddressID = T2.AddressID WHERE T2.BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchall()
    if not result:
        return {"address_details": []}
    return {"address_details": result}

# Endpoint to get the expiration year of a credit card based on CreditCardID
@app.get("/v1/works_cycles/credit_card_exp_year", operation_id="get_credit_card_exp_year", summary="Retrieves the expiration year of a specific credit card. The operation requires the unique identifier of the credit card as input. This identifier is used to locate the corresponding credit card record and extract the expiration year.")
async def get_credit_card_exp_year(credit_card_id: int = Query(..., description="CreditCardID to filter the expiration year")):
    cursor.execute("SELECT ExpYear FROM CreditCard WHERE CreditCardID = ?", (credit_card_id,))
    result = cursor.fetchone()
    if not result:
        return {"exp_year": []}
    return {"exp_year": result[0]}

# Endpoint to get BusinessEntityID based on first and last name
@app.get("/v1/works_cycles/business_entity_id_by_name", operation_id="get_business_entity_id_by_name", summary="Retrieves the unique identifier (BusinessEntityID) associated with a person, based on their first and last names. This operation requires both names to be provided as input parameters.")
async def get_business_entity_id_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT BusinessEntityID FROM Person WHERE FirstName = ? AND LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"business_entity_id": []}
    return {"business_entity_id": result[0]}

# Endpoint to get LocationID based on location name
@app.get("/v1/works_cycles/location_id_by_name", operation_id="get_location_id_by_name", summary="Retrieves the unique identifier (LocationID) for a specified location based on its name. The operation searches for the location with the provided name and returns its corresponding identifier.")
async def get_location_id_by_name(name: str = Query(..., description="Name of the location")):
    cursor.execute("SELECT LocationID FROM Location WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"location_id": []}
    return {"location_id": result[0]}

# Endpoint to get DepartmentID based on group name
@app.get("/v1/works_cycles/department_id_by_group_name", operation_id="get_department_id_by_group_name", summary="Get the DepartmentID for a department given its group name")
async def get_department_id_by_group_name(group_name: str = Query(..., description="Group name of the department")):
    cursor.execute("SELECT DepartmentID FROM Department WHERE GroupName = ?", (group_name,))
    result = cursor.fetchone()
    if not result:
        return {"department_id": []}
    return {"department_id": result[0]}

# Endpoint to get BusinessEntityID based on last year's sales
@app.get("/v1/works_cycles/business_entity_id_by_sales_last_year", operation_id="get_business_entity_id_by_sales_last_year", summary="Retrieves the unique identifier of a salesperson based on their sales performance from the previous year. The operation requires the sales amount from the last year as input to accurately identify the salesperson.")
async def get_business_entity_id_by_sales_last_year(sales_last_year: float = Query(..., description="Last year's sales amount")):
    cursor.execute("SELECT BusinessEntityID FROM SalesPerson WHERE SalesLastYear = ?", (sales_last_year,))
    result = cursor.fetchone()
    if not result:
        return {"business_entity_id": []}
    return {"business_entity_id": result[0]}

# Endpoint to get the start time of a shift based on ShiftID
@app.get("/v1/works_cycles/shift_start_time", operation_id="get_shift_start_time", summary="Retrieves the start time of a specific shift, identified by its unique ShiftID. This operation allows you to obtain the exact time a shift began, providing valuable information for scheduling, tracking, and analysis purposes.")
async def get_shift_start_time(shift_id: int = Query(..., description="ShiftID to filter the start time")):
    cursor.execute("SELECT StartTime FROM Shift WHERE ShiftID = ?", (shift_id,))
    result = cursor.fetchone()
    if not result:
        return {"start_time": []}
    return {"start_time": result[0]}

# Endpoint to get the ship base of a ship method based on its name
@app.get("/v1/works_cycles/ship_base_by_name", operation_id="get_ship_base_by_name", summary="Retrieves the associated ship base for a specific ship method, identified by its unique name. This operation returns the ship base value from the ShipMethod table, providing essential information about the ship method's base location or origin.")
async def get_ship_base_by_name(name: str = Query(..., description="Name of the ship method")):
    cursor.execute("SELECT ShipBase FROM ShipMethod WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"ship_base": []}
    return {"ship_base": result[0]}

# Endpoint to get the name of a culture based on CultureID
@app.get("/v1/works_cycles/culture_name_by_id", operation_id="get_culture_name_by_id", summary="Retrieves the name of a specific culture based on the provided CultureID.")
async def get_culture_name_by_id(culture_id: str = Query(..., description="CultureID to filter the name")):
    cursor.execute("SELECT Name FROM Culture WHERE CultureID = ?", (culture_id,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the currency code based on the currency name
@app.get("/v1/works_cycles/currency_code_by_name", operation_id="get_currency_code", summary="Retrieves the unique currency code associated with a specified currency name. The operation searches for the currency name in the database and returns the corresponding currency code.")
async def get_currency_code(name: str = Query(..., description="Name of the currency")):
    cursor.execute("SELECT CurrencyCode FROM Currency WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"currency_code": []}
    return {"currency_code": result[0]}

# Endpoint to get the phone number type ID based on the phone number type name
@app.get("/v1/works_cycles/phone_number_type_id_by_name", operation_id="get_phone_number_type_id", summary="Retrieves the unique identifier for a specific phone number type, based on the provided name. This operation allows you to look up the ID associated with a given phone number type name.")
async def get_phone_number_type_id(name: str = Query(..., description="Name of the phone number type")):
    cursor.execute("SELECT PhoneNumberTypeID FROM PhoneNumberType WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"phone_number_type_id": []}
    return {"phone_number_type_id": result[0]}

# Endpoint to get the birth date of an employee based on the hire date
@app.get("/v1/works_cycles/employee_birth_date_by_hire_date", operation_id="get_employee_birth_date", summary="Get the birth date of an employee for a given hire date")
async def get_employee_birth_date(hire_date: str = Query(..., description="Hire date of the employee in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT BirthDate FROM Employee WHERE HireDate = ?", (hire_date,))
    result = cursor.fetchone()
    if not result:
        return {"birth_date": []}
    return {"birth_date": result[0]}

# Endpoint to get the product name based on the product ID
@app.get("/v1/works_cycles/product_name_by_id", operation_id="get_product_name", summary="Retrieves the name of a specific product using its unique identifier. The product name is obtained by querying the product and product model tables, ensuring that the product ID matches the provided input parameter.")
async def get_product_name(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN ProductModel AS T2 ON T1.ProductModelID = T2.ProductModelID WHERE T1.ProductID = ?", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the unit measure codes for a product based on the product ID
@app.get("/v1/works_cycles/unit_measure_codes_by_product_id", operation_id="get_unit_measure_codes", summary="Retrieves the distinct unit measure codes associated with a specific product, based on its ID. The unit measure codes can be derived from either the size or weight of the product. This operation is useful for obtaining the various units of measurement applicable to a product.")
async def get_unit_measure_codes(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T2.UnitMeasureCode FROM Product AS T1 INNER JOIN UnitMeasure AS T2 ON T1.SizeUnitMeasureCode = T2.UnitMeasureCode OR T1.WeightUnitMeasureCode = T2.UnitMeasureCode WHERE T1.ProductID = ? GROUP BY T1.ProductID, T2.UnitMeasureCode", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"unit_measure_codes": []}
    return {"unit_measure_codes": [row[0] for row in result]}

# Endpoint to get the address lines for a business entity based on the business entity ID
@app.get("/v1/works_cycles/address_lines_by_business_entity_id", operation_id="get_address_lines", summary="Retrieves the first and second address lines associated with a specific business entity. The operation uses the provided business entity ID to look up the corresponding address lines in the database.")
async def get_address_lines(business_entity_id: int = Query(..., description="Business Entity ID")):
    cursor.execute("SELECT AddressLine1, AddressLine2 FROM Address WHERE AddressID IN ( SELECT AddressID FROM BusinessEntityAddress WHERE BusinessEntityID = ? )", (business_entity_id,))
    result = cursor.fetchall()
    if not result:
        return {"address_lines": []}
    return {"address_lines": [{"address_line1": row[0], "address_line2": row[1]} for row in result]}

# Endpoint to get the unit measure names for a bill of materials component based on the component ID
@app.get("/v1/works_cycles/unit_measure_names_by_component_id", operation_id="get_unit_measure_names", summary="Retrieves the names of the unit measures associated with a specific bill of materials component. The component is identified by its unique ID. The operation returns a list of distinct unit measure names that are currently in use for the specified component.")
async def get_unit_measure_names(component_id: int = Query(..., description="Component ID")):
    cursor.execute("SELECT T2.Name FROM BillOfMaterials AS T1 INNER JOIN UnitMeasure AS T2 ON T1.UnitMeasureCode = T2.UnitMeasureCode WHERE T1.ComponentID = ? AND T1.EndDate IS NULL GROUP BY T2.name", (component_id,))
    result = cursor.fetchall()
    if not result:
        return {"unit_measure_names": []}
    return {"unit_measure_names": [row[0] for row in result]}

# Endpoint to get the count of employees based on job title, birth date, and document summary
@app.get("/v1/works_cycles/employee_count_by_job_title_birth_date", operation_id="get_employee_count", summary="Retrieves the total number of employees with a specific job title and birth date who do not have a document summary. The job title and birth date are provided as input parameters.")
async def get_employee_count(job_title: str = Query(..., description="Job title of the employee"), birth_date: str = Query(..., description="Birth date of the employee in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.BusinessEntityID) FROM Document AS T1 INNER JOIN Employee AS T2 ON T1.Owner = T2.BusinessEntityID WHERE T2.JobTitle = ? AND T2.BirthDate = ? AND T1.DocumentSummary IS NULL", (job_title, birth_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the list price of products based on the difference between list price and standard cost, and start date
@app.get("/v1/works_cycles/product_list_price_by_cost_difference_start_date", operation_id="get_product_list_price", summary="Retrieves the list prices of products for which the difference between the list price and the standard cost surpasses a given threshold, and the start date is on or after a specified date.")
async def get_product_list_price(cost_difference: float = Query(..., description="Difference between list price and standard cost"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.listPrice FROM Product AS T1 INNER JOIN ProductlistPriceHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.listPrice - T1.StandardCost > ? AND STRFTIME('%Y-%m-%d', T2.StartDate) >= ?", (cost_difference, start_date))
    result = cursor.fetchall()
    if not result:
        return {"list_prices": []}
    return {"list_prices": [row[0] for row in result]}

# Endpoint to get the thumbnail photo of a product based on the product ID
@app.get("/v1/works_cycles/product_thumbnail_photo_by_id", operation_id="get_product_thumbnail_photo", summary="Retrieves the thumbnail photo associated with a specific product. The operation requires the product's unique identifier to locate the corresponding thumbnail photo in the database.")
async def get_product_thumbnail_photo(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T1.ThumbNailPhoto FROM ProductPhoto AS T1 INNER JOIN ProductProductPhoto AS T2 ON T1.ProductPhotoID = T2.ProductPhotoID WHERE T2.ProductID = ?", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"thumbnail_photo": []}
    return {"thumbnail_photo": result[0]}

# Endpoint to get the length of the password hash for a person with a specific first and last name
@app.get("/v1/works_cycles/password_hash_length", operation_id="get_password_hash_length", summary="Retrieves the length of the password hash associated with a person identified by their first and last name. The operation requires the first and last name of the person as input parameters to locate the corresponding password hash and calculate its length.")
async def get_password_hash_length(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT LENGTH(T2.PasswordHash) FROM Person AS T1 INNER JOIN Password AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"length": []}
    return {"length": result[0]}

# Endpoint to get the rating of a product review by a specific reviewer for a specific product
@app.get("/v1/works_cycles/product_review_rating", operation_id="get_product_review_rating", summary="Retrieves the rating of a product review submitted by a specific reviewer for a particular product. The operation requires the reviewer's name and the product's name as input parameters to accurately locate the desired review and its associated rating.")
async def get_product_review_rating(reviewer_name: str = Query(..., description="Name of the reviewer"), product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T1.Rating FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ReviewerName = ? AND T2.Name = ?", (reviewer_name, product_name))
    result = cursor.fetchone()
    if not result:
        return {"rating": []}
    return {"rating": result[0]}

# Endpoint to get the difference between the last receipt cost and the standard price for a specific product
@app.get("/v1/works_cycles/product_cost_difference", operation_id="get_product_cost_difference", summary="Retrieves the cost difference between the most recent receipt and the standard price for a specified product. The product is identified by its name.")
async def get_product_cost_difference(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T1.LastReceiptCost - T1.StandardPrice FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the bonus of a salesperson with a specific first and last name
@app.get("/v1/works_cycles/salesperson_bonus", operation_id="get_salesperson_bonus", summary="Retrieves the bonus of a salesperson identified by their first and last names. The operation uses the provided names to locate the salesperson and return their associated bonus.")
async def get_salesperson_bonus(first_name: str = Query(..., description="First name of the salesperson"), last_name: str = Query(..., description="Last name of the salesperson")):
    cursor.execute("SELECT T1.Bonus FROM SalesPerson AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.FirstName = ? AND T2.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"bonus": []}
    return {"bonus": result[0]}

# Endpoint to get the count of distinct sales tax rates for a specific state or province
@app.get("/v1/works_cycles/sales_tax_rate_count", operation_id="get_sales_tax_rate_count", summary="Retrieves the number of unique sales tax rates applicable in a specified state or province.")
async def get_sales_tax_rate_count(state_province_name: str = Query(..., description="Name of the state or province")):
    cursor.execute("SELECT COUNT(DISTINCT T1.Name) FROM SalesTaxRate AS T1 INNER JOIN StateProvince AS T2 ON T1.StateProvinceID = T2.StateProvinceID WHERE T2.Name = ?", (state_province_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the email address of a person with a specific first, middle, and last name
@app.get("/v1/works_cycles/person_email_address", operation_id="get_person_email_address", summary="Retrieves the email address associated with a person identified by their first, middle, and last names. The endpoint uses the provided names to locate the person in the database and returns the corresponding email address.")
async def get_person_email_address(first_name: str = Query(..., description="First name of the person"), middle_name: str = Query(..., description="Middle name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T2.EmailAddress FROM Person AS T1 INNER JOIN EmailAddress AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.FirstName = ? AND T1.MiddleName = ? AND T1.LastName = ?", (first_name, middle_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"email_address": []}
    return {"email_address": result[0]}

# Endpoint to get the country region code for a specific country and state/province flag
@app.get("/v1/works_cycles/country_region_code", operation_id="get_country_region_code", summary="Retrieves the country region code for a specific country and state/province flag. The operation requires the name of the country and a flag indicating if it is the only state/province. The result is a single country region code that corresponds to the provided country and state/province flag.")
async def get_country_region_code(country_name: str = Query(..., description="Name of the country"), is_only_state_province_flag: int = Query(..., description="Flag indicating if it is the only state/province (1 for true, 0 for false)")):
    cursor.execute("SELECT T1.CountryRegionCode FROM StateProvince AS T1 INNER JOIN CountryRegion AS T2 ON T1.CountryRegionCode = T2.CountryRegionCode WHERE T2.Name = ? AND T1.IsOnlyStateProvinceFlag = ?", (country_name, is_only_state_province_flag))
    result = cursor.fetchone()
    if not result:
        return {"country_region_code": []}
    return {"country_region_code": result[0]}

# Endpoint to get the transaction type for a specific product on a specific date
@app.get("/v1/works_cycles/transaction_type", operation_id="get_transaction_type", summary="Get the transaction type for a specific product on a specific date")
async def get_transaction_type(product_name: str = Query(..., description="Name of the product"), transaction_date: str = Query(..., description="Transaction date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.TransactionType FROM TransactionHistory AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ? AND STRFTIME('%Y-%m-%d',T1.TransactionDate) = ?", (product_name, transaction_date))
    result = cursor.fetchone()
    if not result:
        return {"transaction_type": []}
    return {"transaction_type": result[0]}

# Endpoint to get the transaction type from the archive for a specific product on a specific date
@app.get("/v1/works_cycles/transaction_type_archive", operation_id="get_transaction_type_archive", summary="Retrieves the transaction type associated with a specific product on a given date from the archived transaction history. The operation requires the product's name and the transaction date in 'YYYY-MM-DD' format to accurately locate and return the corresponding transaction type.")
async def get_transaction_type_archive(product_name: str = Query(..., description="Name of the product"), transaction_date: str = Query(..., description="Transaction date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.TransactionType FROM TransactionHistoryArchive AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ? AND STRFTIME('%Y-%m-%d',T1.TransactionDate) = ?", (product_name, transaction_date))
    result = cursor.fetchone()
    if not result:
        return {"transaction_type": []}
    return {"transaction_type": result[0]}

# Endpoint to get the credit rating of a vendor with a specific rowguid
@app.get("/v1/works_cycles/vendor_credit_rating", operation_id="get_vendor_credit_rating", summary="Retrieves the credit rating of a vendor associated with a specific business entity. The business entity is identified by its unique rowguid. The response will contain the credit rating of the vendor.")
async def get_vendor_credit_rating(rowguid: str = Query(..., description="Rowguid of the business entity")):
    cursor.execute("SELECT T1.CreditRating FROM Vendor AS T1 INNER JOIN BusinessEntity AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.rowguid = ?", (rowguid,))
    result = cursor.fetchone()
    if not result:
        return {"credit_rating": []}
    return {"credit_rating": result[0]}

# Endpoint to get the preferred vendor status based on rowguid
@app.get("/v1/works_cycles/preferred_vendor_status", operation_id="get_preferred_vendor_status", summary="Retrieves the preferred vendor status associated with a specific business entity, identified by its unique rowguid.")
async def get_preferred_vendor_status(rowguid: str = Query(..., description="Rowguid of the business entity")):
    cursor.execute("SELECT T1.PreferredVendorStatus FROM Vendor AS T1 INNER JOIN BusinessEntity AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.rowguid = ?", (rowguid,))
    result = cursor.fetchone()
    if not result:
        return {"preferred_vendor_status": []}
    return {"preferred_vendor_status": result[0]}

# Endpoint to get the active flag of a vendor based on person ID
@app.get("/v1/works_cycles/vendor_active_flag", operation_id="get_vendor_active_flag", summary="Retrieves the active status of a vendor associated with the specified person ID. This operation returns a boolean value indicating whether the vendor is currently active or not.")
async def get_vendor_active_flag(person_id: int = Query(..., description="Person ID")):
    cursor.execute("SELECT T1.ActiveFlag FROM Vendor AS T1 INNER JOIN BusinessEntityContact AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.PersonID = ?", (person_id,))
    result = cursor.fetchone()
    if not result:
        return {"active_flag": []}
    return {"active_flag": result[0]}

# Endpoint to get the phone number based on first name and last name
@app.get("/v1/works_cycles/phone_number_by_name", operation_id="get_phone_number_by_name", summary="Get the phone number based on the first name and last name of a person")
async def get_phone_number_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T2.PhoneNumber FROM Person AS T1 INNER JOIN PersonPhone AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"phone_number": []}
    return {"phone_number": result[0]}

# Endpoint to get the phone number type name based on business entity ID
@app.get("/v1/works_cycles/phone_number_type_name", operation_id="get_phone_number_type_name", summary="Get the phone number type name based on the business entity ID")
async def get_phone_number_type_name(business_entity_id: int = Query(..., description="Business entity ID")):
    cursor.execute("SELECT T2.Name FROM PersonPhone AS T1 INNER JOIN PhoneNumberType AS T2 USING (PhoneNumberTypeID) WHERE T1.BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchone()
    if not result:
        return {"phone_number_type_name": []}
    return {"phone_number_type_name": result[0]}

# Endpoint to get the job title based on first name, middle name, and last name
@app.get("/v1/works_cycles/job_title_by_name", operation_id="get_job_title_by_name", summary="Retrieves the job title of an employee by matching their full name. The operation requires the first, middle, and last names of the person to accurately identify the corresponding job title.")
async def get_job_title_by_name(first_name: str = Query(..., description="First name of the person"), middle_name: str = Query(..., description="Middle name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T2.JobTitle FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.FirstName = ? AND T1.MiddleName = ? AND T1.LastName = ?", (first_name, middle_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"job_title": []}
    return {"job_title": result[0]}

# Endpoint to get the count of product subcategories based on product category name
@app.get("/v1/works_cycles/count_product_subcategories", operation_id="get_count_product_subcategories", summary="Retrieves the total number of distinct product subcategories associated with a specific product category. The count is determined by matching the provided product category name with the corresponding entries in the ProductCategory table and then counting the associated subcategories in the ProductSubcategory table.")
async def get_count_product_subcategories(product_category_name: str = Query(..., description="Name of the product category")):
    cursor.execute("SELECT COUNT(*) FROM ProductCategory AS T1 INNER JOIN ProductSubcategory AS T2 ON T1.ProductCategoryID = T2.ProductCategoryID WHERE T1.Name = ?", (product_category_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of documents with a summary based on job title and hire date
@app.get("/v1/works_cycles/percentage_documents_with_summary", operation_id="get_percentage_documents_with_summary", summary="Retrieves the proportion of documents with a summary for a specific employee role and hire date. The calculation is based on the total number of documents owned by employees with the given job title and hire date.")
async def get_percentage_documents_with_summary(job_title: str = Query(..., description="Job title of the employee"), hire_date: str = Query(..., description="Hire date of the employee in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.DocumentSummary IS NOT NULL THEN 1 ELSE 0 END) AS REAL) / COUNT(T1.DocumentSummary) FROM Document AS T1 INNER JOIN Employee AS T2 ON T1.Owner = T2.BusinessEntityID WHERE T2.JobTitle = ? AND T2.HireDate = ?", (job_title, hire_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the price difference percentage based on product name
@app.get("/v1/works_cycles/price_difference_percentage", operation_id="get_price_difference_percentage", summary="Retrieves the percentage difference between the last receipt cost and the standard price for a specific product. The product is identified by its name.")
async def get_price_difference_percentage(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT (T1.LastReceiptCost - T1.StandardPrice) / T1.StandardPrice FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"price_difference_percentage": []}
    return {"price_difference_percentage": result[0]}

# Endpoint to get the percentage of phone numbers of a specific type
@app.get("/v1/works_cycles/percentage_phone_numbers_by_type", operation_id="get_percentage_phone_numbers_by_type", summary="Get the percentage of phone numbers of a specific type")
async def get_percentage_phone_numbers_by_type(phone_number_type: str = Query(..., description="Type of the phone number")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.Name) FROM PersonPhone AS T1 INNER JOIN PhoneNumberType AS T2 ON T1.PhoneNumberTypeID = T2.PhoneNumberTypeID", (phone_number_type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the product assembly ID based on unit measure code, BOM level, and per assembly quantity
@app.get("/v1/works_cycles/product_assembly_id", operation_id="get_product_assembly_id", summary="Get the product assembly ID based on the unit measure code, BOM level, and per assembly quantity")
async def get_product_assembly_id(unit_measure_code: str = Query(..., description="Unit measure code"), bom_level: int = Query(..., description="BOM level"), per_assembly_qty: int = Query(..., description="Per assembly quantity")):
    cursor.execute("SELECT ProductAssemblyID FROM BillOfMaterials WHERE UnitMeasureCode = ? AND BOMLevel = ? AND PerAssemblyQty > ?", (unit_measure_code, bom_level, per_assembly_qty))
    result = cursor.fetchone()
    if not result:
        return {"product_assembly_id": []}
    return {"product_assembly_id": result[0]}

# Endpoint to get the count of work order routings based on actual resource hours
@app.get("/v1/works_cycles/count_work_order_routing_by_hours", operation_id="get_count_work_order_routing_by_hours", summary="Retrieves the total number of work order routings that have the specified actual resource hours. This operation provides a count of work order routings based on the actual hours spent by resources, offering insights into resource utilization and work order progress.")
async def get_count_work_order_routing_by_hours(actual_resource_hrs: int = Query(..., description="Actual resource hours")):
    cursor.execute("SELECT COUNT(LocationID) FROM WorkOrderRouting WHERE ActualResourceHrs = ?", (actual_resource_hrs,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of work order routings based on location ID
@app.get("/v1/works_cycles/count_work_order_routing_by_location", operation_id="get_count_work_order_routing_by_location", summary="Retrieves the total number of work order routings associated with a specific location. This operation considers the location ID and filters the routings based on the provided ID. The count is determined by examining the WorkOrderRouting table and correlating it with the BillOfMaterials and WorkOrder tables using the ProductAssemblyID and WorkOrderID fields, respectively.")
async def get_count_work_order_routing_by_location(location_id: int = Query(..., description="Location ID")):
    cursor.execute("SELECT COUNT(*) FROM WorkOrderRouting AS T1 INNER JOIN BillOfMaterials AS T2 ON T1.LocationID = T2.ProductAssemblyID INNER JOIN WorkOrder AS T3 ON T3.WorkOrderID = T1.WorkOrderID WHERE T1.LocationID = ?", (location_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of per assembly quantity grouped by unit measure name
@app.get("/v1/works_cycles/sum_per_assembly_qty_by_unit_measure", operation_id="get_sum_per_assembly_qty_by_unit_measure", summary="Retrieves the total sum of per assembly quantity, grouped by the name of the unit measure. The operation accepts multiple unit measure codes as input parameters, which are used to filter the results. The response includes the sum of per assembly quantity and the corresponding unit measure name.")
async def get_sum_per_assembly_qty_by_unit_measure(unit_measure_code1: str = Query(..., description="First unit measure code"), unit_measure_code2: str = Query(..., description="Second unit measure code"), unit_measure_code3: str = Query(..., description="Third unit measure code")):
    cursor.execute("SELECT SUM(T1.PerAssemblyQty), T2.Name FROM BillOfMaterials AS T1 INNER JOIN UnitMeasure AS T2 ON T1.UnitMeasureCode = T2.UnitMeasureCode WHERE T1.UnitMeasureCode IN (?, ?, ?) GROUP BY T2.Name", (unit_measure_code1, unit_measure_code2, unit_measure_code3))
    result = cursor.fetchall()
    if not result:
        return {"sum_per_assembly_qty": []}
    return {"sum_per_assembly_qty": result}

# Endpoint to get product IDs not in work orders
@app.get("/v1/works_cycles/product_ids_not_in_work_orders", operation_id="get_product_ids_not_in_work_orders", summary="Retrieves a list of unique product identifiers that are not associated with any work orders. This operation is useful for identifying products that have not been included in any work orders, which can help in managing inventory and production planning.")
async def get_product_ids_not_in_work_orders():
    cursor.execute("SELECT ProductID FROM Product WHERE ProductID NOT IN ( SELECT T1.ProductID FROM Product AS T1 INNER JOIN WorkOrder AS T2 ON T1.ProductID = T2.ProductID )")
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get product IDs based on transaction type
@app.get("/v1/works_cycles/product_ids_by_transaction_type", operation_id="get_product_ids_by_transaction_type", summary="Retrieves a list of product IDs associated with a specific transaction type. The operation filters products based on the provided transaction type and returns their corresponding IDs.")
async def get_product_ids_by_transaction_type(transaction_type: str = Query(..., description="Transaction type")):
    cursor.execute("SELECT ProductID FROM Product WHERE ProductID IN ( SELECT ProductID FROM TransactionHistory WHERE TransactionType = ? )", (transaction_type,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get employee names based on job title
@app.get("/v1/works_cycles/employee_names_by_job_title", operation_id="get_employee_names_by_job_title", summary="Retrieves the first, middle, and last names of employees who hold a specific job title. The job title is provided as an input parameter.")
async def get_employee_names_by_job_title(job_title: str = Query(..., description="Job title")):
    cursor.execute("SELECT T2.FirstName, T2.MiddleName, T2.LastName FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.JobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": result}

# Endpoint to get job title based on employee name
@app.get("/v1/works_cycles/job_title_by_employee_name", operation_id="get_job_title_by_employee_name", summary="Retrieves the job title of an employee by matching the provided first, middle, and last names. The operation searches for an employee with the given names and returns the associated job title.")
async def get_job_title_by_employee_name(first_name: str = Query(..., description="First name"), middle_name: str = Query(..., description="Middle name"), last_name: str = Query(..., description="Last name")):
    cursor.execute("SELECT T1.JobTitle FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.FirstName = ? AND T2.MiddleName = ? AND T2.LastName = ?", (first_name, middle_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"job_title": []}
    return {"job_title": result[0]}

# Endpoint to get the count of employees based on email promotion and gender
@app.get("/v1/works_cycles/count_employees_by_email_promotion_gender", operation_id="get_count_employees_by_email_promotion_gender", summary="Retrieves the total number of employees who meet the specified email promotion and gender criteria. The email promotion parameter filters employees based on their email promotion status (0 or 1), while the gender parameter filters employees based on their gender.")
async def get_count_employees_by_email_promotion_gender(email_promotion: int = Query(..., description="Email promotion (0 or 1)"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.EmailPromotion = ? AND T1.Gender = ?", (email_promotion, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top salesperson based on quota year
@app.get("/v1/works_cycles/top_salesperson_by_quota_year", operation_id="get_top_salesperson_by_quota_year", summary="Retrieves the top salesperson based on the sales quota performance for a specific year. The salesperson with the highest ratio of actual sales to sales quota is returned. The year is specified in the 'YYYY' format.")
async def get_top_salesperson_by_quota_year(quota_year: str = Query(..., description="Quota year in 'YYYY' format")):
    cursor.execute("SELECT BusinessEntityID FROM SalesPerson WHERE BusinessEntityID IN ( SELECT BusinessEntityID FROM SalesPersonQuotaHistory WHERE STRFTIME('%Y', QuotaDate) = ? ) ORDER BY CAST(SalesLastYear AS REAL) / SalesQuota DESC LIMIT 1", (quota_year,))
    result = cursor.fetchone()
    if not result:
        return {"business_entity_id": []}
    return {"business_entity_id": result[0]}

# Endpoint to get the count of employees based on person type and marital status
@app.get("/v1/works_cycles/count_employees_by_person_type_marital_status", operation_id="get_count_employees_by_person_type_marital_status", summary="Retrieves the total number of employees based on the specified person type and marital status. The person type and marital status are used as filters to calculate the count.")
async def get_count_employees_by_person_type_marital_status(person_type: str = Query(..., description="Person type"), marital_status: str = Query(..., description="Marital status")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.PersonType = ? AND T1.MaritalStatus = ?", (person_type, marital_status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total sick leave hours for employees based on email promotion status
@app.get("/v1/works_cycles/total_sick_leave_hours_by_email_promotion", operation_id="get_total_sick_leave_hours", summary="Retrieves the cumulative sick leave hours for employees who have a specified email promotion status. The status is represented as a binary value, where 0 indicates no promotion and 1 indicates a promotion.")
async def get_total_sick_leave_hours(email_promotion: int = Query(..., description="Email promotion status of the employee (0 or 1)")):
    cursor.execute("SELECT SUM(T1.SickLeaveHours) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.EmailPromotion = ?", (email_promotion,))
    result = cursor.fetchone()
    if not result:
        return {"total_hours": []}
    return {"total_hours": result[0]}

# Endpoint to get the count of employees based on person type and hire date
@app.get("/v1/works_cycles/employee_count_by_person_type_hire_date", operation_id="get_employee_count_by_person_type_hire_date", summary="Retrieves the total number of employees with a specified person type who were hired before a given year. The person type and hire year are provided as input parameters.")
async def get_employee_count_by_person_type_hire_date(person_type: str = Query(..., description="Person type of the employee"), hire_year: int = Query(..., description="Year before which the employee was hired (YYYY)")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.PersonType = ? AND SUBSTR(T1.HireDate, 0, 4) < ?", (person_type, hire_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top salesperson by sales quota for a specific year
@app.get("/v1/works_cycles/top_salesperson_by_sales_quota", operation_id="get_top_salesperson_by_sales_quota", summary="Retrieves the top salesperson based on their total sales quota for a specific year. The operation calculates the total sales quota for each salesperson in the given year and returns the salesperson with the highest sales quota. The input parameter is the year for which the sales quota is considered.")
async def get_top_salesperson_by_sales_quota(year: str = Query(..., description="Year for which the sales quota is considered (YYYY)")):
    cursor.execute("SELECT T1.BusinessEntityID, SUM(T1.SalesQuota) FROM SalesPerson AS T1 INNER JOIN SalesPersonQuotaHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE STRFTIME('%Y', T2.QuotaDate) = ? GROUP BY T1.BusinessEntityID ORDER BY SUM(T1.SalesYTD) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"top_salesperson": []}
    return {"top_salesperson": {"BusinessEntityID": result[0], "SalesQuota": result[1]}}

# Endpoint to get the count of employees based on first name, marital status, and organization level
@app.get("/v1/works_cycles/employee_count_by_first_name_marital_status_organization_level", operation_id="get_employee_count_by_first_name_marital_status_organization_level", summary="Retrieves the total number of employees who share a specific first name, marital status, and organization level. This operation provides a count of employees based on the provided first name, marital status, and organization level parameters.")
async def get_employee_count_by_first_name_marital_status_organization_level(first_name: str = Query(..., description="First name of the employee"), marital_status: str = Query(..., description="Marital status of the employee"), organization_level: int = Query(..., description="Organization level of the employee")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.FirstName = ? AND T1.MaritalStatus = ? AND T1.OrganizationLevel = ?", (first_name, marital_status, organization_level))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the last name and job title of employees based on document title
@app.get("/v1/works_cycles/employee_last_name_job_title_by_document_title", operation_id="get_employee_last_name_job_title", summary="Retrieves the last names and job titles of employees who own a document with the specified title. The document title is provided as an input parameter.")
async def get_employee_last_name_job_title(document_title: str = Query(..., description="Title of the document")):
    cursor.execute("SELECT T1.LastName, T3.JobTitle FROM Person AS T1 INNER JOIN Document AS T2 ON T1.BusinessEntityID = T2.Owner INNER JOIN Employee AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T2.Title = ?", (document_title,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"LastName": row[0], "JobTitle": row[1]} for row in result]}

# Endpoint to get the count of employees without a suffix
@app.get("/v1/works_cycles/employee_count_without_suffix", operation_id="get_employee_count_without_suffix", summary="Retrieves the total number of employees who do not have a suffix in their personal information. This operation performs a database query to count the unique employee records that lack a suffix entry.")
async def get_employee_count_without_suffix():
    cursor.execute("SELECT COUNT(T3.BusinessEntityID) FROM ( SELECT T1.BusinessEntityID FROM Employee AS T1 INNER JOIN Person AS T2 USING (BusinessEntityID) WHERE T2.Suffix IS NULL GROUP BY T1.BusinessEntityID ) AS T3")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct salesperson IDs based on territory and year
@app.get("/v1/works_cycles/distinct_salesperson_ids_by_territory_year", operation_id="get_distinct_salesperson_ids", summary="Retrieves a unique list of salesperson identifiers associated with a given territory and year. The operation filters salespeople based on their territory and the year of their quota date, ensuring that only distinct identifiers are returned.")
async def get_distinct_salesperson_ids(territory_id: int = Query(..., description="Territory ID of the salesperson"), year: str = Query(..., description="Year for which the quota is considered (YYYY)")):
    cursor.execute("SELECT DISTINCT T1.BusinessEntityID FROM SalesPerson AS T1 INNER JOIN SalesPersonQuotaHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.TerritoryID = ? AND STRFTIME('%Y', QuotaDate) = ?", (territory_id, year))
    result = cursor.fetchall()
    if not result:
        return {"salesperson_ids": []}
    return {"salesperson_ids": [row[0] for row in result]}

# Endpoint to get the last names of employees based on a list of business entity IDs
@app.get("/v1/works_cycles/employee_last_names_by_business_entity_ids", operation_id="get_employee_last_names", summary="Retrieve the last names of employees associated with the provided business entity IDs. This operation fetches the last names from the Person table, joining it with the Employee and JobCandidate tables based on the BusinessEntityID. The input parameters specify the BusinessEntityIDs to filter the results.")
async def get_employee_last_names(business_entity_id_1: int = Query(..., description="First business entity ID"), business_entity_id_2: int = Query(..., description="Second business entity ID")):
    cursor.execute("SELECT T3.LastName FROM Employee AS T1 INNER JOIN JobCandidate AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Person AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T1.BusinessEntityID IN (?, ?)", (business_entity_id_1, business_entity_id_2))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get email addresses of employees based on gender and marital status
@app.get("/v1/works_cycles/employee_email_addresses", operation_id="get_employee_email_addresses", summary="Retrieves the email addresses of employees who match the specified gender and marital status. The operation filters employees based on the provided gender and marital status parameters, and returns their corresponding email addresses.")
async def get_employee_email_addresses(gender: str = Query(..., description="Gender of the employee"), marital_status: str = Query(..., description="Marital status of the employee")):
    cursor.execute("SELECT T3.EmailAddress FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN EmailAddress AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T1.Gender = ? AND T1.MaritalStatus = ?", (gender, marital_status))
    result = cursor.fetchall()
    if not result:
        return {"email_addresses": []}
    return {"email_addresses": [row[0] for row in result]}

# Endpoint to get product colors based on product IDs
@app.get("/v1/works_cycles/product_colors", operation_id="get_product_colors", summary="Retrieves the colors of up to three products specified by their unique identifiers. The operation returns the color information for each product, if available, based on the provided product IDs.")
async def get_product_colors(product_id1: int = Query(..., description="First product ID"), product_id2: int = Query(..., description="Second product ID"), product_id3: int = Query(..., description="Third product ID")):
    cursor.execute("SELECT T1.Color FROM Product AS T1 INNER JOIN ProductReview AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductID IN (?, ?, ?)", (product_id1, product_id2, product_id3))
    result = cursor.fetchall()
    if not result:
        return {"colors": []}
    return {"colors": [row[0] for row in result]}

# Endpoint to get the sum of sales quotas for a specific business entity ID and year
@app.get("/v1/works_cycles/sales_quota_sum", operation_id="get_sales_quota_sum", summary="Retrieves the total sales quota for a given business entity in a specific year. The operation calculates the sum of sales quotas based on the provided business entity ID and year, providing a comprehensive view of the sales target for the specified period.")
async def get_sales_quota_sum(business_entity_id: int = Query(..., description="Business entity ID"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(T1.SalesQuota) FROM SalesPerson AS T1 INNER JOIN SalesPersonQuotaHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.BusinessEntityID = ? AND STRFTIME('%Y', QuotaDate) = ?", (business_entity_id, year))
    result = cursor.fetchone()
    if not result:
        return {"sales_quota_sum": []}
    return {"sales_quota_sum": result[0]}

# Endpoint to get distinct business entity IDs based on year and sales quota comparison
@app.get("/v1/works_cycles/distinct_business_entity_ids", operation_id="get_distinct_business_entity_ids", summary="Retrieves a unique list of business entity IDs that meet the specified year and sales quota criteria. The operation filters entities based on a provided year and compares their sales quota to the previous year's sales. Only entities with a sales quota lower than the previous year's sales are included in the result.")
async def get_distinct_business_entity_ids(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T1.BusinessEntityID FROM SalesPerson AS T1 INNER JOIN SalesPersonQuotaHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE STRFTIME('%Y', T2.QuotaDate) = ? AND T1.SalesQuota < T1.SalesLastYear", (year,))
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": [row[0] for row in result]}

# Endpoint to get the percentage of female employees with email promotion
@app.get("/v1/works_cycles/female_employee_percentage", operation_id="get_female_employee_percentage", summary="Retrieves the percentage of female employees who have email promotion enabled. The calculation is based on the total number of employees with the specified email promotion flag.")
async def get_female_employee_percentage(email_promotion: int = Query(..., description="Email promotion flag (1 for yes, 0 for no)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Gender = 'F' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.EmailPromotion = ?", (email_promotion,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of married to single employees based on person type
@app.get("/v1/works_cycles/married_to_single_ratio", operation_id="get_married_to_single_ratio", summary="Retrieves the percentage of married employees relative to single employees for a specified person type. The calculation is based on the marital status of employees and their corresponding person type.")
async def get_married_to_single_ratio(person_type: str = Query(..., description="Person type (e.g., 'EM' for employee)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.MaritalStatus = 'M' THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN T1.MaritalStatus = 'S' THEN 1 ELSE 0 END) FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.PersonType = ?", (person_type,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the sum of bonuses and the percentage of bonuses to sales quotas for a specific year
@app.get("/v1/works_cycles/bonus_and_percentage", operation_id="get_bonus_and_percentage", summary="Retrieves the total bonus amount and the percentage of bonuses relative to sales quotas for a given year. The calculation is based on the sum of bonuses and sales quotas for all salespeople in the specified year.")
async def get_bonus_and_percentage(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(T1.Bonus), CAST(SUM(T1.Bonus) AS REAL) * 100 / SUM(T1.SalesQuota) FROM SalesPerson AS T1 INNER JOIN SalesPersonQuotaHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE STRFTIME('%Y', T2.QuotaDate) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"bonus_sum": [], "bonus_percentage": []}
    return {"bonus_sum": result[0], "bonus_percentage": result[1]}

# Endpoint to get the count of credit cards based on card type
@app.get("/v1/works_cycles/credit_card_count", operation_id="get_credit_card_count", summary="Retrieves the total number of credit cards of a specific type. The operation requires the card type as input and returns the corresponding count.")
async def get_credit_card_count(card_type: str = Query(..., description="Card type (e.g., 'vista')")):
    cursor.execute("SELECT COUNT(CardNumber) FROM CreditCard WHERE CardType = ?", (card_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the store based on SalesPersonID
@app.get("/v1/works_cycles/store_name_by_salesperson_id", operation_id="get_store_name", summary="Retrieves the name of the store associated with the provided SalesPersonID.")
async def get_store_name(salesperson_id: int = Query(..., description="SalesPersonID of the store")):
    cursor.execute("SELECT Name FROM Store WHERE SalesPersonID = ?", (salesperson_id,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the sum of quantities for a specific product in a given month and transaction type
@app.get("/v1/works_cycles/sum_quantity_by_date_type_product", operation_id="get_sum_quantity", summary="Retrieves the total quantity of a specific product involved in a certain transaction type during a given month. The month is specified in 'YYYY-MM%' format, the transaction type is provided, and the product is identified by its unique ID.")
async def get_sum_quantity(transaction_date: str = Query(..., description="Transaction date in 'YYYY-MM%' format"), transaction_type: str = Query(..., description="Transaction type"), product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT SUM(Quantity) FROM TransactionHistory WHERE TransactionDate LIKE ? AND TransactionType = ? AND ProductID = ?", (transaction_date, transaction_type, product_id))
    result = cursor.fetchone()
    if not result:
        return {"sum_quantity": []}
    return {"sum_quantity": result[0]}

# Endpoint to get credit card numbers expiring before a given year
@app.get("/v1/works_cycles/credit_cards_expiring_before_year", operation_id="get_credit_cards", summary="Retrieves the credit card numbers that will expire before the specified year. The expiration year is provided as an input parameter.")
async def get_credit_cards(exp_year: int = Query(..., description="Expiration year")):
    cursor.execute("SELECT CardNumber FROM CreditCard WHERE ExpYear < ?", (exp_year,))
    result = cursor.fetchall()
    if not result:
        return {"card_numbers": []}
    return {"card_numbers": [row[0] for row in result]}

# Endpoint to get the pay rate of employees hired at a specific age
@app.get("/v1/works_cycles/employee_pay_rate_by_hire_age", operation_id="get_employee_pay_rate", summary="Retrieves the pay rate of employees who were hired at a specified age. The age is calculated as the difference between the year of hire and the year of birth. This operation provides insights into the compensation structure for employees based on their age at the time of hiring.")
async def get_employee_pay_rate(hire_age: int = Query(..., description="Age at which the employee was hired")):
    cursor.execute("SELECT T2.Rate FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE STRFTIME('%Y', T1.HireDate) - STRFTIME('%Y', T1.BirthDate) = ?", (hire_age,))
    result = cursor.fetchall()
    if not result:
        return {"rates": []}
    return {"rates": [row[0] for row in result]}

# Endpoint to get the sales territory name for a specific salesperson
@app.get("/v1/works_cycles/sales_territory_by_salesperson", operation_id="get_sales_territory", summary="Retrieves the name of the sales territory associated with a specific salesperson, identified by their BusinessEntityID.")
async def get_sales_territory(business_entity_id: int = Query(..., description="BusinessEntityID of the salesperson")):
    cursor.execute("SELECT T2.Name FROM SalesPerson AS T1 INNER JOIN SalesTerritory AS T2 ON T1.TerritoryID = T2.TerritoryID WHERE T1.BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchall()
    if not result:
        return {"territory_names": []}
    return {"territory_names": [row[0] for row in result]}

# Endpoint to get purchase order IDs for vendors with a specific credit rating
@app.get("/v1/works_cycles/purchase_order_ids_by_credit_rating", operation_id="get_purchase_order_ids", summary="Retrieves the purchase order IDs associated with vendors who have a specified credit rating. The credit rating is provided as an input parameter.")
async def get_purchase_order_ids(credit_rating: int = Query(..., description="Credit rating of the vendor")):
    cursor.execute("SELECT T2.PurchaseOrderID FROM Vendor AS T1 INNER JOIN PurchaseOrderHeader AS T2 ON T1.BusinessEntityID = T2.VendorID WHERE T1.CreditRating = ?", (credit_rating,))
    result = cursor.fetchall()
    if not result:
        return {"purchase_order_ids": []}
    return {"purchase_order_ids": [row[0] for row in result]}

# Endpoint to get the phone number type name for a specific phone number
@app.get("/v1/works_cycles/phone_number_type_by_phone_number", operation_id="get_phone_number_type", summary="Get the phone number type name for a specific phone number")
async def get_phone_number_type(phone_number: str = Query(..., description="Phone number")):
    cursor.execute("SELECT T2.Name FROM PersonPhone AS T1 INNER JOIN PhoneNumberType AS T2 ON T1.PhoneNumberTypeID = T2.PhoneNumberTypeID WHERE T1.PhoneNumber = ?", (phone_number,))
    result = cursor.fetchall()
    if not result:
        return {"phone_number_types": []}
    return {"phone_number_types": [row[0] for row in result]}

# Endpoint to get the total freight cost for a specific ship method and order date
@app.get("/v1/works_cycles/total_freight_by_ship_method_order_date", operation_id="get_total_freight", summary="Retrieves the total freight cost for a given ship method and order date. The operation calculates the sum of freight costs from the PurchaseOrderHeader table, filtered by the specified ship method name and order date. The ship method name and order date are provided as input parameters.")
async def get_total_freight(ship_method_name: str = Query(..., description="Name of the ship method"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(t2.freight) FROM ShipMethod AS t1 INNER JOIN PurchaseOrderHeader AS t2 ON t1.shipmethodid = t2.shipmethodid WHERE t1.name = ? AND t2.orderdate = ?", (ship_method_name, order_date))
    result = cursor.fetchone()
    if not result:
        return {"total_freight": []}
    return {"total_freight": result[0]}

# Endpoint to get the count of total due based on person details
@app.get("/v1/works_cycles/count_total_due_by_person_details", operation_id="get_count_total_due", summary="Retrieves the total count of due items associated with a specific individual, identified by their first, middle, and last names, as well as their person type. This operation provides a comprehensive overview of the individual's outstanding due items.")
async def get_count_total_due(first_name: str = Query(..., description="First name of the person"), middle_name: str = Query(..., description="Middle name of the person"), last_name: str = Query(..., description="Last name of the person"), person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT COUNT(T2.TotalDue) FROM Person AS T1 INNER JOIN SalesOrderHeader AS T2 ON T1.ModifiedDate = T2.DueDate WHERE T1.FirstName = ? AND T1.MiddleName = ? AND T1.LastName = ? AND T1.PersonType = ?", (first_name, middle_name, last_name, person_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get email addresses based on job title
@app.get("/v1/works_cycles/email_addresses_by_job_title", operation_id="get_email_addresses", summary="Retrieves the email addresses of employees who hold a specific job title. The job title is provided as an input parameter.")
async def get_email_addresses(job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT T3.EmailAddress FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN EmailAddress AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T1.JobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"email_addresses": []}
    return {"email_addresses": [row[0] for row in result]}

# Endpoint to get the count of customers in a sales territory
@app.get("/v1/works_cycles/count_customers_in_sales_territory", operation_id="get_count_customers", summary="Retrieves the total number of customers associated with a specific sales territory. The operation requires the name of the sales territory as input and returns the count of customers within that territory.")
async def get_count_customers(territory_name: str = Query(..., description="Name of the sales territory")):
    cursor.execute("SELECT COUNT(T2.CustomerID) FROM SalesTerritory AS T1 INNER JOIN Customer AS T2 ON T1.TerritoryID = T2.TerritoryID WHERE T1.Name = ?", (territory_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get ship-to address IDs based on sales order ID
@app.get("/v1/works_cycles/ship_to_address_ids_by_sales_order", operation_id="get_ship_to_address_ids", summary="Retrieves the unique identifiers of the shipping addresses associated with a specific sales order. The operation filters the sales order records based on the provided sales order ID and returns the distinct shipping address identifiers.")
async def get_ship_to_address_ids(sales_order_id: int = Query(..., description="Sales order ID")):
    cursor.execute("SELECT T1.ShipToAddressID FROM SalesOrderHeader AS T1 INNER JOIN Address AS T2 ON T1.BillToAddressID = T2.AddressID WHERE T1.SalesOrderID = ? GROUP BY T1.ShipToAddressID", (sales_order_id,))
    result = cursor.fetchall()
    if not result:
        return {"ship_to_address_ids": []}
    return {"ship_to_address_ids": [row[0] for row in result]}

# Endpoint to get the count of employees with a specific job title and below-average pay rate
@app.get("/v1/works_cycles/count_employees_below_average_pay", operation_id="get_count_employees_below_average_pay", summary="Retrieves the number of employees with a specified job title who are paid below the average rate for that job title. The job title can be partially matched using a wildcard character (%).")
async def get_count_employees_below_average_pay(job_title: str = Query(..., description="Job title of the employee (use % for wildcard)")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.JobTitle LIKE ? AND T2.Rate < ( SELECT AVG(T2.Rate) FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.JobTitle LIKE ? )", (job_title, job_title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of sales orders in a specific territory
@app.get("/v1/works_cycles/percentage_sales_orders_in_territory", operation_id="get_percentage_sales_orders", summary="Retrieves the percentage of sales orders associated with a specific sales territory. The calculation is based on the total count of sales orders in the given territory compared to the overall count of sales orders. The territory is identified by its name.")
async def get_percentage_sales_orders(territory_name: str = Query(..., description="Name of the sales territory")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.SalesOrderID) FROM SalesOrderHeader AS T1 INNER JOIN SalesTerritory AS T2 ON T1.TerritoryID = T2.TerritoryID", (territory_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the modified date based on phone number
@app.get("/v1/works_cycles/modified_date_by_phone_number", operation_id="get_modified_date", summary="Get the modified date based on a specific phone number")
async def get_modified_date(phone_number: str = Query(..., description="Phone number")):
    cursor.execute("SELECT ModifiedDate FROM PersonPhone WHERE PhoneNumber = ?", (phone_number,))
    result = cursor.fetchall()
    if not result:
        return {"modified_dates": []}
    return {"modified_dates": [row[0] for row in result]}

# Endpoint to get the top salesperson by sales YTD
@app.get("/v1/works_cycles/top_salesperson_by_sales_ytd", operation_id="get_top_salesperson", summary="Retrieves the ID of the top-performing salesperson based on their year-to-date sales. The salesperson with the highest sales total is returned.")
async def get_top_salesperson():
    cursor.execute("SELECT BusinessEntityID FROM SalesPerson ORDER BY SalesYTD DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"business_entity_id": []}
    return {"business_entity_id": result[0]}

# Endpoint to get vendor names based on active flag
@app.get("/v1/works_cycles/vendor_names_by_active_flag", operation_id="get_vendor_names", summary="Retrieves the names of vendors based on their active status. The operation allows for filtering vendors by their active status, which can be either active (1) or inactive (0).")
async def get_vendor_names(active_flag: int = Query(..., description="Active flag (1 for active, 0 for inactive)")):
    cursor.execute("SELECT Name FROM Vendor WHERE ActiveFlag = ?", (active_flag,))
    result = cursor.fetchall()
    if not result:
        return {"vendor_names": []}
    return {"vendor_names": [row[0] for row in result]}

# Endpoint to get the TerritoryID with the highest count of customers modified before a given date
@app.get("/v1/works_cycles/territory_with_most_customers_before_date", operation_id="get_territory_with_most_customers", summary="Retrieves the TerritoryID that has the most customers who were modified before the provided date. The date should be in the 'YYYY-MM-DD' format.")
async def get_territory_with_most_customers(modified_date: str = Query(..., description="Date in 'YYYY-MM-DD' format before which customers were modified")):
    cursor.execute("SELECT TerritoryID FROM Customer WHERE ModifiedDate < ? GROUP BY TerritoryID ORDER BY COUNT(TerritoryID) DESC LIMIT 1", (modified_date,))
    result = cursor.fetchone()
    if not result:
        return {"TerritoryID": []}
    return {"TerritoryID": result[0]}

# Endpoint to get the total due for purchase orders on a specific date
@app.get("/v1/works_cycles/total_due_on_date", operation_id="get_total_due", summary="Retrieves the total sum of all due amounts for purchase orders placed on a specific date. The date must be provided in the 'YYYY-MM-DD' format.")
async def get_total_due(order_date: str = Query(..., description="Date in 'YYYY-MM-DD' format for which to get the total due")):
    cursor.execute("SELECT SUM(TotalDue) FROM PurchaseOrderHeader WHERE OrderDate LIKE ?", (order_date + '%',))
    result = cursor.fetchone()
    if not result:
        return {"TotalDue": []}
    return {"TotalDue": result[0]}

# Endpoint to get the most common first name for a specific person type
@app.get("/v1/works_cycles/most_common_first_name_by_type", operation_id="get_most_common_first_name", summary="Retrieves the most frequently occurring first name for a given person type. The person type is specified as an input parameter, and the operation returns the first name that appears most frequently in the database for that type of person.")
async def get_most_common_first_name(person_type: str = Query(..., description="Type of person (e.g., 'VC')")):
    cursor.execute("SELECT FirstName FROM Person WHERE PersonType = ? GROUP BY FirstName ORDER BY COUNT(*) DESC LIMIT 1", (person_type,))
    result = cursor.fetchone()
    if not result:
        return {"FirstName": []}
    return {"FirstName": result[0]}

# Endpoint to get the count of order quantities for a specific product name
@app.get("/v1/works_cycles/order_quantity_count_by_product_name", operation_id="get_order_quantity_count", summary="Retrieves the total number of order quantities associated with a specified product. The product is identified by its name.")
async def get_order_quantity_count(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT COUNT(OrderQty) FROM SalesOrderDetail WHERE ProductID IN ( SELECT ProductID FROM Product WHERE Name = ? )", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the BusinessEntityID for a specific credit card number
@app.get("/v1/works_cycles/business_entity_by_card_number", operation_id="get_business_entity_by_card_number", summary="Retrieves the unique identifier of the business entity associated with a given credit card number. This operation fetches the BusinessEntityID from the PersonCreditCard table, which is linked to the CreditCard table via the CreditCardID. The input parameter is the credit card number, which is used to filter the results.")
async def get_business_entity_by_card_number(card_number: int = Query(..., description="Credit card number")):
    cursor.execute("SELECT T2.BusinessEntityID FROM CreditCard AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.CreditCardID = T2.CreditCardID WHERE T1.CardNumber = ?", (card_number,))
    result = cursor.fetchone()
    if not result:
        return {"BusinessEntityID": []}
    return {"BusinessEntityID": result[0]}

# Endpoint to get the city and address for a specific business entity and address type
@app.get("/v1/works_cycles/address_by_business_entity_and_type", operation_id="get_address_by_business_entity_and_type", summary="Retrieves the city and address line 1 for a given business entity and address type. The operation requires the business entity ID and the type of address (e.g., 'Home') as input parameters. The result is a list of city and address line 1 combinations that match the provided business entity and address type.")
async def get_address_by_business_entity_and_type(business_entity_id: int = Query(..., description="Business entity ID"), address_type: str = Query(..., description="Type of address (e.g., 'Home')")):
    cursor.execute("SELECT T3.City, T3.AddressLine1 FROM BusinessEntityAddress AS T1 INNER JOIN AddressType AS T2 ON T1.AddressTypeID = T2.AddressTypeID INNER JOIN Address AS T3 ON T1.AddressID = T3.AddressID WHERE T1.BusinessEntityID = ? AND T2.Name = ?", (business_entity_id, address_type))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"City": result[0], "AddressLine1": result[1]}

# Endpoint to get the product names for a specific special offer ID
@app.get("/v1/works_cycles/product_names_by_special_offer", operation_id="get_product_names_by_special_offer", summary="Retrieves the names of products associated with a given special offer. The operation uses the provided special offer ID to look up the corresponding products and returns their names.")
async def get_product_names_by_special_offer(special_offer_id: int = Query(..., description="Special offer ID")):
    cursor.execute("SELECT T2.Name FROM SpecialOfferProduct AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.SpecialOfferID = ?", (special_offer_id,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get the credit card ID for a specific person
@app.get("/v1/works_cycles/credit_card_id_by_person_name", operation_id="get_credit_card_id_by_person_name", summary="Retrieves the unique credit card identifier associated with a person, based on their full name. The operation requires the first, middle, and last names of the person to accurately identify the corresponding credit card.")
async def get_credit_card_id_by_person_name(first_name: str = Query(..., description="First name of the person"), middle_name: str = Query(..., description="Middle name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T3.CreditCardID FROM Person AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN CreditCard AS T3 ON T2.CreditCardID = T3.CreditCardID WHERE T1.FirstName = ? AND T1.MiddleName = ? AND T1.LastName = ?", (first_name, middle_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"CreditCardID": []}
    return {"CreditCardID": result[0]}

# Endpoint to get the sales reason names for a specific sales order ID
@app.get("/v1/works_cycles/sales_reason_names_by_order_id", operation_id="get_sales_reason_names_by_order_id", summary="Retrieve the names of sales reasons associated with a specific sales order. The operation requires the sales order ID as input and returns the corresponding sales reason names.")
async def get_sales_reason_names_by_order_id(sales_order_id: int = Query(..., description="Sales order ID")):
    cursor.execute("SELECT T2.Name FROM SalesOrderHeaderSalesReason AS T1 INNER JOIN SalesReason AS T2 ON T1.SalesReasonID = T2.SalesReasonID WHERE T1.SalesOrderID = ?", (sales_order_id,))
    result = cursor.fetchall()
    if not result:
        return {"sales_reasons": []}
    return {"sales_reasons": [row[0] for row in result]}

# Endpoint to get the card number for a specific sales order ID
@app.get("/v1/works_cycles/card_number_by_sales_order_id", operation_id="get_card_number_by_sales_order_id", summary="Retrieves the card number linked to a given sales order ID. This operation fetches the card number from the CreditCard table, which is associated with the provided sales order ID in the SalesOrderHeader table.")
async def get_card_number_by_sales_order_id(sales_order_id: int = Query(..., description="Sales order ID")):
    cursor.execute("SELECT T2.CardNumber FROM SalesOrderHeader AS T1 INNER JOIN CreditCard AS T2 ON T1.CreditCardID = T2.CreditCardID WHERE T1.SalesOrderID = ?", (sales_order_id,))
    result = cursor.fetchone()
    if not result:
        return {"card_number": []}
    return {"card_number": result[0]}

# Endpoint to get the business entity IDs for salespersons in a specific territory and country region
@app.get("/v1/works_cycles/business_entity_ids_by_territory_and_country", operation_id="get_business_entity_ids_by_territory_and_country", summary="Retrieve the unique identifiers of business entities associated with salespersons operating in a specified territory and country region. The operation requires the territory name and the country region code as input parameters to filter the results.")
async def get_business_entity_ids_by_territory_and_country(territory_name: str = Query(..., description="Territory name"), country_region_code: str = Query(..., description="Country region code")):
    cursor.execute("SELECT T2.BusinessEntityID FROM SalesTerritory AS T1 INNER JOIN SalesPerson AS T2 ON T1.TerritoryID = T2.TerritoryID WHERE T1.Name = ? AND T1.CountryRegionCode = ?", (territory_name, country_region_code))
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": [row[0] for row in result]}

# Endpoint to get the business entity ID of the highest paid employee with a specific job title
@app.get("/v1/works_cycles/highest_paid_employee_by_job_title", operation_id="get_highest_paid_employee_by_job_title", summary="Retrieves the unique identifier of the employee with the highest pay rate for a given job title. The job title can be specified using wildcard characters for partial matches. The result is determined by joining the Employee and EmployeePayHistory tables and ordering the records by pay rate in descending order.")
async def get_highest_paid_employee_by_job_title(job_title: str = Query(..., description="Job title (use % for wildcard)")):
    cursor.execute("SELECT T1.BusinessEntityID FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.JobTitle LIKE ? ORDER BY T2.Rate DESC LIMIT 1", (job_title,))
    result = cursor.fetchone()
    if not result:
        return {"business_entity_id": []}
    return {"business_entity_id": result[0]}

# Endpoint to get the names of salespersons in a specific territory
@app.get("/v1/works_cycles/salesperson_names_by_territory", operation_id="get_salesperson_names_by_territory", summary="Retrieves the full names of salespersons associated with a given territory. The operation filters salespersons based on the provided territory ID and returns their first, middle, and last names.")
async def get_salesperson_names_by_territory(territory_id: int = Query(..., description="Territory ID")):
    cursor.execute("SELECT T2.FirstName, T2.MiddleName, T2.LastName FROM SalesPerson AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.TerritoryID = ?", (territory_id,))
    result = cursor.fetchall()
    if not result:
        return {"salespersons": []}
    return {"salespersons": [{"first_name": row[0], "middle_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get the description of a special offer for a specific product
@app.get("/v1/works_cycles/special_offer_description_by_product", operation_id="get_special_offer_description_by_product", summary="Get the description of a special offer for a specific product")
async def get_special_offer_description_by_product(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T2.Description FROM SpecialOfferProduct AS T1 INNER JOIN SpecialOffer AS T2 ON T1.SpecialOfferID = T2.SpecialOfferID WHERE T1.ProductID = ?", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the average pay rate grouped by gender
@app.get("/v1/works_cycles/average_pay_rate_by_gender", operation_id="get_average_pay_rate_by_gender", summary="Get the average pay rate grouped by gender")
async def get_average_pay_rate_by_gender():
    cursor.execute("SELECT AVG(T2.Rate) FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID GROUP BY T1.Gender")
    result = cursor.fetchall()
    if not result:
        return {"average_pay_rates": []}
    return {"average_pay_rates": [{"gender": row[0], "average_rate": row[1]} for row in result]}

# Endpoint to get the percentage of employees working a specific shift
@app.get("/v1/works_cycles/percentage_employees_by_shift", operation_id="get_percentage_employees_by_shift", summary="Retrieves the percentage of employees working a specific shift by comparing the total number of employees working that shift to the total number of employees. The shift name is used to filter the data.")
async def get_percentage_employees_by_shift(shift_name: str = Query(..., description="Shift name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.Name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.BusinessEntityID) FROM Shift AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.ShiftId = T2.ShiftId", (shift_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of employees based on marital status, birth year, and gender
@app.get("/v1/works_cycles/employee_count_by_marital_status_birth_year_gender", operation_id="get_employee_count_by_marital_status_birth_year_gender", summary="Retrieves the total number of employees who meet the specified marital status, birth year, and gender criteria. The response is based on the provided marital status, birth year (in 'YYYY' format), and gender parameters.")
async def get_employee_count_by_marital_status_birth_year_gender(marital_status: str = Query(..., description="Marital status"), birth_year: str = Query(..., description="Birth year in 'YYYY' format"), gender: str = Query(..., description="Gender")):
    cursor.execute("SELECT COUNT(BusinessEntityID) FROM Employee WHERE MaritalStatus = ? AND STRFTIME('%Y', BirthDate) < ? AND Gender = ?", (marital_status, birth_year, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of the top N most expensive products
@app.get("/v1/works_cycles/top_expensive_products", operation_id="get_top_expensive_products", summary="Retrieves the names of the top N most expensive products, where N is determined by the provided limit parameter. The products are ranked by their list prices in descending order.")
async def get_top_expensive_products(limit: int = Query(..., description="Number of top expensive products to retrieve")):
    cursor.execute("SELECT Name FROM Product ORDER BY listPrice DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get the earliest hire date for a specific job title
@app.get("/v1/works_cycles/earliest_hire_date_by_job_title", operation_id="get_earliest_hire_date_by_job_title", summary="Retrieves the earliest hire date for a given job title from the employee records.")
async def get_earliest_hire_date_by_job_title(job_title: str = Query(..., description="Job title")):
    cursor.execute("SELECT MIN(HireDate) FROM Employee WHERE JobTitle = ?", (job_title,))
    result = cursor.fetchone()
    if not result:
        return {"hire_date": []}
    return {"hire_date": result[0]}

# Endpoint to get the most common job title for employees hired in a specific year
@app.get("/v1/works_cycles/most_common_job_title_by_hire_year", operation_id="get_most_common_job_title", summary="Retrieves the most frequently occurring job title among employees hired in a particular year. The year of hire should be provided in 'YYYY' format.")
async def get_most_common_job_title(hire_year: str = Query(..., description="Year of hire in 'YYYY' format")):
    cursor.execute("SELECT JobTitle FROM Employee WHERE STRFTIME('%Y', HireDate) = ? GROUP BY HireDate ORDER BY COUNT(JobTitle) DESC LIMIT 1", (hire_year,))
    result = cursor.fetchone()
    if not result:
        return {"job_title": []}
    return {"job_title": result[0]}

# Endpoint to get the difference between LastReceiptCost and StandardPrice for the product with the highest StandardPrice
@app.get("/v1/works_cycles/highest_standard_price_difference", operation_id="get_highest_standard_price_difference", summary="Retrieves the price difference between the last receipt cost and the standard price for the product with the highest standard price.")
async def get_highest_standard_price_difference():
    cursor.execute("SELECT LastReceiptCost - StandardPrice FROM ProductVendor ORDER BY StandardPrice DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of departments modified within a specific year range for a specific person
@app.get("/v1/works_cycles/department_count_by_person_and_year_range", operation_id="get_department_count", summary="Retrieve the total number of departments that were modified within a specified year range for a given person. The operation requires the first and last name of the person, as well as the start and end years of the range, in 'YYYY' format. The result is a count of departments that underwent changes during the specified period.")
async def get_department_count(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T3.Name) FROM Person AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID WHERE T1.FirstName = ? AND T1.LastName = ? AND STRFTIME('%Y', T3.ModifiedDate) BETWEEN ? AND ?", (first_name, last_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average age of employees of a specific type in a given year
@app.get("/v1/works_cycles/average_age_by_person_type", operation_id="get_average_age", summary="Retrieves the average age of employees of a specified type, calculated based on the provided year. The age is determined by subtracting the birth year from the given year.")
async def get_average_age(year: int = Query(..., description="Year to calculate the age"), person_type: str = Query(..., description="Type of person (e.g., 'SP')")):
    cursor.execute("SELECT AVG(? - STRFTIME('%Y', T2.BirthDate)) FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.PersonType = ?", (year, person_type))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the department group with the highest number of employees
@app.get("/v1/works_cycles/most_popular_department_group", operation_id="get_most_popular_department_group", summary="Retrieves the department group that has the most employees. This operation returns the name of the department group with the highest employee count, based on the historical data of employee department assignments. The data is derived from the aggregation and ordering of employee records by department group.")
async def get_most_popular_department_group():
    cursor.execute("SELECT T2.GroupName FROM EmployeeDepartmentHistory AS T1 INNER JOIN Department AS T2 ON T1.DepartmentID = T2.DepartmentID GROUP BY T2.GroupName ORDER BY COUNT(T1.BusinessEntityID) LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"group_name": []}
    return {"group_name": result[0]}

# Endpoint to get the age and pay rate of the oldest employee with a specific job title in a given year
@app.get("/v1/works_cycles/oldest_employee_age_and_rate_by_job_title", operation_id="get_oldest_employee_age_and_rate", summary="Retrieves the age and pay rate of the oldest employee with a specified job title in a given year. The age is calculated based on the provided year and the employee's birth date. The pay rate is determined by matching the employee's ID with their pay history.")
async def get_oldest_employee_age_and_rate(year: int = Query(..., description="Year to calculate the age"), job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT ? - STRFTIME('%Y', T1.BirthDate), T2.Rate FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.JobTitle = ? ORDER BY ? - STRFTIME('%Y', T1.BirthDate) DESC LIMIT 1", (year, job_title, year))
    result = cursor.fetchone()
    if not result:
        return {"age": [], "rate": []}
    return {"age": result[0], "rate": result[1]}

# Endpoint to get the vendor with the lowest unit price for a purchase order
@app.get("/v1/works_cycles/lowest_unit_price_vendor", operation_id="get_lowest_unit_price_vendor", summary="Retrieves the vendor with the lowest unit price for a purchase order. This operation identifies the vendor with the most cost-effective unit price by analyzing the purchase order details and corresponding vendor information. The result provides the name of the vendor offering the lowest unit price.")
async def get_lowest_unit_price_vendor():
    cursor.execute("SELECT T1.UnitPrice, T3.Name FROM PurchaseOrderDetail AS T1 INNER JOIN PurchaseOrderHeader AS T2 ON T1.PurchaseOrderID = T2.PurchaseOrderID INNER JOIN Vendor AS T3 ON T2.VendorID = T3.BusinessEntityID ORDER BY T1.UnitPrice LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"unit_price": [], "vendor_name": []}
    return {"unit_price": result[0], "vendor_name": result[1]}

# Endpoint to get the employee with the highest total due and their age at hire
@app.get("/v1/works_cycles/highest_total_due_employee", operation_id="get_highest_total_due_employee", summary="Retrieves the employee who has the highest total due amount in purchase orders and their age at the time of hire. The response includes the employee's ID and the age at hire, calculated as the difference between the hire date and birth date in years.")
async def get_highest_total_due_employee():
    cursor.execute("SELECT T2.BusinessEntityID, STRFTIME('%Y', T2.HireDate) - STRFTIME('%Y', T2.BirthDate) FROM PurchaseOrderHeader AS T1 INNER JOIN Employee AS T2 ON T1.EmployeeID = T2.BusinessEntityID ORDER BY T1.TotalDue DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"business_entity_id": [], "age_at_hire": []}
    return {"business_entity_id": result[0], "age_at_hire": result[1]}

# Endpoint to get the employee with the nth highest pay rate and their details
@app.get("/v1/works_cycles/nth_highest_pay_rate_employee", operation_id="get_nth_highest_pay_rate_employee", summary="Retrieves the employee with the nth highest pay rate, along with their job title and personal details. The nth value is determined by the provided offset, which specifies the number of top-paying employees to skip before returning the next highest-paying employee's information.")
async def get_nth_highest_pay_rate_employee(offset: int = Query(..., description="Offset to get the nth highest pay rate")):
    cursor.execute("SELECT T2.JobTitle, T1.Rate, T3.FirstName, T3.MiddleName, T3.LastName FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Person AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID ORDER BY T1.Rate DESC LIMIT ?, 1", (offset,))
    result = cursor.fetchone()
    if not result:
        return {"job_title": [], "rate": [], "first_name": [], "middle_name": [], "last_name": []}
    return {"job_title": result[0], "rate": result[1], "first_name": result[2], "middle_name": result[3], "last_name": result[4]}

# Endpoint to get the product with the highest rating by a specific reviewer and its price difference
@app.get("/v1/works_cycles/highest_rated_product_by_reviewer", operation_id="get_highest_rated_product_by_reviewer", summary="Retrieves the product with the highest rating given by a specific reviewer, along with the price difference between its list price and standard cost. The reviewer is identified by the provided name.")
async def get_highest_rated_product_by_reviewer(reviewer_name: str = Query(..., description="Name of the reviewer")):
    cursor.execute("SELECT T1.listPrice - T1.StandardCost, T1.Name FROM Product AS T1 INNER JOIN ProductReview AS T2 ON T1.ProductID = T2.ProductID WHERE T2.ReviewerName = ? ORDER BY T2.Rating DESC LIMIT 1", (reviewer_name,))
    result = cursor.fetchone()
    if not result:
        return {"price_difference": [], "product_name": []}
    return {"price_difference": result[0], "product_name": result[1]}

# Endpoint to get employee pay rates ordered by hire date
@app.get("/v1/works_cycles/employee_pay_rates", operation_id="get_employee_pay_rates", summary="Retrieves a paginated list of employee pay rates, ordered by their hire date. The 'offset' parameter determines the starting point of the data set, while the 'limit' parameter specifies the maximum number of records to return. This operation is useful for viewing employee pay rates in a controlled and manageable manner.")
async def get_employee_pay_rates(offset: int = Query(..., description="Offset for pagination"), limit: int = Query(..., description="Limit for pagination")):
    cursor.execute("SELECT T1.Rate FROM EmployeePayHistory AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Person AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID ORDER BY T2.HireDate ASC LIMIT ?, ?", (offset, limit))
    result = cursor.fetchall()
    if not result:
        return {"rates": []}
    return {"rates": [row[0] for row in result]}

# Endpoint to get vendor names based on order quantity range
@app.get("/v1/works_cycles/vendor_names_by_order_quantity", operation_id="get_vendor_names_by_order_quantity", summary="Retrieves a list of vendor names that have a specified order quantity range. The results are sorted by the difference between the last receipt cost and the standard price in descending order. Pagination is supported through offset and limit parameters.")
async def get_vendor_names_by_order_quantity(min_order_qty: int = Query(..., description="Minimum order quantity"), max_order_qty: int = Query(..., description="Maximum order quantity"), offset: int = Query(..., description="Offset for pagination"), limit: int = Query(..., description="Limit for pagination")):
    cursor.execute("SELECT T2.Name FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.MaxOrderQty BETWEEN ? AND ? ORDER BY T1.LastReceiptCost - T1.StandardPrice DESC LIMIT ?, ?", (min_order_qty, max_order_qty, offset, limit))
    result = cursor.fetchall()
    if not result:
        return {"vendor_names": []}
    return {"vendor_names": [row[0] for row in result]}

# Endpoint to get the years since hire date for documents with a specific status
@app.get("/v1/works_cycles/years_since_hire_date", operation_id="get_years_since_hire_date", summary="Retrieves the number of years since the hire date of the employee who owns a document, given a specific document status. The status of the document is a required input parameter.")
async def get_years_since_hire_date(status: int = Query(..., description="Status of the document")):
    cursor.execute("SELECT 2011 - STRFTIME('%Y', T2.HireDate) FROM Document AS T1 INNER JOIN Employee AS T2 ON T1.Owner = T2.BusinessEntityID WHERE T1.Status = ?", (status,))
    result = cursor.fetchall()
    if not result:
        return {"years_since_hire": []}
    return {"years_since_hire": [row[0] for row in result]}

# Endpoint to get days to manufacture for products with a specific rating and class
@app.get("/v1/works_cycles/days_to_manufacture", operation_id="get_days_to_manufacture", summary="Retrieves the manufacturing duration for products that meet the specified rating and class criteria. The results are sorted by rating and limited to the specified number of records for pagination purposes.")
async def get_days_to_manufacture(rating: int = Query(..., description="Rating of the product"), product_class: str = Query(..., description="Class of the product"), limit: int = Query(..., description="Limit for pagination")):
    cursor.execute("SELECT T1.DaysToManufacture FROM Product AS T1 INNER JOIN ProductReview AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Rating = ? AND T1.Class = ? ORDER BY T2.Rating LIMIT ?", (rating, product_class, limit))
    result = cursor.fetchall()
    if not result:
        return {"days_to_manufacture": []}
    return {"days_to_manufacture": [row[0] for row in result]}

# Endpoint to get the average pay rate for employees of a specific gender
@app.get("/v1/works_cycles/average_pay_rate", operation_id="get_average_pay_rate", summary="Retrieves the average pay rate for employees of a specified gender. The calculation is based on the pay history of employees whose gender matches the provided input. This endpoint provides a statistical overview of the average pay rate for a particular gender.")
async def get_average_pay_rate(gender: str = Query(..., description="Gender of the employee")):
    cursor.execute("SELECT AVG(T2.Rate) FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.Gender = ?", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"average_rate": []}
    return {"average_rate": result[0]}

# Endpoint to get distinct vendor names based on product criteria
@app.get("/v1/works_cycles/distinct_vendor_names", operation_id="get_distinct_vendor_names", summary="Retrieve a unique list of vendor names that match the specified product make flag, style, and subcategory name. This operation filters products based on the provided parameters and returns the distinct names of vendors associated with those products.")
async def get_distinct_vendor_names(make_flag: int = Query(..., description="Make flag of the product"), style: str = Query(..., description="Style of the product"), subcategory_name: str = Query(..., description="Name of the product subcategory")):
    cursor.execute("SELECT DISTINCT T4.Name FROM Product AS T1 INNER JOIN ProductVendor AS T2 ON T1.ProductID = T2.ProductID INNER JOIN ProductSubcategory AS T3 ON T1.ProductSubcategoryID = T3.ProductSubcategoryID INNER JOIN Vendor AS T4 ON T2.BusinessEntityID = T4.BusinessEntityID WHERE T1.MakeFlag = ? AND T1.Style = ? AND T3.Name = ?", (make_flag, style, subcategory_name))
    result = cursor.fetchall()
    if not result:
        return {"vendor_names": []}
    return {"vendor_names": [row[0] for row in result]}

# Endpoint to get pay frequency for employees with a specific job title
@app.get("/v1/works_cycles/pay_frequency", operation_id="get_pay_frequency", summary="Retrieves the pay frequency for employees with a specified job title, ordered by their hire date. The response is paginated, with the number of records returned determined by the provided limit parameter.")
async def get_pay_frequency(job_title: str = Query(..., description="Job title of the employee"), limit: int = Query(..., description="Limit for pagination")):
    cursor.execute("SELECT T2.PayFrequency FROM Employee AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.JobTitle = ? ORDER BY T1.HireDate LIMIT ?", (job_title, limit))
    result = cursor.fetchall()
    if not result:
        return {"pay_frequency": []}
    return {"pay_frequency": [row[0] for row in result]}

# Endpoint to get product details based on shopping cart items
@app.get("/v1/works_cycles/product_details", operation_id="get_product_details", summary="Retrieves detailed information about products in a shopping cart, sorted by the total quantity of each product. The response includes the product's class, product line, and list price. The limit parameter can be used to control the number of products returned in the response.")
async def get_product_details(limit: int = Query(..., description="Limit for pagination")):
    cursor.execute("SELECT T2.Class, T2.ProductLine, T2.listPrice FROM ShoppingCartItem AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID GROUP BY T1.ProductID ORDER BY SUM(Quantity) LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"product_details": []}
    return {"product_details": [{"class": row[0], "product_line": row[1], "list_price": row[2]} for row in result]}

# Endpoint to get the employee with the longest tenure
@app.get("/v1/works_cycles/employee_longest_tenure", operation_id="get_employee_longest_tenure", summary="Get the employee with the longest tenure")
async def get_employee_longest_tenure(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.FirstName, T2.MiddleName, T2.LastName FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID ORDER BY STRFTIME('%Y', T1.HireDate) - STRFTIME('%Y', T1.BirthDate) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the product with the highest total profit margin
@app.get("/v1/works_cycles/highest_profit_margin_product", operation_id="get_highest_profit_margin_product", summary="Retrieves the product with the highest total profit margin, calculated as the difference between the list price and standard cost, multiplied by the total quantity sold. The number of results can be limited by specifying the 'limit' parameter.")
async def get_highest_profit_margin_product(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT (T2.listPrice - T2.StandardCost) * SUM(T1.Quantity), T2.Name FROM ShoppingCartItem AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID GROUP BY T1.ProductID, T2.Name, T2.listPrice, T2.StandardCost, T1.Quantity ORDER BY SUM(T1.Quantity) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the vendor with the highest outstanding order quantity
@app.get("/v1/works_cycles/highest_outstanding_order_quantity", operation_id="get_highest_outstanding_order_quantity", summary="Retrieves the vendor with the highest outstanding order quantity, based on the difference between the total order quantity and the received quantity. The operation returns the top result(s) as determined by the provided limit parameter.")
async def get_highest_outstanding_order_quantity(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.OrderQty - T2.ReceivedQty, VendorID FROM PurchaseOrderHeader AS T1 INNER JOIN PurchaseOrderDetail AS T2 ON T1.PurchaseOrderID = T2.PurchaseOrderID ORDER BY T2.OrderQty - T2.ReceivedQty DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"vendors": []}
    return {"vendors": result}

# Endpoint to get vendor details for a specific product
@app.get("/v1/works_cycles/vendor_details_by_product", operation_id="get_vendor_details_by_product", summary="Retrieves vendor details for a specific product, including average lead time, cost difference, and address information. The product is identified by its unique ID.")
async def get_vendor_details_by_product(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T1.AverageLeadTime, T1.LastReceiptCost - T1.StandardPrice, T4.AddressLine1, T4.AddressLine2, T4.City, T4.PostalCode FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN BusinessEntityAddress AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID INNER JOIN Address AS T4 ON T3.AddressID = T4.AddressID WHERE T1.ProductID = ?", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"vendor_details": []}
    return {"vendor_details": result}

# Endpoint to get the count of salespersons with a specific bonus
@app.get("/v1/works_cycles/count_salespersons_by_bonus", operation_id="get_count_salespersons_by_bonus", summary="Retrieves the total number of salespersons who have a specified bonus amount. The bonus parameter is used to filter the salespersons and calculate the count.")
async def get_count_salespersons_by_bonus(bonus: float = Query(..., description="Bonus amount")):
    cursor.execute("SELECT COUNT(BusinessEntityID) FROM SalesPerson WHERE Bonus = ?", (bonus,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of sales tax rates with a specific name pattern
@app.get("/v1/works_cycles/count_sales_tax_rates_by_name", operation_id="get_count_sales_tax_rates_by_name", summary="Get the count of sales tax rates with a specific name pattern")
async def get_count_sales_tax_rates_by_name(name_pattern: str = Query(..., description="Name pattern to match (use % for wildcard)")):
    cursor.execute("SELECT COUNT(SalesTaxRateID) FROM SalesTaxRate WHERE Name LIKE ?", (name_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest actual cost for a specific transaction type
@app.get("/v1/works_cycles/highest_actual_cost_by_transaction_type", operation_id="get_highest_actual_cost_by_transaction_type", summary="Retrieves the highest actual cost associated with a specified transaction type, allowing the user to limit the number of results returned.")
async def get_highest_actual_cost_by_transaction_type(transaction_type: str = Query(..., description="Transaction type"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT ActualCost FROM TransactionHistory WHERE TransactionType = ? ORDER BY ActualCost DESC LIMIT ?", (transaction_type, limit))
    result = cursor.fetchall()
    if not result:
        return {"actual_costs": []}
    return {"actual_costs": result}

# Endpoint to get the status of the sales order with the highest freight cost
@app.get("/v1/works_cycles/sales_order_status_by_highest_freight", operation_id="get_sales_order_status_by_highest_freight", summary="Retrieves the status of the sales order with the highest freight cost, allowing the user to specify the number of results to return. The operation returns the status of the sales order(s) with the highest freight cost, sorted in descending order.")
async def get_sales_order_status_by_highest_freight(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Status FROM SalesOrderHeader ORDER BY Freight DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"statuses": []}
    return {"statuses": result}

# Endpoint to get the count of products with a specific on-order quantity
@app.get("/v1/works_cycles/count_products_by_on_order_quantity", operation_id="get_count_products_by_on_order_quantity", summary="Retrieves the total number of products that have a specified quantity currently on order. The count is determined by filtering the products based on the provided on-order quantity.")
async def get_count_products_by_on_order_quantity(on_order_qty: int = Query(..., description="On-order quantity")):
    cursor.execute("SELECT COUNT(ProductID) FROM ProductVendor WHERE OnOrderQty = ?", (on_order_qty,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference between LastReceiptCost and StandardPrice for the top N products
@app.get("/v1/works_cycles/top_n_cost_difference", operation_id="get_top_n_cost_difference", summary="Retrieves the cost difference between the last receipt cost and the standard price for the top N products, sorted in descending order. The 'limit' parameter determines the number of top results to return.")
async def get_top_n_cost_difference(limit: int = Query(..., description="Number of top results to return")):
    cursor.execute("SELECT LastReceiptCost - StandardPrice FROM ProductVendor ORDER BY LastReceiptCost - StandardPrice DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"differences": []}
    return {"differences": [row[0] for row in result]}

# Endpoint to get the reviewer names with the highest rating
@app.get("/v1/works_cycles/top_reviewer_names", operation_id="get_top_reviewer_names", summary="Retrieves the names of reviewers who have received the highest ratings, up to a specified limit. The limit parameter determines the number of top-rated reviewers to return.")
async def get_top_reviewer_names(limit: int = Query(..., description="Number of top ratings to consider")):
    cursor.execute("SELECT ReviewerName FROM ProductReview WHERE Rating = ( SELECT Rating FROM ProductReview ORDER BY Rating DESC LIMIT ? )", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"reviewer_names": []}
    return {"reviewer_names": [row[0] for row in result]}

# Endpoint to get the product name with the most reviews
@app.get("/v1/works_cycles/top_product_by_reviews", operation_id="get_top_product_by_reviews", summary="Retrieves the names of the top products with the most reviews, sorted in descending order. The number of products returned can be specified using the provided limit parameter.")
async def get_top_product_by_reviews(limit: int = Query(..., description="Number of top products to return")):
    cursor.execute("SELECT T2.Name FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID GROUP BY T1.ProductID ORDER BY COUNT(T1.ProductReviewID) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the count of products with specific make flag and rating
@app.get("/v1/works_cycles/product_count_make_flag_rating", operation_id="get_product_count_make_flag_rating", summary="Retrieves the total count of products that match a specified make flag and rating. The make flag indicates whether the product is made in-house or outsourced, and the rating represents the average user rating for the product. This operation does not return individual product details, but rather a single count value.")
async def get_product_count_make_flag_rating(make_flag: int = Query(..., description="Make flag value (0 or 1)"), rating: int = Query(..., description="Rating value")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.MakeFlag = ? AND T1.Rating != ?", (make_flag, rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common comment for finished goods
@app.get("/v1/works_cycles/most_common_comment_finished_goods", operation_id="get_most_common_comment_finished_goods", summary="Retrieves the most frequently occurring comments for finished goods products. The operation filters products based on the provided finished goods flag value and returns the top comments as specified by the limit parameter. The comments are ranked by their frequency of occurrence.")
async def get_most_common_comment_finished_goods(finished_goods_flag: int = Query(..., description="Finished goods flag value (0 or 1)"), limit: int = Query(..., description="Number of top comments to return")):
    cursor.execute("SELECT T1.Comments FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.FinishedGoodsFlag = ? GROUP BY T1.Comments ORDER BY COUNT(T1.ProductReviewID) DESC LIMIT ?", (finished_goods_flag, limit))
    result = cursor.fetchall()
    if not result:
        return {"comments": []}
    return {"comments": [row[0] for row in result]}

# Endpoint to get product names based on finished goods flag, make flag, and comment count
@app.get("/v1/works_cycles/product_names_finished_make_comment_count", operation_id="get_product_names_finished_make_comment_count", summary="Retrieves the names of products that meet the specified criteria for finished goods, make, and comment count. The operation filters products based on the provided finished goods flag and make flag values, and returns the names of products with a comment count greater than the specified minimum. The results are ordered by the number of comments in descending order.")
async def get_product_names_finished_make_comment_count(finished_goods_flag: int = Query(..., description="Finished goods flag value (0 or 1)"), make_flag: int = Query(..., description="Make flag value (0 or 1)"), comment_count: int = Query(..., description="Minimum number of comments")):
    cursor.execute("SELECT T2.Name FROM ProductReview AS T1 INNER JOIN Product AS T2 USING (ProductID) WHERE T2.FinishedGoodsFlag = ? AND T2.MakeFlag = ? GROUP BY T2.Name ORDER BY COUNT(T1.COMMENTS) > ?", (finished_goods_flag, make_flag, comment_count))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get product names based on finished goods flag and comment count
@app.get("/v1/works_cycles/product_names_finished_comment_count", operation_id="get_product_names_finished_comment_count", summary="Retrieves the names of products that meet the specified finished goods flag and have a comment count greater than the provided minimum. The results are ordered by the number of comments in descending order.")
async def get_product_names_finished_comment_count(finished_goods_flag: int = Query(..., description="Finished goods flag value (0 or 1)"), comment_count: int = Query(..., description="Minimum number of comments")):
    cursor.execute("SELECT T2.NAME FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.FinishedGoodsFlag = ? GROUP BY T2.NAME ORDER BY COUNT(T1.comments) > ?", (finished_goods_flag, comment_count))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get reviewer names based on product class, style, and rating
@app.get("/v1/works_cycles/reviewer_names_class_style_rating", operation_id="get_reviewer_names_class_style_rating", summary="Retrieves the names of reviewers who have reviewed products of a specific class, style, and rating. The class, style, and rating are provided as input parameters, allowing for a targeted search of reviewer names.")
async def get_reviewer_names_class_style_rating(product_class: str = Query(..., description="Product class"), product_style: str = Query(..., description="Product style"), rating: int = Query(..., description="Rating value")):
    cursor.execute("SELECT T1.ReviewerName FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Class = ? AND T2.Style = ? AND T1.Rating = ?", (product_class, product_style, rating))
    result = cursor.fetchall()
    if not result:
        return {"reviewer_names": []}
    return {"reviewer_names": [row[0] for row in result]}

# Endpoint to get the top product class by review count
@app.get("/v1/works_cycles/top_product_class_by_reviews", operation_id="get_top_product_class_by_reviews", summary="Retrieves the top product classes ranked by the number of reviews they have received. The operation returns the specified number of classes with the highest review count. The input parameter determines the number of top classes to include in the response.")
async def get_top_product_class_by_reviews(limit: int = Query(..., description="Number of top classes to return")):
    cursor.execute("SELECT T2.Class FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID GROUP BY T2.Class ORDER BY COUNT(T1.ProductReviewID) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"product_classes": []}
    return {"product_classes": [row[0] for row in result]}

# Endpoint to get email addresses based on product class
@app.get("/v1/works_cycles/email_addresses_by_product_class", operation_id="get_email_addresses_by_product_class", summary="Retrieves email addresses of users who have reviewed products belonging to a specific class. The operation filters the product reviews based on the provided product class and returns the corresponding email addresses.")
async def get_email_addresses_by_product_class(product_class: str = Query(..., description="Product class")):
    cursor.execute("SELECT T1.EmailAddress FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Class = ?", (product_class,))
    result = cursor.fetchall()
    if not result:
        return {"email_addresses": []}
    return {"email_addresses": [row[0] for row in result]}

# Endpoint to get product names based on list price count
@app.get("/v1/works_cycles/product_names_by_list_price_count", operation_id="get_product_names_by_list_price_count", summary="Retrieves product names based on the specified count of list prices. The operation groups products by name and orders them based on the count of their list prices, returning those with a count greater than the provided input parameter.")
async def get_product_names_by_list_price_count(list_price_count: int = Query(..., description="Count of list prices")):
    cursor.execute("SELECT T2.Name FROM ProductlistPriceHistory AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID GROUP BY T2.Name ORDER BY COUNT(T1.listPrice) > ?", (list_price_count,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get product description based on product name
@app.get("/v1/works_cycles/product_description_by_name", operation_id="get_product_description_by_name", summary="Retrieves the description of a product based on the provided product name. This operation fetches the product description from the ProductDescription table, which is associated with the product in the Product table using the product's name as the key.")
async def get_product_description_by_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T1.Description FROM ProductDescription AS T1 INNER JOIN Product AS T2 WHERE T2.Name = ? AND T1.productDescriptionID = T2.ProductID", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the highest standard price of a product
@app.get("/v1/works_cycles/highest_standard_price_by_product_name", operation_id="get_highest_standard_price_by_product_name", summary="Retrieves the maximum standard price for a product, identified by its name. The operation filters the product vendor records based on the provided product name, sorts them in descending order by standard price, and returns the highest price.")
async def get_highest_standard_price_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T1.StandardPrice FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ? ORDER BY T1.StandardPrice DESC LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"standard_price": []}
    return {"standard_price": result[0]}

# Endpoint to get the most common standard price of a product
@app.get("/v1/works_cycles/most_common_standard_price_by_product_name", operation_id="get_most_common_standard_price_by_product_name", summary="Retrieves the most frequently occurring standard price for a specified product. The product is identified by its name. The operation returns the standard price that appears most frequently in the product vendor records associated with the product.")
async def get_most_common_standard_price_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T1.StandardPrice FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ? GROUP BY T1.StandardPrice ORDER BY COUNT(T1.StandardPrice) DESC LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"standard_price": []}
    return {"standard_price": result[0]}

# Endpoint to get the count of distinct vendors for a product with a specific credit rating
@app.get("/v1/works_cycles/count_distinct_vendors_by_product_name_and_credit_rating", operation_id="get_count_distinct_vendors_by_product_name_and_credit_rating", summary="Retrieves the number of unique vendors associated with a specific product, filtered by the product's name and the vendors' credit rating.")
async def get_count_distinct_vendors_by_product_name_and_credit_rating(product_name: str = Query(..., description="Name of the product"), credit_rating: int = Query(..., description="Credit rating of the vendor")):
    cursor.execute("SELECT COUNT(DISTINCT T3.Name) FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Vendor AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T2.Name = ? AND T3.CreditRating = ?", (product_name, credit_rating))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the purchasing web service URL of vendors for a product
@app.get("/v1/works_cycles/purchasing_web_service_url_by_product_name", operation_id="get_purchasing_web_service_url_by_product_name", summary="Retrieves the purchasing web service URLs of vendors associated with a specific product, based on the provided product name. This operation allows you to find the relevant vendor URLs for a given product, facilitating the purchasing process.")
async def get_purchasing_web_service_url_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T3.PurchasingWebServiceURL FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Vendor AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T2.Name = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"urls": []}
    return {"urls": [row[0] for row in result]}

# Endpoint to get the vendor name with the lowest standard price for a product
@app.get("/v1/works_cycles/vendor_name_by_product_name_and_lowest_standard_price", operation_id="get_vendor_name_by_product_name_and_lowest_standard_price", summary="Retrieves the name of the vendor offering the lowest standard price for a specific product, based on the provided product name.")
async def get_vendor_name_by_product_name_and_lowest_standard_price(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T3.Name FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Vendor AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T2.Name = ? ORDER BY T1.StandardPrice LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"vendor_name": []}
    return {"vendor_name": result[0]}

# Endpoint to get the count of products from preferred vendors with a specific class
@app.get("/v1/works_cycles/count_products_by_preferred_vendor_and_class", operation_id="get_count_products_by_preferred_vendor_and_class", summary="Retrieves the total count of products from preferred vendors that belong to a specific class. The operation requires the preferred vendor status and the product class as input parameters to filter the results.")
async def get_count_products_by_preferred_vendor_and_class(preferred_vendor_status: int = Query(..., description="Preferred vendor status"), product_class: str = Query(..., description="Class of the product")):
    cursor.execute("SELECT COUNT(T2.Name) FROM ProductVendor AS T1 INNER JOIN Product AS T2 USING (ProductID) INNER JOIN Vendor AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T3.PreferredVendorStatus = ? AND T2.Class = ?", (preferred_vendor_status, product_class))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the list price history of a product
@app.get("/v1/works_cycles/list_price_history_by_product_name", operation_id="get_list_price_history_by_product_name", summary="Retrieves the historical list prices for a specific product, identified by its name. The response includes a chronological record of the product's list prices.")
async def get_list_price_history_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.listPrice FROM Product AS T1 INNER JOIN ProductlistPriceHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"list_prices": []}
    return {"list_prices": [row[0] for row in result]}

# Endpoint to get the count of products in a specific product line with more than a certain number of names
@app.get("/v1/works_cycles/count_products_in_product_line_with_name_count", operation_id="get_count_products_in_product_line_with_name_count", summary="Get the count of products in a specific product line with more than a certain number of names")
async def get_count_products_in_product_line_with_name_count(product_line: str = Query(..., description="Product line"), name_count: int = Query(..., description="Count of names")):
    cursor.execute("SELECT SUM(CASE WHEN T1.ProductLine = ? THEN 1 ELSE 0 END) FROM Product AS T1 INNER JOIN ProductVendor AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Vendor AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID GROUP BY T1.ProductID HAVING COUNT(T1.Name) > ?", (product_line, name_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of product reviews for a specific product line with more than a specified number of reviews
@app.get("/v1/works_cycles/sum_product_reviews_by_product_line", operation_id="get_sum_product_reviews", summary="Retrieves the total count of product reviews for a given product line, considering only products that have received more than a specified minimum number of reviews.")
async def get_sum_product_reviews(product_line: str = Query(..., description="Product line"), min_reviews: int = Query(..., description="Minimum number of reviews")):
    cursor.execute("SELECT SUM(CASE WHEN T2.ProductLine = ? THEN 1 ELSE 0 END) FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID GROUP BY T1.ProductID HAVING COUNT(T1.ProductReviewID) > ?", (product_line, min_reviews))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the email address of the reviewer for a specific product name ordered by rating
@app.get("/v1/works_cycles/email_address_by_product_name", operation_id="get_email_address", summary="Retrieves the email address of the reviewer who has given the highest rating to the specified product. The product is identified by its name.")
async def get_email_address(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT T1.EmailAddress FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ? ORDER BY T1.Rating LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"email_address": []}
    return {"email_address": result[0]}

# Endpoint to get the count of products with a specific on-order quantity or null
@app.get("/v1/works_cycles/count_products_by_on_order_qty", operation_id="get_count_products", summary="Retrieves the total count of products that have a specified on-order quantity or no on-order quantity at all. The response is based on the provided on-order quantity or a flag indicating whether to include products with null on-order quantity.")
async def get_count_products(on_order_qty: int = Query(..., description="On order quantity"), null_on_order_qty: bool = Query(..., description="Include null on order quantity")):
    if null_on_order_qty:
        cursor.execute("SELECT COUNT(T2.ProductID) FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.OnOrderQty IS NULL OR T1.OnOrderQty = ?", (on_order_qty,))
    else:
        cursor.execute("SELECT COUNT(T2.ProductID) FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.OnOrderQty = ?", (on_order_qty,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of products with a specific make flag and on-order quantity or null
@app.get("/v1/works_cycles/product_names_by_make_flag_and_on_order_qty", operation_id="get_product_names", summary="Retrieves the names of products that match a specified make flag and on-order quantity, or include those with a null on-order quantity. The make flag is a binary value indicating whether the product is made in-house or not. The on-order quantity represents the number of units currently on order. If the null_on_order_qty parameter is set to true, the operation will also return products with a null on-order quantity.")
async def get_product_names(make_flag: int = Query(..., description="Make flag"), on_order_qty: int = Query(..., description="On order quantity"), null_on_order_qty: bool = Query(..., description="Include null on order quantity")):
    if null_on_order_qty:
        cursor.execute("SELECT T2.Name FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.MakeFlag = ? AND (T1.OnOrderQty IS NULL OR T1.OnOrderQty = ?)", (make_flag, on_order_qty))
    else:
        cursor.execute("SELECT T2.Name FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.MakeFlag = ? AND T1.OnOrderQty = ?", (make_flag, on_order_qty))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the sum of product reviews for a specific product line and finished goods flag
@app.get("/v1/works_cycles/sum_product_reviews_by_product_line_and_finished_goods_flag", operation_id="get_sum_product_reviews_finished_goods", summary="Retrieves the total number of reviews for the product line with the highest review count, filtered by a specified finished goods flag.")
async def get_sum_product_reviews_finished_goods(product_line: str = Query(..., description="Product line"), finished_goods_flag: int = Query(..., description="Finished goods flag")):
    cursor.execute("SELECT SUM(CASE WHEN T2.ProductLine = ? THEN 1 ELSE 0 END) FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.FinishedGoodsFlag = ? GROUP BY T1.ProductID ORDER BY COUNT(T1.ProductReviewID) DESC LIMIT 1", (product_line, finished_goods_flag))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the average standard price for a specific product name
@app.get("/v1/works_cycles/average_standard_price_by_product_name", operation_id="get_average_standard_price", summary="Retrieves the average standard price for a specific product by calculating the sum of all standard prices for the product and dividing it by the count of business entities associated with the product.")
async def get_average_standard_price(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT SUM(T1.StandardPrice) / COUNT(T1.BusinessEntityID) FROM ProductVendor AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the product name with the highest average rating for a specific product line
@app.get("/v1/works_cycles/highest_average_rating_product_by_product_line", operation_id="get_highest_average_rating_product", summary="Retrieves the name of the product with the highest average rating from a specified product line. The product line is determined by the provided input parameter. The operation calculates the average rating for each product within the product line and returns the product with the highest average rating.")
async def get_highest_average_rating_product(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT T2.Name FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.ProductLine = ? GROUP BY T2.Name ORDER BY CAST(SUM(T1.Rating) AS REAL) / COUNT(T1.ProductID) DESC LIMIT 1", (product_line,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the product name with the highest total rating for a specific make flag
@app.get("/v1/works_cycles/highest_total_rating_product_by_make_flag", operation_id="get_highest_total_rating_product", summary="Retrieves the name of the product with the highest cumulative rating for a given make flag. The make flag is used to filter the products, and the product with the highest total rating is determined by summing up individual ratings and sorting in descending order.")
async def get_highest_total_rating_product(make_flag: int = Query(..., description="Make flag")):
    cursor.execute("SELECT T2.Name FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T2.MakeFlag = ? GROUP BY T2.Name ORDER BY SUM(T1.Rating) DESC LIMIT 1", (make_flag,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the first and last names of persons of a specific type
@app.get("/v1/works_cycles/person_names_by_type", operation_id="get_person_names", summary="Retrieves the first and last names of individuals categorized under a specific type, sorted by their business entity ID. The operation allows for filtering by person type, enabling targeted retrieval of relevant records.")
async def get_person_names(person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT FirstName, LastName FROM Person WHERE PersonType = ? ORDER BY BusinessEntityID", (person_type,))
    result = cursor.fetchall()
    if not result:
        return {"person_names": []}
    return {"person_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the full name of persons with a specific last name and person type
@app.get("/v1/works_cycles/full_name_by_last_name_and_type", operation_id="get_full_name", summary="Retrieves the full names of individuals who share a specified last name and belong to a particular person type. The full name includes the first, middle, and last names.")
async def get_full_name(last_name: str = Query(..., description="Last name"), person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT FirstName, MiddleName, LastName FROM Person WHERE LastName = ? AND PersonType = ?", (last_name, person_type))
    result = cursor.fetchall()
    if not result:
        return {"full_names": []}
    return {"full_names": [{"first_name": row[0], "middle_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get the first name, last name, and hire date of employees whose department history end date is not null
@app.get("/v1/works_cycles/employee_department_history_end_date", operation_id="get_employee_department_history_end_date", summary="Retrieves the first name, last name, and hire date of employees who have a recorded department history end date. This operation provides a list of employees who have left their respective departments, along with their personal details and hire dates.")
async def get_employee_department_history_end_date():
    cursor.execute("SELECT T1.FirstName, T1.LastName, T2.HireDate FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN EmployeeDepartmentHistory AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T3.EndDate IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the first name and last name of employees hired after a specific year and are salaried
@app.get("/v1/works_cycles/employee_hired_after_year_salaried", operation_id="get_employee_hired_after_year_salaried", summary="Retrieves the full names of employees who were hired after a specified year and are salaried. The year of hire is provided in 'YYYY' format, and the salaried flag indicates whether the employee is salaried (1) or non-salaried (0).")
async def get_employee_hired_after_year_salaried(hire_year: str = Query(..., description="Year of hire in 'YYYY' format"), salaried_flag: int = Query(..., description="Salaried flag (1 for salaried, 0 for non-salaried)")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE STRFTIME('%Y', T2.HireDate) >= ? AND T2.SalariedFlag = ?", (hire_year, salaried_flag))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the first name and last name of employees with a specific marital status and fewer vacation hours than a specified number
@app.get("/v1/works_cycles/employee_marital_status_vacation_hours", operation_id="get_employee_marital_status_vacation_hours", summary="Retrieves the first and last names of employees who have a specified marital status and fewer vacation hours than a given number. This operation is useful for identifying employees who meet certain criteria based on their marital status and vacation hours.")
async def get_employee_marital_status_vacation_hours(marital_status: str = Query(..., description="Marital status of the employee"), vacation_hours: int = Query(..., description="Maximum vacation hours")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.MaritalStatus = ? AND T2.VacationHours < ?", (marital_status, vacation_hours))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the first name, last name, and age of the oldest employee in a specific shift
@app.get("/v1/works_cycles/oldest_employee_in_shift", operation_id="get_oldest_employee_in_shift", summary="Retrieves the full name and age of the oldest employee working in a specified shift. The age is calculated based on the current year and the employee's birth year.")
async def get_oldest_employee_in_shift(shift_id: int = Query(..., description="Shift ID of the employee")):
    cursor.execute("SELECT T1.FirstName, T1.LastName, STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', BirthDate) FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN EmployeeDepartmentHistory AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T3.ShiftId = ? ORDER BY STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', BirthDate) DESC LIMIT 1", (shift_id,))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get the first name and last name of employees hired in a specific year and working in a specific department
@app.get("/v1/works_cycles/employee_hired_year_department", operation_id="get_employee_hired_year_department", summary="Retrieves the full names of employees who were hired in a specific year and are currently working in a designated department. The year of hire and department name are required as input parameters.")
async def get_employee_hired_year_department(hire_year: str = Query(..., description="Year of hire in 'YYYY' format"), department_name: str = Query(..., description="Name of the department")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN EmployeeDepartmentHistory AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID INNER JOIN Department AS T4 ON T3.DepartmentID = T4.DepartmentID WHERE STRFTIME('%Y', T2.HireDate) = ? AND T4.Name = ?", (hire_year, department_name))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the job title and department name of the most recently hired employee
@app.get("/v1/works_cycles/most_recently_hired_employee", operation_id="get_most_recently_hired_employee", summary="Get the job title and department name of the most recently hired employee")
async def get_most_recently_hired_employee():
    cursor.execute("SELECT T2.JobTitle, T4.Name FROM Person AS T1 INNER JOIN Employee AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN EmployeeDepartmentHistory AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID INNER JOIN Department AS T4 ON T3.DepartmentID = T4.DepartmentID ORDER BY T2.HireDate LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get the average pay rate of employees on the most recent rate change date
@app.get("/v1/works_cycles/average_pay_rate_recent_change", operation_id="get_average_pay_rate_recent_change", summary="Retrieves the average pay rate of employees based on the most recent rate change date. This operation calculates the average pay rate by considering the pay history of employees and their departmental history. The result reflects the average pay rate across all departments on the most recent date when a pay rate change was recorded.")
async def get_average_pay_rate_recent_change():
    cursor.execute("SELECT AVG(T1.Rate) FROM EmployeePayHistory AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID WHERE T1.RateChangeDate = ( SELECT MAX(T1.RateChangeDate) FROM EmployeePayHistory AS T1 INNER JOIN Department AS T2 ON T1.BusinessEntityID = T2.DepartmentID )")
    result = cursor.fetchone()
    if not result:
        return {"average_rate": []}
    return {"average_rate": result[0]}

# Endpoint to get the first name and last name of the highest paid employee excluding a specific job title
@app.get("/v1/works_cycles/highest_paid_employee_excluding_job_title", operation_id="get_highest_paid_employee_excluding_job_title", summary="Retrieves the full name of the highest paid employee, excluding those with a specified job title. The operation filters out employees with the given job title and returns the first and last name of the highest paid remaining employee.")
async def get_highest_paid_employee_excluding_job_title(job_title: str = Query(..., description="Job title to exclude")):
    cursor.execute("SELECT T2.FirstName, T2.LastName FROM EmployeePayHistory AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Employee AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T3.JobTitle NOT LIKE ? ORDER BY T1.Rate DESC LIMIT 1", (job_title,))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get the name of the vendor with the shortest average lead time for a specific product
@app.get("/v1/works_cycles/vendor_shortest_lead_time_product", operation_id="get_vendor_shortest_lead_time_product", summary="Retrieves the name of the vendor with the shortest average lead time for a given product. The product is identified by its unique ID. The operation returns the name of the vendor who can deliver the product in the least amount of time, based on historical data.")
async def get_vendor_shortest_lead_time_product(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T1.Name FROM Vendor AS T1 INNER JOIN ProductVendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.ProductID = ? ORDER BY T2.AverageLeadTime LIMIT 1", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"vendor": []}
    return {"vendor": result[0]}

# Endpoint to get the names of products supplied by a specific vendor
@app.get("/v1/works_cycles/products_supplied_by_vendor", operation_id="get_products_supplied_by_vendor", summary="Retrieves the names of products that a specified vendor supplies. The operation requires the vendor's name as input and returns a list of product names associated with the vendor.")
async def get_products_supplied_by_vendor(vendor_name: str = Query(..., description="Name of the vendor")):
    cursor.execute("SELECT T3.Name FROM Vendor AS T1 INNER JOIN ProductVendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Product AS T3 ON T2.ProductID = T3.ProductID WHERE T1.Name = ?", (vendor_name,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [item[0] for item in result]}

# Endpoint to get the vendor name for a specific product ID
@app.get("/v1/works_cycles/vendor_name_by_product_id", operation_id="get_vendor_name_by_product_id", summary="Retrieves the name of the vendor associated with the specified product ID, prioritizing the vendor with the greatest difference between the last receipt cost and the standard price. The result is limited to a single vendor.")
async def get_vendor_name_by_product_id(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T1.Name FROM Vendor AS T1 INNER JOIN ProductVendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T2.ProductID = ? ORDER BY T2.LastReceiptCost - T2.StandardPrice DESC LIMIT 1", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"vendor_name": []}
    return {"vendor_name": result[0]}

# Endpoint to get the rate and rate variation for a specific employee
@app.get("/v1/works_cycles/employee_rate_variation", operation_id="get_employee_rate_variation", summary="Retrieves the rate and rate variation for a specific employee, identified by their first and last name. The rate variation is calculated as the percentage difference between the maximum and minimum rates for the employee. The endpoint returns the current rate and the rate variation, providing insights into the employee's pay history.")
async def get_employee_rate_variation(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.Rate , (MAX(T2.Rate) - MIN(T2.Rate)) * 100 / MAX(T2.Rate) FROM Person AS T1 INNER JOIN EmployeePayHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"rate_variation": []}
    return {"rate": result[0], "rate_variation": result[1]}

# Endpoint to get the average years of service in a specific department
@app.get("/v1/works_cycles/avg_years_of_service_by_department", operation_id="get_avg_years_of_service_by_department", summary="Retrieves the average number of years that employees have been serving in a specific department. The calculation is based on the current year and the year of hire for each employee in the given department.")
async def get_avg_years_of_service_by_department(department_name: str = Query(..., description="Name of the department")):
    cursor.execute("SELECT AVG(STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.HireDate)) FROM Employee AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID WHERE T3.Name = ?", (department_name,))
    result = cursor.fetchone()
    if not result:
        return {"avg_years_of_service": []}
    return {"avg_years_of_service": result[0]}

# Endpoint to get the average age of employees
@app.get("/v1/works_cycles/avg_employee_age", operation_id="get_avg_employee_age", summary="Retrieves the average age of employees by calculating the difference between the current year and the birth year of each employee.")
async def get_avg_employee_age():
    cursor.execute("SELECT AVG(STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', BirthDate)) FROM Employee")
    result = cursor.fetchone()
    if not result:
        return {"avg_age": []}
    return {"avg_age": result[0]}

# Endpoint to get the percentage of employees with a specific job title
@app.get("/v1/works_cycles/percentage_employees_by_job_title", operation_id="get_percentage_employees_by_job_title", summary="Retrieves the percentage of employees with a specified job title from the total number of employees. The job title is provided as an input parameter.")
async def get_percentage_employees_by_job_title(job_title: str = Query(..., description="Job title of the employees")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN JobTitle = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(BusinessEntityID) FROM Employee", (job_title,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the contact details of employees with a specific job title
@app.get("/v1/works_cycles/employee_contact_details_by_job_title", operation_id="get_employee_contact_details_by_job_title", summary="Get the contact details of employees with a specific job title.")
async def get_employee_contact_details_by_job_title(job_title: str = Query(..., description="Job title of the employees")):
    cursor.execute("SELECT T1.FirstName, T1.LastName, T2.PhoneNumber FROM Person AS T1 INNER JOIN PersonPhone AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Employee AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T3.JobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"contact_details": []}
    return {"contact_details": result}

# Endpoint to get the names of people with a specific credit card type
@app.get("/v1/works_cycles/person_names_by_credit_card_type", operation_id="get_person_names_by_credit_card_type", summary="Retrieves the first and last names of individuals who possess a specific type of credit card. The operation filters the results based on the provided credit card type.")
async def get_person_names_by_credit_card_type(card_type: str = Query(..., description="Type of the credit card")):
    cursor.execute("SELECT T3.FirstName, T3.LastName FROM CreditCard AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.CreditCardID = T2.CreditCardID INNER JOIN Person AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T1.CardType = ?", (card_type,))
    result = cursor.fetchall()
    if not result:
        return {"person_names": []}
    return {"person_names": result}

# Endpoint to get the sales territory name for a specific store
@app.get("/v1/works_cycles/sales_territory_by_store_name", operation_id="get_sales_territory_by_store_name", summary="Retrieves the sales territory name associated with a specific store. The operation requires the store's name as input and returns the corresponding sales territory name by querying the Store, SalesPerson, Person, and SalesTerritory tables.")
async def get_sales_territory_by_store_name(store_name: str = Query(..., description="Name of the store")):
    cursor.execute("SELECT T4.Name FROM Store AS T1 INNER JOIN SalesPerson AS T2 ON T1.SalesPersonID = T2.BusinessEntityID INNER JOIN Person AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID INNER JOIN SalesTerritory AS T4 ON T2.TerritoryID = T4.TerritoryID WHERE T1.Name = ?", (store_name,))
    result = cursor.fetchone()
    if not result:
        return {"sales_territory": []}
    return {"sales_territory": result[0]}

# Endpoint to get the store and person details for a specific sales territory
@app.get("/v1/works_cycles/store_person_details_by_sales_territory", operation_id="get_store_person_details_by_sales_territory", summary="Retrieves the names of stores and the corresponding first and last names of their associated persons for a given sales territory.")
async def get_store_person_details_by_sales_territory(territory_name: str = Query(..., description="Name of the sales territory")):
    cursor.execute("SELECT T3.Name, T4.FirstName, T4.LastName FROM SalesTerritory AS T1 INNER JOIN Customer AS T2 ON T1.TerritoryID = T2.TerritoryID INNER JOIN Store AS T3 ON T2.StoreID = T3.BusinessEntityID INNER JOIN Person AS T4 ON T2.PersonID = T4.BusinessEntityID WHERE T1.Name = ?", (territory_name,))
    result = cursor.fetchall()
    if not result:
        return {"store_person_details": []}
    return {"store_person_details": result}

# Endpoint to get the product and vendor names for products starting to sell in a specific year
@app.get("/v1/works_cycles/product_vendor_by_sell_start_year", operation_id="get_product_vendor_by_sell_start_year", summary="Retrieve the names of products and their respective vendors that began selling in a specified year. The operation filters products based on the year they started selling and returns the product and vendor names.")
async def get_product_vendor_by_sell_start_year(sell_start_year: str = Query(..., description="Year when the product started selling in 'YYYY' format")):
    cursor.execute("SELECT T1.Name, T3.Name FROM Product AS T1 INNER JOIN ProductVendor AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Vendor AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE STRFTIME('%Y', T1.SellStartDate) = ?", (sell_start_year,))
    result = cursor.fetchall()
    if not result:
        return {"product_vendor": []}
    return {"product_vendor": result}

# Endpoint to get the oldest employee by gender and marital status
@app.get("/v1/works_cycles/oldest_employee_by_gender_marital_status", operation_id="get_oldest_employee", summary="Retrieves the details of the oldest employee based on the provided gender and marital status. The response includes the employee's first name, last name, and job title.")
async def get_oldest_employee(gender: str = Query(..., description="Gender of the employee"), marital_status: str = Query(..., description="Marital status of the employee")):
    cursor.execute("SELECT T2.FirstName, T2.LastName, T1.JobTitle FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.Gender = ? AND T1.MaritalStatus = ? ORDER BY T1.BirthDate LIMIT 1", (gender, marital_status))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get the vendor with the shortest lead time for a specific product
@app.get("/v1/works_cycles/shortest_lead_time_vendor", operation_id="get_shortest_lead_time_vendor", summary="Retrieves the name of the vendor with the shortest lead time for a specific product. The product is identified by its unique ID. The lead time is calculated based on the average lead time for the product-vendor pair.")
async def get_shortest_lead_time_vendor(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T2.Name FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.ProductID = ? ORDER BY T1.AverageLeadTime ASC LIMIT 1", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"vendor": []}
    return {"vendor": result[0]}

# Endpoint to get the employee with the most sick leave hours born after a specific year
@app.get("/v1/works_cycles/most_sick_leave_hours_by_birth_year", operation_id="get_most_sick_leave_hours", summary="Retrieves the employee with the highest number of sick leave hours who was born after the specified year. The birth year is provided in 'YYYY' format. The response includes the employee's first and last names.")
async def get_most_sick_leave_hours(birth_year: str = Query(..., description="Birth year in 'YYYY' format")):
    cursor.execute("SELECT T2.FirstName, T2.LastName FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE STRFTIME('%Y', T1.BirthDate) > ? ORDER BY T1.SickLeaveHours LIMIT 1", (birth_year,))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get the youngest employee's age and department name
@app.get("/v1/works_cycles/youngest_employee_age_department", operation_id="get_youngest_employee_age_department", summary="Retrieves the age of the youngest employee and the name of their department. The age is calculated based on the current year and the employee's birth year. The department name is determined by the employee's most recent department assignment.")
async def get_youngest_employee_age_department():
    cursor.execute("SELECT STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.BirthDate) + 1 , T3.Name FROM Employee AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 USING (BusinessEntityID) INNER JOIN Department AS T3 USING (DepartmentID) ORDER BY T1.BirthDate DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get distinct product category IDs
@app.get("/v1/works_cycles/distinct_product_category_ids", operation_id="get_distinct_product_category_ids", summary="Retrieves a unique set of up to three product category identifiers from the product subcategory data.")
async def get_distinct_product_category_ids():
    cursor.execute("SELECT DISTINCT ProductCategoryID FROM ProductSubcategory LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"category_ids": []}
    return {"category_ids": [row[0] for row in result]}

# Endpoint to get the difference between SalesYTD and SalesQuota for a specific salesperson
@app.get("/v1/works_cycles/sales_difference", operation_id="get_sales_difference", summary="Retrieves the difference between the year-to-date sales and the sales quota for a specific salesperson, identified by their Business Entity ID.")
async def get_sales_difference(business_entity_id: int = Query(..., description="Business Entity ID of the salesperson")):
    cursor.execute("SELECT SalesYTD - SalesQuota FROM SalesPerson WHERE BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchone()
    if not result:
        return {"sales_difference": []}
    return {"sales_difference": result[0]}

# Endpoint to get the top 3 salespersons by SalesLastYear
@app.get("/v1/works_cycles/top_salespersons_last_year", operation_id="get_top_salespersons_last_year", summary="Retrieves the top three salespersons based on their sales performance over the past year. The salespersons are ranked in descending order, with the highest-performing salesperson listed first.")
async def get_top_salespersons_last_year():
    cursor.execute("SELECT BusinessEntityID FROM SalesPerson ORDER BY SalesLastYear LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"salespersons": []}
    return {"salespersons": [row[0] for row in result]}

# Endpoint to get top N territories by SalesYTD
@app.get("/v1/works_cycles/top_territories_by_sales", operation_id="get_top_territories_by_sales", summary="Retrieves the top N territories, ranked by their total sales year-to-date. The 'limit' parameter determines the number of top territories to return.")
async def get_top_territories_by_sales(limit: int = Query(..., description="Number of top territories to return")):
    cursor.execute("SELECT TerritoryID FROM SalesTerritory ORDER BY SalesYTD DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"territories": []}
    return {"territories": [row[0] for row in result]}

# Endpoint to get special offer types by category
@app.get("/v1/works_cycles/special_offer_types_by_category", operation_id="get_special_offer_types_by_category", summary="Retrieves the distinct types of special offers available for a specified category. The category is provided as an input parameter.")
async def get_special_offer_types_by_category(category: str = Query(..., description="Category of the special offer")):
    cursor.execute("SELECT Type FROM SpecialOffer WHERE Category = ?", (category,))
    result = cursor.fetchall()
    if not result:
        return {"types": []}
    return {"types": [row[0] for row in result]}

# Endpoint to get the highest discount percentage for a given special offer type
@app.get("/v1/works_cycles/highest_discount_by_type", operation_id="get_highest_discount_by_type", summary="Retrieves the top discount percentages for a specific special offer type, sorted in descending order. The number of discounts returned can be limited by specifying the desired quantity.")
async def get_highest_discount_by_type(type: str = Query(..., description="Type of the special offer"), limit: int = Query(..., description="Number of top discounts to return")):
    cursor.execute("SELECT DiscountPct FROM SpecialOffer WHERE Type = ? ORDER BY DiscountPct DESC LIMIT ?", (type, limit))
    result = cursor.fetchall()
    if not result:
        return {"discounts": []}
    return {"discounts": [row[0] for row in result]}

# Endpoint to get the cost difference for a given product ID
@app.get("/v1/works_cycles/cost_difference_by_product", operation_id="get_cost_difference_by_product", summary="Retrieves the cost difference between planned and actual expenses for a specific product. The product is identified by its unique ID. This endpoint provides a comparison of projected and actual costs, aiding in budgeting and financial analysis.")
async def get_cost_difference_by_product(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT PlannedCost - ActualCost FROM WorkOrderRouting WHERE ProductID = ?", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"cost_differences": []}
    return {"cost_differences": [row[0] for row in result]}

# Endpoint to get the count of bill of materials with a specific BOMLevel and no end date
@app.get("/v1/works_cycles/bom_count_by_level", operation_id="get_bom_count_by_level", summary="Retrieves the total count of active bill of materials that are at a specified level in the hierarchy. This operation is useful for understanding the composition and structure of the bill of materials at a given level.")
async def get_bom_count_by_level(bom_level: int = Query(..., description="BOMLevel")):
    cursor.execute("SELECT COUNT(*) FROM BillOfMaterials WHERE BOMLevel = ? AND EndDate IS NULL", (bom_level,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of documents with a specific status and no document summary
@app.get("/v1/works_cycles/document_count_by_status", operation_id="get_document_count_by_status", summary="Retrieves the total count of documents that have a specified status and lack a document summary.")
async def get_document_count_by_status(status: int = Query(..., description="Status of the document")):
    cursor.execute("SELECT COUNT(DocumentNode) FROM Document WHERE Status = ? AND DocumentSummary IS NULL", (status,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get sales tax rate IDs by name pattern
@app.get("/v1/works_cycles/sales_tax_rate_ids_by_name", operation_id="get_sales_tax_rate_ids_by_name", summary="Get the sales tax rate IDs for names matching a given pattern")
async def get_sales_tax_rate_ids_by_name(name_pattern: str = Query(..., description="Pattern to match in the name (use %% for wildcard)")):
    cursor.execute("SELECT SalesTaxRateID FROM SalesTaxRate WHERE Name LIKE ?", (name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"sales_tax_rate_ids": []}
    return {"sales_tax_rate_ids": [row[0] for row in result]}

# Endpoint to get the top transaction ID by quantity for a given transaction type
@app.get("/v1/works_cycles/top_transaction_by_type", operation_id="get_top_transaction_by_type", summary="Retrieves the top transaction IDs, sorted by quantity, for a specified transaction type. The number of returned transactions is determined by the provided limit.")
async def get_top_transaction_by_type(transaction_type: str = Query(..., description="Type of the transaction"), limit: int = Query(..., description="Number of top transactions to return")):
    cursor.execute("SELECT TransactionID FROM TransactionHistory WHERE TransactionType = ? ORDER BY Quantity DESC LIMIT ?", (transaction_type, limit))
    result = cursor.fetchall()
    if not result:
        return {"transaction_ids": []}
    return {"transaction_ids": [row[0] for row in result]}

# Endpoint to get vendor names by preferred vendor status
@app.get("/v1/works_cycles/vendor_names_by_status", operation_id="get_vendor_names_by_status", summary="Retrieves a list of vendor names that match a specified preferred vendor status, up to a defined limit. This operation is useful for obtaining a subset of vendors based on their preferred status, allowing for targeted vendor management or analysis.")
async def get_vendor_names_by_status(preferred_vendor_status: int = Query(..., description="Preferred vendor status"), limit: int = Query(..., description="Number of vendors to return")):
    cursor.execute("SELECT Name FROM Vendor WHERE PreferredVendorStatus = ? LIMIT ?", (preferred_vendor_status, limit))
    result = cursor.fetchall()
    if not result:
        return {"vendor_names": []}
    return {"vendor_names": [row[0] for row in result]}

# Endpoint to get the count of vendors by preferred vendor status and active flag
@app.get("/v1/works_cycles/vendor_count_by_status_and_active", operation_id="get_vendor_count_by_status_and_active", summary="Retrieves the total number of vendors that match a given preferred vendor status and active flag. The preferred vendor status and active flag are provided as input parameters, allowing for a targeted count of vendors based on these criteria.")
async def get_vendor_count_by_status_and_active(preferred_vendor_status: int = Query(..., description="Preferred vendor status"), active_flag: int = Query(..., description="Active flag (1 for active, 0 for inactive)")):
    cursor.execute("SELECT COUNT(BusinessEntityID) FROM Vendor WHERE PreferredVendorStatus = ? AND ActiveFlag = ?", (preferred_vendor_status, active_flag))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of employees in a specific shift and year
@app.get("/v1/works_cycles/employee_count_shift_year", operation_id="get_employee_count_shift_year", summary="Retrieves the total number of employees who have worked in a specific shift during a given year. The operation requires the Shift ID and the year in 'YYYY' format as input parameters.")
async def get_employee_count_shift_year(shift_id: int = Query(..., description="Shift ID"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM EmployeeDepartmentHistory AS T1 INNER JOIN Shift AS T2 ON T1.ShiftId = T2.ShiftId WHERE T2.ShiftId = ? AND STRFTIME('%Y', T2.StartTime) >= ?", (shift_id, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the department with the most employees in a specific shift
@app.get("/v1/works_cycles/department_most_employees_shift", operation_id="get_department_most_employees_shift", summary="Retrieves the department with the highest number of employees working in the specified shift. The shift is identified by its name.")
async def get_department_most_employees_shift(shift_name: str = Query(..., description="Shift name")):
    cursor.execute("SELECT T3.Name FROM EmployeeDepartmentHistory AS T1 INNER JOIN Shift AS T2 ON T1.ShiftId = T2.ShiftId INNER JOIN Department AS T3 ON T1.DepartmentID = T3.DepartmentID WHERE T2.Name = ? GROUP BY T3.Name ORDER BY COUNT(T1.BusinessEntityID) DESC LIMIT 1", (shift_name,))
    result = cursor.fetchone()
    if not result:
        return {"department": []}
    return {"department": result[0]}

# Endpoint to get the count of sales orders using a specific ship method
@app.get("/v1/works_cycles/sales_order_count_ship_method", operation_id="get_sales_order_count_ship_method", summary="Retrieves the total number of sales orders associated with a specified shipping method. The response is based on a count of records from the SalesOrderHeader table that match the provided shipping method name.")
async def get_sales_order_count_ship_method(ship_method_name: str = Query(..., description="Ship method name")):
    cursor.execute("SELECT COUNT(*) FROM ShipMethod AS T1 INNER JOIN SalesOrderHeader AS T2 USING (ShipMethodID) WHERE T1.Name = ?", (ship_method_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the product subcategories for a specific product category
@app.get("/v1/works_cycles/product_subcategories_category", operation_id="get_product_subcategories_category", summary="Retrieves the names of all product subcategories associated with a given product category. The operation filters the product subcategories based on the provided product category name.")
async def get_product_subcategories_category(product_category_name: str = Query(..., description="Product category name")):
    cursor.execute("SELECT T1.Name FROM ProductSubcategory AS T1 INNER JOIN ProductCategory AS T2 ON T1.ProductCategoryID = T2.ProductCategoryID WHERE T2.name = ?", (product_category_name,))
    result = cursor.fetchall()
    if not result:
        return {"subcategories": []}
    return {"subcategories": [row[0] for row in result]}

# Endpoint to get the sales territory with the highest sales quota
@app.get("/v1/works_cycles/top_sales_territory", operation_id="get_top_sales_territory", summary="Retrieves the name of the sales territory with the highest total sales quota from all salespeople. This operation returns the top-performing sales territory based on the combined sales quotas of all associated salespeople.")
async def get_top_sales_territory():
    cursor.execute("SELECT T2.Name FROM SalesPerson AS T1 INNER JOIN SalesTerritory AS T2 ON T1.TerritoryID = T2.TerritoryID GROUP BY T1.TerritoryID ORDER BY SUM(T1.SalesQuota) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"territory": []}
    return {"territory": result[0]}

# Endpoint to get the product names for a specific sales order
@app.get("/v1/works_cycles/product_names_sales_order", operation_id="get_product_names_sales_order", summary="Retrieves the names of products associated with a specific sales order. The operation uses the provided sales order ID to filter the relevant products from the database.")
async def get_product_names_sales_order(sales_order_id: int = Query(..., description="Sales order ID")):
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN SalesOrderDetail AS T2 ON T1.ProductID = T2.ProductID WHERE T2.SalesOrderID = ?", (sales_order_id,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get the top products by order quantity
@app.get("/v1/works_cycles/top_products_order_quantity", operation_id="get_top_products_order_quantity", summary="Retrieves the top five products with the highest total order quantity. The response is paginated, with the 'offset' parameter determining the starting point and the 'limit' parameter defining the number of products to return per page.")
async def get_top_products_order_quantity(offset: int = Query(..., description="Offset for pagination"), limit: int = Query(..., description="Limit for pagination")):
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN SalesOrderDetail AS T2 ON T1.ProductID = T2.ProductID GROUP BY T1.Name ORDER BY SUM(T2.OrderQty) DESC LIMIT ?, ?", (offset, limit))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [row[0] for row in result]}

# Endpoint to get the shelf location for a specific product
@app.get("/v1/works_cycles/product_shelf_location", operation_id="get_product_shelf_location", summary="Get the shelf location for a specific product")
async def get_product_shelf_location(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT T2.Shelf FROM Product AS T1 INNER JOIN ProductInventory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"shelf": []}
    return {"shelf": result[0]}

# Endpoint to get the address type for a specific store
@app.get("/v1/works_cycles/store_address_type", operation_id="get_store_address_type", summary="Retrieves the type of address associated with a specific store. The operation requires the store name as input and returns the corresponding address type.")
async def get_store_address_type(store_name: str = Query(..., description="Store name")):
    cursor.execute("SELECT T2.Name FROM BusinessEntityAddress AS T1 INNER JOIN AddressType AS T2 ON T1.AddressTypeID = T2.AddressTypeID INNER JOIN Store AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T3.Name = ?", (store_name,))
    result = cursor.fetchone()
    if not result:
        return {"address_type": []}
    return {"address_type": result[0]}

# Endpoint to get the contact type names for a specific store
@app.get("/v1/works_cycles/contact_type_names_by_store", operation_id="get_contact_type_names_by_store", summary="Retrieves the names of all contact types associated with a specific store. The store is identified by its name, which is provided as an input parameter.")
async def get_contact_type_names_by_store(store_name: str = Query(..., description="Name of the store")):
    cursor.execute("SELECT T1.Name FROM ContactType AS T1 INNER JOIN BusinessEntityContact AS T2 ON T1.ContactTypeID = T2.ContactTypeID INNER JOIN Store AS T3 ON T2.BusinessEntityID = T3.BusinessEntityID WHERE T3.Name = ?", (store_name,))
    result = cursor.fetchall()
    if not result:
        return {"contact_type_names": []}
    return {"contact_type_names": [row[0] for row in result]}

# Endpoint to get the reference order IDs for a specific product
@app.get("/v1/works_cycles/reference_order_ids_by_product", operation_id="get_reference_order_ids_by_product", summary="Retrieves the reference order IDs associated with a specific product. The operation uses the provided product name to search for corresponding records in the Product and TransactionHistory tables, returning the reference order IDs linked to the product.")
async def get_reference_order_ids_by_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.ReferenceOrderID FROM Product AS T1 INNER JOIN TransactionHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"reference_order_ids": []}
    return {"reference_order_ids": [row[0] for row in result]}

# Endpoint to get the address line for a specific business entity
@app.get("/v1/works_cycles/address_line_by_business_entity", operation_id="get_address_line_by_business_entity", summary="Retrieves the first line of the address associated with the specified business entity. The operation uses the provided business entity ID to locate the corresponding address record and returns the first line of the address.")
async def get_address_line_by_business_entity(business_entity_id: int = Query(..., description="ID of the business entity")):
    cursor.execute("SELECT T1.AddressLine1 FROM Address AS T1 INNER JOIN BusinessEntityAddress AS T2 ON T1.AddressID = T2.AddressID WHERE T2.BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchall()
    if not result:
        return {"address_lines": []}
    return {"address_lines": [row[0] for row in result]}

# Endpoint to get the state/province names for a specific city
@app.get("/v1/works_cycles/state_province_names_by_city", operation_id="get_state_province_names_by_city", summary="Retrieves the names of all states or provinces associated with the specified city. The operation uses the provided city name to search for corresponding addresses and returns the names of the states or provinces linked to those addresses.")
async def get_state_province_names_by_city(city: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT T2.Name FROM Address AS T1 INNER JOIN StateProvince AS T2 ON T1.StateProvinceID = T2.StateProvinceID WHERE T1.City = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"state_province_names": []}
    return {"state_province_names": [row[0] for row in result]}

# Endpoint to get the percentage of salespersons exceeding a sales quota in a specific territory
@app.get("/v1/works_cycles/sales_quota_percentage", operation_id="get_sales_quota_percentage", summary="Retrieves the percentage of salespersons in a specific territory who have exceeded a given sales quota threshold. The calculation is based on the provided country region code and territory name.")
async def get_sales_quota_percentage(sales_quota: int = Query(..., description="Sales quota threshold"), country_region_code: str = Query(..., description="Country region code"), territory_name: str = Query(..., description="Name of the territory")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.SalesQuota > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.BusinessEntityID) FROM SalesPerson AS T1 INNER JOIN SalesTerritory AS T2 ON T1.TerritoryID = T2.TerritoryID WHERE T2.CountryRegionCode = ? AND T2.Name = ?", (sales_quota, country_region_code, territory_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference between last receipt cost and standard price for products with a specific name pattern
@app.get("/v1/works_cycles/cost_difference_by_product_name", operation_id="get_cost_difference_by_product_name", summary="Retrieves the cost difference between the last receipt cost and the standard price for products whose names match the provided pattern. The pattern can include wildcards for flexible matching.")
async def get_cost_difference_by_product_name(product_name_pattern: str = Query(..., description="Pattern to match product names (use % for wildcard)")):
    cursor.execute("SELECT T2.LastReceiptCost - T2.StandardPrice FROM Product AS T1 INNER JOIN ProductVendor AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name LIKE ?", (product_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"cost_differences": []}
    return {"cost_differences": [row[0] for row in result]}

# Endpoint to get the names of products with the highest rating
@app.get("/v1/works_cycles/highest_rated_products", operation_id="get_highest_rated_products", summary="Retrieves the names of products that have received the highest rating from customer reviews. This operation returns a list of product names that have been rated the highest by customers, based on the reviews they have submitted.")
async def get_highest_rated_products():
    cursor.execute("SELECT T1.Name FROM Product AS T1 INNER JOIN ProductReview AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Rating = ( SELECT Rating FROM ProductReview ORDER BY Rating DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the second highest pay rate for a specific pay frequency
@app.get("/v1/works_cycles/second_highest_pay_rate", operation_id="get_second_highest_pay_rate", summary="Retrieves the second highest pay rate for a given pay frequency. The pay frequency is specified as an input parameter, which determines the pay rate data to be considered. The operation returns the rate that ranks second highest among all pay rates for the specified pay frequency.")
async def get_second_highest_pay_rate(pay_frequency: int = Query(..., description="Pay frequency code")):
    cursor.execute("SELECT Rate FROM EmployeePayHistory WHERE PayFrequency = ? ORDER BY Rate DESC LIMIT 1, 1", (pay_frequency,))
    result = cursor.fetchone()
    if not result:
        return {"pay_rate": []}
    return {"pay_rate": result[0]}

# Endpoint to get the weight of products based on the weight unit measure code
@app.get("/v1/works_cycles/product_weight", operation_id="get_product_weight", summary="Retrieves the weight of products that share a specified weight unit measure code, grouped by weight and ordered by the count of associated styles in descending order. The result is limited to the top entry.")
async def get_product_weight(weight_unit_measure_code: str = Query(..., description="Weight unit measure code of the product")):
    cursor.execute("SELECT Weight FROM Product WHERE WeightUnitMeasureCode = ? GROUP BY Weight ORDER BY COUNT(Style) DESC LIMIT 1", (weight_unit_measure_code,))
    result = cursor.fetchone()
    if not result:
        return {"weight": []}
    return {"weight": result[0]}

# Endpoint to get the name and reorder point of products based on the size unit measure code
@app.get("/v1/works_cycles/product_reorder_point", operation_id="get_product_reorder_point", summary="Retrieves the names and reorder points of products with a specified size unit measure code, sorted by size in descending order. The response is limited to the top 6 results.")
async def get_product_reorder_point(size_unit_measure_code: str = Query(..., description="Size unit measure code of the product")):
    cursor.execute("SELECT Name, ReorderPoint FROM Product WHERE SizeUnitMeasureCode = ? ORDER BY Size DESC LIMIT 6", (size_unit_measure_code,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [{"name": row[0], "reorder_point": row[1]} for row in result]}

# Endpoint to get the total due for purchase orders based on freight
@app.get("/v1/works_cycles/purchase_order_total_due", operation_id="get_purchase_order_total_due", summary="Get the total due for purchase orders ordered by freight in descending order, with an offset of 1")
async def get_purchase_order_total_due():
    cursor.execute("SELECT TotalDue FROM PurchaseOrderHeader ORDER BY Freight DESC LIMIT 2 OFFSET 1")
    result = cursor.fetchall()
    if not result:
        return {"total_due": []}
    return {"total_due": [row[0] for row in result]}

# Endpoint to get the profit margin of products based on weight
@app.get("/v1/works_cycles/product_profit_margin", operation_id="get_product_profit_margin", summary="Retrieves the profit margin of products with a defined weight, sorted by weight in ascending order. The result is multiplied by 10 for scaling purposes.")
async def get_product_profit_margin():
    cursor.execute("SELECT 10 * (listPrice - StandardCost) FROM Product WHERE Weight IS NOT NULL ORDER BY Weight LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"profit_margin": []}
    return {"profit_margin": result[0]}

# Endpoint to get the tax amount and purchase order ID of the top purchase order
@app.get("/v1/works_cycles/top_purchase_order_tax", operation_id="get_top_purchase_order_tax", summary="Retrieves the tax amount and corresponding purchase order ID of the purchase order with the highest tax amount. The data is ordered in descending order based on the tax amount.")
async def get_top_purchase_order_tax():
    cursor.execute("SELECT TaxAmt, PurchaseOrderID FROM PurchaseOrderHeader ORDER BY TaxAmt DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"tax_amount": [], "purchase_order_id": []}
    return {"tax_amount": result[0], "purchase_order_id": result[1]}

# Endpoint to get the count of salespersons based on sales quota
@app.get("/v1/works_cycles/salesperson_count", operation_id="get_salesperson_count", summary="Retrieves the number of salespersons who have a sales quota below a given value. This operation is useful for identifying salespersons with lower sales targets, which can help in resource allocation and performance tracking.")
async def get_salesperson_count(sales_quota: int = Query(..., description="Sales quota value")):
    cursor.execute("SELECT COUNT(BusinessEntityID) FROM SalesPersonQuotaHistory WHERE SalesQuota < ?", (sales_quota,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the start and end times of shifts for employees based on birth year
@app.get("/v1/works_cycles/employee_shift_times", operation_id="get_employee_shift_times", summary="Get the start and end times of shifts for employees born before a specified year, ordered by birth date with an offset of 1")
async def get_employee_shift_times(birth_year: str = Query(..., description="Birth year of the employee in 'YYYY' format")):
    cursor.execute("SELECT T3.StartTime, T3.EndTime FROM Employee AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Shift AS T3 ON T2.ShiftId = T3.ShiftId WHERE STRFTIME('%Y', T1.BirthDate) < ? ORDER BY T1.BirthDate LIMIT 5 OFFSET 1", (birth_year,))
    result = cursor.fetchall()
    if not result:
        return {"shift_times": []}
    return {"shift_times": [{"start_time": row[0], "end_time": row[1]} for row in result]}

# Endpoint to get the product name and rating for the product with the shortest manufacturing time and highest profit margin
@app.get("/v1/works_cycles/product_name_rating_shortest_manufacture_highest_margin", operation_id="get_product_name_rating_shortest_manufacture_highest_margin", summary="Retrieves the name and rating of the product that has the shortest manufacturing time and the highest profit margin. The profit margin is determined by subtracting the standard cost from the list price.")
async def get_product_name_rating_shortest_manufacture_highest_margin():
    cursor.execute("SELECT T1.Name, T2.Rating FROM Product AS T1 INNER JOIN ProductReview AS T2 ON T1.ProductID = T2.ProductID WHERE T1.DaysToManufacture = ( SELECT DaysToManufacture FROM Product ORDER BY DaysToManufacture LIMIT 1 ) ORDER BY T1.listPrice - T1.StandardCost DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": {"name": result[0], "rating": result[1]}}

# Endpoint to get the names of salespersons with a bonus less than a specified amount
@app.get("/v1/works_cycles/salespersons_with_bonus_less_than", operation_id="get_salespersons_with_bonus_less_than", summary="Retrieves the full names of salespersons who have a bonus amount less than the specified input value. The input parameter represents the bonus threshold.")
async def get_salespersons_with_bonus_less_than(bonus: int = Query(..., description="Bonus amount")):
    cursor.execute("SELECT T2.FirstName, T2.MiddleName, T2.LastName FROM SalesPerson AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.Bonus < ?", (bonus,))
    result = cursor.fetchall()
    if not result:
        return {"salespersons": []}
    return {"salespersons": [{"first_name": row[0], "middle_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get the end dates of employees with a specific job title and age difference
@app.get("/v1/works_cycles/employee_end_dates_by_job_title_age_difference", operation_id="get_employee_end_dates_by_job_title_age_difference", summary="Retrieves the termination dates of employees who held a specific job title and had a certain age at the time of hire. The age is calculated as the difference between the hire date and birth date. Only employees with a recorded termination date are included in the results.")
async def get_employee_end_dates_by_job_title_age_difference(job_title: str = Query(..., description="Job title of the employee"), age_difference: int = Query(..., description="Age difference between hire date and birth date")):
    cursor.execute("SELECT T2.EndDate FROM Employee AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID WHERE T1.JobTitle = ? AND STRFTIME('%Y', T1.HireDate) - STRFTIME('%Y', T1.BirthDate) = ? AND T2.EndDate IS NOT NULL", (job_title, age_difference))
    result = cursor.fetchall()
    if not result:
        return {"end_dates": []}
    return {"end_dates": [row[0] for row in result]}

# Endpoint to get the total due for vendors with a specific credit rating and preferred vendor status
@app.get("/v1/works_cycles/total_due_by_credit_rating_preferred_status", operation_id="get_total_due_by_credit_rating_preferred_status", summary="Retrieves the total amount due for vendors who have a specified credit rating and preferred vendor status. The calculation is based on the sum of the total due from purchase order headers linked to the vendor's business entity ID.")
async def get_total_due_by_credit_rating_preferred_status(credit_rating: int = Query(..., description="Credit rating of the vendor"), preferred_vendor_status: int = Query(..., description="Preferred vendor status (0 or 1)")):
    cursor.execute("SELECT SUM(T2.TotalDue) FROM Vendor AS T1 INNER JOIN PurchaseOrderHeader AS T2 ON T1.BusinessEntityID = T2.VendorID WHERE T1.CreditRating = ? AND T1.PreferredVendorStatus = ?", (credit_rating, preferred_vendor_status))
    result = cursor.fetchone()
    if not result:
        return {"total_due": []}
    return {"total_due": result[0]}

# Endpoint to get the department with the most night shifts
@app.get("/v1/works_cycles/department_with_most_night_shifts", operation_id="get_department_with_most_night_shifts", summary="Retrieves the name of the department with the highest count of a specific shift, as identified by the provided shift name. The operation filters shifts by the given name, groups them by department, and orders the results in descending order based on the count of the specified shift. The department with the most occurrences of the shift is returned.")
async def get_department_with_most_night_shifts(shift_name: str = Query(..., description="Name of the shift (e.g., 'Night')")):
    cursor.execute("SELECT T3.Name FROM Shift AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.ShiftId = T2.ShiftId INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID GROUP BY T2.DepartmentID ORDER BY COUNT(T1.Name = ?) DESC LIMIT 1", (shift_name,))
    result = cursor.fetchone()
    if not result:
        return {"department": []}
    return {"department": result[0]}

# Endpoint to get the double profit margin for products with specific class, color, size, and subcategory
@app.get("/v1/works_cycles/double_profit_margin_by_class_color_size_subcategory", operation_id="get_double_profit_margin_by_class_color_size_subcategory", summary="Retrieves the double profit margin for products that match the specified class, color, size, and subcategory. The calculation is based on the difference between the list price and standard cost of the product.")
async def get_double_profit_margin_by_class_color_size_subcategory(product_class: str = Query(..., description="Class of the product"), color: str = Query(..., description="Color of the product"), size: int = Query(..., description="Size of the product"), subcategory_name: str = Query(..., description="Name of the product subcategory")):
    cursor.execute("SELECT 2 * (T1.listPrice - T1.StandardCost) FROM Product AS T1 INNER JOIN ProductSubcategory AS T2 ON T1.ProductSubcategoryID = T2.ProductSubcategoryID WHERE T1.Class = ? AND T1.Color = ? AND T1.Size = ? AND T2.Name = ?", (product_class, color, size, subcategory_name))
    result = cursor.fetchone()
    if not result:
        return {"double_profit_margin": []}
    return {"double_profit_margin": result[0]}

# Endpoint to get the names of employees with a specific job title, gender, and marital status
@app.get("/v1/works_cycles/employee_names_by_job_title_gender_marital_status", operation_id="get_employee_names_by_job_title_gender_marital_status", summary="Retrieves the names of the ten youngest employees who match the specified job title, gender, and marital status. The job title can include a wildcard character for partial matches. The results are sorted by birth date in descending order.")
async def get_employee_names_by_job_title_gender_marital_status(job_title: str = Query(..., description="Job title of the employee (use % for wildcard)"), gender: str = Query(..., description="Gender of the employee"), marital_status: str = Query(..., description="Marital status of the employee")):
    cursor.execute("SELECT T2.FirstName, T2.MiddleName, T2.LastName FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.JobTitle LIKE ? AND T1.Gender = ? AND T1.MaritalStatus = ? ORDER BY T1.BirthDate DESC LIMIT 10", (job_title, gender, marital_status))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"first_name": row[0], "middle_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get the vendor name and credit rating with a specific average lead time and highest profit margin
@app.get("/v1/works_cycles/vendor_name_credit_rating_by_average_lead_time", operation_id="get_vendor_name_credit_rating_by_average_lead_time", summary="Retrieves the name and credit rating of the vendor with the highest profit margin for a given average lead time. The profit margin is calculated as the difference between the last receipt cost and the standard price.")
async def get_vendor_name_credit_rating_by_average_lead_time(average_lead_time: int = Query(..., description="Average lead time of the vendor")):
    cursor.execute("SELECT T2.Name, T2.CreditRating FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.AverageLeadTime = ? ORDER BY T1.LastReceiptCost - T1.StandardPrice DESC LIMIT 1", (average_lead_time,))
    result = cursor.fetchone()
    if not result:
        return {"vendor": []}
    return {"vendor": {"name": result[0], "credit_rating": result[1]}}

# Endpoint to get the profit margin and vendor name for vendors with a specific maximum order quantity
@app.get("/v1/works_cycles/profit_margin_vendor_name_by_max_order_qty", operation_id="get_profit_margin_vendor_name_by_max_order_qty", summary="Retrieves the profit margin and vendor name for vendors who have a specified maximum order quantity. The profit margin is calculated as the difference between the last receipt cost and the standard price of the product. The vendor name is obtained from the Vendor table.")
async def get_profit_margin_vendor_name_by_max_order_qty(max_order_qty: int = Query(..., description="Maximum order quantity")):
    cursor.execute("SELECT T1.LastReceiptCost - T1.StandardPrice, T2.Name FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.MaxOrderQty = ?", (max_order_qty,))
    result = cursor.fetchall()
    if not result:
        return {"vendors": []}
    return {"vendors": [{"profit_margin": row[0], "name": row[1]} for row in result]}

# Endpoint to get the employee with the most purchase orders of a specific status and person type
@app.get("/v1/works_cycles/employee_with_most_purchase_orders", operation_id="get_employee_with_most_purchase_orders", summary="Retrieves the employee who has initiated the highest number of purchase orders with a specific status and person type. The response includes the first and last name of the employee. The person type and purchase order status are used to filter the results.")
async def get_employee_with_most_purchase_orders(person_type: str = Query(..., description="Type of the person (e.g., 'EM' for employee)"), status: int = Query(..., description="Status of the purchase order")):
    cursor.execute("SELECT T2.FirstName, T2.LastName FROM PurchaseOrderHeader AS T1 INNER JOIN Person AS T2 ON T1.EmployeeID = T2.BusinessEntityID WHERE T2.PersonType = ? AND T1.Status = ? GROUP BY T2.FirstName, T2.LastName ORDER BY COUNT(T1.PurchaseOrderID) DESC LIMIT 1", (person_type, status))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get vendor names ordered by maximum order quantity
@app.get("/v1/works_cycles/vendor_names_by_max_order_qty", operation_id="get_vendor_names_by_max_order_qty", summary="Retrieves a list of vendor names, ordered by their maximum order quantity, with the ability to limit and offset the results. This operation allows you to view a specific range of vendors based on their maximum order quantity, providing a targeted view of vendor capacity.")
async def get_vendor_names_by_max_order_qty(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset the results")):
    cursor.execute("SELECT T2.Name FROM ProductVendor AS T1 INNER JOIN Vendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID ORDER BY T1.MaxOrderQty ASC LIMIT ?, ?", (offset, limit))
    result = cursor.fetchall()
    if not result:
        return {"vendor_names": []}
    return {"vendor_names": [row[0] for row in result]}

# Endpoint to get minimum order quantity for vendors based on active flag
@app.get("/v1/works_cycles/min_order_qty_by_active_flag", operation_id="get_min_order_qty_by_active_flag", summary="Retrieves the minimum order quantity for active or inactive vendors, sorted in ascending order. The number of results returned can be limited by specifying a value for the limit parameter.")
async def get_min_order_qty_by_active_flag(active_flag: int = Query(..., description="Active flag of the vendor (0 or 1)"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.MinOrderQty FROM Vendor AS T1 INNER JOIN ProductVendor AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.ActiveFlag = ? ORDER BY T2.MinOrderQty LIMIT ?", (active_flag, limit))
    result = cursor.fetchall()
    if not result:
        return {"min_order_qty": []}
    return {"min_order_qty": [row[0] for row in result]}

# Endpoint to get employee details based on vacation hours
@app.get("/v1/works_cycles/employee_details_by_vacation_hours", operation_id="get_employee_details_by_vacation_hours", summary="Retrieves a list of employee details, sorted by the difference between their hire year and birth year in descending order. The list is filtered to include only employees with vacation hours less than or equal to the provided maximum. The number of results returned can be limited by specifying a maximum number of records.")
async def get_employee_details_by_vacation_hours(vacation_hours: int = Query(..., description="Maximum vacation hours"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT STRFTIME('%Y', T1.HireDate) - STRFTIME('%Y', T1.BirthDate), T2.FirstName, T2.MiddleName, T2.LastName FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.VacationHours <= ? ORDER BY STRFTIME('%Y', T1.HireDate) - STRFTIME('%Y', T1.BirthDate) DESC LIMIT ?", (vacation_hours, limit))
    result = cursor.fetchall()
    if not result:
        return {"employee_details": []}
    return {"employee_details": [{"years_worked": row[0], "first_name": row[1], "middle_name": row[2], "last_name": row[3]} for row in result]}

# Endpoint to get person types based on gender, marital status, and hire date range
@app.get("/v1/works_cycles/person_types_by_gender_marital_status_hire_date", operation_id="get_person_types_by_gender_marital_status_hire_date", summary="Retrieves the types of persons categorized by gender, marital status, and hire date range. The results are ordered by the frequency of each person type, with the most common types listed first. The number of results can be limited by specifying a maximum count.")
async def get_person_types_by_gender_marital_status_hire_date(gender: str = Query(..., description="Gender of the employee"), marital_status: str = Query(..., description="Marital status of the employee"), start_date: str = Query(..., description="Start date of the hire date range in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the hire date range in 'YYYY-MM-DD' format"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.PersonType FROM Employee AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.Gender = ? AND T1.MaritalStatus = ? AND STRFTIME('%Y-%m-%d', T1.HireDate) BETWEEN ? AND ? GROUP BY T2.PersonType ORDER BY COUNT(T2.PersonType) DESC LIMIT ?", (gender, marital_status, start_date, end_date, limit))
    result = cursor.fetchall()
    if not result:
        return {"person_types": []}
    return {"person_types": [row[0] for row in result]}

# Endpoint to get the difference between list price and standard cost based on reviewer name
@app.get("/v1/works_cycles/price_difference_by_reviewer_name", operation_id="get_price_difference_by_reviewer_name", summary="Retrieves the price difference between the list price and standard cost for products reviewed by a specified reviewer, ordered by rating in descending order. The number of results returned is limited by the provided limit parameter.")
async def get_price_difference_by_reviewer_name(reviewer_name: str = Query(..., description="Name of the reviewer"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.listPrice - T2.StandardCost FROM ProductReview AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ReviewerName = ? ORDER BY T1.Rating DESC LIMIT ?", (reviewer_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"price_difference": []}
    return {"price_difference": [row[0] for row in result]}

# Endpoint to get sales territory groups and state province names ordered by tax rate
@app.get("/v1/works_cycles/sales_territory_groups_by_tax_rate", operation_id="get_sales_territory_groups_by_tax_rate", summary="Get sales territory groups and state province names ordered by tax rate with a specified limit and offset")
async def get_sales_territory_groups_by_tax_rate(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset the results")):
    cursor.execute("SELECT T3.Group, T2.Name FROM SalesTaxRate AS T1 INNER JOIN StateProvince AS T2 ON T1.StateProvinceID = T2.StateProvinceID INNER JOIN SalesTerritory AS T3 ON T2.TerritoryID = T3.TerritoryID ORDER BY T1.TaxRate LIMIT ?, ?", (offset, limit))
    result = cursor.fetchall()
    if not result:
        return {"sales_territory_groups": []}
    return {"sales_territory_groups": [{"group": row[0], "name": row[1]} for row in result]}

# Endpoint to get average price difference, count of distinct sizes, and count of distinct styles based on product class and subcategory name
@app.get("/v1/works_cycles/product_stats_by_class_subcategory", operation_id="get_product_stats_by_class_subcategory", summary="Retrieves statistical data for a specific product class and subcategory, including the average price difference between list price and standard cost, the count of distinct sizes, and the count of distinct styles. The data is grouped by product class and color.")
async def get_product_stats_by_class_subcategory(product_class: str = Query(..., description="Class of the product"), subcategory_name: str = Query(..., description="Name of the product subcategory")):
    cursor.execute("SELECT AVG(T1.listPrice - T1.StandardCost), COUNT(DISTINCT T1.Size), COUNT(DISTINCT T1.Style) FROM Product AS T1 INNER JOIN ProductSubcategory AS T2 ON T1.ProductSubcategoryID = T2.ProductSubcategoryID WHERE T1.Class = ? AND T2.Name = ? GROUP BY T1.Class, T1.Color", (product_class, subcategory_name))
    result = cursor.fetchall()
    if not result:
        return {"product_stats": []}
    return {"product_stats": [{"avg_price_difference": row[0], "distinct_sizes": row[1], "distinct_styles": row[2]} for row in result]}

# Endpoint to get discount percentage based on special offer description
@app.get("/v1/works_cycles/discount_percentage_by_description", operation_id="get_discount_percentage_by_description", summary="Retrieves the discount percentage associated with a specific special offer, based on the provided description.")
async def get_discount_percentage_by_description(description: str = Query(..., description="Description of the special offer")):
    cursor.execute("SELECT DiscountPct FROM SpecialOffer WHERE Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"discount_percentage": []}
    return {"discount_percentage": [row[0] for row in result]}

# Endpoint to get the count of special offers based on type
@app.get("/v1/works_cycles/count_special_offers_by_type", operation_id="get_count_special_offers_by_type", summary="Retrieves the total count of special offers of a specific type. The operation requires the type of the special offer as an input parameter to filter the count accordingly.")
async def get_count_special_offers_by_type(offer_type: str = Query(..., description="Type of the special offer")):
    cursor.execute("SELECT COUNT(SpecialOfferID) FROM SpecialOffer WHERE Type = ?", (offer_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get special offer descriptions based on type and ordered by discount percentage
@app.get("/v1/works_cycles/special_offer_descriptions_by_type", operation_id="get_special_offer_descriptions_by_type", summary="Retrieves a list of special offer descriptions of a specified type, ordered by the discount percentage in descending order. The number of results returned is limited by the provided limit parameter.")
async def get_special_offer_descriptions_by_type(offer_type: str = Query(..., description="Type of the special offer"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT Description FROM SpecialOffer WHERE Type = ? ORDER BY DiscountPct DESC LIMIT ?", (offer_type, limit))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get special offers based on category and pagination
@app.get("/v1/works_cycles/special_offers_by_category", operation_id="get_special_offers_by_category", summary="Retrieves the top three special offers for a given category, sorted by the highest discount percentage. Pagination is supported through the use of offset and limit parameters.")
async def get_special_offers_by_category(category: str = Query(..., description="Category of the special offer"), offset: int = Query(..., description="Offset for pagination"), limit: int = Query(..., description="Limit for pagination")):
    cursor.execute("SELECT Description, DiscountPct FROM SpecialOffer WHERE Category = ? ORDER BY DiscountPct DESC LIMIT ?, ?", (category, offset, limit))
    result = cursor.fetchall()
    if not result:
        return {"offers": []}
    return {"offers": result}

# Endpoint to get store demographics based on store name
@app.get("/v1/works_cycles/store_demographics_by_name", operation_id="get_store_demographics_by_name", summary="Retrieves the demographic data associated with a specific store, identified by its name. The demographic data provides insights into the characteristics of the store's customer base.")
async def get_store_demographics_by_name(name: str = Query(..., description="Name of the store")):
    cursor.execute("SELECT Demographics FROM Store WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"demographics": []}
    return {"demographics": result[0]}

# Endpoint to get the sum of finished goods flag based on make flag
@app.get("/v1/works_cycles/sum_finished_goods_flag_by_make_flag", operation_id="get_sum_finished_goods_flag_by_make_flag", summary="Retrieves the total count of finished goods that match the specified make flag. The make flag is used to filter the products and calculate the sum of the finished goods flag.")
async def get_sum_finished_goods_flag_by_make_flag(make_flag: int = Query(..., description="Make flag value")):
    cursor.execute("SELECT SUM(FinishedGoodsFlag) FROM Product WHERE MakeFlag = ?", (make_flag,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get safety stock level based on product name
@app.get("/v1/works_cycles/safety_stock_level_by_product_name", operation_id="get_safety_stock_level_by_product_name", summary="Retrieves the safety stock level for a specific product, identified by its name. This operation allows you to check the minimum stock level required for a product to avoid stockouts and maintain optimal inventory levels.")
async def get_safety_stock_level_by_product_name(name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SafetyStockLevel FROM Product WHERE Name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"safety_stock_level": []}
    return {"safety_stock_level": result[0]}

# Endpoint to get product names ordered by standard cost with a limit
@app.get("/v1/works_cycles/product_names_by_standard_cost", operation_id="get_product_names_by_standard_cost", summary="Retrieves a list of product names, ordered by their standard cost in descending order. The number of products returned is limited by the provided limit parameter. This endpoint is useful for obtaining a ranked list of products based on their standard cost.")
async def get_product_names_by_standard_cost(limit: int = Query(..., description="Limit for the number of products")):
    cursor.execute("SELECT Name FROM Product ORDER BY StandardCost DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get person type based on first name and last name
@app.get("/v1/works_cycles/person_type_by_name", operation_id="get_person_type_by_name", summary="Get person type based on first name and last name")
async def get_person_type_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT PersonType FROM Person WHERE FirstName = ? AND LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"person_type": []}
    return {"person_type": result[0]}

# Endpoint to get the count of persons based on person type and title
@app.get("/v1/works_cycles/count_persons_by_type_and_title", operation_id="get_count_persons_by_type_and_title", summary="Retrieves the total number of individuals categorized by a specific person type and title. The response is based on the provided person type and title parameters.")
async def get_count_persons_by_type_and_title(person_type: str = Query(..., description="Person type"), title: str = Query(..., description="Title of the person")):
    cursor.execute("SELECT COUNT(BusinessEntityID) FROM Person WHERE PersonType = ? AND Title = ?", (person_type, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get address details based on store name
@app.get("/v1/works_cycles/address_details_by_store_name", operation_id="get_address_details_by_store_name", summary="Retrieves the address details of a specific store, identified by its name. The response includes the first and second lines of the store's address.")
async def get_address_details_by_store_name(name: str = Query(..., description="Name of the store")):
    cursor.execute("SELECT T2.AddressLine1, T2.AddressLine2 FROM BusinessEntityAddress AS T1 INNER JOIN Address AS T2 ON T1.AddressID = T2.AddressID INNER JOIN Store AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID WHERE T3.Name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"address_details": []}
    return {"address_details": result}

# Endpoint to get email address based on first name and last name
@app.get("/v1/works_cycles/email_address_by_name", operation_id="get_email_address_by_name", summary="Retrieves the email address associated with a person, identified by their first and last names. The endpoint searches for a match in the Person and EmailAddress tables, using the provided first and last names as criteria.")
async def get_email_address_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T2.EmailAddress FROM Person AS T1 INNER JOIN EmailAddress AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"email_addresses": []}
    return {"email_addresses": result}

# Endpoint to get phone numbers based on person type
@app.get("/v1/works_cycles/phone_numbers_by_person_type", operation_id="get_phone_numbers_by_person_type", summary="Get phone numbers based on person type")
async def get_phone_numbers_by_person_type(person_type: str = Query(..., description="Person type")):
    cursor.execute("SELECT T2.PhoneNumber FROM Person AS T1 INNER JOIN PersonPhone AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.PersonType = ?", (person_type,))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": result}

# Endpoint to get the password hash for a person based on their first and last name
@app.get("/v1/works_cycles/password_hash_by_name", operation_id="get_password_hash_by_name", summary="Get the password hash for a person based on their first and last name")
async def get_password_hash_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T2.PasswordHash FROM Person AS T1 INNER JOIN Password AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"password_hash": []}
    return {"password_hash": result[0]}

# Endpoint to get the email address for a person based on their email promotion status
@app.get("/v1/works_cycles/email_address_by_promotion", operation_id="get_email_address_by_promotion", summary="Get the email address for a person based on their email promotion status")
async def get_email_address_by_promotion(email_promotion: int = Query(..., description="Email promotion status of the person")):
    cursor.execute("SELECT T2.EmailAddress FROM Person AS T1 INNER JOIN EmailAddress AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.EmailPromotion = ?", (email_promotion,))
    result = cursor.fetchone()
    if not result:
        return {"email_address": []}
    return {"email_address": result[0]}

# Endpoint to get the credit card number for a person based on their first and last name
@app.get("/v1/works_cycles/credit_card_number_by_name", operation_id="get_credit_card_number_by_name", summary="Retrieves the credit card number associated with a person, identified by their first and last name. The operation searches for the person using the provided names and returns the corresponding credit card number.")
async def get_credit_card_number_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T3.CardNumber FROM Person AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN CreditCard AS T3 ON T2.CreditCardID = T3.CreditCardID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"card_number": []}
    return {"card_number": result[0]}

# Endpoint to get the expiration year of a credit card for a person based on their first and last name
@app.get("/v1/works_cycles/credit_card_exp_year_by_name", operation_id="get_credit_card_exp_year_by_name", summary="Retrieves the expiration year of a credit card associated with a person, identified by their first and last name. The operation uses the provided names to search for the corresponding person and their linked credit card details.")
async def get_credit_card_exp_year_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T3.ExpYear FROM Person AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN CreditCard AS T3 ON T2.CreditCardID = T3.CreditCardID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"exp_year": []}
    return {"exp_year": result[0]}

# Endpoint to get the first and last name of persons with a specific credit card expiration year and person type
@app.get("/v1/works_cycles/person_names_by_exp_year_and_type", operation_id="get_person_names_by_exp_year_and_type", summary="Retrieves the first and last names of individuals who have a specific credit card expiration year and person type. The operation filters the Person table based on the provided expiration year and person type, and returns the corresponding first and last names.")
async def get_person_names_by_exp_year_and_type(exp_year: int = Query(..., description="Expiration year of the credit card"), person_type: str = Query(..., description="Type of the person")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Person AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN CreditCard AS T3 ON T2.CreditCardID = T3.CreditCardID WHERE T3.ExpYear = ? AND T1.PersonType = ?", (exp_year, person_type))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the count of persons with a specific credit card type and person type
@app.get("/v1/works_cycles/count_persons_by_card_type_and_person_type", operation_id="get_count_persons_by_card_type_and_person_type", summary="Retrieves the total number of individuals of a specified type who possess a certain credit card type. The count is determined by matching the provided card type and person type with corresponding records in the database.")
async def get_count_persons_by_card_type_and_person_type(card_type: str = Query(..., description="Type of the credit card"), person_type: str = Query(..., description="Type of the person")):
    cursor.execute("SELECT COUNT(T1.FirstName) FROM Person AS T1 INNER JOIN PersonCreditCard AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN CreditCard AS T3 ON T2.CreditCardID = T3.CreditCardID WHERE T3.CardType = ? AND T1.PersonType = ?", (card_type, person_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of departments for a person based on their first and last name
@app.get("/v1/works_cycles/count_departments_by_name", operation_id="get_count_departments_by_name", summary="Retrieves the total number of departments a person has been associated with, based on their first and last name. The endpoint uses the provided names to identify the person and calculate the department count.")
async def get_count_departments_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT COUNT(T3.DepartmentID) FROM Person AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the department IDs for a person based on their first and last name
@app.get("/v1/works_cycles/department_ids_by_name", operation_id="get_department_ids_by_name", summary="Retrieves the department IDs associated with a person, identified by their first and last name. This operation returns a list of department IDs that the person has been a part of, based on their employment history.")
async def get_department_ids_by_name(first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT T2.DepartmentID FROM Person AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"department_ids": []}
    return {"department_ids": [row[0] for row in result]}

# Endpoint to get the count of persons in a specific department with employment history spanning a specific year
@app.get("/v1/works_cycles/count_persons_by_department_and_year", operation_id="get_count_persons_by_department_and_year", summary="Retrieves the number of employees who have worked in a specified department during a given year. The department is identified by its name, and the year is provided in 'YYYY' format.")
async def get_count_persons_by_department_and_year(department_name: str = Query(..., description="Name of the department"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM Person AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID WHERE T3.Name = ? AND STRFTIME('%Y', T2.EndDate) > ? AND STRFTIME('%Y', T2.StartDate) < ?", (department_name, year, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the person with the longest employment duration in a specific department
@app.get("/v1/works_cycles/longest_employment_in_department", operation_id="get_longest_employment_in_department", summary="Retrieves the name of the employee with the longest employment duration in the specified department. The department is identified by its name, and the duration is calculated based on the difference between the end and start dates of the employment period.")
async def get_longest_employment_in_department(department_name: str = Query(..., description="Name of the department")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Person AS T1 INNER JOIN EmployeeDepartmentHistory AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T2.DepartmentID = T3.DepartmentID WHERE T3.Name = ? ORDER BY T2.EndDate - T2.StartDate DESC LIMIT 1", (department_name,))
    result = cursor.fetchone()
    if not result:
        return {"person": []}
    return {"person": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get credit card numbers for employees in a specific department with a non-null end date
@app.get("/v1/works_cycles/credit_card_numbers_department_non_null_end_date", operation_id="get_credit_card_numbers", summary="Retrieves the credit card numbers associated with employees who have worked in the specified department and have a recorded end date for their employment. The department name is used to filter the results.")
async def get_credit_card_numbers(department_name: str = Query(..., description="Department name (e.g., 'Finance')")):
    cursor.execute("SELECT T3.CardNumber FROM EmployeeDepartmentHistory AS T1 INNER JOIN Department AS T2 ON T1.DepartmentID = T2.DepartmentID INNER JOIN CreditCard AS T3 ON T1.ModifiedDate = T3.ModifiedDate INNER JOIN PersonCreditCard AS T4 ON T3.CreditCardID = T4.CreditCardID WHERE T2.Name = ? AND T1.EndDate IS NOT NULL", (department_name,))
    result = cursor.fetchall()
    if not result:
        return {"card_numbers": []}
    return {"card_numbers": [row[0] for row in result]}

# Endpoint to get the count of employees with credit cards expiring in a specific year and department
@app.get("/v1/works_cycles/employee_count_credit_card_expiry_year_department", operation_id="get_employee_count_credit_card_expiry", summary="Retrieves the number of employees in a specific department who have credit cards expiring in a given year. The response is based on the provided expiry year and department name.")
async def get_employee_count_credit_card_expiry(exp_year: int = Query(..., description="Expiry year of the credit card (e.g., 2007)"), department_name: str = Query(..., description="Department name (e.g., 'Engineering')")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM EmployeeDepartmentHistory AS T1 INNER JOIN Department AS T2 ON T1.DepartmentID = T2.DepartmentID INNER JOIN PersonCreditCard AS T3 ON T1.BusinessEntityID = T3.BusinessEntityID INNER JOIN CreditCard AS T4 ON T3.CreditCardID = T4.CreditCardID WHERE T4.ExpYear = ? AND T2.Name = ?", (exp_year, department_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the email address of the employee with the most department changes
@app.get("/v1/works_cycles/email_most_department_changes", operation_id="get_email_most_department_changes", summary="Retrieves the email address of the employee who has changed departments the most frequently. This operation uses historical department change data to identify the employee with the highest number of department changes and returns their email address.")
async def get_email_most_department_changes():
    cursor.execute("SELECT T2.EmailAddress FROM EmployeeDepartmentHistory AS T1 INNER JOIN EmailAddress AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID GROUP BY T2.BusinessEntityID ORDER BY COUNT(T1.DepartmentID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"email": []}
    return {"email": result[0]}

# Endpoint to get the count of employees in a specific department with a specific email promotion setting
@app.get("/v1/works_cycles/employee_count_department_email_promotion", operation_id="get_employee_count_email_promotion", summary="Retrieves the total number of employees in a specified department who have a particular email promotion setting. The department is identified by its name, and the email promotion setting is represented by a binary value (0 or 1).")
async def get_employee_count_email_promotion(department_name: str = Query(..., description="Department name (e.g., 'Engineering')"), email_promotion: int = Query(..., description="Email promotion setting (0 or 1)")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM EmployeeDepartmentHistory AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID INNER JOIN Department AS T3 ON T1.DepartmentID = T3.DepartmentID WHERE T3.Name = ? AND T2.EmailPromotion = ?", (department_name, email_promotion))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of employees in a specific department group who started in a specific year
@app.get("/v1/works_cycles/employee_count_department_group_start_year", operation_id="get_employee_count_start_year", summary="Retrieves the number of employees who started in a specific year and belong to a given department group. The department group is identified by its name, and the start year is provided in 'YYYY' format.")
async def get_employee_count_start_year(group_name: str = Query(..., description="Department group name (e.g., 'Quality Assurance')"), start_year: str = Query(..., description="Start year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.BusinessEntityID) FROM EmployeeDepartmentHistory AS T1 INNER JOIN Department AS T2 ON T1.DepartmentID = T2.DepartmentID WHERE T2.GroupName = ? AND STRFTIME('%Y', T1.StartDate) = ?", (group_name, start_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to check if a product is part of any special offer
@app.get("/v1/works_cycles/check_product_special_offer", operation_id="check_product_special_offer", summary="Determines whether a specific product is included in any active special offers. The operation checks the product name against the database and returns a 'Yes' or 'No' response based on the presence of the product in any special offer.")
async def check_product_special_offer(product_name: str = Query(..., description="Product name (e.g., 'Chainring Bolts')")):
    cursor.execute("SELECT CASE WHEN COUNT(T1.Description) >= 1 THEN 'Yes' ELSE 'No' END FROM SpecialOffer AS T1 INNER JOIN SpecialOfferProduct AS T2 ON T1.SpecialOfferID = T2.SpecialOfferID INNER JOIN Product AS T3 ON T2.ProductID = T3.ProductID WHERE T3.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"is_part_of_special_offer": []}
    return {"is_part_of_special_offer": result[0]}

# Endpoint to get the count of products in a specific subcategory for a special offer description
@app.get("/v1/works_cycles/product_count_subcategory_special_offer", operation_id="get_product_count_subcategory", summary="Retrieves the total number of products in a specified subcategory that are part of a given special offer. The operation requires the special offer description and the name of the product subcategory as input parameters. The response will provide a count of the products that meet the specified criteria.")
async def get_product_count_subcategory(description: str = Query(..., description="Special offer description (e.g., 'LL Road Frame Sale')"), subcategory_name: str = Query(..., description="Product subcategory name (e.g., 'Clothing')")):
    cursor.execute("SELECT COUNT(T2.ProductID) FROM SpecialOffer AS T1 INNER JOIN SpecialOfferProduct AS T2 ON T1.SpecialOfferID = T2.SpecialOfferID INNER JOIN Product AS T3 ON T2.ProductID = T3.ProductID INNER JOIN ProductSubcategory AS T4 ON T3.ProductSubcategoryID = T4.ProductSubcategoryID INNER JOIN ProductCategory AS T5 ON T4.ProductCategoryID = T5.ProductCategoryID WHERE T1.Description = ? AND T4.Name = ?", (description, subcategory_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get product names based on product subcategory ID
@app.get("/v1/works_cycles/product_names_by_subcategory", operation_id="get_product_names_by_subcategory", summary="Retrieves the names of products that belong to the specified product subcategory. The operation filters products based on the provided product subcategory ID and returns the names of matching products, excluding those marked as 'MakeFlag'.")
async def get_product_names_by_subcategory(product_subcategory_id: int = Query(..., description="Product subcategory ID")):
    cursor.execute("SELECT CASE WHEN T1.MakeFlag = 1 THEN T1.Name END FROM Product AS T1 INNER JOIN ProductSubcategory AS T2 ON T1.ProductSubcategoryID = T2.ProductSubcategoryID INNER JOIN ProductCategory AS T3 ON T2.ProductCategoryID = T3.ProductCategoryID WHERE T2.ProductSubcategoryID = ?", (product_subcategory_id,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result if row[0] is not None]}

# Endpoint to get the average duration of employment in a specific department
@app.get("/v1/works_cycles/average_employment_duration", operation_id="get_average_employment_duration", summary="Retrieves the average duration of employment for employees in a specified department. The calculation considers the start and end dates of each employment cycle, excluding ongoing cycles. The result is expressed in years.")
async def get_average_employment_duration(department_name: str = Query(..., description="Name of the department")):
    cursor.execute("SELECT CAST(SUM(365 * (STRFTIME('%Y', T1.EndDate) - STRFTIME('%Y', T1.StartDate)) + 30 * (STRFTIME('%m', T1.EndDate) - STRFTIME('%m', T1.StartDate)) + STRFTIME('%d', T1.EndDate) - STRFTIME('%d', T1.StartDate)) AS REAL) / COUNT(T1.BusinessEntityID) FROM EmployeeDepartmentHistory AS T1 INNER JOIN Department AS T2 ON T1.DepartmentID = T2.DepartmentID WHERE T2.Name = ? AND T1.EndDate IS NOT NULL", (department_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_duration": []}
    return {"average_duration": result[0]}

# Endpoint to get the percentage difference between the maximum and minimum list prices for products in a specific category
@app.get("/v1/works_cycles/list_price_percentage_difference", operation_id="get_list_price_percentage_difference", summary="Retrieves the percentage difference between the highest and lowest list prices for products within a specified category. This operation calculates the difference as a percentage of the lowest list price, providing a measure of price variation within the category. The category is identified by its name.")
async def get_list_price_percentage_difference(category_name: str = Query(..., description="Name of the product category")):
    cursor.execute("SELECT (MAX(T1.listPrice) - MIN(T1.listPrice)) * 100 / MIN(T1.listPrice) FROM Product AS T1 INNER JOIN ProductSubcategory AS T2 ON T1.ProductSubcategoryID = T2.ProductSubcategoryID INNER JOIN ProductCategory AS T3 ON T2.ProductCategoryID = T3.ProductCategoryID WHERE T3.Name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the average profit margin for products in a specific category
@app.get("/v1/works_cycles/average_profit_margin", operation_id="get_average_profit_margin", summary="Retrieves the average profit margin for products within a specified category. The calculation is based on the sum of the difference between the list price and standard cost of each product, divided by the total count of products in the category. The category is identified by its name.")
async def get_average_profit_margin(category_name: str = Query(..., description="Name of the product category")):
    cursor.execute("SELECT SUM(T1.listPrice - T1.StandardCost) / COUNT(T1.ProductID) FROM Product AS T1 INNER JOIN ProductSubcategory AS T2 ON T1.ProductSubcategoryID = T2.ProductSubcategoryID INNER JOIN ProductCategory AS T3 ON T2.ProductCategoryID = T3.ProductCategoryID WHERE T3.Name = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_profit_margin": []}
    return {"average_profit_margin": result[0]}

# Endpoint to get the product ID with the lowest standard cost for a given start date pattern
@app.get("/v1/works_cycles/product_id_by_start_date", operation_id="get_product_id_by_start_date", summary="Retrieves the product ID associated with the lowest standard cost for a specified start date pattern. The start date pattern should be provided in the 'YYYY%' format. The operation returns the product ID that meets the criteria, sorted by the standard cost in ascending order.")
async def get_product_id_by_start_date(start_date_pattern: str = Query(..., description="Start date pattern in 'YYYY%' format")):
    cursor.execute("SELECT ProductID FROM ProductCostHistory WHERE StartDate LIKE ? ORDER BY StandardCost LIMIT 1", (start_date_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"product_id": []}
    return {"product_id": result[0]}

# Endpoint to get product IDs based on color
@app.get("/v1/works_cycles/product_ids_by_color", operation_id="get_product_ids_by_color", summary="Retrieves a list of product IDs that match the specified color. The color parameter is used to filter the products and return only those that have the given color.")
async def get_product_ids_by_color(color: str = Query(..., description="Color of the product")):
    cursor.execute("SELECT ProductID FROM Product WHERE Color = ?", (color,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get business entity IDs based on title, first name, and last name
@app.get("/v1/works_cycles/business_entity_ids_by_name", operation_id="get_business_entity_ids_by_name", summary="Retrieves the unique business entity IDs associated with a person, based on their title, first name, and last name. This operation allows you to identify the business entities linked to a specific individual, enabling efficient data retrieval and management.")
async def get_business_entity_ids_by_name(title: str = Query(..., description="Title of the person"), first_name: str = Query(..., description="First name of the person"), last_name: str = Query(..., description="Last name of the person")):
    cursor.execute("SELECT BusinessEntityID FROM Person WHERE Title = ? AND FirstName = ? AND LastName = ?", (title, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"business_entity_ids": []}
    return {"business_entity_ids": [row[0] for row in result]}

# Endpoint to get phone numbers based on business entity ID
@app.get("/v1/works_cycles/phone_numbers_by_business_entity", operation_id="get_phone_numbers_by_business_entity", summary="Get phone numbers based on business entity ID")
async def get_phone_numbers_by_business_entity(business_entity_id: int = Query(..., description="Business entity ID")):
    cursor.execute("SELECT PhoneNumber FROM PersonPhone WHERE BusinessEntityID = ?", (business_entity_id,))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": [row[0] for row in result]}

# Endpoint to get list prices based on product ID
@app.get("/v1/works_cycles/list_prices_by_product_id", operation_id="get_list_prices_by_product_id", summary="Retrieves the list prices associated with a specific product, identified by its unique product ID. This operation provides a comprehensive view of the historical list prices for the product, enabling users to track price changes over time.")
async def get_list_prices_by_product_id(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT listPrice FROM ProductlistPriceHistory WHERE ProductID = ?", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"list_prices": []}
    return {"list_prices": [row[0] for row in result]}

# Endpoint to check if actual cost matches planned cost in work order routing
@app.get("/v1/works_cycles/check_cost_match", operation_id="check_cost_match", summary="This operation verifies if the actual cost of a work order routing matches the planned cost. It returns a 'No' if the costs are equal and a 'Yes' if they differ. This can be used to identify discrepancies in cost planning and execution.")
async def check_cost_match():
    cursor.execute("SELECT CASE WHEN ActualCost = PlannedCost THEN 'No' ELSE 'Yes' END FROM WorkOrderRouting")
    result = cursor.fetchall()
    if not result:
        return {"result": []}
    return {"result": result}

# Endpoint to get thumbnail photo file name for a given product ID
@app.get("/v1/works_cycles/get_thumbnail_photo_file_name", operation_id="get_thumbnail_photo_file_name", summary="Retrieves the file name of the thumbnail photo associated with the specified product. The product is identified by its unique ID.")
async def get_thumbnail_photo_file_name(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T2.ThumbnailPhotoFileName FROM ProductProductPhoto AS T1 INNER JOIN ProductPhoto AS T2 ON T1.ProductPhotoID = T2.ProductPhotoID WHERE T1.ProductID = ?", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"thumbnail_photo_file_name": []}
    return {"thumbnail_photo_file_name": result}

# Endpoint to get distinct product names with list price greater than a specified value
@app.get("/v1/works_cycles/get_distinct_product_names_by_list_price", operation_id="get_distinct_product_names_by_list_price", summary="Retrieve a unique set of product names with a list price exceeding a specified minimum value. This operation filters products based on their list price and returns the distinct names of the products that meet the specified criteria.")
async def get_distinct_product_names_by_list_price(min_list_price: float = Query(..., description="Minimum list price")):
    cursor.execute("SELECT DISTINCT T2.Name FROM ProductlistPriceHistory AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.listPrice > ?", (min_list_price,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the product ID with the highest difference between list price and standard cost
@app.get("/v1/works_cycles/get_product_id_highest_price_difference", operation_id="get_product_id_highest_price_difference", summary="Retrieves the product ID associated with the highest discrepancy between its list price and standard cost. This operation compares historical pricing data to determine the product with the most significant price difference.")
async def get_product_id_highest_price_difference():
    cursor.execute("SELECT T1.ProductID FROM ProductlistPriceHistory AS T1 INNER JOIN ProductCostHistory AS T2 ON T1.ProductID = T2.ProductID ORDER BY T1.listPrice - T2.StandardCost DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product_id": []}
    return {"product_id": result[0]}

# Endpoint to get product names based on location ID, shelf, and bin
@app.get("/v1/works_cycles/get_product_names_by_location_shelf_bin", operation_id="get_product_names_by_location_shelf_bin", summary="Retrieves the names of products stored at a specific location, shelf, and bin. The operation filters the product inventory based on the provided location ID, shelf, and bin, and returns the corresponding product names.")
async def get_product_names_by_location_shelf_bin(location_id: int = Query(..., description="Location ID"), shelf: str = Query(..., description="Shelf"), bin: int = Query(..., description="Bin")):
    cursor.execute("SELECT T2.Name FROM ProductInventory AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.LocationID = ? AND T1.Shelf = ? AND T1.Bin = ?", (location_id, shelf, bin))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get location names based on work order ID
@app.get("/v1/works_cycles/get_location_names_by_work_order_id", operation_id="get_location_names_by_work_order_id", summary="Retrieves the names of all locations associated with a specific work order. The operation uses the provided work order ID to look up the corresponding locations in the database and returns their names.")
async def get_location_names_by_work_order_id(work_order_id: int = Query(..., description="Work Order ID")):
    cursor.execute("SELECT T2.Name FROM WorkOrderRouting AS T1 INNER JOIN Location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.WorkOrderID = ?", (work_order_id,))
    result = cursor.fetchall()
    if not result:
        return {"location_names": []}
    return {"location_names": result}

# Endpoint to get product IDs based on large photo file name pattern
@app.get("/v1/works_cycles/get_product_ids_by_large_photo_file_name", operation_id="get_product_ids_by_large_photo_file_name", summary="Retrieves the product IDs associated with large photos whose file names match the provided pattern. The pattern should include wildcard characters (%%) to specify the desired file name format. This operation uses an inner join between the ProductPhoto and ProductProductPhoto tables to identify the relevant product IDs.")
async def get_product_ids_by_large_photo_file_name(large_photo_file_name_pattern: str = Query(..., description="Pattern for large photo file name (use %% for wildcard)")):
    cursor.execute("SELECT T2.ProductID FROM ProductPhoto AS T1 INNER JOIN ProductProductPhoto AS T2 ON T1.ProductPhotoID = T2.ProductPhotoID WHERE T1.LargePhotoFileName LIKE ?", (large_photo_file_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": result}

# Endpoint to get product IDs based on product subcategory name
@app.get("/v1/works_cycles/get_product_ids_by_subcategory_name", operation_id="get_product_ids_by_subcategory_name", summary="Retrieves a list of product IDs associated with the specified product subcategory. The subcategory is identified by its name, which is provided as an input parameter.")
async def get_product_ids_by_subcategory_name(subcategory_name: str = Query(..., description="Product subcategory name")):
    cursor.execute("SELECT T2.ProductID FROM ProductSubcategory AS T1 INNER JOIN Product AS T2 ON T1.ProductSubcategoryID = T2.ProductSubcategoryID WHERE T1.Name = ?", (subcategory_name,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": result}

# Endpoint to get the percentage of list price for a given product name
@app.get("/v1/works_cycles/get_percentage_of_list_price", operation_id="get_percentage_of_list_price", summary="Retrieves the percentage of the list price for a specific product. The operation calculates the ratio of the current list price to the original list price for the product identified by the provided product name. The result is expressed as a percentage.")
async def get_percentage_of_list_price(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT 100 / T2.listPrice FROM Product AS T1 INNER JOIN ProductlistPriceHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"percentage_of_list_price": []}
    return {"percentage_of_list_price": result}

# Endpoint to get scrap reason names based on work order ID
@app.get("/v1/works_cycles/get_scrap_reason_names_by_work_order_id", operation_id="get_scrap_reason_names_by_work_order_id", summary="Retrieves the names of all scrap reasons associated with a specific work order. The operation requires the work order ID as input and returns a list of corresponding scrap reason names.")
async def get_scrap_reason_names_by_work_order_id(work_order_id: int = Query(..., description="Work Order ID")):
    cursor.execute("SELECT T2.Name FROM WorkOrder AS T1 INNER JOIN ScrapReason AS T2 ON T1.ScrapReasonID = T2.ScrapReasonID WHERE T1.WorkOrderID = ?", (work_order_id,))
    result = cursor.fetchall()
    if not result:
        return {"scrap_reason_names": []}
    return {"scrap_reason_names": result}

# Endpoint to get the standard cost of products with names containing a specific substring
@app.get("/v1/works_cycles/product_standard_cost_by_name", operation_id="get_product_standard_cost_by_name", summary="Retrieves the standard cost of products whose names contain a specified substring. This operation searches for products with names that include the provided substring and returns their respective standard costs. The substring parameter is used to filter the product names.")
async def get_product_standard_cost_by_name(name_substring: str = Query(..., description="Substring to search in product names")):
    cursor.execute("SELECT T2.StandardCost FROM Product AS T1 INNER JOIN ProductCostHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name LIKE ?", ('%' + name_substring + '%',))
    result = cursor.fetchall()
    if not result:
        return {"standard_costs": []}
    return {"standard_costs": [row[0] for row in result]}

# Endpoint to get the quantity of a specific product in a specific location
@app.get("/v1/works_cycles/product_quantity_by_location", operation_id="get_product_quantity_by_location", summary="Retrieves the quantity of a specific product in a given location. The operation requires the product's ID and the location's name as input parameters. It returns the quantity of the specified product available in the provided location.")
async def get_product_quantity_by_location(product_id: int = Query(..., description="ID of the product"), location_name: str = Query(..., description="Name of the location")):
    cursor.execute("SELECT T2.Quantity FROM Location AS T1 INNER JOIN ProductInventory AS T2 ON T1.LocationID = T2.LocationID WHERE T2.ProductID = ? AND T1.Name = ?", (product_id, location_name))
    result = cursor.fetchone()
    if not result:
        return {"quantity": []}
    return {"quantity": result[0]}

# Endpoint to get distinct product IDs with standard cost below the average
@app.get("/v1/works_cycles/product_ids_below_average_cost", operation_id="get_product_ids_below_average_cost", summary="Retrieves a list of unique product identifiers whose standard cost is less than the average cost of all products. This operation uses an inner join to combine data from the ProductCostHistory and Product tables, filtering for products with a standard cost below the average.")
async def get_product_ids_below_average_cost():
    cursor.execute("SELECT DISTINCT T2.ProductID FROM ProductCostHistory AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.StandardCost < ( SELECT SUM(StandardCost) / COUNT(ProductID) FROM Product )")
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get the count of products with a specific product photo ID
@app.get("/v1/works_cycles/product_count_by_photo_id", operation_id="get_product_count_by_photo_id", summary="Retrieves the total count of distinct products associated with a given product photo ID. The input parameter specifies the ID of the product photo for which the count is calculated.")
async def get_product_count_by_photo_id(product_photo_id: int = Query(..., description="ID of the product photo")):
    cursor.execute("SELECT COUNT(ProductID) FROM ProductProductPhoto WHERE ProductPhotoID != ?", (product_photo_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the duration of a work order in days
@app.get("/v1/works_cycles/work_order_duration", operation_id="get_work_order_duration", summary="Retrieves the total duration of a specific work order in days. The calculation is based on the difference between the actual start and end dates of the work order. The work order duration is determined by providing the unique work order ID.")
async def get_work_order_duration(work_order_id: int = Query(..., description="ID of the work order")):
    cursor.execute("SELECT 365 * (STRFTIME('%Y', ActualEndDate) - STRFTIME('%Y', ActualStartDate)) + 30 * (STRFTIME('%m', ActualEndDate) - STRFTIME('%m', ActualStartDate)) + STRFTIME('%d', ActualEndDate) - STRFTIME('%d', ActualStartDate) FROM WorkOrderRouting WHERE WorkOrderID = ?", (work_order_id,))
    result = cursor.fetchone()
    if not result:
        return {"duration": []}
    return {"duration": result[0]}

# Endpoint to get the product ID with the highest list price for a specific start date
@app.get("/v1/works_cycles/highest_list_price_product", operation_id="get_highest_list_price_product", summary="Retrieves the product ID associated with the highest list price for a given start date. The start date is provided in the 'YYYY%' format. This operation returns the product ID that has the highest list price on or after the specified start date.")
async def get_highest_list_price_product(start_date: str = Query(..., description="Start date in 'YYYY%' format")):
    cursor.execute("SELECT ProductID FROM ProductlistPriceHistory WHERE StartDate LIKE ? ORDER BY listPrice DESC LIMIT 1", (start_date + '%',))
    result = cursor.fetchone()
    if not result:
        return {"product_id": []}
    return {"product_id": result[0]}

# Endpoint to get the standard cost of a specific product
@app.get("/v1/works_cycles/product_standard_cost", operation_id="get_product_standard_cost", summary="Retrieves the standard cost associated with a specific product. The product is identified by its unique ID, which is provided as an input parameter. This operation returns the standard cost value from the product cost history.")
async def get_product_standard_cost(product_id: int = Query(..., description="ID of the product")):
    cursor.execute("SELECT StandardCost FROM ProductCostHistory WHERE ProductID = ?", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"standard_cost": []}
    return {"standard_cost": result[0]}

# Endpoint to get the organization level of employees with a specific job title
@app.get("/v1/works_cycles/employee_organization_level", operation_id="get_employee_organization_level", summary="Retrieves the organization level of employees who hold a specified job title. The job title is provided as an input parameter.")
async def get_employee_organization_level(job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT OrganizationLevel FROM Employee WHERE JobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"organization_levels": []}
    return {"organization_levels": [row[0] for row in result]}

# Endpoint to get the count of work orders where the end date is after the due date
@app.get("/v1/works_cycles/work_order_count_end_after_due", operation_id="get_work_order_count_end_after_due", summary="Retrieves the total number of work orders that have been completed after their respective due dates. This operation provides a quantitative measure of work order delays, which can be useful for tracking project progress and identifying potential bottlenecks.")
async def get_work_order_count_end_after_due():
    cursor.execute("SELECT COUNT(WorkOrderID) FROM WorkOrder WHERE EndDate > DueDate")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the standard cost and product number for a given product ID
@app.get("/v1/works_cycles/product_cost_history", operation_id="get_product_cost_history", summary="Retrieves the standard cost and product number associated with a specific product, identified by its unique product ID.")
async def get_product_cost_history(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T2.StandardCost, T2.ProductNumber FROM ProductCostHistory AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductID = ?", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of distinct product IDs for a given large photo file name
@app.get("/v1/works_cycles/product_photo_count", operation_id="get_product_photo_count", summary="Retrieves the total number of unique products associated with a specified large photo file name. This operation is useful for determining the product diversity linked to a particular large photo file.")
async def get_product_photo_count(large_photo_file_name: str = Query(..., description="Large photo file name")):
    cursor.execute("SELECT COUNT(DISTINCT T2.ProductID) FROM ProductPhoto AS T1 INNER JOIN ProductProductPhoto AS T2 ON T1.ProductPhotoID = T2.ProductPhotoID WHERE T1.LargePhotoFileName = ?", (large_photo_file_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the location, shelf, and bin for products with a given name
@app.get("/v1/works_cycles/product_inventory_location", operation_id="get_product_inventory_location", summary="Retrieves the location, shelf, and bin details for products that match the provided product name. The endpoint performs a search using the specified product name and returns the corresponding inventory details.")
async def get_product_inventory_location(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT T2.LocationID, T2.Shelf, T2.Bin FROM Product AS T1 INNER JOIN ProductInventory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name LIKE ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the product category name for a given subcategory name
@app.get("/v1/works_cycles/product_category_name", operation_id="get_product_category_name", summary="Retrieves the name of the product category associated with the provided subcategory name. This operation returns the category name by querying the product subcategory and category tables using the given subcategory name as a filter.")
async def get_product_category_name(subcategory_name: str = Query(..., description="Subcategory name")):
    cursor.execute("SELECT T2.Name FROM ProductSubcategory AS T1 INNER JOIN ProductCategory AS T2 ON T1.ProductCategoryID = T2.ProductCategoryID WHERE T1.Name = ?", (subcategory_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the work order IDs for a given scrap reason name
@app.get("/v1/works_cycles/work_order_ids_by_scrap_reason", operation_id="get_work_order_ids_by_scrap_reason", summary="Retrieves the work order identifiers associated with a specific scrap reason. The operation filters work orders based on the provided scrap reason name and returns their unique identifiers.")
async def get_work_order_ids_by_scrap_reason(scrap_reason_name: str = Query(..., description="Scrap reason name")):
    cursor.execute("SELECT T2.WorkOrderID FROM ScrapReason AS T1 INNER JOIN WorkOrder AS T2 ON T1.ScrapReasonID = T2.ScrapReasonID WHERE T1.Name = ?", (scrap_reason_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the difference between list price and standard cost for a given product ID
@app.get("/v1/works_cycles/price_difference", operation_id="get_price_difference", summary="Retrieves the price difference between the list price and the standard cost for a specific product. The product is identified by its unique ID, which is provided as an input parameter. This operation returns the calculated difference, offering insights into the product's pricing dynamics.")
async def get_price_difference(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T1.listPrice - T2.StandardCost FROM ProductlistPriceHistory AS T1 INNER JOIN ProductCostHistory AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductID = ?", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the first and last name for a given email address
@app.get("/v1/works_cycles/person_by_email", operation_id="get_person_by_email", summary="Retrieves the first and last name of a person associated with the provided email address. The operation searches for the email address in the EmailAddress table and returns the corresponding first and last name from the Person table.")
async def get_person_by_email(email_address: str = Query(..., description="Email address")):
    cursor.execute("SELECT T2.FirstName, T2.LastName FROM EmailAddress AS T1 INNER JOIN Person AS T2 ON T1.BusinessEntityID = T2.BusinessEntityID WHERE T1.EmailAddress = ?", (email_address,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the location name for a given product ID
@app.get("/v1/works_cycles/location_by_product_id", operation_id="get_location_by_product_id", summary="Retrieves the name of the location where a specific product is stored. The product is identified by its unique ID.")
async def get_location_by_product_id(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT T2.Name FROM ProductInventory AS T1 INNER JOIN Location AS T2 ON T1.LocationID = T2.LocationID WHERE T1.ProductID = ?", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the product name for a given work order ID
@app.get("/v1/works_cycles/product_by_work_order_id", operation_id="get_product_by_work_order_id", summary="Retrieves the name of the product associated with the specified work order ID. This operation fetches the product name from the Product table by joining it with the WorkOrder table using the ProductID as the common key. The work order ID is used to filter the results.")
async def get_product_by_work_order_id(work_order_id: int = Query(..., description="Work order ID")):
    cursor.execute("SELECT T2.Name FROM WorkOrder AS T1 INNER JOIN Product AS T2 ON T1.ProductID = T2.ProductID WHERE T1.WorkOrderID = ?", (work_order_id,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get work order IDs for a specific product name
@app.get("/v1/works_cycles/work_order_ids_by_product_name", operation_id="get_work_order_ids_by_product_name", summary="Retrieves the work order IDs associated with a specific product name. The operation filters work orders based on the provided product name and returns the corresponding work order IDs.")
async def get_work_order_ids_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.WorkOrderID FROM Product AS T1 INNER JOIN WorkOrder AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"work_order_ids": []}
    return {"work_order_ids": [row[0] for row in result]}

# Endpoint to get the average percentage difference between list price and standard cost
@app.get("/v1/works_cycles/avg_percentage_difference_list_price_standard_cost", operation_id="get_avg_percentage_difference", summary="Retrieves the average percentage difference between the list price and standard cost of products. This operation calculates the average percentage difference by comparing the list price history with the standard cost history of each product. The result provides a single value representing the overall average percentage difference.")
async def get_avg_percentage_difference():
    cursor.execute("SELECT AVG((T1.listPrice - T2.StandardCost) * 100 / T2.StandardCost) FROM ProductlistPriceHistory AS T1 INNER JOIN ProductCostHistory AS T2 ON T1.ProductID = T2.ProductID")
    result = cursor.fetchone()
    if not result:
        return {"average_percentage_difference": []}
    return {"average_percentage_difference": result[0]}

# Endpoint to get the percentage of work orders routed through a specific location type
@app.get("/v1/works_cycles/percentage_work_orders_by_location_type", operation_id="get_percentage_work_orders_by_location_type", summary="Retrieves the percentage of work orders that have been routed through a specified location type. This operation calculates the ratio of work orders associated with the given location type to the total number of work orders, providing a quantitative measure of work order distribution across different location types.")
async def get_percentage_work_orders_by_location_type(location_type: str = Query(..., description="Type of the location")):
    cursor.execute("SELECT 100.0 * SUM(CASE WHEN T1.Name = ? THEN 1 ELSE 0 END) / COUNT(T2.WorkOrderID) FROM Location AS T1 INNER JOIN WorkOrderRouting AS T2 ON T1.LocationID = T2.LocationID", (location_type,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/works_cycles/avg_standard_cost_by_product_number?product_number=CA-1098",
    "/v1/works_cycles/product_names_start_dates_no_end_date",
    "/v1/works_cycles/product_names_by_cost_difference?cost_difference=80",
    "/v1/works_cycles/product_names_quantities_by_shopping_cart?shopping_cart_id=14951",
    "/v1/works_cycles/product_names_by_quantity?quantity=5",
    "/v1/works_cycles/distinct_product_names_by_class_transaction_type?product_class=L&transaction_type=P",
    "/v1/works_cycles/distinct_product_names_list_prices_by_quantity?quantity=10000",
    "/v1/works_cycles/product_name_lowest_quantity_by_class?product_class=H",
    "/v1/works_cycles/transaction_count_by_product_line?product_line=M",
    "/v1/works_cycles/total_profit_by_shopping_cart?shopping_cart_id=20621",
    "/v1/works_cycles/product_names_list_prices_by_class?class_name=H",
    "/v1/works_cycles/most_common_product_line_finished_goods?finished_goods_flag=1",
    "/v1/works_cycles/product_reviews_by_reviewer_name?reviewer_name_pattern=J%",
    "/v1/works_cycles/product_with_lowest_rating",
    "/v1/works_cycles/distinct_product_names_by_price_difference?price_difference=100",
    "/v1/works_cycles/product_reviews_by_product_line?product_line=R",
    "/v1/works_cycles/product_count_avg_rating?product_name=HL%20Mountain%20Pedal",
    "/v1/works_cycles/product_names_rejected_equals_received",
    "/v1/works_cycles/product_highest_line_total_zero_rejected?rejected_qty=0",
    "/v1/works_cycles/product_names_lines_by_order_quantity?order_qty=4999",
    "/v1/works_cycles/total_order_quantity_product_line?product_line=T",
    "/v1/works_cycles/product_line_total_highest_order_quantity_unit_price?class_type=L",
    "/v1/works_cycles/product_highest_cost_difference",
    "/v1/works_cycles/product_highest_rating_profit_margin",
    "/v1/works_cycles/currency_pair_highest_average_rate",
    "/v1/works_cycles/order_quantity_highest_unit_price",
    "/v1/works_cycles/sales_territory_highest_sales_last_year?country_region_code=US&name1=Northwest&name2=Southeast",
    "/v1/works_cycles/employee_names_job_title_document_level_status?job_title=Document%20Control%20Manager&document_level=1&status=2",
    "/v1/works_cycles/customer_with_highest_subtotal",
    "/v1/works_cycles/total_price_special_offer?description=Volume%20Discount%2011%20to%2014&special_offer_id=2&product_id=716&sales_order_id=46625",
    "/v1/works_cycles/count_products_manufacturing_criteria?make_flag=1&days_to_manufacture=1&bom_level=4&reorder_point=600",
    "/v1/works_cycles/highest_bonus_country_region?country_region_code=CA",
    "/v1/works_cycles/product_name_lowest_rating",
    "/v1/works_cycles/count_employees_department_date_range?start_date_min=2009-01-01&start_date_max=2010-01-01&department_name=Production",
    "/v1/works_cycles/highest_paid_employee_marital_gender?marital_status=S&gender=F",
    "/v1/works_cycles/employee_details_job_title?job_title=Vice%20President%20of%20Engineering",
    "/v1/works_cycles/count_current_employees_pay_rate?current_flag=1&rate=30",
    "/v1/works_cycles/most_recent_department",
    "/v1/works_cycles/employee_details_by_person_type?person_type=SP",
    "/v1/works_cycles/pay_frequency_least_sick_leave",
    "/v1/works_cycles/job_title_lowest_pay_rate",
    "/v1/works_cycles/employee_count_by_department?department_name=Finance",
    "/v1/works_cycles/highest_lowest_profit_margins",
    "/v1/works_cycles/vendor_details_by_credit_rating?credit_rating=3",
    "/v1/works_cycles/count_addresses_with_second_line",
    "/v1/works_cycles/postal_code_most_recent_address",
    "/v1/works_cycles/longest_duration_bill_of_materials",
    "/v1/works_cycles/count_bill_of_materials_no_end_date",
    "/v1/works_cycles/unit_measure_code_highest_per_assembly_qty?limit=1",
    "/v1/works_cycles/count_documents_null_summary",
    "/v1/works_cycles/document_titles_by_status?status=1",
    "/v1/works_cycles/employee_details_by_document_status?status=2",
    "/v1/works_cycles/pay_frequency_earliest_birthdate?limit=1",
    "/v1/works_cycles/count_employees_marital_status_highest_pay_frequency?marital_status=M&limit=1",
    "/v1/works_cycles/rate_most_recent_hire_date?limit=1",
    "/v1/works_cycles/count_employees_gender_rate?gender=M&rate=40",
    "/v1/works_cycles/highest_rate_salaried_employees?salaried_flag=1&limit=1",
    "/v1/works_cycles/vacation_hours_highest_rate_employee?limit=1",
    "/v1/works_cycles/highest_rate_by_vacation_hours",
    "/v1/works_cycles/count_employees_sick_leave_rate?sick_leave_hours=10&rate=35",
    "/v1/works_cycles/count_employees_current_flag_gender_pay_frequency?current_flag=1&gender=M&pay_frequency=2",
    "/v1/works_cycles/count_employees_gender_person_type?gender=M&person_type=SP",
    "/v1/works_cycles/youngest_employee_person_type",
    "/v1/works_cycles/lowest_rate_employee_name_style",
    "/v1/works_cycles/count_employees_name_style_marital_status?name_style=0&marital_status=M",
    "/v1/works_cycles/count_employees_email_promotion_sick_leave?email_promotion=1&sick_leave_hours=10",
    "/v1/works_cycles/employee_ids_email_promotion_vacation_hours?email_promotion=1&vacation_hours=20",
    "/v1/works_cycles/youngest_employee_additional_contact_info?person_type=SP",
    "/v1/works_cycles/employee_first_names?name_style=0&gender=M",
    "/v1/works_cycles/current_employee_count?current_flag=1&title=Mr.",
    "/v1/works_cycles/highest_paid_employee_demographics?marital_status=M",
    "/v1/works_cycles/employee_suffix_most_sick_leave?person_type=SP",
    "/v1/works_cycles/employee_count_marital_name_style_highest_rate?marital_status=M&name_style=1",
    "/v1/works_cycles/current_employee_count_email_promotion?current_flag=1&email_promotion=1",
    "/v1/works_cycles/person_credit_card_ids?person_type=SC",
    "/v1/works_cycles/average_vacation_hours?gender=M&person_type=EM",
    "/v1/works_cycles/pay_rate_difference?email_promotion=2&marital_status=M",
    "/v1/works_cycles/proportion_person_type?person_type=SC&name_style=0&marital_status=M",
    "/v1/works_cycles/employee_vacation_hours_percentage?vacation_hours=20&current_flag=1&sick_leave_hours=10",
    "/v1/works_cycles/average_last_receipt_cost?average_lead_time=60",
    "/v1/works_cycles/average_actual_cost?transaction_type=P&start_date=2012-01-01&end_date=2012-07-01",
    "/v1/works_cycles/employee_marital_status_percentage?marital_status=M&hire_year=2009&gender=M",
    "/v1/works_cycles/email_promotion_percentage?email_promotion=2&person_type=SC&first_name=Mary",
    "/v1/works_cycles/distinct_product_ids?price_difference=0",
    "/v1/works_cycles/average_total_due?status=2",
    "/v1/works_cycles/sales_order_percentage?order_qty=3&unit_price_discount=0.2",
    "/v1/works_cycles/salesperson_business_entity_ids?sales_ytd=SalesLastYear+SalesLastyear*0.6&bonus=3000",
    "/v1/works_cycles/address_type_counts?address_type_1=Home&address_type_2=Shipping",
    "/v1/works_cycles/customer_ids_by_unit_price_order_qty?unit_price=35&order_qty=32",
    "/v1/works_cycles/business_entity_ids_by_credit_card_details?card_type=ColonialVoice&exp_month=3&exp_year=2005",
    "/v1/works_cycles/credit_ratings_by_product_vendor_details?standard_price=18.9900&average_lead_time=16&last_receipt_date=2011-08-27",
    "/v1/works_cycles/product_count_by_name_patterns?name_pattern1=%25accessories%20%25&name_pattern2=%25components%25",
    "/v1/works_cycles/latest_job_title_by_department_id?department_id=12",
    "/v1/works_cycles/location_count_by_name?location_name=Subassembly",
    "/v1/works_cycles/total_scrapped_qty_by_scrap_reason?scrap_reason_name=Trim%20length%20too%20long",
    "/v1/works_cycles/total_order_qty_by_status?status=1",
    "/v1/works_cycles/distinct_product_count_by_order_qty_discount?order_qty=2&unit_price_discount=0",
    "/v1/works_cycles/distinct_transaction_types?size=62&color=Yellow&safety_stock_level=500",
    "/v1/works_cycles/product_subcategory_names?color=Grey",
    "/v1/works_cycles/heaviest_product_end_date?weight_unit_measure_code=G",
    "/v1/works_cycles/average_order_quantity?ship_method_id=5",
    "/v1/works_cycles/highest_sales_growth_state",
    "/v1/works_cycles/employee_count_above_pay_rate?pay_rate_threshold=50",
    "/v1/works_cycles/person_name?business_entity_id=1&person_type=EM",
    "/v1/works_cycles/vendor_name?business_entity_id=1492",
    "/v1/works_cycles/product_vendor_count?min_order_qty=500&max_order_qty=15000",
    "/v1/works_cycles/department_name?group_name=Executive%20General%20and%20Administration",
    "/v1/works_cycles/persons/last_names_by_type_and_middle_name?person_type=EM&middle_name_prefix=C",
    "/v1/works_cycles/product_vendors/count_distinct_business_entities?average_lead_time=25",
    "/v1/works_cycles/product_cost_history/product_ids_by_standard_cost?limit=3",
    "/v1/works_cycles/products/count_by_finished_goods_flag_and_color?finished_goods_flag=0&color=Black",
    "/v1/works_cycles/employees/job_titles_by_sick_leave_hours?limit=3",
    "/v1/works_cycles/addresses/address_lines_by_id?address_id=11906",
    "/v1/works_cycles/product_vendors/cost_difference_by_business_entity?business_entity_id=1580",
    "/v1/works_cycles/products/price_difference_by_product_id?product_id=740",
    "/v1/works_cycles/product_reviews/reviewer_names_by_rating?rating=5",
    "/v1/works_cycles/vendors/business_entity_ids_by_highest_credit_rating",
    "/v1/works_cycles/contact_type_name?business_entity_id=332",
    "/v1/works_cycles/business_entity_ids_by_phone_type?phone_number_type=Cell&limit=3",
    "/v1/works_cycles/currency_names_by_country?country_name=Brazil",
    "/v1/works_cycles/average_lead_time_vendor_name?business_entity_id=1496",
    "/v1/works_cycles/address_count_difference_by_cities?city1=Bothell&city2=Kenmore",
    "/v1/works_cycles/product_model_ids_by_description_culture?description=Chromoly%20steel%25&culture_name=English",
    "/v1/works_cycles/sales_person_territory_by_commission?commission_pct=0.018",
    "/v1/works_cycles/phone_number_types_ordered",
    "/v1/works_cycles/most_common_contact_type",
    "/v1/works_cycles/email_addresses_by_person_type?person_type=SP",
    "/v1/works_cycles/job_titles_by_first_name?first_name=Suchitra",
    "/v1/works_cycles/employee_count_by_person_type_marital_status?person_type=EM&marital_status=S",
    "/v1/works_cycles/culture_count_difference?culture1=English&culture2=Arabic",
    "/v1/works_cycles/address_line_by_business_entity_id?business_entity_id=1",
    "/v1/works_cycles/business_entity_ids_by_city?city=Duvall",
    "/v1/works_cycles/culture_percentage?culture_name=Thai",
    "/v1/works_cycles/employee_gender_percentage?gender=M&person_type=EM",
    "/v1/works_cycles/state_province_details_by_address_id?address_id=15873",
    "/v1/works_cycles/address_details_by_business_entity_id?business_entity_id=24",
    "/v1/works_cycles/credit_card_exp_year?credit_card_id=9648",
    "/v1/works_cycles/business_entity_id_by_name?first_name=Emma&last_name=Harris",
    "/v1/works_cycles/location_id_by_name?name=Debur%20and%20Polish",
    "/v1/works_cycles/department_id_by_group_name?group_name=Sales%20and%20Marketing",
    "/v1/works_cycles/business_entity_id_by_sales_last_year?sales_last_year=1635823.3967",
    "/v1/works_cycles/shift_start_time?shift_id=2",
    "/v1/works_cycles/ship_base_by_name?name=OVERSEAS%20-%20DELUXE",
    "/v1/works_cycles/culture_name_by_id?culture_id=fr",
    "/v1/works_cycles/currency_code_by_name?name=Mauritius%20Rupee",
    "/v1/works_cycles/phone_number_type_id_by_name?name=Cell",
    "/v1/works_cycles/employee_birth_date_by_hire_date?hire_date=2008-12-07",
    "/v1/works_cycles/product_name_by_id?product_id=793",
    "/v1/works_cycles/unit_measure_codes_by_product_id?product_id=762",
    "/v1/works_cycles/address_lines_by_business_entity_id?business_entity_id=4",
    "/v1/works_cycles/unit_measure_names_by_component_id?component_id=494",
    "/v1/works_cycles/employee_count_by_job_title_birth_date?job_title=Document%20Control%20Assistant&birth_date=1975-12-25",
    "/v1/works_cycles/product_list_price_by_cost_difference_start_date?cost_difference=21.9037&start_date=2012-10-01",
    "/v1/works_cycles/product_thumbnail_photo_by_id?product_id=1",
    "/v1/works_cycles/password_hash_length?first_name=Catherine&last_name=Ward",
    "/v1/works_cycles/product_review_rating?reviewer_name=Jill&product_name=HL%20Mountain%20Pedal",
    "/v1/works_cycles/product_cost_difference?product_name=Freewheel",
    "/v1/works_cycles/salesperson_bonus?first_name=Rachel&last_name=Valdez",
    "/v1/works_cycles/sales_tax_rate_count?state_province_name=Quebec",
    "/v1/works_cycles/person_email_address?first_name=Kevin&middle_name=A&last_name=Wright",
    "/v1/works_cycles/country_region_code?country_name=France&is_only_state_province_flag=1",
    "/v1/works_cycles/transaction_type?product_name=HL%20Road%20Frame%20-%20Black%2C%2048&transaction_date=2013-07-31",
    "/v1/works_cycles/transaction_type_archive?product_name=LL%20Road%20Handlebars&transaction_date=2012-11-03",
    "/v1/works_cycles/vendor_credit_rating?rowguid=33671A4E-DF2B-4879-807B-E3F930DD5C0C",
    "/v1/works_cycles/preferred_vendor_status?rowguid=684F328D-C185-43B9-AF9A-37ACC680D2AF",
    "/v1/works_cycles/vendor_active_flag?person_id=2054",
    "/v1/works_cycles/phone_number_by_name?first_name=Gerald&last_name=Patel",
    "/v1/works_cycles/phone_number_type_name?business_entity_id=13626",
    "/v1/works_cycles/job_title_by_name?first_name=Lynn&middle_name=N&last_name=Tsoflias",
    "/v1/works_cycles/count_product_subcategories?product_category_name=Bikes",
    "/v1/works_cycles/percentage_documents_with_summary?job_title=Document%20Control%20Assistant&hire_date=2009-01-22",
    "/v1/works_cycles/price_difference_percentage?product_name=HL%20Grip%20Tape",
    "/v1/works_cycles/percentage_phone_numbers_by_type?phone_number_type=Cell",
    "/v1/works_cycles/product_assembly_id?unit_measure_code=EA&bom_level=2&per_assembly_qty=10",
    "/v1/works_cycles/count_work_order_routing_by_hours?actual_resource_hrs=2",
    "/v1/works_cycles/count_work_order_routing_by_location?location_id=40",
    "/v1/works_cycles/sum_per_assembly_qty_by_unit_measure?unit_measure_code1=EA&unit_measure_code2=IN&unit_measure_code3=OZ",
    "/v1/works_cycles/product_ids_not_in_work_orders",
    "/v1/works_cycles/product_ids_by_transaction_type?transaction_type=P",
    "/v1/works_cycles/employee_names_by_job_title?job_title=Accountant",
    "/v1/works_cycles/job_title_by_employee_name?first_name=Ken&middle_name=J&last_name=S\u00e1nchez",
    "/v1/works_cycles/count_employees_by_email_promotion_gender?email_promotion=0&gender=M",
    "/v1/works_cycles/top_salesperson_by_quota_year?quota_year=2013",
    "/v1/works_cycles/count_employees_by_person_type_marital_status?person_type=EM&marital_status=M",
    "/v1/works_cycles/total_sick_leave_hours_by_email_promotion?email_promotion=0",
    "/v1/works_cycles/employee_count_by_person_type_hire_date?person_type=SP&hire_year=2010",
    "/v1/works_cycles/top_salesperson_by_sales_quota?year=2011",
    "/v1/works_cycles/employee_count_by_first_name_marital_status_organization_level?first_name=Alex&marital_status=S&organization_level=1",
    "/v1/works_cycles/employee_last_name_job_title_by_document_title?document_title=Crank%20Arm%20and%20Tire%20Maintenance",
    "/v1/works_cycles/employee_count_without_suffix",
    "/v1/works_cycles/distinct_salesperson_ids_by_territory_year?territory_id=1&year=2013",
    "/v1/works_cycles/employee_last_names_by_business_entity_ids?business_entity_id_1=212&business_entity_id_2=274",
    "/v1/works_cycles/employee_email_addresses?gender=F&marital_status=S",
    "/v1/works_cycles/product_colors?product_id1=709&product_id2=937&product_id3=798",
    "/v1/works_cycles/sales_quota_sum?business_entity_id=275&year=2013",
    "/v1/works_cycles/distinct_business_entity_ids?year=2013",
    "/v1/works_cycles/female_employee_percentage?email_promotion=1",
    "/v1/works_cycles/married_to_single_ratio?person_type=EM",
    "/v1/works_cycles/bonus_and_percentage?year=2013",
    "/v1/works_cycles/credit_card_count?card_type=vista",
    "/v1/works_cycles/store_name_by_salesperson_id?salesperson_id=277",
    "/v1/works_cycles/sum_quantity_by_date_type_product?transaction_date=2013-08%25&transaction_type=S&product_id=989",
    "/v1/works_cycles/credit_cards_expiring_before_year?exp_year=2007",
    "/v1/works_cycles/employee_pay_rate_by_hire_age?hire_age=20",
    "/v1/works_cycles/sales_territory_by_salesperson?business_entity_id=277",
    "/v1/works_cycles/purchase_order_ids_by_credit_rating?credit_rating=5",
    "/v1/works_cycles/phone_number_type_by_phone_number?phone_number=114-555-0100",
    "/v1/works_cycles/total_freight_by_ship_method_order_date?ship_method_name=cargo%20transport%205&order_date=2011-12-14",
    "/v1/works_cycles/count_total_due_by_person_details?first_name=David&middle_name=R&last_name=Campbell&person_type=SP",
    "/v1/works_cycles/email_addresses_by_job_title?job_title=Facilities%20Manager",
    "/v1/works_cycles/count_customers_in_sales_territory?territory_name=Canada",
    "/v1/works_cycles/ship_to_address_ids_by_sales_order?sales_order_id=43873",
    "/v1/works_cycles/count_employees_below_average_pay?job_title=Production%20Technician%25",
    "/v1/works_cycles/percentage_sales_orders_in_territory?territory_name=United%20Kingdom",
    "/v1/works_cycles/modified_date_by_phone_number?phone_number=1%20(11)%20500%20555-0143",
    "/v1/works_cycles/top_salesperson_by_sales_ytd",
    "/v1/works_cycles/vendor_names_by_active_flag?active_flag=1",
    "/v1/works_cycles/territory_with_most_customers_before_date?modified_date=2014-12-09",
    "/v1/works_cycles/total_due_on_date?order_date=2013-05-29",
    "/v1/works_cycles/most_common_first_name_by_type?person_type=VC",
    "/v1/works_cycles/order_quantity_count_by_product_name?product_name=Minipump",
    "/v1/works_cycles/business_entity_by_card_number?card_number=11113366963373",
    "/v1/works_cycles/address_by_business_entity_and_type?business_entity_id=5555&address_type=Home",
    "/v1/works_cycles/product_names_by_special_offer?special_offer_id=15",
    "/v1/works_cycles/credit_card_id_by_person_name?first_name=Michelle&middle_name=E&last_name=Cox",
    "/v1/works_cycles/sales_reason_names_by_order_id?sales_order_id=51883",
    "/v1/works_cycles/card_number_by_sales_order_id?sales_order_id=45793",
    "/v1/works_cycles/business_entity_ids_by_territory_and_country?territory_name=Northwest&country_region_code=US",
    "/v1/works_cycles/highest_paid_employee_by_job_title?job_title=Production%20Technician%25",
    "/v1/works_cycles/salesperson_names_by_territory?territory_id=9",
    "/v1/works_cycles/special_offer_description_by_product?product_id=762",
    "/v1/works_cycles/average_pay_rate_by_gender",
    "/v1/works_cycles/percentage_employees_by_shift?shift_name=Night",
    "/v1/works_cycles/employee_count_by_marital_status_birth_year_gender?marital_status=M&birth_year=1960&gender=M",
    "/v1/works_cycles/top_expensive_products?limit=5",
    "/v1/works_cycles/earliest_hire_date_by_job_title?job_title=Accountant",
    "/v1/works_cycles/most_common_job_title_by_hire_year?hire_year=2007",
    "/v1/works_cycles/highest_standard_price_difference",
    "/v1/works_cycles/department_count_by_person_and_year_range?first_name=Sheela&last_name=Word&start_year=2011&end_year=2012",
    "/v1/works_cycles/average_age_by_person_type?year=2009&person_type=SP",
    "/v1/works_cycles/most_popular_department_group",
    "/v1/works_cycles/oldest_employee_age_and_rate_by_job_title?year=2015&job_title=Marketing%20Specialist",
    "/v1/works_cycles/lowest_unit_price_vendor",
    "/v1/works_cycles/highest_total_due_employee",
    "/v1/works_cycles/nth_highest_pay_rate_employee?offset=9",
    "/v1/works_cycles/highest_rated_product_by_reviewer?reviewer_name=John%20Smith",
    "/v1/works_cycles/employee_pay_rates?offset=0&limit=5",
    "/v1/works_cycles/vendor_names_by_order_quantity?min_order_qty=500&max_order_qty=750&offset=9&limit=1",
    "/v1/works_cycles/years_since_hire_date?status=1",
    "/v1/works_cycles/days_to_manufacture?rating=5&product_class=M&limit=1",
    "/v1/works_cycles/average_pay_rate?gender=F",
    "/v1/works_cycles/distinct_vendor_names?make_flag=0&style=W&subcategory_name=Tights",
    "/v1/works_cycles/pay_frequency?job_title=Scheduling%20Assistant&limit=1",
    "/v1/works_cycles/product_details?limit=1",
    "/v1/works_cycles/employee_longest_tenure?limit=1",
    "/v1/works_cycles/highest_profit_margin_product?limit=1",
    "/v1/works_cycles/highest_outstanding_order_quantity?limit=1",
    "/v1/works_cycles/vendor_details_by_product?product_id=843",
    "/v1/works_cycles/count_salespersons_by_bonus?bonus=0",
    "/v1/works_cycles/count_sales_tax_rates_by_name?name_pattern=%25+%25",
    "/v1/works_cycles/highest_actual_cost_by_transaction_type?transaction_type=P&limit=1",
    "/v1/works_cycles/sales_order_status_by_highest_freight?limit=1",
    "/v1/works_cycles/count_products_by_on_order_quantity?on_order_qty=0",
    "/v1/works_cycles/top_n_cost_difference?limit=1",
    "/v1/works_cycles/top_reviewer_names?limit=1",
    "/v1/works_cycles/top_product_by_reviews?limit=1",
    "/v1/works_cycles/product_count_make_flag_rating?make_flag=0&rating=5",
    "/v1/works_cycles/most_common_comment_finished_goods?finished_goods_flag=1&limit=1",
    "/v1/works_cycles/product_names_finished_make_comment_count?finished_goods_flag=1&make_flag=1&comment_count=10",
    "/v1/works_cycles/product_names_finished_comment_count?finished_goods_flag=1&comment_count=10",
    "/v1/works_cycles/reviewer_names_class_style_rating?product_class=M&product_style=W&rating=5",
    "/v1/works_cycles/top_product_class_by_reviews?limit=1",
    "/v1/works_cycles/email_addresses_by_product_class?product_class=H",
    "/v1/works_cycles/product_names_by_list_price_count?list_price_count=3",
    "/v1/works_cycles/product_description_by_name?product_name=Headset%20Ball%20Bearings",
    "/v1/works_cycles/highest_standard_price_by_product_name?product_name=Hex%20Nut%205",
    "/v1/works_cycles/most_common_standard_price_by_product_name?product_name=Hex%20Nut%205",
    "/v1/works_cycles/count_distinct_vendors_by_product_name_and_credit_rating?product_name=Hex%20Nut%205&credit_rating=1",
    "/v1/works_cycles/purchasing_web_service_url_by_product_name?product_name=Hex%20Nut%205",
    "/v1/works_cycles/vendor_name_by_product_name_and_lowest_standard_price?product_name=Hex%20Nut%205",
    "/v1/works_cycles/count_products_by_preferred_vendor_and_class?preferred_vendor_status=1&product_class=M",
    "/v1/works_cycles/list_price_history_by_product_name?product_name=LL%20Fork",
    "/v1/works_cycles/count_products_in_product_line_with_name_count?product_line=M&name_count=2",
    "/v1/works_cycles/sum_product_reviews_by_product_line?product_line=M&min_reviews=1",
    "/v1/works_cycles/email_address_by_product_name?product_name=HL%20Mountain%20Pedal",
    "/v1/works_cycles/count_products_by_on_order_qty?on_order_qty=0&null_on_order_qty=true",
    "/v1/works_cycles/product_names_by_make_flag_and_on_order_qty?make_flag=0&on_order_qty=0&null_on_order_qty=true",
    "/v1/works_cycles/sum_product_reviews_by_product_line_and_finished_goods_flag?product_line=M&finished_goods_flag=1",
    "/v1/works_cycles/average_standard_price_by_product_name?product_name=Hex%20Nut%205",
    "/v1/works_cycles/highest_average_rating_product_by_product_line?product_line=M",
    "/v1/works_cycles/highest_total_rating_product_by_make_flag?make_flag=1",
    "/v1/works_cycles/person_names_by_type?person_type=EM",
    "/v1/works_cycles/full_name_by_last_name_and_type?last_name=Anderson&person_type=IN",
    "/v1/works_cycles/employee_department_history_end_date",
    "/v1/works_cycles/employee_hired_after_year_salaried?hire_year=2007&salaried_flag=1",
    "/v1/works_cycles/employee_marital_status_vacation_hours?marital_status=M&vacation_hours=20",
    "/v1/works_cycles/oldest_employee_in_shift?shift_id=3",
    "/v1/works_cycles/employee_hired_year_department?hire_year=2009&department_name=Shipping%20and%20Receiving",
    "/v1/works_cycles/most_recently_hired_employee",
    "/v1/works_cycles/average_pay_rate_recent_change",
    "/v1/works_cycles/highest_paid_employee_excluding_job_title?job_title=Chief%20Executive%20Officer",
    "/v1/works_cycles/vendor_shortest_lead_time_product?product_id=319",
    "/v1/works_cycles/products_supplied_by_vendor?vendor_name=Australia%20Bike%20Retailer",
    "/v1/works_cycles/vendor_name_by_product_id?product_id=342",
    "/v1/works_cycles/employee_rate_variation?first_name=Rob&last_name=Walters",
    "/v1/works_cycles/avg_years_of_service_by_department?department_name=Research%20and%20Development",
    "/v1/works_cycles/avg_employee_age",
    "/v1/works_cycles/percentage_employees_by_job_title?job_title=Sales%20Representative",
    "/v1/works_cycles/employee_contact_details_by_job_title?job_title=Marketing%20Manager",
    "/v1/works_cycles/person_names_by_credit_card_type?card_type=Distinguish",
    "/v1/works_cycles/sales_territory_by_store_name?store_name=Area%20Bike%20Accessories",
    "/v1/works_cycles/store_person_details_by_sales_territory?territory_name=France",
    "/v1/works_cycles/product_vendor_by_sell_start_year?sell_start_year=2013",
    "/v1/works_cycles/oldest_employee_by_gender_marital_status?gender=M&marital_status=M",
    "/v1/works_cycles/shortest_lead_time_vendor?product_id=348",
    "/v1/works_cycles/most_sick_leave_hours_by_birth_year?birth_year=1970",
    "/v1/works_cycles/youngest_employee_age_department",
    "/v1/works_cycles/distinct_product_category_ids",
    "/v1/works_cycles/sales_difference?business_entity_id=288",
    "/v1/works_cycles/top_salespersons_last_year",
    "/v1/works_cycles/top_territories_by_sales?limit=3",
    "/v1/works_cycles/special_offer_types_by_category?category=Reseller",
    "/v1/works_cycles/highest_discount_by_type?type=Excess%20Inventory&limit=1",
    "/v1/works_cycles/cost_difference_by_product?product_id=818",
    "/v1/works_cycles/bom_count_by_level?bom_level=2",
    "/v1/works_cycles/document_count_by_status?status=2",
    "/v1/works_cycles/sales_tax_rate_ids_by_name?name_pattern=%25+%25",
    "/v1/works_cycles/top_transaction_by_type?transaction_type=W&limit=1",
    "/v1/works_cycles/vendor_names_by_status?preferred_vendor_status=0&limit=3",
    "/v1/works_cycles/vendor_count_by_status_and_active?preferred_vendor_status=0&active_flag=1",
    "/v1/works_cycles/employee_count_shift_year?shift_id=3&year=2009",
    "/v1/works_cycles/department_most_employees_shift?shift_name=Night",
    "/v1/works_cycles/sales_order_count_ship_method?ship_method_name=XRQ%20-%20TRUCK%20GROUND",
    "/v1/works_cycles/product_subcategories_category?product_category_name=Bikes",
    "/v1/works_cycles/top_sales_territory",
    "/v1/works_cycles/product_names_sales_order?sales_order_id=43660",
    "/v1/works_cycles/top_products_order_quantity?offset=0&limit=5",
    "/v1/works_cycles/product_shelf_location?product_name=Down%20Tube",
    "/v1/works_cycles/store_address_type?store_name=Fun%20Toys%20and%20Bikes",
    "/v1/works_cycles/contact_type_names_by_store?store_name=Next-Door%20Bike%20Store",
    "/v1/works_cycles/reference_order_ids_by_product?product_name=Mountain%20End%20Caps",
    "/v1/works_cycles/address_line_by_business_entity?business_entity_id=12",
    "/v1/works_cycles/state_province_names_by_city?city=Racine",
    "/v1/works_cycles/sales_quota_percentage?sales_quota=300000&country_region_code=US&territory_name=Northwest",
    "/v1/works_cycles/cost_difference_by_product_name?product_name_pattern=%25Crankarm%25",
    "/v1/works_cycles/highest_rated_products",
    "/v1/works_cycles/second_highest_pay_rate?pay_frequency=1",
    "/v1/works_cycles/product_weight?weight_unit_measure_code=LB",
    "/v1/works_cycles/product_reorder_point?size_unit_measure_code=CM",
    "/v1/works_cycles/purchase_order_total_due",
    "/v1/works_cycles/product_profit_margin",
    "/v1/works_cycles/top_purchase_order_tax",
    "/v1/works_cycles/salesperson_count?sales_quota=500000",
    "/v1/works_cycles/employee_shift_times?birth_year=1969",
    "/v1/works_cycles/product_name_rating_shortest_manufacture_highest_margin",
    "/v1/works_cycles/salespersons_with_bonus_less_than?bonus=1000",
    "/v1/works_cycles/employee_end_dates_by_job_title_age_difference?job_title=Senior%20Tool%20Designer&age_difference=33",
    "/v1/works_cycles/total_due_by_credit_rating_preferred_status?credit_rating=4&preferred_vendor_status=0",
    "/v1/works_cycles/department_with_most_night_shifts?shift_name=Night",
    "/v1/works_cycles/double_profit_margin_by_class_color_size_subcategory?product_class=H&color=Black&size=58&subcategory_name=Road%20Bikes",
    "/v1/works_cycles/employee_names_by_job_title_gender_marital_status?job_title=Production%20Technician%25&gender=M&marital_status=M",
    "/v1/works_cycles/vendor_name_credit_rating_by_average_lead_time?average_lead_time=60",
    "/v1/works_cycles/profit_margin_vendor_name_by_max_order_qty?max_order_qty=200",
    "/v1/works_cycles/employee_with_most_purchase_orders?person_type=EM&status=3",
    "/v1/works_cycles/vendor_names_by_max_order_qty?limit=1&offset=1",
    "/v1/works_cycles/min_order_qty_by_active_flag?active_flag=0&limit=1",
    "/v1/works_cycles/employee_details_by_vacation_hours?vacation_hours=10&limit=1",
    "/v1/works_cycles/person_types_by_gender_marital_status_hire_date?gender=F&marital_status=S&start_date=2008-01-01&end_date=2008-12-31&limit=1",
    "/v1/works_cycles/price_difference_by_reviewer_name?reviewer_name=David&limit=1",
    "/v1/works_cycles/sales_territory_groups_by_tax_rate?limit=1&offset=1",
    "/v1/works_cycles/product_stats_by_class_subcategory?product_class=L&subcategory_name=Road%20Frames",
    "/v1/works_cycles/discount_percentage_by_description?description=LL%20Road%20Frame%20Sale",
    "/v1/works_cycles/count_special_offers_by_type?offer_type=Excess%20Inventory",
    "/v1/works_cycles/special_offer_descriptions_by_type?offer_type=Seasonal%20Discount&limit=1",
    "/v1/works_cycles/special_offers_by_category?category=Reseller&offset=0&limit=3",
    "/v1/works_cycles/store_demographics_by_name?name=Valley%20Bicycle%20Specialists",
    "/v1/works_cycles/sum_finished_goods_flag_by_make_flag?make_flag=1",
    "/v1/works_cycles/safety_stock_level_by_product_name?name=Chainring%20Bolts",
    "/v1/works_cycles/product_names_by_standard_cost?limit=1",
    "/v1/works_cycles/person_type_by_name?first_name=David&last_name=Bradley",
    "/v1/works_cycles/count_persons_by_type_and_title?person_type=SC&title=Mr.",
    "/v1/works_cycles/address_details_by_store_name?name=Valley%20Bicycle%20Specialists",
    "/v1/works_cycles/email_address_by_name?first_name=David&last_name=Bradley",
    "/v1/works_cycles/phone_numbers_by_person_type?person_type=SC",
    "/v1/works_cycles/password_hash_by_name?first_name=David&last_name=Bradley",
    "/v1/works_cycles/email_address_by_promotion?email_promotion=2",
    "/v1/works_cycles/credit_card_number_by_name?first_name=David&last_name=Bradley",
    "/v1/works_cycles/credit_card_exp_year_by_name?first_name=David&last_name=Bradley",
    "/v1/works_cycles/person_names_by_exp_year_and_type?exp_year=2007&person_type=SC",
    "/v1/works_cycles/count_persons_by_card_type_and_person_type?card_type=Vista&person_type=SC",
    "/v1/works_cycles/count_departments_by_name?first_name=David&last_name=Bradley",
    "/v1/works_cycles/department_ids_by_name?first_name=David&last_name=Bradley",
    "/v1/works_cycles/count_persons_by_department_and_year?department_name=Engineering&year=2009",
    "/v1/works_cycles/longest_employment_in_department?department_name=Engineering",
    "/v1/works_cycles/credit_card_numbers_department_non_null_end_date?department_name=Finance",
    "/v1/works_cycles/employee_count_credit_card_expiry_year_department?exp_year=2007&department_name=Engineering",
    "/v1/works_cycles/email_most_department_changes",
    "/v1/works_cycles/employee_count_department_email_promotion?department_name=Engineering&email_promotion=0",
    "/v1/works_cycles/employee_count_department_group_start_year?group_name=Quality%20Assurance&start_year=2007",
    "/v1/works_cycles/check_product_special_offer?product_name=Chainring%20Bolts",
    "/v1/works_cycles/product_count_subcategory_special_offer?description=LL%20Road%20Frame%20Sale&subcategory_name=Clothing",
    "/v1/works_cycles/product_names_by_subcategory?product_subcategory_id=3",
    "/v1/works_cycles/average_employment_duration?department_name=Engineering",
    "/v1/works_cycles/list_price_percentage_difference?category_name=Clothing",
    "/v1/works_cycles/average_profit_margin?category_name=Clothing",
    "/v1/works_cycles/product_id_by_start_date?start_date_pattern=2013%",
    "/v1/works_cycles/product_ids_by_color?color=Yellow",
    "/v1/works_cycles/business_entity_ids_by_name?title=Mr.&first_name=Hung-Fu&last_name=Ting",
    "/v1/works_cycles/phone_numbers_by_business_entity?business_entity_id=12597",
    "/v1/works_cycles/list_prices_by_product_id?product_id=912",
    "/v1/works_cycles/check_cost_match",
    "/v1/works_cycles/get_thumbnail_photo_file_name?product_id=979",
    "/v1/works_cycles/get_distinct_product_names_by_list_price?min_list_price=1000",
    "/v1/works_cycles/get_product_id_highest_price_difference",
    "/v1/works_cycles/get_product_names_by_location_shelf_bin?location_id=1&shelf=L&bin=6",
    "/v1/works_cycles/get_location_names_by_work_order_id?work_order_id=35493",
    "/v1/works_cycles/get_product_ids_by_large_photo_file_name?large_photo_file_name_pattern=%25large.gif",
    "/v1/works_cycles/get_product_ids_by_subcategory_name?subcategory_name=Socks",
    "/v1/works_cycles/get_percentage_of_list_price?product_name=Cable%20Lock",
    "/v1/works_cycles/get_scrap_reason_names_by_work_order_id?work_order_id=57788",
    "/v1/works_cycles/product_standard_cost_by_name?name_substring=Sport",
    "/v1/works_cycles/product_quantity_by_location?product_id=476&location_name=Metal%20Storage",
    "/v1/works_cycles/product_ids_below_average_cost",
    "/v1/works_cycles/product_count_by_photo_id?product_photo_id=1",
    "/v1/works_cycles/work_order_duration?work_order_id=425",
    "/v1/works_cycles/highest_list_price_product?start_date=2012",
    "/v1/works_cycles/product_standard_cost?product_id=847",
    "/v1/works_cycles/employee_organization_level?job_title=Human%20Resources%20Manager",
    "/v1/works_cycles/work_order_count_end_after_due",
    "/v1/works_cycles/product_cost_history?product_id=888",
    "/v1/works_cycles/product_photo_count?large_photo_file_name=roadster_black_large.gif",
    "/v1/works_cycles/product_inventory_location?product_name=Lock%20Ring",
    "/v1/works_cycles/product_category_name?subcategory_name=Road%20Frames",
    "/v1/works_cycles/work_order_ids_by_scrap_reason?scrap_reason_name=Handling%20damage",
    "/v1/works_cycles/price_difference?product_id=792",
    "/v1/works_cycles/person_by_email?email_address=regina7@adventure-works.com",
    "/v1/works_cycles/location_by_product_id?product_id=810",
    "/v1/works_cycles/product_by_work_order_id?work_order_id=2540",
    "/v1/works_cycles/work_order_ids_by_product_name?product_name=Down%20Tube",
    "/v1/works_cycles/avg_percentage_difference_list_price_standard_cost",
    "/v1/works_cycles/percentage_work_orders_by_location_type?location_type=Subassembly"
]
