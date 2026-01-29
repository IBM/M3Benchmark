from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/sales/sales.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of sales based on customer ID
@app.get("/v1/sales/count_sales_by_customer_id", operation_id="get_count_sales_by_customer_id", summary="Retrieves the total number of sales associated with a specific customer, identified by their unique customer ID.")
async def get_count_sales_by_customer_id(customer_id: int = Query(..., description="Customer ID")):
    cursor.execute("SELECT COUNT(SalesID) FROM Sales WHERE CustomerID = ?", (customer_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of quantities for a range of sales IDs
@app.get("/v1/sales/sum_quantities_by_sales_id_range", operation_id="get_sum_quantities_by_sales_id_range", summary="Retrieves the total quantity of sales within a specified range of sales IDs. The range is defined by the minimum and maximum sales IDs provided as input parameters.")
async def get_sum_quantities_by_sales_id_range(min_sales_id: int = Query(..., description="Minimum Sales ID"), max_sales_id: int = Query(..., description="Maximum Sales ID")):
    cursor.execute("SELECT SUM(Quantity) FROM Sales WHERE SalesID BETWEEN ? AND ?", (min_sales_id, max_sales_id))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the average quantity for a range of sales IDs
@app.get("/v1/sales/avg_quantity_by_sales_id_range", operation_id="get_avg_quantity_by_sales_id_range", summary="Retrieves the average quantity of sales within a specified range of sales IDs. The range is defined by a minimum and maximum sales ID, both of which are required as input parameters. This operation calculates the average quantity of sales that fall within this range, providing a statistical overview of sales volume.")
async def get_avg_quantity_by_sales_id_range(min_sales_id: int = Query(..., description="Minimum Sales ID"), max_sales_id: int = Query(..., description="Maximum Sales ID")):
    cursor.execute("SELECT AVG(Quantity) FROM Sales WHERE SalesID BETWEEN ? AND ?", (min_sales_id, max_sales_id))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get distinct product IDs with the maximum quantity
@app.get("/v1/sales/distinct_product_ids_with_max_quantity", operation_id="get_distinct_product_ids_with_max_quantity", summary="Retrieves a list of unique product IDs that have the highest quantity in the sales records.")
async def get_distinct_product_ids_with_max_quantity():
    cursor.execute("SELECT DISTINCT ProductID FROM Sales WHERE Quantity = ( SELECT MAX(Quantity) FROM Sales )")
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct product IDs with the maximum price
@app.get("/v1/sales/count_distinct_product_ids_with_max_price", operation_id="get_count_distinct_product_ids_with_max_price", summary="Retrieves the total number of unique products that have the highest price in the product catalog.")
async def get_count_distinct_product_ids_with_max_price():
    cursor.execute("SELECT COUNT(DISTINCT ProductID) FROM Products WHERE Price = ( SELECT MAX(Price) FROM Products )")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get product names based on price
@app.get("/v1/sales/product_names_by_price", operation_id="get_product_names_by_price", summary="Retrieves the names of products that are priced at the specified value. The operation filters the product catalog based on the provided price and returns the corresponding product names.")
async def get_product_names_by_price(price: float = Query(..., description="Price of the product")):
    cursor.execute("SELECT Name FROM Products WHERE Price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get product names for a range of product IDs
@app.get("/v1/sales/product_names_by_product_id_range", operation_id="get_product_names_by_product_id_range", summary="Retrieves the names of products that have a Product ID within the specified range. The range is defined by a minimum and maximum Product ID, both of which are required as input parameters.")
async def get_product_names_by_product_id_range(min_product_id: int = Query(..., description="Minimum Product ID"), max_product_id: int = Query(..., description="Maximum Product ID")):
    cursor.execute("SELECT Name FROM Products WHERE ProductID BETWEEN ? AND ?", (min_product_id, max_product_id))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the product name with the highest quantity in sales
@app.get("/v1/sales/product_name_with_highest_quantity", operation_id="get_product_name_with_highest_quantity", summary="Retrieves the name of the product with the highest sales quantity. This operation fetches the product with the most units sold by joining the Sales and Products tables and sorting by the quantity sold. The result is the name of the top-selling product.")
async def get_product_name_with_highest_quantity():
    cursor.execute("SELECT T2.Name FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID ORDER BY T1.Quantity LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of customers who bought a specific product
@app.get("/v1/sales/count_customers_by_product_name", operation_id="get_count_customers_by_product_name", summary="Retrieves the total number of unique customers who have purchased a product with the specified name. The operation uses the product name to identify relevant sales records and calculate the count of distinct customers.")
async def get_count_customers_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of sales for a specific product
@app.get("/v1/sales/count_sales_by_product_name", operation_id="get_count_sales_by_product_name", summary="Retrieves the total number of sales for a specified product. The operation calculates the count based on the provided product name.")
async def get_count_sales_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT COUNT(T1.SalesID) FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct product names based on salesperson ID
@app.get("/v1/sales/distinct_product_names_by_salesperson", operation_id="get_distinct_product_names", summary="Retrieves a list of unique product names sold by a specific salesperson. The operation requires the salesperson's ID as input to filter the sales records and identify the distinct products they have sold.")
async def get_distinct_product_names(salesperson_id: int = Query(..., description="ID of the salesperson")):
    cursor.execute("SELECT DISTINCT T1.Name FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T2.SalesPersonID = ?", (salesperson_id,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get customer first names based on product ID
@app.get("/v1/sales/customer_first_names_by_product", operation_id="get_customer_first_names", summary="Retrieves the first names of customers who have purchased a specific product, identified by its unique product ID.")
async def get_customer_first_names(product_id: int = Query(..., description="ID of the product")):
    cursor.execute("SELECT T1.FirstName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T2.ProductID = ?", (product_id,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get customer last names based on sales ID
@app.get("/v1/sales/customer_last_names_by_sales_id", operation_id="get_customer_last_names", summary="Retrieves the last names of customers who participated in a sale identified by the provided sales ID. This operation allows you to obtain customer information linked to a specific sale, facilitating sales analysis or customer relationship management.")
async def get_customer_last_names(sales_id: int = Query(..., description="ID of the sale")):
    cursor.execute("SELECT T1.LastName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.SalesID = ?", (sales_id,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get distinct product IDs based on customer first name
@app.get("/v1/sales/distinct_product_ids_by_customer_first_name", operation_id="get_distinct_product_ids", summary="Retrieves a unique set of product identifiers associated with sales made by customers who share a specified first name. This operation allows for the identification of products that are popular among customers with a particular first name.")
async def get_distinct_product_ids(first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT DISTINCT T1.ProductID FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.FirstName = ?", (first_name,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get distinct product IDs based on employee first name
@app.get("/v1/sales/distinct_product_ids_by_employee_first_name", operation_id="get_distinct_product_ids_by_employee", summary="Retrieve a unique set of product IDs associated with sales made by employees sharing a common first name. The operation filters sales data based on the provided employee first name.")
async def get_distinct_product_ids_by_employee(first_name: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT DISTINCT T2.ProductID FROM Employees AS T1 INNER JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID WHERE T1.FirstName = ?", (first_name,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get employee last names based on sales ID
@app.get("/v1/sales/employee_last_names_by_sales_id", operation_id="get_employee_last_names", summary="Retrieves the last names of employees who were involved in a sale identified by the provided sales ID. This operation returns a list of last names, providing insight into the employees associated with the specified sale.")
async def get_employee_last_names(sales_id: int = Query(..., description="ID of the sale")):
    cursor.execute("SELECT T1.LastName FROM Employees AS T1 INNER JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID WHERE T2.SalesID = ?", (sales_id,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get distinct employee first names based on customer first name
@app.get("/v1/sales/distinct_employee_first_names_by_customer_first_name", operation_id="get_distinct_employee_first_names", summary="Retrieves a list of unique first names of employees who have made sales to customers with a specified first name. This operation allows you to identify the distinct salespeople who have interacted with customers sharing a common first name.")
async def get_distinct_employee_first_names(customer_first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT DISTINCT T3.FirstName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Employees AS T3 ON T2.SalesPersonID = T3.EmployeeID WHERE T1.FirstName = ?", (customer_first_name,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the count of products based on customer ID and product price
@app.get("/v1/sales/count_products_by_customer_and_price", operation_id="get_count_products", summary="Retrieves the total number of products that a particular customer has purchased at a specific price. The operation requires the customer's ID and the product's price as input parameters.")
async def get_count_products(customer_id: int = Query(..., description="ID of the customer"), price: float = Query(..., description="Price of the product")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T2.CustomerID = ? AND T1.Price = ?", (customer_id, price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get customer details based on salesperson ID
@app.get("/v1/sales/customer_details_by_salesperson", operation_id="get_customer_details", summary="Retrieves the full names of customers who were served by the specified salesperson. The salesperson is identified by their unique ID.")
async def get_customer_details(salesperson_id: int = Query(..., description="ID of the salesperson")):
    cursor.execute("SELECT T1.FirstName, T1.MiddleInitial, T1.LastName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.SalesPersonID = ?", (salesperson_id,))
    result = cursor.fetchall()
    if not result:
        return {"customer_details": []}
    return {"customer_details": [{"first_name": row[0], "middle_initial": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get sales IDs based on the first letter of the employee's first name
@app.get("/v1/sales/sales_ids_by_employee_first_letter", operation_id="get_sales_ids", summary="Retrieves the sales IDs for sales made by employees whose first name starts with a specified letter. This operation filters sales data based on the first letter of the employee's first name, providing a targeted list of sales IDs.")
async def get_sales_ids(first_letter: str = Query(..., description="First letter of the employee's first name")):
    cursor.execute("SELECT T1.SalesID FROM Sales AS T1 INNER JOIN Employees AS T2 ON T1.SalesPersonID = T2.EmployeeID WHERE SUBSTR(T2.FirstName, 1, 1) = ?", (first_letter,))
    result = cursor.fetchall()
    if not result:
        return {"sales_ids": []}
    return {"sales_ids": [row[0] for row in result]}

# Endpoint to get the highest product price for sales within a customer ID range
@app.get("/v1/sales/highest_product_price_in_customer_range", operation_id="get_highest_product_price", summary="Retrieves the highest product price from sales records within a specified customer ID range. The operation filters sales data based on the provided minimum and maximum customer IDs, then identifies the product with the highest price from the filtered results.")
async def get_highest_product_price(min_customer_id: int = Query(..., description="Minimum customer ID"), max_customer_id: int = Query(..., description="Maximum customer ID")):
    cursor.execute("SELECT T1.Price FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T2.CustomerID BETWEEN ? AND ? ORDER BY T1.Price DESC LIMIT 1", (min_customer_id, max_customer_id))
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get the first name of the customer with the highest quantity of sales for a given last name
@app.get("/v1/sales/customer_first_name_highest_quantity", operation_id="get_customer_first_name", summary="Retrieves the first name of the customer who has made the most sales with the specified last name. The operation filters customers by the provided last name and sorts them by the quantity of sales in descending order. The first name of the customer with the highest sales quantity is then returned.")
async def get_customer_first_name(last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T1.FirstName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.LastName = ? ORDER BY T2.Quantity DESC LIMIT 1", (last_name,))
    result = cursor.fetchone()
    if not result:
        return {"first_name": []}
    return {"first_name": result[0]}

# Endpoint to get the count of sales by specific employees
@app.get("/v1/sales/count_sales_by_employees", operation_id="get_count_sales_by_employees", summary="Retrieves the total number of sales made by three specific employees. The operation accepts the first names of these employees as input parameters and returns the combined sales count.")
async def get_count_sales_by_employees(first_name_1: str = Query(..., description="First name of the first employee"), first_name_2: str = Query(..., description="First name of the second employee"), first_name_3: str = Query(..., description="First name of the third employee")):
    cursor.execute("SELECT SUM(IIF(T2.FirstName = ?, 1, 0)) + SUM(IIF(T2.FirstName = ?, 1, 0)) + SUM(IIF(T2.FirstName = ?, 1, 0)) AS num FROM Sales AS T1 INNER JOIN Employees AS T2 ON T1.SalesPersonID = T2.EmployeeID", (first_name_1, first_name_2, first_name_3))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to check if a specific product was sold to a customer with a given first name
@app.get("/v1/sales/check_product_sold_to_customer", operation_id="check_product_sold", summary="Check if a specific product was sold to a customer with a given first name")
async def check_product_sold(product_id: int = Query(..., description="Product ID"), first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT IIF(T1.ProductID = ?, 'YES', 'NO') FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.FirstName = ?", (product_id, first_name))
    result = cursor.fetchone()
    if not result:
        return {"sold": []}
    return {"sold": result[0]}

# Endpoint to get the total sales amount for a customer with a given first name
@app.get("/v1/sales/total_sales_amount_for_customer", operation_id="get_total_sales_amount", summary="Retrieves the total sales amount for a customer based on their first name. This operation calculates the sum of the product of the price and quantity for all sales made by the customer, providing a comprehensive view of their total sales value.")
async def get_total_sales_amount(first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT SUM(T3.Price * T2.quantity) FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T1.FirstName = ?", (first_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the total sales amount for products within a given ID range
@app.get("/v1/sales/total_sales_amount_for_product_range", operation_id="get_total_sales_amount_for_range", summary="Retrieves the total sales amount for products within a specified ID range. The operation calculates the sum of the product price multiplied by the quantity sold for all products whose IDs fall within the provided range. The range is defined by the minimum and maximum product IDs.")
async def get_total_sales_amount_for_range(min_product_id: int = Query(..., description="Minimum product ID"), max_product_id: int = Query(..., description="Maximum product ID")):
    cursor.execute("SELECT SUM(T1.Price * T2.quantity) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductID BETWEEN ? AND ?", (min_product_id, max_product_id))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the total quantity sold for products starting with a specific letter
@app.get("/v1/sales/total_quantity_sold_by_product_name", operation_id="get_total_quantity_sold", summary="Retrieves the total quantity sold for all products whose names start with a specified letter. The input parameter determines the starting letter of the product names.")
async def get_total_quantity_sold(starting_letter: str = Query(..., description="Starting letter of the product name")):
    cursor.execute("SELECT SUM(T2.Quantity) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE SUBSTR(T1.Name, 1, 1) = ?", (starting_letter,))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the total quantity sold to a customer with a given first name
@app.get("/v1/sales/total_quantity_sold_to_customer", operation_id="get_total_quantity_sold_to_customer", summary="Retrieves the total quantity of items sold to a customer with a specified first name. This operation calculates the sum of all sales made to the customer, providing a comprehensive view of their purchase history.")
async def get_total_quantity_sold_to_customer(first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT SUM(T2.Quantity) FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.FirstName = ?", (first_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the top 5 product IDs by price
@app.get("/v1/sales/top_5_products_by_price", operation_id="get_top_5_products", summary="Retrieves the identifiers of the five most expensive products in the catalog.")
async def get_top_5_products():
    cursor.execute("SELECT ProductID FROM Products ORDER BY Price DESC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get the count of products with a specific price
@app.get("/v1/sales/count_products_by_price", operation_id="get_count_products_by_price", summary="Retrieves the total number of products that have a specified price. The price is provided as an input parameter.")
async def get_count_products_by_price(price: float = Query(..., description="Price of the product")):
    cursor.execute("SELECT COUNT(ProductID) FROM Products WHERE Price = ?", (price,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct employee names based on sales quantity
@app.get("/v1/sales/distinct_employee_names_by_quantity", operation_id="get_distinct_employee_names", summary="Retrieve a unique list of employee names who have made sales of a specific quantity. The quantity is provided as an input parameter.")
async def get_distinct_employee_names(quantity: int = Query(..., description="Sales quantity")):
    cursor.execute("SELECT DISTINCT T2.FirstName, T2.MiddleInitial, T2.LastName FROM Sales AS T1 INNER JOIN Employees AS T2 ON T1.SalesPersonID = T2.EmployeeID WHERE T1.Quantity = ?", (quantity,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"first_name": row[0], "middle_initial": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get product names and quantities for sales within a specified range of sales IDs
@app.get("/v1/sales/product_quantities_by_sales_id_range", operation_id="get_product_quantities", summary="Retrieves the names and quantities of products sold within a specified range of sales IDs. The range is defined by the minimum and maximum sales IDs provided as input parameters. This operation returns a list of product names and their corresponding quantities sold within the specified sales ID range.")
async def get_product_quantities(min_sales_id: int = Query(..., description="Minimum sales ID"), max_sales_id: int = Query(..., description="Maximum sales ID")):
    cursor.execute("SELECT T3.Name, T2.Quantity FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T2.SalesID BETWEEN ? AND ?", (min_sales_id, max_sales_id))
    result = cursor.fetchall()
    if not result:
        return {"product_quantities": []}
    return {"product_quantities": [{"product_name": row[0], "quantity": row[1]} for row in result]}

# Endpoint to compare sales of two products and return the more popular one
@app.get("/v1/sales/compare_product_sales", operation_id="compare_product_sales", summary="This operation compares the sales of two specified products and returns the name of the product with higher sales. The comparison is based on the total sales quantity of each product. The result is returned as a string, which can be customized using the 'result_if_greater' and 'result_if_lesser' parameters.")
async def compare_product_sales(product_name_1: str = Query(..., description="Name of the first product"), product_name_2: str = Query(..., description="Name of the second product"), result_if_greater: str = Query(..., description="Result if the first product has more sales"), result_if_lesser: str = Query(..., description="Result if the second product has more sales")):
    cursor.execute("SELECT IIF(SUM(IIF(T1.Name = ?, T2.SalesID, 0)) - SUM(IIF(T1.Name = ?, T2.SalesID, 0)) > 0, ?, ?) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID", (product_name_1, product_name_2, result_if_greater, result_if_lesser))
    result = cursor.fetchone()
    if not result:
        return {"comparison_result": []}
    return {"comparison_result": result[0]}

# Endpoint to get the price difference between two products
@app.get("/v1/sales/price_difference_between_products", operation_id="get_price_difference", summary="Retrieves the price difference between two specified products. The operation compares the prices of the products provided as input parameters and returns the difference.")
async def get_price_difference(product_name_1: str = Query(..., description="Name of the first product"), product_name_2: str = Query(..., description="Name of the second product")):
    cursor.execute("SELECT ( SELECT Price FROM Products WHERE Name = ? ) - ( SELECT Price FROM Products WHERE Name = ? ) AS num", (product_name_1, product_name_2))
    result = cursor.fetchone()
    if not result:
        return {"price_difference": []}
    return {"price_difference": result[0]}

# Endpoint to get the count of sales by a specific employee
@app.get("/v1/sales/count_sales_by_employee", operation_id="get_count_sales_by_employee", summary="Retrieves the total number of sales made by a specific employee, identified by their first name, middle initial, and last name.")
async def get_count_sales_by_employee(first_name: str = Query(..., description="First name of the employee"), middle_initial: str = Query(..., description="Middle initial of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT COUNT(T1.SalesID) FROM Sales AS T1 INNER JOIN Employees AS T2 ON T1.SalesPersonID = T2.EmployeeID WHERE T2.FirstName = ? AND T2.MiddleInitial = ? AND T2.LastName = ?", (first_name, middle_initial, last_name))
    result = cursor.fetchone()
    if not result:
        return {"sales_count": []}
    return {"sales_count": result[0]}

# Endpoint to get the ratio of customers to employees
@app.get("/v1/sales/customer_to_employee_ratio", operation_id="get_customer_to_employee_ratio", summary="Retrieves the ratio of the total number of unique customers to the total number of employees involved in sales. This operation calculates the ratio by counting the unique customers who have made at least one purchase and the employees who have made at least one sale.")
async def get_customer_to_employee_ratio():
    cursor.execute("SELECT CAST(COUNT(T1.CustomerID) AS REAL) / COUNT(T3.EmployeeID) FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Employees AS T3 ON T2.SalesPersonID = T3.EmployeeID")
    result = cursor.fetchone()
    if not result:
        return {"customer_to_employee_ratio": []}
    return {"customer_to_employee_ratio": result[0]}

# Endpoint to get the count of customers for a specific product sold by a specific employee
@app.get("/v1/sales/count_customers_by_product_and_employee", operation_id="get_count_customers_by_product_and_employee", summary="Retrieves the total number of customers who have purchased a specific product sold by a particular employee. The employee is identified by their first name, last name, and middle initial. The product is specified by its name.")
async def get_count_customers_by_product_and_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), product_name: str = Query(..., description="Name of the product"), middle_initial: str = Query(..., description="Middle initial of the employee")):
    cursor.execute("SELECT COUNT(T2.CustomerID) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Employees AS T3 ON T2.SalesPersonID = T3.EmployeeID WHERE T3.FirstName = ? AND T3.LastName = ? AND T1.Name = ? AND T3.MiddleInitial = ?", (first_name, last_name, product_name, middle_initial))
    result = cursor.fetchone()
    if not result:
        return {"customer_count": []}
    return {"customer_count": result[0]}

# Endpoint to get distinct employee names based on customer and product criteria
@app.get("/v1/sales/distinct_employee_names_by_customer_and_product", operation_id="get_distinct_employee_names_by_customer_and_product", summary="Retrieves a list of unique employee names who have sold a specific product to a customer with a given name. The customer's full name and the product name are used as criteria to filter the results.")
async def get_distinct_employee_names_by_customer_and_product(customer_middle_initial: str = Query(..., description="Middle initial of the customer"), customer_last_name: str = Query(..., description="Last name of the customer"), product_name: str = Query(..., description="Name of the product"), customer_first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT DISTINCT T3.FirstName, T3.MiddleInitial, T3.LastName FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Employees AS T3 ON T2.SalesPersonID = T3.EmployeeID INNER JOIN Customers AS T4 ON T2.CustomerID = T4.CustomerID WHERE T4.MiddleInitial = ? AND T4.LastName = ? AND T1.Name = ? AND T4.FirstName = ?", (customer_middle_initial, customer_last_name, product_name, customer_first_name))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"first_name": row[0], "middle_initial": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get the count of salespersons for a specific product name
@app.get("/v1/sales/count_salespersons_by_product_name", operation_id="get_count_salespersons_by_product_name", summary="Retrieves the total number of salespersons associated with a specific product. The product is identified by its name, which is provided as an input parameter.")
async def get_count_salespersons_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT COUNT(T2.SalesPersonID) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total sales amount for a specific product name
@app.get("/v1/sales/total_sales_amount_by_product_name", operation_id="get_total_sales_amount_by_product_name", summary="Retrieves the total sales amount for a specific product. The calculation is based on the product's price multiplied by its quantity sold. The product is identified by its name.")
async def get_total_sales_amount_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(T2.Quantity * T1.Price) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_sales_amount": []}
    return {"total_sales_amount": result[0]}

# Endpoint to get the count of sales for a specific customer
@app.get("/v1/sales/count_sales_by_customer_name", operation_id="get_count_sales_by_customer_name", summary="Retrieves the total number of sales made by a customer identified by their full name. The customer's first name, middle initial, and last name are required as input parameters to accurately determine the sales count.")
async def get_count_sales_by_customer_name(first_name: str = Query(..., description="First name of the customer"), middle_initial: str = Query(..., description="Middle initial of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T1.SalesID) FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.FirstName = ? AND T2.MiddleInitial = ? AND T2.LastName = ?", (first_name, middle_initial, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct customer names who bought a specific product at a specific price
@app.get("/v1/sales/distinct_customer_names_by_product_and_price", operation_id="get_distinct_customer_names_by_product_and_price", summary="Retrieves a list of unique customer names who have purchased a specific product at a given price. The operation filters customers based on the provided product name and price, ensuring that only those who have bought the specified product at the exact price are included in the result set.")
async def get_distinct_customer_names_by_product_and_price(product_name: str = Query(..., description="Name of the product"), product_price: float = Query(..., description="Price of the product")):
    cursor.execute("SELECT DISTINCT T1.FirstName, T1.MiddleInitial, T1.LastName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T3.Name = ? AND T3.Price = ?", (product_name, product_price))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the count of customers grouped by salesperson ID
@app.get("/v1/sales/count_customers_by_salesperson", operation_id="get_count_customers_by_salesperson", summary="Retrieves the total number of customers associated with each salesperson. The operation groups customers by their respective salesperson IDs and returns the count for each group.")
async def get_count_customers_by_salesperson():
    cursor.execute("SELECT COUNT(CustomerID) FROM Sales GROUP BY SalesPersonID")
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": result}

# Endpoint to get the count of employees
@app.get("/v1/sales/count_employees", operation_id="get_count_employees", summary="Retrieves the total number of employees in the system. This operation does not require any input parameters and returns a single integer value representing the count of employees.")
async def get_count_employees():
    cursor.execute("SELECT COUNT(EmployeeID) FROM Employees")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of an employee by employee ID
@app.get("/v1/sales/employee_name_by_id", operation_id="get_employee_name_by_id", summary="Retrieves the full name of an employee based on the provided employee ID. The response includes the first name, middle initial, and last name of the employee.")
async def get_employee_name_by_id(employee_id: int = Query(..., description="ID of the employee")):
    cursor.execute("SELECT FirstName, MiddleInitial, LastName FROM Employees WHERE EmployeeID = ?", (employee_id,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result}

# Endpoint to get the names of products with the maximum and minimum prices
@app.get("/v1/sales/product_names_by_max_min_price", operation_id="get_product_names_by_max_min_price", summary="Get the names of products with the maximum and minimum prices")
async def get_product_names_by_max_min_price(max_price: float = Query(..., description="Maximum price of the product"), min_price: float = Query(..., description="Minimum price of the product")):
    cursor.execute("SELECT Name FROM Products WHERE Price IN (?, ?)", (max_price, min_price))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the percentage of sales by a specific employee for a specific product
@app.get("/v1/sales/percentage_sales_by_employee_and_product", operation_id="get_percentage_sales_by_employee_and_product", summary="Retrieves the percentage of total sales attributed to a specific employee for a given product. The calculation is based on the count of sales made by the employee for the product, divided by the total sales count for that product.")
async def get_percentage_sales_by_employee_and_product(first_name: str = Query(..., description="First name of the employee"), middle_initial: str = Query(..., description="Middle initial of the employee"), last_name: str = Query(..., description="Last name of the employee"), product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT CAST(SUM(IIF(T3.FirstName = ? AND T3.MiddleInitial = ? AND T3.LastName = ?, 1, 0)) AS REAL) * 100 / COUNT(T2.CustomerID) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Employees AS T3 ON T2.SalesPersonID = T3.EmployeeID WHERE T1.Name = ?", (first_name, middle_initial, last_name, product_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of customers with a specific first name
@app.get("/v1/sales/count_customers_by_first_name", operation_id="get_count_customers_by_first_name", summary="Retrieves the total number of customers with a specified first name from the database.")
async def get_count_customers_by_first_name(first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT COUNT(CustomerID) FROM Customers WHERE FirstName = ?", (first_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct quantities of a specific product
@app.get("/v1/sales/distinct_quantities_by_product_name", operation_id="get_distinct_quantities", summary="Retrieves the unique quantities associated with a specific product, identified by its name. This operation provides a concise overview of the different quantities in which the product has been sold.")
async def get_distinct_quantities(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT DISTINCT T2.Quantity FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"quantities": []}
    return {"quantities": [row[0] for row in result]}

# Endpoint to get the top employee by sales quantity
@app.get("/v1/sales/top_employee_by_sales_quantity", operation_id="get_top_employee", summary="Retrieves the employee with the highest sales quantity. This operation fetches the first name and last name of the employee who has the highest sales quantity based on the sales data. The result is determined by joining the Employees and Sales tables and ordering the sales quantity in descending order.")
async def get_top_employee():
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Employees AS T1 INNER JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID ORDER BY T2.Quantity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": {}}
    return {"employee": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the top customer by sales quantity
@app.get("/v1/sales/top_customer_by_sales_quantity", operation_id="get_top_customer", summary="Retrieves the customer who has made the highest sales quantity. The operation returns the first name and last name of the top customer.")
async def get_top_customer():
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID ORDER BY T2.Quantity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": {}}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the top product by sales quantity for a specific salesperson
@app.get("/v1/sales/top_product_by_salesperson", operation_id="get_top_product_by_salesperson", summary="Retrieves the product with the highest sales quantity for a given salesperson. The operation requires the salesperson's unique identifier as input and returns the name of the top-selling product.")
async def get_top_product_by_salesperson(salesperson_id: int = Query(..., description="ID of the salesperson")):
    cursor.execute("SELECT T1.Name FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T2.SalesPersonID = ? ORDER BY T2.Quantity DESC LIMIT 1", (salesperson_id,))
    result = cursor.fetchone()
    if not result:
        return {"product": {}}
    return {"product": {"name": result[0]}}

# Endpoint to get the total sales quantity for a specific salesperson, customer, and employee
@app.get("/v1/sales/total_sales_quantity_by_salesperson_customer_employee", operation_id="get_total_sales_quantity", summary="Retrieves the total quantity of sales made by a specific salesperson to a particular customer, as recorded by a designated employee. The operation requires the salesperson's ID, the customer's first and last names, and the employee's first name to accurately calculate and return the total sales quantity.")
async def get_total_sales_quantity(salesperson_id: int = Query(..., description="ID of the salesperson"), customer_first_name: str = Query(..., description="First name of the customer"), customer_last_name: str = Query(..., description="Last name of the customer"), employee_first_name: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT SUM(T2.Quantity) FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Employees AS T3 ON T2.SalesPersonID = T3.EmployeeID WHERE T2.SalesPersonID = ? AND T1.FirstName = ? AND T1.LastName = ? AND T3.FirstName = ?", (salesperson_id, customer_first_name, customer_last_name, employee_first_name))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get customers with sales quantity greater than a specified value
@app.get("/v1/sales/customers_by_min_sales_quantity", operation_id="get_customers_by_min_sales_quantity", summary="Retrieves a list of customers who have made purchases exceeding a specified minimum sales quantity. The response includes the first and last names of the qualifying customers.")
async def get_customers_by_min_sales_quantity(min_quantity: int = Query(..., description="Minimum sales quantity")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Quantity > ?", (min_quantity,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the top customer by sales quantity with a specific first name
@app.get("/v1/sales/top_customer_by_first_name", operation_id="get_top_customer_by_first_name", summary="Retrieves the customer with the highest sales quantity who has the specified first name. The response includes the customer's first and last names.")
async def get_top_customer_by_first_name(first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.FirstName = ? ORDER BY T2.Quantity DESC LIMIT 1", (first_name,))
    result = cursor.fetchone()
    if not result:
        return {"customer": {}}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get sales IDs for products with a specific name pattern and price greater than a specified value
@app.get("/v1/sales/sales_ids_by_product_name_and_price", operation_id="get_sales_ids_by_product_name_and_price", summary="Retrieves the sales IDs for products that match a specified name pattern and have a price greater than a given value. The name pattern supports wildcard characters for flexible matching. The minimum price filter ensures that only products with prices above the specified threshold are included in the results.")
async def get_sales_ids_by_product_name_and_price(product_name_pattern: str = Query(..., description="Pattern for the product name (use % for wildcard)"), min_price: float = Query(..., description="Minimum price of the product")):
    cursor.execute("SELECT T2.SalesID FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name LIKE ? AND T1.Price > ?", (product_name_pattern, min_price))
    result = cursor.fetchall()
    if not result:
        return {"sales_ids": []}
    return {"sales_ids": [row[0] for row in result]}

# Endpoint to get distinct customer IDs based on product price range
@app.get("/v1/sales/distinct_customer_ids_by_price_range", operation_id="get_distinct_customer_ids", summary="Retrieve a unique set of customer identifiers for products that fall within a specified price range. The operation filters products based on the provided minimum and maximum price values, and returns the distinct customer IDs associated with these products.")
async def get_distinct_customer_ids(min_price: int = Query(..., description="Minimum price of the product"), max_price: int = Query(..., description="Maximum price of the product")):
    cursor.execute("SELECT DISTINCT T2.CustomerID FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Price BETWEEN ? AND ?", (min_price, max_price))
    result = cursor.fetchall()
    if not result:
        return {"customer_ids": []}
    return {"customer_ids": [row[0] for row in result]}

# Endpoint to get the total quantity of products sold at a specific price
@app.get("/v1/sales/total_quantity_by_price", operation_id="get_total_quantity_by_price", summary="Retrieves the total quantity of products sold at a specified price. The operation calculates the sum of all quantities sold for products with the given price, providing a comprehensive view of sales volume at that price point.")
async def get_total_quantity_by_price(price: int = Query(..., description="Price of the product")):
    cursor.execute("SELECT SUM(T2.Quantity) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Price = ?", (price,))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the percentage of products sold at a specific price
@app.get("/v1/sales/percentage_sold_by_price", operation_id="get_percentage_sold_by_price", summary="Retrieves the percentage of total sales that correspond to a specific product price. This operation calculates the proportion of sales for a given price by comparing the total quantity sold at that price to the overall sales volume. The input parameter specifies the product price for which the sales percentage is determined.")
async def get_percentage_sold_by_price(price: int = Query(..., description="Price of the product")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.Price = ?, T2.Quantity, 0)) AS REAL) * 100 / SUM(T2.Quantity) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID", (price,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of a specific product sold
@app.get("/v1/sales/percentage_sold_by_product_name", operation_id="get_percentage_sold_by_product_name", summary="Retrieves the percentage of total sales attributed to a specific product. The product is identified by its name, which is provided as an input parameter. The calculation is based on the sum of the quantities sold for the specified product divided by the total quantity of all products sold.")
async def get_percentage_sold_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.Name = ?, T2.Quantity, 0)) AS REAL) * 100 / SUM(T2.Quantity) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of employees with a specific last name
@app.get("/v1/sales/employee_count_by_last_name", operation_id="get_employee_count_by_last_name", summary="Retrieves the total number of employees who share a specified last name. The operation filters the employee records based on the provided last name and returns the count.")
async def get_employee_count_by_last_name(last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT COUNT(LastName) FROM Employees WHERE LastName = ?", (last_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of products with a specific product ID and price
@app.get("/v1/sales/product_count_by_id_and_price", operation_id="get_product_count_by_id_and_price", summary="Retrieves the total number of products that have a product ID less than the provided value and a price equal to or less than the given price.")
async def get_product_count_by_id_and_price(product_id: int = Query(..., description="Product ID"), price: int = Query(..., description="Price of the product")):
    cursor.execute("SELECT COUNT(ProductID) FROM Products WHERE ProductID < ? AND Price <= ?", (product_id, price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the product with the highest price within a specific quantity range
@app.get("/v1/sales/highest_priced_product_by_quantity_range", operation_id="get_highest_priced_product", summary="Retrieves the product with the highest price that has been sold within the specified quantity range. The range is defined by the minimum and maximum quantity parameters, which determine the lower and upper bounds of the quantity, respectively.")
async def get_highest_priced_product(min_quantity: int = Query(..., description="Minimum quantity"), max_quantity: int = Query(..., description="Maximum quantity")):
    cursor.execute("SELECT T1.ProductID, T1.Name FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T2.quantity BETWEEN ? AND ? ORDER BY T1.Price DESC LIMIT 1", (min_quantity, max_quantity))
    result = cursor.fetchone()
    if not result:
        return {"product": {}}
    return {"product": {"product_id": result[0], "name": result[1]}}

# Endpoint to get the customer with the highest quantity of sales based on first name
@app.get("/v1/sales/highest_quantity_customer_by_first_name", operation_id="get_highest_quantity_customer", summary="Retrieves the customer with the highest sales quantity based on the provided first name. The operation returns the full name of the customer who has made the most sales with the given first name.")
async def get_highest_quantity_customer(first_name: str = Query(..., description="First name of the customer")):
    cursor.execute("SELECT T2.FirstName, T2.LastName FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.FirstName = ? ORDER BY T1.Quantity DESC LIMIT 1", (first_name,))
    result = cursor.fetchone()
    if not result:
        return {"customer": {}}
    return {"customer": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get customer and sales IDs based on product price range and quantity
@app.get("/v1/sales/customer_sales_ids_by_price_range_and_quantity", operation_id="get_customer_sales_ids", summary="Retrieves the customer and sales IDs for products that fall within a specified price range and have a quantity less than a given maximum. The price range is defined by a minimum and maximum price, and the maximum quantity is provided as an input parameter.")
async def get_customer_sales_ids(min_price: int = Query(..., description="Minimum price of the product"), max_price: int = Query(..., description="Maximum price of the product"), max_quantity: int = Query(..., description="Maximum quantity")):
    cursor.execute("SELECT T2.CustomerID, T2.SalesID FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Price BETWEEN ? AND ? AND T2.Quantity < ?", (min_price, max_price, max_quantity))
    result = cursor.fetchall()
    if not result:
        return {"customer_sales_ids": []}
    return {"customer_sales_ids": [{"customer_id": row[0], "sales_id": row[1]} for row in result]}

# Endpoint to get the quantity and price of products sold to a specific customer
@app.get("/v1/sales/product_quantity_price_by_customer", operation_id="get_product_quantity_price_by_customer", summary="Retrieve the total quantity and individual prices of products sold to a customer identified by their first and last name. The response includes a list of products, their respective quantities, and prices.")
async def get_product_quantity_price_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T2.Quantity, T1.Price FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Customers AS T3 ON T2.CustomerID = T3.CustomerID WHERE T3.FirstName = ? AND T3.LastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of products sold with a specific quantity and price limit
@app.get("/v1/sales/product_count_by_quantity_price", operation_id="get_product_count_by_quantity_price", summary="Retrieves the total count of products that have been sold in a specific quantity and are priced at or below a given limit. This operation provides a quantitative overview of product sales based on the provided quantity and price limit.")
async def get_product_count_by_quantity_price(quantity: int = Query(..., description="Quantity of the product"), price_limit: float = Query(..., description="Price limit of the product")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T2.quantity = ? AND T1.Price <= ?", (quantity, price_limit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of products sold to a specific customer with a quantity less than a specified value
@app.get("/v1/sales/product_count_by_customer_quantity", operation_id="get_product_count_by_customer_quantity", summary="Retrieves the total number of distinct products sold to a customer, filtered by a specified first name, where the quantity of each product sold is less than a given limit.")
async def get_product_count_by_customer_quantity(first_name: str = Query(..., description="First name of the customer"), quantity_limit: int = Query(..., description="Quantity limit of the product")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.FirstName = ? AND T1.Quantity < ?", (first_name, quantity_limit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the name of the product with the highest quantity sold to a specific customer
@app.get("/v1/sales/top_product_by_customer", operation_id="get_top_product_by_customer", summary="Retrieves the name of the product that has been sold the most to a specific customer. The operation requires the first and last name of the customer to identify the sales records and determine the top-selling product.")
async def get_top_product_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T1.Name FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Customers AS T3 ON T2.CustomerID = T3.CustomerID WHERE T3.FirstName = ? AND T3.LastName = ? ORDER BY T2.Quantity DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get distinct prices and quantities of a specific product sold
@app.get("/v1/sales/product_price_quantity", operation_id="get_product_price_quantity", summary="Retrieves unique price and quantity combinations for a specified product from the sales records.")
async def get_product_price_quantity(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT DISTINCT T2.Price, T1.Quantity FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the price and name of products sold to a specific customer
@app.get("/v1/sales/product_price_name_by_customer", operation_id="get_product_price_name_by_customer", summary="Retrieves the price and name of products sold to a specific customer, identified by their first and last names. The operation filters sales records based on the provided customer names and returns the corresponding product details.")
async def get_product_price_name_by_customer(first_name: str = Query(..., description="First name of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT T3.Price, T3.Name FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Products AS T3 ON T1.ProductID = T3.ProductID WHERE T2.FirstName = ? AND T2.LastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the sales IDs of a specific product sold with a specific quantity
@app.get("/v1/sales/sales_id_by_product_quantity", operation_id="get_sales_id_by_product_quantity", summary="Retrieves the unique sales identifiers for a specific product sold in a specific quantity. The operation filters sales data based on the provided product name and quantity, returning the corresponding sales IDs.")
async def get_sales_id_by_product_quantity(product_name: str = Query(..., description="Name of the product"), quantity: int = Query(..., description="Quantity of the product")):
    cursor.execute("SELECT T1.SalesID FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ? AND T1.Quantity = ?", (product_name, quantity))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the names of customers who bought a specific quantity of products within a range of sales IDs
@app.get("/v1/sales/customer_names_by_quantity_sales_id_range", operation_id="get_customer_names_by_quantity_sales_id_range", summary="Retrieve the first and last names of customers who purchased a specified quantity of products within a defined range of sales IDs. The operation filters sales data based on the provided quantity and sales ID range, and then matches the filtered sales with corresponding customer records.")
async def get_customer_names_by_quantity_sales_id_range(quantity: int = Query(..., description="Quantity of the product"), min_sales_id: int = Query(..., description="Minimum sales ID"), max_sales_id: int = Query(..., description="Maximum sales ID")):
    cursor.execute("SELECT T2.FirstName, T2.LastName FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Quantity = ? AND T1.SalesID BETWEEN ? AND ?", (quantity, min_sales_id, max_sales_id))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the customer IDs and last names of customers who bought more than a certain percentage of the average quantity
@app.get("/v1/sales/customer_ids_by_quantity_percentage", operation_id="get_customer_ids_by_quantity_percentage", summary="Retrieves the IDs and last names of customers who have purchased more than the specified percentage of the average sales quantity. This operation provides a list of customers who are above the given percentage threshold, offering insights into high-volume buyers.")
async def get_customer_ids_by_quantity_percentage(percentage: float = Query(..., description="Percentage of the average quantity")):
    cursor.execute("SELECT T2.CustomerID, T2.LastName FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Quantity > ( SELECT AVG(Quantity) FROM Sales ) * ?", (percentage,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of products sold within a specific price range and sales ID range
@app.get("/v1/sales/percentage_products_by_price_sales_id_range", operation_id="get_percentage_products_by_price_sales_id_range", summary="Retrieves the percentage of products sold within a specified price range and sales ID range. The calculation is based on the count of products sold within the given price range and the total number of products sold within the provided sales ID range.")
async def get_percentage_products_by_price_sales_id_range(min_price: float = Query(..., description="Minimum price of the product"), max_price: float = Query(..., description="Maximum price of the product"), min_sales_id: int = Query(..., description="Minimum sales ID"), max_sales_id: int = Query(..., description="Maximum sales ID")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Price BETWEEN ? AND ?, 1, 0)) AS REAL) * 100 / COUNT(T2.Price) FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID WHERE T1.SalesID BETWEEN ? AND ?", (min_price, max_price, min_sales_id, max_sales_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the name of the most expensive product
@app.get("/v1/sales/most_expensive_product", operation_id="get_most_expensive_product", summary="Get the name of the most expensive product")
async def get_most_expensive_product():
    cursor.execute("SELECT Name FROM Products WHERE Price = ( SELECT MAX(Price) FROM Products )")
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the count of products with a specific name pattern
@app.get("/v1/sales/product_count_by_name_pattern", operation_id="get_product_count_by_name_pattern", summary="Retrieves the total count of products whose names match the provided pattern. The pattern can include wildcard characters to broaden the search scope.")
async def get_product_count_by_name_pattern(name_pattern: str = Query(..., description="Pattern to match in the product name (use %% for wildcard)")):
    cursor.execute("SELECT COUNT(ProductID) FROM Products WHERE Name LIKE ?", (name_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers with the most common last name
@app.get("/v1/sales/most_common_last_name_customer_count", operation_id="get_most_common_last_name_customer_count", summary="Retrieves the total number of customers who share the most frequently occurring last name in the database. This operation provides a statistical insight into the customer base, helping to identify the most common surname among customers.")
async def get_most_common_last_name_customer_count():
    cursor.execute("SELECT COUNT(CustomerID) FROM Customers GROUP BY LastName ORDER BY COUNT(LastName) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the employee with the most sales
@app.get("/v1/sales/top_employee_by_sales", operation_id="get_top_employee_by_sales", summary="Retrieves the top-performing sales employee based on the highest number of sales. The operation returns the full name of the employee with the most sales records in the system.")
async def get_top_employee_by_sales():
    cursor.execute("SELECT T1.FirstName, T1.MiddleInitial, T1.LastName FROM Employees AS T1 INNER JOIN Sales AS T2 ON T2.SalesPersonID = T1.EmployeeID GROUP BY T2.SalesPersonID, T1.FirstName, T1.MiddleInitial, T1.LastName ORDER BY COUNT(T2.SalesID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "middle_initial": result[1], "last_name": result[2]}}

# Endpoint to get the customer with the highest total sales value
@app.get("/v1/sales/top_customer_by_sales_value", operation_id="get_top_customer_by_sales_value", summary="Retrieves the customer who has made the highest total sales value. This operation calculates the total sales value by multiplying the quantity of each product sold by its price, then sums these values for each customer. The customer with the highest total sales value is returned, including their full name.")
async def get_top_customer_by_sales_value():
    cursor.execute("SELECT T2.FirstName, T2.MiddleInitial, T2.LastName FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN Products AS T3 ON T1.ProductID = T3.ProductID GROUP BY T1.SalesID, T1.Quantity, T3.Price, FirstName, MiddleInitial, LastName ORDER BY T1.Quantity * T3.Price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"first_name": result[0], "middle_initial": result[1], "last_name": result[2]}}

# Endpoint to get the total sales value for a specific employee
@app.get("/v1/sales/total_sales_value_by_employee", operation_id="get_total_sales_value_by_employee", summary="Retrieves the total sales value generated by a specific employee. This operation calculates the sum of the sales value for all products sold by the employee, based on the quantity of each product sold and its respective price. The employee is identified by their first and last names.")
async def get_total_sales_value_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT SUM(T2.Quantity * T3.Price) FROM Employees AS T1 INNER JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"total_sales_value": []}
    return {"total_sales_value": result[0]}

# Endpoint to get the total quantity sold by a specific employee for a specific product
@app.get("/v1/sales/total_quantity_sold_by_employee_and_product", operation_id="get_total_quantity_sold_by_employee_and_product", summary="Retrieves the total quantity of a specific product sold by a particular employee. The operation requires the first and last name of the employee, as well as the name of the product. The result is the sum of the quantities sold for the specified product by the given employee.")
async def get_total_quantity_sold_by_employee_and_product(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(T2.Quantity) FROM Employees AS T1 INNER JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T1.FirstName = ? AND T1.LastName = ? AND T3.Name = ?", (first_name, last_name, product_name))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the count of products purchased by a specific customer
@app.get("/v1/sales/product_count_by_customer", operation_id="get_product_count_by_customer", summary="Retrieves the total number of products purchased by a customer identified by their full name. The customer's first name, middle initial, and last name are used to accurately determine the product count.")
async def get_product_count_by_customer(first_name: str = Query(..., description="First name of the customer"), middle_initial: str = Query(..., description="Middle initial of the customer"), last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT COUNT(T2.ProductID) FROM Customers AS T1 INNER JOIN Sales AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.FirstName = ? AND T1.MiddleInitial = ? AND T1.LastName = ?", (first_name, middle_initial, last_name))
    result = cursor.fetchone()
    if not result:
        return {"product_count": []}
    return {"product_count": result[0]}

# Endpoint to get the top N employees by sales count
@app.get("/v1/sales/top_employees_by_sales", operation_id="get_top_employees_by_sales", summary="Retrieves a list of the top N employees with the highest sales count. The list is sorted in descending order based on the number of sales made by each employee. The 'limit' parameter determines the number of top employees to return.")
async def get_top_employees_by_sales(limit: int = Query(..., description="Number of top employees to return")):
    cursor.execute("SELECT T1.FirstName, T1.MiddleInitial, T1.LastName FROM Employees AS T1 INNER JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID GROUP BY T2.SalesPersonID, T1.FirstName, T1.MiddleInitial, T1.LastName ORDER BY COUNT(T2.SalesID) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the top product by sales quantity with a specific name pattern
@app.get("/v1/sales/top_product_by_name_pattern", operation_id="get_top_product_by_name_pattern", summary="Retrieves the top-selling products with names matching the provided pattern, sorted by total sales quantity in descending order. The number of products returned is limited by the specified quantity.")
async def get_top_product_by_name_pattern(name_pattern: str = Query(..., description="Name pattern to match (use % for wildcard)"), limit: int = Query(..., description="Number of top products to return")):
    cursor.execute("SELECT T1.Name FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Name LIKE ? GROUP BY T2.Quantity, T1.Name ORDER BY SUM(T2.Quantity) DESC LIMIT ?", (name_pattern, limit))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the quantity of a specific product in a specific sale
@app.get("/v1/sales/product_quantity_in_sale", operation_id="get_product_quantity_in_sale", summary="Retrieves the quantity of a specific product in a given sale. The operation requires the product's name and the sale's unique identifier to accurately determine the quantity.")
async def get_product_quantity_in_sale(product_name: str = Query(..., description="Name of the product"), sales_id: int = Query(..., description="Sales ID")):
    cursor.execute("SELECT T1.Quantity FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Name = ? AND T1.SalesID = ?", (product_name, sales_id))
    result = cursor.fetchone()
    if not result:
        return {"quantity": []}
    return {"quantity": result[0]}

# Endpoint to get the count of salespersons with total quantity sold above a threshold for a specific product
@app.get("/v1/sales/count_salespersons_above_quantity_threshold", operation_id="get_count_salespersons_above_quantity_threshold", summary="Retrieves the number of salespersons who have sold a specific product in quantities exceeding a defined threshold. The product is identified by its name, and the threshold is set as a quantity value. This operation provides insights into sales performance by individual salespersons for a particular product.")
async def get_count_salespersons_above_quantity_threshold(product_name: str = Query(..., description="Name of the product"), quantity_threshold: int = Query(..., description="Quantity threshold")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT SUM(Quantity) FROM Sales WHERE ProductID IN ( SELECT ProductID FROM Products WHERE Name = ? ) GROUP BY Quantity, SalesPersonID HAVING SUM(Quantity) > ? )", (product_name, quantity_threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total sales amount for a specific employee and product
@app.get("/v1/sales/total_sales_amount_employee_product", operation_id="get_total_sales_amount_employee_product", summary="Retrieves the total sales amount for a specific employee and product. The operation calculates the sum of the sales quantity multiplied by the product price for the given employee and product. The employee is identified by their first name, middle initial, and last name. The product is specified by its name.")
async def get_total_sales_amount_employee_product(first_name: str = Query(..., description="First name of the employee"), middle_initial: str = Query(..., description="Middle initial of the employee"), last_name: str = Query(..., description="Last name of the employee"), product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(T2.Quantity * T3.Price) FROM Employees AS T1 INNER JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T1.FirstName = ? AND T1.MiddleInitial = ? AND T1.LastName = ? AND T3.Name = ?", (first_name, middle_initial, last_name, product_name))
    result = cursor.fetchone()
    if not result:
        return {"total_sales_amount": []}
    return {"total_sales_amount": result[0]}

# Endpoint to get the top product by total sales amount
@app.get("/v1/sales/top_product_by_sales_amount", operation_id="get_top_product_by_sales_amount", summary="Retrieves the top-selling products, ranked by total sales amount, up to a specified limit. The operation calculates the total sales amount for each product by multiplying the quantity sold by the product's price, then summing the results. The products are then ordered by their total sales amount in descending order, and the top products, up to the specified limit, are returned.")
async def get_top_product_by_sales_amount(limit: int = Query(..., description="Number of top products to return")):
    cursor.execute("SELECT T1.Name, SUM(T2.Quantity * T1.Price) FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID GROUP BY T1.ProductID, T1.Name ORDER BY SUM(T2.Quantity) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get customers with a specific last name
@app.get("/v1/sales/customers_by_last_name", operation_id="get_customers_by_last_name", summary="Retrieves the first and last names of customers who share a specified last name.")
async def get_customers_by_last_name(last_name: str = Query(..., description="Last name of the customer")):
    cursor.execute("SELECT FirstName, LastName FROM Customers WHERE LastName = ?", (last_name,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the most common middle initial among employees
@app.get("/v1/sales/most_common_middle_initial", operation_id="get_most_common_middle_initial", summary="Retrieves the most frequently occurring middle initials among employees, sorted in descending order of frequency. The number of top middle initials to return can be specified.")
async def get_most_common_middle_initial(limit: int = Query(..., description="Number of top middle initials to return")):
    cursor.execute("SELECT MiddleInitial FROM Employees GROUP BY MiddleInitial ORDER BY COUNT(MiddleInitial) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"middle_initials": []}
    return {"middle_initials": result}

# Endpoint to get the average price of products within a price range
@app.get("/v1/sales/average_price_range", operation_id="get_average_price_range", summary="Retrieves the average price of products that fall within the specified price range. The price range is defined by the minimum and maximum price values provided as input parameters.")
async def get_average_price_range(min_price: float = Query(..., description="Minimum price"), max_price: float = Query(..., description="Maximum price")):
    cursor.execute("SELECT AVG(Price) FROM Products WHERE Price BETWEEN ? AND ?", (min_price, max_price))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get customer details who purchased more than the average quantity
@app.get("/v1/sales/customer_details_above_avg_quantity", operation_id="get_customer_details_above_avg_quantity", summary="Retrieves the full names of customers who have purchased more than the average quantity of products. This operation calculates the average quantity of products sold and returns the names of customers who have exceeded this average. The data is sourced from the Sales and Customers tables, which are joined on the CustomerID field.")
async def get_customer_details_above_avg_quantity():
    cursor.execute("SELECT T2.FirstName, T2.MiddleInitial, T2.LastName FROM Sales AS T1 INNER JOIN Customers AS T2 ON T1.CustomerID = T2.CustomerID GROUP BY T1.Quantity HAVING T1.Quantity > ( SELECT AVG(Quantity) FROM Sales )")
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the top employee by total sales value
@app.get("/v1/sales/top_employee_by_sales_value", operation_id="get_top_employee_by_sales_value", summary="Get the top employee by total sales value")
async def get_top_employee_by_sales_value():
    cursor.execute("SELECT T1.FirstName, T1.MiddleInitial, T1.LastName FROM Employees AS T1 INNER JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID ORDER BY T2.Quantity * T3.Price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get distinct customers who spent more than a specified amount
@app.get("/v1/sales/customers_spent_above_amount", operation_id="get_customers_spent_above_amount", summary="Retrieves unique customer records for individuals who have spent more than a specified amount on purchases. This operation considers the total cost of each sale, calculated as the product quantity multiplied by the product price, to determine qualifying customers.")
async def get_customers_spent_above_amount(amount: int = Query(..., description="Amount spent by the customer")):
    cursor.execute("SELECT DISTINCT T3.FirstName, T3.MiddleInitial, T3.LastName FROM Products AS T1 INNER JOIN Sales AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Customers AS T3 ON T2.CustomerID = T3.CustomerID WHERE T2.Quantity * T1.Price > ?", (amount,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the top product by quantity sold
@app.get("/v1/sales/top_product_by_quantity", operation_id="get_top_product_by_quantity", summary="Retrieves the name of the product that has been sold in the highest quantity. The operation calculates the total quantity sold for each product and returns the product with the maximum quantity.")
async def get_top_product_by_quantity():
    cursor.execute("SELECT T2.Name FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID ORDER BY T1.Quantity DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result}

# Endpoint to get distinct products sold below the average quantity
@app.get("/v1/sales/products_below_avg_quantity", operation_id="get_products_below_avg_quantity", summary="Retrieves a list of unique product names that have been sold in quantities less than the average sales quantity. This operation does not require any input parameters and returns a JSON array of product names.")
async def get_products_below_avg_quantity():
    cursor.execute("SELECT DISTINCT T2.Name FROM Sales AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Quantity < ( SELECT AVG(Quantity) FROM Sales )")
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

api_calls = [
    "/v1/sales/count_sales_by_customer_id?customer_id=80",
    "/v1/sales/sum_quantities_by_sales_id_range?min_sales_id=1&max_sales_id=10",
    "/v1/sales/avg_quantity_by_sales_id_range?min_sales_id=20&max_sales_id=30",
    "/v1/sales/distinct_product_ids_with_max_quantity",
    "/v1/sales/count_distinct_product_ids_with_max_price",
    "/v1/sales/product_names_by_price?price=0",
    "/v1/sales/product_names_by_product_id_range?min_product_id=1&max_product_id=10",
    "/v1/sales/product_name_with_highest_quantity",
    "/v1/sales/count_customers_by_product_name?product_name=Hex%20Nut%209",
    "/v1/sales/count_sales_by_product_name?product_name=Flat%20Washer%208",
    "/v1/sales/distinct_product_names_by_salesperson?salesperson_id=10",
    "/v1/sales/customer_first_names_by_product?product_id=1",
    "/v1/sales/customer_last_names_by_sales_id?sales_id=178",
    "/v1/sales/distinct_product_ids_by_customer_first_name?first_name=Abby",
    "/v1/sales/distinct_product_ids_by_employee_first_name?first_name=Meander",
    "/v1/sales/employee_last_names_by_sales_id?sales_id=100",
    "/v1/sales/distinct_employee_first_names_by_customer_first_name?customer_first_name=Abigail",
    "/v1/sales/count_products_by_customer_and_price?customer_id=11782&price=0",
    "/v1/sales/customer_details_by_salesperson?salesperson_id=5",
    "/v1/sales/sales_ids_by_employee_first_letter?first_letter=s",
    "/v1/sales/highest_product_price_in_customer_range?min_customer_id=1&max_customer_id=100",
    "/v1/sales/customer_first_name_highest_quantity?last_name=Valdez",
    "/v1/sales/count_sales_by_employees?first_name_1=Morningstar&first_name_2=Heather&first_name_3=Dean",
    "/v1/sales/check_product_sold_to_customer?product_id=498&first_name=Alex",
    "/v1/sales/total_sales_amount_for_customer?first_name=Adam",
    "/v1/sales/total_sales_amount_for_product_range?min_product_id=400&max_product_id=500",
    "/v1/sales/total_quantity_sold_by_product_name?starting_letter=C",
    "/v1/sales/total_quantity_sold_to_customer?first_name=Adam",
    "/v1/sales/top_5_products_by_price",
    "/v1/sales/count_products_by_price?price=0",
    "/v1/sales/distinct_employee_names_by_quantity?quantity=1000",
    "/v1/sales/product_quantities_by_sales_id_range?min_sales_id=1&max_sales_id=10",
    "/v1/sales/compare_product_sales?product_name_1=HL%20Mountain%20Frame%20-%20Silver,%2042&product_name_2=HL%20Mountain%20Frame%20-%20Black,%2042&result_if_greater=Silver&result_if_lesser=Black",
    "/v1/sales/price_difference_between_products?product_name_1=HL%20Mountain%20Frame%20-%20Black,%2042&product_name_2=LL%20Mountain%20Frame%20-%20Black,%2042",
    "/v1/sales/count_sales_by_employee?first_name=Michel&middle_initial=e&last_name=DeFrance",
    "/v1/sales/customer_to_employee_ratio",
    "/v1/sales/count_customers_by_product_and_employee?first_name=Innes&last_name=del%20Castillo&product_name=Short-Sleeve%20Classic%20Jersey,%20L&middle_initial=e",
    "/v1/sales/distinct_employee_names_by_customer_and_product?customer_middle_initial=A&customer_last_name=White&product_name=Road-250%20Black,%2048&customer_first_name=Elizabeth",
    "/v1/sales/count_salespersons_by_product_name?product_name=Headlights%20-%20Weatherproof",
    "/v1/sales/total_sales_amount_by_product_name?product_name=HL%20Road%20Frame%20-%20Red%2C%2056",
    "/v1/sales/count_sales_by_customer_name?first_name=Joe&middle_initial=L&last_name=Lopez",
    "/v1/sales/distinct_customer_names_by_product_and_price?product_name=Touring%20Rim&product_price=0",
    "/v1/sales/count_customers_by_salesperson",
    "/v1/sales/count_employees",
    "/v1/sales/employee_name_by_id?employee_id=7",
    "/v1/sales/product_names_by_max_min_price?max_price=1000&min_price=10",
    "/v1/sales/percentage_sales_by_employee_and_product?first_name=Albert&middle_initial=I&last_name=Ringer&product_name=ML%20Bottom%20Bracket",
    "/v1/sales/count_customers_by_first_name?first_name=Abigail",
    "/v1/sales/distinct_quantities_by_product_name?product_name=Blade",
    "/v1/sales/top_employee_by_sales_quantity",
    "/v1/sales/top_customer_by_sales_quantity",
    "/v1/sales/top_product_by_salesperson?salesperson_id=20",
    "/v1/sales/total_sales_quantity_by_salesperson_customer_employee?salesperson_id=1&customer_first_name=Aaron&customer_last_name=Alexander&employee_first_name=Abraham",
    "/v1/sales/customers_by_min_sales_quantity?min_quantity=600",
    "/v1/sales/top_customer_by_first_name?first_name=Cameron",
    "/v1/sales/sales_ids_by_product_name_and_price?product_name_pattern=Hex%20Nut%25&min_price=100",
    "/v1/sales/distinct_customer_ids_by_price_range?min_price=1000&max_price=2000",
    "/v1/sales/total_quantity_by_price?price=0",
    "/v1/sales/percentage_sold_by_price?price=0",
    "/v1/sales/percentage_sold_by_product_name?product_name=Blade",
    "/v1/sales/employee_count_by_last_name?last_name=Ringer",
    "/v1/sales/product_count_by_id_and_price?product_id=15&price=10",
    "/v1/sales/highest_priced_product_by_quantity_range?min_quantity=400&max_quantity=500",
    "/v1/sales/highest_quantity_customer_by_first_name?first_name=Kate",
    "/v1/sales/customer_sales_ids_by_price_range_and_quantity?min_price=100&max_price=150&max_quantity=25",
    "/v1/sales/product_quantity_price_by_customer?first_name=Abigail&last_name=Henderson",
    "/v1/sales/product_count_by_quantity_price?quantity=60&price_limit=500",
    "/v1/sales/product_count_by_customer_quantity?first_name=Erica&quantity_limit=200",
    "/v1/sales/top_product_by_customer?first_name=Kathryn&last_name=Ashe",
    "/v1/sales/product_price_quantity?product_name=Seat%20Tube",
    "/v1/sales/product_price_name_by_customer?first_name=Erica&last_name=Xu",
    "/v1/sales/sales_id_by_product_quantity?product_name=External%20Lock%20Washer%207&quantity=590",
    "/v1/sales/customer_names_by_quantity_sales_id_range?quantity=403&min_sales_id=30&max_sales_id=40",
    "/v1/sales/customer_ids_by_quantity_percentage?percentage=0.9",
    "/v1/sales/percentage_products_by_price_sales_id_range?min_price=200&max_price=300&min_sales_id=1&max_sales_id=200",
    "/v1/sales/most_expensive_product",
    "/v1/sales/product_count_by_name_pattern?name_pattern=%25HL%20Touring%20Frame%25",
    "/v1/sales/most_common_last_name_customer_count",
    "/v1/sales/top_employee_by_sales",
    "/v1/sales/top_customer_by_sales_value",
    "/v1/sales/total_sales_value_by_employee?first_name=Heather&last_name=McBadden",
    "/v1/sales/total_quantity_sold_by_employee_and_product?first_name=Stearns&last_name=MacFeather&product_name=Mountain-100%20Silver%2C%2038",
    "/v1/sales/product_count_by_customer?first_name=Dalton&middle_initial=M&last_name=Coleman",
    "/v1/sales/top_employees_by_sales?limit=3",
    "/v1/sales/top_product_by_name_pattern?name_pattern=Mountain-500%20Black%25&limit=1",
    "/v1/sales/product_quantity_in_sale?product_name=Chainring%20Bolts&sales_id=551971",
    "/v1/sales/count_salespersons_above_quantity_threshold?product_name=Touring-2000%20Blue%2C%2050&quantity_threshold=20000",
    "/v1/sales/total_sales_amount_employee_product?first_name=Abraham&middle_initial=e&last_name=Bennet&product_name=Road-650%20Red%2C%2060",
    "/v1/sales/top_product_by_sales_amount?limit=1",
    "/v1/sales/customers_by_last_name?last_name=Chen",
    "/v1/sales/most_common_middle_initial?limit=1",
    "/v1/sales/average_price_range?min_price=100&max_price=200",
    "/v1/sales/customer_details_above_avg_quantity",
    "/v1/sales/top_employee_by_sales_value",
    "/v1/sales/customers_spent_above_amount?amount=50000",
    "/v1/sales/top_product_by_quantity",
    "/v1/sales/products_below_avg_quantity"
]
