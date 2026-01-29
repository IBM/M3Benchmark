from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/retail_world/retail_world.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of employees based on title of courtesy
@app.get("/v1/retail_world/employee_count_by_title_of_courtesy", operation_id="get_employee_count_by_title_of_courtesy", summary="Get the count of employees with a specific title of courtesy")
async def get_employee_count_by_title_of_courtesy(title_of_courtesy: str = Query(..., description="Title of courtesy of the employee")):
    cursor.execute("SELECT COUNT(EmployeeID) FROM Employees WHERE TitleOfCourtesy = ?", (title_of_courtesy,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last name of an employee based on the manager's name
@app.get("/v1/retail_world/employee_name_by_manager_name", operation_id="get_employee_name_by_manager_name", summary="Get the first and last name of an employee based on the manager's name")
async def get_employee_name_by_manager_name(manager_last_name: str = Query(..., description="Last name of the manager"), manager_first_name: str = Query(..., description="First name of the manager")):
    cursor.execute("SELECT FirstName, LastName FROM Employees WHERE EmployeeID = ( SELECT ReportsTo FROM Employees WHERE LastName = ? AND FirstName = ? )", (manager_last_name, manager_first_name))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the home phone number of an employee based on their name
@app.get("/v1/retail_world/employee_home_phone_by_name", operation_id="get_employee_home_phone_by_name", summary="Get the home phone number of an employee based on their name")
async def get_employee_home_phone_by_name(last_name: str = Query(..., description="Last name of the employee"), first_name: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT HomePhone FROM Employees WHERE LastName = ? AND FirstName = ?", (last_name, first_name))
    result = cursor.fetchone()
    if not result:
        return {"home_phone": []}
    return {"home_phone": result[0]}

# Endpoint to get the count of employees reporting to a specific manager
@app.get("/v1/retail_world/employee_count_by_manager_name", operation_id="get_employee_count_by_manager_name", summary="Get the count of employees reporting to a specific manager")
async def get_employee_count_by_manager_name(manager_last_name: str = Query(..., description="Last name of the manager"), manager_first_name: str = Query(..., description="First name of the manager")):
    cursor.execute("SELECT COUNT(EmployeeID) FROM Employees WHERE ReportsTo = ( SELECT EmployeeID FROM Employees WHERE LastName = ? AND FirstName = ? )", (manager_last_name, manager_first_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last name of the employee with the highest salary
@app.get("/v1/retail_world/employee_with_highest_salary", operation_id="get_employee_with_highest_salary", summary="Get the first and last name of the employee with the highest salary")
async def get_employee_with_highest_salary():
    cursor.execute("SELECT FirstName, LastName FROM Employees WHERE Salary = ( SELECT MAX(Salary) FROM Employees )")
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the salary difference between two employees
@app.get("/v1/retail_world/salary_difference_between_employees", operation_id="get_salary_difference_between_employees", summary="Get the salary difference between two employees")
async def get_salary_difference_between_employees(last_name_1: str = Query(..., description="Last name of the first employee"), first_name_1: str = Query(..., description="First name of the first employee"), last_name_2: str = Query(..., description="Last name of the second employee"), first_name_2: str = Query(..., description="First name of the second employee")):
    cursor.execute("SELECT ( SELECT Salary FROM Employees WHERE LastName = ? AND FirstName = ? ) - ( SELECT Salary FROM Employees WHERE LastName = ? AND FirstName = ? ) AS RESULT", (last_name_1, first_name_1, last_name_2, first_name_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of employees with a specific title and country
@app.get("/v1/retail_world/employee_count_by_title_and_country", operation_id="get_employee_count_by_title_and_country", summary="Get the count of employees with a specific title and country")
async def get_employee_count_by_title_and_country(title: str = Query(..., description="Title of the employee"), country: str = Query(..., description="Country of the employee")):
    cursor.execute("SELECT COUNT(Country) FROM Employees WHERE Title = ? AND Country = ?", (title, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last name of employees working in a specific territory
@app.get("/v1/retail_world/employee_names_by_territory", operation_id="get_employee_names_by_territory", summary="Get the first and last name of employees working in a specific territory")
async def get_employee_names_by_territory(territory_description: str = Query(..., description="Description of the territory")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T3.TerritoryDescription = ?", (territory_description,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the count of territories an employee is responsible for
@app.get("/v1/retail_world/territory_count_by_employee_name", operation_id="get_territory_count_by_employee_name", summary="Get the count of territories an employee is responsible for")
async def get_territory_count_by_employee_name(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT COUNT(T2.TerritoryID) FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the territory descriptions an employee is responsible for
@app.get("/v1/retail_world/territory_descriptions_by_employee_name", operation_id="get_territory_descriptions_by_employee_name", summary="Get the territory descriptions an employee is responsible for")
async def get_territory_descriptions_by_employee_name(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T3.TerritoryDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"territories": []}
    return {"territories": result}

# Endpoint to get the count of territories for employees in a specific city
@app.get("/v1/retail_world/count_territories_by_city", operation_id="get_count_territories_by_city", summary="Get the count of territories for employees in a specific city")
async def get_count_territories_by_city(city: str = Query(..., description="City of the employee")):
    cursor.execute("SELECT COUNT(T2.TerritoryID) FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.City = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the territory descriptions for employees reporting to a specific employee
@app.get("/v1/retail_world/territory_descriptions_by_reporting_employee", operation_id="get_territory_descriptions_by_reporting_employee", summary="Get the territory descriptions for employees reporting to a specific employee")
async def get_territory_descriptions_by_reporting_employee(first_name: str = Query(..., description="First name of the reporting employee"), last_name: str = Query(..., description="Last name of the reporting employee")):
    cursor.execute("SELECT T3.TerritoryDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T1.ReportsTo = ( SELECT EmployeeID FROM Employees WHERE FirstName = ? AND LastName = ? )", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"territory_descriptions": []}
    return {"territory_descriptions": [row[0] for row in result]}

# Endpoint to get the count of employees with more than a specified number of territories in a specific country
@app.get("/v1/retail_world/count_employees_by_country_and_territory_count", operation_id="get_count_employees_by_country_and_territory_count", summary="Get the count of employees with more than a specified number of territories in a specific country")
async def get_count_employees_by_country_and_territory_count(country: str = Query(..., description="Country of the employee"), min_territory_count: int = Query(..., description="Minimum number of territories")):
    cursor.execute("SELECT COUNT(COUNTEID) FROM ( SELECT T1.EmployeeID AS COUNTEID FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.Country = ? GROUP BY T1.EmployeeID HAVING COUNT(T2.TerritoryID) > ? ) T1", (country, min_territory_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of territories in a specific region
@app.get("/v1/retail_world/count_territories_by_region", operation_id="get_count_territories_by_region", summary="Get the count of territories in a specific region")
async def get_count_territories_by_region(region_description: str = Query(..., description="Description of the region")):
    cursor.execute("SELECT COUNT(T1.TerritoryID) FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ?", (region_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct territory descriptions in a specific region
@app.get("/v1/retail_world/distinct_territory_descriptions_by_region", operation_id="get_distinct_territory_descriptions_by_region", summary="Get distinct territory descriptions in a specific region")
async def get_distinct_territory_descriptions_by_region(region_description: str = Query(..., description="Description of the region")):
    cursor.execute("SELECT DISTINCT T1.TerritoryDescription FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ?", (region_description,))
    result = cursor.fetchall()
    if not result:
        return {"territory_descriptions": []}
    return {"territory_descriptions": [row[0] for row in result]}

# Endpoint to get the count of employees in a specific region
@app.get("/v1/retail_world/count_employees_by_region", operation_id="get_count_employees_by_region", summary="Get the count of employees in a specific region")
async def get_count_employees_by_region(region_description: str = Query(..., description="Description of the region")):
    cursor.execute("SELECT COUNT(T.EmployeeID) FROM ( SELECT T3.EmployeeID FROM Region AS T1 INNER JOIN Territories AS T2 ON T1.RegionID = T2.RegionID INNER JOIN EmployeeTerritories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T1.RegionDescription = ? GROUP BY T3.EmployeeID ) T", (region_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the home phones of employees in a specific region
@app.get("/v1/retail_world/home_phones_by_region", operation_id="get_home_phones_by_region", summary="Get the home phones of employees in a specific region")
async def get_home_phones_by_region(region_description: str = Query(..., description="Description of the region")):
    cursor.execute("SELECT T1.HomePhone FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID INNER JOIN Region AS T4 ON T3.RegionID = T4.RegionID WHERE T4.RegionDescription = ? GROUP BY T1.HomePhone", (region_description,))
    result = cursor.fetchall()
    if not result:
        return {"home_phones": []}
    return {"home_phones": [row[0] for row in result]}

# Endpoint to get the difference in the count of territories between two regions
@app.get("/v1/retail_world/territory_count_difference_by_regions", operation_id="get_territory_count_difference_by_regions", summary="Get the difference in the count of territories between two regions")
async def get_territory_count_difference_by_regions(region_description_1: str = Query(..., description="Description of the first region"), region_description_2: str = Query(..., description="Description of the second region")):
    cursor.execute("SELECT ( SELECT COUNT(T1.TerritoryID) FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ? ) - ( SELECT COUNT(T1.TerritoryID) FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ? ) AS Calu", (region_description_1, region_description_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the product names for a specific order
@app.get("/v1/retail_world/product_names_by_order", operation_id="get_product_names_by_order", summary="Get the product names for a specific order")
async def get_product_names_by_order(order_id: int = Query(..., description="ID of the order")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the quantity of a specific product in a specific order
@app.get("/v1/retail_world/product_quantity_by_order", operation_id="get_product_quantity_by_order", summary="Get the quantity of a specific product in a specific order")
async def get_product_quantity_by_order(order_id: int = Query(..., description="ID of the order"), product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.Quantity FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderID = ? AND T1.ProductName = ?", (order_id, product_name))
    result = cursor.fetchone()
    if not result:
        return {"quantity": []}
    return {"quantity": result[0]}

# Endpoint to get the total price of a product in a specific order
@app.get("/v1/retail_world/total_price_product_order", operation_id="get_total_price_product_order", summary="Get the total price of a product in a specific order")
async def get_total_price_product_order(order_id: int = Query(..., description="Order ID"), product_name: str = Query(..., description="Product Name")):
    cursor.execute("SELECT T2.UnitPrice * T2.Quantity FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderID = ? AND T1.ProductName = ?", (order_id, product_name))
    result = cursor.fetchone()
    if not result:
        return {"total_price": []}
    return {"total_price": result[0]}

# Endpoint to get the total stock and on-order units for the product with the highest quantity in a specific order
@app.get("/v1/retail_world/total_stock_on_order_highest_quantity", operation_id="get_total_stock_on_order_highest_quantity", summary="Get the total stock and on-order units for the product with the highest quantity in a specific order")
async def get_total_stock_on_order_highest_quantity(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.UnitsInStock + T1.UnitsOnOrder FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderID = ? ORDER BY T2.Quantity DESC LIMIT 1", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"total_stock_on_order": []}
    return {"total_stock_on_order": result[0]}

# Endpoint to get the product name with the highest reorder level in a specific order
@app.get("/v1/retail_world/product_name_highest_reorder_level", operation_id="get_product_name_highest_reorder_level", summary="Get the product name with the highest reorder level in a specific order")
async def get_product_name_highest_reorder_level(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderID = ? ORDER BY T1.ReorderLevel DESC LIMIT 1", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the count of orders for a specific product
@app.get("/v1/retail_world/count_orders_product", operation_id="get_count_orders_product", summary="Get the count of orders for a specific product")
async def get_count_orders_product(product_name: str = Query(..., description="Product Name")):
    cursor.execute("SELECT COUNT(T2.OrderID) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the order IDs for discontinued products
@app.get("/v1/retail_world/order_ids_discontinued_products", operation_id="get_order_ids_discontinued_products", summary="Get the order IDs for discontinued products")
async def get_order_ids_discontinued_products(discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT T2.OrderID FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Discontinued = ?", (discontinued,))
    result = cursor.fetchall()
    if not result:
        return {"order_ids": []}
    return {"order_ids": [row[0] for row in result]}

# Endpoint to get the count of orders where the unit price is less than the product's unit price
@app.get("/v1/retail_world/count_orders_unit_price_less", operation_id="get_count_orders_unit_price_less", summary="Get the count of orders where the unit price is less than the product's unit price")
async def get_count_orders_unit_price_less(product_name: str = Query(..., description="Product Name")):
    cursor.execute("SELECT COUNT(T2.OrderID) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductName = ? AND T2.UnitPrice < T1.UnitPrice", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the company name of the supplier for a specific product
@app.get("/v1/retail_world/supplier_company_name", operation_id="get_supplier_company_name", summary="Get the company name of the supplier for a specific product")
async def get_supplier_company_name(product_name: str = Query(..., description="Product Name")):
    cursor.execute("SELECT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the product names from suppliers in a specific country
@app.get("/v1/retail_world/product_names_by_country", operation_id="get_product_names_by_country", summary="Retrieves the names of products supplied by vendors based in a specified country. The country parameter is used to filter the results.")
async def get_product_names_by_country(country: str = Query(..., description="Country")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the distinct phone numbers of suppliers where the stock is less than the on-order units
@app.get("/v1/retail_world/supplier_phones_low_stock", operation_id="get_supplier_phones_low_stock", summary="Get the distinct phone numbers of suppliers where the stock is less than the on-order units")
async def get_supplier_phones_low_stock():
    cursor.execute("SELECT DISTINCT T2.Phone FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.UnitsInStock < T1.UnitsOnOrder")
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": [row[0] for row in result]}

# Endpoint to get the count of discontinued products from suppliers in a specific country
@app.get("/v1/retail_world/count_discontinued_products_by_country", operation_id="get_count_discontinued_products_by_country", summary="Get the count of discontinued products from suppliers in a specific country")
async def get_count_discontinued_products_by_country(country: str = Query(..., description="Country"), discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT COUNT(T1.Discontinued) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.Country = ? AND T1.Discontinued = ?", (country, discontinued))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the company names of suppliers with the highest unit price product
@app.get("/v1/retail_world/suppliers_with_highest_unit_price_product", operation_id="get_suppliers_with_highest_unit_price_product", summary="Get the company names of suppliers whose products have the highest unit price")
async def get_suppliers_with_highest_unit_price_product():
    cursor.execute("SELECT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.UnitPrice = ( SELECT MAX(UnitPrice) FROM Products )")
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get the average salary of employees managing more than a specified number of territories
@app.get("/v1/retail_world/avg_salary_employees_managing_territories", operation_id="get_avg_salary_employees_managing_territories", summary="Get the average salary of employees who manage more than a specified number of territories")
async def get_avg_salary_employees_managing_territories(min_territories: int = Query(..., description="Minimum number of territories managed by an employee")):
    cursor.execute("SELECT AVG(T1.Salary) FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID GROUP BY T1.EmployeeID HAVING COUNT(T2.TerritoryID) > ?", (min_territories,))
    result = cursor.fetchall()
    if not result:
        return {"avg_salaries": []}
    return {"avg_salaries": [row[0] for row in result]}

# Endpoint to get the percentage difference in unit price for a specific product in a given order
@app.get("/v1/retail_world/percentage_difference_unit_price", operation_id="get_percentage_difference_unit_price", summary="Get the percentage difference in unit price for a specific product in a given order")
async def get_percentage_difference_unit_price(order_id: int = Query(..., description="Order ID"), product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT CAST((T1.UnitPrice - T2.UnitPrice) AS REAL) * 100 / T1.UnitPrice FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderID = ? AND T1.ProductName = ?", (order_id, product_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the average quantity per order for a specific product
@app.get("/v1/retail_world/avg_quantity_per_order", operation_id="get_avg_quantity_per_order", summary="Get the average quantity per order for a specific product")
async def get_avg_quantity_per_order(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT CAST(SUM(T2.Quantity) AS REAL) / COUNT(T2.OrderID) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"avg_quantity": []}
    return {"avg_quantity": result[0]}

# Endpoint to get the product with the highest units on order to units in stock ratio for a specific order
@app.get("/v1/retail_world/product_highest_units_on_order_ratio", operation_id="get_product_highest_units_on_order_ratio", summary="Get the product with the highest units on order to units in stock ratio for a specific order")
async def get_product_highest_units_on_order_ratio(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderID = ? ORDER BY T1.UnitsOnOrder / T1.UnitsInStock DESC LIMIT 1", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the employee with the latest birth date
@app.get("/v1/retail_world/employee_latest_birth_date", operation_id="get_employee_latest_birth_date", summary="Retrieves the first and last name of the employee with the most recent birth date from the Employees table.")
async def get_employee_latest_birth_date():
    cursor.execute("SELECT FirstName, LastName FROM Employees WHERE BirthDate = ( SELECT MAX(BirthDate) FROM Employees )")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the employee with the latest hire date
@app.get("/v1/retail_world/employee_latest_hire_date", operation_id="get_employee_latest_hire_date", summary="Get the employee with the latest hire date")
async def get_employee_latest_hire_date():
    cursor.execute("SELECT FirstName, LastName FROM Employees WHERE HireDate = ( SELECT MAX(HireDate) FROM Employees )")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of orders for a specific employee
@app.get("/v1/retail_world/count_orders_employee", operation_id="get_count_orders_employee", summary="Retrieves the total number of orders associated with a specific employee, identified by their first and last names.")
async def get_count_orders_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT COUNT(T2.OrderID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the territory descriptions for a specific employee
@app.get("/v1/retail_world/territory_descriptions_employee", operation_id="get_territory_descriptions_employee", summary="Get the territory descriptions for a specific employee")
async def get_territory_descriptions_employee(title_of_courtesy: str = Query(..., description="Title of courtesy"), first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T3.TerritoryDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T1.TitleOfCourtesy = ? AND T1.FirstName = ? AND T1.LastName = ?", (title_of_courtesy, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"territory_descriptions": []}
    return {"territory_descriptions": [row[0] for row in result]}

# Endpoint to get the territory descriptions for the manager of a specific employee
@app.get("/v1/retail_world/territory_descriptions_manager", operation_id="get_territory_descriptions_manager", summary="Get the territory descriptions for the manager of a specific employee")
async def get_territory_descriptions_manager(title_of_courtesy: str = Query(..., description="Title of courtesy"), first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T3.TerritoryDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T1.EmployeeID = ( SELECT ReportsTo FROM Employees WHERE TitleOfCourtesy = ? AND FirstName = ? AND LastName = ? )", (title_of_courtesy, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"territory_descriptions": []}
    return {"territory_descriptions": [row[0] for row in result]}

# Endpoint to get the home phone numbers of employees in a specific territory
@app.get("/v1/retail_world/employee_home_phones_by_territory", operation_id="get_employee_home_phones_by_territory", summary="Get the home phone numbers of employees in a specific territory")
async def get_employee_home_phones_by_territory(territory_description: str = Query(..., description="Description of the territory")):
    cursor.execute("SELECT T1.HomePhone FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T3.TerritoryDescription = ?", (territory_description,))
    result = cursor.fetchall()
    if not result:
        return {"home_phones": []}
    return {"home_phones": result}

# Endpoint to get the total quantity of orders for a specific customer on a specific date
@app.get("/v1/retail_world/total_order_quantity_by_customer_and_date", operation_id="get_total_order_quantity", summary="Get the total quantity of orders for a specific customer on a specific date")
async def get_total_order_quantity(customer_id: str = Query(..., description="Customer ID"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD%' format")):
    cursor.execute("SELECT SUM(T2.Quantity) FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID WHERE T1.CustomerID = ? AND T1.OrderDate LIKE ?", (customer_id, order_date))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the count of product IDs for orders with the maximum freight
@app.get("/v1/retail_world/product_count_max_freight", operation_id="get_product_count_max_freight", summary="Get the count of product IDs for orders with the maximum freight")
async def get_product_count_max_freight():
    cursor.execute("SELECT COUNT(T2.ProductID) FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID WHERE T1.Freight = ( SELECT MAX(Freight) FROM Orders ) GROUP BY T1.OrderID")
    result = cursor.fetchall()
    if not result:
        return {"product_counts": []}
    return {"product_counts": result}

# Endpoint to get the company name of the shipper for a specific order
@app.get("/v1/retail_world/shipper_company_by_order", operation_id="get_shipper_company_by_order", summary="Get the company name of the shipper for a specific order")
async def get_shipper_company_by_order(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T2.CompanyName FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T1.OrderID = ?", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the count of orders shipped by a specific company
@app.get("/v1/retail_world/order_count_by_shipper", operation_id="get_order_count_by_shipper", summary="Get the count of orders shipped by a specific company")
async def get_order_count_by_shipper(company_name: str = Query(..., description="Company name of the shipper")):
    cursor.execute("SELECT COUNT(T1.OrderID) FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get the count of products in a specific category
@app.get("/v1/retail_world/product_count_by_category", operation_id="get_product_count_by_category", summary="Retrieves the total number of products that belong to a specified category. The category is identified by its name.")
async def get_product_count_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.CategoryName = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"product_count": []}
    return {"product_count": result[0]}

# Endpoint to get the category name of a specific product
@app.get("/v1/retail_world/category_by_product", operation_id="get_category_by_product", summary="Retrieves the category name associated with a specific product. The operation requires the product name as input and returns the corresponding category name.")
async def get_category_by_product(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT T2.CategoryName FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get the total units in stock and on order for products supplied by a specific company
@app.get("/v1/retail_world/total_units_by_supplier", operation_id="get_total_units_by_supplier", summary="Get the total units in stock and on order for products supplied by a specific company")
async def get_total_units_by_supplier(company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT SUM(T1.UnitsInStock + T1.UnitsOnOrder) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_units": []}
    return {"total_units": result[0]}

# Endpoint to get the reorder level of products supplied by a specific company
@app.get("/v1/retail_world/reorder_level_by_supplier", operation_id="get_reorder_level_by_supplier", summary="Get the reorder level of products supplied by a specific company")
async def get_reorder_level_by_supplier(company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT T1.ReorderLevel FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"reorder_levels": []}
    return {"reorder_levels": result}

# Endpoint to get the count of products from a specific supplier
@app.get("/v1/retail_world/product_count_by_supplier", operation_id="get_product_count_by_supplier", summary="Get the count of products supplied by a specific company")
async def get_product_count_by_supplier(company_name: str = Query(..., description="Name of the supplier company")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the contact name of the supplier for a specific product
@app.get("/v1/retail_world/supplier_contact_by_product", operation_id="get_supplier_contact_by_product", summary="Retrieves the contact name of the supplier associated with a specific product. The product is identified by its name, which is provided as an input parameter.")
async def get_supplier_contact_by_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.ContactName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"contact_name": []}
    return {"contact_name": result[0]}

# Endpoint to get the country of the supplier for a specific product
@app.get("/v1/retail_world/supplier_country_by_product", operation_id="get_supplier_country_by_product", summary="Retrieves the country of the supplier associated with a specific product. The product is identified by its name, which is provided as an input parameter.")
async def get_supplier_country_by_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.Country FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the region description for a specific territory
@app.get("/v1/retail_world/region_description_by_territory", operation_id="get_region_description_by_territory", summary="Get the region description for a specific territory")
async def get_region_description_by_territory(territory_description: str = Query(..., description="Description of the territory")):
    cursor.execute("SELECT T2.RegionDescription FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T1.TerritoryDescription = ?", (territory_description,))
    result = cursor.fetchone()
    if not result:
        return {"region_description": []}
    return {"region_description": result[0]}

# Endpoint to get the percentage of orders shipped by a specific company for a specific customer
@app.get("/v1/retail_world/order_percentage_by_shipper_and_customer", operation_id="get_order_percentage_by_shipper_and_customer", summary="Get the percentage of orders shipped by a specific company for a specific customer")
async def get_order_percentage_by_shipper_and_customer(company_name: str = Query(..., description="Name of the shipping company"), customer_id: str = Query(..., description="ID of the customer")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.CompanyName = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.OrderID) FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T1.CustomerID = ?", (company_name, customer_id))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage difference in orders between two shipping companies
@app.get("/v1/retail_world/order_percentage_difference_by_shippers", operation_id="get_order_percentage_difference_by_shippers", summary="Get the percentage difference in orders between two shipping companies")
async def get_order_percentage_difference_by_shippers(company_name_1: str = Query(..., description="Name of the first shipping company"), company_name_2: str = Query(..., description="Name of the second shipping company")):
    cursor.execute("SELECT CAST((COUNT(CASE WHEN T2.CompanyName = ? THEN 1 ELSE NULL END) - COUNT(CASE WHEN T2.CompanyName = ? THEN 1 ELSE NULL END)) AS REAL) * 100 / COUNT(CASE WHEN T2.CompanyName = ? THEN 1 ELSE NULL END) FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID", (company_name_1, company_name_2, company_name_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage_difference": []}
    return {"percentage_difference": result[0]}

# Endpoint to get the ratio of territory counts between two regions
@app.get("/v1/retail_world/territory_count_ratio_by_regions", operation_id="get_territory_count_ratio_by_regions", summary="Get the ratio of territory counts between two regions")
async def get_territory_count_ratio_by_regions(region_description_1: str = Query(..., description="Description of the first region"), region_description_2: str = Query(..., description="Description of the second region")):
    cursor.execute("SELECT CAST(( SELECT COUNT(T1.TerritoryID) FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ? ) AS REAL) / ( SELECT COUNT(T1.TerritoryID) FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ? ) AS Calu", (region_description_1, region_description_2))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the country with the highest number of customers
@app.get("/v1/retail_world/country_with_most_customers", operation_id="get_country_with_most_customers", summary="Retrieves the country with the most customers. This operation returns the country that has the highest number of customers in the database, based on the count of unique customer IDs. The result is determined by grouping customers by their country and ordering the count in descending order, with the top result being returned.")
async def get_country_with_most_customers():
    cursor.execute("SELECT COUNT(CustomerID) FROM Customers GROUP BY Country ORDER BY COUNT(CustomerID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get order IDs with freight cost greater than a specified value
@app.get("/v1/retail_world/orders_by_freight_cost", operation_id="get_orders_by_freight_cost", summary="Get order IDs with freight cost greater than a specified value")
async def get_orders_by_freight_cost(min_freight: float = Query(..., description="Minimum freight cost")):
    cursor.execute("SELECT OrderID FROM Orders WHERE Freight > ?", (min_freight,))
    result = cursor.fetchall()
    if not result:
        return {"order_ids": []}
    return {"order_ids": [row[0] for row in result]}

# Endpoint to get customer details based on company name
@app.get("/v1/retail_world/customer_details_by_company_name", operation_id="get_customer_details", summary="Get customer details such as address, city, region, country, and postal code based on the company name")
async def get_customer_details(company_name: str = Query(..., description="Name of the company")):
    cursor.execute("SELECT Address, City, Region, Country, PostalCode FROM Customers WHERE CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the product with the highest unit price
@app.get("/v1/retail_world/product_with_highest_unit_price", operation_id="get_product_with_highest_unit_price", summary="Get the product name with the highest unit price")
async def get_product_with_highest_unit_price():
    cursor.execute("SELECT ProductName FROM Products WHERE UnitPrice = ( SELECT MAX(UnitPrice) FROM Products )")
    result = cursor.fetchall()
    if not result:
        return {"product_name": []}
    return {"product_name": result}

# Endpoint to get the count of suppliers based on country
@app.get("/v1/retail_world/supplier_count_by_country", operation_id="get_supplier_count", summary="Retrieves the total number of suppliers based in the specified country.")
async def get_supplier_count(country: str = Query(..., description="Country of the supplier")):
    cursor.execute("SELECT COUNT(SupplierID) FROM Suppliers WHERE Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the product with the highest reorder level
@app.get("/v1/retail_world/product_with_highest_reorder_level", operation_id="get_product_with_highest_reorder_level", summary="Get the product name with the highest reorder level")
async def get_product_with_highest_reorder_level():
    cursor.execute("SELECT ProductName FROM Products WHERE ReorderLevel = ( SELECT MAX(ReorderLevel) FROM Products )")
    result = cursor.fetchall()
    if not result:
        return {"product_name": []}
    return {"product_name": result}

# Endpoint to get the product names based on discontinued status
@app.get("/v1/retail_world/product_names_by_discontinued_status", operation_id="get_product_names_by_discontinued_status", summary="Get the product names based on whether they are discontinued or not")
async def get_product_names_by_discontinued_status(discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT ProductName FROM Products WHERE Discontinued = ?", (discontinued,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the average salary of employees based on title and country
@app.get("/v1/retail_world/average_salary_by_title_and_country", operation_id="get_average_salary", summary="Get the average salary of employees with a specific title and country")
async def get_average_salary(title: str = Query(..., description="Title of the employee"), country: str = Query(..., description="Country of the employee")):
    cursor.execute("SELECT AVG(Salary) FROM Employees WHERE Title = ? AND Country = ?", (title, country))
    result = cursor.fetchone()
    if not result:
        return {"average_salary": []}
    return {"average_salary": result[0]}

# Endpoint to get the most frequent shipper company name for a given ship country
@app.get("/v1/retail_world/most_frequent_shipper_by_country", operation_id="get_most_frequent_shipper", summary="Get the most frequent shipper company name for a given ship country")
async def get_most_frequent_shipper(ship_country: str = Query(..., description="Ship country")):
    cursor.execute("SELECT T2.CompanyName FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T1.ShipCountry = ? GROUP BY T2.CompanyName ORDER BY COUNT(T2.CompanyName) DESC LIMIT 1", (ship_country,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the count of orders shipped by a specific shipper for a specific customer
@app.get("/v1/retail_world/order_count_by_customer_and_shipper", operation_id="get_order_count_by_customer_and_shipper", summary="Get the count of orders shipped by a specific shipper for a specific customer")
async def get_order_count_by_customer_and_shipper(customer_id: str = Query(..., description="Customer ID"), company_name: str = Query(..., description="Name of the shipper company")):
    cursor.execute("SELECT COUNT(T1.OrderID) FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T1.CustomerID = ? AND T2.CompanyName = ?", (customer_id, company_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct supplier company names for products with the highest reorder level
@app.get("/v1/retail_world/supplier_names_with_highest_reorder_level", operation_id="get_supplier_names_with_highest_reorder_level", summary="Get distinct supplier company names for products with the highest reorder level")
async def get_supplier_names_with_highest_reorder_level():
    cursor.execute("SELECT DISTINCT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.ReorderLevel = ( SELECT MAX(ReorderLevel) FROM Products )")
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": result}

# Endpoint to get the count of discontinued products from a specific country
@app.get("/v1/retail_world/discontinued_product_count_by_country", operation_id="get_discontinued_product_count_by_country", summary="Get the count of discontinued products supplied from a specific country")
async def get_discontinued_product_count_by_country(discontinued: int = Query(..., description="Discontinued status (1 for discontinued)"), country: str = Query(..., description="Country of the supplier")):
    cursor.execute("SELECT COUNT(T1.Discontinued) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.Discontinued = ? AND T2.Country = ?", (discontinued, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top orders by total cost
@app.get("/v1/retail_world/top_orders_by_total_cost", operation_id="get_top_orders_by_total_cost", summary="Get the top orders by total cost (unit price * quantity + freight)")
async def get_top_orders_by_total_cost(limit: int = Query(..., description="Number of top orders to retrieve")):
    cursor.execute("SELECT T2.UnitPrice * T2.Quantity + T1.Freight FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID ORDER BY T2.UnitPrice * T2.Quantity + T1.Freight DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"orders": []}
    return {"orders": result}

# Endpoint to get the top supplier by product count
@app.get("/v1/retail_world/top_supplier_by_product_count", operation_id="get_top_supplier_by_product_count", summary="Retrieves the top supplier(s) who have supplied the most products, based on the specified limit. The response is sorted in descending order by the count of products supplied.")
async def get_top_supplier_by_product_count(limit: int = Query(..., description="Number of top suppliers to retrieve")):
    cursor.execute("SELECT T1.SupplierID FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID GROUP BY T1.SupplierID ORDER BY COUNT(*) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"suppliers": []}
    return {"suppliers": result}

# Endpoint to get distinct product names with unit price below a certain value
@app.get("/v1/retail_world/distinct_product_names_by_unit_price", operation_id="get_distinct_product_names_by_unit_price", summary="Get distinct product names with unit price below a specified value")
async def get_distinct_product_names_by_unit_price(unit_price: float = Query(..., description="Maximum unit price")):
    cursor.execute("SELECT DISTINCT T1.ProductName FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.UnitPrice < ?", (unit_price,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the top employee title by territory count
@app.get("/v1/retail_world/top_employee_title_by_territory_count", operation_id="get_top_employee_title_by_territory_count", summary="Get the top employee title based on the number of territories managed")
async def get_top_employee_title_by_territory_count(limit: int = Query(..., description="Number of top employee titles to retrieve")):
    cursor.execute("SELECT T1.Title FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID GROUP BY T1.Title ORDER BY COUNT(T2.TerritoryID) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get the top product by order count
@app.get("/v1/retail_world/top_product_by_order_count", operation_id="get_top_product_by_order_count", summary="Get the top product based on the number of orders")
async def get_top_product_by_order_count(limit: int = Query(..., description="Number of top products to retrieve")):
    cursor.execute("SELECT T1.ProductID FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID GROUP BY T1.ProductID ORDER BY COUNT(*) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get territory descriptions based on title of courtesy
@app.get("/v1/retail_world/territory_descriptions_by_title_of_courtesy", operation_id="get_territory_descriptions_by_title_of_courtesy", summary="Get territory descriptions for employees with a specific title of courtesy")
async def get_territory_descriptions_by_title_of_courtesy(title_of_courtesy: str = Query(..., description="Title of courtesy (e.g., Dr.)")):
    cursor.execute("SELECT T3.TerritoryDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T1.TitleOfCourtesy = ?", (title_of_courtesy,))
    result = cursor.fetchall()
    if not result:
        return {"territory_descriptions": []}
    return {"territory_descriptions": result}

# Endpoint to get the average number of territories per employee by title
@app.get("/v1/retail_world/average_territories_per_employee_by_title", operation_id="get_average_territories_per_employee_by_title", summary="Get the average number of territories managed by employees with a specific title")
async def get_average_territories_per_employee_by_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT CAST(COUNT(T2.TerritoryID) AS REAL) / COUNT(DISTINCT T1.EmployeeID) FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.Title = ?", (title,))
    result = cursor.fetchone()
    if not result:
        return {"average_territories": []}
    return {"average_territories": result[0]}

# Endpoint to get the employee with the highest number of orders
@app.get("/v1/retail_world/employee_highest_orders", operation_id="get_employee_highest_orders", summary="Retrieves the employee who has handled the most orders. This operation returns the first and last name of the employee with the highest order count, based on the joined data from the Employees and Orders tables.")
async def get_employee_highest_orders():
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID GROUP BY T1.FirstName, T1.LastName ORDER BY COUNT(*) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the product with the highest freight count
@app.get("/v1/retail_world/product_highest_freight_count", operation_id="get_product_highest_freight_count", summary="Get the product with the highest freight count")
async def get_product_highest_freight_count():
    cursor.execute("SELECT COUNT(T2.ProductID) FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID GROUP BY T2.ProductID ORDER BY COUNT(T1.Freight) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product_count": []}
    return {"product_count": result[0]}

# Endpoint to get the product name with the highest combined units in stock and on order in a specific category
@app.get("/v1/retail_world/product_name_highest_combined_units", operation_id="get_product_name_highest_combined_units", summary="Get the product name with the highest combined units in stock and on order in a specific category")
async def get_product_name_highest_combined_units(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.CategoryName = ? ORDER BY T1.UnitsInStock + T1.UnitsOnOrder DESC LIMIT 1", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get the salary range of the top two employees by number of territories
@app.get("/v1/retail_world/salary_range_top_two_employees", operation_id="get_salary_range_top_two_employees", summary="Get the salary range of the top two employees by number of territories")
async def get_salary_range_top_two_employees():
    cursor.execute("SELECT MAX(Salary) - MIN(Salary) FROM ( SELECT T1.Salary FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID GROUP BY T1.EmployeeID, T1.Salary ORDER BY COUNT(T2.TerritoryID) DESC LIMIT 2 ) T1")
    result = cursor.fetchone()
    if not result:
        return {"salary_range": []}
    return {"salary_range": result[0]}

# Endpoint to get the average unit price of products with stock greater than a specified amount
@app.get("/v1/retail_world/average_unit_price_above_stock", operation_id="get_average_unit_price_above_stock", summary="Get the average unit price of products with stock greater than a specified amount")
async def get_average_unit_price_above_stock(units_in_stock: int = Query(..., description="Minimum units in stock")):
    cursor.execute("SELECT SUM(UnitPrice) / COUNT(UnitPrice) FROM Products WHERE UnitsInStock > ?", (units_in_stock,))
    result = cursor.fetchone()
    if not result:
        return {"average_unit_price": []}
    return {"average_unit_price": result[0]}

# Endpoint to get the company names in the city with the highest number of relationships
@app.get("/v1/retail_world/company_names_highest_relationships_city", operation_id="get_company_names_highest_relationships_city", summary="Get the company names in the city with the highest number of relationships")
async def get_company_names_highest_relationships_city():
    cursor.execute("SELECT CompanyName FROM `Customer and Suppliers by City` WHERE CITY = ( SELECT City FROM `Customer and Suppliers by City` GROUP BY City ORDER BY COUNT(Relationship) DESC LIMIT 1 )")
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get customer details in a specific city
@app.get("/v1/retail_world/customer_details_by_city", operation_id="get_customer_details_by_city", summary="Get customer details in a specific city")
async def get_customer_details_by_city(city: str = Query(..., description="Name of the city")):
    cursor.execute("SELECT CompanyName, ContactName, ContactTitle FROM Customers WHERE City = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"customer_details": []}
    return {"customer_details": [{"company_name": row[0], "contact_name": row[1], "contact_title": row[2]} for row in result]}

# Endpoint to get employee details ordered by birth date
@app.get("/v1/retail_world/employee_details_by_birthdate", operation_id="get_employee_details_by_birthdate", summary="Get employee details ordered by birth date")
async def get_employee_details_by_birthdate():
    cursor.execute("SELECT FirstName, LastName, HomePhone FROM Employees ORDER BY BirthDate DESC")
    result = cursor.fetchall()
    if not result:
        return {"employee_details": []}
    return {"employee_details": [{"first_name": row[0], "last_name": row[1], "home_phone": row[2]} for row in result]}

# Endpoint to get the average unit price of invoices in a specific country
@app.get("/v1/retail_world/average_unit_price_by_country", operation_id="get_average_unit_price_by_country", summary="Get the average unit price of invoices in a specific country")
async def get_average_unit_price_by_country(country: str = Query(..., description="Name of the country")):
    cursor.execute("SELECT AVG(UnitPrice) AS avg FROM Invoices WHERE Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"average_unit_price": []}
    return {"average_unit_price": result[0]}

# Endpoint to get the top product by total quantity ordered
@app.get("/v1/retail_world/top_product_by_quantity", operation_id="get_top_product_by_quantity", summary="Get the product ID with the highest total quantity ordered")
async def get_top_product_by_quantity():
    cursor.execute("SELECT ProductID FROM `Order Details` GROUP BY ProductID ORDER BY SUM(Quantity) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product_id": []}
    return {"product_id": result[0]}

# Endpoint to get supplier details for products with a unit price greater than a specified value
@app.get("/v1/retail_world/supplier_details_by_unit_price", operation_id="get_supplier_details_by_unit_price", summary="Get supplier details for products with a unit price greater than the specified value")
async def get_supplier_details_by_unit_price(unit_price: float = Query(..., description="Unit price threshold")):
    cursor.execute("SELECT T2.CompanyName, T2.Address, T2.Phone FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.UnitPrice > ?", (unit_price,))
    result = cursor.fetchall()
    if not result:
        return {"supplier_details": []}
    return {"supplier_details": result}

# Endpoint to get distinct customer details for orders with freight greater than the average freight
@app.get("/v1/retail_world/customer_details_above_avg_freight", operation_id="get_customer_details_above_avg_freight", summary="Get distinct customer details for orders with freight greater than the average freight")
async def get_customer_details_above_avg_freight():
    cursor.execute("SELECT DISTINCT T1.CompanyName, T1.Address FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Freight > ( SELECT AVG(Freight) FROM Orders )")
    result = cursor.fetchall()
    if not result:
        return {"customer_details": []}
    return {"customer_details": result}

# Endpoint to get distinct supplier company names for discontinued products from a specific country
@app.get("/v1/retail_world/supplier_companies_discontinued_products", operation_id="get_supplier_companies_discontinued_products", summary="Get distinct supplier company names for discontinued products from a specific country")
async def get_supplier_companies_discontinued_products(discontinued: int = Query(..., description="Discontinued status (1 for discontinued)"), country: str = Query(..., description="Country")):
    cursor.execute("SELECT DISTINCT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.Discontinued = ? AND T2.Country != ?", (discontinued, country))
    result = cursor.fetchall()
    if not result:
        return {"supplier_companies": []}
    return {"supplier_companies": result}

# Endpoint to get product names ordered in a specific year, sorted by total price and freight
@app.get("/v1/retail_world/product_names_by_order_date", operation_id="get_product_names_by_order_date", summary="Get product names ordered in a specific year, sorted by total price and freight")
async def get_product_names_by_order_date(order_date: str = Query(..., description="Order date in 'YYYY%' format"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.ProductName FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T1.OrderDate LIKE ? ORDER BY T3.UnitPrice + T1.Freight DESC LIMIT ?", (order_date, limit))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the count of customers for a specific shipper name
@app.get("/v1/retail_world/customer_count_by_shipper", operation_id="get_customer_count_by_shipper", summary="Get the count of customers for a specific shipper name")
async def get_customer_count_by_shipper(ship_name: str = Query(..., description="Shipper name")):
    cursor.execute("SELECT COUNT(T3.CustomerID) FROM Shippers AS T1 INNER JOIN Orders AS T2 ON T1.ShipperID = T2.ShipVia INNER JOIN Customers AS T3 ON T2.CustomerID = T3.CustomerID WHERE T2.ShipName = ?", (ship_name,))
    result = cursor.fetchone()
    if not result:
        return {"customer_count": []}
    return {"customer_count": result[0]}

# Endpoint to get the top products by order count
@app.get("/v1/retail_world/top_products_by_order_count", operation_id="get_top_products_by_order_count", summary="Get the top products by order count")
async def get_top_products_by_order_count(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T3.ProductName FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID GROUP BY T3.ProductName ORDER BY COUNT(*) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get distinct employee details with region description
@app.get("/v1/retail_world/employee_details_with_region", operation_id="get_employee_details_with_region", summary="Get distinct employee details with region description")
async def get_employee_details_with_region():
    cursor.execute("SELECT DISTINCT T1.FirstName, T1.LastName, T4.RegionDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID INNER JOIN Region AS T4 ON T3.RegionID = T4.RegionID ORDER BY T1.FirstName")
    result = cursor.fetchall()
    if not result:
        return {"employee_details": []}
    return {"employee_details": result}

# Endpoint to get employee details based on order date range
@app.get("/v1/retail_world/employee_details_by_order_date_range", operation_id="get_employee_details_by_order_date_range", summary="Get employee details who had the most orders within a specified date range")
async def get_employee_details_by_order_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD HH:MM:SS' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT FirstName, LastName, Title, address FROM Employees WHERE EmployeeID = ( SELECT T1.EmployeeID FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.OrderDate BETWEEN ? AND ? GROUP BY T1.EmployeeID ORDER BY COUNT(T2.OrderID) DESC LIMIT 1 )", (start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"employee_details": []}
    return {"employee_details": result}

# Endpoint to get the average unit price of orders shipped by a specific company in a given year
@app.get("/v1/retail_world/avg_unit_price_by_shipper_year", operation_id="get_avg_unit_price_by_shipper_year", summary="Get the average unit price of orders shipped by a specific company in a given year")
async def get_avg_unit_price_by_shipper_year(year: str = Query(..., description="Year in 'YYYY' format"), company_name: str = Query(..., description="Name of the shipping company")):
    cursor.execute("SELECT AVG(T2.UnitPrice) FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Shippers AS T3 ON T1.ShipVia = T3.ShipperID WHERE T1.OrderDate LIKE ? AND T3.CompanyName = ?", (year + '%', company_name))
    result = cursor.fetchone()
    if not result:
        return {"avg_unit_price": []}
    return {"avg_unit_price": result[0]}

# Endpoint to get the percentage of orders shipped by a specific company
@app.get("/v1/retail_world/percentage_orders_by_shipper", operation_id="get_percentage_orders_by_shipper", summary="Get the percentage of orders shipped by a specific company")
async def get_percentage_orders_by_shipper(company_name: str = Query(..., description="Name of the shipping company")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.CompanyName = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.ShipVia) FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the top 3 titles of courtesy based on salary
@app.get("/v1/retail_world/top_titles_of_courtesy_by_salary", operation_id="get_top_titles_of_courtesy_by_salary", summary="Get the top 3 titles of courtesy based on salary")
async def get_top_titles_of_courtesy_by_salary():
    cursor.execute("SELECT TitleOfCourtesy FROM Employees ORDER BY Salary LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"titles_of_courtesy": []}
    return {"titles_of_courtesy": result}

# Endpoint to get the last names of employees reporting to a specific title
@app.get("/v1/retail_world/employee_last_names_by_reporting_title", operation_id="get_employee_last_names_by_reporting_title", summary="Get the last names of employees reporting to a specific title")
async def get_employee_last_names_by_reporting_title(title: str = Query(..., description="Title of the employee to whom others report")):
    cursor.execute("SELECT LastName FROM Employees WHERE ReportsTo = ( SELECT EmployeeID FROM Employees WHERE Title = ? )", (title,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": result}

# Endpoint to get the top order detail by total price
@app.get("/v1/retail_world/top_order_detail_by_total_price", operation_id="get_top_order_detail_by_total_price", summary="Get the top order detail by total price")
async def get_top_order_detail_by_total_price():
    cursor.execute("SELECT UnitPrice * Quantity * (1 - Discount) AS THETOP FROM `Order Details` ORDER BY UnitPrice * Quantity * (1 - Discount) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"top_order_detail": []}
    return {"top_order_detail": result[0]}

# Endpoint to get the top 3 products by total units in stock and on order
@app.get("/v1/retail_world/top_products_by_total_units", operation_id="get_top_products_by_total_units", summary="Get the top 3 products by total units in stock and on order")
async def get_top_products_by_total_units():
    cursor.execute("SELECT ProductName FROM Products ORDER BY UnitsInStock + UnitsOnOrder DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the product ID with the lowest reorder level and highest unit price
@app.get("/v1/retail_world/product_id_by_reorder_level_and_unit_price", operation_id="get_product_id_by_reorder_level_and_unit_price", summary="Get the product ID with the lowest reorder level and highest unit price")
async def get_product_id_by_reorder_level_and_unit_price():
    cursor.execute("SELECT ProductID FROM Products ORDER BY ReorderLevel ASC, UnitPrice DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product_id": []}
    return {"product_id": result[0]}

# Endpoint to get the count of products in a specific category that are not discontinued
@app.get("/v1/retail_world/count_products_by_category", operation_id="get_count_products_by_category", summary="Get the count of products in a specific category that are not discontinued")
async def get_count_products_by_category(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT COUNT(T1.CategoryID) FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID WHERE T1.CategoryName = ? AND T2.Discontinued = 0", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the territory descriptions for employees with a specific title
@app.get("/v1/retail_world/territory_descriptions_by_employee_title", operation_id="get_territory_descriptions_by_employee_title", summary="Get the territory descriptions for employees with a specific title")
async def get_territory_descriptions_by_employee_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT T3.TerritoryDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T1.Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"territory_descriptions": []}
    return {"territory_descriptions": result}

# Endpoint to get company names of customers with orders exceeding a specified freight value
@app.get("/v1/retail_world/customer_company_names_by_freight", operation_id="get_customer_company_names_by_freight", summary="Get company names of customers with orders exceeding a specified freight value")
async def get_customer_company_names_by_freight(freight: int = Query(..., description="Freight value to filter orders")):
    cursor.execute("SELECT T1.CompanyName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.Freight > ?", (freight,))
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get the top shipper company name by ship city
@app.get("/v1/retail_world/top_shipper_by_ship_city", operation_id="get_top_shipper_by_ship_city", summary="Get the top shipper company name by ship city")
async def get_top_shipper_by_ship_city(ship_city: str = Query(..., description="Ship city to filter orders")):
    cursor.execute("SELECT T2.CompanyName FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T1.ShipCity = ? GROUP BY T2.CompanyName ORDER BY COUNT(T1.ShipVia) DESC LIMIT 1", (ship_city,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get distinct employee names by region description
@app.get("/v1/retail_world/employee_names_by_region", operation_id="get_employee_names_by_region", summary="Get distinct employee names by region description")
async def get_employee_names_by_region(region_description: str = Query(..., description="Region description to filter employees")):
    cursor.execute("SELECT DISTINCT T1.FirstName, T1.LastName FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID INNER JOIN Region AS T4 ON T3.RegionID = T4.RegionID WHERE T4.RegionDescription = ?", (region_description,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get product names by ship city
@app.get("/v1/retail_world/product_names_by_ship_city", operation_id="get_product_names_by_ship_city", summary="Get product names by ship city")
async def get_product_names_by_ship_city(ship_city: str = Query(..., description="Ship city to filter products")):
    cursor.execute("SELECT T3.ProductName FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T1.ShipCity = ?", (ship_city,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the top product name by postal code
@app.get("/v1/retail_world/top_product_by_postal_code", operation_id="get_top_product_by_postal_code", summary="Get the top product name by postal code")
async def get_top_product_by_postal_code(postal_code: int = Query(..., description="Postal code to filter products")):
    cursor.execute("SELECT T4.ProductName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID INNER JOIN Products AS T4 ON T3.ProductID = T4.ProductID WHERE T1.PostalCode = ? ORDER BY T3.Quantity LIMIT 1", (postal_code,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get distinct employee names by region description and reporting manager
@app.get("/v1/retail_world/employee_names_by_region_and_manager", operation_id="get_employee_names_by_region_and_manager", summary="Get distinct employee names by region description and reporting manager")
async def get_employee_names_by_region_and_manager(region_description: str = Query(..., description="Region description to filter employees"), manager_first_name: str = Query(..., description="First name of the reporting manager"), manager_last_name: str = Query(..., description="Last name of the reporting manager")):
    cursor.execute("SELECT DISTINCT T1.FirstName, T1.LastName FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID INNER JOIN Region AS T4 ON T3.RegionID = T4.RegionID WHERE T4.RegionDescription = ? AND T1.ReportsTo = (SELECT EmployeeID FROM Employees WHERE FirstName = ? AND LastName = ?)", (region_description, manager_first_name, manager_last_name))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get order dates by product name, quantity, and company name
@app.get("/v1/retail_world/order_dates_by_product_quantity_company", operation_id="get_order_dates_by_product_quantity_company", summary="Get order dates by product name, quantity, and company name")
async def get_order_dates_by_product_quantity_company(product_name: str = Query(..., description="Product name to filter orders"), quantity: int = Query(..., description="Quantity to filter orders"), company_name: str = Query(..., description="Company name to filter orders")):
    cursor.execute("SELECT T2.OrderDate FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID INNER JOIN Products AS T4 ON T3.ProductID = T4.ProductID WHERE T4.ProductName = ? AND T3.Quantity = ? AND T1.CompanyName = ?", (product_name, quantity, company_name))
    result = cursor.fetchall()
    if not result:
        return {"order_dates": []}
    return {"order_dates": [row[0] for row in result]}

# Endpoint to get category names by order ID
@app.get("/v1/retail_world/category_names_by_order_id", operation_id="get_category_names_by_order_id", summary="Get category names by order ID")
async def get_category_names_by_order_id(order_id: int = Query(..., description="Order ID to filter categories")):
    cursor.execute("SELECT T3.CategoryName FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Categories AS T3 ON T1.CategoryID = T3.CategoryID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the average quantity per order for a specific shipping date pattern and shipper company name
@app.get("/v1/retail_world/average_quantity_per_order", operation_id="get_average_quantity_per_order", summary="Get the average quantity per order for a specific shipping date pattern and shipper company name")
async def get_average_quantity_per_order(shipped_date_pattern: str = Query(..., description="Shipped date pattern in 'YYYY-MM%' format"), company_name: str = Query(..., description="Company name of the shipper")):
    cursor.execute("SELECT CAST(SUM(T2.Quantity) AS REAL) / COUNT(T2.OrderID) FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Shippers AS T3 ON T1.ShipVia = T3.ShipperID WHERE T1.ShippedDate LIKE ? AND T3.CompanyName = ?", (shipped_date_pattern, company_name))
    result = cursor.fetchone()
    if not result:
        return {"average_quantity": []}
    return {"average_quantity": result[0]}

# Endpoint to get the percentage of orders shipped in a specific month by a specific shipper
@app.get("/v1/retail_world/percentage_orders_shipped", operation_id="get_percentage_orders_shipped", summary="Get the percentage of orders shipped in a specific month by a specific shipper")
async def get_percentage_orders_shipped(shipped_date_pattern: str = Query(..., description="Shipped date pattern in 'YYYY-MM%' format"), company_name: str = Query(..., description="Company name of the shipper"), year_pattern: str = Query(..., description="Year pattern in 'YYYY%' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.ShippedDate LIKE ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.ShipVia) FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T2.CompanyName = ? AND T1.ShippedDate LIKE ?", (shipped_date_pattern, company_name, year_pattern))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of customers with a specific contact title in a specific country
@app.get("/v1/retail_world/count_customers_by_contact_title", operation_id="get_count_customers_by_contact_title", summary="Get the count of customers with a specific contact title in a specific country")
async def get_count_customers_by_contact_title(country: str = Query(..., description="Country of the customer"), contact_title: str = Query(..., description="Contact title of the customer")):
    cursor.execute("SELECT COUNT(ContactTitle) FROM Customers WHERE Country = ? AND ContactTitle = ?", (country, contact_title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the address details of a customer by contact name
@app.get("/v1/retail_world/customer_address_by_contact_name", operation_id="get_customer_address_by_contact_name", summary="Get the address details of a customer by contact name")
async def get_customer_address_by_contact_name(contact_name: str = Query(..., description="Contact name of the customer")):
    cursor.execute("SELECT Address, City, Region, PostalCode, Country FROM Customers WHERE ContactName = ?", (contact_name,))
    result = cursor.fetchone()
    if not result:
        return {"address_details": []}
    return {"address_details": result}

# Endpoint to get the company names of customers with a specific phone number pattern
@app.get("/v1/retail_world/company_names_by_phone_pattern", operation_id="get_company_names_by_phone_pattern", summary="Get the company names of customers with a specific phone number pattern")
async def get_company_names_by_phone_pattern(phone_pattern: str = Query(..., description="Phone number pattern in '(171)%' format")):
    cursor.execute("SELECT CompanyName FROM Customers WHERE Phone LIKE ?", (phone_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get the difference in the count of employees with a specific title between two countries
@app.get("/v1/retail_world/employee_title_count_difference", operation_id="get_employee_title_count_difference", summary="Get the difference in the count of employees with a specific title between two countries")
async def get_employee_title_count_difference(country1: str = Query(..., description="First country"), title: str = Query(..., description="Title of the employee"), country2: str = Query(..., description="Second country")):
    cursor.execute("SELECT ( SELECT COUNT(Title) FROM Employees WHERE Country = ? AND Title = ? ) - ( SELECT COUNT(Title) FROM Employees WHERE Country = ? AND Title = ? ) AS DIFFERENCE", (country1, title, country2, title))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of customers with a specific contact title in a specific city and country
@app.get("/v1/retail_world/count_customers_by_city_country_contact_title", operation_id="get_count_customers_by_city_country_contact_title", summary="Get the count of customers with a specific contact title in a specific city and country")
async def get_count_customers_by_city_country_contact_title(city: str = Query(..., description="City of the customer"), country: str = Query(..., description="Country of the customer"), contact_title: str = Query(..., description="Contact title of the customer")):
    cursor.execute("SELECT COUNT(CustomerID) FROM Customers WHERE City = ? AND Country = ? AND ContactTitle = ?", (city, country, contact_title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the last name of employees associated with a specific order and customer
@app.get("/v1/retail_world/employee_last_name_by_order_customer", operation_id="get_employee_last_name_by_order_customer", summary="Retrieves the last name of the employee who handled a specific order for a given customer. The operation requires the unique identifiers for the order and the customer as input parameters.")
async def get_employee_last_name_by_order_customer(order_id: int = Query(..., description="Order ID"), customer_id: str = Query(..., description="Customer ID")):
    cursor.execute("SELECT T1.LastName FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.OrderID = ? AND T2.CustomerID = ?", (order_id, customer_id))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get the freight cost for a specific order and company name
@app.get("/v1/retail_world/freight_cost_by_order_company", operation_id="get_freight_cost_by_order_company", summary="Get the freight cost for a specific order and company name")
async def get_freight_cost_by_order_company(order_id: int = Query(..., description="Order ID"), company_name: str = Query(..., description="Company name of the customer")):
    cursor.execute("SELECT T2.Freight FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.OrderID = ? AND T1.CompanyName = ?", (order_id, company_name))
    result = cursor.fetchone()
    if not result:
        return {"freight_cost": []}
    return {"freight_cost": result[0]}

# Endpoint to get order IDs based on shipper company name and limit
@app.get("/v1/retail_world/order_ids_by_shipper_company", operation_id="get_order_ids_by_shipper_company", summary="Get order IDs based on shipper company name and limit")
async def get_order_ids_by_shipper_company(company_name: str = Query(..., description="Company name of the shipper"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.OrderID FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T2.CompanyName = ? LIMIT ?", (company_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"order_ids": []}
    return {"order_ids": [row[0] for row in result]}

# Endpoint to get product names based on category name
@app.get("/v1/retail_world/product_names_by_category", operation_id="get_product_names_by_category", summary="Retrieves a list of product names that belong to the specified category. The category is identified by its name.")
async def get_product_names_by_category(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T2.ProductName FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID WHERE T1.CategoryName = ?", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get category descriptions based on product name
@app.get("/v1/retail_world/category_descriptions_by_product", operation_id="get_category_descriptions_by_product", summary="Retrieves the category descriptions associated with a specific product. The operation uses the provided product name to search for matching products and returns the corresponding category descriptions.")
async def get_category_descriptions_by_product(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT T1.Description FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.ProductName = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get product names based on supplier company name
@app.get("/v1/retail_world/product_names_by_supplier_company", operation_id="get_product_names_by_supplier_company", summary="Get product names based on supplier company name")
async def get_product_names_by_supplier_company(company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get unit prices based on supplier company name pattern and product name
@app.get("/v1/retail_world/unit_prices_by_supplier_company_pattern_and_product", operation_id="get_unit_prices_by_supplier_company_pattern_and_product", summary="Get unit prices based on supplier company name pattern and product name")
async def get_unit_prices_by_supplier_company_pattern_and_product(company_name_pattern: str = Query(..., description="Pattern for company name of the supplier"), product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT T1.UnitPrice FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName LIKE ? AND T1.ProductName = ?", (company_name_pattern, product_name))
    result = cursor.fetchall()
    if not result:
        return {"unit_prices": []}
    return {"unit_prices": [row[0] for row in result]}

# Endpoint to get discontinued product names based on category name and limit
@app.get("/v1/retail_world/discontinued_product_names_by_category", operation_id="get_discontinued_product_names_by_category", summary="Get discontinued product names based on category name and limit")
async def get_discontinued_product_names_by_category(discontinued: int = Query(..., description="Discontinued status (1 for discontinued)"), category_name: str = Query(..., description="Category name"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.ProductName FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.Discontinued = ? AND T1.CategoryName = ? LIMIT ?", (discontinued, category_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get product names based on supplier company name, ordered by reorder level and limited
@app.get("/v1/retail_world/product_names_by_supplier_company_ordered", operation_id="get_product_names_by_supplier_company_ordered", summary="Get product names based on supplier company name, ordered by reorder level and limited")
async def get_product_names_by_supplier_company_ordered(company_name: str = Query(..., description="Company name of the supplier"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ? ORDER BY T1.ReorderLevel DESC LIMIT ?", (company_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get supplier contact names based on company name
@app.get("/v1/retail_world/supplier_contact_names_by_company", operation_id="get_supplier_contact_names_by_company", summary="Get supplier contact names based on company name")
async def get_supplier_contact_names_by_company(company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT ContactName FROM Suppliers WHERE CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"contact_names": []}
    return {"contact_names": [row[0] for row in result]}

# Endpoint to get the country of a customer based on the company name
@app.get("/v1/retail_world/customer_country_by_company_name", operation_id="get_customer_country", summary="Get the country of a customer based on the company name")
async def get_customer_country(company_name: str = Query(..., description="Company name of the customer")):
    cursor.execute("SELECT Country FROM Customers WHERE CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of territories
@app.get("/v1/retail_world/count_territories", operation_id="get_count_territories", summary="Get the count of territories")
async def get_count_territories():
    cursor.execute("SELECT COUNT(TerritoryID) FROM Territories")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest total unit price from order details
@app.get("/v1/retail_world/highest_total_unit_price", operation_id="get_highest_total_unit_price", summary="Get the highest total unit price from order details")
async def get_highest_total_unit_price():
    cursor.execute("SELECT SUM(UnitPrice) FROM `Order Details` GROUP BY OrderID ORDER BY SUM(UnitPrice) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"total_unit_price": []}
    return {"total_unit_price": result[0]}

# Endpoint to get the count of orders shipped to a specific country
@app.get("/v1/retail_world/count_orders_by_ship_country", operation_id="get_count_orders_by_ship_country", summary="Get the count of orders shipped to a specific country")
async def get_count_orders_by_ship_country(ship_country: str = Query(..., description="Country to which the orders were shipped")):
    cursor.execute("SELECT COUNT(ShipCountry) FROM Orders WHERE ShipCountry = ?", (ship_country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the supplier's homepage based on the product name
@app.get("/v1/retail_world/supplier_homepage_by_product_name", operation_id="get_supplier_homepage_by_product_name", summary="Get the supplier's homepage based on the product name")
async def get_supplier_homepage_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.HomePage FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"homepage": []}
    return {"homepage": result[0]}

# Endpoint to get distinct first names of employees who shipped to a specific city
@app.get("/v1/retail_world/distinct_employee_first_names_by_ship_city", operation_id="get_distinct_employee_first_names_by_ship_city", summary="Get distinct first names of employees who shipped to a specific city")
async def get_distinct_employee_first_names_by_ship_city(ship_city: str = Query(..., description="City to which the orders were shipped")):
    cursor.execute("SELECT DISTINCT T1.FirstName FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.ShipCity = ?", (ship_city,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the highest quantity of a specific product ordered
@app.get("/v1/retail_world/highest_quantity_by_product_name", operation_id="get_highest_quantity_by_product_name", summary="Get the highest quantity of a specific product ordered")
async def get_highest_quantity_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.Quantity FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T2.ProductID = T1.ProductID WHERE T1.ProductName = ? ORDER BY T2.Quantity DESC LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"quantity": []}
    return {"quantity": result[0]}

# Endpoint to get the unit price of products in a specific category
@app.get("/v1/retail_world/product_unit_price_by_category", operation_id="get_product_unit_price_by_category", summary="Get the unit price of products in a specific category")
async def get_product_unit_price_by_category(description: str = Query(..., description="Description of the category")):
    cursor.execute("SELECT T2.UnitPrice FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID WHERE T1.Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"unit_prices": []}
    return {"unit_prices": [row[0] for row in result]}

# Endpoint to get the count of orders for a specific customer
@app.get("/v1/retail_world/order_count_by_customer", operation_id="get_order_count_by_customer", summary="Get the count of orders for a specific customer")
async def get_order_count_by_customer(company_name: str = Query(..., description="Company name of the customer")):
    cursor.execute("SELECT COUNT(T2.OrderID) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get product names shipped to a specific address
@app.get("/v1/retail_world/product_names_by_ship_address", operation_id="get_product_names_by_ship_address", summary="Get product names shipped to a specific address")
async def get_product_names_by_ship_address(ship_address: str = Query(..., description="Shipping address")):
    cursor.execute("SELECT T3.ProductName FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T1.ShipAddress = ? GROUP BY T3.ProductName", (ship_address,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get territory descriptions for a specific employee
@app.get("/v1/retail_world/territory_descriptions_by_employee", operation_id="get_territory_descriptions_by_employee", summary="Get territory descriptions for a specific employee")
async def get_territory_descriptions_by_employee(last_name: str = Query(..., description="Last name of the employee"), first_name: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT T3.TerritoryDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID WHERE T1.LastName = ? AND T1.FirstName = ?", (last_name, first_name))
    result = cursor.fetchall()
    if not result:
        return {"territory_descriptions": []}
    return {"territory_descriptions": [row[0] for row in result]}

# Endpoint to get contact names of customers who have orders shipped to a specific country
@app.get("/v1/retail_world/contact_names_by_ship_country", operation_id="get_contact_names_by_ship_country", summary="Get contact names of customers who have orders shipped to a specific country")
async def get_contact_names_by_ship_country(ship_country: str = Query(..., description="Shipping country")):
    cursor.execute("SELECT T1.ContactName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.ShipCountry = ? GROUP BY T1.ContactName", (ship_country,))
    result = cursor.fetchall()
    if not result:
        return {"contact_names": []}
    return {"contact_names": [row[0] for row in result]}

# Endpoint to get the percentage of orders handled by employees with a specific title
@app.get("/v1/retail_world/order_percentage_by_employee_title", operation_id="get_order_percentage_by_employee_title", summary="Get the percentage of orders handled by employees with a specific title")
async def get_order_percentage_by_employee_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.Title = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.OrderID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID", (title,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average salary of employees within a specific range of employee IDs
@app.get("/v1/retail_world/average_salary_by_employee_id_range", operation_id="get_average_salary_by_employee_id_range", summary="Get the average salary of employees within a specific range of employee IDs")
async def get_average_salary_by_employee_id_range(min_employee_id: int = Query(..., description="Minimum employee ID"), max_employee_id: int = Query(..., description="Maximum employee ID")):
    cursor.execute("SELECT AVG(Salary) FROM Employees WHERE EmployeeID BETWEEN ? AND ?", (min_employee_id, max_employee_id))
    result = cursor.fetchone()
    if not result:
        return {"average_salary": []}
    return {"average_salary": result[0]}

# Endpoint to get the total salary of employees in a specific country
@app.get("/v1/retail_world/total_salary_by_country", operation_id="get_total_salary_by_country", summary="Get the total salary of employees in a specific country")
async def get_total_salary_by_country(country: str = Query(..., description="Country of the employee")):
    cursor.execute("SELECT SUM(Salary) FROM Employees WHERE Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"total_salary": []}
    return {"total_salary": result[0]}

# Endpoint to check if an employee has a specific home phone number
@app.get("/v1/retail_world/check_employee_home_phone", operation_id="check_employee_home_phone", summary="Check if an employee with a given first name and last name has a specific home phone number")
async def check_employee_home_phone(home_phone: str = Query(..., description="Home phone number to check"), first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT CASE WHEN HomePhone = ? THEN 'YES' ELSE 'NO' END FROM Employees WHERE FirstName = ? AND LastName = ?", (home_phone, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"result": []}
    return {"result": result[0]}

# Endpoint to get notes of the employee with the highest salary
@app.get("/v1/retail_world/get_notes_highest_salary", operation_id="get_notes_highest_salary", summary="Get notes of the employee with the highest salary")
async def get_notes_highest_salary():
    cursor.execute("SELECT Notes FROM Employees WHERE Salary = ( SELECT MAX(Salary) FROM Employees )")
    result = cursor.fetchone()
    if not result:
        return {"notes": []}
    return {"notes": result[0]}

# Endpoint to get customer IDs for orders handled by a specific employee
@app.get("/v1/retail_world/get_customer_ids_by_employee", operation_id="get_customer_ids_by_employee", summary="Retrieves the IDs of customers who have placed orders handled by an employee with the specified first and last names. The operation uses the provided employee's first and last names to identify the relevant orders and extract the associated customer IDs.")
async def get_customer_ids_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T2.CustomerID FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"customer_ids": []}
    return {"customer_ids": [row[0] for row in result]}

# Endpoint to get distinct ship countries for orders handled by a specific employee
@app.get("/v1/retail_world/get_distinct_ship_countries_by_employee", operation_id="get_distinct_ship_countries_by_employee", summary="Get distinct ship countries for orders handled by an employee with a given first name and last name")
async def get_distinct_ship_countries_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT DISTINCT T2.ShipCountry FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"ship_countries": []}
    return {"ship_countries": [row[0] for row in result]}

# Endpoint to get the count of orders handled by a specific employee
@app.get("/v1/retail_world/get_order_count_by_employee", operation_id="get_order_count_by_employee", summary="Retrieves the total number of orders processed by an employee identified by their first and last names. The response provides a count of orders associated with the specified employee.")
async def get_order_count_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT COUNT(T2.EmployeeID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average salary of an employee based on orders handled
@app.get("/v1/retail_world/get_average_salary_by_employee", operation_id="get_average_salary_by_employee", summary="Get the average salary of an employee with a given first name and last name based on orders handled")
async def get_average_salary_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT CAST(SUM(T1.Salary) AS REAL) / COUNT(T2.EmployeeID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.FirstName = ? AND T1.LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"average_salary": []}
    return {"average_salary": result[0]}

# Endpoint to get quantity per unit of products supplied by a specific company
@app.get("/v1/retail_world/get_quantity_per_unit_by_supplier", operation_id="get_quantity_per_unit_by_supplier", summary="Get quantity per unit of products supplied by a company with a given company name")
async def get_quantity_per_unit_by_supplier(company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT T1.QuantityPerUnit FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"quantity_per_unit": []}
    return {"quantity_per_unit": [row[0] for row in result]}

# Endpoint to get the count of discontinued products supplied by a specific company
@app.get("/v1/retail_world/get_discontinued_product_count_by_supplier", operation_id="get_discontinued_product_count_by_supplier", summary="Get the count of discontinued products supplied by a company with a given company name")
async def get_discontinued_product_count_by_supplier(company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT COUNT(T1.Discontinued) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average unit price of products supplied by a specific company
@app.get("/v1/retail_world/get_average_unit_price_by_supplier", operation_id="get_average_unit_price_by_supplier", summary="Get the average unit price of products supplied by a company with a given company name")
async def get_average_unit_price_by_supplier(company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT SUM(T1.UnitPrice) / COUNT(T1.SupplierID) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_unit_price": []}
    return {"average_unit_price": result[0]}

# Endpoint to get the contact title of the supplier for a specific quantity per unit
@app.get("/v1/retail_world/supplier_contact_title_by_quantity", operation_id="get_supplier_contact_title_by_quantity", summary="Get the contact title of the supplier for a specific quantity per unit")
async def get_supplier_contact_title_by_quantity(quantity_per_unit: str = Query(..., description="Quantity per unit of the product")):
    cursor.execute("SELECT T2.ContactTitle FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.QuantityPerUnit = ?", (quantity_per_unit,))
    result = cursor.fetchone()
    if not result:
        return {"contact_title": []}
    return {"contact_title": result[0]}

# Endpoint to get the total units on order for a specific company
@app.get("/v1/retail_world/total_units_on_order_by_company", operation_id="get_total_units_on_order_by_company", summary="Get the total units on order for a specific company")
async def get_total_units_on_order_by_company(company_name: str = Query(..., description="Name of the company")):
    cursor.execute("SELECT SUM(T1.UnitsOnOrder) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_units_on_order": []}
    return {"total_units_on_order": result[0]}

# Endpoint to get the percentage of products supplied by a specific company
@app.get("/v1/retail_world/percentage_products_by_company", operation_id="get_percentage_products_by_company", summary="Get the percentage of products supplied by a specific company")
async def get_percentage_products_by_company(company_name: str = Query(..., description="Name of the company")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.CompanyName = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.SupplierID) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the top products by units in stock
@app.get("/v1/retail_world/top_products_by_stock", operation_id="get_top_products_by_stock", summary="Get the top products by units in stock")
async def get_top_products_by_stock(limit: int = Query(..., description="Number of top products to retrieve")):
    cursor.execute("SELECT ProductID FROM Products ORDER BY UnitsInStock DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": [row[0] for row in result]}

# Endpoint to get the count of discontinued products
@app.get("/v1/retail_world/discontinued_product_count", operation_id="get_discontinued_product_count", summary="Get the count of discontinued products")
async def get_discontinued_product_count(discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT COUNT(*) FROM Products WHERE Discontinued = ?", (discontinued,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get employee details by title
@app.get("/v1/retail_world/employee_details_by_title", operation_id="get_employee_details_by_title", summary="Get employee details by title")
async def get_employee_details_by_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT Address, HomePhone, Salary FROM Employees WHERE Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"employee_details": []}
    return {"employee_details": [{"address": row[0], "home_phone": row[1], "salary": row[2]} for row in result]}

# Endpoint to get employee names by title
@app.get("/v1/retail_world/employee_names_by_title", operation_id="get_employee_names_by_title", summary="Get employee names by title")
async def get_employee_names_by_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT FirstName, LastName FROM Employees WHERE Title = ?", (title,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the top companies by sales amount
@app.get("/v1/retail_world/top_companies_by_sales", operation_id="get_top_companies_by_sales", summary="Get the top companies by sales amount")
async def get_top_companies_by_sales(limit: int = Query(..., description="Number of top companies to retrieve")):
    cursor.execute("SELECT CompanyName FROM `Sales Totals by Amount` ORDER BY SaleAmount DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get the average product sales grouped by category name
@app.get("/v1/retail_world/average_product_sales_by_category", operation_id="get_average_product_sales_by_category", summary="Get the average product sales grouped by category name")
async def get_average_product_sales_by_category():
    cursor.execute("SELECT AVG(ProductSales) FROM `Sales by Category` GROUP BY CategoryName")
    result = cursor.fetchall()
    if not result:
        return {"average_sales": []}
    return {"average_sales": result}

# Endpoint to get the total quantity of a product with a specific name pattern
@app.get("/v1/retail_world/total_quantity_by_product_name", operation_id="get_total_quantity_by_product_name", summary="Get the total quantity of a product with a specific name pattern")
async def get_total_quantity_by_product_name(product_name_pattern: str = Query(..., description="Pattern to match the product name (use %% for wildcard)")):
    cursor.execute("SELECT SUM(T2.Quantity) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductName LIKE ?", (product_name_pattern,))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the count of products in a specific category with quantity greater than a specified value
@app.get("/v1/retail_world/count_products_by_category_and_quantity", operation_id="get_count_products_by_category_and_quantity", summary="Get the count of products in a specific category with quantity greater than a specified value")
async def get_count_products_by_category_and_quantity(category_name: str = Query(..., description="Name of the category"), min_quantity: int = Query(..., description="Minimum quantity")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Categories AS T3 ON T1.CategoryID = T3.CategoryID WHERE T3.CategoryName = ? AND T2.Quantity > ?", (category_name, min_quantity))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get product details from a specific supplier
@app.get("/v1/retail_world/product_details_by_supplier", operation_id="get_product_details_by_supplier", summary="Get product details from a specific supplier")
async def get_product_details_by_supplier(company_name: str = Query(..., description="Name of the supplier company")):
    cursor.execute("SELECT T1.ProductName, T1.ProductID, T1.ReorderLevel FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"product_details": []}
    return {"product_details": result}

# Endpoint to get distinct supplier company names for discontinued products
@app.get("/v1/retail_world/distinct_suppliers_for_discontinued_products", operation_id="get_distinct_suppliers_for_discontinued_products", summary="Get distinct supplier company names for discontinued products")
async def get_distinct_suppliers_for_discontinued_products(discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT DISTINCT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.Discontinued = ?", (discontinued,))
    result = cursor.fetchall()
    if not result:
        return {"supplier_names": []}
    return {"supplier_names": result}

# Endpoint to get employee names with employee IDs less than a specified value
@app.get("/v1/retail_world/employee_names_by_id", operation_id="get_employee_names_by_id", summary="Get employee names with employee IDs less than a specified value")
async def get_employee_names_by_id(max_employee_id: int = Query(..., description="Maximum employee ID")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.EmployeeID < ?", (max_employee_id,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": result}

# Endpoint to get the count of distinct employee first names in a specific region
@app.get("/v1/retail_world/count_distinct_employee_first_names_by_region", operation_id="get_count_distinct_employee_first_names_by_region", summary="Get the count of distinct employee first names in a specific region")
async def get_count_distinct_employee_first_names_by_region(region_description: str = Query(..., description="Description of the region")):
    cursor.execute("SELECT COUNT(DISTINCT T1.FirstName) FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID INNER JOIN Region AS T4 ON T3.RegionID = T4.RegionID WHERE T4.RegionDescription = ?", (region_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of orders shipped by a specific company within a date range
@app.get("/v1/retail_world/count_orders_by_shipper_and_date_range", operation_id="get_count_orders_by_shipper_and_date_range", summary="Get the count of orders shipped by a specific company within a date range")
async def get_count_orders_by_shipper_and_date_range(company_name: str = Query(..., description="Name of the shipping company"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD HH:MM:SS' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT COUNT(T1.OrderID) FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T2.CompanyName = ? AND T1.ShippedDate BETWEEN ? AND ?", (company_name, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct customer IDs for orders handled by a specific employee within a date range and shipped to a specific country
@app.get("/v1/retail_world/distinct_customer_ids_by_employee_and_date_range", operation_id="get_distinct_customer_ids_by_employee_and_date_range", summary="Get distinct customer IDs for orders handled by a specific employee within a date range and shipped to a specific country")
async def get_distinct_customer_ids_by_employee_and_date_range(last_name: str = Query(..., description="Last name of the employee"), first_name: str = Query(..., description="First name of the employee"), ship_country: str = Query(..., description="Country where the order was shipped"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD HH:MM:SS' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT DISTINCT T2.CustomerID FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.LastName = ? AND T1.FirstName = ? AND T2.ShipCountry = ? AND T2.ShippedDate BETWEEN ? AND ?", (last_name, first_name, ship_country, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"customer_ids": []}
    return {"customer_ids": result}

# Endpoint to get the reorder level of products with a specific order quantity
@app.get("/v1/retail_world/reorder_level_by_order_quantity", operation_id="get_reorder_level_by_order_quantity", summary="Get the reorder level of products with a specific order quantity")
async def get_reorder_level_by_order_quantity(quantity: int = Query(..., description="Order quantity")):
    cursor.execute("SELECT T1.ReorderLevel FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.Quantity = ?", (quantity,))
    result = cursor.fetchall()
    if not result:
        return {"reorder_levels": []}
    return {"reorder_levels": result}

# Endpoint to get the total value of products in stock
@app.get("/v1/retail_world/product_value_in_stock", operation_id="get_product_value_in_stock", summary="Get the total value of products in stock")
async def get_product_value_in_stock():
    cursor.execute("SELECT T1.UnitPrice * T1.UnitsInStock FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID")
    result = cursor.fetchall()
    if not result:
        return {"value": []}
    return {"value": result}

# Endpoint to get customers with orders shipped late in a specific country
@app.get("/v1/retail_world/late_shipped_orders_by_country", operation_id="get_late_shipped_orders_by_country", summary="Get customers with orders shipped late in a specific country")
async def get_late_shipped_orders_by_country(country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT T1.CompanyName, TIMESTAMPDIFF(DAY, T2.ShippedDate, T2.RequiredDate) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Country = ? AND TIMESTAMPDIFF(DAY, T2.ShippedDate, T2.RequiredDate) < 0", (country,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the customer with the highest order value
@app.get("/v1/retail_world/top_customer_by_order_value", operation_id="get_top_customer_by_order_value", summary="Get the customer with the highest order value")
async def get_top_customer_by_order_value():
    cursor.execute("SELECT T1.ContactName, T1.Phone FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID GROUP BY T2.OrderID, T1.ContactName, T1.Phone ORDER BY SUM(T3.UnitPrice * T3.Quantity * (1 - T3.Discount)) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": result}

# Endpoint to get the employee with the highest payment from order details
@app.get("/v1/retail_world/top_employee_by_payment", operation_id="get_top_employee_by_payment", summary="Get the employee with the highest payment from order details")
async def get_top_employee_by_payment():
    cursor.execute("SELECT T4.LastName, T4.FirstName, T4.ReportsTo , T1.Quantity * T1.UnitPrice * (1 - T1.Discount) AS payment FROM `Order Details` AS T1 INNER JOIN Orders AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Customers AS T3 ON T2.CustomerID = T3.CustomerID INNER JOIN Employees AS T4 ON T2.EmployeeID = T4.EmployeeID ORDER BY payment DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get the count of customers in a specific city and country
@app.get("/v1/retail_world/count_customers_by_city_and_country", operation_id="get_count_customers_by_city_and_country", summary="Retrieves the total number of customers residing in a specified city within a given country. The operation requires the country and city as input parameters to filter the customer data.")
async def get_count_customers_by_city_and_country(country: str = Query(..., description="Country of the customer"), city: str = Query(..., description="City of the customer")):
    cursor.execute("SELECT COUNT(City) FROM Customers WHERE Country = ? AND City = ?", (country, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of products supplied by a specific supplier
@app.get("/v1/retail_world/count_products_by_supplier", operation_id="get_count_products_by_supplier", summary="Get the count of products supplied by a specific supplier")
async def get_count_products_by_supplier(company_name: str = Query(..., description="Name of the supplier company")):
    cursor.execute("SELECT COUNT(T1.ProductName) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the product with the minimum unit price and its supplier
@app.get("/v1/retail_world/min_price_product_and_supplier", operation_id="get_min_price_product_and_supplier", summary="Get the product with the minimum unit price and its supplier")
async def get_min_price_product_and_supplier():
    cursor.execute("SELECT T2.CompanyName, T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.UnitPrice = ( SELECT MIN(UnitPrice) FROM Products )")
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result}

# Endpoint to get the top region by the number of territories
@app.get("/v1/retail_world/top_region_by_territories", operation_id="get_top_region_by_territories", summary="Get the top region by the number of territories")
async def get_top_region_by_territories(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.RegionID FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID GROUP BY T2.RegionID ORDER BY COUNT(T1.TerritoryID) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"region_id": []}
    return {"region_id": result[0]}

# Endpoint to get the region description for a given territory ID
@app.get("/v1/retail_world/region_description_by_territory_id", operation_id="get_region_description_by_territory_id", summary="Get the region description for a given territory ID")
async def get_region_description_by_territory_id(territory_id: int = Query(..., description="Territory ID")):
    cursor.execute("SELECT T2.RegionDescription FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T1.TerritoryID = ?", (territory_id,))
    result = cursor.fetchone()
    if not result:
        return {"region_description": []}
    return {"region_description": result[0]}

# Endpoint to get the percentage of customers from a specific city in a given year
@app.get("/v1/retail_world/percentage_customers_by_city_year", operation_id="get_percentage_customers_by_city_year", summary="Get the percentage of customers from a specific city in a given year")
async def get_percentage_customers_by_city_year(city: str = Query(..., description="City name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.City = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.City) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE STRFTIME('%Y', T2.OrderDate) = ?", (city, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the list of employees with their first name, last name, and title
@app.get("/v1/retail_world/employee_list", operation_id="get_employee_list", summary="Get the list of employees with their first name, last name, and title")
async def get_employee_list():
    cursor.execute("SELECT FirstName, LastName, Title FROM Employees")
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the count of employees with a specific title and salary greater than a given amount
@app.get("/v1/retail_world/employee_count_by_title_salary", operation_id="get_employee_count_by_title_salary", summary="Get the count of employees with a specific title and salary greater than a given amount")
async def get_employee_count_by_title_salary(salary: int = Query(..., description="Minimum salary"), title: str = Query(..., description="Employee title")):
    cursor.execute("SELECT COUNT(Title) FROM Employees WHERE Salary > ? AND Title = ?", (salary, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers from a specific country who placed orders in a given year
@app.get("/v1/retail_world/customer_count_by_country_year", operation_id="get_customer_count_by_country_year", summary="Retrieves the total number of customers from a specified country who have placed orders in a particular year. The operation filters customers based on the provided year and country parameters.")
async def get_customer_count_by_country_year(year: str = Query(..., description="Year in 'YYYY' format"), country: str = Query(..., description="Country")):
    cursor.execute("SELECT COUNT(T1.CustomerID) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE STRFTIME('%Y', T2.OrderDate) = ? AND T1.Country = ?", (year, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top company by the number of orders in a given year
@app.get("/v1/retail_world/top_company_by_orders_year", operation_id="get_top_company_by_orders_year", summary="Get the top company by the number of orders in a given year")
async def get_top_company_by_orders_year(year: str = Query(..., description="Year in 'YYYY' format"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.CompanyName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE STRFTIME('%Y', T2.OrderDate) = ? GROUP BY T1.CompanyName ORDER BY COUNT(T2.OrderID) DESC LIMIT ?", (year, limit))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the count of customers who placed orders in a given year, grouped by country
@app.get("/v1/retail_world/customer_count_by_year_grouped_by_country", operation_id="get_customer_count_by_year_grouped_by_country", summary="Retrieves the total number of customers who have placed orders in a specific year, categorized by their respective countries. The year is provided in the 'YYYY' format.")
async def get_customer_count_by_year_grouped_by_country(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.CustomerID) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE STRFTIME('%Y', T2.OrderDate) = ? GROUP BY T1.Country", (year,))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": result}

# Endpoint to get the count of orders placed by a specific company in a given year
@app.get("/v1/retail_world/order_count_by_company_year", operation_id="get_order_count_by_company_year", summary="Get the count of orders placed by a specific company in a given year")
async def get_order_count_by_company_year(year: str = Query(..., description="Year in 'YYYY' format"), company_name: str = Query(..., description="Company name")):
    cursor.execute("SELECT COUNT(T2.OrderID) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE STRFTIME('%Y', T2.OrderDate) = ? AND T1.CompanyName = ?", (year, company_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in days between order date and shipped date for a specific company, ordered by the difference
@app.get("/v1/retail_world/order_shipping_difference_by_company", operation_id="get_order_shipping_difference_by_company", summary="Get the difference in days between order date and shipped date for a specific company, ordered by the difference")
async def get_order_shipping_difference_by_company(company_name: str = Query(..., description="Company name"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT datediff(T2.ShippedDate, T2.OrderDate) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.CompanyName = ? ORDER BY datediff(T2.ShippedDate, T2.OrderDate) LIMIT ?", (company_name, limit))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the company name based on order ID
@app.get("/v1/retail_world/company_name_by_order_id", operation_id="get_company_name_by_order_id", summary="Get the company name based on the order ID")
async def get_company_name_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.CompanyName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the year with the most orders for a given company
@app.get("/v1/retail_world/most_orders_year_by_company", operation_id="get_most_orders_year_by_company", summary="Get the year with the most orders for a given company")
async def get_most_orders_year_by_company(company_name: str = Query(..., description="Company name")):
    cursor.execute("SELECT STRFTIME('%Y', T2.OrderDate) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.CompanyName = ? GROUP BY STRFTIME('%Y', T2.OrderDate) ORDER BY COUNT(T2.OrderID) DESC LIMIT 1", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the country with the most suppliers
@app.get("/v1/retail_world/top_supplier_country", operation_id="get_top_supplier_country", summary="Retrieves the country with the highest number of suppliers. The operation ranks countries based on the count of their respective suppliers and returns the top-ranked country.")
async def get_top_supplier_country():
    cursor.execute("SELECT Country FROM Suppliers GROUP BY Country ORDER BY COUNT(SupplierID) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get employees reporting to a specific title
@app.get("/v1/retail_world/employees_reporting_to_title", operation_id="get_employees_reporting_to_title", summary="Get employees reporting to a specific title")
async def get_employees_reporting_to_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT FirstName, LastName FROM Employees WHERE ReportsTo = ( SELECT EmployeeID FROM Employees WHERE Title = ? )", (title,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the customer with the most orders in a specific month and year
@app.get("/v1/retail_world/top_customer_by_month_year", operation_id="get_top_customer_by_month_year", summary="Retrieves the customer who placed the highest number of orders in a given month and year. The month and year should be provided in the 'YYYY-MM' format.")
async def get_top_customer_by_month_year(month_year: str = Query(..., description="Month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(OrderID) FROM Orders WHERE OrderDate LIKE ? GROUP BY CustomerID ORDER BY COUNT(OrderID) DESC LIMIT 1", (month_year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get the salary of the earliest hired employee
@app.get("/v1/retail_world/salary_of_earliest_hired_employee", operation_id="get_salary_of_earliest_hired_employee", summary="Get the salary of the earliest hired employee")
async def get_salary_of_earliest_hired_employee():
    cursor.execute("SELECT Salary FROM Employees WHERE HireDate = ( SELECT MIN(HireDate) FROM Employees )")
    result = cursor.fetchone()
    if not result:
        return {"salary": []}
    return {"salary": result[0]}

# Endpoint to get the maximum age difference between birth date and hire date
@app.get("/v1/retail_world/max_age_difference_birth_hire", operation_id="get_max_age_difference_birth_hire", summary="Get the maximum age difference between birth date and hire date")
async def get_max_age_difference_birth_hire():
    cursor.execute("SELECT MAX(TIMESTAMPDIFF(YEAR, BirthDate, HireDate)) FROM Employees")
    result = cursor.fetchone()
    if not result:
        return {"age_difference": []}
    return {"age_difference": result[0]}

# Endpoint to get the total revenue from discontinued products
@app.get("/v1/retail_world/total_revenue_discontinued_products", operation_id="get_total_revenue_discontinued_products", summary="Get the total revenue from discontinued products")
async def get_total_revenue_discontinued_products(discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT SUM(T2.UnitPrice * T2.Quantity) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T1.Discontinued = ?", (discontinued,))
    result = cursor.fetchone()
    if not result:
        return {"total_revenue": []}
    return {"total_revenue": result[0]}

# Endpoint to get the category with the most discontinued products
@app.get("/v1/retail_world/top_category_discontinued_products", operation_id="get_top_category_discontinued_products", summary="Get the category with the most discontinued products")
async def get_top_category_discontinued_products(discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT T2.CategoryName FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T1.Discontinued = ? GROUP BY T2.CategoryName ORDER BY COUNT(T1.ProductID) DESC LIMIT 1", (discontinued,))
    result = cursor.fetchone()
    if not result:
        return {"category_name": []}
    return {"category_name": result[0]}

# Endpoint to get the count of products in a specific category and year
@app.get("/v1/retail_world/product_count_category_year", operation_id="get_product_count_category_year", summary="Get the count of products in a specific category and year")
async def get_product_count_category_year(category_name: str = Query(..., description="Category name"), category_id: int = Query(..., description="Category ID"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.ProductID) FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID INNER JOIN `Order Details` AS T3 ON T2.ProductID = T3.ProductID INNER JOIN Orders AS T4 ON T3.OrderID = T4.OrderID WHERE T1.CategoryName = ? AND T1.CategoryID = ? AND T4.OrderDate LIKE ?", (category_name, category_id, year + '%'))
    result = cursor.fetchone()
    if not result:
        return {"product_count": []}
    return {"product_count": result[0]}

# Endpoint to get the top company by the number of products ordered
@app.get("/v1/retail_world/top_company_by_products_ordered", operation_id="get_top_company_by_products_ordered", summary="Get the top company by the number of products ordered")
async def get_top_company_by_products_ordered(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.CompanyName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID GROUP BY T1.CompanyName ORDER BY COUNT(T3.ProductID) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the average ship via for a specific company and year
@app.get("/v1/retail_world/average_ship_via_by_company_and_year", operation_id="get_average_ship_via", summary="Get the average ship via for a specific company and year")
async def get_average_ship_via(divisor: int = Query(..., description="Divisor for the average calculation"), company_name: str = Query(..., description="Company name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(T1.ShipVia) AS REAL) / ? FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T2.CompanyName = ? AND T1.ShippedDate LIKE ?", (divisor, company_name, year + '%'))
    result = cursor.fetchone()
    if not result:
        return {"average_ship_via": []}
    return {"average_ship_via": result[0]}

# Endpoint to get product names from suppliers with a specific company name pattern
@app.get("/v1/retail_world/product_names_by_supplier_company_name", operation_id="get_product_names_by_supplier", summary="Get product names from suppliers with a specific company name pattern")
async def get_product_names_by_supplier(company_name_pattern: str = Query(..., description="Company name pattern")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName LIKE ?", (company_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the top territory by the number of regions
@app.get("/v1/retail_world/top_territory_by_region_count", operation_id="get_top_territory_by_region_count", summary="Get the top territory by the number of regions")
async def get_top_territory_by_region_count(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT COUNT(T2.RegionDescription), T1.TerritoryDescription, COUNT(*) AS num FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID GROUP BY T1.TerritoryDescription ORDER BY num DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"territory_description": []}
    return {"territory_description": result[1]}

# Endpoint to get the count of employees with specific titles and a minimum number of territories
@app.get("/v1/retail_world/employee_count_by_title_and_territory", operation_id="get_employee_count_by_title_and_territory", summary="Get the count of employees with specific titles and a minimum number of territories")
async def get_employee_count_by_title_and_territory(title1: str = Query(..., description="First title of courtesy"), title2: str = Query(..., description="Second title of courtesy"), min_territories: int = Query(..., description="Minimum number of territories")):
    cursor.execute("SELECT COUNT(EID) FROM ( SELECT T1.EmployeeID AS EID FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.TitleOfCourtesy IN (?, ?) GROUP BY T1.EmployeeID HAVING COUNT(T2.TerritoryID) >= ? ) T1", (title1, title2, min_territories))
    result = cursor.fetchone()
    if not result:
        return {"employee_count": []}
    return {"employee_count": result[0]}

# Endpoint to get supplier company names ordered by reorder level
@app.get("/v1/retail_world/supplier_company_names_by_reorder_level", operation_id="get_supplier_company_names_by_reorder_level", summary="Get supplier company names ordered by reorder level")
async def get_supplier_company_names_by_reorder_level(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID ORDER BY T1.ReorderLevel DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get the top company by total order value with a specific discount
@app.get("/v1/retail_world/top_company_by_order_value_and_discount", operation_id="get_top_company_by_order_value_and_discount", summary="Get the top company by total order value with a specific discount")
async def get_top_company_by_order_value_and_discount(discount: float = Query(..., description="Discount value"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.CompanyName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID WHERE T3.Discount = ? GROUP BY T1.CompanyName ORDER BY SUM(T3.UnitPrice * T3.Quantity) DESC LIMIT ?", (discount, limit))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the total order value for a specific employee and order date pattern
@app.get("/v1/retail_world/total_order_value_by_employee_and_date", operation_id="get_total_order_value_by_employee_and_date", summary="Get the total order value for a specific employee and order date pattern")
async def get_total_order_value_by_employee_and_date(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), order_date_pattern: str = Query(..., description="Order date pattern in 'YYYY-MM' format"), discount: float = Query(..., description="Discount value")):
    cursor.execute("SELECT SUM(T3.UnitPrice * T3.Quantity) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID WHERE T1.FirstName = ? AND T1.LastName = ? AND T2.OrderDate LIKE ? AND T3.Discount = ?", (first_name, last_name, order_date_pattern + '%', discount))
    result = cursor.fetchone()
    if not result:
        return {"total_order_value": []}
    return {"total_order_value": result[0]}

# Endpoint to get the total order value for a specific year
@app.get("/v1/retail_world/total_order_value_by_year", operation_id="get_total_order_value_by_year", summary="Get the total order value for a specific year")
async def get_total_order_value_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID WHERE T1.OrderDate LIKE ?", (year + '%',))
    result = cursor.fetchone()
    if not result:
        return {"total_order_value": []}
    return {"total_order_value": result[0]}

# Endpoint to get the average order value for a specific date range
@app.get("/v1/retail_world/average_order_value_by_date_range", operation_id="get_average_order_value_by_date_range", summary="Get the average order value for a specific date range")
async def get_average_order_value_by_date_range(divisor: int = Query(..., description="Divisor for the average calculation"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD HH:MM:SS' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) / ? FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID WHERE T1.ShippedDate BETWEEN ? AND ?", (divisor, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_order_value": []}
    return {"average_order_value": result[0]}

# Endpoint to get the count of orders shipped to a specific country in a specific year
@app.get("/v1/retail_world/order_count_by_country_year", operation_id="get_order_count", summary="Get the count of orders shipped to a specific country in a specific year")
async def get_order_count(ship_country: str = Query(..., description="Country to which the order was shipped"), ship_year: str = Query(..., description="Year in which the order was shipped in 'YYYY' format")):
    cursor.execute("SELECT COUNT(OrderID) FROM Orders WHERE ShipCountry = ? AND STRFTIME('%Y', ShippedDate) = ?", (ship_country, ship_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get category details based on category name
@app.get("/v1/retail_world/category_details", operation_id="get_category_details", summary="Retrieves detailed information about a specific category, including its unique identifier and description. The category is identified by its name, which is provided as an input parameter.")
async def get_category_details(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT CategoryID, Description FROM Categories WHERE CategoryName = ?", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get order details where the total price exceeds a specified amount
@app.get("/v1/retail_world/order_details_by_total_price", operation_id="get_order_details", summary="Get order details where the total price exceeds a specified amount")
async def get_order_details(total_price: float = Query(..., description="Total price threshold")):
    cursor.execute("SELECT ProductID, OrderID, UnitPrice FROM `Order Details` WHERE UnitPrice * Quantity * (1 - Discount) > ?", (total_price,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get territory IDs for a specific employee
@app.get("/v1/retail_world/employee_territories", operation_id="get_employee_territories", summary="Get territory IDs for a specific employee")
async def get_employee_territories(employee_id: int = Query(..., description="Employee ID")):
    cursor.execute("SELECT TerritoryID FROM EmployeeTerritories WHERE EmployeeID = ?", (employee_id,))
    result = cursor.fetchall()
    if not result:
        return {"territories": []}
    return {"territories": result}

# Endpoint to get supplier details based on city
@app.get("/v1/retail_world/supplier_details_by_city", operation_id="get_supplier_details", summary="Get supplier details based on city")
async def get_supplier_details(city: str = Query(..., description="City of the supplier")):
    cursor.execute("SELECT CompanyName, HomePage FROM Suppliers WHERE City = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get supplier names based on country
@app.get("/v1/retail_world/supplier_names_by_country", operation_id="get_supplier_names", summary="Get supplier names based on country")
async def get_supplier_names(country: str = Query(..., description="Country of the supplier")):
    cursor.execute("SELECT CompanyName FROM Suppliers WHERE Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get employee details based on city
@app.get("/v1/retail_world/employee_details_by_city", operation_id="get_employee_details_city", summary="Get employee details based on city")
async def get_employee_details_city(city: str = Query(..., description="City of the employee")):
    cursor.execute("SELECT TitleOfCourtesy, FirstName, LastName, TIMESTAMPDIFF(YEAR, BirthDate, NOW()) AS ages FROM Employees WHERE City = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get employee details based on the title of their manager
@app.get("/v1/retail_world/employee_details_by_manager_title", operation_id="get_employee_details_manager", summary="Get employee details based on the title of their manager")
async def get_employee_details_manager(manager_title: str = Query(..., description="Title of the manager")):
    cursor.execute("SELECT FirstName, LastName, Title FROM Employees WHERE ReportsTo = (SELECT EmployeeID FROM Employees WHERE Title = ?)", (manager_title,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get company names and cities of customers from a specific country
@app.get("/v1/retail_world/customers_by_country", operation_id="get_customers_by_country", summary="Get company names and cities of customers from a specific country")
async def get_customers_by_country(country: str = Query(..., description="Country of the customers")):
    cursor.execute("SELECT CompanyName, City FROM Customers WHERE Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the total stock and product names based on quantity per unit
@app.get("/v1/retail_world/product_stock_by_quantity_per_unit", operation_id="get_product_stock_by_quantity_per_unit", summary="Get the total stock and product names based on quantity per unit")
async def get_product_stock_by_quantity_per_unit(quantity_per_unit: str = Query(..., description="Quantity per unit of the product")):
    cursor.execute("SELECT UnitsInStock + UnitsOnOrder, ProductName FROM Products WHERE QuantityPerUnit = ?", (quantity_per_unit,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the product name and category name of the product with the highest reorder level
@app.get("/v1/retail_world/top_reorder_product", operation_id="get_top_reorder_product", summary="Get the product name and category name of the product with the highest reorder level")
async def get_top_reorder_product():
    cursor.execute("SELECT T2.ProductName, T1.CategoryName FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID ORDER BY T2.ReorderLevel DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"product": []}
    return {"product": result}

# Endpoint to get supplier details and product names based on total stock
@app.get("/v1/retail_world/suppliers_by_total_stock", operation_id="get_suppliers_by_total_stock", summary="Get supplier details and product names based on total stock")
async def get_suppliers_by_total_stock(total_stock: int = Query(..., description="Total stock of the product")):
    cursor.execute("SELECT T2.CompanyName, T2.City, T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.UnitsInStock + UnitsOnOrder > ?", (total_stock,))
    result = cursor.fetchall()
    if not result:
        return {"suppliers": []}
    return {"suppliers": result}

# Endpoint to get contact details and product names based on supplier company name
@app.get("/v1/retail_world/contacts_by_supplier_company", operation_id="get_contacts_by_supplier_company", summary="Get contact details and product names based on supplier company name")
async def get_contacts_by_supplier_company(company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT T2.ContactName, T2.ContactTitle, T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"contacts": []}
    return {"contacts": result}

# Endpoint to get territory and region details based on employee title, last name, and first name
@app.get("/v1/retail_world/territory_region_by_employee", operation_id="get_territory_region_by_employee", summary="Get territory and region details based on employee title, last name, and first name")
async def get_territory_region_by_employee(title_of_courtesy: str = Query(..., description="Title of courtesy of the employee"), last_name: str = Query(..., description="Last name of the employee"), first_name: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT T3.TerritoryID, T3.TerritoryDescription, T4.RegionDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID INNER JOIN Region AS T4 ON T3.RegionID = T4.RegionID WHERE T1.TitleOfCourtesy = ? AND T1.LastName = ? AND T1.FirstName = ?", (title_of_courtesy, last_name, first_name))
    result = cursor.fetchall()
    if not result:
        return {"territories": []}
    return {"territories": result}

# Endpoint to get product names based on required date and customer ID
@app.get("/v1/retail_world/products_by_required_date_customer", operation_id="get_products_by_required_date_customer", summary="Get product names based on required date and customer ID")
async def get_products_by_required_date_customer(required_date: str = Query(..., description="Required date in 'YYYY-MM-DD%' format"), customer_id: str = Query(..., description="Customer ID")):
    cursor.execute("SELECT T3.ProductName FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T1.RequiredDate LIKE ? AND T1.CustomerID = ?", (required_date, customer_id))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get product names and category names based on the maximum reorder level
@app.get("/v1/retail_world/products_by_max_reorder_level", operation_id="get_products_by_max_reorder_level", summary="Get product names and category names based on the maximum reorder level")
async def get_products_by_max_reorder_level():
    cursor.execute("SELECT T2.ProductName, T1.CategoryName FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.ReorderLevel = ( SELECT MAX(ReorderLevel) FROM Products )")
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the count of orders based on customer country
@app.get("/v1/retail_world/order_count_by_customer_country", operation_id="get_order_count_by_customer_country", summary="Retrieves the total number of orders placed by customers from a specific country. The operation uses the provided country parameter to filter the customer data and calculate the order count.")
async def get_order_count_by_customer_country(country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT COUNT(T2.OrderID) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get product names for customers in a specific country and order year
@app.get("/v1/retail_world/product_names_by_country_and_year", operation_id="get_product_names", summary="Get product names for customers in a specific country and order year")
async def get_product_names(country: str = Query(..., description="Country of the customer"), order_year: str = Query(..., description="Year of the order in 'YYYY' format")):
    cursor.execute("SELECT T4.ProductName FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID INNER JOIN Products AS T4 ON T3.ProductID = T4.ProductID WHERE T1.Country = ? AND STRFTIME('%Y', T2.OrderDate) = ?", (country, order_year))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get supplier company names based on employee details and order criteria
@app.get("/v1/retail_world/supplier_company_names_by_employee_and_order", operation_id="get_supplier_company_names", summary="Get supplier company names based on employee details and order criteria")
async def get_supplier_company_names(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee"), ship_country: str = Query(..., description="Ship country of the order"), order_date_pattern: str = Query(..., description="Order date pattern in 'YYYY-MM%' format")):
    cursor.execute("SELECT T5.CompanyName FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID INNER JOIN Products AS T4 ON T3.ProductID = T4.ProductID INNER JOIN Suppliers AS T5 ON T4.SupplierID = T5.SupplierID WHERE T1.FirstName = ? AND T1.LastName = ? AND T2.ShipCountry = ? AND T2.OrderDate LIKE ?", (first_name, last_name, ship_country, order_date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get employee details and order count for a specific ship country
@app.get("/v1/retail_world/employee_details_and_order_count_by_ship_country", operation_id="get_employee_details_and_order_count", summary="Get employee details and order count for a specific ship country")
async def get_employee_details_and_order_count(ship_country: str = Query(..., description="Ship country of the order")):
    cursor.execute("SELECT T1.FirstName, T1.LastName, T1.Title, T1.Salary , COUNT(T2.OrderID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE ShipCountry = ? GROUP BY T1.FirstName, T1.LastName, T1.Title, T1.Salary, T1.BirthDate ORDER BY T1.BirthDate LIMIT 1", (ship_country,))
    result = cursor.fetchone()
    if not result:
        return {"employee_details": []}
    return {"employee_details": {"first_name": result[0], "last_name": result[1], "title": result[2], "salary": result[3], "order_count": result[4]}}

# Endpoint to get territory details based on region description
@app.get("/v1/retail_world/territory_details_by_region_description", operation_id="get_territory_details", summary="Get territory details based on region description")
async def get_territory_details(region_description: str = Query(..., description="Description of the region")):
    cursor.execute("SELECT T1.TerritoryID, T1.TerritoryDescription FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ?", (region_description,))
    result = cursor.fetchall()
    if not result:
        return {"territory_details": []}
    return {"territory_details": [{"territory_id": row[0], "territory_description": row[1]} for row in result]}

# Endpoint to get the average payment per product in a specific category
@app.get("/v1/retail_world/average_payment_per_product_by_category", operation_id="get_average_payment_per_product", summary="Get the average payment per product in a specific category")
async def get_average_payment_per_product(category_name: str = Query(..., description="Name of the category")):
    cursor.execute("SELECT SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) / COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Categories AS T3 ON T1.CategoryID = T3.CategoryID WHERE T3.CategoryName = ?", (category_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_payment": []}
    return {"average_payment": result[0]}

# Endpoint to get the total payment for customers in a specific city
@app.get("/v1/retail_world/total_payment_by_city", operation_id="get_total_payment_by_city", summary="Get the total payment for customers in a specific city")
async def get_total_payment_by_city(city: str = Query(..., description="City of the customer")):
    cursor.execute("SELECT SUM(T3.UnitPrice * T3.Quantity * (1 - T3.Discount)) AS TOTALPAYMENT FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID WHERE T1.City = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"total_payment": []}
    return {"total_payment": result[0]}

# Endpoint to get the total units in stock and on order for suppliers in a specific country
@app.get("/v1/retail_world/total_units_by_supplier_country", operation_id="get_total_units_by_supplier_country", summary="Get the total units in stock and on order for suppliers in a specific country")
async def get_total_units_by_supplier_country(country: str = Query(..., description="Country of the supplier")):
    cursor.execute("SELECT SUM(T1.UnitsInStock + T1.UnitsOnOrder) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.Country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"total_units": []}
    return {"total_units": result[0]}

# Endpoint to get product and category names for suppliers in a specific country and discontinued status
@app.get("/v1/retail_world/product_and_category_names_by_supplier_country_and_discontinued", operation_id="get_product_and_category_names", summary="Get product and category names for suppliers in a specific country and discontinued status")
async def get_product_and_category_names(country: str = Query(..., description="Country of the supplier"), discontinued: int = Query(..., description="Discontinued status of the product (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT T2.ProductName, T3.CategoryName FROM Suppliers AS T1 INNER JOIN Products AS T2 ON T1.SupplierID = T2.SupplierID INNER JOIN Categories AS T3 ON T2.CategoryID = T3.CategoryID WHERE T1.Country = ? AND T2.Discontinued = ?", (country, discontinued))
    result = cursor.fetchall()
    if not result:
        return {"product_and_category_names": []}
    return {"product_and_category_names": [{"product_name": row[0], "category_name": row[1]} for row in result]}

# Endpoint to get the country and order ID for the highest payment order of a specific product
@app.get("/v1/retail_world/highest_payment_order_by_product", operation_id="get_highest_payment_order_by_product", summary="Get the country and order ID for the highest payment order of a specific product")
async def get_highest_payment_order_by_product(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T3.Country, T1.OrderID FROM `Order Details` AS T1 INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID INNER JOIN Suppliers AS T3 ON T2.SupplierID = T3.SupplierID WHERE T2.ProductName = ? ORDER BY T1.UnitPrice * T1.Quantity * (1 - T1.Discount) DESC LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"order_details": []}
    return {"order_details": {"country": result[0], "order_id": result[1]}}

# Endpoint to get the total payment for each product in a specific order
@app.get("/v1/retail_world/total_payment_per_product_by_order", operation_id="get_total_payment_per_product_by_order", summary="Get the total payment for each product in a specific order")
async def get_total_payment_per_product_by_order(order_id: int = Query(..., description="ID of the order")):
    cursor.execute("SELECT T1.ProductName , SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T2.OrderID = ? GROUP BY T1.ProductName", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"product_payments": []}
    return {"product_payments": [{"product_name": row[0], "total_payment": row[1]} for row in result]}

# Endpoint to get distinct contact names and titles of suppliers for a specific category and supplier ID range
@app.get("/v1/retail_world/supplier_contacts_by_category_and_id_range", operation_id="get_supplier_contacts", summary="Get distinct contact names and titles of suppliers for a specific category and supplier ID range")
async def get_supplier_contacts(category_name: str = Query(..., description="Category name"), min_supplier_id: int = Query(..., description="Minimum supplier ID"), max_supplier_id: int = Query(..., description="Maximum supplier ID")):
    cursor.execute("SELECT DISTINCT T1.ContactName, T1.ContactTitle FROM Suppliers AS T1 INNER JOIN Products AS T2 ON T1.SupplierID = T2.SupplierID INNER JOIN Categories AS T3 ON T2.CategoryID = T3.CategoryID WHERE T3.CategoryName = ? AND T1.SupplierID BETWEEN ? AND ? LIMIT 1", (category_name, min_supplier_id, max_supplier_id))
    result = cursor.fetchall()
    if not result:
        return {"contacts": []}
    return {"contacts": result}

# Endpoint to get distinct product names where the shipped date is before the required date
@app.get("/v1/retail_world/products_shipped_before_required_date", operation_id="get_products_shipped_before_required_date", summary="Get distinct product names where the shipped date is before the required date")
async def get_products_shipped_before_required_date():
    cursor.execute("SELECT DISTINCT T3.ProductName FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE DATEDIFF(T1.ShippedDate, T1.RequiredDate) < 0")
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get product names for orders handled by employees with a specific title, shipped date, and country
@app.get("/v1/retail_world/products_by_employee_title_shipped_date_country", operation_id="get_products_by_employee_title_shipped_date_country", summary="Get product names for orders handled by employees with a specific title, shipped date, and country")
async def get_products_by_employee_title_shipped_date_country(title: str = Query(..., description="Employee title"), shipped_date: str = Query(..., description="Shipped date in 'YYYY%' format"), ship_country: str = Query(..., description="Ship country")):
    cursor.execute("SELECT T4.ProductName FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID INNER JOIN Products AS T4 ON T3.ProductID = T4.ProductID WHERE T1.Title = ? AND T2.ShippedDate LIKE ? AND T2.ShipCountry = ?", (title, shipped_date, ship_country))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get product names in a specific category with the maximum reorder level
@app.get("/v1/retail_world/products_by_category_max_reorder_level", operation_id="get_products_by_category_max_reorder_level", summary="Get product names in a specific category with the maximum reorder level")
async def get_products_by_category_max_reorder_level(category_name: str = Query(..., description="Category name")):
    cursor.execute("SELECT T2.ProductName FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID WHERE T1.CategoryName = ? AND T2.ReorderLevel = ( SELECT MAX(ReorderLevel) FROM Products )", (category_name,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the total units in stock and on order for a specific product
@app.get("/v1/retail_world/total_units_by_product_name", operation_id="get_total_units_by_product_name", summary="Get the total units in stock and on order for a specific product")
async def get_total_units_by_product_name(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT SUM(UnitsInStock + UnitsOnOrder) FROM Products WHERE ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_units": []}
    return {"total_units": result[0]}

# Endpoint to get product names and category names for products with the minimum and maximum unit prices
@app.get("/v1/retail_world/products_by_min_max_unit_price", operation_id="get_products_by_min_max_unit_price", summary="Get product names and category names for products with the minimum and maximum unit prices")
async def get_products_by_min_max_unit_price():
    cursor.execute("SELECT T2.ProductName, T1.CategoryName FROM Categories AS T1 INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.UnitPrice IN (( SELECT MIN(UnitPrice) FROM Products ), ( SELECT MAX(UnitPrice) FROM Products ))")
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the count of employees in a specific region
@app.get("/v1/retail_world/employee_count_by_region", operation_id="get_employee_count_by_region", summary="Get the count of employees in a specific region")
async def get_employee_count_by_region(region_description: str = Query(..., description="Region description")):
    cursor.execute("SELECT COUNT(T2.EmployeeID) FROM Territories AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.TerritoryID = T2.TerritoryID INNER JOIN Region AS T3 ON T1.RegionID = T3.RegionID WHERE T3.RegionDescription = ?", (region_description,))
    result = cursor.fetchone()
    if not result:
        return {"employee_count": []}
    return {"employee_count": result[0]}

# Endpoint to get the average order value
@app.get("/v1/retail_world/average_order_value", operation_id="get_average_order_value", summary="Get the average order value")
async def get_average_order_value():
    cursor.execute("SELECT SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(OrderID) FROM `Order Details`")
    result = cursor.fetchone()
    if not result:
        return {"average_order_value": []}
    return {"average_order_value": result[0]}

# Endpoint to get the percentage of discontinued products
@app.get("/v1/retail_world/percentage_discontinued_products", operation_id="get_percentage_discontinued_products", summary="Get the percentage of discontinued products")
async def get_percentage_discontinued_products(discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN Discontinued = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(ProductID) FROM Products", (discontinued,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get employee details based on order ID
@app.get("/v1/retail_world/employee_details_by_order_id", operation_id="get_employee_details_by_order_id", summary="Retrieves the first and last name of the employee associated with a given order ID. This operation uses the provided order ID to look up the corresponding employee record in the database.")
async def get_employee_details_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.FirstName, T1.LastName FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get product details based on employee and company name
@app.get("/v1/retail_world/product_details_by_employee_and_company", operation_id="get_product_details_by_employee_and_company", summary="Get the product name and quantity based on employee first name, last name, and company name")
async def get_product_details_by_employee_and_company(first_name: str = Query(..., description="Employee first name"), last_name: str = Query(..., description="Employee last name"), company_name: str = Query(..., description="Company name")):
    cursor.execute("SELECT T4.ProductName, T3.Quantity FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID INNER JOIN Products AS T4 ON T3.ProductID = T4.ProductID INNER JOIN Customers AS T5 ON T2.CustomerID = T5.CustomerID WHERE T1.FirstName = ? AND T1.LastName = ? AND T5.CompanyName = ?", (first_name, last_name, company_name))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get order statistics for a specific company
@app.get("/v1/retail_world/order_statistics_by_company", operation_id="get_order_statistics_by_company", summary="Get the count of orders and average order value for a specific company")
async def get_order_statistics_by_company(company_name: str = Query(..., description="Company name")):
    cursor.execute("SELECT COUNT(T2.OrderID), SUM(T3.UnitPrice * T3.Quantity * (1 - T3.Discount)) / COUNT(T2.OrderID) FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID WHERE T1.CompanyName = ?", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"statistics": []}
    return {"statistics": {"order_count": result[0], "average_order_value": result[1]}}

# Endpoint to get the count of quantities for a specific product
@app.get("/v1/retail_world/quantity_count_by_product", operation_id="get_quantity_count_by_product", summary="Get the count of quantities for a specific product")
async def get_quantity_count_by_product(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT COUNT(T2.Quantity) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the minimum and maximum salary for a specific job title
@app.get("/v1/retail_world/salary_range_by_title", operation_id="get_salary_range_by_title", summary="Get the minimum and maximum salary for a specific job title")
async def get_salary_range_by_title(title: str = Query(..., description="Job title")):
    cursor.execute("SELECT (SELECT MIN(Salary) FROM Employees WHERE Title = ?) AS MIN, (SELECT MAX(Salary) FROM Employees WHERE Title = ?) AS MAX", (title, title))
    result = cursor.fetchone()
    if not result:
        return {"salary_range": []}
    return {"salary_range": {"min": result[0], "max": result[1]}}

# Endpoint to get contact names based on company name and contact title
@app.get("/v1/retail_world/contact_names_by_company_and_title", operation_id="get_contact_names_by_company_and_title", summary="Get the contact names based on company name and contact title")
async def get_contact_names_by_company_and_title(company_name: str = Query(..., description="Company name"), contact_title: str = Query(..., description="Contact title")):
    cursor.execute("SELECT ContactName FROM Customers WHERE CompanyName = ? AND ContactTitle = ?", (company_name, contact_title))
    result = cursor.fetchall()
    if not result:
        return {"contact_names": []}
    return {"contact_names": result}

# Endpoint to get the count of shippers
@app.get("/v1/retail_world/shipper_count", operation_id="get_shipper_count", summary="Retrieves the total number of shippers in the system. This operation does not require any input parameters and returns a single integer value representing the count of shippers.")
async def get_shipper_count():
    cursor.execute("SELECT COUNT(ShipperID) FROM Shippers")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of orders shipped to a specific country by a specific employee
@app.get("/v1/retail_world/percentage_orders_shipped_by_country_and_employee", operation_id="get_percentage_orders_shipped_by_country_and_employee", summary="Get the percentage of orders shipped to a specific country by a specific employee")
async def get_percentage_orders_shipped_by_country_and_employee(ship_country: str = Query(..., description="Ship country"), first_name: str = Query(..., description="Employee first name"), last_name: str = Query(..., description="Employee last name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.ShipCountry = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.OrderID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.FirstName = ? AND T1.LastName = ?", (ship_country, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get category names based on description
@app.get("/v1/retail_world/category_name_by_description", operation_id="get_category_name_by_description", summary="Retrieves the names of categories that match a given description. The description is used to filter the categories and return the corresponding category names.")
async def get_category_name_by_description(description: str = Query(..., description="Description of the category")):
    cursor.execute("SELECT CategoryName FROM Categories WHERE Description = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get phone numbers based on company name
@app.get("/v1/retail_world/phone_by_company_name", operation_id="get_phone_by_company_name", summary="Get phone numbers based on a specific company name")
async def get_phone_by_company_name(company_name: str = Query(..., description="Name of the company")):
    cursor.execute("SELECT Phone FROM Customers WHERE CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": [row[0] for row in result]}

# Endpoint to get fax numbers based on company name and city
@app.get("/v1/retail_world/fax_by_company_name_and_city", operation_id="get_fax_by_company_name_and_city", summary="Get fax numbers based on a specific company name and city")
async def get_fax_by_company_name_and_city(company_name: str = Query(..., description="Name of the company"), city: str = Query(..., description="City of the company")):
    cursor.execute("SELECT Fax FROM Customers WHERE CompanyName = ? AND City = ?", (company_name, city))
    result = cursor.fetchall()
    if not result:
        return {"fax_numbers": []}
    return {"fax_numbers": [row[0] for row in result]}

# Endpoint to get the count of companies based on city
@app.get("/v1/retail_world/count_companies_by_city", operation_id="get_count_companies_by_city", summary="Get the count of companies based on a specific city")
async def get_count_companies_by_city(city: str = Query(..., description="City of the company")):
    cursor.execute("SELECT COUNT(CompanyName) FROM Customers WHERE City = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get addresses based on company name and contact name
@app.get("/v1/retail_world/address_by_company_name_and_contact_name", operation_id="get_address_by_company_name_and_contact_name", summary="Get addresses based on a specific company name and contact name")
async def get_address_by_company_name_and_contact_name(company_name: str = Query(..., description="Name of the company"), contact_name: str = Query(..., description="Name of the contact person")):
    cursor.execute("SELECT Address FROM Customers WHERE CompanyName = ? AND ContactName = ?", (company_name, contact_name))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [row[0] for row in result]}

# Endpoint to get company names based on country
@app.get("/v1/retail_world/company_name_by_country", operation_id="get_company_name_by_country", summary="Get company names based on a specific country")
async def get_company_name_by_country(country: str = Query(..., description="Country of the company")):
    cursor.execute("SELECT CompanyName FROM Customers WHERE Country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"company_names": []}
    return {"company_names": [row[0] for row in result]}

# Endpoint to get the company name of the supplier with the highest units in stock in a specific city
@app.get("/v1/retail_world/supplier_with_highest_stock_by_city", operation_id="get_supplier_with_highest_stock_by_city", summary="Get the company name of the supplier with the highest units in stock in a specific city")
async def get_supplier_with_highest_stock_by_city(city: str = Query(..., description="City of the supplier")):
    cursor.execute("SELECT T2.CompanyName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.City = ? ORDER BY T1.UnitsInStock DESC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the product name with the highest reorder level from a specific supplier
@app.get("/v1/retail_world/product_with_highest_reorder_level_by_supplier", operation_id="get_product_with_highest_reorder_level_by_supplier", summary="Get the product name with the highest reorder level from a specific supplier")
async def get_product_with_highest_reorder_level_by_supplier(company_name: str = Query(..., description="Name of the supplier company")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ? ORDER BY T1.ReorderLevel DESC LIMIT 1", (company_name,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get category names based on supplier company name and product name pattern
@app.get("/v1/retail_world/category_name_by_supplier_and_product_pattern", operation_id="get_category_name_by_supplier_and_product_pattern", summary="Get category names based on a specific supplier company name and product name pattern")
async def get_category_name_by_supplier_and_product_pattern(company_name: str = Query(..., description="Name of the supplier company"), product_name_pattern: str = Query(..., description="Pattern of the product name (use %% for wildcard)")):
    cursor.execute("SELECT T3.CategoryName FROM Suppliers AS T1 INNER JOIN Products AS T2 ON T1.SupplierID = T2.SupplierID INNER JOIN Categories AS T3 ON T2.CategoryID = T3.CategoryID WHERE T1.CompanyName = ? AND T2.ProductName LIKE ?", (company_name, product_name_pattern))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the country of the supplier based on product name and supplier company name
@app.get("/v1/retail_world/supplier_country_by_product_and_company_name", operation_id="get_supplier_country_by_product_and_company_name", summary="Get the country of the supplier based on a specific product name and supplier company name")
async def get_supplier_country_by_product_and_company_name(product_name: str = Query(..., description="Name of the product"), company_name: str = Query(..., description="Name of the supplier company")):
    cursor.execute("SELECT T2.Country FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.ProductName = ? AND T2.CompanyName = ?", (product_name, company_name))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the category name of the product with the maximum units on order
@app.get("/v1/retail_world/category_name_max_units_on_order", operation_id="get_category_name_max_units_on_order", summary="Get the category name of the product with the maximum units on order")
async def get_category_name_max_units_on_order():
    cursor.execute("SELECT T2.CategoryName FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T1.UnitsOnOrder = ( SELECT MAX(T1.UnitsOnOrder) FROM Products )")
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the price difference between two products from the same supplier
@app.get("/v1/retail_world/price_difference_products", operation_id="get_price_difference_products", summary="Get the price difference between two products from the same supplier")
async def get_price_difference_products(company_name: str = Query(..., description="Company name of the supplier"), product_name_1: str = Query(..., description="Product name 1"), product_name_2: str = Query(..., description="Product name 2")):
    cursor.execute("SELECT ( SELECT T1.UnitPrice FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ? AND T1.ProductName LIKE ? ) - ( SELECT T1.UnitPrice FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ? AND T1.ProductName LIKE ? ) AS calu", (company_name, product_name_1, company_name, product_name_2))
    result = cursor.fetchone()
    if not result:
        return {"price_difference": []}
    return {"price_difference": result[0]}

# Endpoint to get product names from a supplier with a unit price greater than a specified value
@app.get("/v1/retail_world/product_names_supplier_unit_price", operation_id="get_product_names_supplier_unit_price", summary="Get product names from a supplier with a unit price greater than a specified value")
async def get_product_names_supplier_unit_price(company_name: str = Query(..., description="Company name of the supplier"), min_unit_price: float = Query(..., description="Minimum unit price")):
    cursor.execute("SELECT T1.ProductName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName LIKE ? AND T1.UnitPrice > ?", (company_name, min_unit_price))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get distinct product names from a supplier with order quantities greater than a specified value
@app.get("/v1/retail_world/distinct_product_names_supplier_quantity", operation_id="get_distinct_product_names_supplier_quantity", summary="Get distinct product names from a supplier with order quantities greater than a specified value")
async def get_distinct_product_names_supplier_quantity(company_name: str = Query(..., description="Company name of the supplier"), min_quantity: int = Query(..., description="Minimum order quantity")):
    cursor.execute("SELECT DISTINCT T2.ProductName FROM Suppliers AS T1 INNER JOIN Products AS T2 ON T1.SupplierID = T2.SupplierID INNER JOIN `Order Details` AS T3 ON T2.ProductID = T3.ProductID WHERE T1.CompanyName = ? AND T3.Quantity > ?", (company_name, min_quantity))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get category names of products from a supplier with units in stock greater than a specified value
@app.get("/v1/retail_world/category_names_supplier_units_in_stock", operation_id="get_category_names_supplier_units_in_stock", summary="Get category names of products from a supplier with units in stock greater than a specified value")
async def get_category_names_supplier_units_in_stock(min_units_in_stock: int = Query(..., description="Minimum units in stock"), company_name: str = Query(..., description="Company name of the supplier")):
    cursor.execute("SELECT T3.CategoryName FROM Suppliers AS T1 INNER JOIN Products AS T2 ON T1.SupplierID = T2.SupplierID INNER JOIN Categories AS T3 ON T2.CategoryID = T3.CategoryID WHERE T2.UnitsInStock > ? AND T1.CompanyName = ?", (min_units_in_stock, company_name))
    result = cursor.fetchall()
    if not result:
        return {"category_names": []}
    return {"category_names": [row[0] for row in result]}

# Endpoint to get the count and percentage of products with order quantities less than a specified value
@app.get("/v1/retail_world/count_percentage_products_quantity", operation_id="get_count_percentage_products_quantity", summary="Get the count and percentage of products with order quantities less than a specified value")
async def get_count_percentage_products_quantity(max_quantity: int = Query(..., description="Maximum order quantity")):
    cursor.execute("SELECT SUM(CASE WHEN T2.Quantity < ? THEN 1 ELSE 0 END) , CAST(SUM(IF(T2.Quantity < ?, 1, 0)) AS REAL) / COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID", (max_quantity, max_quantity))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "percentage": []}
    return {"count": result[0], "percentage": result[1]}

# Endpoint to get the count and percentage of a specific product in order details
@app.get("/v1/retail_world/count_percentage_specific_product", operation_id="get_count_percentage_specific_product", summary="Get the count and percentage of a specific product in order details")
async def get_count_percentage_specific_product(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT SUM(IF(T1.ProductName = ?, 1, 0)) AS sum , CAST(SUM(IF(T1.ProductName = ?, 1, 0)) AS REAL) / COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID", (product_name, product_name))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "percentage": []}
    return {"count": result[0], "percentage": result[1]}

# Endpoint to get the title of an employee based on first and last name
@app.get("/v1/retail_world/employee_title", operation_id="get_employee_title", summary="Get the title of an employee based on first and last name")
async def get_employee_title(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT Title FROM Employees WHERE FirstName = ? AND LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"title": []}
    return {"title": result[0]}

# Endpoint to get the company name of a shipper based on phone number
@app.get("/v1/retail_world/shipper_company_name", operation_id="get_shipper_company_name", summary="Get the company name of a shipper based on phone number")
async def get_shipper_company_name(phone: str = Query(..., description="Phone number of the shipper")):
    cursor.execute("SELECT CompanyName FROM Shippers WHERE Phone = ?", (phone,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the address and home phone of an employee based on first and last name
@app.get("/v1/retail_world/employee_address_homephone", operation_id="get_employee_address_homephone", summary="Get the address and home phone of an employee based on first and last name")
async def get_employee_address_homephone(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT Address, HomePhone FROM Employees WHERE FirstName = ? AND LastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"address": [], "home_phone": []}
    return {"address": result[0], "home_phone": result[1]}

# Endpoint to get distinct shipping addresses based on ship name
@app.get("/v1/retail_world/distinct_shipping_addresses", operation_id="get_distinct_shipping_addresses", summary="Get distinct shipping addresses based on ship name")
async def get_distinct_shipping_addresses(ship_name: str = Query(..., description="Name of the shipper")):
    cursor.execute("SELECT DISTINCT ShipAddress, ShipCity, ShipRegion, ShipPostalCode, ShipCountry FROM Orders WHERE ShipName = ?", (ship_name,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get supplier details based on product name pattern
@app.get("/v1/retail_world/supplier_details_by_product_name", operation_id="get_supplier_details_by_product_name", summary="Get supplier details based on product name pattern")
async def get_supplier_details_by_product_name(product_name_pattern: str = Query(..., description="Pattern of the product name (use %% for wildcard)")):
    cursor.execute("SELECT T2.CompanyName, T2.ContactName FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T1.ProductName LIKE ?", (product_name_pattern,))
    result = cursor.fetchall()
    if not result:
        return {"supplier_details": []}
    return {"supplier_details": result}

# Endpoint to get category details based on product name
@app.get("/v1/retail_world/category_details_by_product_name", operation_id="get_category_details_by_product_name", summary="Retrieves the category details, including the category name and description, for a specific product. The product is identified by its name, which is provided as an input parameter.")
async def get_category_details_by_product_name(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T2.CategoryName, T2.Description FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"category_details": []}
    return {"category_details": result}

# Endpoint to get distinct supplier company names based on category description
@app.get("/v1/retail_world/distinct_supplier_names_by_category_description", operation_id="get_distinct_supplier_names_by_category_description", summary="Get distinct supplier company names based on category description")
async def get_distinct_supplier_names_by_category_description(category_description: str = Query(..., description="Description of the category")):
    cursor.execute("SELECT DISTINCT T1.CompanyName FROM Suppliers AS T1 INNER JOIN Products AS T2 ON T1.SupplierID = T2.SupplierID INNER JOIN Categories AS T3 ON T2.CategoryID = T3.CategoryID WHERE T3.Description = ?", (category_description,))
    result = cursor.fetchall()
    if not result:
        return {"supplier_names": []}
    return {"supplier_names": result}

# Endpoint to get the unit price of products from a specific supplier that are discontinued
@app.get("/v1/retail_world/product_unit_price_by_supplier_discontinued", operation_id="get_product_unit_price", summary="Get the unit price of products from a specific supplier that are discontinued")
async def get_product_unit_price(company_name: str = Query(..., description="Name of the supplier company"), discontinued: int = Query(..., description="Discontinued status (1 for discontinued, 0 for not discontinued)")):
    cursor.execute("SELECT T1.UnitPrice FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ? AND T1.Discontinued = ?", (company_name, discontinued))
    result = cursor.fetchall()
    if not result:
        return {"unit_prices": []}
    return {"unit_prices": [row[0] for row in result]}

# Endpoint to get the shipping cities for orders containing a specific product
@app.get("/v1/retail_world/shipping_cities_by_product", operation_id="get_shipping_cities", summary="Get the shipping cities for orders containing a specific product")
async def get_shipping_cities(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT T1.ShipCity FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID WHERE T3.ProductName = ?", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"shipping_cities": []}
    return {"shipping_cities": [row[0] for row in result]}

# Endpoint to get the percentage of orders shipped to a specific country by a specific shipper
@app.get("/v1/retail_world/order_percentage_by_country_shipper", operation_id="get_order_percentage", summary="Get the percentage of orders shipped to a specific country by a specific shipper")
async def get_order_percentage(ship_country: str = Query(..., description="Country to which the orders are shipped"), company_name: str = Query(..., description="Name of the shipper company")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T1.ShipCountry = ? THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.OrderID) FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T2.CompanyName = ?", (ship_country, company_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of territories in a specific region
@app.get("/v1/retail_world/territory_count_by_region", operation_id="get_territory_count", summary="Get the count of territories in a specific region")
async def get_territory_count(region_id: int = Query(..., description="ID of the region")):
    cursor.execute("SELECT COUNT(TerritoryID) FROM Territories WHERE RegionID = ?", (region_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of territory descriptions in specified regions
@app.get("/v1/retail_world/territory_description_count_by_regions", operation_id="get_territory_description_count", summary="Get the count of territory descriptions in specified regions")
async def get_territory_description_count(region_id1: int = Query(..., description="ID of the first region"), region_id2: int = Query(..., description="ID of the second region"), region_id3: int = Query(..., description="ID of the third region"), region_id4: int = Query(..., description="ID of the fourth region")):
    cursor.execute("SELECT COUNT(TerritoryDescription) FROM Territories WHERE RegionID IN (?, ?, ?, ?) GROUP BY RegionID", (region_id1, region_id2, region_id3, region_id4))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": [row[0] for row in result]}

# Endpoint to get the count of employees with a specific title in a specific country
@app.get("/v1/retail_world/employee_count_by_country_title", operation_id="get_employee_count", summary="Get the count of employees with a specific title in a specific country")
async def get_employee_count(country: str = Query(..., description="Country of the employees"), title: str = Query(..., description="Title of the employees")):
    cursor.execute("SELECT COUNT(Country) FROM Employees WHERE Country = ? AND Title = ?", (country, title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the salary and title of the highest-paid employee
@app.get("/v1/retail_world/highest_paid_employee", operation_id="get_highest_paid_employee", summary="Get the salary and title of the highest-paid employee")
async def get_highest_paid_employee():
    cursor.execute("SELECT Salary, Title FROM Employees WHERE Salary = ( SELECT MAX(Salary) FROM Employees )")
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"salary": row[0], "title": row[1]} for row in result]}

# Endpoint to get the count of products from a specific supplier with specific stock and order status
@app.get("/v1/retail_world/product_count_by_supplier_stock_order", operation_id="get_product_count", summary="Get the count of products from a specific supplier with specific stock and order status")
async def get_product_count(company_name: str = Query(..., description="Name of the supplier company"), units_in_stock: int = Query(..., description="Units in stock"), units_on_order: int = Query(..., description="Units on order")):
    cursor.execute("SELECT COUNT(T1.ProductID) FROM Products AS T1 INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID WHERE T2.CompanyName = ? AND T1.UnitsInStock = ? AND T1.UnitsOnOrder = ?", (company_name, units_in_stock, units_on_order))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get product name and quantity based on unit price
@app.get("/v1/retail_world/product_name_quantity_by_unit_price", operation_id="get_product_name_quantity", summary="Get product name and quantity based on unit price")
async def get_product_name_quantity(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.ProductName, T2.Quantity FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID ORDER BY T1.UnitPrice DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get employee titles based on the number of orders
@app.get("/v1/retail_world/employee_titles_by_order_count", operation_id="get_employee_titles", summary="Get employee titles based on the number of orders")
async def get_employee_titles(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Title FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID GROUP BY T1.Title ORDER BY COUNT(T2.OrderID) LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": result}

# Endpoint to get product IDs based on customer country
@app.get("/v1/retail_world/product_ids_by_customer_country", operation_id="get_product_ids", summary="Get product IDs based on customer country")
async def get_product_ids(country: str = Query(..., description="Country of the customer"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.ProductID FROM Customers AS T1 INNER JOIN `Order Details` AS T2 WHERE T1.Country = ? GROUP BY T2.ProductID ORDER BY COUNT(T2.ProductID) DESC LIMIT ?", (country, limit))
    result = cursor.fetchall()
    if not result:
        return {"product_ids": []}
    return {"product_ids": result}

# Endpoint to get total order value based on shipper and ship country
@app.get("/v1/retail_world/total_order_value_by_shipper_and_country", operation_id="get_total_order_value", summary="Get total order value based on shipper and ship country")
async def get_total_order_value(company_name: str = Query(..., description="Company name of the shipper"), ship_country: str = Query(..., description="Ship country")):
    cursor.execute("SELECT SUM(T2.Quantity * T2.UnitPrice) FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Shippers AS T3 ON T1.ShipVia = T3.ShipperID WHERE T3.CompanyName = ? AND T1.ShipCountry = ?", (company_name, ship_country))
    result = cursor.fetchone()
    if not result:
        return {"total_value": []}
    return {"total_value": result[0]}

# Endpoint to get employee count and total order value based on reporting manager
@app.get("/v1/retail_world/employee_count_and_order_value_by_manager", operation_id="get_employee_count_and_order_value", summary="Get employee count and total order value based on reporting manager")
async def get_employee_count_and_order_value(reports_to: int = Query(..., description="Employee ID of the reporting manager"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT COUNT(T1.EmployeeID), SUM(T3.Quantity * T3.UnitPrice) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID WHERE T1.ReportsTo = ? ORDER BY SUM(T3.UnitPrice * T3.Quantity) DESC LIMIT ?", (reports_to, limit))
    result = cursor.fetchall()
    if not result:
        return {"employee_count_and_order_value": []}
    return {"employee_count_and_order_value": result}

# Endpoint to get total order value based on employee title
@app.get("/v1/retail_world/total_order_value_by_employee_title", operation_id="get_total_order_value_by_title", summary="Get total order value based on employee title")
async def get_total_order_value_by_title(title: str = Query(..., description="Title of the employee")):
    cursor.execute("SELECT SUM(T3.UnitPrice * T3.Quantity) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID WHERE T1.Title = ? ORDER BY SUM(T3.UnitPrice * T3.Quantity)", (title,))
    result = cursor.fetchone()
    if not result:
        return {"total_value": []}
    return {"total_value": result[0]}

# Endpoint to get company names and ship countries based on product count
@app.get("/v1/retail_world/company_names_and_ship_countries", operation_id="get_company_names_and_ship_countries", summary="Get company names and ship countries based on product count")
async def get_company_names_and_ship_countries(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.CompanyName, T2.ShipCountry FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID INNER JOIN `Order Details` AS T3 ON T2.OrderID = T3.OrderID GROUP BY T1.CompanyName, T2.ShipCountry ORDER BY COUNT(T3.ProductID) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"company_names_and_ship_countries": []}
    return {"company_names_and_ship_countries": result}

# Endpoint to get customer IDs and category names
@app.get("/v1/retail_world/customer_ids_and_category_names", operation_id="get_customer_ids_and_category_names", summary="Get customer IDs and category names")
async def get_customer_ids_and_category_names():
    cursor.execute("SELECT T1.CustomerID, T4.CategoryName FROM Orders AS T1 INNER JOIN `Order Details` AS T2 ON T1.OrderID = T2.OrderID INNER JOIN Products AS T3 ON T2.ProductID = T3.ProductID INNER JOIN Categories AS T4 ON T3.CategoryID = T4.CategoryID ORDER BY T1.CustomerID DESC, T4.CategoryName DESC")
    result = cursor.fetchall()
    if not result:
        return {"customer_ids_and_category_names": []}
    return {"customer_ids_and_category_names": result}

# Endpoint to get the count of product unit prices based on category name
@app.get("/v1/retail_world/count_product_unit_prices_by_category", operation_id="get_count_product_unit_prices", summary="Get the count of product unit prices based on category name")
async def get_count_product_unit_prices(category_name: str = Query(..., description="Category name of the product"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT COUNT(T1.UnitPrice * T3.Quantity) FROM Products AS T1 INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID INNER JOIN `Order Details` AS T3 ON T1.ProductID = T3.ProductID WHERE T2.CategoryName = ? GROUP BY T3.Quantity ORDER BY T3.Quantity DESC LIMIT ?", (category_name, limit))
    result = cursor.fetchall()
    if not result:
        return {"count_product_unit_prices": []}
    return {"count_product_unit_prices": result}

# Endpoint to get the product name based on product ID
@app.get("/v1/retail_world/product_name_by_id", operation_id="get_product_name_by_id", summary="Retrieves the name of a specific product using its unique identifier. The product ID is required as an input parameter to identify the product and return its corresponding name.")
async def get_product_name_by_id(product_id: int = Query(..., description="Product ID")):
    cursor.execute("SELECT ProductName FROM Products WHERE ProductID = ?", (product_id,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get distinct employee names based on company name
@app.get("/v1/retail_world/employee_names_by_company", operation_id="get_employee_names_by_company", summary="Get distinct employee names based on the company name")
async def get_employee_names_by_company(company_name: str = Query(..., description="Company name")):
    cursor.execute("SELECT DISTINCT T1.FirstName, T1.LastName FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Customers AS T3 ON T2.CustomerID = T3.CustomerID WHERE T3.CompanyName = ?", (company_name,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": result}

# Endpoint to get the percentage of territories in one region compared to another
@app.get("/v1/retail_world/territory_percentage_comparison", operation_id="get_territory_percentage_comparison", summary="Get the percentage of territories in one region compared to another")
async def get_territory_percentage_comparison(region_description_1: str = Query(..., description="First region description"), region_description_2: str = Query(..., description="Second region description")):
    cursor.execute("SELECT CAST(( SELECT COUNT(T1.TerritoryID) FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ? ) AS REAL) * 100 / ( SELECT COUNT(T1.TerritoryID) FROM Territories AS T1 INNER JOIN Region AS T2 ON T1.RegionID = T2.RegionID WHERE T2.RegionDescription = ? ) AS Calu", (region_description_1, region_description_2))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get employee IDs based on a range of territory IDs
@app.get("/v1/retail_world/employee_ids_by_territory_range", operation_id="get_employee_ids_by_territory_range", summary="Get employee IDs based on a range of territory IDs")
async def get_employee_ids_by_territory_range(min_territory_id: int = Query(..., description="Minimum territory ID"), max_territory_id: int = Query(..., description="Maximum territory ID")):
    cursor.execute("SELECT EmployeeID FROM EmployeeTerritories WHERE TerritoryID BETWEEN ? AND ?", (min_territory_id, max_territory_id))
    result = cursor.fetchall()
    if not result:
        return {"employee_ids": []}
    return {"employee_ids": result}

# Endpoint to get region and territory information based on employee name
@app.get("/v1/retail_world/region_territory_by_employee_name", operation_id="get_region_territory_by_employee_name", summary="Get region and territory information based on the employee's first and last name")
async def get_region_territory_by_employee_name(last_name: str = Query(..., description="Last name of the employee"), first_name: str = Query(..., description="First name of the employee")):
    cursor.execute("SELECT T3.RegionID, T3.TerritoryDescription, T4.RegionDescription FROM Employees AS T1 INNER JOIN EmployeeTerritories AS T2 ON T1.EmployeeID = T2.EmployeeID INNER JOIN Territories AS T3 ON T2.TerritoryID = T3.TerritoryID INNER JOIN Region AS T4 ON T3.RegionID = T4.RegionID WHERE T1.LastName = ? AND T1.FirstName = ?", (last_name, first_name))
    result = cursor.fetchall()
    if not result:
        return {"region_territory_info": []}
    return {"region_territory_info": result}

# Endpoint to get the count of orders by employee based on title and hire year
@app.get("/v1/retail_world/order_count_by_employee_title_hire_year", operation_id="get_order_count_by_employee_title_hire_year", summary="Get the count of orders by employee based on their title and hire year")
async def get_order_count_by_employee_title_hire_year(title: str = Query(..., description="Title of the employee"), hire_year: str = Query(..., description="Hire year in 'YYYY' format")):
    cursor.execute("SELECT T1.FirstName, T1.LastName, COUNT(T2.OrderID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.Title = ? AND STRFTIME('%Y', T1.HireDate) = ? GROUP BY T1.EmployeeID, T1.FirstName, T1.LastName", (title, hire_year))
    result = cursor.fetchall()
    if not result:
        return {"order_counts": []}
    return {"order_counts": result}

# Endpoint to get the total sales amount for a specific product
@app.get("/v1/retail_world/total_sales_amount_by_product", operation_id="get_total_sales_amount", summary="Get the total sales amount for a specific product")
async def get_total_sales_amount(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) AS sum FROM Products AS T1 INNER JOIN `Order Details` AS T2 ON T1.ProductID = T2.ProductID WHERE T1.ProductName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get product and supplier company name based on order ID
@app.get("/v1/retail_world/product_supplier_by_order_id", operation_id="get_product_supplier", summary="Get product name and supplier company name based on a specific order ID")
async def get_product_supplier(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T2.ProductName, T1.CompanyName FROM Suppliers AS T1 INNER JOIN Products AS T2 ON T1.SupplierID = T2.SupplierID INNER JOIN `Order Details` AS T3 ON T2.ProductID = T3.ProductID WHERE T3.OrderID = ? ORDER BY T2.ReorderLevel DESC LIMIT 1", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": [], "company_name": []}
    return {"product_name": result[0], "company_name": result[1]}

# Endpoint to get the top shipper company name based on shipped year
@app.get("/v1/retail_world/top_shipper_by_year", operation_id="get_top_shipper", summary="Get the top shipper company name based on a specific shipped year")
async def get_top_shipper(shipped_year: str = Query(..., description="Shipped year in 'YYYY' format")):
    cursor.execute("SELECT T1.CompanyName FROM Shippers AS T1 INNER JOIN Orders AS T2 ON T1.ShipperID = T2.ShipVia WHERE STRFTIME('%Y', T2.ShippedDate) = ? GROUP BY T1.CompanyName ORDER BY COUNT(T2.OrderID) DESC LIMIT 1", (shipped_year,))
    result = cursor.fetchone()
    if not result:
        return {"company_name": []}
    return {"company_name": result[0]}

# Endpoint to get the count of customers based on city
@app.get("/v1/retail_world/customer_count_by_city", operation_id="get_customer_count", summary="Retrieves the total number of customers residing in a specified city. The city is provided as an input parameter.")
async def get_customer_count(city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(CustomerID) FROM Customers WHERE City = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last name of employees based on birth date
@app.get("/v1/retail_world/employee_names_by_birthdate", operation_id="get_employee_names", summary="Retrieves the first and last names of employees who share a specific birth date. The birth date must be provided in 'YYYY-MM-DD HH:MM:SS' format.")
async def get_employee_names(birth_date: str = Query(..., description="Birth date in 'YYYY-MM-DD HH:MM:SS' format")):
    cursor.execute("SELECT FirstName, LastName FROM Employees WHERE BirthDate = ?", (birth_date,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of orders based on employee city
@app.get("/v1/retail_world/order_count_by_employee_city", operation_id="get_order_count_by_city", summary="Get the count of orders based on a specific employee city")
async def get_order_count_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(T2.OrderID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.City = ?", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country of employees based on order ID
@app.get("/v1/retail_world/employee_country_by_order_id", operation_id="get_employee_country", summary="Get the country of employees based on a specific order ID")
async def get_employee_country(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.Country FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the title of employees based on order ID
@app.get("/v1/retail_world/employee_title_by_order_id", operation_id="get_employee_title_by_order_id", summary="Get the title of employees based on a specific order ID")
async def get_employee_title_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.Title FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"titles": []}
    return {"titles": [row[0] for row in result]}

# Endpoint to get the phone number of customers based on order ID
@app.get("/v1/retail_world/customer_phone_by_order_id", operation_id="get_customer_phone_by_order_id", summary="Get the phone number of customers based on a specific order ID")
async def get_customer_phone_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.Phone FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": [row[0] for row in result]}

# Endpoint to get the region of customers based on order ID
@app.get("/v1/retail_world/customer_region_by_order_id", operation_id="get_customer_region_by_order_id", summary="Get the region of customers based on a specific order ID")
async def get_customer_region_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.Region FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"regions": []}
    return {"regions": [row[0] for row in result]}

# Endpoint to get the highest average salary of employees by first and last name based on ship country
@app.get("/v1/retail_world/highest_avg_salary_by_ship_country", operation_id="get_highest_avg_salary_by_ship_country", summary="Get the highest average salary of employees by first and last name based on a specific ship country")
async def get_highest_avg_salary_by_ship_country(ship_country: str = Query(..., description="Ship country")):
    cursor.execute("SELECT T1.FirstName, T1.LastName, AVG(T1.Salary) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.ShipCountry = ? GROUP BY T1.FirstName, T1.LastName ORDER BY SUM(T1.Salary) DESC LIMIT 1", (ship_country,))
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": {"first_name": result[0], "last_name": result[1], "avg_salary": result[2]}}

# Endpoint to get the percentage of salary for a specific year
@app.get("/v1/retail_world/percentage_salary_by_year", operation_id="get_percentage_salary_by_year", summary="Get the percentage of salary for a specific year")
async def get_percentage_salary_by_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN STRFTIME('%Y', T2.ShippedDate) = ? THEN T1.Salary ELSE 0 END) AS REAL) * 100 / SUM(T1.Salary) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID", (year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the hire date of employees based on order ID
@app.get("/v1/retail_world/employee_hire_date_by_order_id", operation_id="get_employee_hire_date_by_order_id", summary="Get the hire date of employees based on a specific order ID")
async def get_employee_hire_date_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T1.HireDate FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T2.OrderID = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"hire_dates": []}
    return {"hire_dates": [row[0] for row in result]}

# Endpoint to get the count of orders based on the employee's report to
@app.get("/v1/retail_world/order_count_by_reports_to", operation_id="get_order_count_by_reports_to", summary="Get the count of orders based on the employee's report to")
async def get_order_count_by_reports_to(reports_to: int = Query(..., description="Employee ID to whom the employee reports")):
    cursor.execute("SELECT COUNT(T2.OrderID) FROM Employees AS T1 INNER JOIN Orders AS T2 ON T1.EmployeeID = T2.EmployeeID WHERE T1.ReportsTo = ?", (reports_to,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone number of shippers based on order ID
@app.get("/v1/retail_world/shipper_phone_by_order_id", operation_id="get_shipper_phone_by_order_id", summary="Get the phone number of shippers based on a specific order ID")
async def get_shipper_phone_by_order_id(order_id: int = Query(..., description="Order ID")):
    cursor.execute("SELECT T2.Phone FROM Orders AS T1 INNER JOIN Shippers AS T2 ON T1.ShipVia = T2.ShipperID WHERE T1.OrderID = ?", (order_id,))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": [row[0] for row in result]}

api_calls = [
    "/v1/retail_world/employee_count_by_title_of_courtesy?title_of_courtesy=Dr.",
    "/v1/retail_world/employee_name_by_manager_name?manager_last_name=Davolio&manager_first_name=Nancy",
    "/v1/retail_world/employee_home_phone_by_name?last_name=Davolio&first_name=Nancy",
    "/v1/retail_world/employee_count_by_manager_name?manager_last_name=Fuller&manager_first_name=Andrew",
    "/v1/retail_world/employee_with_highest_salary",
    "/v1/retail_world/salary_difference_between_employees?last_name_1=Fuller&first_name_1=Andrew&last_name_2=Davolio&first_name_2=Nancy",
    "/v1/retail_world/employee_count_by_title_and_country?title=Sales%20Representative&country=UK",
    "/v1/retail_world/employee_names_by_territory?territory_description=Hollis",
    "/v1/retail_world/territory_count_by_employee_name?first_name=Nancy&last_name=Davolio",
    "/v1/retail_world/territory_descriptions_by_employee_name?first_name=Nancy&last_name=Davolio",
    "/v1/retail_world/count_territories_by_city?city=London",
    "/v1/retail_world/territory_descriptions_by_reporting_employee?first_name=Andrew&last_name=Fuller",
    "/v1/retail_world/count_employees_by_country_and_territory_count?country=UK&min_territory_count=4",
    "/v1/retail_world/count_territories_by_region?region_description=Eastern",
    "/v1/retail_world/distinct_territory_descriptions_by_region?region_description=Eastern",
    "/v1/retail_world/count_employees_by_region?region_description=Eastern",
    "/v1/retail_world/home_phones_by_region?region_description=Eastern",
    "/v1/retail_world/territory_count_difference_by_regions?region_description_1=Eastern&region_description_2=Southern",
    "/v1/retail_world/product_names_by_order?order_id=10248",
    "/v1/retail_world/product_quantity_by_order?order_id=10273&product_name=Ikura",
    "/v1/retail_world/total_price_product_order?order_id=10273&product_name=Ikura",
    "/v1/retail_world/total_stock_on_order_highest_quantity?order_id=10248",
    "/v1/retail_world/product_name_highest_reorder_level?order_id=10248",
    "/v1/retail_world/count_orders_product?product_name=Tofu",
    "/v1/retail_world/order_ids_discontinued_products?discontinued=1",
    "/v1/retail_world/count_orders_unit_price_less?product_name=Ikura",
    "/v1/retail_world/supplier_company_name?product_name=Aniseed%20Syrup",
    "/v1/retail_world/product_names_by_country?country=Japan",
    "/v1/retail_world/supplier_phones_low_stock",
    "/v1/retail_world/count_discontinued_products_by_country?country=USA&discontinued=1",
    "/v1/retail_world/suppliers_with_highest_unit_price_product",
    "/v1/retail_world/avg_salary_employees_managing_territories?min_territories=4",
    "/v1/retail_world/percentage_difference_unit_price?order_id=10273&product_name=Ikura",
    "/v1/retail_world/avg_quantity_per_order?product_name=Ikura",
    "/v1/retail_world/product_highest_units_on_order_ratio?order_id=10248",
    "/v1/retail_world/employee_latest_birth_date",
    "/v1/retail_world/employee_latest_hire_date",
    "/v1/retail_world/count_orders_employee?first_name=Michael&last_name=Suyama",
    "/v1/retail_world/territory_descriptions_employee?title_of_courtesy=Mr.&first_name=Robert&last_name=King",
    "/v1/retail_world/territory_descriptions_manager?title_of_courtesy=Ms.&first_name=Laura&last_name=Callahan",
    "/v1/retail_world/employee_home_phones_by_territory?territory_description=Savannah",
    "/v1/retail_world/total_order_quantity_by_customer_and_date?customer_id=HILAA&order_date=1997-12-25%",
    "/v1/retail_world/product_count_max_freight",
    "/v1/retail_world/shipper_company_by_order?order_id=10585",
    "/v1/retail_world/order_count_by_shipper?company_name=Federal%20Shipping",
    "/v1/retail_world/product_count_by_category?category_name=Dairy%20Products",
    "/v1/retail_world/category_by_product?product_name=Tofu",
    "/v1/retail_world/total_units_by_supplier?company_name=Escargots%20Nouveaux",
    "/v1/retail_world/reorder_level_by_supplier?company_name=Nord-Ost-Fisch%20Handelsgesellschaft%20mbH",
    "/v1/retail_world/product_count_by_supplier?company_name=Karkki%20Oy",
    "/v1/retail_world/supplier_contact_by_product?product_name=Gudbrandsdalsost",
    "/v1/retail_world/supplier_country_by_product?product_name=Scottish%20Longbreads",
    "/v1/retail_world/region_description_by_territory?territory_description=Columbia",
    "/v1/retail_world/order_percentage_by_shipper_and_customer?company_name=United%20Package&customer_id=WHITC",
    "/v1/retail_world/order_percentage_difference_by_shippers?company_name_1=United%20Package&company_name_2=Speedy%20Express",
    "/v1/retail_world/territory_count_ratio_by_regions?region_description_1=Eastern&region_description_2=Southern",
    "/v1/retail_world/country_with_most_customers",
    "/v1/retail_world/orders_by_freight_cost?min_freight=800",
    "/v1/retail_world/customer_details_by_company_name?company_name=Island%20Trading",
    "/v1/retail_world/product_with_highest_unit_price",
    "/v1/retail_world/supplier_count_by_country?country=USA",
    "/v1/retail_world/product_with_highest_reorder_level",
    "/v1/retail_world/product_names_by_discontinued_status?discontinued=1",
    "/v1/retail_world/average_salary_by_title_and_country?title=Sales%20Representative&country=UK",
    "/v1/retail_world/most_frequent_shipper_by_country?ship_country=USA",
    "/v1/retail_world/order_count_by_customer_and_shipper?customer_id=GREAL&company_name=United%20Package",
    "/v1/retail_world/supplier_names_with_highest_reorder_level",
    "/v1/retail_world/discontinued_product_count_by_country?discontinued=1&country=Australia",
    "/v1/retail_world/top_orders_by_total_cost?limit=2",
    "/v1/retail_world/top_supplier_by_product_count?limit=1",
    "/v1/retail_world/distinct_product_names_by_unit_price?unit_price=5",
    "/v1/retail_world/top_employee_title_by_territory_count?limit=1",
    "/v1/retail_world/top_product_by_order_count?limit=1",
    "/v1/retail_world/territory_descriptions_by_title_of_courtesy?title_of_courtesy=Dr.",
    "/v1/retail_world/average_territories_per_employee_by_title?title=Sales%20Representative",
    "/v1/retail_world/employee_highest_orders",
    "/v1/retail_world/product_highest_freight_count",
    "/v1/retail_world/product_name_highest_combined_units?category_name=Seafood",
    "/v1/retail_world/salary_range_top_two_employees",
    "/v1/retail_world/average_unit_price_above_stock?units_in_stock=50",
    "/v1/retail_world/company_names_highest_relationships_city",
    "/v1/retail_world/customer_details_by_city?city=Madrid",
    "/v1/retail_world/employee_details_by_birthdate",
    "/v1/retail_world/average_unit_price_by_country?country=UK",
    "/v1/retail_world/top_product_by_quantity",
    "/v1/retail_world/supplier_details_by_unit_price?unit_price=30",
    "/v1/retail_world/customer_details_above_avg_freight",
    "/v1/retail_world/supplier_companies_discontinued_products?discontinued=1&country=USA",
    "/v1/retail_world/product_names_by_order_date?order_date=1998%25&limit=5",
    "/v1/retail_world/customer_count_by_shipper?ship_name=Federal%20Shipping",
    "/v1/retail_world/top_products_by_order_count?limit=10",
    "/v1/retail_world/employee_details_with_region",
    "/v1/retail_world/employee_details_by_order_date_range?start_date=1996-01-01%2000:00:00&end_date=1997-01-01%2000:00:00",
    "/v1/retail_world/avg_unit_price_by_shipper_year?year=1997&company_name=United%20Package",
    "/v1/retail_world/percentage_orders_by_shipper?company_name=Speedy%20Express",
    "/v1/retail_world/top_titles_of_courtesy_by_salary",
    "/v1/retail_world/employee_last_names_by_reporting_title?title=Vice%20President%2C%20Sales",
    "/v1/retail_world/top_order_detail_by_total_price",
    "/v1/retail_world/top_products_by_total_units",
    "/v1/retail_world/product_id_by_reorder_level_and_unit_price",
    "/v1/retail_world/count_products_by_category?category_name=Dairy%20Products",
    "/v1/retail_world/territory_descriptions_by_employee_title?title=Inside%20Sales%20Coordinator",
    "/v1/retail_world/customer_company_names_by_freight?freight=2000000",
    "/v1/retail_world/top_shipper_by_ship_city?ship_city=Aachen",
    "/v1/retail_world/employee_names_by_region?region_description=Northern",
    "/v1/retail_world/product_names_by_ship_city?ship_city=Paris",
    "/v1/retail_world/top_product_by_postal_code?postal_code=28023",
    "/v1/retail_world/employee_names_by_region_and_manager?region_description=Southern&manager_first_name=Andrew&manager_last_name=Fuller",
    "/v1/retail_world/order_dates_by_product_quantity_company?product_name=Filo%20Mix&quantity=9&company_name=Du%20monde%20entier",
    "/v1/retail_world/category_names_by_order_id?order_id=10933",
    "/v1/retail_world/average_quantity_per_order?shipped_date_pattern=1996-11%25&company_name=Federal%20Shipping",
    "/v1/retail_world/percentage_orders_shipped?shipped_date_pattern=1996-09%25&company_name=United%20Package&year_pattern=1996%25",
    "/v1/retail_world/count_customers_by_contact_title?country=Mexico&contact_title=Owner",
    "/v1/retail_world/customer_address_by_contact_name?contact_name=Andr%20Fonseca",
    "/v1/retail_world/company_names_by_phone_pattern?phone_pattern=(171)%25",
    "/v1/retail_world/employee_title_count_difference?country1=UK&title=Sales%20Representative&country2=USA",
    "/v1/retail_world/count_customers_by_city_country_contact_title?city=Sao%20Paulo&country=Brazil&contact_title=Sales%20Associate",
    "/v1/retail_world/employee_last_name_by_order_customer?order_id=10521&customer_id=CACTU",
    "/v1/retail_world/freight_cost_by_order_company?order_id=10692&company_name=Alfreds%20Futterkiste",
    "/v1/retail_world/order_ids_by_shipper_company?company_name=Speedy%20Express&limit=3",
    "/v1/retail_world/product_names_by_category?category_name=Beverages",
    "/v1/retail_world/category_descriptions_by_product?product_name=tofu",
    "/v1/retail_world/product_names_by_supplier_company?company_name=Aux%20joyeux%20ecclsiastiques",
    "/v1/retail_world/unit_prices_by_supplier_company_pattern_and_product?company_name_pattern=Mayumi%25&product_name=Konbu",
    "/v1/retail_world/discontinued_product_names_by_category?discontinued=1&category_name=Meat%2FPoultry&limit=3",
    "/v1/retail_world/product_names_by_supplier_company_ordered?company_name=Heli%20Swaren%20GmbH%20%26%20Co.%20KG&limit=2",
    "/v1/retail_world/supplier_contact_names_by_company?company_name=Heli%20Swaren%20GmbH%20%26%20Co.%20KG",
    "/v1/retail_world/customer_country_by_company_name?company_name=Drachenblut%20Delikatessen",
    "/v1/retail_world/count_territories",
    "/v1/retail_world/highest_total_unit_price",
    "/v1/retail_world/count_orders_by_ship_country?ship_country=France",
    "/v1/retail_world/supplier_homepage_by_product_name?product_name=Thringer%20Rostbratwurst",
    "/v1/retail_world/distinct_employee_first_names_by_ship_city?ship_city=Reims",
    "/v1/retail_world/highest_quantity_by_product_name?product_name=Manjimup%20Dried%20Apples",
    "/v1/retail_world/product_unit_price_by_category?description=Cheeses",
    "/v1/retail_world/order_count_by_customer?company_name=Laughing%20Bacchus%20Wine%20Cellars",
    "/v1/retail_world/product_names_by_ship_address?ship_address=Starenweg%205",
    "/v1/retail_world/territory_descriptions_by_employee?last_name=King&first_name=Robert",
    "/v1/retail_world/contact_names_by_ship_country?ship_country=Switzerland",
    "/v1/retail_world/order_percentage_by_employee_title?title=Sales%20Representative",
    "/v1/retail_world/average_salary_by_employee_id_range?min_employee_id=1&max_employee_id=9",
    "/v1/retail_world/total_salary_by_country?country=UK",
    "/v1/retail_world/check_employee_home_phone?home_phone=(206)%20555-1189&first_name=Laura&last_name=Callahan",
    "/v1/retail_world/get_notes_highest_salary",
    "/v1/retail_world/get_customer_ids_by_employee?first_name=Michael&last_name=Suyama",
    "/v1/retail_world/get_distinct_ship_countries_by_employee?first_name=Janet&last_name=Leverling",
    "/v1/retail_world/get_order_count_by_employee?first_name=Margaret&last_name=Peacock",
    "/v1/retail_world/get_average_salary_by_employee?first_name=Andrew&last_name=Fuller",
    "/v1/retail_world/get_quantity_per_unit_by_supplier?company_name=Tokyo%20Traders",
    "/v1/retail_world/get_discontinued_product_count_by_supplier?company_name=New%20Orleans%20Cajun%20Delights",
    "/v1/retail_world/get_average_unit_price_by_supplier?company_name=Formaggi%20Fortini%20s.r.l.",
    "/v1/retail_world/supplier_contact_title_by_quantity?quantity_per_unit=10%20boxes%20x%2012%20pieces",
    "/v1/retail_world/total_units_on_order_by_company?company_name=Exotic%20Liquids",
    "/v1/retail_world/percentage_products_by_company?company_name=Gai%20pturage",
    "/v1/retail_world/top_products_by_stock?limit=5",
    "/v1/retail_world/discontinued_product_count?discontinued=1",
    "/v1/retail_world/employee_details_by_title?title=Sales%20Manager",
    "/v1/retail_world/employee_names_by_title?title=Vice%20President%2C%20Sales",
    "/v1/retail_world/top_companies_by_sales?limit=10",
    "/v1/retail_world/average_product_sales_by_category",
    "/v1/retail_world/total_quantity_by_product_name?product_name_pattern=Uncle%20Bob%25s%20Organic%20Dried%20Pears",
    "/v1/retail_world/count_products_by_category_and_quantity?category_name=Seafood&min_quantity=50",
    "/v1/retail_world/product_details_by_supplier?company_name=Pavlova%2C%20Ltd.",
    "/v1/retail_world/distinct_suppliers_for_discontinued_products?discontinued=1",
    "/v1/retail_world/employee_names_by_id?max_employee_id=4",
    "/v1/retail_world/count_distinct_employee_first_names_by_region?region_description=Eastern",
    "/v1/retail_world/count_orders_by_shipper_and_date_range?company_name=Federal%20Shipping&start_date=1997-03-01%2000:00:00&end_date=1997-10-08%2023:59:59",
    "/v1/retail_world/distinct_customer_ids_by_employee_and_date_range?last_name=Peacock&first_name=Margaret&ship_country=Brazil&start_date=1997-03-31%2000:00:00&end_date=1997-12-10%2023:59:59",
    "/v1/retail_world/reorder_level_by_order_quantity?quantity=1",
    "/v1/retail_world/product_value_in_stock",
    "/v1/retail_world/late_shipped_orders_by_country?country=USA",
    "/v1/retail_world/top_customer_by_order_value",
    "/v1/retail_world/top_employee_by_payment",
    "/v1/retail_world/count_customers_by_city_and_country?country=Germany&city=Berlin",
    "/v1/retail_world/count_products_by_supplier?company_name=Exotic%20Liquids",
    "/v1/retail_world/min_price_product_and_supplier",
    "/v1/retail_world/top_region_by_territories?limit=1",
    "/v1/retail_world/region_description_by_territory_id?territory_id=2116",
    "/v1/retail_world/percentage_customers_by_city_year?city=Madrid&year=1996",
    "/v1/retail_world/employee_list",
    "/v1/retail_world/employee_count_by_title_salary?salary=2000&title=Sales%20Representative",
    "/v1/retail_world/customer_count_by_country_year?year=1996&country=UK",
    "/v1/retail_world/top_company_by_orders_year?year=1998&limit=1",
    "/v1/retail_world/customer_count_by_year_grouped_by_country?year=1996",
    "/v1/retail_world/order_count_by_company_year?year=1999&company_name=Hanna%20Moos",
    "/v1/retail_world/order_shipping_difference_by_company?company_name=Berglunds%20snabbkp&limit=1",
    "/v1/retail_world/company_name_by_order_id?order_id=10257",
    "/v1/retail_world/most_orders_year_by_company?company_name=Around%20the%20Horn",
    "/v1/retail_world/top_supplier_country",
    "/v1/retail_world/employees_reporting_to_title?title=Sales%20Manager",
    "/v1/retail_world/top_customer_by_month_year?month_year=1996-08",
    "/v1/retail_world/salary_of_earliest_hired_employee",
    "/v1/retail_world/max_age_difference_birth_hire",
    "/v1/retail_world/total_revenue_discontinued_products?discontinued=1",
    "/v1/retail_world/top_category_discontinued_products?discontinued=1",
    "/v1/retail_world/product_count_category_year?category_name=Condiments&category_id=2&year=1997",
    "/v1/retail_world/top_company_by_products_ordered?limit=1",
    "/v1/retail_world/average_ship_via_by_company_and_year?divisor=12&company_name=Federal%20Shipping&year=1996",
    "/v1/retail_world/product_names_by_supplier_company_name?company_name_pattern=G%day,%20Mate",
    "/v1/retail_world/top_territory_by_region_count?limit=1",
    "/v1/retail_world/employee_count_by_title_and_territory?title1=Ms.&title2=Mrs.&min_territories=3",
    "/v1/retail_world/supplier_company_names_by_reorder_level?limit=8",
    "/v1/retail_world/top_company_by_order_value_and_discount?discount=0&limit=1",
    "/v1/retail_world/total_order_value_by_employee_and_date?first_name=Nancy&last_name=Davolio&order_date_pattern=1996-12&discount=0",
    "/v1/retail_world/total_order_value_by_year?year=1997",
    "/v1/retail_world/average_order_value_by_date_range?divisor=3&start_date=1996-01-01%2000:00:00&end_date=1998-12-31%2023:59:59",
    "/v1/retail_world/order_count_by_country_year?ship_country=Venezuela&ship_year=1996",
    "/v1/retail_world/category_details?category_name=Condiments",
    "/v1/retail_world/order_details_by_total_price?total_price=15000",
    "/v1/retail_world/employee_territories?employee_id=7",
    "/v1/retail_world/supplier_details_by_city?city=Sydney",
    "/v1/retail_world/supplier_names_by_country?country=Germany",
    "/v1/retail_world/employee_details_by_city?city=London",
    "/v1/retail_world/employee_details_by_manager_title?manager_title=Sales%20Manager",
    "/v1/retail_world/customers_by_country?country=Canada",
    "/v1/retail_world/product_stock_by_quantity_per_unit?quantity_per_unit=10%20-%20500%20g%20pkgs.",
    "/v1/retail_world/top_reorder_product",
    "/v1/retail_world/suppliers_by_total_stock?total_stock=120",
    "/v1/retail_world/contacts_by_supplier_company?company_name=Escargots%20Nouveaux",
    "/v1/retail_world/territory_region_by_employee?title_of_courtesy=Mrs.&last_name=Peacock&first_name=Margaret",
    "/v1/retail_world/products_by_required_date_customer?required_date=1998-03-26%25&customer_id=WILMK",
    "/v1/retail_world/products_by_max_reorder_level",
    "/v1/retail_world/order_count_by_customer_country?country=Ireland",
    "/v1/retail_world/product_names_by_country_and_year?country=Norway&order_year=1996",
    "/v1/retail_world/supplier_company_names_by_employee_and_order?first_name=Anne&last_name=Dodsworth&ship_country=Brazil&order_date_pattern=1996-12%",
    "/v1/retail_world/employee_details_and_order_count_by_ship_country?ship_country=USA",
    "/v1/retail_world/territory_details_by_region_description?region_description=Southern",
    "/v1/retail_world/average_payment_per_product_by_category?category_name=Confections",
    "/v1/retail_world/total_payment_by_city?city=San%20Francisco",
    "/v1/retail_world/total_units_by_supplier_country?country=Japan",
    "/v1/retail_world/product_and_category_names_by_supplier_country_and_discontinued?country=Australia&discontinued=1",
    "/v1/retail_world/highest_payment_order_by_product?product_name=Ipoh%20Coffee",
    "/v1/retail_world/total_payment_per_product_by_order?order_id=10979",
    "/v1/retail_world/supplier_contacts_by_category_and_id_range?category_name=Grains/Cereals&min_supplier_id=1&max_supplier_id=10",
    "/v1/retail_world/products_shipped_before_required_date",
    "/v1/retail_world/products_by_employee_title_shipped_date_country?title=Inside%20Sales%20Coordinator&shipped_date=1996%&ship_country=Mexico",
    "/v1/retail_world/products_by_category_max_reorder_level?category_name=Dairy%20Products",
    "/v1/retail_world/total_units_by_product_name?product_name=Mascarpone%20Fabioli",
    "/v1/retail_world/products_by_min_max_unit_price",
    "/v1/retail_world/employee_count_by_region?region_description=Northern",
    "/v1/retail_world/average_order_value",
    "/v1/retail_world/percentage_discontinued_products?discontinued=1",
    "/v1/retail_world/employee_details_by_order_id?order_id=10274",
    "/v1/retail_world/product_details_by_employee_and_company?first_name=Nancy&last_name=Davolio&company_name=GROSELLA-Restaurante",
    "/v1/retail_world/order_statistics_by_company?company_name=Laughing%20Bacchus%20Wine%20Cellars",
    "/v1/retail_world/quantity_count_by_product?product_name=Pavlova",
    "/v1/retail_world/salary_range_by_title?title=Sales%20Representative",
    "/v1/retail_world/contact_names_by_company_and_title?company_name=Eastern%20Connection&contact_title=Sales%20Agent",
    "/v1/retail_world/shipper_count",
    "/v1/retail_world/percentage_orders_shipped_by_country_and_employee?ship_country=Austria&first_name=Andrew&last_name=Fuller",
    "/v1/retail_world/category_name_by_description?description=Soft%20drinks%2C%20coffees%2C%20teas%2C%20beers%2C%20and%20ales",
    "/v1/retail_world/phone_by_company_name?company_name=Around%20the%20Horn",
    "/v1/retail_world/fax_by_company_name_and_city?company_name=Blondesddsl%20pre%20et%20fils&city=Strasbourg",
    "/v1/retail_world/count_companies_by_city?city=London",
    "/v1/retail_world/address_by_company_name_and_contact_name?company_name=Eastern%20Connection&contact_name=Ann%20Devon",
    "/v1/retail_world/company_name_by_country?country=France",
    "/v1/retail_world/supplier_with_highest_stock_by_city?city=London",
    "/v1/retail_world/product_with_highest_reorder_level_by_supplier?company_name=Exotic%20Liquids",
    "/v1/retail_world/category_name_by_supplier_and_product_pattern?company_name=New%20Orleans%20Cajun%20Delights&product_name_pattern=Chef%20Anton%25s%20Gumbo%20Mix",
    "/v1/retail_world/supplier_country_by_product_and_company_name?product_name=Ipoh%20Coffee&company_name=Leka%20Trading",
    "/v1/retail_world/category_name_max_units_on_order",
    "/v1/retail_world/price_difference_products?company_name=New%20Orleans%20Cajun%20Delights&product_name_1=Chef%20Anton%25s%20Cajun%20Seasoning&product_name_2=Chef%20Anton%25s%20Gumbo%20Mix",
    "/v1/retail_world/product_names_supplier_unit_price?company_name=Cooperativa%20de%20Quesos%25&min_unit_price=20",
    "/v1/retail_world/distinct_product_names_supplier_quantity?company_name=Tokyo%20Traders&min_quantity=40",
    "/v1/retail_world/category_names_supplier_units_in_stock?min_units_in_stock=100&company_name=Exotic%20Liquids",
    "/v1/retail_world/count_percentage_products_quantity?max_quantity=50",
    "/v1/retail_world/count_percentage_specific_product?product_name=Geitost",
    "/v1/retail_world/employee_title?first_name=Robert&last_name=King",
    "/v1/retail_world/shipper_company_name?phone=(503)%20555-9931",
    "/v1/retail_world/employee_address_homephone?first_name=Margaret&last_name=Peacock",
    "/v1/retail_world/distinct_shipping_addresses?ship_name=Rattlesnake%20Canyon%20Grocery",
    "/v1/retail_world/supplier_details_by_product_name?product_name_pattern=Sir%20Rodney%25s%20Marmalade",
    "/v1/retail_world/category_details_by_product_name?product_name=Mozzarella%20di%20Giovanni",
    "/v1/retail_world/distinct_supplier_names_by_category_description?category_description=Cheeses",
    "/v1/retail_world/product_unit_price_by_supplier_discontinued?company_name=Plutzer%20Lebensmittelgromrkte%20AG&discontinued=1",
    "/v1/retail_world/shipping_cities_by_product?product_name=Mishi%20Kobe%20Niku",
    "/v1/retail_world/order_percentage_by_country_shipper?ship_country=Sweden&company_name=Speedy%20Express",
    "/v1/retail_world/territory_count_by_region?region_id=1",
    "/v1/retail_world/territory_description_count_by_regions?region_id1=1&region_id2=2&region_id3=3&region_id4=4",
    "/v1/retail_world/employee_count_by_country_title?country=USA&title=Sales%20Representative",
    "/v1/retail_world/highest_paid_employee",
    "/v1/retail_world/product_count_by_supplier_stock_order?company_name=Plutzer%20Lebensmittelgromrkte%20AG&units_in_stock=0&units_on_order=0",
    "/v1/retail_world/product_name_quantity_by_unit_price?limit=1",
    "/v1/retail_world/employee_titles_by_order_count?limit=1",
    "/v1/retail_world/product_ids_by_customer_country?country=Germany&limit=1",
    "/v1/retail_world/total_order_value_by_shipper_and_country?company_name=Speedy%20Express&ship_country=Brazil",
    "/v1/retail_world/employee_count_and_order_value_by_manager?reports_to=2&limit=1",
    "/v1/retail_world/total_order_value_by_employee_title?title=Sales%20Representative",
    "/v1/retail_world/company_names_and_ship_countries?limit=1",
    "/v1/retail_world/customer_ids_and_category_names",
    "/v1/retail_world/count_product_unit_prices_by_category?category_name=Confections&limit=1",
    "/v1/retail_world/product_name_by_id?product_id=77",
    "/v1/retail_world/employee_names_by_company?company_name=Victuailles%20en%20stock",
    "/v1/retail_world/territory_percentage_comparison?region_description_1=Northern&region_description_2=Westerns",
    "/v1/retail_world/employee_ids_by_territory_range?min_territory_id=1000&max_territory_id=2000",
    "/v1/retail_world/region_territory_by_employee_name?last_name=Davolio&first_name=Nancy",
    "/v1/retail_world/order_count_by_employee_title_hire_year?title=Sales%20Representative&hire_year=1992",
    "/v1/retail_world/total_sales_amount_by_product?product_name=Vegie-spread",
    "/v1/retail_world/product_supplier_by_order_id?order_id=10337",
    "/v1/retail_world/top_shipper_by_year?shipped_year=1998",
    "/v1/retail_world/customer_count_by_city?city=London",
    "/v1/retail_world/employee_names_by_birthdate?birth_date=1955-03-04%2000:00:00",
    "/v1/retail_world/order_count_by_employee_city?city=Tacoma",
    "/v1/retail_world/employee_country_by_order_id?order_id=10257",
    "/v1/retail_world/employee_title_by_order_id?order_id=10257",
    "/v1/retail_world/customer_phone_by_order_id?order_id=10264",
    "/v1/retail_world/customer_region_by_order_id?order_id=10276",
    "/v1/retail_world/highest_avg_salary_by_ship_country?ship_country=Brazil",
    "/v1/retail_world/percentage_salary_by_year?year=1996",
    "/v1/retail_world/employee_hire_date_by_order_id?order_id=10281",
    "/v1/retail_world/order_count_by_reports_to?reports_to=5",
    "/v1/retail_world/shipper_phone_by_order_id?order_id=10260"
]
