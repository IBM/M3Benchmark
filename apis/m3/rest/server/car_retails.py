from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/car_retails/car_retails.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of customers grouped by country
@app.get("/v1/car_retails/customer_count_by_country", operation_id="get_customer_count_by_country", summary="Retrieves the total number of customers categorized by their respective countries. This operation provides a comprehensive overview of the customer distribution across different countries.")
async def get_customer_count_by_country():
    cursor.execute("SELECT country, COUNT(customerNumber) FROM customers GROUP BY country")
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": result}

# Endpoint to get the distinct product vendors and their price differences ordered by quantity ordered
@app.get("/v1/car_retails/product_vendor_price_difference", operation_id="get_product_vendor_price_difference", summary="Retrieves a list of unique product vendors along with their respective price differences, sorted by the total quantity ordered. The number of results can be limited using the input parameter.")
async def get_product_vendor_price_difference(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T1.productVendor, T1.MSRP - T1.buyPrice FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode GROUP BY T1.productVendor, T1.MSRP, T1.buyPrice ORDER BY COUNT(T2.quantityOrdered) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"vendors": []}
    return {"vendors": result}

# Endpoint to get the distinct last name and first name of employees based on payment year
@app.get("/v1/car_retails/employee_names_by_payment_year", operation_id="get_employee_names_by_payment_year", summary="Retrieves a list of distinct employee names, sorted by their total payment amounts in the specified year. The number of results returned can be limited.")
async def get_employee_names_by_payment_year(year: str = Query(..., description="Year in 'YYYY' format"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T3.lastName, T3.firstName FROM payments AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber INNER JOIN employees AS T3 ON T2.salesRepEmployeeNumber = T3.employeeNumber WHERE STRFTIME('%Y', T1.paymentDate) = ? ORDER BY T1.amount DESC LIMIT ?", (year, limit))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get the total price of ordered items for a specific customer, order status, and shipped year
@app.get("/v1/car_retails/total_price_ordered_items", operation_id="get_total_price_ordered_items", summary="Retrieves the total price of items ordered by a specific customer, based on the order status and the year the order was shipped. The calculation is done by multiplying the price of each item by its ordered quantity.")
async def get_total_price_ordered_items(customer_name: str = Query(..., description="Customer name"), status: str = Query(..., description="Order status"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T3.priceEach * T3.quantityOrdered FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customerNumber = T2.customerNumber INNER JOIN orderdetails AS T3 ON T2.orderNumber = T3.orderNumber WHERE T1.customerName = ? AND T2.status = ? AND STRFTIME('%Y', T2.shippedDate) = ?", (customer_name, status, year))
    result = cursor.fetchall()
    if not result:
        return {"total_prices": []}
    return {"total_prices": result}

# Endpoint to get the count of employees in a specific city
@app.get("/v1/car_retails/employee_count_by_city", operation_id="get_employee_count_by_city", summary="Retrieves the total number of employees working in a specified city. The city is identified by its name, which is provided as an input parameter.")
async def get_employee_count_by_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(employeeNumber) FROM employees WHERE officeCode = ( SELECT officeCode FROM offices WHERE city = ? )", (city,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the employee numbers based on reporting manager and city
@app.get("/v1/car_retails/employee_numbers_by_manager_city", operation_id="get_employee_numbers_by_manager_city", summary="Retrieves the number of employees who report to a specific manager in a given city. The manager is identified by their employee number, and the city is specified by name. This operation provides a count of employees under a manager in a particular location.")
async def get_employee_numbers_by_manager_city(reports_to: int = Query(..., description="Employee number of the manager"), city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.employeeNumber FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T1.reportsTo = ? AND T2.city = ?", (reports_to, city))
    result = cursor.fetchall()
    if not result:
        return {"employee_numbers": []}
    return {"employee_numbers": result}

# Endpoint to get payment statistics between two dates
@app.get("/v1/car_retails/payment_statistics", operation_id="get_payment_statistics", summary="Retrieves payment statistics for a specified date range. The operation calculates the average payment amount, maximum payment amount, and minimum payment amount for all payments made between the provided start and end dates. The results are based on the payments made by customers.")
async def get_payment_statistics(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(T1.amount) AS REAL) / 3, MAX(T1.amount) , MIN(T1.amount) FROM payments AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber WHERE T1.paymentDate BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"statistics": []}
    return {"statistics": result}

# Endpoint to get distinct customer addresses ordered by payment amount
@app.get("/v1/car_retails/customer_addresses_by_payment_amount", operation_id="get_customer_addresses_by_payment_amount", summary="Retrieves a list of unique customer addresses, sorted by the total payment amount in descending order. The number of results returned can be limited by specifying the 'limit' parameter.")
async def get_customer_addresses_by_payment_amount(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T2.country, T2.addressLine1, T2.addressLine2 FROM payments AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber ORDER BY T1.amount DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get the count of check numbers for a specific customer and payment year
@app.get("/v1/car_retails/check_number_count_by_customer_year", operation_id="get_check_number_count_by_customer_year", summary="Retrieves the total number of unique check numbers associated with a specific customer for a given payment year. This operation requires the customer's name and the desired year in 'YYYY' format as input parameters.")
async def get_check_number_count_by_customer_year(customer_name: str = Query(..., description="Customer name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.checkNumber) FROM payments AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber WHERE customerName = ? AND STRFTIME('%Y', T1.paymentDate) = ?", (customer_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the product names based on product scale, product line, and ordered quantity
@app.get("/v1/car_retails/product_names_by_scale_line_quantity", operation_id="get_product_names_by_scale_line_quantity", summary="Retrieves a list of product names that match the specified product scale and line, sorted by the total quantity ordered in descending order. The number of results returned can be limited by the provided limit parameter.")
async def get_product_names_by_scale_line_quantity(product_scale: str = Query(..., description="Product scale"), product_line: str = Query(..., description="Product line"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.productName FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode WHERE T1.productScale = ? AND T1.productLine = ? GROUP BY T1.productName ORDER BY SUM(T2.quantityOrdered) DESC LIMIT ?", (product_scale, product_line, limit))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": result}

# Endpoint to get the customer with the highest payment amount in a specific year and credit limit
@app.get("/v1/car_retails/top_customer_by_payment", operation_id="get_top_customer_by_payment", summary="Get the customer with the highest payment amount in a specific year and credit limit")
async def get_top_customer_by_payment(credit_limit: int = Query(..., description="Credit limit of the customer"), year: str = Query(..., description="Year of the payment in 'YYYY' format")):
    cursor.execute("SELECT ( SELECT COUNT(customerNumber) FROM customers WHERE creditLimit <= ? AND customerNumber IN ( SELECT customerNumber FROM payments WHERE STRFTIME('%Y', paymentDate) = ? ) ), T1.customerName FROM customers AS T1 INNER JOIN payments AS T2 ON T1.customerNumber = T2.customerNumber WHERE T1.creditLimit <= ? AND STRFTIME('%Y', T2.paymentDate) = ? GROUP BY T1.customerNumber, T1.customerName ORDER BY SUM(T2.amount) DESC LIMIT 1", (credit_limit, year, credit_limit, year))
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": result[1]}

# Endpoint to get the top customer by payment amount for a specific city and reporting employee
@app.get("/v1/car_retails/top_customer_by_city_and_employee", operation_id="get_top_customer_by_city_and_employee", summary="Get the top customer by payment amount for a specific city and reporting employee")
async def get_top_customer_by_city_and_employee(city: str = Query(..., description="City of the office"), reports_to: int = Query(..., description="Employee number to whom the sales representative reports")):
    cursor.execute("SELECT T2.customerName, T2.contactFirstName, T2.contactLastName, SUM(T3.amount) FROM employees AS T1 INNER JOIN customers AS T2 ON T2.salesRepEmployeeNumber = T1.employeeNumber INNER JOIN payments AS T3 ON T2.customerNumber = T3.customerNumber INNER JOIN offices AS T4 ON T1.officeCode = T4.officeCode WHERE T4.city = ? AND T1.reportsTo = ? GROUP BY T2.customerName, T2.contactFirstName, T2.contactLastName ORDER BY amount DESC LIMIT 1", (city, reports_to))
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": result[0]}

# Endpoint to get the total revenue for the most and least profitable products
@app.get("/v1/car_retails/revenue_most_least_profitable_products", operation_id="get_revenue_most_least_profitable_products", summary="Retrieves the total revenue generated by the most and least profitable products. The profitability is determined by the difference between the manufacturer's suggested retail price (MSRP) and the buy price. The total revenue is calculated by summing the product of the quantity ordered and the price each for each product.")
async def get_revenue_most_least_profitable_products():
    cursor.execute("SELECT T2.productName, SUM(T1.quantityOrdered * T1.priceEach) FROM orderdetails AS T1 INNER JOIN ( SELECT productCode, productName FROM products ORDER BY MSRP - buyPrice DESC LIMIT 1 ) AS T2 ON T1.productCode = T2.productCode UNION SELECT T2.productName, SUM(quantityOrdered * priceEach) FROM orderdetails AS T1 INNER JOIN ( SELECT productCode, productName FROM products ORDER BY MSRP - buyPrice ASC LIMIT 1 ) AS T2 ON T1.productCode = T2.productCode")
    result = cursor.fetchall()
    if not result:
        return {"revenue": []}
    return {"revenue": result}

# Endpoint to get the count of distinct orders with a total order value greater than a specified amount and a specific status
@app.get("/v1/car_retails/count_orders_by_value_and_status", operation_id="get_count_orders_by_value_and_status", summary="Retrieves the count of unique orders that surpass a given total value and have a specific status. The order value is determined by multiplying the quantity ordered by the price per unit. The order status is also considered in the count.")
async def get_count_orders_by_value_and_status(order_value: float = Query(..., description="Total order value"), status: str = Query(..., description="Status of the order")):
    cursor.execute("SELECT COUNT(DISTINCT T1.orderNumber) FROM orderdetails AS T1 INNER JOIN orders AS T2 ON T1.orderNumber = T2.orderNumber WHERE T1.quantityOrdered * T1.priceEach > ? AND T2.status = ?", (order_value, status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct orders with a quantity ordered less than a specified amount and a specific order year
@app.get("/v1/car_retails/count_orders_by_quantity_and_year", operation_id="get_count_orders_by_quantity_and_year", summary="Retrieves the count of unique orders that have a quantity less than the specified amount and were placed in a particular year. The input parameters include the maximum quantity and the year of the order.")
async def get_count_orders_by_quantity_and_year(quantity_ordered: int = Query(..., description="Quantity ordered"), year: str = Query(..., description="Year of the order in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.orderNumber) FROM orderdetails AS T1 INNER JOIN orders AS T2 ON T1.orderNumber = T2.orderNumber WHERE T1.quantityOrdered < ? AND STRFTIME('%Y', T2.orderDate) = ?", (quantity_ordered, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total revenue for orders with a specific status
@app.get("/v1/car_retails/total_revenue_by_status", operation_id="get_total_revenue_by_status", summary="Retrieves the total revenue generated from orders with a specified status. The calculation is based on the quantity ordered and the price per unit for each order detail, summed up for orders with the given status.")
async def get_total_revenue_by_status(status: str = Query(..., description="Status of the order")):
    cursor.execute("SELECT SUM(T1.quantityOrdered * T1.priceEach) FROM orderdetails AS T1 INNER JOIN orders AS T2 ON T1.orderNumber = T2.orderNumber WHERE T2.status = ?", (status,))
    result = cursor.fetchone()
    if not result:
        return {"revenue": []}
    return {"revenue": result[0]}

# Endpoint to get the total revenue for a specific product line
@app.get("/v1/car_retails/total_revenue_by_product_line", operation_id="get_total_revenue_by_product_line", summary="Retrieves the total revenue generated by a specific product line. This operation calculates the sum of the total revenue for each product within the specified product line. The product line is identified by the provided input parameter.")
async def get_total_revenue_by_product_line(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT SUM(T1.quantityOrdered * T1.priceEach) FROM orderdetails AS T1 INNER JOIN products AS T2 ON T1.productCode = T2.productCode WHERE T2.productLine = ?", (product_line,))
    result = cursor.fetchone()
    if not result:
        return {"revenue": []}
    return {"revenue": result[0]}

# Endpoint to get the count of products in a specific product line
@app.get("/v1/car_retails/count_products_by_product_line", operation_id="get_count_products_by_product_line", summary="Retrieves the total number of products in a specified product line. The product line is identified by the provided input parameter.")
async def get_count_products_by_product_line(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT COUNT(T1.productCode) FROM orderdetails AS T1 INNER JOIN products AS T2 ON T1.productCode = T2.productCode WHERE T2.productLine = ?", (product_line,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of products with a profit margin greater than a specified amount
@app.get("/v1/car_retails/count_products_by_profit_margin", operation_id="get_count_products_by_profit_margin", summary="Retrieves the total number of products that have a profit margin exceeding the provided value. The profit margin is calculated as the difference between the manufacturer's suggested retail price (MSRP) and the buy price.")
async def get_count_products_by_profit_margin(profit_margin: float = Query(..., description="Profit margin")):
    cursor.execute("SELECT COUNT(T1.productCode) FROM orderdetails AS T1 INNER JOIN products AS T2 ON T1.productCode = T2.productCode WHERE T2.MSRP - T2.buyPrice > ?", (profit_margin,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average revenue per order for a specific product vendor and year
@app.get("/v1/car_retails/average_revenue_per_order", operation_id="get_average_revenue_per_order", summary="Retrieves the average revenue per order for a specific product vendor in a given year. This operation calculates the total revenue generated by the vendor's products and divides it by the total number of orders placed in the specified year. The input parameters include the product vendor's name and the year of the order in 'YYYY' format.")
async def get_average_revenue_per_order(product_vendor: str = Query(..., description="Product vendor"), year: str = Query(..., description="Year of the order in 'YYYY' format")):
    cursor.execute("SELECT SUM(T2.quantityOrdered * T2.priceEach) / COUNT(T3.orderNumber) FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode INNER JOIN orders AS T3 ON T2.orderNumber = T3.orderNumber WHERE T1.productVendor = ? AND STRFTIME('%Y', T3.orderDate) = ?", (product_vendor, year))
    result = cursor.fetchone()
    if not result:
        return {"average_revenue": []}
    return {"average_revenue": result[0]}

# Endpoint to get the count of employees based on job title and office code
@app.get("/v1/car_retails/employee_count_by_job_title_office_code", operation_id="get_employee_count", summary="Retrieves the total number of employees with a specified job title and office code. The response is based on the provided job title and office code parameters.")
async def get_employee_count(job_title: str = Query(..., description="Job title of the employee"), office_code: int = Query(..., description="Office code of the employee")):
    cursor.execute("SELECT COUNT(officeCode) FROM employees WHERE jobTitle = ? AND officeCode = ?", (job_title, office_code))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct customer names based on payment amount
@app.get("/v1/car_retails/distinct_customer_names_by_payment_amount", operation_id="get_distinct_customer_names", summary="Retrieves a list of unique customer names who have made payments exceeding a specified amount threshold. This operation is useful for identifying customers with significant spending patterns.")
async def get_distinct_customer_names(amount: float = Query(..., description="Payment amount threshold")):
    cursor.execute("SELECT DISTINCT T2.customerName FROM payments AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber WHERE T1.amount > ?", (amount,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the total payment amount for customers from a specific country
@app.get("/v1/car_retails/total_payment_amount_by_country", operation_id="get_total_payment_amount", summary="Retrieves the total payment amount made by customers from a specified country. The operation calculates the sum of all payments made by customers residing in the provided country.")
async def get_total_payment_amount(country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT SUM(T1.amount) FROM payments AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber WHERE T2.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the total payment amount for a specific year and credit limit
@app.get("/v1/car_retails/total_payment_amount_by_year_credit_limit", operation_id="get_total_payment_amount_by_year", summary="Retrieves the total payment amount for a given year and credit limit. This operation calculates the sum of all payments made by customers with a specific credit limit in the specified year. The year should be provided in 'YYYY' format, and the credit limit should correspond to the customer's credit limit.")
async def get_total_payment_amount_by_year(year: str = Query(..., description="Year in 'YYYY' format"), credit_limit: float = Query(..., description="Credit limit of the customer")):
    cursor.execute("SELECT SUM(amount) FROM payments WHERE STRFTIME('%Y', paymentDate) = ? AND customerNumber IN ( SELECT customerNumber FROM customers WHERE creditLimit = ? )", (year, credit_limit))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get customer names and phone numbers based on order status
@app.get("/v1/car_retails/customer_info_by_order_status", operation_id="get_customer_info", summary="Retrieves the names and phone numbers of customers who have orders with a specified status. The status parameter is used to filter the results.")
async def get_customer_info(status: str = Query(..., description="Status of the order")):
    cursor.execute("SELECT T2.customerName, T2.phone FROM orders AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber WHERE T1.status = ?", (status,))
    result = cursor.fetchall()
    if not result:
        return {"customer_info": []}
    return {"customer_info": [{"name": row[0], "phone": row[1]} for row in result]}

# Endpoint to get the count of countries for orders based on status and country
@app.get("/v1/car_retails/count_countries_by_order_status_country", operation_id="get_count_countries", summary="Get the count of countries for orders based on status and country")
async def get_count_countries(status: str = Query(..., description="Status of the order"), country: str = Query(..., description="Country of the customer"), count_status: int = Query(..., description="Count of statuses")):
    cursor.execute("SELECT COUNT(T2.country) FROM orders AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber WHERE T1.status = ? AND T2.country = ? GROUP BY T2.customerNumber HAVING COUNT(T1.status) = ?", (status, country, count_status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average order value for a specific status and country
@app.get("/v1/car_retails/average_order_value_by_status_country", operation_id="get_average_order_value", summary="Retrieves the average order value for a specific order status and customer country. This operation calculates the total order value by summing the product of quantity ordered and price per unit, then dividing by the total number of orders for the given order status and customer country.")
async def get_average_order_value(status: str = Query(..., description="Status of the order"), country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT SUM(T3.quantityOrdered * T3.priceEach) / COUNT(T2.orderNumber) FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customerNumber = T2.customerNumber INNER JOIN orderdetails AS T3 ON T2.orderNumber = T3.orderNumber WHERE T2.status = ? AND T1.country = ?", (status, country))
    result = cursor.fetchone()
    if not result:
        return {"average_value": []}
    return {"average_value": result[0]}

# Endpoint to get employee names based on office city
@app.get("/v1/car_retails/employee_names_by_office_city", operation_id="get_employee_names", summary="Retrieves the first and last names of employees working in the specified office city. The city is provided as an input parameter.")
async def get_employee_names(city: str = Query(..., description="City of the office")):
    cursor.execute("SELECT T1.firstName, T1.lastName FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T2.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get employee details based on office city and job title
@app.get("/v1/car_retails/employee_details_by_office_city_job_title", operation_id="get_employee_details", summary="Retrieves the first name, last name, and email of employees who work in a specific office city and hold a particular job title. The city and job title are provided as input parameters.")
async def get_employee_details(city: str = Query(..., description="City of the office"), job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT T1.firstName, T1.lastName, T1.email FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T2.city = ? AND T1.jobTitle = ?", (city, job_title))
    result = cursor.fetchall()
    if not result:
        return {"employee_details": []}
    return {"employee_details": [{"first_name": row[0], "last_name": row[1], "email": row[2]} for row in result]}

# Endpoint to get employee emails based on reporting manager and office city
@app.get("/v1/car_retails/employee_emails_by_reports_to_city", operation_id="get_employee_emails", summary="Retrieves the email addresses of employees who report to a specific manager and work in a particular city. The operation requires the manager's employee ID and the desired city as input parameters.")
async def get_employee_emails(reports_to: int = Query(..., description="Employee ID of the reporting manager"), city: str = Query(..., description="City of the office")):
    cursor.execute("SELECT T1.email FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T1.reportsTo = ? AND T2.city = ?", (reports_to, city))
    result = cursor.fetchall()
    if not result:
        return {"employee_emails": []}
    return {"employee_emails": [row[0] for row in result]}

# Endpoint to get employee emails and numbers based on office state and country
@app.get("/v1/car_retails/employee_emails_numbers_by_office_state_country", operation_id="get_employee_emails_numbers", summary="Retrieves the email addresses and employee numbers of employees working in offices located in a specific state and country. The operation filters the employee data based on the provided state and country parameters.")
async def get_employee_emails_numbers(state: str = Query(..., description="State of the office"), country: str = Query(..., description="Country of the office")):
    cursor.execute("SELECT T1.email, T1.employeeNumber FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T2.state = ? AND T2.country = ?", (state, country))
    result = cursor.fetchall()
    if not result:
        return {"emails_numbers": []}
    return {"emails_numbers": result}

# Endpoint to get employee emails and office countries based on customer credit limit and job title
@app.get("/v1/car_retails/employee_emails_office_countries_by_credit_limit_job_title", operation_id="get_employee_emails_office_countries", summary="Retrieves the email addresses of employees and their respective office countries, filtered by a specified country, customer credit limit, and employee job title. This operation is useful for identifying employees who handle customers with specific credit limits and job titles in a particular country.")
async def get_employee_emails_office_countries(country: str = Query(..., description="Country of the office"), credit_limit: int = Query(..., description="Credit limit of the customer"), job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT T2.email, T3.country FROM customers AS T1 INNER JOIN employees AS T2 ON T1.salesRepEmployeeNumber = T2.employeeNumber INNER JOIN offices AS T3 ON T2.officeCode = T3.officeCode WHERE T3.country = ? AND T1.creditLimit < ? AND T2.jobTitle = ?", (country, credit_limit, job_title))
    result = cursor.fetchall()
    if not result:
        return {"emails_countries": []}
    return {"emails_countries": result}

# Endpoint to get distinct employee emails based on job title and customer country
@app.get("/v1/car_retails/distinct_employee_emails_by_job_title_country", operation_id="get_distinct_employee_emails", summary="Retrieves a list of unique email addresses of employees who have a specific job title and serve customers in a particular country. The list is ordered by the customers' credit limit and limited to the top 10 results.")
async def get_distinct_employee_emails(job_title: str = Query(..., description="Job title of the employee"), country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT DISTINCT T2.email FROM customers AS T1 INNER JOIN employees AS T2 ON T1.salesRepEmployeeNumber = T2.employeeNumber WHERE T2.jobTitle = ? AND T1.country = ? ORDER BY T1.creditLimit LIMIT 10", (job_title, country))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": result}

# Endpoint to get the count of customers based on credit limit and country
@app.get("/v1/car_retails/customer_count_by_credit_limit_country", operation_id="get_customer_count", summary="Retrieves the number of customers who have a credit limit below the specified value and reside in the provided country.")
async def get_customer_count(credit_limit: int = Query(..., description="Credit limit of the customer"), country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT COUNT(creditLimit) FROM customers WHERE creditLimit < ? AND country = ?", (credit_limit, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get top 3 customer numbers based on payment amount
@app.get("/v1/car_retails/top_customer_numbers_by_payment_amount", operation_id="get_top_customer_numbers", summary="Retrieves the top three customer numbers that have made the highest payments. The data is sorted in descending order based on the payment amount.")
async def get_top_customer_numbers():
    cursor.execute("SELECT customerNumber FROM payments ORDER BY amount DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"customer_numbers": []}
    return {"customer_numbers": result}

# Endpoint to get top 3 employee emails based on customer credit limit and job title
@app.get("/v1/car_retails/top_employee_emails_by_credit_limit_job_title", operation_id="get_top_employee_emails", summary="Retrieves the top 3 employee emails who have the highest total customer credit limit for a specific job title and country. The employees are ranked based on the sum of their customers' credit limits.")
async def get_top_employee_emails(job_title: str = Query(..., description="Job title of the employee"), country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT T2.email FROM customers AS T1 INNER JOIN employees AS T2 ON T1.salesRepEmployeeNumber = T2.employeeNumber WHERE T2.jobTitle = ? AND T1.country = ? GROUP BY T1.customerName, T2.email ORDER BY SUM(T1.creditLimit) DESC LIMIT 3", (job_title, country))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": result}

# Endpoint to get employee details based on office city
@app.get("/v1/car_retails/employee_details_by_city", operation_id="get_employee_details_by_city", summary="Retrieves the first name, last name, and email of employees working in the specified city. The city is determined by the office location of the employees.")
async def get_employee_details_by_city(city: str = Query(..., description="City of the office")):
    cursor.execute("SELECT T1.firstName, T1.lastName, T1.email FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T2.city = ?", (city,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": result}

# Endpoint to get top 5 products by stock quantity in a specific product line
@app.get("/v1/car_retails/top_products_by_stock", operation_id="get_top_products_by_stock", summary="Retrieves the top 5 products with the highest stock quantities from a specified product line. The product line is determined by the provided input parameter.")
async def get_top_products_by_stock(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT productCode, quantityInStock FROM products WHERE productLine = ? ORDER BY quantityInStock DESC LIMIT 5", (product_line,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the count of customers with payments in a specific year range and more than a certain number of payments
@app.get("/v1/car_retails/customer_count_by_payment_year_range", operation_id="get_customer_count_by_payment_year_range", summary="Retrieves the count of unique customers who made payments within a specified year range and had more than a certain number of payments. The year range is defined by the start and end year, and the minimum number of payments is also provided as input.")
async def get_customer_count_by_payment_year_range(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), min_payments: int = Query(..., description="Minimum number of payments")):
    cursor.execute("SELECT COUNT(customernumber) FROM ( SELECT customernumber FROM payments WHERE STRFTIME('%Y', paymentDate) >= ? AND STRFTIME('%Y', paymentDate) <= ? GROUP BY customernumber HAVING COUNT(customernumber) > ? ) T", (start_year, end_year, min_payments))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average profit margin for a specific product
@app.get("/v1/car_retails/average_profit_margin", operation_id="get_average_profit_margin", summary="Retrieves the average profit margin for a specific product by calculating the sum of the difference between the selling price and the purchase price for each order detail, then dividing by the total number of orders for that product.")
async def get_average_profit_margin(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(T1.priceEach - T2.buyPrice) / COUNT(*) FROM orderdetails AS T1 INNER JOIN products AS T2 ON T1.productCode = T2.productCode WHERE T2.productName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_profit_margin": []}
    return {"average_profit_margin": result[0]}

# Endpoint to get the most ordered product name in a specific product line, scale, and country
@app.get("/v1/car_retails/most_ordered_product", operation_id="get_most_ordered_product", summary="Retrieves the name of the product that has been ordered the most in a specific product line, scale, and country. The product line, scale, and customer's country are used to filter the results. The product with the highest total quantity ordered is returned.")
async def get_most_ordered_product(product_line: str = Query(..., description="Product line"), product_scale: str = Query(..., description="Product scale"), country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT T1.productName FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode INNER JOIN orders AS T3 ON T2.orderNumber = T3.orderNumber INNER JOIN customers AS T4 ON T3.customerNumber = T4.customerNumber WHERE T1.productLine = ? AND T1.productScale = ? AND T4.country = ? GROUP BY T1.productName ORDER BY SUM(T2.quantityOrdered) DESC LIMIT 1", (product_line, product_scale, country))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get check numbers for payments made by a specific customer within a date range
@app.get("/v1/car_retails/check_numbers_by_customer_date_range", operation_id="get_check_numbers_by_customer_date_range", summary="Retrieves the check numbers associated with payments made by a specific customer within a given date range. The operation requires the customer's name and the start and end dates of the range, both in 'YYYY-MM-DD' format. The response includes a list of check numbers corresponding to the payments made by the specified customer during the provided date range.")
async def get_check_numbers_by_customer_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT T1.checkNumber FROM payments AS T1 INNER JOIN customers AS T2 ON T1.customerNumber = T2.customerNumber WHERE T1.paymentDate >= ? AND T1.paymentDate <= ? AND T2.customerName = ?", (start_date, end_date, customer_name))
    result = cursor.fetchall()
    if not result:
        return {"check_numbers": []}
    return {"check_numbers": result}

# Endpoint to get the maximum profit margin for a specific product line
@app.get("/v1/car_retails/max_profit_margin_by_product_line", operation_id="get_max_profit_margin_by_product_line", summary="Retrieves the maximum profit margin for a given product line. The profit margin is calculated by subtracting the average unit price from the total quantity ordered multiplied by the unit price. The product line is specified as an input parameter.")
async def get_max_profit_margin_by_product_line(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT MAX(quantityOrdered * priceEach) - AVG(priceEach) FROM orderdetails WHERE productCode IN ( SELECT productCode FROM products WHERE productLine = ? )", (product_line,))
    result = cursor.fetchone()
    if not result:
        return {"max_profit_margin": []}
    return {"max_profit_margin": result[0]}

# Endpoint to get the total sales amount for shipped orders within a date range
@app.get("/v1/car_retails/total_sales_amount_shipped_orders", operation_id="get_total_sales_amount_shipped_orders", summary="Retrieves the total sales amount for shipped orders within a specified date range. The operation considers the order status and calculates the sum of the product price multiplied by the quantity ordered for each order that meets the criteria. The date range is inclusive and must be provided in 'YYYY-MM-DD' format.")
async def get_total_sales_amount_shipped_orders(status: str = Query(..., description="Order status"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(T2.priceEach * T2.quantityOrdered) FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode INNER JOIN orders AS T3 ON T2.orderNumber = T3.orderNumber WHERE T3.status = ? AND T3.orderDate BETWEEN ? AND ?", (status, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"total_sales_amount": []}
    return {"total_sales_amount": result[0]}

# Endpoint to get the top employee by payment amount
@app.get("/v1/car_retails/top_employee_by_payment_amount", operation_id="get_top_employee_by_payment_amount", summary="Retrieves the employee with the highest total payment amount from sales. This operation identifies the top-performing employee based on the total payments received from their associated customers. The result includes the employee's first name, last name, and their direct supervisor.")
async def get_top_employee_by_payment_amount():
    cursor.execute("SELECT T1.firstName, T1.lastName, T1.reportsTo FROM employees AS T1 INNER JOIN customers AS T2 ON T1.employeeNumber = T2.salesRepEmployeeNumber INNER JOIN payments AS T3 ON T2.customerNumber = T3.customerNumber ORDER BY T3.amount DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"employee": []}
    return {"employee": result}

# Endpoint to get the product name and total price for the highest quantity ordered in a specific city by a sales rep
@app.get("/v1/car_retails/product_highest_quantity_ordered", operation_id="get_product_highest_quantity_ordered", summary="Retrieves the name and total price of the product with the highest quantity ordered by a sales representative in a specific city. The city and job title of the sales representative are used to filter the results.")
async def get_product_highest_quantity_ordered(city: str = Query(..., description="City of the customer"), job_title: str = Query(..., description="Job title of the sales rep")):
    cursor.execute("SELECT T2.productName, T1.quantityOrdered * T1.priceEach FROM orderdetails AS T1 INNER JOIN products AS T2 ON T1.productCode = T2.productCode INNER JOIN orders AS T3 ON T1.orderNumber = T3.orderNumber INNER JOIN customers AS T4 ON T3.customerNumber = T4.customerNumber WHERE T4.city = ? AND T4.salesRepEmployeeNumber IN ( SELECT employeeNumber FROM employees WHERE jobTitle = ? ) ORDER BY T1.quantityOrdered DESC LIMIT 1", (city, job_title))
    result = cursor.fetchone()
    if not result:
        return {"product_name": [], "total_price": []}
    return {"product_name": result[0], "total_price": result[1]}

# Endpoint to get the total profit from orders within a specific date range
@app.get("/v1/car_retails/total_profit_date_range", operation_id="get_total_profit_date_range", summary="Retrieves the total profit generated from car retail orders placed within a specified date range. The calculation considers the selling price of each product and its corresponding purchase price. The date range is defined by the provided start and end dates.")
async def get_total_profit_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(T2.priceEach - T1.buyPrice) FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode INNER JOIN orders AS T3 ON T2.orderNumber = T3.orderNumber INNER JOIN customers AS T4 ON T3.customerNumber = T4.customerNumber WHERE T3.orderDate > ? AND T3.orderDate < ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"total_profit": []}
    return {"total_profit": result[0]}

# Endpoint to get the address and phone of the customer with the highest quantity ordered in a specific year by a sales rep
@app.get("/v1/car_retails/customer_highest_quantity_ordered", operation_id="get_customer_highest_quantity_ordered", summary="Retrieves the address and phone number of the customer who ordered the highest quantity of items in a specific year, as reported by a sales representative with a given job title. The response is based on the order details, orders, customers, employees, and offices tables.")
async def get_customer_highest_quantity_ordered(year: str = Query(..., description="Year in 'YYYY' format"), job_title: str = Query(..., description="Job title of the sales rep")):
    cursor.execute("SELECT T3.addressLine1, T3.addressLine2, T3.phone FROM orderdetails AS T1 INNER JOIN orders AS T2 ON T1.orderNumber = T2.orderNumber INNER JOIN customers AS T3 ON T2.customerNumber = T3.customerNumber INNER JOIN employees AS T4 ON T3.salesRepEmployeeNumber = T4.employeeNumber INNER JOIN offices AS T5 ON T4.officeCode = T5.officeCode WHERE STRFTIME('%Y', T2.orderDate) = ? AND T4.jobTitle = ? ORDER BY T1.quantityOrdered DESC LIMIT 1", (year, job_title))
    result = cursor.fetchone()
    if not result:
        return {"address_line1": [], "address_line2": [], "phone": []}
    return {"address_line1": result[0], "address_line2": result[1], "phone": result[2]}

# Endpoint to get the phone numbers of customers with a specific last name pattern and not from a specific country
@app.get("/v1/car_retails/customer_phones_lastname_country", operation_id="get_customer_phones_lastname_country", summary="Retrieves the phone numbers of customers whose last names match the provided pattern and who are not from the specified country.")
async def get_customer_phones_lastname_country(last_name_pattern: str = Query(..., description="Last name pattern (e.g., 'M%')"), country: str = Query(..., description="Country to exclude")):
    cursor.execute("SELECT phone FROM customers WHERE contactLastName LIKE ? AND country != ?", (last_name_pattern, country))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": [row[0] for row in result]}

# Endpoint to get the average payment amount within a specific date range
@app.get("/v1/car_retails/average_payment_date_range", operation_id="get_average_payment_date_range", summary="Retrieves the average payment amount for a specified date range. The operation calculates the average payment amount from the payments table, considering only those payments made between the provided start and end dates.")
async def get_average_payment_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT AVG(amount) FROM payments WHERE paymentDate BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_payment": []}
    return {"average_payment": result[0]}

# Endpoint to get the percentage of orders shipped to a specific customer within a date range
@app.get("/v1/car_retails/percentage_orders_shipped_customer", operation_id="get_percentage_orders_shipped_customer", summary="Retrieves the percentage of orders shipped to a specific customer within a given date range and order status. The calculation is based on the total number of orders that meet the specified criteria.")
async def get_percentage_orders_shipped_customer(customer_number: int = Query(..., description="Customer number"), status: str = Query(..., description="Order status"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN customerNumber = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(orderNumber) FROM orders WHERE status = ? AND shippedDate BETWEEN ? AND ?", (customer_number, status, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of customers with cancelled orders and a credit limit above a specific value
@app.get("/v1/car_retails/count_customers_cancelled_orders_credit_limit", operation_id="get_count_customers_cancelled_orders_credit_limit", summary="Retrieves the number of customers who have cancelled orders and a credit limit exceeding a specified value. The operation filters customers based on the provided order status and credit limit.")
async def get_count_customers_cancelled_orders_credit_limit(status: str = Query(..., description="Order status"), credit_limit: float = Query(..., description="Credit limit")):
    cursor.execute("SELECT COUNT(T1.customerNumber) FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customerNumber = T2.customerNumber WHERE T2.status = ? AND T1.creditLimit > ?", (status, credit_limit))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the earliest payment date for a customer of a specific sales rep
@app.get("/v1/car_retails/earliest_payment_date_sales_rep", operation_id="get_earliest_payment_date_sales_rep", summary="Retrieves the earliest payment date for a customer associated with a specific sales representative, sorted by the customer's credit limit in ascending order. The sales representative is identified by their first name, last name, and job title.")
async def get_earliest_payment_date_sales_rep(first_name: str = Query(..., description="First name of the sales rep"), last_name: str = Query(..., description="Last name of the sales rep"), job_title: str = Query(..., description="Job title of the sales rep")):
    cursor.execute("SELECT T3.paymentDate FROM employees AS T1 INNER JOIN customers AS T2 ON T1.employeeNumber = T2.salesRepEmployeeNumber INNER JOIN payments AS T3 ON T2.customerNumber = T3.customerNumber WHERE T1.firstName = ? AND T1.lastName = ? AND T1.jobTitle = ? ORDER BY T2.creditLimit ASC LIMIT 1", (first_name, last_name, job_title))
    result = cursor.fetchone()
    if not result:
        return {"payment_date": []}
    return {"payment_date": result[0]}

# Endpoint to get the employee numbers of sales reps for customers in a specific country
@app.get("/v1/car_retails/sales_rep_employee_numbers_country", operation_id="get_sales_rep_employee_numbers_country", summary="Retrieves the employee numbers of sales representatives who serve customers in a specified country. The input parameter is the country of the customer.")
async def get_sales_rep_employee_numbers_country(country: str = Query(..., description="Country of the customer")):
    cursor.execute("SELECT T1.reportsTo FROM employees AS T1 INNER JOIN customers AS T2 ON T1.employeeNumber = T2.salesRepEmployeeNumber WHERE T2.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"employee_numbers": []}
    return {"employee_numbers": [row[0] for row in result]}

# Endpoint to get the address of customers with orders shipped on a specific date and status
@app.get("/v1/car_retails/customer_address_shipped_date_status", operation_id="get_customer_address_shipped_date_status", summary="Retrieves the addresses of customers who have orders shipped on a specific date and with a particular status. The operation filters orders based on the provided shipped date and status, and returns the corresponding customer addresses.")
async def get_customer_address_shipped_date_status(shipped_date: str = Query(..., description="Shipped date in 'YYYY-MM-DD' format"), status: str = Query(..., description="Order status")):
    cursor.execute("SELECT T1.addressLine1, T1.addressLine2 FROM customers AS T1 INNER JOIN orders AS T2 ON T1.customerNumber = T2.customerNumber WHERE T2.shippedDate = ? AND T2.status = ?", (shipped_date, status))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [{"address_line1": row[0], "address_line2": row[1]} for row in result]}

# Endpoint to get customer addresses based on city and job title
@app.get("/v1/car_retails/customer_addresses_by_city_job_title", operation_id="get_customer_addresses", summary="Retrieves the addresses of customers who reside in a specific city and are associated with employees holding a particular job title. The operation filters customers based on the provided city and job title, and returns the first and second lines of their addresses.")
async def get_customer_addresses(city: str = Query(..., description="City of the customer"), job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT T2.addressLine1, T2.addressLine2 FROM employees AS T1 INNER JOIN customers AS T2 ON T1.employeeNumber = T2.salesRepEmployeeNumber INNER JOIN offices AS T3 ON T1.officeCode = T3.officeCode WHERE T2.city = ? AND T1.jobTitle = ?", (city, job_title))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get customer addresses based on job title
@app.get("/v1/car_retails/customer_addresses_by_job_title", operation_id="get_customer_addresses_by_job_title", summary="Retrieves the addresses of customers who are associated with employees of a specific job title. The job title is provided as an input parameter.")
async def get_customer_addresses_by_job_title(job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT T1.addressLine1, T1.addressLine2 FROM customers AS T1 INNER JOIN employees AS T2 ON T1.salesRepEmployeeNumber = T2.employeeNumber WHERE T2.jobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get the total profit margin for products from a specific vendor and description
@app.get("/v1/car_retails/total_profit_margin_by_vendor_description", operation_id="get_total_profit_margin", summary="Retrieves the total profit margin for products from a specified vendor that match a given description. The profit margin is calculated as the sum of the difference between the manufacturer's suggested retail price (MSRP) and the buy price for each product. The vendor and description are provided as input parameters.")
async def get_total_profit_margin(product_vendor: str = Query(..., description="Product vendor"), text_description: str = Query(..., description="Text description containing specific keywords")):
    cursor.execute("SELECT SUM(T2.MSRP - T2.buyPrice) FROM productlines AS T1 INNER JOIN products AS T2 ON T1.productLine = T2.productLine WHERE T2.productVendor = ? AND T1.textDescription LIKE ?", (product_vendor, text_description))
    result = cursor.fetchone()
    if not result:
        return {"total_profit_margin": []}
    return {"total_profit_margin": result[0]}

# Endpoint to get the customer number with the highest average payment amount in a specific city
@app.get("/v1/car_retails/top_customer_by_average_payment", operation_id="get_top_customer_by_average_payment", summary="Retrieves the customer number of the individual with the highest average payment in the specified city. This operation calculates the average payment amount for each customer in the given city and returns the customer number with the highest average. The city is a required input parameter.")
async def get_top_customer_by_average_payment(city: str = Query(..., description="City of the customer")):
    cursor.execute("SELECT T1.customerNumber FROM customers AS T1 INNER JOIN payments AS T2 ON T1.customerNumber = T2.customerNumber WHERE T1.city = ? GROUP BY T1.customerNumber ORDER BY SUM(T2.amount) / COUNT(T2.paymentDate) DESC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"customer_number": []}
    return {"customer_number": result[0]}

# Endpoint to get the total quantity ordered and average price for a specific product
@app.get("/v1/car_retails/total_quantity_average_price_by_product", operation_id="get_total_quantity_average_price", summary="Retrieves the total quantity ordered and the average price for a specific product. The product is identified by its name, which is provided as an input parameter. The operation calculates the total quantity ordered by summing up the quantityOrdered field from the orderdetails table. It also calculates the average price by dividing the sum of the product of quantityOrdered and priceEach by the total quantity ordered. The products and orderdetails tables are joined on the productCode field.")
async def get_total_quantity_average_price(product_name: str = Query(..., description="Name of the product")):
    cursor.execute("SELECT SUM(T2.quantityOrdered), SUM(T2.quantityOrdered * T2.priceEach) / SUM(T2.quantityOrdered) FROM products AS T1 INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode WHERE T1.productName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": [], "average_price": []}
    return {"total_quantity": result[0], "average_price": result[1]}

# Endpoint to get the count of product codes for a specific order number
@app.get("/v1/car_retails/count_product_codes_by_order_number", operation_id="get_count_product_codes", summary="Retrieves the total number of unique product codes associated with a specific order. The order is identified by its unique order number.")
async def get_count_product_codes(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT COUNT(t.productCode) FROM orderdetails t WHERE t.orderNumber = ?", (order_number,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get employee names based on customer address
@app.get("/v1/car_retails/employee_names_by_customer_address", operation_id="get_employee_names_by_address", summary="Retrieves the first and last names of employees who have sold to customers residing at a specific address. The address is determined by the provided first and second lines of the address.")
async def get_employee_names_by_address(address_line1: str = Query(..., description="First line of the address"), address_line2: str = Query(..., description="Second line of the address")):
    cursor.execute("SELECT T2.firstName, T2.lastName FROM customers AS T1 INNER JOIN employees AS T2 ON T1.salesRepEmployeeNumber = T2.employeeNumber WHERE T1.addressLine1 = ? AND T1.addressLine2 = ?", (address_line1, address_line2))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": result}

# Endpoint to get office addresses based on employee name
@app.get("/v1/car_retails/office_addresses_by_employee_name", operation_id="get_office_addresses_by_employee_name", summary="Retrieves the office addresses associated with a specific employee, identified by their first and last names. The operation returns the first and second lines of the office address.")
async def get_office_addresses_by_employee_name(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT T1.addressLine1, T1.addressLine2 FROM offices AS T1 INNER JOIN employees AS T2 ON T1.officeCode = T2.officeCode WHERE T2.firstName = ? AND T2.lastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"office_addresses": []}
    return {"office_addresses": result}

# Endpoint to get the order date with the highest total order value on specific dates
@app.get("/v1/car_retails/highest_order_value_by_dates", operation_id="get_highest_order_value_by_dates", summary="Retrieves the date with the highest total order value from two specified dates. The order value is calculated by multiplying the quantity ordered and the price per unit. The result is sorted in descending order and the date with the highest order value is returned.")
async def get_highest_order_value_by_dates(date1: str = Query(..., description="First date in 'YYYY-MM-DD' format"), date2: str = Query(..., description="Second date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.orderDate FROM orderdetails AS T1 INNER JOIN orders AS T2 ON T1.orderNumber = T2.orderNumber WHERE STRFTIME('%Y-%m-%d', T2.orderDate) = ? OR STRFTIME('%Y-%m-%d', T2.orderDate) = ? ORDER BY T1.quantityOrdered * T1.priceEach DESC LIMIT 1", (date1, date2))
    result = cursor.fetchone()
    if not result:
        return {"order_date": []}
    return {"order_date": result[0]}

# Endpoint to get the total quantity ordered for the product with the highest profit margin
@app.get("/v1/car_retails/total_quantity_highest_profit_margin", operation_id="get_total_quantity_highest_profit_margin", summary="Retrieves the total quantity ordered for the product with the highest profit margin. This operation calculates the profit margin for each product and identifies the product with the highest margin. It then sums up the total quantity ordered for this product.")
async def get_total_quantity_highest_profit_margin():
    cursor.execute("SELECT SUM(t2.quantityOrdered) FROM orderdetails AS t2 INNER JOIN ( SELECT t1.productCode FROM products AS t1 ORDER BY t1.MSRP - t1.buyPrice DESC LIMIT 1 ) AS t3 ON t2.productCode = t3.productCode")
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the payment amount for a specific customer and payment date
@app.get("/v1/car_retails/payment_amount_by_customer_and_date", operation_id="get_payment_amount", summary="Retrieves the total payment amount made by a specific customer on a given date. The operation requires the customer's name and the payment date in 'YYYY-MM-DD' format to accurately fetch the payment amount.")
async def get_payment_amount(customer_name: str = Query(..., description="Name of the customer"), payment_date: str = Query(..., description="Payment date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t1.amount FROM payments AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber WHERE t2.customerName = ? AND t1.paymentDate = ?", (customer_name, payment_date))
    result = cursor.fetchone()
    if not result:
        return {"amount": []}
    return {"amount": result[0]}

# Endpoint to get the contact details of a customer based on check number
@app.get("/v1/car_retails/customer_contact_by_check_number", operation_id="get_customer_contact", summary="Retrieves the first and last name of the customer associated with the provided check number. This operation fetches the customer's contact details from the database using the check number as a reference.")
async def get_customer_contact(check_number: str = Query(..., description="Check number")):
    cursor.execute("SELECT t2.contactFirstName, t2.contactLastName FROM payments AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber WHERE t1.checkNumber = ?", (check_number,))
    result = cursor.fetchone()
    if not result:
        return {"contact": []}
    return {"contact": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the address of a customer based on order number
@app.get("/v1/car_retails/customer_address_by_order_number", operation_id="get_customer_address", summary="Retrieves the address of a customer associated with a specific order number. The order number is used to identify the customer and retrieve their address details, which include the first and second lines of the address.")
async def get_customer_address(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT t2.addressLine1, t2.addressLine2 FROM orders AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber WHERE t1.orderNumber = ?", (order_number,))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": {"line1": result[0], "line2": result[1]}}

# Endpoint to get the product line description based on product code
@app.get("/v1/car_retails/product_line_description_by_product_code", operation_id="get_product_line_description", summary="Retrieves the detailed description of a product line associated with the provided product code. This operation fetches the textual description from the product lines table, which is linked to the products table using the product line identifier. The product code is used to pinpoint the specific product line and its corresponding description.")
async def get_product_line_description(product_code: str = Query(..., description="Product code")):
    cursor.execute("SELECT t1.textDescription FROM productlines AS t1 INNER JOIN products AS t2 ON t1.productLine = t2.productLine WHERE t2.productCode = ?", (product_code,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the email of the sales representative for a specific customer
@app.get("/v1/car_retails/sales_rep_email_by_customer_name", operation_id="get_sales_rep_email", summary="Retrieves the email address of the sales representative assigned to a specific customer. The customer is identified by their name, which is provided as an input parameter.")
async def get_sales_rep_email(customer_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT t2.email FROM customers AS t1 INNER JOIN employees AS t2 ON t1.salesRepEmployeeNumber = t2.employeeNumber WHERE t1.customerName = ?", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"email": []}
    return {"email": result[0]}

# Endpoint to get the most recent product name ordered by a specific customer
@app.get("/v1/car_retails/most_recent_product_by_customer", operation_id="get_most_recent_product", summary="Retrieves the name of the most recent product ordered by a specific customer. The operation uses the provided customer number to identify the customer and returns the name of the product from the most recent order. The order date is used to determine the most recent order.")
async def get_most_recent_product(customer_number: str = Query(..., description="Customer number")):
    cursor.execute("SELECT t3.productName FROM orderdetails AS t1 INNER JOIN orders AS t2 ON t1.orderNumber = t2.orderNumber INNER JOIN products AS t3 ON t1.productCode = t3.productCode WHERE t2.customerNumber = ? ORDER BY t2.orderDate DESC LIMIT 1", (customer_number,))
    result = cursor.fetchone()
    if not result:
        return {"productName": []}
    return {"productName": result[0]}

# Endpoint to get the discount percentage for a specific product in a specific order
@app.get("/v1/car_retails/discount_percentage_by_product_and_order", operation_id="get_discount_percentage", summary="Retrieves the discount percentage for a specific product within a given order. This operation calculates the discount by comparing the MSRP (Manufacturer's Suggested Retail Price) of the product with the actual price paid in the order. The product is identified by its unique code, and the order is specified by its number.")
async def get_discount_percentage(product_code: str = Query(..., description="Product code"), order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT (t1.MSRP - t2.priceEach) / t1.MSRP FROM products AS t1 INNER JOIN orderdetails AS t2 ON t1.productCode = t2.productCode WHERE t1.productCode = ? AND t2.orderNumber = ?", (product_code, order_number))
    result = cursor.fetchone()
    if not result:
        return {"discountPercentage": []}
    return {"discountPercentage": result[0]}

# Endpoint to get the email of employees based on job title
@app.get("/v1/car_retails/employee_email_by_job_title", operation_id="get_employee_email_by_job_title", summary="Retrieves the email addresses of employees who hold a specific job title. The job title is provided as an input parameter.")
async def get_employee_email_by_job_title(job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT t.email FROM employees t WHERE t.jobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get the first and last name of employees based on customer name
@app.get("/v1/car_retails/employee_names_by_customer", operation_id="get_employee_names_by_customer", summary="Retrieves the first and last names of employees who have sold to a specific customer. The customer is identified by their name.")
async def get_employee_names_by_customer(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT t2.firstName, t2.lastName FROM customers AS t1 INNER JOIN employees AS t2 ON t1.salesRepEmployeeNumber = t2.employeeNumber WHERE t1.customerName = ?", (customer_name,))
    result = cursor.fetchall()
    if not result:
        return {"employee_names": []}
    return {"employee_names": [{"firstName": row[0], "lastName": row[1]} for row in result]}

# Endpoint to get the customer names based on employee first and last name
@app.get("/v1/car_retails/customer_names_by_employee", operation_id="get_customer_names_by_employee", summary="Retrieves the names of customers associated with a specific employee, identified by their first and last names. The operation returns a list of customer names that were sold to by the specified employee.")
async def get_customer_names_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT t1.customerName FROM customers AS t1 INNER JOIN employees AS t2 ON t1.salesRepEmployeeNumber = t2.employeeNumber WHERE t2.firstName = ? AND t2.lastName = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the count of customers based on employee first and last name
@app.get("/v1/car_retails/customer_count_by_employee", operation_id="get_customer_count_by_employee", summary="Retrieves the total number of customers associated with a specific employee, identified by their first and last names. This operation calculates the count based on the relationship between the customers and employees tables, considering the sales representative assigned to each customer.")
async def get_customer_count_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT COUNT(t1.customerNumber) FROM customers AS t1 INNER JOIN employees AS t2 ON t1.salesRepEmployeeNumber = t2.employeeNumber WHERE t2.firstName = ? AND t2.lastName = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the top 3 customer phones based on employee first and last name, ordered by credit limit
@app.get("/v1/car_retails/top_customer_phones_by_employee", operation_id="get_top_customer_phones_by_employee", summary="Retrieves the top three customer phone numbers associated with the specified employee, sorted by the customers' credit limits in descending order. The employee is identified by their first and last names.")
async def get_top_customer_phones_by_employee(first_name: str = Query(..., description="First name of the employee"), last_name: str = Query(..., description="Last name of the employee")):
    cursor.execute("SELECT t1.phone FROM customers AS t1 INNER JOIN employees AS t2 ON t1.salesRepEmployeeNumber = t2.employeeNumber WHERE t2.firstName = ? AND t2.lastName = ? ORDER BY t1.creditLimit DESC LIMIT 3", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": [row[0] for row in result]}

# Endpoint to get the count of employees based on country and job title
@app.get("/v1/car_retails/employee_count_by_country_job_title", operation_id="get_employee_count_by_country_job_title", summary="Retrieves the total number of employees in a specific country and job title. The operation uses the provided country and job title to filter the employees and calculate the count.")
async def get_employee_count_by_country_job_title(country: str = Query(..., description="Country of the office"), job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT COUNT(t1.employeeNumber) FROM employees AS t1 INNER JOIN offices AS t2 ON t1.officeCode = t2.officeCode WHERE t2.country = ? AND t1.jobTitle = ?", (country, job_title))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the office address based on job title
@app.get("/v1/car_retails/office_address_by_job_title", operation_id="get_office_address_by_job_title", summary="Retrieves the office address associated with a specific job title. The job title is used to identify the corresponding office address from the employees and offices tables.")
async def get_office_address_by_job_title(job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT t2.addressLine1, t2.addressLine2 FROM employees AS t1 INNER JOIN offices AS t2 ON t1.officeCode = t2.officeCode WHERE t1.jobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [{"addressLine1": row[0], "addressLine2": row[1]} for row in result]}

# Endpoint to get the postal code based on job title
@app.get("/v1/car_retails/postal_code_by_job_title", operation_id="get_postal_code_by_job_title", summary="Retrieves the postal code of the office where the employee with the specified job title works. The job title is used to identify the employee and subsequently the office location.")
async def get_postal_code_by_job_title(job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT t2.postalCode FROM employees AS t1 INNER JOIN offices AS t2 ON t1.officeCode = t2.officeCode WHERE t1.jobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"postal_codes": []}
    return {"postal_codes": [row[0] for row in result]}

# Endpoint to get the total order value based on customer name and order date
@app.get("/v1/car_retails/total_order_value_by_customer_order_date", operation_id="get_total_order_value_by_customer_order_date", summary="Retrieves the total value of all orders placed by a specific customer on a given date. The calculation is based on the price and quantity of each item in the orders. The customer name and order date are required as input parameters.")
async def get_total_order_value_by_customer_order_date(customer_name: str = Query(..., description="Name of the customer"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(t1.priceEach * t1.quantityOrdered) FROM orderdetails AS t1 INNER JOIN orders AS t2 ON t1.orderNumber = t2.orderNumber INNER JOIN customers AS t3 ON t2.customerNumber = t3.customerNumber WHERE t3.customerName = ? AND t2.orderDate = ?", (customer_name, order_date))
    result = cursor.fetchone()
    if not result:
        return {"total_value": []}
    return {"total_value": result[0]}

# Endpoint to get product names for a specific customer and order date
@app.get("/v1/car_retails/product_names_by_customer_order_date", operation_id="get_product_names", summary="Retrieves the names of products ordered by a specific customer on a given date. The operation requires the customer's name and the order date in 'YYYY-MM-DD' format as input parameters.")
async def get_product_names(customer_name: str = Query(..., description="Name of the customer"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t4.productName FROM orderdetails AS t1 INNER JOIN orders AS t2 ON t1.orderNumber = t2.orderNumber INNER JOIN customers AS t3 ON t2.customerNumber = t3.customerNumber INNER JOIN products AS t4 ON t1.productCode = t4.productCode WHERE t3.customerName = ? AND t2.orderDate = ?", (customer_name, order_date))
    result = cursor.fetchall()
    if not result:
        return {"product_names": []}
    return {"product_names": [row[0] for row in result]}

# Endpoint to get the most expensive product for a specific customer
@app.get("/v1/car_retails/most_expensive_product_by_customer", operation_id="get_most_expensive_product", summary="Retrieves the product with the highest total cost that a specific customer has purchased. The product is identified by comparing the total cost of each product (price per unit multiplied by quantity ordered) for the given customer. The total cost is calculated based on the customer's order history.")
async def get_most_expensive_product(customer_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT t4.productName FROM orderdetails AS t1 INNER JOIN orders AS t2 ON t1.orderNumber = t2.orderNumber INNER JOIN customers AS t3 ON t2.customerNumber = t3.customerNumber INNER JOIN products AS t4 ON t1.productCode = t4.productCode WHERE t3.customerName = ? ORDER BY t1.priceEach * t1.quantityOrdered DESC LIMIT 1", (customer_name,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": []}
    return {"product_name": result[0]}

# Endpoint to get shipped dates for a specific customer and order date
@app.get("/v1/car_retails/shipped_dates_by_customer_order_date", operation_id="get_shipped_dates", summary="Retrieves the shipping dates for orders placed by a specific customer on a given date. The operation requires the customer's name and the order date in 'YYYY-MM-DD' format to filter the results.")
async def get_shipped_dates(customer_name: str = Query(..., description="Name of the customer"), order_date: str = Query(..., description="Order date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT t1.shippedDate FROM orders AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber WHERE t2.customerName = ? AND t1.orderDate = ?", (customer_name, order_date))
    result = cursor.fetchall()
    if not result:
        return {"shipped_dates": []}
    return {"shipped_dates": [row[0] for row in result]}

# Endpoint to get the product with the highest profit margin in a specific product line
@app.get("/v1/car_retails/highest_profit_margin_product", operation_id="get_highest_profit_margin_product", summary="Retrieves the product with the highest profit margin from a specified product line. The profit margin is calculated as the difference between the Manufacturer's Suggested Retail Price (MSRP) and the buy price. The product line is determined by the provided input parameter.")
async def get_highest_profit_margin_product(product_line: str = Query(..., description="Product line")):
    cursor.execute("SELECT t.productName, t.MSRP - t.buyPrice FROM products AS t WHERE t.productLine = ? ORDER BY t.MSRP - t.buyPrice DESC LIMIT 1", (product_line,))
    result = cursor.fetchone()
    if not result:
        return {"product_name": [], "profit_margin": []}
    return {"product_name": result[0], "profit_margin": result[1]}

# Endpoint to get customer names for orders with a specific status
@app.get("/v1/car_retails/customer_names_by_order_status", operation_id="get_customer_names", summary="Retrieves the names of customers who have orders with a specified status. The status parameter is used to filter the orders and identify the relevant customers.")
async def get_customer_names(status: str = Query(..., description="Status of the order")):
    cursor.execute("SELECT t2.customerName FROM orders AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber WHERE t1.status = ?", (status,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the count of orders shipped more than 3 days after the required date
@app.get("/v1/car_retails/count_late_shipped_orders", operation_id="get_count_late_shipped_orders", summary="Retrieves the count of orders that were shipped more than 3 days after the required date, based on the provided order status.")
async def get_count_late_shipped_orders(status: str = Query(..., description="Status of the order")):
    cursor.execute("SELECT COUNT(CASE WHEN JULIANDAY(t1.shippeddate) - JULIANDAY(t1.requireddate) > 3 THEN T1.customerNumber ELSE NULL END) FROM orders AS T1 INNER JOIN orderdetails AS T2 ON T1.orderNumber = T2.orderNumber WHERE T1.status = ?", (status,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer with the highest payment amount in a specific year
@app.get("/v1/car_retails/highest_paying_customer_by_year", operation_id="get_highest_paying_customer", summary="Retrieves the customer who made the highest total payment in a specified year. The year is provided in the 'YYYY' format. The operation returns the name of the customer with the highest cumulative payment amount for the given year.")
async def get_highest_paying_customer(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT t2.customerName FROM payments AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber WHERE STRFTIME('%Y', t1.paymentDate) = ? GROUP BY t2.customerNumber, t2.customerName ORDER BY SUM(t1.amount) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the product with the highest quantity ordered
@app.get("/v1/car_retails/highest_quantity_ordered_product", operation_id="get_highest_quantity_ordered_product", summary="Retrieves the product with the highest recorded order quantity and returns its name along with the difference between its Manufacturer's Suggested Retail Price (MSRP) and buy price.")
async def get_highest_quantity_ordered_product():
    cursor.execute("SELECT productName, MSRP - buyPrice FROM products WHERE productCode = ( SELECT productCode FROM orderdetails ORDER BY quantityOrdered DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"product_name": [], "profit_margin": []}
    return {"product_name": result[0], "profit_margin": result[1]}

# Endpoint to get the employee with the highest employee number
@app.get("/v1/car_retails/highest_employee_number", operation_id="get_highest_employee_number", summary="Retrieves the first and last name of the employee with the highest employee number from the database. This operation uses the office and employee tables to identify the employee with the maximum employee number.")
async def get_highest_employee_number():
    cursor.execute("SELECT T2.firstName, T2.lastName FROM offices AS T1 INNER JOIN employees AS T2 ON T1.officeCode = T2.officeCode WHERE T2.employeeNumber = ( SELECT MAX(employeeNumber) FROM employees )")
    result = cursor.fetchone()
    if not result:
        return {"first_name": [], "last_name": []}
    return {"first_name": result[0], "last_name": result[1]}

# Endpoint to get the first and last names of employees associated with disputed orders
@app.get("/v1/car_retails/employee_names_disputed_orders", operation_id="get_employee_names_disputed_orders", summary="Retrieves the first and last names of employees who have been associated with orders that are currently in dispute. The operation filters the results based on the provided order status.")
async def get_employee_names_disputed_orders(status: str = Query(..., description="Status of the order")):
    cursor.execute("SELECT t3.firstName, t3.lastName FROM orders AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber INNER JOIN employees AS t3 ON t2.salesRepEmployeeNumber = t3.employeeNumber WHERE t1.status = ?", (status,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"firstName": row[0], "lastName": row[1]} for row in result]}

# Endpoint to get the percentage of employees in a specific city
@app.get("/v1/car_retails/percentage_employees_city", operation_id="get_percentage_employees_city", summary="Retrieves the percentage of employees working in a specified city. This operation calculates the ratio of employees in the given city to the total number of employees across all offices. The city is identified by its name.")
async def get_percentage_employees_city(city: str = Query(..., description="City name")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN t1.city = ? THEN t2.employeeNumber ELSE NULL END) AS REAL) * 100 / COUNT(t2.employeeNumber) FROM offices AS t1 INNER JOIN employees AS t2 ON t1.officeCode = t2.officeCode", (city,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the first and last names of employees with a specific job title
@app.get("/v1/car_retails/employee_names_job_title", operation_id="get_employee_names_job_title", summary="Retrieves the first and last names of employees who hold a specific job title. The job title is provided as an input parameter, allowing for targeted retrieval of employee names.")
async def get_employee_names_job_title(job_title: str = Query(..., description="Job title of the employee")):
    cursor.execute("SELECT t2.firstName, t2.lastName FROM offices AS t1 INNER JOIN employees AS t2 ON t1.officeCode = t2.officeCode WHERE t2.jobTitle = ?", (job_title,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"firstName": row[0], "lastName": row[1]} for row in result]}

# Endpoint to get the first and last names and reportsTo of employees in a specific country
@app.get("/v1/car_retails/employee_details_country", operation_id="get_employee_details_country", summary="Retrieves the first and last names and reporting hierarchy of employees working in a specified country. The country is provided as an input parameter.")
async def get_employee_details_country(country: str = Query(..., description="Country name")):
    cursor.execute("SELECT t2.firstName, t2.lastName, t2.reportsTo FROM offices AS t1 INNER JOIN employees AS t2 ON t1.officeCode = t2.officeCode WHERE t1.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"firstName": row[0], "lastName": row[1], "reportsTo": row[2]} for row in result]}

# Endpoint to get the customer name for the highest priced order of a specific product
@app.get("/v1/car_retails/customer_highest_priced_order", operation_id="get_customer_highest_priced_order", summary="Retrieves the name of the customer who placed the highest priced order for a specific product. The product is identified by its name, and the customer with the highest order price is returned.")
async def get_customer_highest_priced_order(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT t4.customerName FROM products AS t1 INNER JOIN orderdetails AS t2 ON t1.productCode = t2.productCode INNER JOIN orders AS t3 ON t2.orderNumber = t3.orderNumber INNER JOIN customers AS t4 ON t3.customerNumber = t4.customerNumber WHERE t1.productName = ? ORDER BY t2.priceEach DESC LIMIT 1", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"customerName": []}
    return {"customerName": result[0]}

# Endpoint to get the percentage of total payments made by a specific customer in a specific year
@app.get("/v1/car_retails/percentage_payments_customer_year", operation_id="get_percentage_payments_customer_year", summary="Retrieves the proportion of total annual payments made by a particular customer. The calculation is based on the customer's payments in the specified year.")
async def get_percentage_payments_customer_year(customer_name: str = Query(..., description="Customer name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(CASE WHEN t1.customerName = ? THEN t2.amount ELSE 0 END) * 100 / SUM(t2.amount) FROM customers AS t1 INNER JOIN payments AS t2 ON t1.customerNumber = t2.customerNumber WHERE STRFTIME('%Y', t2.paymentDate) = ?", (customer_name, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total profit from a specific order
@app.get("/v1/car_retails/total_profit_order", operation_id="get_total_profit_order", summary="Retrieves the total profit generated from a specific order by calculating the sum of the profit from each product in the order. The profit is determined by subtracting the buy price of each product from its selling price and then multiplying by the quantity ordered. The order number is required to identify the relevant order.")
async def get_total_profit_order(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT SUM((t1.priceEach - t2.buyPrice) * t1.quantityOrdered) FROM orderdetails AS t1 INNER JOIN products AS t2 ON t1.productCode = t2.productCode WHERE t1.orderNumber = ?", (order_number,))
    result = cursor.fetchone()
    if not result:
        return {"total_profit": []}
    return {"total_profit": result[0]}

# Endpoint to get the total amount paid by a specific customer
@app.get("/v1/car_retails/total_amount_paid_customer", operation_id="get_total_amount_paid_customer", summary="Retrieves the total amount paid by a specific customer. The operation calculates the sum of all payments made by the customer identified by the provided customer number.")
async def get_total_amount_paid_customer(customer_number: str = Query(..., description="Customer number")):
    cursor.execute("SELECT SUM(t.amount) FROM payments t WHERE t.customerNumber = ?", (customer_number,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the total amount for a specific order
@app.get("/v1/car_retails/total_amount_order", operation_id="get_total_amount_order", summary="Retrieves the total amount for a specific order by summing the product of the price per unit and the quantity ordered for the given order number.")
async def get_total_amount_order(order_number: str = Query(..., description="Order number")):
    cursor.execute("SELECT SUM(t.priceEach * t.quantityOrdered) FROM orderdetails t WHERE t.orderNumber = ?", (order_number,))
    result = cursor.fetchone()
    if not result:
        return {"total_amount": []}
    return {"total_amount": result[0]}

# Endpoint to get the top 3 most expensive products
@app.get("/v1/car_retails/top_expensive_products", operation_id="get_top_expensive_products", summary="Retrieves the names of the top three most expensive products, based on the highest price per unit. The products are determined by comparing the prices of all items in the order details.")
async def get_top_expensive_products():
    cursor.execute("SELECT t1.productName FROM products AS t1 INNER JOIN orderdetails AS t2 ON t1.productCode = t2.productCode ORDER BY t2.priceEach DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [{"productName": row[0]} for row in result]}

# Endpoint to get the contact details of the customer with the highest credit limit for a given employee
@app.get("/v1/car_retails/customer_contact_highest_credit_limit", operation_id="get_customer_contact_highest_credit_limit", summary="Retrieves the first name and last name of the customer with the highest credit limit associated with a specific employee. The employee is identified by the provided employee number.")
async def get_customer_contact_highest_credit_limit(employee_number: str = Query(..., description="Employee number")):
    cursor.execute("SELECT t2.contactFirstName, t2.contactLastName FROM employees AS t1 INNER JOIN customers AS t2 ON t1.employeeNumber = t2.salesRepEmployeeNumber WHERE t1.employeeNumber = ? ORDER BY t2.creditLimit DESC LIMIT 1", (employee_number,))
    result = cursor.fetchone()
    if not result:
        return {"contact": []}
    return {"contact": {"firstName": result[0], "lastName": result[1]}}

# Endpoint to get the total quantity ordered for a specific product
@app.get("/v1/car_retails/total_quantity_ordered_product", operation_id="get_total_quantity_ordered_product", summary="Retrieves the total quantity ordered for a specific product by aggregating the quantity ordered from the order details table. The product is identified by its name.")
async def get_total_quantity_ordered_product(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT SUM(t2.quantityOrdered) FROM products AS t1 INNER JOIN orderdetails AS t2 ON t1.productCode = t2.productCode WHERE t1.productName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the shipped date of the order with the highest price per item
@app.get("/v1/car_retails/shipped_date_highest_price_item", operation_id="get_shipped_date_highest_price_item", summary="Retrieves the date when the order with the highest priced item was shipped. This operation identifies the order with the most expensive item and returns the date it was shipped. The order details are cross-referenced to determine the highest priced item.")
async def get_shipped_date_highest_price_item():
    cursor.execute("SELECT t1.shippedDate FROM orders AS t1 INNER JOIN orderdetails AS t2 ON t1.orderNumber = t2.orderNumber ORDER BY t2.priceEach DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"shipped_date": []}
    return {"shipped_date": result[0]}

# Endpoint to get the total quantity ordered for a specific product line in a given year
@app.get("/v1/car_retails/total_quantity_ordered_product_line_year", operation_id="get_total_quantity_ordered_product_line_year", summary="Retrieves the total quantity of products ordered for a specific product line in a given year. The operation calculates the sum of quantities ordered from the order details, considering only the orders that belong to the specified product line and were placed in the requested year.")
async def get_total_quantity_ordered_product_line_year(product_line: str = Query(..., description="Product line"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(t2.quantityOrdered) FROM orders AS t1 INNER JOIN orderdetails AS t2 ON t1.orderNumber = t2.orderNumber INNER JOIN products AS t3 ON t2.productCode = t3.productCode WHERE t3.productLine = ? AND STRFTIME('%Y', t1.orderDate) = ?", (product_line, year))
    result = cursor.fetchone()
    if not result:
        return {"total_quantity": []}
    return {"total_quantity": result[0]}

# Endpoint to get the order numbers for customers with a specific credit limit
@app.get("/v1/car_retails/order_numbers_credit_limit", operation_id="get_order_numbers_credit_limit", summary="Retrieves the order numbers for customers who have a specified credit limit. The operation filters orders based on the provided credit limit and returns the corresponding order numbers.")
async def get_order_numbers_credit_limit(credit_limit: int = Query(..., description="Credit limit")):
    cursor.execute("SELECT t1.orderNumber FROM orders AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber WHERE t2.creditLimit = ?", (credit_limit,))
    result = cursor.fetchall()
    if not result:
        return {"order_numbers": []}
    return {"order_numbers": [row[0] for row in result]}

# Endpoint to get the customer with the highest profit margin
@app.get("/v1/car_retails/customer_highest_profit_margin", operation_id="get_customer_highest_profit_margin", summary="Retrieves the customer who has generated the highest profit margin based on their order history. The profit margin is calculated by subtracting the buy price from the selling price and then multiplying by the quantity ordered. The result is then grouped by customer name, selling price, buy price, and quantity ordered before being ordered in descending order to find the customer with the highest profit margin.")
async def get_customer_highest_profit_margin():
    cursor.execute("SELECT t3.customerName, (t1.priceEach - t4.buyPrice) * t1.quantityOrdered FROM orderdetails AS t1 INNER JOIN orders AS t2 ON t1.orderNumber = t2.orderNumber INNER JOIN customers AS t3 ON t2.customerNumber = t3.customerNumber INNER JOIN products AS t4 ON t1.productCode = t4.productCode GROUP BY t3.customerName, t1.priceEach, t4.buyPrice, t1.quantityOrdered ORDER BY (t1.priceEach - t4.buyPrice) * t1.quantityOrdered DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"customer": []}
    return {"customer": {"name": result[0], "profit_margin": result[1]}}

# Endpoint to get the number of payments below a certain amount grouped by year
@app.get("/v1/car_retails/payments_below_amount_by_year", operation_id="get_payments_below_amount_by_year", summary="Retrieves the count of payments below a specified amount, grouped by year. This operation allows you to analyze payment trends over time, focusing on transactions under a certain threshold. The input parameter determines the maximum amount for the count.")
async def get_payments_below_amount_by_year(amount: float = Query(..., description="Amount threshold")):
    cursor.execute("SELECT STRFTIME('%Y', t1.paymentDate), COUNT(t1.customerNumber) FROM payments AS t1 WHERE t1.amount < ? GROUP BY STRFTIME('%Y', t1.paymentDate)", (amount,))
    result = cursor.fetchall()
    if not result:
        return {"payments": []}
    return {"payments": [{"year": row[0], "count": row[1]} for row in result]}

# Endpoint to get the top 3 products ordered in a specific year
@app.get("/v1/car_retails/top_products_ordered_year", operation_id="get_top_products_ordered_year", summary="Retrieves the top three products with the highest total quantity ordered in a specified year. The year is provided in the 'YYYY' format. The operation calculates the total quantity ordered for each product and ranks them in descending order, returning the top three.")
async def get_top_products_ordered_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT t3.productName, SUM(t2.quantityOrdered) FROM orders AS t1 INNER JOIN orderdetails AS t2 ON t1.orderNumber = t2.orderNumber INNER JOIN products AS t3 ON t2.productCode = t3.productCode WHERE STRFTIME('%Y', t1.orderDate) = ? GROUP BY t3.productName ORDER BY SUM(t2.quantityOrdered) DESC LIMIT 3", (year,))
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": [{"name": row[0], "total_quantity": row[1]} for row in result]}

# Endpoint to get the total quantity ordered by sales representatives for a specific product
@app.get("/v1/car_retails/total_quantity_ordered_sales_reps_product", operation_id="get_total_quantity_ordered_sales_reps_product", summary="Retrieves the total quantity of a specific product ordered by each sales representative. The product is identified by its name.")
async def get_total_quantity_ordered_sales_reps_product(product_name: str = Query(..., description="Product name")):
    cursor.execute("SELECT t5.firstName, t5.lastName, SUM(t2.quantityOrdered) FROM products AS t1 INNER JOIN orderdetails AS t2 ON t1.productCode = t2.productCode INNER JOIN orders AS t3 ON t2.orderNumber = t3.orderNumber INNER JOIN customers AS t4 ON t3.customerNumber = t4.customerNumber INNER JOIN employees AS t5 ON t4.salesRepEmployeeNumber = t5.employeeNumber WHERE t1.productName = ? GROUP BY t5.lastName, t5.firstName", (product_name,))
    result = cursor.fetchall()
    if not result:
        return {"sales_reps": []}
    return {"sales_reps": [{"firstName": row[0], "lastName": row[1], "total_quantity": row[2]} for row in result]}

# Endpoint to get the names of employees with a specific job title in a specific city
@app.get("/v1/car_retails/employee_names_job_title_city", operation_id="get_employee_names_job_title_city", summary="Retrieves the first and last names of employees who hold a specific job title in a given city. The city and job title are provided as input parameters.")
async def get_employee_names_job_title_city(city: str = Query(..., description="City"), job_title: str = Query(..., description="Job title")):
    cursor.execute("SELECT t1.lastName, t1.firstName FROM employees AS t1 INNER JOIN offices AS t2 ON t1.officeCode = t2.officeCode WHERE t2.city = ? AND t1.jobTitle = ?", (city, job_title))
    result = cursor.fetchall()
    if not result:
        return {"employees": []}
    return {"employees": [{"lastName": row[0], "firstName": row[1]} for row in result]}

# Endpoint to get customer details based on check number
@app.get("/v1/car_retails/customer_details_by_check_number", operation_id="get_customer_details", summary="Retrieves the name and country of a customer who made a payment using a specific check number. The check number is used to filter the results.")
async def get_customer_details(check_number: str = Query(..., description="Check number to filter the results")):
    cursor.execute("SELECT t2.customerName, t2.country FROM payments AS t1 INNER JOIN customers AS t2 ON t1.customerNumber = t2.customerNumber WHERE t1.checkNumber = ?", (check_number,))
    result = cursor.fetchall()
    if not result:
        return {"customers": []}
    return {"customers": result}

# Endpoint to get the sum of order numbers for a specific product
@app.get("/v1/car_retails/sum_order_numbers_by_product", operation_id="get_sum_order_numbers", summary="Retrieves the total number of orders for a specific product. The product is identified by its name, which is used to filter the results.")
async def get_sum_order_numbers(product_name: str = Query(..., description="Product name to filter the results")):
    cursor.execute("SELECT SUM(t1.orderNumber) FROM orderdetails AS t1 INNER JOIN products AS t2 ON t1.productCode = t2.productCode WHERE t2.productName = ?", (product_name,))
    result = cursor.fetchone()
    if not result:
        return {"sum_order_numbers": []}
    return {"sum_order_numbers": result[0]}

# Endpoint to get the top 5 products ordered by quantity
@app.get("/v1/car_retails/top_5_products_by_quantity", operation_id="get_top_5_products", summary="Retrieves the top 5 products with the lowest total quantity ordered. The products are ranked based on the sum of their respective quantities ordered, with the least ordered product appearing first. This operation does not require any input parameters.")
async def get_top_5_products():
    cursor.execute("SELECT t2.productName FROM orderdetails AS t1 INNER JOIN products AS t2 ON t1.productCode = t2.productCode GROUP BY t2.productName ORDER BY SUM(t1.quantityOrdered) ASC LIMIT 5")
    result = cursor.fetchall()
    if not result:
        return {"products": []}
    return {"products": result}

# Endpoint to get the count of orders based on status and country
@app.get("/v1/car_retails/order_count_by_status_and_country", operation_id="get_order_count", summary="Retrieves the total number of orders that match a specific status and are associated with a particular country. The operation filters the orders based on the provided status and country parameters.")
async def get_order_count(status: str = Query(..., description="Order status to filter the results"), country: str = Query(..., description="Country to filter the results")):
    cursor.execute("SELECT COUNT(t2.orderNumber) FROM customers AS t1 INNER JOIN orders AS t2 ON t1.customerNumber = t2.customerNumber WHERE t2.status = ? AND t1.country = ?", (status, country))
    result = cursor.fetchone()
    if not result:
        return {"order_count": []}
    return {"order_count": result[0]}

# Endpoint to get the total sales amount for a specific product line, customer, and order status
@app.get("/v1/car_retails/total_sales_amount", operation_id="get_total_sales_amount", summary="Retrieves the total sales amount for a specific product line, customer, and order status. This operation calculates the sum of the product of the price and quantity ordered for the specified product line, customer, and order status. The input parameters allow filtering the results based on the product line, customer name, and order status.")
async def get_total_sales_amount(product_line: str = Query(..., description="Product line to filter the results"), customer_name: str = Query(..., description="Customer name to filter the results"), status: str = Query(..., description="Order status to filter the results")):
    cursor.execute("SELECT SUM(t3.priceEach * t3.quantityOrdered) FROM customers AS t1 INNER JOIN orders AS t2 ON t1.customerNumber = t2.customerNumber INNER JOIN orderdetails AS t3 ON t2.orderNumber = t3.orderNumber INNER JOIN products AS t4 ON t3.productCode = t4.productCode WHERE t4.productLine = ? AND t1.customerName = ? AND t2.status = ?", (product_line, customer_name, status))
    result = cursor.fetchone()
    if not result:
        return {"total_sales_amount": []}
    return {"total_sales_amount": result[0]}

api_calls = [
    "/v1/car_retails/customer_count_by_country",
    "/v1/car_retails/product_vendor_price_difference?limit=1",
    "/v1/car_retails/employee_names_by_payment_year?year=2004&limit=1",
    "/v1/car_retails/total_price_ordered_items?customer_name=Rovelli%20Gifts&status=Shipped&year=2003",
    "/v1/car_retails/employee_count_by_city?city=Sydney",
    "/v1/car_retails/employee_numbers_by_manager_city?reports_to=1143&city=NYC",
    "/v1/car_retails/payment_statistics?start_date=2003-01-01&end_date=2005-12-31",
    "/v1/car_retails/customer_addresses_by_payment_amount?limit=5",
    "/v1/car_retails/check_number_count_by_customer_year?customer_name=Euro%2B%20Shopping%20Channel&year=2004",
    "/v1/car_retails/product_names_by_scale_line_quantity?product_scale=1%3A18&product_line=Classic%20Cars&limit=1",
    "/v1/car_retails/top_customer_by_payment?credit_limit=100000&year=2004",
    "/v1/car_retails/top_customer_by_city_and_employee?city=Tokyo&reports_to=1056",
    "/v1/car_retails/revenue_most_least_profitable_products",
    "/v1/car_retails/count_orders_by_value_and_status?order_value=4000&status=Cancelled",
    "/v1/car_retails/count_orders_by_quantity_and_year?quantity_ordered=30&year=2003",
    "/v1/car_retails/total_revenue_by_status?status=Cancelled",
    "/v1/car_retails/total_revenue_by_product_line?product_line=Motorcycles",
    "/v1/car_retails/count_products_by_product_line?product_line=Planes",
    "/v1/car_retails/count_products_by_profit_margin?profit_margin=100",
    "/v1/car_retails/average_revenue_per_order?product_vendor=Exoto%20Designs&year=2005",
    "/v1/car_retails/employee_count_by_job_title_office_code?job_title=Sales%20Rep&office_code=1",
    "/v1/car_retails/distinct_customer_names_by_payment_amount?amount=50000",
    "/v1/car_retails/total_payment_amount_by_country?country=USA",
    "/v1/car_retails/total_payment_amount_by_year_credit_limit?year=2003&credit_limit=0",
    "/v1/car_retails/customer_info_by_order_status?status=Cancelled",
    "/v1/car_retails/count_countries_by_order_status_country?status=Shipped&country=France&count_status=2",
    "/v1/car_retails/average_order_value_by_status_country?status=Shipped&country=Germany",
    "/v1/car_retails/employee_names_by_office_city?city=Tokyo",
    "/v1/car_retails/employee_details_by_office_city_job_title?city=Tokyo&job_title=Sales%20Rep",
    "/v1/car_retails/employee_emails_by_reports_to_city?reports_to=1002&city=San%20Francisco",
    "/v1/car_retails/employee_emails_numbers_by_office_state_country?state=MA&country=USA",
    "/v1/car_retails/employee_emails_office_countries_by_credit_limit_job_title?country=Australia&credit_limit=200000&job_title=Sales%20Rep",
    "/v1/car_retails/distinct_employee_emails_by_job_title_country?job_title=Sales%20Rep&country=UK",
    "/v1/car_retails/customer_count_by_credit_limit_country?credit_limit=220000&country=Australia",
    "/v1/car_retails/top_customer_numbers_by_payment_amount",
    "/v1/car_retails/top_employee_emails_by_credit_limit_job_title?job_title=Sales%20Rep&country=UK",
    "/v1/car_retails/employee_details_by_city?city=Paris",
    "/v1/car_retails/top_products_by_stock?product_line=Motorcycles",
    "/v1/car_retails/customer_count_by_payment_year_range?start_year=2003&end_year=2004&min_payments=3",
    "/v1/car_retails/average_profit_margin?product_name=1937%20Lincoln%20Berline",
    "/v1/car_retails/most_ordered_product?product_line=Motorcycles&product_scale=1:10&country=USA",
    "/v1/car_retails/check_numbers_by_customer_date_range?start_date=2003-08-01&end_date=2004-08-30&customer_name=Mini%20Gifts%20Distributors%20Ltd.",
    "/v1/car_retails/max_profit_margin_by_product_line?product_line=Planes",
    "/v1/car_retails/total_sales_amount_shipped_orders?status=Shipped&start_date=2003-01-01&end_date=2004-12-31",
    "/v1/car_retails/top_employee_by_payment_amount",
    "/v1/car_retails/product_highest_quantity_ordered?city=Boston&job_title=Sales%20Rep",
    "/v1/car_retails/total_profit_date_range?start_date=2003-01-05&end_date=2005-05-10",
    "/v1/car_retails/customer_highest_quantity_ordered?year=2005&job_title=Sales%20Rep",
    "/v1/car_retails/customer_phones_lastname_country?last_name_pattern=M%25&country=Germany",
    "/v1/car_retails/average_payment_date_range?start_date=2004-01-01&end_date=2004-06-30",
    "/v1/car_retails/percentage_orders_shipped_customer?customer_number=186&status=Shipped&start_date=2005-01-01&end_date=2005-12-31",
    "/v1/car_retails/count_customers_cancelled_orders_credit_limit?status=Cancelled&credit_limit=115000",
    "/v1/car_retails/earliest_payment_date_sales_rep?first_name=Barry&last_name=Jones&job_title=Sales%20Rep",
    "/v1/car_retails/sales_rep_employee_numbers_country?country=France",
    "/v1/car_retails/customer_address_shipped_date_status?shipped_date=2005-04-04&status=Shipped",
    "/v1/car_retails/customer_addresses_by_city_job_title?city=NYC&job_title=Sales%20Rep",
    "/v1/car_retails/customer_addresses_by_job_title?job_title=Sales%20Rep",
    "/v1/car_retails/total_profit_margin_by_vendor_description?product_vendor=Carousel%20DieCast%20Legends&text_description=%25perfect%20holiday%20or%20anniversary%20gift%20for%20executives%25",
    "/v1/car_retails/top_customer_by_average_payment?city=Boston",
    "/v1/car_retails/total_quantity_average_price_by_product?product_name=18th%20Century%20Vintage%20Horse%20Carriage",
    "/v1/car_retails/count_product_codes_by_order_number?order_number=10252",
    "/v1/car_retails/employee_names_by_customer_address?address_line1=25%20Maiden%20Lane&address_line2=Floor%20No.%204",
    "/v1/car_retails/office_addresses_by_employee_name?first_name=Foon%20Yue&last_name=Tseng",
    "/v1/car_retails/highest_order_value_by_dates?date1=2005-04-08&date2=2005-04-10",
    "/v1/car_retails/total_quantity_highest_profit_margin",
    "/v1/car_retails/payment_amount_by_customer_and_date?customer_name=Petit%20Auto&payment_date=2004-08-09",
    "/v1/car_retails/customer_contact_by_check_number?check_number=NR157385",
    "/v1/car_retails/customer_address_by_order_number?order_number=10383",
    "/v1/car_retails/product_line_description_by_product_code?product_code=S18_2949",
    "/v1/car_retails/sales_rep_email_by_customer_name?customer_name=Dragon%20Souveniers%2C%20Ltd.",
    "/v1/car_retails/most_recent_product_by_customer?customer_number=114",
    "/v1/car_retails/discount_percentage_by_product_and_order?product_code=S18_3482&order_number=10108",
    "/v1/car_retails/employee_email_by_job_title?job_title=President",
    "/v1/car_retails/employee_names_by_customer?customer_name=Muscle%20Machine%20Inc",
    "/v1/car_retails/customer_names_by_employee?first_name=Steve&last_name=Patterson",
    "/v1/car_retails/customer_count_by_employee?first_name=William&last_name=Patterson",
    "/v1/car_retails/top_customer_phones_by_employee?first_name=Leslie&last_name=Jennings",
    "/v1/car_retails/employee_count_by_country_job_title?country=USA&job_title=Sales%20Rep",
    "/v1/car_retails/office_address_by_job_title?job_title=President",
    "/v1/car_retails/postal_code_by_job_title?job_title=VP%20Sales",
    "/v1/car_retails/total_order_value_by_customer_order_date?customer_name=Cruz%20%26%20Sons%20Co.&order_date=2003-03-03",
    "/v1/car_retails/product_names_by_customer_order_date?customer_name=Cruz%20%26%20Sons%20Co.&order_date=2003-03-03",
    "/v1/car_retails/most_expensive_product_by_customer?customer_name=Cruz%20%26%20Sons%20Co.",
    "/v1/car_retails/shipped_dates_by_customer_order_date?customer_name=Cruz%20%26%20Sons%20Co.&order_date=2003-03-03",
    "/v1/car_retails/highest_profit_margin_product?product_line=Classic%20Cars",
    "/v1/car_retails/customer_names_by_order_status?status=In%20Process",
    "/v1/car_retails/count_late_shipped_orders?status=Shipped",
    "/v1/car_retails/highest_paying_customer_by_year?year=2005",
    "/v1/car_retails/highest_quantity_ordered_product",
    "/v1/car_retails/highest_employee_number",
    "/v1/car_retails/employee_names_disputed_orders?status=Disputed",
    "/v1/car_retails/percentage_employees_city?city=Paris",
    "/v1/car_retails/employee_names_job_title?job_title=Sale%20Manager%20(EMEA)",
    "/v1/car_retails/employee_details_country?country=Japan",
    "/v1/car_retails/customer_highest_priced_order?product_name=1939%20Chevrolet%20Deluxe%20Coupe",
    "/v1/car_retails/percentage_payments_customer_year?customer_name=Atelier%20graphique&year=2004",
    "/v1/car_retails/total_profit_order?order_number=10100",
    "/v1/car_retails/total_amount_paid_customer?customer_number=103",
    "/v1/car_retails/total_amount_order?order_number=10100",
    "/v1/car_retails/top_expensive_products",
    "/v1/car_retails/customer_contact_highest_credit_limit?employee_number=1370",
    "/v1/car_retails/total_quantity_ordered_product?product_name=2003%20Harley-Davidson%20Eagle%20Drag%20Bike",
    "/v1/car_retails/shipped_date_highest_price_item",
    "/v1/car_retails/total_quantity_ordered_product_line_year?product_line=motorcycles&year=2004",
    "/v1/car_retails/order_numbers_credit_limit?credit_limit=45300",
    "/v1/car_retails/customer_highest_profit_margin",
    "/v1/car_retails/payments_below_amount_by_year?amount=10000",
    "/v1/car_retails/top_products_ordered_year?year=2003",
    "/v1/car_retails/total_quantity_ordered_sales_reps_product?product_name=1969%20Harley%20Davidson%20Ultimate%20Chopper",
    "/v1/car_retails/employee_names_job_title_city?city=NYC&job_title=Sales%20Rep",
    "/v1/car_retails/customer_details_by_check_number?check_number=GG31455",
    "/v1/car_retails/sum_order_numbers_by_product?product_name=2001%20Ferrari%20Enzo",
    "/v1/car_retails/top_5_products_by_quantity",
    "/v1/car_retails/order_count_by_status_and_country?status=On%20Hold&country=USA",
    "/v1/car_retails/total_sales_amount?product_line=Classic%20Cars&customer_name=Land%20of%20Toys%20Inc.&status=Shipped"
]
