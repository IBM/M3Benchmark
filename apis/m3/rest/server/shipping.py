from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/shipping/shipping.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of shipments for a specific customer in a given year
@app.get("/v1/shipping/count_shipments_customer_year", operation_id="get_count_shipments_customer_year", summary="Retrieves the total number of shipments associated with a particular customer during a specified year. The operation requires the customer's name and the year (in 'YYYY' format) as input parameters to filter the shipment records accordingly.")
async def get_count_shipments_customer_year(cust_name: str = Query(..., description="Customer name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T2.ship_id) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T1.cust_name = ? AND STRFTIME('%Y', T2.ship_date) = ?", (cust_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the total weight of shipments for a specific customer
@app.get("/v1/shipping/total_weight_shipments_customer", operation_id="get_total_weight_shipments_customer", summary="Retrieves the cumulative weight of all shipments associated with a particular customer. The operation requires the customer's name as input to accurately calculate and return the total weight.")
async def get_total_weight_shipments_customer(cust_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT SUM(T2.weight) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T1.cust_name = ?", (cust_name,))
    result = cursor.fetchone()
    if not result:
        return {"total_weight": []}
    return {"total_weight": result[0]}

# Endpoint to get the count of shipments for a specific customer handled by a specific driver
@app.get("/v1/shipping/count_shipments_customer_driver", operation_id="get_count_shipments_customer_driver", summary="Retrieve the total number of shipments associated with a specific customer that were handled by a particular driver. The operation requires the customer's full name and the driver's first and last names as input parameters.")
async def get_count_shipments_customer_driver(cust_name: str = Query(..., description="Customer name"), first_name: str = Query(..., description="Driver's first name"), last_name: str = Query(..., description="Driver's last name")):
    cursor.execute("SELECT COUNT(*) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id INNER JOIN driver AS T3 ON T3.driver_id = T2.driver_id WHERE T1.cust_name = ? AND T3.first_name = ? AND T3.last_name = ?", (cust_name, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers in a specific state
@app.get("/v1/shipping/count_customers_state", operation_id="get_count_customers_state", summary="Retrieves the total number of customers who have made a shipment in a specified state. The state is provided as an input parameter.")
async def get_count_customers_state(state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(T1.cust_id) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T1.state = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the shipment IDs for a specific customer type
@app.get("/v1/shipping/shipment_ids_customer_type", operation_id="get_shipment_ids_customer_type", summary="Retrieves the unique identifiers of all shipments associated with a specific customer type. The customer type is used to filter the shipments and return only those that match the provided type.")
async def get_shipment_ids_customer_type(cust_type: str = Query(..., description="Customer type")):
    cursor.execute("SELECT T2.ship_id FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T1.cust_type = ?", (cust_type,))
    result = cursor.fetchall()
    if not result:
        return {"shipment_ids": []}
    return {"shipment_ids": [row[0] for row in result]}

# Endpoint to get the count of customers with a specific annual revenue and number of shipments in a given year
@app.get("/v1/shipping/count_customers_revenue_shipments_year", operation_id="get_count_customers_revenue_shipments_year", summary="Retrieves the count of customers who have an annual revenue exceeding a specified amount and a minimum number of shipments in a given year. The year, annual revenue, and minimum number of shipments are provided as input parameters.")
async def get_count_customers_revenue_shipments_year(year: str = Query(..., description="Year in 'YYYY' format"), annual_revenue: float = Query(..., description="Annual revenue"), min_shipments: int = Query(..., description="Minimum number of shipments")):
    cursor.execute("SELECT COUNT(COUNTCUSID) FROM ( SELECT COUNT(T1.cust_id) AS COUNTCUSID FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE STRFTIME('%Y', T2.ship_date) = ? AND T1.annual_revenue > ? GROUP BY T1.cust_id HAVING COUNT(T2.ship_id) >= ? ) T3", (year, annual_revenue, min_shipments))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of shipments handled by a specific driver in a given year
@app.get("/v1/shipping/count_shipments_driver_year", operation_id="get_count_shipments_driver_year", summary="Retrieves the total number of shipments managed by a driver in a specified year. The operation requires the driver's first and last names, along with the year in 'YYYY' format. It returns a count of shipments that the driver handled during the given year.")
async def get_count_shipments_driver_year(year: str = Query(..., description="Year in 'YYYY' format"), first_name: str = Query(..., description="Driver's first name"), last_name: str = Query(..., description="Driver's last name")):
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE STRFTIME('%Y', T1.ship_date) = ? AND T2.first_name = ? AND T2.last_name = ?", (year, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the driver with the most shipments in a given year
@app.get("/v1/shipping/top_driver_year", operation_id="get_top_driver_year", summary="Retrieves the driver who handled the most shipments in a specified year. The year must be provided in the 'YYYY' format. The operation returns the first and last name of the top driver.")
async def get_top_driver_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE STRFTIME('%Y', T1.ship_date) = ? GROUP BY T2.first_name, T2.last_name ORDER BY COUNT(*) DESC LIMIT 1", (year,))
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of shipments in a specific state in a given year
@app.get("/v1/shipping/count_shipments_state_year", operation_id="get_count_shipments_state_year", summary="Retrieves the total number of shipments that occurred in a specified state during a given year. The operation requires the year (in 'YYYY' format) and the state as input parameters. The result is a single count value representing the total number of shipments that meet the specified criteria.")
async def get_count_shipments_state_year(year: str = Query(..., description="Year in 'YYYY' format"), state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE STRFTIME('%Y', T1.ship_date) = ? AND T2.state = ?", (year, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the maximum weight of shipments in a specific city
@app.get("/v1/shipping/max_weight_shipments_city", operation_id="get_max_weight_shipments_city", summary="Retrieves the maximum weight of shipments for a specified city. The operation calculates the maximum weight from the shipment records associated with the provided city name.")
async def get_max_weight_shipments_city(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT MAX(T1.weight) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T2.city_name = ?", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"max_weight": []}
    return {"max_weight": result[0]}

# Endpoint to get the difference in total weight between two cities
@app.get("/v1/shipping/weight_difference_between_cities", operation_id="get_weight_difference", summary="Retrieves the difference in total weight of shipments between two specified cities. The operation calculates the sum of weights for shipments in the first city and subtracts the sum of weights for shipments in the second city.")
async def get_weight_difference(city1: str = Query(..., description="First city name"), city2: str = Query(..., description="Second city name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.city_name = ? THEN T1.weight ELSE 0 END) - SUM(CASE WHEN T2.city_name = ? THEN T1.weight ELSE 0 END) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id", (city1, city2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get distinct city names for a given customer
@app.get("/v1/shipping/distinct_cities_for_customer", operation_id="get_distinct_cities", summary="Retrieve a unique list of city names where a given customer has made shipments. The operation requires the customer's name as input to filter the shipment records and identify the distinct cities.")
async def get_distinct_cities(cust_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT DISTINCT T3.city_name FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id INNER JOIN city AS T3 ON T3.city_id = T2.city_id WHERE T1.cust_name = ?", (cust_name,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the average weight of shipments for a given customer
@app.get("/v1/shipping/average_weight_for_customer", operation_id="get_average_weight", summary="Retrieves the average weight of shipments associated with a particular customer. The customer is identified by their name, which is provided as an input parameter.")
async def get_average_weight(cust_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT AVG(T2.weight) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T1.cust_name = ?", (cust_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

# Endpoint to get the percentage of shipments to a specific city within a state
@app.get("/v1/shipping/percentage_shipments_to_city_in_state", operation_id="get_percentage_shipments", summary="Retrieves the percentage of total shipments that are destined for a specific city within a given state. The operation calculates this percentage by counting the number of shipments to the specified city and dividing it by the total number of shipments to the state.")
async def get_percentage_shipments(city_name: str = Query(..., description="City name"), state: str = Query(..., description="State")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.city_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T2.state = ?", (city_name, state))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the make of the truck used in a specific shipment
@app.get("/v1/shipping/truck_make_for_shipment", operation_id="get_truck_make", summary="Retrieves the make of the truck associated with a specific shipment. The operation uses the provided shipment ID to identify the corresponding shipment and returns the make of the truck used for that shipment.")
async def get_truck_make(ship_id: int = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T1.make FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T2.ship_id = ?", (ship_id,))
    result = cursor.fetchone()
    if not result:
        return {"make": []}
    return {"make": result[0]}

# Endpoint to get the count of shipments for the earliest model year of trucks
@app.get("/v1/shipping/count_shipments_earliest_model_year", operation_id="get_count_earliest_model_year", summary="Retrieves the total number of shipments associated with the oldest truck model year available in the database. This operation returns a single value representing the count of shipments for the earliest model year of trucks.")
async def get_count_earliest_model_year():
    cursor.execute("SELECT COUNT(*) FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id GROUP BY T1.model_year ORDER BY T1.model_year ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the customer name for a specific shipment
@app.get("/v1/shipping/customer_name_for_shipment", operation_id="get_customer_name", summary="Retrieves the name of the customer associated with a specific shipment. The operation uses the provided shipment ID to look up the corresponding customer name in the database.")
async def get_customer_name(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T1.cust_name FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T2.ship_id = ?", (ship_id,))
    result = cursor.fetchone()
    if not result:
        return {"customer_name": []}
    return {"customer_name": result[0]}

# Endpoint to get the city name for a specific shipment
@app.get("/v1/shipping/city_name_for_shipment", operation_id="get_city_name", summary="Retrieves the city name associated with a specific shipment. The shipment is identified by its unique shipment ID. The operation returns the name of the city where the shipment is located.")
async def get_city_name(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.city_name FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the driver's name for a specific shipment
@app.get("/v1/shipping/driver_name_for_shipment", operation_id="get_driver_name", summary="Retrieves the first and last name of the driver associated with a specific shipment, identified by its unique shipment ID.")
async def get_driver_name(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchone()
    if not result:
        return {"driver_name": []}
    return {"driver_name": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the driver's name for shipments on a specific date
@app.get("/v1/shipping/driver_name_for_ship_date", operation_id="get_driver_name_by_date", summary="Retrieves the first and last name of the driver responsible for shipments on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_driver_name_by_date(ship_date: str = Query(..., description="Shipment date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.first_name, T2.last_name FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T1.ship_date = ?", (ship_date,))
    result = cursor.fetchall()
    if not result:
        return {"driver_names": []}
    return {"driver_names": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the population of a city based on shipment ID
@app.get("/v1/shipping/city_population_by_shipment_id", operation_id="get_city_population", summary="Retrieves the population of the city associated with the provided shipment ID. This operation fetches the city's population from the database using the shipment ID as a reference.")
async def get_city_population(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.population FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchone()
    if not result:
        return {"population": []}
    return {"population": result[0]}

# Endpoint to get the earliest shipment date for a given state
@app.get("/v1/shipping/earliest_shipment_date_by_state", operation_id="get_earliest_shipment_date", summary="Retrieves the earliest shipment date for a specified state. This operation identifies the minimum shipment date from the shipment table, filtered by the state of the customer. The state is provided as an input parameter.")
async def get_earliest_shipment_date(state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT MIN(T1.ship_date) FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id WHERE T2.state = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"earliest_shipment_date": []}
    return {"earliest_shipment_date": result[0]}

# Endpoint to get the weight of a shipment based on driver's name and shipment date
@app.get("/v1/shipping/shipment_weight_by_driver_name_and_date", operation_id="get_shipment_weight", summary="Retrieve the weight of a shipment associated with a specific driver and shipment date. The driver is identified by their first and last name, and the shipment date is provided in 'YYYY-MM-DD' format.")
async def get_shipment_weight(first_name: str = Query(..., description="Driver's first name"), last_name: str = Query(..., description="Driver's last name"), ship_date: str = Query(..., description="Shipment date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.weight FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T2.first_name = ? AND T2.last_name = ? AND T1.ship_date = ?", (first_name, last_name, ship_date))
    result = cursor.fetchone()
    if not result:
        return {"weight": []}
    return {"weight": result[0]}

# Endpoint to get the area of a city based on shipment ID
@app.get("/v1/shipping/city_area_by_shipment_id", operation_id="get_city_area", summary="Retrieves the area of the city associated with the specified shipment. The shipment ID is used to identify the city and subsequently determine its area.")
async def get_city_area(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.area FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchone()
    if not result:
        return {"area": []}
    return {"area": result[0]}

# Endpoint to get the weight of a shipment based on customer name and shipment date
@app.get("/v1/shipping/shipment_weight_by_customer_name_and_date", operation_id="get_shipment_weight_by_customer", summary="Retrieves the total weight of a shipment associated with a specific customer on a given date. The operation requires the customer's name and the shipment date in 'YYYY-MM-DD' format to accurately locate the shipment and calculate its weight.")
async def get_shipment_weight_by_customer(cust_name: str = Query(..., description="Customer name"), ship_date: str = Query(..., description="Shipment date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.weight FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id WHERE T2.cust_name = ? AND T1.ship_date = ?", (cust_name, ship_date))
    result = cursor.fetchone()
    if not result:
        return {"weight": []}
    return {"weight": result[0]}

# Endpoint to get the area per population ratio of a city based on shipment ID
@app.get("/v1/shipping/area_per_population_by_shipment_id", operation_id="get_area_per_population", summary="Retrieves the ratio of the area to population for the city associated with the provided shipment ID. This operation calculates the ratio by dividing the city's total area by its population, providing a measure of population density.")
async def get_area_per_population(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.area / T2.population FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchone()
    if not result:
        return {"area_per_population": []}
    return {"area_per_population": result[0]}

# Endpoint to get the average number of shipments per truck for a given truck make
@app.get("/v1/shipping/avg_shipments_per_truck_by_make", operation_id="get_avg_shipments_per_truck", summary="Retrieves the average number of shipments per truck for a specific truck make. This operation calculates the ratio of total shipments to the number of unique trucks for the provided make, providing insights into the utilization of trucks from that manufacturer.")
async def get_avg_shipments_per_truck(make: str = Query(..., description="Truck make")):
    cursor.execute("SELECT CAST(COUNT(T2.ship_id) AS REAL) / COUNT(DISTINCT T1.truck_id) FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T1.make = ?", (make,))
    result = cursor.fetchone()
    if not result:
        return {"avg_shipments_per_truck": []}
    return {"avg_shipments_per_truck": result[0]}

# Endpoint to get the weight of the earliest shipment for a given driver's name
@app.get("/v1/shipping/earliest_shipment_weight_by_driver_name", operation_id="get_earliest_shipment_weight", summary="Retrieves the weight of the earliest shipment handled by a driver identified by their first and last names. The driver's name is used to filter the shipment records and the earliest shipment is determined based on the shipment date. The weight of this shipment is then returned.")
async def get_earliest_shipment_weight(first_name: str = Query(..., description="Driver's first name"), last_name: str = Query(..., description="Driver's last name")):
    cursor.execute("SELECT T1.weight FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T2.first_name = ? AND T2.last_name = ? ORDER BY T1.ship_date ASC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"weight": []}
    return {"weight": result[0]}

# Endpoint to get the customer name with the heaviest shipment
@app.get("/v1/shipping/heaviest_shipment_customer", operation_id="get_heaviest_shipment_customer", summary="Retrieves the name of the customer who has the shipment with the highest weight. The operation calculates the weight of all shipments and identifies the customer associated with the heaviest shipment.")
async def get_heaviest_shipment_customer():
    cursor.execute("SELECT T2.cust_name FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id ORDER BY T1.weight DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"cust_name": []}
    return {"cust_name": result[0]}

# Endpoint to get the first driver's name based on the earliest shipment date
@app.get("/v1/shipping/first_driver_by_ship_date", operation_id="get_first_driver_by_ship_date", summary="Retrieves the name of the driver who has the earliest shipment date. The driver's first and last names are returned.")
async def get_first_driver_by_ship_date():
    cursor.execute("SELECT T2.first_name, T2.last_name FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id ORDER BY T1.ship_date ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of shipments for a specific customer
@app.get("/v1/shipping/count_shipments_by_customer", operation_id="get_count_shipments_by_customer", summary="Retrieves the total number of shipments associated with a particular customer. The customer is identified by their name.")
async def get_count_shipments_by_customer(cust_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT COUNT(T2.ship_id) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T1.cust_name = ?", (cust_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of shipments for a specific customer in a specific city
@app.get("/v1/shipping/count_shipments_by_customer_city", operation_id="get_count_shipments_by_customer_city", summary="Retrieves the total number of shipments associated with a specific customer in a given city. The operation requires the customer's name and the city's name as input parameters to accurately calculate the shipment count.")
async def get_count_shipments_by_customer_city(city_name: str = Query(..., description="City name"), cust_name: str = Query(..., description="Customer name")):
    cursor.execute("SELECT COUNT(*) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id INNER JOIN city AS T3 ON T3.city_id = T2.city_id WHERE T3.city_name = ? AND T1.cust_name = ?", (city_name, cust_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of shipments for a specific truck make, ordered by weight
@app.get("/v1/shipping/count_shipments_by_truck_make", operation_id="get_count_shipments_by_truck_make", summary="Retrieves the total number of shipments associated with a specific truck make, sorted by weight in descending order. The response includes the count of shipments for the given truck make, with the heaviest shipment considered first.")
async def get_count_shipments_by_truck_make(make: str = Query(..., description="Truck make")):
    cursor.execute("SELECT COUNT(T2.ship_id) FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T1.make = ? ORDER BY T2.weight DESC LIMIT 1", (make,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the driver with the highest population city
@app.get("/v1/shipping/driver_highest_population_city", operation_id="get_driver_highest_population_city", summary="Retrieves the driver who has the most shipments in the city with the highest population. The driver's first and last name are returned.")
async def get_driver_highest_population_city():
    cursor.execute("SELECT T1.first_name, T1.last_name FROM driver AS T1 INNER JOIN shipment AS T2 ON T1.driver_id = T2.driver_id INNER JOIN city AS T3 ON T3.city_id = T2.city_id GROUP BY T1.first_name, T1.last_name, T3.population HAVING T3.population = MAX(T3.population) ORDER BY COUNT(*) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of shipments for trucks with weight less than a specified value, ordered by model year
@app.get("/v1/shipping/count_shipments_by_truck_weight", operation_id="get_count_shipments_by_truck_weight", summary="Retrieves the total number of shipments associated with trucks that weigh less than the provided weight limit, sorted by the model year of the trucks in ascending order. The result is limited to the first record.")
async def get_count_shipments_by_truck_weight(weight: int = Query(..., description="Weight limit")):
    cursor.execute("SELECT COUNT(*) FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T2.weight < ? ORDER BY T1.model_year ASC LIMIT 1", (weight,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the annual revenue of the customer with the most shipments
@app.get("/v1/shipping/annual_revenue_most_shipments", operation_id="get_annual_revenue_most_shipments", summary="Retrieves the annual revenue of the customer who has made the most shipments. This operation calculates the total number of shipments made by each customer and identifies the customer with the highest count. The annual revenue of this customer is then returned.")
async def get_annual_revenue_most_shipments():
    cursor.execute("SELECT T2.annual_revenue FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id GROUP BY T1.cust_id ORDER BY COUNT(T1.cust_id) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"annual_revenue": []}
    return {"annual_revenue": result[0]}

# Endpoint to get the count of shipments for a specific city, year, and state
@app.get("/v1/shipping/count_shipments_by_city_year_state", operation_id="get_count_shipments_by_city_year_state", summary="Retrieves the total number of shipments for a given city, year, and state. The operation filters shipments based on the specified city name, year, and state, and returns the count of matching records.")
async def get_count_shipments_by_city_year_state(city_name: str = Query(..., description="City name"), year: str = Query(..., description="Year in 'YYYY' format"), state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id INNER JOIN customer AS T3 ON T3.cust_id = T1.cust_id WHERE T2.city_name = ? AND STRFTIME('%Y', T1.ship_date) = ? AND T3.state = ?", (city_name, year, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of shipments for a specific driver in a specific city
@app.get("/v1/shipping/count_shipments_by_driver_city", operation_id="get_count_shipments_by_driver_city", summary="Retrieves the total number of shipments handled by a specific driver in a given city. The operation requires the driver's first and last names, as well as the name of the city, to accurately determine the shipment count.")
async def get_count_shipments_by_driver_city(first_name: str = Query(..., description="Driver's first name"), last_name: str = Query(..., description="Driver's last name"), city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT COUNT(*) FROM driver AS T1 INNER JOIN shipment AS T2 ON T1.driver_id = T2.driver_id INNER JOIN city AS T3 ON T3.city_id = T2.city_id WHERE T1.first_name = ? AND T1.last_name = ? AND T3.city_name = ?", (first_name, last_name, city_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of shipments for the city with the highest area per population
@app.get("/v1/shipping/count_shipments_highest_area_per_population", operation_id="get_count_shipments_highest_area_per_population", summary="Retrieves the total number of shipments associated with the city that has the highest area-to-population ratio. This operation provides a quantitative measure of shipping activity in the most spacious city per capita.")
async def get_count_shipments_highest_area_per_population():
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id ORDER BY T2.area / T2.population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of customers of a specific type in a given state
@app.get("/v1/shipping/customer_percentage_by_type_and_state", operation_id="get_customer_percentage", summary="Retrieves the percentage of customers of a specific type residing in a given state. The operation calculates this percentage by summing the number of customers of the specified type in the provided state and dividing it by the total number of customers in that state.")
async def get_customer_percentage(cust_type: str = Query(..., description="Customer type"), state: str = Query(..., description="State")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN cust_type = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM customer WHERE state = ?", (cust_type, state))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the total weight of shipments to a specific city in a given year
@app.get("/v1/shipping/total_weight_by_city_and_year", operation_id="get_total_weight", summary="Retrieves the cumulative weight of all shipments sent to a specified city during a particular year. The city is identified by its name, and the year is provided in the 'YYYY' format.")
async def get_total_weight(city_name: str = Query(..., description="City name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(T1.weight) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T2.city_name = ? AND STRFTIME('%Y', T1.ship_date) = ?", (city_name, year))
    result = cursor.fetchone()
    if not result:
        return {"total_weight": []}
    return {"total_weight": result[0]}

# Endpoint to get the total weight of shipments by a specific truck make in a given year
@app.get("/v1/shipping/total_weight_by_truck_make_and_year", operation_id="get_total_weight_by_truck", summary="Retrieves the total weight of shipments handled by a specific truck make in a given year, sorted by the most recent model year. The make and year parameters are required to filter the results.")
async def get_total_weight_by_truck(make: str = Query(..., description="Truck make"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT SUM(T2.weight) FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T1.make = ? AND STRFTIME('%Y', T2.ship_date) = ? ORDER BY T1.model_year DESC LIMIT 1", (make, year))
    result = cursor.fetchone()
    if not result:
        return {"total_weight": []}
    return {"total_weight": result[0]}

# Endpoint to get the heaviest shipment to a specific city
@app.get("/v1/shipping/heaviest_shipment_by_city", operation_id="get_heaviest_shipment", summary="Retrieves the heaviest shipment to a specified city, including the shipment weight and the name of the customer who ordered it. The city is identified by its name.")
async def get_heaviest_shipment(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.weight, T2.cust_name FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id INNER JOIN city AS T3 ON T3.city_id = T1.city_id WHERE T3.city_name = ? ORDER BY T1.weight DESC LIMIT 1", (city_name,))
    result = cursor.fetchone()
    if not result:
        return {"heaviest_shipment": []}
    return {"heaviest_shipment": {"weight": result[0], "customer_name": result[1]}}

# Endpoint to get driver details for shipments to a specific city in a given month and year
@app.get("/v1/shipping/driver_details_by_city_and_month", operation_id="get_driver_details", summary="Retrieves the first and last names of drivers who have made deliveries to a specified city during a given month and year. The city is identified by its name, and the month and year are determined by the shipment date in the 'YYYY-MM%' format.")
async def get_driver_details(city_name: str = Query(..., description="City name"), ship_date: str = Query(..., description="Ship date in 'YYYY-MM%' format")):
    cursor.execute("SELECT T3.first_name, T3.last_name FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id INNER JOIN driver AS T3 ON T3.driver_id = T1.driver_id WHERE T2.city_name = ? AND T1.ship_date LIKE ?", (city_name, ship_date))
    result = cursor.fetchall()
    if not result:
        return {"driver_details": []}
    return {"driver_details": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get customer names for shipments to a specific city
@app.get("/v1/shipping/customer_names_by_city", operation_id="get_customer_names", summary="Retrieves the names of customers who have shipments to a specified city. The operation filters shipments by the provided city name and returns the corresponding customer names.")
async def get_customer_names(city_name: str = Query(..., description="City name")):
    cursor.execute("SELECT T2.cust_name FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id INNER JOIN city AS T3 ON T3.city_id = T1.city_id WHERE T3.city_name = ?", (city_name,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get the count of shipments by a specific driver to a specific city in a given year
@app.get("/v1/shipping/shipment_count_by_driver_city_and_year", operation_id="get_shipment_count", summary="Retrieve the total number of shipments handled by a specific driver in a particular city during a given year. The operation requires the driver's first and last names, the city name, and the year in 'YYYY' format to accurately filter and count the shipments.")
async def get_shipment_count(first_name: str = Query(..., description="Driver's first name"), last_name: str = Query(..., description="Driver's last name"), city_name: str = Query(..., description="City name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id INNER JOIN driver AS T3 ON T3.driver_id = T1.driver_id WHERE T3.first_name = ? AND T3.last_name = ? AND T2.city_name = ? AND STRFTIME('%Y', T1.ship_date) = ?", (first_name, last_name, city_name, year))
    result = cursor.fetchone()
    if not result:
        return {"shipment_count": []}
    return {"shipment_count": result[0]}

# Endpoint to get the average number of shipments per driver per month
@app.get("/v1/shipping/average_shipments_per_driver_per_month", operation_id="get_average_shipments_per_driver", summary="Retrieves the average number of shipments handled by each driver per month. This operation calculates the ratio of total shipments to the total number of drivers, considering a 12-month period. The result provides a monthly average of shipments per driver, offering insights into driver workload distribution.")
async def get_average_shipments_per_driver():
    cursor.execute("SELECT CAST(COUNT(*) AS REAL) / (12 * COUNT(T2.driver_id)) FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id")
    result = cursor.fetchone()
    if not result:
        return {"average_shipments": []}
    return {"average_shipments": result[0]}

# Endpoint to get the percentage of shipments with weight greater than or equal to a specified value for a given customer in a specific year
@app.get("/v1/shipping/percentage_heavy_shipments", operation_id="get_percentage_heavy_shipments", summary="Retrieves the percentage of shipments for a specific customer in a given year that have a weight equal to or greater than a specified minimum value. This operation calculates the proportion of qualifying shipments by comparing the total count of shipments meeting the weight criterion to the overall number of shipments for the customer in the specified year.")
async def get_percentage_heavy_shipments(min_weight: int = Query(..., description="Minimum weight of the shipment"), cust_name: str = Query(..., description="Customer name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.weight >= ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id WHERE T2.cust_name = ? AND STRFTIME('%Y', T1.ship_date) = ?", (min_weight, cust_name, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of shipments for a specific customer handled by a driver with a given first and last name
@app.get("/v1/shipping/percentage_shipments_by_driver", operation_id="get_percentage_shipments_by_driver", summary="Retrieve the percentage of shipments associated with a specific customer that were handled by a driver with the given first and last name. This operation calculates the proportion of shipments for the specified customer that were managed by the identified driver, providing insights into the driver's contribution to the customer's shipment volume.")
async def get_percentage_shipments_by_driver(cust_name: str = Query(..., description="Customer name"), first_name: str = Query(..., description="Driver's first name"), last_name: str = Query(..., description="Driver's last name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.cust_name = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) AS per FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id INNER JOIN customer AS T3 ON T3.cust_id = T1.cust_id WHERE T2.first_name = ? AND T2.last_name = ?", (cust_name, first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of cities in a given state with total shipment weight greater than a specified value
@app.get("/v1/shipping/count_cities_by_state_weight", operation_id="get_count_cities_by_state_weight", summary="Retrieves the count of cities in a specified state where the total shipment weight surpasses a given minimum value. This operation provides insights into the distribution of shipment weights across cities within a state.")
async def get_count_cities_by_state_weight(state: str = Query(..., description="State name"), min_weight: int = Query(..., description="Minimum total shipment weight")):
    cursor.execute("SELECT COUNT(*) FROM ( SELECT T2.city_id AS CITYID FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T2.state = ? GROUP BY T2.city_id HAVING SUM(T1.weight) > ? )", (state, min_weight))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of shipments in cities with a population greater than a specified value in a given year
@app.get("/v1/shipping/count_shipments_by_city_population_year", operation_id="get_count_shipments_by_city_population_year", summary="Retrieves the total number of shipments that occurred in cities with a population exceeding a specified threshold during a particular year. The operation filters cities based on their population and shipments based on the year they were made, then returns the count of qualifying shipments.")
async def get_count_shipments_by_city_population_year(min_population: int = Query(..., description="Minimum population of the city"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM city AS T1 INNER JOIN shipment AS T2 ON T1.city_id = T2.city_id WHERE T1.population > ? AND STRFTIME('%Y', T2.ship_date) = ?", (min_population, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the weights of shipments for trucks of a specific make
@app.get("/v1/shipping/shipment_weights_by_truck_make", operation_id="get_shipment_weights_by_truck_make", summary="Retrieve the total weights of shipments associated with a specific truck make. The operation filters shipments based on the provided truck make and returns the aggregated weight of these shipments.")
async def get_shipment_weights_by_truck_make(make: str = Query(..., description="Make of the truck")):
    cursor.execute("SELECT T2.weight FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE make = ?", (make,))
    result = cursor.fetchall()
    if not result:
        return {"weights": []}
    return {"weights": [row[0] for row in result]}

# Endpoint to get the model year of trucks used in a specific shipment
@app.get("/v1/shipping/truck_model_year_by_shipment", operation_id="get_truck_model_year_by_shipment", summary="Retrieves the model year of trucks associated with a specific shipment. The shipment is identified by its unique ID, which is used to look up the corresponding truck IDs and their respective model years.")
async def get_truck_model_year_by_shipment(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T1.model_year FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T2.ship_id = ?", (ship_id,))
    result = cursor.fetchall()
    if not result:
        return {"model_years": []}
    return {"model_years": [row[0] for row in result]}

# Endpoint to get the state of the driver for a specific shipment
@app.get("/v1/shipping/driver_state_by_shipment", operation_id="get_driver_state_by_shipment", summary="Retrieves the current state of the driver associated with a specific shipment. The shipment ID is required to identify the driver and determine their state.")
async def get_driver_state_by_shipment(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.state FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get the addresses of drivers who have handled shipments with a total weight greater than a specified value
@app.get("/v1/shipping/driver_addresses_by_total_weight", operation_id="get_driver_addresses_by_total_weight", summary="Retrieve the addresses of drivers who have managed shipments with a combined weight surpassing a specified minimum threshold.")
async def get_driver_addresses_by_total_weight(min_weight: int = Query(..., description="Minimum total shipment weight")):
    cursor.execute("SELECT T2.address FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id GROUP BY T2.driver_id HAVING SUM(T1.weight) > ?", (min_weight,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [row[0] for row in result]}

# Endpoint to get the average annual revenue of customers based on shipment weight
@app.get("/v1/shipping/avg_annual_revenue_by_weight", operation_id="get_avg_annual_revenue_by_weight", summary="Retrieves the average annual revenue of customers who have shipments weighing less than the specified maximum weight. This operation calculates the average revenue from a dataset that combines customer and shipment information, filtering for shipments under the provided weight limit.")
async def get_avg_annual_revenue_by_weight(weight: int = Query(..., description="Maximum weight of the shipment")):
    cursor.execute("SELECT AVG(T1.annual_revenue) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T2.weight < ?", (weight,))
    result = cursor.fetchone()
    if not result:
        return {"average_revenue": []}
    return {"average_revenue": result[0]}

# Endpoint to get the percentage of shipments below a certain weight for a specific customer type
@app.get("/v1/shipping/percentage_shipments_below_weight", operation_id="get_percentage_shipments_below_weight", summary="Retrieves the percentage of shipments with a weight less than the specified maximum for a given customer type. This operation calculates the proportion of shipments that meet the weight criteria for a specific customer type, providing insights into the distribution of shipment weights for that customer segment.")
async def get_percentage_shipments_below_weight(weight: int = Query(..., description="Maximum weight of the shipment"), cust_type: str = Query(..., description="Customer type")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.weight < ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T1.cust_type = ?", (weight, cust_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get driver last name based on shipment ID
@app.get("/v1/shipping/driver_last_name_by_ship_id", operation_id="get_driver_last_name_by_ship_id", summary="Retrieves the last name of the driver associated with the specified shipment. The shipment is identified by its unique ID.")
async def get_driver_last_name_by_ship_id(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.last_name FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": result}

# Endpoint to get driver phone numbers based on total shipment weight
@app.get("/v1/shipping/driver_phones_by_total_weight", operation_id="get_driver_phones_by_total_weight", summary="Retrieve the contact information of drivers who have handled shipments collectively weighing more than the specified total weight. This operation provides a means to identify drivers with substantial shipping experience based on the cumulative weight of their handled shipments.")
async def get_driver_phones_by_total_weight(total_weight: int = Query(..., description="Total shipment weight")):
    cursor.execute("SELECT T2.phone FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id GROUP BY T2.driver_id HAVING SUM(T1.weight) > ?", (total_weight,))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": result}

# Endpoint to get truck make and model year based on shipment ID
@app.get("/v1/shipping/truck_details_by_ship_id", operation_id="get_truck_details_by_ship_id", summary="Retrieves the make and model year of trucks associated with a specific shipment. The shipment is identified by its unique ID.")
async def get_truck_details_by_ship_id(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T1.make, T1.model_year FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id WHERE T2.ship_id = ?", (ship_id,))
    result = cursor.fetchall()
    if not result:
        return {"truck_details": []}
    return {"truck_details": result}

# Endpoint to get the count of trucks based on model year
@app.get("/v1/shipping/truck_count_by_model_year", operation_id="get_truck_count_by_model_year", summary="Retrieves the total number of trucks in the system that match the specified model year. The model year is a required input parameter.")
async def get_truck_count_by_model_year(model_year: int = Query(..., description="Model year of the truck")):
    cursor.execute("SELECT COUNT(truck_id) FROM truck WHERE model_year = ?", (model_year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on customer type
@app.get("/v1/shipping/customer_count_by_type", operation_id="get_customer_count_by_type", summary="Retrieves the total number of customers categorized by a specific customer type. The customer type is provided as an input parameter.")
async def get_customer_count_by_type(cust_type: str = Query(..., description="Customer type")):
    cursor.execute("SELECT COUNT(*) FROM customer WHERE cust_type = ?", (cust_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of customers based on customer type and state
@app.get("/v1/shipping/customer_count_by_type_and_state", operation_id="get_customer_count_by_type_and_state", summary="Retrieves the total number of customers categorized by a specific customer type and state. The operation filters customers based on the provided customer type and state, then returns the count of customers that match the criteria.")
async def get_customer_count_by_type_and_state(cust_type: str = Query(..., description="Customer type"), state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(*) FROM customer WHERE cust_type = ? AND state = ?", (cust_type, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cities based on state
@app.get("/v1/shipping/city_count_by_state", operation_id="get_city_count_by_state", summary="Retrieves the total number of cities within a specified state. The state is provided as an input parameter.")
async def get_city_count_by_state(state: str = Query(..., description="State")):
    cursor.execute("SELECT COUNT(*) FROM city WHERE state = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city with the highest population in a given state
@app.get("/v1/shipping/city_with_highest_population", operation_id="get_city_with_highest_population", summary="Retrieves the name of the city with the highest population in the specified state. The operation filters the cities based on the provided state and identifies the one with the maximum population.")
async def get_city_with_highest_population(state: str = Query(..., description="State to filter the city by")):
    cursor.execute("SELECT city_name FROM city WHERE state = ? AND population = ( SELECT MAX(population) FROM city WHERE state = ? )", (state, state))
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get the annual revenue of a customer by name
@app.get("/v1/shipping/customer_annual_revenue", operation_id="get_customer_annual_revenue", summary="Retrieves the annual revenue of a specific customer. The customer is identified by their name, which is provided as an input parameter. The operation returns the annual revenue value associated with the customer.")
async def get_customer_annual_revenue(cust_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT annual_revenue FROM customer WHERE cust_name = ?", (cust_name,))
    result = cursor.fetchone()
    if not result:
        return {"annual_revenue": []}
    return {"annual_revenue": result[0]}

# Endpoint to get the driver of the lightest shipment
@app.get("/v1/shipping/driver_of_lightest_shipment", operation_id="get_driver_of_lightest_shipment", summary="Retrieves the first name and last name of the driver responsible for the shipment with the lowest weight. The shipment weight is used as the primary sorting criterion to determine the lightest shipment.")
async def get_driver_of_lightest_shipment():
    cursor.execute("SELECT T2.first_name, T2.last_name FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id ORDER BY T1.weight ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"driver": []}
    return {"driver": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of shipments in a given year and state
@app.get("/v1/shipping/count_shipments_year_state", operation_id="get_count_shipments_year_state", summary="Retrieves the total number of shipments for a specific year and state. The operation filters shipments based on the provided year and state, and returns the count of shipments that meet the criteria.")
async def get_count_shipments_year_state(year: str = Query(..., description="Year in 'YYYY' format"), state: str = Query(..., description="State to filter the shipments by")):
    cursor.execute("SELECT COUNT(*) AS per FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE STRFTIME('%Y', T2.ship_date) = ? AND T1.state = ?", (year, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct truck makes driven by a specific driver
@app.get("/v1/shipping/distinct_truck_makes_by_driver", operation_id="get_distinct_truck_makes_by_driver", summary="Retrieve a list of unique truck makes that a specific driver has operated. The operation requires the driver's first and last names as input parameters to filter the results.")
async def get_distinct_truck_makes_by_driver(first_name: str = Query(..., description="First name of the driver"), last_name: str = Query(..., description="Last name of the driver")):
    cursor.execute("SELECT DISTINCT T1.make FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id INNER JOIN driver AS T3 ON T3.driver_id = T2.driver_id WHERE T3.first_name = ? AND T3.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"truck_makes": []}
    return {"truck_makes": [row[0] for row in result]}

# Endpoint to get customer names with shipments in a specific month and year
@app.get("/v1/shipping/customer_names_by_shipment_date", operation_id="get_customer_names_by_shipment_date", summary="Retrieves the names of customers who have shipments in a specified month and year. The input parameter is used to filter the shipment date, allowing for a targeted search of customer names.")
async def get_customer_names_by_shipment_date(ship_date: str = Query(..., description="Shipment date in 'YYYY-MM%' format")):
    cursor.execute("SELECT T1.cust_name FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id WHERE T2.ship_date LIKE ?", (ship_date,))
    result = cursor.fetchall()
    if not result:
        return {"customer_names": []}
    return {"customer_names": [row[0] for row in result]}

# Endpoint to get truck make and driver details for a specific customer
@app.get("/v1/shipping/truck_make_driver_details_by_customer", operation_id="get_truck_make_driver_details_by_customer", summary="Retrieves the make of trucks and the first and last names of the drivers associated with a specific customer's shipments. The customer is identified by their name.")
async def get_truck_make_driver_details_by_customer(cust_name: str = Query(..., description="Name of the customer")):
    cursor.execute("SELECT T3.make, T4.first_name, T4.last_name FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id INNER JOIN truck AS T3 ON T3.truck_id = T2.truck_id INNER JOIN driver AS T4 ON T4.driver_id = T2.driver_id WHERE T1.cust_name = ?", (cust_name,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": [{"make": row[0], "first_name": row[1], "last_name": row[2]} for row in result]}

# Endpoint to get the heaviest shipment ID for a specific driver
@app.get("/v1/shipping/heaviest_shipment_by_driver", operation_id="get_heaviest_shipment_by_driver", summary="Retrieves the ID of the heaviest shipment handled by a specific driver. The driver is identified by their first and last names. The shipment with the highest weight is returned.")
async def get_heaviest_shipment_by_driver(first_name: str = Query(..., description="First name of the driver"), last_name: str = Query(..., description="Last name of the driver")):
    cursor.execute("SELECT T1.ship_id FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T2.first_name = ? AND T2.last_name = ? ORDER BY T1.weight DESC LIMIT 1", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"ship_id": []}
    return {"ship_id": result[0]}

# Endpoint to get the count of shipments for a specific driver in a specific city and year
@app.get("/v1/shipping/count_shipments_driver_city_year", operation_id="get_count_shipments_driver_city_year", summary="Retrieves the total number of shipments handled by a specific driver in a given city during a particular year. The driver is identified by their first and last names, while the city is specified by its name. The year is provided in the 'YYYY' format.")
async def get_count_shipments_driver_city_year(first_name: str = Query(..., description="First name of the driver"), last_name: str = Query(..., description="Last name of the driver"), city_name: str = Query(..., description="Name of the city"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM city AS T1 INNER JOIN shipment AS T2 ON T1.city_id = T2.city_id INNER JOIN driver AS T3 ON T3.driver_id = T2.driver_id WHERE T3.first_name = ? AND T3.last_name = ? AND T1.city_name = ? AND STRFTIME('%Y', T2.ship_date) = ?", (first_name, last_name, city_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the state with the most shipments for a specific truck make in a given year
@app.get("/v1/shipping/state_most_shipments_truck_make_year", operation_id="get_state_most_shipments_truck_make_year", summary="Retrieve the state with the highest number of shipments for a specific truck make in a given year. The operation considers three truck makes and their corresponding states. The year is provided in the 'YYYY' format.")
async def get_state_most_shipments_truck_make_year(make1: str = Query(..., description="First truck make"), state1: str = Query(..., description="State for the first truck make"), make2: str = Query(..., description="Second truck make"), state2: str = Query(..., description="State for the second truck make"), make3: str = Query(..., description="Third truck make"), state3: str = Query(..., description="State for the third truck make"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CASE WHEN T2.make = ? THEN ? WHEN T2.make = ? THEN ? WHEN T2.make = ? THEN ? END AS 'result' FROM shipment AS T1 INNER JOIN truck AS T2 ON T1.truck_id = T2.truck_id WHERE CAST(T1.ship_date AS DATE) = ? GROUP BY T2.make ORDER BY COUNT(T1.ship_id) DESC LIMIT 1", (make1, state1, make2, state2, make3, state3, year))
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the count of cities in a specific state with the lowest population
@app.get("/v1/shipping/count_cities_by_state", operation_id="get_count_cities_by_state", summary="Retrieves the number of cities in a specified state that has the lowest population. The state is identified by its name. The count is determined by considering the cities associated with the customers and their shipments.")
async def get_count_cities_by_state(state: str = Query(..., description="State name")):
    cursor.execute("SELECT COUNT(T3.city_name) FROM customer AS T1 INNER JOIN shipment AS T2 ON T1.cust_id = T2.cust_id INNER JOIN city AS T3 ON T3.city_id = T2.city_id WHERE T3.state = ? ORDER BY T3.population ASC LIMIT 1", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the city name with the heaviest shipment
@app.get("/v1/shipping/heaviest_shipment_city", operation_id="get_heaviest_shipment_city", summary="Retrieves the name of the city with the heaviest total shipment weight. The operation calculates the total weight of shipments for each city and returns the name of the city with the highest total weight.")
async def get_heaviest_shipment_city():
    cursor.execute("SELECT T2.city_name FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id ORDER BY T1.weight DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"city_name": []}
    return {"city_name": result[0]}

# Endpoint to get distinct city names for shipments by a specific driver
@app.get("/v1/shipping/distinct_cities_by_driver", operation_id="get_distinct_cities_by_driver", summary="Retrieve a unique list of city names where a driver with the specified first and last names has made deliveries. This operation returns the distinct city names associated with shipments handled by the driver, providing insights into their delivery coverage.")
async def get_distinct_cities_by_driver(first_name: str = Query(..., description="First name of the driver"), last_name: str = Query(..., description="Last name of the driver")):
    cursor.execute("SELECT DISTINCT T3.city_name FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id INNER JOIN city AS T3 ON T1.city_id = T3.city_id WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"city_names": []}
    return {"city_names": [row[0] for row in result]}

# Endpoint to get the percentage of shipments by a specific driver in a given year
@app.get("/v1/shipping/percentage_shipments_by_driver_year", operation_id="get_percentage_shipments_by_driver_year", summary="Retrieves the percentage of shipments handled by a specific driver in a given year. The calculation is based on the total number of shipments in the specified year. The driver is identified by their first and last names.")
async def get_percentage_shipments_by_driver_year(first_name: str = Query(..., description="First name of the driver"), last_name: str = Query(..., description="Last name of the driver"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.first_name = ? AND T2.last_name = ? THEN T1.ship_id ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE STRFTIME('%Y', T1.ship_date) = ?", (first_name, last_name, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of total weight shipped by a specific driver in a given year
@app.get("/v1/shipping/percentage_weight_by_driver_year", operation_id="get_percentage_weight_by_driver_year", summary="Retrieves the proportion of the total weight shipped by a specific driver in a given year. The calculation is based on the driver's first and last name, and the year of the shipment. The result is expressed as a percentage.")
async def get_percentage_weight_by_driver_year(first_name: str = Query(..., description="First name of the driver"), last_name: str = Query(..., description="Last name of the driver"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.first_name = ? AND T2.last_name = ? THEN T1.weight ELSE 0 END) AS REAL) * 100 / SUM(T1.weight) FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE STRFTIME('%Y', T1.ship_date) = ?", (first_name, last_name, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of shipments by a specific driver
@app.get("/v1/shipping/count_shipments_by_driver", operation_id="get_count_shipments_by_driver", summary="Retrieves the total number of shipments associated with a specific driver, identified by the provided driver_id.")
async def get_count_shipments_by_driver(driver_id: int = Query(..., description="Driver ID")):
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T1.driver_id = ?", (driver_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the shipment ID with the largest city area
@app.get("/v1/shipping/largest_city_area_shipment", operation_id="get_largest_city_area_shipment", summary="Retrieves the shipment ID associated with the city that has the largest area. This operation returns the shipment ID that corresponds to the city with the highest area value, as determined by the area attribute of the city table. The result is obtained by joining the shipment and city tables on the city_id field and ordering the results by the area in descending order. The top result is then returned.")
async def get_largest_city_area_shipment():
    cursor.execute("SELECT T1.ship_id FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id ORDER BY T2.area DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ship_id": []}
    return {"ship_id": result[0]}

# Endpoint to get the driver's name for the shipment to the city with the smallest population
@app.get("/v1/shipping/driver_smallest_city_population", operation_id="get_driver_smallest_city_population", summary="Retrieves the name of the driver who has the shipment to the city with the smallest population. The operation returns the first name and last name of the driver. The city population is used to determine the smallest populated city.")
async def get_driver_smallest_city_population():
    cursor.execute("SELECT T3.first_name, T3.last_name FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id INNER JOIN driver AS T3 ON T3.driver_id = T1.driver_id ORDER BY T2.population ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"driver_name": []}
    return {"driver_name": {"first_name": result[0], "last_name": result[1]}}

# Endpoint to get the count of shipments to a specific city and state within a weight range
@app.get("/v1/shipping/count_shipments_city_state_weight_range", operation_id="get_count_shipments_city_state_weight_range", summary="Retrieves the total number of shipments to a specified city and state that fall within a given weight range. The city and state are identified by their names, while the weight range is defined by a minimum and maximum value.")
async def get_count_shipments_city_state_weight_range(city_name: str = Query(..., description="City name"), state: str = Query(..., description="State name"), min_weight: int = Query(..., description="Minimum weight"), max_weight: int = Query(..., description="Maximum weight")):
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id WHERE T2.city_name = ? AND T2.state = ? AND T1.weight BETWEEN ? AND ?", (city_name, state, min_weight, max_weight))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the driver address for a shipment based on customer address, city, and state
@app.get("/v1/shipping/driver_address_by_customer_location", operation_id="get_driver_address", summary="Get the driver address for a shipment based on customer address, city, and state")
async def get_driver_address(address: str = Query(..., description="Customer address"), city: str = Query(..., description="Customer city"), state: str = Query(..., description="Customer state")):
    cursor.execute("SELECT T3.address FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id INNER JOIN driver AS T3 ON T3.driver_id = T1.driver_id WHERE T2.address = ? AND T2.city = ? AND T2.state = ?", (address, city, state))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get the count of shipments by driver name and year
@app.get("/v1/shipping/count_shipments_by_driver_and_year", operation_id="get_count_shipments", summary="Retrieves the total number of shipments handled by a specific driver in a given year. The driver is identified by their first and last names. The year is specified in the 'YYYY' format.")
async def get_count_shipments(first_name: str = Query(..., description="Driver first name"), last_name: str = Query(..., description="Driver last name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T2.first_name = ? AND T2.last_name = ? AND STRFTIME('%Y', T1.ship_date) = ?", (first_name, last_name, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the driver address by shipment ID
@app.get("/v1/shipping/driver_address_by_shipment_id", operation_id="get_driver_address_by_shipment_id", summary="Retrieves the address of the driver associated with the specified shipment ID. The operation returns the address of the driver who is handling the shipment identified by the provided shipment ID.")
async def get_driver_address_by_shipment_id(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.address FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchone()
    if not result:
        return {"address": []}
    return {"address": result[0]}

# Endpoint to get the count of shipments by customer state
@app.get("/v1/shipping/count_shipments_by_customer_state", operation_id="get_count_shipments_by_state", summary="Retrieves the total number of shipments associated with customers from a specific state. The state is provided as an input parameter.")
async def get_count_shipments_by_state(state: str = Query(..., description="Customer state")):
    cursor.execute("SELECT COUNT(*) FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id WHERE T2.state = ?", (state,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the shipment ID of the most populous city
@app.get("/v1/shipping/shipment_id_most_populous_city", operation_id="get_shipment_id_most_populous_city", summary="Retrieves the shipment ID associated with the city that has the highest population. This operation returns the shipment ID of the most populous city by joining the shipment and city tables and ordering the results by population in descending order. The shipment ID of the top result is then returned.")
async def get_shipment_id_most_populous_city():
    cursor.execute("SELECT T1.ship_id FROM shipment AS T1 INNER JOIN city AS T2 ON T1.city_id = T2.city_id ORDER BY T2.population DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"ship_id": []}
    return {"ship_id": result[0]}

# Endpoint to get shipment weights based on customer annual revenue
@app.get("/v1/shipping/shipment_weights_by_annual_revenue", operation_id="get_shipment_weights_by_annual_revenue", summary="Retrieves the total weight of shipments associated with a customer, filtered by the customer's annual revenue. The response provides a comprehensive view of shipment weights for customers within a specific revenue bracket.")
async def get_shipment_weights_by_annual_revenue(annual_revenue: int = Query(..., description="Annual revenue of the customer")):
    cursor.execute("SELECT T1.weight FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id WHERE T2.annual_revenue = ?", (annual_revenue,))
    result = cursor.fetchall()
    if not result:
        return {"weights": []}
    return {"weights": result}

# Endpoint to get customer address based on shipment ID
@app.get("/v1/shipping/customer_address_by_ship_id", operation_id="get_customer_address_by_ship_id", summary="Retrieves the address of the customer associated with the provided shipment ID. This operation fetches the customer's address from the database by joining the shipment and customer tables using the shipment ID.")
async def get_customer_address_by_ship_id(ship_id: str = Query(..., description="Shipment ID")):
    cursor.execute("SELECT T2.address FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id WHERE T1.ship_id = ?", (ship_id,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": result}

# Endpoint to get the percentage of shipments in a specific year and state
@app.get("/v1/shipping/percentage_shipments_by_year_state", operation_id="get_percentage_shipments_by_year_state", summary="Retrieves the percentage of total shipments that occurred in a specific year and state. The year is provided in 'YYYY' format, and the state is identified by its abbreviation. This operation calculates the percentage by dividing the number of shipments in the given year and state by the total number of shipments, then multiplying by 100.")
async def get_percentage_shipments_by_year_state(year: str = Query(..., description="Year in 'YYYY' format"), state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN STRFTIME('%Y', T1.ship_date) = ? THEN 1 ELSE 0 END) AS REAL ) * 100 / COUNT(*) FROM shipment AS T1 INNER JOIN customer AS T2 ON T1.cust_id = T2.cust_id WHERE T2.state = ?", (year, state))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in the number of trucks between two model years
@app.get("/v1/shipping/truck_count_difference_by_model_year", operation_id="get_truck_count_difference_by_model_year", summary="Retrieve the difference in the total number of trucks used in shipments between two specified model years. This operation compares the count of trucks from the first model year with the count from the second model year, providing a numerical difference.")
async def get_truck_count_difference_by_model_year(model_year_1: str = Query(..., description="First model year"), model_year_2: str = Query(..., description="Second model year")):
    cursor.execute("SELECT SUM(CASE WHEN T1.model_year = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T1.model_year = ? THEN 1 ELSE 0 END) FROM truck AS T1 INNER JOIN shipment AS T2 ON T1.truck_id = T2.truck_id", (model_year_1, model_year_2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get driver details for shipments heavier than 95% of the average weight
@app.get("/v1/shipping/driver_details_heavy_shipments", operation_id="get_driver_details_heavy_shipments", summary="Retrieves the full names of drivers who have handled shipments that are heavier than 95% of the average shipment weight. This operation provides a list of drivers who have managed exceptionally heavy shipments, which can be useful for recognizing high-capacity drivers or identifying potential safety concerns.")
async def get_driver_details_heavy_shipments():
    cursor.execute("SELECT T2.first_name, T2.last_name FROM shipment AS T1 INNER JOIN driver AS T2 ON T1.driver_id = T2.driver_id WHERE T1.weight * 100 > ( SELECT 95 * AVG(weight) FROM shipment )")
    result = cursor.fetchall()
    if not result:
        return {"drivers": []}
    return {"drivers": result}

api_calls = [
    "/v1/shipping/count_shipments_customer_year?cust_name=S%20K%20L%20Enterprises%20Inc&year=2017",
    "/v1/shipping/total_weight_shipments_customer?cust_name=S%20K%20L%20Enterprises%20Inc",
    "/v1/shipping/count_shipments_customer_driver?cust_name=S%20K%20L%20Enterprises%20Inc&first_name=Sue&last_name=Newell",
    "/v1/shipping/count_customers_state?state=FL",
    "/v1/shipping/shipment_ids_customer_type?cust_type=retailer",
    "/v1/shipping/count_customers_revenue_shipments_year?year=2017&annual_revenue=30000000&min_shipments=1",
    "/v1/shipping/count_shipments_driver_year?year=2017&first_name=Sue&last_name=Newell",
    "/v1/shipping/top_driver_year?year=2017",
    "/v1/shipping/count_shipments_state_year?year=2017&state=New%20Jersey",
    "/v1/shipping/max_weight_shipments_city?city_name=New%20York",
    "/v1/shipping/weight_difference_between_cities?city1=New%20York&city2=Chicago",
    "/v1/shipping/distinct_cities_for_customer?cust_name=S%20K%20L%20Enterprises%20Inc",
    "/v1/shipping/average_weight_for_customer?cust_name=S%20K%20L%20Enterprises%20Inc",
    "/v1/shipping/percentage_shipments_to_city_in_state?city_name=Jacksonville&state=Florida",
    "/v1/shipping/truck_make_for_shipment?ship_id=1045",
    "/v1/shipping/count_shipments_earliest_model_year",
    "/v1/shipping/customer_name_for_shipment?ship_id=1275",
    "/v1/shipping/city_name_for_shipment?ship_id=1701",
    "/v1/shipping/driver_name_for_shipment?ship_id=1021",
    "/v1/shipping/driver_name_for_ship_date?ship_date=2017-11-05",
    "/v1/shipping/city_population_by_shipment_id?ship_id=1398",
    "/v1/shipping/earliest_shipment_date_by_state?state=SC",
    "/v1/shipping/shipment_weight_by_driver_name_and_date?first_name=Leszek&last_name=Kieltyka&ship_date=2017-09-25",
    "/v1/shipping/city_area_by_shipment_id?ship_id=1346",
    "/v1/shipping/shipment_weight_by_customer_name_and_date?cust_name=U-haul%20Center%20Of%20N%20Syracuse&ship_date=2016-09-21",
    "/v1/shipping/area_per_population_by_shipment_id?ship_id=1369",
    "/v1/shipping/avg_shipments_per_truck_by_make?make=Kenworth",
    "/v1/shipping/earliest_shipment_weight_by_driver_name?first_name=Sue&last_name=Newell",
    "/v1/shipping/heaviest_shipment_customer",
    "/v1/shipping/first_driver_by_ship_date",
    "/v1/shipping/count_shipments_by_customer?cust_name=Olympic%20Camper%20Sales%20Inc",
    "/v1/shipping/count_shipments_by_customer_city?city_name=New%20York&cust_name=Harry%27s%20Hot%20Rod%20Auto%20%26%20Truck%20Accessories",
    "/v1/shipping/count_shipments_by_truck_make?make=Mack",
    "/v1/shipping/driver_highest_population_city",
    "/v1/shipping/count_shipments_by_truck_weight?weight=1000",
    "/v1/shipping/annual_revenue_most_shipments",
    "/v1/shipping/count_shipments_by_city_year_state?city_name=Downey&year=2016&state=CA",
    "/v1/shipping/count_shipments_by_driver_city?first_name=Holger&last_name=Nohr&city_name=North%20Las%20Vegas",
    "/v1/shipping/count_shipments_highest_area_per_population",
    "/v1/shipping/customer_percentage_by_type_and_state?cust_type=manufacturer&state=TX",
    "/v1/shipping/total_weight_by_city_and_year?city_name=San%20Mateo&year=2016",
    "/v1/shipping/total_weight_by_truck_make_and_year?make=Peterbilt&year=2016",
    "/v1/shipping/heaviest_shipment_by_city?city_name=Boston",
    "/v1/shipping/driver_details_by_city_and_month?city_name=New%20York&ship_date=2016-02%25",
    "/v1/shipping/customer_names_by_city?city_name=Oak%20Park",
    "/v1/shipping/shipment_count_by_driver_city_and_year?first_name=Andrea&last_name=Simons&city_name=Huntsville&year=2016",
    "/v1/shipping/average_shipments_per_driver_per_month",
    "/v1/shipping/percentage_heavy_shipments?min_weight=10000&cust_name=Sunguard%20Window%20Tinting%20%26%20Truck%20Accessories&year=2017",
    "/v1/shipping/percentage_shipments_by_driver?cust_name=Autoware%20Inc&first_name=Sue&last_name=Newell",
    "/v1/shipping/count_cities_by_state_weight?state=New%20Jersey&min_weight=20000",
    "/v1/shipping/count_shipments_by_city_population_year?min_population=50000&year=2017",
    "/v1/shipping/shipment_weights_by_truck_make?make=Peterbilt",
    "/v1/shipping/truck_model_year_by_shipment?ship_id=1003",
    "/v1/shipping/driver_state_by_shipment?ship_id=1055",
    "/v1/shipping/driver_addresses_by_total_weight?min_weight=50000",
    "/v1/shipping/avg_annual_revenue_by_weight?weight=65000",
    "/v1/shipping/percentage_shipments_below_weight?weight=70000&cust_type=wholesaler",
    "/v1/shipping/driver_last_name_by_ship_id?ship_id=1088",
    "/v1/shipping/driver_phones_by_total_weight?total_weight=20000",
    "/v1/shipping/truck_details_by_ship_id?ship_id=1055",
    "/v1/shipping/truck_count_by_model_year?model_year=2009",
    "/v1/shipping/customer_count_by_type?cust_type=manufacturer",
    "/v1/shipping/customer_count_by_type_and_state?cust_type=retailer&state=CA",
    "/v1/shipping/city_count_by_state?state=Connecticut",
    "/v1/shipping/city_with_highest_population?state=California",
    "/v1/shipping/customer_annual_revenue?cust_name=Klett%20%26%20Sons%20Repair",
    "/v1/shipping/driver_of_lightest_shipment",
    "/v1/shipping/count_shipments_year_state?year=2016&state=CA",
    "/v1/shipping/distinct_truck_makes_by_driver?first_name=Zachery&last_name=Hicks",
    "/v1/shipping/customer_names_by_shipment_date?ship_date=2017-02%",
    "/v1/shipping/truck_make_driver_details_by_customer?cust_name=Klett%20%26%20Sons%20Repair",
    "/v1/shipping/heaviest_shipment_by_driver?first_name=Zachery&last_name=Hicks",
    "/v1/shipping/count_shipments_driver_city_year?first_name=Zachery&last_name=Hicks&city_name=New%20York&year=2016",
    "/v1/shipping/state_most_shipments_truck_make_year?make1=Peterbilt&state1=Texas%20(TX)&make2=Mack&state2=North%20Carolina%20(NC)&make3=Kenworth&state3=Washington%20(WA)&year=2016",
    "/v1/shipping/count_cities_by_state?state=California",
    "/v1/shipping/heaviest_shipment_city",
    "/v1/shipping/distinct_cities_by_driver?first_name=Zachery&last_name=Hicks",
    "/v1/shipping/percentage_shipments_by_driver_year?first_name=Zachery&last_name=Hicks&year=2017",
    "/v1/shipping/percentage_weight_by_driver_year?first_name=Zachery&last_name=Hicks&year=2016",
    "/v1/shipping/count_shipments_by_driver?driver_id=23",
    "/v1/shipping/largest_city_area_shipment",
    "/v1/shipping/driver_smallest_city_population",
    "/v1/shipping/count_shipments_city_state_weight_range?city_name=Cicero&state=Illinois&min_weight=9000&max_weight=15000",
    "/v1/shipping/driver_address_by_customer_location?address=7052%20Carroll%20Road&city=San%20Diego&state=CA",
    "/v1/shipping/count_shipments_by_driver_and_year?first_name=Maria&last_name=Craft&year=2017",
    "/v1/shipping/driver_address_by_shipment_id?ship_id=1127",
    "/v1/shipping/count_shipments_by_customer_state?state=NY",
    "/v1/shipping/shipment_id_most_populous_city",
    "/v1/shipping/shipment_weights_by_annual_revenue?annual_revenue=39448581",
    "/v1/shipping/customer_address_by_ship_id?ship_id=1117",
    "/v1/shipping/percentage_shipments_by_year_state?year=2017&state=TX",
    "/v1/shipping/truck_count_difference_by_model_year?model_year_1=2005&model_year_2=2006",
    "/v1/shipping/driver_details_heavy_shipments"
]
