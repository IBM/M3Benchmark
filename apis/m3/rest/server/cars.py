from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/cars/cars.sqlite')
cursor = conn.cursor()

# Endpoint to get the car name with the highest price for a given number of cylinders
@app.get("/v1/cars/highest_price_car_by_cylinders", operation_id="get_highest_price_car_by_cylinders", summary="Retrieves the name of the car with the highest price for a specified number of cylinders. The input parameter determines the number of cylinders for which the car with the highest price is returned.")
async def get_highest_price_car_by_cylinders(cylinders: int = Query(..., description="Number of cylinders")):
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.cylinders = ? ORDER BY T2.price DESC LIMIT 1", (cylinders,))
    result = cursor.fetchone()
    if not result:
        return {"car_name": []}
    return {"car_name": result[0]}

# Endpoint to get the count of cars with weight greater than a given value and price less than a given value
@app.get("/v1/cars/count_cars_by_weight_and_price", operation_id="get_count_cars_by_weight_and_price", summary="Retrieves the total number of cars that weigh more than the specified weight and cost less than the provided price. The weight and price parameters are used to filter the cars.")
async def get_count_cars_by_weight_and_price(weight: int = Query(..., description="Weight of the car"), price: int = Query(..., description="Price of the car")):
    cursor.execute("SELECT COUNT(T1.car_name) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.weight > ? AND T2.price < ?", (weight, price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the acceleration of the highest priced car
@app.get("/v1/cars/highest_price_car_acceleration", operation_id="get_highest_price_car_acceleration", summary="Retrieves the acceleration of the car with the highest price. This operation fetches the acceleration data from the database and identifies the car with the highest price, returning its acceleration value.")
async def get_highest_price_car_acceleration():
    cursor.execute("SELECT T1.acceleration FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T2.price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"acceleration": []}
    return {"acceleration": result[0]}

# Endpoint to get the price of a car by its name
@app.get("/v1/cars/price_by_car_name", operation_id="get_price_by_car_name", summary="Retrieves the price of a specific car by its name. The operation uses the provided car name to search for a matching record in the data table and returns the corresponding price from the price table.")
async def get_price_by_car_name(car_name: str = Query(..., description="Name of the car")):
    cursor.execute("SELECT T2.price FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.car_name = ?", (car_name,))
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get the country of production for a car by its name and model year
@app.get("/v1/cars/country_by_car_name_and_model_year", operation_id="get_country_by_car_name_and_model_year", summary="Retrieves the country of production for a specific car based on its name and model year. The operation uses the provided car name and model year to search for a match in the production data. Once a match is found, the corresponding country of production is returned.")
async def get_country_by_car_name_and_model_year(car_name: str = Query(..., description="Name of the car"), model_year: int = Query(..., description="Model year of the car")):
    cursor.execute("SELECT T3.country FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T1.car_name = ? AND T2.model_year = ?", (car_name, model_year))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of cars produced in a specific country and model year
@app.get("/v1/cars/count_cars_by_country_and_model_year", operation_id="get_count_cars_by_country_and_model_year", summary="Retrieves the total number of cars produced in a specified country during a particular model year. The operation requires the model year and the country of production as input parameters.")
async def get_count_cars_by_country_and_model_year(model_year: int = Query(..., description="Model year of the car"), country: str = Query(..., description="Country of production")):
    cursor.execute("SELECT COUNT(*) FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin WHERE T1.model_year = ? AND T2.country = ?", (model_year, country))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct car names produced in a specific country
@app.get("/v1/cars/distinct_car_names_by_country", operation_id="get_distinct_car_names_by_country", summary="Retrieves a unique list of car names that are produced in a specified country. The operation filters the car data based on the provided country of production.")
async def get_distinct_car_names_by_country(country: str = Query(..., description="Country of production")):
    cursor.execute("SELECT DISTINCT T1.car_name FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T2.country = T3.origin WHERE T3.country = ?", (country,))
    result = cursor.fetchall()
    if not result:
        return {"car_names": []}
    return {"car_names": [row[0] for row in result]}

# Endpoint to get the top 3 highest priced cars
@app.get("/v1/cars/top_3_highest_priced_cars", operation_id="get_top_3_highest_priced_cars", summary="Retrieves the names of the three most expensive cars from the database, sorted in descending order by price.")
async def get_top_3_highest_priced_cars():
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T2.price DESC LIMIT 3")
    result = cursor.fetchall()
    if not result:
        return {"car_names": []}
    return {"car_names": [row[0] for row in result]}

# Endpoint to get distinct model years for a specific car name
@app.get("/v1/cars/distinct_model_years_by_car_name", operation_id="get_distinct_model_years_by_car_name", summary="Retrieves a list of unique model years associated with a specific car name. The operation filters the production data based on the provided car name and returns the distinct model years for that car.")
async def get_distinct_model_years_by_car_name(car_name: str = Query(..., description="Name of the car")):
    cursor.execute("SELECT DISTINCT T1.model_year FROM production AS T1 INNER JOIN data AS T2 ON T1.ID = T2.ID WHERE T2.car_name = ?", (car_name,))
    result = cursor.fetchall()
    if not result:
        return {"model_years": []}
    return {"model_years": [row[0] for row in result]}

# Endpoint to get the count of cars with acceleration greater than a given value and price within a given range
@app.get("/v1/cars/count_cars_by_acceleration_and_price_range", operation_id="get_count_cars_by_acceleration_and_price_range", summary="Retrieves the total number of cars that meet the specified acceleration and price criteria. The acceleration must exceed a given value, and the price must fall within a provided range.")
async def get_count_cars_by_acceleration_and_price_range(acceleration: float = Query(..., description="Acceleration of the car"), min_price: int = Query(..., description="Minimum price of the car"), max_price: int = Query(..., description="Maximum price of the car")):
    cursor.execute("SELECT COUNT(*) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.acceleration > ? AND T2.price BETWEEN ? AND ?", (acceleration, min_price, max_price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the weight of cars with a price greater than a specified value
@app.get("/v1/cars/weight_by_price", operation_id="get_weight_by_price", summary="Retrieves the weight of cars that have a price greater than the specified minimum price. This operation allows you to filter cars based on their price and obtain their corresponding weights.")
async def get_weight_by_price(min_price: float = Query(..., description="Minimum price of the car")):
    cursor.execute("SELECT T1.weight FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T2.price > ?", (min_price,))
    result = cursor.fetchall()
    if not result:
        return {"weights": []}
    return {"weights": [row[0] for row in result]}

# Endpoint to get the maximum acceleration of cars with a price greater than a specified value
@app.get("/v1/cars/max_acceleration_by_price", operation_id="get_max_acceleration_by_price", summary="Retrieves the highest acceleration value among cars that exceed a specified minimum price. The operation filters cars based on their price and identifies the one with the maximum acceleration.")
async def get_max_acceleration_by_price(min_price: float = Query(..., description="Minimum price of the car")):
    cursor.execute("SELECT MAX(T1.acceleration) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T2.price > ?", (min_price,))
    result = cursor.fetchone()
    if not result:
        return {"max_acceleration": []}
    return {"max_acceleration": result[0]}

# Endpoint to get the average price of cars with a specified number of cylinders
@app.get("/v1/cars/avg_price_by_cylinders", operation_id="get_avg_price_by_cylinders", summary="Retrieves the average price of cars with a specified number of cylinders. The operation calculates the average price from the 'price' table, filtering the results based on the provided number of cylinders. This endpoint is useful for obtaining a statistical overview of car prices based on their cylinder count.")
async def get_avg_price_by_cylinders(cylinders: int = Query(..., description="Number of cylinders in the car")):
    cursor.execute("SELECT AVG(T2.price) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.cylinders = ?", (cylinders,))
    result = cursor.fetchone()
    if not result:
        return {"avg_price": []}
    return {"avg_price": result[0]}

# Endpoint to get the maximum displacement per cylinder of cars with a price less than a specified value
@app.get("/v1/cars/max_displacement_per_cylinder_by_price", operation_id="get_max_displacement_per_cylinder_by_price", summary="Retrieves the maximum displacement per cylinder for cars priced below a specified value. This operation allows you to determine the highest displacement-to-cylinder ratio among cars that fall within a given price range.")
async def get_max_displacement_per_cylinder_by_price(max_price: float = Query(..., description="Maximum price of the car")):
    cursor.execute("SELECT MAX(T1.displacement / T1.cylinders) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T2.price < ?", (max_price,))
    result = cursor.fetchone()
    if not result:
        return {"max_displacement_per_cylinder": []}
    return {"max_displacement_per_cylinder": result[0]}

# Endpoint to get the count of cars produced in a specified country
@app.get("/v1/cars/count_by_country", operation_id="get_count_by_country", summary="Retrieves the total number of cars produced in a specified country. The operation calculates the count based on the provided country parameter, which represents the country of production.")
async def get_count_by_country(country: str = Query(..., description="Country of production")):
    cursor.execute("SELECT COUNT(*) FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin WHERE T2.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country of production for a specified car name
@app.get("/v1/cars/country_by_car_name", operation_id="get_country_by_car_name", summary="Retrieves the country of production for a specific car. The operation requires the car's name as input and returns the corresponding country of production.")
async def get_country_by_car_name(car_name: str = Query(..., description="Name of the car")):
    cursor.execute("SELECT T3.country FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T1.car_name = ?", (car_name,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the MPG of the most expensive car
@app.get("/v1/cars/mpg_most_expensive", operation_id="get_mpg_most_expensive", summary="Retrieves the miles per gallon (MPG) of the car with the highest price. The operation compares the prices of all cars and returns the MPG of the most expensive one.")
async def get_mpg_most_expensive():
    cursor.execute("SELECT T1.mpg FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T2.price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"mpg": []}
    return {"mpg": result[0]}

# Endpoint to get the distinct countries of production for a specified car ID
@app.get("/v1/cars/distinct_countries_by_id", operation_id="get_distinct_countries_by_id", summary="Retrieve the unique countries of production for a specific car, identified by its unique ID. This operation provides a list of distinct countries where the specified car has been produced.")
async def get_distinct_countries_by_id(car_id: int = Query(..., description="ID of the car")):
    cursor.execute("SELECT DISTINCT T2.country FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin WHERE T1.ID = ?", (car_id,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the country of production for a specified car price
@app.get("/v1/cars/country_by_price", operation_id="get_country_by_price", summary="Retrieves the country of origin for cars that match the specified price. This operation returns the country where the cars are produced, based on the provided price parameter.")
async def get_country_by_price(price: float = Query(..., description="Price of the car")):
    cursor.execute("SELECT T3.country FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T1.price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"countries": []}
    return {"countries": [row[0] for row in result]}

# Endpoint to get the price of a car with a specified name and acceleration
@app.get("/v1/cars/price_by_car_name_and_acceleration", operation_id="get_price_by_car_name_and_acceleration", summary="Retrieves the price of a specific car model based on its name and acceleration. The operation uses the provided car name and acceleration to search for a matching record in the data table, then fetches the corresponding price from the price table.")
async def get_price_by_car_name_and_acceleration(car_name: str = Query(..., description="Name of the car"), acceleration: str = Query(..., description="Acceleration of the car")):
    cursor.execute("SELECT T2.price FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.car_name = ? AND T1.acceleration = ?", (car_name, acceleration))
    result = cursor.fetchall()
    if not result:
        return {"prices": []}
    return {"prices": [row[0] for row in result]}

# Endpoint to get the displacement of cars based on price
@app.get("/v1/cars/displacement_by_price", operation_id="get_displacement_by_price", summary="Retrieves the displacement of cars that match a given price. The operation filters the available car data based on the provided price and returns the corresponding displacement values.")
async def get_displacement_by_price(price: float = Query(..., description="Price of the car")):
    cursor.execute("SELECT T1.displacement FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T2.price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"displacement": []}
    return {"displacement": [row[0] for row in result]}

# Endpoint to get the model of cars based on price
@app.get("/v1/cars/model_by_price", operation_id="get_model_by_price", summary="Retrieves the model of cars that match a specified price. The operation filters the car models based on the provided price and returns the corresponding models.")
async def get_model_by_price(price: float = Query(..., description="Price of the car")):
    cursor.execute("SELECT T1.model FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T2.price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"model": []}
    return {"model": [row[0] for row in result]}

# Endpoint to get the cylinders of the car with the lowest price
@app.get("/v1/cars/cylinders_by_lowest_price", operation_id="get_cylinders_by_lowest_price", summary="Retrieves the number of cylinders of the car model with the lowest price. The operation compares car models based on their prices and returns the cylinder count of the most affordable option.")
async def get_cylinders_by_lowest_price():
    cursor.execute("SELECT T1.cylinders FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY price ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"cylinders": []}
    return {"cylinders": result[0]}

# Endpoint to get the ID and price of the heaviest car
@app.get("/v1/cars/id_price_by_heaviest_car", operation_id="get_id_price_by_heaviest_car", summary="Retrieves the unique identifier and price of the car with the highest weight from the available data. The operation returns the most weighted car's ID and corresponding price by joining the 'data' and 'price' tables on the 'ID' field and sorting the results in descending order based on weight.")
async def get_id_price_by_heaviest_car():
    cursor.execute("SELECT T1.ID, T2.price FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T1.weight DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"id": [], "price": []}
    return {"id": result[0], "price": result[1]}

# Endpoint to get the country of the car with the highest horsepower
@app.get("/v1/cars/country_by_highest_horsepower", operation_id="get_country_by_highest_horsepower", summary="Retrieves the country of origin for the car model with the highest horsepower. This operation fetches data from the 'data' and 'production' tables, joining them based on a common identifier. The 'country' table is then used to determine the country of origin for the car model with the highest horsepower. The result is a single country name.")
async def get_country_by_highest_horsepower():
    cursor.execute("SELECT T3.country FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country ORDER BY T1.horsepower DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the percentage of cars produced in a specific country
@app.get("/v1/cars/percentage_produced_in_country", operation_id="get_percentage_produced_in_country", summary="Retrieves the percentage of cars produced in a specified country. This operation calculates the proportion of cars produced in the given country by comparing the total number of cars produced in that country to the overall production count. The result is expressed as a percentage.")
async def get_percentage_produced_in_country(country: str = Query(..., description="Country of production")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.country = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin", (country,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the displacement per cylinder based on price
@app.get("/v1/cars/displacement_per_cylinder_by_price", operation_id="get_displacement_per_cylinder_by_price", summary="Retrieves the average displacement per cylinder for cars within a specified price range. The operation calculates this value by dividing the total displacement by the number of cylinders for each car that matches the provided price.")
async def get_displacement_per_cylinder_by_price(price: float = Query(..., description="Price of the car")):
    cursor.execute("SELECT T1.displacement / T1.cylinders FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T2.price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"displacement_per_cylinder": []}
    return {"displacement_per_cylinder": [row[0] for row in result]}

# Endpoint to get the car name with the highest price
@app.get("/v1/cars/car_name_by_highest_price", operation_id="get_car_name_by_highest_price", summary="Retrieves the name of the car with the highest price from the database. The operation returns the car name that corresponds to the highest price in the price table, which is joined with the data table using the ID field.")
async def get_car_name_by_highest_price():
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T2.price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"car_name": []}
    return {"car_name": result[0]}

# Endpoint to get the count of cars produced in a specific country and model year
@app.get("/v1/cars/count_produced_in_country_and_year", operation_id="get_count_produced_in_country_and_year", summary="Retrieves the total number of cars produced in a specified country and model year. The operation requires the country of production and the model year as input parameters to filter the production data.")
async def get_count_produced_in_country_and_year(country: str = Query(..., description="Country of production"), model_year: int = Query(..., description="Model year of the car")):
    cursor.execute("SELECT COUNT(*) FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin WHERE T2.country = ? AND T1.model_year = ?", (country, model_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the price of the car with the highest displacement to cylinders ratio
@app.get("/v1/cars/price_highest_displacement_cylinders_ratio", operation_id="get_price_highest_displacement_cylinders_ratio", summary="Retrieves the price of the car model with the highest displacement-to-cylinders ratio. This operation calculates the ratio of displacement to cylinders for each car model and returns the price of the car with the highest calculated ratio.")
async def get_price_highest_displacement_cylinders_ratio():
    cursor.execute("SELECT T2.price FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T1.displacement / T1.cylinders DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get the car name with the highest horsepower from a specific country
@app.get("/v1/cars/car_name_highest_horsepower_by_country", operation_id="get_car_name_highest_horsepower_by_country", summary="Retrieves the name of the car with the highest horsepower produced in a specified country. The operation filters cars based on the provided country and selects the one with the highest horsepower.")
async def get_car_name_highest_horsepower_by_country(country: str = Query(..., description="Country of the car")):
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T3.country = ? ORDER BY T1.horsepower DESC LIMIT 1", (country,))
    result = cursor.fetchone()
    if not result:
        return {"car_name": []}
    return {"car_name": result[0]}

# Endpoint to get the count of model years for a specific car name
@app.get("/v1/cars/count_model_years_by_car_name", operation_id="get_count_model_years_by_car_name", summary="Retrieves the total number of unique model years associated with a specific car name. The operation filters the data based on the provided car name and calculates the count of distinct model years from the production records.")
async def get_count_model_years_by_car_name(car_name: str = Query(..., description="Name of the car")):
    cursor.execute("SELECT COUNT(T2.model_year) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE T1.car_name = ?", (car_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the country of the car with the highest MPG
@app.get("/v1/cars/country_highest_mpg", operation_id="get_country_highest_mpg", summary="Retrieves the country of origin for the car model with the highest fuel efficiency (miles per gallon) from the available data.")
async def get_country_highest_mpg():
    cursor.execute("SELECT T3.country FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country ORDER BY T1.mpg DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the cheapest car name starting with a specific prefix
@app.get("/v1/cars/cheapest_car_name_by_prefix", operation_id="get_cheapest_car_name_by_prefix", summary="Retrieves the name of the least expensive car whose name begins with the provided prefix. The operation filters cars based on the specified prefix and sorts them by ascending price to identify the cheapest option.")
async def get_cheapest_car_name_by_prefix(prefix: str = Query(..., description="Prefix of the car name")):
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.car_name LIKE ? ORDER BY T2.price ASC LIMIT 1", (prefix + '%',))
    result = cursor.fetchone()
    if not result:
        return {"car_name": []}
    return {"car_name": result[0]}

# Endpoint to get the most expensive car name from a specific country
@app.get("/v1/cars/most_expensive_car_name_by_country", operation_id="get_most_expensive_car_name_by_country", summary="Retrieves the name of the most expensive car produced in a specified country. The operation considers the production and price data to determine the highest-priced car. The country of origin is used to filter the results.")
async def get_most_expensive_car_name_by_country(country: str = Query(..., description="Country of the car")):
    cursor.execute("SELECT T4.car_name FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country INNER JOIN data AS T4 ON T4.ID = T1.ID WHERE T3.country = ? ORDER BY T1.price DESC LIMIT 1", (country,))
    result = cursor.fetchone()
    if not result:
        return {"car_name": []}
    return {"car_name": result[0]}

# Endpoint to get the count of cars with displacement and price above specific values
@app.get("/v1/cars/count_cars_by_displacement_price", operation_id="get_count_cars_by_displacement_price", summary="Retrieves the total number of cars that have a displacement greater than the specified minimum displacement and a price higher than the specified minimum price. This operation is useful for understanding the distribution of cars based on their displacement and price.")
async def get_count_cars_by_displacement_price(min_displacement: int = Query(..., description="Minimum displacement"), min_price: int = Query(..., description="Minimum price")):
    cursor.execute("SELECT COUNT(*) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.displacement > ? AND T2.price > ?", (min_displacement, min_price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common model year for cars from a specific country
@app.get("/v1/cars/most_common_model_year_by_country", operation_id="get_most_common_model_year_by_country", summary="Retrieves the most frequently occurring model year for cars originating from a specified country. The operation groups cars by their model year and orders them by the frequency of occurrence in descending order. The model year with the highest frequency is then returned as the result.")
async def get_most_common_model_year_by_country(country: str = Query(..., description="Country of the car")):
    cursor.execute("SELECT T1.model_year FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin WHERE T2.country = ? GROUP BY T1.model_year ORDER BY COUNT(T1.model_year) DESC LIMIT 1", (country,))
    result = cursor.fetchone()
    if not result:
        return {"model_year": []}
    return {"model_year": result[0]}

# Endpoint to get the acceleration of the cheapest car from a specific country
@app.get("/v1/cars/acceleration_cheapest_car_by_country", operation_id="get_acceleration_cheapest_car_by_country", summary="Retrieves the acceleration of the most affordable car produced in a specified country. The operation considers the price and production data to determine the cheapest car and its acceleration.")
async def get_acceleration_cheapest_car_by_country(country: str = Query(..., description="Country of the car")):
    cursor.execute("SELECT T4.acceleration FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country INNER JOIN data AS T4 ON T4.ID = T1.ID WHERE T3.country = ? ORDER BY T1.price ASC LIMIT 1", (country,))
    result = cursor.fetchone()
    if not result:
        return {"acceleration": []}
    return {"acceleration": result[0]}

# Endpoint to get the average number of cars produced per model year by country
@app.get("/v1/cars/average_cars_per_model_year_by_country", operation_id="get_average_cars_per_model_year_by_country", summary="Retrieves the average number of cars produced per model year, grouped by country. The results are ordered in descending order based on the total count of cars per country. The limit parameter can be used to restrict the number of results returned.")
async def get_average_cars_per_model_year_by_country(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.country, CAST(COUNT(T1.ID) AS REAL) / COUNT(DISTINCT T1.model_year) FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin GROUP BY T2.country ORDER BY COUNT(T2.country) DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"average_cars_per_model_year": []}
    return {"average_cars_per_model_year": result}

# Endpoint to get the percentage of cars from a specific country with a displacement to cylinders ratio greater than a specified value
@app.get("/v1/cars/percentage_cars_by_country_displacement_ratio", operation_id="get_percentage_cars_by_country_displacement_ratio", summary="Retrieves the percentage of cars from a specified country that have a displacement to cylinders ratio greater than a given value. This operation calculates the ratio based on the provided country and displacement to cylinders ratio, and returns the corresponding percentage of cars that meet the specified criteria.")
async def get_percentage_cars_by_country_displacement_ratio(country: str = Query(..., description="Country of origin"), displacement_ratio: float = Query(..., description="Displacement to cylinders ratio")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.country = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T1.displacement / T1.cylinders > ?", (country, displacement_ratio))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get car names for a specific model year
@app.get("/v1/cars/car_names_by_model_year", operation_id="get_car_names_by_model_year", summary="Retrieves the names of cars that were produced in a specified model year. The operation filters the car data based on the provided model year and returns the corresponding car names.")
async def get_car_names_by_model_year(model_year: int = Query(..., description="Model year of the car")):
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE T2.model_year = ?", (model_year,))
    result = cursor.fetchall()
    if not result:
        return {"car_names": []}
    return {"car_names": [row[0] for row in result]}

# Endpoint to get the average price of cars from a specific country
@app.get("/v1/cars/average_price_by_country", operation_id="get_average_price_by_country", summary="Retrieves the average price of cars produced in a specified country. The calculation is based on the prices of all cars from the given country, as recorded in the price and production tables. The country of origin is provided as an input parameter.")
async def get_average_price_by_country(country: str = Query(..., description="Country of origin")):
    cursor.execute("SELECT AVG(T1.price) FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T3.country = ?", (country,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the price of a car by its ID
@app.get("/v1/cars/price_by_car_id", operation_id="get_price_by_car_id", summary="Retrieves the price of a specific car, identified by its unique ID. The operation fetches the price from the associated data and returns it.")
async def get_price_by_car_id(car_id: int = Query(..., description="ID of the car")):
    cursor.execute("SELECT T2.price FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.ID = ?", (car_id,))
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get the count of cars from a specific country with weight less than a specified value
@app.get("/v1/cars/count_cars_by_country_weight", operation_id="get_count_cars_by_country_weight", summary="Retrieves the total number of cars from a specified country with a weight less than the provided value. This operation considers the car's price, production details, and country of origin to accurately determine the count.")
async def get_count_cars_by_country_weight(country: str = Query(..., description="Country of origin"), weight: float = Query(..., description="Weight of the car")):
    cursor.execute("SELECT COUNT(*) FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country INNER JOIN data AS T4 ON T4.ID = T1.ID WHERE T3.country = ? AND T4.weight < ?", (country, weight))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cars from a specific model year with horsepower less than a specified value
@app.get("/v1/cars/count_cars_by_model_year_horsepower", operation_id="get_count_cars_by_model_year_horsepower", summary="Retrieves the total number of cars from a specified model year that have a horsepower value less than the provided limit. This operation is useful for understanding the distribution of cars based on their model year and horsepower.")
async def get_count_cars_by_model_year_horsepower(model_year: int = Query(..., description="Model year of the car"), horsepower: float = Query(..., description="Horsepower of the car")):
    cursor.execute("SELECT COUNT(*) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE T2.model_year = ? AND T1.horsepower < ?", (model_year, horsepower))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get car IDs from a specific country with a price greater than a specified value and a specific acceleration
@app.get("/v1/cars/car_ids_by_country_price_acceleration", operation_id="get_car_ids_by_country_price_acceleration", summary="Retrieves the IDs of cars from a specified country that have a price greater than a given value and a specific acceleration. The operation filters cars based on their country of origin, price, and acceleration, returning a list of matching car IDs.")
async def get_car_ids_by_country_price_acceleration(country: str = Query(..., description="Country of origin"), price: float = Query(..., description="Price of the car"), acceleration: float = Query(..., description="Acceleration of the car")):
    cursor.execute("SELECT T4.ID FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country INNER JOIN data AS T4 ON T4.ID = T1.ID WHERE T3.country = ? AND T1.price > ? AND T4.acceleration = ?", (country, price, acceleration))
    result = cursor.fetchall()
    if not result:
        return {"car_ids": []}
    return {"car_ids": [row[0] for row in result]}

# Endpoint to get the model year of the heaviest car
@app.get("/v1/cars/model_year_heaviest_car", operation_id="get_model_year_heaviest_car", summary="Retrieves the model year of the heaviest car(s) based on the provided weight limit. The operation returns the model year(s) of the car(s) with the highest weight, up to the specified limit.")
async def get_model_year_heaviest_car(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.model_year FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID ORDER BY T1.weight DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"model_year": []}
    return {"model_year": result[0]}

# Endpoint to get the horsepower and model year of a specific car
@app.get("/v1/cars/horsepower_model_year_by_car_name", operation_id="get_horsepower_model_year_by_car_name", summary="Retrieves the horsepower and model year of a specific car by its name. The operation uses the provided car name to search for matching records in the data and production tables, returning the horsepower and model year of the corresponding car.")
async def get_horsepower_model_year_by_car_name(car_name: str = Query(..., description="Name of the car")):
    cursor.execute("SELECT T1.horsepower, T2.model_year FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE T1.car_name = ?", (car_name,))
    result = cursor.fetchall()
    if not result:
        return {"horsepower_model_year": []}
    return {"horsepower_model_year": result}

# Endpoint to get car names based on a specific price
@app.get("/v1/cars/car_names_by_price", operation_id="get_car_names_by_price", summary="Retrieves the names of cars that match a given price. The operation filters the available car data based on the provided price and returns the corresponding car names.")
async def get_car_names_by_price(price: int = Query(..., description="Price of the car")):
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T2.price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"car_names": []}
    return {"car_names": [row[0] for row in result]}

# Endpoint to get the count of cars from a specific country with a price above a certain threshold
@app.get("/v1/cars/count_cars_by_country_and_price", operation_id="get_count_cars_by_country_and_price", summary="Retrieves the total number of cars from a specified country that have a price greater than a given threshold. The operation filters cars based on their country of origin and price, providing a count of the qualifying vehicles.")
async def get_count_cars_by_country_and_price(country: str = Query(..., description="Country of origin"), min_price: int = Query(..., description="Minimum price of the car")):
    cursor.execute("SELECT COUNT(*) FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T3.country = ? AND T1.price > ?", (country, min_price))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct prices and countries for a specific car name
@app.get("/v1/cars/distinct_prices_countries_by_car_name", operation_id="get_distinct_prices_countries_by_car_name", summary="Retrieves unique price and country combinations for a specific car model. The operation filters data based on the provided car name, returning a list of distinct prices and their corresponding countries of origin.")
async def get_distinct_prices_countries_by_car_name(car_name: str = Query(..., description="Name of the car")):
    cursor.execute("SELECT DISTINCT T1.price, T3.country FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country INNER JOIN data AS T4 ON T4.ID = T1.ID WHERE T4.car_name = ?", (car_name,))
    result = cursor.fetchall()
    if not result:
        return {"prices_countries": []}
    return {"prices_countries": [{"price": row[0], "country": row[1]} for row in result]}

# Endpoint to get car names where the price is significantly higher than the average price
@app.get("/v1/cars/car_names_above_average_price", operation_id="get_car_names_above_average_price", summary="Retrieves the names of cars that have a price significantly higher than the average price. This operation compares each car's price to 85% of the average price and returns the names of cars that exceed this threshold. The result is a list of car names that are priced well above the average.")
async def get_car_names_above_average_price():
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T2.price * 100 > ( SELECT AVG(price) * 85 FROM price )")
    result = cursor.fetchall()
    if not result:
        return {"car_names": []}
    return {"car_names": [row[0] for row in result]}

# Endpoint to get the difference in the number of cars between two model years for a specific horsepower
@app.get("/v1/cars/difference_cars_by_model_year_horsepower", operation_id="get_difference_cars_by_model_year_horsepower", summary="Retrieve the difference in the count of cars with a specific horsepower between two distinct model years. This operation compares the number of cars produced in the first model year with the number of cars produced in the second model year, both sharing the same horsepower. The result is a single value representing the difference in the count of cars between the two model years.")
async def get_difference_cars_by_model_year_horsepower(model_year_1: int = Query(..., description="First model year"), model_year_2: int = Query(..., description="Second model year"), horsepower: int = Query(..., description="Horsepower of the car")):
    cursor.execute("SELECT SUM(CASE WHEN T2.model_year = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.model_year = ? THEN 1 ELSE 0 END) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE T1.horsepower = ?", (model_year_1, model_year_2, horsepower))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get car IDs from a specific country and model year
@app.get("/v1/cars/car_ids_by_country_model_year", operation_id="get_car_ids_by_country_model_year", summary="Retrieves the unique identifiers of cars that originate from a specified country and were manufactured in a given year. The operation filters the production data based on the provided country and model year.")
async def get_car_ids_by_country_model_year(country: str = Query(..., description="Country of origin"), model_year: int = Query(..., description="Model year of the car")):
    cursor.execute("SELECT T1.ID FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin WHERE T2.country = ? AND T1.model_year = ?", (country, model_year))
    result = cursor.fetchall()
    if not result:
        return {"car_ids": []}
    return {"car_ids": [row[0] for row in result]}

# Endpoint to get the country of the car with the lowest MPG
@app.get("/v1/cars/country_of_lowest_mpg", operation_id="get_country_of_lowest_mpg", summary="Retrieves the country of origin for the car model with the lowest miles per gallon (MPG) rating. This operation considers data from both the 'data' and 'production' tables, joining them based on a common identifier. The 'country' table is also referenced to determine the country of origin for each car model. The result is the country with the lowest MPG rating.")
async def get_country_of_lowest_mpg():
    cursor.execute("SELECT T3.country FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country ORDER BY T1.mpg ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the heaviest car details
@app.get("/v1/cars/heaviest_car_details", operation_id="get_heaviest_car_details", summary="Retrieves the details of the car with the highest weight, including its name, model, displacement per cylinder, and model year. The data is sourced from the 'data' and 'production' tables, which are joined on the 'ID' field. The results are ordered by weight in descending order and limited to the top record.")
async def get_heaviest_car_details():
    cursor.execute("SELECT T1.car_name, T1.model, T1.displacement / T1.cylinders, T2.model_year FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID ORDER BY T1.weight DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"car_details": []}
    return {"car_details": {"car_name": result[0], "model": result[1], "displacement_per_cylinder": result[2], "model_year": result[3]}}

# Endpoint to get car names and horsepower from a specific country and model year
@app.get("/v1/cars/car_names_horsepower_by_country_model_year", operation_id="get_car_names_horsepower_by_country_model_year", summary="Retrieves the names and horsepower of cars from a specific country and model year. The operation filters cars based on the provided model year and country of origin, returning a list of car names and their corresponding horsepower.")
async def get_car_names_horsepower_by_country_model_year(model_year: int = Query(..., description="Model year of the car"), country: str = Query(..., description="Country of origin")):
    cursor.execute("SELECT T1.car_name, T1.horsepower FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T2.model_year = ? AND T3.country = ?", (model_year, country))
    result = cursor.fetchall()
    if not result:
        return {"car_details": []}
    return {"car_details": [{"car_name": row[0], "horsepower": row[1]} for row in result]}

# Endpoint to get the most expensive car
@app.get("/v1/cars/most_expensive_car", operation_id="get_most_expensive_car", summary="Retrieves the car model with the highest price from the available data. The operation returns the name and model of the most expensive car.")
async def get_most_expensive_car():
    cursor.execute("SELECT T1.car_name, T1.model FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T2.price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"car": []}
    return {"car": {"car_name": result[0], "model": result[1]}}

# Endpoint to get the country of the cheapest car
@app.get("/v1/cars/country_of_cheapest_car", operation_id="get_country_of_cheapest_car", summary="Retrieves the country of origin for the least expensive car, based on the lowest price in the price table. This is determined by joining the price and production tables on their shared ID, and then joining the resulting set with the country table based on the production country. The result is sorted by price in ascending order, and the country of the cheapest car is returned.")
async def get_country_of_cheapest_car():
    cursor.execute("SELECT T3.country FROM price AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country ORDER BY T1.price ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the car with the most cylinders for a specific model year
@app.get("/v1/cars/most_cylinders_by_model_year", operation_id="get_most_cylinders_by_model_year", summary="Retrieves the car with the highest number of cylinders produced in a specific model year. The response includes the car's ID, name, and the country of origin. The model year is provided as an input parameter.")
async def get_most_cylinders_by_model_year(model_year: int = Query(..., description="Model year of the car")):
    cursor.execute("SELECT T1.ID, T1.car_name, T3.country FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T2.model_year = ? ORDER BY T1.cylinders DESC LIMIT 1", (model_year,))
    result = cursor.fetchone()
    if not result:
        return {"car": []}
    return {"car": {"ID": result[0], "car_name": result[1], "country": result[2]}}

# Endpoint to get the car with the lowest price
@app.get("/v1/cars/cheapest_car_details", operation_id="get_cheapest_car_details", summary="Retrieves the acceleration, number of cylinders, and model year of the least expensive car available. The data is sourced from the 'data', 'production', and 'price' tables, which are joined based on their respective IDs. The results are ordered by price in ascending order, and only the cheapest car's details are returned.")
async def get_cheapest_car_details():
    cursor.execute("SELECT T1.acceleration, T1.cylinders, T2.model_year FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN price AS T3 ON T3.ID = T2.ID ORDER BY T3.price ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"car": []}
    return {"car": {"acceleration": result[0], "cylinders": result[1], "model_year": result[2]}}

# Endpoint to get the price of cars with a specific model and MPG greater than a specified value
@app.get("/v1/cars/price_by_model_and_mpg", operation_id="get_price_by_model_and_mpg", summary="Retrieves the price of cars that match a specific model and have a miles per gallon (MPG) rating greater than a given value. The model and MPG parameters are used to filter the results.")
async def get_price_by_model_and_mpg(model: int = Query(..., description="Model of the car"), mpg: float = Query(..., description="MPG of the car")):
    cursor.execute("SELECT T2.car_name, T1.price FROM price AS T1 INNER JOIN data AS T2 ON T1.ID = T2.ID WHERE T2.model = ? AND T2.mpg > ?", (model, mpg))
    result = cursor.fetchall()
    if not result:
        return {"cars": []}
    return {"cars": [{"car_name": row[0], "price": row[1]} for row in result]}

# Endpoint to get the percentage of cars produced in the USA
@app.get("/v1/cars/percentage_produced_in_usa", operation_id="get_percentage_produced_in_usa", summary="Retrieves the percentage of cars produced in the USA by calculating the ratio of cars produced in the USA to the total number of cars produced. This operation considers the production data and the country of origin for each car.")
async def get_percentage_produced_in_usa():
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.country = 'USA' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM production AS T1 INNER JOIN country AS T2 ON T1.country = T2.origin")
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average number of cars produced per year within a specified range and distinct car names with weight less than a specified value
@app.get("/v1/cars/average_production_and_car_names", operation_id="get_average_production_and_car_names", summary="Retrieves the average annual production of cars within a specified model year range and distinct car names with weight below a given threshold. The operation calculates the average production by dividing the total number of cars produced within the range by the number of years in the range. It also returns distinct car names that meet the weight criteria within the same model year range.")
async def get_average_production_and_car_names(start_year: int = Query(..., description="Start year of the model year range"), end_year: int = Query(..., description="End year of the model year range"), weight: float = Query(..., description="Weight of the car")):
    cursor.execute("SELECT CAST(COUNT(T1.ID) AS REAL) / 9 FROM production AS T1 INNER JOIN data AS T2 ON T2.ID = T1.ID WHERE T1.model_year BETWEEN ? AND ? UNION ALL SELECT DISTINCT T2.car_name FROM production AS T1 INNER JOIN data AS T2 ON T2.ID = T1.ID WHERE T1.model_year BETWEEN ? AND ? AND T2.weight < ?", (start_year, end_year, start_year, end_year, weight))
    result = cursor.fetchall()
    if not result:
        return {"average_production": [], "car_names": []}
    average_production = result[0][0] if result[0][0] is not None else 0
    car_names = [row[0] for row in result[1:]]
    return {"average_production": average_production, "car_names": car_names}

# Endpoint to get the average price of cars based on model
@app.get("/v1/cars/average_price_by_model", operation_id="get_average_price_by_model", summary="Retrieves the average price of a specific car model from the database. The operation calculates the average price based on the provided car model.")
async def get_average_price_by_model(model: int = Query(..., description="Model of the car")):
    cursor.execute("SELECT AVG(T2.price) FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID WHERE T1.model = ?", (model,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get distinct prices of cars based on name pattern and model year range
@app.get("/v1/cars/distinct_prices_by_name_and_year", operation_id="get_distinct_prices_by_name_and_year", summary="Retrieve a unique set of prices for cars that match a specified name pattern and fall within a given model year range. The operation filters cars based on the provided name pattern and year range, then returns the distinct prices associated with the filtered cars.")
async def get_distinct_prices_by_name_and_year(car_name_pattern: str = Query(..., description="Pattern of the car name (use % for wildcard)"), start_year: int = Query(..., description="Start year of the model year range"), end_year: int = Query(..., description="End year of the model year range")):
    cursor.execute("SELECT DISTINCT T3.price FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN price AS T3 ON T3.ID = T2.ID WHERE T1.car_name LIKE ? AND T2.model_year BETWEEN ? AND ?", (car_name_pattern, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"prices": []}
    return {"prices": [row[0] for row in result]}

# Endpoint to get the car with the highest MPG for a given model year
@app.get("/v1/cars/highest_mpg_by_model_year", operation_id="get_highest_mpg_by_model_year", summary="Retrieves the car model with the highest fuel efficiency (MPG) for a specific model year. The operation filters the car data by the provided model year and returns the car name with the highest MPG value.")
async def get_highest_mpg_by_model_year(model_year: int = Query(..., description="Model year of the car")):
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE T2.model_year = ? ORDER BY T1.mpg DESC LIMIT 1", (model_year,))
    result = cursor.fetchone()
    if not result:
        return {"car_name": []}
    return {"car_name": result[0]}

# Endpoint to get the car with the highest MPG and price
@app.get("/v1/cars/highest_mpg_and_price", operation_id="get_highest_mpg_and_price", summary="Retrieves the name of the car that has the highest fuel efficiency (MPG) and the highest price. The data is fetched from the 'data' and 'price' tables, which are joined on the 'ID' field. The results are ordered by MPG in descending order and then by price in descending order. Only the top result is returned.")
async def get_highest_mpg_and_price():
    cursor.execute("SELECT T1.car_name FROM data AS T1 INNER JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T1.mpg DESC, T2.price DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"car_name": []}
    return {"car_name": result[0]}

# Endpoint to get the count of cars from a specific country with acceleration below a certain value
@app.get("/v1/cars/count_by_country_and_acceleration", operation_id="get_count_by_country_and_acceleration", summary="Retrieves the total number of cars from a specified country that have an acceleration value less than a given threshold. This operation considers data from both the 'data' and 'production' tables, joined by the 'ID' field, and further filtered by the 'country' table using the 'origin' and 'country' fields.")
async def get_count_by_country_and_acceleration(country: str = Query(..., description="Country of the car"), max_acceleration: float = Query(..., description="Maximum acceleration value")):
    cursor.execute("SELECT COUNT(*) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T3.country = ? AND T1.acceleration < ?", (country, max_acceleration))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cars from a specific country with weight above a certain value
@app.get("/v1/cars/count_by_country_and_weight", operation_id="get_count_by_country_and_weight", summary="Retrieves the total number of cars from a specified country that weigh more than a given minimum weight. The operation filters cars based on their country of origin and weight, then returns the count of cars that meet the criteria.")
async def get_count_by_country_and_weight(country: str = Query(..., description="Country of the car"), min_weight: float = Query(..., description="Minimum weight value")):
    cursor.execute("SELECT COUNT(*) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T3.country = ? AND T1.weight > ?", (country, min_weight))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the model year of a car based on its name
@app.get("/v1/cars/model_year_by_car_name", operation_id="get_model_year_by_car_name", summary="Retrieves the model year of a specific car by its name. The operation uses the provided car name to search for a match in the data table and returns the corresponding model year from the production table.")
async def get_model_year_by_car_name(car_name: str = Query(..., description="Name of the car")):
    cursor.execute("SELECT T2.model_year FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE T1.car_name = ?", (car_name,))
    result = cursor.fetchone()
    if not result:
        return {"model_year": []}
    return {"model_year": result[0]}

# Endpoint to get the country of the most expensive car for a given model year
@app.get("/v1/cars/most_expensive_country_by_model_year", operation_id="get_most_expensive_country_by_model_year", summary="Retrieves the country of origin for the most expensive car of a specified model year. The operation compares car prices from the given year and identifies the country with the highest priced vehicle. The model year is a required input parameter.")
async def get_most_expensive_country_by_model_year(model_year: int = Query(..., description="Model year of the car")):
    cursor.execute("SELECT T3.country FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country INNER JOIN price AS T4 ON T4.ID = T1.ID WHERE T2.model_year = ? ORDER BY T4.price DESC LIMIT 1", (model_year,))
    result = cursor.fetchone()
    if not result:
        return {"country": []}
    return {"country": result[0]}

# Endpoint to get the count of model years for cars with a specific horsepower and model year
@app.get("/v1/cars/count_model_years_by_horsepower_and_year", operation_id="get_count_model_years", summary="Retrieves the number of cars with a horsepower greater than the specified value and a specific model year. This operation provides a quantitative measure of cars that meet the given horsepower and model year criteria.")
async def get_count_model_years(horsepower: int = Query(..., description="Horsepower of the car"), model_year: int = Query(..., description="Model year of the car")):
    cursor.execute("SELECT COUNT(T2.model_year) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID WHERE T1.horsepower > ? AND T2.model_year = ?", (horsepower, model_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of cars produced in a specific country
@app.get("/v1/cars/percentage_cars_by_country", operation_id="get_percentage_cars_by_country", summary="Retrieves the percentage of cars produced in a specified country. This operation calculates the proportion of cars produced in the given country relative to the total number of cars in the database. The calculation is based on the data from the 'data' and 'production' tables, joined by their respective IDs, and further linked to the 'country' table via the 'origin' and 'country' fields.")
async def get_percentage_cars_by_country(country: str = Query(..., description="Country of production")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T3.country = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country", (country,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average weight of cars produced in a specific country within a range of model years and with a specific number of cylinders
@app.get("/v1/cars/average_weight_by_year_cylinders_country", operation_id="get_average_weight", summary="Retrieve the average weight of cars manufactured within a specified range of model years, possessing a certain number of cylinders, and produced in a particular country.")
async def get_average_weight(start_year: int = Query(..., description="Start year of the model year range"), end_year: int = Query(..., description="End year of the model year range"), cylinders: int = Query(..., description="Number of cylinders"), country: str = Query(..., description="Country of production")):
    cursor.execute("SELECT AVG(T1.weight) FROM data AS T1 INNER JOIN production AS T2 ON T1.ID = T2.ID INNER JOIN country AS T3 ON T3.origin = T2.country WHERE T2.model_year BETWEEN ? AND ? AND T1.cylinders = ? AND T3.country = ?", (start_year, end_year, cylinders, country))
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

api_calls = [
    "/v1/cars/highest_price_car_by_cylinders?cylinders=8",
    "/v1/cars/count_cars_by_weight_and_price?weight=3000&price=30000",
    "/v1/cars/highest_price_car_acceleration",
    "/v1/cars/price_by_car_name?car_name=ford%20torino",
    "/v1/cars/country_by_car_name_and_model_year?car_name=ford%20torino&model_year=1970",
    "/v1/cars/count_cars_by_country_and_model_year?model_year=1970&country=USA",
    "/v1/cars/distinct_car_names_by_country?country=USA",
    "/v1/cars/top_3_highest_priced_cars",
    "/v1/cars/distinct_model_years_by_car_name?car_name=chevrolet%20impala",
    "/v1/cars/count_cars_by_acceleration_and_price_range?acceleration=10&min_price=20000&max_price=30000",
    "/v1/cars/weight_by_price?min_price=40000",
    "/v1/cars/max_acceleration_by_price?min_price=40000",
    "/v1/cars/avg_price_by_cylinders?cylinders=8",
    "/v1/cars/max_displacement_per_cylinder_by_price?max_price=30000",
    "/v1/cars/count_by_country?country=Europe",
    "/v1/cars/country_by_car_name?car_name=chevrolet%20malibu",
    "/v1/cars/mpg_most_expensive",
    "/v1/cars/distinct_countries_by_id?car_id=382",
    "/v1/cars/country_by_price?price=44274.40748",
    "/v1/cars/price_by_car_name_and_acceleration?car_name=volkswagen%20dasher&acceleration=14.1",
    "/v1/cars/displacement_by_price?price=37443.85589",
    "/v1/cars/model_by_price?price=32650.65157",
    "/v1/cars/cylinders_by_lowest_price",
    "/v1/cars/id_price_by_heaviest_car",
    "/v1/cars/country_by_highest_horsepower",
    "/v1/cars/percentage_produced_in_country?country=Japan",
    "/v1/cars/displacement_per_cylinder_by_price?price=34538.97449",
    "/v1/cars/car_name_by_highest_price",
    "/v1/cars/count_produced_in_country_and_year?country=USA&model_year=1981",
    "/v1/cars/price_highest_displacement_cylinders_ratio",
    "/v1/cars/car_name_highest_horsepower_by_country?country=Japan",
    "/v1/cars/count_model_years_by_car_name?car_name=ford%20maverick",
    "/v1/cars/country_highest_mpg",
    "/v1/cars/cheapest_car_name_by_prefix?prefix=dodge",
    "/v1/cars/most_expensive_car_name_by_country?country=USA",
    "/v1/cars/count_cars_by_displacement_price?min_displacement=400&min_price=30000",
    "/v1/cars/most_common_model_year_by_country?country=Europe",
    "/v1/cars/acceleration_cheapest_car_by_country?country=USA",
    "/v1/cars/average_cars_per_model_year_by_country?limit=1",
    "/v1/cars/percentage_cars_by_country_displacement_ratio?country=Japan&displacement_ratio=30",
    "/v1/cars/car_names_by_model_year?model_year=1975",
    "/v1/cars/average_price_by_country?country=Europe",
    "/v1/cars/price_by_car_id?car_id=15",
    "/v1/cars/count_cars_by_country_weight?country=Japan&weight=3000",
    "/v1/cars/count_cars_by_model_year_horsepower?model_year=1973&horsepower=100",
    "/v1/cars/car_ids_by_country_price_acceleration?country=Japan&price=3500&acceleration=14",
    "/v1/cars/model_year_heaviest_car?limit=1",
    "/v1/cars/horsepower_model_year_by_car_name?car_name=subaru%20dl",
    "/v1/cars/car_names_by_price?price=20000",
    "/v1/cars/count_cars_by_country_and_price?country=USA&min_price=40000",
    "/v1/cars/distinct_prices_countries_by_car_name?car_name=ford%20maverick",
    "/v1/cars/car_names_above_average_price",
    "/v1/cars/difference_cars_by_model_year_horsepower?model_year_1=1970&model_year_2=1976&horsepower=130",
    "/v1/cars/car_ids_by_country_model_year?country=Japan&model_year=1979",
    "/v1/cars/country_of_lowest_mpg",
    "/v1/cars/heaviest_car_details",
    "/v1/cars/car_names_horsepower_by_country_model_year?model_year=1977&country=Europe",
    "/v1/cars/most_expensive_car",
    "/v1/cars/country_of_cheapest_car",
    "/v1/cars/most_cylinders_by_model_year?model_year=1975",
    "/v1/cars/cheapest_car_details",
    "/v1/cars/price_by_model_and_mpg?model=82&mpg=30",
    "/v1/cars/percentage_produced_in_usa",
    "/v1/cars/average_production_and_car_names?start_year=1971&end_year=1980&weight=1800",
    "/v1/cars/average_price_by_model?model=70",
    "/v1/cars/distinct_prices_by_name_and_year?car_name_pattern=ford%25&start_year=1970&end_year=1980",
    "/v1/cars/highest_mpg_by_model_year?model_year=1975",
    "/v1/cars/highest_mpg_and_price",
    "/v1/cars/count_by_country_and_acceleration?country=USA&max_acceleration=12",
    "/v1/cars/count_by_country_and_weight?country=Japan&min_weight=2000",
    "/v1/cars/model_year_by_car_name?car_name=buick%20skylark%20320",
    "/v1/cars/most_expensive_country_by_model_year?model_year=1970",
    "/v1/cars/count_model_years_by_horsepower_and_year?horsepower=200&model_year=1975",
    "/v1/cars/percentage_cars_by_country?country=USA",
    "/v1/cars/average_weight_by_year_cylinders_country?start_year=1975&end_year=1980&cylinders=4&country=Japan"
]
