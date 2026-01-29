from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/trains/trains.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of trains based on direction
@app.get("/v1/trains/count_trains_by_direction", operation_id="get_count_trains_by_direction", summary="Retrieves the total number of trains traveling in a specified direction. The direction parameter is used to filter the trains and calculate the count.")
async def get_count_trains_by_direction(direction: str = Query(..., description="Direction of the train")):
    cursor.execute("SELECT COUNT(id) FROM trains WHERE direction = ?", (direction,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cars based on train ID
@app.get("/v1/trains/count_cars_by_train_id", operation_id="get_count_cars_by_train_id", summary="Retrieves the total number of cars associated with a specific train, identified by its unique train ID.")
async def get_count_cars_by_train_id(train_id: int = Query(..., description="ID of the train")):
    cursor.execute("SELECT COUNT(id) FROM cars WHERE train_id = ?", (train_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the shape of a car based on train ID and position
@app.get("/v1/trains/get_car_shape_by_train_id_and_position", operation_id="get_car_shape_by_train_id_and_position", summary="Retrieves the shape of a specific car from a train, based on the provided train ID and car position. The operation uses the train ID and car position to identify the car and return its shape.")
async def get_car_shape_by_train_id_and_position(train_id: int = Query(..., description="ID of the train"), position: int = Query(..., description="Position of the car")):
    cursor.execute("SELECT shape FROM cars WHERE train_id = ? AND position = ?", (train_id, position))
    result = cursor.fetchone()
    if not result:
        return {"shape": []}
    return {"shape": result[0]}

# Endpoint to get train IDs based on car shapes
@app.get("/v1/trains/get_train_ids_by_car_shapes", operation_id="get_train_ids_by_car_shapes", summary="Retrieves the unique identifiers of trains that have cars with the specified shapes. The operation filters the cars based on the provided shapes and returns the train IDs associated with the matching cars. The input parameters define the shapes to be considered for the search.")
async def get_train_ids_by_car_shapes(shape1: str = Query(..., description="First shape of the car"), shape2: str = Query(..., description="Second shape of the car")):
    cursor.execute("SELECT train_id FROM cars WHERE shape IN (?, ?) GROUP BY train_id", (shape1, shape2))
    result = cursor.fetchall()
    if not result:
        return {"train_ids": []}
    return {"train_ids": [row[0] for row in result]}

# Endpoint to get the count of cars based on train ID and roof type
@app.get("/v1/trains/count_cars_by_train_id_and_roof", operation_id="get_count_cars_by_train_id_and_roof", summary="Retrieves the total number of cars associated with a specific train, filtered by a given roof type. This operation requires the train's unique identifier and the desired roof type as input parameters.")
async def get_count_cars_by_train_id_and_roof(train_id: int = Query(..., description="ID of the train"), roof: str = Query(..., description="Roof type of the car")):
    cursor.execute("SELECT COUNT(id) FROM cars WHERE train_id = ? AND roof = ?", (train_id, roof))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get car IDs based on train ID and number of wheels
@app.get("/v1/trains/get_car_ids_by_train_id_and_wheels", operation_id="get_car_ids_by_train_id_and_wheels", summary="Retrieves the unique identifiers of cars that belong to a specific train and have a specified number of wheels. The operation requires the train's ID and the number of wheels as input parameters.")
async def get_car_ids_by_train_id_and_wheels(train_id: int = Query(..., description="ID of the train"), wheels: int = Query(..., description="Number of wheels on the car")):
    cursor.execute("SELECT id FROM cars WHERE train_id = ? AND wheels = ?", (train_id, wheels))
    result = cursor.fetchall()
    if not result:
        return {"car_ids": []}
    return {"car_ids": [row[0] for row in result]}

# Endpoint to get the count of cars with specific shapes in trains going in a specific direction
@app.get("/v1/trains/count_cars_with_shapes_in_direction", operation_id="get_count_cars_with_shapes_in_direction", summary="Retrieves the total number of cars with shapes specified in the input parameters that are present in trains moving in the provided direction. The response includes a count of cars that match the given shapes and direction.")
async def get_count_cars_with_shapes_in_direction(shape1: str = Query(..., description="First shape of the car"), shape2: str = Query(..., description="Second shape of the car"), direction: str = Query(..., description="Direction of the train")):
    cursor.execute("SELECT SUM(CASE WHEN T1.shape IN (?, ?) THEN 1 ELSE 0 end)as count FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T2.direction = ?", (shape1, shape2, direction))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get train IDs based on direction and maximum number of cars
@app.get("/v1/trains/get_train_ids_by_direction_and_max_cars", operation_id="get_train_ids_by_direction_and_max_cars", summary="Retrieves the IDs of trains traveling in a specified direction with a number of cars less than a given maximum. The operation filters trains based on their direction and the maximum number of cars they can accommodate, returning a list of matching train IDs.")
async def get_train_ids_by_direction_and_max_cars(direction: str = Query(..., description="Direction of the train"), max_cars: int = Query(..., description="Maximum number of cars")):
    cursor.execute("SELECT T1.id FROM trains AS T1 INNER JOIN ( SELECT train_id, MAX(position) AS carsNum FROM cars GROUP BY train_id ) AS T2 ON T1.id = T2.train_id WHERE T1.direction = ? AND T2.carsNum < ?", (direction, max_cars))
    result = cursor.fetchall()
    if not result:
        return {"train_ids": []}
    return {"train_ids": [row[0] for row in result]}

# Endpoint to get car IDs based on train direction and car sides
@app.get("/v1/trains/get_car_ids_by_train_direction_and_sides", operation_id="get_car_ids_by_train_direction_and_sides", summary="Retrieves the IDs of cars that match the specified train direction and car sides. This operation filters cars based on the provided direction of the train and the sides of the car, returning a list of IDs for the cars that meet the criteria.")
async def get_car_ids_by_train_direction_and_sides(direction: str = Query(..., description="Direction of the train"), sides: str = Query(..., description="Sides of the car")):
    cursor.execute("SELECT T1.id FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T2.direction = ? AND T1.sides = ?", (direction, sides))
    result = cursor.fetchall()
    if not result:
        return {"car_ids": []}
    return {"car_ids": [row[0] for row in result]}

# Endpoint to get the count of trains with more than a specified number of long cars in a specific direction
@app.get("/v1/trains/count_trains_with_long_cars_in_direction", operation_id="get_count_trains_with_long_cars_in_direction", summary="Retrieves the total count of trains that have more than the specified number of long cars (cars with the given length) traveling in the provided direction.")
async def get_count_trains_with_long_cars_in_direction(long_cars_num: int = Query(..., description="Number of long cars"), car_length: str = Query(..., description="Length of the car"), direction: str = Query(..., description="Direction of the train")):
    cursor.execute("SELECT SUM(CASE WHEN T2.longCarsNum > ? THEN 1 ELSE 0 END)as count FROM trains AS T1 INNER JOIN ( SELECT train_id, COUNT(id) AS longCarsNum FROM cars WHERE len = ? GROUP BY train_id ) AS T2 ON T1.id = T2.train_id WHERE T1.direction = ?", (long_cars_num, car_length, direction))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the direction of trains based on load number
@app.get("/v1/trains/direction_by_load_num", operation_id="get_direction_by_load_num", summary="Retrieves the direction of travel for trains associated with a specified load number of cars. The load number is used to identify the relevant train and determine its direction.")
async def get_direction_by_load_num(load_num: int = Query(..., description="Load number of the car")):
    cursor.execute("SELECT T2.direction FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.load_num = ?", (load_num,))
    result = cursor.fetchall()
    if not result:
        return {"directions": []}
    return {"directions": [row[0] for row in result]}

# Endpoint to get the direction of trains based on car shape
@app.get("/v1/trains/direction_by_shape", operation_id="get_direction_by_shape", summary="Retrieves the direction of trains that have cars matching the specified shape. The input parameter is used to filter the cars by their shape.")
async def get_direction_by_shape(shape: str = Query(..., description="Shape of the car")):
    cursor.execute("SELECT T2.direction FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.shape = ?", (shape,))
    result = cursor.fetchall()
    if not result:
        return {"directions": []}
    return {"directions": [row[0] for row in result]}

# Endpoint to get the count of cars with a specific length and direction
@app.get("/v1/trains/count_cars_by_length_and_direction", operation_id="get_count_cars_by_length_and_direction", summary="Retrieves the total number of cars of a specific length that are part of trains moving in a certain direction. The operation considers the length of the cars and the direction of the trains to calculate the count.")
async def get_count_cars_by_length_and_direction(length: str = Query(..., description="Length of the car"), direction: str = Query(..., description="Direction of the train")):
    cursor.execute("SELECT SUM(CASE WHEN T1.len = ? THEN 1 ELSE 0 END) AS count FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T2.direction = ?", (length, direction))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the shapes of cars based on direction and position
@app.get("/v1/trains/shapes_by_direction_and_position", operation_id="get_shapes_by_direction_and_position", summary="Retrieves the distinct shapes of cars that match the specified train direction and car position. The operation filters cars based on the provided direction of the train and the position of the car, then returns the unique shapes of these cars.")
async def get_shapes_by_direction_and_position(direction: str = Query(..., description="Direction of the train"), position: int = Query(..., description="Position of the car")):
    cursor.execute("SELECT T1.shape FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T2.direction = ? AND T1.position = ? GROUP BY T1.shape", (direction, position))
    result = cursor.fetchall()
    if not result:
        return {"shapes": []}
    return {"shapes": [row[0] for row in result]}

# Endpoint to get the count of cars with a specific roof type and direction
@app.get("/v1/trains/count_cars_by_roof_and_direction", operation_id="get_count_cars_by_roof_and_direction", summary="Retrieves the total number of cars with a specified roof type that are part of trains traveling in a given direction.")
async def get_count_cars_by_roof_and_direction(roof: str = Query(..., description="Roof type of the car"), direction: str = Query(..., description="Direction of the train")):
    cursor.execute("SELECT SUM(CASE WHEN T1.roof = ? THEN 1 ELSE 0 END) AS count FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T2.direction = ?", (roof, direction))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of cars with a specific load shape, direction, and roof type
@app.get("/v1/trains/count_cars_by_load_shape_direction_and_roof", operation_id="get_count_cars_by_load_shape_direction_and_roof", summary="Retrieves the total number of cars that match a specified load shape, direction, and roof type. This operation considers the relationship between cars and trains, ensuring that the count is accurate for the given train direction. The roof type of the cars is also taken into account.")
async def get_count_cars_by_load_shape_direction_and_roof(load_shape: str = Query(..., description="Load shape of the car"), direction: str = Query(..., description="Direction of the train"), roof: str = Query(..., description="Roof type of the car")):
    cursor.execute("SELECT SUM(CASE WHEN T1.load_shape = ? THEN 1 ELSE 0 END) AS count FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T2.direction = ? AND T1.roof = ?", (load_shape, direction, roof))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the direction of trains ordered by the number of cars with a specific shape
@app.get("/v1/trains/direction_by_shape_ordered", operation_id="get_direction_by_shape_ordered", summary="Retrieve the direction of trains, sorted by the count of cars with a specified shape. The operation calculates the number of cars with the given shape for each train and orders the results in descending order based on this count.")
async def get_direction_by_shape_ordered(shape: str = Query(..., description="Shape of the car")):
    cursor.execute("SELECT T1.direction FROM trains AS T1 INNER JOIN (SELECT train_id, COUNT(id) AS rectCarsNum FROM cars WHERE shape = ? GROUP BY train_id) AS T2 ON T1.id = T2.train_id ORDER BY T2.rectCarsNum DESC", (shape,))
    result = cursor.fetchall()
    if not result:
        return {"directions": []}
    return {"directions": [row[0] for row in result]}

# Endpoint to get the direction of trains based on car length and position
@app.get("/v1/trains/direction_by_length_and_position", operation_id="get_direction_by_length_and_position", summary="Retrieves the direction of travel for trains that have cars of a specified length and position. The endpoint uses the length and position of cars to determine the direction of the associated train.")
async def get_direction_by_length_and_position(length: str = Query(..., description="Length of the car"), position: int = Query(..., description="Position of the car")):
    cursor.execute("SELECT T2.direction FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.len = ? AND T1.position = ?", (length, position))
    result = cursor.fetchall()
    if not result:
        return {"directions": []}
    return {"directions": [row[0] for row in result]}

# Endpoint to get the average number of cars per train in a specific direction
@app.get("/v1/trains/average_cars_per_train_by_direction", operation_id="get_average_cars_per_train_by_direction", summary="Retrieves the average number of cars per train traveling in a specified direction. This operation calculates the ratio of the total number of cars to the total number of trains in the given direction.")
async def get_average_cars_per_train_by_direction(direction: str = Query(..., description="Direction of the train")):
    cursor.execute("SELECT CAST(COUNT(T1.id) AS REAL) / COUNT(DISTINCT T1.train_id) FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T2.direction = ?", (direction,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the percentage of trains with a specific direction and car shapes
@app.get("/v1/trains/percentage_trains_by_direction_and_shapes", operation_id="get_percentage_trains_by_direction_and_shapes", summary="Retrieves the percentage of trains that have a specified direction and contain cars of certain shapes. The direction and shapes are provided as input parameters. The result is calculated by counting the distinct trains that meet the criteria and dividing by the total number of distinct trains.")
async def get_percentage_trains_by_direction_and_shapes(direction: str = Query(..., description="Direction of the train"), shape1: str = Query(..., description="First shape of the car"), shape2: str = Query(..., description="Second shape of the car")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT CASE WHEN T2.direction = ? THEN T1.train_id ELSE NULL END) AS REAL) * 100 / COUNT(DISTINCT T1.train_id) FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.shape IN (?, ?)", (direction, shape1, shape2))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of cars based on shape and length
@app.get("/v1/trains/count_cars_by_shape_and_length", operation_id="get_count_cars_by_shape_and_length", summary="Retrieves the total number of cars that match the specified shape and length. The operation filters the cars based on the provided shape and length parameters, and returns the count of cars that meet the criteria.")
async def get_count_cars_by_shape_and_length(shape: str = Query(..., description="Shape of the car"), len: str = Query(..., description="Length of the car")):
    cursor.execute("SELECT COUNT(id) FROM cars WHERE shape = ? AND len = ?", (shape, len))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the load shape of cars based on shape and length
@app.get("/v1/trains/load_shape_by_shape_and_length", operation_id="get_load_shape_by_shape_and_length", summary="Retrieves the load shape of cars that match the specified shape and length. The operation filters cars based on the provided shape and length parameters, and returns the corresponding load shape data.")
async def get_load_shape_by_shape_and_length(shape: str = Query(..., description="Shape of the car"), len: str = Query(..., description="Length of the car")):
    cursor.execute("SELECT load_shape FROM cars WHERE shape = ? AND len = ?", (shape, len))
    result = cursor.fetchall()
    if not result:
        return {"load_shape": []}
    return {"load_shape": [row[0] for row in result]}

# Endpoint to get the IDs of trains based on direction
@app.get("/v1/trains/train_ids_by_direction", operation_id="get_train_ids_by_direction", summary="Retrieves the unique identifiers of trains that are traveling in the specified direction. The direction parameter is used to filter the trains by their travel direction.")
async def get_train_ids_by_direction(direction: str = Query(..., description="Direction of the train")):
    cursor.execute("SELECT id FROM trains WHERE direction = ?", (direction,))
    result = cursor.fetchall()
    if not result:
        return {"train_ids": []}
    return {"train_ids": [row[0] for row in result]}

# Endpoint to get the total number of wheels for cars based on length
@app.get("/v1/trains/total_wheels_by_length", operation_id="get_total_wheels_by_length", summary="Retrieves the cumulative count of wheels for cars of a specified length. The length parameter is used to filter the cars for which the total wheel count is calculated.")
async def get_total_wheels_by_length(len: str = Query(..., description="Length of the car")):
    cursor.execute("SELECT SUM(wheels) FROM cars WHERE len = ?", (len,))
    result = cursor.fetchone()
    if not result:
        return {"total_wheels": []}
    return {"total_wheels": result[0]}

# Endpoint to get the directions of trains ordered by the count of trains in each direction
@app.get("/v1/trains/directions_ordered_by_count", operation_id="get_directions_ordered_by_count", summary="Retrieves a list of unique train directions, sorted in descending order based on the count of trains traveling in each direction. This operation provides insights into the most popular train routes.")
async def get_directions_ordered_by_count():
    cursor.execute("SELECT direction FROM trains GROUP BY direction ORDER BY COUNT(id) DESC")
    result = cursor.fetchall()
    if not result:
        return {"directions": []}
    return {"directions": [row[0] for row in result]}

# Endpoint to get the count of trains with a specific direction and a minimum number of cars
@app.get("/v1/trains/count_trains_by_direction_and_min_cars", operation_id="get_count_trains_by_direction_and_min_cars", summary="Retrieves the total number of trains traveling in a specified direction and having at least a certain number of cars. The direction and minimum number of cars are provided as input parameters.")
async def get_count_trains_by_direction_and_min_cars(direction: str = Query(..., description="Direction of the train"), min_cars: int = Query(..., description="Minimum number of cars")):
    cursor.execute("SELECT SUM(CASE WHEN T1.direction = ? THEN 1 ELSE 0 END) as count FROM trains AS T1 INNER JOIN ( SELECT train_id, COUNT(id) AS carsNum FROM cars GROUP BY train_id ) AS T2 ON T1.id = T2.train_id WHERE T2.carsNum >= ?", (direction, min_cars))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common direction of trains for cars with a specific position and shape
@app.get("/v1/trains/most_common_direction_by_position_and_shape", operation_id="get_most_common_direction_by_position_and_shape", summary="Retrieves the most frequently occurring direction of trains for cars with a specified position and shape. The operation considers the position and shape of the cars to determine the most common direction of the associated trains.")
async def get_most_common_direction_by_position_and_shape(position: int = Query(..., description="Position of the car"), shape: str = Query(..., description="Shape of the car")):
    cursor.execute("SELECT T2.direction FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.position = ? AND T1.shape = ? GROUP BY T2.direction ORDER BY COUNT(T2.id) DESC LIMIT 1", (position, shape))
    result = cursor.fetchone()
    if not result:
        return {"direction": []}
    return {"direction": result[0]}

# Endpoint to get the count of train IDs based on car position, train direction, and car sides
@app.get("/v1/trains/count_train_ids_by_position_direction_sides", operation_id="get_count_train_ids_by_position_direction_sides", summary="Retrieves the total number of unique train IDs that meet the specified car position, train direction, and car sides criteria. The count is determined by grouping the cars based on their train IDs, and then counting the distinct train IDs that match the provided parameters.")
async def get_count_train_ids_by_position_direction_sides(position: int = Query(..., description="Position of the car"), direction: str = Query(..., description="Direction of the train"), sides: str = Query(..., description="Sides of the car")):
    cursor.execute("SELECT COUNT(T.train_id) FROM (SELECT T1.train_id FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.position = ? AND T2.direction = ? AND T1.sides = ? GROUP BY T1.train_id) as T", (position, direction, sides))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of train IDs based on car position, train direction, and car shape
@app.get("/v1/trains/count_train_ids_by_position_direction_shape", operation_id="get_count_train_ids_by_position_direction_shape", summary="Retrieves the total number of unique train IDs that meet the specified car position, train direction, and car shape criteria. This operation aggregates data from the cars and trains tables, filtering by the provided parameters to ensure accurate results.")
async def get_count_train_ids_by_position_direction_shape(position: int = Query(..., description="Position of the car"), direction: str = Query(..., description="Direction of the train"), shape: str = Query(..., description="Shape of the car")):
    cursor.execute("SELECT COUNT(T.train_id) FROM (SELECT T1.train_id FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.position = ? AND T2.direction = ? AND T1.shape = ? GROUP BY T1.train_id) as T", (position, direction, shape))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of trains going in a specific direction with a specific roof type and car count
@app.get("/v1/trains/count_trains_direction_roof_car_count", operation_id="get_count_trains_direction_roof_car_count", summary="Retrieves the total number of trains traveling in a specified direction that have a certain roof type and a specific count of cars. The direction, roof type, and car count are provided as input parameters.")
async def get_count_trains_direction_roof_car_count(direction: str = Query(..., description="Direction of the train"), roof: str = Query(..., description="Roof type of the car"), car_count: int = Query(..., description="Number of cars")):
    cursor.execute("SELECT SUM(CASE WHEN T1.direction = ? THEN 1 ELSE 0 END) as count FROM trains AS T1 INNER JOIN ( SELECT train_id, COUNT(id) FROM cars WHERE roof = ? GROUP BY train_id HAVING COUNT(id) = ? ) AS T2 ON T1.id = T2.train_id", (direction, roof, car_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the directions of trains with a specific number of cars
@app.get("/v1/trains/directions_trains_car_count", operation_id="get_directions_trains_car_count", summary="Retrieves the directions of trains that have a specified number of cars. The operation returns a list of unique train directions where the count of associated cars matches the provided input parameter. This endpoint is useful for identifying the routes of trains with a certain capacity.")
async def get_directions_trains_car_count(car_count: int = Query(..., description="Number of cars")):
    cursor.execute("SELECT T1.direction FROM trains AS T1 INNER JOIN ( SELECT train_id, COUNT(id) AS carsNum FROM cars GROUP BY train_id HAVING carsNum = ? ) AS T2 ON T1.id = T2.train_id GROUP BY T1.direction", (car_count,))
    result = cursor.fetchall()
    if not result:
        return {"directions": []}
    return {"directions": [row[0] for row in result]}

# Endpoint to get the count of distinct train IDs with a specific position and load number
@app.get("/v1/trains/count_distinct_train_ids_position_load_num", operation_id="get_count_distinct_train_ids_position_load_num", summary="Retrieves the number of unique trains that have a car at a specific position with a given load number. This operation considers the relationship between cars and trains, filtering the results based on the provided position and load number of the car.")
async def get_count_distinct_train_ids_position_load_num(position: int = Query(..., description="Position of the car"), load_num: int = Query(..., description="Load number of the car")):
    cursor.execute("SELECT COUNT(DISTINCT T1.train_id) FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.position = ? AND T1.load_num = ?", (position, load_num))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of trains with a specific direction, position, and sides
@app.get("/v1/trains/count_trains_direction_position_sides", operation_id="get_count_trains_direction_position_sides", summary="Retrieves the total number of trains that meet the specified direction and have cars with the given sides at the last position. The count is determined by considering the direction of the train and the sides of the car at the final position of the train.")
async def get_count_trains_direction_position_sides(direction: str = Query(..., description="Direction of the train"), sides: str = Query(..., description="Sides of the car")):
    cursor.execute("SELECT COUNT(T1.id) FROM trains AS T1 INNER JOIN cars AS T2 ON T1.id = T2.train_id INNER JOIN ( SELECT train_id, MAX(position) AS trailPosi FROM cars GROUP BY train_id ) AS T3 ON T1.id = T3.train_id WHERE T1.direction = ? AND T2.position = T3.trailPosi AND T2.sides = ?", (direction, sides))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the directions of trains with a specific position and shape
@app.get("/v1/trains/directions_trains_position_shape", operation_id="get_directions_trains_position_shape", summary="Retrieves the direction of trains associated with a car at a specific position and shape. The position and shape of the car are used to identify the relevant trains and their directions.")
async def get_directions_trains_position_shape(position: int = Query(..., description="Position of the car"), shape: str = Query(..., description="Shape of the car")):
    cursor.execute("SELECT T2.direction FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.position = ? AND T1.shape = ?", (position, shape))
    result = cursor.fetchall()
    if not result:
        return {"directions": []}
    return {"directions": [row[0] for row in result]}

# Endpoint to get the count of trains with a specific direction, wheels, and roof
@app.get("/v1/trains/count_trains_direction_wheels_roof", operation_id="get_count_trains_direction_wheels_roof", summary="Retrieves the total number of trains that meet the specified criteria for direction, number of wheels, and roof type of the cars they carry. This operation considers the direction of the train and the characteristics of its cars, including the number of wheels and roof type.")
async def get_count_trains_direction_wheels_roof(direction: str = Query(..., description="Direction of the train"), wheels: int = Query(..., description="Number of wheels"), roof: str = Query(..., description="Roof type of the car")):
    cursor.execute("SELECT SUM(CASE WHEN T2.direction = ? THEN 1 ELSE 0 END) as count FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.wheels = ? AND T1.roof = ?", (direction, wheels, roof))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the directions of trains with a specific maximum position
@app.get("/v1/trains/directions_trains_max_position", operation_id="get_directions_trains_max_position", summary="Retrieves the directions of trains that have a car with a position less than or equal to the specified maximum position. The operation identifies the maximum position of each train and returns the direction of those trains that meet the specified position criteria.")
async def get_directions_trains_max_position(max_position: int = Query(..., description="Maximum position of the car")):
    cursor.execute("SELECT T1.direction FROM trains AS T1 INNER JOIN ( SELECT train_id, MAX(position) AS trailPosi FROM cars GROUP BY train_id ) AS T2 ON T1.id = T2.train_id WHERE T2.trailPosi <= ?", (max_position,))
    result = cursor.fetchall()
    if not result:
        return {"directions": []}
    return {"directions": [row[0] for row in result]}

# Endpoint to get the percentage of trains with a specific minimum position and their directions
@app.get("/v1/trains/percentage_trains_min_position_directions", operation_id="get_percentage_trains_min_position_directions", summary="Retrieves the percentage of trains that have at least one car at or beyond a specified minimum position. The operation also returns the direction of these trains. The input parameter, min_position, is used to determine the minimum position of the car.")
async def get_percentage_trains_min_position_directions(min_position: int = Query(..., description="Minimum position of the car")):
    cursor.execute("SELECT CAST(COUNT(CASE WHEN T2.trailPosi >= ? THEN T1.id ELSE NULL END) AS REAL) * 100 / COUNT(T1.id) FROM trains AS T1 INNER JOIN ( SELECT train_id, MAX(position) AS trailPosi FROM cars GROUP BY train_id ) AS T2 ON T1.id = T2.train_id UNION ALL SELECT T1.direction FROM trains AS T1 INNER JOIN ( SELECT train_id, MAX(position) AS trailPosi FROM cars t GROUP BY train_id ) AS T2 ON T1.id = T2.train_id AND T2.trailPosi >= ?", (min_position, min_position))
    result = cursor.fetchall()
    if not result:
        return {"percentage": [], "directions": []}
    return {"percentage": result[0][0], "directions": [row[0] for row in result[1:]]}

# Endpoint to get distinct load shapes and percentage of trains with a specific direction and load shape
@app.get("/v1/trains/distinct_load_shapes_percentage_direction_load_shape", operation_id="get_distinct_load_shapes_percentage_direction_load_shape", summary="Retrieves distinct load shapes of cars and the percentage of trains with a specific direction and load shape. The position parameter determines the car's position, while the direction parameter specifies the train's direction. The operation returns a list of unique load shapes and the percentage of trains that match the given direction and load shape.")
async def get_distinct_load_shapes_percentage_direction_load_shape(position: int = Query(..., description="Position of the car"), direction: str = Query(..., description="Direction of the train")):
    cursor.execute("SELECT DISTINCT T3.load_shape FROM ( SELECT load_shape, train_id FROM cars WHERE position = ? ORDER BY train_id DESC ) AS T3 UNION ALL SELECT T4.load_shape FROM ( SELECT load_shape, train_id FROM cars WHERE position = ? ORDER BY train_id DESC LIMIT 1 ) AS T4 UNION ALL SELECT (CAST(COUNT(DISTINCT CASE WHEN T2.direction = ? THEN T2.id ELSE NULL END) AS REAL) * 100 / COUNT(DISTINCT T2.id)) FROM cars AS T1 INNER JOIN trains AS T2 ON T1.train_id = T2.id WHERE T1.position = ? AND T1.load_shape = ( SELECT T4.load_shape FROM ( SELECT load_shape, train_id FROM cars AS T WHERE position = ? ORDER BY train_id DESC LIMIT 1 ) AS T4 )", (position, position, direction, position, position))
    result = cursor.fetchall()
    if not result:
        return {"load_shapes": [], "percentage": []}
    return {"load_shapes": [row[0] for row in result[:-1]], "percentage": result[-1][0]}

api_calls = [
    "/v1/trains/count_trains_by_direction?direction=east",
    "/v1/trains/count_cars_by_train_id?train_id=1",
    "/v1/trains/get_car_shape_by_train_id_and_position?train_id=1&position=4",
    "/v1/trains/get_train_ids_by_car_shapes?shape1=elipse&shape2=bucket",
    "/v1/trains/count_cars_by_train_id_and_roof?train_id=1&roof=none",
    "/v1/trains/get_car_ids_by_train_id_and_wheels?train_id=1&wheels=2",
    "/v1/trains/count_cars_with_shapes_in_direction?shape1=bucket&shape2=elipse&direction=east",
    "/v1/trains/get_train_ids_by_direction_and_max_cars?direction=east&max_cars=4",
    "/v1/trains/get_car_ids_by_train_direction_and_sides?direction=east&sides=double",
    "/v1/trains/count_trains_with_long_cars_in_direction?long_cars_num=2&car_length=long&direction=west",
    "/v1/trains/direction_by_load_num?load_num=0",
    "/v1/trains/direction_by_shape?shape=ellipse",
    "/v1/trains/count_cars_by_length_and_direction?length=short&direction=east",
    "/v1/trains/shapes_by_direction_and_position?direction=east&position=1",
    "/v1/trains/count_cars_by_roof_and_direction?roof=flat&direction=east",
    "/v1/trains/count_cars_by_load_shape_direction_and_roof?load_shape=circle&direction=east&roof=flat",
    "/v1/trains/direction_by_shape_ordered?shape=rectangle",
    "/v1/trains/direction_by_length_and_position?length=short&position=4",
    "/v1/trains/average_cars_per_train_by_direction?direction=east",
    "/v1/trains/percentage_trains_by_direction_and_shapes?direction=east&shape1=bucket&shape2=ellipse",
    "/v1/trains/count_cars_by_shape_and_length?shape=hexagon&len=short",
    "/v1/trains/load_shape_by_shape_and_length?shape=ellipse&len=short",
    "/v1/trains/train_ids_by_direction?direction=east",
    "/v1/trains/total_wheels_by_length?len=long",
    "/v1/trains/directions_ordered_by_count",
    "/v1/trains/count_trains_by_direction_and_min_cars?direction=east&min_cars=4",
    "/v1/trains/most_common_direction_by_position_and_shape?position=2&shape=rectangle",
    "/v1/trains/count_train_ids_by_position_direction_sides?position=3&direction=west&sides=double",
    "/v1/trains/count_train_ids_by_position_direction_shape?position=1&direction=east&shape=rectangle",
    "/v1/trains/count_trains_direction_roof_car_count?direction=west&roof=none&car_count=1",
    "/v1/trains/directions_trains_car_count?car_count=3",
    "/v1/trains/count_distinct_train_ids_position_load_num?position=1&load_num=3",
    "/v1/trains/count_trains_direction_position_sides?direction=east&sides=double",
    "/v1/trains/directions_trains_position_shape?position=2&shape=diamond",
    "/v1/trains/count_trains_direction_wheels_roof?direction=west&wheels=3&roof=jagged",
    "/v1/trains/directions_trains_max_position?max_position=2",
    "/v1/trains/percentage_trains_min_position_directions?min_position=4",
    "/v1/trains/distinct_load_shapes_percentage_direction_load_shape?position=1&direction=east"
]
