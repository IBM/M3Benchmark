from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/menu/menu.sqlite')
cursor = conn.cursor()

# Endpoint to get the count of dishes based on first appeared year range
@app.get("/v1/menu/dish_count_by_year_range", operation_id="get_dish_count_by_year_range", summary="Retrieves the total number of dishes that were introduced before a specified minimum year or after a specified maximum year. This operation allows you to analyze the distribution of dishes based on their year of introduction.")
async def get_dish_count_by_year_range(min_year: int = Query(..., description="Minimum year for first appeared"), max_year: int = Query(..., description="Maximum year for first appeared")):
    cursor.execute("SELECT COUNT(*) FROM Dish WHERE first_appeared < ? OR first_appeared > ?", (min_year, max_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to compare the duration of two dishes
@app.get("/v1/menu/compare_dish_duration", operation_id="compare_dish_duration", summary="This operation compares the total duration between the first and last appearance of two dishes in the menu. It returns the name of the dish that has been on the menu for the longest duration. The comparison is made between the provided dish names and 'Fresh lobsters in every style'.")
async def compare_dish_duration(dish1: str = Query(..., description="Name of the first dish"), dish2: str = Query(..., description="Name of the second dish")):
    cursor.execute("SELECT CASE WHEN SUM(CASE WHEN name = ? THEN last_appeared - first_appeared ELSE 0 END) - SUM(CASE WHEN name = ? THEN last_appeared - first_appeared ELSE 0 END) > 0 THEN ? ELSE ? END FROM Dish WHERE name IN (?, ?)", (dish1, dish2, dish1, dish2, dish1, dish2))
    result = cursor.fetchone()
    if not result:
        return {"dish": []}
    return {"dish": result[0]}

# Endpoint to get the name of the dish with the lowest price
@app.get("/v1/menu/dish_with_lowest_price", operation_id="get_dish_with_lowest_price", summary="Retrieves the name of the dish with the lowest price, as specified in the input parameter, from the database. The results are ordered by the number of menus the dish has appeared in, with the dish appearing in the most menus being returned first. Only the top result is returned.")
async def get_dish_with_lowest_price(lowest_price: int = Query(..., description="Lowest price of the dish")):
    cursor.execute("SELECT name FROM Dish WHERE lowest_price = ? ORDER BY menus_appeared DESC LIMIT 1", (lowest_price,))
    result = cursor.fetchone()
    if not result:
        return {"dish_name": []}
    return {"dish_name": result[0]}

# Endpoint to get the count of menus based on name and page count
@app.get("/v1/menu/menu_count_by_name_and_page_count", operation_id="get_menu_count_by_name_and_page_count", summary="Retrieves the total number of menus that match a given name and page count. The name and page count are provided as input parameters, allowing for a targeted search of the menu database.")
async def get_menu_count_by_name_and_page_count(name: str = Query(..., description="Name of the menu"), page_count: int = Query(..., description="Page count of the menu")):
    cursor.execute("SELECT COUNT(*) FROM Menu WHERE name = ? AND page_count = ?", (name, page_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get dish names based on menu page ID and position
@app.get("/v1/menu/dish_names_by_menu_page_and_position", operation_id="get_dish_names_by_menu_page_and_position", summary="Retrieves the names of dishes that are located within the specified coordinates on a given menu page. The menu page is identified by its unique ID, while the coordinates are defined by X and Y positions.")
async def get_dish_names_by_menu_page_and_position(menu_page_id: int = Query(..., description="Menu page ID"), xpos: float = Query(..., description="X position on the menu page"), ypos: float = Query(..., description="Y position on the menu page")):
    cursor.execute("SELECT T1.name FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T2.menu_page_id = ? AND T2.xpos < ? AND T2.ypos < ?", (menu_page_id, xpos, ypos))
    results = cursor.fetchall()
    if not results:
        return {"dish_names": []}
    return {"dish_names": [result[0] for result in results]}

# Endpoint to get the prices of a specific dish
@app.get("/v1/menu/dish_prices_by_name", operation_id="get_dish_prices_by_name", summary="Retrieves the prices of a specific dish from all menus. The dish is identified by its name, which is provided as an input parameter. The operation returns a list of prices for the specified dish, as it appears in different menus.")
async def get_dish_prices_by_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T2.price FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.name = ?", (dish_name,))
    results = cursor.fetchall()
    if not results:
        return {"prices": []}
    return {"prices": [result[0] for result in results]}

# Endpoint to get the count of a specific dish with no highest price
@app.get("/v1/menu/count_dish_no_highest_price", operation_id="get_count_dish_no_highest_price", summary="Retrieves the total count of a specific dish that does not have a highest price set. The dish is identified by its name, which is provided as an input parameter. This operation is useful for determining the availability of a particular dish without a defined highest price.")
async def get_count_dish_no_highest_price(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT SUM(CASE WHEN T1.name = ? THEN 1 ELSE 0 END) FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.highest_price IS NULL", (dish_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest price of a specific dish
@app.get("/v1/menu/highest_price_of_dish", operation_id="get_highest_price_of_dish", summary="Retrieves the highest price of a specific dish from the menu. The operation requires the name of the dish as input and returns the maximum price associated with that dish.")
async def get_highest_price_of_dish(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T2.price FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.name = ? ORDER BY T2.price DESC LIMIT 1", (dish_name,))
    result = cursor.fetchone()
    if not result:
        return {"highest_price": []}
    return {"highest_price": result[0]}

# Endpoint to get menu IDs containing a specific dish
@app.get("/v1/menu/menu_ids_by_dish_name", operation_id="get_menu_ids_by_dish_name", summary="Retrieve the unique identifiers of menus that include a specific dish. The operation filters menus based on the provided dish name and returns their corresponding identifiers.")
async def get_menu_ids_by_dish_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T1.menu_id FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id INNER JOIN Dish AS T3 ON T2.dish_id = T3.id WHERE T3.name = ?", (dish_name,))
    results = cursor.fetchall()
    if not results:
        return {"menu_ids": []}
    return {"menu_ids": [result[0] for result in results]}

# Endpoint to get the count of menus with a specific currency for a specific dish
@app.get("/v1/menu/count_menus_by_currency_and_dish", operation_id="get_count_menus_by_currency_and_dish", summary="Retrieve the total number of menus that utilize a specified currency for a particular dish. This operation calculates the sum of menus that meet the given currency and dish criteria, providing a quantitative overview of menu distribution.")
async def get_count_menus_by_currency_and_dish(currency: str = Query(..., description="Currency used in the menu"), dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT SUM(CASE WHEN T3.currency = ? THEN 1 ELSE 0 END) FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T4.name = ?", (currency, dish_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of a specific dish in menus without a call number
@app.get("/v1/menu/count_dish_without_call_number", operation_id="get_count_dish_without_call_number", summary="Retrieves the total count of a specified dish across menus that do not have a call number. The dish is identified by its name.")
async def get_count_dish_without_call_number(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT SUM(CASE WHEN T4.name = ? THEN 1 ELSE 0 END) FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T3.call_number IS NULL", (dish_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get dish names from a specific menu
@app.get("/v1/menu/dish_names_from_menu", operation_id="get_dish_names_from_menu", summary="Retrieves the names of dishes associated with a specific menu. The operation filters the menu items based on the provided menu name and returns the corresponding dish names.")
async def get_dish_names_from_menu(menu_name: str = Query(..., description="Name of the menu")):
    cursor.execute("SELECT T4.name FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T3.name = ?", (menu_name,))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": [row[0] for row in result]}

# Endpoint to get the most expensive dish from a specific menu
@app.get("/v1/menu/most_expensive_dish_from_menu", operation_id="get_most_expensive_dish_from_menu", summary="Retrieves the name of the most expensive dish available on a specified menu. The menu is identified by its unique name. The operation returns the name of the dish with the highest price on the given menu, providing a quick reference for the priciest option.")
async def get_most_expensive_dish_from_menu(menu_name: str = Query(..., description="Name of the menu")):
    cursor.execute("SELECT T4.name FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T3.name = ? ORDER BY T1.price DESC LIMIT 1", (menu_name,))
    result = cursor.fetchone()
    if not result:
        return {"dish_name": []}
    return {"dish_name": result[0]}

# Endpoint to get the count of a specific menu
@app.get("/v1/menu/count_specific_menu", operation_id="get_count_specific_menu", summary="Retrieves the total count of menu items associated with a specific menu, identified by its name. This operation provides a quantitative overview of the menu's composition, aiding in menu analysis and management.")
async def get_count_specific_menu(menu_name: str = Query(..., description="Name of the menu")):
    cursor.execute("SELECT SUM(CASE WHEN T3.name = ? THEN 1 ELSE 0 END) FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id", (menu_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get menu IDs for a specific dish without a sponsor
@app.get("/v1/menu/menu_ids_for_dish_without_sponsor", operation_id="get_menu_ids_for_dish_without_sponsor", summary="Retrieves the identifiers of menus that include a specific dish, excluding those with a sponsor. The dish is identified by its name.")
async def get_menu_ids_for_dish_without_sponsor(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T2.menu_id FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T4.name = ? AND T3.sponsor IS NULL", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"menu_ids": []}
    return {"menu_ids": [row[0] for row in result]}

# Endpoint to get the average page number for a specific dish
@app.get("/v1/menu/average_page_number_for_dish", operation_id="get_average_page_number_for_dish", summary="Retrieves the average page number on which a specific dish appears across all menus. The dish is identified by its name.")
async def get_average_page_number_for_dish(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT AVG(T1.page_number) FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id INNER JOIN Dish AS T3 ON T2.dish_id = T3.id WHERE T3.name = ?", (dish_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_page_number": []}
    return {"average_page_number": result[0]}

# Endpoint to get the average price of menu items in a specific menu
@app.get("/v1/menu/average_price_in_menu", operation_id="get_average_price_in_menu", summary="Retrieves the average price of all menu items within a specified menu. The calculation is based on the sum of the prices of all items divided by the total number of items in the menu. The menu is identified by its unique name.")
async def get_average_price_in_menu(menu_name: str = Query(..., description="Name of the menu")):
    cursor.execute("SELECT SUM(T1.price) / COUNT(T1.price) FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id WHERE T3.name = ?", (menu_name,))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the count of menu items created on a specific date
@app.get("/v1/menu/count_menu_items_created_on_date", operation_id="get_count_menu_items_created_on_date", summary="Retrieves the total number of menu items that were created on a specific date. The date should be provided in the 'YYYY-MM-DD' format.")
async def get_count_menu_items_created_on_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(*) FROM MenuItem WHERE created_at LIKE ?", (date + '%',))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of menu items for a specific menu page
@app.get("/v1/menu/count_menu_items_for_menu_page", operation_id="get_count_menu_items_for_menu_page", summary="Retrieves the total number of menu items associated with a specific menu page. The operation requires the unique identifier of the menu page as input, which is used to filter the menu items and calculate the count.")
async def get_count_menu_items_for_menu_page(menu_page_id: int = Query(..., description="ID of the menu page")):
    cursor.execute("SELECT COUNT(*) FROM MenuItem WHERE menu_page_id = ?", (menu_page_id,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of menus based on location
@app.get("/v1/menu/count_menus_by_location", operation_id="get_menu_count_by_location", summary="Retrieves the total number of menus available at a specified location. The location parameter is used to filter the count of menus.")
async def get_menu_count_by_location(location: str = Query(..., description="Location of the menu")):
    cursor.execute("SELECT COUNT(*) FROM Menu WHERE location = ?", (location,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of dishes where times appeared is greater than menus appeared
@app.get("/v1/menu/count_dishes_times_appeared_greater_than_menus_appeared", operation_id="get_count_dishes_times_appeared_greater_than_menus_appeared", summary="Retrieves the total number of dishes that have been served more times than the total number of menus they have appeared on. This operation provides a measure of popularity for dishes based on their frequency of appearance across different menus.")
async def get_count_dishes_times_appeared_greater_than_menus_appeared():
    cursor.execute("SELECT COUNT(*) FROM Dish WHERE times_appeared > menus_appeared")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of menus based on venue
@app.get("/v1/menu/count_menus_by_venue", operation_id="get_menu_count_by_venue", summary="Retrieves the total number of menus associated with a specified venue. The venue is identified using the provided input parameter.")
async def get_menu_count_by_venue(venue: str = Query(..., description="Venue of the menu")):
    cursor.execute("SELECT COUNT(*) FROM Menu WHERE venue = ?", (venue,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of menus on a specific date
@app.get("/v1/menu/sum_menus_by_date", operation_id="get_sum_menus_by_date", summary="Retrieves the total count of menus that were active on a specific date. The date should be provided in the 'YYYY-MM-DD' format.")
async def get_sum_menus_by_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT SUM(CASE WHEN T1.date = ? THEN 1 ELSE 0 END) FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id", (date,))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get dish names based on menu page ID
@app.get("/v1/menu/dish_names_by_menu_page_id", operation_id="get_dish_names_by_menu_page_id", summary="Retrieves the names of dishes associated with a specific menu page. The operation uses the provided menu page ID to look up the corresponding dishes and returns their names.")
async def get_dish_names_by_menu_page_id(menu_page_id: int = Query(..., description="Menu page ID")):
    cursor.execute("SELECT T2.name FROM MenuItem AS T1 INNER JOIN Dish AS T2 ON T2.id = T1.dish_id WHERE T1.menu_page_id = ?", (menu_page_id,))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": [row[0] for row in result]}

# Endpoint to get dish names and IDs based on the year they first appeared
@app.get("/v1/menu/dish_names_ids_by_first_appeared", operation_id="get_dish_names_ids_by_first_appeared", summary="Retrieves the names and corresponding IDs of dishes that first appeared in a specified year. The year is provided as an input parameter.")
async def get_dish_names_ids_by_first_appeared(first_appeared: int = Query(..., description="Year the dish first appeared")):
    cursor.execute("SELECT T2.name, T1.dish_id FROM MenuItem AS T1 INNER JOIN Dish AS T2 ON T2.id = T1.dish_id WHERE T2.first_appeared = ?", (first_appeared,))
    result = cursor.fetchall()
    if not result:
        return {"dish_info": []}
    return {"dish_info": [{"name": row[0], "dish_id": row[1]} for row in result]}

# Endpoint to get the most expensive dish created on a specific date
@app.get("/v1/menu/most_expensive_dish_by_date", operation_id="get_most_expensive_dish_by_date", summary="Retrieves the name and price of the most expensive dish created on a specific date. The date is provided in the 'YYYY-MM-DD%' format. The operation returns the top result based on the price, in descending order.")
async def get_most_expensive_dish_by_date(created_at: str = Query(..., description="Date in 'YYYY-MM-DD%' format")):
    cursor.execute("SELECT T1.name, T2.price FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T2.created_at LIKE ? ORDER BY T2.price DESC LIMIT 1", (created_at,))
    result = cursor.fetchone()
    if not result:
        return {"dish_info": []}
    return {"dish_info": {"name": result[0], "price": result[1]}}

# Endpoint to get the dish name from a specific page number ordered by full height
@app.get("/v1/menu/dish_name_by_page_number", operation_id="get_dish_name_by_page_number", summary="Retrieves the name of the dish that is displayed on a specific page, sorted by the full height of the page in descending order. If multiple dishes are found on the same page, the one with the lowest full height is returned. The page number is provided as an input parameter.")
async def get_dish_name_by_page_number(page_number: int = Query(..., description="Page number")):
    cursor.execute("SELECT T3.name FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id INNER JOIN Dish AS T3 ON T2.dish_id = T3.id WHERE T1.page_number = ? ORDER BY T1.full_height DESC, T1.full_height ASC LIMIT 1", (page_number,))
    result = cursor.fetchone()
    if not result:
        return {"dish_name": []}
    return {"dish_name": result[0]}

# Endpoint to get the page number and menu name with the highest page count
@app.get("/v1/menu/page_number_menu_name_highest_page_count", operation_id="get_page_number_menu_name_highest_page_count", summary="Retrieves the page number and corresponding menu name that have the highest page count. This operation returns the most frequently accessed menu page, providing insights into user behavior and menu popularity.")
async def get_page_number_menu_name_highest_page_count():
    cursor.execute("SELECT T1.page_number, T2.name FROM MenuPage AS T1 INNER JOIN Menu AS T2 ON T2.id = T1.menu_id ORDER BY T2.page_count DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"menu_info": []}
    return {"menu_info": {"page_number": result[0], "name": result[1]}}

# Endpoint to get the count of dishes on a specific page number grouped by menu name
@app.get("/v1/menu/count_dishes_by_page_number", operation_id="get_count_dishes_by_page_number", summary="Retrieves the total number of dishes available on a specific page of the menu, grouped by the menu name. The operation returns the highest count of dishes among all menus for the given page number.")
async def get_count_dishes_by_page_number(page_number: int = Query(..., description="Page number")):
    cursor.execute("SELECT COUNT(T1.dish_id) FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id WHERE T2.page_number = ? GROUP BY T3.name ORDER BY T3.dish_count DESC LIMIT 1", (page_number,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get menu details for a specific dish name
@app.get("/v1/menu/menu_details_by_dish_name", operation_id="get_menu_details_by_dish_name", summary="Retrieves the menu details associated with a specific dish. The response includes the menu ID, the x-coordinate position (xpos), and the y-coordinate position (ypos) of the dish on the menu. To use this endpoint, provide the name of the dish as an input parameter.")
async def get_menu_details_by_dish_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T2.menu_id, T1.xpos, T1.ypos FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T4.name = ?", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"menu_details": []}
    return {"menu_details": result}

# Endpoint to get dish names based on sponsor and position
@app.get("/v1/menu/dish_names_by_sponsor_and_position", operation_id="get_dish_names_by_sponsor_and_position", summary="Retrieves the names of dishes from a menu sponsored by a specific sponsor and positioned within the given X and Y coordinates. The sponsor, X position threshold, and Y position threshold are used to filter the results.")
async def get_dish_names_by_sponsor_and_position(sponsor: str = Query(..., description="Sponsor of the menu"), xpos: float = Query(..., description="X position threshold"), ypos: float = Query(..., description="Y position threshold")):
    cursor.execute("SELECT T4.name FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T3.sponsor = ? AND T1.xpos < ? AND T1.ypos < ?", (sponsor, xpos, ypos))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": result}

# Endpoint to get menu name and event for a specific dish name
@app.get("/v1/menu/menu_name_event_by_dish_name", operation_id="get_menu_name_event_by_dish_name", summary="Retrieves the name of the menu and the associated event for a specific dish. The operation filters the menu items by the provided dish name and returns the corresponding menu name and event.")
async def get_menu_name_event_by_dish_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T3.name, T3.event FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T4.name = ?", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"menu_name_event": []}
    return {"menu_name_event": result}

# Endpoint to get the percentage of menu items within a specific position range for a dish name
@app.get("/v1/menu/percentage_menu_items_by_position_and_dish_name", operation_id="get_percentage_menu_items_by_position_and_dish_name", summary="Retrieve the percentage of menu items that fall within a specified range of positions for a given dish name. The positions are defined by a minimum and maximum X and Y value. The dish name is used to filter the menu items.")
async def get_percentage_menu_items_by_position_and_dish_name(xpos_min: float = Query(..., description="Minimum X position"), xpos_max: float = Query(..., description="Maximum X position"), ypos_min: float = Query(..., description="Minimum Y position"), ypos_max: float = Query(..., description="Maximum Y position"), dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.xpos BETWEEN ? AND ? AND T2.ypos BETWEEN ? AND ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.id) FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.name LIKE ?", (xpos_min, xpos_max, ypos_min, ypos_max, dish_name))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the sponsor of a menu by menu ID
@app.get("/v1/menu/sponsor_by_menu_id", operation_id="get_sponsor_by_menu_id", summary="Retrieves the sponsor associated with a specific menu, identified by its unique menu ID.")
async def get_sponsor_by_menu_id(menu_id: int = Query(..., description="ID of the menu")):
    cursor.execute("SELECT sponsor FROM Menu WHERE id = ?", (menu_id,))
    result = cursor.fetchone()
    if not result:
        return {"sponsor": []}
    return {"sponsor": result[0]}

# Endpoint to get the event of a menu by sponsor
@app.get("/v1/menu/event_by_sponsor", operation_id="get_event_by_sponsor", summary="Retrieves the event associated with a specific menu sponsored by the provided sponsor.")
async def get_event_by_sponsor(sponsor: str = Query(..., description="Sponsor of the menu")):
    cursor.execute("SELECT event FROM Menu WHERE sponsor = ?", (sponsor,))
    result = cursor.fetchone()
    if not result:
        return {"event": []}
    return {"event": result[0]}

# Endpoint to get the location of a menu by menu ID
@app.get("/v1/menu/location_by_menu_id", operation_id="get_location_by_menu_id", summary="Retrieves the location associated with a specific menu, identified by its unique menu ID.")
async def get_location_by_menu_id(menu_id: int = Query(..., description="ID of the menu")):
    cursor.execute("SELECT location FROM Menu WHERE id = ?", (menu_id,))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": result[0]}

# Endpoint to get the physical description of a menu by sponsor
@app.get("/v1/menu/physical_description_by_sponsor", operation_id="get_physical_description_by_sponsor", summary="Retrieves the physical description of a menu associated with a specific sponsor. The sponsor parameter is used to filter the menu records and return the corresponding physical description.")
async def get_physical_description_by_sponsor(sponsor: str = Query(..., description="Sponsor of the menu")):
    cursor.execute("SELECT physical_description FROM Menu WHERE sponsor = ?", (sponsor,))
    result = cursor.fetchone()
    if not result:
        return {"physical_description": []}
    return {"physical_description": result[0]}

# Endpoint to get the occasion of a menu by menu ID
@app.get("/v1/menu/occasion_by_menu_id", operation_id="get_occasion_by_menu_id", summary="Retrieves the occasion associated with a specific menu, identified by its unique menu ID. This operation allows you to determine the occasion for which a particular menu is intended.")
async def get_occasion_by_menu_id(menu_id: int = Query(..., description="ID of the menu")):
    cursor.execute("SELECT occasion FROM Menu WHERE id = ?", (menu_id,))
    result = cursor.fetchone()
    if not result:
        return {"occasion": []}
    return {"occasion": result[0]}

# Endpoint to get the location of a menu by sponsor
@app.get("/v1/menu/location_by_sponsor", operation_id="get_location_by_sponsor", summary="Get the location of a menu by sponsor")
async def get_location_by_sponsor(sponsor: str = Query(..., description="Sponsor of the menu")):
    cursor.execute("SELECT location FROM Menu WHERE sponsor = ?", (sponsor,))
    result = cursor.fetchone()
    if not result:
        return {"location": []}
    return {"location": result[0]}

# Endpoint to get menu IDs based on sponsor
@app.get("/v1/menu/menu_ids_by_sponsor", operation_id="get_menu_ids_by_sponsor", summary="Retrieves a list of menu IDs associated with the specified sponsor. The sponsor parameter is used to filter the results.")
async def get_menu_ids_by_sponsor(sponsor: str = Query(..., description="Sponsor of the menu")):
    cursor.execute("SELECT T2.id FROM MenuPage AS T1 INNER JOIN Menu AS T2 ON T2.id = T1.menu_id WHERE T2.sponsor = ?", (sponsor,))
    result = cursor.fetchall()
    if not result:
        return {"menu_ids": []}
    return {"menu_ids": [row[0] for row in result]}

# Endpoint to get image IDs based on menu location
@app.get("/v1/menu/image_ids_by_location", operation_id="get_image_ids_by_location", summary="Retrieves a list of image IDs associated with a specific menu location. The location parameter is used to filter the results, returning only the image IDs that correspond to the specified location.")
async def get_image_ids_by_location(location: str = Query(..., description="Location of the menu")):
    cursor.execute("SELECT T1.image_id FROM MenuPage AS T1 INNER JOIN Menu AS T2 ON T2.id = T1.menu_id WHERE T2.location = ?", (location,))
    result = cursor.fetchall()
    if not result:
        return {"image_ids": []}
    return {"image_ids": [row[0] for row in result]}

# Endpoint to get full height and width of menu pages based on menu name
@app.get("/v1/menu/dimensions_by_menu_name", operation_id="get_dimensions_by_menu_name", summary="Retrieves the full height and width dimensions of menu pages associated with a specific menu, identified by its name.")
async def get_dimensions_by_menu_name(name: str = Query(..., description="Name of the menu")):
    cursor.execute("SELECT T1.full_height, T1.full_width FROM MenuPage AS T1 INNER JOIN Menu AS T2 ON T2.id = T1.menu_id WHERE T2.name = ?", (name,))
    result = cursor.fetchall()
    if not result:
        return {"dimensions": []}
    return {"dimensions": [{"full_height": row[0], "full_width": row[1]} for row in result]}

# Endpoint to check if a menu page UUID matches a given UUID for a specific menu name
@app.get("/v1/menu/check_uuid_by_menu_name", operation_id="check_uuid_by_menu_name", summary="Verifies if a menu page with the provided UUID exists under a specific menu name. The operation returns 'yes' if the UUID matches the menu page under the given menu name, and 'no' otherwise.")
async def check_uuid_by_menu_name(name: str = Query(..., description="Name of the menu"), uuid: str = Query(..., description="UUID of the menu page")):
    cursor.execute("SELECT CASE WHEN T2.uuid = ? THEN 'yes' ELSE 'no' END AS yn FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T1.name = ? AND T2.uuid = ?", (uuid, name, uuid))
    result = cursor.fetchone()
    if not result:
        return {"match": []}
    return {"match": result[0]}

# Endpoint to get the name of the menu with the highest full height
@app.get("/v1/menu/highest_full_height_menu", operation_id="get_highest_full_height_menu", summary="Retrieves the name of the menu with the maximum full height. This operation fetches the menu name from the database by joining the MenuPage and Menu tables, ordering the results by full height in descending order, and returning the top result.")
async def get_highest_full_height_menu():
    cursor.execute("SELECT T2.name FROM MenuPage AS T1 INNER JOIN Menu AS T2 ON T2.id = T1.menu_id ORDER BY T1.full_height DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"menu_name": []}
    return {"menu_name": result[0]}

# Endpoint to get the page count of a menu based on menu page ID
@app.get("/v1/menu/page_count_by_menu_page_id", operation_id="get_page_count_by_menu_page_id", summary="Retrieves the total number of pages in a specific menu, identified by its unique menu page ID. This operation provides a count of the pages associated with the given menu page, offering insights into the menu's structure and content.")
async def get_page_count_by_menu_page_id(menu_page_id: int = Query(..., description="ID of the menu page")):
    cursor.execute("SELECT T1.page_count FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T2.id = ?", (menu_page_id,))
    result = cursor.fetchone()
    if not result:
        return {"page_count": []}
    return {"page_count": result[0]}

# Endpoint to get prices of menu items based on image ID
@app.get("/v1/menu/prices_by_image_id", operation_id="get_prices_by_image_id", summary="Retrieves the prices of menu items associated with a specific image. The image is identified by its unique ID, which is used to look up the corresponding menu items and their respective prices.")
async def get_prices_by_image_id(image_id: int = Query(..., description="ID of the image")):
    cursor.execute("SELECT T3.price FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id INNER JOIN MenuItem AS T3 ON T2.id = T3.menu_page_id WHERE T2.image_id = ?", (image_id,))
    result = cursor.fetchall()
    if not result:
        return {"prices": []}
    return {"prices": [row[0] for row in result]}

# Endpoint to get page numbers based on menu item position
@app.get("/v1/menu/page_numbers_by_position", operation_id="get_page_numbers_by_position", summary="Retrieves the page numbers of menu items that are positioned within the specified X and Y coordinates. This operation allows you to determine the pages where menu items are located based on their horizontal and vertical positions. The X and Y coordinates are used to filter the menu items and identify their corresponding pages.")
async def get_page_numbers_by_position(xpos: float = Query(..., description="X position of the menu item"), ypos: float = Query(..., description="Y position of the menu item")):
    cursor.execute("SELECT T2.page_number FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id INNER JOIN MenuItem AS T3 ON T2.id = T3.menu_page_id WHERE T3.xpos > ? AND T3.ypos < ?", (xpos, ypos))
    result = cursor.fetchall()
    if not result:
        return {"page_numbers": []}
    return {"page_numbers": [row[0] for row in result]}

# Endpoint to get dish names based on the month of creation
@app.get("/v1/menu/dish_names_by_month", operation_id="get_dish_names_by_month", summary="Retrieves the names of dishes that were created in a specific month. The month is represented by a single digit, which corresponds to the month of creation. The operation returns a list of dish names that match the provided month.")
async def get_dish_names_by_month(month: str = Query(..., description="Month of creation (single digit)")):
    cursor.execute("SELECT T2.name FROM MenuItem AS T1 INNER JOIN Dish AS T2 ON T2.id = T1.dish_id WHERE SUBSTR(T1.created_at, 7, 1) = ?", (month,))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": [row[0] for row in result]}

# Endpoint to get the highest price of a dish within a given range of menu item IDs
@app.get("/v1/menu/highest_price_in_range", operation_id="get_highest_price_in_range", summary="Retrieve the maximum price of a dish that falls within the range of menu items specified by the provided minimum and maximum menu item IDs.")
async def get_highest_price_in_range(min_id: int = Query(..., description="Minimum menu item ID"), max_id: int = Query(..., description="Maximum menu item ID")):
    cursor.execute("SELECT T2.price FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T2.id BETWEEN ? AND ? ORDER BY T2.price DESC LIMIT 1", (min_id, max_id))
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get dish names based on position coordinates
@app.get("/v1/menu/dish_names_by_position", operation_id="get_dish_names_by_position", summary="Retrieves the names of dishes that are located at the specified position coordinates on the menu. The position coordinates are defined by the input parameters xpos and ypos.")
async def get_dish_names_by_position(xpos: float = Query(..., description="X position coordinate"), ypos: float = Query(..., description="Y position coordinate")):
    cursor.execute("SELECT T1.name FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T2.xpos < ? AND T2.ypos < ?", (xpos, ypos))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the difference in dish counts between two UUIDs
@app.get("/v1/menu/dish_count_difference", operation_id="get_dish_count_difference", summary="Retrieves the difference in total dish counts between two menus identified by their respective UUIDs. This operation compares the sum of dish counts in each menu and returns the difference, providing insights into the relative size or composition of the menus.")
async def get_dish_count_difference(uuid1: str = Query(..., description="First UUID"), uuid2: str = Query(..., description="Second UUID")):
    cursor.execute("SELECT SUM(CASE WHEN T2.uuid = ? THEN T1.dish_count ELSE 0 END) - SUM(CASE WHEN T2.uuid = ? THEN T1.dish_count ELSE 0 END) FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id", (uuid1, uuid2))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the total price of menu items for a given menu ID
@app.get("/v1/menu/total_price_by_menu_id", operation_id="get_total_price_by_menu_id", summary="Retrieves the total price of all menu items associated with a specific menu, identified by its unique menu ID. This operation calculates the sum of the prices of all items listed under the given menu, providing a comprehensive cost overview.")
async def get_total_price_by_menu_id(menu_id: int = Query(..., description="Menu ID")):
    cursor.execute("SELECT SUM(T2.price) FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id WHERE T1.menu_id = ?", (menu_id,))
    result = cursor.fetchone()
    if not result:
        return {"total_price": []}
    return {"total_price": result[0]}

# Endpoint to get dish names ordered by highest price with a limit
@app.get("/v1/menu/dish_names_by_highest_price", operation_id="get_dish_names_by_highest_price", summary="Retrieves a specified number of dish names, ordered by their highest price in descending order.")
async def get_dish_names_by_highest_price(limit: int = Query(..., description="Limit of dish names to return")):
    cursor.execute("SELECT name FROM Dish ORDER BY highest_price DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of dishes with a specific lowest price
@app.get("/v1/menu/dish_count_by_lowest_price", operation_id="get_dish_count_by_lowest_price", summary="Retrieves the total number of dishes that have a specified lowest price. The input parameter determines the price point for the count.")
async def get_dish_count_by_lowest_price(lowest_price: float = Query(..., description="Lowest price of the dish")):
    cursor.execute("SELECT COUNT(*) FROM Dish WHERE lowest_price = ?", (lowest_price,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get dish names based on first and last appeared years
@app.get("/v1/menu/dish_names_by_appearance_years", operation_id="get_dish_names_by_appearance_years", summary="Retrieves the names of dishes that first appeared and last appeared in the specified years. This operation allows you to filter dishes based on their appearance years, providing a targeted list of dish names.")
async def get_dish_names_by_appearance_years(first_appeared: int = Query(..., description="First appeared year"), last_appeared: int = Query(..., description="Last appeared year")):
    cursor.execute("SELECT name FROM Dish WHERE first_appeared = ? AND last_appeared = ?", (first_appeared, last_appeared))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get menu names ordered by dish count with a limit
@app.get("/v1/menu/menu_names_by_dish_count", operation_id="get_menu_names_by_dish_count", summary="Retrieves a list of menu names, ordered by the number of dishes each menu contains in descending order. The operation returns a maximum of the specified limit of menu names.")
async def get_menu_names_by_dish_count(limit: int = Query(..., description="Limit of menu names to return")):
    cursor.execute("SELECT name FROM Menu GROUP BY name ORDER BY dish_count DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the count of menu items based on position coordinates
@app.get("/v1/menu/menu_item_count_by_position", operation_id="get_menu_item_count_by_position", summary="Retrieves the total number of menu items located within the specified position coordinates. The position is defined by an X coordinate and a Y coordinate, which determine the range for the count.")
async def get_menu_item_count_by_position(xpos: float = Query(..., description="X position coordinate"), ypos: float = Query(..., description="Y position coordinate")):
    cursor.execute("SELECT COUNT(*) FROM MenuItem AS T1 INNER JOIN Dish AS T2 ON T1.dish_id = T2.id WHERE T1.xpos > ? AND T1.ypos < ?", (xpos, ypos))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the duration a dish appeared and its last update time
@app.get("/v1/menu/dish_appearance_duration", operation_id="get_dish_appearance_duration", summary="Get the duration a dish appeared and its last update time")
async def get_dish_appearance_duration(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T1.last_appeared - T1.first_appeared, T2.updated_at FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.name = ?", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"duration": [], "updated_at": []}
    return {"duration": result[0][0], "updated_at": result[0][1]}

# Endpoint to get dish names where the difference between last_appeared and first_appeared is greater than a specified value
@app.get("/v1/menu/dish_names_by_appearance_difference", operation_id="get_dish_names_by_appearance_difference", summary="Retrieves the names of dishes that have been featured on the menu for a duration longer than the specified difference between their last and first appearance. This operation is useful for identifying dishes that have been consistently popular over an extended period.")
async def get_dish_names_by_appearance_difference(difference: int = Query(..., description="Difference between last_appeared and first_appeared")):
    cursor.execute("SELECT T1.name FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.last_appeared - T1.first_appeared > ?", (difference,))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": [row[0] for row in result]}

# Endpoint to get the count of dishes where the difference between last_appeared and first_appeared is less than a specified value
@app.get("/v1/menu/dish_count_by_appearance_difference", operation_id="get_dish_count_by_appearance_difference", summary="Retrieves the number of dishes that have appeared within a specified time frame. The time frame is determined by the difference between the first and last appearance of a dish on the menu.")
async def get_dish_count_by_appearance_difference(difference: int = Query(..., description="Difference between last_appeared and first_appeared")):
    cursor.execute("SELECT COUNT(*) FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.last_appeared - T1.first_appeared < ?", (difference,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get dish names and prices where the lowest price is a specified value
@app.get("/v1/menu/dish_names_prices_by_lowest_price", operation_id="get_dish_names_prices_by_lowest_price", summary="Retrieves the names and prices of dishes that have a specified lowest price. The input parameter 'lowest_price' is used to filter the results.")
async def get_dish_names_prices_by_lowest_price(lowest_price: int = Query(..., description="Lowest price of the dish")):
    cursor.execute("SELECT T2.name, T1.price FROM MenuItem AS T1 INNER JOIN Dish AS T2 ON T2.id = T1.dish_id WHERE T2.lowest_price = ?", (lowest_price,))
    result = cursor.fetchall()
    if not result:
        return {"dish_names_prices": []}
    return {"dish_names_prices": [{"name": row[0], "price": row[1]} for row in result]}

# Endpoint to get menu item prices based on menu ID and page number
@app.get("/v1/menu/menu_item_prices_by_menu_id_page_number", operation_id="get_menu_item_prices_by_menu_id_page_number", summary="Retrieves the prices of menu items associated with a specific menu and page number. The menu is identified by its unique ID, and the page number specifies the desired page within the menu. This operation returns a list of prices for the menu items on the selected page.")
async def get_menu_item_prices_by_menu_id_page_number(menu_id: int = Query(..., description="Menu ID"), page_number: int = Query(..., description="Page number")):
    cursor.execute("SELECT T1.price FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T2.id = T1.menu_page_id WHERE T2.menu_id = ? AND T2.page_number = ?", (menu_id, page_number))
    result = cursor.fetchall()
    if not result:
        return {"prices": []}
    return {"prices": [row[0] for row in result]}

# Endpoint to get the count of dishes created within a specified date range
@app.get("/v1/menu/dish_count_by_date_range", operation_id="get_dish_count_by_date_range", summary="Retrieve the total number of dishes added to the menu within a specified date range. The date range is defined by the provided start and end dates, both in 'YYYY-MM-DD HH:MM:SS UTC' format. This operation calculates the sum of dishes that were created between these dates, providing an overview of menu growth during the specified period.")
async def get_dish_count_by_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD HH:MM:SS UTC' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD HH:MM:SS UTC' format")):
    cursor.execute("SELECT SUM(CASE WHEN T2.created_at BETWEEN ? AND ? THEN 1 ELSE 0 END) FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get menu sponsors based on full height
@app.get("/v1/menu/menu_sponsors_by_full_height", operation_id="get_menu_sponsors_by_full_height", summary="Retrieves a list of sponsors associated with menu pages that have a specified full height. The full height is a parameter that determines the vertical size of the menu page. This operation returns the sponsor names that meet the given full height criteria.")
async def get_menu_sponsors_by_full_height(full_height: int = Query(..., description="Full height of the menu page")):
    cursor.execute("SELECT T2.sponsor FROM MenuPage AS T1 INNER JOIN Menu AS T2 ON T2.id = T1.menu_id WHERE T1.full_height = ?", (full_height,))
    result = cursor.fetchall()
    if not result:
        return {"sponsors": []}
    return {"sponsors": [row[0] for row in result]}

# Endpoint to get menu page details based on event
@app.get("/v1/menu/menu_page_details_by_event", operation_id="get_menu_page_details_by_event", summary="Retrieves the image details (image ID, full height, and full width) for a specific menu page associated with the provided event. The event name is used to identify the relevant menu and its corresponding page details.")
async def get_menu_page_details_by_event(event: str = Query(..., description="Event name")):
    cursor.execute("SELECT T1.image_id, T1.full_height, T1.full_width FROM MenuPage AS T1 INNER JOIN Menu AS T2 ON T2.id = T1.menu_id WHERE T2.event = ?", (event,))
    result = cursor.fetchall()
    if not result:
        return {"menu_page_details": []}
    return {"menu_page_details": [{"image_id": row[0], "full_height": row[1], "full_width": row[2]} for row in result]}

# Endpoint to get menu events based on full width
@app.get("/v1/menu/menu_events_by_full_width", operation_id="get_menu_events_by_full_width", summary="Retrieves all menu events associated with a menu page of a specified full width. The full width parameter is used to filter the menu events.")
async def get_menu_events_by_full_width(full_width: int = Query(..., description="Full width of the menu page")):
    cursor.execute("SELECT T1.event FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T2.full_width = ?", (full_width,))
    result = cursor.fetchall()
    if not result:
        return {"events": []}
    return {"events": [row[0] for row in result]}

# Endpoint to get dish names based on price
@app.get("/v1/menu/dish_names_by_price", operation_id="get_dish_names_by_price", summary="Retrieves the names of dishes that match the specified price. The operation filters menu items based on the provided price and returns the corresponding dish names.")
async def get_dish_names_by_price(price: int = Query(..., description="Price of the dish")):
    cursor.execute("SELECT T1.name FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T2.price = ?", (price,))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": [row[0] for row in result]}

# Endpoint to get menu item positions and appearance difference based on dish name
@app.get("/v1/menu/menu_item_positions_appearance_difference_by_dish_name", operation_id="get_menu_item_positions_appearance_difference_by_dish_name", summary="Retrieves the x and y coordinates of a menu item and the difference between its last and first appearance dates, based on the provided dish name.")
async def get_menu_item_positions_appearance_difference_by_dish_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T2.xpos, T2.ypos, T1.last_appeared - T1.first_appeared FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.name = ?", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"menu_item_positions_appearance_difference": []}
    return {"menu_item_positions_appearance_difference": [{"xpos": row[0], "ypos": row[1], "appearance_difference": row[2]} for row in result]}

# Endpoint to get the full height and width, page number, and image ID of a menu page for a specific dish name
@app.get("/v1/menu/menu_page_details_by_dish_name", operation_id="get_menu_page_details_by_dish_name", summary="Retrieve the dimensions (height and width), page number, and image ID of a menu page associated with a specific dish. The dish is identified by its name.")
async def get_menu_page_details_by_dish_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T1.full_height * T1.full_width, T1.page_number, T1.image_id FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id INNER JOIN Dish AS T3 ON T2.dish_id = T3.id WHERE T3.name = ?", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the count of menus for a specific date pattern and sponsor
@app.get("/v1/menu/count_menus_by_date_pattern_and_sponsor", operation_id="get_count_menus_by_date_pattern_and_sponsor", summary="Retrieves the total number of menus that match a given date pattern and sponsor. The date pattern should be provided in 'YYYY-MM%' format, and the sponsor name must be specified. This operation is useful for obtaining a quick overview of menu availability based on date and sponsor.")
async def get_count_menus_by_date_pattern_and_sponsor(date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM%' format"), sponsor: str = Query(..., description="Sponsor name")):
    cursor.execute("SELECT COUNT(*) FROM Menu WHERE date LIKE ? AND sponsor = ?", (date_pattern, sponsor))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the dish name with the longest duration between first and last appearance
@app.get("/v1/menu/dish_with_longest_appearance_duration", operation_id="get_dish_with_longest_appearance_duration", summary="Retrieves the name of the dish that has the longest duration between its first and last appearance in the menu. The duration is calculated as the difference between the last and first appearance dates.")
async def get_dish_with_longest_appearance_duration():
    cursor.execute("SELECT name FROM Dish ORDER BY last_appeared - Dish.first_appeared DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"dish_name": []}
    return {"dish_name": result[0]}

# Endpoint to get the event name for a specific date and menu ID
@app.get("/v1/menu/event_by_date_and_menu_id", operation_id="get_event_by_date_and_menu_id", summary="Retrieves the name of the event associated with a specific date and menu ID. The date should be provided in 'YYYY-MM-DD' format, and the menu ID should correspond to a valid menu in the system. This operation allows users to quickly look up event details based on date and menu selection.")
async def get_event_by_date_and_menu_id(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), menu_id: int = Query(..., description="Menu ID")):
    cursor.execute("SELECT event FROM Menu WHERE date = ? AND id = ?", (date, menu_id))
    result = cursor.fetchone()
    if not result:
        return {"event": []}
    return {"event": result[0]}

# Endpoint to get the count of menus with a specific name
@app.get("/v1/menu/count_menus_by_name", operation_id="get_count_menus_by_name", summary="Retrieves the total number of menus that match a given name. This operation considers the relationship between menus and menu pages to provide an accurate count.")
async def get_count_menus_by_name(menu_name: str = Query(..., description="Name of the menu")):
    cursor.execute("SELECT SUM(CASE WHEN T1.name = ? THEN 1 ELSE 0 END) FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id", (menu_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of dishes with a specific name
@app.get("/v1/menu/count_dishes_by_name", operation_id="get_count_dishes_by_name", summary="Retrieves the total number of menu items associated with a specific dish name. The dish name is provided as an input parameter.")
async def get_count_dishes_by_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT SUM(CASE WHEN T1.name = ? THEN 1 ELSE 0 END) FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id", (dish_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of dishes created in a specific year and month with no highest price
@app.get("/v1/menu/dishes_by_year_month_no_highest_price", operation_id="get_dishes_by_year_month_no_highest_price", summary="Retrieves the names of dishes that were added to the menu in a specific year and month, excluding those with a highest price. The year and month are provided in 'YYYY' and 'M' format respectively.")
async def get_dishes_by_year_month_no_highest_price(year: str = Query(..., description="Year in 'YYYY' format"), month: str = Query(..., description="Month in 'M' format")):
    cursor.execute("SELECT T1.name FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE SUBSTR(T2.created_at, 1, 4) = ? AND SUBSTR(T2.created_at, 7, 1) = ? AND T1.highest_price IS NULL", (year, month))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": [row[0] for row in result]}

# Endpoint to get the menu name with the highest number of pages
@app.get("/v1/menu/menu_with_most_pages", operation_id="get_menu_with_most_pages", summary="Retrieves the name of the menu that has the most pages. This operation identifies the menu with the highest number of associated pages and returns its name. The result provides a quick overview of the menu with the most extensive content.")
async def get_menu_with_most_pages():
    cursor.execute("SELECT T1.name FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id GROUP BY T2.menu_id ORDER BY COUNT(T2.page_number) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"menu_name": []}
    return {"menu_name": result[0]}

# Endpoint to get the menu page IDs for a specific dish name
@app.get("/v1/menu/menu_page_ids_by_dish_name", operation_id="get_menu_page_ids_by_dish_name", summary="Retrieves the IDs of menu pages that contain a specific dish. The operation filters menu items by the provided dish name and returns the corresponding menu page IDs.")
async def get_menu_page_ids_by_dish_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T2.menu_page_id FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.name = ?", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"menu_page_ids": []}
    return {"menu_page_ids": [row[0] for row in result]}

# Endpoint to get the count of menus with a specific sponsor and page number count
@app.get("/v1/menu/count_menus_sponsor_page_count", operation_id="get_count_menus_sponsor_page_count", summary="Retrieves the count of menus associated with a specific sponsor that have a page count less than or equal to a given maximum. This operation is useful for determining the number of menus that meet certain criteria, such as having a limited number of pages.")
async def get_count_menus_sponsor_page_count(sponsor: str = Query(..., description="Sponsor of the menu"), page_count: int = Query(..., description="Maximum number of pages")):
    cursor.execute("SELECT COUNT(*) FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T1.sponsor = ? GROUP BY T2.menu_id HAVING COUNT(T2.page_number) <= ?", (sponsor, page_count))
    result = cursor.fetchall()
    if not result:
        return {"count": []}
    return {"count": [row[0] for row in result]}

# Endpoint to get the menu page ID of the most expensive dish with a specific name
@app.get("/v1/menu/most_expensive_dish_menu_page", operation_id="get_most_expensive_dish_menu_page", summary="Retrieves the menu page ID associated with the most expensive dish that matches the provided dish name. The operation compares the input name against the names of dishes in the database, sorts the results by price in descending order, and returns the menu page ID of the top-ranked dish.")
async def get_most_expensive_dish_menu_page(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T1.menu_page_id FROM MenuItem AS T1 INNER JOIN Dish AS T2 ON T2.id = T1.dish_id WHERE T2.name = ? ORDER BY T1.price DESC LIMIT 1", (dish_name,))
    result = cursor.fetchone()
    if not result:
        return {"menu_page_id": []}
    return {"menu_page_id": result[0]}

# Endpoint to get the menu ID with the most pages for a specific sponsor
@app.get("/v1/menu/most_pages_menu_id_sponsor", operation_id="get_most_pages_menu_id_sponsor", summary="Retrieves the unique identifier of the menu with the highest number of pages associated with a specific sponsor. The sponsor is identified by the provided input parameter.")
async def get_most_pages_menu_id_sponsor(sponsor: str = Query(..., description="Sponsor of the menu")):
    cursor.execute("SELECT T2.menu_id FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T1.sponsor = ? GROUP BY T2.menu_id ORDER BY COUNT(T2.page_number) DESC LIMIT 1", (sponsor,))
    result = cursor.fetchone()
    if not result:
        return {"menu_id": []}
    return {"menu_id": result[0]}

# Endpoint to get the position of a specific dish on the menu
@app.get("/v1/menu/dish_position", operation_id="get_dish_position", summary="Retrieves the x and y coordinates of a specific dish on the menu. The dish is identified by its name, which is provided as an input parameter.")
async def get_dish_position(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T2.xpos, T2.ypos FROM Dish AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.dish_id WHERE T1.name = ?", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"positions": []}
    return {"positions": [{"xpos": row[0], "ypos": row[1]} for row in result]}

# Endpoint to get the names of dishes for a specific event and place sponsored by a specific sponsor
@app.get("/v1/menu/dish_names_event_place_sponsor", operation_id="get_dish_names_event_place_sponsor", summary="Retrieves the names of dishes associated with a specific event and place, sponsored by a particular sponsor. The operation filters the menu items based on the provided sponsor, event, and place, and returns the corresponding dish names.")
async def get_dish_names_event_place_sponsor(sponsor: str = Query(..., description="Sponsor of the menu"), event: str = Query(..., description="Event name"), place: str = Query(..., description="Place of the event")):
    cursor.execute("SELECT T4.name FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id INNER JOIN MenuItem AS T3 ON T2.id = T3.menu_page_id INNER JOIN Dish AS T4 ON T3.dish_id = T4.id WHERE T1.sponsor = ? AND T1.event = ? AND T1.place = ?", (sponsor, event, place))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": [row[0] for row in result]}

# Endpoint to get the sponsor of the most expensive menu item with a specific dish name and ID
@app.get("/v1/menu/sponsor_most_expensive_dish", operation_id="get_sponsor_most_expensive_dish", summary="Retrieves the sponsor of the most expensive menu item that matches a specific dish name and ID. The operation filters menu items by the provided dish name and ID, sorts them by price in descending order, and returns the sponsor of the top-priced item.")
async def get_sponsor_most_expensive_dish(dish_name: str = Query(..., description="Name of the dish"), dish_id: int = Query(..., description="ID of the dish")):
    cursor.execute("SELECT T4.sponsor FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id INNER JOIN Dish AS T3 ON T2.dish_id = T3.id INNER JOIN Menu AS T4 ON T4.id = T1.menu_id WHERE T3.name = ? AND T3.id = ? ORDER BY T2.price DESC LIMIT 1", (dish_name, dish_id))
    result = cursor.fetchone()
    if not result:
        return {"sponsor": []}
    return {"sponsor": result[0]}

# Endpoint to get the count and dish ID of the most frequent dish in a specific menu
@app.get("/v1/menu/most_frequent_dish_count", operation_id="get_most_frequent_dish_count", summary="Retrieves the count and ID of the most frequently occurring dish in a specified menu. The operation calculates the frequency of each dish in the menu and returns the dish with the highest count. The menu is identified by its name.")
async def get_most_frequent_dish_count(menu_name: str = Query(..., description="Name of the menu")):
    cursor.execute("SELECT COUNT(*), T1.dish_id FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id INNER JOIN Dish AS T4 ON T1.dish_id = T4.id WHERE T3.name = ? GROUP BY T3.id ORDER BY COUNT(T1.dish_id) DESC LIMIT 1", (menu_name,))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "dish_id": []}
    return {"count": result[0], "dish_id": result[1]}

# Endpoint to get the IDs of menus without a sponsor
@app.get("/v1/menu/menus_without_sponsor", operation_id="get_menus_without_sponsor", summary="Retrieves the unique identifiers of all menus that do not have a sponsor associated with them.")
async def get_menus_without_sponsor():
    cursor.execute("SELECT id FROM Menu WHERE sponsor IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"menu_ids": []}
    return {"menu_ids": [row[0] for row in result]}

# Endpoint to get the count of menus for a specific event
@app.get("/v1/menu/count_menus_event", operation_id="get_count_menus_event", summary="Retrieves the total number of menus associated with a particular event. The event is specified as an input parameter.")
async def get_count_menus_event(event: str = Query(..., description="Event name")):
    cursor.execute("SELECT COUNT(*) FROM Menu WHERE event = ?", (event,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of menus based on page count and dish count
@app.get("/v1/menu/count_menus_by_page_and_dish_count", operation_id="get_menu_count", summary="Retrieves the total number of menus that have a page count exceeding the provided minimum value and a dish count surpassing the specified minimum value. This operation is useful for understanding the distribution of menus based on their page and dish counts.")
async def get_menu_count(page_count: int = Query(..., description="Minimum page count"), dish_count: int = Query(..., description="Minimum dish count")):
    cursor.execute("SELECT COUNT(*) FROM Menu WHERE page_count > ? AND dish_count > ?", (page_count, dish_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the menu ID with the highest dish count
@app.get("/v1/menu/get_menu_with_highest_dish_count", operation_id="get_menu_with_highest_dish_count", summary="Retrieves the unique identifier of the menu that contains the most dishes. This operation does not require any input parameters and returns the menu ID as a single value.")
async def get_menu_with_highest_dish_count():
    cursor.execute("SELECT id FROM Menu ORDER BY dish_count DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"menu_id": []}
    return {"menu_id": result[0]}

# Endpoint to get the count of menus with no call number and date before a specific year
@app.get("/v1/menu/count_menus_no_call_number_before_year", operation_id="get_menu_count_no_call_number_before_year", summary="Retrieves the total number of menus that do not have a call number and were created before a specified year. The year is provided in the 'YYYY' format.")
async def get_menu_count_no_call_number_before_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM Menu WHERE call_number IS NULL AND strftime('%Y', date) < ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the image ID of a menu page based on menu name and page number
@app.get("/v1/menu/get_image_id_by_menu_name_and_page_number", operation_id="get_image_id_by_menu_name_and_page_number", summary="Retrieves the unique identifier of the image associated with a specific menu page, based on the provided menu name and page number. This operation allows you to locate the image ID of a particular menu page, enabling further image-related actions or data retrieval.")
async def get_image_id_by_menu_name_and_page_number(name: str = Query(..., description="Name of the menu"), page_number: int = Query(..., description="Page number")):
    cursor.execute("SELECT T2.image_id FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T1.name = ? AND T2.page_number = ?", (name, page_number))
    result = cursor.fetchone()
    if not result:
        return {"image_id": []}
    return {"image_id": result[0]}

# Endpoint to get the menu name based on image ID
@app.get("/v1/menu/get_menu_name_by_image_id", operation_id="get_menu_name_by_image_id", summary="Retrieves the name of the menu associated with the provided image ID. This operation fetches the menu name from the Menu table by joining it with the MenuPage table using the menu_id and image_id fields.")
async def get_menu_name_by_image_id(image_id: int = Query(..., description="Image ID")):
    cursor.execute("SELECT T1.name FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T2.image_id = ?", (image_id,))
    result = cursor.fetchone()
    if not result:
        return {"menu_name": []}
    return {"menu_name": result[0]}

# Endpoint to compare the full width of two menus
@app.get("/v1/menu/compare_menu_full_width", operation_id="compare_menu_full_width", summary="This operation compares the total full width of two menus and returns the name of the menu with the greater full width. The comparison is based on the sum of the full widths of all pages in each menu. The input parameters specify the names of the two menus to be compared.")
async def compare_menu_full_width(name1: str = Query(..., description="Name of the first menu"), name2: str = Query(..., description="Name of the second menu")):
    cursor.execute("SELECT CASE WHEN SUM(CASE WHEN T1.name = ? THEN T2.full_width ELSE 0 END) - SUM(CASE WHEN T1.name = ? THEN T2.full_width ELSE 0 END) > 0 THEN ? ELSE ? END FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id", (name1, name2, name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"menu_name": []}
    return {"menu_name": result[0]}

# Endpoint to get the page number of the highest full height page in a menu
@app.get("/v1/menu/get_highest_full_height_page_number", operation_id="get_highest_full_height_page_number", summary="Retrieves the page number of the tallest full-height page within a specified menu. The menu is identified by its name, and the page number corresponds to the page with the maximum full height.")
async def get_highest_full_height_page_number(name: str = Query(..., description="Name of the menu")):
    cursor.execute("SELECT T1.page_number FROM MenuPage AS T1 INNER JOIN Menu AS T2 ON T2.id = T1.menu_id WHERE T2.name = ? ORDER BY T1.full_height DESC LIMIT 1", (name,))
    result = cursor.fetchone()
    if not result:
        return {"page_number": []}
    return {"page_number": result[0]}

# Endpoint to get the count of menu pages with full width greater than a specified value
@app.get("/v1/menu/count_menu_pages_by_full_width", operation_id="count_menu_pages_by_full_width", summary="Retrieves the total number of menu pages that have a full width greater than the specified value, within a particular menu.")
async def count_menu_pages_by_full_width(name: str = Query(..., description="Name of the menu"), full_width: int = Query(..., description="Full width")):
    cursor.execute("SELECT SUM(CASE WHEN T1.name = ? THEN 1 ELSE 0 END) FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T2.full_width > ?", (name, full_width))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of menu items on a specific page of a menu
@app.get("/v1/menu/count_menu_items_on_page", operation_id="count_menu_items_on_page", summary="Retrieves the total number of menu items available on a specific page of a given menu. The operation requires the page number and the menu ID as input parameters to accurately determine the count.")
async def count_menu_items_on_page(page_number: int = Query(..., description="Page number"), menu_id: int = Query(..., description="Menu ID")):
    cursor.execute("SELECT SUM(CASE WHEN T1.page_number = ? THEN 1 ELSE 0 END) FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id WHERE T1.menu_id = ?", (page_number, menu_id))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get dish names from a specific menu page
@app.get("/v1/menu/dish_names_by_menu_page", operation_id="get_dish_names_by_menu_page", summary="Retrieve the names of dishes from a specific page of a menu, identified by its unique menu ID and page number. This operation allows you to access the names of dishes available on a particular menu page, providing a concise overview of the menu's offerings.")
async def get_dish_names_by_menu_page(menu_id: int = Query(..., description="ID of the menu"), page_number: int = Query(..., description="Page number of the menu")):
    cursor.execute("SELECT T3.name FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id INNER JOIN Dish AS T3 ON T2.dish_id = T3.id WHERE T1.menu_id = ? AND T1.page_number = ?", (menu_id, page_number))
    result = cursor.fetchall()
    if not result:
        return {"dish_names": []}
    return {"dish_names": [row[0] for row in result]}

# Endpoint to get page numbers where a specific dish is listed
@app.get("/v1/menu/page_numbers_by_dish_name", operation_id="get_page_numbers_by_dish_name", summary="Retrieve the page numbers on which a specific dish is listed in the menu. This operation requires the name of the dish as input and returns a list of page numbers where the dish is featured.")
async def get_page_numbers_by_dish_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T1.page_number FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id INNER JOIN Dish AS T3 ON T2.dish_id = T3.id WHERE T3.name = ?", (dish_name,))
    result = cursor.fetchall()
    if not result:
        return {"page_numbers": []}
    return {"page_numbers": [row[0] for row in result]}

# Endpoint to get the menu page ID with the widest full width for a specific dish
@app.get("/v1/menu/widest_menu_page_by_dish_name", operation_id="get_widest_menu_page_by_dish_name", summary="Retrieves the ID of the menu page that has the widest full width for a specific dish. The dish is identified by its name, which is provided as an input parameter. The menu page with the maximum full width is determined by ordering the menu pages in descending order of their full width and selecting the top result.")
async def get_widest_menu_page_by_dish_name(dish_name: str = Query(..., description="Name of the dish")):
    cursor.execute("SELECT T1.id FROM MenuPage AS T1 INNER JOIN MenuItem AS T2 ON T1.id = T2.menu_page_id INNER JOIN Dish AS T3 ON T2.dish_id = T3.id WHERE T3.name = ? ORDER BY T1.full_width DESC LIMIT 1", (dish_name,))
    result = cursor.fetchone()
    if not result:
        return {"menu_page_id": []}
    return {"menu_page_id": result[0]}

# Endpoint to get the area of a specific menu page
@app.get("/v1/menu/menu_page_area", operation_id="get_menu_page_area", summary="Retrieves the total area of a specific menu page, determined by the menu name and page number. The area is calculated as the product of the full height and full width of the menu page.")
async def get_menu_page_area(menu_name: str = Query(..., description="Name of the menu"), page_number: int = Query(..., description="Page number of the menu")):
    cursor.execute("SELECT T2.full_height * T2.full_width FROM Menu AS T1 INNER JOIN MenuPage AS T2 ON T1.id = T2.menu_id WHERE T1.name = ? AND T2.page_number = ?", (menu_name, page_number))
    result = cursor.fetchone()
    if not result:
        return {"area": []}
    return {"area": result[0]}

# Endpoint to get the average number of dishes per page in a specific menu
@app.get("/v1/menu/average_dishes_per_page", operation_id="get_average_dishes_per_page", summary="Retrieves the average number of dishes per page for a specific menu. This operation calculates the ratio of the total number of dishes to the total number of pages in the given menu, providing a useful metric for menu planning and organization.")
async def get_average_dishes_per_page(menu_id: int = Query(..., description="ID of the menu")):
    cursor.execute("SELECT CAST(COUNT(dish_id) AS REAL) / COUNT(T3.page_count) FROM MenuItem AS T1 INNER JOIN MenuPage AS T2 ON T1.menu_page_id = T2.id INNER JOIN Menu AS T3 ON T2.menu_id = T3.id WHERE T2.menu_id = ?", (menu_id,))
    result = cursor.fetchone()
    if not result:
        return {"average_dishes_per_page": []}
    return {"average_dishes_per_page": result[0]}

api_calls = [
    "/v1/menu/dish_count_by_year_range?min_year=1851&max_year=2012",
    "/v1/menu/compare_dish_duration?dish1=Anchovies&dish2=Fresh%20lobsters%20in%20every%20style",
    "/v1/menu/dish_with_lowest_price?lowest_price=0",
    "/v1/menu/menu_count_by_name_and_page_count?name=Waldorf%20Astoria&page_count=4",
    "/v1/menu/dish_names_by_menu_page_and_position?menu_page_id=1389&xpos=0.25&ypos=0.25",
    "/v1/menu/dish_prices_by_name?dish_name=Clear%20green%20turtle",
    "/v1/menu/count_dish_no_highest_price?dish_name=Clear%20green%20turtle",
    "/v1/menu/highest_price_of_dish?dish_name=Clear%20green%20turtle",
    "/v1/menu/menu_ids_by_dish_name?dish_name=Clear%20green%20turtle",
    "/v1/menu/count_menus_by_currency_and_dish?currency=Dollars&dish_name=Clear%20green%20turtle",
    "/v1/menu/count_dish_without_call_number?dish_name=Clear%20green%20turtle",
    "/v1/menu/dish_names_from_menu?menu_name=Zentral%20Theater%20Terrace",
    "/v1/menu/most_expensive_dish_from_menu?menu_name=Zentral%20Theater%20Terrace",
    "/v1/menu/count_specific_menu?menu_name=Zentral%20Theater%20Terrace",
    "/v1/menu/menu_ids_for_dish_without_sponsor?dish_name=Clear%20green%20turtle",
    "/v1/menu/average_page_number_for_dish?dish_name=Clear%20green%20turtle",
    "/v1/menu/average_price_in_menu?menu_name=Zentral%20Theater%20Terrace",
    "/v1/menu/count_menu_items_created_on_date?date=2011-03-28",
    "/v1/menu/count_menu_items_for_menu_page?menu_page_id=144",
    "/v1/menu/count_menus_by_location?location=Dutcher%20House",
    "/v1/menu/count_dishes_times_appeared_greater_than_menus_appeared",
    "/v1/menu/count_menus_by_venue?venue=STEAMSHIP",
    "/v1/menu/sum_menus_by_date?date=1898-11-17",
    "/v1/menu/dish_names_by_menu_page_id?menu_page_id=174",
    "/v1/menu/dish_names_ids_by_first_appeared?first_appeared=1861",
    "/v1/menu/most_expensive_dish_by_date?created_at=2011-05-23%",
    "/v1/menu/dish_name_by_page_number?page_number=30",
    "/v1/menu/page_number_menu_name_highest_page_count",
    "/v1/menu/count_dishes_by_page_number?page_number=2",
    "/v1/menu/menu_details_by_dish_name?dish_name=Fresh%20lobsters%20in%20every%20style",
    "/v1/menu/dish_names_by_sponsor_and_position?sponsor=CHAS.BRADLEY%27S%20OYSTER%20%26%20DINING%20ROOM&xpos=0.25&ypos=0.25",
    "/v1/menu/menu_name_event_by_dish_name?dish_name=Cerealine%20with%20Milk",
    "/v1/menu/percentage_menu_items_by_position_and_dish_name?xpos_min=0.25&xpos_max=0.75&ypos_min=0.25&ypos_max=0.75&dish_name=%25BLuefish%25",
    "/v1/menu/sponsor_by_menu_id?menu_id=12463",
    "/v1/menu/event_by_sponsor?sponsor=REPUBLICAN%20HOUSE",
    "/v1/menu/location_by_menu_id?menu_id=12472",
    "/v1/menu/physical_description_by_sponsor?sponsor=Noviomagus",
    "/v1/menu/occasion_by_menu_id?menu_id=12463",
    "/v1/menu/location_by_sponsor?sponsor=Norddeutscher%20Lloyd%20Bremen",
    "/v1/menu/menu_ids_by_sponsor?sponsor=Occidental%20%26%20Oriental",
    "/v1/menu/image_ids_by_location?location=Manhattan%20Hotel",
    "/v1/menu/dimensions_by_menu_name?name=El%20Fuerte%20Del%20Palmar",
    "/v1/menu/check_uuid_by_menu_name?name=The%20Biltmore&uuid=c02c9a3b-6881-7080-e040-e00a180631aa",
    "/v1/menu/highest_full_height_menu",
    "/v1/menu/page_count_by_menu_page_id?menu_page_id=130",
    "/v1/menu/prices_by_image_id?image_id=4000009194",
    "/v1/menu/page_numbers_by_position?xpos=0.75&ypos=0.25",
    "/v1/menu/dish_names_by_month?month=4",
    "/v1/menu/highest_price_in_range?min_id=1&max_id=5",
    "/v1/menu/dish_names_by_position?xpos=0.25&ypos=0.25",
    "/v1/menu/dish_count_difference?uuid1=510d47e4-2958-a3d9-e040-e00a18064a99&uuid2=510d47e4-295a-a3d9-e040-e00a18064a99",
    "/v1/menu/total_price_by_menu_id?menu_id=12882",
    "/v1/menu/dish_names_by_highest_price?limit=5",
    "/v1/menu/dish_count_by_lowest_price?lowest_price=0",
    "/v1/menu/dish_names_by_appearance_years?first_appeared=1855&last_appeared=1900",
    "/v1/menu/menu_names_by_dish_count?limit=10",
    "/v1/menu/menu_item_count_by_position?xpos=0.75&ypos=0.25",
    "/v1/menu/dish_appearance_duration?dish_name=Clear%20green%20turtle",
    "/v1/menu/dish_names_by_appearance_difference?difference=100",
    "/v1/menu/dish_count_by_appearance_difference?difference=5",
    "/v1/menu/dish_names_prices_by_lowest_price?lowest_price=0",
    "/v1/menu/menu_item_prices_by_menu_id_page_number?menu_id=12474&page_number=2",
    "/v1/menu/dish_count_by_date_range?start_date=2011-03-31%2020:24:46%20UTC&end_date=2011-04-15%2023:09:51%20UTC",
    "/v1/menu/menu_sponsors_by_full_height?full_height=10000",
    "/v1/menu/menu_page_details_by_event?event=100TH%20ANNIVERSARY%20OF%20BIRTH%20OF%20DANIEL%20WEBSTER",
    "/v1/menu/menu_events_by_full_width?full_width=2000",
    "/v1/menu/dish_names_by_price?price=180000",
    "/v1/menu/menu_item_positions_appearance_difference_by_dish_name?dish_name=Small%20Hominy",
    "/v1/menu/menu_page_details_by_dish_name?dish_name=Baked%20Stuffed%20Mullet%20%26%20Sauce%20Pomard",
    "/v1/menu/count_menus_by_date_pattern_and_sponsor?date_pattern=2015-04%25&sponsor=Krogs%20Fiskerestaurant",
    "/v1/menu/dish_with_longest_appearance_duration",
    "/v1/menu/event_by_date_and_menu_id?date=1887-07-21&menu_id=21380",
    "/v1/menu/count_menus_by_name?menu_name=Emil%20Kuehn",
    "/v1/menu/count_dishes_by_name?dish_name=Puree%20of%20split%20peas%20aux%20croutons",
    "/v1/menu/dishes_by_year_month_no_highest_price?year=2011&month=4",
    "/v1/menu/menu_with_most_pages",
    "/v1/menu/menu_page_ids_by_dish_name?dish_name=Mashed%20potatoes",
    "/v1/menu/count_menus_sponsor_page_count?sponsor=PACIFIC%20MAIL%20STEAMSHIP%20COMPANY&page_count=2",
    "/v1/menu/most_expensive_dish_menu_page?dish_name=Milk",
    "/v1/menu/most_pages_menu_id_sponsor?sponsor=OCCIDENTAL%20%26%20ORIENTAL%20STEAMSHIP%20COMPANY",
    "/v1/menu/dish_position?dish_name=breaded%20veal%20cutlet%20with%20peas",
    "/v1/menu/dish_names_event_place_sponsor?sponsor=THE%20SOCIETY%20OF%20THE%20CUMBERLAND&event=19NTH%20REUNION&place=GRAND%20PACIFIC%20HOTEL,CHICAGO,ILL",
    "/v1/menu/sponsor_most_expensive_dish?dish_name=Baked%20apples%20with%20cream&dish_id=107",
    "/v1/menu/most_frequent_dish_count?menu_name=Souper%20de%20Luxe",
    "/v1/menu/menus_without_sponsor",
    "/v1/menu/count_menus_event?event=LUNCH",
    "/v1/menu/count_menus_by_page_and_dish_count?page_count=10&dish_count=20",
    "/v1/menu/get_menu_with_highest_dish_count",
    "/v1/menu/count_menus_no_call_number_before_year?year=1950",
    "/v1/menu/get_image_id_by_menu_name_and_page_number?name=Zentral%20Theater%20Terrace&page_number=1",
    "/v1/menu/get_menu_name_by_image_id?image_id=5189412",
    "/v1/menu/compare_menu_full_width?name1=Zentral%20Theater%20Terrace&name2=Young%27s%20Hotel",
    "/v1/menu/get_highest_full_height_page_number?name=Ritz%20Carlton",
    "/v1/menu/count_menu_pages_by_full_width?name=Ritz%20Carlton&full_width=1000",
    "/v1/menu/count_menu_items_on_page?page_number=1&menu_id=12882",
    "/v1/menu/dish_names_by_menu_page?menu_id=12882&page_number=1",
    "/v1/menu/page_numbers_by_dish_name?dish_name=Chicken%20gumbo",
    "/v1/menu/widest_menu_page_by_dish_name?dish_name=Chicken%20gumbo",
    "/v1/menu/menu_page_area?menu_name=Zentral%20Theater%20Terrace&page_number=1",
    "/v1/menu/average_dishes_per_page?menu_id=12882"
]
