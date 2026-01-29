from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/coinmarketcap/coinmarketcap.sqlite')
cursor = conn.cursor()

# Endpoint to get the name of coins with the highest market cap in a specific year
@app.get("/v1/coinmarketcap/coin_name_highest_market_cap", operation_id="get_coin_name_highest_market_cap", summary="Retrieves the name of the coin with the highest market capitalization in a specified year. The year is provided in the 'YYYY' format. This operation returns the coin name by querying the coins and historical tables, filtering for the specified year and the maximum market capitalization.")
async def get_coin_name_highest_market_cap(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date LIKE ? AND T2.market_cap = ( SELECT MAX(market_cap) FROM historical WHERE STRFTIME('%Y', date) = ? )", (year + '%', year))
    result = cursor.fetchall()
    if not result:
        return {"coin_names": []}
    return {"coin_names": [row[0] for row in result]}

# Endpoint to get the 24-hour volume of a specific coin on a specific date
@app.get("/v1/coinmarketcap/coin_volume_24h", operation_id="get_coin_volume_24h", summary="Retrieves the 24-hour trading volume of a specified coin on a given date. The operation requires the coin's name and the date in 'YYYY-MM-DD' format as input parameters. The returned volume reflects the total trading activity for the coin within the 24-hour period preceding the provided date.")
async def get_coin_volume_24h(coin_name: str = Query(..., description="Name of the coin"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.volume_24h FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND T2.date = ?", (coin_name, date))
    result = cursor.fetchone()
    if not result:
        return {"volume_24h": []}
    return {"volume_24h": result[0]}

# Endpoint to get the price and average price of a specific coin within a date range
@app.get("/v1/coinmarketcap/coin_price_and_avg_price", operation_id="get_coin_price_and_avg_price", summary="Retrieve the current price and average price of a specified cryptocurrency within a given date range. The operation fetches the price data from the historical records of the coin, and calculates the average price for the provided date range. The input parameters include the name of the cryptocurrency, and the start and end dates of the date range in 'YYYY-MM-DD' format.")
async def get_coin_price_and_avg_price(coin_name: str = Query(..., description="Name of the coin"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.price FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND T2.date BETWEEN ? AND ? UNION ALL SELECT AVG(T2.price) FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND T2.date BETWEEN ? AND ?", (coin_name, start_date, end_date, coin_name, start_date, end_date))
    result = cursor.fetchall()
    if not result:
        return {"prices": []}
    return {"prices": [row[0] for row in result]}

# Endpoint to get the high and low times and date of a specific coin in a specific month and year
@app.get("/v1/coinmarketcap/coin_time_high_low", operation_id="get_coin_time_high_low", summary="Retrieves the highest and lowest prices, along with their respective dates, for a specified coin during a given month and year. The coin is identified by its name, and the time period is defined by a year and month in 'YYYY-MM' format.")
async def get_coin_time_high_low(coin_name: str = Query(..., description="Name of the coin"), year_month: str = Query(..., description="Year and month in 'YYYY-MM' format")):
    cursor.execute("SELECT T2.time_high, T2.time_low, T2.date FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND STRFTIME('%Y-%m', T2.date) = ?", (coin_name, year_month))
    result = cursor.fetchall()
    if not result:
        return {"time_high_low": []}
    return {"time_high_low": [{"time_high": row[0], "time_low": row[1], "date": row[2]} for row in result]}

# Endpoint to get the date of the highest price for a specific coin
@app.get("/v1/coinmarketcap/coin_highest_price_date", operation_id="get_coin_highest_price_date", summary="Retrieves the date when a specified coin reached its highest price. The operation filters the coin by its name and sorts the historical price data in descending order to identify the date of the highest price.")
async def get_coin_highest_price_date(coin_name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.date FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? ORDER BY T2.price DESC LIMIT 1", (coin_name,))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the coin with the highest 24-hour percent change
@app.get("/v1/coinmarketcap/coin_highest_percent_change_24h", operation_id="get_coin_highest_percent_change_24h", summary="Retrieves the coin that has experienced the highest percentage change in price over the past 24 hours. The response includes the coin's name, the date of the price data, and the corresponding price.")
async def get_coin_highest_percent_change_24h():
    cursor.execute("SELECT T1.name, T2.date, T2.price FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.percent_change_24h = ( SELECT MAX(percent_change_24h) FROM historical )")
    result = cursor.fetchall()
    if not result:
        return {"coins": []}
    return {"coins": [{"name": row[0], "date": row[1], "price": row[2]} for row in result]}

# Endpoint to get the average monthly circulating supply of a specific coin in a specific year
@app.get("/v1/coinmarketcap/coin_avg_monthly_circulating_supply", operation_id="get_coin_avg_monthly_circulating_supply", summary="Retrieves the average monthly circulating supply of a specific coin for a given year. The calculation is based on the historical data of the coin's circulating supply. The coin is identified by its name, and the year is specified in the 'YYYY' format.")
async def get_coin_avg_monthly_circulating_supply(coin_name: str = Query(..., description="Name of the coin"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(T2.circulating_supply) AS REAL) / 12 FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND STRFTIME('%Y', T2.date) = ?", (coin_name, year))
    result = cursor.fetchone()
    if not result:
        return {"avg_monthly_circulating_supply": []}
    return {"avg_monthly_circulating_supply": result[0]}

# Endpoint to get the latest date of an inactive coin
@app.get("/v1/coinmarketcap/latest_date_inactive_coin", operation_id="get_latest_date_inactive_coin", summary="Retrieves the most recent date when a coin with the specified status was marked as inactive. The status parameter is used to filter the coins by their current status.")
async def get_latest_date_inactive_coin(status: str = Query(..., description="Status of the coin")):
    cursor.execute("SELECT T1.name, MAX(T2.date) FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.status = ? ORDER BY T2.date DESC LIMIT 1", (status,))
    result = cursor.fetchone()
    if not result:
        return {"latest_date": []}
    return {"latest_date": {"name": result[0], "date": result[1]}}

# Endpoint to get the average price of a specific coin in a specific year
@app.get("/v1/coinmarketcap/coin_avg_price_year", operation_id="get_coin_avg_price_year", summary="Retrieves the average price of a specified coin for a given year. The calculation is based on historical price data. The coin is identified by its name, and the year is provided in 'YYYY' format.")
async def get_coin_avg_price_year(coin_name: str = Query(..., description="Name of the coin"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT AVG(T2.price) FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND STRFTIME('%Y', T2.date) = ?", (coin_name, year))
    result = cursor.fetchone()
    if not result:
        return {"avg_price": []}
    return {"avg_price": result[0]}

# Endpoint to get the date and price of the lowest price for a specific coin
@app.get("/v1/coinmarketcap/coin_lowest_price_date", operation_id="get_coin_lowest_price_date", summary="Retrieves the date and corresponding lowest price for a specified coin. The operation filters historical data for the coin and returns the record with the lowest price.")
async def get_coin_lowest_price_date(coin_name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.date, T2.price FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? ORDER BY T2.price LIMIT 1", (coin_name,))
    result = cursor.fetchone()
    if not result:
        return {"lowest_price_date": []}
    return {"lowest_price_date": {"date": result[0], "price": result[1]}}

# Endpoint to get the status of coins with an average price greater than a specified value
@app.get("/v1/coinmarketcap/coin_status_avg_price", operation_id="get_coin_status_avg_price", summary="Retrieve the status of coins whose average historical price surpasses a given threshold. The operation calculates the average historical price for each coin and returns the status of those coins that meet the specified price threshold.")
async def get_coin_status_avg_price(avg_price: float = Query(..., description="Average price threshold")):
    cursor.execute("SELECT T1.status FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id GROUP BY T1.name HAVING AVG(T2.price) > ?", (avg_price,))
    result = cursor.fetchall()
    if not result:
        return {"status": []}
    return {"status": [row[0] for row in result]}

# Endpoint to get the name and date of coins with the minimum 1-hour percent change
@app.get("/v1/coinmarketcap/coin_min_percent_change_1h", operation_id="get_coin_min_percent_change_1h", summary="Retrieves the names and corresponding dates of the coins that have experienced the smallest 1-hour percentage change in value. This operation identifies the coins with the least fluctuation in their value over the past hour and provides their names along with the specific dates when this minimal change occurred.")
async def get_coin_min_percent_change_1h():
    cursor.execute("SELECT T1.name, T2.date FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.percent_change_1h = ( SELECT MIN(percent_change_1h) FROM historical )")
    result = cursor.fetchall()
    if not result:
        return {"coins": []}
    return {"coins": [{"name": row[0], "date": row[1]} for row in result]}

# Endpoint to get the name of the coin with the highest price range in a specified category
@app.get("/v1/coinmarketcap/coin_highest_price_range", operation_id="get_coin_highest_price_range", summary="Retrieves the name of the coin that has the largest price range within a specified category. The price range is determined by the difference between the highest and lowest historical prices of the coin. The category of the coin is provided as an input parameter.")
async def get_coin_highest_price_range(category: str = Query(..., description="Category of the coin")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.category = ? ORDER BY T2.high - T2.low DESC LIMIT 1", (category,))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the names of coins with an average 24-hour percent change greater than the price on a specified date
@app.get("/v1/coinmarketcap/coin_avg_percent_change_24h", operation_id="get_coin_avg_percent_change_24h", summary="Retrieve the names of cryptocurrencies that have experienced an average 24-hour percent change greater than their price on a specified date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_coin_avg_percent_change_24h(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? GROUP BY T1.name HAVING AVG(T2.percent_change_24h) > T2.PRICE", (date,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the names of coins with a specific CMC rank on a specified date
@app.get("/v1/coinmarketcap/coin_cmc_rank", operation_id="get_coin_cmc_rank", summary="Retrieve the names of cryptocurrencies that held a specific CoinMarketCap (CMC) rank on a given date. This operation fetches the required data from the coins and historical tables, filtering results based on the provided date and CMC rank.")
async def get_coin_cmc_rank(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), cmc_rank: int = Query(..., description="CMC rank")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T2.cmc_rank = ?", (date, cmc_rank))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the market cap of a specific coin on a specified date
@app.get("/v1/coinmarketcap/coin_market_cap", operation_id="get_coin_market_cap", summary="Retrieves the market capitalization of a specific cryptocurrency on a given date. The operation requires the coin's name and the date in 'YYYY-MM-DD' format to accurately fetch the market cap from the historical data.")
async def get_coin_market_cap(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.market_cap FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T1.name = ?", (date, name))
    result = cursor.fetchone()
    if not result:
        return {"market_cap": []}
    return {"market_cap": result[0]}

# Endpoint to get the names of coins with a NULL open value on a specified date
@app.get("/v1/coinmarketcap/coin_null_open", operation_id="get_coin_null_open", summary="Retrieves the names of cryptocurrencies that have no recorded opening price on a specified date. The date must be provided in the 'YYYY-MM-DD' format.")
async def get_coin_null_open(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T2.open IS NULL", (date,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the closing price of a specific coin on a specified date
@app.get("/v1/coinmarketcap/coin_close_price", operation_id="get_coin_close_price", summary="Retrieves the closing price of a specific cryptocurrency on a given date. The operation requires the name of the coin and the date in 'YYYY-MM-DD' format. The closing price is fetched from historical data, which is associated with the coin's unique identifier.")
async def get_coin_close_price(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.close FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T1.name = ?", (date, name))
    result = cursor.fetchone()
    if not result:
        return {"close_price": []}
    return {"close_price": result[0]}

# Endpoint to get the time of the highest price of a specific coin on a specified date
@app.get("/v1/coinmarketcap/coin_time_high", operation_id="get_coin_time_high", summary="Retrieves the exact time when a specific coin reached its highest price on a given date. The operation requires the coin's name and the date in 'YYYY-MM-DD' format.")
async def get_coin_time_high(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.time_high FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T1.name = ?", (date, name))
    result = cursor.fetchone()
    if not result:
        return {"time_high": []}
    return {"time_high": result[0]}

# Endpoint to get the price range of a specific coin on a specified date
@app.get("/v1/coinmarketcap/coin_price_range", operation_id="get_coin_price_range", summary="Retrieves the price range (i.e., the difference between the highest and lowest prices) of a specific coin on a given date. The date must be provided in 'YYYY-MM-DD' format, and the coin's name is required to identify the correct coin.")
async def get_coin_price_range(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.high - T2.low FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T1.name = ?", (date, name))
    result = cursor.fetchone()
    if not result:
        return {"price_range": []}
    return {"price_range": result[0]}

# Endpoint to get the difference between max supply and total supply for a specific coin on a specific date
@app.get("/v1/coinmarketcap/max_total_supply_difference", operation_id="get_max_total_supply_difference", summary="Retrieves the difference between the maximum supply and the total supply of a specified coin on a given date. This operation requires the date in 'YYYY-MM-DD' format and the name of the coin as input parameters.")
async def get_max_total_supply_difference(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.max_supply - T2.total_supply FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T1.name = ?", (date, name))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the name of the coin with the highest 24-hour volume on a specific date
@app.get("/v1/coinmarketcap/coin_with_max_volume", operation_id="get_coin_with_max_volume", summary="Retrieves the name of the coin that had the highest 24-hour trading volume on a specified date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_coin_with_max_volume(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T2.volume_24h = ( SELECT MAX(volume_24h) FROM historical WHERE date = ? )", (date, date))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the names of coins with a total supply greater than a specified amount on a specific date
@app.get("/v1/coinmarketcap/coins_with_total_supply_greater_than", operation_id="get_coins_with_total_supply_greater_than", summary="Retrieves the names of cryptocurrencies that have a total supply greater than a specified threshold on a given date. The date and total supply threshold are provided as input parameters.")
async def get_coins_with_total_supply_greater_than(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), total_supply: int = Query(..., description="Total supply threshold")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T2.total_supply > ?", (date, total_supply))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the 7-day percent change status for a specific coin on a specific date
@app.get("/v1/coinmarketcap/percent_change_7d_status", operation_id="get_percent_change_7d_status", summary="Retrieves the 7-day percent change status (either 'INCREASED' or 'DECREASED') for a specific coin on a given date. The status is determined by comparing the coin's price on the given date to its price 7 days prior. The operation requires the coin's name and the date in 'YYYY-MM-DD' format as input parameters.")
async def get_percent_change_7d_status(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT (CASE WHEN T2.percent_change_7d > 0 THEN 'INCREASED' ELSE 'DECREASED' END) FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T1.name = ?", (date, name))
    result = cursor.fetchone()
    if not result:
        return {"status": []}
    return {"status": result[0]}

# Endpoint to get the name of the coin with the highest circulating supply on a specific date among a list of coins
@app.get("/v1/coinmarketcap/coin_with_highest_circulating_supply", operation_id="get_coin_with_highest_circulating_supply", summary="Retrieves the name of the coin with the highest circulating supply on a specified date, from a list of provided coins. The date must be in 'YYYY-MM-DD' format.")
async def get_coin_with_highest_circulating_supply(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name1: str = Query(..., description="Name of the first coin"), name2: str = Query(..., description="Name of the second coin")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T1.name IN (?, ?) ORDER BY T2.circulating_supply DESC LIMIT 1", (date, name1, name2))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"name": result[0]}

# Endpoint to get the price of a specific coin on a specific date
@app.get("/v1/coinmarketcap/coin_price", operation_id="get_coin_price", summary="Retrieves the price of a specific cryptocurrency on a given date. The operation requires the name of the coin and the date in 'YYYY-MM-DD' format. The returned price reflects the historical value of the coin on the specified date.")
async def get_coin_price(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.price FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T1.name = ?", (date, name))
    result = cursor.fetchone()
    if not result:
        return {"price": []}
    return {"price": result[0]}

# Endpoint to get the average price of a specific coin for a specific year
@app.get("/v1/coinmarketcap/average_coin_price_year", operation_id="get_average_coin_price_year", summary="Retrieves the average price of a specific cryptocurrency for a given year. The operation calculates the average price based on historical data, taking into account the specified year and the name of the coin. The result provides a single value representing the average price for the specified year.")
async def get_average_coin_price_year(year: str = Query(..., description="Year in 'YYYY' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT AVG(T2.price) FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE STRFTIME('%Y', T2.date) = ? AND T1.name = ?", (year, name))
    result = cursor.fetchone()
    if not result:
        return {"average_price": []}
    return {"average_price": result[0]}

# Endpoint to get the ratio of the difference between max supply and total supply to total supply for a specific coin before a specific date
@app.get("/v1/coinmarketcap/supply_ratio_before_date", operation_id="get_supply_ratio_before_date", summary="Retrieves the ratio of the difference between the maximum supply and the total supply to the total supply for a specific coin, based on historical data before a specified date.")
async def get_supply_ratio_before_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT CAST((SUM(T2.max_supply) - SUM(T2.total_supply)) AS REAL) / SUM(T2.total_supply) FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date < ? AND T1.name = ?", (date, name))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the names of coins with a specific status
@app.get("/v1/coinmarketcap/coins_by_status", operation_id="get_coins_by_status", summary="Retrieves the names of coins that match the specified status. The status parameter is used to filter the coins and return only those that meet the provided criteria.")
async def get_coins_by_status(status: str = Query(..., description="Status of the coin")):
    cursor.execute("SELECT name FROM coins WHERE status = ?", (status,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [row[0] for row in result]}

# Endpoint to get the description of a specific coin
@app.get("/v1/coinmarketcap/coin_description", operation_id="get_coin_description", summary="Retrieves the description of a specific coin from the database. The operation requires the name of the coin as an input parameter to identify the coin and return its description.")
async def get_coin_description(name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT description FROM coins WHERE name = ?", (name,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the count of coins and their names added in a specific month and year
@app.get("/v1/coinmarketcap/count_and_names_by_month_year", operation_id="get_count_and_names_by_month_year", summary="Retrieves the total number of coins and their respective names that were added during a specified month and year. The input parameter 'month_year' in 'YYYY-MM' format is used to filter the coins added in the given month and year.")
async def get_count_and_names_by_month_year(month_year: str = Query(..., description="Month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(id) num FROM coins WHERE STRFTIME('%Y-%m', date_added) = ? UNION ALL SELECT name FROM coins WHERE STRFTIME('%Y-%m', date_added) = ?", (month_year, month_year))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the name and symbol of coins added on a specific date
@app.get("/v1/coinmarketcap/name_symbol_by_date", operation_id="get_name_symbol_by_date", summary="Retrieve the names and symbols of coins that were added on a specified date. The date should be provided in the 'YYYY-MM-DD' format.")
async def get_name_symbol_by_date(date_added: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT name, symbol FROM coins WHERE date_added LIKE ?", (date_added + '%',))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the name of coins added in a specific year and with a specific status
@app.get("/v1/coinmarketcap/name_by_year_status", operation_id="get_name_by_year_status", summary="Retrieves the names of coins that were added in a specified year and have a specified status. The year should be provided in 'YYYY' format, and the status should indicate the coin's current standing.")
async def get_name_by_year_status(year: str = Query(..., description="Year in 'YYYY' format"), status: str = Query(..., description="Status of the coin")):
    cursor.execute("SELECT name FROM coins WHERE date_added LIKE ? AND status = ?", (year + '%', status))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the name of coins with a specific number of tags
@app.get("/v1/coinmarketcap/name_by_tag_count", operation_id="get_name_by_tag_count", summary="Retrieves the names of coins that have a specified number of tags. The tag count is provided as an input parameter, allowing for targeted results. This operation is useful for identifying coins with a particular number of associated tags.")
async def get_name_by_tag_count(tag_count: int = Query(..., description="Number of tags")):
    cursor.execute("SELECT name FROM coins WHERE LENGTH(tag_names) - LENGTH(replace(tag_names, ',', '')) = ?", (tag_count,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the name of the coin with the highest price in history
@app.get("/v1/coinmarketcap/name_by_max_price", operation_id="get_name_by_max_price", summary="Retrieves the name of the cryptocurrency that has achieved the highest price in its history. This operation identifies the coin with the maximum historical price and returns its name.")
async def get_name_by_max_price():
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.price = ( SELECT MAX(price) FROM historical )")
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the date with the lowest price for a specific coin
@app.get("/v1/coinmarketcap/date_by_lowest_price", operation_id="get_date_by_lowest_price", summary="Retrieves the date on which the specified coin had its lowest recorded price. The coin is identified by its name, and the date is determined based on historical price data.")
async def get_date_by_lowest_price(coin_name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.date FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? ORDER BY T2.low LIMIT 1", (coin_name,))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result[0]}

# Endpoint to get the name of the coin with the highest 24-hour volume in history
@app.get("/v1/coinmarketcap/name_by_max_volume", operation_id="get_name_by_max_volume", summary="Retrieves the name of the cryptocurrency that has achieved the highest 24-hour trading volume in history. This operation does not require any input parameters and returns the name of the coin that has had the most significant trading activity in a single day.")
async def get_name_by_max_volume():
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.volume_24h = ( SELECT MAX(volume_24h) FROM historical )")
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the name of coins with no opening price in a specific month and year
@app.get("/v1/coinmarketcap/name_by_no_open_price", operation_id="get_name_by_no_open_price", summary="Retrieves the names of coins that have no recorded opening price for a specified month and year. The month and year should be provided in the 'YYYY-MM' format.")
async def get_name_by_no_open_price(month_year: str = Query(..., description="Month and year in 'YYYY-MM' format")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE STRFTIME('%Y-%m', T2.date) = ? AND T2.open IS NULL", (month_year,))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the date with the highest closing price for a specific coin
@app.get("/v1/coinmarketcap/date_by_highest_close", operation_id="get_date_by_highest_close", summary="Retrieves the date with the highest closing price for a specific coin. The operation uses the provided coin name to filter the data and returns the date with the highest closing price.")
async def get_date_by_highest_close(coin_name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.date FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? ORDER BY T2.close DESC LIMIT 1", (coin_name,))
    result = cursor.fetchone()
    if not result:
        return {"data": []}
    return {"data": result[0]}

# Endpoint to get the dates for a specific coin name and CMC rank
@app.get("/v1/coinmarketcap/dates_by_coin_name_and_rank", operation_id="get_dates_by_coin_name_and_rank", summary="Retrieves the dates for a specific coin, identified by its name and CMC rank. The operation returns a list of dates for which historical data is available for the specified coin and its corresponding CMC rank.")
async def get_dates_by_coin_name_and_rank(coin_name: str = Query(..., description="Name of the coin"), cmc_rank: int = Query(..., description="CMC rank of the coin")):
    cursor.execute("SELECT T2.date FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND T2.cmc_rank = ?", (coin_name, cmc_rank))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": [row[0] for row in result]}

# Endpoint to get the date with the highest market cap for a specific coin name
@app.get("/v1/coinmarketcap/date_highest_market_cap_by_coin_name", operation_id="get_date_highest_market_cap_by_coin_name", summary="Retrieves the date on which a specific coin reached its highest market capitalization. The coin is identified by its name.")
async def get_date_highest_market_cap_by_coin_name(coin_name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT T2.date FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? ORDER BY T2.market_cap DESC LIMIT 1", (coin_name,))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the names of coins with a specific date and CMC rank
@app.get("/v1/coinmarketcap/coin_names_by_date_and_rank", operation_id="get_coin_names_by_date_and_rank", summary="Retrieves the names of coins that have a CMC rank less than or equal to the specified rank on a given date. The date should be provided in 'YYYY-MM-DD' format.")
async def get_coin_names_by_date_and_rank(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), cmc_rank: int = Query(..., description="CMC rank of the coin")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T2.cmc_rank <= ?", (date, cmc_rank))
    result = cursor.fetchall()
    if not result:
        return {"coin_names": []}
    return {"coin_names": [row[0] for row in result]}

# Endpoint to get distinct dates for a specific coin name where the open price is null or zero
@app.get("/v1/coinmarketcap/distinct_dates_by_coin_name_and_open_price", operation_id="get_distinct_dates_by_coin_name_and_open_price", summary="Retrieves unique dates for a specified coin where the opening price is either missing or zero. This operation is useful for identifying gaps in historical price data for a particular coin.")
async def get_distinct_dates_by_coin_name_and_open_price(coin_name: str = Query(..., description="Name of the coin")):
    cursor.execute("SELECT DISTINCT T2.date FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND (T2.open IS NULL OR T2.open = 0)", (coin_name,))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": [row[0] for row in result]}

# Endpoint to get the count of coin IDs for a specific coin name and year-month
@app.get("/v1/coinmarketcap/count_coin_ids_by_name_and_year_month", operation_id="get_count_coin_ids_by_name_and_year_month", summary="Retrieves the total number of unique coin IDs associated with a specified coin name and a given year-month. The year-month should be provided in 'YYYY-MM' format. This operation is useful for understanding the historical prevalence of a particular coin during a specific time period.")
async def get_count_coin_ids_by_name_and_year_month(coin_name: str = Query(..., description="Name of the coin"), year_month: str = Query(..., description="Year and month in 'YYYY-MM' format")):
    cursor.execute("SELECT COUNT(T2.coin_id) FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T1.name = ? AND STRFTIME('%Y-%m', T2.date) = ?", (coin_name, year_month))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of coins with a price above the average price on a specific date
@app.get("/v1/coinmarketcap/coin_names_above_avg_price_by_date", operation_id="get_coin_names_above_avg_price_by_date", summary="Retrieves the names of cryptocurrencies that had a price higher than the average price on a specific date. The date must be provided in 'YYYY-MM-DD' format.")
async def get_coin_names_above_avg_price_by_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T2.price > ( SELECT AVG(price) FROM historical WHERE date = ? )", (date, date))
    result = cursor.fetchall()
    if not result:
        return {"coin_names": []}
    return {"coin_names": [row[0] for row in result]}

# Endpoint to get the names of coins with a positive percent change in the last hour on a specific date
@app.get("/v1/coinmarketcap/coin_names_positive_percent_change_by_date", operation_id="get_coin_names_positive_percent_change_by_date", summary="Retrieve the names of cryptocurrencies that have experienced a positive price change in the last hour on a specified date. The date must be provided in the 'YYYY-MM-DD' format.")
async def get_coin_names_positive_percent_change_by_date(date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.name FROM coins AS T1 INNER JOIN historical AS T2 ON T1.id = T2.coin_id WHERE T2.date = ? AND T2.percent_change_1h > 0", (date,))
    result = cursor.fetchall()
    if not result:
        return {"coin_names": []}
    return {"coin_names": [row[0] for row in result]}

api_calls = [
    "/v1/coinmarketcap/coin_name_highest_market_cap?year=2018",
    "/v1/coinmarketcap/coin_volume_24h?coin_name=Argentum&date=2016-10-11",
    "/v1/coinmarketcap/coin_price_and_avg_price?coin_name=Zetacoin&start_date=2013-11-01&end_date=2013-11-07",
    "/v1/coinmarketcap/coin_time_high_low?coin_name=WARP&year_month=2016-08",
    "/v1/coinmarketcap/coin_highest_price_date?coin_name=DigixDAO",
    "/v1/coinmarketcap/coin_highest_percent_change_24h",
    "/v1/coinmarketcap/coin_avg_monthly_circulating_supply?coin_name=Frozen&year=2014",
    "/v1/coinmarketcap/latest_date_inactive_coin?status=inactive",
    "/v1/coinmarketcap/coin_avg_price_year?coin_name=Bitcoin&year=2016",
    "/v1/coinmarketcap/coin_lowest_price_date?coin_name=Bitcoin",
    "/v1/coinmarketcap/coin_status_avg_price?avg_price=1000",
    "/v1/coinmarketcap/coin_min_percent_change_1h",
    "/v1/coinmarketcap/coin_highest_price_range?category=token",
    "/v1/coinmarketcap/coin_avg_percent_change_24h?date=2020-06-22",
    "/v1/coinmarketcap/coin_cmc_rank?date=2013-04-28&cmc_rank=1",
    "/v1/coinmarketcap/coin_market_cap?date=2013-04-28&name=Bitcoin",
    "/v1/coinmarketcap/coin_null_open?date=2013-05-03",
    "/v1/coinmarketcap/coin_close_price?date=2013-04-29&name=Bitcoin",
    "/v1/coinmarketcap/coin_time_high?date=2013-04-29&name=Bitcoin",
    "/v1/coinmarketcap/coin_price_range?date=2013-04-28&name=Bitcoin",
    "/v1/coinmarketcap/max_total_supply_difference?date=2013-04-28&name=Bitcoin",
    "/v1/coinmarketcap/coin_with_max_volume?date=2016-01-08",
    "/v1/coinmarketcap/coins_with_total_supply_greater_than?date=2013-04-28&total_supply=10000000",
    "/v1/coinmarketcap/percent_change_7d_status?date=2013-05-05&name=Bitcoin",
    "/v1/coinmarketcap/coin_with_highest_circulating_supply?date=2013-04-28&name1=Bitcoin&name2=Litecoin",
    "/v1/coinmarketcap/coin_price?date=2013-04-28&name=Bitcoin",
    "/v1/coinmarketcap/average_coin_price_year?year=2013&name=Bitcoin",
    "/v1/coinmarketcap/supply_ratio_before_date?date=2018-04-28&name=Bitcoin",
    "/v1/coinmarketcap/coins_by_status?status=extinct",
    "/v1/coinmarketcap/coin_description?name=BitBar",
    "/v1/coinmarketcap/count_and_names_by_month_year?month_year=2013-05",
    "/v1/coinmarketcap/name_symbol_by_date?date_added=2013-06-14",
    "/v1/coinmarketcap/name_by_year_status?year=2014&status=untracked",
    "/v1/coinmarketcap/name_by_tag_count?tag_count=2",
    "/v1/coinmarketcap/name_by_max_price",
    "/v1/coinmarketcap/date_by_lowest_price?coin_name=Bitcoin",
    "/v1/coinmarketcap/name_by_max_volume",
    "/v1/coinmarketcap/name_by_no_open_price?month_year=2013-05",
    "/v1/coinmarketcap/date_by_highest_close?coin_name=CHNCoin",
    "/v1/coinmarketcap/dates_by_coin_name_and_rank?coin_name=Peercoin&cmc_rank=5",
    "/v1/coinmarketcap/date_highest_market_cap_by_coin_name?coin_name=Devcoin",
    "/v1/coinmarketcap/coin_names_by_date_and_rank?date=2014-01-01&cmc_rank=5",
    "/v1/coinmarketcap/distinct_dates_by_coin_name_and_open_price?coin_name=Lebowskis",
    "/v1/coinmarketcap/count_coin_ids_by_name_and_year_month?coin_name=Bytecoin&year_month=2013-06",
    "/v1/coinmarketcap/coin_names_above_avg_price_by_date?date=2018-04-28",
    "/v1/coinmarketcap/coin_names_positive_percent_change_by_date?date=2013-05-29"
]
