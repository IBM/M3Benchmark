from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/college_completion/college_completion.sqlite')
cursor = conn.cursor()

# Endpoint to get the most common institution name for a given year and race
@app.get("/v1/college_completion/most_common_institution_name", operation_id="get_most_common_institution_name", summary="Retrieves the name of the institution with the highest number of graduates for a specific year and race. The year and race are used to filter the graduation cohort data, and the institution with the most graduates is determined based on the count of occurrences in the filtered dataset.")
async def get_most_common_institution_name(year: int = Query(..., description="Year of the graduation cohort"), race: str = Query(..., description="Race of the graduates")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T1.state = T2.state WHERE T2.year = ? AND T2.race = ? GROUP BY T1.chronname ORDER BY COUNT(T1.chronname) DESC LIMIT 1", (year, race))
    result = cursor.fetchone()
    if not result:
        return {"chronname": []}
    return {"chronname": result[0]}

# Endpoint to get institution details for a given year and graduation cohort
@app.get("/v1/college_completion/institution_details_by_year_cohort", operation_id="get_institution_details_by_year_cohort", summary="Retrieves institution details for a specific year and graduation cohort. The operation filters institutions based on the provided year and graduation cohort, and returns the institution name and site.")
async def get_institution_details_by_year_cohort(year: int = Query(..., description="Year of the graduation cohort"), grad_cohort: int = Query(..., description="Graduation cohort")):
    cursor.execute("SELECT T1.chronname, T1.site FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T1.state = T2.state WHERE T2.year = ? AND T2.grad_cohort = ?", (year, grad_cohort))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the count of graduation cohorts for a given year range, institution name, and gender
@app.get("/v1/college_completion/count_grad_cohorts_by_year_range_institution_gender", operation_id="get_count_grad_cohorts_by_year_range_institution_gender", summary="Retrieves the total number of graduation cohorts within a specified year range for a particular institution and gender. The response is based on the provided start and end years, institution name, and gender.")
async def get_count_grad_cohorts_by_year_range_institution_gender(start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), chronname: str = Query(..., description="Institution name"), gender: str = Query(..., description="Gender of the graduates")):
    cursor.execute("SELECT COUNT(T2.grad_cohort) FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T1.state = T2.state WHERE T2.year BETWEEN ? AND ? AND T1.chronname = ? AND T2.gender = ?", (start_year, end_year, chronname, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of graduation cohorts for a given year and gender, ordered by aid value
@app.get("/v1/college_completion/count_grad_cohorts_by_year_gender_aid_value", operation_id="get_count_grad_cohorts_by_year_gender_aid_value", summary="Retrieves the total number of graduation cohorts for a specific year and gender, sorted by the descending order of aid value. The result is limited to the top record.")
async def get_count_grad_cohorts_by_year_gender_aid_value(year: int = Query(..., description="Year of the graduation cohort"), gender: str = Query(..., description="Gender of the graduates")):
    cursor.execute("SELECT COUNT(T2.grad_cohort) FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T1.state = T2.state WHERE T2.year = ? AND T2.gender = ? ORDER BY T1.aid_value DESC LIMIT 1", (year, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average median SAT value for a given year and gender, grouped by graduation cohort
@app.get("/v1/college_completion/avg_med_sat_value_by_year_gender", operation_id="get_avg_med_sat_value_by_year_gender", summary="Retrieves the average median SAT score for a specific year and gender, focusing on the graduation cohort with the highest number of graduates. The data is aggregated by graduation cohort, providing insights into the academic performance of students based on their year of graduation and gender.")
async def get_avg_med_sat_value_by_year_gender(year: int = Query(..., description="Year of the graduation cohort"), gender: str = Query(..., description="Gender of the graduates")):
    cursor.execute("SELECT AVG(T1.med_sat_value) FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T1.state = T2.state WHERE T2.year = ? AND T2.gender = ? GROUP BY T2.grad_cohort ORDER BY COUNT(T2.grad_cohort) DESC LIMIT 1", (year, gender))
    result = cursor.fetchone()
    if not result:
        return {"avg_med_sat_value": []}
    return {"avg_med_sat_value": result[0]}

# Endpoint to get the state and institution name for a given year and control type, grouped by graduation cohort
@app.get("/v1/college_completion/state_institution_by_year_control", operation_id="get_state_institution_by_year_control", summary="Retrieves the state and institution name for the specified year and control type, grouped by graduation cohort. The results are ordered by the count of graduates in each cohort, with the cohort with the highest number of graduates listed first.")
async def get_state_institution_by_year_control(year: int = Query(..., description="Year of the graduation cohort"), control: str = Query(..., description="Control type of the institution")):
    cursor.execute("SELECT T1.state, T1.chronname FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T1.state = T2.state WHERE T2.year = ? AND T1.control = ? GROUP BY T2.grad_cohort ORDER BY COUNT(T2.grad_cohort) DESC LIMIT 1", (year, control))
    result = cursor.fetchone()
    if not result:
        return {"state": [], "chronname": []}
    return {"state": result[0], "chronname": result[1]}

# Endpoint to get the institution name for a given year and control type, ordered by graduation cohort
@app.get("/v1/college_completion/institution_name_by_year_control", operation_id="get_institution_name_by_year_control", summary="Retrieve the name of the institution that matches the specified year and control type, sorted by graduation cohort. The result is limited to a single entry.")
async def get_institution_name_by_year_control(year: int = Query(..., description="Year of the graduation cohort"), control: str = Query(..., description="Control type of the institution")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T1.state = T2.state WHERE T2.year = ? AND T1.control = ? ORDER BY T2.grad_cohort LIMIT 1", (year, control))
    result = cursor.fetchone()
    if not result:
        return {"chronname": []}
    return {"chronname": result[0]}

# Endpoint to get institution details for a given year, gender, race, and maximum graduation cohort
@app.get("/v1/college_completion/institution_details_by_year_gender_race_cohort", operation_id="get_institution_details_by_year_gender_race_cohort", summary="Retrieves institution details, including the chronological name and graduation cohort, for a specific year, gender, and race. The results are filtered to include only those with a graduation cohort less than the provided maximum cohort.")
async def get_institution_details_by_year_gender_race_cohort(year: int = Query(..., description="Year of the graduation cohort"), gender: str = Query(..., description="Gender of the graduates"), race: str = Query(..., description="Race of the graduates"), max_grad_cohort: int = Query(..., description="Maximum graduation cohort")):
    cursor.execute("SELECT T1.chronname, T2.grad_cohort FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T2.year = ? AND T2.gender = ? AND T2.race = ? AND T2.grad_cohort < ?", (year, gender, race, max_grad_cohort))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the count of graduates for a given year, gender, race, and institution name
@app.get("/v1/college_completion/count_graduates_by_year_gender_race_institution", operation_id="get_count_graduates_by_year_gender_race_institution", summary="Retrieves the total number of graduates from a specific institution, filtered by a particular year, gender, and race. This operation provides a detailed breakdown of graduation statistics, enabling users to analyze graduation trends across different demographics and time periods.")
async def get_count_graduates_by_year_gender_race_institution(year: int = Query(..., description="Year of the graduation cohort"), gender: str = Query(..., description="Gender of the graduates"), race: str = Query(..., description="Race of the graduates"), chronname: str = Query(..., description="Institution name")):
    cursor.execute("SELECT COUNT(*) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T2.year = ? AND T2.gender = ? AND T2.race = ? AND T1.chronname = ?", (year, gender, race, chronname))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of graduates for a given gender, race, institution name, and cohort
@app.get("/v1/college_completion/count_graduates_by_gender_race_institution_cohort", operation_id="get_count_graduates_by_gender_race_institution_cohort", summary="Retrieves the total number of graduates from a specific institution, categorized by gender, race, and cohort type. This operation allows you to analyze graduation statistics for a particular institution, gender, race, and cohort.")
async def get_count_graduates_by_gender_race_institution_cohort(gender: str = Query(..., description="Gender of the graduates"), race: str = Query(..., description="Race of the graduates"), chronname: str = Query(..., description="Institution name"), cohort: str = Query(..., description="Cohort type")):
    cursor.execute("SELECT COUNT(*) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T2.gender = ? AND T2.race = ? AND T1.chronname = ? AND T2.cohort = ?", (gender, race, chronname, cohort))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of graduation cohorts for specific years, gender, race, and institution name
@app.get("/v1/college_completion/sum_grad_cohorts_by_year_gender_race_institution", operation_id="get_sum_grad_cohorts", summary="Retrieves the total number of graduation cohorts for three specified years, filtered by gender, race, and institution name. This operation aggregates data from the institution details and graduation records, providing a comprehensive view of graduation cohort sums for the given criteria.")
async def get_sum_grad_cohorts(year1: int = Query(..., description="First year"), year2: int = Query(..., description="Second year"), year3: int = Query(..., description="Third year"), gender: str = Query(..., description="Gender"), race: str = Query(..., description="Race"), institution_name: str = Query(..., description="Institution name")):
    cursor.execute("SELECT SUM(CASE WHEN T2.year = ? THEN T2.grad_cohort ELSE 0 END), SUM(CASE WHEN T2.year = ? THEN T2.grad_cohort ELSE 0 END), SUM(CASE WHEN T2.year = ? THEN T2.grad_cohort ELSE 0 END) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T2.gender = ? AND T2.race = ? AND T1.chronname = ?", (year1, year2, year3, gender, race, institution_name))
    result = cursor.fetchone()
    if not result:
        return {"sums": []}
    return {"sums": result}

# Endpoint to get the percentage of graduates of a specific race from institutions with a specific control type
@app.get("/v1/college_completion/percentage_graduates_by_race_control", operation_id="get_percentage_graduates", summary="Retrieves the percentage of graduates of a specific race from institutions with a particular control type. The operation calculates this percentage by summing the number of graduates of the specified race and dividing it by the total number of graduates in the cohort. The race and control type are provided as input parameters.")
async def get_percentage_graduates(race: str = Query(..., description="Race"), control_type: str = Query(..., description="Control type")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.race = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(T2.grad_cohort) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T2.race = ? AND T1.control = ?", (race, race, control_type))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of graduates of a specific race from institutions in a specific state and year
@app.get("/v1/college_completion/percentage_graduates_by_race_state_year", operation_id="get_percentage_graduates_state_year", summary="Retrieves the percentage of graduates of a specific race from institutions in a given state and year. The operation calculates this percentage by summing the number of graduates of the specified race and dividing it by the total number of graduates in the given year and state.")
async def get_percentage_graduates_state_year(race: str = Query(..., description="Race"), year: int = Query(..., description="Year"), state: str = Query(..., description="State")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.race = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(T2.grad_cohort) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T2.year = ? AND T1.state = ?", (race, year, state))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the ratio of graduation cohorts by gender for a specific institution, year, and race
@app.get("/v1/college_completion/grad_cohort_ratio_by_gender_institution_year_race", operation_id="get_grad_cohort_ratio", summary="Retrieves the ratio of graduation cohorts for two specified genders at a particular institution, during a given year, and for a specific race. The calculation is based on the sum of graduation cohorts for each gender, divided by the total sum of graduation cohorts for both genders.")
async def get_grad_cohort_ratio(gender1: str = Query(..., description="First gender"), gender2: str = Query(..., description="Second gender"), institution_name: str = Query(..., description="Institution name"), year: int = Query(..., description="Year"), race: str = Query(..., description="Race")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.Gender = ? THEN T2.grad_cohort ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.Gender = ? THEN T2.grad_cohort ELSE 0 END) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T1.chronname = ? AND T2.year = ? AND T2.race = ?", (gender1, gender2, institution_name, year, race))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the institution name with the highest grad_100_value from a list of institution names
@app.get("/v1/college_completion/top_institution_by_grad_value", operation_id="get_top_institution", summary="Retrieves the name of the institution with the highest 10-year graduation value from a provided list of institution names. The list should contain at least two institution names.")
async def get_top_institution(institution_name1: str = Query(..., description="First institution name"), institution_name2: str = Query(..., description="Second institution name")):
    cursor.execute("SELECT chronname FROM institution_details WHERE chronname IN (?, ?) ORDER BY grad_100_value LIMIT 1", (institution_name1, institution_name2))
    result = cursor.fetchone()
    if not result:
        return {"institution": []}
    return {"institution": result[0]}

# Endpoint to get the count of institutions with a specific control type
@app.get("/v1/college_completion/count_institutions_by_control", operation_id="get_count_institutions", summary="Retrieves the total number of institutions that match the specified control type. The control type is a parameter that filters the institutions to be counted.")
async def get_count_institutions(control_type: str = Query(..., description="Control type")):
    cursor.execute("SELECT COUNT(*) FROM institution_details WHERE control = ?", (control_type,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of graduation cohorts for a specific institution, gender, and race
@app.get("/v1/college_completion/sum_grad_cohorts_by_institution_gender_race", operation_id="get_sum_grad_cohorts_institution", summary="Retrieves the total number of graduation cohorts for a specific institution, categorized by gender and race. The endpoint uses the provided institution name, gender, and race to filter the data and calculate the sum of graduation cohorts.")
async def get_sum_grad_cohorts_institution(institution_name: str = Query(..., description="Institution name"), gender: str = Query(..., description="Gender"), race: str = Query(..., description="Race")):
    cursor.execute("SELECT SUM(T2.grad_cohort) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T1.chronname = ? AND T2.gender = ? AND T2.race = ?", (institution_name, gender, race))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the count of institutions with a specific name, year, and cohort
@app.get("/v1/college_completion/count_institutions_by_name_year_cohort", operation_id="get_count_institutions_name_year_cohort", summary="Retrieves the total number of institutions that match a specific name, year, and cohort. The response is based on a combination of institution details and graduation data.")
async def get_count_institutions_name_year_cohort(institution_name: str = Query(..., description="Institution name"), year: int = Query(..., description="Year"), cohort: str = Query(..., description="Cohort")):
    cursor.execute("SELECT COUNT(T1.unitid) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T1.chronname = ? AND T2.year = ? AND T2.cohort = ?", (institution_name, year, cohort))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the sum of graduation rates for a specific institution, year, gender, and race
@app.get("/v1/college_completion/sum_grad_rates_by_institution_year_gender_race", operation_id="get_sum_grad_rates", summary="Retrieves the aggregated graduation rates for a specific institution, year, gender, and race. The operation calculates the sum of graduation rates based on the provided institution name, year, gender, and race.")
async def get_sum_grad_rates(institution_name: str = Query(..., description="Institution name"), year: int = Query(..., description="Year"), gender: str = Query(..., description="Gender"), race: str = Query(..., description="Race")):
    cursor.execute("SELECT SUM(T2.grad_100) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T1.chronname = ? AND T2.year = ? AND T2.gender = ? AND T2.race = ?", (institution_name, year, gender, race))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get the graduation cohorts for a specific institution and year
@app.get("/v1/college_completion/grad_cohorts_by_institution_year", operation_id="get_grad_cohorts", summary="Retrieves the graduation cohorts for a specific institution and year. The operation requires the institution name and the year as input parameters. It returns the graduation cohorts for the specified institution and year, providing a comprehensive view of the graduation trends for that particular institution and year.")
async def get_grad_cohorts(institution_name: str = Query(..., description="Institution name"), year: int = Query(..., description="Year")):
    cursor.execute("SELECT T2.grad_cohort FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T1.unitid = T2.unitid WHERE T1.chronname = ? AND T2.year = ?", (institution_name, year))
    result = cursor.fetchall()
    if not result:
        return {"cohorts": []}
    return {"cohorts": [row[0] for row in result]}

# Endpoint to determine if female or male White students have a higher graduation rate in a specific institution and year
@app.get("/v1/college_completion/graduation_rate_comparison", operation_id="get_graduation_rate_comparison", summary="This operation compares the graduation rates of female and male White students in a specific institution and year. It returns the gender group with the higher graduation rate. The input parameters include the institution name, the year of the graduation data, and the race of the students.")
async def get_graduation_rate_comparison(chronname: str = Query(..., description="Name of the institution"), year: int = Query(..., description="Year of the graduation data"), race: str = Query(..., description="Race of the students")):
    cursor.execute("SELECT IIF(SUM(CASE WHEN T2.gender = 'F' THEN T2.grad_150 ELSE 0 END) > SUM(CASE WHEN T2.gender = 'M' THEN T2.grad_150 ELSE 0 END), 'female White students', 'male White students') FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname = ? AND T2.year = ? AND T2.race = ?", (chronname, year, race))
    result = cursor.fetchone()
    if not result:
        return {"comparison": []}
    return {"comparison": result[0]}

# Endpoint to get the institution with the highest graduation rate for a specific gender and race in a given year
@app.get("/v1/college_completion/highest_graduation_rate_institution", operation_id="get_highest_graduation_rate_institution", summary="Get the institution with the highest graduation rate for a specific gender and race in a given year")
async def get_highest_graduation_rate_institution(year: int = Query(..., description="Year of the graduation data"), gender: str = Query(..., description="Gender of the students"), race: str = Query(..., description="Race of the students")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.year = ? AND T2.gender = ? AND T2.race = ? AND T2.grad_150 = ( SELECT MAX(T2.grad_150) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.year = ? AND T2.gender = ? AND T2.race = ? )", (year, gender, race, year, gender, race))
    result = cursor.fetchone()
    if not result:
        return {"institution": []}
    return {"institution": result[0]}

# Endpoint to get institutions with graduation rates above a certain threshold for a specific gender and race in a given year
@app.get("/v1/college_completion/institutions_above_graduation_threshold", operation_id="get_institutions_above_graduation_threshold", summary="Retrieves a list of institutions that have graduation rates surpassing a specified threshold for a particular gender and race in a given year. The data returned is distinct and excludes institutions with null graduation rates.")
async def get_institutions_above_graduation_threshold(grad_threshold: int = Query(..., description="Graduation rate threshold"), year: int = Query(..., description="Year of the graduation data"), gender: str = Query(..., description="Gender of the students"), race: str = Query(..., description="Race of the students")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE WHEN T2.grad_150 > ? THEN T1.chronname ELSE NULL END AS T FROM institution_details T1 INNER JOIN institution_grads T2 ON T2.unitid = T1.unitid WHERE T2.year = ? AND T2.gender = ? AND T2.race = ? ) WHERE T IS NOT NULL", (grad_threshold, year, gender, race))
    result = cursor.fetchall()
    if not result:
        return {"institutions": []}
    return {"institutions": [row[0] for row in result]}

# Endpoint to get the total graduation cohort for a specific cohort, year, and state
@app.get("/v1/college_completion/total_graduation_cohort", operation_id="get_total_graduation_cohort", summary="Retrieves the total number of students in a specific graduation cohort, based on the provided cohort type, year, and state. This operation aggregates graduation data from multiple institutions within the specified state, providing a comprehensive view of the graduation cohort size for the given year and cohort type.")
async def get_total_graduation_cohort(cohort: str = Query(..., description="Cohort type"), year: int = Query(..., description="Year of the graduation data"), state: str = Query(..., description="State of the institution")):
    cursor.execute("SELECT SUM(T2.grad_cohort) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.cohort = ? AND T2.year = ? AND T1.state = ?", (cohort, year, state))
    result = cursor.fetchone()
    if not result:
        return {"total_cohort": []}
    return {"total_cohort": result[0]}

# Endpoint to get the difference in graduation cohort between two years for a specific institution
@app.get("/v1/college_completion/graduation_cohort_difference", operation_id="get_graduation_cohort_difference", summary="Retrieve the difference in graduation cohort between two specified years for a given institution. This operation calculates the sum of graduation cohorts for the first year and subtracts the sum of graduation cohorts for the second year, providing a comparison of graduation cohort sizes between the two years for the selected institution.")
async def get_graduation_cohort_difference(year1: int = Query(..., description="First year of the graduation data"), year2: int = Query(..., description="Second year of the graduation data"), chronname: str = Query(..., description="Name of the institution")):
    cursor.execute("SELECT SUM(CASE WHEN T2.year = ? THEN T2.grad_cohort ELSE 0 END) - SUM(CASE WHEN T2.year = ? THEN T2.grad_cohort ELSE 0 END) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname = ?", (year1, year2, chronname))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the count of distinct institutions with a specific SAT percentile and graduation cohort above a threshold in a given state and year
@app.get("/v1/college_completion/count_distinct_institutions", operation_id="get_count_distinct_institutions", summary="Retrieve the count of unique institutions in a specified state that have a given median SAT percentile and a graduation cohort above a certain threshold for a particular year.")
async def get_count_distinct_institutions(state: str = Query(..., description="State of the institution"), med_sat_percentile: str = Query(..., description="Median SAT percentile"), year: int = Query(..., description="Year of the graduation data"), grad_cohort_threshold: int = Query(..., description="Graduation cohort threshold")):
    cursor.execute("SELECT COUNT(DISTINCT T1.chronname) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.state = ? AND T1.med_sat_percentile = ? AND T2.year = ? AND T2.grad_cohort > ?", (state, med_sat_percentile, year, grad_cohort_threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of institutions with a specific control type and graduation rate above a threshold in a given state and year
@app.get("/v1/college_completion/count_institutions_by_control_type", operation_id="get_count_institutions_by_control_type", summary="Retrieve the number of institutions in a specified state that meet a certain control type and have a graduation rate surpassing a given threshold for a particular year.")
async def get_count_institutions_by_control_type(state: str = Query(..., description="State of the institution"), control: str = Query(..., description="Control type of the institution"), year: int = Query(..., description="Year of the graduation data"), grad_100_threshold: int = Query(..., description="Graduation rate threshold")):
    cursor.execute("SELECT COUNT(T1.chronname) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.state = ? AND T1.control = ? AND T2.year = ? AND T2.grad_100 > ?", (state, control, year, grad_100_threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct institutions with a specific race and graduation cohort above a threshold in a given state and year
@app.get("/v1/college_completion/distinct_institutions_by_race", operation_id="get_distinct_institutions_by_race", summary="Retrieve a unique list of institutions in a specified state that meet the given criteria for race and graduation cohort size in a particular year. The criteria include a minimum graduation cohort size threshold.")
async def get_distinct_institutions_by_race(state: str = Query(..., description="State of the institution"), year: int = Query(..., description="Year of the graduation data"), race: str = Query(..., description="Race of the students"), grad_cohort_threshold: int = Query(..., description="Graduation cohort threshold")):
    cursor.execute("SELECT DISTINCT T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.state = ? AND T2.year = ? AND T2.race = ? AND T2.grad_cohort > ?", (state, year, race, grad_cohort_threshold))
    result = cursor.fetchall()
    if not result:
        return {"institutions": []}
    return {"institutions": [row[0] for row in result]}

# Endpoint to get the average graduation cohort for a specific institution, gender, and race over multiple years
@app.get("/v1/college_completion/average_graduation_cohort", operation_id="get_average_graduation_cohort", summary="Retrieves the average graduation cohort for a specific institution, considering the gender and race of the students, over a range of years. The operation calculates the average graduation cohort based on the provided institution name, gender, and race, and the years of graduation data.")
async def get_average_graduation_cohort(chronname: str = Query(..., description="Name of the institution"), year1: int = Query(..., description="First year of the graduation data"), year2: int = Query(..., description="Second year of the graduation data"), year3: int = Query(..., description="Third year of the graduation data"), gender: str = Query(..., description="Gender of the students"), race: str = Query(..., description="Race of the students")):
    cursor.execute("SELECT AVG(T2.grad_cohort) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname = ? AND T2.year IN (?, ?, ?) AND T2.gender = ? AND T2.race = ?", (chronname, year1, year2, year3, gender, race))
    result = cursor.fetchone()
    if not result:
        return {"average_cohort": []}
    return {"average_cohort": result[0]}

# Endpoint to get the average graduation rate for a specific institution
@app.get("/v1/college_completion/average_graduation_rate", operation_id="get_average_graduation_rate", summary="Retrieves the average graduation rate for a specified institution. The operation calculates this rate by aggregating graduation data from the institution's details and graduation records.")
async def get_average_graduation_rate(chronname: str = Query(..., description="Name of the institution")):
    cursor.execute("SELECT AVG(T2.grad_100_rate) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname = ?", (chronname,))
    result = cursor.fetchone()
    if not result:
        return {"average_rate": []}
    return {"average_rate": result[0]}

# Endpoint to get the site of a specific institution
@app.get("/v1/college_completion/institution_site", operation_id="get_institution_site", summary="Retrieves the site of a specific institution by its name. The operation filters the institution details to find the unique site associated with the provided institution name.")
async def get_institution_site(chronname: str = Query(..., description="Name of the institution")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE WHEN chronname = ? THEN site ELSE NULL END AS T FROM institution_details ) WHERE T IS NOT NULL", (chronname,))
    result = cursor.fetchall()
    if not result:
        return {"sites": []}
    return {"sites": [row[0] for row in result]}

# Endpoint to get the state of a specific institution
@app.get("/v1/college_completion/institution_state", operation_id="get_institution_state", summary="Retrieve the state of a specific institution by providing its name. This operation returns a distinct state value for the given institution, ensuring that no duplicate entries are included in the response.")
async def get_institution_state(chronname: str = Query(..., description="Name of the institution")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE WHEN chronname = ? THEN state ELSE NULL END AS T FROM institution_details ) WHERE T IS NOT NULL", (chronname,))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get the city of a specific institution
@app.get("/v1/college_completion/institution_city", operation_id="get_institution_city", summary="Retrieve the city of a specific institution by providing its name. This operation returns the unique city associated with the given institution name, ensuring no duplicate entries are considered.")
async def get_institution_city(chronname: str = Query(..., description="Name of the institution")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE WHEN chronname = ? THEN city ELSE NULL END AS T FROM institution_details ) WHERE T IS NOT NULL", (chronname,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the state abbreviation for a specific state
@app.get("/v1/college_completion/state_abbreviation", operation_id="get_state_abbreviation", summary="Retrieves the abbreviation of a specified state. The operation filters the distinct state abbreviations from the state_sector_grads table and returns the abbreviation corresponding to the provided state name.")
async def get_state_abbreviation(state: str = Query(..., description="Name of the state")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE WHEN state = ? THEN state_abbr ELSE NULL END AS T FROM state_sector_grads ) WHERE T IS NOT NULL", (state,))
    result = cursor.fetchall()
    if not result:
        return {"state_abbreviations": []}
    return {"state_abbreviations": [row[0] for row in result]}

# Endpoint to get the count of state IDs based on state, level, and control
@app.get("/v1/college_completion/count_state_ids", operation_id="get_count_state_ids", summary="Retrieves the total number of unique state IDs that match the specified state, institution level, and control type. This operation is useful for understanding the distribution of institutions across different states, levels, and control types.")
async def get_count_state_ids(state: str = Query(..., description="Name of the state"), level: str = Query(..., description="Level of the institution"), control: str = Query(..., description="Control type of the institution")):
    cursor.execute("SELECT COUNT(stateid) FROM state_sector_details WHERE state = ? AND level = ? AND control = ?", (state, level, control))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the state post for a specific state
@app.get("/v1/college_completion/state_post", operation_id="get_state_post", summary="Retrieves the distinct state post for a given state. This operation fetches the unique state post associated with the specified state from the state_sector_details table. The state parameter is used to filter the results.")
async def get_state_post(state: str = Query(..., description="Name of the state")):
    cursor.execute("SELECT T FROM ( SELECT DISTINCT CASE WHEN state = ? THEN state_post ELSE NULL END AS T FROM state_sector_details ) WHERE T IS NOT NULL", (state,))
    result = cursor.fetchall()
    if not result:
        return {"state_posts": []}
    return {"state_posts": [row[0] for row in result]}

# Endpoint to get the institution with the maximum student count in a specific state and level
@app.get("/v1/college_completion/max_student_count_institution", operation_id="get_max_student_count_institution", summary="Retrieves the name of the institution with the highest student count in a specified state and level. The state is identified by its abbreviation, and the level refers to the type of institution. This operation does not return any student-specific data.")
async def get_max_student_count_institution(state_abbr: str = Query(..., description="State abbreviation"), level: str = Query(..., description="Level of the institution")):
    cursor.execute("SELECT DISTINCT T1.chronname FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.state = T1.state WHERE T2.state_abbr = ? AND T1.level = ? AND T1.student_count = ( SELECT MAX(T1.student_count) FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.state = T1.state WHERE T2.state_abbr = ? AND T1.level = ? )", (state_abbr, level, state_abbr, level))
    result = cursor.fetchall()
    if not result:
        return {"institutions": []}
    return {"institutions": [row[0] for row in result]}

# Endpoint to get the site with the maximum latitude in a specific state
@app.get("/v1/college_completion/max_latitude_site", operation_id="get_max_latitude_site", summary="Retrieves the site with the highest latitude in a specified state. This operation requires the state abbreviation as input and returns the distinct site that matches the given state and has the maximum latitude value.")
async def get_max_latitude_site(state_abbr: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT DISTINCT T1.site FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.state = T1.state WHERE T2.state_abbr = ? AND T1.lat_y = ( SELECT MAX(T1.lat_y) FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.state = T1.state WHERE T2.state_abbr = ? )", (state_abbr, state_abbr))
    result = cursor.fetchall()
    if not result:
        return {"sites": []}
    return {"sites": [row[0] for row in result]}

# Endpoint to get the count of institutions with awards per value greater than awards per state value in a specific state, level, and control
@app.get("/v1/college_completion/count_institutions_awards_per_value", operation_id="get_count_institutions_awards_per_value", summary="Retrieves the number of unique institutions in a specified state, level, and control type that have a higher awards-per-value ratio than the state average. This operation helps identify institutions with exceptional performance in award distribution.")
async def get_count_institutions_awards_per_value(state_abbr: str = Query(..., description="State abbreviation"), level: str = Query(..., description="Level of the institution"), control: str = Query(..., description="Control type of the institution")):
    cursor.execute("SELECT COUNT(DISTINCT T1.chronname) FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.state = T1.state WHERE T2.state_abbr = ? AND T1.level = ? AND T1.control = ? AND T1.awards_per_value > T1.awards_per_state_value", (state_abbr, level, control))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of institutions with awards per value less than awards per national value in a specific state, level, and control
@app.get("/v1/college_completion/count_institutions_awards_per_natl_value", operation_id="get_count_institutions_awards_per_natl_value", summary="Retrieve the number of unique institutions in a specified state, level, and control type that have awarded less than the national average. The state is identified by its abbreviation, the level refers to the institution's level, and the control type indicates the institution's control type.")
async def get_count_institutions_awards_per_natl_value(state_abbr: str = Query(..., description="State abbreviation"), level: str = Query(..., description="Level of the institution"), control: str = Query(..., description="Control type of the institution")):
    cursor.execute("SELECT COUNT(DISTINCT T1.chronname) FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.state = T1.state WHERE T2.state_abbr = ? AND T2.level = ? AND T1.control = ? AND T1.awards_per_value < T1.awards_per_natl_value", (state_abbr, level, control))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the institution name with the lowest graduation value for a given state abbreviation, level, and control type
@app.get("/v1/college_completion/institution_lowest_grad_value", operation_id="get_institution_lowest_grad_value", summary="Retrieve the name of the institution with the lowest graduation rate for a specified state, institution level, and control type. The graduation rate is calculated based on the sum of graduation values for the given state, level, and control type. The result is ordered in ascending order and limited to the top 1 record.")
async def get_institution_lowest_grad_value(state_abbr: str = Query(..., description="State abbreviation"), level: str = Query(..., description="Institution level"), control: str = Query(..., description="Control type")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.state = T1.state WHERE T2.state_abbr = ? AND T1.level = ? AND T1.control = ? GROUP BY T1.chronname ORDER BY SUM(T1.grad_100_value) ASC LIMIT 1", (state_abbr, level, control))
    result = cursor.fetchone()
    if not result:
        return {"institution_name": []}
    return {"institution_name": result[0]}

# Endpoint to get the institution name and unit ID with the highest graduation value for a given state abbreviation, level, and control type
@app.get("/v1/college_completion/institution_highest_grad_value", operation_id="get_institution_highest_grad_value", summary="Retrieves the name and unique identifier of the institution with the highest graduation value in a specified state, based on the provided level and control type. The graduation value is calculated as the sum of graduation rates for the institution.")
async def get_institution_highest_grad_value(state_abbr: str = Query(..., description="State abbreviation"), level: str = Query(..., description="Institution level"), control: str = Query(..., description="Control type")):
    cursor.execute("SELECT T1.chronname, T1.unitid FROM institution_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.state = T1.state WHERE T2.state_abbr = ? AND T1.level = ? AND T1.control = ? GROUP BY T1.chronname ORDER BY SUM(T1.grad_150_value) DESC LIMIT 1", (state_abbr, level, control))
    result = cursor.fetchone()
    if not result:
        return {"institution_name": [], "unit_id": []}
    return {"institution_name": result[0], "unit_id": result[1]}

# Endpoint to get the sum of graduates for a given institution name, gender, and race
@app.get("/v1/college_completion/sum_graduates_by_institution_gender_race", operation_id="get_sum_graduates", summary="Retrieves the total number of graduates from a specific institution, filtered by gender and race. This operation calculates the sum of graduates based on the provided institution name, gender, and race.")
async def get_sum_graduates(institution_name: str = Query(..., description="Institution name"), gender: str = Query(..., description="Gender"), race: str = Query(..., description="Race")):
    cursor.execute("SELECT SUM(T2.grad_100) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname = ? AND T2.gender = ? AND T2.race = ?", (institution_name, gender, race))
    result = cursor.fetchone()
    if not result:
        return {"sum_graduates": []}
    return {"sum_graduates": result[0]}

# Endpoint to get the institution names with the maximum cohort size
@app.get("/v1/college_completion/institutions_max_cohort_size", operation_id="get_institutions_max_cohort_size", summary="Retrieves the names of institutions that have the largest cohort size. This operation identifies the maximum cohort size across all institutions and returns the names of institutions that match this size. The cohort size is determined by considering the details of each institution and their graduation data.")
async def get_institutions_max_cohort_size():
    cursor.execute("SELECT DISTINCT T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.cohort_size = ( SELECT MAX(T1.cohort_size) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid )")
    result = cursor.fetchall()
    if not result:
        return {"institution_names": []}
    return {"institution_names": [row[0] for row in result]}

# Endpoint to get the sum of graduates for a given state, year, level, control type, and race
@app.get("/v1/college_completion/sum_graduates_by_state_year_level_control_race", operation_id="get_sum_graduates_by_state", summary="Retrieves the total number of graduates for a specific state, year, institution level, control type, and race. This operation aggregates data from state-sector details and state-sector graduates, filtering results based on the provided parameters.")
async def get_sum_graduates_by_state(state: str = Query(..., description="State name"), year: int = Query(..., description="Year"), level: str = Query(..., description="Institution level"), control: str = Query(..., description="Control type"), race: str = Query(..., description="Race")):
    cursor.execute("SELECT SUM(T2.grad_cohort) FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T1.state = ? AND T2.year = ? AND T1.level = ? AND T1.control = ? AND T2.race = ?", (state, year, level, control, race))
    result = cursor.fetchone()
    if not result:
        return {"sum_graduates": []}
    return {"sum_graduates": result[0]}

# Endpoint to get the count of graduates for a given level, control type, gender, race, cohort, and schools count
@app.get("/v1/college_completion/count_graduates_by_level_control_gender_race_cohort_schools", operation_id="get_count_graduates", summary="Retrieves the total number of graduates based on specified institution level, control type, gender, race, cohort, and number of schools. This operation aggregates data from state sector details and graduation records, providing a comprehensive count of graduates that meet the given criteria.")
async def get_count_graduates(level: str = Query(..., description="Institution level"), control: str = Query(..., description="Control type"), gender: str = Query(..., description="Gender"), race: str = Query(..., description="Race"), cohort: str = Query(..., description="Cohort"), schools_count: int = Query(..., description="Number of schools")):
    cursor.execute("SELECT COUNT(T2.grad_cohort) FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T2.level = ? AND T2.control = ? AND T2.gender = ? AND T2.race = ? AND T2.cohort = ? AND T1.schools_count = ?", (level, control, gender, race, cohort, schools_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of institutions in a given state with a specific level and control type
@app.get("/v1/college_completion/percentage_institutions_by_state_level_control", operation_id="get_percentage_institutions", summary="Retrieves the percentage of institutions in a specified state that match a given level and control type. The calculation is based on the total number of distinct institutions in the state.")
async def get_percentage_institutions(institution_name: str = Query(..., description="Institution name"), level: str = Query(..., description="Institution level"), control: str = Query(..., description="Control type"), state: str = Query(..., description="State name")):
    cursor.execute("SELECT CAST(COUNT(DISTINCT CASE WHEN T1.state = ( SELECT T1.state FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 ON T2.state = T1.state WHERE T1.chronname = ? ) AND T1.level = ? AND T1.control = ? THEN T1.chronname ELSE NULL END) AS REAL) * 100 / COUNT(DISTINCT CASE WHEN T2.state = ? THEN T1.chronname ELSE NULL END) FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 ON T2.state = T1.state", (institution_name, level, control, state))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct states and institution names for a given race and year range
@app.get("/v1/college_completion/distinct_states_institutions_by_race_year_range", operation_id="get_distinct_states_institutions", summary="Retrieves a list of unique state and institution names that match the specified race and fall within the provided year range. The data is sourced from institution details and graduation records.")
async def get_distinct_states_institutions(race: str = Query(..., description="Race"), start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year")):
    cursor.execute("SELECT DISTINCT T1.state, T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.race = ? AND T2.year BETWEEN ? AND ?", (race, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"states_institutions": []}
    return {"states_institutions": [{"state": row[0], "institution_name": row[1]} for row in result]}

# Endpoint to get distinct states for a given year and awards per national value
@app.get("/v1/college_completion/distinct_states_by_year_awards_value", operation_id="get_distinct_states", summary="Retrieve a list of unique states that meet the specified criteria for a given year and awards per national value. The operation filters states based on the provided year and awards per national value, returning only those that match the input parameters.")
async def get_distinct_states(year: int = Query(..., description="Year"), awards_per_natl_value: int = Query(..., description="Awards per national value")):
    cursor.execute("SELECT DISTINCT T1.state FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T2.year = ? AND T1.awards_per_natl_value <= ?", (year, awards_per_natl_value))
    result = cursor.fetchall()
    if not result:
        return {"states": []}
    return {"states": [row[0] for row in result]}

# Endpoint to get distinct control and level of institutions with the maximum student count for a given race
@app.get("/v1/college_completion/distinct_control_level_max_student_count", operation_id="get_distinct_control_level_max_student_count", summary="Retrieves the unique control and level combinations of institutions that have the highest student count for a specified race. The data is sourced from institution details and graduation records, with the race filter applied to both.")
async def get_distinct_control_level_max_student_count(race: str = Query(..., description="Race of the students")):
    cursor.execute("SELECT DISTINCT T1.control, T1.level FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.race = ? AND T1.student_count = ( SELECT MAX(T1.student_count) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.race = ? )", (race, race))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct races of graduates from states with a given maximum number of schools and control type
@app.get("/v1/college_completion/distinct_races_schools_count_control", operation_id="get_distinct_races_schools_count_control", summary="Retrieve the unique racial identities of graduates from states where the number of schools does not exceed a specified limit and the control type matches the provided criteria. This operation is useful for analyzing diversity in educational institutions based on their size and management structure.")
async def get_distinct_races_schools_count_control(schools_count: int = Query(..., description="Maximum number of schools"), control: str = Query(..., description="Control type of the schools")):
    cursor.execute("SELECT DISTINCT T2.race FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T1.schools_count <= ? AND T1.control = ?", (schools_count, control))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct basic details and race of graduates for a given year, gender, and race
@app.get("/v1/college_completion/distinct_basic_race_year_gender", operation_id="get_distinct_basic_race_year_gender", summary="Retrieve unique combinations of basic institution details and race for graduates of a specific year, gender, and race. This operation filters graduates based on the provided year, gender, and race, and returns the distinct basic details and race of the corresponding institutions.")
async def get_distinct_basic_race_year_gender(year: int = Query(..., description="Year of graduation"), gender: str = Query(..., description="Gender of the graduates"), race: str = Query(..., description="Race of the graduates")):
    cursor.execute("SELECT DISTINCT T1.basic, T2.race FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.year = ? AND T2.gender = ? AND T2.race = ?", (year, gender, race))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of graduates of a specific race in a given state with a specific number of schools within a year range
@app.get("/v1/college_completion/count_race_schools_count_year_range_state", operation_id="get_count_race_schools_count_year_range_state", summary="Retrieve the total number of graduates of a specific race from schools in a given state, within a specified year range and a defined number of schools. This operation allows you to analyze graduation data based on race, state, and the number of schools, providing insights into educational trends and outcomes.")
async def get_count_race_schools_count_year_range_state(schools_count: int = Query(..., description="Number of schools"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), race: str = Query(..., description="Race of the graduates"), state: str = Query(..., description="State of the schools")):
    cursor.execute("SELECT COUNT(T2.race) FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T1.schools_count = ? AND T2.year BETWEEN ? AND ? AND T2.race = ? AND T1.state = ?", (schools_count, start_year, end_year, race, state))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct chronological names of institutions for a given race and graduation cohort range
@app.get("/v1/college_completion/distinct_chronname_grad_cohort_range_race", operation_id="get_distinct_chronname_grad_cohort_range_race", summary="Retrieve a unique list of institution names that fall within a specified graduation cohort range and race. This operation filters institutions based on the provided graduation cohort range and race, ensuring that only distinct institution names are returned.")
async def get_distinct_chronname_grad_cohort_range_race(start_cohort: int = Query(..., description="Start of the graduation cohort range"), end_cohort: int = Query(..., description="End of the graduation cohort range"), race: str = Query(..., description="Race of the graduates")):
    cursor.execute("SELECT DISTINCT T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.grad_cohort BETWEEN ? AND ? AND T2.race = ?", (start_cohort, end_cohort, race))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the sum of graduation cohorts for states with a specific pattern, awards per national value, and year
@app.get("/v1/college_completion/sum_grad_cohort_state_pattern_awards_year", operation_id="get_sum_grad_cohort_state_pattern_awards_year", summary="Retrieves the total number of graduation cohorts for states matching a specified pattern, with a particular awards-to-national-value ratio, and for a given year.")
async def get_sum_grad_cohort_state_pattern_awards_year(state_pattern: str = Query(..., description="Pattern for the state name (use % for wildcard)"), awards_per_natl_value: float = Query(..., description="Awards per national value"), year: int = Query(..., description="Year of graduation")):
    cursor.execute("SELECT SUM(T2.grad_cohort) FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T2.state LIKE ? AND T1.awards_per_natl_value = ? AND T2.year = ?", (state_pattern, awards_per_natl_value, year))
    result = cursor.fetchone()
    if not result:
        return {"sum": []}
    return {"sum": result[0]}

# Endpoint to get distinct sites of institutions with a student count within a given range and the maximum graduation year
@app.get("/v1/college_completion/distinct_sites_student_count_range_max_year", operation_id="get_distinct_sites_student_count_range_max_year", summary="Retrieves a list of unique institution sites that have a student count within the specified range and the latest graduation year. The range is defined by the minimum and maximum student count parameters.")
async def get_distinct_sites_student_count_range_max_year(min_student_count: int = Query(..., description="Minimum student count"), max_student_count: int = Query(..., description="Maximum student count")):
    cursor.execute("SELECT DISTINCT T1.site FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.student_count BETWEEN ? AND ? AND T2.year = ( SELECT MAX(T2.year) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid )", (min_student_count, max_student_count))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct states with specific gender, race, control type, and year
@app.get("/v1/college_completion/distinct_states_gender_race_control_year", operation_id="get_distinct_states_gender_race_control_year", summary="Retrieve a list of unique states that have graduates of a specific gender, race, school control type, and graduation year. This operation filters the data based on the provided parameters and returns the distinct states that meet the criteria.")
async def get_distinct_states_gender_race_control_year(gender: str = Query(..., description="Gender of the graduates"), race: str = Query(..., description="Race of the graduates"), control: str = Query(..., description="Control type of the schools"), year: int = Query(..., description="Year of graduation")):
    cursor.execute("SELECT DISTINCT T1.state FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T2.gender = ? AND T2.race = ? AND T1.control = ? AND T2.year = ?", (gender, race, control, year))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get distinct sites and chronological names of institutions for a given year, race, and graduation cohort range
@app.get("/v1/college_completion/distinct_sites_chronname_year_race_cohort_range", operation_id="get_distinct_sites_chronname_year_race_cohort_range", summary="Retrieve unique institution sites and their chronological names based on specific graduation year, race, and cohort range. This operation filters institution details by the provided year, race, and graduation cohort range, returning only distinct sites and corresponding chronological names.")
async def get_distinct_sites_chronname_year_race_cohort_range(year: int = Query(..., description="Year of graduation"), race: str = Query(..., description="Race of the graduates"), start_cohort: int = Query(..., description="Start of the graduation cohort range"), end_cohort: int = Query(..., description="End of the graduation cohort range")):
    cursor.execute("SELECT DISTINCT T1.site, T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.year = ? AND T2.race = ? AND T2.grad_cohort BETWEEN ? AND ?", (year, race, start_cohort, end_cohort))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of graduates of a specific gender within a range of schools count for a given year
@app.get("/v1/college_completion/count_race_gender_schools_count_range_year", operation_id="get_count_race_gender_schools_count_range_year", summary="Retrieves the total number of graduates of a specific gender from a range of schools, based on the provided minimum and maximum school count, for a given year. This operation provides insights into the graduation rates across different genders and school sizes.")
async def get_count_race_gender_schools_count_range_year(gender: str = Query(..., description="Gender of the graduates"), min_schools_count: int = Query(..., description="Minimum number of schools"), max_schools_count: int = Query(..., description="Maximum number of schools"), year: int = Query(..., description="Year of graduation")):
    cursor.execute("SELECT COUNT(T2.race) FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T2.gender = ? AND schools_count BETWEEN ? AND ? AND T2.year = ?", (gender, min_schools_count, max_schools_count, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct races from institutions with student count above a certain threshold in a specific state
@app.get("/v1/college_completion/distinct_races_above_student_count", operation_id="get_distinct_races", summary="Retrieve a list of unique racial categories from institutions in a specified state where the student population exceeds a certain threshold. The threshold is determined by multiplying the average student count of institutions in the state by a given multiplier.")
async def get_distinct_races(student_count_multiplier: float = Query(..., description="Multiplier for the average student count"), state: str = Query(..., description="State of the institution")):
    cursor.execute("SELECT DISTINCT T2.race FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.student_count > ( SELECT AVG(T1.student_count) * ? FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.state = ? ) AND T1.state = ?", (student_count_multiplier, state, state))
    result = cursor.fetchall()
    if not result:
        return {"races": []}
    return {"races": [row[0] for row in result]}

# Endpoint to get the percentage of 2-year level graduates in a specific state and year
@app.get("/v1/college_completion/percentage_2year_graduates", operation_id="get_percentage_2year_graduates", summary="Retrieves the percentage of graduates who completed a 2-year level education program in a specific state and year. The calculation is based on the total number of graduates in the given state and year. The level of education, state, and year of graduation are required as input parameters.")
async def get_percentage_2year_graduates(level: str = Query(..., description="Level of education"), state: str = Query(..., description="State of the institution"), year: int = Query(..., description="Year of graduation")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.level = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.level) FROM state_sector_details AS T1 INNER JOIN state_sector_grads AS T2 ON T2.stateid = T1.stateid WHERE T2.state = ? AND T2.year = ?", (level, state, year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the institution with the highest retain percentile in a specific state
@app.get("/v1/college_completion/highest_retain_percentile_institution", operation_id="get_highest_retain_percentile_institution", summary="Retrieves the name of the institution with the highest retain percentile in a specified state. The state is provided as an input parameter.")
async def get_highest_retain_percentile_institution(state: str = Query(..., description="State of the institution")):
    cursor.execute("SELECT chronname FROM institution_details WHERE state = ? AND retain_percentile = ( SELECT MAX(retain_percentile) FROM institution_details WHERE state = ? )", (state, state))
    result = cursor.fetchone()
    if not result:
        return {"institution": []}
    return {"institution": result[0]}

# Endpoint to get the site of the institution with the highest graduation cohort for a specific race, cohort, and year
@app.get("/v1/college_completion/highest_grad_cohort_site", operation_id="get_highest_grad_cohort_site", summary="Retrieves the site of the institution with the highest graduation cohort for a specific race, cohort, and year. The graduation cohort is determined by the provided race, cohort, and year. The result is ordered in descending order based on the graduation cohort and the top result is returned.")
async def get_highest_grad_cohort_site(race: str = Query(..., description="Race of the graduates"), cohort: str = Query(..., description="Cohort of the graduates"), year: int = Query(..., description="Year of graduation")):
    cursor.execute("SELECT T1.site FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.race = ? AND T2.cohort = ? AND T2.year = ? ORDER BY T2.grad_cohort DESC LIMIT 1", (race, cohort, year))
    result = cursor.fetchone()
    if not result:
        return {"site": []}
    return {"site": result[0]}

# Endpoint to get the year with the highest total graduation cohort for a specific institution
@app.get("/v1/college_completion/highest_grad_cohort_year", operation_id="get_highest_grad_cohort_year", summary="Retrieves the year with the highest total graduation cohort for a specific institution. The input parameter is used to identify the institution. The operation calculates the total graduation cohort for each year and returns the year with the highest total.")
async def get_highest_grad_cohort_year(chronname: str = Query(..., description="Name of the institution")):
    cursor.execute("SELECT T2.year FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname = ? GROUP BY T2.year ORDER BY SUM(T2.grad_cohort) DESC LIMIT 1", (chronname,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the institution with the minimum student count in a specific year, grouped by state
@app.get("/v1/college_completion/min_student_count_institution", operation_id="get_min_student_count_institution", summary="Retrieves the institution with the lowest student count for a given year, grouped by state. The result is ordered by the sum of state approval values in descending order, with the top institution returned.")
async def get_min_student_count_institution(year: int = Query(..., description="Year of the student count")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 ON T2.state = T1.state INNER JOIN institution_grads AS T3 ON T3.unitid = T1.unitid WHERE T1.student_count = ( SELECT MIN(T1.student_count) FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 ON T2.state = T1.state INNER JOIN institution_grads AS T3 ON T3.unitid = T1.unitid WHERE T3.year = ? ) AND T3.year = ? GROUP BY T1.state ORDER BY SUM(T2.state_appr_value) DESC LIMIT 1", (year, year))
    result = cursor.fetchone()
    if not result:
        return {"institution": []}
    return {"institution": result[0]}

# Endpoint to get the average graduation cohort for a specific institution, year range, race, and cohort
@app.get("/v1/college_completion/avg_grad_cohort", operation_id="get_avg_grad_cohort", summary="Retrieves the average graduation cohort for a specific institution within a given year range, race, and cohort. This operation calculates the average graduation cohort based on the provided institution, year range, race, and cohort parameters.")
async def get_avg_grad_cohort(chronname: str = Query(..., description="Name of the institution"), start_year: int = Query(..., description="Start year of the range"), end_year: int = Query(..., description="End year of the range"), race: str = Query(..., description="Race of the graduates"), cohort: str = Query(..., description="Cohort of the graduates")):
    cursor.execute("SELECT AVG(T2.grad_cohort) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname = ? AND T2.year BETWEEN ? AND ? AND T2.race = ? AND T2.cohort = ?", (chronname, start_year, end_year, race, cohort))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the institution with the highest graduation cohort for a specific race among a list of institutions
@app.get("/v1/college_completion/highest_grad_cohort_institution", operation_id="get_highest_grad_cohort_institution", summary="Retrieves the institution with the highest graduation cohort for a specific race among a list of up to eight institutions. The operation considers the total sum of graduation cohorts for the specified race at each institution and returns the institution with the highest sum.")
async def get_highest_grad_cohort_institution(institution1: str = Query(..., description="Name of the first institution"), institution2: str = Query(..., description="Name of the second institution"), institution3: str = Query(..., description="Name of the third institution"), institution4: str = Query(..., description="Name of the fourth institution"), institution5: str = Query(..., description="Name of the fifth institution"), institution6: str = Query(..., description="Name of the sixth institution"), institution7: str = Query(..., description="Name of the seventh institution"), institution8: str = Query(..., description="Name of the eighth institution"), race: str = Query(..., description="Race of the graduates")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname IN (?, ?, ?, ?, ?, ?, ?, ?) AND T2.race = ? GROUP BY T1.chronname ORDER BY SUM(T2.grad_cohort) DESC LIMIT 1", (institution1, institution2, institution3, institution4, institution5, institution6, institution7, institution8, race))
    result = cursor.fetchone()
    if not result:
        return {"institution": []}
    return {"institution": result[0]}

# Endpoint to get institutions based on level, control, and state
@app.get("/v1/college_completion/institutions_by_level_control_state", operation_id="get_institutions_by_level_control_state", summary="Retrieves a list of institutions that match the specified level of education, control type, and state. The level parameter filters institutions based on their educational level, the control parameter filters based on the type of control (e.g., public, private), and the state parameter filters based on the institution's location. The endpoint returns the names of the institutions that meet all the specified criteria.")
async def get_institutions_by_level_control_state(level: str = Query(..., description="Level of education"), control: str = Query(..., description="Control type"), state: str = Query(..., description="State of the institution")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 WHERE T2.level = ? AND T2.control = ? AND T2.state = ?", (level, control, state))
    result = cursor.fetchall()
    if not result:
        return {"institutions": []}
    return {"institutions": [row[0] for row in result]}

# Endpoint to get the state with the lowest total state approval value for a list of institutions
@app.get("/v1/college_completion/lowest_state_appr_value_state", operation_id="get_lowest_state_appr_value_state", summary="Retrieve the state with the lowest cumulative state approval value for a specified list of up to eight institutions. The state approval value is calculated by aggregating the state_appr_value for each institution within a state. The result is determined by grouping institutions by state and ordering the sum of state_appr_values in ascending order, with the lowest value being returned.")
async def get_lowest_state_appr_value_state(institution1: str = Query(..., description="Name of the first institution"), institution2: str = Query(..., description="Name of the second institution"), institution3: str = Query(..., description="Name of the third institution"), institution4: str = Query(..., description="Name of the fourth institution"), institution5: str = Query(..., description="Name of the fifth institution"), institution6: str = Query(..., description="Name of the sixth institution"), institution7: str = Query(..., description="Name of the seventh institution"), institution8: str = Query(..., description="Name of the eighth institution")):
    cursor.execute("SELECT T1.state FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 ON T2.state = T1.state WHERE T1.chronname IN (?, ?, ?, ?, ?, ?, ?, ?) GROUP BY T1.state ORDER BY SUM(T2.state_appr_value) ASC LIMIT 1", (institution1, institution2, institution3, institution4, institution5, institution6, institution7, institution8))
    result = cursor.fetchone()
    if not result:
        return {"state": []}
    return {"state": result[0]}

# Endpoint to get the count of institutions and their names with a full-time percentage greater than a specified value
@app.get("/v1/college_completion/institution_count_ft_pct", operation_id="get_institution_count_ft_pct", summary="Retrieves the number of institutions and their names where the full-time student percentage surpasses a given threshold. The results are ordered by the descending count of schools in the respective state, with the top result being returned.")
async def get_institution_count_ft_pct(ft_pct: int = Query(..., description="Full-time percentage threshold")):
    cursor.execute("SELECT COUNT(t1.unitid), t1.chronname FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 ON t1.state = t2.state WHERE t1.ft_pct > ? ORDER BY t2.schools_count DESC LIMIT 1", (ft_pct,))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "chronname": []}
    return {"count": result[0], "chronname": result[1]}

# Endpoint to get the average median SAT value of institutions ordered by state approval value
@app.get("/v1/college_completion/avg_med_sat_value", operation_id="get_avg_med_sat_value", summary="Retrieves the average median SAT value of institutions, ordered by the state approval value. The result represents a single value that provides an overview of the median SAT scores across institutions, with a focus on states with higher approval values.")
async def get_avg_med_sat_value():
    cursor.execute("SELECT AVG(t1.med_sat_value) FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 ON t1.state = t2.state ORDER BY t2.state_appr_value LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"avg_med_sat_value": []}
    return {"avg_med_sat_value": result[0]}

# Endpoint to get the institution name based on gender and cohort
@app.get("/v1/college_completion/institution_name_gender_cohort", operation_id="get_institution_name_gender_cohort", summary="Retrieves the most recent name of the institution where graduates of a specific gender and cohort type were enrolled. The name is obtained from the institution details table, filtered by the gender and cohort of the graduates from the institution graduates table. The result is sorted by the graduation cohort in descending order and limited to one record.")
async def get_institution_name_gender_cohort(gender: str = Query(..., description="Gender of the graduates"), cohort: str = Query(..., description="Cohort type")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T2.gender = ? AND T2.cohort = ? ORDER BY T2.grad_cohort DESC LIMIT 1", (gender, cohort))
    result = cursor.fetchone()
    if not result:
        return {"chronname": []}
    return {"chronname": result[0]}

# Endpoint to get the institution name based on specific criteria
@app.get("/v1/college_completion/institution_name_specific_criteria", operation_id="get_institution_name_specific_criteria", summary="Retrieves the name of the institution that meets the specified criteria, including a list of potential institution names, a year, a race, and a cohort type. The result is ordered by the graduation cohort in descending order and limited to one entry.")
async def get_institution_name_specific_criteria(chronname1: str = Query(..., description="Institution name 1"), chronname2: str = Query(..., description="Institution name 2"), chronname3: str = Query(..., description="Institution name 3"), chronname4: str = Query(..., description="Institution name 4"), chronname5: str = Query(..., description="Institution name 5"), chronname6: str = Query(..., description="Institution name 6"), chronname7: str = Query(..., description="Institution name 7"), chronname8: str = Query(..., description="Institution name 8"), year: int = Query(..., description="Year"), race: str = Query(..., description="Race"), cohort: str = Query(..., description="Cohort type")):
    cursor.execute("SELECT T1.chronname FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname IN (?, ?, ?, ?, ?, ?, ?, ?) AND T2.year = ? AND T2.race = ? AND T2.cohort = ? ORDER BY T2.grad_cohort DESC LIMIT 1", (chronname1, chronname2, chronname3, chronname4, chronname5, chronname6, chronname7, chronname8, year, race, cohort))
    result = cursor.fetchone()
    if not result:
        return {"chronname": []}
    return {"chronname": result[0]}

# Endpoint to get the average graduation rate within 150% of normal time for a specific institution and criteria
@app.get("/v1/college_completion/avg_grad_150", operation_id="get_avg_grad_150", summary="Retrieves the average graduation rate within 150% of normal time for a specific institution, considering a range of years and demographic criteria. The operation factors in the institution's name, the start and end years of the period, the gender, and the race of the graduates.")
async def get_avg_grad_150(chronname: str = Query(..., description="Institution name"), start_year: int = Query(..., description="Start year"), end_year: int = Query(..., description="End year"), gender: str = Query(..., description="Gender of the graduates"), race: str = Query(..., description="Race of the graduates")):
    cursor.execute("SELECT AVG(T2.grad_150) FROM institution_details AS T1 INNER JOIN institution_grads AS T2 ON T2.unitid = T1.unitid WHERE T1.chronname = ? AND T2.year BETWEEN ? AND ? AND T2.gender = ? AND T2.race = ?", (chronname, start_year, end_year, gender, race))
    result = cursor.fetchone()
    if not result:
        return {"avg_grad_150": []}
    return {"avg_grad_150": result[0]}

# Endpoint to get the institution name and state approval value based on awards per value difference
@app.get("/v1/college_completion/institution_state_appr_value", operation_id="get_institution_state_appr_value", summary="Retrieves the name and state approval value of the institution with the highest difference between its awards per value and the national average awards per value. The data is sorted in descending order based on this difference.")
async def get_institution_state_appr_value():
    cursor.execute("SELECT T1.chronname, T2.state_appr_value FROM institution_details AS T1 INNER JOIN state_sector_details AS T2 ON T2.state = T1.state ORDER BY T1.awards_per_value - T2.awards_per_natl_value DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"chronname": [], "state_appr_value": []}
    return {"chronname": result[0], "state_appr_value": result[1]}

api_calls = [
    "/v1/college_completion/most_common_institution_name?year=2007&race=Ai",
    "/v1/college_completion/institution_details_by_year_cohort?year=2011&grad_cohort=209",
    "/v1/college_completion/count_grad_cohorts_by_year_range_institution_gender?start_year=2011&end_year=2013&chronname=Gateway%20Community%20College&gender=F",
    "/v1/college_completion/count_grad_cohorts_by_year_gender_aid_value?year=2012&gender=M",
    "/v1/college_completion/avg_med_sat_value_by_year_gender?year=2013&gender=M",
    "/v1/college_completion/state_institution_by_year_control?year=2012&control=Private%20for-profit",
    "/v1/college_completion/institution_name_by_year_control?year=2013&control=Public",
    "/v1/college_completion/institution_details_by_year_gender_race_cohort?year=2013&gender=B&race=X&max_grad_cohort=200",
    "/v1/college_completion/count_graduates_by_year_gender_race_institution?year=2013&gender=F&race=X&chronname=Oakwood%20University",
    "/v1/college_completion/count_graduates_by_gender_race_institution_cohort?gender=F&race=A&chronname=University%20of%20Alaska%20at%20Anchorage&cohort=4y%20other",
    "/v1/college_completion/sum_grad_cohorts_by_year_gender_race_institution?year1=2011&year2=2012&year3=2013&gender=B&race=X&institution_name=Auburn%20University",
    "/v1/college_completion/percentage_graduates_by_race_control?race=B&control_type=Private%20for-profit",
    "/v1/college_completion/percentage_graduates_by_race_state_year?race=A&year=2013&state=Alabama",
    "/v1/college_completion/grad_cohort_ratio_by_gender_institution_year_race?gender1=M&gender2=F&institution_name=Harvard%20University&year=2013&race=A",
    "/v1/college_completion/top_institution_by_grad_value?institution_name1=Amridge%20University&institution_name2=Auburn%20University",
    "/v1/college_completion/count_institutions_by_control?control_type=Private%20not-for-profit",
    "/v1/college_completion/sum_grad_cohorts_by_institution_gender_race?institution_name=Amridge%20University&gender=M&race=H",
    "/v1/college_completion/count_institutions_by_name_year_cohort?institution_name=Lincoln%20College&year=2011&cohort=4y%20bach",
    "/v1/college_completion/sum_grad_rates_by_institution_year_gender_race?institution_name=Central%20Alabama%20Community%20College&year=2011&gender=M&race=H",
    "/v1/college_completion/grad_cohorts_by_institution_year?institution_name=Central%20Alabama%20Community%20College&year=2011",
    "/v1/college_completion/graduation_rate_comparison?chronname=Central%20Alabama%20Community%20College&year=2011&race=W",
    "/v1/college_completion/highest_graduation_rate_institution?year=2011&gender=M&race=W",
    "/v1/college_completion/institutions_above_graduation_threshold?grad_threshold=20&year=2011&gender=M&race=W",
    "/v1/college_completion/total_graduation_cohort?cohort=2y%20all&year=2011&state=Alabama",
    "/v1/college_completion/graduation_cohort_difference?year1=2012&year2=2011&chronname=Central%20Alabama%20Community%20College",
    "/v1/college_completion/count_distinct_institutions?state=Alabama&med_sat_percentile=100&year=2011&grad_cohort_threshold=500",
    "/v1/college_completion/count_institutions_by_control_type?state=Alabama&control=Public&year=2011&grad_100_threshold=30",
    "/v1/college_completion/distinct_institutions_by_race?state=Alabama&year=2011&race=X&grad_cohort_threshold=500",
    "/v1/college_completion/average_graduation_cohort?chronname=Central%20Alabama%20Community%20College&year1=2011&year2=2012&year3=2013&gender=B&race=X",
    "/v1/college_completion/average_graduation_rate?chronname=Central%20Alabama%20Community%20College",
    "/v1/college_completion/institution_site?chronname=Swarthmore%20College",
    "/v1/college_completion/institution_state?chronname=Mercer%20University",
    "/v1/college_completion/institution_city?chronname=Rensselaer%20Polytechnic%20Institute",
    "/v1/college_completion/state_abbreviation?state=Delaware",
    "/v1/college_completion/count_state_ids?state=California&level=2-year&control=Public",
    "/v1/college_completion/state_post?state=Idaho",
    "/v1/college_completion/max_student_count_institution?state_abbr=NJ&level=4-year",
    "/v1/college_completion/max_latitude_site?state_abbr=PA",
    "/v1/college_completion/count_institutions_awards_per_value?state_abbr=UT&level=4-year&control=Public",
    "/v1/college_completion/count_institutions_awards_per_natl_value?state_abbr=CT&level=2-year&control=Private%20not-for-profit",
    "/v1/college_completion/institution_lowest_grad_value?state_abbr=ID&level=4-year&control=Public",
    "/v1/college_completion/institution_highest_grad_value?state_abbr=KY&level=4-year&control=Private%20for-profit",
    "/v1/college_completion/sum_graduates_by_institution_gender_race?institution_name=Pennsylvania%20State%20University-Altoona&gender=F&race=H",
    "/v1/college_completion/institutions_max_cohort_size",
    "/v1/college_completion/sum_graduates_by_state_year_level_control_race?state=Alabama&year=2011&level=2-year&control=Public&race=X",
    "/v1/college_completion/count_graduates_by_level_control_gender_race_cohort_schools?level=2-year&control=Public&gender=B&race=A&cohort=2y%20all&schools_count=113",
    "/v1/college_completion/percentage_institutions_by_state_level_control?institution_name=Madison%20Area%20Technical%20College&level=4-year&control=Public&state=Alabama",
    "/v1/college_completion/distinct_states_institutions_by_race_year_range?race=B&start_year=2010&end_year=2012",
    "/v1/college_completion/distinct_states_by_year_awards_value?year=2011&awards_per_natl_value=20",
    "/v1/college_completion/distinct_control_level_max_student_count?race=X",
    "/v1/college_completion/distinct_races_schools_count_control?schools_count=20&control=Public",
    "/v1/college_completion/distinct_basic_race_year_gender?year=2012&gender=M&race=X",
    "/v1/college_completion/count_race_schools_count_year_range_state?schools_count=1&start_year=2011&end_year=2013&race=W&state=Alaska",
    "/v1/college_completion/distinct_chronname_grad_cohort_range_race?start_cohort=1&end_cohort=3&race=Ai",
    "/v1/college_completion/sum_grad_cohort_state_pattern_awards_year?state_pattern=A%25&awards_per_natl_value=16.5&year=2012",
    "/v1/college_completion/distinct_sites_student_count_range_max_year?min_student_count=500&max_student_count=1000",
    "/v1/college_completion/distinct_states_gender_race_control_year?gender=M&race=B&control=Private%20for-profit&year=2011",
    "/v1/college_completion/distinct_sites_chronname_year_race_cohort_range?year=2011&race=B&start_cohort=20&end_cohort=30",
    "/v1/college_completion/count_race_gender_schools_count_range_year?gender=F&min_schools_count=10&max_schools_count=20&year=2012",
    "/v1/college_completion/distinct_races_above_student_count?student_count_multiplier=0.9&state=Alabama",
    "/v1/college_completion/percentage_2year_graduates?level=2-year&state=Hawaii&year=2010",
    "/v1/college_completion/highest_retain_percentile_institution?state=Connecticut",
    "/v1/college_completion/highest_grad_cohort_site?race=W&cohort=2y%20all&year=2008",
    "/v1/college_completion/highest_grad_cohort_year?chronname=Harvard%20University",
    "/v1/college_completion/min_student_count_institution?year=2010",
    "/v1/college_completion/avg_grad_cohort?chronname=Yale%20University&start_year=2002&end_year=2005&race=B&cohort=4y%20bach",
    "/v1/college_completion/highest_grad_cohort_institution?institution1=Brown%20University&institution2=Columbia%20University&institution3=Cornell%20University&institution4=Dartmouth%20College&institution5=Harvard%20University&institution6=Princeton%20University&institution7=University%20of%20Pennsylvania&institution8=Yale%20University&race=H",
    "/v1/college_completion/institutions_by_level_control_state?level=4-year&control=Public&state=Florida",
    "/v1/college_completion/lowest_state_appr_value_state?institution1=Brown%20University&institution2=Columbia%20University&institution3=Cornell%20University&institution4=Dartmouth%20College&institution5=Harvard%20University&institution6=Princeton%20University&institution7=University%20of%20Pennsylvania&institution8=Yale%20University",
    "/v1/college_completion/institution_count_ft_pct?ft_pct=90",
    "/v1/college_completion/avg_med_sat_value",
    "/v1/college_completion/institution_name_gender_cohort?gender=F&cohort=4y%20other",
    "/v1/college_completion/institution_name_specific_criteria?chronname1=Brown%20University&chronname2=Columbia%20University&chronname3=Cornell%20University&chronname4=Dartmouth%20College&chronname5=Harvard%20University&chronname6=Princeton%20University&chronname7=University%20of%20Pennsylvania&chronname8=Yale%20University&year=2013&race=B&cohort=4y%20bach",
    "/v1/college_completion/avg_grad_150?chronname=United%20Education%20Institute-Huntington%20Park%20Campus&start_year=2011&end_year=2013&gender=M&race=H",
    "/v1/college_completion/institution_state_appr_value"
]
