from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/california_schools/california_schools.sqlite')
cursor = conn.cursor()

# Endpoint to get the highest free meal ratio in a given county
@app.get("/v1/california_schools/highest_free_meal_ratio_by_county", operation_id="get_highest_free_meal_ratio", summary="Retrieves the highest ratio of free meals (K-12) to total enrollment in a specified county. The operation returns the single highest ratio, calculated by dividing the number of students receiving free meals by the total enrollment in the county.")
async def get_highest_free_meal_ratio(county_name: str = Query(..., description="Name of the county")):
    cursor.execute("SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = ? ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1", (county_name,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the lowest free meal ratios for a given educational option type
@app.get("/v1/california_schools/lowest_free_meal_ratios_by_educational_option", operation_id="get_lowest_free_meal_ratios", summary="Retrieve the three lowest ratios of free meals to enrollment for students aged 5-17 in California schools, filtered by a specified educational option type.")
async def get_lowest_free_meal_ratios(educational_option_type: str = Query(..., description="Type of educational option")):
    cursor.execute("SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = ? AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3", (educational_option_type,))
    result = cursor.fetchall()
    if not result:
        return {"ratios": []}
    return {"ratios": [row[0] for row in result]}

# Endpoint to get the zip codes of schools in a given district with a specific charter school status
@app.get("/v1/california_schools/zip_codes_by_district_and_charter_status", operation_id="get_zip_codes", summary="Retrieves the zip codes of schools in a specific district based on their charter school status. The district is identified by its name, and the charter school status is determined by a binary value (1 for Yes, 0 for No).")
async def get_zip_codes(district_name: str = Query(..., description="Name of the district"), charter_school_status: int = Query(..., description="Charter school status (1 for Yes, 0 for No)")):
    cursor.execute("SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = ? AND T1.`Charter School (Y/N)` = ?", (district_name, charter_school_status))
    result = cursor.fetchall()
    if not result:
        return {"zip_codes": []}
    return {"zip_codes": [row[0] for row in result]}

# Endpoint to get the mailing street of the school with the highest FRPM count (K-12)
@app.get("/v1/california_schools/mailing_street_highest_frpm_count", operation_id="get_mailing_street_highest_frpm_count", summary="Retrieves the mailing street of the school with the highest number of students eligible for free or reduced-price meals (FRPM) in California. The data is obtained by joining the 'frpm' and 'schools' tables using the CDSCode, and then ordering the results by the FRPM count in descending order. The mailing street of the top-ranked school is returned.")
async def get_mailing_street_highest_frpm_count():
    cursor.execute("SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"mailing_street": []}
    return {"mailing_street": result[0]}

# Endpoint to get the phone numbers of charter schools with a specific funding type and open date
@app.get("/v1/california_schools/phone_numbers_charter_schools", operation_id="get_phone_numbers_charter_schools", summary="Retrieve the phone numbers of charter schools that meet the specified funding type, status, and open date criteria. The funding type and status are used to filter the results, while the open date is used to ensure the schools are operational as of the provided date.")
async def get_phone_numbers_charter_schools(charter_funding_type: str = Query(..., description="Charter funding type"), charter_school_status: int = Query(..., description="Charter school status (1 for Yes, 0 for No)"), open_date: str = Query(..., description="Open date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Charter Funding Type` = ? AND T1.`Charter School (Y/N)` = ? AND T2.OpenDate > ?", (charter_funding_type, charter_school_status, open_date))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": [row[0] for row in result]}

# Endpoint to get the count of distinct schools with a specific virtual status and average math score
@app.get("/v1/california_schools/count_distinct_schools_virtual_math_score", operation_id="get_count_distinct_schools", summary="Retrieve the number of unique schools in California that have a specified virtual status and an average math score surpassing a given value. This operation considers the virtual status of the school and the average math score of students in the school.")
async def get_count_distinct_schools(virtual_status: str = Query(..., description="Virtual status ('F' for False, 'T' for True)"), avg_math_score: int = Query(..., description="Average math score")):
    cursor.execute("SELECT COUNT(DISTINCT T2.School) FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.Virtual = ? AND T1.AvgScrMath > ?", (virtual_status, avg_math_score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the names of magnet schools with a specific number of test takers
@app.get("/v1/california_schools/magnet_schools_test_takers", operation_id="get_magnet_schools", summary="Retrieve the names of magnet schools in California that have more than a specified number of test takers. The operation filters schools based on their magnet status and the number of test takers, providing a targeted list of schools that meet the specified criteria.")
async def get_magnet_schools(magnet_status: int = Query(..., description="Magnet status (1 for Yes, 0 for No)"), num_test_takers: int = Query(..., description="Number of test takers")):
    cursor.execute("SELECT T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.Magnet = ? AND T1.NumTstTakr > ?", (magnet_status, num_test_takers))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [row[0] for row in result]}

# Endpoint to get the phone number of the school with the highest number of students scoring 1500 or above
@app.get("/v1/california_schools/phone_highest_ge1500_scorers", operation_id="get_phone_highest_ge1500_scorers", summary="Retrieves the phone number of the school with the highest count of students who scored 1500 or above in the SAT exam. The data is obtained by joining the 'satscores' and 'schools' tables on the 'cds' and 'CDSCode' fields, respectively. The results are then ordered in descending order based on the number of students who scored 1500 or above, and the phone number of the top-ranked school is returned.")
async def get_phone_highest_ge1500_scorers():
    cursor.execute("SELECT T2.Phone FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the number of test takers for the school with the highest FRPM count (K-12)
@app.get("/v1/california_schools/test_takers_highest_frpm_count", operation_id="get_test_takers_highest_frpm_count", summary="Retrieves the total number of test takers from the school with the highest FRPM count (K-12). This operation identifies the school with the highest FRPM count and returns the corresponding number of test takers.")
async def get_test_takers_highest_frpm_count():
    cursor.execute("SELECT NumTstTakr FROM satscores WHERE cds = ( SELECT CDSCode FROM frpm ORDER BY `FRPM Count (K-12)` DESC LIMIT 1 )")
    result = cursor.fetchone()
    if not result:
        return {"test_takers": []}
    return {"test_takers": result[0]}

# Endpoint to get the count of school codes with a specific average math score and charter funding type
@app.get("/v1/california_schools/count_school_codes_math_score_charter_funding", operation_id="get_count_school_codes", summary="Retrieves the number of schools in California with an average math score exceeding a specified value and a particular charter funding type.")
async def get_count_school_codes(avg_math_score: int = Query(..., description="Average math score"), charter_funding_type: str = Query(..., description="Charter funding type")):
    cursor.execute("SELECT COUNT(T2.`School Code`) FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrMath > ? AND T2.`Charter Funding Type` = ?", (avg_math_score, charter_funding_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the FRPM count for the school with the highest average reading score
@app.get("/v1/california_schools/frpm_count_highest_avg_reading_score", operation_id="get_frpm_count_highest_avg_reading_score", summary="Retrieves the number of students receiving free or reduced-price meals (FRPM) in the school with the highest average reading score. The data is obtained by joining the 'satscores' and 'frpm' tables using the common 'cds' field and sorting the results by the average reading score in descending order. The operation returns the FRPM count of the top-ranked school.")
async def get_frpm_count_highest_avg_reading_score():
    cursor.execute("SELECT T2.`FRPM Count (Ages 5-17)` FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.AvgScrRead DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"frpm_count": []}
    return {"frpm_count": result[0]}

# Endpoint to get CDS codes of schools with total enrollment greater than a specified number
@app.get("/v1/california_schools/cds_codes_total_enrollment", operation_id="get_cds_codes_total_enrollment", summary="Retrieve the CDS codes of California schools with a total enrollment surpassing a specified threshold. The operation considers both the K-12 and 5-17 age group enrollments to determine the total enrollment.")
async def get_cds_codes_total_enrollment(total_enrollment: int = Query(..., description="Total enrollment threshold")):
    cursor.execute("SELECT T2.CDSCode FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Enrollment (K-12)` + T2.`Enrollment (Ages 5-17)` > ?", (total_enrollment,))
    result = cursor.fetchall()
    if not result:
        return {"cds_codes": []}
    return {"cds_codes": [row[0] for row in result]}

# Endpoint to get the maximum free meal ratio for schools with a specified SAT score ratio
@app.get("/v1/california_schools/max_free_meal_ratio", operation_id="get_max_free_meal_ratio", summary="Retrieves the highest free meal ratio among schools where the ratio of students scoring above 1500 on the SAT to the total number of test takers surpasses the provided threshold.")
async def get_max_free_meal_ratio(sat_score_ratio: float = Query(..., description="SAT score ratio threshold")):
    cursor.execute("SELECT MAX(CAST(T1.`Free Meal Count (Ages 5-17)` AS REAL) / T1.`Enrollment (Ages 5-17)`) FROM frpm AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE CAST(T2.NumGE1500 AS REAL) / T2.NumTstTakr > ?", (sat_score_ratio,))
    result = cursor.fetchone()
    if not result:
        return {"max_free_meal_ratio": []}
    return {"max_free_meal_ratio": result[0]}

# Endpoint to get phone numbers of schools with the highest SAT score ratios
@app.get("/v1/california_schools/phone_numbers_highest_sat_ratios", operation_id="get_phone_numbers_highest_sat_ratios", summary="Retrieves the phone numbers of the top schools in California with the highest SAT score ratios. The SAT score ratio is calculated by dividing the number of students scoring 1500 or above by the total number of test takers. The number of schools to return can be specified using the 'limit' input parameter.")
async def get_phone_numbers_highest_sat_ratios(limit: int = Query(..., description="Number of top schools to return")):
    cursor.execute("SELECT T1.Phone FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds ORDER BY CAST(T2.NumGE1500 AS REAL) / T2.NumTstTakr DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"phone_numbers": []}
    return {"phone_numbers": [row[0] for row in result]}

# Endpoint to get NCES school codes of schools with the highest enrollment
@app.get("/v1/california_schools/nces_school_codes_highest_enrollment", operation_id="get_nces_school_codes_highest_enrollment", summary="Retrieves the NCES school codes of the top schools in California with the highest enrollment, as determined by the number of students aged 5-17. The number of schools returned can be specified using the 'limit' input parameter.")
async def get_nces_school_codes_highest_enrollment(limit: int = Query(..., description="Number of top schools to return")):
    cursor.execute("SELECT T1.NCESSchool FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T2.`Enrollment (Ages 5-17)` DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"nces_school_codes": []}
    return {"nces_school_codes": [row[0] for row in result]}

# Endpoint to get the district of the school with the highest average reading score and a specified status type
@app.get("/v1/california_schools/district_highest_avg_reading_score", operation_id="get_district_highest_avg_reading_score", summary="Retrieves the district of the school with the highest average reading score among schools with the specified status type.")
async def get_district_highest_avg_reading_score(status_type: str = Query(..., description="Status type of the school")):
    cursor.execute("SELECT T1.District FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.StatusType = ? ORDER BY T2.AvgScrRead DESC LIMIT 1", (status_type,))
    result = cursor.fetchone()
    if not result:
        return {"district": []}
    return {"district": result[0]}

# Endpoint to get the number of schools (by CDS code) in a given county with a given status type and fewer than a specified number of SAT test takers
@app.get("/v1/california_schools/count_cds_codes_status_test_takers_county", operation_id="get_count_cds_codes_status_test_takers_county", summary="Retrieves the count of schools in a specific county that match a given status type and have fewer than a specified number of SAT test takers. This operation is useful for analyzing the distribution of schools based on their status and the number of SAT test takers within a county.")
async def get_count_cds_codes_status_test_takers_county(status_type: str = Query(..., description="Status type"), num_test_takers: int = Query(..., description="Number of test takers"), county: str = Query(..., description="County")):
    cursor.execute("SELECT COUNT(T1.CDSCode) FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.StatusType = ? AND T2.NumTstTakr < ? AND T1.County = ?", (status_type, num_test_takers, county))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get charter numbers, average writing scores, and their ranks for schools with average writing scores above a specified threshold
@app.get("/v1/california_schools/charter_numbers_avg_writing_scores", operation_id="get_charter_numbers_avg_writing_scores", summary="Retrieves the charter numbers, average writing scores, and their corresponding ranks for California schools with average writing scores surpassing a specified threshold. The threshold is provided as an input parameter.")
async def get_charter_numbers_avg_writing_scores(avg_writing_score: int = Query(..., description="Average writing score threshold")):
    cursor.execute("SELECT CharterNum, AvgScrWrite, RANK() OVER (ORDER BY AvgScrWrite DESC) AS WritingScoreRank FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.AvgScrWrite > ? AND CharterNum is not null", (avg_writing_score,))
    result = cursor.fetchall()
    if not result:
        return {"charter_numbers": []}
    return {"charter_numbers": [{"CharterNum": row[0], "AvgScrWrite": row[1], "WritingScoreRank": row[2]} for row in result]}

# Endpoint to get the count of schools with a specified charter funding type, county, and number of test takers
@app.get("/v1/california_schools/count_cds_codes_charter_funding_county_test_takers", operation_id="get_count_cds_codes_charter_funding_county_test_takers", summary="Retrieve the count of schools in a specific county with a given charter funding type and a maximum number of test takers. This operation provides a quantitative overview of schools based on their charter funding, county, and test taker population.")
async def get_count_cds_codes_charter_funding_county_test_takers(charter_funding_type: str = Query(..., description="Charter funding type"), county_name: str = Query(..., description="County name"), num_test_takers: int = Query(..., description="Number of test takers")):
    cursor.execute("SELECT COUNT(T1.CDSCode) FROM frpm AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.`Charter Funding Type` = ? AND T1.`County Name` = ? AND T2.NumTstTakr <= ?", (charter_funding_type, county_name, num_test_takers))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the phone number of the school with the highest average math score
@app.get("/v1/california_schools/phone_number_highest_avg_math_score", operation_id="get_phone_number_highest_avg_math_score", summary="Retrieves the phone number of the school in California with the highest average math score. The operation uses the provided data to determine the school with the highest average math score and returns its phone number.")
async def get_phone_number_highest_avg_math_score():
    cursor.execute("SELECT T1.Phone FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds ORDER BY T2.AvgScrMath DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"phone_number": []}
    return {"phone_number": result[0]}

# Endpoint to get the count of schools in a specific county with specific low and high grades
@app.get("/v1/california_schools/count_schools_by_county_grades", operation_id="get_count_schools_by_county_grades", summary="Retrieves the total number of schools in a specified county that cater to students within a defined grade range. The operation filters schools based on the provided county name, lowest grade, and highest grade.")
async def get_count_schools_by_county_grades(county: str = Query(..., description="County name"), low_grade: int = Query(..., description="Low grade"), high_grade: int = Query(..., description="High grade")):
    cursor.execute("SELECT COUNT(T1.`School Name`) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = ? AND T1.`Low Grade` = ? AND T1.`High Grade` = ?", (county, low_grade, high_grade))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of CDS codes in a specific county with specific free meal and FRPM counts
@app.get("/v1/california_schools/count_cds_codes_by_county_meal_counts", operation_id="get_count_cds_codes_by_county_meal_counts", summary="Retrieve the total number of California Department of Education (CDE) codes in a specified county that meet the given criteria for free meal and FRPM counts. The criteria include a minimum free meal count and a maximum FRPM count.")
async def get_count_cds_codes_by_county_meal_counts(county_name: str = Query(..., description="County name"), free_meal_count: int = Query(..., description="Free meal count (K-12)"), frpm_count: int = Query(..., description="FRPM count (K-12)")):
    cursor.execute("SELECT COUNT(CDSCode) FROM frpm WHERE `County Name` = ? AND `Free Meal Count (K-12)` > ? AND `FRPM Count (K-12)` < ?", (county_name, free_meal_count, frpm_count))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the school name with the highest number of test takers in a specific county
@app.get("/v1/california_schools/top_school_by_test_takers", operation_id="get_top_school_by_test_takers", summary="Retrieves the name of the school with the highest number of test takers in a specified county. The operation filters schools by county and excludes those with missing names, then sorts the remaining schools by the number of test takers in descending order. The top school is returned.")
async def get_top_school_by_test_takers(county_name: str = Query(..., description="County name")):
    cursor.execute("SELECT sname FROM satscores WHERE cname = ? AND sname IS NOT NULL ORDER BY NumTstTakr DESC LIMIT 1", (county_name,))
    result = cursor.fetchone()
    if not result:
        return {"school_name": []}
    return {"school_name": result[0]}

# Endpoint to get schools and their streets with a specific enrollment difference
@app.get("/v1/california_schools/schools_by_enrollment_difference", operation_id="get_schools_by_enrollment_difference", summary="Retrieves a list of schools and their corresponding streets where the enrollment difference between K-12 and ages 5-17 exceeds the provided threshold.")
async def get_schools_by_enrollment_difference(enrollment_difference: int = Query(..., description="Enrollment difference between K-12 and Ages 5-17")):
    cursor.execute("SELECT T1.School, T1.Street FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Enrollment (K-12)` - T2.`Enrollment (Ages 5-17)` > ?", (enrollment_difference,))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [{"school": row[0], "street": row[1]} for row in result]}

# Endpoint to get school names with a specific free meal ratio and SAT score count
@app.get("/v1/california_schools/schools_by_free_meal_ratio_sat_score", operation_id="get_schools_by_free_meal_ratio_sat_score", summary="Retrieves the names of schools in California where the ratio of students receiving free meals (K-12) surpasses the specified threshold and the number of SAT scores at or above 1500 meets the given count.")
async def get_schools_by_free_meal_ratio_sat_score(free_meal_ratio: float = Query(..., description="Free meal ratio (K-12)"), num_ge_1500: int = Query(..., description="Number of SAT scores >= 1500")):
    cursor.execute("SELECT T2.`School Name` FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode WHERE CAST(T2.`Free Meal Count (K-12)` AS REAL) / T2.`Enrollment (K-12)` > ? AND T1.NumGE1500 > ?", (free_meal_ratio, num_ge_1500))
    result = cursor.fetchall()
    if not result:
        return {"school_names": []}
    return {"school_names": [row[0] for row in result]}

# Endpoint to get school names and charter funding types with a specific average math score
@app.get("/v1/california_schools/schools_by_district_avg_math_score", operation_id="get_schools_by_district_avg_math_score", summary="Retrieves the names of schools and their charter funding types within a specified district, filtering results based on a minimum average math score. The district is identified using a pattern, and the average math score is a numerical value.")
async def get_schools_by_district_avg_math_score(district_name: str = Query(..., description="District name pattern"), avg_math_score: float = Query(..., description="Average math score")):
    cursor.execute("SELECT T1.sname, T2.`Charter Funding Type` FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode WHERE T2.`District Name` LIKE ? GROUP BY T1.sname, T2.`Charter Funding Type` HAVING CAST(SUM(T1.AvgScrMath) AS REAL) / COUNT(T1.cds) > ?", (district_name, avg_math_score))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [{"school_name": row[0], "charter_funding_type": row[1]} for row in result]}

# Endpoint to get school details in a specific county with specific free meal count and school type
@app.get("/v1/california_schools/school_details_by_county_meal_count_type", operation_id="get_school_details_by_county_meal_count_type", summary="Retrieves detailed information about schools in a specified county that offer a certain number of free meals and belong to a particular school type. The response includes the school's name, street address, city, state, and zip code.")
async def get_school_details_by_county_meal_count_type(county: str = Query(..., description="County name"), free_meal_count: int = Query(..., description="Free meal count (Ages 5-17)"), school_type: str = Query(..., description="School type")):
    cursor.execute("SELECT T1.`School Name`, T2.Street, T2.City, T2.State, T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = ? AND T1.`Free Meal Count (Ages 5-17)` > ? AND T1.`School Type` = ?", (county, free_meal_count, school_type))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [{"school_name": row[0], "street": row[1], "city": row[2], "state": row[3], "zip": row[4]} for row in result]}

# Endpoint to get school details with specific open and close dates
@app.get("/v1/california_schools/school_details_by_open_close_dates", operation_id="get_school_details_by_open_close_dates", summary="Retrieves details of schools that opened after the specified year or closed before the specified year, including the school name, average SAT writing score, and phone number. The open and close years are provided in 'YYYY' format.")
async def get_school_details_by_open_close_dates(open_year: str = Query(..., description="Open year in 'YYYY' format"), close_year: str = Query(..., description="Close year in 'YYYY' format")):
    cursor.execute("SELECT T2.School, T1.AvgScrWrite, T2.Phone FROM schools AS T2 LEFT JOIN satscores AS T1 ON T2.CDSCode = T1.cds WHERE strftime('%Y', T2.OpenDate) > ? OR strftime('%Y', T2.ClosedDate) < ?", (open_year, close_year))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [{"school": row[0], "avg_scr_write": row[1], "phone": row[2]} for row in result]}

# Endpoint to get school details with specific funding type and enrollment difference
@app.get("/v1/california_schools/school_details_by_funding_type_enrollment_difference", operation_id="get_school_details_by_funding_type_enrollment_difference", summary="Retrieves details of schools with a specified funding type and an enrollment difference that exceeds the average enrollment difference of schools with the same funding type. The enrollment difference is calculated as the difference between the total enrollment and the enrollment of students aged 5-17.")
async def get_school_details_by_funding_type_enrollment_difference(funding_type: str = Query(..., description="Funding type")):
    cursor.execute("SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = ? AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > (SELECT AVG(T3.`Enrollment (K-12)` - T3.`Enrollment (Ages 5-17)`) FROM frpm AS T3 INNER JOIN schools AS T4 ON T3.CDSCode = T4.CDSCode WHERE T4.FundingType = ?)", (funding_type, funding_type))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [{"school": row[0], "doc": row[1]} for row in result]}

# Endpoint to get the open date of the school with the highest enrollment
@app.get("/v1/california_schools/open_date_highest_enrollment", operation_id="get_open_date_highest_enrollment", summary="Retrieves the open date of the school with the highest enrollment in California. This operation identifies the school with the most students and returns the date it began operations. The data is sourced from a comprehensive database of California schools.")
async def get_open_date_highest_enrollment():
    cursor.execute("SELECT T2.OpenDate FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`Enrollment (K-12)` DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"open_date": []}
    return {"open_date": result[0]}

# Endpoint to get the top cities with the lowest total enrollment (K-12)
@app.get("/v1/california_schools/top_cities_lowest_enrollment", operation_id="get_top_cities_lowest_enrollment", summary="Retrieve a list of cities in California with the lowest total enrollment in K-12 schools. The list is sorted in ascending order based on the total enrollment, and the number of cities returned can be limited by specifying the 'limit' parameter.")
async def get_top_cities_lowest_enrollment(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.City FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode GROUP BY T2.City ORDER BY SUM(T1.`Enrollment (K-12)`) ASC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get the ratio of free meal count to enrollment (K-12) for schools with the highest enrollment
@app.get("/v1/california_schools/free_meal_ratio_highest_enrollment", operation_id="get_free_meal_ratio_highest_enrollment", summary="Get the ratio of free meal count to enrollment (K-12) for schools with the highest enrollment")
async def get_free_meal_ratio_highest_enrollment(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset for pagination")):
    cursor.execute("SELECT CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)` FROM frpm ORDER BY `Enrollment (K-12)` DESC LIMIT ? OFFSET ?", (limit, offset))
    result = cursor.fetchall()
    if not result:
        return {"ratios": []}
    return {"ratios": [row[0] for row in result]}

# Endpoint to get the ratio of FRPM count to enrollment (K-12) for schools with a specific SOC code
@app.get("/v1/california_schools/frpm_ratio_by_soc", operation_id="get_frpm_ratio_by_soc", summary="Retrieves the proportion of students receiving free or reduced-price meals (FRPM) to the total enrollment in schools with a specified SOC code. The results are sorted in descending order based on the FRPM count. The number of results can be limited by providing a value for the limit parameter.")
async def get_frpm_ratio_by_soc(soc: int = Query(..., description="SOC code"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT CAST(T1.`FRPM Count (K-12)` AS REAL) / T1.`Enrollment (K-12)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.SOC = ? ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT ?", (soc, limit))
    result = cursor.fetchall()
    if not result:
        return {"ratios": []}
    return {"ratios": [row[0] for row in result]}

# Endpoint to get school names and websites with a specific range of free meal counts (ages 5-17)
@app.get("/v1/california_schools/schools_by_free_meal_count", operation_id="get_schools_by_free_meal_count", summary="Retrieves a list of school names and their corresponding websites that offer a specific range of free meals to students aged 5-17. The range is determined by the provided minimum and maximum free meal counts.")
async def get_schools_by_free_meal_count(min_count: int = Query(..., description="Minimum free meal count"), max_count: int = Query(..., description="Maximum free meal count")):
    cursor.execute("SELECT T2.Website, T1.`School Name` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Free Meal Count (Ages 5-17)` BETWEEN ? AND ? AND T2.Website IS NOT NULL", (min_count, max_count))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [{"website": row[0], "school_name": row[1]} for row in result]}

# Endpoint to get the ratio of free meal count to enrollment (ages 5-17) for schools with specific admin names
@app.get("/v1/california_schools/free_meal_ratio_by_admin_names", operation_id="get_free_meal_ratio_by_admin_names", summary="Retrieve the proportion of students receiving free meals (ages 5-17) in schools managed by a specific administrator. The calculation is based on the ratio of the number of students receiving free meals to the total enrollment in the specified age range.")
async def get_free_meal_ratio_by_admin_names(adm_fname1: str = Query(..., description="First name of the admin"), adm_lname1: str = Query(..., description="Last name of the admin")):
    cursor.execute("SELECT CAST(T2.`Free Meal Count (Ages 5-17)` AS REAL) / T2.`Enrollment (Ages 5-17)` FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.AdmFName1 = ? AND T1.AdmLName1 = ?", (adm_fname1, adm_lname1))
    result = cursor.fetchall()
    if not result:
        return {"ratios": []}
    return {"ratios": [row[0] for row in result]}

# Endpoint to get admin emails for charter schools with the lowest enrollment (K-12)
@app.get("/v1/california_schools/admin_emails_charter_schools", operation_id="get_admin_emails_charter_schools", summary="Retrieves the primary administrative email addresses of charter schools with the lowest enrollment (K-12). The operation allows filtering by charter school status and limiting the number of results returned. The data is sourced from the 'frpm' and 'schools' tables, with the 'CDSCode' field serving as the link between them.")
async def get_admin_emails_charter_schools(charter_school: int = Query(..., description="Charter school indicator (1 for yes, 0 for no)"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.AdmEmail1 FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Charter School (Y/N)` = ? ORDER BY T1.`Enrollment (K-12)` ASC LIMIT ?", (charter_school, limit))
    result = cursor.fetchall()
    if not result:
        return {"emails": []}
    return {"emails": [row[0] for row in result]}

# Endpoint to get admin names for schools with the highest number of students scoring 1500 or above on the SAT
@app.get("/v1/california_schools/admin_names_highest_sat_scores", operation_id="get_admin_names_highest_sat_scores", summary="Retrieve the names of school administrators from institutions with the highest count of students who scored 1500 or above on the SAT. The response is sorted in descending order based on the number of students achieving this score. The limit parameter can be used to restrict the number of results returned.")
async def get_admin_names_highest_sat_scores(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.AdmFName1, T2.AdmLName1, T2.AdmFName2, T2.AdmLName2, T2.AdmFName3, T2.AdmLName3 FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"admin_names": []}
    return {"admin_names": [{"adm_fname1": row[0], "adm_lname1": row[1], "adm_fname2": row[2], "adm_lname2": row[3], "adm_fname3": row[4], "adm_lname3": row[5]} for row in result]}

# Endpoint to get school addresses with the lowest ratio of students scoring 1500 or above on the SAT
@app.get("/v1/california_schools/school_addresses_lowest_sat_ratio", operation_id="get_school_addresses_lowest_sat_ratio", summary="Retrieves the addresses of schools in California with the lowest ratio of students scoring 1500 or above on the SAT. The results are sorted in ascending order based on the ratio of students scoring 1500 or above to the total number of test takers. The number of results can be limited by specifying the 'limit' parameter.")
async def get_school_addresses_lowest_sat_ratio(limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T2.Street, T2.City, T2.State, T2.Zip FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY CAST(T1.NumGE1500 AS REAL) / T1.NumTstTakr ASC LIMIT ?", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [{"street": row[0], "city": row[1], "state": row[2], "zip": row[3]} for row in result]}

# Endpoint to get school websites with a specific range of SAT test takers in a given county
@app.get("/v1/california_schools/school_websites_by_test_takers_county", operation_id="get_school_websites_by_test_takers_county", summary="Retrieve a list of school websites in a specified county where the number of SAT test takers falls within a given range. The range is defined by a minimum and maximum number of test takers.")
async def get_school_websites_by_test_takers_county(min_test_takers: int = Query(..., description="Minimum number of SAT test takers"), max_test_takers: int = Query(..., description="Maximum number of SAT test takers"), county: str = Query(..., description="County")):
    cursor.execute("SELECT T2.Website FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.NumTstTakr BETWEEN ? AND ? AND T2.County = ?", (min_test_takers, max_test_takers, county))
    result = cursor.fetchall()
    if not result:
        return {"websites": []}
    return {"websites": [row[0] for row in result]}

# Endpoint to get the average number of SAT test takers for schools opened in a specific year in a given county
@app.get("/v1/california_schools/avg_sat_test_takers_by_year_county", operation_id="get_avg_sat_test_takers_by_year_county", summary="Retrieves the average number of SAT test takers for schools that opened in a specific year within a given county. The operation calculates this average based on data from the schools and their corresponding SAT scores.")
async def get_avg_sat_test_takers_by_year_county(year: str = Query(..., description="Year the school was opened (format: 'YYYY')"), county: str = Query(..., description="County")):
    cursor.execute("SELECT AVG(T1.NumTstTakr) FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE strftime('%Y', T2.OpenDate) = ? AND T2.County = ?", (year, county))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the phone number of the school with the lowest average reading score in a given district
@app.get("/v1/california_schools/phone_lowest_avg_reading_score", operation_id="get_phone_lowest_avg_reading_score", summary="Retrieves the phone number of the school with the lowest average reading score in a specified district. The operation filters schools based on the provided district name and identifies the one with the lowest non-null average reading score. The result is the phone number of the school that meets these criteria.")
async def get_phone_lowest_avg_reading_score(district: str = Query(..., description="District name")):
    cursor.execute("SELECT T2.Phone FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.District = ? AND T1.AvgScrRead IS NOT NULL ORDER BY T1.AvgScrRead ASC LIMIT 1", (district,))
    result = cursor.fetchone()
    if not result:
        return {"phone": []}
    return {"phone": result[0]}

# Endpoint to get the top schools based on average reading score in each county
@app.get("/v1/california_schools/top_schools_avg_reading_score", operation_id="get_top_schools_avg_reading_score", summary="Retrieve the top-performing schools in each county based on their average reading scores. The operation considers only schools with a specified virtual status and returns a maximum of the top-ranked schools as determined by the provided rank limit.")
async def get_top_schools_avg_reading_score(virtual: str = Query(..., description="Virtual status of the school ('F' for non-virtual)"), rank_limit: int = Query(..., description="Limit for the rank of schools")):
    cursor.execute("SELECT School FROM (SELECT T2.School,T1.AvgScrRead, RANK() OVER (PARTITION BY T2.County ORDER BY T1.AvgScrRead DESC) AS rnk FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.Virtual = ? ) ranked_schools WHERE rnk <= ?", (virtual, rank_limit))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [row[0] for row in result]}

# Endpoint to get the name of the educational operations with the highest average math score
@app.get("/v1/california_schools/highest_avg_math_score_edops", operation_id="get_highest_avg_math_score_edops", summary="Retrieves the name of the educational operation with the highest average math score, based on the SAT scores of schools in California. The operation uses the average math scores to rank the educational operations and returns the top-ranked operation.")
async def get_highest_avg_math_score_edops():
    cursor.execute("SELECT T2.EdOpsName FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.AvgScrMath DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"edops_name": []}
    return {"edops_name": result[0]}

# Endpoint to get the average math score and county of the school with the lowest combined average scores
@app.get("/v1/california_schools/lowest_combined_avg_scores", operation_id="get_lowest_combined_avg_scores", summary="Retrieves the average math score and the corresponding county of the school with the lowest combined average scores in math, reading, and writing. The scores are calculated based on the SAT scores of the students in the school.")
async def get_lowest_combined_avg_scores():
    cursor.execute("SELECT T1.AvgScrMath, T2.County FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrMath IS NOT NULL ORDER BY T1.AvgScrMath + T1.AvgScrRead + T1.AvgScrWrite ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"avg_math_score": [], "county": []}
    return {"avg_math_score": result[0], "county": result[1]}

# Endpoint to get the average writing score and city of the school with the highest number of students scoring above 1500
@app.get("/v1/california_schools/highest_num_ge1500_avg_writing_score", operation_id="get_highest_num_ge1500_avg_writing_score", summary="Retrieves the average writing score and city of the school with the highest number of students who scored above 1500 in the writing section of the SAT. The data is sourced from the SAT scores and schools databases, and the results are ordered by the number of students who scored above 1500 in descending order, with only the top result being returned.")
async def get_highest_num_ge1500_avg_writing_score():
    cursor.execute("SELECT T1.AvgScrWrite, T2.City FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"avg_writing_score": [], "city": []}
    return {"avg_writing_score": result[0], "city": result[1]}

# Endpoint to get the school and average writing score based on administrator's first and last name
@app.get("/v1/california_schools/school_avg_writing_score_by_admin", operation_id="get_school_avg_writing_score_by_admin", summary="Retrieves the school name and its average writing score based on the provided administrator's first and last name. This operation uses the administrator's name to filter the data and returns the corresponding school and its average writing score.")
async def get_school_avg_writing_score_by_admin(adm_fname1: str = Query(..., description="Administrator's first name"), adm_lname1: str = Query(..., description="Administrator's last name")):
    cursor.execute("SELECT T2.School, T1.AvgScrWrite FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.AdmFName1 = ? AND T2.AdmLName1 = ?", (adm_fname1, adm_lname1))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": [{"school": row[0], "avg_writing_score": row[1]} for row in result]}

# Endpoint to get the school with the highest enrollment in a given district
@app.get("/v1/california_schools/highest_enrollment_school", operation_id="get_highest_enrollment_school", summary="Retrieves the school with the highest enrollment in a specified district. The district is identified by its operational code (DOC). The operation returns the name of the school with the highest number of students enrolled in grades K-12 within the given district.")
async def get_highest_enrollment_school(doc: int = Query(..., description="District Operational Code (DOC)")):
    cursor.execute("SELECT T2.School FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.DOC = ? ORDER BY T1.`Enrollment (K-12)` DESC LIMIT 1", (doc,))
    result = cursor.fetchone()
    if not result:
        return {"school": []}
    return {"school": result[0]}

# Endpoint to get the average number of schools opened in a given year in a specific county and district
@app.get("/v1/california_schools/avg_schools_opened_year", operation_id="get_avg_schools_opened_year", summary="Retrieves the average number of schools opened per month in a given year within a specific county and district. The calculation is based on the total count of schools divided by 12 months. The operation requires the District Operational Code (DOC), county name, and the year in 'YYYY' format as input parameters.")
async def get_avg_schools_opened_year(doc: int = Query(..., description="District Operational Code (DOC)"), county: str = Query(..., description="County name"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(COUNT(School) AS REAL) / 12 FROM schools WHERE DOC = ? AND County = ? AND strftime('%Y', OpenDate) = ?", (doc, county, year))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the ratio of schools with a specific DOC to another DOC in a given county and status type
@app.get("/v1/california_schools/ratio_schools_doc", operation_id="get_ratio_schools_doc", summary="Retrieves the ratio of schools with a specific operational code (DOC1) to another operational code (DOC2) in a given county and status type. This operation calculates the ratio by dividing the total number of schools with DOC1 by the total number of schools with DOC2, where both sets of schools share the same status type and county.")
async def get_ratio_schools_doc(doc1: int = Query(..., description="First District Operational Code (DOC)"), doc2: int = Query(..., description="Second District Operational Code (DOC)"), status_type: str = Query(..., description="Status type of the school"), county: str = Query(..., description="County name")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN DOC = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN DOC = ? THEN 1 ELSE 0 END) FROM schools WHERE StatusType = ? AND County = ?", (doc1, doc2, status_type, county))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the details of closed schools in the county with the highest number of closed schools
@app.get("/v1/california_schools/closed_schools_highest_county", operation_id="get_closed_schools_highest_county", summary="Retrieves the details of closed schools in the county with the highest number of closed schools. The response includes the county name, school name, and the date each school was closed. The data is filtered to exclude schools with a null value for the school name.")
async def get_closed_schools_highest_county():
    cursor.execute("SELECT DISTINCT County, School, ClosedDate FROM schools WHERE County = ( SELECT County FROM schools WHERE StatusType = 'Closed' GROUP BY County ORDER BY COUNT(School) DESC LIMIT 1 ) AND StatusType = 'Closed' AND school IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"closed_schools": []}
    return {"closed_schools": [{"county": row[0], "school": row[1], "closed_date": row[2]} for row in result]}

# Endpoint to get school details ordered by average math score
@app.get("/v1/california_schools/school_details_by_avg_math_score", operation_id="get_school_details_by_avg_math_score", summary="Retrieves detailed information about California schools, ordered by their average math scores in descending order. The response is paginated, allowing you to specify the number of results to return (limit) and the starting point of the results (offset).")
async def get_school_details_by_avg_math_score(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset the results")):
    cursor.execute("SELECT T2.MailStreet, T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.AvgScrMath DESC LIMIT ? OFFSET ?", (limit, offset))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": result}

# Endpoint to get school details ordered by average reading score
@app.get("/v1/california_schools/school_details_by_avg_reading_score", operation_id="get_school_details_by_avg_reading_score", summary="Retrieve a limited number of school details, ordered by their average reading scores. The results can be offset to retrieve different subsets of the data. This operation is useful for obtaining a ranked list of schools based on their reading scores.")
async def get_school_details_by_avg_reading_score(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset the results")):
    cursor.execute("SELECT T2.MailStreet, T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrRead IS NOT NULL ORDER BY T1.AvgScrRead ASC LIMIT ? OFFSET ?", (limit, offset))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": result}

# Endpoint to get the count of schools based on city and combined average scores
@app.get("/v1/california_schools/count_schools_by_city_and_scores", operation_id="get_count_schools_by_city_and_scores", summary="Retrieve the number of schools in a specified city that have a combined average score of at least 1500 in reading, math, and writing. The city is identified by its mailing city name, and the combined average score is calculated as the sum of the average scores in reading, math, and writing.")
async def get_count_schools_by_city_and_scores(mail_city: str = Query(..., description="City name"), combined_score: int = Query(..., description="Combined average score threshold")):
    cursor.execute("SELECT COUNT(T1.cds) FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.MailCity = ? AND (T1.AvgScrRead + T1.AvgScrMath + T1.AvgScrWrite) >= ?", (mail_city, combined_score))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the number of SAT test takers by city
@app.get("/v1/california_schools/num_sat_test_takers_by_city", operation_id="get_num_sat_test_takers_by_city", summary="Retrieves the total number of students who took the SAT in a specified city in California. The city is identified by its name, which is provided as an input parameter.")
async def get_num_sat_test_takers_by_city(mail_city: str = Query(..., description="City name")):
    cursor.execute("SELECT T1.NumTstTakr FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.MailCity = ?", (mail_city,))
    result = cursor.fetchall()
    if not result:
        return {"test_takers": []}
    return {"test_takers": result}

# Endpoint to get school and mail zip based on administrator's name
@app.get("/v1/california_schools/school_and_mail_zip_by_admin_name", operation_id="get_school_and_mail_zip_by_admin_name", summary="Retrieves the school name and its mailing zip code associated with the specified administrator. The administrator is identified by their first and last names.")
async def get_school_and_mail_zip_by_admin_name(adm_fname1: str = Query(..., description="Administrator's first name"), adm_lname1: str = Query(..., description="Administrator's last name")):
    cursor.execute("SELECT School, MailZip FROM schools WHERE AdmFName1 = ? AND AdmLName1 = ?", (adm_fname1, adm_lname1))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": result}

# Endpoint to get the ratio of schools in two counties within a state
@app.get("/v1/california_schools/ratio_of_schools_in_counties", operation_id="get_ratio_of_schools_in_counties", summary="Retrieve the ratio of schools in two specified counties within a given state. The operation compares the number of schools in the first county to the number of schools in the second county, both within the provided state. The result is a numerical value representing the ratio of schools between the two counties.")
async def get_ratio_of_schools_in_counties(county1: str = Query(..., description="First county name"), county2: str = Query(..., description="Second county name"), mail_state: str = Query(..., description="State abbreviation")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN County = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN County = ? THEN 1 ELSE 0 END) FROM schools WHERE MailState = ?", (county1, county2, mail_state))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of schools based on city, state, and status type
@app.get("/v1/california_schools/count_schools_by_city_state_status", operation_id="get_count_schools_by_city_state_status", summary="Retrieves the total number of schools in a given city and state that have a specific status. The operation requires the city name, state abbreviation, and status type as input parameters to filter the schools and calculate the count.")
async def get_count_schools_by_city_state_status(city: str = Query(..., description="City name"), mail_state: str = Query(..., description="State abbreviation"), status_type: str = Query(..., description="Status type")):
    cursor.execute("SELECT COUNT(CDSCode) FROM schools WHERE City = ? AND MailState = ? AND StatusType = ?", (city, mail_state, status_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get school phone and extension ordered by average writing score
@app.get("/v1/california_schools/school_phone_ext_by_avg_writing_score", operation_id="get_school_phone_ext_by_avg_writing_score", summary="Get school phone and extension ordered by average writing score with limit and offset")
async def get_school_phone_ext_by_avg_writing_score(limit: int = Query(..., description="Limit the number of results"), offset: int = Query(..., description="Offset the results")):
    cursor.execute("SELECT T2.Phone, T2.Ext FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.AvgScrWrite DESC LIMIT ? OFFSET ?", (limit, offset))
    result = cursor.fetchall()
    if not result:
        return {"phones": []}
    return {"phones": result}

# Endpoint to get school phone, extension, and name based on zip code
@app.get("/v1/california_schools/school_details_by_zip", operation_id="get_school_details_by_zip", summary="Retrieves the phone number, extension, and name of schools located in the specified zip code.")
async def get_school_details_by_zip(zip: str = Query(..., description="Zip code")):
    cursor.execute("SELECT Phone, Ext, School FROM schools WHERE Zip = ?", (zip,))
    result = cursor.fetchall()
    if not result:
        return {"schools": []}
    return {"schools": result}

# Endpoint to get school websites based on administrator names
@app.get("/v1/california_schools/school_websites_by_admin_names", operation_id="get_school_websites_by_admin_names", summary="Retrieves the websites of schools that are administered by the specified individuals. The operation accepts the first and last names of two administrators and returns the corresponding school websites. This endpoint is useful for identifying schools based on their administrative staff.")
async def get_school_websites_by_admin_names(adm_fname1_1: str = Query(..., description="First administrator's first name"), adm_lname1_1: str = Query(..., description="First administrator's last name"), adm_fname1_2: str = Query(..., description="Second administrator's first name"), adm_lname1_2: str = Query(..., description="Second administrator's last name")):
    cursor.execute("SELECT Website FROM schools WHERE (AdmFName1 = ? AND AdmLName1 = ?) OR (AdmFName1 = ? AND AdmLName1 = ?)", (adm_fname1_1, adm_lname1_1, adm_fname1_2, adm_lname1_2))
    result = cursor.fetchall()
    if not result:
        return {"websites": []}
    return {"websites": result}

# Endpoint to get the website of schools in a specific county with specific virtual and charter status
@app.get("/v1/california_schools/get_website_by_county_virtual_charter", operation_id="get_website_by_county_virtual_charter", summary="Retrieves the website of schools located in a specified county that have a particular virtual status and charter status. The operation filters schools based on the provided county, virtual status, and charter status, and returns the website URLs of the matching schools.")
async def get_website_by_county_virtual_charter(county: str = Query(..., description="County of the school"), virtual: str = Query(..., description="Virtual status of the school"), charter: int = Query(..., description="Charter status of the school")):
    cursor.execute("SELECT Website FROM schools WHERE County = ? AND Virtual = ? AND Charter = ?", (county, virtual, charter))
    result = cursor.fetchall()
    if not result:
        return {"websites": []}
    return {"websites": [row[0] for row in result]}

# Endpoint to get the count of schools in a specific city with specific DOC and charter status
@app.get("/v1/california_schools/count_schools_by_doc_charter_city", operation_id="count_schools_by_doc_charter_city", summary="Retrieves the total number of schools in a specified city that have a particular DOC status and charter status. The response is based on the provided DOC, charter status, and city parameters.")
async def count_schools_by_doc_charter_city(doc: int = Query(..., description="DOC of the school"), charter: int = Query(..., description="Charter status of the school"), city: str = Query(..., description="City of the school")):
    cursor.execute("SELECT COUNT(School) FROM schools WHERE DOC = ? AND Charter = ? AND City = ?", (doc, charter, city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of schools in a specific county with a specific charter status and free meal percentage
@app.get("/v1/california_schools/count_schools_by_county_charter_free_meal_percentage", operation_id="count_schools_by_county_charter_free_meal_percentage", summary="Retrieve the number of schools in a specified county that meet the given charter status and have a free meal percentage below a certain threshold. This operation provides a count of schools that satisfy the input criteria, offering insights into the distribution of schools based on charter status and free meal availability.")
async def count_schools_by_county_charter_free_meal_percentage(county: str = Query(..., description="County of the school"), charter: int = Query(..., description="Charter status of the school"), free_meal_percentage: float = Query(..., description="Free meal percentage threshold")):
    cursor.execute("SELECT COUNT(T2.School) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = ? AND T2.Charter = ? AND CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < ?", (county, charter, free_meal_percentage))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the administrative details of schools with specific charter status and charter number
@app.get("/v1/california_schools/get_admin_details_by_charter_charter_num", operation_id="get_admin_details_by_charter_charter_num", summary="Retrieves the first names and last names of the administrators, along with the school name and city, for schools that have a specific charter status and charter number. The charter status and charter number are used to filter the results.")
async def get_admin_details_by_charter_charter_num(charter: int = Query(..., description="Charter status of the school"), charter_num: str = Query(..., description="Charter number of the school")):
    cursor.execute("SELECT AdmFName1, AdmLName1, School, City FROM schools WHERE Charter = ? AND CharterNum = ?", (charter, charter_num))
    result = cursor.fetchall()
    if not result:
        return {"admin_details": []}
    return {"admin_details": [{"AdmFName1": row[0], "AdmLName1": row[1], "School": row[2], "City": row[3]} for row in result]}

# Endpoint to get the count of schools with specific charter number and mail city
@app.get("/v1/california_schools/count_schools_by_charter_num_mail_city", operation_id="count_schools_by_charter_num_mail_city", summary="Retrieves the total number of schools that match the specified charter number and mail city. This operation is useful for determining the quantity of schools in a particular charter and city combination.")
async def count_schools_by_charter_num_mail_city(charter_num: str = Query(..., description="Charter number of the school"), mail_city: str = Query(..., description="Mail city of the school")):
    cursor.execute("SELECT COUNT(*) FROM schools WHERE CharterNum = ? AND MailCity = ?", (charter_num, mail_city))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of schools with a specific funding type in a specific county with a specific charter status
@app.get("/v1/california_schools/percentage_funding_type_by_county_charter", operation_id="percentage_funding_type_by_county_charter", summary="Retrieve the percentage of schools in a given county with a specific charter status that have a particular funding type. The calculation is based on the total number of schools with different funding types in the specified county and charter status.")
async def percentage_funding_type_by_county_charter(funding_type: str = Query(..., description="Funding type of the school"), county: str = Query(..., description="County of the school"), charter: int = Query(..., description="Charter status of the school")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN FundingType = ? THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN FundingType != ? THEN 1 ELSE 0 END) FROM schools WHERE County = ? AND Charter = ?", (funding_type, funding_type, county, charter))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of schools opened within a specific year range in a specific county with a specific funding type
@app.get("/v1/california_schools/count_schools_opened_by_year_range_county_funding_type", operation_id="count_schools_opened_by_year_range_county_funding_type", summary="Retrieves the total number of schools that opened between the specified start and end years in a given county, filtered by a particular funding type.")
async def count_schools_opened_by_year_range_county_funding_type(start_year: str = Query(..., description="Start year of the range (YYYY)"), end_year: str = Query(..., description="End year of the range (YYYY)"), county: str = Query(..., description="County of the school"), funding_type: str = Query(..., description="Funding type of the school")):
    cursor.execute("SELECT COUNT(School) FROM schools WHERE strftime('%Y', OpenDate) BETWEEN ? AND ? AND County = ? AND FundingType = ?", (start_year, end_year, county, funding_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of schools closed in a specific year in a specific city with a specific DOC type
@app.get("/v1/california_schools/count_schools_closed_by_year_city_doc_type", operation_id="count_schools_closed_by_year_city_doc_type", summary="Retrieve the number of schools that closed in a given year within a specific city, filtered by a particular DOC type.")
async def count_schools_closed_by_year_city_doc_type(year: str = Query(..., description="Year the school was closed (YYYY)"), city: str = Query(..., description="City of the school"), doc_type: str = Query(..., description="DOC type of the school")):
    cursor.execute("SELECT COUNT(School) FROM schools WHERE strftime('%Y', ClosedDate) = ? AND City = ? AND DOCType = ?", (year, city, doc_type))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the county with the highest number of closed schools within a specific year range with a specific status type and SOC
@app.get("/v1/california_schools/top_county_closed_schools_by_year_range_status_soc", operation_id="top_county_closed_schools_by_year_range_status_soc", summary="Retrieves the county with the most closed schools within a specified year range, status type, and SOC. The response is based on the count of schools that meet the given criteria, sorted in descending order. The start and end years define the range, while the status type and SOC further filter the results.")
async def top_county_closed_schools_by_year_range_status_soc(start_year: str = Query(..., description="Start year of the range (YYYY)"), end_year: str = Query(..., description="End year of the range (YYYY)"), status_type: str = Query(..., description="Status type of the school"), soc: int = Query(..., description="SOC of the school")):
    cursor.execute("SELECT County FROM schools WHERE strftime('%Y', ClosedDate) BETWEEN ? AND ? AND StatusType = ? AND SOC = ? GROUP BY County ORDER BY COUNT(School) DESC LIMIT 1", (start_year, end_year, status_type, soc))
    result = cursor.fetchone()
    if not result:
        return {"county": []}
    return {"county": result[0]}

# Endpoint to get the NCES district of schools with a specific SOC
@app.get("/v1/california_schools/get_nces_district_by_soc", operation_id="get_nces_district_by_soc", summary="Retrieves the NCES district identifier for schools in California that match the provided SOC code. The SOC code is a unique identifier for the school's Standard Occupational Classification.")
async def get_nces_district_by_soc(soc: int = Query(..., description="SOC of the school")):
    cursor.execute("SELECT NCESDist FROM schools WHERE SOC = ?", (soc,))
    result = cursor.fetchall()
    if not result:
        return {"nces_districts": []}
    return {"nces_districts": [row[0] for row in result]}

# Endpoint to get the count of schools based on status type, SOC, and county
@app.get("/v1/california_schools/count_schools_status_soc_county", operation_id="get_count_schools_status_soc_county", summary="This operation retrieves the total number of schools in a specific county that match two given status types and a specified SOC code. The SOC code and county are used to filter the schools, while the status types determine the criteria for counting the schools.")
async def get_count_schools_status_soc_county(status_type1: str = Query(..., description="First status type"), status_type2: str = Query(..., description="Second status type"), soc: int = Query(..., description="SOC code"), county: str = Query(..., description="County")):
    cursor.execute("SELECT COUNT(School) FROM schools WHERE (StatusType = ? OR StatusType = ?) AND SOC = ? AND County = ?", (status_type1, status_type2, soc, county))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get district codes based on city and magnet status
@app.get("/v1/california_schools/district_codes_city_magnet", operation_id="get_district_codes_city_magnet", summary="Retrieves the district codes for schools in a specified city that have a specified magnet status. The city and magnet status are provided as input parameters, allowing for a targeted search of the school database.")
async def get_district_codes_city_magnet(city: str = Query(..., description="City"), magnet: int = Query(..., description="Magnet status (0 or 1)")):
    cursor.execute("SELECT T1.`District Code` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.City = ? AND T2.Magnet = ?", (city, magnet))
    result = cursor.fetchall()
    if not result:
        return {"district_codes": []}
    return {"district_codes": [row[0] for row in result]}

# Endpoint to get enrollment data based on EdOpsCode, city, and academic year range
@app.get("/v1/california_schools/enrollment_edopscode_city_academic_year", operation_id="get_enrollment_edopscode_city_academic_year", summary="Retrieves the enrollment data for schools in a specific city, identified by their EdOpsCode, within a given academic year range.")
async def get_enrollment_edopscode_city_academic_year(edopscode: str = Query(..., description="EdOpsCode"), city: str = Query(..., description="City"), start_year: int = Query(..., description="Start of academic year range"), end_year: int = Query(..., description="End of academic year range")):
    cursor.execute("SELECT T1.`Enrollment (Ages 5-17)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.EdOpsCode = ? AND T2.City = ? AND T1.`Academic Year` BETWEEN ? AND ?", (edopscode, city, start_year, end_year))
    result = cursor.fetchall()
    if not result:
        return {"enrollment": []}
    return {"enrollment": [row[0] for row in result]}

# Endpoint to get FRPM count based on mail street and SOC type
@app.get("/v1/california_schools/frpm_count_mailstreet_soctype", operation_id="get_frpm_count_mailstreet_soctype", summary="Retrieves the number of students eligible for free or reduced-price meals (FRPM) in California schools that share a specified mailing street and school type (SOC). The response is based on the intersection of data from the FRPM and schools datasets, filtered by the provided mailing street and school type.")
async def get_frpm_count_mailstreet_soctype(mail_street: str = Query(..., description="Mail street"), soc_type: str = Query(..., description="SOC type")):
    cursor.execute("SELECT T1.`FRPM Count (Ages 5-17)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.MailStreet = ? AND T2.SOCType = ?", (mail_street, soc_type))
    result = cursor.fetchall()
    if not result:
        return {"frpm_count": []}
    return {"frpm_count": [row[0] for row in result]}

# Endpoint to get the minimum low grade based on NCESDist and EdOpsCode
@app.get("/v1/california_schools/min_low_grade_ncesdist_edopscode", operation_id="get_min_low_grade_ncesdist_edopscode", summary="Retrieves the lowest grade level offered by any school in a specified district and education operation code. The district is identified by its NCESDist code, and the education operation code is specified by the EdOpsCode. This operation is useful for understanding the minimum grade level offered in a particular district and education operation context.")
async def get_min_low_grade_ncesdist_edopscode(ncesdist: str = Query(..., description="NCESDist code"), edopscode: str = Query(..., description="EdOpsCode")):
    cursor.execute("SELECT MIN(T1.`Low Grade`) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.NCESDist = ? AND T2.EdOpsCode = ?", (ncesdist, edopscode))
    result = cursor.fetchone()
    if not result:
        return {"min_low_grade": []}
    return {"min_low_grade": result[0]}

# Endpoint to get EILName and School based on NSLP provision status and county code
@app.get("/v1/california_schools/eilname_school_nslp_provision_county_code", operation_id="get_eilname_school_nslp_provision_county_code", summary="Retrieves the names and schools of educational institutions in California that meet the specified NSLP provision status and county code criteria. The operation filters schools based on the provided NSLP provision status and county code, and returns the corresponding EILName and School information.")
async def get_eilname_school_nslp_provision_county_code(nslp_provision_status: str = Query(..., description="NSLP provision status"), county_code: int = Query(..., description="County code")):
    cursor.execute("SELECT T2.EILName, T2.School FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`NSLP Provision Status` = ? AND T1.`County Code` = ?", (nslp_provision_status, county_code))
    result = cursor.fetchall()
    if not result:
        return {"eilname_school": []}
    return {"eilname_school": [{"EILName": row[0], "School": row[1]} for row in result]}

# Endpoint to get city based on NSLP provision status, county, grade range, and EIL code
@app.get("/v1/california_schools/city_nslp_provision_county_grade_eilcode", operation_id="get_city_nslp_provision_county_grade_eilcode", summary="This operation retrieves the city names of schools that meet specific criteria. These criteria include a particular NSLP provision status, a specific county, a range of grades, and a specific EIL code. The operation returns a list of cities that match these conditions.")
async def get_city_nslp_provision_county_grade_eilcode(nslp_provision_status: str = Query(..., description="NSLP provision status"), county: str = Query(..., description="County"), low_grade: int = Query(..., description="Low grade"), high_grade: int = Query(..., description="High grade"), eil_code: str = Query(..., description="EIL code")):
    cursor.execute("SELECT T2.City FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`NSLP Provision Status` = ? AND T2.County = ? AND T1.`Low Grade` = ? AND T1.`High Grade` = ? AND T2.EILCode = ?", (nslp_provision_status, county, low_grade, high_grade, eil_code))
    result = cursor.fetchall()
    if not result:
        return {"cities": []}
    return {"cities": [row[0] for row in result]}

# Endpoint to get school and FRPM percentage based on county and GSserved
@app.get("/v1/california_schools/school_frpm_percentage_county_gsserved", operation_id="get_school_frpm_percentage_county_gsserved", summary="Retrieves the school names and their corresponding FRPM percentages for schools in a specified county that serve a specific grade span. The FRPM percentage is calculated as the ratio of the FRPM count to the total enrollment for ages 5-17, multiplied by 100.")
async def get_school_frpm_percentage_county_gsserved(county: str = Query(..., description="County"), gsserved: str = Query(..., description="GSserved")):
    cursor.execute("SELECT T2.School, T1.`FRPM Count (Ages 5-17)` * 100 / T1.`Enrollment (Ages 5-17)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = ? AND T2.GSserved = ?", (county, gsserved))
    result = cursor.fetchall()
    if not result:
        return {"school_frpm_percentage": []}
    return {"school_frpm_percentage": [{"School": row[0], "FRPM_Percentage": row[1]} for row in result]}

# Endpoint to get the most common GSserved in a city
@app.get("/v1/california_schools/most_common_gsserved_city", operation_id="get_most_common_gsserved_city", summary="Retrieves the most frequently served grade span (GSserved) in a specified city within California schools. The operation returns the grade span that is most commonly served in the given city, based on the data from the schools database.")
async def get_most_common_gsserved_city(city: str = Query(..., description="City")):
    cursor.execute("SELECT GSserved FROM schools WHERE City = ? GROUP BY GSserved ORDER BY COUNT(GSserved) DESC LIMIT 1", (city,))
    result = cursor.fetchone()
    if not result:
        return {"most_common_gsserved": []}
    return {"most_common_gsserved": result[0]}

# Endpoint to get the county with the most virtual schools
@app.get("/v1/california_schools/county_most_virtual_schools", operation_id="get_county_most_virtual_schools", summary="Retrieves the county with the highest number of schools offering virtual learning from a given set of two counties. The operation considers the virtual status of schools, which can be specified as 'F' for false or any other value for true.")
async def get_county_most_virtual_schools(county1: str = Query(..., description="First county"), county2: str = Query(..., description="Second county"), virtual: str = Query(..., description="Virtual status (e.g., 'F' for false)")):
    cursor.execute("SELECT County, COUNT(Virtual) FROM schools WHERE (County = ? OR County = ?) AND Virtual = ? GROUP BY County ORDER BY COUNT(Virtual) DESC LIMIT 1", (county1, county2, virtual))
    result = cursor.fetchone()
    if not result:
        return {"county_most_virtual_schools": []}
    return {"county_most_virtual_schools": {"County": result[0], "Count": result[1]}}

# Endpoint to get school type, name, and latitude of the school with the highest latitude
@app.get("/v1/california_schools/school_info_highest_latitude", operation_id="get_school_info_highest_latitude", summary="Retrieves the school type, name, and latitude of the school located furthest north in California. The operation allows you to limit the number of results returned, ensuring you receive only the most relevant information.")
async def get_school_info_highest_latitude(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T1.`School Type`, T1.`School Name`, T2.Latitude FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T2.Latitude DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"school_info": []}
    return {"school_info": result}

# Endpoint to get city, low grade, and school name of the school with the lowest latitude in a given state
@app.get("/v1/california_schools/school_info_lowest_latitude_state", operation_id="get_school_info_lowest_latitude_state", summary="Retrieves the city, lowest grade, and name of the school with the lowest latitude in a specified state. The operation allows limiting the number of results returned. The data is sourced from the 'frpm' and 'schools' tables, with the 'CDSCode' serving as the link between them.")
async def get_school_info_lowest_latitude_state(state: str = Query(..., description="State code (e.g., 'CA')"), limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T2.City, T1.`Low Grade`, T1.`School Name` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.State = ? ORDER BY T2.Latitude ASC LIMIT ?", (state, limit))
    result = cursor.fetchone()
    if not result:
        return {"school_info": []}
    return {"school_info": result}

# Endpoint to get the grade span offered by the school with the highest absolute longitude
@app.get("/v1/california_schools/grade_span_highest_absolute_longitude", operation_id="get_grade_span_highest_absolute_longitude", summary="Retrieves the grade span offered by the school located furthest east or west, based on the provided limit for the number of results returned.")
async def get_grade_span_highest_absolute_longitude(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT GSoffered FROM schools ORDER BY ABS(longitude) DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"grade_span": []}
    return {"grade_span": result[0]}

# Endpoint to get the count of schools in each city with specific criteria
@app.get("/v1/california_schools/count_schools_city_criteria", operation_id="get_count_schools_city_criteria", summary="Retrieves the number of schools in each city that meet specific criteria. These criteria include being a magnet school, offering a particular grade span, and having a certain NSLP provision status. The response will provide a breakdown of the count of schools per city that match these conditions.")
async def get_count_schools_city_criteria(magnet: int = Query(..., description="Magnet status (1 for magnet schools)"), gs_offered: str = Query(..., description="Grade span offered (e.g., 'K-8')"), nslp_provision_status: str = Query(..., description="NSLP provision status (e.g., 'Multiple Provision Types')")):
    cursor.execute("SELECT T2.City, COUNT(T2.CDSCode) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.Magnet = ? AND T2.GSoffered = ? AND T1.`NSLP Provision Status` = ? GROUP BY T2.City", (magnet, gs_offered, nslp_provision_status))
    result = cursor.fetchall()
    if not result:
        return {"counts": []}
    return {"counts": result}

# Endpoint to get distinct administrator first names and their districts for the top N administrators by count
@app.get("/v1/california_schools/top_admin_first_names_districts", operation_id="get_top_admin_first_names_districts", summary="Retrieves a list of unique administrator first names and their respective districts for the top N administrators, ranked by the count of their occurrences in the database. The 'limit' parameter allows specifying the number of top administrators to include in the result set.")
async def get_top_admin_first_names_districts(limit: int = Query(..., description="Limit the number of top administrators by count")):
    cursor.execute("SELECT DISTINCT T1.AdmFName1, T1.District FROM schools AS T1 INNER JOIN ( SELECT admfname1 FROM schools GROUP BY admfname1 ORDER BY COUNT(admfname1) DESC LIMIT ? ) AS T2 ON T1.AdmFName1 = T2.admfname1", (limit,))
    result = cursor.fetchall()
    if not result:
        return {"admin_info": []}
    return {"admin_info": result}

# Endpoint to get the percentage of free meal count to enrollment and district code for a specific administrator first name
@app.get("/v1/california_schools/free_meal_percentage_district_code", operation_id="get_free_meal_percentage_district_code", summary="Retrieves the percentage of students eligible for free meals and the district code for a specific school administrator. The calculation is based on the ratio of the total number of students eligible for free meals to the total enrollment in the school district. The administrator is identified by their first name.")
async def get_free_meal_percentage_district_code(adm_fname1: str = Query(..., description="Administrator first name")):
    cursor.execute("SELECT T1.`Free Meal Count (K-12)` * 100 / T1.`Enrollment (K-12)`, T1.`District Code` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.AdmFName1 = ?", (adm_fname1,))
    result = cursor.fetchone()
    if not result:
        return {"free_meal_percentage": []}
    return {"free_meal_percentage": result[0], "district_code": result[1]}

# Endpoint to get administrator last name, district, county, and school for a specific charter number
@app.get("/v1/california_schools/admin_info_charter_number", operation_id="get_admin_info_charter_number", summary="Retrieves the last name of the administrator, the district, the county, and the school associated with a specific charter number.")
async def get_admin_info_charter_number(charter_num: str = Query(..., description="Charter number")):
    cursor.execute("SELECT AdmLName1, District, County, School FROM schools WHERE CharterNum = ?", (charter_num,))
    result = cursor.fetchone()
    if not result:
        return {"admin_info": []}
    return {"admin_info": result}

# Endpoint to get administrator emails based on county, city, DOC, open date range, and SOC
@app.get("/v1/california_schools/admin_emails_criteria", operation_id="get_admin_emails_criteria", summary="Retrieves the primary and secondary email addresses of school administrators in a specified county, city, and DOC, within a given date range and SOC. The operation filters results based on the provided parameters, including the county and city names, DOC value, start and end years for the open date, and SOC value.")
async def get_admin_emails_criteria(county: str = Query(..., description="County name"), city: str = Query(..., description="City name"), doc: int = Query(..., description="DOC value"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), soc: int = Query(..., description="SOC value")):
    cursor.execute("SELECT T2.AdmEmail1, T2.AdmEmail2 FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = ? AND T2.City = ? AND T2.DOC = ? AND strftime('%Y', T2.OpenDate) BETWEEN ? AND ? AND T2.SOC = ?", (county, city, doc, start_year, end_year, soc))
    result = cursor.fetchone()
    if not result:
        return {"admin_emails": []}
    return {"admin_emails": result}

# Endpoint to get administrator email and school name for the school with the highest number of students scoring above 1500 in SAT
@app.get("/v1/california_schools/admin_email_school_highest_sat_score", operation_id="get_admin_email_school_highest_sat_score", summary="Retrieves the administrator email and school name for the top-performing school(s) in terms of students scoring above 1500 in the SAT. The number of results returned can be limited by specifying the 'limit' parameter.")
async def get_admin_email_school_highest_sat_score(limit: int = Query(..., description="Limit the number of results returned")):
    cursor.execute("SELECT T2.AdmEmail1, T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT ?", (limit,))
    result = cursor.fetchone()
    if not result:
        return {"admin_email_school": []}
    return {"admin_email_school": result}

api_calls = [
    "/v1/california_schools/highest_free_meal_ratio_by_county?county_name=Alameda",
    "/v1/california_schools/lowest_free_meal_ratios_by_educational_option?educational_option_type=Continuation%20School",
    "/v1/california_schools/zip_codes_by_district_and_charter_status?district_name=Fresno%20County%20Office%20of%20Education&charter_school_status=1",
    "/v1/california_schools/mailing_street_highest_frpm_count",
    "/v1/california_schools/phone_numbers_charter_schools?charter_funding_type=Directly%20funded&charter_school_status=1&open_date=2000-01-01",
    "/v1/california_schools/count_distinct_schools_virtual_math_score?virtual_status=F&avg_math_score=400",
    "/v1/california_schools/magnet_schools_test_takers?magnet_status=1&num_test_takers=500",
    "/v1/california_schools/phone_highest_ge1500_scorers",
    "/v1/california_schools/test_takers_highest_frpm_count",
    "/v1/california_schools/count_school_codes_math_score_charter_funding?avg_math_score=560&charter_funding_type=Directly%20funded",
    "/v1/california_schools/frpm_count_highest_avg_reading_score",
    "/v1/california_schools/cds_codes_total_enrollment?total_enrollment=500",
    "/v1/california_schools/max_free_meal_ratio?sat_score_ratio=0.3",
    "/v1/california_schools/phone_numbers_highest_sat_ratios?limit=3",
    "/v1/california_schools/nces_school_codes_highest_enrollment?limit=5",
    "/v1/california_schools/district_highest_avg_reading_score?status_type=Active",
    "/v1/california_schools/count_cds_codes_status_test_takers_county?status_type=Merged&num_test_takers=100&county=Lake",
    "/v1/california_schools/charter_numbers_avg_writing_scores?avg_writing_score=499",
    "/v1/california_schools/count_cds_codes_charter_funding_county_test_takers?charter_funding_type=Directly%20funded&county_name=Fresno&num_test_takers=250",
    "/v1/california_schools/phone_number_highest_avg_math_score",
    "/v1/california_schools/count_schools_by_county_grades?county=Amador&low_grade=9&high_grade=12",
    "/v1/california_schools/count_cds_codes_by_county_meal_counts?county_name=Los%20Angeles&free_meal_count=500&frpm_count=700",
    "/v1/california_schools/top_school_by_test_takers?county_name=Contra%20Costa",
    "/v1/california_schools/schools_by_enrollment_difference?enrollment_difference=30",
    "/v1/california_schools/schools_by_free_meal_ratio_sat_score?free_meal_ratio=0.1&num_ge_1500=0",
    "/v1/california_schools/schools_by_district_avg_math_score?district_name=Riverside%25&avg_math_score=400",
    "/v1/california_schools/school_details_by_county_meal_count_type?county=Monterey&free_meal_count=800&school_type=High%20Schools%20(Public)",
    "/v1/california_schools/school_details_by_open_close_dates?open_year=1991&close_year=2000",
    "/v1/california_schools/school_details_by_funding_type_enrollment_difference?funding_type=Locally%20funded",
    "/v1/california_schools/open_date_highest_enrollment",
    "/v1/california_schools/top_cities_lowest_enrollment?limit=5",
    "/v1/california_schools/free_meal_ratio_highest_enrollment?limit=9&offset=2",
    "/v1/california_schools/frpm_ratio_by_soc?soc=66&limit=5",
    "/v1/california_schools/schools_by_free_meal_count?min_count=1900&max_count=2000",
    "/v1/california_schools/free_meal_ratio_by_admin_names?adm_fname1=Kacey&adm_lname1=Gibson",
    "/v1/california_schools/admin_emails_charter_schools?charter_school=1&limit=1",
    "/v1/california_schools/admin_names_highest_sat_scores?limit=1",
    "/v1/california_schools/school_addresses_lowest_sat_ratio?limit=1",
    "/v1/california_schools/school_websites_by_test_takers_county?min_test_takers=2000&max_test_takers=3000&county=Los%20Angeles",
    "/v1/california_schools/avg_sat_test_takers_by_year_county?year=1980&county=Fresno",
    "/v1/california_schools/phone_lowest_avg_reading_score?district=Fresno%20Unified",
    "/v1/california_schools/top_schools_avg_reading_score?virtual=F&rank_limit=5",
    "/v1/california_schools/highest_avg_math_score_edops",
    "/v1/california_schools/lowest_combined_avg_scores",
    "/v1/california_schools/highest_num_ge1500_avg_writing_score",
    "/v1/california_schools/school_avg_writing_score_by_admin?adm_fname1=Ricci&adm_lname1=Ulrich",
    "/v1/california_schools/highest_enrollment_school?doc=31",
    "/v1/california_schools/avg_schools_opened_year?doc=52&county=Alameda&year=1980",
    "/v1/california_schools/ratio_schools_doc?doc1=54&doc2=52&status_type=Merged&county=Orange",
    "/v1/california_schools/closed_schools_highest_county",
    "/v1/california_schools/school_details_by_avg_math_score?limit=6&offset=1",
    "/v1/california_schools/school_details_by_avg_reading_score?limit=1&offset=0",
    "/v1/california_schools/count_schools_by_city_and_scores?mail_city=Lakeport&combined_score=1500",
    "/v1/california_schools/num_sat_test_takers_by_city?mail_city=Fresno",
    "/v1/california_schools/school_and_mail_zip_by_admin_name?adm_fname1=Avetik&adm_lname1=Atoian",
    "/v1/california_schools/ratio_of_schools_in_counties?county1=Colusa&county2=Humboldt&mail_state=CA",
    "/v1/california_schools/count_schools_by_city_state_status?city=San%20Joaquin&mail_state=CA&status_type=Active",
    "/v1/california_schools/school_phone_ext_by_avg_writing_score?limit=332&offset=1",
    "/v1/california_schools/school_details_by_zip?zip=95203-3704",
    "/v1/california_schools/school_websites_by_admin_names?adm_fname1_1=Mike&adm_lname1_1=Larson&adm_fname1_2=Dante&adm_lname1_2=Alvarez",
    "/v1/california_schools/get_website_by_county_virtual_charter?county=San%20Joaquin&virtual=P&charter=1",
    "/v1/california_schools/count_schools_by_doc_charter_city?doc=52&charter=1&city=Hickman",
    "/v1/california_schools/count_schools_by_county_charter_free_meal_percentage?county=Los%20Angeles&charter=0&free_meal_percentage=0.18",
    "/v1/california_schools/get_admin_details_by_charter_charter_num?charter=1&charter_num=00D2",
    "/v1/california_schools/count_schools_by_charter_num_mail_city?charter_num=00D4&mail_city=Hickman",
    "/v1/california_schools/percentage_funding_type_by_county_charter?funding_type=Locally%20funded&county=Santa%20Clara&charter=1",
    "/v1/california_schools/count_schools_opened_by_year_range_county_funding_type?start_year=2000&end_year=2005&county=Stanislaus&funding_type=Directly%20funded",
    "/v1/california_schools/count_schools_closed_by_year_city_doc_type?year=1989&city=San%20Francisco&doc_type=Community%20College%20District",
    "/v1/california_schools/top_county_closed_schools_by_year_range_status_soc?start_year=1980&end_year=1989&status_type=Closed&soc=11",
    "/v1/california_schools/get_nces_district_by_soc?soc=31",
    "/v1/california_schools/count_schools_status_soc_county?status_type1=Closed&status_type2=Active&soc=69&county=Alpine",
    "/v1/california_schools/district_codes_city_magnet?city=Fresno&magnet=0",
    "/v1/california_schools/enrollment_edopscode_city_academic_year?edopscode=SSS&city=Fremont&start_year=2014&end_year=2015",
    "/v1/california_schools/frpm_count_mailstreet_soctype?mail_street=PO%20Box%201040&soc_type=Youth%20Authority%20Facilities",
    "/v1/california_schools/min_low_grade_ncesdist_edopscode?ncesdist=0613360&edopscode=SPECON",
    "/v1/california_schools/eilname_school_nslp_provision_county_code?nslp_provision_status=Breakfast%20Provision%202&county_code=37",
    "/v1/california_schools/city_nslp_provision_county_grade_eilcode?nslp_provision_status=Lunch%20Provision%202&county=Merced&low_grade=9&high_grade=12&eil_code=HS",
    "/v1/california_schools/school_frpm_percentage_county_gsserved?county=Los%20Angeles&gsserved=K-9",
    "/v1/california_schools/most_common_gsserved_city?city=Adelanto",
    "/v1/california_schools/county_most_virtual_schools?county1=San%20Diego&county2=Santa%20Barbara&virtual=F",
    "/v1/california_schools/school_info_highest_latitude?limit=1",
    "/v1/california_schools/school_info_lowest_latitude_state?state=CA&limit=1",
    "/v1/california_schools/grade_span_highest_absolute_longitude?limit=1",
    "/v1/california_schools/count_schools_city_criteria?magnet=1&gs_offered=K-8&nslp_provision_status=Multiple%20Provision%20Types",
    "/v1/california_schools/top_admin_first_names_districts?limit=2",
    "/v1/california_schools/free_meal_percentage_district_code?adm_fname1=Alusine",
    "/v1/california_schools/admin_info_charter_number?charter_num=0040",
    "/v1/california_schools/admin_emails_criteria?county=San%20Bernardino&city=San%20Bernardino&doc=54&start_year=2009&end_year=2010&soc=62",
    "/v1/california_schools/admin_email_school_highest_sat_score?limit=1"
]
