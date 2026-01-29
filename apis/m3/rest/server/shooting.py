from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/shooting/shooting.sqlite')
cursor = conn.cursor()

# Endpoint to get the percentage of officers of a specific race and gender
@app.get("/v1/shooting/percentage_officers_race_gender", operation_id="get_percentage_officers_race_gender", summary="Retrieves the percentage of officers of a specific race and gender from the total number of officers. The operation requires the race and gender as input parameters to filter the data and calculate the corresponding percentage.")
async def get_percentage_officers_race_gender(race: str = Query(..., description="Race of the officer"), gender: str = Query(..., description="Gender of the officer")):
    cursor.execute("SELECT CAST(SUM(IIF(race = ?, 1, 0)) AS REAL) * 100 / COUNT(case_number) FROM officers WHERE gender = ?", (race, gender))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of incidents with more than a specified number of officers within a date range
@app.get("/v1/shooting/percentage_incidents_officer_count_date_range", operation_id="get_percentage_incidents_officer_count_date_range", summary="Get the percentage of incidents with more than a specified number of officers within a date range")
async def get_percentage_incidents_officer_count_date_range(officer_count: int = Query(..., description="Number of officers"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(IIF(officer_count > ?, 1, 0)) AS REAL) * 100 / COUNT(case_number) FROM incidents WHERE STRFTIME('%Y', date) BETWEEN ? AND ?", (officer_count, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the year with the highest number of incidents involving a specific subject weapon
@app.get("/v1/shooting/year_highest_incidents_subject_weapon", operation_id="get_year_highest_incidents_subject_weapon", summary="Retrieves the year with the highest count of incidents associated with a given subject weapon. The input parameter specifies the subject weapon for which the year is determined.")
async def get_year_highest_incidents_subject_weapon(subject_weapon: str = Query(..., description="Subject weapon")):
    cursor.execute("SELECT STRFTIME('%Y', date) FROM incidents WHERE subject_weapon = ? GROUP BY STRFTIME('%Y', date) ORDER BY COUNT(case_number) DESC LIMIT 1", (subject_weapon,))
    result = cursor.fetchone()
    if not result:
        return {"year": []}
    return {"year": result[0]}

# Endpoint to get the percentage of incidents with a specific subject status and grand jury disposition
@app.get("/v1/shooting/percentage_incidents_subject_status_grand_jury_disposition", operation_id="get_percentage_incidents_subject_status_grand_jury_disposition", summary="Get the percentage of incidents with a specific subject status and grand jury disposition")
async def get_percentage_incidents_subject_status_grand_jury_disposition(subject_status: str = Query(..., description="Subject status"), grand_jury_disposition: str = Query(..., description="Grand jury disposition")):
    cursor.execute("SELECT CAST(SUM(subject_statuses = ?) AS REAL) * 100 / COUNT(case_number) FROM incidents WHERE grand_jury_disposition = ?", (subject_status, grand_jury_disposition))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the difference in the number of incidents involving a specific subject weapon between two years
@app.get("/v1/shooting/difference_incidents_subject_weapon_years", operation_id="get_difference_incidents_subject_weapon_years", summary="Retrieve the difference in the count of incidents involving a specific subject weapon between two input years. The endpoint calculates the total number of incidents for each year and returns the difference.")
async def get_difference_incidents_subject_weapon_years(year1: str = Query(..., description="First year in 'YYYY' format"), year2: str = Query(..., description="Second year in 'YYYY' format"), subject_weapon: str = Query(..., description="Subject weapon")):
    cursor.execute("SELECT SUM(IIF(STRFTIME('%Y', date) = ?, 1, 0)) - SUM(IIF(STRFTIME('%Y', date) = ?, 1, 0)) FROM incidents WHERE subject_weapon = ?", (year1, year2, subject_weapon))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the percentage of incidents with a specific subject status and subject weapon
@app.get("/v1/shooting/percentage_incidents_subject_status_subject_weapon", operation_id="get_percentage_incidents_subject_status_subject_weapon", summary="Retrieves the percentage of incidents where the subject's status matches the provided status and the subject was using the specified weapon. The calculation is based on the total number of incidents involving the given weapon.")
async def get_percentage_incidents_subject_status_subject_weapon(subject_status: str = Query(..., description="Subject status"), subject_weapon: str = Query(..., description="Subject weapon")):
    cursor.execute("SELECT CAST(SUM(subject_statuses = ?) AS REAL) * 100 / COUNT(case_number) FROM incidents WHERE subject_weapon = ?", (subject_status, subject_weapon))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the last name and gender of officers involved in incidents with a specific grand jury disposition
@app.get("/v1/shooting/officers_last_name_gender_grand_jury_disposition", operation_id="get_officers_last_name_gender_grand_jury_disposition", summary="Get the last name and gender of officers involved in incidents with a specific grand jury disposition")
async def get_officers_last_name_gender_grand_jury_disposition(grand_jury_disposition: str = Query(..., description="Grand jury disposition")):
    cursor.execute("SELECT T2.last_name, T2.gender FROM incidents AS T1 INNER JOIN officers AS T2 ON T1.case_number = T2.case_number WHERE T1.grand_jury_disposition = ?", (grand_jury_disposition,))
    result = cursor.fetchall()
    if not result:
        return {"officers": []}
    return {"officers": result}

# Endpoint to get the case number, subject statuses, and subject weapon of incidents involving subjects of a specific gender
@app.get("/v1/shooting/incidents_subject_gender", operation_id="get_incidents_subject_gender", summary="Retrieves the case number, subject statuses, and subject weapon for incidents involving subjects of a specified gender. The gender parameter is used to filter the incidents by the gender of the involved subjects.")
async def get_incidents_subject_gender(gender: str = Query(..., description="Gender of the subject")):
    cursor.execute("SELECT T1.case_number, T1.subject_statuses, T1.subject_weapon FROM incidents AS T1 INNER JOIN subjects AS T2 ON T1.case_number = T2.case_number WHERE T2.gender = ?", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"incidents": []}
    return {"incidents": result}

# Endpoint to get the case number, location, and subject statuses of incidents involving subjects of a specific gender
@app.get("/v1/shooting/incidents_location_subject_gender", operation_id="get_incidents_location_subject_gender", summary="Retrieves the case number, location, and subject statuses of incidents involving subjects of a specified gender. The gender parameter is used to filter the incidents by the gender of the involved subjects.")
async def get_incidents_location_subject_gender(gender: str = Query(..., description="Gender of the subject")):
    cursor.execute("SELECT T1.case_number, T1.location, T1.subject_statuses FROM incidents AS T1 INNER JOIN subjects AS T2 ON T1.case_number = T2.case_number WHERE T2.gender = ?", (gender,))
    result = cursor.fetchall()
    if not result:
        return {"incidents": []}
    return {"incidents": result}

# Endpoint to get the case number and grand jury disposition of incidents involving officers with a specific first and last name
@app.get("/v1/shooting/incidents_officer_name", operation_id="get_incidents_officer_name", summary="Get the case number and grand jury disposition of incidents involving officers with a specific first and last name")
async def get_incidents_officer_name(first_name: str = Query(..., description="First name of the officer"), last_name: str = Query(..., description="Last name of the officer")):
    cursor.execute("SELECT T1.case_number, T1.grand_jury_disposition FROM incidents AS T1 INNER JOIN officers AS T2 ON T1.case_number = T2.case_number WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"incidents": []}
    return {"incidents": result}

# Endpoint to get the last name and subject statuses of officers involved in a specific case
@app.get("/v1/shooting/officer_details_by_case_number", operation_id="get_officer_details_by_case_number", summary="Retrieves the last names and subject statuses of officers associated with a given case number. This operation provides a concise summary of the officers involved in a specific incident, enabling users to understand their roles and statuses in the case.")
async def get_officer_details_by_case_number(case_number: str = Query(..., description="Case number of the incident")):
    cursor.execute("SELECT T2.last_name, T1.subject_statuses FROM incidents AS T1 INNER JOIN officers AS T2 ON T1.case_number = T2.case_number WHERE T1.case_number = ?", (case_number,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get details of subjects with a specific status
@app.get("/v1/shooting/subject_details_by_status", operation_id="get_subject_details_by_status", summary="Retrieves the last name, gender, race, and case number of subjects who have a specified status. The status is used to filter the subjects and return only those who match the provided status.")
async def get_subject_details_by_status(subject_statuses: str = Query(..., description="Status of the subject")):
    cursor.execute("SELECT T2.last_name, T2.gender, T2.race, T2.case_number FROM incidents AS T1 INNER JOIN subjects AS T2 ON T1.case_number = T2.case_number WHERE T1.subject_statuses = ?", (subject_statuses,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the percentage of incidents involving a specific weapon and gender
@app.get("/v1/shooting/percentage_weapon_gender", operation_id="get_percentage_weapon_gender", summary="Retrieves the percentage of incidents involving a specific weapon and gender. The operation calculates this percentage by comparing the count of incidents where the subject used a particular weapon to the total number of incidents. The gender of the subject is also considered in the calculation.")
async def get_percentage_weapon_gender(subject_weapon: str = Query(..., description="Weapon used by the subject"), gender: str = Query(..., description="Gender of the subject")):
    cursor.execute("SELECT CAST(SUM(T1.subject_weapon = ?) AS REAL) * 100 / COUNT(T1.case_number) FROM incidents T1 INNER JOIN subjects T2 ON T1.case_number = T2.case_number WHERE T2.gender = ?", (subject_weapon, gender))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of incidents involving a specific weapon compared to another weapon for a specific subject status
@app.get("/v1/shooting/percentage_weapon_comparison", operation_id="get_percentage_weapon_comparison", summary="Retrieves the percentage of incidents involving a specific subject status where a given weapon was used compared to another weapon. This operation calculates the proportion of incidents where the first weapon was used out of all incidents involving the second weapon and the specified subject status.")
async def get_percentage_weapon_comparison(subject_weapon_1: str = Query(..., description="First weapon used by the subject"), subject_weapon_2: str = Query(..., description="Second weapon used by the subject"), subject_statuses: str = Query(..., description="Status of the subject")):
    cursor.execute("SELECT CAST(SUM(T1.subject_weapon = ?) AS REAL) * 100 / SUM(T1.subject_weapon = ?) FROM incidents AS T1 INNER JOIN subjects AS T2 ON T1.case_number = T2.case_number WHERE T1.subject_statuses = ?", (subject_weapon_1, subject_weapon_2, subject_statuses))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get case numbers of incidents after a specific year with a specific subject status
@app.get("/v1/shooting/case_numbers_by_year_status", operation_id="get_case_numbers_by_year_status", summary="Retrieves the case numbers of shooting incidents that occurred after a specified year and involved subjects with a particular status. The year is provided in 'YYYY' format, and the subject status is a specific condition or category.")
async def get_case_numbers_by_year_status(year: str = Query(..., description="Year in 'YYYY' format"), subject_statuses: str = Query(..., description="Status of the subject")):
    cursor.execute("SELECT case_number FROM incidents WHERE STRFTIME('%Y', date) > ? AND subject_statuses = ?", (year, subject_statuses))
    result = cursor.fetchall()
    if not result:
        return {"case_numbers": []}
    return {"case_numbers": [item[0] for item in result]}

# Endpoint to get the count of incidents involving a specific weapon and officer gender
@app.get("/v1/shooting/count_incidents_weapon_officer_gender", operation_id="get_count_incidents_weapon_officer_gender", summary="Retrieves the total number of incidents involving a specific weapon used by a subject and the gender of the involved officer. The response provides a quantitative measure of incidents that meet the specified criteria.")
async def get_count_incidents_weapon_officer_gender(subject_weapon: str = Query(..., description="Weapon used by the subject"), gender: str = Query(..., description="Gender of the officer")):
    cursor.execute("SELECT COUNT(T1.case_number) FROM incidents AS T1 INNER JOIN officers AS T2 ON T1.case_number = T2.case_number WHERE T1.subject_weapon = ? AND T2.gender = ?", (subject_weapon, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of incidents involving a specific subject gender and status
@app.get("/v1/shooting/count_incidents_subject_gender_status", operation_id="get_count_incidents_subject_gender_status", summary="Retrieves the total number of incidents involving subjects of a specific gender and status. The gender and status of the subjects are used to filter the incidents and calculate the count.")
async def get_count_incidents_subject_gender_status(gender: str = Query(..., description="Gender of the subject"), subject_statuses: str = Query(..., description="Status of the subject")):
    cursor.execute("SELECT COUNT(T1.case_number) FROM incidents AS T1 INNER JOIN subjects AS T2 ON T1.case_number = T2.case_number WHERE T2.gender = ? AND T1.subject_statuses = ?", (gender, subject_statuses))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of incidents involving officers of a specific race within a date range
@app.get("/v1/shooting/count_incidents_officer_race_date_range", operation_id="get_count_incidents_officer_race_date_range", summary="Retrieves the total number of incidents involving officers of a specified race that occurred within a given date range. The date range is inclusive and should be provided in 'YYYY-MM-DD' format.")
async def get_count_incidents_officer_race_date_range(race: str = Query(..., description="Race of the officer"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T1.case_number) FROM officers AS T1 INNER JOIN incidents AS T2 ON T2.case_number = T1.case_number WHERE T1.race = ? AND T2.date BETWEEN ? AND ?", (race, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of incidents within a date range
@app.get("/v1/shooting/count_incidents_date_range", operation_id="get_count_incidents_date_range", summary="Retrieves the total number of incidents that occurred between the specified start and end dates. The dates should be provided in 'YYYY-MM-DD' format.")
async def get_count_incidents_date_range(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(case_number) FROM incidents WHERE date BETWEEN ? AND ?", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of incidents based on subject weapon, subject status, and date range
@app.get("/v1/shooting/incident_count_by_weapon_status_date", operation_id="get_incident_count", summary="Retrieves the total number of incidents that match the specified subject weapon, subject status, and date range. The response is based on the count of incident locations that meet the provided criteria.")
async def get_incident_count(subject_weapon: str = Query(..., description="Subject weapon"), subject_statuses: str = Query(..., description="Subject statuses"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(location) FROM incidents WHERE subject_weapon = ? AND subject_statuses = ? AND date BETWEEN ? AND ?", (subject_weapon, subject_statuses, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common subject weapon for deceased subjects
@app.get("/v1/shooting/most_common_weapon_deceased", operation_id="get_most_common_weapon", summary="Retrieves the most frequently used weapon by deceased subjects in shooting incidents. The operation filters incidents based on the provided subject statuses and returns the weapon that appears most frequently in these incidents.")
async def get_most_common_weapon(subject_statuses: str = Query(..., description="Subject statuses")):
    cursor.execute("SELECT subject_weapon FROM incidents WHERE subject_statuses = ? GROUP BY subject_weapon ORDER BY COUNT(case_number) DESC LIMIT 1", (subject_statuses,))
    result = cursor.fetchone()
    if not result:
        return {"subject_weapon": []}
    return {"subject_weapon": result[0]}

# Endpoint to get the ratio of male to female officers of a specific race
@app.get("/v1/shooting/male_to_female_ratio_by_race", operation_id="get_male_to_female_ratio", summary="Retrieves the ratio of male to female officers for a specified race. This operation calculates the proportion of male officers to female officers based on the provided gender parameters and race. The result is a numerical value representing the ratio of male to female officers of the given race.")
async def get_male_to_female_ratio(gender_m: str = Query(..., description="Gender for male"), gender_f: str = Query(..., description="Gender for female"), race: str = Query(..., description="Race of the officers")):
    cursor.execute("SELECT CAST(SUM(gender = ?) AS REAL) / SUM(gender = ?) FROM officers WHERE race = ?", (gender_m, gender_f, race))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the difference in counts of two races for a specific gender
@app.get("/v1/shooting/race_difference_by_gender", operation_id="get_race_difference", summary="Retrieves the difference in the number of subjects between two specified races for a given gender. The response indicates the net count difference, which can be positive, negative, or zero.")
async def get_race_difference(race1: str = Query(..., description="First race"), race2: str = Query(..., description="Second race"), gender: str = Query(..., description="Gender of the subjects")):
    cursor.execute("SELECT SUM(race = ?) - SUM(race = ?) FROM subjects WHERE gender = ?", (race1, race2, gender))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the subject weapon, race, and gender for a specific case number
@app.get("/v1/shooting/subject_details_by_case_number", operation_id="get_subject_details", summary="Retrieves the weapon used by the subject, as well as the subject's race and gender, for a given case number. This operation requires the case number as an input parameter to accurately fetch the requested details.")
async def get_subject_details(case_number: str = Query(..., description="Case number")):
    cursor.execute("SELECT T1.subject_weapon, T2.race, T2.gender FROM incidents AS T1 INNER JOIN subjects AS T2 ON T1.case_number = T2.case_number WHERE T1.case_number = ?", (case_number,))
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"subject_weapon": result[0], "race": result[1], "gender": result[2]}

# Endpoint to get the case number, race, and gender for a specific officer's name
@app.get("/v1/shooting/case_details_by_officer_name", operation_id="get_case_details", summary="Retrieves the case number, race, and gender of subjects involved in cases where the specified officer was present. The officer is identified by their first and last names.")
async def get_case_details(first_name: str = Query(..., description="First name of the officer"), last_name: str = Query(..., description="Last name of the officer")):
    cursor.execute("SELECT T1.case_number, T3.race, T3.gender FROM incidents AS T1 INNER JOIN officers AS T2 ON T1.case_number = T2.case_number INNER JOIN subjects AS T3 ON T1.case_number = T3.case_number WHERE T2.first_name = ? AND T2.last_name = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"details": []}
    return {"case_number": result[0], "race": result[1], "gender": result[2]}

# Endpoint to get the percentage of male officers in incidents with a specific subject status
@app.get("/v1/shooting/percentage_male_officers_by_subject_status", operation_id="get_percentage_male_officers", summary="Retrieves the percentage of male officers involved in incidents with a specific subject status. The calculation is based on the total number of incidents with the given subject status and the count of male officers in those incidents.")
async def get_percentage_male_officers(gender: str = Query(..., description="Gender of the officers"), subject_statuses: str = Query(..., description="Subject statuses")):
    cursor.execute("SELECT CAST(SUM(T2.gender = ?) AS REAL) * 100 / COUNT(T1.case_number) FROM incidents T1 INNER JOIN officers T2 ON T1.case_number = T2.case_number WHERE T1.subject_statuses = ?", (gender, subject_statuses))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/shooting/percentage_officers_race_gender?race=W&gender=M",
    "/v1/shooting/percentage_incidents_officer_count_date_range?officer_count=3&start_year=2010&end_year=2015",
    "/v1/shooting/year_highest_incidents_subject_weapon?subject_weapon=Handgun",
    "/v1/shooting/percentage_incidents_subject_status_grand_jury_disposition?subject_status=Injured&grand_jury_disposition=No%20Bill",
    "/v1/shooting/difference_incidents_subject_weapon_years?year1=2007&year2=2008&subject_weapon=Vehicle",
    "/v1/shooting/percentage_incidents_subject_status_subject_weapon?subject_status=Shoot%20and%20Miss&subject_weapon=Handgun",
    "/v1/shooting/officers_last_name_gender_grand_jury_disposition?grand_jury_disposition=No%20Bill",
    "/v1/shooting/incidents_subject_gender?gender=F",
    "/v1/shooting/incidents_location_subject_gender?gender=M",
    "/v1/shooting/incidents_officer_name?first_name=George&last_name=Evenden",
    "/v1/shooting/officer_details_by_case_number?case_number=134472-2015",
    "/v1/shooting/subject_details_by_status?subject_statuses=Deceased",
    "/v1/shooting/percentage_weapon_gender?subject_weapon=Vehicle&gender=F",
    "/v1/shooting/percentage_weapon_comparison?subject_weapon_1=Knife&subject_weapon_2=Handgun&subject_statuses=Injured",
    "/v1/shooting/case_numbers_by_year_status?year=2011&subject_statuses=Deceased",
    "/v1/shooting/count_incidents_weapon_officer_gender?subject_weapon=Vehicle&gender=F",
    "/v1/shooting/count_incidents_subject_gender_status?gender=F&subject_statuses=Deceased",
    "/v1/shooting/count_incidents_officer_race_date_range?race=B&start_date=2010-01-01&end_date=2015-12-31",
    "/v1/shooting/count_incidents_date_range?start_date=2015-06-01&end_date=2015-06-30",
    "/v1/shooting/incident_count_by_weapon_status_date?subject_weapon=Handgun&subject_statuses=Injured&start_date=2006-01-01&end_date=2013-12-31",
    "/v1/shooting/most_common_weapon_deceased?subject_statuses=Deceased",
    "/v1/shooting/male_to_female_ratio_by_race?gender_m=M&gender_f=F&race=W",
    "/v1/shooting/race_difference_by_gender?race1=B&race2=W&gender=F",
    "/v1/shooting/subject_details_by_case_number?case_number=031347-2015",
    "/v1/shooting/case_details_by_officer_name?first_name=Fredirick&last_name=Ruben",
    "/v1/shooting/percentage_male_officers_by_subject_status?gender=M&subject_statuses=Injured"
]
