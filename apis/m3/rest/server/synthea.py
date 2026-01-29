from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/synthea/synthea.sqlite')
cursor = conn.cursor()

# Endpoint to get the value and units of observations for a specific patient on a specific date and description
@app.get("/v1/synthea/observations_value_units", operation_id="get_observations_value_units", summary="Retrieves the value and units of a specific observation for a patient, identified by their first and last names, on a given date and description. The observation details returned correspond to the provided patient and observation criteria.")
async def get_observations_value_units(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), date: str = Query(..., description="Date of the observation in 'YYYY-MM-DD' format"), description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT T2.value, T2.units FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.date = ? AND T2.description = ?", (first, last, date, description))
    result = cursor.fetchall()
    if not result:
        return {"observations": []}
    return {"observations": result}

# Endpoint to get the increase in observation value between two years for a specific patient and description
@app.get("/v1/synthea/observations_increase", operation_id="get_observations_increase", summary="Retrieve the difference in observation values for a specific patient and description between two given years. The operation calculates the total observation value for each year and returns the increase or decrease in value between them. The result is presented in the units of the observation.")
async def get_observations_increase(year1: str = Query(..., description="First year in 'YYYY' format"), year2: str = Query(..., description="Second year in 'YYYY' format"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT SUM(CASE WHEN strftime('%Y', T2.date) = ? THEN T2.VALUE END) - SUM(CASE WHEN strftime('%Y', T2.date) = ? THEN T2.VALUE END) AS increase , T2.units FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.description = ?", (year1, year2, first, last, description))
    result = cursor.fetchall()
    if not result:
        return {"increase": []}
    return {"increase": result}

# Endpoint to get the highest value and units of observations for a specific patient and description
@app.get("/v1/synthea/observations_max_value", operation_id="get_observations_max_value", summary="Retrieves the maximum value and corresponding units of a specific observation for a patient identified by their first and last names. The observation is determined by the provided description.")
async def get_observations_max_value(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT T2.value, T2.units FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.description = ? ORDER BY T2.VALUE DESC LIMIT 1", (first, last, description))
    result = cursor.fetchone()
    if not result:
        return {"observations": []}
    return {"observations": result}

# Endpoint to get the count of observations for a specific patient and description
@app.get("/v1/synthea/observations_count", operation_id="get_observations_count", summary="Retrieves the total count of a specific observation for a patient, identified by their first and last names. This operation is useful for tracking the frequency of a particular observation for a given patient.")
async def get_observations_count(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT COUNT(T2.description) FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.description = ?", (first, last, description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the patient with the highest value for a specific observation description
@app.get("/v1/synthea/patient_max_observation_value", operation_id="get_patient_max_observation_value", summary="Retrieves the patient with the highest recorded value for a specified observation. The observation description is provided as an input parameter to identify the relevant data.")
async def get_patient_max_observation_value(description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT T1.first, T1.last FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T2.VALUE = ( SELECT MAX(VALUE) FROM observations WHERE description = ? ) LIMIT 1", (description,))
    result = cursor.fetchone()
    if not result:
        return {"patient": []}
    return {"patient": result}

# Endpoint to get the duration of medication for a specific patient and medication description
@app.get("/v1/synthea/medication_duration", operation_id="get_medication_duration", summary="Retrieves the duration in days of a specific medication for a patient, based on their first and last names and a partial medication description. The medication duration is calculated as the difference between the medication's stop and start dates.")
async def get_medication_duration(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the medication")):
    cursor.execute("SELECT strftime('%J', T2.STOP) - strftime('%J', T2.START) AS days FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.description LIKE ?", (first, last, description))
    result = cursor.fetchall()
    if not result:
        return {"duration": []}
    return {"duration": result}

# Endpoint to get distinct medication descriptions for a specific patient
@app.get("/v1/synthea/patient_medications", operation_id="get_patient_medications", summary="Retrieve unique medication descriptions associated with a patient identified by their first and last names.")
async def get_patient_medications(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT DISTINCT T2.description FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first, last))
    result = cursor.fetchall()
    if not result:
        return {"medications": []}
    return {"medications": result}

# Endpoint to get the reason descriptions for a specific patient and medication description
@app.get("/v1/synthea/medication_reasons", operation_id="get_medication_reasons", summary="Retrieves the reason descriptions associated with a specific patient and medication. The operation requires the first and last name of the patient, as well as a description of the medication. It returns the reason descriptions that match the provided medication description for the specified patient.")
async def get_medication_reasons(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the medication")):
    cursor.execute("SELECT T2.REASONDESCRIPTION FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.description LIKE ?", (first, last, description))
    result = cursor.fetchall()
    if not result:
        return {"reasons": []}
    return {"reasons": result}

# Endpoint to get the medication descriptions for a specific patient and reason description
@app.get("/v1/synthea/medications_by_reason", operation_id="get_medications_by_reason", summary="Retrieves the descriptions of medications prescribed to a specific patient for a given reason. The operation requires the patient's first and last names, as well as the description of the reason for the medication. The returned data includes the descriptions of the medications that match the provided criteria.")
async def get_medications_by_reason(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), reasondescription: str = Query(..., description="Reason description for the medication")):
    cursor.execute("SELECT T2.description FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.reasondescription = ?", (first, last, reasondescription))
    result = cursor.fetchall()
    if not result:
        return {"medications": []}
    return {"medications": result}

# Endpoint to get distinct patients for a specific medication description
@app.get("/v1/synthea/patients_by_medication", operation_id="get_patients_by_medication", summary="Retrieve a unique list of patients who have been prescribed a medication with a specific description. The description parameter is used to filter the patients based on their medication records.")
async def get_patients_by_medication(description: str = Query(..., description="Description of the medication")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.description LIKE ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": result}

# Endpoint to get the description of conditions for a patient with a specific first name, last name, and start date
@app.get("/v1/synthea/patient_condition_description", operation_id="get_patient_condition_description", summary="Retrieves the description of conditions associated with a patient identified by their first name, last name, and start date. The operation returns a list of condition descriptions that match the provided patient details.")
async def get_patient_condition_description(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.description FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.START = ?", (first, last, start_date))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the number of days between the start and stop dates of a condition for a patient with a specific first name, last name, and condition description
@app.get("/v1/synthea/condition_duration", operation_id="get_condition_duration", summary="Retrieves the duration in days between the start and stop dates of a specific condition for a patient identified by their first and last names. The condition is determined by its description.")
async def get_condition_duration(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT strftime('%J', T2.STOP) - strftime('%J', T2.START) AS days FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.description = ?", (first, last, description))
    result = cursor.fetchone()
    if not result:
        return {"days": []}
    return {"days": result[0]}

# Endpoint to get the average value and units of observations for a patient with a specific first name, last name, and observation description
@app.get("/v1/synthea/average_observation_value", operation_id="get_average_observation_value", summary="Retrieves the average value and corresponding units of a specific observation for a patient identified by their first and last name. The operation filters the patient records based on the provided first and last names, and further narrows down the observations to the one matching the given description. The average value and units of this observation are then calculated and returned.")
async def get_average_observation_value(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT AVG(T2.VALUE), T2.units FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.description = ?", (first, last, description))
    result = cursor.fetchone()
    if not result:
        return {"average_value": [], "units": []}
    return {"average_value": result[0], "units": result[1]}

# Endpoint to get the percentage of married patients with a specific condition description
@app.get("/v1/synthea/married_patients_percentage", operation_id="get_married_patients_percentage", summary="Retrieves the percentage of patients who are married and have a specific medical condition. The condition is identified by its description. The calculation is based on the total number of unique patients with the specified condition.")
async def get_married_patients_percentage(description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.marital = 'M' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T2.description = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the description, value, and units of observations for a patient with a specific prefix, first name, last name, date, and observation description
@app.get("/v1/synthea/patient_observation_details", operation_id="get_patient_observation_details", summary="Retrieves the description, value, and units of a specific observation for a patient identified by their prefix, first name, last name, and observation date. The observation is further specified by its description.")
async def get_patient_observation_details(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT T2.description, T2.VALUE, T2.units FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ? AND T2.date = ? AND T2.description = ?", (prefix, first, last, date, description))
    result = cursor.fetchall()
    if not result:
        return {"observations": []}
    return {"observations": [{"description": row[0], "value": row[1], "units": row[2]} for row in result]}

# Endpoint to get the count of care plans for a patient with a specific prefix, first name, and last name
@app.get("/v1/synthea/care_plan_count", operation_id="get_care_plan_count", summary="Retrieves the total number of care plans associated with a patient identified by a specific prefix, first name, and last name.")
async def get_care_plan_count(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT COUNT(T2.PATIENT) FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ?", (prefix, first, last))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the reason description for medications for a patient with a specific prefix, first name, last name, start date, and medication description
@app.get("/v1/synthea/medication_reason_description", operation_id="get_medication_reason_description", summary="Retrieves the reason description for a specific medication given to a patient identified by their prefix, first name, last name, medication start date, and medication description. The response includes the reason description associated with the medication.")
async def get_medication_reason_description(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), description: str = Query(..., description="Description of the medication")):
    cursor.execute("SELECT T2.reasondescription FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ? AND T2.start = ? AND T2.description = ?", (prefix, first, last, start_date, description))
    result = cursor.fetchall()
    if not result:
        return {"reason_descriptions": []}
    return {"reason_descriptions": [row[0] for row in result]}

# Endpoint to get the distinct prevalence percentages for a condition with a specific code
@app.get("/v1/synthea/prevalence_percentage", operation_id="get_prevalence_percentage", summary="Retrieve the unique prevalence percentages associated with a specific medical condition, identified by its code. This operation returns a list of distinct prevalence percentages, which can be used to understand the condition's prevalence distribution.")
async def get_prevalence_percentage(code: str = Query(..., description="Code of the condition")):
    cursor.execute("SELECT DISTINCT T1.\"PREVALENCE PERCENTAGE\" FROM all_prevalences AS T1 INNER JOIN conditions AS T2 ON lower(T1.ITEM) = lower(T2.DESCRIPTION) WHERE T2.code = ?", (code,))
    result = cursor.fetchall()
    if not result:
        return {"prevalence_percentages": []}
    return {"prevalence_percentages": [row[0] for row in result]}

# Endpoint to get the distinct prevalence rates for a condition with a specific code
@app.get("/v1/synthea/prevalence_rate", operation_id="get_prevalence_rate", summary="Retrieve the unique prevalence rates associated with a specific medical condition, identified by its code. This operation returns a list of distinct prevalence rates, which are determined by matching the condition's description with the corresponding entries in the all_prevalences table.")
async def get_prevalence_rate(code: str = Query(..., description="Code of the condition")):
    cursor.execute("SELECT DISTINCT T1.\"PREVALENCE RATE\" FROM all_prevalences AS T1 INNER JOIN conditions AS T2 ON lower(T1.ITEM) = lower(T2.DESCRIPTION) WHERE T2.code = ?", (code,))
    result = cursor.fetchall()
    if not result:
        return {"prevalence_rates": []}
    return {"prevalence_rates": [row[0] for row in result]}

# Endpoint to get the distinct procedure descriptions for a patient with a specific prefix, first name, last name, and date
@app.get("/v1/synthea/patient_procedure_descriptions", operation_id="get_patient_procedure_descriptions", summary="Retrieve the unique procedure descriptions associated with a specific patient, identified by their prefix, first name, last name, and a given date. This operation returns a list of distinct procedure descriptions that match the provided patient and date criteria.")
async def get_patient_procedure_descriptions(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT DISTINCT T2.description FROM patients AS T1 INNER JOIN procedures AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ? AND T2.DATE = ?", (prefix, first, last, date))
    result = cursor.fetchall()
    if not result:
        return {"procedure_descriptions": []}
    return {"procedure_descriptions": [row[0] for row in result]}

# Endpoint to get the count of billable periods for a patient with specific name and date range
@app.get("/v1/synthea/count_billable_periods", operation_id="get_count_billable_periods", summary="Retrieves the total number of billable periods for a patient, based on their full name and a specified date range. The patient's name is composed of a prefix, first name, and last name. The date range is defined by a start and end date, both in 'YYYY-MM-DD' format.")
async def get_count_billable_periods(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(T2.BILLABLEPERIOD) FROM patients AS T1 INNER JOIN claims AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ? AND T2.BILLABLEPERIOD BETWEEN ? AND ?", (prefix, first, last, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct allergy codes for a patient with specific name
@app.get("/v1/synthea/count_distinct_allergy_codes", operation_id="get_count_distinct_allergy_codes", summary="Retrieves the total number of unique allergy codes associated with a patient identified by a specific prefix, first name, and last name.")
async def get_count_distinct_allergy_codes(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T2.code) FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ?", (prefix, first, last))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last names of patients with claims on a specific billable period
@app.get("/v1/synthea/patient_names_by_billable_period", operation_id="get_patient_names_by_billable_period", summary="Retrieves the first and last names of patients who have claims associated with a specified billable period. The billable period must be provided in 'YYYY-MM-DD' format.")
async def get_patient_names_by_billable_period(billable_period: str = Query(..., description="Billable period in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.first, T1.last FROM patients AS T1 INNER JOIN claims AS T2 ON T1.patient = T2.PATIENT WHERE T2.billableperiod = ?", (billable_period,))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": result}

# Endpoint to get the encounter descriptions for a patient with specific name and date
@app.get("/v1/synthea/encounter_descriptions", operation_id="get_encounter_descriptions", summary="Retrieve the descriptions of encounters for a patient with a specific name and date. The operation uses the provided patient's prefix, first name, last name, and encounter date to filter the results. The encounter descriptions returned correspond to the matching patient and date.")
async def get_encounter_descriptions(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), encounter_date: str = Query(..., description="Encounter date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.description FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ? AND T2.date = ?", (prefix, first, last, encounter_date))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get the condition descriptions for a patient with specific name and condition period
@app.get("/v1/synthea/condition_descriptions", operation_id="get_condition_descriptions", summary="Retrieves the descriptions of conditions for a patient identified by their first and last names and the period during which the conditions were active. The condition descriptions are fetched from the database based on the provided patient details and condition period.")
async def get_condition_descriptions(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), start_date: str = Query(..., description="Start date of the condition in 'YYYY-MM-DD' format"), stop_date: str = Query(..., description="Stop date of the condition in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.description FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.start = ? AND T2.stop = ?", (first, last, start_date, stop_date))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get the procedure dates for a patient with specific name and procedure description
@app.get("/v1/synthea/procedure_dates", operation_id="get_procedure_dates", summary="Retrieves the dates of a specific procedure for a patient identified by their full name. The procedure is determined by its description. This operation returns a list of dates when the procedure was performed for the specified patient.")
async def get_procedure_dates(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the procedure")):
    cursor.execute("SELECT T2.date FROM patients AS T1 INNER JOIN procedures AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ? AND T2.description = ?", (prefix, first, last, description))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": result}

# Endpoint to get the average duration of care plans for a patient with specific name
@app.get("/v1/synthea/average_careplan_duration", operation_id="get_average_careplan_duration", summary="Retrieves the average duration of care plans for a patient identified by a specific prefix, first name, and last name. The calculation is based on the difference between the start and stop dates of each care plan associated with the patient.")
async def get_average_careplan_duration(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT CAST(SUM(strftime('%J', T2.STOP) - strftime('%J', T2.START)) AS REAL) / COUNT(T1.patient) FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ?", (prefix, first, last))
    result = cursor.fetchone()
    if not result:
        return {"average_duration": []}
    return {"average_duration": result[0]}

# Endpoint to get the average duration of conditions for a patient with specific name and condition description
@app.get("/v1/synthea/average_condition_duration", operation_id="get_average_condition_duration", summary="Retrieves the average duration of a specific condition for a patient identified by their full name. The calculation is based on the start and stop dates of the condition occurrences for the patient.")
async def get_average_condition_duration(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT CAST(SUM(strftime('%J', T2.STOP) - strftime('%J', T2.START)) AS REAL) / COUNT(T2.PATIENT) FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ? AND T2.description = ?", (prefix, first, last, description))
    result = cursor.fetchone()
    if not result:
        return {"average_duration": []}
    return {"average_duration": result[0]}

# Endpoint to get the patient with the most recent allergy stop date
@app.get("/v1/synthea/most_recent_allergy_stop", operation_id="get_most_recent_allergy_stop", summary="Retrieves the patient who has the most recent allergy stop date. This operation considers the year of the stop date and the year of the start date to determine the most recent allergy stop. The patient with the longest duration between the start and stop dates is returned if multiple patients have the same stop year.")
async def get_most_recent_allergy_stop():
    cursor.execute("SELECT PATIENT FROM allergies WHERE STOP IS NOT NULL GROUP BY PATIENT ORDER BY CASE WHEN SUBSTR(STOP, -2, 1) != '9' THEN SUBSTR(STOP, LENGTH(STOP) - 1) + 2000 END - CASE WHEN SUBSTR(START, -2, 1) = '9' THEN SUBSTR(START, LENGTH(START) - 1) + 1900 ELSE SUBSTR(START, LENGTH(START) - 1) + 2000 END LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"patient": []}
    return {"patient": result[0]}

# Endpoint to get the count of patients with a specific condition and start year
@app.get("/v1/synthea/count_patients_by_condition_and_year", operation_id="get_count_patients_by_condition_and_year", summary="Retrieves the total number of patients who have a specific medical condition and whose condition started in a given year. The condition is identified by its description, and the start year is provided in 'YYYY' format.")
async def get_count_patients_by_condition_and_year(description: str = Query(..., description="Description of the condition"), start_year: str = Query(..., description="Start year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(PATIENT) FROM conditions WHERE DESCRIPTION = ? AND strftime('%Y', START) = ?", (description, start_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients with a specific allergy description
@app.get("/v1/synthea/allergy_count", operation_id="get_allergy_count", summary="Retrieves the total number of patients who have a specified allergy. The allergy is identified by its description.")
async def get_allergy_count(description: str = Query(..., description="Description of the allergy")):
    cursor.execute("SELECT COUNT(PATIENT) FROM allergies WHERE DESCRIPTION = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the patient with the latest start date for a specific condition
@app.get("/v1/synthea/latest_condition_start", operation_id="get_latest_condition_start", summary="Retrieves the patient record with the most recent start date for a specified medical condition. The condition is identified by its description.")
async def get_latest_condition_start(description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT PATIENT FROM conditions WHERE START = ( SELECT MAX(START) FROM conditions WHERE DESCRIPTION = ? )", (description,))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [row[0] for row in result]}

# Endpoint to get the most common allergy description
@app.get("/v1/synthea/most_common_allergy", operation_id="get_most_common_allergy", summary="Retrieves the description of the most frequently occurring allergy in the database. The operation returns the allergy description that appears most frequently in the allergies table, sorted in descending order by the count of occurrences.")
async def get_most_common_allergy():
    cursor.execute("SELECT DESCRIPTION FROM allergies GROUP BY DESCRIPTION ORDER BY COUNT(DESCRIPTION) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the patient with the highest observation value for a specific description and units
@app.get("/v1/synthea/highest_observation_value", operation_id="get_highest_observation_value", summary="Retrieves the patient with the highest recorded value for a specific observation type, as defined by the provided description and units. The returned patient data reflects the highest value observed for the specified observation type.")
async def get_highest_observation_value(description: str = Query(..., description="Description of the observation"), units: str = Query(..., description="Units of the observation")):
    cursor.execute("SELECT PATIENT FROM observations WHERE DESCRIPTION = ? AND UNITS = ? ORDER BY VALUE DESC LIMIT 1", (description, units))
    result = cursor.fetchone()
    if not result:
        return {"patient": []}
    return {"patient": result[0]}

# Endpoint to get the most common condition description for patients of a specific gender and ethnicity
@app.get("/v1/synthea/most_common_condition_gender_ethnicity", operation_id="get_most_common_condition_gender_ethnicity", summary="Retrieves the most frequently occurring medical condition description for patients of a specific gender and ethnicity. The gender and ethnicity of the patients are used to filter the results.")
async def get_most_common_condition_gender_ethnicity(gender: str = Query(..., description="Gender of the patient"), ethnicity: str = Query(..., description="Ethnicity of the patient")):
    cursor.execute("SELECT T2.DESCRIPTION FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.gender = ? AND T1.ethnicity = ? GROUP BY T2.DESCRIPTION ORDER BY COUNT(T2.DESCRIPTION) DESC LIMIT 1", (gender, ethnicity))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the count of distinct patients with a specific medication description, ethnicity, start year, and duration
@app.get("/v1/synthea/medication_count_ethnicity_year_duration", operation_id="get_medication_count_ethnicity_year_duration", summary="Retrieves the number of unique patients who have been prescribed a specific medication, based on the provided medication description, patient ethnicity, medication start year, and medication duration in months. The response is a count of distinct patients who meet these criteria.")
async def get_medication_count_ethnicity_year_duration(description: str = Query(..., description="Description of the medication"), ethnicity: str = Query(..., description="Ethnicity of the patient"), start_year: str = Query(..., description="Start year of the medication in 'YYYY' format"), duration: int = Query(..., description="Duration in months")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.ethnicity = ? AND strftime('%Y', T2.START) = ? AND strftime('%m', T2.STOP) - strftime('%m', T2.START) = ?", (description, ethnicity, start_year, duration))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients with a specific condition and birth year pattern
@app.get("/v1/synthea/condition_count_birth_year", operation_id="get_condition_count_birth_year", summary="Retrieves the number of unique patients who have a specific medical condition and were born in a year matching a given pattern. The condition is identified by its description, and the birth year pattern is provided in 'YYYY' format.")
async def get_condition_count_birth_year(description: str = Query(..., description="Description of the condition"), birth_year_pattern: str = Query(..., description="Birth year pattern in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE DESCRIPTION = ? AND strftime('%Y', T1.birthdate) LIKE ?", (description, birth_year_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct patient names with a specific medication and start year
@app.get("/v1/synthea/patient_names_medication_year", operation_id="get_patient_names_medication_year", summary="Retrieves unique patient names who have been prescribed a specific medication and started it in a given year. The medication description and start year are required as input parameters.")
async def get_patient_names_medication_year(description: str = Query(..., description="Description of the medication"), start_year: str = Query(..., description="Start year of the medication in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last, T1.suffix FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND strftime('%Y', T2.START) = ?", (description, start_year))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"first": row[0], "last": row[1], "suffix": row[2]} for row in result]}

# Endpoint to get the count of distinct patients with a specific race, immunization description, and year
@app.get("/v1/synthea/immunization_count_race_year", operation_id="get_immunization_count_race_year", summary="Retrieves the number of unique patients of a specified race who received a particular immunization in a given year. The response is based on the count of distinct patients that match the provided race, immunization description, and year.")
async def get_immunization_count_race_year(race: str = Query(..., description="Race of the patient"), description: str = Query(..., description="Description of the immunization"), year: str = Query(..., description="Year of the immunization in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T1.race = ? AND T2.DESCRIPTION = ? AND strftime('%Y', T2.DATE) = ?", (race, description, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients with the highest prevalence rate and latest condition start date
@app.get("/v1/synthea/highest_prevalence_latest_start", operation_id="get_highest_prevalence_latest_start", summary="Retrieves the count of patients with the highest prevalence rate and the most recent condition start date. This operation considers all prevalences and conditions, and filters patients who have received immunizations. The result is sorted by condition start date and prevalence rate in descending order, with the top record returned.")
async def get_highest_prevalence_latest_start():
    cursor.execute("SELECT COUNT(T2.patient) FROM all_prevalences AS T1 INNER JOIN conditions AS T2 ON lower(T1.ITEM) = lower(T2.DESCRIPTION) INNER JOIN immunizations AS T3 ON T2.PATIENT = T3.PATIENT GROUP BY T2.PATIENT ORDER BY T2.START DESC, T1.\"PREVALENCE RATE\" DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients based on prevalence rate
@app.get("/v1/synthea/patient_count_prevalence_rate", operation_id="get_patient_count_prevalence_rate", summary="Retrieves the total number of unique patients who have the condition with the highest prevalence rate. The prevalence rate is determined by comparing the condition descriptions in the 'all_prevalences' table with the 'conditions' table. The count is based on distinct patient entries.")
async def get_patient_count_prevalence_rate():
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM all_prevalences AS T1 INNER JOIN conditions AS T2 ON lower(T1.ITEM) = lower(T2.DESCRIPTION) ORDER BY T1.\"PREVALENCE RATE\" DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common condition description for a given immunization description
@app.get("/v1/synthea/most_common_condition_description", operation_id="get_most_common_condition_description", summary="Get the most common condition description for a given immunization description")
async def get_most_common_condition_description(immunization_description: str = Query(..., description="Description of the immunization")):
    cursor.execute("SELECT T2.DESCRIPTION FROM immunizations AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.DESCRIPTION = ? GROUP BY T2.DESCRIPTION ORDER BY COUNT(T2.DESCRIPTION) DESC LIMIT 1", (immunization_description,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get distinct patients based on prevalence percentage
@app.get("/v1/synthea/distinct_patients_prevalence_percentage", operation_id="get_distinct_patients_prevalence_percentage", summary="Retrieves a list of unique patients who have a condition with a specified prevalence percentage. The prevalence percentage is provided as a float value.")
async def get_distinct_patients_prevalence_percentage(prevalence_percentage: float = Query(..., description="Prevalence percentage as a float")):
    cursor.execute("SELECT DISTINCT T1.PATIENT FROM conditions AS T1 INNER JOIN all_prevalences AS T2 ON lower(T2.ITEM) = lower(T1.DESCRIPTION) WHERE T2.\"PREVALENCE PERCENTAGE\" = CAST(? AS float)", (prevalence_percentage,))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [row[0] for row in result]}

# Endpoint to get the count of distinct condition descriptions for a given patient name
@app.get("/v1/synthea/condition_description_count_patient_name", operation_id="get_condition_description_count_patient_name", summary="Retrieves the number of unique condition descriptions associated with a patient, based on the provided first and last name.")
async def get_condition_description_count_patient_name(first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T2.DESCRIPTION) FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients based on immunization and condition descriptions
@app.get("/v1/synthea/patient_count_immunization_condition", operation_id="get_patient_count_immunization_condition", summary="Retrieves the number of unique patients who have a specific immunization and a specific medical condition. The immunization and condition are identified by their descriptions.")
async def get_patient_count_immunization_condition(immunization_description: str = Query(..., description="Description of the immunization"), condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM immunizations AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.DESCRIPTION = ? AND T2.DESCRIPTION = ?", (immunization_description, condition_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients based on gender and condition description
@app.get("/v1/synthea/patient_count_gender_condition", operation_id="get_patient_count_gender_condition", summary="Retrieves the number of unique patients categorized by gender and a specific medical condition. The gender and condition description are used as filters to narrow down the patient count.")
async def get_patient_count_gender_condition(gender: str = Query(..., description="Gender of the patient"), condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.gender = ? AND T1.DESCRIPTION = ?", (gender, condition_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last name of a patient based on observation description and units
@app.get("/v1/synthea/patient_name_observation", operation_id="get_patient_name_observation", summary="Retrieves the first and last name of a patient associated with a specific observation description and units. The operation filters patients based on the provided observation description and units, and returns the first and last name of the first matching patient.")
async def get_patient_name_observation(observation_description: str = Query(..., description="Description of the observation"), observation_units: str = Query(..., description="Units of the observation")):
    cursor.execute("SELECT T1.first, T1.last FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T2.UNITS = ? ORDER BY T2.VALUE LIMIT 1", (observation_description, observation_units))
    result = cursor.fetchone()
    if not result:
        return {"name": []}
    return {"first": result[0], "last": result[1]}

# Endpoint to get the age of a patient based on name and condition description
@app.get("/v1/synthea/patient_age_name_condition", operation_id="get_patient_age_name_condition", summary="Retrieves the age of a patient with the specified first and last name who has a condition matching the provided description. The age is calculated based on the difference between the patient's death date and birth date.")
async def get_patient_age_name_condition(first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient"), condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT strftime('%Y', T2.deathdate) - strftime('%Y', T2.birthdate) AS age FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.first = ? AND T2.last = ? AND T1.DESCRIPTION = ?", (first_name, last_name, condition_description))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the count of distinct patients based on medication description, race, and gender
@app.get("/v1/synthea/patient_count_medication_race_gender", operation_id="get_patient_count_medication_race_gender", summary="Retrieves the number of unique patients who have been prescribed a specific medication, categorized by their race and gender. This operation provides a breakdown of patient demographics based on the provided medication description, race, and gender.")
async def get_patient_count_medication_race_gender(medication_description: str = Query(..., description="Description of the medication"), race: str = Query(..., description="Race of the patient"), gender: str = Query(..., description="Gender of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM medications AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND T2.race = ? AND T2.gender = ?", (medication_description, race, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients based on condition description and alive status
@app.get("/v1/synthea/patient_count_condition_alive", operation_id="get_patient_count_condition_alive", summary="Retrieves the number of unique patients who have a specific medical condition and are currently alive. The condition is identified by its description.")
async def get_patient_count_condition_alive(condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.description = ? AND T2.deathdate IS NULL", (condition_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients with a specific allergy and race
@app.get("/v1/synthea/allergy_patient_count", operation_id="get_allergy_patient_count", summary="Retrieves the total number of patients with a specified allergy and race. The operation filters patients based on the provided allergy description and race, then returns the count of patients who meet these criteria.")
async def get_allergy_patient_count(allergy_description: str = Query(..., description="Description of the allergy"), race: str = Query(..., description="Race of the patient")):
    cursor.execute("SELECT COUNT(T2.patient) FROM allergies AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND T2.race = ?", (allergy_description, race))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average value of a specific observation for patients with a specific condition
@app.get("/v1/synthea/avg_observation_value", operation_id="get_avg_observation_value", summary="Retrieves the average value of a specific observation for patients diagnosed with a particular condition. The operation requires the description of the condition and the description of the observation as input parameters.")
async def get_avg_observation_value(condition_description: str = Query(..., description="Description of the condition"), observation_description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT AVG(T1.VALUE) FROM observations AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient INNER JOIN conditions AS T3 ON T2.patient = T3.PATIENT WHERE T3.DESCRIPTION = ? AND T1.DESCRIPTION = ?", (condition_description, observation_description))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the description of medications for a patient with a specific SSN
@app.get("/v1/synthea/medications_by_ssn", operation_id="get_medications_by_ssn", summary="Retrieves the descriptions of all medications associated with a patient identified by their unique Social Security Number (SSN).")
async def get_medications_by_ssn(ssn: str = Query(..., description="SSN of the patient")):
    cursor.execute("SELECT T1.DESCRIPTION FROM medications AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.ssn = ?", (ssn,))
    result = cursor.fetchall()
    if not result:
        return {"medications": []}
    return {"medications": [row[0] for row in result]}

# Endpoint to get the SSNs of patients with a specific allergy, ethnicity, and gender
@app.get("/v1/synthea/ssn_by_allergy_ethnicity_gender", operation_id="get_ssn_by_allergy_ethnicity_gender", summary="Retrieves the Social Security Numbers (SSNs) of patients who have a specific allergy, belong to a certain ethnic group, and identify with a particular gender. The allergy is identified by its description, while the ethnicity and gender are specified directly.")
async def get_ssn_by_allergy_ethnicity_gender(allergy_description: str = Query(..., description="Description of the allergy"), ethnicity: str = Query(..., description="Ethnicity of the patient"), gender: str = Query(..., description="Gender of the patient")):
    cursor.execute("SELECT T2.ssn FROM allergies AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND T2.ethnicity = ? AND T2.gender = ?", (allergy_description, ethnicity, gender))
    result = cursor.fetchall()
    if not result:
        return {"ssns": []}
    return {"ssns": [row[0] for row in result]}

# Endpoint to get the first and last names of patients with a specific care plan code
@app.get("/v1/synthea/patient_names_by_careplan_code", operation_id="get_patient_names_by_careplan_code", summary="Retrieves the first and last names of patients associated with a specific care plan code. The operation filters patients based on the provided care plan code and returns their names.")
async def get_patient_names_by_careplan_code(careplan_code: int = Query(..., description="Code of the care plan")):
    cursor.execute("SELECT T2.first, T2.last FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.CODE = ?", (careplan_code,))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the description of the condition of the patient with the longest lifespan
@app.get("/v1/synthea/longest_lifespan_condition", operation_id="get_longest_lifespan_condition", summary="Retrieves the description of the medical condition associated with the patient who has the longest lifespan. This operation identifies the patient with the longest lifespan based on the difference between their death and birth dates, and then returns the description of their medical condition.")
async def get_longest_lifespan_condition():
    cursor.execute("SELECT T1.DESCRIPTION FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.deathdate IS NOT NULL ORDER BY strftime('%Y', T2.deathdate) - strftime('%Y', T2.birthdate) DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"condition": []}
    return {"condition": result[0]}

# Endpoint to get the code of the most prevalent condition
@app.get("/v1/synthea/most_prevalent_condition_code", operation_id="get_most_prevalent_condition_code", summary="Retrieves the code of the condition that occurs most frequently in the database. This operation uses the all_prevalences table to determine the most prevalent condition and then retrieves the corresponding code from the conditions table.")
async def get_most_prevalent_condition_code():
    cursor.execute("SELECT T2.code FROM all_prevalences AS T1 INNER JOIN conditions AS T2 ON T1.ITEM = T2.DESCRIPTION ORDER BY T1.OCCURRENCES DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"code": []}
    return {"code": result[0]}

# Endpoint to get distinct observations for patients at a specific address
@app.get("/v1/synthea/observations_by_address", operation_id="get_observations_by_address", summary="Retrieve unique observations for patients residing at a specified address. The operation requires the description of the observation and the patient's address as input parameters. It returns the description, value, and units of the distinct observations that match the provided criteria.")
async def get_observations_by_address(observation_description: str = Query(..., description="Description of the observation"), address: str = Query(..., description="Address of the patient")):
    cursor.execute("SELECT DISTINCT T2.DESCRIPTION, T2.VALUE, T2.UNITS FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.address = ?", (observation_description, address))
    result = cursor.fetchall()
    if not result:
        return {"observations": []}
    return {"observations": [{"description": row[0], "value": row[1], "units": row[2]} for row in result]}

# Endpoint to get distinct SSNs of patients with conditions having a prevalence percentage below a certain threshold
@app.get("/v1/synthea/ssn_by_prevalence_threshold", operation_id="get_ssn_by_prevalence_threshold", summary="Get distinct SSNs of patients with conditions having a prevalence percentage below a certain threshold")
async def get_ssn_by_prevalence_threshold(prevalence_threshold: float = Query(..., description="Prevalence percentage threshold")):
    cursor.execute("SELECT DISTINCT T2.ssn FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient INNER JOIN all_prevalences AS T3 ON lower(T1.DESCRIPTION) = lower(T3.ITEM) WHERE CAST(T3.\"PREVALENCE PERCENTAGE\" AS REAL) * 100 / ( SELECT AVG(\"PREVALENCE PERCENTAGE\") FROM all_prevalences ) < ? LIMIT 5", (prevalence_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"ssns": []}
    return {"ssns": [row[0] for row in result]}

# Endpoint to get the percentage of patients with a specific condition, gender, and race
@app.get("/v1/synthea/condition_gender_race_percentage", operation_id="get_condition_gender_race_percentage", summary="Retrieves the percentage of patients with a specific medical condition, gender, and race. The calculation is based on the total number of patients with the given condition, filtered by gender and race.")
async def get_condition_gender_race_percentage(gender: str = Query(..., description="Gender of the patient"), race: str = Query(..., description="Race of the patient"), condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.gender = ? AND T2.race = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.gender) FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ?", (gender, race, condition_description))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of encounters for a patient with a specific first and last name
@app.get("/v1/synthea/encounter_count_by_patient_name", operation_id="get_encounter_count", summary="Retrieves the total number of encounters associated with a patient, identified by their first and last name. The operation filters patients based on the provided first and last names, and then counts the corresponding encounters.")
async def get_encounter_count(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT COUNT(T2.ID) FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first, last))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the descriptions of procedures for a patient with a specific first and last name
@app.get("/v1/synthea/procedure_descriptions_by_patient_name", operation_id="get_procedure_descriptions", summary="Retrieve the descriptions of all procedures performed on a patient identified by a specific first and last name. The operation filters the patient records based on the provided first and last names, and then fetches the corresponding procedure descriptions.")
async def get_procedure_descriptions(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT T2.DESCRIPTION FROM patients AS T1 INNER JOIN procedures AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first, last))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the first and last names of patients who had a specific procedure
@app.get("/v1/synthea/patient_names_by_procedure", operation_id="get_patient_names_by_procedure", summary="Retrieves the first and last names of patients who underwent a procedure matching the provided description. The procedure description is used to filter the patients.")
async def get_patient_names_by_procedure(description: str = Query(..., description="Description of the procedure")):
    cursor.execute("SELECT T1.first, T1.last FROM patients AS T1 INNER JOIN procedures AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the observations for a patient with a specific first and last name and a specific observation description
@app.get("/v1/synthea/observations_by_patient_name_and_description", operation_id="get_observations", summary="Retrieve the observations for a patient identified by a specific first and last name, filtered by a given observation description. The response includes the description, value, and units of the matching observations.")
async def get_observations(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT T2.DESCRIPTION, T2.VALUE, T2.UNITS FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.DESCRIPTION = ?", (first, last, description))
    result = cursor.fetchall()
    if not result:
        return {"observations": []}
    return {"observations": [{"description": row[0], "value": row[1], "units": row[2]} for row in result]}

# Endpoint to get the first and last names of patients with a specific allergy
@app.get("/v1/synthea/patient_names_by_allergy", operation_id="get_patient_names_by_allergy", summary="Retrieves the first and last names of patients who have a specified allergy. The allergy description is used to filter the patients.")
async def get_patient_names_by_allergy(description: str = Query(..., description="Description of the allergy")):
    cursor.execute("SELECT T1.first, T1.last FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the count of immunizations for a patient with a specific first and last name and a specific immunization description
@app.get("/v1/synthea/immunization_count_by_patient_name_and_description", operation_id="get_immunization_count", summary="Retrieves the total number of immunizations administered to a patient with a specific first and last name that match a given immunization description. This operation is useful for tracking the immunization history of a particular patient.")
async def get_immunization_count(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the immunization")):
    cursor.execute("SELECT COUNT(T2.CODE) FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.DESCRIPTION = ?", (first, last, description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last names of patients who have been on a specific medication for more than a specified number of years
@app.get("/v1/synthea/patient_names_by_medication_duration", operation_id="get_patient_names_by_medication_duration", summary="Retrieve the first and last names of patients who have been prescribed a specific medication for a duration exceeding a given number of years. The medication is identified by its description, and the duration is specified in years.")
async def get_patient_names_by_medication_duration(description: str = Query(..., description="Description of the medication"), duration: int = Query(..., description="Duration in years")):
    cursor.execute("SELECT T1.first, T1.last FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND strftime('%Y', T2.STOP) - strftime('%Y', T2.START) > ?", (description, duration))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the distinct descriptions of procedures and medications for patients with a specific condition
@app.get("/v1/synthea/procedure_medication_descriptions_by_condition", operation_id="get_procedure_medication_descriptions", summary="Retrieve the unique descriptions of procedures and medications associated with patients diagnosed with a specific medical condition. The condition description is used to filter the results.")
async def get_procedure_medication_descriptions(description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT DISTINCT T1.DESCRIPTION, T3.DESCRIPTION FROM procedures AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT INNER JOIN medications AS T3 ON T2.patient = T3.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [{"procedure": row[0], "medication": row[1]} for row in result]}

# Endpoint to get the descriptions of medications for patients with a specific allergy and start date
@app.get("/v1/synthea/medication_descriptions_by_allergy_and_start_date", operation_id="get_medication_descriptions", summary="Retrieves the descriptions of medications prescribed to patients who have a specific allergy and started on a given date. The operation requires the start date of the allergy and its description as input parameters.")
async def get_medication_descriptions(start_date: str = Query(..., description="Start date of the allergy in 'MM/DD/YY' format"), description: str = Query(..., description="Description of the allergy")):
    cursor.execute("SELECT T2.DESCRIPTION FROM allergies AS T1 INNER JOIN medications AS T2 ON T1.PATIENT = T2.PATIENT WHERE T1.START = ? AND T1.DESCRIPTION = ?", (start_date, description))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the distinct descriptions of care plans for patients with a specific condition
@app.get("/v1/synthea/careplan_descriptions_by_condition", operation_id="get_careplan_descriptions", summary="Retrieve unique care plan descriptions for patients diagnosed with a specific medical condition. The condition description is used to filter the results.")
async def get_careplan_descriptions(description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT DISTINCT T1.DESCRIPTION FROM careplans AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get patient details based on observation criteria
@app.get("/v1/synthea/patient_details_by_observation", operation_id="get_patient_details", summary="Retrieves patient details, including first name, last name, and age, based on a specific observation. The observation is identified by its description, value, units, and the year it was recorded. The age is calculated as the difference between the year of the observation and the patient's birth year, or the year of death if applicable.")
async def get_patient_details(description: str = Query(..., description="Observation description"), value: int = Query(..., description="Observation value"), units: str = Query(..., description="Observation units"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.first, T2.last , CASE WHEN T2.deathdate IS NULL THEN strftime('%Y', T1.DATE) - strftime('%Y', T2.birthdate) ELSE strftime('%Y', T2.deathdate) - strftime('%Y', T2.birthdate) END AS age FROM observations AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND T1.VALUE = ? AND T1.UNITS = ? AND strftime('%Y', T1.DATE) = ?", (description, value, units, year))
    result = cursor.fetchall()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get immunization and ethnicity statistics for a given year
@app.get("/v1/synthea/immunization_ethnicity_stats", operation_id="get_immunization_ethnicity_stats", summary="Retrieves the percentage of patients with a specific immunization and ethnicity for a given year. The operation calculates the proportion of patients with a certain immunization and ethnicity, based on the provided immunization description and ethnicity. The year parameter is used to filter the data for the specified year.")
async def get_immunization_ethnicity_stats(description: str = Query(..., description="Immunization description"), ethnicity: str = Query(..., description="Ethnicity"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.DESCRIPTION = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.patient), SUM(CASE WHEN T1.ethnicity = ? THEN 1 ELSE 0 END) FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE strftime('%Y', T2.DATE) = ?", (description, ethnicity, year))
    result = cursor.fetchone()
    if not result:
        return {"stats": []}
    return {"stats": result}

# Endpoint to get distinct patient first names based on encounter reason
@app.get("/v1/synthea/distinct_patient_first_names_by_encounter_reason", operation_id="get_distinct_patient_first_names", summary="Retrieves a list of unique first names of patients who have had an encounter with the specified reason description.")
async def get_distinct_patient_first_names(reason_description: str = Query(..., description="Encounter reason description")):
    cursor.execute("SELECT DISTINCT T1.first FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T2.REASONDESCRIPTION = ?", (reason_description,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get distinct patient birthdates based on encounter description
@app.get("/v1/synthea/distinct_patient_birthdates_by_encounter_description", operation_id="get_distinct_patient_birthdates", summary="Retrieves the unique birthdates of patients who have encounters with the specified description. This operation is useful for identifying the distinct birthdates of patients associated with a particular encounter type or scenario.")
async def get_distinct_patient_birthdates(description: str = Query(..., description="Encounter description")):
    cursor.execute("SELECT DISTINCT T1.birthdate FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"birthdates": []}
    return {"birthdates": [row[0] for row in result]}

# Endpoint to get distinct patient first names based on condition description
@app.get("/v1/synthea/distinct_patient_first_names_by_condition_description", operation_id="get_distinct_patient_first_names_by_condition", summary="Retrieves a list of unique first names of patients who have been diagnosed with a specific medical condition. The condition is identified by its description.")
async def get_distinct_patient_first_names_by_condition(description: str = Query(..., description="Condition description")):
    cursor.execute("SELECT DISTINCT T1.first FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"first_names": []}
    return {"first_names": [row[0] for row in result]}

# Endpoint to get the count of distinct patients based on condition description and marital status
@app.get("/v1/synthea/count_distinct_patients_by_condition_marital", operation_id="get_count_distinct_patients", summary="Retrieves the number of unique patients who have a specific medical condition and a particular marital status. The condition is identified by its description, and the marital status is specified as a parameter.")
async def get_count_distinct_patients(description: str = Query(..., description="Condition description"), marital: str = Query(..., description="Marital status")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.marital = ?", (description, marital))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct patient addresses based on billable period
@app.get("/v1/synthea/distinct_patient_addresses_by_billable_period", operation_id="get_distinct_patient_addresses", summary="Retrieves a list of unique patient addresses associated with claims from a specified billable period. The billable period should be provided in the 'YYYY%' format.")
async def get_distinct_patient_addresses(billable_period: str = Query(..., description="Billable period in 'YYYY%' format")):
    cursor.execute("SELECT DISTINCT T1.address FROM patients AS T1 INNER JOIN claims AS T2 ON T1.patient = T2.PATIENT WHERE T2.BILLABLEPERIOD LIKE ?", (billable_period,))
    result = cursor.fetchall()
    if not result:
        return {"addresses": []}
    return {"addresses": [row[0] for row in result]}

# Endpoint to get distinct patient last names based on allergy description
@app.get("/v1/synthea/distinct_patient_last_names_by_allergy_description", operation_id="get_distinct_patient_last_names", summary="Retrieves a list of unique last names of patients who have the specified allergy description. The operation filters patients based on the provided allergy description and returns their distinct last names.")
async def get_distinct_patient_last_names(description: str = Query(..., description="Allergy description")):
    cursor.execute("SELECT DISTINCT T1.last FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"last_names": []}
    return {"last_names": [row[0] for row in result]}

# Endpoint to get allergy start dates based on patient first and last name
@app.get("/v1/synthea/allergy_start_dates_by_patient_name", operation_id="get_allergy_start_dates", summary="Retrieves the start dates of allergies for a patient identified by their first and last name. The operation filters allergies based on the provided patient name and returns the start dates of allergies that have a recorded end date.")
async def get_allergy_start_dates(first_name: str = Query(..., description="Patient first name"), last_name: str = Query(..., description="Patient last name")):
    cursor.execute("SELECT T2.START FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.STOP IS NOT NULL", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"start_dates": []}
    return {"start_dates": [row[0] for row in result]}

# Endpoint to get the count of distinct patients with a specific allergy and gender
@app.get("/v1/synthea/patient_count_allergy_gender", operation_id="get_patient_count_allergy_gender", summary="Retrieves the number of unique patients who have a specific allergy and are of a certain gender. The allergy is identified by its description, and the gender is specified as either 'male' or 'female'.")
async def get_patient_count_allergy_gender(allergy_description: str = Query(..., description="Description of the allergy"), gender: str = Query(..., description="Gender of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.gender = ?", (allergy_description, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common allergy description for patients of a specific race
@app.get("/v1/synthea/most_common_allergy_by_race", operation_id="get_most_common_allergy_by_race", summary="Retrieves the most frequently occurring allergy description for patients of a specified race. The operation groups allergies by description and orders them by frequency, returning the most common allergy for the given race.")
async def get_most_common_allergy_by_race(race: str = Query(..., description="Race of the patient")):
    cursor.execute("SELECT T2.DESCRIPTION FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T1.race = ? GROUP BY T2.DESCRIPTION ORDER BY COUNT(T2.DESCRIPTION) DESC LIMIT 1", (race,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get distinct first and last names of patients with a specific immunization
@app.get("/v1/synthea/patient_names_by_immunization", operation_id="get_patient_names_by_immunization", summary="Get distinct first and last names of patients with a specific immunization")
async def get_patient_names_by_immunization(immunization_description: str = Query(..., description="Description of the immunization")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (immunization_description,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the count of distinct patients with a specific immunization and gender
@app.get("/v1/synthea/patient_count_immunization_gender", operation_id="get_patient_count_immunization_gender", summary="Get the count of distinct patients with a specific immunization and gender")
async def get_patient_count_immunization_gender(immunization_description: str = Query(..., description="Description of the immunization"), gender: str = Query(..., description="Gender of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.gender = ?", (immunization_description, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct encounter descriptions for patients born in a specific place
@app.get("/v1/synthea/encounter_descriptions_by_birthplace", operation_id="get_encounter_descriptions_by_birthplace", summary="Retrieves unique encounter descriptions for patients who were born in a specified location. The birthplace parameter is used to filter the results.")
async def get_encounter_descriptions_by_birthplace(birthplace: str = Query(..., description="Birthplace of the patient")):
    cursor.execute("SELECT DISTINCT T2.DESCRIPTION FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T1.birthplace = ?", (birthplace,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": result}

# Endpoint to get the count of distinct allergy descriptions for patients of a specific ethnicity
@app.get("/v1/synthea/allergy_count_by_ethnicity", operation_id="get_allergy_count_by_ethnicity", summary="Retrieves the number of unique allergy descriptions for patients of a specified ethnicity. The ethnicity parameter is used to filter the patient population.")
async def get_allergy_count_by_ethnicity(ethnicity: str = Query(..., description="Ethnicity of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T2.DESCRIPTION) FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T1.ethnicity = ?", (ethnicity,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average age of patients with a specific care plan reason
@app.get("/v1/synthea/average_age_by_careplan_reason", operation_id="get_average_age_by_careplan_reason", summary="Retrieves the average age of patients who have a care plan with a specific reason. The calculation considers the patient's birthdate and deathdate, if applicable, and the care plan's stop date. The result is the average age in years.")
async def get_average_age_by_careplan_reason(careplan_reason: str = Query(..., description="Reason for the care plan")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.deathdate IS NULL THEN strftime('%Y', T2.STOP) - strftime('%Y', T1.birthdate) ELSE strftime('%Y', T1.deathdate) - strftime('%Y', T1.birthdate) END) AS REAL) / COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T2.REASONDESCRIPTION = ?", (careplan_reason,))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the count of distinct patients with a specific medication reason who are alive and have stopped the medication
@app.get("/v1/synthea/patient_count_medication_reason_alive", operation_id="get_patient_count_medication_reason_alive", summary="Retrieves the number of unique patients who have a specific reason for taking a medication, are currently alive, and have stopped taking the medication. The reason for the medication is provided as an input parameter.")
async def get_patient_count_medication_reason_alive(medication_reason: str = Query(..., description="Reason for the medication")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM medications AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.REASONDESCRIPTION = ? AND T1.STOP IS NOT NULL AND T2.deathdate IS NULL", (medication_reason,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients with a specific procedure and who have drivers
@app.get("/v1/synthea/patient_count_procedure_drivers", operation_id="get_patient_count_procedure_drivers", summary="Retrieve the number of unique patients who have undergone a specific procedure and possess a valid driver's license.")
async def get_patient_count_procedure_drivers(procedure_description: str = Query(..., description="Description of the procedure")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN procedures AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.drivers IS NOT NULL", (procedure_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last names of patients with more than a specified number of distinct allergies
@app.get("/v1/synthea/patient_names_by_allergy_count", operation_id="get_patient_names_by_allergy_count", summary="Retrieve the first and last names of patients who have more than the specified number of distinct allergies. The endpoint returns a list of patient names, sorted by the count of their unique allergies in descending order.")
async def get_patient_names_by_allergy_count(allergy_count: int = Query(..., description="Number of distinct allergies")):
    cursor.execute("SELECT T1.first, T1.last FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT GROUP BY T1.patient HAVING COUNT(DISTINCT T2.DESCRIPTION) > ?", (allergy_count,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": result}

# Endpoint to get the count of distinct patients with specific allergies and immunizations
@app.get("/v1/synthea/patient_count_allergies_immunizations", operation_id="get_patient_count_allergies_immunizations", summary="Retrieves the number of unique patients who have a specific allergy and a specific immunization. The operation filters patients based on the provided descriptions of the allergy and immunization.")
async def get_patient_count_allergies_immunizations(allergy_description: str = Query(..., description="Description of the allergy"), immunization_description: str = Query(..., description="Description of the immunization")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM allergies AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient INNER JOIN immunizations AS T3 ON T2.patient = T3.PATIENT WHERE T1.DESCRIPTION = ? AND T3.DESCRIPTION = ?", (allergy_description, immunization_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients with specific care plans and observations
@app.get("/v1/synthea/patient_count_careplans_observations", operation_id="get_patient_count_careplans_observations", summary="Retrieve the number of unique patients who have a specific care plan and a corresponding observation that meets the provided criteria. The criteria include the description of the observation, the description of the care plan, the value of the observation, and the units of the observation.")
async def get_patient_count_careplans_observations(observation_description: str = Query(..., description="Description of the observation"), careplan_description: str = Query(..., description="Description of the care plan"), observation_value: float = Query(..., description="Value of the observation"), observation_units: str = Query(..., description="Units of the observation")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient INNER JOIN observations AS T3 ON T2.patient = T3.PATIENT WHERE T3.DESCRIPTION = ? AND T1.DESCRIPTION = ? AND T3.VALUE > ? AND T3.UNITS = ?", (observation_description, careplan_description, observation_value, observation_units))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common gender of patients with a specific allergy
@app.get("/v1/synthea/most_common_gender_allergy", operation_id="get_most_common_gender_allergy", summary="Retrieves the gender that is most commonly associated with a specific allergy among patients. The allergy is identified by its description.")
async def get_most_common_gender_allergy(allergy_description: str = Query(..., description="Description of the allergy")):
    cursor.execute("SELECT T1.gender FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? GROUP BY T1.gender ORDER BY COUNT(T1.gender) DESC LIMIT 1", (allergy_description,))
    result = cursor.fetchone()
    if not result:
        return {"gender": []}
    return {"gender": result[0]}

# Endpoint to get distinct billable periods for patients with a specific last name
@app.get("/v1/synthea/distinct_billable_periods", operation_id="get_distinct_billable_periods", summary="Retrieves the unique billing periods for patients who share a specified last name. The endpoint returns a list of distinct billable periods associated with the given last name, providing insights into the billing cycles of patients with the same last name.")
async def get_distinct_billable_periods(last_name: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT DISTINCT T2.BILLABLEPERIOD FROM patients AS T1 INNER JOIN claims AS T2 ON T1.patient = T2.PATIENT WHERE T1.last = ?", (last_name,))
    result = cursor.fetchall()
    if not result:
        return {"billable_periods": []}
    return {"billable_periods": [item[0] for item in result]}

# Endpoint to get distinct patient names with a specific condition
@app.get("/v1/synthea/distinct_patient_names_condition", operation_id="get_distinct_patient_names_condition", summary="Retrieves a list of unique patient names who have been diagnosed with a specific medical condition. The condition is identified by its description.")
async def get_distinct_patient_names_condition(condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (condition_description,))
    result = cursor.fetchall()
    if not result:
        return {"patient_names": []}
    return {"patient_names": [{"first": item[0], "last": item[1]} for item in result]}

# Endpoint to get the count of distinct patients with specific medications and ethnicity
@app.get("/v1/synthea/patient_count_medications_ethnicity", operation_id="get_patient_count_medications_ethnicity", summary="Retrieves the number of unique patients who have a specific medication reason and belong to a certain ethnic group. The medication reason and ethnicity are provided as input parameters.")
async def get_patient_count_medications_ethnicity(medication_reason: str = Query(..., description="Reason for the medication"), ethnicity: str = Query(..., description="Ethnicity of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.REASONDESCRIPTION = ? AND T1.ethnicity = ?", (medication_reason, ethnicity))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients with specific marital status and care plan reason
@app.get("/v1/synthea/patient_count_marital_careplan", operation_id="get_patient_count_marital_careplan", summary="Retrieves the number of unique patients who have a specific marital status and a care plan with a particular reason. The marital status and care plan reason are provided as input parameters.")
async def get_patient_count_marital_careplan(marital_status: str = Query(..., description="Marital status of the patient"), careplan_reason: str = Query(..., description="Reason for the care plan")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T1.marital = ? AND T2.REASONDESCRIPTION = ?", (marital_status, careplan_reason))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients with specific immunizations and care plan stop date
@app.get("/v1/synthea/patient_count_immunizations_careplan_stop", operation_id="get_patient_count_immunizations_careplan_stop", summary="Get the count of distinct patients with specific immunizations and care plan stop date")
async def get_patient_count_immunizations_careplan_stop(immunization_description: str = Query(..., description="Description of the immunization"), careplan_stop_date: str = Query(..., description="Stop date of the care plan in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM careplans AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.STOP = ?", (immunization_description, careplan_stop_date))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients with specific medications and gender
@app.get("/v1/synthea/patient_count_medications_gender", operation_id="get_patient_count_medications_gender", summary="Retrieve the number of unique patients who have a specific medication and belong to a certain gender. The medication is identified by its description, and the gender is specified as either 'Male' or 'Female'.")
async def get_patient_count_medications_gender(medication_description: str = Query(..., description="Description of the medication"), gender: str = Query(..., description="Gender of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.gender = ?", (medication_description, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of patients with a specific allergy and birthplace
@app.get("/v1/synthea/percentage_patients_allergy_birthplace", operation_id="get_percentage_patients_allergy_birthplace", summary="Retrieves the percentage of patients who have a specific allergy and were born in a particular location. The allergy is identified by its description, and the birthplace is specified by its name. The calculation is based on the total number of patients with the given birthplace.")
async def get_percentage_patients_allergy_birthplace(allergy_description: str = Query(..., description="Description of the allergy"), birthplace: str = Query(..., description="Birthplace of the patient")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.DESCRIPTION = ? THEN 1 ELSE 0 END) AS REL) * 100 / COUNT(T1.patient) FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T1.birthplace = ?", (allergy_description, birthplace))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average body weight of patients of a specific race
@app.get("/v1/synthea/average_body_weight", operation_id="get_average_body_weight", summary="Retrieves the average body weight of patients belonging to a specific race. The calculation is based on the sum of all body weight observations for the specified race, divided by the total number of patients of that race. The input parameters allow for specifying the race, the description of the body weight observation, and the units of the observation.")
async def get_average_body_weight(race: str = Query(..., description="Race of the patient"), description: str = Query(..., description="Description of the observation"), units: str = Query(..., description="Units of the observation")):
    cursor.execute("SELECT SUM(T2.VALUE) / COUNT(T1.patient) FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.race = ? AND T2.DESCRIPTION = ? AND T2.UNITS = ?", (race, description, units))
    result = cursor.fetchone()
    if not result:
        return {"average_weight": []}
    return {"average_weight": result[0]}

# Endpoint to get SSNs of patients with a specific allergy
@app.get("/v1/synthea/ssn_by_allergy", operation_id="get_ssn_by_allergy", summary="Retrieves the Social Security Numbers (SSNs) of patients who have a specific allergy, as identified by the provided allergy description.")
async def get_ssn_by_allergy(description: str = Query(..., description="Description of the allergy")):
    cursor.execute("SELECT T1.ssn FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"ssns": []}
    return {"ssns": [row[0] for row in result]}

# Endpoint to get the duration of allergies for a specific patient
@app.get("/v1/synthea/allergy_duration", operation_id="get_allergy_duration", summary="Retrieves the duration of allergies for a specific patient, given their first and last names. The response includes the allergy description and the duration in years.")
async def get_allergy_duration(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT CASE WHEN SUBSTR(T1.STOP, -2, 1) != '9' THEN SUBSTR(T1.STOP, LENGTH(T1.STOP) - 1) + 2000 END - CASE WHEN SUBSTR(T1.START, -2, 1) = '9' THEN SUBSTR(T1.START, LENGTH(T1.START) - 1) + 1900 ELSE SUBSTR(T1.START, LENGTH(T1.START) - 1) + 2000 END AS years , T1.DESCRIPTION FROM allergies AS T1 INNER JOIN patients AS T2 ON T2.patient = T1.PATIENT WHERE T1.STOP IS NOT NULL AND T1.START IS NOT NULL AND T2.first = ? AND T2.last = ?", (first, last))
    result = cursor.fetchall()
    if not result:
        return {"allergy_duration": []}
    return {"allergy_duration": [{"years": row[0], "description": row[1]} for row in result]}

# Endpoint to get distinct patients with a specific marital status and care plan duration
@app.get("/v1/synthea/patients_by_marital_status_and_duration", operation_id="get_patients_by_marital_status_and_duration", summary="Retrieves a list of unique patients who have a specified marital status and a care plan duration that exceeds the provided value. The response includes the first and last names of the patients.")
async def get_patients_by_marital_status_and_duration(marital: str = Query(..., description="Marital status of the patient"), duration: int = Query(..., description="Duration in days")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T1.marital = ? AND strftime('%J', T2.STOP) - strftime('%J', T2.START) > ?", (marital, duration))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get immunization dates for a specific patient and immunization type
@app.get("/v1/synthea/immunization_dates", operation_id="get_immunization_dates", summary="Get immunization dates for a specific patient and immunization type")
async def get_immunization_dates(description: str = Query(..., description="Description of the immunization"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT T2.DATE FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.first = ? AND T1.last = ?", (description, first, last))
    result = cursor.fetchall()
    if not result:
        return {"dates": []}
    return {"dates": [row[0] for row in result]}

# Endpoint to get the count of distinct patients with a specific immunization within a date range and race
@app.get("/v1/synthea/immunization_count_by_race", operation_id="get_immunization_count_by_race", summary="Retrieve the number of unique patients who have received a specific immunization within a specified date range and belong to a particular race. The immunization is identified by its description, and the date range is defined by a start and end date. The patient's race is also considered in the count.")
async def get_immunization_count_by_race(description: str = Query(..., description="Description of the immunization"), start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), race: str = Query(..., description="Race of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T2.DATE BETWEEN ? AND ? AND T1.race = ?", (description, start_date, end_date, race))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct immunization codes and dates for a specific patient and immunization type
@app.get("/v1/synthea/immunization_codes_and_dates", operation_id="get_immunization_codes_and_dates", summary="Get distinct immunization codes and dates for a specific patient and immunization type")
async def get_immunization_codes_and_dates(prefix: str = Query(..., description="Prefix of the patient"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the immunization")):
    cursor.execute("SELECT DISTINCT T2.CODE, T2.DATE FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T1.prefix = ? AND T1.first = ? AND T1.last = ? AND T2.DESCRIPTION = ?", (prefix, first, last, description))
    result = cursor.fetchall()
    if not result:
        return {"immunizations": []}
    return {"immunizations": [{"code": row[0], "date": row[1]} for row in result]}

# Endpoint to get the count of distinct patients with a specific marital status, medication reason, and medication description in a specific year
@app.get("/v1/synthea/medication_count_by_marital_status", operation_id="get_medication_count_by_marital_status", summary="Retrieve the number of unique patients who have a certain marital status, medication reason, and medication description in a given year. This operation provides insights into medication usage patterns based on marital status and medication details.")
async def get_medication_count_by_marital_status(marital: str = Query(..., description="Marital status of the patient"), reason_description: str = Query(..., description="Reason for the medication"), description: str = Query(..., description="Description of the medication"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T1.marital = ? AND T2.REASONDESCRIPTION = ? AND T2.DESCRIPTION = ? AND strftime('%Y', T2.START) = ?", (marital, reason_description, description, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the reason description for encounters based on date, first name, and last name
@app.get("/v1/synthea/encounter_reason_description", operation_id="get_encounter_reason_description", summary="Retrieves the description of the reason for a patient's encounter on a specific date. The operation requires the patient's first and last name to identify the correct encounter and its associated reason description.")
async def get_encounter_reason_description(date: str = Query(..., description="Date of the encounter in 'YYYY-MM-DD' format"), first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT T2.REASONDESCRIPTION FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T2.DATE = ? AND T1.first = ? AND T1.last = ?", (date, first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"reason_descriptions": []}
    return {"reason_descriptions": [row[0] for row in result]}

# Endpoint to get SSN and address of patients based on encounter date, reason description, and encounter description
@app.get("/v1/synthea/patient_ssn_address", operation_id="get_patient_ssn_address", summary="Retrieves the SSN and address of patients who had an encounter on a specific date with a given reason and description. The encounter date, reason, and description are used to filter the results.")
async def get_patient_ssn_address(date: str = Query(..., description="Date of the encounter in 'YYYY-MM-DD' format"), reason_description: str = Query(..., description="Reason description of the encounter"), description: str = Query(..., description="Description of the encounter")):
    cursor.execute("SELECT T1.ssn, T1.address FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T2.DATE = ? AND T2.REASONDESCRIPTION = ? AND T2.DESCRIPTION = ?", (date, reason_description, description))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"ssn": row[0], "address": row[1]} for row in result]}

# Endpoint to get medication details based on encounter ID
@app.get("/v1/synthea/medication_details", operation_id="get_medication_details", summary="Retrieves detailed information about medications administered during a specific encounter. The response includes the reason for medication, medication description, duration in days, and the patient's status (alive or dead). The encounter is identified by the provided encounter_id.")
async def get_medication_details(encounter_id: str = Query(..., description="Encounter ID")):
    cursor.execute("SELECT T2.REASONDESCRIPTION, T2.DESCRIPTION, strftime('%J', T2.STOP) - strftime('%J', T2.START) AS days, CASE WHEN T1.deathdate IS NULL THEN 'alive' ELSE 'dead' END FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.ENCOUNTER = ?", (encounter_id,))
    result = cursor.fetchall()
    if not result:
        return {"medications": []}
    return {"medications": [{"reason_description": row[0], "description": row[1], "days": row[2], "status": row[3]} for row in result]}

# Endpoint to get patient names based on allergy description and age at death
@app.get("/v1/synthea/patient_names_allergy_age", operation_id="get_patient_names_allergy_age", summary="Retrieves the first and last names of patients who have a specific allergy and died before a certain age. The allergy is identified by its description, and the age at death is provided in years.")
async def get_patient_names_allergy_age(allergy_description: str = Query(..., description="Description of the allergy"), age_at_death: int = Query(..., description="Age at death in years")):
    cursor.execute("SELECT T1.first, T1.last FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND CAST((strftime('%J', T1.deathdate) - strftime('%J', T1.birthdate)) AS REAL) / 365 < ?", (allergy_description, age_at_death))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get the count of distinct medication descriptions based on first name and last name
@app.get("/v1/synthea/count_distinct_medications", operation_id="get_count_distinct_medications", summary="Retrieves the number of unique medication descriptions associated with a patient, based on their first and last names. This operation considers the patient's medications and returns the count of distinct medication descriptions.")
async def get_count_distinct_medications(first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T2.DESCRIPTION) FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first_name, last_name))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct observation details based on first name and last name
@app.get("/v1/synthea/observation_details", operation_id="get_observation_details", summary="Retrieve unique observation details, including description, value, and units, for a patient identified by their first and last names.")
async def get_observation_details(first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT DISTINCT T2.DESCRIPTION, T2.VALUE, T2.UNITS FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"observations": []}
    return {"observations": [{"description": row[0], "value": row[1], "units": row[2]} for row in result]}

# Endpoint to get patient names based on observation description and value
@app.get("/v1/synthea/patient_names_observation", operation_id="get_patient_names_observation", summary="Retrieve the distinct first and last names of patients who have a specific observation description and a value less than a given threshold. This operation allows you to filter patients based on a particular observation and its associated value.")
async def get_patient_names_observation(description: str = Query(..., description="Description of the observation"), value: float = Query(..., description="Value of the observation")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T2.VALUE < ?", (description, value))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": [{"first_name": row[0], "last_name": row[1]} for row in result]}

# Endpoint to get top 5 observation details based on description
@app.get("/v1/synthea/top_observation_details", operation_id="get_top_observation_details", summary="Retrieves the top five distinct observation details based on a specified description. The details include the description, value, and units of the observations. The results are grouped by value and ordered by the count of each value in descending order.")
async def get_top_observation_details(description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT DISTINCT T2.DESCRIPTION, T2.VALUE, T2.UNITS FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? GROUP BY T2.VALUE ORDER BY COUNT(T2.VALUE) LIMIT 5", (description,))
    result = cursor.fetchall()
    if not result:
        return {"observations": []}
    return {"observations": [{"description": row[0], "value": row[1], "units": row[2]} for row in result]}

# Endpoint to get the count of distinct patients with a specific reason description and matching encounter and immunization dates
@app.get("/v1/synthea/patient_count_reason_description_date", operation_id="get_patient_count_reason_description_date", summary="Retrieves the number of unique patients who have a specific reason description for their encounter and whose encounter and immunization dates match. The input parameter specifies the reason description for the encounter.")
async def get_patient_count_reason_description_date(reason_description: str = Query(..., description="Reason description for the encounter")):
    cursor.execute("SELECT COUNT(DISTINCT T2.PATIENT) FROM encounters AS T1 INNER JOIN immunizations AS T2 ON T1.PATIENT = T2.PATIENT WHERE T1.REASONDESCRIPTION = ? AND T1.DATE = T2.DATE", (reason_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get care plan descriptions for a patient with a specific first and last name and a specific year
@app.get("/v1/synthea/careplan_descriptions_by_name_year", operation_id="get_careplan_descriptions_by_name_year", summary="Retrieves the descriptions of care plans associated with a patient who has a specific first and last name and whose care plan started in a specific year. The response includes a list of descriptions for the matching care plans.")
async def get_careplan_descriptions_by_name_year(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.DESCRIPTION FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND strftime('%Y', T2.START) = ?", (first, last, year))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct descriptions and patient names for a specific encounter ID
@app.get("/v1/synthea/patient_descriptions_by_encounter_id", operation_id="get_patient_descriptions_by_encounter_id", summary="Retrieves unique descriptions and patient names associated with a specific encounter ID. This operation fetches distinct descriptions from careplans, procedures, and medications, along with the first and last names of the patient linked to the provided encounter ID.")
async def get_patient_descriptions_by_encounter_id(encounter_id: str = Query(..., description="Encounter ID")):
    cursor.execute("SELECT DISTINCT T3.DESCRIPTION, T4.DESCRIPTION, T5.DESCRIPTION, T1.first, T1.last FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT INNER JOIN careplans AS T3 ON T1.patient = T3.PATIENT INNER JOIN procedures AS T4 ON T1.patient = T4.PATIENT INNER JOIN medications AS T5 ON T1.patient = T5.PATIENT WHERE T2.ID = ?", (encounter_id,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [{"careplan": row[0], "procedure": row[1], "medication": row[2], "first_name": row[3], "last_name": row[4]} for row in result]}

# Endpoint to get the count of male and female patients with a specific condition
@app.get("/v1/synthea/gender_count_by_condition", operation_id="get_gender_count_by_condition", summary="Retrieves the distinct count of male and female patients diagnosed with a specific medical condition. The condition is identified by its description.")
async def get_gender_count_by_condition(condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT COUNT(DISTINCT CASE WHEN T2.gender = 'M' THEN T2.patient END) AS Male , COUNT(DISTINCT CASE WHEN T2.gender = 'F' THEN T2.patient END) AS Female FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ?", (condition_description,))
    result = cursor.fetchone()
    if not result:
        return {"count": {"Male": 0, "Female": 0}}
    return {"count": {"Male": result[0], "Female": result[1]}}

# Endpoint to get the count of patients with a specific condition, gender, and marital status
@app.get("/v1/synthea/patient_count_by_condition_gender_marital", operation_id="get_patient_count_by_condition_gender_marital", summary="Retrieves the number of unique patients who have a specific medical condition, gender, and marital status. The condition is identified by its description, and the gender and marital status are provided as input parameters. This operation returns a count of patients who meet all three criteria.")
async def get_patient_count_by_condition_gender_marital(condition_description: str = Query(..., description="Description of the condition"), gender: str = Query(..., description="Gender of the patient"), marital_status: str = Query(..., description="Marital status of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND T2.gender = ? AND T2.marital = ?", (condition_description, gender, marital_status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct procedure and medication descriptions for a specific encounter reason
@app.get("/v1/synthea/descriptions_by_encounter_reason", operation_id="get_descriptions_by_encounter_reason", summary="Retrieve unique procedure and medication descriptions associated with a given encounter reason. This operation returns a list of distinct descriptions for procedures and medications linked to a specific encounter reason, providing valuable insights into the treatments and interventions related to that reason.")
async def get_descriptions_by_encounter_reason(reason_description: str = Query(..., description="Reason description for the encounter")):
    cursor.execute("SELECT DISTINCT T2.DESCRIPTION, T3.DESCRIPTION FROM encounters AS T1 INNER JOIN procedures AS T2 ON T1.PATIENT = T2.PATIENT INNER JOIN medications AS T3 ON T1.PATIENT = T3.PATIENT WHERE T1.REASONDESCRIPTION = ?", (reason_description,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [{"procedure": row[0], "medication": row[1]} for row in result]}

# Endpoint to get the difference in average observation values between two age ranges for a specific observation description
@app.get("/v1/synthea/observation_value_difference_by_age_range", operation_id="get_observation_value_difference_by_age_range", summary="Retrieve the difference in average observation values between two specified age ranges for a given observation description. The operation calculates the average observation value for each age range and returns the difference between them.")
async def get_observation_value_difference_by_age_range(age_range_start_1: int = Query(..., description="Start of the first age range"), age_range_end_1: int = Query(..., description="End of the first age range"), age_range_start_2: int = Query(..., description="Start of the second age range"), age_range_end_2: int = Query(..., description="End of the second age range"), observation_description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT SUM(CASE WHEN ROUND((strftime('%J', T2.DATE) - strftime('%J', T1.birthdate)) / 365) BETWEEN ? AND ? THEN T2.VALUE ELSE 0 END) / COUNT(CASE WHEN ROUND((strftime('%J', T2.DATE) - strftime('%J', T1.birthdate)) / 365) BETWEEN ? AND ? THEN T2.PATIENT END) - SUM(CASE WHEN ROUND((strftime('%J', T2.DATE) - strftime('%J', T1.birthdate)) / 365) BETWEEN ? AND ? THEN T2.VALUE ELSE 0 END) / COUNT(CASE WHEN ROUND((strftime('%J', T2.DATE) - strftime('%J', T1.birthdate)) / 365) BETWEEN ? AND ? THEN T2.PATIENT END) AS difference FROM patients AS T1 INNER JOIN observations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (age_range_start_1, age_range_end_1, age_range_start_1, age_range_end_1, age_range_start_2, age_range_end_2, age_range_start_2, age_range_end_2, observation_description))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get the percentage of patients with the most common condition after a certain age
@app.get("/v1/synthea/percentage_most_common_condition_by_age", operation_id="get_percentage_most_common_condition_by_age", summary="Retrieves the percentage of patients who have the most common medical condition after reaching a specified age threshold. The calculation is based on the number of patients with the most frequently occurring condition, divided by the total number of patients who meet the age threshold.")
async def get_percentage_most_common_condition_by_age(age_threshold: int = Query(..., description="Age threshold in years")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T5.DESCRIPTION = T3.DESCRIPTION THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T3.patient) FROM ( SELECT T2.DESCRIPTION, T1.patient FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE ROUND((strftime('%J', T2.START) - strftime('%J', T1.birthdate)) / 365) > ? GROUP BY T2.DESCRIPTION ORDER BY COUNT(T2.DESCRIPTION) DESC LIMIT 1 ) AS T3 INNER JOIN patients AS T4 ON T3.patient = T4.patient INNER JOIN conditions AS T5 ON T4.patient = T5.PATIENT WHERE ROUND((strftime('%J', T5.START) - strftime('%J', T4.birthdate)) / 365) > ?", (age_threshold, age_threshold))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get encounter reason descriptions for a patient with a specific first and last name and a specific date pattern
@app.get("/v1/synthea/encounter_reasons_by_name_date", operation_id="get_encounter_reasons_by_name_date", summary="Retrieves the descriptions of encounter reasons for a patient with a specific first and last name and a date that matches the provided pattern. The date pattern should be in the 'YYYY-MM%' format.")
async def get_encounter_reasons_by_name_date(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM%' format")):
    cursor.execute("SELECT T2.REASONDESCRIPTION FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.DATE LIKE ?", (first, last, date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"reasons": []}
    return {"reasons": [row[0] for row in result]}

# Endpoint to get the age of a patient at the time of an encounter
@app.get("/v1/synthea/patient_age_at_encounter", operation_id="get_patient_age_at_encounter", summary="Retrieves the age of a patient at the time of a specific encounter, based on the provided first and last name of the patient, the description of the encounter, and the reason for the encounter. The age is calculated as the difference between the encounter date and the patient's birthdate.")
async def get_patient_age_at_encounter(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the encounter"), reason_description: str = Query(..., description="Reason for the encounter")):
    cursor.execute("SELECT T2.DATE - T1.birthdate AS age FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND T2.DESCRIPTION = ? AND T2.REASONDESCRIPTION = ? ORDER BY T2.DATE LIMIT 1", (first, last, description, reason_description))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get the most common medication description for a given reason
@app.get("/v1/synthea/most_common_medication_description", operation_id="get_most_common_medication_description", summary="Retrieves the most frequently prescribed medication description associated with a specific medical reason. The operation returns the description of the medication that is most commonly used for the provided medical reason. The input parameter specifies the medical reason for which the most common medication description is sought.")
async def get_most_common_medication_description(reason_description: str = Query(..., description="Reason for the medication")):
    cursor.execute("SELECT DESCRIPTION FROM medications WHERE REASONDESCRIPTION = ? GROUP BY DESCRIPTION ORDER BY COUNT(DESCRIPTION) DESC LIMIT 1", (reason_description,))
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get the average number of patients per year for a specific procedure
@app.get("/v1/synthea/average_patients_per_year_by_procedure", operation_id="get_average_patients_per_year_by_procedure", summary="Retrieves the average annual patient count for a specific medical procedure. The procedure is identified by its description. The result is calculated by dividing the total number of patients by the number of distinct years in the procedure data.")
async def get_average_patients_per_year_by_procedure(description: str = Query(..., description="Description of the procedure")):
    cursor.execute("SELECT CAST(COUNT(PATIENT) AS REAL) / COUNT(DISTINCT strftime('%Y', DATE)) FROM procedures WHERE DESCRIPTION = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the timeframe of the most recent care plan for a specific patient
@app.get("/v1/synthea/careplan_timeframe", operation_id="get_careplan_timeframe", summary="Retrieves the duration of the most recent care plan for a specific patient, identified by their first and last names. The duration is calculated as the difference between the care plan's end and start dates. The description of the care plan is also returned.")
async def get_careplan_timeframe(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT strftime('%J', T2.STOP) - strftime('%J', T2.START) AS timeFrame , T2.DESCRIPTION FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? ORDER BY T2.START DESC LIMIT 1", (first, last))
    result = cursor.fetchone()
    if not result:
        return {"timeframe": [], "description": []}
    return {"timeframe": result[0], "description": result[1]}

# Endpoint to get allergy details for a specific patient
@app.get("/v1/synthea/patient_allergy_details", operation_id="get_patient_allergy_details", summary="Retrieves detailed allergy information for a patient identified by their first and last names. The response includes the start and end dates of the allergies, as well as a description of each allergy.")
async def get_patient_allergy_details(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT T2.START, T2.STOP, T2.DESCRIPTION FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first, last))
    results = cursor.fetchall()
    if not results:
        return {"allergies": []}
    return {"allergies": [{"start": row[0], "stop": row[1], "description": row[2]} for row in results]}

# Endpoint to get the age of a patient at the most recent encounter
@app.get("/v1/synthea/patient_age_at_recent_encounter", operation_id="get_patient_age_at_recent_encounter", summary="Retrieves the age of a patient at their most recent encounter, based on the provided first and last names. The age is calculated by subtracting the patient's birthdate from the date of the most recent encounter.")
async def get_patient_age_at_recent_encounter(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT T2.DATE - T1.birthdate AS age FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? ORDER BY T2.DATE DESC LIMIT 1", (first, last))
    result = cursor.fetchone()
    if not result:
        return {"age": []}
    return {"age": result[0]}

# Endpoint to get conditions for a specific patient in a given year
@app.get("/v1/synthea/patient_conditions_by_year", operation_id="get_patient_conditions_by_year", summary="Retrieves the conditions associated with a specific patient for a given year. The operation requires the patient's first and last names, as well as the year in 'YYYY' format. The response includes a list of conditions that the patient had during the specified year.")
async def get_patient_conditions_by_year(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT T2.DESCRIPTION FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND strftime('%Y', T2.START) = ?", (first, last, year))
    results = cursor.fetchall()
    if not results:
        return {"conditions": []}
    return {"conditions": [row[0] for row in results]}

# Endpoint to get the most recent immunization date for a specific patient and immunization type
@app.get("/v1/synthea/recent_immunization_date", operation_id="get_recent_immunization_date", summary="Get the most recent immunization date for a specific patient and immunization type")
async def get_recent_immunization_date(description: str = Query(..., description="Description of the immunization"), first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT T2.DATE FROM patients AS T1 INNER JOIN immunizations AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND T1.first = ? AND T1.last = ? ORDER BY T2.DATE DESC LIMIT 1", (description, first, last))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the count of distinct patients and their care plan descriptions based on a specific reason description
@app.get("/v1/synthea/patient_count_careplan_description", operation_id="get_patient_count_careplan_description", summary="Retrieve the number of unique patients and their associated care plan descriptions that match a given reason description. This operation provides a count of distinct patients and their care plan descriptions based on a specific reason description.")
async def get_patient_count_careplan_description(reason_description: str = Query(..., description="Reason description for the care plan")):
    cursor.execute("SELECT COUNT(DISTINCT T2.PATIENT), T2.DESCRIPTION FROM encounters AS T1 INNER JOIN careplans AS T2 ON T1.PATIENT = T2.PATIENT WHERE T2.REASONDESCRIPTION = ?", (reason_description,))
    result = cursor.fetchall()
    if not result:
        return {"count": [], "descriptions": []}
    return {"count": result[0][0], "descriptions": [row[1] for row in result]}

# Endpoint to get the percentage of female patients with a specific reason description in a given year
@app.get("/v1/synthea/percentage_female_patients", operation_id="get_percentage_female_patients", summary="Retrieves the percentage of female patients who had an encounter with a specific reason description in a given year. The calculation is based on the total number of patients who had an encounter with the same reason description in the specified year.")
async def get_percentage_female_patients(year: str = Query(..., description="Year in 'YYYY' format"), reason_description: str = Query(..., description="Reason description for the encounter")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.gender = 'F' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.PATIENT) FROM encounters AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE strftime('%Y', T1.DATE) = ? AND T1.REASONDESCRIPTION = ?", (year, reason_description))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of patients and the percentage of a specific encounter description within a date range
@app.get("/v1/synthea/patient_count_encounter_percentage", operation_id="get_patient_count_encounter_percentage", summary="Retrieves the total number of patients and the percentage of a specific encounter type within a specified date range. The operation filters patients by their first and last names, and encounters by their description. The date range is defined by the start and end years.")
async def get_patient_count_encounter_percentage(first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient"), description: str = Query(..., description="Description of the encounter"), start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.patient), CAST(SUM(CASE WHEN T2.DESCRIPTION = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.patient) FROM patients AS T1 INNER JOIN encounters AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ? AND strftime('%Y', T2.DATE) BETWEEN ? AND ?", (description, first_name, last_name, start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "percentage": []}
    return {"count": result[0], "percentage": result[1]}

# Endpoint to get distinct start dates of care plans for patients with a specific maiden name
@app.get("/v1/synthea/careplan_start_dates", operation_id="get_careplan_start_dates", summary="Retrieve the unique start dates of care plans for patients who share a specific maiden name. This operation allows you to filter care plan start dates based on the maiden name of the patients.")
async def get_careplan_start_dates(maiden_name: str = Query(..., description="Maiden name of the patient")):
    cursor.execute("SELECT DISTINCT T1.START FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.maiden = ?", (maiden_name,))
    result = cursor.fetchall()
    if not result:
        return {"start_dates": []}
    return {"start_dates": [row[0] for row in result]}

# Endpoint to get the first and last name of the first patient with a specific gender based on care plan start date
@app.get("/v1/synthea/first_patient_by_gender", operation_id="get_first_patient_by_gender", summary="Retrieves the first and last name of the patient with the earliest care plan start date for a specified gender.")
async def get_first_patient_by_gender(gender: str = Query(..., description="Gender of the patient")):
    cursor.execute("SELECT T2.first, T2.last FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.gender = ? ORDER BY T1.START LIMIT 1", (gender,))
    result = cursor.fetchone()
    if not result:
        return {"first_name": [], "last_name": []}
    return {"first_name": result[0], "last_name": result[1]}

# Endpoint to get the percentage of Hispanic patients with care plans stopped in a specific year
@app.get("/v1/synthea/percentage_hispanic_patients", operation_id="get_percentage_hispanic_patients", summary="Retrieves the percentage of Hispanic patients whose care plans were stopped in a specific year. The calculation is based on the total number of patients with stopped care plans in the given year.")
async def get_percentage_hispanic_patients(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.race = 'hispanic' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.PATIENT) FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE strftime('%Y', T1.stop) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct reason descriptions for care plans of a specific patient
@app.get("/v1/synthea/careplan_reason_descriptions", operation_id="get_careplan_reason_descriptions", summary="Retrieve unique reason descriptions associated with the care plans of a specific patient. This operation requires the patient's first and last names as input parameters to filter the care plans and return the distinct reason descriptions.")
async def get_careplan_reason_descriptions(first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT DISTINCT T1.REASONDESCRIPTION FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.first = ? AND T2.last = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"reason_descriptions": []}
    return {"reason_descriptions": [row[0] for row in result]}

# Endpoint to get distinct care plan descriptions for patients of a specific ethnicity
@app.get("/v1/synthea/careplan_descriptions_by_ethnicity", operation_id="get_careplan_descriptions_by_ethnicity", summary="Retrieve unique care plan descriptions for patients of a specified ethnicity. This operation filters care plans based on the provided ethnicity and returns a list of distinct descriptions.")
async def get_careplan_descriptions_by_ethnicity(ethnicity: str = Query(..., description="Ethnicity of the patient")):
    cursor.execute("SELECT DISTINCT T1.DESCRIPTION FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.ethnicity = ?", (ethnicity,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get distinct encounter IDs for patients born in a specific place
@app.get("/v1/synthea/encounter_ids_by_birthplace", operation_id="get_encounter_ids_by_birthplace", summary="Retrieve a unique set of encounter IDs for patients who were born in a specified location. This operation filters encounters based on the birthplace of the associated patients, providing a focused list of encounter records.")
async def get_encounter_ids_by_birthplace(birthplace: str = Query(..., description="Birthplace of the patient")):
    cursor.execute("SELECT DISTINCT T1.ENCOUNTER FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.birthplace = ?", (birthplace,))
    result = cursor.fetchall()
    if not result:
        return {"encounter_ids": []}
    return {"encounter_ids": [row[0] for row in result]}

# Endpoint to get distinct start dates of care plans for patients who are still alive
@app.get("/v1/synthea/careplans/distinct_start_dates_alive_patients", operation_id="get_distinct_start_dates_alive_patients", summary="Retrieves the unique start dates of care plans for patients who are currently alive. This operation provides a comprehensive list of distinct start dates, offering insights into the initiation of care plans for living patients.")
async def get_distinct_start_dates_alive_patients():
    cursor.execute("SELECT DISTINCT T1.START FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.deathdate IS NULL")
    result = cursor.fetchall()
    if not result:
        return {"start_dates": []}
    return {"start_dates": [row[0] for row in result]}

# Endpoint to get the count of distinct patients with a specific race and reason code in their care plans
@app.get("/v1/synthea/careplans/count_patients_race_reasoncode", operation_id="get_count_patients_race_reasoncode", summary="Retrieves the number of unique patients who have a specified race and reason code in their care plans. The race and reason code are provided as input parameters.")
async def get_count_patients_race_reasoncode(race: str = Query(..., description="Race of the patient"), reason_code: str = Query(..., description="Reason code in the care plan")):
    cursor.execute("SELECT COUNT(DISTINCT T1.PATIENT) FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.race = ? AND T1.REASONCODE = ?", (race, reason_code))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct first and last names of patients with a specific care plan description
@app.get("/v1/synthea/careplans/distinct_patient_names_description", operation_id="get_distinct_patient_names_description", summary="Retrieves a list of unique first and last names of patients who have a care plan with the specified description. This operation does not return any duplicate names.")
async def get_distinct_patient_names_description(description: str = Query(..., description="Description of the care plan")):
    cursor.execute("SELECT DISTINCT T2.first, T2.last FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get distinct stop dates of care plans for deceased patients
@app.get("/v1/synthea/careplans/distinct_stop_dates_deceased_patients", operation_id="get_distinct_stop_dates_deceased_patients", summary="Retrieves the unique end dates of care plans associated with patients who have passed away. This operation provides a comprehensive view of the last dates care plans were active for deceased patients, without duplication.")
async def get_distinct_stop_dates_deceased_patients():
    cursor.execute("SELECT DISTINCT T1.STOP FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.deathdate IS NOT NULL AND T1.STOP IS NOT NULL")
    result = cursor.fetchall()
    if not result:
        return {"stop_dates": []}
    return {"stop_dates": [row[0] for row in result]}

# Endpoint to get the count of distinct patients with a specific ethnicity and care plan code
@app.get("/v1/synthea/careplans/count_patients_ethnicity_code", operation_id="get_count_patients_ethnicity_code", summary="Retrieves the number of unique patients who have a specified ethnicity and a particular code in their care plan.")
async def get_count_patients_ethnicity_code(ethnicity: str = Query(..., description="Ethnicity of the patient"), code: str = Query(..., description="Code in the care plan")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.ethnicity = ? AND T1.CODE = ?", (ethnicity, code))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of female patients with care plans starting in a specific year
@app.get("/v1/synthea/careplans/percentage_female_patients_year", operation_id="get_percentage_female_patients_year", summary="Retrieves the percentage of female patients who have care plans starting in a specific year. The calculation is based on the total number of patients with care plans in the given year. The year is provided as a four-digit input parameter.")
async def get_percentage_female_patients_year(year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.gender = 'F' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.PATIENT) AS percentage FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE strftime('%Y', T1.START) = ?", (year,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of distinct patients with a specific race and care plans stopping in a specific year
@app.get("/v1/synthea/careplans/count_patients_race_stop_year", operation_id="get_count_patients_race_stop_year", summary="Retrieves the number of unique patients who have a specified race and care plans that ended in a given year. The input parameters include the race of the patient and the year (in 'YYYY' format) when the care plans concluded.")
async def get_count_patients_race_stop_year(race: str = Query(..., description="Race of the patient"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.race = ? AND strftime('%Y', T1.STOP) = ?", (race, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients with a specific condition and gender
@app.get("/v1/synthea/patients/count_patients_condition_gender", operation_id="get_count_patients_condition_gender", summary="Retrieves the number of unique patients who have a specified medical condition and are of a certain gender. The condition is identified by its description, and the gender is specified as either 'male' or 'female'.")
async def get_count_patients_condition_gender(description: str = Query(..., description="Description of the condition"), gender: str = Query(..., description="Gender of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN conditions AS T2 WHERE T2.DESCRIPTION = ? AND T1.gender = ?", (description, gender))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct first and last names of patients with a specific allergy description
@app.get("/v1/synthea/patients/distinct_patient_names_allergy", operation_id="get_distinct_patient_names_allergy", summary="Retrieves a list of unique first and last names of patients who have a specified allergy. The allergy description is used to filter the patients.")
async def get_distinct_patient_names_allergy(description: str = Query(..., description="Description of the allergy")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last FROM patients AS T1 INNER JOIN allergies AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get condition descriptions for a patient with a specific first and last name
@app.get("/v1/synthea/patients/conditions_by_name", operation_id="get_conditions_by_name", summary="Retrieves the descriptions of all conditions associated with a patient, identified by their first and last name. The endpoint returns a list of condition descriptions for the specified patient.")
async def get_conditions_by_name(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT T2.DESCRIPTION FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first, last))
    result = cursor.fetchall()
    if not result:
        return {"conditions": []}
    return {"conditions": [row[0] for row in result]}

# Endpoint to get distinct patient names who have taken a specific medication starting from a given year
@app.get("/v1/synthea/patient_names_by_medication_year", operation_id="get_patient_names_by_medication_year", summary="Retrieve a list of unique patient names who have been prescribed a specific medication from a specified year onwards. The medication is identified by its description, and the start year is provided in 'YYYY' format.")
async def get_patient_names_by_medication_year(medication_description: str = Query(..., description="Description of the medication"), start_year: str = Query(..., description="Start year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ? AND strftime('%Y', T2.START) >= ?", (medication_description, start_year))
    result = cursor.fetchall()
    if not result:
        return {"patient_names": []}
    return {"patient_names": result}

# Endpoint to get the count of distinct patients with a specific condition, birth year, and race
@app.get("/v1/synthea/patient_count_by_condition_birthyear_race", operation_id="get_patient_count_by_condition_birthyear_race", summary="Retrieves the number of unique patients who have a specific medical condition, were born in a certain year, and belong to a particular racial group. The response is based on the provided birth year, race, and condition description.")
async def get_patient_count_by_condition_birthyear_race(birth_year: str = Query(..., description="Birth year in 'YYYY' format"), race: str = Query(..., description="Race of the patient"), condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT COUNT(DISTINCT T1.patient) FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.patient WHERE strftime('%Y', T1.birthdate) = ? AND T1.race = ? AND T2.DESCRIPTION = ?", (birth_year, race, condition_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct condition descriptions for patients with a specific immunization
@app.get("/v1/synthea/condition_descriptions_by_immunization", operation_id="get_condition_descriptions_by_immunization", summary="Retrieve unique condition descriptions for patients who have received a specific immunization. The immunization is identified by its description.")
async def get_condition_descriptions_by_immunization(immunization_description: str = Query(..., description="Description of the immunization")):
    cursor.execute("SELECT DISTINCT T2.DESCRIPTION FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT INNER JOIN immunizations AS T3 ON T1.patient = T3.PATIENT WHERE T3.DESCRIPTION = ?", (immunization_description,))
    result = cursor.fetchall()
    if not result:
        return {"condition_descriptions": []}
    return {"condition_descriptions": result}

# Endpoint to get the patient with the highest occurrence of a specific condition
@app.get("/v1/synthea/top_patient_by_condition_occurrence", operation_id="get_top_patient_by_condition_occurrence", summary="Retrieves the patient with the highest recorded occurrences of a specific medical condition. This operation identifies the patient with the most instances of a particular condition by cross-referencing patient, condition, and prevalence data. The result is the patient with the highest number of occurrences for the specified condition.")
async def get_top_patient_by_condition_occurrence():
    cursor.execute("SELECT T1.patient FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT INNER JOIN all_prevalences AS T3 ON T3.ITEM = T2.DESCRIPTION ORDER BY T3.OCCURRENCES DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"patient": []}
    return {"patient": result[0]}

# Endpoint to get distinct patient names with a specific condition
@app.get("/v1/synthea/patient_names_by_condition", operation_id="get_patient_names_by_condition", summary="Retrieve a list of unique patient names who have been diagnosed with a specific medical condition. The condition is identified by its description.")
async def get_patient_names_by_condition(condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT DISTINCT T1.first, T1.last FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.patient WHERE T2.DESCRIPTION = ?", (condition_description,))
    result = cursor.fetchall()
    if not result:
        return {"patient_names": []}
    return {"patient_names": result}

# Endpoint to get distinct care plan start dates for a specific patient
@app.get("/v1/synthea/careplan_start_dates_by_patient", operation_id="get_careplan_start_dates_by_patient", summary="Retrieve the unique start dates of care plans associated with a specific patient, identified by their first and last names.")
async def get_careplan_start_dates_by_patient(first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT DISTINCT T2.start FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"start_dates": []}
    return {"start_dates": result}

# Endpoint to get distinct care plan descriptions for a specific patient
@app.get("/v1/synthea/careplan_descriptions_by_patient", operation_id="get_careplan_descriptions_by_patient", summary="Retrieves unique care plan descriptions associated with a specific patient, identified by their first and last names.")
async def get_careplan_descriptions_by_patient(first_name: str = Query(..., description="First name of the patient"), last_name: str = Query(..., description="Last name of the patient")):
    cursor.execute("SELECT DISTINCT T2.DESCRIPTION FROM patients AS T1 INNER JOIN careplans AS T2 ON T1.patient = T2.PATIENT WHERE T1.first = ? AND T1.last = ?", (first_name, last_name))
    result = cursor.fetchall()
    if not result:
        return {"careplan_descriptions": []}
    return {"careplan_descriptions": result}

# Endpoint to get the percentage of male patients with a specific condition
@app.get("/v1/synthea/percentage_male_patients_by_condition", operation_id="get_percentage_male_patients_by_condition", summary="Retrieves the percentage of male patients diagnosed with a specific medical condition. The condition is identified by its description. The result is calculated by dividing the count of male patients with the condition by the total number of patients with the condition, then multiplying by 100.")
async def get_percentage_male_patients_by_condition(condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.gender = 'M' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.patient) FROM patients AS T1 INNER JOIN conditions AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (condition_description,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of female patients prescribed a specific medication
@app.get("/v1/synthea/percentage_female_patients_medication", operation_id="get_percentage_female_patients_medication", summary="Retrieves the percentage of female patients who have been prescribed a specific medication. The medication is identified by its description. The result is calculated by dividing the count of female patients with the specified medication by the total number of patients with that medication, then multiplying by 100 to obtain a percentage.")
async def get_percentage_female_patients_medication(description: str = Query(..., description="Description of the medication")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.gender = 'F' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.patient) FROM patients AS T1 INNER JOIN medications AS T2 ON T1.patient = T2.PATIENT WHERE T2.DESCRIPTION = ?", (description,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average value of a specific observation for patients of a specific race
@app.get("/v1/synthea/average_observation_value_by_race", operation_id="get_average_observation_value_by_race", summary="Retrieves the average value of a specified observation for patients belonging to a particular race. The operation requires the race of the patients and the description of the observation as input parameters.")
async def get_average_observation_value_by_race(race: str = Query(..., description="Race of the patient"), description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT AVG(T1.VALUE) FROM observations AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.race = ? AND T1.DESCRIPTION = ?", (race, description))
    result = cursor.fetchone()
    if not result:
        return {"average_value": []}
    return {"average_value": result[0]}

# Endpoint to get care plan descriptions for patients at a specific address
@app.get("/v1/synthea/care_plan_descriptions_by_address", operation_id="get_care_plan_descriptions_by_address", summary="Retrieves the descriptions of care plans for patients residing at the specified address. The operation filters the care plans based on the provided address and returns the corresponding descriptions.")
async def get_care_plan_descriptions_by_address(address: str = Query(..., description="Address of the patient")):
    cursor.execute("SELECT T1.DESCRIPTION FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.address = ?", (address,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get allergy descriptions for patients with specific first name, last name, and ethnicity
@app.get("/v1/synthea/allergy_descriptions_by_name_ethnicity", operation_id="get_allergy_descriptions_by_name_ethnicity", summary="Retrieves detailed descriptions of allergies for patients who share a specific first name, last name, and ethnicity. This operation allows for targeted allergy information retrieval based on these demographic parameters.")
async def get_allergy_descriptions_by_name_ethnicity(first: str = Query(..., description="First name of the patient"), last: str = Query(..., description="Last name of the patient"), ethnicity: str = Query(..., description="Ethnicity of the patient")):
    cursor.execute("SELECT T1.DESCRIPTION FROM allergies AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.first = ? AND T2.last = ? AND T2.ethnicity = ?", (first, last, ethnicity))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the count of distinct patients with specific conditions and allergies
@app.get("/v1/synthea/count_patients_conditions_allergies", operation_id="get_count_patients_conditions_allergies", summary="Retrieves the count of unique patients who have a specific medical condition and a particular allergy. The operation filters patients based on the provided descriptions of the condition and allergy.")
async def get_count_patients_conditions_allergies(condition_description: str = Query(..., description="Description of the condition"), allergy_description: str = Query(..., description="Description of the allergy")):
    cursor.execute("SELECT COUNT(DISTINCT T2.patient) FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient INNER JOIN allergies AS T3 ON T2.patient = T3.PATIENT WHERE T1.DESCRIPTION = ? AND T3.DESCRIPTION = ?", (condition_description, allergy_description))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first and last names of patients with a specific condition and age at death
@app.get("/v1/synthea/patient_names_condition_age_at_death", operation_id="get_patient_names_condition_age_at_death", summary="Retrieves the first and last names of patients who have a specified medical condition and died at the age of 44. The condition is identified by its description, and the age at death is calculated based on the patient's birthdate and deathdate.")
async def get_patient_names_condition_age_at_death(condition_description: str = Query(..., description="Description of the condition"), age_at_death: int = Query(..., description="Age at death in years")):
    cursor.execute("SELECT T2.first, T2.last FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND ROUND((strftime('%J', T2.deathdate) - strftime('%J', T2.birthdate)) / 365) = ?", (condition_description, age_at_death))
    result = cursor.fetchall()
    if not result:
        return {"names": []}
    return {"names": [{"first": row[0], "last": row[1]} for row in result]}

# Endpoint to get the SSN of the patient with the highest value for a specific observation
@app.get("/v1/synthea/ssn_highest_observation_value", operation_id="get_ssn_highest_observation_value", summary="Retrieves the Social Security Number (SSN) of the patient who has the highest recorded value for a specific observation. The observation is identified by its description. The result is determined by comparing the values of the specified observation across all patients and selecting the patient with the highest value.")
async def get_ssn_highest_observation_value(description: str = Query(..., description="Description of the observation")):
    cursor.execute("SELECT T2.ssn FROM observations AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? ORDER BY T1.VALUE DESC LIMIT 1", (description,))
    result = cursor.fetchone()
    if not result:
        return {"ssn": []}
    return {"ssn": result[0]}

# Endpoint to get the care plan description with the highest prevalence percentage
@app.get("/v1/synthea/highest_prevalence_care_plan", operation_id="get_highest_prevalence_care_plan", summary="Retrieves the description of the care plan with the highest prevalence percentage. This operation identifies the condition with the highest prevalence percentage, then determines the associated care plan. The prevalence percentage is calculated based on the frequency of the condition in the database.")
async def get_highest_prevalence_care_plan():
    cursor.execute("SELECT T4.DESCRIPTION FROM all_prevalences AS T1 INNER JOIN conditions AS T2 ON T2.DESCRIPTION = T1.ITEM INNER JOIN encounters AS T3 ON T2.ENCOUNTER = T3.ID INNER JOIN careplans AS T4 ON T4.ENCOUNTER = T3.ID ORDER BY T1.\"PREVALENCE PERCENTAGE\" DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"description": []}
    return {"description": result[0]}

# Endpoint to get distinct care plan descriptions for a patient with a specific SSN
@app.get("/v1/synthea/care_plan_descriptions_by_ssn", operation_id="get_care_plan_descriptions_by_ssn", summary="Retrieves unique care plan descriptions associated with a patient identified by their Social Security Number (SSN).")
async def get_care_plan_descriptions_by_ssn(ssn: str = Query(..., description="SSN of the patient")):
    cursor.execute("SELECT DISTINCT T1.DESCRIPTION FROM careplans AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T2.ssn = ?", (ssn,))
    result = cursor.fetchall()
    if not result:
        return {"descriptions": []}
    return {"descriptions": [row[0] for row in result]}

# Endpoint to get the average age of patients with a specific condition and gender
@app.get("/v1/synthea/average_age_condition_gender", operation_id="get_average_age_condition_gender", summary="Get the average age of patients with a specific condition and gender")
async def get_average_age_condition_gender(condition_description: str = Query(..., description="Description of the condition"), gender: str = Query(..., description="Gender of the patient")):
    cursor.execute("SELECT SUM(CASE WHEN T2.deathdate IS NULL THEN ROUND((strftime('%J', date('now')) - strftime('%J', T2.birthdate)) / 365) ELSE ROUND((strftime('%J', T2.deathdate) - strftime('%J', T2.birthdate)) / 365) END) / COUNT(T2.patient) FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND T2.gender = ?", (condition_description, gender))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get distinct first and last names of patients based on medication reason description
@app.get("/v1/synthea/distinct_patient_names_by_reason", operation_id="get_distinct_patient_names_by_reason", summary="Retrieves a list of unique patient names, based on a specified medication reason description. The number of results returned can be limited by the user.")
async def get_distinct_patient_names_by_reason(reason_description: str = Query(..., description="Reason description for the medication"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T2.first, T2.last FROM medications AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.REASONDESCRIPTION = ? LIMIT ?", (reason_description, limit))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": result}

# Endpoint to get patient names based on observation description, units, and value
@app.get("/v1/synthea/patient_names_by_observation", operation_id="get_patient_names_by_observation", summary="Retrieves the first and last names of patients who have a specific observation, as defined by the provided description, units, and value. This operation filters the observations based on the input parameters and returns the corresponding patient names.")
async def get_patient_names_by_observation(description: str = Query(..., description="Description of the observation"), units: str = Query(..., description="Units of the observation"), value: float = Query(..., description="Value of the observation")):
    cursor.execute("SELECT T2.first, T2.last FROM observations AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND T1.UNITS = ? AND T1.VALUE = ?", (description, units, value))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": result}

# Endpoint to get distinct birthdates of patients based on medication description and gender
@app.get("/v1/synthea/distinct_birthdates_by_medication", operation_id="get_distinct_birthdates_by_medication", summary="Retrieves a list of unique birthdates for patients who have been prescribed a specific medication and match a given gender. The number of results returned can be limited by specifying a maximum count.")
async def get_distinct_birthdates_by_medication(description: str = Query(..., description="Description of the medication"), gender: str = Query(..., description="Gender of the patient"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T2.birthdate FROM medications AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ? AND T2.gender = ? LIMIT ?", (description, gender, limit))
    result = cursor.fetchall()
    if not result:
        return {"birthdates": []}
    return {"birthdates": result}

# Endpoint to get distinct patient names based on ethnicity and prevalence threshold
@app.get("/v1/synthea/distinct_patient_names_by_ethnicity_prevalence", operation_id="get_distinct_patient_names_by_ethnicity_prevalence", summary="Retrieves a list of distinct patient names that meet the specified ethnicity and prevalence threshold criteria. The prevalence threshold is used to filter patients based on the occurrence of their conditions relative to the average occurrence across all conditions. The ethnicity parameter is used to further narrow down the results to a specific ethnic group.")
async def get_distinct_patient_names_by_ethnicity_prevalence(ethnicity: str = Query(..., description="Ethnicity of the patient"), prevalence_threshold: float = Query(..., description="Prevalence threshold percentage")):
    cursor.execute("SELECT DISTINCT T2.first, T2.last FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient INNER JOIN all_prevalences AS T3 ON T1.DESCRIPTION = T3.ITEM WHERE T2.ethnicity = ? AND 100 * CAST(T3.OCCURRENCES AS REAL) / ( SELECT AVG(OCCURRENCES) FROM all_prevalences ) > ?", (ethnicity, prevalence_threshold))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": result}

# Endpoint to get the difference in counts of married and single patients with a specific condition
@app.get("/v1/synthea/marital_status_difference_by_condition", operation_id="get_marital_status_difference_by_condition", summary="Retrieve the difference in the number of patients with a specific condition who are in the first marital status versus the second marital status. This operation allows for a comparison of the prevalence of a given condition between two marital statuses.")
async def get_marital_status_difference_by_condition(marital_status_1: str = Query(..., description="First marital status (e.g., 'M' for married)"), marital_status_2: str = Query(..., description="Second marital status (e.g., 'S' for single)"), condition_description: str = Query(..., description="Description of the condition")):
    cursor.execute("SELECT SUM(CASE WHEN T2.marital = ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.marital = ? THEN 1 ELSE 0 END) FROM conditions AS T1 INNER JOIN patients AS T2 ON T1.PATIENT = T2.patient WHERE T1.DESCRIPTION = ?", (marital_status_1, marital_status_2, condition_description))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

api_calls = [
    "/v1/synthea/observations_value_units?first=Elly&last=Koss&date=2008-03-11&description=Body%20Height",
    "/v1/synthea/observations_increase?year1=2009&year2=2008&first=Elly&last=Koss&description=Body%20Height",
    "/v1/synthea/observations_max_value?first=Elly&last=Koss&description=Systolic%20Blood%20Pressure",
    "/v1/synthea/observations_count?first=Elly&last=Koss&description=Systolic%20Blood%20Pressure",
    "/v1/synthea/patient_max_observation_value?description=Systolic%20Blood%20Pressure",
    "/v1/synthea/medication_duration?first=Elly&last=Koss&description=Acetaminophen%25",
    "/v1/synthea/patient_medications?first=Elly&last=Koss",
    "/v1/synthea/medication_reasons?first=Elly&last=Koss&description=Acetaminophen%25",
    "/v1/synthea/medications_by_reason?first=Elly&last=Koss&reasondescription=Streptococcal%20sore%20throat%20(disorder)",
    "/v1/synthea/patients_by_medication?description=Acetaminophen%25",
    "/v1/synthea/patient_condition_description?first=Elly&last=Koss&start_date=2009-01-08",
    "/v1/synthea/condition_duration?first=Elly&last=Koss&description=Cystitis",
    "/v1/synthea/average_observation_value?first=Elly&last=Koss&description=Body%20Weight",
    "/v1/synthea/married_patients_percentage?description=Cystitis",
    "/v1/synthea/patient_observation_details?prefix=Mr.&first=Vincent&last=Wyman&date=2010-08-02&description=Body%20Height",
    "/v1/synthea/care_plan_count?prefix=Mrs.&first=Norman&last=Berge",
    "/v1/synthea/medication_reason_description?prefix=Mrs.&first=Annabelle&last=Pouros&start_date=1970-12-19&description=Leucovorin%20100%20MG%20Injection",
    "/v1/synthea/prevalence_percentage?code=64859006",
    "/v1/synthea/prevalence_rate?code=368581000119106",
    "/v1/synthea/patient_procedure_descriptions?prefix=Ms.&first=Jacquelyn&last=Shanahan&date=2009-08-09",
    "/v1/synthea/count_billable_periods?prefix=Ms.&first=Abbie&last=Cole&start_date=2010-12-31&end_date=2012-01-01",
    "/v1/synthea/count_distinct_allergy_codes?prefix=Mrs.&first=Saundra&last=Monahan",
    "/v1/synthea/patient_names_by_billable_period?billable_period=1947-09-11",
    "/v1/synthea/encounter_descriptions?prefix=Mr.&first=Hubert&last=Baumbach&encounter_date=2008-10-25",
    "/v1/synthea/condition_descriptions?first=Keven&last=Kuhn&start_date=2016-09-24&stop_date=2016-10-10",
    "/v1/synthea/procedure_dates?prefix=Mrs.&first=Ira&last=Deckow&description=Standard%20pregnancy%20test",
    "/v1/synthea/average_careplan_duration?prefix=Mr.&first=Wesley&last=Lemke",
    "/v1/synthea/average_condition_duration?prefix=Ms.&first=Angelena&last=Kertzmann&description=Normal%20pregnancy",
    "/v1/synthea/most_recent_allergy_stop",
    "/v1/synthea/count_patients_by_condition_and_year?description=Diabetes&start_year=1988",
    "/v1/synthea/allergy_count?description=Allergy%20to%20eggs",
    "/v1/synthea/latest_condition_start?description=Hypertension",
    "/v1/synthea/most_common_allergy",
    "/v1/synthea/highest_observation_value?description=Body%20Height&units=cm",
    "/v1/synthea/most_common_condition_gender_ethnicity?gender=F&ethnicity=american",
    "/v1/synthea/medication_count_ethnicity_year_duration?description=Ibuprofen%20200%20MG%20Oral%20Tablet&ethnicity=dominican&start_year=2016&duration=1",
    "/v1/synthea/condition_count_birth_year?description=Pneumonia&birth_year_pattern=192%",
    "/v1/synthea/patient_names_medication_year?description=Yaz%2028%20Day%20Pack&start_year=2011",
    "/v1/synthea/immunization_count_race_year?race=black&description=DTaP&year=2013",
    "/v1/synthea/highest_prevalence_latest_start",
    "/v1/synthea/patient_count_prevalence_rate",
    "/v1/synthea/most_common_condition_description?immunization_description=Influenza%20seasonal%20injectable%20preservative%20free",
    "/v1/synthea/distinct_patients_prevalence_percentage?prevalence_percentage=18.8",
    "/v1/synthea/condition_description_count_patient_name?first_name=Tyree&last_name=Eichmann",
    "/v1/synthea/patient_count_immunization_condition?immunization_description=meningococcal%20MCV4P&condition_description=Viral%20sinusitis%20(disorder)",
    "/v1/synthea/patient_count_gender_condition?gender=F&condition_description=Prediabetes",
    "/v1/synthea/patient_name_observation?observation_description=Body%20Mass%20Index&observation_units=kg%2Fm2",
    "/v1/synthea/patient_age_name_condition?first_name=Giovanni&last_name=Russel&condition_description=Hypertension",
    "/v1/synthea/patient_count_medication_race_gender?medication_description=oxaliplatin%205%20MG%2FML%20%5BEloxatin%5D&race=asian&gender=F",
    "/v1/synthea/patient_count_condition_alive?condition_description=Stroke",
    "/v1/synthea/allergy_patient_count?allergy_description=Allergy%20to%20peanuts&race=asian",
    "/v1/synthea/avg_observation_value?condition_description=Hypertension&observation_description=Diastolic%20Blood%20Pressure",
    "/v1/synthea/medications_by_ssn?ssn=999-94-3751",
    "/v1/synthea/ssn_by_allergy_ethnicity_gender?allergy_description=Allergy%20to%20grass%20pollen&ethnicity=irish&gender=F",
    "/v1/synthea/patient_names_by_careplan_code?careplan_code=315043002",
    "/v1/synthea/longest_lifespan_condition",
    "/v1/synthea/most_prevalent_condition_code",
    "/v1/synthea/observations_by_address?observation_description=Glucose&address=365%20Della%20Crossroad%20Suite%20202%20Deerfield%20MA%2001342%20US",
    "/v1/synthea/ssn_by_prevalence_threshold?prevalence_threshold=30",
    "/v1/synthea/condition_gender_race_percentage?gender=F&race=asian&condition_description=Acute%20bronchitis%20(disorder)",
    "/v1/synthea/encounter_count_by_patient_name?first=Major&last=D'Amore",
    "/v1/synthea/procedure_descriptions_by_patient_name?first=Emmy&last=Waelchi",
    "/v1/synthea/patient_names_by_procedure?description=Extraction%20of%20wisdom%20tooth",
    "/v1/synthea/observations_by_patient_name_and_description?first=Elly&last=Koss&description=Body%20Weight",
    "/v1/synthea/patient_names_by_allergy?description=Allergy%20to%20soya",
    "/v1/synthea/immunization_count_by_patient_name_and_description?first=Keven&last=Kuhn&description=DTaP",
    "/v1/synthea/patient_names_by_medication_duration?description=Clopidogrel%2075%20MG%20Oral%20Tablet&duration=10",
    "/v1/synthea/procedure_medication_descriptions_by_condition?description=Third%20degree%20burn",
    "/v1/synthea/medication_descriptions_by_allergy_and_start_date?start_date=6/6/16&description=Allergy%20to%20mould",
    "/v1/synthea/careplan_descriptions_by_condition?description=Secondary%20malignant%20neoplasm%20of%20colon",
    "/v1/synthea/patient_details_by_observation?description=Systolic%20Blood%20Pressure&value=200&units=mmHg&year=2011",
    "/v1/synthea/immunization_ethnicity_stats?description=Influenza%20seasonal%20injectable%20preservative%20free&ethnicity=english&year=2017",
    "/v1/synthea/distinct_patient_first_names_by_encounter_reason?reason_description=Normal%20pregnancy",
    "/v1/synthea/distinct_patient_birthdates_by_encounter_description?description=Outpatient%20Encounter",
    "/v1/synthea/distinct_patient_first_names_by_condition_description?description=Cystitis",
    "/v1/synthea/count_distinct_patients_by_condition_marital?description=Stroke&marital=M",
    "/v1/synthea/distinct_patient_addresses_by_billable_period?billable_period=2010%25",
    "/v1/synthea/distinct_patient_last_names_by_allergy_description?description=Allergy%20to%20dairy%20product",
    "/v1/synthea/allergy_start_dates_by_patient_name?first_name=Adolfo&last_name=Schmitt",
    "/v1/synthea/patient_count_allergy_gender?allergy_description=House%20dust%20mite%20allergy&gender=M",
    "/v1/synthea/most_common_allergy_by_race?race=white",
    "/v1/synthea/patient_names_by_immunization?immunization_description=Influenza%20seasonal%20injectable%20preservative%20free",
    "/v1/synthea/patient_count_immunization_gender?immunization_description=HPV%20quadrivalent&gender=F",
    "/v1/synthea/encounter_descriptions_by_birthplace?birthplace=Pittsfield%20MA%20US",
    "/v1/synthea/allergy_count_by_ethnicity?ethnicity=german",
    "/v1/synthea/average_age_by_careplan_reason?careplan_reason=Prediabetes",
    "/v1/synthea/patient_count_medication_reason_alive?medication_reason=Coronary%20Heart%20Disease",
    "/v1/synthea/patient_count_procedure_drivers?procedure_description=Bone%20immobilization",
    "/v1/synthea/patient_names_by_allergy_count?allergy_count=3",
    "/v1/synthea/patient_count_allergies_immunizations?allergy_description=Allergy%20to%20eggs&immunization_description=Td%20(adult)%20preservative%20free",
    "/v1/synthea/patient_count_careplans_observations?observation_description=Body%20Weight&careplan_description=Diabetes%20self%20management%20plan&observation_value=100&observation_units=kg",
    "/v1/synthea/most_common_gender_allergy?allergy_description=Dander%20(animal)%20allergy",
    "/v1/synthea/distinct_billable_periods?last_name=Dickinson",
    "/v1/synthea/distinct_patient_names_condition?condition_description=Otitis%20media",
    "/v1/synthea/patient_count_medications_ethnicity?medication_reason=Myocardial%20Infarction&ethnicity=irish",
    "/v1/synthea/patient_count_marital_careplan?marital_status=M&careplan_reason=Concussion%20with%20loss%20of%20consciousness",
    "/v1/synthea/patient_count_immunizations_careplan_stop?immunization_description=rotavirus%20monovalent&careplan_stop_date=2013-11-23",
    "/v1/synthea/patient_count_medications_gender?medication_description=Nitroglycerin%200.4%20MG/ACTUAT%20[Nitrolingual]&gender=F",
    "/v1/synthea/percentage_patients_allergy_birthplace?allergy_description=Allergy%20to%20grass%20pollen&birthplace=Pembroke%20MA%20US",
    "/v1/synthea/average_body_weight?race=asian&description=Body%20Weight&units=kg",
    "/v1/synthea/ssn_by_allergy?description=Latex%20allergy",
    "/v1/synthea/allergy_duration?first=Isadora&last=Moen",
    "/v1/synthea/patients_by_marital_status_and_duration?marital=M&duration=60",
    "/v1/synthea/immunization_dates?description=Influenza%20seasonal%20injectable%20preservative%20free&first=Elly&last=Koss",
    "/v1/synthea/immunization_count_by_race?description=meningococcal%20MCV4P&start_date=2010-07-09&end_date=2013-10-29&race=black",
    "/v1/synthea/immunization_codes_and_dates?prefix=Ms.&first=Jacquelyn&last=Shanahan&description=Influenza%20seasonal%20injectable%20preservative%20free",
    "/v1/synthea/medication_count_by_marital_status?marital=S&reason_description=Cystitis&description=Nitrofurantoin%205%20MG/ML%20[Furadantin]&year=2010",
    "/v1/synthea/encounter_reason_description?date=2013-11-20&first_name=Lavelle&last_name=Vandervort",
    "/v1/synthea/patient_ssn_address?date=2008-06-13&reason_description=Viral%20sinusitis%20(disorder)&description=Encounter%20for%20symptom",
    "/v1/synthea/medication_details?encounter_id=23c293ec-dbae-4a22-896e-f12cf3c8bac3",
    "/v1/synthea/patient_names_allergy_age?allergy_description=Shellfish%20allergy&age_at_death=12",
    "/v1/synthea/count_distinct_medications?first_name=Major&last_name=D'Amore",
    "/v1/synthea/observation_details?first_name=Bella&last_name=Rolfson",
    "/v1/synthea/patient_names_observation?description=Calcium&value=8.6",
    "/v1/synthea/top_observation_details?description=Body%20Mass%20Index",
    "/v1/synthea/patient_count_reason_description_date?reason_description=Normal%20pregnancy",
    "/v1/synthea/careplan_descriptions_by_name_year?first=Elly&last=Koss&year=2013",
    "/v1/synthea/patient_descriptions_by_encounter_id?encounter_id=6f2e3935-b203-493e-a9c0-f23e847b9798",
    "/v1/synthea/gender_count_by_condition?condition_description=Hypertension",
    "/v1/synthea/patient_count_by_condition_gender_marital?condition_description=Normal%20pregnancy&gender=F&marital_status=S",
    "/v1/synthea/descriptions_by_encounter_reason?reason_description=Drug%20overdose",
    "/v1/synthea/observation_value_difference_by_age_range?age_range_start_1=20&age_range_end_1=30&age_range_start_2=50&age_range_end_2=60&observation_description=Glucose",
    "/v1/synthea/percentage_most_common_condition_by_age?age_threshold=60",
    "/v1/synthea/encounter_reasons_by_name_date?first=Walter&last=Bahringer&date_pattern=2009-07%",
    "/v1/synthea/patient_age_at_encounter?first=Stacy&last=Morar&description=Emergency%20Room%20Admission&reason_description=Drug%20overdose",
    "/v1/synthea/most_common_medication_description?reason_description=Child%20attention%20deficit%20disorder",
    "/v1/synthea/average_patients_per_year_by_procedure?description=Combined%20chemotherapy%20and%20radiation%20therapy%20(procedure)",
    "/v1/synthea/careplan_timeframe?first=Jacquelyn&last=Shanahan",
    "/v1/synthea/patient_allergy_details?first=Isadora&last=Moen",
    "/v1/synthea/patient_age_at_recent_encounter?first=Laronda&last=Bernier",
    "/v1/synthea/patient_conditions_by_year?first=Joye&last=Homenick&year=2017",
    "/v1/synthea/recent_immunization_date?description=Influenza%20seasonal%20injectable%20preservative%20free&first=Joye&last=Homenick",
    "/v1/synthea/patient_count_careplan_description?reason_description=Second%20degree%20burn",
    "/v1/synthea/percentage_female_patients?year=2010&reason_description=Contact%20dermatitis",
    "/v1/synthea/patient_count_encounter_percentage?first_name=Lorri&last_name=Simonis&description=Prenatal%20visit&start_year=2010&end_year=2017",
    "/v1/synthea/careplan_start_dates?maiden_name=Adams",
    "/v1/synthea/first_patient_by_gender?gender=M",
    "/v1/synthea/percentage_hispanic_patients?year=2011",
    "/v1/synthea/careplan_reason_descriptions?first_name=Angelo&last_name=Buckridge",
    "/v1/synthea/careplan_descriptions_by_ethnicity?ethnicity=american",
    "/v1/synthea/encounter_ids_by_birthplace?birthplace=Pembroke%20MA%20US",
    "/v1/synthea/careplans/distinct_start_dates_alive_patients",
    "/v1/synthea/careplans/count_patients_race_reasoncode?race=white&reason_code=10509002",
    "/v1/synthea/careplans/distinct_patient_names_description?description=Diabetic%20diet",
    "/v1/synthea/careplans/distinct_stop_dates_deceased_patients",
    "/v1/synthea/careplans/count_patients_ethnicity_code?ethnicity=italian&code=304510005",
    "/v1/synthea/careplans/percentage_female_patients_year?year=2010",
    "/v1/synthea/careplans/count_patients_race_stop_year?race=black&year=2017",
    "/v1/synthea/patients/count_patients_condition_gender?description=Prediabetes&gender=M",
    "/v1/synthea/patients/distinct_patient_names_allergy?description=Allergy%20to%20nut",
    "/v1/synthea/patients/conditions_by_name?first=Wilmer&last=Koepp",
    "/v1/synthea/patient_names_by_medication_year?medication_description=Penicillin%20V%20Potassium%20250%20MG&start_year=1948",
    "/v1/synthea/patient_count_by_condition_birthyear_race?birth_year=1935&race=white&condition_description=Stroke",
    "/v1/synthea/condition_descriptions_by_immunization?immunization_description=IPV",
    "/v1/synthea/top_patient_by_condition_occurrence",
    "/v1/synthea/patient_names_by_condition?condition_description=Cystitis",
    "/v1/synthea/careplan_start_dates_by_patient?first_name=Walter&last_name=Bahringer",
    "/v1/synthea/careplan_descriptions_by_patient?first_name=Major&last_name=D'Amore",
    "/v1/synthea/percentage_male_patients_by_condition?condition_description=Viral%20sinusitis%20(disorder)",
    "/v1/synthea/percentage_female_patients_medication?description=Penicillin%20V%20Potassium%20250%20MG",
    "/v1/synthea/average_observation_value_by_race?race=white&description=Body%20Height",
    "/v1/synthea/care_plan_descriptions_by_address?address=179%20Sydni%20Roads%20Taunton%20MA%2002780%20US",
    "/v1/synthea/allergy_descriptions_by_name_ethnicity?first=Dirk&last=Langosh&ethnicity=dominican",
    "/v1/synthea/count_patients_conditions_allergies?condition_description=Asthma&allergy_description=Allergy%20to%20peanuts",
    "/v1/synthea/patient_names_condition_age_at_death?condition_description=Drug%20overdose&age_at_death=44",
    "/v1/synthea/ssn_highest_observation_value?description=Systolic%20Blood%20Pressure",
    "/v1/synthea/highest_prevalence_care_plan",
    "/v1/synthea/care_plan_descriptions_by_ssn?ssn=999-15-3685",
    "/v1/synthea/average_age_condition_gender?condition_description=Hypertension&gender=M",
    "/v1/synthea/distinct_patient_names_by_reason?reason_description=Streptococcal%20sore%20throat%20(disorder)&limit=5",
    "/v1/synthea/patient_names_by_observation?description=Body%20Weight&units=kg&value=61.97",
    "/v1/synthea/distinct_birthdates_by_medication?description=Penicillin%20V%20Potassium%20250%20MG&gender=M&limit=5",
    "/v1/synthea/distinct_patient_names_by_ethnicity_prevalence?ethnicity=irish&prevalence_threshold=96",
    "/v1/synthea/marital_status_difference_by_condition?marital_status_1=M&marital_status_2=S&condition_description=Diabetes"
]
