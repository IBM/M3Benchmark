from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/thrombosis_prediction/thrombosis_prediction.sqlite')
cursor = conn.cursor()

# Endpoint to get the admission ratio for male patients
@app.get("/v1/thrombosis_prediction/admission_ratio_male", operation_id="get_admission_ratio_male", summary="Retrieves the percentage of male patients who were admitted, out of the total number of male patients. The sex of the patient is specified as an input parameter.")
async def get_admission_ratio_male(sex: str = Query(..., description="Sex of the patient (M for male)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Admission = '+' THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN Admission = '-' THEN 1 ELSE 0 END) FROM Patient WHERE SEX = ?", (sex,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the percentage of female patients born after a specific year
@app.get("/v1/thrombosis_prediction/percentage_female_born_after_year", operation_id="get_percentage_female_born_after_year", summary="Retrieves the percentage of female patients in the database who were born after a specified year. The year is provided in the 'YYYY' format. The calculation is based on the total number of female patients in the database.")
async def get_percentage_female_born_after_year(year: str = Query(..., description="Year in 'YYYY' format"), sex: str = Query(..., description="Sex of the patient (F for female)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN STRFTIME('%Y', Birthday) > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Patient WHERE SEX = ?", (year, sex))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the admission percentage for patients born between two years
@app.get("/v1/thrombosis_prediction/admission_percentage_between_years", operation_id="get_admission_percentage_between_years", summary="Retrieves the percentage of admitted patients born between the specified start and end years. The calculation is based on the total number of patients born within the given range.")
async def get_admission_percentage_between_years(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Admission = '+' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Patient WHERE STRFTIME('%Y', Birthday) BETWEEN ? AND ?", (start_year, end_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the admission ratio for patients with a specific diagnosis
@app.get("/v1/thrombosis_prediction/admission_ratio_diagnosis", operation_id="get_admission_ratio_diagnosis", summary="Retrieves the ratio of admitted patients to discharged patients for a specific medical diagnosis. The diagnosis is provided as an input parameter.")
async def get_admission_ratio_diagnosis(diagnosis: str = Query(..., description="Diagnosis of the patient")):
    cursor.execute("SELECT SUM(CASE WHEN Admission = '+' THEN 1.0 ELSE 0 END) / SUM(CASE WHEN Admission = '-' THEN 1 ELSE 0 END) FROM Patient WHERE Diagnosis = ?", (diagnosis,))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the diagnosis and date for a specific patient ID
@app.get("/v1/thrombosis_prediction/diagnosis_date_by_patient_id", operation_id="get_diagnosis_date_by_patient_id", summary="Retrieves the diagnosis and corresponding date for a specific patient, identified by the provided patient ID.")
async def get_diagnosis_date_by_patient_id(patient_id: int = Query(..., description="ID of the patient")):
    cursor.execute("SELECT T1.Diagnosis, T2.Date FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.ID = ?", (patient_id,))
    result = cursor.fetchall()
    if not result:
        return {"diagnosis_date": []}
    return {"diagnosis_date": result}

# Endpoint to get the sex, birthday, examination date, and symptoms for a specific patient ID
@app.get("/v1/thrombosis_prediction/patient_details_by_id", operation_id="get_patient_details_by_id", summary="Retrieves the gender, birth date, examination date, and symptoms of a patient identified by the provided patient ID.")
async def get_patient_details_by_id(patient_id: int = Query(..., description="ID of the patient")):
    cursor.execute("SELECT T1.SEX, T1.Birthday, T2.`Examination Date`, T2.Symptoms FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T1.ID = ?", (patient_id,))
    result = cursor.fetchall()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get distinct patient details with LDH greater than a specific value
@app.get("/v1/thrombosis_prediction/patient_details_ldh_greater_than", operation_id="get_patient_details_ldh_greater_than", summary="Retrieves unique patient details for individuals with a LDH value greater than the provided input. The response includes the patient's ID, sex, and birthday.")
async def get_patient_details_ldh_greater_than(ldh_value: int = Query(..., description="LDH value")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.LDH > ?", (ldh_value,))
    result = cursor.fetchall()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get distinct patient IDs and their age with a specific RVVT value
@app.get("/v1/thrombosis_prediction/patient_age_rvvt_value", operation_id="get_patient_age_rvvt_value", summary="Get distinct patient IDs and their age with a specific RVVT value")
async def get_patient_age_rvvt_value(rvvt_value: str = Query(..., description="RVVT value")):
    cursor.execute("SELECT DISTINCT T1.ID, STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.Birthday) FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T2.RVVT = ?", (rvvt_value,))
    result = cursor.fetchall()
    if not result:
        return {"patient_age": []}
    return {"patient_age": result}

# Endpoint to get distinct patient details with a specific thrombosis value
@app.get("/v1/thrombosis_prediction/patient_details_thrombosis_value", operation_id="get_patient_details_thrombosis_value", summary="Retrieve unique patient records, including their ID, sex, and diagnosis, based on a specified thrombosis value. This operation filters patients who have undergone examinations and returns those with the provided thrombosis value.")
async def get_patient_details_thrombosis_value(thrombosis_value: int = Query(..., description="Thrombosis value")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX, T1.Diagnosis FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T2.Thrombosis = ?", (thrombosis_value,))
    result = cursor.fetchall()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get distinct patient IDs born in a specific year with T-CHO greater than or equal to a specific value
@app.get("/v1/thrombosis_prediction/patient_ids_birthyear_tcho", operation_id="get_patient_ids_birthyear_tcho", summary="Retrieves unique patient identifiers born in a specified year who have a T-CHO level at or above a given value. This operation filters patients based on their birth year and T-CHO level, providing a list of distinct patient IDs that meet the criteria.")
async def get_patient_ids_birthyear_tcho(birth_year: str = Query(..., description="Birth year in 'YYYY' format"), tcho_value: int = Query(..., description="T-CHO value")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', T1.Birthday) = ? AND T2.`T-CHO` >= ?", (birth_year, tcho_value))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": result}

# Endpoint to get distinct patient details based on ALB level
@app.get("/v1/thrombosis_prediction/patient_details_by_alb", operation_id="get_patient_details_by_alb", summary="Retrieve unique patient records with ALB levels below the specified threshold. The response includes patient ID, sex, and diagnosis.")
async def get_patient_details_by_alb(alb_level: float = Query(..., description="ALB level threshold")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX, T1.Diagnosis FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.ALB < ?", (alb_level,))
    result = cursor.fetchall()
    if not result:
        return {"details": []}
    return {"details": result}

# Endpoint to get the percentage of female patients with TP levels outside a specified range
@app.get("/v1/thrombosis_prediction/percentage_female_tp_range", operation_id="get_percentage_female_tp_range", summary="Retrieves the percentage of female patients whose TP levels fall outside the specified range. The range is defined by a lower and upper threshold. The calculation is based on the total number of patients in the database.")
async def get_percentage_female_tp_range(sex: str = Query(..., description="Sex of the patient"), tp_lower: float = Query(..., description="Lower threshold for TP level"), tp_upper: float = Query(..., description="Upper threshold for TP level")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T1.SEX = ? AND (T2.TP < ? OR T2.TP > ?) THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.SEX = ?", (sex, tp_lower, tp_upper, sex))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average aCL IgG level for patients above a certain age and with a specific admission status
@app.get("/v1/thrombosis_prediction/avg_acl_igg_by_age_admission", operation_id="get_avg_acl_igg_by_age_admission", summary="Get the average aCL IgG level for patients above a certain age and with a specific admission status")
async def get_avg_acl_igg_by_age_admission(age_threshold: int = Query(..., description="Age threshold"), admission_status: str = Query(..., description="Admission status")):
    cursor.execute("SELECT AVG(T2.`aCL IgG`) FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.Birthday) >= ? AND T1.Admission = ?", (age_threshold, admission_status))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get the count of patients based on description year, sex, and admission status
@app.get("/v1/thrombosis_prediction/patient_count_by_description_year_sex_admission", operation_id="get_patient_count_by_description_year_sex_admission", summary="Retrieves the total number of patients who match the specified description year, sex, and admission status. The description year is a four-digit year derived from the patient's description. The sex and admission status parameters further refine the patient count.")
async def get_patient_count_by_description_year_sex_admission(description_year: str = Query(..., description="Description year in 'YYYY' format"), sex: str = Query(..., description="Sex of the patient"), admission_status: str = Query(..., description="Admission status")):
    cursor.execute("SELECT COUNT(*) FROM Patient WHERE STRFTIME('%Y', Description) = ? AND SEX = ? AND Admission = ?", (description_year, sex, admission_status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the minimum age at first date
@app.get("/v1/thrombosis_prediction/min_age_at_first_date", operation_id="get_min_age_at_first_date", summary="Retrieves the minimum age difference in years between the earliest recorded 'First Date' and the 'Birthday' of any patient in the database.")
async def get_min_age_at_first_date():
    cursor.execute("SELECT MIN(STRFTIME('%Y', `First Date`) - STRFTIME('%Y', Birthday)) FROM Patient")
    result = cursor.fetchone()
    if not result:
        return {"min_age": []}
    return {"min_age": result[0]}

# Endpoint to get the count of patients with specific sex, examination year, and thrombosis status
@app.get("/v1/thrombosis_prediction/patient_count_by_sex_examination_year_thrombosis", operation_id="get_patient_count_by_sex_examination_year_thrombosis", summary="Retrieves the total number of patients who match the specified sex, examination year, and thrombosis status. The response includes a count of patients that meet the provided criteria.")
async def get_patient_count_by_sex_examination_year_thrombosis(sex: str = Query(..., description="Sex of the patient"), examination_year: str = Query(..., description="Examination year in 'YYYY' format"), thrombosis_status: int = Query(..., description="Thrombosis status")):
    cursor.execute("SELECT COUNT(*) FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T1.SEX = ? AND STRFTIME('%Y', T2.`Examination Date`) = ? AND T2.Thrombosis = ?", (sex, examination_year, thrombosis_status))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the age range of patients with TG level above a specified value
@app.get("/v1/thrombosis_prediction/age_range_by_tg_level", operation_id="get_age_range_by_tg_level", summary="Retrieves the age range of patients with TG levels at or above a specified threshold. The age range is calculated as the difference between the maximum and minimum birth years of qualifying patients.")
async def get_age_range_by_tg_level(tg_level: int = Query(..., description="TG level threshold")):
    cursor.execute("SELECT STRFTIME('%Y', MAX(T1.Birthday)) - STRFTIME('%Y', MIN(T1.Birthday)) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.TG >= ?", (tg_level,))
    result = cursor.fetchone()
    if not result:
        return {"age_range": []}
    return {"age_range": result[0]}

# Endpoint to get the latest patient diagnosis and symptoms
@app.get("/v1/thrombosis_prediction/latest_patient_diagnosis_symptoms", operation_id="get_latest_patient_diagnosis_symptoms", summary="Retrieves the most recent diagnosis and associated symptoms for a patient. The operation returns the latest record from the patient and examination tables, based on the patient's birthday, where symptoms are not null.")
async def get_latest_patient_diagnosis_symptoms():
    cursor.execute("SELECT T2.Symptoms, T1.Diagnosis FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T2.Symptoms IS NOT NULL ORDER BY T1.Birthday DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"diagnosis_symptoms": []}
    return {"diagnosis_symptoms": result}

# Endpoint to get the average monthly count of patients based on year and sex
@app.get("/v1/thrombosis_prediction/avg_monthly_patient_count_by_year_sex", operation_id="get_avg_monthly_patient_count_by_year_sex", summary="Retrieves the average monthly count of patients for a given year and sex. This operation calculates the total number of patients for the specified year and sex, then divides it by 12 to provide an average monthly count. The year should be provided in 'YYYY' format, and the sex of the patient is also required.")
async def get_avg_monthly_patient_count_by_year_sex(year: str = Query(..., description="Year in 'YYYY' format"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT CAST(COUNT(T1.ID) AS REAL) / 12 FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', T2.Date) = ? AND T1.SEX = ?", (year, sex))
    result = cursor.fetchone()
    if not result:
        return {"average_monthly_count": []}
    return {"average_monthly_count": result[0]}

# Endpoint to get the earliest patient details based on diagnosis
@app.get("/v1/thrombosis_prediction/earliest_patient_details_by_diagnosis", operation_id="get_earliest_patient_details_by_diagnosis", summary="Retrieves the earliest patient details, including date of birth and age, based on a specified diagnosis. The patient's age is calculated as the difference between the current year and their birth year. The data is fetched from the Laboratory and Patient tables, with a join on the patient's ID. Only the patient with the earliest birth date is returned.")
async def get_earliest_patient_details_by_diagnosis(diagnosis: str = Query(..., description="Diagnosis of the patient")):
    cursor.execute("SELECT T1.Date, STRFTIME('%Y', T2.`First Date`) - STRFTIME('%Y', T2.Birthday), T2.Birthday FROM Laboratory AS T1 INNER JOIN Patient AS T2 ON T1.ID = T2.ID WHERE T2.Diagnosis = ? AND T2.Birthday IS NOT NULL ORDER BY T2.Birthday ASC LIMIT 1", (diagnosis,))
    result = cursor.fetchone()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get the ratio of male patients with UA <= 8.0 to female patients with UA <= 6.5
@app.get("/v1/thrombosis_prediction/ua_ratio_by_sex", operation_id="get_ua_ratio_by_sex", summary="Retrieves the ratio of male patients with uric acid (UA) levels at or below a specified threshold to female patients with UA levels at or below another specified threshold. The calculation is based on the total number of patients who meet the respective UA criteria for each sex.")
async def get_ua_ratio_by_sex(ua_male: float = Query(..., description="UA threshold for male patients"), sex_male: str = Query(..., description="Sex of the patient (M for male)"), ua_female: float = Query(..., description="UA threshold for female patients"), sex_female: str = Query(..., description="Sex of the patient (F for female)")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.UA <= ? AND T1.SEX = ? THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.UA <= ? AND T1.SEX = ? THEN 1 ELSE 0 END) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID", (ua_male, sex_male, ua_female, sex_female))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of distinct patients admitted with a specific admission status and a minimum duration since the first date
@app.get("/v1/thrombosis_prediction/patient_count_by_admission_duration", operation_id="get_patient_count_by_admission_duration", summary="Get the count of distinct patients admitted with a specific admission status and a minimum duration since the first date")
async def get_patient_count_by_admission_duration(admission: str = Query(..., description="Admission status of the patient"), min_duration: int = Query(..., description="Minimum duration since the first date in years")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T1.Admission = ? AND STRFTIME('%Y', T2.`Examination Date`) - STRFTIME('%Y', T1.`First Date`) >= ?", (admission, min_duration))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients examined within a specific year range and age limit
@app.get("/v1/thrombosis_prediction/patient_count_by_examination_year_age", operation_id="get_patient_count_by_examination_year_age", summary="Retrieve the total number of patients who underwent an examination within a specified year range and were below a certain age limit at the time of examination. The start and end years define the examination date range, while the maximum age sets the upper limit for patient age.")
async def get_patient_count_by_examination_year_age(start_year: str = Query(..., description="Start year of the examination date range"), end_year: str = Query(..., description="End year of the examination date range"), max_age: int = Query(..., description="Maximum age of the patient")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', T2.`Examination Date`) BETWEEN ? AND ? AND STRFTIME('%Y', T2.`Examination Date`) - STRFTIME('%Y', T1.Birthday) < ?", (start_year, end_year, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patients with a specific T-BIL level and sex
@app.get("/v1/thrombosis_prediction/patient_count_by_tbil_sex", operation_id="get_patient_count_by_tbil_sex", summary="Retrieves the number of unique patients who have a T-BIL level equal to or above the specified value and belong to the given sex category.")
async def get_patient_count_by_tbil_sex(tbil: float = Query(..., description="T-BIL level"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`T-BIL` >= ? AND T1.SEX = ?", (tbil, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common diagnosis within a specific examination date range
@app.get("/v1/thrombosis_prediction/most_common_diagnosis_by_examination_date", operation_id="get_most_common_diagnosis_by_examination_date", summary="Retrieve the most frequently occurring diagnosis among patients who had examinations within the specified date range. The date range is defined by the start_date and end_date parameters, both in 'YYYY-MM-DD' format. The diagnosis is determined by aggregating and ordering the diagnoses of patients who had examinations within the provided date range, then selecting the most common one.")
async def get_most_common_diagnosis_by_examination_date(start_date: str = Query(..., description="Start date of the examination date range in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the examination date range in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Diagnosis FROM Examination AS T1 INNER JOIN Patient AS T2 ON T1.ID = T2.ID WHERE T1.`Examination Date` BETWEEN ? AND ? GROUP BY T2.Diagnosis ORDER BY COUNT(T2.Diagnosis) DESC LIMIT 1", (start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"diagnosis": []}
    return {"diagnosis": result[0]}

# Endpoint to get the average age of patients within a specific laboratory date range
@app.get("/v1/thrombosis_prediction/average_age_by_laboratory_date", operation_id="get_average_age_by_laboratory_date", summary="Retrieves the average age of patients whose laboratory tests were conducted within a specified date range. The age is calculated based on the provided year and the patients' birthdays. The start and end dates of the range are also provided to filter the laboratory tests.")
async def get_average_age_by_laboratory_date(year: int = Query(..., description="Year to calculate the age difference"), start_date: str = Query(..., description="Start date of the laboratory date range in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date of the laboratory date range in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT AVG(? - STRFTIME('%Y', T2.Birthday)) FROM Laboratory AS T1 INNER JOIN Patient AS T2 ON T1.ID = T2.ID WHERE T1.Date BETWEEN ? AND ?", (year, start_date, end_date))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get the age and diagnosis of the patient with the highest HGB level
@app.get("/v1/thrombosis_prediction/age_diagnosis_by_highest_hgb", operation_id="get_age_diagnosis_by_highest_hgb", summary="Retrieves the age and diagnosis of the patient with the highest recorded hemoglobin (HGB) level. The age is calculated based on the difference between the patient's birth year and the year of the laboratory test. The diagnosis refers to the patient's medical condition.")
async def get_age_diagnosis_by_highest_hgb():
    cursor.execute("SELECT STRFTIME('%Y', T2.Date) - STRFTIME('%Y', T1.Birthday), T1.Diagnosis FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID ORDER BY T2.HGB DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"age": [], "diagnosis": []}
    return {"age": result[0], "diagnosis": result[1]}

# Endpoint to get the ANA value for a specific patient and examination date
@app.get("/v1/thrombosis_prediction/ana_by_patient_examination_date", operation_id="get_ana_by_patient_examination_date", summary="Retrieves the ANA value for a specific patient on a given examination date. The operation requires the patient's unique ID and the examination date in 'YYYY-MM-DD' format to locate the corresponding record in the database.")
async def get_ana_by_patient_examination_date(patient_id: int = Query(..., description="ID of the patient"), examination_date: str = Query(..., description="Examination date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT ANA FROM Examination WHERE ID = ? AND `Examination Date` = ?", (patient_id, examination_date))
    result = cursor.fetchone()
    if not result:
        return {"ana": []}
    return {"ana": result[0]}

# Endpoint to get the T-CHO status for a specific patient and laboratory date
@app.get("/v1/thrombosis_prediction/tcho_status_by_patient_laboratory_date", operation_id="get_tcho_status_by_patient_laboratory_date", summary="Retrieves the T-CHO status (normal or abnormal) for a specific patient on a given laboratory date. The T-CHO status is determined by comparing the patient's T-CHO value to a provided threshold. The patient is identified by their unique ID, and the laboratory date is provided in 'YYYY-MM-DD' format.")
async def get_tcho_status_by_patient_laboratory_date(tcho_threshold: float = Query(..., description="T-CHO threshold value"), patient_id: int = Query(..., description="ID of the patient"), laboratory_date: str = Query(..., description="Laboratory date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CASE WHEN `T-CHO` < ? THEN 'Normal' ELSE 'Abnormal' END FROM Laboratory WHERE ID = ? AND Date = ?", (tcho_threshold, patient_id, laboratory_date))
    result = cursor.fetchone()
    if not result:
        return {"tcho_status": []}
    return {"tcho_status": result[0]}

# Endpoint to get the sex of the earliest diagnosed patient with a specific diagnosis
@app.get("/v1/thrombosis_prediction/sex_by_earliest_diagnosis", operation_id="get_sex_by_earliest_diagnosis", summary="Retrieves the sex of the patient who was first diagnosed with the specified condition. The diagnosis parameter is used to filter the patients, and the result is ordered by the earliest diagnosis date.")
async def get_sex_by_earliest_diagnosis(diagnosis: str = Query(..., description="Diagnosis of the patient")):
    cursor.execute("SELECT SEX FROM Patient WHERE Diagnosis = ? AND `First Date` IS NOT NULL ORDER BY `First Date` ASC LIMIT 1", (diagnosis,))
    result = cursor.fetchone()
    if not result:
        return {"sex": []}
    return {"sex": result[0]}

# Endpoint to get aCL IgA, aCL IgG, aCL IgM values for patients with a specific diagnosis, description, and examination date
@app.get("/v1/thrombosis_prediction/aCL_values", operation_id="get_aCL_values", summary="Retrieves the aCL IgA, aCL IgG, and aCL IgM values for patients who have a specific diagnosis and description, and were examined on a particular date. The response includes the requested values for all patients who meet the specified criteria.")
async def get_aCL_values(diagnosis: str = Query(..., description="Diagnosis of the patient"), description: str = Query(..., description="Description of the patient"), examination_date: str = Query(..., description="Examination date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT `aCL IgA`, `aCL IgG`, `aCL IgM` FROM Examination WHERE ID IN ( SELECT ID FROM Patient WHERE Diagnosis = ? AND Description = ? ) AND `Examination Date` = ?", (diagnosis, description, examination_date))
    result = cursor.fetchall()
    if not result:
        return {"aCL_values": []}
    return {"aCL_values": result}

# Endpoint to get the sex of patients with specific GPT value and date
@app.get("/v1/thrombosis_prediction/patient_sex", operation_id="get_patient_sex", summary="Retrieves the sex of patients who have a specific GPT value and date. The GPT value and date are used to filter the patients, ensuring that only those with the specified GPT value and date are included in the results.")
async def get_patient_sex(gpt: float = Query(..., description="GPT value"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.SEX FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GPT = ? AND T2.Date = ?", (gpt, date))
    result = cursor.fetchall()
    if not result:
        return {"sex": []}
    return {"sex": result}

# Endpoint to get the age difference in years between the laboratory date and the patient's birthday for a specific UA value and date
@app.get("/v1/thrombosis_prediction/age_difference", operation_id="get_age_difference", summary="Retrieves the age difference in years between a patient's birthday and a specific laboratory date, given a particular UA value and date. This operation calculates the age difference by subtracting the year of the patient's birthday from the year of the laboratory date. The input parameters include the UA value and the laboratory date in 'YYYY-MM-DD' format.")
async def get_age_difference(ua: float = Query(..., description="UA value"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT STRFTIME('%Y', T2.Date) - STRFTIME('%Y', T1.Birthday) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.UA = ? AND T2.Date = ?", (ua, date))
    result = cursor.fetchall()
    if not result:
        return {"age_difference": []}
    return {"age_difference": result}

# Endpoint to get the count of laboratory records for a specific first date, diagnosis, and year
@app.get("/v1/thrombosis_prediction/lab_record_count", operation_id="get_lab_record_count", summary="Retrieves the total number of laboratory records associated with a specific patient diagnosis and year. The patient is identified by a first date, and the year is determined by the date of the laboratory record. This operation is useful for tracking the volume of laboratory records for a given diagnosis and year.")
async def get_lab_record_count(first_date: str = Query(..., description="First date in 'YYYY-MM-DD' format"), diagnosis: str = Query(..., description="Diagnosis of the patient"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(*) FROM Laboratory WHERE ID = ( SELECT ID FROM Patient WHERE `First Date` = ? AND Diagnosis = ? ) AND STRFTIME('%Y', Date) = ?", (first_date, diagnosis, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the diagnosis of patients with a specific examination date and diagnosis
@app.get("/v1/thrombosis_prediction/patient_diagnosis", operation_id="get_patient_diagnosis", summary="Retrieves the diagnosis of a patient who had their first examination on a specific date and was diagnosed with a particular condition. The examination date and diagnosis are provided as input parameters.")
async def get_patient_diagnosis(examination_date: str = Query(..., description="Examination date in 'YYYY-MM-DD' format"), diagnosis: str = Query(..., description="Diagnosis of the patient")):
    cursor.execute("SELECT T1.Diagnosis FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T1.ID = ( SELECT ID FROM Examination WHERE `Examination Date` = ? AND Diagnosis = ? ) AND T2.`Examination Date` = T1.`First Date`", (examination_date, diagnosis))
    result = cursor.fetchall()
    if not result:
        return {"diagnosis": []}
    return {"diagnosis": result}

# Endpoint to get the symptoms of patients with a specific birthday and examination date
@app.get("/v1/thrombosis_prediction/patient_symptoms", operation_id="get_patient_symptoms", summary="Retrieves the symptoms of patients who were born on a specific date and underwent an examination on a particular date. The operation requires the patient's birthday and examination date as input parameters.")
async def get_patient_symptoms(birthday: str = Query(..., description="Birthday in 'YYYY-MM-DD' format"), examination_date: str = Query(..., description="Examination date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T2.Symptoms FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T1.Birthday = ? AND T2.`Examination Date` = ?", (birthday, examination_date))
    result = cursor.fetchall()
    if not result:
        return {"symptoms": []}
    return {"symptoms": result}

# Endpoint to calculate the difference in T-CHO values between two months for patients with a specific birthday
@app.get("/v1/thrombosis_prediction/tcho_difference", operation_id="get_tcho_difference", summary="Retrieve the relative difference in T-CHO values for a specific patient between two given months. The calculation is based on the sum of T-CHO values for each month, with the result expressed as a real number. The patient is identified by their birthday.")
async def get_tcho_difference(month1: str = Query(..., description="First month in 'YYYY-MM-%' format"), month2: str = Query(..., description="Second month in 'YYYY-MM-%' format"), birthday: str = Query(..., description="Birthday in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT CAST((SUM(CASE WHEN T2.Date LIKE ? THEN T2.`T-CHO` ELSE 0 END) - SUM(CASE WHEN T2.Date LIKE ? THEN T2.`T-CHO` ELSE 0 END)) AS REAL) / SUM(CASE WHEN T2.Date LIKE ? THEN T2.`T-CHO` ELSE 0 END) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.Birthday = ?", (month1, month2, month2, birthday))
    result = cursor.fetchone()
    if not result:
        return {"tcho_difference": []}
    return {"tcho_difference": result[0]}

# Endpoint to get the IDs of examinations within a specific date range and diagnosis
@app.get("/v1/thrombosis_prediction/examination_ids", operation_id="get_examination_ids", summary="Retrieves the unique identifiers of examinations that fall within a specified date range and match a given diagnosis. The start and end dates should be provided in 'YYYY-MM-DD' format, and the diagnosis must be an exact match.")
async def get_examination_ids(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), diagnosis: str = Query(..., description="Diagnosis of the patient")):
    cursor.execute("SELECT ID FROM Examination WHERE `Examination Date` BETWEEN ? AND ? AND Diagnosis = ?", (start_date, end_date, diagnosis))
    result = cursor.fetchall()
    if not result:
        return {"examination_ids": []}
    return {"examination_ids": result}

# Endpoint to get distinct IDs of laboratory records within a specific date range, GPT value, and ALB value
@app.get("/v1/thrombosis_prediction/lab_ids", operation_id="get_lab_ids", summary="Retrieves unique laboratory record identifiers that fall within a specified date range and meet the given GPT and ALB value criteria. The operation filters records based on the provided start and end dates, GPT value, and ALB value, returning only the distinct IDs of the records that match these conditions.")
async def get_lab_ids(start_date: str = Query(..., description="Start date in 'YYYY-MM-DD' format"), end_date: str = Query(..., description="End date in 'YYYY-MM-DD' format"), gpt: float = Query(..., description="GPT value"), alb: float = Query(..., description="ALB value")):
    cursor.execute("SELECT DISTINCT ID FROM Laboratory WHERE Date BETWEEN ? AND ? AND GPT > ? AND ALB < ?", (start_date, end_date, gpt, alb))
    result = cursor.fetchall()
    if not result:
        return {"lab_ids": []}
    return {"lab_ids": result}

# Endpoint to get the IDs of patients with a specific birth year, sex, and admission status
@app.get("/v1/thrombosis_prediction/patient_ids", operation_id="get_patient_ids", summary="Get the IDs of patients with a specific birth year, sex, and admission status")
async def get_patient_ids(birth_year: str = Query(..., description="Birth year in 'YYYY' format"), sex: str = Query(..., description="Sex of the patient"), admission: str = Query(..., description="Admission status of the patient")):
    cursor.execute("SELECT ID FROM Patient WHERE STRFTIME('%Y', Birthday) = ? AND SEX = ? AND Admission = ?", (birth_year, sex, admission))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": result}

# Endpoint to get the count of examinations with specific thrombosis, ANA pattern, and aCL IgM criteria
@app.get("/v1/thrombosis_prediction/examination_count_thrombosis_ana_acl", operation_id="get_examination_count", summary="Retrieves the count of examinations that meet specific thrombosis, ANA pattern, and aCL IgM criteria. The aCL IgM value must be greater than 1.2 times the average aCL IgM value for the specified thrombosis and ANA pattern. The thrombosis value, ANA pattern, and aCL IgM multiplier are provided as input parameters.")
async def get_examination_count(thrombosis: int = Query(..., description="Thrombosis value"), ana_pattern: str = Query(..., description="ANA pattern"), acl_multiplier: float = Query(..., description="Multiplier for average aCL IgM")):
    cursor.execute("SELECT COUNT(*) FROM Examination WHERE Thrombosis = ? AND `ANA Pattern` = ? AND `aCL IgM` > (SELECT AVG(`aCL IgM`) * ? FROM Examination WHERE Thrombosis = ? AND `ANA Pattern` = ?)", (thrombosis, ana_pattern, acl_multiplier, thrombosis, ana_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of laboratory records with UA below a threshold and U-PRO within a range
@app.get("/v1/thrombosis_prediction/percentage_ua_u_pro_range", operation_id="get_percentage_ua_u_pro", summary="Get the percentage of laboratory records with UA below a threshold and U-PRO within a range")
async def get_percentage_ua_u_pro(ua_threshold: float = Query(..., description="UA threshold"), u_pro_min: float = Query(..., description="Minimum U-PRO value"), u_pro_max: float = Query(..., description="Maximum U-PRO value")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN UA <= ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(ID) FROM Laboratory WHERE `U-PRO` > ? AND `U-PRO` < ?", (ua_threshold, u_pro_min, u_pro_max))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of patients with a specific diagnosis, first date year, and sex
@app.get("/v1/thrombosis_prediction/percentage_diagnosis_first_date_sex", operation_id="get_percentage_diagnosis_first_date_sex", summary="Retrieves the percentage of patients with a specific medical condition, first date year, and sex. The calculation is based on the total number of patients who meet the specified criteria.")
async def get_percentage_diagnosis_first_date_sex(diagnosis: str = Query(..., description="Diagnosis"), first_date_year: str = Query(..., description="First date year in 'YYYY' format"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Diagnosis = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(ID) FROM Patient WHERE STRFTIME('%Y', `First Date`) = ? AND SEX = ?", (diagnosis, first_date_year, sex))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get distinct patient IDs based on admission status, T-BIL value, and date pattern
@app.get("/v1/thrombosis_prediction/distinct_patient_ids_admission_tbil_date", operation_id="get_distinct_patient_ids", summary="Retrieves a list of unique patient identifiers who meet the specified admission status, have a T-BIL value below the given threshold, and whose laboratory test date matches the provided pattern. The date pattern should be in 'YYYY-MM-%' format.")
async def get_distinct_patient_ids(admission: str = Query(..., description="Admission status"), tbil_threshold: float = Query(..., description="T-BIL threshold"), date_pattern: str = Query(..., description="Date pattern in 'YYYY-MM-%' format")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.Admission = ? AND T2.`T-BIL` < ? AND T2.Date LIKE ?", (admission, tbil_threshold, date_pattern))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct patient IDs based on ANA pattern, birth year range, and sex
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_ana_birth_sex", operation_id="get_count_distinct_patient_ids", summary="Retrieve the count of unique patients who meet specific criteria: an ANA pattern, a birth year within a specified range, and a particular sex. This operation provides a quantitative overview of patients with these characteristics.")
async def get_count_distinct_patient_ids(ana_pattern: str = Query(..., description="ANA pattern"), birth_year_start: str = Query(..., description="Start of birth year range in 'YYYY' format"), birth_year_end: str = Query(..., description="End of birth year range in 'YYYY' format"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T2.`ANA Pattern` != ? AND STRFTIME('%Y', T1.Birthday) BETWEEN ? AND ? AND T1.SEX = ?", (ana_pattern, birth_year_start, birth_year_end, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the average ALB value based on PLT threshold, diagnosis, and sex
@app.get("/v1/thrombosis_prediction/average_alb_plt_diagnosis_sex", operation_id="get_average_alb", summary="Retrieves the average albumin (ALB) value for patients with a platelet (PLT) count exceeding the specified threshold, a particular diagnosis, and a specific sex.")
async def get_average_alb(plt_threshold: int = Query(..., description="PLT threshold"), diagnosis: str = Query(..., description="Diagnosis"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT AVG(T2.ALB) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.PLT > ? AND T1.Diagnosis = ? AND T1.SEX = ?", (plt_threshold, diagnosis, sex))
    result = cursor.fetchone()
    if not result:
        return {"average_alb": []}
    return {"average_alb": result[0]}

# Endpoint to get the most common symptom for a specific diagnosis
@app.get("/v1/thrombosis_prediction/most_common_symptom_diagnosis", operation_id="get_most_common_symptom", summary="Retrieves the most frequently occurring symptom associated with a given diagnosis. The diagnosis is specified as an input parameter, and the operation returns the symptom that is most commonly reported in conjunction with that diagnosis.")
async def get_most_common_symptom(diagnosis: str = Query(..., description="Diagnosis")):
    cursor.execute("SELECT Symptoms FROM Examination WHERE Diagnosis = ? GROUP BY Symptoms ORDER BY COUNT(Symptoms) DESC LIMIT 1", (diagnosis,))
    result = cursor.fetchone()
    if not result:
        return {"symptom": []}
    return {"symptom": result[0]}

# Endpoint to get the first date and diagnosis of a patient by ID
@app.get("/v1/thrombosis_prediction/patient_first_date_diagnosis", operation_id="get_patient_first_date_diagnosis", summary="Retrieves the earliest date of diagnosis and the corresponding diagnosis for a specific patient, identified by their unique ID.")
async def get_patient_first_date_diagnosis(patient_id: int = Query(..., description="Patient ID")):
    cursor.execute("SELECT `First Date`, Diagnosis FROM Patient WHERE ID = ?", (patient_id,))
    result = cursor.fetchone()
    if not result:
        return {"first_date": [], "diagnosis": []}
    return {"first_date": result[0], "diagnosis": result[1]}

# Endpoint to get the count of patients based on sex and diagnosis
@app.get("/v1/thrombosis_prediction/patient_count_sex_diagnosis", operation_id="get_patient_count", summary="Retrieves the total number of patients categorized by their sex and specific diagnosis. The response is based on the provided sex and diagnosis parameters.")
async def get_patient_count(sex: str = Query(..., description="Sex of the patient"), diagnosis: str = Query(..., description="Diagnosis")):
    cursor.execute("SELECT COUNT(ID) FROM Patient WHERE SEX = ? AND Diagnosis = ?", (sex, diagnosis))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of laboratory records based on ALB levels and year
@app.get("/v1/thrombosis_prediction/laboratory_count_alb_year", operation_id="get_laboratory_count", summary="Retrieves the total number of laboratory records that fall within a specified range of ALB levels and correspond to a given year. The ALB range is defined by a minimum and maximum value, while the year is specified in the 'YYYY' format.")
async def get_laboratory_count(alb_min: float = Query(..., description="Minimum ALB level"), alb_max: float = Query(..., description="Maximum ALB level"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(ID) FROM Laboratory WHERE (ALB <= ? OR ALB >= ?) AND STRFTIME('%Y', Date) = ?", (alb_min, alb_max, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of examinations with a specific diagnosis and symptoms
@app.get("/v1/thrombosis_prediction/examination_percentage_diagnosis_symptoms", operation_id="get_examination_percentage", summary="Retrieves the percentage of examinations that have a specified diagnosis and symptoms. The calculation is based on the total number of examinations with the given symptoms, and the count of examinations with the specified diagnosis among those with the given symptoms.")
async def get_examination_percentage(diagnosis: str = Query(..., description="Diagnosis"), symptoms: str = Query(..., description="Symptoms")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN Diagnosis = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(ID) FROM Examination WHERE Symptoms = ?", (diagnosis, symptoms))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of patients with a specific diagnosis and birth year
@app.get("/v1/thrombosis_prediction/patient_percentage_diagnosis_birthyear", operation_id="get_patient_percentage", summary="Retrieves the percentage of patients with a specific diagnosis and birth year, filtered by sex. The endpoint calculates this percentage by dividing the count of patients with the specified diagnosis, birth year, and sex by the total count of patients with the same diagnosis and birth year, then multiplying the result by 100.")
async def get_patient_percentage(sex: str = Query(..., description="Sex of the patient"), diagnosis: str = Query(..., description="Diagnosis"), birth_year: str = Query(..., description="Birth year in 'YYYY' format")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN SEX = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(ID) FROM Patient WHERE Diagnosis = ? AND STRFTIME('%Y', Birthday) = ?", (sex, diagnosis, birth_year))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the count of patients with a specific WBC level and sex
@app.get("/v1/thrombosis_prediction/patient_count_wbc_sex", operation_id="get_patient_count_wbc_sex", summary="Retrieves the number of patients with a white blood cell (WBC) count below a specified level and of a particular sex. The WBC level and sex are provided as input parameters.")
async def get_patient_count_wbc_sex(wbc_level: float = Query(..., description="WBC level"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.WBC < ? AND T1.SEX = ?", (wbc_level, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the difference in days between examination date and first date for a specific patient
@app.get("/v1/thrombosis_prediction/examination_date_difference", operation_id="get_examination_date_difference", summary="Retrieves the duration in days between the patient's first recorded date and the date of a specific examination. The operation requires the patient's unique identifier to accurately calculate and return the time difference.")
async def get_examination_date_difference(patient_id: int = Query(..., description="Patient ID")):
    cursor.execute("SELECT STRFTIME('%d', T3.`Examination Date`) - STRFTIME('%d', T1.`First Date`) FROM Patient AS T1 INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T1.ID = ?", (patient_id,))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to check if a patient meets specific UA criteria based on sex
@app.get("/v1/thrombosis_prediction/patient_ua_criteria", operation_id="get_patient_ua_criteria", summary="Check if a patient meets specific UA criteria based on sex")
async def get_patient_ua_criteria(sex_f: str = Query(..., description="Sex of the patient (F)"), ua_f: float = Query(..., description="UA level for female"), sex_m: str = Query(..., description="Sex of the patient (M)"), ua_m: float = Query(..., description="UA level for male"), patient_id: int = Query(..., description="Patient ID")):
    cursor.execute("SELECT CASE WHEN (T1.SEX = ? AND T2.UA > ?) OR (T1.SEX = ? AND T2.UA > ?) THEN true ELSE false END FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.ID = ?", (sex_f, ua_f, sex_m, ua_m, patient_id))
    result = cursor.fetchone()
    if not result:
        return {"criteria_met": []}
    return {"criteria_met": result[0]}

# Endpoint to get the date of a laboratory record based on ID and GOT level
@app.get("/v1/thrombosis_prediction/laboratory_date_id_got", operation_id="get_laboratory_date", summary="Get the date of a laboratory record based on ID and GOT level")
async def get_laboratory_date(id: int = Query(..., description="Laboratory ID"), got_level: int = Query(..., description="GOT level")):
    cursor.execute("SELECT Date FROM Laboratory WHERE ID = ? AND GOT >= ?", (id, got_level))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get distinct sex and birthday of patients based on GOT level and year
@app.get("/v1/thrombosis_prediction/patient_sex_birthday_got_year", operation_id="get_patient_sex_birthday", summary="Retrieve unique combinations of patient sex and birthdays associated with GOT levels below a specified threshold and a particular year. This operation provides a concise overview of patient demographics meeting the specified criteria.")
async def get_patient_sex_birthday(got_level: int = Query(..., description="GOT level"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GOT < ? AND STRFTIME('%Y', T2.Date) = ?", (got_level, year))
    result = cursor.fetchall()
    if not result:
        return {"patients": []}
    return {"patients": result}

# Endpoint to get distinct diagnoses for patients with GPT above a certain value
@app.get("/v1/thrombosis_prediction/distinct_diagnoses_gpt", operation_id="get_distinct_diagnoses_gpt", summary="Retrieves a list of unique diagnoses for patients whose GPT value exceeds a specified threshold. The diagnoses are ordered by the patients' birthdays in ascending order.")
async def get_distinct_diagnoses_gpt(gpt_value: int = Query(..., description="GPT value threshold")):
    cursor.execute("SELECT DISTINCT T1.Diagnosis FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GPT > ? ORDER BY T1.Birthday ASC", (gpt_value,))
    result = cursor.fetchall()
    if not result:
        return {"diagnoses": []}
    return {"diagnoses": [row[0] for row in result]}

# Endpoint to get the average LDH value below a certain threshold
@app.get("/v1/thrombosis_prediction/average_ldh", operation_id="get_average_ldh", summary="Retrieves the average value of the LDH (lactate dehydrogenase) enzyme level from laboratory data for patients with LDH levels below a specified threshold.")
async def get_average_ldh(ldh_threshold: int = Query(..., description="LDH threshold value")):
    cursor.execute("SELECT AVG(LDH) FROM Laboratory WHERE LDH < ?", (ldh_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"average_ldh": []}
    return {"average_ldh": result[0]}

# Endpoint to get distinct patient IDs and their ages within a specific LDH range
@app.get("/v1/thrombosis_prediction/patient_ids_ages_ldh_range", operation_id="get_patient_ids_ages_ldh_range", summary="Retrieves unique patient identifiers and their respective ages for patients with LDH values within the specified range. The range is defined by the minimum and maximum LDH values provided as input parameters.")
async def get_patient_ids_ages_ldh_range(min_ldh: int = Query(..., description="Minimum LDH value"), max_ldh: int = Query(..., description="Maximum LDH value")):
    cursor.execute("SELECT DISTINCT T1.ID, STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.Birthday) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.LDH > ? AND T2.LDH < ?", (min_ldh, max_ldh))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"id": row[0], "age": row[1]} for row in result]}

# Endpoint to get admission details for patients with ALP below a certain value
@app.get("/v1/thrombosis_prediction/admission_details_alp", operation_id="get_admission_details_alp", summary="Retrieves the admission details of patients whose ALP (Alkaline Phosphatase) level is below the specified threshold. This operation is useful for identifying patients with lower ALP levels, which can be relevant in various medical contexts, including thrombosis prediction.")
async def get_admission_details_alp(alp_value: int = Query(..., description="ALP value threshold")):
    cursor.execute("SELECT T1.Admission FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.ALP < ?", (alp_value,))
    result = cursor.fetchall()
    if not result:
        return {"admission_details": []}
    return {"admission_details": [row[0] for row in result]}

# Endpoint to get patient IDs and ALP status based on birthday
@app.get("/v1/thrombosis_prediction/patient_alp_status_birthday", operation_id="get_patient_alp_status_birthday", summary="Retrieves a list of patient IDs along with their corresponding ALP status (normal or abnormal) based on the provided birthday. The ALP status is determined by comparing the patient's ALP value to the given threshold.")
async def get_patient_alp_status_birthday(alp_threshold: int = Query(..., description="ALP threshold value"), birthday: str = Query(..., description="Birthday in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.ID, CASE WHEN T2.ALP < ? THEN 'normal' ELSE 'abNormal' END FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.Birthday = ?", (alp_threshold, birthday))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"id": row[0], "status": row[1]} for row in result]}

# Endpoint to get distinct patient details with TP below a certain value
@app.get("/v1/thrombosis_prediction/patient_details_tp", operation_id="get_patient_details_tp", summary="Retrieves unique patient records, including their ID, sex, and birthdate, for patients with a TP value below the specified threshold. This operation filters the patient data based on the provided TP value threshold and returns only the distinct patient records that meet the criteria.")
async def get_patient_details_tp(tp_value: float = Query(..., description="TP value threshold")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.TP < ?", (tp_value,))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"id": row[0], "sex": row[1], "birthday": row[2]} for row in result]}

# Endpoint to get the difference between TP and a threshold for patients of a specific sex
@app.get("/v1/thrombosis_prediction/tp_difference_sex", operation_id="get_tp_difference_sex", summary="Retrieve the difference between a patient's TP value and a specified threshold, filtered by the patient's sex. The operation returns the difference for patients whose TP value exceeds the provided threshold.")
async def get_tp_difference_sex(tp_threshold: float = Query(..., description="TP threshold value"), sex: str = Query(..., description="Sex of the patient"), tp_value: float = Query(..., description="TP value threshold")):
    cursor.execute("SELECT T2.TP - ? FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.SEX = ? AND T2.TP > ?", (tp_threshold, sex, tp_value))
    result = cursor.fetchall()
    if not result:
        return {"tp_differences": []}
    return {"tp_differences": [row[0] for row in result]}

# Endpoint to get distinct patient IDs based on sex and ALB range, ordered by birthday
@app.get("/v1/thrombosis_prediction/patient_ids_sex_alb_range", operation_id="get_patient_ids_sex_alb_range", summary="Retrieves a list of unique patient identifiers, filtered by sex and a specified range of ALB values. The results are ordered by patient's birthday in descending order.")
async def get_patient_ids_sex_alb_range(sex: str = Query(..., description="Sex of the patient"), min_alb: float = Query(..., description="Minimum ALB value"), max_alb: float = Query(..., description="Maximum ALB value")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.SEX = ? AND (T2.ALB <= ? OR T2.ALB >= ?) ORDER BY T1.Birthday DESC", (sex, min_alb, max_alb))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": [row[0] for row in result]}

# Endpoint to get ALB status for patients born in a specific year
@app.get("/v1/thrombosis_prediction/alb_status_birth_year", operation_id="get_alb_status_birth_year", summary="Retrieves the albumin (ALB) status of patients born in a specific year, based on their ALB values. The status is determined by comparing the ALB value to a provided range. If the ALB value falls within the range, the status is 'normal'; otherwise, it is 'abnormal'.")
async def get_alb_status_birth_year(min_alb: float = Query(..., description="Minimum ALB value"), max_alb: float = Query(..., description="Maximum ALB value"), birth_year: str = Query(..., description="Birth year in 'YYYY' format")):
    cursor.execute("SELECT CASE WHEN T2.ALB >= ? AND T2.ALB <= ? THEN 'normal' ELSE 'abnormal' END FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', T1.Birthday) = ?", (min_alb, max_alb, birth_year))
    result = cursor.fetchall()
    if not result:
        return {"alb_status": []}
    return {"alb_status": [row[0] for row in result]}

# Endpoint to get the percentage of patients with UA above a certain value based on sex
@app.get("/v1/thrombosis_prediction/percentage_ua_sex", operation_id="get_percentage_ua_sex", summary="Retrieves the percentage of patients with uric acid (UA) levels exceeding a specified threshold, categorized by sex. This operation calculates the proportion of patients with UA levels above the given threshold, relative to the total number of patients of the specified sex.")
async def get_percentage_ua_sex(ua_threshold: float = Query(..., description="UA threshold value"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.UA > ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.SEX = ?", (ua_threshold, sex))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average UA value based on gender and UA thresholds
@app.get("/v1/thrombosis_prediction/average_ua_by_gender_and_ua_thresholds", operation_id="get_average_ua", summary="Retrieves the average UA value for male and female patients who have UA levels below the specified thresholds. The UA value is obtained from the most recent laboratory data.")
async def get_average_ua(ua_threshold_female: float = Query(..., description="UA threshold for female patients"), sex_female: str = Query(..., description="Sex of the patient (F)"), ua_threshold_male: float = Query(..., description="UA threshold for male patients"), sex_male: str = Query(..., description="Sex of the patient (M)")):
    cursor.execute("SELECT AVG(T2.UA) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE (T2.UA < ? AND T1.SEX = ?) OR (T2.UA < ? AND T1.SEX = ?) AND T2.Date = ( SELECT MAX(Date) FROM Laboratory )", (ua_threshold_female, sex_female, ua_threshold_male, sex_male))
    result = cursor.fetchone()
    if not result:
        return {"average_ua": []}
    return {"average_ua": result[0]}

# Endpoint to get distinct patient details based on UN value
@app.get("/v1/thrombosis_prediction/distinct_patient_details_by_un", operation_id="get_distinct_patient_details", summary="Retrieves unique patient details, including ID, sex, and birthday, for patients who have a specific UN value in their laboratory records. This operation is useful for obtaining distinct patient information based on a given UN value.")
async def get_distinct_patient_details(un_value: int = Query(..., description="UN value")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.UN = ?", (un_value,))
    result = cursor.fetchall()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get distinct patient details based on UN threshold and diagnosis
@app.get("/v1/thrombosis_prediction/distinct_patient_details_by_un_and_diagnosis", operation_id="get_distinct_patient_details_by_un_and_diagnosis", summary="Retrieves unique patient records, including their ID, sex, and birthday, based on a specified UN threshold and diagnosis. This operation filters patients with UN values below the provided threshold and matches the given diagnosis, ensuring only distinct patient details are returned.")
async def get_distinct_patient_details_by_un_and_diagnosis(un_threshold: int = Query(..., description="UN threshold"), diagnosis: str = Query(..., description="Diagnosis of the patient")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.UN < ? AND T1.Diagnosis = ?", (un_threshold, diagnosis))
    result = cursor.fetchall()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get the count of patients based on CRE threshold and sex
@app.get("/v1/thrombosis_prediction/count_patients_by_cre_and_sex", operation_id="get_count_patients_by_cre_and_sex", summary="Retrieves the total number of patients who have a CRE level equal to or above the specified threshold and belong to the given sex category. This operation is useful for understanding the distribution of patients based on their CRE levels and sex.")
async def get_count_patients_by_cre_and_sex(cre_threshold: float = Query(..., description="CRE threshold"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.CRE >= ? AND T1.SEX = ?", (cre_threshold, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to compare the number of male and female patients based on CRE threshold
@app.get("/v1/thrombosis_prediction/compare_male_female_count_by_cre", operation_id="compare_male_female_count_by_cre", summary="This endpoint compares the number of male and female patients who have a CRE value equal to or above a specified threshold. It returns a boolean value indicating whether the count of male patients surpasses the count of female patients meeting the CRE criterion.")
async def compare_male_female_count_by_cre(sex_male: str = Query(..., description="Sex of the patient (M)"), sex_female: str = Query(..., description="Sex of the patient (F)"), cre_threshold: float = Query(..., description="CRE threshold")):
    cursor.execute("SELECT CASE WHEN SUM(CASE WHEN T1.SEX = ? THEN 1 ELSE 0 END) > SUM(CASE WHEN T1.SEX = ? THEN 1 ELSE 0 END) THEN 'True' ELSE 'False' END FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.CRE >= ?", (sex_male, sex_female, cre_threshold))
    result = cursor.fetchone()
    if not result:
        return {"comparison": []}
    return {"comparison": result[0]}

# Endpoint to get the patient with the highest T-BIL value
@app.get("/v1/thrombosis_prediction/patient_with_highest_t_bil", operation_id="get_patient_with_highest_t_bil", summary="Retrieves the patient with the highest T-BIL (total bilirubin) value, along with their ID, sex, and birthdate. This operation returns the patient's demographic information and the corresponding T-BIL value from the laboratory data.")
async def get_patient_with_highest_t_bil():
    cursor.execute("SELECT T2.`T-BIL`, T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID ORDER BY T2.`T-BIL` DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get patient IDs and sex based on T-BIL threshold
@app.get("/v1/thrombosis_prediction/patient_ids_sex_by_t_bil", operation_id="get_patient_ids_sex_by_t_bil", summary="Retrieves the IDs and sex of patients who have a T-BIL value equal to or greater than the specified threshold. The data is grouped by sex and patient ID.")
async def get_patient_ids_sex_by_t_bil(t_bil_threshold: float = Query(..., description="T-BIL threshold")):
    cursor.execute("SELECT T1.ID, T1.SEX FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`T-BIL` >= ? GROUP BY T1.SEX, T1.ID", (t_bil_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get the patient with the highest T-CHO value and earliest birthday
@app.get("/v1/thrombosis_prediction/patient_with_highest_t_cho_earliest_birthday", operation_id="get_patient_with_highest_t_cho_earliest_birthday", summary="Retrieves the patient with the highest T-CHO (total cholesterol) level and the earliest birthday from the database. The operation considers both the T-CHO value and the patient's birthday to determine the result.")
async def get_patient_with_highest_t_cho_earliest_birthday():
    cursor.execute("SELECT T1.ID, T2.`T-CHO` FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID ORDER BY T2.`T-CHO` DESC, T1.Birthday ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get the average age of patients based on T-CHO threshold and sex
@app.get("/v1/thrombosis_prediction/average_age_by_t_cho_and_sex", operation_id="get_average_age_by_t_cho_and_sex", summary="Retrieves the average age of patients who meet the specified T-CHO threshold and sex criteria. The calculation is based on the current year minus the patient's birth year.")
async def get_average_age_by_t_cho_and_sex(t_cho_threshold: int = Query(..., description="T-CHO threshold"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT AVG(STRFTIME('%Y', date('NOW')) - STRFTIME('%Y', T1.Birthday)) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`T-CHO` >= ? AND T1.SEX = ?", (t_cho_threshold, sex))
    result = cursor.fetchone()
    if not result:
        return {"average_age": []}
    return {"average_age": result[0]}

# Endpoint to get patient IDs and diagnoses based on TG threshold
@app.get("/v1/thrombosis_prediction/patient_ids_diagnoses_by_tg", operation_id="get_patient_ids_diagnoses_by_tg", summary="Retrieves patient IDs and their corresponding diagnoses for patients with TG levels above the specified threshold. The TG threshold is a critical parameter that determines the patients included in the result set.")
async def get_patient_ids_diagnoses_by_tg(tg_threshold: int = Query(..., description="TG threshold")):
    cursor.execute("SELECT T1.ID, T1.Diagnosis FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.TG > ?", (tg_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"patient_details": []}
    return {"patient_details": result}

# Endpoint to get the count of distinct patients with TG level above a certain threshold and age above a certain number of years
@app.get("/v1/thrombosis_prediction/count_patients_tg_age", operation_id="get_count_patients_tg_age", summary="Retrieves the number of unique patients who have a TG level equal to or above the specified threshold and are older than the given age. The count is determined by comparing the TG level and age of each patient to the provided thresholds.")
async def get_count_patients_tg_age(tg_threshold: int = Query(..., description="TG level threshold"), age_threshold: int = Query(..., description="Age threshold in years")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.TG >= ? AND STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.Birthday) > ?", (tg_threshold, age_threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct patient IDs with CPK level below a certain threshold and specific admission status
@app.get("/v1/thrombosis_prediction/patient_ids_cpk_admission", operation_id="get_patient_ids_cpk_admission", summary="Retrieves unique patient identifiers who have a CPK level below a specified threshold and a particular admission status. This operation is useful for identifying patients who meet specific criteria, enabling targeted analysis or intervention.")
async def get_patient_ids_cpk_admission(cpk_threshold: int = Query(..., description="CPK level threshold"), admission_status: str = Query(..., description="Admission status")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.CPK < ? AND T1.Admission = ?", (cpk_threshold, admission_status))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct patients born between specific years, of a specific gender, and with CPK level above a certain threshold
@app.get("/v1/thrombosis_prediction/count_patients_birth_year_sex_cpk", operation_id="get_count_patients_birth_year_sex_cpk", summary="Retrieves the number of unique patients born between the specified start and end years, of the given gender, and with a CPK level at or above the provided threshold.")
async def get_count_patients_birth_year_sex_cpk(start_year: str = Query(..., description="Start year in 'YYYY' format"), end_year: str = Query(..., description="End year in 'YYYY' format"), sex: str = Query(..., description="Gender of the patient"), cpk_threshold: int = Query(..., description="CPK level threshold")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', T1.Birthday) BETWEEN ? AND ? AND T1.SEX = ? AND T2.CPK >= ?", (start_year, end_year, sex, cpk_threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct patient IDs, sex, and age with GLU level above a certain threshold and T-CHO level below a certain threshold
@app.get("/v1/thrombosis_prediction/patient_ids_sex_age_glu_tcho", operation_id="get_patient_ids_sex_age_glu_tcho", summary="Retrieve unique patient records, including their ID, sex, and age, that meet specific criteria for GLU and T-CHO levels. The GLU level must be equal to or above a provided threshold, while the T-CHO level must be below another given threshold.")
async def get_patient_ids_sex_age_glu_tcho(glu_threshold: int = Query(..., description="GLU level threshold"), tcho_threshold: int = Query(..., description="T-CHO level threshold")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX , STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.Birthday) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GLU >= ? AND T2.`T-CHO` < ?", (glu_threshold, tcho_threshold))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"id": row[0], "sex": row[1], "age": row[2]} for row in result]}

# Endpoint to get distinct patient IDs and GLU levels with a specific first date year and GLU level below a certain threshold
@app.get("/v1/thrombosis_prediction/patient_ids_glu_first_date", operation_id="get_patient_ids_glu_first_date", summary="Retrieves unique patient identifiers and their corresponding glucose (GLU) levels that meet the specified criteria. The criteria include a specific year for the patient's first date and a GLU level below a certain threshold.")
async def get_patient_ids_glu_first_date(first_date_year: str = Query(..., description="First date year in 'YYYY' format"), glu_threshold: int = Query(..., description="GLU level threshold")):
    cursor.execute("SELECT DISTINCT T1.ID, T2.GLU FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', T1.`First Date`) = ? AND T2.GLU < ?", (first_date_year, glu_threshold))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"id": row[0], "glu": row[1]} for row in result]}

# Endpoint to get distinct patient IDs, sex, and birthday with WBC levels within a certain range, grouped by sex and ID, ordered by birthday
@app.get("/v1/thrombosis_prediction/patient_ids_sex_birthday_wbc", operation_id="get_patient_ids_sex_birthday_wbc", summary="Retrieves unique patient records, including their ID, sex, and birthday, with WBC levels outside a specified range. The results are grouped by sex and ID, and sorted by birthday in ascending order.")
async def get_patient_ids_sex_birthday_wbc(wbc_lower_threshold: float = Query(..., description="Lower threshold for WBC level"), wbc_upper_threshold: float = Query(..., description="Upper threshold for WBC level")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX, T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.WBC <= ? OR T2.WBC >= ? GROUP BY T1.SEX,T1.ID ORDER BY T1.Birthday ASC", (wbc_lower_threshold, wbc_upper_threshold))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"id": row[0], "sex": row[1], "birthday": row[2]} for row in result]}

# Endpoint to get distinct patient diagnoses, IDs, and age with RBC level below a certain threshold
@app.get("/v1/thrombosis_prediction/patient_diagnoses_ids_age_rbc", operation_id="get_patient_diagnoses_ids_age_rbc", summary="Retrieve unique patient diagnoses, IDs, and ages for individuals with RBC levels below a specified threshold. This operation filters patients based on their RBC levels and returns distinct records, providing a concise overview of relevant patient information.")
async def get_patient_diagnoses_ids_age_rbc(rbc_threshold: float = Query(..., description="RBC level threshold")):
    cursor.execute("SELECT DISTINCT T1.Diagnosis, T1.ID , STRFTIME('%Y', CURRENT_TIMESTAMP) -STRFTIME('%Y', T1.Birthday) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.RBC < ?", (rbc_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"diagnosis": row[0], "id": row[1], "age": row[2]} for row in result]}

# Endpoint to get distinct patient IDs and admission status with specific sex, RBC level within a certain range, and age above a certain threshold
@app.get("/v1/thrombosis_prediction/patient_ids_admission_sex_rbc_age", operation_id="get_patient_ids_admission_sex_rbc_age", summary="Retrieve unique patient identifiers and their admission statuses based on specific gender, red blood cell (RBC) count within a defined range, and age surpassing a certain threshold. This operation filters patients by their sex, RBC levels, and age, providing a concise list of qualifying patients and their admission statuses.")
async def get_patient_ids_admission_sex_rbc_age(sex: str = Query(..., description="Gender of the patient"), rbc_lower_threshold: float = Query(..., description="Lower threshold for RBC level"), rbc_upper_threshold: float = Query(..., description="Upper threshold for RBC level"), age_threshold: int = Query(..., description="Age threshold in years")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.Admission FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.SEX = ? AND (T2.RBC <= ? OR T2.RBC >= ?) AND STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.Birthday) >= ?", (sex, rbc_lower_threshold, rbc_upper_threshold, age_threshold))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"id": row[0], "admission": row[1]} for row in result]}

# Endpoint to get distinct patient IDs and sex with HGB level below a certain threshold and specific admission status
@app.get("/v1/thrombosis_prediction/patient_ids_sex_hgb_admission", operation_id="get_patient_ids_sex_hgb_admission", summary="Retrieves unique patient identifiers and their sex, filtered by hemoglobin (HGB) levels below a specified threshold and a particular admission status.")
async def get_patient_ids_sex_hgb_admission(hgb_threshold: float = Query(..., description="HGB level threshold"), admission_status: str = Query(..., description="Admission status")):
    cursor.execute("SELECT DISTINCT T1.ID, T1.SEX FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.HGB < ? AND T1.Admission = ?", (hgb_threshold, admission_status))
    result = cursor.fetchall()
    if not result:
        return {"patient_data": []}
    return {"patient_data": [{"id": row[0], "sex": row[1]} for row in result]}

# Endpoint to get patient ID and sex with a specific diagnosis and HGB level within a certain range, ordered by birthday, limited to one result
@app.get("/v1/thrombosis_prediction/patient_id_sex_diagnosis_hgb", operation_id="get_patient_id_sex_diagnosis_hgb", summary="Retrieves the ID and sex of a patient with a specific diagnosis and hemoglobin (HGB) level within a given range. The result is ordered by the patient's birthday and limited to one record.")
async def get_patient_id_sex_diagnosis_hgb(diagnosis: str = Query(..., description="Diagnosis of the patient"), hgb_lower_threshold: float = Query(..., description="Lower threshold for HGB level"), hgb_upper_threshold: float = Query(..., description="Upper threshold for HGB level")):
    cursor.execute("SELECT T1.ID, T1.SEX FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.Diagnosis = ? AND T2.HGB > ? AND T2.HGB < ? ORDER BY T1.Birthday ASC LIMIT 1", (diagnosis, hgb_lower_threshold, hgb_upper_threshold))
    result = cursor.fetchone()
    if not result:
        return {"patient_data": []}
    return {"patient_data": {"id": result[0], "sex": result[1]}}

# Endpoint to get distinct patient IDs and their ages based on HCT and count criteria
@app.get("/v1/thrombosis_prediction/patient_ids_ages_hct_count", operation_id="get_patient_ids_ages_hct_count", summary="Retrieve unique patient identifiers and their corresponding ages, filtered by a specified HCT threshold and a minimum count of records meeting that threshold. The age is calculated as the difference between the current year and the patient's birth year.")
async def get_patient_ids_ages_hct_count(hct_threshold: int = Query(..., description="HCT threshold value"), count_threshold: int = Query(..., description="Count threshold value")):
    cursor.execute("SELECT DISTINCT T1.ID, STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.Birthday) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.ID IN ( SELECT ID FROM Laboratory WHERE HCT >= ? GROUP BY ID HAVING COUNT(ID) >= ? )", (hct_threshold, count_threshold))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the average HCT value for patients based on HCT threshold and year
@app.get("/v1/thrombosis_prediction/avg_hct_threshold_year", operation_id="get_avg_hct_threshold_year", summary="Retrieves the average Hematocrit (HCT) value for patients whose HCT levels are below a specified threshold in a given year. This operation is useful for analyzing trends in HCT levels over time and identifying potential thresholds for thrombosis prediction.")
async def get_avg_hct_threshold_year(hct_threshold: int = Query(..., description="HCT threshold value"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT AVG(T2.HCT) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.HCT < ? AND STRFTIME('%Y', T2.Date) = ?", (hct_threshold, year))
    result = cursor.fetchone()
    if not result:
        return {"average_hct": []}
    return {"average_hct": result[0]}

# Endpoint to get the difference in counts of PLT values below and above specified thresholds
@app.get("/v1/thrombosis_prediction/plt_count_difference", operation_id="get_plt_count_difference", summary="Retrieves the difference in the number of PLT values that fall below the specified lower threshold and above the specified upper threshold. This operation calculates the sum of instances where PLT values are less than or equal to the lower threshold and subtracts the sum of instances where PLT values are greater than or equal to the upper threshold. The calculation is performed on PLT values associated with patients in the database.")
async def get_plt_count_difference(plt_lower_threshold: int = Query(..., description="Lower PLT threshold value"), plt_upper_threshold: int = Query(..., description="Upper PLT threshold value")):
    cursor.execute("SELECT SUM(CASE WHEN T2.PLT <= ? THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.PLT >= ? THEN 1 ELSE 0 END) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID", (plt_lower_threshold, plt_upper_threshold))
    result = cursor.fetchone()
    if not result:
        return {"difference": []}
    return {"difference": result[0]}

# Endpoint to get distinct patient IDs based on PLT range, age difference, and year
@app.get("/v1/thrombosis_prediction/patient_ids_plt_range_age_year", operation_id="get_patient_ids_plt_range_age_year", summary="Retrieve unique patient identifiers that meet specific criteria: platelet count within a defined range, age difference below a certain value, and the year of the laboratory test matches the provided year.")
async def get_patient_ids_plt_range_age_year(plt_lower_bound: int = Query(..., description="Lower bound of PLT range"), plt_upper_bound: int = Query(..., description="Upper bound of PLT range"), age_difference: int = Query(..., description="Age difference threshold"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.PLT BETWEEN ? AND ? AND STRFTIME('%Y', T2.Date) - STRFTIME('%Y', T1.Birthday) < ? AND STRFTIME('%Y', T2.Date) = ?", (plt_lower_bound, plt_upper_bound, age_difference, year))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the percentage of patients with PT above a threshold and specific sex, based on age difference
@app.get("/v1/thrombosis_prediction/pt_percentage_sex_age", operation_id="get_pt_percentage_sex_age", summary="Retrieves the percentage of patients with prothrombin time (PT) above a specified threshold and of a certain sex, where the age difference between the current year and the patient's birth year exceeds a given value.")
async def get_pt_percentage_sex_age(pt_threshold: int = Query(..., description="PT threshold value"), sex: str = Query(..., description="Sex of the patient"), age_difference: int = Query(..., description="Age difference threshold")):
    cursor.execute("SELECT CAST(SUM(CASE WHEN T2.PT >= ? AND T1.SEX = ? THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', CURRENT_TIMESTAMP) - STRFTIME('%Y', T1.Birthday) > ?", (pt_threshold, sex, age_difference))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get patient IDs based on first date year and PT threshold
@app.get("/v1/thrombosis_prediction/patient_ids_first_date_pt", operation_id="get_patient_ids_first_date_pt", summary="Retrieves patient identifiers for individuals whose first recorded date is later than the specified year and whose PT (Prothrombin Time) value is less than the provided threshold.")
async def get_patient_ids_first_date_pt(first_date_year: str = Query(..., description="First date year in 'YYYY' format"), pt_threshold: int = Query(..., description="PT threshold value")):
    cursor.execute("SELECT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE STRFTIME('%Y', T1.`First Date`) > ? AND T2.PT < ?", (first_date_year, pt_threshold))
    result = cursor.fetchall()
    if not result:
        return {"data": []}
    return {"data": result}

# Endpoint to get the count of patient IDs based on date and APTT threshold
@app.get("/v1/thrombosis_prediction/count_patient_ids_date_aptt", operation_id="get_count_patient_ids_date_aptt", summary="Retrieves the number of patients who have had a laboratory test after a specified date and with an APTT value equal to or above a given threshold.")
async def get_count_patient_ids_date_aptt(date: str = Query(..., description="Date in 'YYYY-MM-DD' format"), aptt_threshold: int = Query(..., description="APTT threshold value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.Date > ? AND T2.APTT >= ?", (date, aptt_threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs based on thrombosis status and APTT threshold
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_thrombosis_aptt", operation_id="get_count_distinct_patient_ids_thrombosis_aptt", summary="Retrieve the number of unique patients who have a specified thrombosis status and an APTT value exceeding a given threshold.")
async def get_count_distinct_patient_ids_thrombosis_aptt(thrombosis_status: int = Query(..., description="Thrombosis status (0 or 1)"), aptt_threshold: int = Query(..., description="APTT threshold value")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T3.ID = T2.ID WHERE T3.Thrombosis = ? AND T2.APTT > ?", (thrombosis_status, aptt_threshold))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs based on FG range, WBC range, and sex
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_fg_wbc_sex", operation_id="get_count_distinct_patient_ids_fg_wbc_sex", summary="Get the count of distinct patient IDs where FG is within a specified range, WBC is within a specified range, and sex is specified.")
async def get_count_distinct_patient_ids_fg_wbc_sex(fg_lower_threshold: int = Query(..., description="Lower FG threshold value"), fg_upper_threshold: int = Query(..., description="Upper FG threshold value"), wbc_lower_threshold: float = Query(..., description="Lower WBC threshold value"), wbc_upper_threshold: float = Query(..., description="Upper WBC threshold value"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE (T2.FG <= ? OR T2.FG >= ?) AND T2.WBC > ? AND T2.WBC < ? AND T1.SEX = ?", (fg_lower_threshold, fg_upper_threshold, wbc_lower_threshold, wbc_upper_threshold, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs based on FG range and birthday
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_fg_birthday", operation_id="get_count_distinct_patient_ids_fg_birthday", summary="Get the count of distinct patient IDs where FG is within a specified range and birthday is after a specified date.")
async def get_count_distinct_patient_ids_fg_birthday(fg_lower_threshold: int = Query(..., description="Lower FG threshold value"), fg_upper_threshold: int = Query(..., description="Upper FG threshold value"), birthday: str = Query(..., description="Birthday in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE (T2.FG <= ? OR T2.FG >= ?) AND T1.Birthday > ?", (fg_lower_threshold, fg_upper_threshold, birthday))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get diagnoses of patients with U-PRO levels above a certain threshold
@app.get("/v1/thrombosis_prediction/diagnoses_by_u_pro_threshold", operation_id="get_diagnoses_by_u_pro_threshold", summary="Retrieves diagnoses of patients who have U-PRO levels equal to or above a specified threshold. The threshold is provided as an input parameter.")
async def get_diagnoses_by_u_pro_threshold(u_pro_threshold: int = Query(..., description="U-PRO threshold value")):
    cursor.execute("SELECT T1.Diagnosis FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`U-PRO` >= ?", (u_pro_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"diagnoses": []}
    return {"diagnoses": [row[0] for row in result]}

# Endpoint to get distinct patient IDs with U-PRO levels within a certain range and a specific diagnosis
@app.get("/v1/thrombosis_prediction/patient_ids_by_u_pro_range_and_diagnosis", operation_id="get_patient_ids_by_u_pro_range_and_diagnosis", summary="Retrieve a unique list of patient identifiers who have U-PRO levels between the provided minimum and maximum values and have been diagnosed with the specified condition.")
async def get_patient_ids_by_u_pro_range_and_diagnosis(u_pro_min: int = Query(..., description="Minimum U-PRO value"), u_pro_max: int = Query(..., description="Maximum U-PRO value"), diagnosis: str = Query(..., description="Diagnosis of the patient")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`U-PRO` > ? AND T2.`U-PRO` < ? AND T1.Diagnosis = ?", (u_pro_min, u_pro_max, diagnosis))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct patient IDs with IGG levels above a certain threshold
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_by_igg_threshold", operation_id="get_count_distinct_patient_ids_by_igg_threshold", summary="Retrieve the number of unique patients who have IGG levels equal to or above a specified threshold. This operation considers laboratory and examination data to determine the count.")
async def get_count_distinct_patient_ids_by_igg_threshold(igg_threshold: int = Query(..., description="IGG threshold value")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T3.ID = T2.ID WHERE T2.IGG >= ?", (igg_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patient IDs with IGG levels within a certain range and non-null symptoms
@app.get("/v1/thrombosis_prediction/count_patient_ids_by_igg_range_and_symptoms", operation_id="get_count_patient_ids_by_igg_range_and_symptoms", summary="Retrieves the total number of patients with IGG levels between the provided minimum and maximum values, who also have non-null symptoms. This operation considers data from the Patient, Laboratory, and Examination tables.")
async def get_count_patient_ids_by_igg_range_and_symptoms(igg_min: int = Query(..., description="Minimum IGG value"), igg_max: int = Query(..., description="Maximum IGG value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T3.ID = T2.ID WHERE T2.IGG BETWEEN ? AND ? AND T3.Symptoms IS NOT NULL", (igg_min, igg_max))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the diagnosis of the patient with the highest IGA level within a certain range
@app.get("/v1/thrombosis_prediction/diagnosis_by_highest_iga_in_range", operation_id="get_diagnosis_by_highest_iga_in_range", summary="Retrieves the diagnosis of the patient with the highest IGA level within the specified IGA range. The range is defined by the minimum and maximum IGA values provided as input parameters. The operation returns the diagnosis of the patient with the highest IGA level that falls within this range.")
async def get_diagnosis_by_highest_iga_in_range(iga_min: int = Query(..., description="Minimum IGA value"), iga_max: int = Query(..., description="Maximum IGA value")):
    cursor.execute("SELECT patientData.Diagnosis FROM Patient AS patientData INNER JOIN Laboratory AS labData ON patientData.ID = labData.ID WHERE labData.IGA BETWEEN ? AND ? ORDER BY labData.IGA DESC LIMIT 1", (iga_min, iga_max))
    result = cursor.fetchone()
    if not result:
        return {"diagnosis": []}
    return {"diagnosis": result[0]}

# Endpoint to get the count of patient IDs with IGA levels within a certain range and first date after a specified year
@app.get("/v1/thrombosis_prediction/count_patient_ids_by_iga_range_and_first_date", operation_id="get_count_patient_ids_by_iga_range_and_first_date", summary="Retrieve the number of unique patient IDs that have IGA levels within the provided range and a first date occurring after the specified year. This operation is useful for understanding the distribution of patients based on their IGA levels and first date of record.")
async def get_count_patient_ids_by_iga_range_and_first_date(iga_min: int = Query(..., description="Minimum IGA value"), iga_max: int = Query(..., description="Maximum IGA value"), first_date_year: str = Query(..., description="Year of the first date in 'YYYY' format")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.IGA BETWEEN ? AND ? AND strftime('%Y', T1.`First Date`) > ?", (iga_min, iga_max, first_date_year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the most common diagnosis for patients with IGM levels outside a certain range
@app.get("/v1/thrombosis_prediction/most_common_diagnosis_by_igm_range", operation_id="get_most_common_diagnosis_by_igm_range", summary="Retrieves the most frequently occurring diagnosis among patients whose IGM levels fall outside the specified range. The range is defined by the minimum and maximum IGM values provided as input parameters.")
async def get_most_common_diagnosis_by_igm_range(igm_min: int = Query(..., description="Minimum IGM value"), igm_max: int = Query(..., description="Maximum IGM value")):
    cursor.execute("SELECT T1.Diagnosis FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.IGM NOT BETWEEN ? AND ? GROUP BY T1.Diagnosis ORDER BY COUNT(T1.Diagnosis) DESC LIMIT 1", (igm_min, igm_max))
    result = cursor.fetchone()
    if not result:
        return {"diagnosis": []}
    return {"diagnosis": result[0]}

# Endpoint to get the count of patient IDs with a specific CRP value and null description
@app.get("/v1/thrombosis_prediction/count_patient_ids_by_crp_and_null_description", operation_id="get_count_patient_ids_by_crp_and_null_description", summary="Get the count of patient IDs with a specific CRP value and null description")
async def get_count_patient_ids_by_crp_and_null_description(crp_value: str = Query(..., description="CRP value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.CRP = ? AND T1.Description IS NULL", (crp_value,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs with CRE levels above a certain threshold and age below a certain value
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_by_cre_and_age", operation_id="get_count_distinct_patient_ids_by_cre_and_age", summary="Retrieve the number of unique patients who have CRE levels at or above a specified threshold and are younger than a given age. This operation provides a count of distinct patient IDs that meet these criteria, offering insights into the prevalence of certain conditions or characteristics within a specific age range and CRE level.")
async def get_count_distinct_patient_ids_by_cre_and_age(cre_threshold: float = Query(..., description="CRE threshold value"), max_age: int = Query(..., description="Maximum age of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.CRE >= ? AND STRFTIME('%Y', Date('now')) - STRFTIME('%Y', T1.Birthday) < ?", (cre_threshold, max_age))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs with specific RA values and KCT value
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_by_ra_and_kct", operation_id="get_count_distinct_patient_ids_by_ra_and_kct", summary="Get the count of distinct patient IDs with specific RA values and KCT value")
async def get_count_distinct_patient_ids_by_ra_and_kct(ra_value_1: str = Query(..., description="First RA value"), ra_value_2: str = Query(..., description="Second RA value"), kct_value: str = Query(..., description="KCT value")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T3.ID = T2.ID WHERE (T2.RA = ? OR T2.RA = ?) AND T3.KCT = ?", (ra_value_1, ra_value_2, kct_value))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get diagnoses of patients with specific RA values and birthdays after a certain date
@app.get("/v1/thrombosis_prediction/diagnoses_by_ra_and_birthday", operation_id="get_diagnoses_by_ra_and_birthday", summary="Retrieves diagnoses of patients who have either of the specified RA values and were born after the provided date. The RA values can include a specific value or a range, and the birthday is provided in 'YYYY-MM-DD' format.")
async def get_diagnoses_by_ra_and_birthday(ra_value1: str = Query(..., description="First RA value"), ra_value2: str = Query(..., description="Second RA value"), birthday: str = Query(..., description="Birthday in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.Diagnosis FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE (T2.RA = ? OR T2.RA = ?) AND T1.Birthday > ?", (ra_value1, ra_value2, birthday))
    result = cursor.fetchall()
    if not result:
        return {"diagnoses": []}
    return {"diagnoses": [row[0] for row in result]}

# Endpoint to get patient IDs with RF values below a threshold and age above a certain number of years
@app.get("/v1/thrombosis_prediction/patient_ids_by_rf_and_age", operation_id="get_patient_ids_by_rf_and_age", summary="Retrieves the IDs of patients who have RF values below a specified threshold and are older than a certain age. The operation filters patients based on their RF values and age, returning a list of IDs that meet the criteria.")
async def get_patient_ids_by_rf_and_age(rf_threshold: int = Query(..., description="RF threshold value"), age_threshold: int = Query(..., description="Age threshold in years")):
    cursor.execute("SELECT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.RF < ? AND STRFTIME('%Y', DATE('now')) - STRFTIME('%Y', T1.Birthday) > ?", (rf_threshold, age_threshold))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct patient IDs with RF values below a threshold and no thrombosis
@app.get("/v1/thrombosis_prediction/count_distinct_ids_by_rf_and_thrombosis", operation_id="get_count_distinct_ids_by_rf_and_thrombosis", summary="Retrieves the count of unique patients who have RF values less than a specified threshold and do not have thrombosis. The RF threshold and thrombosis status are provided as input parameters.")
async def get_count_distinct_ids_by_rf_and_thrombosis(rf_threshold: int = Query(..., description="RF threshold value"), thrombosis: int = Query(..., description="Thrombosis value (0 for no thrombosis)")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Examination AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.RF < ? AND T1.Thrombosis = ?", (rf_threshold, thrombosis))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs with C3 values above a threshold and a specific ANA pattern
@app.get("/v1/thrombosis_prediction/count_distinct_ids_by_c3_and_ana_pattern", operation_id="get_count_distinct_ids_by_c3_and_ana_pattern", summary="Retrieves the count of unique patients who have C3 values exceeding a specified threshold and a particular ANA pattern. The C3 threshold and ANA pattern are provided as input parameters.")
async def get_count_distinct_ids_by_c3_and_ana_pattern(c3_threshold: int = Query(..., description="C3 threshold value"), ana_pattern: str = Query(..., description="ANA pattern value")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Examination AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.C3 > ? AND T1.`ANA Pattern` = ?", (c3_threshold, ana_pattern))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct patient IDs with HCT values within a range, ordered by aCL IgA values
@app.get("/v1/thrombosis_prediction/distinct_ids_by_hct_range_ordered_by_acl_iga", operation_id="get_distinct_ids_by_hct_range_ordered_by_acl_iga", summary="Retrieve a distinct set of patient IDs with HCT values within a specified range, sorted in descending order by their corresponding aCL IgA values. The number of results returned can be limited by providing a specific value.")
async def get_distinct_ids_by_hct_range_ordered_by_acl_iga(hct_min: int = Query(..., description="Minimum HCT value"), hct_max: int = Query(..., description="Maximum HCT value"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID INNER JOIN Laboratory AS T3 on T1.ID = T3.ID WHERE (T3.HCT >= ? OR T3.HCT <= ?) ORDER BY T2.`aCL IgA` DESC LIMIT ?", (hct_min, hct_max, limit))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct patient IDs with C4 values above a threshold and a specific diagnosis
@app.get("/v1/thrombosis_prediction/count_distinct_ids_by_c4_and_diagnosis", operation_id="get_count_distinct_ids_by_c4_and_diagnosis", summary="Retrieve the number of unique patients who have a C4 value exceeding a specified threshold and a particular diagnosis. The C4 threshold and diagnosis are provided as input parameters.")
async def get_count_distinct_ids_by_c4_and_diagnosis(c4_threshold: int = Query(..., description="C4 threshold value"), diagnosis: str = Query(..., description="Diagnosis value")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.C4 > ? AND T1.Diagnosis = ?", (c4_threshold, diagnosis))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs with specific RNP values and admission status
@app.get("/v1/thrombosis_prediction/count_distinct_ids_by_rnp_and_admission", operation_id="get_count_distinct_ids_by_rnp_and_admission", summary="Get the count of distinct patient IDs with specific RNP values and admission status")
async def get_count_distinct_ids_by_rnp_and_admission(rnp_value1: str = Query(..., description="First RNP value"), rnp_value2: str = Query(..., description="Second RNP value"), admission: str = Query(..., description="Admission status")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE (T2.RNP = ? OR T2.RNP = ?) AND T1.Admission = ?", (rnp_value1, rnp_value2, admission))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the birthdays of patients with RNP values not equal to a specific value, ordered by birthday
@app.get("/v1/thrombosis_prediction/birthdays_by_rnp_not_equal_ordered_by_birthday", operation_id="get_birthdays_by_rnp_not_equal_ordered_by_birthday", summary="Retrieve the birthdays of patients who have RNP values different from the provided value, sorted in descending order by birthday. The number of results can be limited by specifying the maximum number of records to return.")
async def get_birthdays_by_rnp_not_equal_ordered_by_birthday(rnp_value: str = Query(..., description="RNP value to exclude"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.RNP != ? ORDER BY T1.Birthday DESC LIMIT ?", (rnp_value, limit))
    result = cursor.fetchall()
    if not result:
        return {"birthdays": []}
    return {"birthdays": [row[0] for row in result]}

# Endpoint to get the count of patient IDs with specific SM values and no thrombosis
@app.get("/v1/thrombosis_prediction/count_ids_by_sm_and_thrombosis", operation_id="get_count_ids_by_sm_and_thrombosis", summary="Retrieves the count of unique patient identifiers that have the specified SM values and no thrombosis. The SM values are provided as a pair, and the thrombosis value is set to 0 to indicate the absence of thrombosis.")
async def get_count_ids_by_sm_and_thrombosis(sm_value1: str = Query(..., description="First SM value"), sm_value2: str = Query(..., description="Second SM value"), thrombosis: int = Query(..., description="Thrombosis value (0 for no thrombosis)")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Examination AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.SM IN (?,?) AND T1.Thrombosis = ?", (sm_value1, sm_value2, thrombosis))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get patient IDs with SM values not in a specific set, ordered by birthday
@app.get("/v1/thrombosis_prediction/ids_by_sm_not_in_ordered_by_birthday", operation_id="get_ids_by_sm_not_in_ordered_by_birthday", summary="Retrieve a list of patient IDs, excluding those with specific SM values, sorted by birthday in descending order. The list is limited to a specified number of results.")
async def get_ids_by_sm_not_in_ordered_by_birthday(sm_value1: str = Query(..., description="First SM value to exclude"), sm_value2: str = Query(..., description="Second SM value to exclude"), limit: int = Query(..., description="Limit the number of results")):
    cursor.execute("SELECT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.SM NOT IN (?,?) ORDER BY T1.Birthday DESC LIMIT ?", (sm_value1, sm_value2, limit))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": [row[0] for row in result]}

# Endpoint to get patient IDs based on laboratory SC170 values and date
@app.get("/v1/thrombosis_prediction/patient_ids_by_sc170_and_date", operation_id="get_patient_ids_by_sc170_and_date", summary="Get patient IDs where laboratory SC170 values are in the specified list and date is after a specified date")
async def get_patient_ids_by_sc170_and_date(sc170_1: str = Query(..., description="First SC170 value"), sc170_2: str = Query(..., description="Second SC170 value"), date: str = Query(..., description="Date in 'YYYY-MM-DD' format")):
    cursor.execute("SELECT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.SC170 IN (?, ?) AND T2.Date > ?", (sc170_1, sc170_2, date))
    result = cursor.fetchall()
    if not result:
        return {"patient_ids": []}
    return {"patient_ids": [row[0] for row in result]}

# Endpoint to get the count of distinct patient IDs based on laboratory SC170 values, sex, and examination symptoms
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_by_sc170_sex_symptoms", operation_id="get_count_distinct_patient_ids_by_sc170_sex_symptoms", summary="Retrieve the count of unique patients who have either of the specified SC170 laboratory values, belong to the specified sex, and have no recorded symptoms.")
async def get_count_distinct_patient_ids_by_sc170_sex_symptoms(sc170_1: str = Query(..., description="First SC170 value"), sc170_2: str = Query(..., description="Second SC170 value"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T3.ID = T2.ID WHERE (T2.SC170 = ? OR T2.SC170 = ?) AND T1.SEX = ? AND T3.Symptoms IS NULL", (sc170_1, sc170_2, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs based on laboratory SSA values and year
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_by_ssa_year", operation_id="get_count_distinct_patient_ids_by_ssa_year", summary="Retrieves the number of unique patients who have laboratory SSA values within a specified range and whose records were created before a given year. The input parameters define the range of SSA values and the year threshold.")
async def get_count_distinct_patient_ids_by_ssa_year(ssa_1: str = Query(..., description="First SSA value"), ssa_2: str = Query(..., description="Second SSA value"), year: str = Query(..., description="Year in 'YYYY' format")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.SSA IN (?, ?) AND STRFTIME('%Y', T2.Date) < ?", (ssa_1, ssa_2, year))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the first patient ID based on laboratory SSA values and first date
@app.get("/v1/thrombosis_prediction/first_patient_id_by_ssa_first_date", operation_id="get_first_patient_id_by_ssa_first_date", summary="Retrieves the ID of the first patient who has a non-null first date and whose laboratory SSA values do not match the provided SSA values. The results are ordered by the first date in ascending order.")
async def get_first_patient_id_by_ssa_first_date(ssa_1: str = Query(..., description="First SSA value"), ssa_2: str = Query(..., description="Second SSA value")):
    cursor.execute("SELECT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.`First Date` IS NOT NULL AND T2.SSA NOT IN (?, ?) ORDER BY T1.`First Date` ASC LIMIT 1", (ssa_1, ssa_2))
    result = cursor.fetchone()
    if not result:
        return {"patient_id": []}
    return {"patient_id": result[0]}

# Endpoint to get the count of distinct patient IDs based on laboratory SSB values and diagnosis
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_by_ssb_diagnosis", operation_id="get_count_distinct_patient_ids_by_ssb_diagnosis", summary="Retrieve the number of unique patients who have at least one of the specified laboratory SSB values and the given diagnosis. The response includes the count of distinct patient IDs that meet these criteria.")
async def get_count_distinct_patient_ids_by_ssb_diagnosis(ssb_1: str = Query(..., description="First SSB value"), ssb_2: str = Query(..., description="Second SSB value"), diagnosis: str = Query(..., description="Diagnosis of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.SSB = ? OR ? AND T1.Diagnosis = ?", (ssb_1, ssb_2, diagnosis))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct examination IDs based on laboratory SSB values and symptoms
@app.get("/v1/thrombosis_prediction/count_distinct_examination_ids_by_ssb_symptoms", operation_id="get_count_distinct_examination_ids_by_ssb_symptoms", summary="Retrieve the count of unique examinations where the laboratory SSB values match the provided list and symptoms are present.")
async def get_count_distinct_examination_ids_by_ssb_symptoms(ssb_1: str = Query(..., description="First SSB value"), ssb_2: str = Query(..., description="Second SSB value")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Examination AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.SSB = ? OR ? AND T1.Symptoms IS NOT NULL", (ssb_1, ssb_2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of distinct patient IDs based on laboratory CENTROMEA, SSB values, and sex
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_by_centromea_ssb_sex", operation_id="get_count_distinct_patient_ids_by_centromea_ssb_sex", summary="Retrieve the count of unique patient records that meet the specified criteria for laboratory CENTROMEA and SSB values and patient sex. The endpoint returns a single integer value representing the count of distinct patient IDs that match the input parameters.")
async def get_count_distinct_patient_ids_by_centromea_ssb_sex(centromea_1: str = Query(..., description="First CENTROMEA value"), centromea_2: str = Query(..., description="Second CENTROMEA value"), ssb_1: str = Query(..., description="First SSB value"), ssb_2: str = Query(..., description="Second SSB value"), sex: str = Query(..., description="Sex of the patient")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.CENTROMEA IN (?, ?) AND T2.SSB IN (?, ?) AND T1.SEX = ?", (centromea_1, centromea_2, ssb_1, ssb_2, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct diagnoses based on laboratory DNA value
@app.get("/v1/thrombosis_prediction/distinct_diagnoses_by_dna", operation_id="get_distinct_diagnoses_by_dna", summary="Retrieves unique diagnoses for patients with a specified DNA value or higher. This operation filters patients based on their DNA value from laboratory records and returns the distinct diagnoses associated with them.")
async def get_distinct_diagnoses_by_dna(dna: int = Query(..., description="DNA value")):
    cursor.execute("SELECT DISTINCT(T1.Diagnosis) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.DNA >= ?", (dna,))
    result = cursor.fetchall()
    if not result:
        return {"diagnoses": []}
    return {"diagnoses": [row[0] for row in result]}

# Endpoint to get the count of distinct patient IDs based on laboratory DNA value and description
@app.get("/v1/thrombosis_prediction/count_distinct_patient_ids_by_dna_description", operation_id="get_count_distinct_patient_ids_by_dna_description", summary="Retrieves the count of unique patients who have a DNA value less than the provided value and lack a description. This operation is useful for identifying the number of patients with incomplete data or specific DNA characteristics.")
async def get_count_distinct_patient_ids_by_dna_description(dna: int = Query(..., description="DNA value")):
    cursor.execute("SELECT COUNT(DISTINCT T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.DNA < ? AND T1.Description IS NULL", (dna,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patient IDs based on laboratory IGG values and admission status
@app.get("/v1/thrombosis_prediction/count_patient_ids_by_igg_admission", operation_id="get_count_patient_ids_by_igg_admission", summary="Get the count of patient IDs where laboratory IGG values are within a specified range and admission status is specified")
async def get_count_patient_ids_by_igg_admission(min_igg: int = Query(..., description="Minimum IGG value"), max_igg: int = Query(..., description="Maximum IGG value"), admission: str = Query(..., description="Admission status")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.IGG > ? AND T2.IGG < ? AND T1.Admission = ?", (min_igg, max_igg, admission))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the ratio of patients with a specific diagnosis and GOT value above a threshold
@app.get("/v1/thrombosis_prediction/diagnosis_ratio_above_got", operation_id="get_diagnosis_ratio_above_got", summary="Retrieves the proportion of patients with a specified diagnosis who have a GOT value equal to or above a given threshold. The calculation is based on the total number of patients with the specified diagnosis and the number of those patients whose GOT value meets or exceeds the provided threshold.")
async def get_diagnosis_ratio_above_got(diagnosis: str = Query(..., description="Diagnosis pattern to match"), got_threshold: int = Query(..., description="GOT threshold value")):
    cursor.execute("SELECT COUNT(CASE WHEN T1.Diagnosis LIKE ? THEN T1.ID ELSE 0 END) / COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.`GOT` >= ?", (diagnosis, got_threshold))
    result = cursor.fetchone()
    if not result:
        return {"ratio": []}
    return {"ratio": result[0]}

# Endpoint to get the count of male patients with GOT value below a threshold
@app.get("/v1/thrombosis_prediction/male_patient_count_below_got", operation_id="get_male_patient_count_below_got", summary="Retrieves the number of male patients who have a GOT value less than the provided threshold. The endpoint uses the provided GOT threshold and sex to filter the patient data and calculate the count.")
async def get_male_patient_count_below_got(got_threshold: int = Query(..., description="GOT threshold value"), sex: str = Query(..., description="Sex of the patient (e.g., 'M' for male)")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GOT < ? AND T1.SEX = ?", (got_threshold, sex))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest birthday of patients with GOT value above a threshold
@app.get("/v1/thrombosis_prediction/latest_birthday_above_got", operation_id="get_latest_birthday_above_got", summary="Retrieves the most recent birthday of patients whose GOT value is equal to or exceeds the provided threshold. This operation fetches data from the Patient and Laboratory tables, filtering for patients with a GOT value at or above the specified threshold. The result is sorted in descending order by birthday, with only the most recent record returned.")
async def get_latest_birthday_above_got(got_threshold: int = Query(..., description="GOT threshold value")):
    cursor.execute("SELECT T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GOT >= ? ORDER BY T1.Birthday DESC LIMIT 1", (got_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"birthday": []}
    return {"birthday": result[0]}

# Endpoint to get the top 3 birthdays of patients with GPT value below a threshold
@app.get("/v1/thrombosis_prediction/top_birthdays_below_gpt", operation_id="get_top_birthdays_below_gpt", summary="Retrieves the birthdays of the top three patients with GPT values below the provided threshold. The patients are ranked based on their GPT values in descending order.")
async def get_top_birthdays_below_gpt(gpt_threshold: int = Query(..., description="GPT threshold value")):
    cursor.execute("SELECT T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GPT < ? ORDER BY T2.GPT DESC LIMIT 3", (gpt_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"birthdays": []}
    return {"birthdays": [row[0] for row in result]}

# Endpoint to get the first date of patients with LDH value below a threshold
@app.get("/v1/thrombosis_prediction/first_date_below_ldh", operation_id="get_first_date_below_ldh", summary="Retrieves the earliest date for a patient whose LDH value is below the specified threshold. The operation compares the LDH value from the Laboratory table with the provided threshold and returns the first date from the Patient table for the patient with the lowest LDH value that meets the threshold condition.")
async def get_first_date_below_ldh(ldh_threshold: int = Query(..., description="LDH threshold value")):
    cursor.execute("SELECT T1.`First Date` FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.LDH < ? ORDER BY T2.LDH ASC LIMIT 1", (ldh_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"first_date": []}
    return {"first_date": result[0]}

# Endpoint to get the latest first date of patients with LDH value above a threshold
@app.get("/v1/thrombosis_prediction/latest_first_date_above_ldh", operation_id="get_latest_first_date_above_ldh", summary="Retrieves the most recent date when a patient first had an LDH value above the specified threshold. This operation fetches the date from the patient records, considering laboratory results where the LDH value meets or exceeds the provided threshold. The result is sorted in descending order and limited to the most recent date.")
async def get_latest_first_date_above_ldh(ldh_threshold: int = Query(..., description="LDH threshold value")):
    cursor.execute("SELECT T1.`First Date` FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.LDH >= ? ORDER BY T1.`First Date` DESC LIMIT 1", (ldh_threshold,))
    result = cursor.fetchone()
    if not result:
        return {"first_date": []}
    return {"first_date": result[0]}

# Endpoint to get the count of patients with ALP value above a threshold and specific admission status
@app.get("/v1/thrombosis_prediction/patient_count_above_alp_admission", operation_id="get_patient_count_above_alp_admission", summary="Get the count of patients with ALP value above a threshold and specific admission status")
async def get_patient_count_above_alp_admission(alp_threshold: int = Query(..., description="ALP threshold value"), admission: str = Query(..., description="Admission status (e.g., '+' for admitted)")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.ALP >= ? AND T1.Admission = ?", (alp_threshold, admission))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients with ALP value below a threshold and specific admission status
@app.get("/v1/thrombosis_prediction/patient_count_below_alp_admission", operation_id="get_patient_count_below_alp_admission", summary="Retrieves the number of patients who have an ALP value less than the provided threshold and a specified admission status. This operation is useful for identifying patients with lower ALP levels and a particular admission status, which can be relevant for various medical analyses or treatments.")
async def get_patient_count_below_alp_admission(alp_threshold: int = Query(..., description="ALP threshold value"), admission: str = Query(..., description="Admission status (e.g., '-' for not admitted)")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.ALP < ? AND T1.Admission = ?", (alp_threshold, admission))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the diagnoses of patients with TP value below a threshold
@app.get("/v1/thrombosis_prediction/diagnoses_below_tp", operation_id="get_diagnoses_below_tp", summary="Retrieves the diagnoses of patients whose TP value is below a specified threshold. The threshold is provided as an input parameter, allowing for customized data retrieval.")
async def get_diagnoses_below_tp(tp_threshold: float = Query(..., description="TP threshold value")):
    cursor.execute("SELECT T1.Diagnosis FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.TP < ?", (tp_threshold,))
    result = cursor.fetchall()
    if not result:
        return {"diagnoses": []}
    return {"diagnoses": [row[0] for row in result]}

# Endpoint to get the count of patients with a specific diagnosis and TP value within a range
@app.get("/v1/thrombosis_prediction/patient_count_diagnosis_tp_range", operation_id="get_patient_count_diagnosis_tp_range", summary="Retrieves the number of patients with a specified diagnosis and TP values within a given range. The diagnosis is a medical condition, while TP refers to a specific laboratory test result. The range is defined by a minimum and maximum TP value.")
async def get_patient_count_diagnosis_tp_range(diagnosis: str = Query(..., description="Diagnosis of the patient"), tp_min: float = Query(..., description="Minimum TP value"), tp_max: float = Query(..., description="Maximum TP value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.Diagnosis = ? AND T2.TP > ? AND T2.TP < ?", (diagnosis, tp_min, tp_max))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the latest date where ALB is within a specified range
@app.get("/v1/thrombosis_prediction/latest_date_alb_range", operation_id="get_latest_date_alb_range", summary="Retrieves the most recent date for which the ALB value falls within the specified range. The operation accepts a minimum and maximum ALB value, and returns the date associated with the ALB value that is closest to the maximum ALB value within the provided range.")
async def get_latest_date_alb_range(min_alb: float = Query(..., description="Minimum ALB value"), max_alb: float = Query(..., description="Maximum ALB value")):
    cursor.execute("SELECT Date FROM Laboratory WHERE ALB > ? AND ALB < ? ORDER BY ALB DESC LIMIT 1", (min_alb, max_alb))
    result = cursor.fetchone()
    if not result:
        return {"date": []}
    return {"date": result[0]}

# Endpoint to get the count of male patients with ALB and TP within specified ranges
@app.get("/v1/thrombosis_prediction/count_male_patients_alb_tp_range", operation_id="get_count_male_patients_alb_tp_range", summary="Retrieves the count of male patients whose ALB and TP values fall within the specified ranges. The operation filters patients based on their sex and laboratory test results for ALB and TP, returning the total count of patients who meet the criteria.")
async def get_count_male_patients_alb_tp_range(sex: str = Query(..., description="Sex of the patient"), min_alb: float = Query(..., description="Minimum ALB value"), max_alb: float = Query(..., description="Maximum ALB value"), min_tp: float = Query(..., description="Minimum TP value"), max_tp: float = Query(..., description="Maximum TP value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.SEX = ? AND T2.ALB > ? AND T2.ALB < ? AND T2.TP BETWEEN ? AND ?", (sex, min_alb, max_alb, min_tp, max_tp))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the aCL IgG, IgM, and IgA values for the female patient with the highest UA value above a specified threshold
@app.get("/v1/thrombosis_prediction/acl_values_highest_ua_female", operation_id="get_acl_values_highest_ua_female", summary="Retrieve the aCL IgG, IgM, and IgA values for the female patient with the highest UA value exceeding the specified minimum threshold. The operation filters patients by sex and UA value, then selects the patient with the highest UA value above the provided threshold. The aCL IgG, IgM, and IgA values for this patient are returned.")
async def get_acl_values_highest_ua_female(sex: str = Query(..., description="Sex of the patient"), min_ua: float = Query(..., description="Minimum UA value")):
    cursor.execute("SELECT T3.`aCL IgG`, T3.`aCL IgM`, T3.`aCL IgA` FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T3.ID = T2.ID WHERE T1.SEX = ? AND T2.UA > ? ORDER BY T2.UA DESC LIMIT 1", (sex, min_ua))
    result = cursor.fetchone()
    if not result:
        return {"aCL IgG": [], "aCL IgM": [], "aCL IgA": []}
    return {"aCL IgG": result[0], "aCL IgM": result[1], "aCL IgA": result[2]}

# Endpoint to get the highest ANA value for patients with CRE below a specified threshold
@app.get("/v1/thrombosis_prediction/highest_ana_cre_below_threshold", operation_id="get_highest_ana_cre_below_threshold", summary="Retrieves the highest ANA value from patients whose CRE value is below the provided maximum CRE threshold. This operation joins data from the Patient, Examination, and Laboratory tables to filter patients based on their CRE values and then selects the highest ANA value.")
async def get_highest_ana_cre_below_threshold(max_cre: float = Query(..., description="Maximum CRE value")):
    cursor.execute("SELECT T2.ANA FROM Patient AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID INNER JOIN Laboratory AS T3 ON T1.ID = T3.ID WHERE T3.CRE < ? ORDER BY T2.ANA DESC LIMIT 1", (max_cre,))
    result = cursor.fetchone()
    if not result:
        return {"ANA": []}
    return {"ANA": result[0]}

# Endpoint to get the ID of the patient with the highest aCL IgA value and CRE below a specified threshold
@app.get("/v1/thrombosis_prediction/patient_id_highest_acl_iga_cre_below_threshold", operation_id="get_patient_id_highest_acl_iga_cre_below_threshold", summary="Retrieves the unique identifier of the patient who has the highest aCL IgA value among those with a CRE value less than the provided maximum CRE value.")
async def get_patient_id_highest_acl_iga_cre_below_threshold(max_cre: float = Query(..., description="Maximum CRE value")):
    cursor.execute("SELECT T2.ID FROM Laboratory AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T1.CRE < ? ORDER BY T2.`aCL IgA` DESC LIMIT 1", (max_cre,))
    result = cursor.fetchone()
    if not result:
        return {"ID": []}
    return {"ID": result[0]}

# Endpoint to get the count of patients with T-BIL above a specified threshold and ANA Pattern containing a specific substring
@app.get("/v1/thrombosis_prediction/count_patients_tbil_ana_pattern", operation_id="get_count_patients_tbil_ana_pattern", summary="Retrieves the number of patients who have a T-BIL value equal to or above a specified minimum threshold and whose ANA Pattern contains a certain substring. This operation is useful for identifying patients with specific liver function and autoimmune disorder characteristics.")
async def get_count_patients_tbil_ana_pattern(min_tbil: float = Query(..., description="Minimum T-BIL value"), ana_pattern: str = Query(..., description="Substring to match in ANA Pattern")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.`T-BIL` >= ? AND T3.`ANA Pattern` LIKE ?", (min_tbil, f'%{ana_pattern}%'))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest ANA value for patients with T-BIL below a specified threshold
@app.get("/v1/thrombosis_prediction/highest_ana_tbil_below_threshold", operation_id="get_highest_ana_tbil_below_threshold", summary="Retrieves the highest ANA (Antinuclear Antibody) value from patients whose T-BIL (Total Bilirubin) is below a specified maximum threshold. The result is ordered by T-BIL in descending order and limited to the top record.")
async def get_highest_ana_tbil_below_threshold(max_tbil: float = Query(..., description="Maximum T-BIL value")):
    cursor.execute("SELECT T3.ANA FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.`T-BIL` < ? ORDER BY T2.`T-BIL` DESC LIMIT 1", (max_tbil,))
    result = cursor.fetchone()
    if not result:
        return {"ANA": []}
    return {"ANA": result[0]}

# Endpoint to get the count of patients with T-CHO above a specified threshold and KCT equal to a specific value
@app.get("/v1/thrombosis_prediction/count_patients_tcho_kct", operation_id="get_count_patients_tcho_kct", summary="Retrieves the number of patients who have a T-CHO level at or above a specified minimum and a KCT value equal to a given input. This operation considers data from the patient, laboratory, and examination tables, filtering for patients with the specified T-CHO and KCT values.")
async def get_count_patients_tcho_kct(min_tcho: float = Query(..., description="Minimum T-CHO value"), kct: str = Query(..., description="KCT value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.`T-CHO` >= ? AND T3.KCT = ?", (min_tcho, kct))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients with a specific ANA Pattern and T-CHO below a specified threshold
@app.get("/v1/thrombosis_prediction/count_patients_ana_pattern_tcho", operation_id="get_count_patients_ana_pattern_tcho", summary="Retrieves the number of patients who have a specific ANA Pattern and a T-CHO level below a given threshold. The ANA Pattern and T-CHO threshold are provided as input parameters.")
async def get_count_patients_ana_pattern_tcho(ana_pattern: str = Query(..., description="ANA Pattern value"), max_tcho: float = Query(..., description="Maximum T-CHO value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T3.`ANA Pattern` = ? AND T2.`T-CHO` < ?", (ana_pattern, max_tcho))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients with TG below a specified threshold and non-null Symptoms
@app.get("/v1/thrombosis_prediction/count_patients_tg_symptoms", operation_id="get_count_patients_tg_symptoms", summary="Retrieves the total number of patients who have a TG level below the specified maximum threshold and have reported symptoms. This operation does not return individual patient data, but rather an aggregate count of patients meeting the specified criteria.")
async def get_count_patients_tg_symptoms(max_tg: float = Query(..., description="Maximum TG value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Examination AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.TG < ? AND T1.Symptoms IS NOT NULL", (max_tg,))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the diagnosis of a patient with a specific TG level
@app.get("/v1/thrombosis_prediction/diagnosis_by_tg_level", operation_id="get_diagnosis_by_tg_level", summary="Retrieves the diagnosis of a patient with the highest TG level below the specified threshold. The TG level is a critical factor in determining the diagnosis, and this operation provides the most relevant diagnosis based on the given TG level.")
async def get_diagnosis_by_tg_level(tg_level: int = Query(..., description="TG level")):
    cursor.execute("SELECT T1.Diagnosis FROM Examination AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.TG < ? ORDER BY T2.TG DESC LIMIT 1", (tg_level,))
    result = cursor.fetchone()
    if not result:
        return {"diagnosis": []}
    return {"diagnosis": result[0]}

# Endpoint to get distinct IDs of patients with specific thrombosis status and CPK level
@app.get("/v1/thrombosis_prediction/distinct_ids_by_thrombosis_cpk", operation_id="get_distinct_ids_by_thrombosis_cpk", summary="Retrieves unique patient identifiers who have a specified thrombosis status and a CPK level below a given threshold. This operation is useful for identifying distinct patients based on their thrombosis condition and CPK level.")
async def get_distinct_ids_by_thrombosis_cpk(thrombosis: int = Query(..., description="Thrombosis status (0 or 1)"), cpk_level: int = Query(..., description="CPK level")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Laboratory AS T1 INNER JOIN Examination AS T2 ON T1.ID = T2.ID WHERE T2.Thrombosis = ? AND T1.CPK < ?", (thrombosis, cpk_level))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get the count of patients with specific CPK level and positive test results
@app.get("/v1/thrombosis_prediction/count_patients_by_cpk_positive_tests", operation_id="get_count_patients_by_cpk_positive_tests", summary="Retrieves the count of patients who have a CPK level below a specified threshold and at least one positive test result from KCT, RVVT, or LAC.")
async def get_count_patients_by_cpk_positive_tests(cpk_level: int = Query(..., description="CPK level"), kct: str = Query(..., description="KCT result ('+' or '-')"), rvvt: str = Query(..., description="RVVT result ('+' or '-')"), lac: str = Query(..., description="LAC result ('+' or '-')")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.CPK < ? AND (T3.KCT = ? OR T3.RVVT = ? OR T3.LAC = ?)", (cpk_level, kct, rvvt, lac))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the birthday of the youngest patient with a specific GLU level
@app.get("/v1/thrombosis_prediction/youngest_patient_by_glu_level", operation_id="get_youngest_patient_by_glu_level", summary="Retrieves the birthdate of the youngest patient who has a GLU level greater than the provided value. The GLU level is a specific laboratory test result.")
async def get_youngest_patient_by_glu_level(glu_level: int = Query(..., description="GLU level")):
    cursor.execute("SELECT T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.GLU > ? ORDER BY T1.Birthday ASC LIMIT 1", (glu_level,))
    result = cursor.fetchone()
    if not result:
        return {"birthday": []}
    return {"birthday": result[0]}

# Endpoint to get the count of patients with specific GLU level and thrombosis status
@app.get("/v1/thrombosis_prediction/count_patients_by_glu_thrombosis", operation_id="get_count_patients_by_glu_thrombosis", summary="Retrieves the total number of patients who have a GLU level below the specified value and a particular thrombosis status. The GLU level and thrombosis status are provided as input parameters.")
async def get_count_patients_by_glu_thrombosis(glu_level: int = Query(..., description="GLU level"), thrombosis: int = Query(..., description="Thrombosis status (0 or 1)")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.GLU < ? AND T3.Thrombosis = ?", (glu_level, thrombosis))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients with specific WBC range and admission status
@app.get("/v1/thrombosis_prediction/count_patients_by_wbc_admission", operation_id="get_count_patients_by_wbc_admission", summary="Retrieves the total number of patients who have a white blood cell (WBC) count within a specified range and a particular admission status. The WBC range is defined by a minimum and maximum value, while the admission status is indicated by a '+' or '-' symbol.")
async def get_count_patients_by_wbc_admission(wbc_min: float = Query(..., description="Minimum WBC level"), wbc_max: float = Query(..., description="Maximum WBC level"), admission: str = Query(..., description="Admission status ('+' or '-')")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.WBC BETWEEN ? AND ? AND T1.Admission = ?", (wbc_min, wbc_max, admission))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of patients with specific diagnosis and WBC range
@app.get("/v1/thrombosis_prediction/count_patients_by_diagnosis_wbc", operation_id="get_count_patients_by_diagnosis_wbc", summary="Retrieves the number of patients with a specific medical diagnosis and a white blood cell (WBC) count within a given range. The diagnosis and WBC range are provided as input parameters.")
async def get_count_patients_by_diagnosis_wbc(diagnosis: str = Query(..., description="Diagnosis"), wbc_min: float = Query(..., description="Minimum WBC level"), wbc_max: float = Query(..., description="Maximum WBC level")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.Diagnosis = ? AND T2.WBC BETWEEN ? AND ?", (diagnosis, wbc_min, wbc_max))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get distinct IDs of patients with specific RBC range and admission status
@app.get("/v1/thrombosis_prediction/distinct_ids_by_rbc_admission", operation_id="get_distinct_ids_by_rbc_admission", summary="Retrieves unique patient identifiers who have RBC levels outside a specified range and a particular admission status. The RBC range and admission status are provided as input parameters.")
async def get_distinct_ids_by_rbc_admission(rbc_min: float = Query(..., description="Minimum RBC level"), rbc_max: float = Query(..., description="Maximum RBC level"), admission: str = Query(..., description="Admission status ('+' or '-')")):
    cursor.execute("SELECT DISTINCT T1.ID FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE (T2.RBC <= ? OR T2.RBC >= ?) AND T1.Admission = ?", (rbc_min, rbc_max, admission))
    result = cursor.fetchall()
    if not result:
        return {"ids": []}
    return {"ids": [row[0] for row in result]}

# Endpoint to get the count of patients with specific PLT range and non-null diagnosis
@app.get("/v1/thrombosis_prediction/count_patients_by_plt_diagnosis", operation_id="get_count_patients_by_plt_diagnosis", summary="Retrieves the number of patients who have a diagnosis and whose platelet (PLT) count falls within a specified range. The range is defined by the minimum and maximum PLT levels provided as input parameters.")
async def get_count_patients_by_plt_diagnosis(plt_min: int = Query(..., description="Minimum PLT level"), plt_max: int = Query(..., description="Maximum PLT level")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.PLT > ? AND T2.PLT < ? AND T1.Diagnosis IS NOT NULL", (plt_min, plt_max))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the PLT levels of patients with specific diagnosis and PLT range
@app.get("/v1/thrombosis_prediction/plt_levels_by_diagnosis_plt_range", operation_id="get_plt_levels_by_diagnosis_plt_range", summary="Retrieves the platelet (PLT) levels of patients diagnosed with a specific condition, within a specified PLT range. The operation filters patients based on their diagnosis and PLT levels, returning the PLT values that fall within the provided range.")
async def get_plt_levels_by_diagnosis_plt_range(diagnosis: str = Query(..., description="Diagnosis"), plt_min: int = Query(..., description="Minimum PLT level"), plt_max: int = Query(..., description="Maximum PLT level")):
    cursor.execute("SELECT T2.PLT FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.Diagnosis = ? AND T2.PLT BETWEEN ? AND ?", (diagnosis, plt_min, plt_max))
    result = cursor.fetchall()
    if not result:
        return {"plt_levels": []}
    return {"plt_levels": [row[0] for row in result]}

# Endpoint to get the average PT value for patients based on PT threshold and sex
@app.get("/v1/thrombosis_prediction/avg_pt_value_by_threshold_sex", operation_id="get_avg_pt_value", summary="Retrieves the average PT value for patients with PT values less than a specified threshold and a particular sex. The operation filters patients based on the provided PT threshold and sex, then calculates the average PT value from the filtered set.")
async def get_avg_pt_value(pt_threshold: float = Query(..., description="PT threshold value"), sex: str = Query(..., description="Sex of the patient (e.g., 'M' for male, 'F' for female)")):
    cursor.execute("SELECT AVG(T2.PT) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T2.PT < ? AND T1.SEX = ?", (pt_threshold, sex))
    result = cursor.fetchone()
    if not result:
        return {"average_pt": []}
    return {"average_pt": result[0]}

# Endpoint to get the count of patients based on PT threshold and thrombosis range
@app.get("/v1/thrombosis_prediction/count_patients_by_pt_threshold_thrombosis_range", operation_id="get_count_patients", summary="Retrieves the number of patients who have PT values less than a specified threshold and thrombosis values that fall within a defined range. The PT threshold and thrombosis range are provided as input parameters.")
async def get_count_patients(pt_threshold: float = Query(..., description="PT threshold value"), max_thrombosis: float = Query(..., description="Maximum thrombosis value"), min_thrombosis: float = Query(..., description="Minimum thrombosis value")):
    cursor.execute("SELECT COUNT(T1.ID) FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID INNER JOIN Examination AS T3 ON T1.ID = T3.ID WHERE T2.PT < ? AND T3.Thrombosis < ? AND T3.Thrombosis > ?", (pt_threshold, max_thrombosis, min_thrombosis))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

api_calls = [
    "/v1/thrombosis_prediction/admission_ratio_male?sex=M",
    "/v1/thrombosis_prediction/percentage_female_born_after_year?year=1930&sex=F",
    "/v1/thrombosis_prediction/admission_percentage_between_years?start_year=1930&end_year=1940",
    "/v1/thrombosis_prediction/admission_ratio_diagnosis?diagnosis=SLE",
    "/v1/thrombosis_prediction/diagnosis_date_by_patient_id?patient_id=30609",
    "/v1/thrombosis_prediction/patient_details_by_id?patient_id=163109",
    "/v1/thrombosis_prediction/patient_details_ldh_greater_than?ldh_value=500",
    "/v1/thrombosis_prediction/patient_age_rvvt_value?rvvt_value=+",
    "/v1/thrombosis_prediction/patient_details_thrombosis_value?thrombosis_value=2",
    "/v1/thrombosis_prediction/patient_ids_birthyear_tcho?birth_year=1937&tcho_value=250",
    "/v1/thrombosis_prediction/patient_details_by_alb?alb_level=3.5",
    "/v1/thrombosis_prediction/percentage_female_tp_range?sex=F&tp_lower=6.0&tp_upper=8.5",
    "/v1/thrombosis_prediction/avg_acl_igg_by_age_admission?age_threshold=50&admission_status=+",
    "/v1/thrombosis_prediction/patient_count_by_description_year_sex_admission?description_year=1997&sex=F&admission_status=-",
    "/v1/thrombosis_prediction/min_age_at_first_date",
    "/v1/thrombosis_prediction/patient_count_by_sex_examination_year_thrombosis?sex=F&examination_year=1997&thrombosis_status=1",
    "/v1/thrombosis_prediction/age_range_by_tg_level?tg_level=200",
    "/v1/thrombosis_prediction/latest_patient_diagnosis_symptoms",
    "/v1/thrombosis_prediction/avg_monthly_patient_count_by_year_sex?year=1998&sex=M",
    "/v1/thrombosis_prediction/earliest_patient_details_by_diagnosis?diagnosis=SJS",
    "/v1/thrombosis_prediction/ua_ratio_by_sex?ua_male=8.0&sex_male=M&ua_female=6.5&sex_female=F",
    "/v1/thrombosis_prediction/patient_count_by_admission_duration?admission=+&min_duration=1",
    "/v1/thrombosis_prediction/patient_count_by_examination_year_age?start_year=1990&end_year=1993&max_age=18",
    "/v1/thrombosis_prediction/patient_count_by_tbil_sex?tbil=2.0&sex=M",
    "/v1/thrombosis_prediction/most_common_diagnosis_by_examination_date?start_date=1985-01-01&end_date=1995-12-31",
    "/v1/thrombosis_prediction/average_age_by_laboratory_date?year=1999&start_date=1991-10-01&end_date=1991-10-30",
    "/v1/thrombosis_prediction/age_diagnosis_by_highest_hgb",
    "/v1/thrombosis_prediction/ana_by_patient_examination_date?patient_id=3605340&examination_date=1996-12-02",
    "/v1/thrombosis_prediction/tcho_status_by_patient_laboratory_date?tcho_threshold=250&patient_id=2927464&laboratory_date=1995-09-04",
    "/v1/thrombosis_prediction/sex_by_earliest_diagnosis?diagnosis=AORTITIS",
    "/v1/thrombosis_prediction/aCL_values?diagnosis=SLE&description=1994-02-19&examination_date=1993-11-12",
    "/v1/thrombosis_prediction/patient_sex?gpt=9.0&date=1992-06-12",
    "/v1/thrombosis_prediction/age_difference?ua=8.4&date=1991-10-21",
    "/v1/thrombosis_prediction/lab_record_count?first_date=1991-06-13&diagnosis=SJS&year=1995",
    "/v1/thrombosis_prediction/patient_diagnosis?examination_date=1997-01-27&diagnosis=SLE",
    "/v1/thrombosis_prediction/patient_symptoms?birthday=1959-03-01&examination_date=1993-09-27",
    "/v1/thrombosis_prediction/tcho_difference?month1=1981-11-%&month2=1981-12-%&birthday=1959-02-18",
    "/v1/thrombosis_prediction/examination_ids?start_date=1997-01-01&end_date=1997-12-31&diagnosis=Behcet",
    "/v1/thrombosis_prediction/lab_ids?start_date=1987-07-06&end_date=1996-01-31&gpt=30&alb=4",
    "/v1/thrombosis_prediction/patient_ids?birth_year=1964&sex=F&admission=+",
    "/v1/thrombosis_prediction/examination_count_thrombosis_ana_acl?thrombosis=2&ana_pattern=S&acl_multiplier=1.2",
    "/v1/thrombosis_prediction/percentage_ua_u_pro_range?ua_threshold=6.5&u_pro_min=0&u_pro_max=30",
    "/v1/thrombosis_prediction/percentage_diagnosis_first_date_sex?diagnosis=BEHCET&first_date_year=1981&sex=M",
    "/v1/thrombosis_prediction/distinct_patient_ids_admission_tbil_date?admission=-&tbil_threshold=2.0&date_pattern=1991-10-%",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_ana_birth_sex?ana_pattern=P&birth_year_start=1980&birth_year_end=1989&sex=F",
    "/v1/thrombosis_prediction/average_alb_plt_diagnosis_sex?plt_threshold=400&diagnosis=SLE&sex=F",
    "/v1/thrombosis_prediction/most_common_symptom_diagnosis?diagnosis=SLE",
    "/v1/thrombosis_prediction/patient_first_date_diagnosis?patient_id=48473",
    "/v1/thrombosis_prediction/patient_count_sex_diagnosis?sex=F&diagnosis=APS",
    "/v1/thrombosis_prediction/laboratory_count_alb_year?alb_min=6.0&alb_max=8.5&year=1997",
    "/v1/thrombosis_prediction/examination_percentage_diagnosis_symptoms?diagnosis=SLE&symptoms=thrombocytopenia",
    "/v1/thrombosis_prediction/patient_percentage_diagnosis_birthyear?sex=F&diagnosis=RA&birth_year=1980",
    "/v1/thrombosis_prediction/patient_count_wbc_sex?wbc_level=3.5&sex=F",
    "/v1/thrombosis_prediction/examination_date_difference?patient_id=821298",
    "/v1/thrombosis_prediction/patient_ua_criteria?sex_f=F&ua_f=6.5&sex_m=M&ua_m=8.0&patient_id=57266",
    "/v1/thrombosis_prediction/laboratory_date_id_got?id=48473&got_level=60",
    "/v1/thrombosis_prediction/patient_sex_birthday_got_year?got_level=60&year=1994",
    "/v1/thrombosis_prediction/distinct_diagnoses_gpt?gpt_value=60",
    "/v1/thrombosis_prediction/average_ldh?ldh_threshold=500",
    "/v1/thrombosis_prediction/patient_ids_ages_ldh_range?min_ldh=600&max_ldh=800",
    "/v1/thrombosis_prediction/admission_details_alp?alp_value=300",
    "/v1/thrombosis_prediction/patient_alp_status_birthday?alp_threshold=300&birthday=1982-04-01",
    "/v1/thrombosis_prediction/patient_details_tp?tp_value=6.0",
    "/v1/thrombosis_prediction/tp_difference_sex?tp_threshold=8.5&sex=F&tp_value=8.5",
    "/v1/thrombosis_prediction/patient_ids_sex_alb_range?sex=M&min_alb=3.5&max_alb=5.5",
    "/v1/thrombosis_prediction/alb_status_birth_year?min_alb=3.5&max_alb=5.5&birth_year=1982",
    "/v1/thrombosis_prediction/percentage_ua_sex?ua_threshold=6.5&sex=F",
    "/v1/thrombosis_prediction/average_ua_by_gender_and_ua_thresholds?ua_threshold_female=6.5&sex_female=F&ua_threshold_male=8.0&sex_male=M",
    "/v1/thrombosis_prediction/distinct_patient_details_by_un?un_value=29",
    "/v1/thrombosis_prediction/distinct_patient_details_by_un_and_diagnosis?un_threshold=30&diagnosis=RA",
    "/v1/thrombosis_prediction/count_patients_by_cre_and_sex?cre_threshold=1.5&sex=M",
    "/v1/thrombosis_prediction/compare_male_female_count_by_cre?sex_male=M&sex_female=F&cre_threshold=1.5",
    "/v1/thrombosis_prediction/patient_with_highest_t_bil",
    "/v1/thrombosis_prediction/patient_ids_sex_by_t_bil?t_bil_threshold=2.0",
    "/v1/thrombosis_prediction/patient_with_highest_t_cho_earliest_birthday",
    "/v1/thrombosis_prediction/average_age_by_t_cho_and_sex?t_cho_threshold=250&sex=M",
    "/v1/thrombosis_prediction/patient_ids_diagnoses_by_tg?tg_threshold=300",
    "/v1/thrombosis_prediction/count_patients_tg_age?tg_threshold=200&age_threshold=50",
    "/v1/thrombosis_prediction/patient_ids_cpk_admission?cpk_threshold=250&admission_status=-",
    "/v1/thrombosis_prediction/count_patients_birth_year_sex_cpk?start_year=1936&end_year=1956&sex=M&cpk_threshold=250",
    "/v1/thrombosis_prediction/patient_ids_sex_age_glu_tcho?glu_threshold=180&tcho_threshold=250",
    "/v1/thrombosis_prediction/patient_ids_glu_first_date?first_date_year=1991&glu_threshold=180",
    "/v1/thrombosis_prediction/patient_ids_sex_birthday_wbc?wbc_lower_threshold=3.5&wbc_upper_threshold=9.0",
    "/v1/thrombosis_prediction/patient_diagnoses_ids_age_rbc?rbc_threshold=3.5",
    "/v1/thrombosis_prediction/patient_ids_admission_sex_rbc_age?sex=F&rbc_lower_threshold=3.5&rbc_upper_threshold=6.0&age_threshold=50",
    "/v1/thrombosis_prediction/patient_ids_sex_hgb_admission?hgb_threshold=10&admission_status=-",
    "/v1/thrombosis_prediction/patient_id_sex_diagnosis_hgb?diagnosis=SLE&hgb_lower_threshold=10&hgb_upper_threshold=17",
    "/v1/thrombosis_prediction/patient_ids_ages_hct_count?hct_threshold=52&count_threshold=2",
    "/v1/thrombosis_prediction/avg_hct_threshold_year?hct_threshold=29&year=1991",
    "/v1/thrombosis_prediction/plt_count_difference?plt_lower_threshold=100&plt_upper_threshold=400",
    "/v1/thrombosis_prediction/patient_ids_plt_range_age_year?plt_lower_bound=100&plt_upper_bound=400&age_difference=50&year=1984",
    "/v1/thrombosis_prediction/pt_percentage_sex_age?pt_threshold=14&sex=F&age_difference=55",
    "/v1/thrombosis_prediction/patient_ids_first_date_pt?first_date_year=1992&pt_threshold=14",
    "/v1/thrombosis_prediction/count_patient_ids_date_aptt?date=1997-01-01&aptt_threshold=45",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_thrombosis_aptt?thrombosis_status=0&aptt_threshold=45",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_fg_wbc_sex?fg_lower_threshold=150&fg_upper_threshold=450&wbc_lower_threshold=3.5&wbc_upper_threshold=9.0&sex=M",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_fg_birthday?fg_lower_threshold=150&fg_upper_threshold=450&birthday=1980-01-01",
    "/v1/thrombosis_prediction/diagnoses_by_u_pro_threshold?u_pro_threshold=30",
    "/v1/thrombosis_prediction/patient_ids_by_u_pro_range_and_diagnosis?u_pro_min=0&u_pro_max=30&diagnosis=SLE",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_by_igg_threshold?igg_threshold=2000",
    "/v1/thrombosis_prediction/count_patient_ids_by_igg_range_and_symptoms?igg_min=900&igg_max=2000",
    "/v1/thrombosis_prediction/diagnosis_by_highest_iga_in_range?iga_min=80&iga_max=500",
    "/v1/thrombosis_prediction/count_patient_ids_by_iga_range_and_first_date?iga_min=80&iga_max=500&first_date_year=1990",
    "/v1/thrombosis_prediction/most_common_diagnosis_by_igm_range?igm_min=40&igm_max=400",
    "/v1/thrombosis_prediction/count_patient_ids_by_crp_and_null_description?crp_value=+",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_by_cre_and_age?cre_threshold=1.5&max_age=70",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_by_ra_and_kct?ra_value_1=-&ra_value_2=+-&kct_value=+",
    "/v1/thrombosis_prediction/diagnoses_by_ra_and_birthday?ra_value1=-&ra_value2=+-&birthday=1985-01-01",
    "/v1/thrombosis_prediction/patient_ids_by_rf_and_age?rf_threshold=20&age_threshold=60",
    "/v1/thrombosis_prediction/count_distinct_ids_by_rf_and_thrombosis?rf_threshold=20&thrombosis=0",
    "/v1/thrombosis_prediction/count_distinct_ids_by_c3_and_ana_pattern?c3_threshold=35&ana_pattern=P",
    "/v1/thrombosis_prediction/distinct_ids_by_hct_range_ordered_by_acl_iga?hct_min=52&hct_max=29&limit=1",
    "/v1/thrombosis_prediction/count_distinct_ids_by_c4_and_diagnosis?c4_threshold=10&diagnosis=APS",
    "/v1/thrombosis_prediction/count_distinct_ids_by_rnp_and_admission?rnp_value1=negative&rnp_value2=0&admission=+",
    "/v1/thrombosis_prediction/birthdays_by_rnp_not_equal_ordered_by_birthday?rnp_value=-&limit=1",
    "/v1/thrombosis_prediction/count_ids_by_sm_and_thrombosis?sm_value1=negative&sm_value2=0&thrombosis=0",
    "/v1/thrombosis_prediction/ids_by_sm_not_in_ordered_by_birthday?sm_value1=negative&sm_value2=0&limit=3",
    "/v1/thrombosis_prediction/patient_ids_by_sc170_and_date?sc170_1=negative&sc170_2=0&date=1997-01-01",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_by_sc170_sex_symptoms?sc170_1=negative&sc170_2=0&sex=F",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_by_ssa_year?ssa_1=negative&ssa_2=0&year=2000",
    "/v1/thrombosis_prediction/first_patient_id_by_ssa_first_date?ssa_1=negative&ssa_2=0",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_by_ssb_diagnosis?ssb_1=negative&ssb_2=0&diagnosis=SLE",
    "/v1/thrombosis_prediction/count_distinct_examination_ids_by_ssb_symptoms?ssb_1=negative&ssb_2=0",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_by_centromea_ssb_sex?centromea_1=negative&centromea_2=0&ssb_1=negative&ssb_2=0&sex=M",
    "/v1/thrombosis_prediction/distinct_diagnoses_by_dna?dna=8",
    "/v1/thrombosis_prediction/count_distinct_patient_ids_by_dna_description?dna=8",
    "/v1/thrombosis_prediction/count_patient_ids_by_igg_admission?min_igg=900&max_igg=2000&admission=+",
    "/v1/thrombosis_prediction/diagnosis_ratio_above_got?diagnosis=%25SLE%25&got_threshold=60",
    "/v1/thrombosis_prediction/male_patient_count_below_got?got_threshold=60&sex=M",
    "/v1/thrombosis_prediction/latest_birthday_above_got?got_threshold=60",
    "/v1/thrombosis_prediction/top_birthdays_below_gpt?gpt_threshold=60",
    "/v1/thrombosis_prediction/first_date_below_ldh?ldh_threshold=500",
    "/v1/thrombosis_prediction/latest_first_date_above_ldh?ldh_threshold=500",
    "/v1/thrombosis_prediction/patient_count_above_alp_admission?alp_threshold=300&admission=+",
    "/v1/thrombosis_prediction/patient_count_below_alp_admission?alp_threshold=300&admission=-",
    "/v1/thrombosis_prediction/diagnoses_below_tp?tp_threshold=6.0",
    "/v1/thrombosis_prediction/patient_count_diagnosis_tp_range?diagnosis=SJS&tp_min=6.0&tp_max=8.5",
    "/v1/thrombosis_prediction/latest_date_alb_range?min_alb=3.5&max_alb=5.5",
    "/v1/thrombosis_prediction/count_male_patients_alb_tp_range?sex=M&min_alb=3.5&max_alb=5.5&min_tp=6.0&max_tp=8.5",
    "/v1/thrombosis_prediction/acl_values_highest_ua_female?sex=F&min_ua=6.5",
    "/v1/thrombosis_prediction/highest_ana_cre_below_threshold?max_cre=1.5",
    "/v1/thrombosis_prediction/patient_id_highest_acl_iga_cre_below_threshold?max_cre=1.5",
    "/v1/thrombosis_prediction/count_patients_tbil_ana_pattern?min_tbil=2&ana_pattern=P",
    "/v1/thrombosis_prediction/highest_ana_tbil_below_threshold?max_tbil=2.0",
    "/v1/thrombosis_prediction/count_patients_tcho_kct?min_tcho=250&kct=-",
    "/v1/thrombosis_prediction/count_patients_ana_pattern_tcho?ana_pattern=P&max_tcho=250",
    "/v1/thrombosis_prediction/count_patients_tg_symptoms?max_tg=200",
    "/v1/thrombosis_prediction/diagnosis_by_tg_level?tg_level=200",
    "/v1/thrombosis_prediction/distinct_ids_by_thrombosis_cpk?thrombosis=0&cpk_level=250",
    "/v1/thrombosis_prediction/count_patients_by_cpk_positive_tests?cpk_level=250&kct=%2B&rvvt=%2B&lac=%2B",
    "/v1/thrombosis_prediction/youngest_patient_by_glu_level?glu_level=180",
    "/v1/thrombosis_prediction/count_patients_by_glu_thrombosis?glu_level=180&thrombosis=0",
    "/v1/thrombosis_prediction/count_patients_by_wbc_admission?wbc_min=3.5&wbc_max=9&admission=%2B",
    "/v1/thrombosis_prediction/count_patients_by_diagnosis_wbc?diagnosis=SLE&wbc_min=3.5&wbc_max=9",
    "/v1/thrombosis_prediction/distinct_ids_by_rbc_admission?rbc_min=3.5&rbc_max=6&admission=%2D",
    "/v1/thrombosis_prediction/count_patients_by_plt_diagnosis?plt_min=100&plt_max=400",
    "/v1/thrombosis_prediction/plt_levels_by_diagnosis_plt_range?diagnosis=MCTD&plt_min=100&plt_max=400",
    "/v1/thrombosis_prediction/avg_pt_value_by_threshold_sex?pt_threshold=14&sex=M",
    "/v1/thrombosis_prediction/count_patients_by_pt_threshold_thrombosis_range?pt_threshold=14&max_thrombosis=3&min_thrombosis=0"
]
