from fastapi import APIRouter, HTTPException, Query, Request
import sqlite3

app = APIRouter()

conn = sqlite3.connect('db/genes/genes.sqlite')
cursor = conn.cursor()

# Endpoint to get chromosome information based on gene localization
@app.get("/v1/genes/chromosome_by_localization", operation_id="get_chromosome_by_localization", summary="Retrieves the chromosome information for genes that are localized in the specified region. The localization parameter is used to filter the genes and return the corresponding chromosome data.")
async def get_chromosome_by_localization(localization: str = Query(..., description="Localization of the gene")):
    cursor.execute("SELECT T1.Chromosome FROM Genes AS T1 INNER JOIN Classification AS T2 ON T1.GeneID = T2.GeneID WHERE T2.Localization = ?", (localization,))
    result = cursor.fetchall()
    if not result:
        return {"chromosomes": []}
    return {"chromosomes": [row[0] for row in result]}

# Endpoint to get the count of genes based on localization and essentiality
@app.get("/v1/genes/count_genes_by_localization_essentiality", operation_id="get_count_genes_by_localization_essentiality", summary="Retrieves the total number of genes that are located in a specific area and have a certain essentiality status. The localization and essentiality parameters are used to filter the genes and calculate the count.")
async def get_count_genes_by_localization_essentiality(localization: str = Query(..., description="Localization of the gene"), essentiality: str = Query(..., description="Essentiality of the gene")):
    cursor.execute("SELECT COUNT(T1.GeneID) FROM Genes AS T1 INNER JOIN Classification AS T2 ON T1.GeneID = T2.GeneID WHERE T2.Localization = ? AND T1.Essential = ?", (localization, essentiality))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of genes based on localization and phenotype
@app.get("/v1/genes/count_genes_by_localization_phenotype", operation_id="get_count_genes_by_localization_phenotype", summary="Retrieves the total number of genes that share a specified localization and phenotype. The localization and phenotype are provided as input parameters, allowing for a targeted count of genes that meet these criteria.")
async def get_count_genes_by_localization_phenotype(localization: str = Query(..., description="Localization of the gene"), phenotype: str = Query(..., description="Phenotype of the gene")):
    cursor.execute("SELECT COUNT(T1.GeneID) FROM Genes AS T1 INNER JOIN Classification AS T2 ON T1.GeneID = T2.GeneID WHERE T2.Localization = ? AND T1.Phenotype = ?", (localization, phenotype))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the localization of the gene with the highest chromosome number
@app.get("/v1/genes/localization_highest_chromosome", operation_id="get_localization_highest_chromosome", summary="Retrieves the localization of the gene with the highest chromosome number. This operation identifies the gene with the highest chromosome number and returns its associated localization data.")
async def get_localization_highest_chromosome():
    cursor.execute("SELECT T2.Localization FROM Genes AS T1 INNER JOIN Classification AS T2 ON T1.GeneID = T2.GeneID ORDER BY T1.Chromosome DESC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"localization": []}
    return {"localization": result[0]}

# Endpoint to get the highest expression correlation between genes with specific localizations
@app.get("/v1/genes/highest_expression_corr_by_localization", operation_id="get_highest_expression_corr_by_localization", summary="Retrieves the highest correlation in expression between two genes, each with a specific localization. The localizations are provided as input parameters. The result is the highest correlation value found in the interactions between genes with the given localizations.")
async def get_highest_expression_corr_by_localization(localization1: str = Query(..., description="Localization of the first gene"), localization2: str = Query(..., description="Localization of the second gene")):
    cursor.execute("SELECT T2.Expression_Corr FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 INNER JOIN Genes AS T3 ON T3.GeneID = T2.GeneID2 WHERE T1.Localization = ? AND T3.Localization = ? ORDER BY T2.Expression_Corr DESC LIMIT 1", (localization1, localization2))
    result = cursor.fetchone()
    if not result:
        return {"expression_corr": []}
    return {"expression_corr": result[0]}

# Endpoint to get the function of the gene with the lowest expression correlation
@app.get("/v1/genes/function_lowest_expression_corr", operation_id="get_function_lowest_expression_corr", summary="Retrieves the biological function of the gene that exhibits the lowest correlation in expression levels. This operation considers the interactions between genes and their respective expression correlations to determine the gene with the least correlated expression pattern.")
async def get_function_lowest_expression_corr():
    cursor.execute("SELECT T1.Function FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 ORDER BY T2.Expression_Corr ASC LIMIT 1")
    result = cursor.fetchone()
    if not result:
        return {"function": []}
    return {"function": result[0]}

# Endpoint to get the count of genes with expression correlation below a threshold and specific class
@app.get("/v1/genes/count_genes_by_expression_corr_class", operation_id="get_count_genes_by_expression_corr_class", summary="Retrieves the number of genes that have an expression correlation below a specified threshold and belong to a particular class. The correlation threshold and gene class are provided as input parameters.")
async def get_count_genes_by_expression_corr_class(expression_corr_threshold: float = Query(..., description="Threshold for expression correlation"), gene_class: str = Query(..., description="Class of the gene")):
    cursor.execute("SELECT COUNT(T1.GeneID) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T2.Expression_Corr < ? AND T1.Class = ?", (expression_corr_threshold, gene_class))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the highest expression correlation for genes on specific chromosomes
@app.get("/v1/genes/highest_expression_corr_by_chromosomes", operation_id="get_highest_expression_corr_by_chromosomes", summary="Retrieves the highest expression correlation value for genes located on the specified pair of chromosomes. The correlation data is derived from the interactions of genes, with the highest correlation value being returned. The chromosomes are identified by their respective numbers.")
async def get_highest_expression_corr_by_chromosomes(chromosome1: int = Query(..., description="First chromosome number"), chromosome2: int = Query(..., description="Second chromosome number")):
    cursor.execute("SELECT T2.Expression_Corr FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T1.Chromosome = ? OR T1.Chromosome = ? ORDER BY T2.Expression_Corr DESC LIMIT 1", (chromosome1, chromosome2))
    result = cursor.fetchone()
    if not result:
        return {"expression_corr": []}
    return {"expression_corr": result[0]}

# Endpoint to get gene interactions based on localization and chromosome
@app.get("/v1/genes/gene_interactions_by_localization_chromosome", operation_id="get_gene_interactions_by_localization_chromosome", summary="Retrieves a list of gene interactions based on the specified gene localization and chromosome number. The operation filters genes by their localization and chromosome, and returns the corresponding gene-gene interactions.")
async def get_gene_interactions_by_localization_chromosome(localization: str = Query(..., description="Localization of the gene"), chromosome: int = Query(..., description="Chromosome number")):
    cursor.execute("SELECT T2.GeneID1, T2.GeneID2 FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T1.Localization = ? AND T1.Chromosome = ?", (localization, chromosome))
    result = cursor.fetchall()
    if not result:
        return {"interactions": []}
    return {"interactions": [{"gene_id1": row[0], "gene_id2": row[1]} for row in result]}

# Endpoint to get the count of genes based on localization, function, and essentiality
@app.get("/v1/genes/count_genes_by_localization_function_essentiality", operation_id="get_count_genes_by_localization_function_essentiality", summary="Retrieves the count of genes that meet the specified localization, function, and essentiality criteria. The localization, function, and essentiality parameters are used to filter the genes. The endpoint returns the total number of genes that match the provided criteria.")
async def get_count_genes_by_localization_function_essentiality(localization: str = Query(..., description="Localization of the gene"), function: str = Query(..., description="Function of the gene"), essentiality: str = Query(..., description="Essentiality of the gene")):
    cursor.execute("SELECT COUNT(T1.GeneID) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T1.Localization != ? AND T1.Function = ? AND T1.Essential = ?", (localization, function, essentiality))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the count of gene interactions based on expression correlation and essentiality
@app.get("/v1/genes/count_gene_interactions", operation_id="get_count_gene_interactions", summary="Retrieves the total number of gene interactions that surpass a specified expression correlation threshold and involve genes with a certain essentiality status.")
async def get_count_gene_interactions(expression_corr: float = Query(..., description="Expression correlation threshold"), essential: str = Query(..., description="Essentiality of the gene")):
    cursor.execute("SELECT COUNT(T2.GeneID2) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T2.Expression_Corr > ? AND T1.Essential = ?", (expression_corr, essential))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of genes with chromosome number greater than a threshold and positive expression correlation
@app.get("/v1/genes/percentage_chromosome_expression_corr", operation_id="get_percentage_chromosome_expression_corr", summary="Retrieves the percentage of genes that have a chromosome number greater than a specified threshold and a positive expression correlation above a certain value. The calculation is based on the total number of genes in the database.")
async def get_percentage_chromosome_expression_corr(chromosome_threshold: int = Query(..., description="Chromosome number threshold"), expression_corr: float = Query(..., description="Expression correlation threshold")):
    cursor.execute("SELECT CAST(SUM(IIF(T1.Chromosome > ? AND T3.Chromosome > ?, 1, 0)) AS REAL) * 100 / COUNT(T1.GeneID) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 INNER JOIN Genes AS T3 ON T3.GeneID = T2.GeneID2 WHERE T2.Expression_Corr > ?", (chromosome_threshold, chromosome_threshold, expression_corr))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the average expression correlation for a specific gene class
@app.get("/v1/genes/average_expression_corr", operation_id="get_average_expression_corr", summary="Retrieves the average correlation of expression levels for a specific gene class. This operation calculates the mean expression correlation for all genes within the provided gene class, offering insights into the overall correlation trend for that class.")
async def get_average_expression_corr(gene_class: str = Query(..., description="Class of the gene")):
    cursor.execute("SELECT AVG(T2.Expression_Corr) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T1.Class = ?", (gene_class,))
    result = cursor.fetchone()
    if not result:
        return {"average": []}
    return {"average": result[0]}

# Endpoint to get distinct gene IDs based on localization and function
@app.get("/v1/genes/distinct_gene_ids", operation_id="get_distinct_gene_ids", summary="Retrieves a list of unique gene identifiers based on the specified gene localization and function. This operation filters the genes database to return only distinct gene IDs that match the provided localization and functional criteria.")
async def get_distinct_gene_ids(localization: str = Query(..., description="Localization of the gene"), function: str = Query(..., description="Function of the gene")):
    cursor.execute("SELECT DISTINCT GeneID FROM Genes WHERE Localization = ? AND Function = ?", (localization, function))
    result = cursor.fetchall()
    if not result:
        return {"gene_ids": []}
    return {"gene_ids": [row[0] for row in result]}

# Endpoint to get the count of genes based on localization
@app.get("/v1/genes/count_genes_by_localization", operation_id="get_count_genes_by_localization", summary="Retrieves the total count of genes that are localized in either of the two specified locations. The operation considers the provided localization parameters to filter and count the genes accordingly.")
async def get_count_genes_by_localization(localization1: str = Query(..., description="First localization of the gene"), localization2: str = Query(..., description="Second localization of the gene")):
    cursor.execute("SELECT COUNT(GeneID) FROM Classification WHERE Localization IN (?, ?)", (localization1, localization2))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get expression correlation and percentage of negative correlations for a specific interaction type
@app.get("/v1/genes/expression_corr_and_percentage", operation_id="get_expression_corr_and_percentage", summary="Retrieves the expression correlation and the percentage of negative correlations for a specific interaction type. The interaction type is used to filter the data, and the expression correlation threshold is used to calculate the percentage of negative correlations. The response includes the average expression correlation and the percentage of correlations below the specified threshold.")
async def get_expression_corr_and_percentage(interaction_type: str = Query(..., description="Type of interaction"), expression_corr_threshold: float = Query(..., description="Expression correlation threshold")):
    cursor.execute("SELECT Expression_Corr FROM Interactions WHERE Type = ? UNION ALL SELECT CAST(SUM(Expression_Corr < ?) AS REAL) * 100 / COUNT(*) FROM Interactions WHERE Type = ?", (interaction_type, expression_corr_threshold, interaction_type))
    result = cursor.fetchall()
    if not result:
        return {"expression_corr": [], "percentage": []}
    return {"expression_corr": [row[0] for row in result[:-1]], "percentage": result[-1][0]}

# Endpoint to get the count and percentage of genes with specific localization and phenotype
@app.get("/v1/genes/count_and_percentage_localization_phenotype", operation_id="get_count_and_percentage_localization_phenotype", summary="Retrieves the total count and percentage of genes that exhibit a specified localization and phenotype. The operation calculates these values based on the provided localization and phenotype parameters, which determine the specific gene characteristics to consider.")
async def get_count_and_percentage_localization_phenotype(localization: str = Query(..., description="Localization of the gene"), phenotype: str = Query(..., description="Phenotype of the gene")):
    cursor.execute("SELECT SUM(Localization = ? AND Phenotype = ?) , CAST(SUM(Localization = ?) AS REAL) * 100 / COUNT(GeneID) FROM Genes", (localization, phenotype, localization))
    result = cursor.fetchone()
    if not result:
        return {"count": [], "percentage": []}
    return {"count": result[0], "percentage": result[1]}

# Endpoint to get interaction types for genes with specific function and essentiality
@app.get("/v1/genes/interaction_types_by_function_essentiality", operation_id="get_interaction_types_by_function_essentiality", summary="Retrieves the interaction types associated with genes that have a specified function and essentiality. The function and essentiality of the gene are used to filter the results.")
async def get_interaction_types_by_function_essentiality(function: str = Query(..., description="Function of the gene"), essential: str = Query(..., description="Essentiality of the gene")):
    cursor.execute("SELECT T2.Type FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T1.Function = ? AND T1.Essential = ?", (function, essential))
    result = cursor.fetchall()
    if not result:
        return {"interaction_types": []}
    return {"interaction_types": [row[0] for row in result]}

# Endpoint to get gene IDs based on expression correlation and localization
@app.get("/v1/genes/gene_ids_by_expression_corr_localization", operation_id="get_gene_ids_by_expression_corr_localization", summary="Retrieves a list of gene IDs that meet the specified expression correlation threshold and localization criteria. The expression correlation threshold filters genes based on their correlation coefficient, while the localization criteria narrows down the results to genes found in the specified location.")
async def get_gene_ids_by_expression_corr_localization(expression_corr: float = Query(..., description="Expression correlation threshold"), localization: str = Query(..., description="Localization of the gene")):
    cursor.execute("SELECT T1.GeneID FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T2.Expression_Corr > ? AND T1.Localization = ?", (expression_corr, localization))
    result = cursor.fetchall()
    if not result:
        return {"gene_ids": []}
    return {"gene_ids": [row[0] for row in result]}

# Endpoint to get gene IDs based on localization, class, essentiality, and expression correlation
@app.get("/v1/genes/gene_ids_by_localization_class_essentiality_expression_corr", operation_id="get_gene_ids_by_localization_class_essentiality_expression_corr", summary="Retrieves a list of gene IDs that meet the specified localization, class, essentiality, and expression correlation criteria. The localization, class, and essentiality parameters filter the genes, while the expression correlation threshold determines the minimum correlation value for the gene interactions to be considered.")
async def get_gene_ids_by_localization_class_essentiality_expression_corr(localization: str = Query(..., description="Localization of the gene"), gene_class: str = Query(..., description="Class of the gene"), essential: str = Query(..., description="Essentiality of the gene"), expression_corr: float = Query(..., description="Expression correlation threshold")):
    cursor.execute("SELECT T2.GeneID1 FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T1.Localization = ? AND T1.Class = ? AND T1.Essential = ? AND T2.Expression_Corr != ?", (localization, gene_class, essential, expression_corr))
    result = cursor.fetchall()
    if not result:
        return {"gene_ids": []}
    return {"gene_ids": [row[0] for row in result]}

# Endpoint to get the count of genes based on interaction type, phenotype, class, and essentiality
@app.get("/v1/genes/count_genes_by_interaction_type_phenotype_class_essentiality", operation_id="get_count_genes_by_interaction_type_phenotype_class_essentiality", summary="Retrieves the total number of genes that meet the specified criteria for interaction type, phenotype, class, and essentiality. The interaction type must not match the provided value, while the phenotype, class, and essentiality must match the provided values.")
async def get_count_genes_by_interaction_type_phenotype_class_essentiality(interaction_type: str = Query(..., description="Interaction type"), phenotype: str = Query(..., description="Phenotype"), class_type: str = Query(..., description="Class type"), essentiality: str = Query(..., description="Essentiality")):
    cursor.execute("SELECT COUNT(T1.GeneID) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T2.Type != ? AND T1.Phenotype = ? AND T1.Class != ? AND T1.Essential = ?", (interaction_type, phenotype, class_type, essentiality))
    result = cursor.fetchone()
    if not result:
        return {"count": []}
    return {"count": result[0]}

# Endpoint to get the percentage of genes with positive expression correlation based on phenotype and motif
@app.get("/v1/genes/percentage_positive_expression_correlation_by_phenotype_motif", operation_id="get_percentage_positive_expression_correlation_by_phenotype_motif", summary="Retrieves the percentage of genes that exhibit a positive correlation in expression, filtered by a specific phenotype and motif. The phenotype and motif parameters are used to narrow down the selection of genes for analysis.")
async def get_percentage_positive_expression_correlation_by_phenotype_motif(phenotype: str = Query(..., description="Phenotype"), motif: str = Query(..., description="Motif")):
    cursor.execute("SELECT CAST(SUM(IIF(T2.Expression_Corr > 0, 1, 0)) AS REAL) * 100 / COUNT(T2.GeneID1) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T1.Phenotype = ? AND T1.Motif = ?", (phenotype, motif))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

# Endpoint to get the percentage of non-essential genes with negative expression correlation
@app.get("/v1/genes/percentage_non_essential_genes_negative_expression_correlation", operation_id="get_percentage_non_essential_genes_negative_expression_correlation", summary="Retrieves the percentage of non-essential genes that exhibit a negative correlation in their expression. The essentiality parameter is used to filter the genes based on their essentiality status.")
async def get_percentage_non_essential_genes_negative_expression_correlation(essentiality: str = Query(..., description="Essentiality")):
    cursor.execute("SELECT CAST(COUNT(T1.GeneID) AS REAL) * 100 / ( SELECT COUNT(T1.GeneID) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T2.Expression_Corr < 0 ) FROM Genes AS T1 INNER JOIN Interactions AS T2 ON T1.GeneID = T2.GeneID1 WHERE T2.Expression_Corr < 0 AND T1.Essential = ?", (essentiality,))
    result = cursor.fetchone()
    if not result:
        return {"percentage": []}
    return {"percentage": result[0]}

api_calls = [
    "/v1/genes/chromosome_by_localization?localization=plasma%20membrane",
    "/v1/genes/count_genes_by_localization_essentiality?localization=nucleus&essentiality=Non-Essential",
    "/v1/genes/count_genes_by_localization_phenotype?localization=vacuole&phenotype=Nucleic%20acid%20metabolism%20defects",
    "/v1/genes/localization_highest_chromosome",
    "/v1/genes/highest_expression_corr_by_localization?localization1=nucleus&localization2=nucleus",
    "/v1/genes/function_lowest_expression_corr",
    "/v1/genes/count_genes_by_expression_corr_class?expression_corr_threshold=0&gene_class=Motorproteins",
    "/v1/genes/highest_expression_corr_by_chromosomes?chromosome1=6&chromosome2=8",
    "/v1/genes/gene_interactions_by_localization_chromosome?localization=cytoplasm&chromosome=7",
    "/v1/genes/count_genes_by_localization_function_essentiality?localization=cytoplasm&function=TRANSCRIPTION&essentiality=NON-Essential",
    "/v1/genes/count_gene_interactions?expression_corr=0&essential=Non-Essential",
    "/v1/genes/percentage_chromosome_expression_corr?chromosome_threshold=10&expression_corr=0",
    "/v1/genes/average_expression_corr?gene_class=ATPases",
    "/v1/genes/distinct_gene_ids?localization=cytoplasm&function=METABOLISM",
    "/v1/genes/count_genes_by_localization?localization1=plasma&localization2=nucleus",
    "/v1/genes/expression_corr_and_percentage?interaction_type=Physical&expression_corr_threshold=0",
    "/v1/genes/count_and_percentage_localization_phenotype?localization=cytoskeleton&phenotype=Conditional%20phenotypes",
    "/v1/genes/interaction_types_by_function_essentiality?function=TRANSCRIPTION&essential=Non-Essential",
    "/v1/genes/gene_ids_by_expression_corr_localization?expression_corr=0&localization=nucleus",
    "/v1/genes/gene_ids_by_localization_class_essentiality_expression_corr?localization=nucleus&gene_class=Transcription%20factors&essential=Essential&expression_corr=0",
    "/v1/genes/count_genes_by_interaction_type_phenotype_class_essentiality?interaction_type=Physical&phenotype=Cell%20cycle%20defects&class_type=Motorproteins&essentiality=Non-Essential",
    "/v1/genes/percentage_positive_expression_correlation_by_phenotype_motif?phenotype=Nucleic%20acid%20metabolism%20defects&motif=PS00107",
    "/v1/genes/percentage_non_essential_genes_negative_expression_correlation?essentiality=Non-Essential"
]
