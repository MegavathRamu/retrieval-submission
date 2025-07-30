


!pip install pymongo
!pip install faiss-cpu
!pip install openpyxl


### code 



import pymongo
import requests
import numpy as np
import pandas as pd
import os
import re
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EMBEDDING_DIM = 1024

def connect_to_mongodb(uri, db_name, collection_name):
    try:
        client = pymongo.MongoClient(uri)
        db = client[db_name]
        collection = db[collection_name]
        print(f"Successfully connected to MongoDB: {db_name}/{collection_name}")
        return collection
    except pymongo.errors.ConnectionFailure as e:
        print(f"MongoDB Connection Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during MongoDB connection: {e}")
        return None

def embed_query_with_voyage(text, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "voyage-3",
        "input": [text]
    }
    try:
        response = requests.post("https://api.voyageai.com/v1/embeddings", json=payload, headers=headers)
        response.raise_for_status()
        embedding = np.array(response.json()["data"][0]["embedding"], dtype=np.float32)
        return embedding
    except requests.exceptions.HTTPError as e:
        print(f"Voyage AI HTTP Error for query '{text[:50]}...': {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Voyage AI Connection Error for query '{text[:50]}...': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Voyage AI embedding for query '{text[:50]}...': {e}")
        return None

def parse_criteria_string_to_mongo_query(criteria_str):
    mongo_conditions = []
    if not criteria_str or criteria_str.lower() == "nan":
        return {}

    if ' AND ' in criteria_str.upper():
        individual_criteria = [c.strip() for c in criteria_str.split(' AND ')]
        for criterion in individual_criteria:
            if '>=' in criterion:
                parts = [p.strip() for p in criterion.split('>=')]
                field, value = parts[0], float(parts[1])
                mongo_conditions.append({field: {'$gte': value}})
            elif '<=' in criterion:
                parts = [p.strip() for p in criterion.split('<=')]
                field, value = parts[0], float(parts[1])
                mongo_conditions.append({field: {'$lte': value}})
            elif '>' in criterion:
                parts = [p.strip() for p in criterion.split('>')]
                field, value = parts[0], float(parts[1])
                mongo_conditions.append({field: {'$gt': value}})
            elif '<' in criterion:
                parts = [p.strip() for p in criterion.split('<')]
                field, value = parts[0], float(parts[1])
                mongo_conditions.append({field: {'$lt': value}})
            elif 'CONTAINS' in criterion.upper():
                parts = [p.strip() for p in re.split(r'\bCONTAINS\b', criterion, 1, flags=re.IGNORECASE)]
                if len(parts) == 2:
                    field, value = parts[0], parts[1].strip("'\" ").lower()
                    mongo_conditions.append({field: {'$regex': re.escape(value), '$options': 'i'}})
                else:
                    print(f"Malformed CONTAINS criterion: {criterion}")
            elif 'EXACTLY' in criterion.upper():
                parts = [p.strip() for p in re.split(r'\bEXACTLY\b', criterion, 1, flags=re.IGNORECASE)]
                if len(parts) == 2:
                    field, value = parts[0], parts[1].strip("'\" ").lower()
                    mongo_conditions.append({field: {'$regex': f"^{re.escape(value)}$", '$options': 'i'}})
                else:
                    print(f"Malformed EXACTLY criterion: {criterion}")
            elif 'IS' in criterion.upper():
                parts = [p.strip() for p in re.split(r'\bIS\b', criterion, 1, flags=re.IGNORECASE)]
                if len(parts) == 2:
                    field, value_str = parts[0], parts[1].strip().lower()
                    value = True if value_str == 'true' else (False if value_str == 'false' else value_str)
                    mongo_conditions.append({field: value})
                else:
                    print(f"Malformed IS criterion: {criterion}")
            else:
                print(f"Unrecognized structured criterion: {criterion}")
        if mongo_conditions:
            return {'$and': mongo_conditions}

    individual_criteria = [re.sub(r'^\d+\.\s*', '', c).strip() for c in criteria_str.split('\n') if c.strip()]
    for criterion in individual_criteria:
        degree_match = re.search(r'(JD|MD|MBA|PhD|Master\'s|Bachelor\'s|Higher degree)\s*(?:degree)?\s*(?:in\s*([\w\s,]+))?(?:from\s*([\w\s,.()-]+))?', criterion, re.IGNORECASE)
        if degree_match:
            degree_type = degree_match.group(1).replace("’s", "'s")
            field_of_study = degree_match.group(2).strip() if degree_match.group(2) else None
            location = degree_match.group(3).strip() if degree_match.group(3) else None
            regex_parts = [re.escape(degree_type.replace("JD", "J\.?D\.?"))]
            if field_of_study:
                field_of_study = field_of_study.replace(',', '|')
                regex_parts.append(f"({field_of_study})")
            if location:
                location_terms = '|'.join([re.escape(loc.strip()) for loc in location.split('or') if loc.strip()])
                location_terms = location_terms.replace('U.S.', 'U\.S\.|United States|USA').replace('U.K.', 'U\.K\.|United Kingdom|UK')
                regex_parts.append(f"({location_terms})")
            regex_str = '.*'.join(regex_parts)
            mongo_conditions.append({"education": {"$regex": regex_str, "$options": "i"}})
            continue

        years_match = re.search(r'(\d+)(?:\+|\s*or more)?\s*years?\s*(?:of)?\s*(?:experience\s*(?:in\s*([\w\s,]+))|prior work experience\s*(?:in\s*([\w\s,]+)))?', criterion, re.IGNORECASE)
        if years_match:
            years = float(years_match.group(1))
            role = years_match.group(2) or years_match.group(3)
            role = role.strip() if role else None
            mongo_conditions.append({"yearsOfExperience": {"$gte": years}})
            if role:
                role_terms = '|'.join([re.escape(r.strip()) for r in role.split('or') if r.strip()])
                mongo_conditions.append({"experience": {"$regex": role_terms, "$options": "i"}})
            continue

        range_match = re.search(r'(\d+)-(\d+)\s*years?\s*(?:of)?\s*experience\s*(?:in\s*([\w\s,]+))?', criterion, re.IGNORECASE)
        if range_match:
            min_years = float(range_match.group(1))
            max_years = float(range_match.group(2))
            role = range_match.group(3).strip() if range_match.group(3) else None
            mongo_conditions.append({"yearsOfExperience": {"$gte": min_years, "$lte": max_years}})
            if role:
                role_terms = '|'.join([re.escape(r.strip()) for r in role.split('or') if r.strip()])
                mongo_conditions.append({"experience": {"$regex": role_terms, "$options": "i"}})
            continue

        role_match = re.search(r'experience\s*working\s*as\s*(?:a\s*)?([\w\s()]+)', criterion, re.IGNORECASE)
        if role_match:
            role = role_match.group(1).strip()
            mongo_conditions.append({"experience": {"$regex": re.escape(role), "$options": "i"}})
            continue

        location_match = re.search(r'(?:undergraduate studies|graduate of|completed)\s*(?:in|from)\s*([\w\s,.()-]+)', criterion, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
            location_terms = '|'.join([re.escape(loc.strip()) for loc in location.split('or') if loc.strip()])
            location_terms = location_terms.replace('U.S.', 'U\.S\.|United States|USA').replace('U.K.', 'U\.K\.|United Kingdom|UK')
            mongo_conditions.append({"education": {"$regex": location_terms, "$options": "i"}})
            continue

        phd_status_match = re.search(r'PhD\s*\(in progress or completed\)\s*(?:from\s*a\s*([\w\s,]+)\s*program\s*(?:in\s*([\w\s,]+)))?', criterion, re.IGNORECASE)
        if phd_status_match:
            program_type = phd_status_match.group(1).strip() if phd_status_match.group(1) else None
            fields = phd_status_match.group(2).strip() if phd_status_match.group(2) else None
            regex_parts = ["PhD"]
            if fields:
                fields = fields.replace(',', '|')
                regex_parts.append(f"({fields})")
            if program_type:
                regex_parts.append(re.escape(program_type))
            regex_str = '.*'.join(regex_parts)
            mongo_conditions.append({"education": {"$regex": regex_str, "$options": "i"}})
            continue

        phd_time_match = re.search(r'PhD program started within the last (\d+) years', criterion, re.IGNORECASE)
        if phd_time_match:
            mongo_conditions.append({"education": {"$regex": "PhD", "$options": "i"}})
            continue

        print(f"Unrecognized criterion: {criterion}")

    return {"$and": mongo_conditions} if mongo_conditions else {}

def get_fallback_candidates(collection, description, limit_count):
    keywords = re.findall(r'\w+', description.lower())
    useful_keywords = [kw for kw in keywords if len(kw) > 3 and kw not in ('with', 'from', 'experience', 'years')]
    if not useful_keywords:
        return []
    query = {
        "$or": [
            {"education": {"$regex": '|'.join(useful_keywords), "$options": "i"}},
            {"experience": {"$regex": '|'.join(useful_keywords), "$options": "i"}},
            {"rerankSummary": {"$regex": '|'.join(useful_keywords), "$options": "i"}}
        ]
    }
    try:
        projection = {
            "_id": 1,
            "rerankSummary": 1,
            "embedding": 1,
            "education": 1,
            "yearsOfExperience": 1,
            "experience": 1
        }
        cursor = collection.find(query, projection).limit(limit_count)
        results = [doc for doc in cursor]
        print(f"Fallback: Fetched {len(results)} documents using keywords: {useful_keywords[:5]}...")
        return results
    except Exception as e:
        print(f"Error in fallback query: {e}")
        return []

def get_hard_filtered_docs_from_mongo(collection, mongo_query, description, limit_count):
    try:
        projection = {
            "_id": 1,
            "rerankSummary": 1,
            "embedding": 1,
            "education": 1,
            "yearsOfExperience": 1,
            "experience": 1
        }
        cursor = collection.find(mongo_query, projection).limit(limit_count)
        results = [doc for doc in cursor]
        print(f"Fetched {len(results)} documents from MongoDB matching hard criteria (limit={limit_count}).")
        if not results:
            print("No documents matched hard criteria. Attempting fallback query...")
            results = get_fallback_candidates(collection, description, limit_count)
        for doc in results:
            doc['text'] = doc.get('rerankSummary', '')
        return results
    except Exception as e:
        print(f"Error fetching hard-filtered documents from MongoDB: {e}")
        return []

def calculate_l2_distance(vec1, vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1, dtype=np.float32)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2, dtype=np.float32)
    return np.linalg.norm(vec1 - vec2)

def send_evaluation_request(config_path, object_ids, your_email):
    eval_url = "https://mercor-dev--search-eng-interview.modal.run/evaluate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": your_email
    }
    payload = {
        "config_path": config_path,
        "object_ids": object_ids
    }
    print(f"\nSending evaluation for config: {config_path} with {len(object_ids)} IDs: {object_ids}")
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.post(eval_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        eval_result = response.json()
        print(f"Evaluation response for {config_path}:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response JSON: {json.dumps(eval_result, indent=2)}")
        print(f"   Score: {eval_result.get('score', 'N/A')}")
        print(f"   Message: {eval_result.get('message', 'No message')}")
        return eval_result
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error during evaluation for {config_path}: {e.response.status_code} - {e.response.text}")
        return {"score": "N/A", "message": f"HTTP Error: {e.response.status_code} - {e.response.text}"}
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error during evaluation for {config_path}: {e}")
        return {"score": "N/A", "message": f"Connection Error: {e}"}
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error during evaluation for {config_path}: {e}")
        return {"score": "N/A", "message": f"Timeout Error: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred during evaluation for {config_path}: {e}")
        return {"score": "N/A", "message": f"Unexpected Error: {e}"}

def main():
    mongo_uri = "mongodb+srv://candidate:aQ7hHSLV9QqvQutP@hardfiltering.awwim.mongodb.net/"
    db_name = "interview_data"
    collection_name = "linkedin_data_subset"
    VOYAGE_API_KEY = "pa-vNEmoJfc5evP_SSvpxIAj3uFzs9dfppEZkpx-3kOFZy"
    excel_file = r"/content/drive/MyDrive/queries.xlsx"
    YOUR_EMAIL_FOR_EVALUATION = "ramumegavath823@gmail.com"

    MONGO_FETCH_LIMIT = 500
    FINAL_OUTPUT_TOP_K = 10

    if not os.path.exists(excel_file):
        print(f"Excel file not found at: {excel_file}")
        return

    try:
        df = pd.read_excel(excel_file)
        print(f"Successfully loaded Excel file: {excel_file}")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    required_cols = ["Natural Language Description", "Hard Criteria", "Soft Criteria", "Yaml File"]
    if not all(col in df.columns for col in required_cols):
        print(f"One or more required columns missing from Excel: {required_cols}")
        return

    queries_data = []
    for index, row in df.iterrows():
        description = str(row["Natural Language Description"]).strip()
        hard_criteria = str(row["Hard Criteria"]).strip()
        soft_criteria = str(row["Soft Criteria"]).strip()
        yaml_file = str(row["Yaml File"]).strip()
        if description:
            queries_data.append({
                "description": description,
                "hard_criteria": hard_criteria,
                "soft_criteria": soft_criteria,
                "yaml_file": yaml_file
            })

    if not queries_data:
        print("No valid queries found in the Excel file after filtering empty descriptions.")
        return

    collection = connect_to_mongodb(mongo_uri, db_name, collection_name)
    if collection is None:
        return

    print(f"Total documents in MongoDB collection: {collection.count_documents({})}")

    all_query_results_for_submission = {}
    evaluation_results = []  # Store results for table

    for i, query_item in enumerate(queries_data):
        query_text = f"{query_item['description']} {query_item['soft_criteria']}"
        hard_criteria_str = query_item["hard_criteria"]
        yaml_file = query_item["yaml_file"]

        print(f"\n--- Processing Query {i+1} of {len(queries_data)} ---")
        print(f"    Description: '{query_text[:100]}...'")
        print(f"    Hard Criteria: '{hard_criteria_str}'")
        print(f"    YAML File: '{yaml_file}'")

        mongo_hard_query = parse_criteria_string_to_mongo_query(hard_criteria_str)
        print(f"    Generated MongoDB Hard Query: {mongo_hard_query}")

        hard_filtered_candidates = get_hard_filtered_docs_from_mongo(collection, mongo_hard_query, query_item["description"], MONGO_FETCH_LIMIT)

        if not hard_filtered_candidates:
            print(f"No documents found for query '{query_text[:50]}...'. Using PAD_IDs for submission.")
            all_query_results_for_submission[yaml_file] = [f"PAD_ID_{i}" for i in range(FINAL_OUTPUT_TOP_K)]
            evaluation_results.append({"Query": yaml_file, "Score": "N/A", "Status": "No documents found"})
            continue

        query_embedding = embed_query_with_voyage(query_text, VOYAGE_API_KEY)
        if query_embedding is None:
            print(f"Skipping semantic re-ranking for query '{query_text[:50]}...' due to embedding failure.")
            all_query_results_for_submission[yaml_file] = [f"PAD_ID_{i}" for i in range(FINAL_OUTPUT_TOP_K)]
            evaluation_results.append({"Query": yaml_file, "Score": "N/A", "Status": "Embedding failure"})
            continue

        results_with_distances = []
        for doc in hard_filtered_candidates:
            doc_embedding_list = doc.get('embedding')
            if doc_embedding_list and isinstance(doc_embedding_list, list) and len(doc_embedding_list) == EMBEDDING_DIM:
                doc_embedding = np.array(doc_embedding_list, dtype=np.float32)
                distance = calculate_l2_distance(query_embedding, doc_embedding)
                doc_copy = doc.copy()
                doc_copy['distance'] = distance
                results_with_distances.append(doc_copy)

        semantically_ranked_results = sorted(results_with_distances, key=lambda x: x['distance'])
        final_results_for_query = semantically_ranked_results[:FINAL_OUTPUT_TOP_K]
        object_ids_for_submission = [str(doc.get('_id')) for doc in final_results_for_query if doc.get('_id')]

        while len(object_ids_for_submission) < FINAL_OUTPUT_TOP_K:
            object_ids_for_submission.append(f"PAD_ID_{len(object_ids_for_submission)}")

        all_query_results_for_submission[yaml_file] = object_ids_for_submission

        print(f"    Top {len(final_results_for_query)} final results (after hard filtering & semantic re-ranking):")
        for j, result in enumerate(final_results_for_query):
            display_text = result.get('text', 'No text available')
            print(f"    Result {j+1} - ID: {result.get('_id', 'N/A')}, Distance: {result.get('distance', 'N/A'):.4f}, Text: {display_text[:100]}...")
            print(f"      Education: {result.get('education', 'N/A')}")
            print(f"      Years of Experience: {result.get('yearsOfExperience', 'N/A')}")
            print(f"      Experience: {result.get('experience', 'N/A')[:100]}...")

        
        eval_result = send_evaluation_request(yaml_file, object_ids_for_submission, YOUR_EMAIL_FOR_EVALUATION)
        time.sleep(2)
        score = eval_result.get('average_final_score', 'N/A') if eval_result else 'N/A'
       # status = eval_result.get('message', 'No message') if eval_result else 'Evaluation failed'
        evaluation_results.append({"Query": yaml_file, "Score": score})

    print("\n--- Processing complete. Results for each query are stored in 'all_query_results_for_submission'. ---")
    print(f"Submission JSON: {json.dumps(all_query_results_for_submission, indent=2)}")

   
    print("\n--- Evaluation Scores for Public Queries ---")
    scores_df = pd.DataFrame(evaluation_results)
    print(scores_df.to_string(index=False))


    scores_df.to_csv("evaluation_scores.csv", index=False)
    print("\nSaved scores table to 'evaluation_scores.csv'")

if __name__ == "__main__":
    main()