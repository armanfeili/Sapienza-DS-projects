import os
import torch
import pandas as pd
from flask import Blueprint, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
from models import db, PersonalityRecord
import json
from sqlalchemy import inspect

# Create a blueprint
api = Blueprint('api', __name__)

# Load the tokenizer and model for MBTI classification
tokenizer_mbti = AutoTokenizer.from_pretrained("JanSt/albert-base-v2_mbti-classification")
model_mbti = AutoModelForSequenceClassification.from_pretrained("JanSt/albert-base-v2_mbti-classification")

# Load the tokenizer and model for Big Five personality traits
model_big_five_personality_traits = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality", num_labels=5)
tokenizer_big_five_personality_traits = BertTokenizer.from_pretrained('Minej/bert-base-personality', do_lower_case=True)

# Determine the absolute path to the datasets
base_dir = os.path.abspath(os.path.dirname(__file__))
dataset_mbti = os.path.join(base_dir, "datasets/MBTI-career.xlsx")
dataset_big_five = os.path.join(base_dir, "datasets/5-big-traits-career.xlsx")

def classify_mbti(text):
    try:
        # Tokenize the input text
        inputs = tokenizer_mbti(text, return_tensors="pt", truncation=True, max_length=512)

        # Perform the classification
        with torch.no_grad():  # Disables gradient calculation to save memory and speed up
            outputs = model_mbti(**inputs)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = ["INTP", "ISTP", "ENTP", "ESTP", "INFP", "ISFP", "INFJ", "ESFP",
                  "INTJ", "ISTJ", "ENTJ", "ESTJ", "ENFP", "ISFJ", "ENFJ", "ESFJ"]

        # Convert probabilities to percentages and link them with labels
        percentages = (probabilities.squeeze() * 100).tolist()  # Convert to percentages
        label_percentages = {label: round(float(percent) / 100, 3) for label, percent in zip(labels, percentages)}

        # Sort by highest probability
        sorted_label_percentages = dict(sorted(label_percentages.items(), key=lambda item: item[1], reverse=True))

        return sorted_label_percentages

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

def classify_big_five_personality_traits(model_input: str) -> dict:
    '''
    Performs personality prediction on the given input text

    Args:
        model_input (str): The text conversation

    Returns:
        dict: A dictionary where keys are speaker labels and values are their personality predictions
    '''

    # Tokenize input
    inputs = tokenizer_big_five_personality_traits(model_input, truncation=True, padding=True, return_tensors="pt")

    # Get model outputs
    with torch.no_grad():
        outputs = model_big_five_personality_traits(**inputs)

    # Get logits
    logits = outputs.logits.squeeze().detach().numpy()

    # Apply softmax to get probabilities
    probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()

    # Map the predictions to labels
    id2label = {
        0: "Extraversion",
        1: "Neuroticism",
        2: "Agreeableness",
        3: "Conscientiousness",
        4: "Openness"
    }

    result = {id2label[i]: round(float(probabilities[i]), 3) for i in range(len(id2label))}

    # Order the result based on their values in descending order
    ordered_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

    return ordered_result

def get_sorted_career_data(result_MBTI, results_big_five_personality_traits):
    # Read the Excel files into DataFrames
    df_mbti = pd.read_excel(dataset_mbti, engine='openpyxl')
    df_big_five = pd.read_excel(dataset_big_five, engine='openpyxl')

    # Extract the top 2 MBTI types and Big Five traits
    top_mbtis = list(result_MBTI.keys())[:2]
    top_big_five_traits = list(results_big_five_personality_traits.keys())[:2]

    # Initialize result object
    result_object = {
        "MBTI": [],
        "big_5": []
    }

    # Query the MBTI DataFrame for each top MBTI type and sort by 'Average Salary (USD)'
    for mbti in top_mbtis:
        query_mbti = df_mbti[df_mbti['Type'] == mbti][['Type', 'Career', 'Average Salary (USD)']]
        query_mbti_sorted = query_mbti.sort_values(by='Average Salary (USD)', ascending=False)
        result_object["MBTI"].append({mbti: query_mbti_sorted.to_dict(orient='records')})

    # Query the Big Five DataFrame for each top Big Five trait and sort by 'Average Salary (USD)'
    for trait in top_big_five_traits:
        query_big_five = df_big_five[df_big_five['Type'] == trait][['Type', 'Career', 'Average Salary (USD)']]
        query_big_five_sorted = query_big_five.sort_values(by='Average Salary (USD)', ascending=False)
        result_object["big_5"].append({trait: query_big_five_sorted.to_dict(orient='records')})

    return result_object


@api.route('/classify_mbti', methods=['POST'])
def classify_mbti_route():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'message': 'No text provided'}), 400
        
        result = classify_mbti(text)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'message': 'Error processing request', 'error': str(e)}), 500

@api.route('/classify_big_five', methods=['POST'])
def classify_big_five_route():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'message': 'No text provided'}), 400
        
        result = classify_big_five_personality_traits(text)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({'message': 'Error processing request', 'error': str(e)}), 500

@api.route('/get_career_recommendations', methods=['POST'])
def get_career_recommendations_route():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'message': 'No text provided'}), 400

        result_mbti = classify_mbti(text)
        result_big_five = classify_big_five_personality_traits(text)

        career_data = get_sorted_career_data(result_mbti, result_big_five)

        # Check if the personality_records table exists
        inspector = inspect(db.engine)
        if not inspector.has_table('personality_records'):
            return jsonify({'message': 'The table "personality_records" does not exist in the database'}), 500

        # Create a new PersonalityRecord
        new_record = PersonalityRecord(
            text=text,
            result_mbti=json.dumps(result_mbti),
            result_big_five=json.dumps(result_big_five),
            career_data=json.dumps(career_data)
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify(career_data), 200

    except Exception as e:
        print(f"An error occurred: {e}")  # Print the error to the logs
        return jsonify({'message': 'Error processing request', 'error': str(e)}), 500
