#!/bin/bash

# OpenAI API key

# Prompt to send to ChatGPT
TEXT='{
  "text": "Retrieve data from series.json file, convert data types, sort by total_books_count in descending order, and return the top 5 elements.
  Steps:
  1. Install jq with 'brew install jq' to read and process the series.json file.
  2. Convert the id and books_count fields to integers.
  3. Map the works array to sum the books_count values.
  4. Sort the resulting data by total_books_count in descending order.
  5. Return the first 5 elements of the sorted data."
}'

# Send a POST request to the OpenAI API
data_report=$(curl -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" -d "$TEXT" "https://api.openai.com/v1/engines/davinci-codex/completions")

# Extract the response from the API
response=$(echo "$data_report" | jq -r '.choices[0].text')

echo "$response"
