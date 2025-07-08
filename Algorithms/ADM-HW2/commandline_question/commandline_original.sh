#!/bin/bash

# Requirements:

# Download jq package. 
# in Linux: 
	# $ sudo apt-get update 
	# $ sudo apt-get install jq
# in mac:
	# $ brew install wget
	# $ wget https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
	# $ chmod u+x install.sh
	# $ ./install.sh
	# $ brew install jq

# run this file with: $ bash commandline_LLM.sh

# Store the code in an array of objects named 'data'
data=("$(cat series.json | jq '.id |= tonumber | .works |= map(.books_count |= tonumber) | {id, title, total_books_count: (.works | map(.books_count) | add)}')")

# Sort the 'data' array based on 'total_books_count' property in descending order
sorted_data=$(echo "${data[@]}" | jq -s 'sort_by(.total_books_count) | reverse')

# Return the first 5 elements in a readable format
echo "$sorted_data" | jq '.[:5]'