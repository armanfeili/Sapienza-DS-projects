#!/bin/bash

# Create an empty merged file
touch merged_courses.tsv

for ((i = 1; i <= 6000; i++)); do
  folder="HTML_folders/page_${i}"
  file="html_${i}.html.tsv"

  if [ $i -eq 1 ]; then
    # For the first file, copy the whole content
    cat "${folder}/${file}" >> merged_courses.tsv
  else
    # For files 2 to 6000, omit the first row
    tail -n +2 "${folder}/${file}" >> merged_courses.tsv
  fi
done

printf "The merged_courses.tsv file is generated. \n"

# Question-1:

printf "# Question-1: \n"

# Read 'merged_courses.tsv' file and generate counts for each country
country_count=$(awk -F'\t' 'NR>1 { countries[$11]++ } END { for (country in countries) print "{\"country\": \"" country "\", \"counts\": " countries[country] "}" }' merged_courses.tsv)

# Sort the country count based on counts in descending order
sorted_country_count=$(echo "$country_count" | jq -s 'sort_by(-.counts)')

echo "$sorted_country_count" | jq '.[:5]'

# Extract the value of "country" from the first cell and store it in most_frequent_country
most_frequent_country=$(echo "$sorted_country_count" | jq -r '.[0].country')

echo "Most frequent country: $most_frequent_country"

# Loop over 'merged_courses.tsv' file and filter rows by most frequent country
city_list=$(awk -F'\t' -v most_frequent_country="$most_frequent_country" 'NR>1 && $11 == most_frequent_country {
    cities[$10]++;
}
END {
    for (city in cities) {
        printf "{\"country\": \"%s\", \"city\": \"%s\", \"city_occurrence\": %d}\n", most_frequent_country, city, cities[city];
    }
}' merged_courses.tsv | jq -s '.')

# Sort the city_list based on city_occurrence
sorted_city_list=$(echo "$city_list" | jq 'sort_by(-.city_occurrence)')

echo "$sorted_city_list" | jq '.[:5]'

# Extract the maximum city_occurrence value
max_occurrence=$(echo "$sorted_city_list" | jq '[.[] | .city_occurrence] | max')

# Find cities with the maximum city_occurrence
max_cities=$(echo "$sorted_city_list" | jq --arg max_occurrence "$max_occurrence" "[.[] | select(.city_occurrence == $max_occurrence) | .city]")

printf "\n The most Master's Degrees are in the following cities: "
echo "$max_cities"

# Question-2:

printf "# Question-2: \n"

# Initialize an empty array to store 'part_time' rows
part_time=()

# Read 'merged_courses.tsv' line by line
while IFS=$'\t' read -r col1 col2 col3 isItFullTime col5; do
    # Check if the value in the 'isItFullTime' column is 'Part time'
    if [ "$isItFullTime" = "Part time" ]; then
        # If true, add the entire row to the 'part_time' array
        part_time+=("$col1" "$col2" "$col3" "$isItFullTime" "$col5")
    fi
done < merged_courses.tsv

# Print the length of the 'part_time' array
echo "Number of colleges offering Part-Time education is: ${#part_time[@]}"

# Question-3:

printf "# Question-3: \n"

# Initialize an empty array to store rows containing 'Engineer' in 'courseName'
contain_engineer=()

# Read 'merged_courses.tsv' line by line
while IFS=$'\t' read -r courseName col2 col3 col4 col5; do
    # Check if 'courseName' column contains 'Engineer'
    if [[ "$courseName" == *"Engineer"* ]]; then
        # If true, add the entire row to 'contain_engineer' array
        contain_engineer+=("$courseName" "$col2" "$col3" "$col4" "$col5")
    fi
done < merged_courses.tsv

# Print the length of the 'contain_engineer' array
echo "Length of 'contain_engineer' list: ${#contain_engineer[@]}"

# Calculate the length of the 'contain_engineer' array
length=${#contain_engineer[@]}

# Perform the calculation using bc for floating-point arithmetic
result=$(echo "scale=2; $length / 6000 * 100" | bc)

echo "The percentage of courses in Engineering is: $result%"
