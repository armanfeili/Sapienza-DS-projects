#!/bin/bash

#-----------------  Q-1  --------------------

echo "Q-1 Answer: "

file='vodclickstream_uk_movies_03.csv'

# we use 'awk' as a text-processing tool.
# -F is the field-separator. we separate the fields by ','
# we then, print the 4th column of each row
# The output of the awk which are titles, should pipe to 'sort', using '|' operator.
# uniq -c counts the occurrences of each unique title
# The output of 'uniq -c' (which includes the count and the title) should be piped to 'sort -nr' which sorts numerically (by the count) and reverse the order
# '| head -1:' returns the first line of the result.

# Extracting the fourth column ('title') and counting occurrences
most_occurred_title=$(awk -F ',' '{print $4}' "$file" | sort | uniq -c | sort -nr | head -1)

echo "Most occurred title and its count:"
echo "$most_occurred_title"


#-----------------  Q-2  --------------------

echo "Q-2 Answer: "

# split($2, date, /[:-]/): Splits the second column ($2) of the current line into an array named date, using /[:-]/ as the separator.
# Then we convert the date and time components to seconds and sum up all these values to get the total time in seconds.
# diff += time - prev: Calculates the difference between consecutive rows in seconds and accumulates it in the diff variable.
# count++: counts the number of rows.
# prev = time: Updates the previous time variable for the next iteration.
# Once all codes executed before END, compiler runs the code after END.

awk -F ',' 'NR > 1 {
    split($2, date, /[:-]/); 
    time = date[1]*365*24*60*60 + date[2]*30*24*60*60 + date[3]*24*60*60 + date[4]*60*60 + date[5]*60 + date[6]; 
    diff += time - prev; 
    count++; 
    prev = time
} 
END {
    avg_diff = diff/count; 
    hours = int(avg_diff / 3600); 
    minutes = int((avg_diff % 3600) / 60); 
    seconds = int(avg_diff % 60); 
    print "Average difference:", diff/count;
    print "Average difference:", hours " hours, " minutes " minutes, " seconds " seconds";
}' vodclickstream_uk_movies_03.csv


#-----------------  Q-3  --------------------

echo "Q-3 Answer: "

# Read the CSV file
# awk -F',' means that he input fields in the file are separated by commas 
# {sums[$NF]+=$3} creates an array called sums. it sum up the durations ($3) for each unique user_id. ($NF represents the last column). 
# After END, compiler iterates through the sums array and prints each user_id and its total duration.

# Read the CSV file and use awk to sum up durations for each user_id
result=$(awk -F ',' '{sums[$NF]+=$3} END {for (id in sums) print id, sums[id]}' vodclickstream_uk_movies_03.csv)

# Find the user_id with the maximum total duration
# sort -nrk2 sorts the output numerically (-n) based on the second column (-k2) in reverse order (-r)
# head -n1 selects only the first line
max_duration=$(echo "$result" | sort -nrk2 | head -n1)

# Extract user_id and total duration
max_user_id=$(echo "$max_duration" | awk '{print $1}')
total_seconds=$(echo "$max_duration" | awk '{print $2}')

# Convert total duration to hours, minutes, and seconds
hours=$((total_seconds / 3600))
minutes=$(( (total_seconds % 3600) / 60 ))
seconds=$((total_seconds % 60))

echo "User_id with the most total duration: $max_user_id"
echo "Total duration: $hours hours, $minutes minutes, $seconds seconds"


