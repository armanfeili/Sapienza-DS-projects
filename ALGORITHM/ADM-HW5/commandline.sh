#!/bin/bash

echo "#-----------------  Q-1  --------------------:"

echo "Nodes with High Degrees: "
# This line returns only top 10 selected nodes from sorted degree counts.
grep -o '<edge.*source="[^"]*"' Citation_Graph.graphml | sed 's/.*source="\([^"]*\)"/\1/' | sort | uniq -c | awk '{print $2, $1}' | sort -k2 -nr | head -n 10


echo "#-----------------  Q-2  --------------------:"

# Extract edges and source nodes, count occurrences of each node
node_degrees=$(grep -o '<edge.*source="[^"]*"' Citation_Graph.graphml | sed 's/.*source="\([^"]*\)"/\1/' | sort | uniq -c | awk '{print $1}')

# Calculate total degrees
total_degrees=$(echo "$node_degrees" | awk '{sum+=$1} END {print sum}')

# Count the number of nodes
total_nodes=$(echo "$node_degrees" | wc -l)

# Calculate average degree
average_degree=$(echo "scale=2; $total_degrees / $total_nodes" | bc)

echo "Total degrees: $total_degrees"
echo "Total nodes: $total_nodes"
echo "Average degree: $average_degree"

echo "Nodes with Low Degrees: "
grep -o '<edge.*source="[^"]*"' Citation_Graph.graphml | sed 's/.*source="\([^"]*\)"/\1/' | sort | uniq -c | awk '{if ($1 <= 5) print $2, $1}' | sort -k2 -rn


echo "#-----------------  Q-3  --------------------:"

python3 -c '
import networkx as nx

# Define your graph or load it from a file
# Example: Load graph from a GraphML file
G = nx.read_graphml("Citation_Graph.graphml")

# Calculate the sum of shortest path lengths between all pairs of nodes
total_length = sum(len(path) - 1 for source in G for target, path in nx.bfs_edges(G, source=source))

# Calculate the total number of paths (excluding self-paths)
total_paths = sum(1 for source in G for _ in nx.bfs_edges(G, source=source))

# Calculate the average shortest path length
average_length = total_length / total_paths if total_paths > 0 else 0

print(f"Average shortest path length: {average_length}")
'