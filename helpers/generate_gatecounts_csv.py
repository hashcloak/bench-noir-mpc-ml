import csv
import os
import argparse

# Input file remains hardcoded
input_file = "output/benchmarks.txt"

# Parse arguments for output file
parser = argparse.ArgumentParser(
    description="Parse benchmark results and save to a CSV file."
)
parser.add_argument("--output", required=True, help="Path to the output CSV file")
args = parser.parse_args()
output_file = args.output

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize storage for parsed data
results = []

# Read the benchmarking file line by line
with open(input_file, "r") as file:
    lines = file.readlines()

# Iterate through the lines to find entries
i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("epochs="):
        entry = {}

        # Extract epochs and samples_train
        parts = line.split(", ")
        entry["epochs"] = parts[0].split("=")[1]
        entry["samples_train"] = parts[1].split("=")[1]

        # Extract acir_opcodes and circuit_size
        try:
            acir_opcodes_line = lines[i + 3].strip()
            circuit_size_line = lines[i + 4].strip()

            # Extract numerical values directly
            if "acir_opcodes" in acir_opcodes_line:
                entry["acir_opcodes"] = (
                    acir_opcodes_line.split(":")[1].strip().strip(",")
                )
            else:
                entry["acir_opcodes"] = "N/A"

            if "circuit_size" in circuit_size_line:
                entry["circuit_size"] = (
                    circuit_size_line.split(":")[1].strip().strip(",")
                )
            else:
                entry["circuit_size"] = "N/A"
        except IndexError:
            # If parsing fails due to insufficient lines, set values to N/A
            entry["acir_opcodes"] = "N/A"
            entry["circuit_size"] = "N/A"

        # Add the entry to results
        results.append(entry)

        # Move to the next entry (skip processed lines)
        i += 4
    if line.startswith("real"):
        parts = line.split("\t")
        entry["proving_time"] = parts[1]
        i += 1
    else:
        i += 1  # Move to the next line if not `epochs=`

# Write the results to a CSV file
with open(output_file, "w", newline="") as csvfile:
    fieldnames = [
        "epochs",
        "samples_train",
        "acir_opcodes",
        "circuit_size",
        "proving_time",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header and rows
    writer.writeheader()
    writer.writerows(results)

print(f"Benchmarking results saved to {output_file}")
