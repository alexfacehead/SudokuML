import argparse
import re
import os

def extract_ms_per_step(file_name: str):
    extracted_numbers = []
    print("Attempting to open " + file_name)
    with open(file_name, 'r') as file:
        print("file_name")
        for line in file:
            if "ms/step" in line:
                match = re.search(r'(\d+)\s?ms/step', line)
                if match:
                    number = float(match.group(1))
                    extracted_numbers.append(number)
                else:
                    print(f"No match found in line: {line}")

    return extracted_numbers

def calculate_average(numbers: list):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def main():
    print("test")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the base .csv file")
    args = parser.parse_args()
    file_name = args.file
    # Edited code: use the current working directory plus the file path on the 32nd line
    result = extract_ms_per_step(os.path.join(os.getcwd(), "debug_output", file_name))
    average = calculate_average(result)
    print(f"Average time taken (ms): {average}")

main()