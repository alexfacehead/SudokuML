import argparse
import random


"""
python3 DataFormatter.py --base_file /home/dev/sudoku/resources/sudoku_large.csv \
                         --output_file /home/dev/SudokuML/resources/sudoku_mini_easy.csv \
                         --num_lines 100 \
                         --difficulty 8
"""
def insert_zeros(s, num_zeros):
    existing_zero_positions = {i for i, char in enumerate(s) if char == '0'}

    zero_positions = set()
    while len(zero_positions) < num_zeros:
        zero_pos = random.randint(0, len(s)-1)
        if zero_pos not in existing_zero_positions:
            zero_positions.add(zero_pos)

    s_list = list(s)

    for zero_pos in zero_positions:
        s_list[zero_pos] = '0'

    s = ''.join(s_list)
    return s

def main(base_file, output_file, num_lines, difficulty):
    with open(base_file, "r") as input_file:
        lines = input_file.readlines()

    with open(output_file, "w") as output_file:
        for i in range(num_lines):
            line = lines[i].strip().split(',')
            first_value = line[0]
            second_value = line[1]

            line_index = i
            if line_index < num_lines // 4:
                num_zeros = difficulty
            elif line_index < num_lines // 2:
                num_zeros = difficulty
            elif line_index < (3 * num_lines) // 4:
                num_zeros = difficulty + 4
            else:
                num_zeros = difficulty + 4

            modified_first_value = insert_zeros(second_value[:], num_zeros)
            output_line = modified_first_value + ',' + second_value + '\n'
            output_file.write(output_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudoku Reinforcement Learning Task")
    parser.add_argument("--base_file", type=str, required=True, help="Path to the base .csv file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .csv file")
    parser.add_argument("--num_lines", type=int, required=True, help="Number of lines to process (between 1 and 1 million)")
    parser.add_argument("--difficulty", type=int, required=True, help="Difficulty level (integer)")

    args = parser.parse_args()

    main(args.base_file, args.output_file, args.num_lines, args.difficulty)