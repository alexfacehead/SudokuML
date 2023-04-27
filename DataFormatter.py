import random

input_file_path = "/home/dev/SudokuML/resources/sudoku_very_small.csv"
output_file_path = "/home/dev/SudokuML/resources/sudoku_mini_easy.csv"

start_line = 0
end_line = 1000
total_lines = end_line - start_line

def insert_zeros(s, num_zeros):
    # Find the positions that already contain zeros in the original string
    existing_zero_positions = {i for i, char in enumerate(s) if char == '0'}
    
    # Generate a set of unique random positions to insert zeros, excluding existing zero positions
    zero_positions = set()
    while len(zero_positions) < num_zeros:
        zero_pos = random.randint(0, len(s)-1)
        if zero_pos not in existing_zero_positions:
            zero_positions.add(zero_pos)
    
    # Convert the string to a list to allow for easy modification
    s_list = list(s)
    
    # Insert zeros at the generated positions
    for zero_pos in zero_positions:
        s_list[zero_pos] = '0'
    
    # Convert the list back to a string
    s = ''.join(s_list)
    return s

with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

with open(output_file_path, "w") as output_file:
    for i in range(start_line, end_line):
        line = lines[i].strip().split(',')
        first_value = line[0]
        second_value = line[1]

        # Determine the number of zeros to insert based on the current line index
        line_index = i - start_line
        if line_index < total_lines // 4:
            num_zeros = 8
        elif line_index < total_lines // 2:
            num_zeros = 8
        elif line_index < (3 * total_lines) // 4:
            num_zeros = 12
        else:
            num_zeros = 12
        modified_first_value = insert_zeros(second_value[:], num_zeros)
        output_line = modified_first_value + ',' + second_value + '\n'
        output_file.write(output_line)