import os
import random

def split_txt_file(input_file, output_file_1, output_file_2, percentage):
    with open(input_file, 'r', encoding='utf-8') as input_txt, \
         open(output_file_1 + ".txt", 'w', encoding='utf-8') as output_txt_1, \
         open(output_file_2 + ".txt", 'w', encoding='utf-8') as output_txt_2, \
         open(output_file_1 + "_FS.txt", 'w', encoding='utf-8') as output_txt_1_FS, \
         open(output_file_2 + "_FS.txt", 'w', encoding='utf-8') as output_txt_2_FS:

        lines = input_txt.readlines()
        total_lines = len(lines)
        lines_per_file_1 = int(total_lines * (percentage / 100) / 2)

        # Create a list of pairs of lines
        line_pairs = [(lines[i].strip(), lines[i+1].strip()) for i in range(0, total_lines, 2)]

        # Randomly shuffle the pairs
        random.shuffle(line_pairs)

        for pair in line_pairs[:lines_per_file_1]:
            output_txt_1.write(f"{pair[0]}\n{pair[1]}\n")

            # Split pair[0] and pair[1] into multiple columns
            columns_0 = pair[0].split('|')
            columns_1 = pair[1].split('|')

            output_txt_1_FS.write(f"{columns_0[0]}|AD|{columns_0[1]}|{columns_0[1]}\n{columns_1[0]}|AD|{columns_1[1]}|{columns_0[1]}\n")

        for pair in line_pairs[lines_per_file_1:]:
            output_txt_2.write(f"{pair[0]}\n{pair[1]}\n")

            # Split pair[0] and pair[1] into multiple columns
            columns_0 = pair[0].split('|')
            columns_1 = pair[1].split('|')

            output_txt_2_FS.write(f"{columns_0[0]}|AD|{columns_0[1]}|{columns_0[1]}\n{columns_1[0]}|AD|{columns_1[1]}|{columns_0[1]}\n")

if __name__ == "__main__":
    # Set the seed for randomization
    random.seed(1234)

    input_file = "AD_ALN_formatted.txt"
    output_file_1 = "train_AD_neutral"
    output_file_2 = "test_AD_neutral"
    percentage_for_file_1 = 95  # Adjust the percentage as needed

    split_txt_file(input_file, output_file_1, output_file_2, percentage_for_file_1)
    print(f"Processing completed. Results saved to {output_file_1}, {output_file_2}, {output_file_1}_FS, and {output_file_2}_FS")