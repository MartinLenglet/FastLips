import csv
import numpy as np

def process_csv(input_file, output_file):
    durations = []

    with open(input_file, 'r', encoding='utf-8') as input_csv, open(output_file, 'w', encoding='utf-8', newline='') as output_csv:
        reader = csv.reader(input_csv, delimiter='|', quotechar=None, doublequote=False)
        writer = csv.writer(output_csv, delimiter='|', quotechar=None, doublequote=False)

        current_element = None
        counter = 1

        for i, row in enumerate(reader):
            # Assuming the input CSV has at least 4 columns
            if len(row) >= 4:
                element = row[0]

                if element != current_element:
                    current_element = element
                    counter = 1

                verif_id = f"{element}_{counter}"
                new_element = f"{prefix}{element}_{counter}{suffix}"
                counter += 1

                if not verif_id in excluded_ids:
                    writer.writerow([new_element, row[3]])
                    if i % 2 == 0:
                        start_time = int(row[1])
                        end_time = int(row[2])
                        duration = (end_time - start_time)/1000
                        durations.append(duration)

    return durations

if __name__ == "__main__":
    input_file = "AD_ALN.csv"
    output_file = "AD_ALN_formatted.txt"

    # prefix = "/research/crissp/lengletm/FastSpeech2-master/raw_data/ALL_corpus/AD/"
    # suffix = ".wav"
    # prefix = "/research/crissp/lengletm/FastSpeech2-master/preprocessed_data/ALL_corpus/mel/AD-mel-"
    # suffix = ".npy"
    prefix = ""
    suffix = ""

    excluded_ids = [
        'DIVERS_BOOK_AD_01_0002_91',
        'DIVERS_BOOK_AD_01_0002_92',
        'ESTYLE_DETERMINE_AD_01_0001_73',
        'ESTYLE_DETERMINE_AD_01_0001_74',
        'ESTYLE_DETERMINE_AD_01_0001_77',
        'ESTYLE_DETERMINE_AD_01_0001_78',
        'DIVERS_PARL_PENSIF_AD_02_0005_305',
        'DIVERS_PARL_PENSIF_AD_02_0005_306',
    ]
    
    durations = process_csv(input_file, output_file)
    print(f"Processing completed. Results saved to {output_file}")

    total_duration = sum(durations)
    mean_duration = np.mean(durations)
    std_dev_duration = np.std(durations)

    print(f"Total Duration: {total_duration} seconds")
    print(f"Mean Duration: {mean_duration} seconds")
    print(f"Standard Deviation of Duration: {std_dev_duration} seconds")