import os
import re
import time

# MISSTATED FILINGS
# 1. https://www.sec.gov/Archives/edgar/data/0001025771/000114420406012804/v039306_10-k.txt
#   20060331_10-K_edgar_data_1025771_0001144204-06-012804_1.txt



PATH = "C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\Data\\10-K Sample"
REGEX_10K = r"Item[\s]+?7\.([\s\S]*?)((Item[\s]+?7A\.)|(Item[\s]+?8\.))"

# 1. Item[\s]+?(%\$%)?7\.([\s\S]*?)Item[\\n\s]+?(%\$%)?7A\.
# 2. Item[\s]+?7\.([\s\S]*?)((Item[\s]+?7A\.)|(Item[\s]+?8\.))
# Original: r"Item[\\n\s]?(%\$%)?7\.([\s\S]*?)Item[\\n\s]?(%\$%)?7A\."
REGEX_10Q = r"Item[\\n\s]?(%\$%)?2\.([\s\S]*?)Item[\\n\s]?(%\$%)?3\."

if __name__ == "__main__":
    _, _, file_names = next(
        os.walk(
            f'C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-K Sample'))


    for file_name in file_names:
        file_path = f'{PATH}\\{file_name}'
        file_text = open(file_path).read()
        match = re.findall(REGEX_10K, file_text, re.IGNORECASE)

        try:
            mda = match[-1][0]
            os.startfile(file_path)
            output_file = open('Text Extraction\\output.txt', 'w')
            output_file.write(mda)
            os.startfile('Text Extraction\\output.txt')
            output_file.close()
        except Exception as e:
            print(e, file_name)

        input('Continue?: 1 or 0')