import os
import re

PATH = "C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-K Sample"


def get_index(raw_text, text_positions, is_open = True):
    if is_open:
        response_text = f'Is this the true opening label?:\n Nothing: Proceed 1\n 0: Yes\n 1: Go back 1\n 5: Proceed 5\n'
    else:
        response_text = f'Is this the true closing label?:\n Nothing: Proceed 1\n 0: Yes\n 1: Go back 1\n 5: Proceed 5\n'

    i = 0
    while i < len(text_positions):
        position = text_positions[i]
        print('\n' * 100)
        print(f'{raw_text[max(0, position - 300):position]}$%$%$%%$%${raw_text[position:position + 4]}$%$%$%%$%${raw_text[position + 4:min(len(raw_text), position + 300)]}')
        response = input(response_text)

        if response == '':
            i += 1
        elif response == '5':
            i += 5
        elif response == '1':
            i -= 1
        elif response == '0':
            if input('Are you sure? Yes: 0') == '0':
                return i

    return None


if __name__ == "__main__":
    with open('supervised.csv', mode='a', encoding='utf-8') as file:
        _, _, file_names = next(
            os.walk(PATH))

        for file_name in file_names:
            file_path = f'{PATH}\\{file_name}'
            file_text = open(file_path).read()
            indices = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
            opening_index = get_index(file_text, indices, is_open=True)
            closing_index = get_index(file_text, indices, is_open=False)
            file.write(f'{file_name},{opening_index},{closing_index}\n')

    file.close()