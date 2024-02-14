def extract_numbers(input_string):
    import re
    """
    Extracts all the numbers from a given string.

    Args:
    input_string (str): The string from which numbers need to be extracted.

    Returns:
    list: A list of numbers extracted from the string.
    """
    # Using regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', input_string)

    # Converting each extracted number from string to integer
    numbers = [int(num) for num in numbers][0]

    return numbers

def extract_scores(string):
    raw_data = string.split(':')[1:]
    output = {}

    output['score']      =    extract_numbers(raw_data[0])
    output['innovation'] =    extract_numbers(raw_data[1])
    output['newness']    =    extract_numbers(raw_data[2])
    output['potential']  =    extract_numbers(raw_data[3])
    output['clarity']    =    extract_numbers(raw_data[4])
    output['relevance']  =    extract_numbers(raw_data[5])

    return output