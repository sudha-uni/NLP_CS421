# CS421: Natural Language Processing
# University of Illinois at Chicago
# Spring 2025
# Assignment 1
#
# Do not rename/delete any functions or global variables provided in this template. Write your implementation
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that test code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment, you will need to obtain approval from the course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================

# Function to calculate the squared list
# num_list: a list containing numerical values
# Returns: a list containing the squared values
def sq_list(num_list):
    squared_list = []
    for i in num_list:
        squared_list.append(i**2)
    return squared_list
    pass

# Function to add two lists
# list_a: a list of numerical values
# list_b: a list of numerical values
# Returns: a list containing the sum of corresponding values from list_a and list_b
def add_lists(list_a, list_b):
    added_list = []
    if len(list_a) != len(list_b):
        raise ValueError("Both lists must have the same length.")
    for i in range(len(list_a)):
        added_list.append(list_a[i] + list_b[i])
    return added_list
    pass

# Function to process the list by adding square of each element to itself
# num_list: a list containing numerical values
# Returns: a list containing the processed values
def process_list(num_list):
    processed_list = add_lists(num_list, sq_list(num_list))
    return processed_list
    pass

# Function to transform the list by adding each element with it's next one in circular fashion
# num_list: a list containing numerical values
# Returns: a list containing the transformed values
def transform_list(num_list):
    transformed_list = num_list[:]

    for i in range(len(num_list) - 1):
        transformed_list[i] += transformed_list[i + 1]

    transformed_list[-1] += transformed_list[0]

    return transformed_list
    pass

# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. If you want, you may run the code from the terminal as:
# python lists.py
# It should produce the following output (with correct solution):
#       $ python lists.py
#       The squared list is: [4, 9, 36]
#       The sum of lists is: [2, 6, 9]
#       The processed list is: [6, 20, 12]
#       The transformed list is: [7, 8, 10]

def main():
    # Test sq_list
    squared_list = sq_list([2, 3, 6])
    print(f'The squared list is: {squared_list}')

    # Test add_lists
    list_sum = add_lists([1, 5, 3], [1, 1, 6])
    print(f'The sum of lists is: {list_sum}')

    # Test process_list
    proc_list = process_list([2, 4, 3])
    print(f'The processed list is: {proc_list}')

    # Test transform_list
    transformed_list = transform_list([2, 5, 3])
    print(f'The transformed list is: {transformed_list}')

################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()
