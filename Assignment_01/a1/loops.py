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


# Function to return the sum of the first n positive odd numbers
# n: the number of initial odd numbers to sum
# Returns: sum as an integer
def sum_odd(n):
    oddsum = 0
    cur = 1

    for _ in range(n):
        oddsum += cur
        cur += 2

    return oddsum
    pass

# Function to calculate the sum of the first N Fibonacci numbers
# n: the number of initial Fibonacci numbers to sum
# Returns: sum as an integer
def sum_fib(n):
    first = 0
    second = 1
    if n <= 1:
        return 0
    fib_sum = first + second
    if n == 2:
        return fib_sum
    for i in range(3, n+1):
        next = first + second
        fib_sum += next
        first = second
        second = next

    return fib_sum
    pass


# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. If you want, you may run the code from the terminal as:
# python loops.py
# It should produce the following output (with correct solution):
# 	    $ python loops.py
#       The sum of the first 5 positive odd numbers is: 25
#       The sum of the first 5 fibonacci numbers is: 7

def main():
    # Call the function to calculate sum
    osum = sum_odd(5) 

    # Print it out
    print(f'The sum of the first 5 positive odd numbers is: {osum}')

    # Call the function to calculate sum of fibonacci numbers
    fsum = sum_fib(5)
    print(f'The sum of the first 5 fibonacci numbers is: {fsum}')

################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()