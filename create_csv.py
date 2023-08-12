import csv
import math

def is_prime(n):
    if n <= 1:
        return 0
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return 0
    return 1


def digit_sum(n):
    return sum(int(digit) for digit in str(n))


with open('Data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Number', 'Modulo 2', 'Modulo 3', 'Modulo 5', 'Modulo 7', 'Modulo 11', 'Modulo 13',
                     'Sum of digits', 'Prime'])
    for num in range(2, 1000000 + 1):
        writer.writerow([num, num % 2, num % 3, num % 5, num % 7, num % 11, num % 13, digit_sum(num), is_prime(num)])

