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


def is_fibonacci(n):
    phi = 0.5 + 0.5 * math.sqrt(5.0)
    a = phi * n
    return 1 if n == 0 or abs(round(a) - a) < 1.0 / n else 0


with open('Data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Number', 'Modulo 3', 'Modulo 5', 'Modulo 7', 'Sum of digits', 'Prime'])
    for num in range(3, 1000000 + 1, 2):
        if not str(num)[-1] == '5':
            writer.writerow([num, num % 3, num % 5, num % 7, digit_sum(num), is_prime(num)])

