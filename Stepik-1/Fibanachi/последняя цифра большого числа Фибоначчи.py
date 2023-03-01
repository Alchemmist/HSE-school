def fib_digit_2(n):
    if n == 0: 
        return 0
    if n == 1: 
        return 1

    before_value, last_value = 0, 1
    for _ in range(1, n):
        before_value = last_value 
        last_value = (before_value + last_value) % 10
    return last_value


def fib_digit(n: int) -> int:
    if n == 0: return 0
    if n == 1: return 1

    before_value, last_value = 0, 1
    for _ in range(1, n):
        before_value = last_value 
        last_value = (before_value + last_value) % 10
    return last_value


def main():
    n = int(input())
    print(fib_digit(n))


if __name__ == "__main__":
    main()
