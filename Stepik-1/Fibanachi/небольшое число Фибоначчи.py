def fib(n):
    fib = [0, 1]
    for _ in range(n):
        a, b = fib[-1], fib[-2] 
        fib.append(a + b)
    return fib[n - 1] + fib[n-2] 


def main():
    n = int(input())
    print(fib(n))


if __name__ == "__main__":
    main()
