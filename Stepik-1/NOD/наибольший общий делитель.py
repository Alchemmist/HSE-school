def gcd(a: int, b: int) -> int:
    if a == 0: return b
    if b == 0: return a
    
    if a > b:
        return gcd(a % b, b)
    elif b > a:
        return gcd(a, b % a)

    return a


def main():
    a, b = map(int, input().split())
    print(gcd(a, b))


if __name__ == "__main__":
    main()
