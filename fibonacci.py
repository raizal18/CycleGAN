def cache(func):
    memory = {}
    def wrapper(n):
        if n in memory:
            return memory[n]
        result = func(n)
        memory[n] = result
        return result
    return wrapper

@cache
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
