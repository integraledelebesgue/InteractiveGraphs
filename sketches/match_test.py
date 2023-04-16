def foo(arr):
    match arr:
        case None:
            arr = []
        case []:
            pass
        case _:
            arr.clear()

    arr.extend(range(10))
    print(arr)


a = []
foo(a)
print(a)

foo(None)

b = [5, 6, 7]
foo(b)
print(b)
