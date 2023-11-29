def test_function() -> int:
    return 4,5

def main(i) -> None:
    a,b = test_function()
    print(a,b)
    if i>0:
        x = 1
    print(x)

if __name__=='__main__':
    main(-1)