from functools import cache

@cache
def getSierpinskiPoints(n):
    if n == 0:
        return [0j, 0.5+1j, 1+0j]
    s = getSierpinskiPoints(n-1)
    idx = len(s)//2 + (n-1)//2
    return s + [p + s[idx] for p in s[1:]]\
             + [p + s[-1] for p in s[1:] if p != s[idx]]
