def char(x):
    if x < -1:
        return '/'
    if x < -0.2:
        return '-'
    if x > 0.2:
        return '+'
    if x > 1:
        return '#'
    return '.'

def show(ar):
    n = min(80, len(ar))
    chars = [char(ar[i]) for i in range(n)]
    return ''.join(chars)
