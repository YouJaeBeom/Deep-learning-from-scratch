# AND gate 퍼셉트론으로 구현
def AND(x1, x2):
    w1, w2, thera = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= thera:
        return 0
    elif tmp > thera:
        return 1

# OR gate 퍼셉트론으로 구현
def OR(x1, x2):
    w1, w2, thera = 1, 1, 1
    tmp = x1 * w1 + x2 * w2
    if tmp < thera:
        return 0
    elif tmp >= thera:
        return 1

print(" AND gate")
print("0,0 =",AND(0,0))
print("0,1 =",AND(0,1))
print("1,0 =",AND(1,0))
print("1,1 =",AND(1,1))

print(" OR gate")
print("0,0 =",OR(0,0))
print("0,1 =",OR(0,1))
print("1,0 =",OR(1,0))
print("1,1 =",OR(1,1))