import numpy as np

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

# AND gate 퍼셉트론으로 구현 + weight
def AND_w(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    thera =-0.7
    tmp=np.sum(x*w)+thera
    if tmp <= 0:
        return 0
    else:
        return 1

# OR gate 퍼셉트론으로 구현 + weight
def OR_w(x1, x2):
    x=np.array([x1,x2])
    w=np.array([1,1])
    thera =-0.7
    tmp=np.sum(x*w)+thera
    if tmp <= 0:
        return 0
    else:
        return 1
# NAND gate 퍼셉트론으로 구현 +weight
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    thera=0.9
    tmp=np.sum(x*w)+thera
    if tmp <=0:
        return 0
    else:
        return 1

#  한계 XOR gate 구현하기
# 비선형이므로 단순 퍼셉트론으로는 구현이 불가함
# 다층 퍼셉트론으로 구현가능
# XOR NAND OR AND 합쳐서
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y


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

print(" AND gate + weight")
print("0,0 =",AND_w(0,0))
print("0,1 =",AND_w(0,1))
print("1,0 =",AND_w(1,0))
print("1,1 =",AND_w(1,1))

print(" OR gate + weight")
print("0,0 =",OR_w(0,0))
print("0,1 =",OR_w(0,1))
print("1,0 =",OR_w(1,0))
print("1,1 =",OR_w(1,1))

print(" NAND gate + weight")
print("0,0 =",NAND(0,0))
print("0,1 =",NAND(0,1))
print("1,0 =",NAND(1,0))
print("1,1 =",NAND(1,1))

print(" XOR gate + weight")
print("0,0 =",XOR(0,0))
print("0,1 =",XOR(0,1))
print("1,0 =",XOR(1,0))
print("1,1 =",XOR(1,1))