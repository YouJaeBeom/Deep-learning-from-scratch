# 덧셈 오차역전파법
class MulLayer:
    def __init__(self):
        self.x=None
        self.y =None

    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y

        return out

    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x

        return dx,dy

apple = 100
apple_num=2
tax=1.1

mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()

apple_price=mul_apple_layer.forward(apple,apple_num)
total_price=mul_tax_layer.forward(apple_price,tax)

print("총 가격은 ", total_price)

## 역전파 example

dprice=1
dapple_price,dtax=mul_tax_layer.backward(dprice)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)

print("미분한 apple갯수와 가격",dapple,dapple_num,dtax)

## 덧셈계층
class AddLayer:
    def __init__(self):
        self.x=None
        self.y=None

    def forward(self,x,y):
        out = x+y

        return out

    def backward(self,dout):
        dx=dout*1
        dy=dout*1

        return dx,dy

## 여러개 계산하는 example

apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1

mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()

#순전파
apple_price=mul_apple_layer.forward(apple,apple_num)
orange_price=mul_orange_layer.forward(orange,orange_num)
all_price=add_apple_orange_layer.forward(apple_price,orange_price)
totoal_price=mul_tax_layer.forward(all_price,tax)

print("totoal_price",totoal_price)

#역전파
dprice=1
dall_price=mul_tax_layer.backward(dprice)
dapple_price,dorange_price=add_apple_orange_layer.backward(dall_price)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)
dorange,dorange_num=mul_orange_layer.backward(dorange_price)

print("dall_price",dall_price)
print("dapple_price,dorange_price",dapple_price,dorange_price)
print("dapple,dapple_num",dapple,dapple_num)
print("dorange,dorange_num",dorange,dorange_num)

