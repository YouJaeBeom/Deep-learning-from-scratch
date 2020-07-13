
# 오차역전파법으로 구현한 것이 제대로 구현되었는지 확인하는 단계
# 수치미분으로 오차역전파법으로 구현한 기울기값이 제대로 업데이트 되는지 확인
import LayerNet
network=LayerNet(input_size=784,hidden_size=50,output_size=10)

x_batch=x_train[:3]
t_batch=t_train[:3]

grad_numerical=network.numerical_graient(x_batch,t_batch)
grad_backprop=network.graient(x_batch,t_batch)

for key in grad_numerical.keys():
    diff=np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key,":",str(diff))