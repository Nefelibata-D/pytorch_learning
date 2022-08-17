# 理解optimizer.zero_grad(), loss.backward(), optimizer.step()的作用及原理

在用pytorch训练模型时，通常会在遍历epochs的过程中依次用到optimizer.zero_grad(),loss.backward()和optimizer.step()三个函数，如下所示：

```python
    model = MyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            output= model(inputs)
            loss = criterion(output, labels)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**总得来说，这三个函数的作用是先将梯度归零（optimizer.zero_grad()），然后反向传播计算得到每个参数的梯度值（loss.backward()），最后通过梯度下降执行一步参数更新（optimizer.step()）**

接下来将通过源码分别理解这三个函数的具体实现过程。在此之前，先简要说明一下函数中常见的参数变量：

**param_groups：**Optimizer类在实例化时会在构造函数中创建一个param_groups列表，列表中有num_groups个长度为6的param_group字典（num_groups取决于你定义optimizer时传入了几组参数），每个param_group包含了 ['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'] 这6组键值对。

**param_group['params']：**由传入的模型参数组成的列表，即实例化Optimizer类时传入该group的参数，如果参数没有分组，则为整个模型的参数model.parameters()，每个参数是一个torch.nn.parameter.Parameter对象。

 

### 一、**optimizer.zero_grad()：**

```python
    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()```
```

optimizer.zero_grad()函数会遍历模型的所有参数，通过p.grad.detach_()方法截断反向传播的梯度流，再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。

**因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前。**

 

### 二、**loss.backward()：**

PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。

具体来说，torch.tensor是autograd包的基础类，如果你设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算，**如果你做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。**

更具体地说，损失函数loss是由模型的所有权重w经过一系列运算得到的，若某个w的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算，然后在使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。

**如果没有进行tensor.backward()的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前。**

 

### 三、**optimizer.step()：**

以SGD为例，torch.optim.SGD().step()源码如下：

```python
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
```

step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以**在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。**

**注意：optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。**
