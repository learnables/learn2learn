# pytorch-hacks

PyTorch Hackathon Repo

# API Demo

~~~python
import learn2learn as l2l

model = Net()
maml = l2l.MAML(model, lr=1e-3)
opt = optim.Adam(maml, lr=4e-3)

for X, y in task_generator:
    learner = maml.new()  # Clones model using clone_module()
    error = loss(learner(X), y)

    # Optionally compute pre-accuracy, etc...
    learner.adapt(error)  # Fast-adapt the parameters with maml_update() and backward(create_graph=True)
    adapt_error = loss(learner(X), y)
    opt.zero_grad()
    adapt_error.backward()
    opt.step()
~~~

# Acknowledgements

1. The RL environments are copied from: https://github.com/tristandeleu/pytorch-maml-rl
