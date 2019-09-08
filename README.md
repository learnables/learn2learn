<p align="center"><img src="https://raw.githubusercontent.com/seba-1511/learn2learn/gh-pages/assets/img/l2l-full.png" height="150px" /></p>

--------------------------------------------------------------------------------

learn2learn is a PyTorch library for meta-learning implementations.
It was developed during the [first PyTorch Hackathon](http://pytorchmpk.devpost.com/).

# Installation

~~~bash
pip install learn2learn
~~~

# API Demo

The following is an example of using the high-level MAML implementation on MNIST.
For more algorithms and lower-level utilities, please refer to [the documentation](http://learn2learn.net/docs/learn2learn/) or the [examples](https://github.com/learnables/learn2learn/tree/master/examples).

~~~python
import learn2learn as l2l

mnist = torchvision.datasets.MNIST(root="/tmp/mnist", train=True)

mnist = l2l.data.MetaDataset(mnist)
task_generator = l2l.data.TaskGenerator(mnist,
                                        ways=3,
                                        classes=[0, 1, 4, 6, 8, 9],
                                        tasks=10)
model = Net()
maml = l2l.MAML(model, lr=1e-3, first_order=False)
opt = optim.Adam(maml.parameters(), lr=4e-3)

for iteration in range(num_iterations):
    learner = maml.clone()  # Creates a clone of model
    adaptation_task = task_generator.sample(shots=1)

    # Fast adapt
    for step in range(adaptation_steps):
        error = compute_loss(adaptation_task)
        learner.adapt(error)

    # Compute evaluation loss
    evaluation_task = task_generator.sample(shots=1,
                                            classes=adaptation_task.sampled_classes)
    evaluation_error = compute_loss(evaluation_task)

    # Meta-update the model parameters
    opt.zero_grad()
    evaluation_error.backward()
    opt.step()
~~~

# Acknowledgements

1. The RL environments are adapted from Tristan Deleu's [implementations](https://github.com/tristandeleu/pytorch-maml-rl) and from the ProMP [repository](https://github.com/jonasrothfuss/ProMP/). Both shared with permission, under the MIT License.
