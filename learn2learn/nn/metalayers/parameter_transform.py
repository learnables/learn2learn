# -*- coding=utf-8 -*-

import torch


class ParameterTransform(torch.nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/metalayers/parameter_transforms.py)

    **Description**

    Calls `module` after have transformed its parameters via `transform`.

    After the forward pass, the parameters of `module` are reverted to their original values.

    Useful to implement learnable (and constrained) updates of module weights (e.g., LoRA).
    Best used in conjunction with `MetaModule`.

    **Arguments**

    * **module** (Module) - The model to wrap.
    * **transform** (callable) - Function to be called on all parameters of `module` *before* its forward
        pass. Possibly a module itself, which is learnable.

    **Example**

    Where we only learn to a scalar factor of the original weights.

    ~~~python
    import learn2learn.nn.metalayers as ml

    model = torch.nn.Sequential(
        torch.nn.Linear(768, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )
    meta_model = ml.MetaModule(
        module=model,
        substitutions={
            torch.nn.Linear: lambda linear: ml.ParameterTransform(
                module=linear,
                transform=lambda param: l2l.nn.Scale(),
            ),
        },
        freeze_module=True,
    )
    ~~~
    """

    def __init__(self, module, transform):
        super(ParameterTransform, self).__init__()
        self.parameter_transforms = {}
        self._parameter_names = []
        self.module = module
        self.transform = transform

        for name, param in module.named_parameters():
            self.parameter_transforms[name] = transform(param)
            self._parameter_names.append(name)
        self.parameter_transforms = torch.nn.ModuleDict(
            self.parameter_transforms
        )

    def forward(self, *args, **kwargs):
        # set transformed parameters
        original_parameters = {}
        for name in self._parameter_names:
            param = getattr(self.module, name)
            original_parameters[name] = param
            transform = self.parameter_transforms[name]
            new_param = transform(param)
            if isinstance(param, torch.nn.Parameter):
                self.module._parameters[name] = new_param
            else:  # DataParallel (replica parameters are frozen)
                setattr(self.module, name, new_param)

        # compute forward pass
        out = self.module(*args, **kwargs)

        # revert to original parameters
        for name in self._parameter_names:
            original_param = original_parameters[name]
            if isinstance(original_param, torch.nn.Parameter):
                self.module._parameters[name] = original_param
            else:  # DataParallel (replica parameters are frozen)
                setattr(self.module, name, original_param)
        return out

    def meta_parameters(self):
        for name, transform in self.parameter_transforms.items():
            for param in transform.parameters():
                yield param
