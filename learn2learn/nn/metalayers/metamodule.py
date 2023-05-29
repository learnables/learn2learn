# -*- coding=utf-8 -*-

import torch


class MetaModule(torch.nn.Module):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/metalayers/metamodule.py)

    ## Description

    Takes a module and recursively replaces its submodules with others.

    The substitution is passed based on a dictionary (`substitutions`) which maps module classes to substitution functions.
    For example, to append a second Linear module after all Linear submodules:

    ```python
    substitutions[torch.nn.Linear] = lambda linear: torch.nn.Sequential(
        linear,
        torch.nn.Linear(linear.out_features, linear.out_features),
    )
    ```

    Optionally, the original module parameters can be frozen (`requires_grad = False`) by setting `freeze_module = True`.
    This is helpful when only the substitution modules need to be updated.

    **Arguments**

    * **module** (Module) - The model to wrap.
    * **substitutions** (dict) - Map of class -> construction substitutions.
    * **freeze_module** (bool, *optional*, default=True) - Whether to freeze the original `module` parameters.

    **Example**

    ~~~python
    import learn2learn.nn.metalayers as ml

    single_layer = torch.nn.Sequential(
        torch.nn.Linear(768, 10),
        torch.nn.ReLU(),
    )

    double_layers = ml.MetaModule(
        module=single_layer,
        substitutions={
            torch.nn.Linear: lambda linear: torch.nn.Sequential(
                linear,
                torch.nn.Linear(linear.out_features, linear.out_features),
            )
        },
        freeze_module=True,
    )
    print(double_layers)
    ~~~

    Output:

    ~~~python
    MetaModule(
      (wrapped_module): Sequential(
        (0): Sequential(
          (0): Linear(in_features=768, out_features=10, bias=True)
          (1): Linear(in_features=10, out_features=10, bias=True)
        )
        (1): ReLU()
      )
    )
    ~~~
    """

    def __init__(self, module, substitutions, freeze_module=True):
        super(MetaModule, self).__init__()
        self.substitutions = substitutions
        self.freeze_module = freeze_module
        self.wrapped_module = self._wrap(module)

    def _wrap(self, module):
        # wrap module with substitutions?
        new_module = module
        module_type = type(module)
        if module_type in self.substitutions:
            new_module = self.substitutions[module_type](new_module)

        # recurse over child modules
        for name, child in module.named_children():
            new_module._modules[name] = self._wrap(child)

        # freeze module parameters?
        if self.freeze_module:
            for p in module._parameters.values():
                if hasattr(p, 'requires_grad'):
                    p.requires_grad = False

        return new_module

    def _unwrap(self, module):
        base_module = module

        if hasattr(module, 'module'):
            base_module = module.module

        for name, child in base_module.named_children():
            base_module._modules[name] = self._unwrap(child)

        return base_module

    def meta_parameters(self, module=None):
        if module is None:
            module = self.wrapped_module
        if hasattr(module, 'meta_parameters'):
            for param in module.meta_parameters():
                yield param
        for name, child in module.named_children():
            for param in self.meta_parameters(child):
                yield param

    def module(self):
        """
        **Description**

        Returns the original `module`.

        **Example**

        (continued from above)

        ~~~python
        single_layer = double_layers.module()
        ~~~
        """
        module = self._unwrap(self.wrapped_module)
        return module

    def forward(self, *args, **kwargs):
        return self.wrapped_module(*args, **kwargs)
