# Extending Sparse

## Changing the Model class

The `Trainer` class placed in `sparsemod.model` contains the functions to train and evaluate a model. 

It should be easy to change it for your own routine. However, there are some points and parameters that should be fixed because Sparse uses them.

If you want to change the whole `Trainer` class follow all this section and you will become acquainted of all the necessary changes and mandatory requirements.

### Changing the class

The `Trainer` class neeeds to have as mandatory:

+ Signature

```
    def __init__(
        self,
        pruning=False,
        ddtype=floatp,
        datasets=None,
        models_path=None,
        cuda="cuda:0",
    ):
```
and common definitions

```

self.datasets = datasets
self.dtype = ddtype
# TODO: choose GPU with less memory
self.devicy = device(cuda if torchcuda.is_available() else "cpu")
self.datasizes = {
    i: len(sett) for i, sett in zip(["train", "val", "test"], self.datasets)
}
self.pruning = pruning
self.models_path = models_path
self.dataloader = None
self.criterion = nn.CrossEntropyLoss()
```

+ `self.load_dataloaders` function. This function needs to have as input arguments the batch size and the collate and it should beuild the object `self.dataloaders`

```
def load_dataloaders(self, batch_size, collate_fn):
    """
    Defines data loaders as a call to be able to define
    collates from outside
    """
    self.dataloader = {
        i: DataLoader(
            sett,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn,
        )
        for i, sett in zip(["train", "val", "test"], self.datasets)
    }
```
+ Train method (explained next) and, inside, the training loop (also explained next)
+ Evaluation method (explained next)


### Changing the training function

The `Trainer` class has two main methods `Trainer.train` and `Trainer.evaluate`. In the `Trainer.train` we can find a call to the training loop, a network reloading, and the setting up of some training hyperparameters.

```
    def train(
        self,
        net: nn.Module,
        parameters: Dict[str, float],
        name: str,
        epochs: int,
        reload: bool,
        old_net: nn.Module.state_dict,
    ) -> nn.Module:

        # Initialize network
        if reload:
            net = copy_weights(old_net, net)
        net.to(dtype=self.dtype, device=self.devicy)  # pyre-ignore [28]
        # Define loss and optimizer
        optimizer = Adam(net.parameters(), lr=parameters.get("learning_rate"))
        # TODO: change to reduce on plateau, is for cifar change 1000
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=parameters.get("learning_step") * 1000,
            gamma=parameters.get("learning_gamma"),
        )

        # Train Network
        net = self.train_loop(
            net,
            optimizer,
            exp_lr_scheduler,
            name,
            epochs,
            parameters.get("prune_threshold"),
        )
        return net

```

The main things that you have to conserve if you change this function are:
+ The function signature

```
    def train(
        self,
        net: nn.Module,
        parameters: Dict[str, float],
        name: str,
        epochs: int,
        reload: bool,
        old_net: nn.Module.state_dict,
    )
```
+ The reloading lines, used in morphisms

```
# Initialize network
if reload:
    net = copy_weights(old_net, net)
```
+ The return of the function which should be the network itself.

The other parts can be arranged as you want. They are mainly:

+ Placing the network in the `self.devicy` object
+ Assigning some hyperaparameters
+ And the training loop

it may be the case that you want only to modify the training loop, that is only modifying the hyperparameters and the trianing loop call:

```
    # Define loss and optimizer
    optimizer = Adam(net.parameters(), lr=parameters.get("learning_rate"))
    # TODO: change to reduce on plateau, is for cifar change 1000
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=parameters.get("learning_step") * 1000,
        gamma=parameters.get("learning_gamma"),
    )

    # Train Network
    net = self.train_loop(
        net,
        optimizer,
        exp_lr_scheduler,
        name,
        epochs,
        parameters.get("prune_threshold"),
    )
```

You could substitute only this part and conserve the rest of the class. In the case that you substitute only this part in the next section we describe how to accoplate a new training loop function.

In case you change this parts the key variables that come from the parametriation are 

+ 'learning_rate'
+ 'learning_step'
+ 'learning_gamma'
+ 'prune_threshold'

### Changing the training loop

The training loop mainly carries out the training of the network. However it also performs pruning. In the current training loop, pruning is made incrementally in the procedure, but in your case it could be totally you choice since the only variable coming from the parametrization is the `pruninng threshold`. 

Hence, you can change the training loop by your own. But keep in mind that this training loop performs:

+ Training: important is that, if you have not modified the dataloaders, they are defined as a dictionary in the `Trainer.load_dataloaders` function`

```

def load_dataloaders(self, batch_size, collate_fn):
    """
    Defines data loaders as a call to be able to define
    collates from outside
    """
    self.dataloader = {
        i: DataLoader(
            sett,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=collate_fn,
        )
        for i, sett in zip(["train", "val", "test"], self.datasets)
    }
```
2. Pruning: pruning is carried through the boolean variable `self.pruning`.

```
if phase == "train" and self.pruning:
    model = prune_net(model, init_threshold + thres_step * cnt)
    cnt += 1

```

The `threshold_init`, `thres_step` and `cnt` are variables for controlling the amount of pruning at each epoch. All those variables are defined at the beginning of the training loop, and are based on the only parameter that defines the pruning: the final value for the threshold 
```
init_threshold = 0.01
thres_step = (threshold - init_threshold) / steps
```

3. Saving models: models should be saved in the `self.models_path` and using the variable `name`
```
save(model.state_dict(), path.join(self.models_path, str(name) + ".pth"))
```
4. The function should return the model itself (the network)

If you use some external parameters you could read the file here and use those parameters.

### Changing the evaluation routine

The `Trainer.evaluate` function serves for both the evaluation of the network on the test set and for quantization purposes.  It should conserve its signature and should be using the `self.dataloaders` for choosing which set is used in each case. It should return an accuracy performane (accuracy, not error) and the network itself.

Also, due to quantization procedures the inputs and labels (in the original implementation) go to the `cpu`.

Hence,

+ Signature: `net: nn.Module, quant_mode: bool`
+ Use of `self.dataloaders` to distinguish between calibration and evaluation modes. If you do not perform quantization you don't need this separation. However, it is better to maintain the signature.

```
if quant_mode:
    data_loader = self.dataloader["train"]
else:
    data_loader = self.dataloader["test"]
```

Also calibration does not need the full dataset, that's why:

```                i
if quant_mode and cnt > 2000:
    break
```

+ Finally, the return values `return accuracy, network`. in the original implementation

```
return correct / total, net
```