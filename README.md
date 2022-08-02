# Pruning
It is a module that performs pruning and has the effect of reducing the model size and amount of computation by actually removing the filter in the model. Pruning can be performed on convolution and linear layers, and is valid only for .pt files created with pytorch framework.  
  
**1. layer_name** :  A list of target layers to be pruned in the model is input, and the name of each layer must be specified as a string type.  
**2. weight_init** : For weight initialization, one of 'lottery', 'random', and 'finetune' must be selected.
Each method is a choice on how to set the initial weight of the subnetwork. Finetune maintains the weights in the traditional way, random initializes them randomly, and lottery brings the initial weights of the original model.  
**3. copy_from_model** : It means the target model from which weights are to be copied, and is valid only in the case of the 'lottery' method.  
**4. global_prune** : Option to select global pruning or local pruning. If True, it determines the filter to be pruned for all input layers. Otherwise, it is decided for each layer.  
**5. arcface** : The choice of whether to use the arcface loss function. If true, prune the arcface weights.  
**6. amount** : Define the percentage of all filters to prune.  
