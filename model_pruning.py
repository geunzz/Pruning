import torch
from structured_prune import *

#model format should be '.pt'
model = torch.load('PATH/TO/THE/MODEL') #Pruning target model
origin_model = torch.load('swin/swin_try4_origin.pt') #Original model with initialized weights before training -> For Lottery

AMOUNT = 0.9 #pruning ratio (0 to 1 value)

###############################################################################################################
# For weight initialization(weight_init), one of 'lottery', 'random', and 'finetune' must be selected.        #
# Each method is a choice on how to set the initial weight of the subnetwork.                                 #
# Finetune maintains the weights in the traditional way, random initializes them randomly,                    #
# and lottery brings the initial weights of the original model.                                               #
# [Reference Papers] ##########################################################################################
# 'Lottery': The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, Jonathan Frankle, 2019 #
# 'Random' : Rethinking the Value of Network Pruning, Z Liu, 2018                                             #
###############################################################################################################

layer_name = ['model.MLP1_exp', 'model.MLP2_exp', 'model.MLP3_exp', 'model.MLP4_exp', 'model.MLP5_exp']       

#lottery #random #finetune
strunctured_prune = Strunctured_prune(model, layer_name, copy_from_model=origin_model, \
                                      weight_init='lottery', global_prune=True, arcface=False, amount=AMOUNT)
prune_model = strunctured_prune.structure_trim()


layer_name = ['model.outblock']

strunctured_prune = Strunctured_prune(prune_model, layer_name, copy_from_model=origin_model, \
                                      weight_init='lottery',global_prune=True, arcface=False, amount=AMOUNT)
prune_model = strunctured_prune.structure_trim()

#save pruned model
torch.save(prune_model, 'PATH/TO/THE/MODEL')
