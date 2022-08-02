import re
import copy
import torch
import numpy as np
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class Strunctured_prune():
    def __init__(self, model, layer_name, layer_type='weight', \
                       copy_from_model ='empty', \
                       amount=0.5, n=1, global_prune=False, arcface=False, feature_length=1000, \
                       weight_init='finetune'):
        super().__init__()
        self.change_dict = {}

        # target layer modify query dictionary
        self.change_dict['Conv2d_output'] ='self.assign_str = torch.nn.Conv2d(eval(self.layer_name).in_channels, \
                                            len(alive_filter), eval(self.layer_name).kernel_size, \
                                            eval(self.layer_name).stride, eval(self.layer_name).padding, \
                                            bias=bias_factor)'
        self.change_dict['Linear_output'] = 'self.assign_str = torch.nn.Linear(eval(self.layer_name).in_features, \
                                            len(alive_filter))'

        # next layer modify query dictionary
        self.change_dict['Conv2d_Conv2d_input'] = "self.next_assign_str = torch.nn.Conv2d(len(alive_filter), \
                                                eval(self.next_assign_str).out_channels,\
                                                eval(self.next_assign_str).kernel_size,\
                                                eval(self.next_assign_str).stride, eval(self.layer_name).padding, \
                                                bias=bias_factor)"
        self.change_dict['Conv2d_Linear_input'] = "self.next_assign_str = torch.nn.Linear(in_shape, \
                                                eval(self.next_assign_str).out_features)"
        self.change_dict['Linear_Linear_input'] = "self.next_assign_str = torch.nn.Linear(len(alive_filter), \
                                                eval(self.next_assign_str).out_features)"
        self.change_dict['Conv2d_BatchNorm2d_input'] = "self.next_assign_str = \
                                                        torch.nn.BatchNorm2d(eval(self.assign_str).out_channels,\
                                                        eval(self.next_assign_str).eps, \
                                                        eval(self.next_assign_str).momentum)"
        self.change_dict['break_cal'] = exec('self.next_layer_break=False')
        
        # before pruning model
        self.model = model
        # after pruning model
        self.prune_model = copy.deepcopy(self.model)
        # weight copy form model (for lottery)
        self.copy_from_model = copy_from_model

        # prune arcface weight
        self.arcface = arcface
        # feature length of arcface
        self.feature_length = feature_length

        # pruning ratio
        self.amount = amount
        # alive ratio
        self.real_amount = 1 - amount 
        # prune criteria (L1 or L2 Norm)
        self.n = n

        # next layer property
        self.next_layer_batchnorm = False
        self.next_layer_linear = False

        # pruning assinged layer & next layer name initialization
        self.assign_str = ''
        self.next_assign_str = ''
        self.search_string = ''

        # last layer bit
        self.last_layer = False

        # for global prune
        self.global_prune = global_prune
        self.global_threshold = 0

        # weight initialization model
        self.weight_init = weight_init

        # parameter name triming
        self.params = list(self.model.state_dict())
        for i in range(len(self.params)):
            while True:
                extract = re.search('[.]\d[.]',self.params[i])
                if extract != None:
                    new_string = extract[0]
                    new_string = new_string.replace('.','[',1)
                    new_string = new_string.replace('.',']',1)
                    new_string = new_string+'.'
                    self.params[i] = self.params[i].replace(extract[0],new_string)
                else:
                    break

        # global pruning
        if self.global_prune == True:
            weight_array = []
            for i in range(len(layer_name)):
                for filter in eval(layer_name[i]).weight:
                    weight_array.append(pow(float(abs(filter).mean().detach().cpu().numpy()),self.n))        
            weight_array = np.sort(weight_array)
            self.global_threshold = weight_array[int(len(weight_array)*self.amount)]

            self.weight_name = layer_name
            self.layer_type = layer_type
            
            for i in range(len(self.weight_name)):
                self.weight_name[i] = 'self.' + self.weight_name[i] + '.weight'

        # local layer pruning (with pytorch library)
        else:
            self.layer_name = 'self.'+layer_name
            self.layer_type = layer_type

            prune.ln_structured(
                eval(self.layer_name),
                name=self.layer_type,
                amount=self.amount,
                n=self.n,
                dim=0)
            prune.remove(eval(self.layer_name), 'weight')

    # change output size of assigned layer
    def shape_change(self, alive_filter):
        self.search_string = re.search('Conv2d|Linear',str(eval(self.assign_str)))[0]+'_output'
        if eval(self.assign_str).bias == None:
            bias_factor = False
        else:
            bias_factor = True
        query = self.change_dict[self.search_string].\
                replace('self.assign_str ='or'self.assign_str=', self.assign_str+'=')
        
        # check the number of pruned number of filters
        self.real_amount = 1 - len(alive_filter)/eval(self.assign_str).out_features

        #query execution : shape change
        exec(query)
        return

    # change input size of next assigned layer
    def next_shape_change(self, alive_filter):
        self.search_string = re.search('Conv2d|Linear',str(eval(self.assign_str)))[0]+'_'\
                    +re.search('Conv2d|Linear|BatchNorm2d',str(eval(self.next_assign_str)))[0]+'_input'
        # check exist of bias weight
        if eval(self.next_assign_str).bias == None:
            bias_factor = False
        else:
            bias_factor = True

        # feature extraction to classifier layer
        if self.search_string == 'Conv2d_Linear_input':
            in_shape = int(eval(self.next_assign_str).in_features*(1-self.real_amount))
        query = self.change_dict[self.search_string].\
                replace('self.next_assign_str ='or'self.next_assign_str=', self.next_assign_str+'=')
        
        #query execution : shape change
        exec(query)

        # next layer property check
        if re.search('BatchNorm2d', self.search_string) != None:
            self.next_layer_batchnorm = False
        elif re.search('Linear', self.search_string) != None:
            self.next_layer_linear = False
        else:
            self.next_layer_batchnorm = False
            self.next_layer_linear = False

        return

    # weight copy for pruned layer
    def pruned_weight_copy(self, prune_object, alive_filter):
        if self.weight_init == 'random':
            pass
        else:
            if self.weight_init == 'lottery':
                prune_object = prune_object.replace('self.model', 'self.copy_from_model')

            for p_f_index, f_index in zip(range(len(alive_filter)), alive_filter):
                with torch.no_grad():
                    exec(self.assign_str+'.weight[p_f_index].copy_('+prune_object+'[f_index])')
                    try:
                        prune_bias = prune_object.replace('.weight', '.bias')
                        exec(self.assign_str+'.bias[p_f_index].copy_('+prune_bias+'[f_index])')
                    except:
                        print('No bias parameter in the model.')
                        pass
        
        return

    # Layers not subject to pruning
    def weight_copy(self, present_layer):
        if self.weight_init == 'random':
            pass

        else:
            if self.weight_init == 'lottery':
                present_layer = present_layer.replace('self.model', 'self.copy_from_model')

            with torch.no_grad():
                target_layer = present_layer.replace('self.model', 'self.prune_model')
                exec(target_layer+'.copy_('+present_layer+')')

        return

    # weight copy for the pruned layer
    def next_weight_copy(self, next_object, alive_filter):
        if self.weight_init == 'random':
            pass
        else:
            if self.weight_init == 'lottery':
                next_object = next_object.replace('self.model', 'self.copy_from_model')
            #batchnorm
            if self.next_layer_batchnorm == True:
                for c_i, alive_channel in zip(range(len(alive_filter)),alive_filter):
                    with torch.no_grad():
                        exec(self.next_assign_str+'.weight[c_i].\
                            copy_('+next_object+'.weight[alive_channel])')
            #linear
            elif self.next_layer_linear == True:
                for f_index in range(eval('len('+self.next_assign_str+'.weight)')):
                    for c_i, alive_channel in zip(range(len(alive_filter)),alive_filter):
                        with torch.no_grad():
                            increment = int(eval(self.next_assign_str).in_features/len(alive_filter))
                            exec(self.next_assign_str+'.weight[f_index][c_i*increment:(c_i+1)*increment].\
                                copy_('+next_object+'.weight[f_index][alive_channel*increment:(alive_channel+1)*increment])')
            #conv
            else:
                # Repeat for the number of filters
                for f_index in range(eval('len('+self.next_assign_str+'.weight)')):
                    # Repeat for the number of alive channels
                    for c_i, alive_channel in zip(range(len(alive_filter)),alive_filter):
                        with torch.no_grad():
                            exec(self.next_assign_str+'.weight[f_index][c_i].\
                                copy_('+next_object+'.weight[f_index][alive_channel])')
        return

    def pruned_index(self, prune_object):
        # identify filters to survive/delete within a layer
        alive_filter = []
        for fil_no,filter in zip(range(eval(prune_object).shape[0]),eval(prune_object)):
            # global prune
            if self.global_prune == True:
                if pow(float(abs(filter).mean().detach().cpu().numpy()),self.n) <= self.global_threshold:
                    print(fil_no,'번째 filter pruned')
                else:
                    print(round(float(filter.sum().detach()), 4))
                    alive_filter.append(fil_no)
            # layer prune
            else:
                if not filter.cpu().detach().numpy().any():
                    print(fil_no,'번째 filter pruned')
                else:
                    print(round(float(filter.sum().detach()), 4))
                    alive_filter.append(fil_no)

        return alive_filter

    # prune arcface
    def prune_arcface(self, alive_filter):
        if (self.arcface == True and self.last_layer and self.global_prune) or (self.arcface == True and self.global_prune == False):
            
            self.prune_model.arcface.weight = nn.Parameter(torch.FloatTensor(self.feature_length, len(alive_filter)))
            nn.init.xavier_uniform_(self.prune_model.arcface.weight)

            for i in range(self.feature_length):
                for p_f_index, f_index in zip(range(len(alive_filter)), alive_filter):
                    with torch.no_grad():
                        if self.weight_init == 'random':
                            pass
                        elif self.weight_init == 'lottery':
                            self.prune_model.arcface.weight[i][p_f_index].copy_(self.copy_from_model.arcface.weight[i][f_index])
                        else:
                            self.prune_model.arcface.weight[i][p_f_index].copy_(self.model.arcface.weight[i][f_index])

        return

    def structure_trim(self):
        next_list = []
        for param_no in range(len(self.params)):
            present_layer = 'self.model.'+self.params[param_no]

            # global prune
            if self.global_prune:
                prune_layer_bit = present_layer in self.weight_name
                prune_object = present_layer
                self.layer_name = present_layer.replace('.weight', '')
                if self.arcface == True and self.last_layer == False:
                    try:
                        self.last_layer = self.weight_name.index(present_layer) == len(self.weight_name) - 1
                    except:
                        self.last_layer = False
            else:
                prune_object = self.layer_name + '.' + self.layer_type
                prune_layer_bit = prune_object == present_layer

            #Pruning 대상 layer인 경우
            if prune_layer_bit:
                alive_filter = self.pruned_index(prune_object)
                self.assign_str = self.layer_name.replace('self.','self.prune_')
                #Shape change
                self.shape_change(alive_filter)
                #Assign weight value
                self.pruned_weight_copy(prune_object, alive_filter)

                #pruning 대상의 next layer에 대한 작업
                while True:
                    if len(self.params) > param_no+1:
                        if re.search('.weight', self.params[param_no+1]) != None:
                            self.next_assign_str = 'self.prune_model.' + re.split('.weight', self.params[param_no+1])[0]
                            next_object = 'self.model.' + re.split('.weight', self.params[param_no+1])[0]
                            #next layer shape 변경
                            self.next_shape_change(alive_filter)
                            #next layer weight 변경
                            self.next_weight_copy(next_object, alive_filter)
                            next_list.append(self.next_assign_str)
                            #다음 weight 조정 후 break
                            if self.next_layer_linear == False and self.next_layer_batchnorm == False:
                                break
                    else:
                        break
                    param_no = param_no + 1
            else:
                if re.search('.weight', self.params[param_no]) != None:
                    if 'self.prune_model.'+self.params[param_no].replace('.weight','') not in next_list:
                        self.weight_copy(present_layer)

        self.prune_arcface(alive_filter)

        return self.prune_model
