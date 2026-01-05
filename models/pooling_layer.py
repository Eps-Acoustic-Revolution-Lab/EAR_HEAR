# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Modifications Copyright (c) 2025 Yuan Jin
#
# Changes made by Yuan Jin:
# - Added attention_mask parameter support to MHASTP.forward() and MQMHASTP.forward()
#
# This file is still licensed under the Apache License, Version 2.0.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHASTP(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 layer_num=2,
                 head_num=2,
                 d_s=1,   
                 bottleneck_dim=64,
                 **kwargs):
        super(MHASTP, self).__init__()
        assert (in_dim % head_num) == 0  
        self.in_dim = in_dim   
        self.head_num = head_num   
        d_model = int(in_dim / head_num) 
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]  
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s  
        heads_att_trans = []
        for i in range(self.head_num):  
            att_trans = nn.Sequential()
            for i in range(layer_num - 1):  
                att_trans.add_module(
                    'att_' + str(i),
                    nn.Conv1d(channel_dims[i], channel_dims[i + 1], 1, 1))
                att_trans.add_module('tanh' + str(i), nn.Tanh())
            att_trans.add_module( 
                'att_' + str(layer_num - 1),
                nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num],
                          1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input, attention_mask=None):

        if attention_mask is None: 
            if len(input.shape) == 4: 
                input = input.reshape(input.shape[0],
                                    input.shape[1] * input.shape[2],
                                    input.shape[3])
            assert len(input.shape) == 3
            chunks = torch.chunk(input, self.head_num, 1)  
            chunks_out = []
            for i, layer in enumerate(self.heads_att_trans):
                att_score = layer(chunks[i]) 
                alpha = F.softmax(att_score, dim=-1) 
                mean = torch.sum(alpha * chunks[i], dim=2) 
                var = torch.sum(alpha * chunks[i]**2, dim=2) - mean**2
                std = torch.sqrt(var.clamp(min=1e-7))
                chunks_out.append(torch.cat((mean, std), dim=1))
            out = torch.cat(chunks_out, dim=1)
            return out
        else: # 考虑 mask
            if len(input.shape) == 4: 
                input = input.reshape(input.shape[0],
                                    input.shape[1] * input.shape[2],
                                    input.shape[3])
            assert len(input.shape) == 3
            chunks = torch.chunk(input, self.head_num, 1)

            chunks_out = []
            for i, layer in enumerate(self.heads_att_trans):
                x = chunks[i] 
                att_score = layer(x)  
                att_score = att_score.masked_fill(attention_mask.unsqueeze(1), -1e4) 
                alpha = F.softmax(att_score, dim=-1)
                mean = torch.sum(alpha * x, dim=2)  
                var = torch.sum(alpha * x**2, dim=2) - mean**2
                std = torch.sqrt(var.clamp(min=1e-7))  
                chunks_out.append(torch.cat((mean, std), dim=1))  
            out = torch.cat(chunks_out, dim=1)
            return out

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MQMHASTP(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 layer_num=2,
                 query_num=2,
                 head_num=8,
                 d_s=2,
                 bottleneck_dim=64,
                 **kwargs):
        super(MQMHASTP, self).__init__()
        self.n_query = nn.ModuleList([
            MHASTP(in_dim,
                   layer_num=layer_num,
                   head_num=head_num,
                   d_s=d_s,
                   bottleneck_dim=bottleneck_dim) for i in range(query_num)  # 2
        ])
        self.query_num = query_num
        self.in_dim = in_dim

    def forward(self, input, attention_mask=None):
        if attention_mask is None:
            if len(input.shape) == 4: 
                input = input.reshape(input.shape[0],
                                    input.shape[1] * input.shape[2],
                                    input.shape[3])
            assert len(input.shape) == 3
            res = []
            for i, layer in enumerate(self.n_query):
                res.append(layer(input))
            out = torch.cat(res, dim=-1)
            return out
        else:  # 考虑 mask
            if len(input.shape) == 4:  
                input = input.reshape(input.shape[0],
                                      input.shape[1] * input.shape[2],
                                      input.shape[3])
            assert len(input.shape) == 3

            res = []
            for i, layer in enumerate(self.n_query):
                res.append(layer(input, attention_mask=attention_mask))
            out = torch.cat(res, dim=-1)
            return out

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2 * self.query_num
        return self.out_dim
