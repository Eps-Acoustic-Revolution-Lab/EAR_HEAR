from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from models.pooling_layer import MQMHASTP
from safetensors.torch import load_file
import numpy as np

class TimeDownsample(nn.Module):
    def __init__(
        self, dim_in, dim_out=None, kernel_size=5, stride=5, padding=0, dropout=0.1
    ):
        super().__init__()
        self.dim_out = dim_out or dim_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.depthwise_conv = nn.Conv1d(
            in_channels=dim_in,   
            out_channels=dim_in, 
            kernel_size=kernel_size,   
            stride=stride,       
            padding=padding,     
            groups=dim_in,    
            bias=False,
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=dim_in, 
            out_channels=self.dim_out,  
            kernel_size=1,
            bias=False,
        )
        self.pool = nn.AvgPool1d(kernel_size, stride, padding=padding)
        self.norm1 = nn.LayerNorm(self.dim_out) 
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        if dim_in != self.dim_out:
            self.residual_conv = nn.Conv1d(
                dim_in, self.dim_out, kernel_size=1, bias=False
            )
        else:
            self.residual_conv = None

    def forward(self, x, attention_mask):
        if attention_mask is None:  
            residual = x  
            x_c = x.transpose(1, 2)  
            x_c = self.depthwise_conv(x_c)  
            x_c = self.pointwise_conv(x_c)  


            res = self.pool(residual.transpose(1, 2))  
            if self.residual_conv:
                res = self.residual_conv(res)  
            x_c = x_c + res  
            x_c = x_c.transpose(1, 2)  
            x_c = self.norm1(x_c)
            x_c = self.act1(x_c)
            x_c = self.dropout1(x_c)
            return x_c, None
        else:  
            residual = x 
            x_c_T = x.transpose(1, 2)  
            x_c = self.depthwise_conv(x_c_T)  
            x_c = self.pointwise_conv(x_c) 

            res = self.pool(residual.transpose(1, 2))  
            if self.residual_conv:
                res = self.residual_conv(res) 
            x_c = x_c + res  
            x_c = x_c.transpose(1, 2) 
            x_c = self.norm1(x_c)  
            x_c = self.act1(x_c)
            x_c = self.dropout1(x_c)  

            B, T = attention_mask.shape

            T_down = (T + 2 * self.padding - self.kernel_size) // self.stride + 1 

            new_mask = []
            for i in range(T_down):
                start = i * self.stride
                end = start + self.kernel_size
                window = attention_mask[:, start:end]  
                new_mask.append(window.all(dim=-1))  

            new_mask = torch.stack(new_mask, dim=1) 
            return x_c, new_mask

class HEAR(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.model = Generator(
            in_features=config['in_features'],   
            ffd_hidden_size=config['ffd_hidden_size'],
            num_classes=config['num_classes'],
            attn_layer_num=config['attn_layer_num'],
        )

        self.mixed_win_downsample = nn.Linear(config['input_dim_raw'], config['input_dim']) 
        self.input_norm = nn.LayerNorm(config['input_dim'])
        self.down_sample_conv = TimeDownsample(
            dim_in=config['input_dim'], 
            dim_out=config['transformer_encoder_input_dim'],
            kernel_size=config['down_sample_conv_kernel_size'],  
            stride=config['down_sample_conv_stride'],  
            dropout=config['down_sample_conv_dropout'],  
            padding=config['down_sample_conv_padding'],  
        )

    def forward(self, audio_tensor_1, labels_1=None, attention_mask_1=None, audio_tensor_2=None, labels_2=None, attention_mask_2=None, return_dict=False, mode='train'):
        if mode == "train":
            if attention_mask_1 is None:
                input_embeddings_1 = self.mixed_win_downsample(audio_tensor_1)
                input_embeddings_1 = self.input_norm(input_embeddings_1)
                logits_1 = self.down_sample_conv(input_embeddings_1)  

                input_embeddings_2 = self.mixed_win_downsample(audio_tensor_2)
                input_embeddings_2 = self.input_norm(input_embeddings_2)
                logits_2 = self.down_sample_conv(input_embeddings_2)  

                scores_g = self.model(logits_1, logits_2)
            else:
                attention_mask_1 = attention_mask_1.bool()
                skip_n_1 = int(attention_mask_1.size(-1) / audio_tensor_1.size(1))  
                attention_mask_1 = attention_mask_1[:, ::skip_n_1]  
                attention_mask_1 = attention_mask_1[:, :audio_tensor_1.size(1)]   
                attention_mask_1 = ~attention_mask_1 

                input_embeddings_1 = self.mixed_win_downsample(audio_tensor_1)  
                input_embeddings_1 = self.input_norm(input_embeddings_1)
                logits_1, attention_mask_1 = self.down_sample_conv(input_embeddings_1, attention_mask_1) 

                attention_mask_2 = attention_mask_2.bool()
                skip_n_2 = int(attention_mask_2.size(-1) / audio_tensor_2.size(1))  
                attention_mask_2 = attention_mask_2[:, ::skip_n_2] 
                attention_mask_2 = attention_mask_2[:, :audio_tensor_2.size(1)]  
                attention_mask_2 = ~attention_mask_2  

                input_embeddings_2 = self.mixed_win_downsample(audio_tensor_2)
                input_embeddings_2 = self.input_norm(input_embeddings_2)
                logits_2, attention_mask_2 = self.down_sample_conv(input_embeddings_2, attention_mask_2)  

                scores_g, labels_mix = self.model(logits_1, labels_1, attention_mask_1, logits_2, labels_2, attention_mask_2, "train")
            if return_dict:
                return SequenceClassifierOutput(
                    loss=None,
                    logits=scores_g,
                    hidden_states=None,
                    attentions=None,
                )
            return scores_g, labels_mix
        else:  
            if attention_mask_1 is None:
                input_embeddings_1 = self.mixed_win_downsample(audio_tensor_1) 
                input_embeddings_1 = self.input_norm(input_embeddings_1)
                logits_1, attention_mask_1 = self.down_sample_conv(input_embeddings_1, attention_mask_1) 
                scores_g = self.model(audio_tensor_1)
            else:
                attention_mask_1 = attention_mask_1.bool()
                skip_n = int(attention_mask_1.size(-1) / audio_tensor_1.size(1)) 
                attention_mask_1 = attention_mask_1[:, ::skip_n] 
                attention_mask_1 = attention_mask_1[:, :audio_tensor_1.size(1)]  
                attention_mask_1 = ~attention_mask_1 

                input_embeddings_1 = self.mixed_win_downsample(audio_tensor_1)  
                input_embeddings_1 = self.input_norm(input_embeddings_1)  
                logits_1, attention_mask_1 = self.down_sample_conv(input_embeddings_1, attention_mask_1) 

                scores_g = self.model(logits_1, None, attention_mask_1, None, None, None, "test")
            if return_dict:
                return SequenceClassifierOutput(
                    loss=None,
                    logits=scores_g,
                    hidden_states=None,
                    attentions=None,
                )
            return scores_g

class Generator(nn.Module):

    def __init__(self,
                 in_features,   # 4096
                 ffd_hidden_size,   
                 num_classes,
                 attn_layer_num,
                 
                 ):
        super(Generator, self).__init__()

        self.attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=in_features,  
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(attn_layer_num)  
            ]
        )
        self.ln_ffd  = nn.LayerNorm(in_features)        
        self.ffd = nn.Sequential(
            nn.Linear(in_features, ffd_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(ffd_hidden_size, in_features)
        )
        
        self.mqmhastp = MQMHASTP(in_features, 2, 2, 8, 2, 64)         

        self.dropout = nn.Dropout(0.0)
        
        self.fc =  nn.Linear(in_features * 4, num_classes)
        self.proj = nn.Tanh()
      
    def forward(self, ssl_feature_1, labels_1=None, attention_mask_1=None, ssl_feature_2=None, labels_2=None, attention_mask_2=None, mode="train"):
        '''
        ssl_feature: [B, T, D]    attention_mask:[B, T] (False表示有效位，True为mask位)
        output: [B, num_classes]
        '''
        if mode == "train":
            ssl_feature_1 = self.ffd(ssl_feature_1)
            tmp_ssl_feature_1 = ssl_feature_1
            for attn in self.attn:  
                tmp_ssl_feature_new_1, _ = attn(self.ln_ffd(tmp_ssl_feature_1), self.ln_ffd(tmp_ssl_feature_1), self.ln_ffd(tmp_ssl_feature_1), key_padding_mask=attention_mask_1)
                tmp_ssl_feature_1 = tmp_ssl_feature_1 + tmp_ssl_feature_new_1
            ssl_feature_1 = self.mqmhastp(tmp_ssl_feature_1.transpose(1,2), attention_mask_1)  
            
            ssl_feature_2 = self.ffd(ssl_feature_2)
            tmp_ssl_feature_2 = ssl_feature_2
            for attn in self.attn:  
                tmp_ssl_feature_new_2, _ = attn(self.ln_ffd(tmp_ssl_feature_2), self.ln_ffd(tmp_ssl_feature_2), self.ln_ffd(tmp_ssl_feature_2), key_padding_mask=attention_mask_2)
                tmp_ssl_feature_2 = tmp_ssl_feature_2 + tmp_ssl_feature_new_2
            ssl_feature_2 = self.mqmhastp(tmp_ssl_feature_2.transpose(1,2), attention_mask_2)  

            lambd = np.random.beta(2.0, 2.0) 
            ssl_feature_mix = lambd * ssl_feature_1 + (1 - lambd) * ssl_feature_2
            label_mix = lambd * labels_1 + (1 - lambd) * labels_2

            x = self.fc(ssl_feature_mix) 
            x = self.proj(x) * 2.0 + 3

            return x, label_mix
        else: 
            ssl_feature_1 = self.ffd(ssl_feature_1)
            tmp_ssl_feature_1 = ssl_feature_1
            for attn in self.attn:   
                tmp_ssl_feature_new_1, _ = attn(self.ln_ffd(tmp_ssl_feature_1), self.ln_ffd(tmp_ssl_feature_1), self.ln_ffd(tmp_ssl_feature_1), key_padding_mask=attention_mask_1)
                tmp_ssl_feature_1 = tmp_ssl_feature_1 + tmp_ssl_feature_new_1
            ssl_feature_1 = self.mqmhastp(tmp_ssl_feature_1.transpose(1,2), attention_mask_1)  

            x = self.fc(ssl_feature_1)  
            x = self.proj(x) * 2.0 + 3
            return x