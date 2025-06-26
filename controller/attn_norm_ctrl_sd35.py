import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.models.attention_processor import Attention
from typing import Dict, List, Optional, Tuple, Union
import abc
import inspect
from controller import logging
from controller import seq_aligner


'''
Replace the processor JointAttnProcessor2_0 in Attention class with controller

Notations:
encoder_hidden_states: text embeds
pooled_projections: pooled text embeds
temb: time embeds + pooled text embeds
'''

logger = logging.get_logger(__name__)


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, max_num_words, 1)
    return alpha_time_words


def calc_mean_std(feat, eps: float = 1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def expand_first(feat, scale=1.,):
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)

def adaln(src_feat, tar_feat):
    src_mean, src_std = calc_mean_std(src_feat)
    tar_mean, tar_std = calc_mean_std(tar_feat)
    # print("shape", src_mean.shape, src_std.shape, src_feat.shape)

    # tar_style_mean = expand_first(tar_mean)
    # tar_style_std = expand_first(tar_std)
    
    feat = (src_feat - src_mean) / src_std
    feat = feat * tar_std + tar_mean

    return feat


class Adalayernorm_replace(abc.ABC):
    '''
    Replace the adaptive layernorm.
    '''
    def __init__(self, prompts, num_steps, self_replace_steps, tokenizer, tokenizer3, device):
        super(Adalayernorm_replace, self).__init__()
        self.cur_step = 0
        self.cur_layer = 0
        self.num_adanorm = 0
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])

        # get clip mappers
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer, max_len=77)
        self.mapper, alphas = self.mapper.squeeze().to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], alphas.shape[1], 1)

        # get t5 mappers
        self.mapper2, alphas2 = seq_aligner.get_refinement_mapper(prompts, tokenizer3, max_len=256)
        self.mapper2, alphas2 = self.mapper2.squeeze().to(device), alphas2.to(device)
        self.alphas2 = alphas2.reshape(alphas2.shape[0], alphas2.shape[1], 1)
        
        # get alpha words
        self.cross_replace_alpha_clip = get_time_words_attention_alpha(prompts, num_steps, self_replace_steps, tokenizer).to(device)
        self.cross_replace_alpha_t5 = get_time_words_attention_alpha(prompts, num_steps, self_replace_steps, tokenizer3, max_num_words=256).to(device)
        

    def replace_adaptive_layernorm(self, base, replace, cur_step):
        '''
        decompose the token dimension based on clip1 + clip2, and T5
        clip1,clip2 [2, 77, 768]
        t5 [2, 256, 4096]
        txt embed: clip1+clip2 [2, 77, 1536] -> pad to [2, 77, 4096] + t5 [2, 156, 4096] > [2, 333, 4096]
        base: [1, 333, 1536]
        replace: [1, 333, 1536]
        mapper1: [1, 77]
        mapper2: [1, 256]
        '''
        # replace the clip 77 token

        alpha_word_clip = self.cross_replace_alpha_clip[cur_step] 
        unchange_base_clip = base[:, :77, :]
        select_unchange_base_clip = unchange_base_clip[:, self.mapper, :]
        replace[:, :77, :] = select_unchange_base_clip * self.alphas + replace[:, :77, :] * (1 - self.alphas)

#         replace_new = select_unchange_base_clip * self.alphas + replace[:, :77, :] * (1 - self.alphas)
#         replace[:, :77, :] = replace_new * alpha_word_clip + (1-alpha_word_clip) * replace[:, :77, :]
        
        # repalce the t5 256 token
        alpha_word_t5 = self.cross_replace_alpha_t5[cur_step] 
        unchange_base_t5 = base[:, 77:, :]
        select_unchange_base_t5 = unchange_base_t5[:, self.mapper2, :]
        replace[:, 77:, :] = select_unchange_base_t5 * self.alphas2 + replace[:, 77:, :] * (1 - self.alphas2)

#         replace_new_t5 = select_unchange_base_t5 * self.alphas2 + replace[:, 77:, :] * (1 - self.alphas2)
#         replace[:, 77:, :] = replace_new_t5 * alpha_word_t5 + (1-alpha_word_t5) * replace[:, 77:, :]

        return replace

    
    def forward(self, x):
        # x shape [4, 333, 1536], txt prompt embeds [4, 333, 4096] >= [negative_prompt, prompt]
        # 333 shape [clip1 + clip2, t5], clip concate in channel and then concate t5 in token [77 + 256, 768 +768 pad to 4096]
    
        if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            # negative/uncond prompt embed does not change
            h = x.shape[0] // 2
            txt_prpt_embed = x[h:] # [2, 333, 4096] with bs==2
            txt_prpt_embed_base = txt_prpt_embed[0,:].unsqueeze(0)
            txt_prpt_embed_replace = txt_prpt_embed[1:,:]
            txt_prpt_embed[1:,:] = self.replace_adaptive_layernorm(txt_prpt_embed_base, txt_prpt_embed_replace, self.cur_step)

        return x

    def __call__(self, x):
        x = self.forward(x)

        self.cur_layer += 1
        if self.cur_layer == self.num_adanorm:
            self.cur_step += 1
            self.cur_layer = 0

        return x


class SD3attention_adaln(abc.ABC):
    def adaln_self_attention(self, attn_base, att_replace):
        '''
        AdaLN self-attention for attn_base and att_replace
        Instead of copying all attn_base to att_repalce, only copy the norm to the target
        '''
        mod_attn = adaln(attn_base, att_replace)

        return mod_attn

    def replace_self_attention(self, attn_base, att_replace,):
        # mod_attn = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        mod_attn = adaln(attn_base, att_replace)
        return mod_attn

    def forward(self, query, key):
        if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            h = query.shape[0] // (self.batch_size)
            query = query.reshape(self.batch_size, h, *query.shape[1:])
            key = key.reshape(self.batch_size, h, *key.shape[1:])
            query_base, query_repalce = query[0], query[1:]
            key_base, key_repalce = key[0], key[1:]

            query[1:] = self.replace_self_attention(query_base, query_repalce,)
            key[1:] = self.replace_self_attention(key_base, key_repalce,)

            query = query.reshape(self.batch_size * h, *query.shape[2:])
            key = key.reshape(self.batch_size * h, *key.shape[2:])

        return query, key

    def __call__(self, query, key):
        query, key = self.forward(query, key)

        self.cur_layer += 1
        if self.cur_layer == self.num_att_layers:
            self.cur_step += 1
            self.cur_layer = 0

        return query, key
    
    def __init__(self, prompts, num_steps, self_replace_steps):
        super(SD3attention_adaln, self).__init__()
        self.cur_step = 0
        self.cur_layer = 0
        self.num_att_layers = 0
        self.batch_size = len(prompts)
        # self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = [0, self_replace_steps]
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])


class SD3attentionreplace(abc.ABC):
    def replace_self_attention(self, attn_base, att_replace,):
        attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        return attn_base

    def forward(self, query, key):
        if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            query_base, query_repalce = query[0], query[1:]
            key_base, key_repalce = key[0], key[1:]

            query[1:] = self.replace_self_attention(query_base, query_repalce,)
            key[1:] = self.replace_self_attention(key_base, key_repalce,)

        return query, key
    
    def forward2(self, x):
        if self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
            x_base, x_repalce = x[0], x[1:]

            x[1:] = self.replace_self_attention(x_base, x_repalce,)

        return x

    def __call__(self, x, y):
        # x, y = self.forward(x, y)
        x = self.forward2(x)

        self.cur_layer += 1
        if self.cur_layer == self.num_att_layers:
            self.cur_step += 1
            self.cur_layer = 0

        return x, y
    
    def __init__(self, prompts, num_steps, self_replace_steps):
        super(SD3attentionreplace, self).__init__()
        self.cur_step = 0
        self.cur_layer = 0
        self.num_att_layers = 0
        self.batch_size = len(prompts)
        # self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])


class P2P35_JointAttnProcessor2_0:
    """P2P Attention processor used typically in processing the SD3-like self-attention projections.
        image attn shape before concate: [4, 4096, 1536], context attn shape [4, 333, 1536]
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            
        # after view, query shape [4, 24, 4429, 64]
        ######################### replace the query and key related to the conditonal embeding by the controller #########################  
        h = query.shape[0]
        value[h // 2:], key[h // 2:] = self.controller(value[h // 2:], key[h // 2:])
        # value[h // 2:] = self.controller(value[h // 2:])

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
    


def register_attention_control_sd35(model, controller, controller_norm):
    '''
    replace the layernorm and self attention control in SD3 attentions
    '''
    
    def ca_forward_self(self):
        def forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **cross_attention_kwargs,
        ) -> torch.Tensor:
            r"""
            The forward method of the `Attention` class.

            Args:
                hidden_states (`torch.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`torch.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`torch.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty

            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            quiet_attn_parameters = {"ip_adapter_masks"}
            unused_kwargs = [
                k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
            ]
            if len(unused_kwargs) > 0:
                logger.warning(
                    f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
                )
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

            self.processor = P2P35_JointAttnProcessor2_0(controller)

            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        
        return forward
    
    
    def ca_forward_norm1(self):
        '''
        Norm1: adalayerzeronorm
        '''
        raise NotImplementedError

    def ca_forward_norm1_cxt(self):
        '''
        AdaLayerNormZero forward function 

        x: encoder_hidden_states
        timestep: fused tembed with the pooled text embeds
        '''
        def forward(
            x: torch.Tensor,
            timestep: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            hidden_dtype: Optional[torch.dtype] = None,
            emb: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.emb is not None:
                emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
            emb = self.linear(self.silu(emb))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            x = controller_norm(x)
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        
        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0], args[1]

        def __init__(self):
            self.num_att_layers = 0

    class DummyController2:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_adanorm = 0

    if controller is None:
        controller = DummyController()
    if controller_norm is None:
        controller_norm = DummyController2()

    def register_recr(net_):
        net_.forward = ca_forward_self(net_)

    def register_norm1(net_):
        net_.forward = ca_forward_norm1(net_)

    def register_norm1_cxt(net_):
        net_.forward = ca_forward_norm1_cxt(net_)

    att_count = 0
    ada_norm_count = 0
    sub_nets = model.transformer.transformer_blocks.named_children()

    for name, net in sub_nets:
        for sub_net in net.named_children():
            # replace the attention
            if sub_net[0] == "attn":
                if att_count < 37:
                    register_recr(sub_net[1])
                    att_count += 1
            if sub_net[0] == "norm1":
                pass
            if sub_net[0] == "norm1_context":
                if ada_norm_count <37:
                    register_norm1_cxt(sub_net[1])
                    ada_norm_count += 1

    controller.num_att_layers = att_count
    controller_norm.num_adanorm = ada_norm_count



