import torch 
import torch.nn as nn
import torch.nn.functional as F
import copy

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        super(CustomTransformerEncoderLayer, self).__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first,
            norm_first=norm_first, device=device, dtype=dtype)


    def forward_show(self, src, src_mask = None,
                src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        # if (src.dim() == 3 and not self.norm_first and not self.training and
        #     self.self_attn.batch_first and
        #     self.self_attn._qkv_same_embed_dim and self.activation_relu_or_gelu and
        #     self.norm1.eps == self.norm2.eps and
        #     src_mask is None and
        #         not (src.is_nested and src_key_padding_mask is not None)):
        #     tensor_args = (
        #         src,
        #         self.self_attn.in_proj_weight,
        #         self.self_attn.in_proj_bias,
        #         self.self_attn.out_proj.weight,
        #         self.self_attn.out_proj.bias,
        #         self.norm1.weight,
        #         self.norm1.bias,
        #         self.norm2.weight,
        #         self.norm2.bias,
        #         self.linear1.weight,
        #         self.linear1.bias,
        #         self.linear2.weight,
        #         self.linear2.bias,
        #     )
        #     if (not torch.overrides.has_torch_function(tensor_args) and
        #             # We have to use a list comprehension here because TorchScript
        #             # doesn't support generator expressions.
        #             all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
        #             (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
        #         return torch._transformer_encoder_layer_fwd(
        #             src,
        #             self.self_attn.embed_dim,
        #             self.self_attn.num_heads,
        #             self.self_attn.in_proj_weight,
        #             self.self_attn.in_proj_bias,
        #             self.self_attn.out_proj.weight,
        #             self.self_attn.out_proj.bias,
        #             self.activation_relu_or_gelu == 2,
        #             False,  # norm_first, currently not supported
        #             self.norm1.eps,
        #             self.norm1.weight,
        #             self.norm1.bias,
        #             self.norm2.weight,
        #             self.norm2.bias,
        #             self.linear1.weight,
        #             self.linear1.bias,
        #             self.linear2.weight,
        #             self.linear2.bias,
        #             src_mask if src_mask is not None else src_key_padding_mask,  # TODO: split into two args
        #         )
        x = src
        if self.norm_first:
            tmp, attn = self._sa_block_show(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + tmp
            x = x + self._ff_block(self.norm2(x))
        else:
            tmp, attn = self._sa_block_show(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + tmp)
            x = self.norm2(x + self._ff_block(x))

        return x, attn
    
    def _sa_block_show(self, x, attn_mask, key_padding_mask):
        tmp = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)
        x = tmp[0]
        attn = tmp[1]
        return self.dropout1(x), attn
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomTransformerEncoder(nn.TransformerEncoder):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``False`` (disabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False):
        super(CustomTransformerEncoder, self).__init__(encoder_layer, num_layers, norm=norm, 
                                                       enable_nested_tensor=enable_nested_tensor)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

    def forward_show(self, src, mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor) :
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())

        for mod in self.layers:
            if convert_to_nested:
                output, attn = mod.forward_show(output, src_mask=mask)
            else:
                output, attn = mod.forward_show(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn