from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.activations import NewGELUActivation
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
)


# HAT-prefixed classes are shared across genome pretraining and downstream prediction code.
# The configuration deviates slightly from standard transformer settings.
# The implementation is adapted from public HAT references with project-specific tweaks.
class HATConfig(PretrainedConfig):
    model_type = "hierarchical-transformer"

    def __init__(
            self,
            hidden_size=640,
            max_blocks=64,
            block_size=128,
            model_max_length=8192,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1280,
            hidden_act="gelu_new",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=10000,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            cls_token_id=1,
            sep_token_id=2,
            unk_token_id=3,
            mask_token_id=4,
            type_vocab_size=2,
            vocab_size=100005,
            position_embedding_type=None,
            encoder_layout=None,
            use_cache=True,
            classifier_dropout=None,
            is_decoder=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.model_max_length = model_max_length
        self.encoder_layout = encoder_layout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.is_decoder = is_decoder


def create_position_ids_from_input_ids(input_ids, padding_idx, position_ids):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    return position_ids[:, :input_ids.size(1)].repeat(input_ids.size(0), 1) * mask


class HATEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config, inputs_embeds_shape=1280):
        super().__init__()
        self.padding_idx = config.pad_token_id
        # Inputs already arrive as ESM-2 embeddings, so replace token embedding lookup with a linear projection.
        self.word_embeddings = nn.Linear(inputs_embeds_shape, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.block_size + self.padding_idx + 1, config.hidden_size,
                                                padding_idx=self.padding_idx)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(self.padding_idx + 1,
                                                          config.block_size + self.padding_idx + 1).repeat(
            config.max_blocks).expand((1, -1)))
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long),
                             persistent=False)

    def forward(
            self,
            input_ids=None,  # Kept for API compatibility; embedding path uses inputs_embeds.
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, self.position_ids)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
                # print(position_ids, flush = True)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = self.word_embeddings(inputs_embeds) + token_type_embeddings
        # Positional embedding type can be swapped by changing this addition.
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class TransformerLayer(nn.Module):
    def __init__(self, config, relative_position=True):
        super().__init__()
        self.attention = RobertaAttention(config, "relative_key_query" if relative_position else None)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


# Convert (seq_len * hidden) into (num_blocks * block_size * hidden).
def transform_tokens2gene_blocks(hidden_states, num_blocks, block_size):
    seg_hidden_states = torch.reshape(hidden_states,
                                      (hidden_states.size(0), num_blocks, block_size, hidden_states.size(-1)))

    hidden_states_reshape = seg_hidden_states.contiguous().view(hidden_states.size(0) * num_blocks,
                                                                block_size, seg_hidden_states.size(-1))

    return hidden_states_reshape


def transform_masks2gene_blocks(hidden_states, num_blocks, block_size):
    seg_hidden_states = torch.reshape(hidden_states, (hidden_states.size(0), 1, 1, num_blocks, block_size))

    hidden_states_reshape = seg_hidden_states.contiguous().view(hidden_states.size(0) * num_blocks,
                                                                1, 1, seg_hidden_states.size(-1))

    return hidden_states_reshape


# Convert (num_blocks * block_size * hidden) back to (seq_len * hidden).
def transform_gene_blocks2tokens(seg_hidden_states, num_blocks, block_size):
    hidden_states = seg_hidden_states.contiguous().view(seg_hidden_states.size(0) // num_blocks, num_blocks,
                                                        block_size, seg_hidden_states.size(-1))

    hidden_states = hidden_states.contiguous().view(hidden_states.size(0), num_blocks * block_size,
                                                    hidden_states.size(-1))
    return hidden_states


class HATLayer(nn.Module):
    def __init__(self, config, use_gene_block_encoder=True, use_genome_encoder=True):
        super().__init__()
        self.block_size = config.block_size
        self.max_blocks = config.max_blocks
        self.hidden_size = config.hidden_size
        self.use_gene_block_encoder = use_gene_block_encoder
        self.use_genome_encoder = use_genome_encoder
        if self.use_gene_block_encoder:
            self.gene_block_encoder = TransformerLayer(config)
        if self.use_genome_encoder:
            self.genome_encoder = TransformerLayer(config, False)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            num_blocks=None,
            output_attentions=False,
    ):

        gene_block_outputs = (None, None)
        # Segment-wise encoder: split tokens/masks into blocks, encode, then stitch back together.
        if self.use_gene_block_encoder:
            gene_block_inputs = transform_tokens2gene_blocks(hidden_states,
                                                             num_blocks=num_blocks,
                                                             block_size=self.block_size)
            gene_block_masks = transform_masks2gene_blocks(attention_mask,
                                                           num_blocks=num_blocks,
                                                           block_size=self.block_size)

            gene_block_outputs = self.gene_block_encoder(gene_block_inputs,
                                                         gene_block_masks,
                                                         output_attentions=output_attentions)

            # Restore block outputs back to the flattened token view.
            outputs = transform_gene_blocks2tokens(gene_block_outputs[0],
                                                   num_blocks=num_blocks,
                                                   block_size=self.block_size)

        else:
            outputs = hidden_states

        genome_outputs = (None, None)
        # Cross-segment encoder consumes block-level CLS tokens and writes them back to the same slots.
        if self.use_genome_encoder:
            gene_block_global_tokens = outputs[:, ::self.block_size].clone()
            gene_block_attention_mask = attention_mask[:, :, :, ::self.block_size].clone()

            genome_outputs = self.genome_encoder(gene_block_global_tokens,
                                                 gene_block_attention_mask,
                                                 output_attentions=output_attentions)

            outputs[:, ::self.block_size] = genome_outputs[0]

        if output_attentions:
            return outputs, gene_block_outputs[1], genome_outputs[1]

        return outputs, None


@dataclass
class BaseModelOutputWithGeneBlockAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    gene_block_attentions: Optional[Tuple[torch.FloatTensor]] = None


# Stack of HAT layers with optional gradient checkpointing.
class HATEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([HATLayer(config,
                                             use_gene_block_encoder=self.config.encoder_layout[str(idx)][
                                                 'gene_block_encoder'],
                                             use_genome_encoder=self.config.encoder_layout[str(idx)]['genome_encoder'])
                                    for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            num_blocks=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_gene_block_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    num_blocks,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_gene_block_attentions = all_gene_block_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_gene_block_attentions
                ]
                if v is not None
            )
        return BaseModelOutputWithGeneBlockAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            gene_block_attentions=all_gene_block_attentions,
        )


class HATPreTrainedModel(PreTrainedModel):
    config_class = HATConfig
    base_model_prefix = "hat"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            with torch.no_grad():
                module.weight.fill_(1.0 / self.config.window_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HATEncoder):
            module.gradient_checkpointing = value

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]

    @classmethod
    def from_config(cls, config):
        return cls._from_config(config)


class HATModel(HATPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = HATEmbeddings(config)
        self.encoder = HATEncoder(config)

        self.post_init()

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, cluster_ids=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Compute number of sentences
        num_blocks = input_shape[-1] // self.config.block_size

        embedding_output = self.embeddings(
            input_ids=cluster_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            num_blocks=num_blocks,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output) + encoder_outputs[1:]

        return BaseModelOutputWithGeneBlockAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            gene_block_attentions=encoder_outputs.gene_block_attentions,
        )


class HATLMHead(nn.Module):
    """Prediction head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = NewGELUActivation()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        x = self.layer_norm(x)

        x = self.decoder(x)

        return x

    def _tie_weights(self):
        self.bias = self.decoder.bias


# Genome masked language model built on top of the hierarchical encoder.
class HATForMaskedLM(HATPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.hi_transformer = HATModel(config)
        self.lm_head = HATLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.hi_transformer.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.hi_transformer.embeddings.word_embeddings = value

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        pass

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cluster_ids=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hi_transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cluster_ids=cluster_ids,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )