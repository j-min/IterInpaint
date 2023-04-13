import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test

def _expand_mask(mask, dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def _build_causal_attention_mask(bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text, embedding_manager=None):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True, embedding_manager=embedding_manager)
        return z

    def encode(self, text, **kwargs):
        # output of length 77
        return self(text, **kwargs)

class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda",
        with_bbox=False,
        num_bins=1000,
        with_class_embedding=False,
        num_classes=48,
        max_length=77, extend_outputlen=None,
        ):
        super().__init__()
        from transformers import CLIPTextModel, logging
        class log_level:
            orig_log_level: int
            log_level: int
            def __init__(self, log_level: int):
                self.log_level = log_level
                self.orig_log_level = logging.get_verbosity()
            def __enter__(self):
                logging.set_verbosity(self.log_level)
            def __exit__(self, exception_type, exception_value, traceback):
                logging.set_verbosity(self.orig_log_level)
        with log_level(logging.ERROR):
            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.transformer = CLIPTextModel.from_pretrained(version)
            print(f'Loaded CLIP text model - {version}')

            

        self.device = device
        self.max_length = max_length
        self.with_bbox = with_bbox
        self.num_bins = num_bins
        self.extend_outputlen = extend_outputlen
        self.with_class_embedding = with_class_embedding
        self.num_classes = num_classes

        
        if with_bbox: ## expand box vocab
            N_bin_vocabs = num_bins
            for i in range(N_bin_vocabs):
                token = f'<bin{str(i).zfill(3)}>'
                self.tokenizer.add_tokens(token)
            print(f"Added {num_bins} bbox bin tokens to tokenizer")

            cliptext_weight = self.transformer.text_model.embeddings.token_embedding.weight

            if cliptext_weight.shape[1] == 768:
                ofa_weight = torch.load(
                    'preload_model_checkpoints/OFA-base/ofa_base.pt',  # 768
                    map_location='cpu')
                print('Loading bins from OFA-base')
            elif cliptext_weight.shape[1] == 1024:
                ofa_weight = torch.load(
                    'preload_model_checkpoints/OFA-large/ofa_large.pt',  # 1024
                    map_location='cpu')
                print('Loading bins from OFA-large')
            else:
                raise ValueError(f'OFA weight not found for dim={cliptext_weight.shape[1]}')

            ofa_weight = ofa_weight['model']['encoder.embed_tokens.weight']

            print('ofa_weight',ofa_weight.shape)
            init_weight = torch.cat([cliptext_weight,ofa_weight[58457:]],dim=0)
            n, d = cliptext_weight.shape[0]+num_bins, cliptext_weight.shape[1]
            expand_word_emb = nn.Embedding(n, d, _weight=init_weight).to(cliptext_weight.device)
            self.transformer.text_model.embeddings.token_embedding = expand_word_emb

            del ofa_weight
        if extend_outputlen is not None:
            clippos_weight = self.transformer.text_model.embeddings.position_embedding.weight
            extend_pos_weight = torch.cat([clippos_weight,clippos_weight[-1,:].view(1,-1).repeat(extend_outputlen-77,1)],dim=0)
            n, d = clippos_weight.shape[0]+extend_outputlen-77, clippos_weight.shape[1]
            expand_pos_emd = nn.Embedding(n, d, _weight=extend_pos_weight).to(clippos_weight.device)
            self.transformer.text_model.embeddings.position_embedding = expand_pos_emd
            self.transformer.text_model.embeddings.position_ids = torch.arange(1000).expand((1, -1)).to(clippos_weight.device)
            if extend_outputlen>1000:
                self.transformer.text_model.embeddings.position_ids = torch.arange(2000).expand((1, -1)).to(clippos_weight.device)

        if with_class_embedding:
            N_class_vocabs = num_classes
            for i in range(N_class_vocabs):
                token = f'<class{str(i).zfill(3)}>'
                self.tokenizer.add_tokens(token)
            print(f"Added {num_classes} class tokens to tokenizer")

            cliptext_weight = self.transformer.text_model.embeddings.token_embedding.weight
            new_class_embedding_weights = torch.randn((num_classes, cliptext_weight.shape[1]))
            init_weight = torch.cat([cliptext_weight,new_class_embedding_weights],dim=0)
            n = cliptext_weight.shape[0]+num_classes
            d = cliptext_weight.shape[1]
            expand_word_emb = nn.Embedding(n, d, _weight=init_weight).to(cliptext_weight.device)
            self.transformer.text_model.embeddings.token_embedding = expand_word_emb


            from ldm.data.clevr import clevr_all_objects

            self.all_objects = clevr_all_objects

            self.class_name_to_token = {}
            self.token_to_class_name = {}

            for i, obj in enumerate(self.all_objects):
                self.class_name_to_token[obj] = f'<class{str(i).zfill(3)}>'
                self.token_to_class_name[f'<class{str(i).zfill(3)}>'] = obj

            assert len(self.class_name_to_token) == self.num_classes, f'num_classes should be {len(self.class_name_to_token)}'

        def embedding_forward(
                self,
                input_ids = None,
                position_ids = None,
                inputs_embeds = None,
                embedding_manager = None,
            ) -> torch.Tensor:

                seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
                if position_ids is None:
                    position_ids = self.position_ids[:, :seq_length]

                if inputs_embeds is None:
                    inputs_embeds = self.token_embedding(input_ids)

                if embedding_manager is not None:
                    inputs_embeds = embedding_manager(input_ids, inputs_embeds)


                position_embeddings = self.position_embedding(position_ids)
                embeddings = inputs_embeds + position_embeddings
                
                return embeddings      

        self.transformer.text_model.embeddings.forward = embedding_forward.__get__(self.transformer.text_model.embeddings)

        def encoder_forward(
            self,
            inputs_embeds,
            attention_mask = None,
            causal_attention_mask = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            return hidden_states

        self.transformer.text_model.encoder.forward = encoder_forward.__get__(self.transformer.text_model.encoder)


        def text_encoder_forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            embedding_manager = None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is None:
                raise ValueError("You have to specify either input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids, embedding_manager=embedding_manager)

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = self.final_layer_norm(last_hidden_state)

            return last_hidden_state

        self.transformer.text_model.forward = text_encoder_forward.__get__(self.transformer.text_model)

        def transformer_forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = None,
            embedding_manager = None,
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_manager = embedding_manager
            )

        self.transformer.forward = transformer_forward.__get__(self.transformer)


    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, **kwargs):
        # print('texttext encoder input',text)
        if type(text)==type([]): ## list of strings

            text = text[0]

            if self.with_class_embedding:
                assert self.class_name_to_token is not None, 'class_name_to_token must be provided'
                for orig_class_name, new_token_name in self.class_name_to_token.items():
                    if orig_class_name in text:
                        text = text.replace(orig_class_name, new_token_name)
                        assert new_token_name in self.tokenizer.get_vocab(), f'{new_token_name} not in tokenizer vocab'
            print("prompt:", text)

            text = [text]

            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.device)        
        else: ## tokenized text
            tokens = text.to(self.device)
        z = self.transformer(input_ids=tokens, **kwargs)
        # print(text, tokens, z.shape) ## ['photo of a container'] tensor([[49406,  1125,   539,   320, 14913, 49407, 49407, 49407, 49407,... torch.Size([1, 77, 768])
        return z

    def encode(self, text, **kwargs):
        return self(text, **kwargs)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)