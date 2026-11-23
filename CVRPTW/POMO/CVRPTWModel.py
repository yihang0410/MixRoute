import torch
import torch.nn as nn
import torch.nn.functional as F


class CVRPTWModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CVRPTW_Encoder(**model_params)
        self.decoder = CVRPTW_Decoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_tw = reset_state.node_tw
        # shape: (batch, problem, 2)
        node_service = reset_state.node_service
        # shape: (batch, problem)

        # Node features: (x, y, demand, tw_start, tw_end, service_time) = 6 dim
        node_features = torch.cat((
            node_xy,
            node_demand[:, :, None],
            node_tw,
            node_service[:, :, None]
        ), dim=2)
        # shape: (batch, problem, 6)

        self.encoded_nodes = self.encoder(depot_xy, node_features)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(
                encoded_last_node, state.load, state.current_time,
                ninf_mask=state.ninf_mask
            )
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


# ENCODER

class CVRPTW_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(6, embedding_dim)
        # node features: (x, y, demand, tw_start, tw_end, service_time)

        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_features):
        # depot_xy.shape: (batch, 1, 2)
        # node_features.shape: (batch, problem, 6)

        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_features)

        out = torch.cat((embedded_depot, embedded_node), dim=1)

        for layer in self.layers:
            out = layer(out)

        return out


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']

        # Expert 1: InstanceNorm
        self.in_norm = nn.InstanceNorm1d(
            embedding_dim,
            affine=True,
            track_running_stats=False
        )

        # Expert 2: LayerNorm
        self.ln_norm = nn.LayerNorm(
            embedding_dim,
            elementwise_affine=True
        )

        # Expert 3: BatchNorm
        self.bn_norm = nn.BatchNorm1d(
            embedding_dim,
            affine=True
        )

        # Gate
        self.gate = nn.Linear(embedding_dim * 2, 3, bias=True)

        self.alpha = nn.Parameter(torch.tensor(0.5))

        with torch.no_grad():
            self.gate.bias[:] = torch.tensor([0.0, 0.0, 1.0])

    def forward(self, input1, input2):

        x = input1 + input2
        B, N, D = x.size()

        # ----- IN -----
        x_in = self.in_norm(x.transpose(1, 2)).transpose(1, 2)

        # ----- LN -----
        x_ln = self.ln_norm(x)

        # ----- BN -----
        x_bn = self.bn_norm(x.transpose(1, 2)).transpose(1, 2)

        # ----- Global context -----
        ctx = x.mean(dim=1, keepdim=True)
        ctx_expanded = ctx.expand(B, N, D)

        gate_input = torch.cat([x, ctx_expanded], dim=-1)

        # ----- Gating -----
        gate_logits = self.gate(gate_input)
        gate = F.softmax(gate_logits, dim=-1)

        g_in = gate[..., 0:1]
        g_ln = gate[..., 1:2]
        g_bn = gate[..., 2:3]

        # ----- Mixture -----
        out = g_in * x_in + g_ln * x_ln + g_bn * x_bn

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        out = (1.0 - alpha) * x + alpha * out

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.norm1 = AddAndInstanceNormalization(**model_params)
        self.norm2 = AddAndInstanceNormalization(**model_params)

        self.feed_forward = FeedForward(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.norm1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.norm2(out1, out2)

        return out3


# DECODER

class CVRPTW_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Query input: [encoded_last_node; load; current_time] = embedding_dim + 2
        self.Wq_last = nn.Linear(embedding_dim + 2, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None
        self.v = None
        self.single_head_key = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def forward(self, encoded_last_node, load, current_time, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # current_time.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem+1)

        head_num = self.model_params['head_num']

        # Concatenate context: [h_last; load; time]
        input_cat = torch.cat((
            encoded_last_node,
            load[:, :, None],
            current_time[:, :, None]
        ), dim=2)
        # shape: (batch, pomo, embedding+2)

        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        # Single-Head Attention for probability
        score = torch.matmul(mh_atten_out, self.single_head_key)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        return probs


# NN SUB CLASS / FUNCTIONS

def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)

    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
