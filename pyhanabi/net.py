# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from itertools import permutations
import common_utils
import math


@torch.jit.script
def duel(v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor) -> torch.Tensor:
    assert a.size() == legal_move.size()
    assert legal_move.dim() == 3  # seq, batch, dim
    legal_a = a * legal_move
    q = v + legal_a - legal_a.mean(2, keepdim=True)
    return q


def cross_entropy(net, lstm_o, target_p, hand_slot_mask, seq_len):
    # target_p: [seq_len, batch, num_player, 5, 3]
    # hand_slot_mask: [seq_len, batch, num_player, 5]
    logit = net(lstm_o).view(target_p.size())
    q = nn.functional.softmax(logit, -1)
    logq = nn.functional.log_softmax(logit, -1)
    plogq = (target_p * logq).sum(-1)
    xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(min=1e-6)

    if xent.dim() == 3:
        # [seq, batch, num_player]
        xent = xent.mean(2)

    # save before sum out
    seq_xent = xent
    xent = xent.sum(0)
    assert xent.size() == seq_len.size()
    avg_xent = (xent / seq_len).mean().item()
    return xent, avg_xent, q, seq_xent.detach()


class FFWDNet(torch.jit.ScriptModule):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)
        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        """fake, only for compatibility"""
        shape = (1, batchsize, 1)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self, priv_s: torch.Tensor, publ_s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2, "dim should be 2, [batch, dim], get %d" % priv_s.dim()
        o = self.net(priv_s)
        a = self.fc_a(o)
        return a, hid

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        o = self.net(priv_s)
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [(seq_len), batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [(seq_len), batch]
        greedy_action = legal_q.argmax(-1).detach()
        return qa, greedy_action, q, o

    def pred_loss_1st(self, o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, o, target, hand_slot_mask, seq_len)


class LSTMNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        ff_layers = [nn.Linear(self.priv_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2
        bsize = hid["h0"].size(0)
        assert hid["h0"].dim() == 4
        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": hid["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        priv_s = priv_s.unsqueeze(0)

        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        a = a.squeeze(0)

        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.num_lstm_layer,
            bsize,
            -1,
            self.hid_dim,
        )

        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True
        x = self.net(priv_s)
        if len(hid) == 0:
            o, _ = self.lstm(x)
        else:
            o, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)


class PublicLSTMNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2

        bsize = hid["h0"].size(0)
        assert hid["h0"].dim() == 4
        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": hid["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        priv_s = priv_s.unsqueeze(0)
        publ_s = publ_s.unsqueeze(0)

        x = self.publ_net(publ_s)
        publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))

        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        a = self.fc_a(o)
        a = a.squeeze(0)

        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.num_lstm_layer,
            bsize,
            -1,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)

class EquivariantLSTMNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        ff_layers = [nn.Linear(self.priv_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)

        # group symmetry
        self.group_type = "cyclic"
        colours = [0, 1, 2, 3, 4]
        if self.group_type == "cyclic":
            self.symmetries = torch.tensor([[0,1,2,3,4],[4,0,1,2,3],[3,4,0,1,2],[2,3,4,0,1],[1,2,3,4,0]])
        elif self.group_type == "dihedral":
            self.symmetries = torch.tensor([[0,1,2,3,4],
              [1,2,3,4,0],
              [2,3,4,0,1],
              [3,4,0,1,2],
              [4,0,1,2,3],
              [0,4,3,2,1],
              [4,3,2,1,0],
              [3,2,1,0,4],
              [2,1,0,4,3],
              [1,0,4,3,2]])
        else: # symmetric group
            self.symmetries = torch.tensor(list(permutations(colours)))
        self.num_symmetries = len(self.symmetries)
        self.inv_symmetries = torch.argsort(self.symmetries)
        self.input_perms = torch.eye((self.priv_in_dim))
        self.output_perms = torch.eye((out_dim))
        # input permutations
        for symm in self.symmetries[1:]:
            perm = torch.eye(self.priv_in_dim)

            # partner hand
            card_perm = torch.zeros(25)
            for i,idx in enumerate(symm):
                temp = torch.tensor([0,1,2,3,4], dtype=torch.long)
                temp = 5*i+temp
                card_perm[5*idx:5*(idx+1)] = temp

            hand_perm = torch.zeros(125, dtype=torch.long)
            for i in range(5):
                hand_perm[25*i:25*(i+1)] = card_perm + 25*i

            perm[0:125] = perm[hand_perm]

             # fireworks
            fireworks_perm = card_perm + 167
            perm[167:192] = perm[fireworks_perm.long()]

            # discards
            discards_perm = torch.zeros(50, dtype=torch.long)
            for i,idx in enumerate(symm):
                temp = torch.tensor([203,204,205,206,207,208,209,210,211,212], dtype=torch.long)
                temp = 10*i+temp
                discards_perm[10*idx:10*(idx+1)] = temp

            perm[203:253] = perm[discards_perm]

            # last action
            which_colour_perm = torch.tensor([261,261,261,261,261], dtype=torch.long) + symm
            perm[261:266] = perm[which_colour_perm]
            perm[281:306] = perm[card_perm.long() + 281]

            # V0
            for i in range(10):
                perm[308+35*i:333+35*i] = perm[card_perm.long() + 308+35*i]
                directly_revealed_colour_perm = torch.tensor([333,333,333,333,333], dtype=torch.long) + 35*i + symm
                perm[333+35*i:338+35*i] = perm[directly_revealed_colour_perm]

            # greedy action
            if (self.priv_in_dim == 713):
                which_colour_perm = torch.tensor([666,666,666,666,666], dtype=torch.long) + symm
                perm[666:671] = perm[which_colour_perm]
                perm[686:711] = perm[card_perm.long() + 686]

            self.input_perms = torch.cat([self.input_perms, perm])
        self.input_perms = self.input_perms.transpose(0, 1)
        # output permutations, inverse of input transforms
        for symm in self.inv_symmetries[1:]:
            perm = torch.eye(self.out_dim)
            temp = torch.tensor([10,10,10,10,10], dtype=torch.long)
            perm[10:15] = perm[temp + symm]
            self.output_perms = torch.cat([self.output_perms, perm])
        self.output_perms = self.output_perms.reshape(self.num_symmetries, self.out_dim, self.out_dim)

        #repeat hidden/cell state
        self.hidden_cell_repeat = torch.eye(self.hid_dim).repeat(1, self.num_symmetries)
 

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        assert priv_s.dim() == 2

        bsize = hid["h0"].size(0)
        assert hid["h0"].dim() == 4
        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": hid["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }

        batchsize = priv_s.size(0)

        priv_s = torch.matmul(priv_s, self.input_perms).reshape(batchsize*self.num_symmetries, self.priv_in_dim)

        hid_h0 = torch.matmul(hid["h0"], self.hidden_cell_repeat).reshape(2, hid["h0"].size(1)*self.num_symmetries, self.hid_dim)

        hid_c0 = torch.matmul(hid["c0"], self.hidden_cell_repeat).reshape(2, hid["c0"].size(1)*self.num_symmetries, self.hid_dim)

        priv_s = priv_s.unsqueeze(0)

        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid_h0, hid_c0))
        a = self.fc_a(o)
        a = a.squeeze(0).reshape(batchsize, self.num_symmetries, self.out_dim)
        a = torch.matmul(a.transpose(0, 1), self.output_perms).sum(0)*1/self.num_symmetries

        h = h.reshape(2, -1, self.num_symmetries, self.hid_dim).sum(2)*1/self.num_symmetries
        c = c.reshape(2, -1, self.num_symmetries, self.hid_dim).sum(2)*1/self.num_symmetries

        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.num_lstm_layer,
            bsize,
            -1,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)
        
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        batchsize = priv_s.size(1)
        priv_s = torch.matmul(priv_s, self.input_perms).reshape(80, batchsize*self.num_symmetries, self.priv_in_dim) 
        x = self.net(priv_s)
        if len(hid) == 0:
            o, _ = self.lstm(x)
        else:
            hid_h0 = torch.matmul(hid["h0"], self.hidden_cell_repeat).reshape(2, hid["h0"].size(1)*self.num_symmetries, self.hid_dim)
            hid_c0 = torch.matmul(hid["c0"], self.hidden_cell_repeat).reshape(2, hid["c0"].size(1)*self.num_symmetries, self.hid_dim)
            o, _ = self.lstm(x, (hid_h0, hid_c0))

        a = self.fc_a(o).reshape(80, batchsize, self.num_symmetries, self.out_dim)
        a = torch.matmul(a.transpose(1, 2), self.output_perms).sum(1)*1/self.num_symmetries
    
        v = self.fc_v(o).reshape(80, batchsize, self.num_symmetries, -1).sum(2)*1/self.num_symmetries
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        o = o.reshape(-1, batchsize, self.num_symmetries, self.hid_dim).sum(2)*1/self.num_symmetries
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)

class EquivariantPublicLSTMNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)
        
        # group symmetry
        self.group_type = "cyclic"
        colours = [0, 1, 2, 3, 4]
        if self.group_type == "cyclic":
            self.symmetries = torch.tensor([[0,1,2,3,4],[4,0,1,2,3],[3,4,0,1,2],[2,3,4,0,1],[1,2,3,4,0]])
        elif self.group_type == "dihedral":
            self.symmetries = torch.tensor([[0,1,2,3,4],
              [1,2,3,4,0],
              [2,3,4,0,1],
              [3,4,0,1,2],
              [4,0,1,2,3],
              [0,4,3,2,1],
              [4,3,2,1,0],
              [3,2,1,0,4],
              [2,1,0,4,3],
              [1,0,4,3,2]])
        else: # symmetric group
            self.symmetries = torch.tensor(list(permutations(colours)))
        self.num_symmetries = len(self.symmetries)
        self.inv_symmetries = torch.argsort(self.symmetries)
        self.priv_input_perms = torch.eye((self.priv_in_dim))
        self.publ_input_perms = torch.eye((self.publ_in_dim))
        self.output_perms = torch.eye((out_dim))
        
        # input permutations
        for symm in self.symmetries[1:]:
            perm_priv = torch.eye(self.priv_in_dim)
            perm_publ = torch.eye(self.publ_in_dim)

            # partner hand
            card_perm = torch.zeros(25)
            for i,idx in enumerate(symm):
                temp = torch.tensor([0,1,2,3,4], dtype=torch.long)
                temp = 5*i+temp
                card_perm[5*idx:5*(idx+1)] = temp

            hand_perm = torch.zeros(125, dtype=torch.long)
            for i in range(5):
                hand_perm[25*i:25*(i+1)] = card_perm + 25*i

            perm_priv[0:125] = perm_priv[hand_perm]

             # fireworks
            fireworks_perm = card_perm + 167
            perm_priv[167:192] = perm_priv[fireworks_perm.long()]
            perm_publ[42:67] = perm_publ[fireworks_perm.long() - 125]

            # discards
            discards_perm = torch.zeros(50, dtype=torch.long)
            for i,idx in enumerate(symm):
                temp = torch.tensor([203,204,205,206,207,208,209,210,211,212], dtype=torch.long)
                temp = 10*i+temp
                discards_perm[10*idx:10*(idx+1)] = temp

            perm_priv[203:253] = perm_priv[discards_perm]
            perm_publ[78:128] = perm_publ[discards_perm - 125]

            # last action
            which_colour_perm = torch.tensor([261,261,261,261,261], dtype=torch.long) + symm
            perm_priv[261:266] = perm_priv[which_colour_perm]
            perm_priv[281:306] = perm_priv[card_perm.long() + 281]
            perm_publ[136:141] = perm_publ[which_colour_perm - 125]
            perm_publ[156:181] = perm_publ[card_perm.long() + 281 - 125]

            # V0
            for i in range(10):
                perm_priv[308+35*i:333+35*i] = perm_priv[card_perm.long() + 308+35*i]
                perm_publ[183+35*i:208+35*i] = perm_publ[card_perm.long() + 183+35*i]
                directly_revealed_colour_perm = torch.tensor([333,333,333,333,333], dtype=torch.long) + 35*i + symm
                perm_priv[333+35*i:338+35*i] = perm_priv[directly_revealed_colour_perm]
                perm_publ[208+35*i:213+35*i] = perm_publ[directly_revealed_colour_perm - 125]

            # greedy action
            if (self.priv_in_dim == 713):
                which_colour_perm = torch.tensor([666,666,666,666,666], dtype=torch.long) + symm
                perm_priv[666:671] = perm_priv[which_colour_perm]
                perm_publ[541:546] = perm_publ[which_colour_perm]
                perm_priv[686:711] = perm_priv[card_perm.long() + 686]
                perm_publ[561:586] = perm_publ[card_perm.long() + 686 - 125]

            self.priv_input_perms = torch.cat([self.priv_input_perms, perm_priv])
            self.publ_input_perms = torch.cat([self.publ_input_perms, perm_publ])
                                              
        self.priv_input_perms = self.priv_input_perms.transpose(0, 1)
        self.publ_input_perms = self.publ_input_perms.transpose(0, 1)                       
        # output permutations, inverse of input transforms
        for symm in self.inv_symmetries[1:]:
            perm = torch.eye(self.out_dim)
            temp = torch.tensor([10,10,10,10,10], dtype=torch.long)
            perm[10:15] = perm[temp + symm]
            self.output_perms = torch.cat([self.output_perms, perm])
        self.output_perms = self.output_perms.reshape(self.num_symmetries, self.out_dim, self.out_dim)

        #repeat hidden/cell state
        self.hidden_cell_repeat = torch.eye(self.hid_dim).repeat(1, self.num_symmetries)
        

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2

        bsize = hid["h0"].size(0)
        assert hid["h0"].dim() == 4
        # hid size: [batch, num_layer, num_player, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].transpose(0, 1).flatten(1, 2).contiguous(),
            "c0": hid["c0"].transpose(0, 1).flatten(1, 2).contiguous(),
        }
                                              
        batchsize = priv_s.size(0)                                      

        priv_s = torch.matmul(priv_s, self.priv_input_perms).reshape(batchsize*self.num_symmetries, self.priv_in_dim)
        publ_s = torch.matmul(publ_s, self.publ_input_perms).reshape(batchsize*self.num_symmetries, self.publ_in_dim)                                     
        hid_h0 = torch.matmul(hid["h0"], self.hidden_cell_repeat).reshape(2, hid["h0"].size(1)*self.num_symmetries, self.hid_dim)
        hid_c0 = torch.matmul(hid["c0"], self.hidden_cell_repeat).reshape(2, hid["c0"].size(1)*self.num_symmetries, self.hid_dim)                        
                                              
        priv_s = priv_s.unsqueeze(0)
        publ_s = publ_s.unsqueeze(0)

        x = self.publ_net(publ_s)
        publ_o, (h, c) = self.lstm(x, (hid_h0, hid_c0))

        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        a = self.fc_a(o)
        a = a.squeeze(0).reshape(batchsize, self.num_symmetries, self.out_dim)
        a = torch.matmul(a.transpose(0, 1), self.output_perms).sum(0)*1/self.num_symmetries
                                              
        h = h.reshape(2, -1, self.num_symmetries, self.hid_dim).sum(2)*1/self.num_symmetries
        c = c.reshape(2, -1, self.num_symmetries, self.hid_dim).sum(2)*1/self.num_symmetries

        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_layer, num_player, dim]
        interim_hid_shape = (
            self.num_lstm_layer,
            bsize,
            -1,
            self.hid_dim,
        )
        h = h.view(*interim_hid_shape).transpose(0, 1)
        c = c.view(*interim_hid_shape).transpose(0, 1)

        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        batchsize = priv_s.size(1)
        priv_s = torch.matmul(priv_s, self.priv_input_perms).reshape(80, batchsize*self.num_symmetries, self.priv_in_dim)
        publ_s = torch.matmul(publ_s, self.publ_input_perms).reshape(80, batchsize*self.num_symmetries, self.publ_in_dim)
                                              
        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            hid_h0 = torch.matmul(hid["h0"], self.hidden_cell_repeat).reshape(2, hid["h0"].size(1)*self.num_symmetries, self.hid_dim)
            hid_c0 = torch.matmul(hid["c0"], self.hidden_cell_repeat).reshape(2, hid["c0"].size(1)*self.num_symmetries, self.hid_dim)                                  
            publ_o, _ = self.lstm(x, (hid_h0, hid_c0))
        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        a = self.fc_a(o).reshape(80, batchsize, self.num_symmetries, self.out_dim)
        a = torch.matmul(a.transpose(1, 2), self.output_perms).sum(1)*1/self.num_symmetries                                      
        v = self.fc_v(o).reshape(80, batchsize, self.num_symmetries, -1).sum(2)*1/self.num_symmetries
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)                                      
        o = o.reshape(-1, batchsize, self.num_symmetries, self.hid_dim).sum(2)*1/self.num_symmetries                                      
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)
