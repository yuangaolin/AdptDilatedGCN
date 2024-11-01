# -*- coding: utf-8 -*-
import dgl
import dgl.function as dglfn
import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from adpt_attention import EncoderLayer


class ModelNew(nn.Module):

    def __init__(self):
        super().__init__()
        self.a_init = nn.Linear(82, 120)
        self.b_init = nn.Linear(12, 8)
        self.r_init_1 = nn.Linear(20, 9)
        self.r_init_2 = nn.Linear(30, 120)
        self.r_mt = Pro_Update(120)
        self.a_mt = Lig_Update(120)
        # ligand vector
        self.A = nn.Linear(120, 120)
        self.B = nn.Linear(240, 120)
        self.a_conv1 = Adaptive_GCN(120, 8, 120, 4)
        self.a_conv2 = Adaptive_GCN(120, 8, 120, 4)
        # protein vector
        self.C = nn.Linear(120, 120)
        self.D = nn.Linear(240, 120)
        self.r_conv1 = Dilated_GCN(120, 0, 120, 4)
        self.r_conv2 = Dilated_GCN(120, 0, 120, 4)

        # Pocket-Ligand vector
        self.i_conf = GCN(120, 8, 3)

        # Predict OutPut
        self.classifier = nn.Sequential(
            nn.Linear(120 + 120 + 120 + 6, 298),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(298, 160),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(160, 1)
        )
        self.sum_pool = dglnn.SumPooling()
        self.mean_pool = dglnn.AvgPooling()
        self.dilated_block_a = DilatedConvNet(120, 120)
        self.dilated_block_b = DilatedConvNet_T(120, 120)
        self.smi_attention_poc = EncoderLayer(120, 120, 0.1, 0.1, 2)

    def forward(self, ga, gr, gi, vina):
        device = torch.device("cuda:0")
        ga = ga.to('cuda:0')
        gr = gr.to('cuda:0')
        gi = gi.to('cuda:0')
        vina = vina.to('cuda:0')

        # ligand features
        va_init = self.a_init(ga.ndata['feat'])
        ea = self.b_init(ga.edata['feat'])

        # protein features
        vr = self.r_init_1(gr.ndata['feat'][:, :20])
        vr = torch.cat((vr, gr.ndata['feat'][:, 20:]), -1)
        vr_init = self.r_init_2(vr)

        # pocket-ligand features
        vi_a = self.a_init(gi.ndata['feat']['atom'])
        vi_r = self.r_init_1(gi.ndata['feat']['residue'][:, :20])
        vi_r = torch.cat((vi_r, gi.ndata['feat']['residue'][:, 20:]), -1)
        vi_r = self.r_init_2(vi_r)
        ei = gi.edata['weight'].reshape(-1)
        ei = torch.cat((ei, ei)).unsqueeze(1)

        # Add reverse edges and set the number of batch nodes and edges
        gii = dgl.add_reverse_edges(dgl.to_homogeneous(gi))
        gii.set_batch_num_nodes(gi.batch_num_nodes('atom') + gi.batch_num_nodes('residue'))
        gii.set_batch_num_edges(gi.batch_num_edges() * 2)

        # Lig update
        va = self.a_mt(gr, vr_init, ga, va_init)

        # MSIC
        vr = self.r_mt(ga, va_init, gr, vr_init)
        vv = vr.permute(1, 0).unsqueeze(0)
        vv = self.dilated_block_a(vv)
        vv = vv.squeeze(0).permute(1, 0)
        vr = vr + vv

        # Update pocket-ligand node features
        vi_init = torch.cat((va + vi_a, vr + vi_r), dim=0)

        # Learn the features of ligand with Adaptive GCN
        va = F.leaky_relu(self.A(va), 0.1)
        sa = self.sum_pool(ga, va)
        va, sa = self.a_conv1(ga, va, ea, sa)
        va, sa = self.a_conv2(ga, va + va_init, ea, sa)
        ca = torch.cat((self.mean_pool(ga, va), sa), dim=-1)
        ca = self.B(ca)
        ca = ca + self.mean_pool(ga, va_init)

        # Learn the features of protein with Dilated GCN
        vr = F.leaky_relu(self.C(vr), 0.1)
        sr = self.sum_pool(gr, vr)
        vr, sr = self.r_conv1(gr, vr, torch.Tensor().reshape(gr.num_edges(), -1).to(device), sr)
        vr, sr = self.r_conv2(gr, vr + vr_init, torch.Tensor().reshape(gr.num_edges(), -1).to(device), sr)
        cr = torch.cat((self.mean_pool(gr, vr), sr), dim=-1)
        cr = self.D(cr)
        cr = cr + self.mean_pool(gr, vr_init)

        # Learn the features of pocket-ligand with GCN
        vi = self.i_conf(gii, vi_init, ei)
        vi = vi + vi_init
        vi1 = self.i_conf(gii, vi, ei)
        vi = vi + vi1
        vi = self.i_conf(gii, vi, ei)
        ci = self.mean_pool(gii, vi)

        #  Predict OutPut
        fusion = torch.cat((ca, cr, ci, vina), dim=-1)
        output = self.classifier(fusion)

        return output


class Lig_Update(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        self.A = nn.Linear(in_dim, 64)
        self.B = nn.Linear(in_dim, 8)
        self.C = nn.Linear(64, in_dim)
        self.sum_pool = dglnn.SumPooling()
        self.D = nn.Linear(in_dim, 120)
        self.E = nn.Linear(in_dim, 120)

    def forward(self, ga, va, gb, vb):
        s = self.A(va)
        h = self.B(va)
        with ga.local_scope():
            ga.ndata['h'] = h
            h = dgl.softmax_nodes(ga, 'h')
            ga.ndata['h'] = h
            ga.ndata['s'] = s
            gga = dgl.unbatch(ga)
            glo = torch.stack([torch.mm(g.ndata['s'].T, g.ndata['h']) for g in gga])
            glo = self.C(glo.mean(dim=-1))
        glo2 = self.D(glo)
        glo3 = dgl.broadcast_nodes(gb, glo2)
        glo3 = glo3.permute(1, 0)
        r_ = torch.sum(torch.mm(self.E(vb), glo3), dim=-1)
        r = torch.sigmoid(r_)
        vbb = vb + vb * r.unsqueeze(1)
        return vbb


class Pro_Update(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        self.A = nn.Linear(in_dim, 64)
        self.B = nn.Linear(in_dim, 8)
        self.C = nn.Linear(64, in_dim)
        self.sum_pool = dglnn.SumPooling()
        self.D = nn.Linear(in_dim, 120)
        self.E = nn.Linear(in_dim, 120)

    def forward(self, ga, va, gb, vb):
        s = self.A(va)
        h = self.B(va)
        with ga.local_scope():
            ga.ndata['h'] = h
            h = dgl.softmax_nodes(ga, 'h')
            ga.ndata['h'] = h
            ga.ndata['s'] = s
            gga = dgl.unbatch(ga)
            glo = torch.stack([torch.mm(g.ndata['s'].T, g.ndata['h']) for g in gga])
            glo = self.C(glo.mean(dim=-1))
        glo2 = self.D(glo)
        glo3 = dgl.broadcast_nodes(gb, glo2)
        glo3 = glo3.permute(1, 0)
        r_ = torch.sum(torch.mm(self.E(vb), glo3), dim=-1)
        r = torch.sigmoid(r_)
        vbb = vb + vb * r.unsqueeze(1)
        return vbb


class Adaptive_GCN(nn.Module):
    def __init__(self, v_dim, e_dim, h_dim, k_head):
        super().__init__()
        self.A = nn.Linear(v_dim, h_dim)
        self.m2s = nn.ModuleList([Adaptive_GCN.Multi_scale_interaction(v_dim, h_dim) for _ in range(k_head)])
        self.B = nn.Linear(h_dim * k_head, h_dim)
        self.C = nn.Linear(v_dim, h_dim)
        self.D = nn.Linear(e_dim + v_dim, h_dim)
        self.E = nn.Linear(h_dim + v_dim, h_dim)
        self.K = nn.Linear(e_dim, h_dim)
        self.gate_update_m = Adaptive_GCN.Adaptive_GRU_T(h_dim)
        self.gate_update_s = Adaptive_GCN.Adaptive_GRU_T(h_dim)
        self.M = nn.Linear(240, 120)
        self.dilated_block_b = DilatedConvNet_T(h_dim, h_dim)

    def __msg_func(self, edges):
        v = edges.src['v']
        e = edges.data['e']
        return {'ve': F.leaky_relu(self.K(e) * v, 0.1)}

    def forward(self, g, v, e, s):
        s2s = torch.tanh(self.A(s))
        m2s = torch.cat([layer(g, v, s) for layer in self.m2s], dim=1)
        m2s = torch.tanh(self.B(m2s))
        s2m = torch.tanh(self.C(s))
        s2m = dgl.broadcast_nodes(g, s2m)
        with g.local_scope():
            g.ndata['v'] = v
            g.edata['e'] = e
            g.update_all(self.__msg_func, dglfn.sum('ve', 'sve'))
            svev = torch.cat((g.ndata['sve'], v), dim=1)
        m2m = F.leaky_relu(self.E(svev), 0.1)
        update_v = self.gate_update_m(m2m, s2m, v)
        update_s = self.gate_update_s(s2s, m2s, s)

        return update_v, update_s

    class Multi_scale_interaction(nn.Module):
        def __init__(self, v_dim, h_dim):
            super().__init__()
            self.A = nn.Linear(v_dim, h_dim)
            self.B = nn.Linear(v_dim, h_dim)
            self.C = nn.Linear(h_dim, 1)
            self.D = nn.Linear(v_dim, h_dim)

        def forward(self, g, v, s):
            d_node = torch.tanh(self.A(v))
            d_super = torch.tanh(self.B(s))
            d_super = dgl.broadcast_nodes(g, d_super)
            a = self.C(d_node * d_super).reshape(-1)

            with g.local_scope():
                g.ndata['a'] = a
                a = dgl.softmax_nodes(g, 'a')
                g.ndata['h'] = self.D(v) * a.unsqueeze(1)
                main2super_i = dgl.sum_nodes(g, 'h')

            return main2super_i

    class Adaptive_GRU_T(nn.Module):

        def __init__(self, h_dim):
            super().__init__()
            self.A = nn.Linear(h_dim, h_dim)
            self.B = nn.Linear(h_dim, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim)

        def forward(self, a, b, c):
            z = torch.sigmoid(self.A(a) + self.B(b))
            h = z * b + (1 - z) * a
            nor = self.gru(c, h)
            return nor


class Dilated_GCN(nn.Module):
    def __init__(self, v_dim, e_dim, h_dim, k_head):
        super().__init__()

        self.A = nn.Linear(v_dim, h_dim)
        self.m2s = nn.ModuleList([Dilated_GCN.Multi_scale_interaction(v_dim, h_dim) for _ in range(k_head)])
        self.B = nn.Linear(h_dim * k_head, h_dim)
        self.C = nn.Linear(v_dim, h_dim)
        self.D = nn.Linear(e_dim + v_dim, h_dim)
        self.E = nn.Linear(h_dim + v_dim, h_dim)
        self.gate_update_m = Dilated_GCN.Adaptive_GRU_T(h_dim)
        self.gate_update_s = Dilated_GCN.Adaptive_GRU_T(h_dim)
        self.dilated_block_a = DilatedConvNet_S(h_dim, h_dim)

    def __msg_func(self, edges):
        v = edges.src['v']
        return {'ve': F.leaky_relu(v, 0.1)}

    def forward(self, g, v, e, s):
        s2s = torch.tanh(self.A(s))
        m2s = torch.cat([layer(g, v, s) for layer in self.m2s], dim=1)
        m2s = torch.tanh(self.B(m2s))
        s2m = torch.tanh(self.C(s))
        s2m = dgl.broadcast_nodes(g, s2m)

        with g.local_scope():
            g.ndata['v'] = v
            g.edata['e'] = e
            g.update_all(self.__msg_func, dglfn.sum('ve', 'sve'))
            svev = torch.cat((g.ndata['sve'], v), dim=1)
        m2m = F.leaky_relu(self.E(svev), 0.1)
        vv1 = self.gate_update_m(m2m, s2m, v)
        update_s = self.gate_update_s(s2s, m2s, s)
        vv = vv1.permute(1, 0).unsqueeze(0)
        vv = self.dilated_block_a(vv)
        vv = vv.squeeze(0).permute(1, 0)
        update_v = vv + vv1

        return update_v, update_s

    class Multi_scale_interaction(nn.Module):
        def __init__(self, v_dim, h_dim):
            super().__init__()
            self.A = nn.Linear(v_dim, h_dim)
            self.B = nn.Linear(v_dim, h_dim)
            self.C = nn.Linear(h_dim, 1)
            self.D = nn.Linear(v_dim, h_dim)

        def forward(self, g, v, s):
            d_node = torch.tanh(self.A(v))
            d_super = torch.tanh(self.B(s))
            d_super = dgl.broadcast_nodes(g, d_super)
            a = self.C(d_node * d_super).reshape(-1)

            with g.local_scope():
                g.ndata['a'] = a
                a = dgl.softmax_nodes(g, 'a')
                g.ndata['h'] = self.D(v) * a.unsqueeze(1)
                main2super_i = dgl.sum_nodes(g, 'h')

            return main2super_i

    class Adaptive_GRU_T(nn.Module):
        def __init__(self, h_dim):
            super().__init__()
            self.A = nn.Linear(h_dim, h_dim)
            self.B = nn.Linear(h_dim, h_dim)
            self.gru = nn.GRUCell(h_dim, h_dim)

        def forward(self, a, b, c):
            z = torch.sigmoid(self.A(a) + self.B(b))
            h = z * b + (1 - z) * a
            nor = self.gru(c, h)
            return nor


class GCN(nn.Module):
    def __init__(self, dim, rc, depth):
        super().__init__()
        self.rs = nn.Parameter(torch.rand(1))
        self.sigma = nn.Parameter(torch.rand(1))
        self.A = nn.Linear(dim, dim)
        self.rc = rc
        self.depth = depth

    def f(self, r):
        return torch.exp((-torch.square(r - self.rs) / torch.square(self.sigma))) * \
               0.5 * torch.cos(np.pi * r / self.rc) * (r < self.rc)

    def __msg_func(self, edges):
        v = edges.src['v']
        f = edges.data['f']
        return {'vf': f * v}

    def forward(self, g, v, e):
        with g.local_scope():
            g.ndata['v'] = v
            g.edata['f'] = self.f(e)
            for _ in range(self.depth):
                g.update_all(self.__msg_func, dglfn.sum('vf', 'svf'))
                g.ndata['v'] = torch.relu(self.A(g.ndata['svf'] + v))
            new_v = g.ndata['v']

        return new_v


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output


class DilatedConvNet(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)
        dc1 = self.d1(output1)
        dc2 = self.d2(output1)
        dc4 = self.d4(output1)
        dc8 = self.d8(output1)
        dc16 = self.d16(output1)
        # Multi-Scale Fusion Strategy
        add1 = dc2
        add2 = add1 + dc4
        add3 = add2 + dc8
        add4 = add3 + dc16
        combine = torch.cat([dc1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        n = self.br2(combine)
        return n


class DilatedConvNet_S(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        # Multi-Scale Fusion Strategy
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        n = self.br2(combine)
        return n


class DilatedConvNet_T(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        # Multi-Scale Fusion Strategy
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        c = torch.cat([d1, add1, add2, add3], 1)
        if self.add:
            c = input + c
        output = self.br2(c)
        return output

# Ml = ModelNew()
# print(Ml)
