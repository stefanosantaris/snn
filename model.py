import torch.nn as nn
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import torch
from snntorch import surrogate

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Initialize layers
        self.fc1 = nn.Linear(self.config.num_inputs, self.config.num_hidden)
        self.lif1 = snn.Leaky(beta=self.config.beta)
        self.fc2 = nn.Linear(self.config.num_hidden, self.config.num_outputs)
        self.lif2 = snn.Leaky(beta=self.config.beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(self.config.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_m = nn.Dropout(p=0.2)
            cur2 = self.fc2(spk1_m(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            m = nn.Sigmoid()
            spk2_rec.append(m(spk2))
            mem2_rec.append(m(mem2))
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
