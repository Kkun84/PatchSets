from logging import getLogger
import hydra
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

from src.patch import make_patch2d


logger = getLogger(__name__)


class Encoder(pl.LightningModule):

    def __init__(self, input_shape, hidden_n_0, hidden_n_1, output_n):
        logger.debug(f"Encoder.__init__()")
        super().__init__()
        self.hparams = {}

        self.input_n = np.prod([input_shape[i] for i in range(len(input_shape))])
        self.output_n = output_n

        self.fc0 = torch.nn.Linear(self.input_n, hidden_n_0)
        self.fc1 = torch.nn.Linear(hidden_n_0, hidden_n_1)
        self.fc2 = torch.nn.Linear(hidden_n_1, output_n)

    def forward(self, input):
        # [batch, sets, channels, x, y]
        logger.debug(f"input.shape={input.shape}")
        logger.debug([input.shape[0]*input.shape[1], self.input_n])
        x = input.reshape([input.shape[0]*input.shape[1], self.input_n])
        logger.debug(f"reshape-x.shape={x.shape}")
        x = self.fc0(x).relu()
        x = self.fc1(x).relu()
        x = self.fc2(x)
        logger.debug(f"linear-x.shape={x.shape}")
        x = x.reshape([input.shape[0], input.shape[1], self.output_n])
        logger.debug(f"reshape-x.shape={x.shape}")
        # [batch, lattent]
        return x


class Decoder(pl.LightningModule):

    def __init__(self, input_n, hidden_n_0, hidden_n_1, output_n, use_tanh):
        logger.debug(f"Decoder.__init__()")
        super().__init__()
        self.hparams = {}
        self.input_n = input_n
        self.output_n = output_n
        self.use_tanh = use_tanh

        self.fc0 = torch.nn.Linear(input_n, hidden_n_0)
        self.fc1 = torch.nn.Linear(hidden_n_0, hidden_n_1)
        self.fc2 = torch.nn.Linear(hidden_n_1, output_n)

    def forward(self, input):
        # [batch, lattent]
        logger.debug(f"input.shape={input.shape}")
        x = input
        if self.use_tanh:
            x = x.tanh()
        x = self.fc0(x).relu()
        x = self.fc1(x).relu()
        x = self.fc2(x)
        logger.debug(f"linear-x.shape={x.shape}")
        return x
