import pytorch_lightning as pl
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn.functional as F
import hydra
import yaml
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image, ImageDraw
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from src.model import Model
from src.set_module import cutout_patch2d
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


@st.cache(allow_output_mutation=True)
def load_model(model_path: str):
    return Model.load_from_checkpoint(model_path)


@st.cache(allow_output_mutation=True)
def load_datamodule(data_config):
    datamodule = hydra.utils.instantiate(data_config)
    datamodule.target_transform = None
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def crop(image: Tensor, x: int, y: int, patch_size: int) -> Tensor:
    assert image.dim() == 3
    patch = image[:, y : y + patch_size, x : x + patch_size]
    assert patch.shape == torch.Size(
        [1, *[patch_size] * 2]
    ), f"{patch.shape} == {torch.Size([1, *[patch_size] * 2])}"
    return patch


def main():
    model_directory = Path(
        st.text_input(
            'Model directory', '/workspace/outputs/Default/2021-01-10/15-52-03'
        )
    )
    model_path = Path(
        st.selectbox('Model path', list(model_directory.glob('**/*.ckpt')))
    )
    with st.beta_expander('Display Yaml'):
        yaml_path = Path(
            st.selectbox('Yaml path', list(model_directory.glob('**/*.yaml')))
        )
        with open(yaml_path) as file:
            st.write(yaml.safe_load(file))
    with open(model_directory / '.hydra' / 'config.yaml') as file:
        config = hydra.utils.DictConfig(yaml.safe_load(file))

    model = load_model(str(model_path))

    patch_size = model.hparams.patch_size

    datamodule = load_datamodule(config.data.datamodule)
    dataset = datamodule.test_dataset

    col1, col2 = st.beta_columns(2)
    font_length = len(dataset.unique_font)
    font_index = col1.number_input(
        f'font_index (0~{font_length - 1})', 0, font_length - 1, 0
    )
    alphabet_length = len(dataset.unique_alphabet)
    alphabet_index = col2.number_input(
        f'alphabet_index (0~{alphabet_length - 1})', 0, alphabet_length - 1, 0
    )

    data_index = font_index * alphabet_length + alphabet_index

    data = dataset[data_index]
    image = data[0]
    label = data[1]

    st.sidebar.image(to_pil_image(image), use_column_width=True, output_format='PNG')

    y_list = torch.linspace(0, image.shape[1] - patch_size + 1, 8, dtype=int)
    x_list = torch.linspace(0, image.shape[2] - patch_size + 1, 8, dtype=int)

    random_mask = st.sidebar.checkbox('Random mask', True)
    shuffle = st.sidebar.checkbox('Shuffle', True)

    mask = torch.zeros([len(y_list), len(x_list)], dtype=bool)
    for i in range(len(y_list)):
        for j, col in enumerate(st.sidebar.beta_columns(len(x_list))):
            if random_mask:
                mask_value = torch.rand(1) > 0.9
            else:
                mask_value = 0
            mask[i, j] = col.checkbox('', mask_value, key=i * len(x_list) + j)

    y_list, x_list = torch.meshgrid(y_list, x_list)
    x_list = x_list[mask]
    y_list = y_list[mask]

    patch_list = []
    for x, y in zip(x_list, y_list):
        patch_list.append(crop(image, x, y, patch_size))
    patch_list = torch.stack(patch_list)
    if shuffle:
        patch_list = patch_list[torch.randperm(len(patch_list))]

    for patch, col in zip(patch_list, st.beta_columns(len(patch_list))):
        col.image(to_pil_image(patch), use_column_width=True, output_format='PNG')
    for patch in patch_list:
        st.sidebar.image(to_pil_image(patch), output_format='PNG')

    feature_list = model.encode(patch_list[None])[0]
    feature = model.pool(feature_list[None])[0]
    predicted = model.decode(feature[None])[0]
    likelihood = predicted.softmax(0)

    x, y = feature_list.cumsum(0).T
    x = torch.cat([torch.zeros_like(x[:1]), x[:-1]])
    y = torch.cat([torch.zeros_like(y[:1]), y[:-1]])

    u, v = feature_list.T * 0.97

    # max_value = feature_list.cumsum(0).abs().max()

    fig = ff.create_quiver(x, y, u, v, scale=1)
    fig.layout = go.Layout(
        autosize=False,
        width=500,
        height=500,
        yaxis=dict(scaleratio=1, scaleanchor='x'),
    )
    st.plotly_chart(fig)

    fig = px.bar(
        x=range(len(feature_list)),
        y=feature_list.norm(dim=1),
        barmode='group',
        labels={'x': 'Patch', 'y': 'L2 norm'},
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        x=range(len(feature_list)),
        y=feature_list @ feature,
        barmode='group',
        labels={'x': 'Patch', 'y': 'Inner product value'},
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        x=[i[3:4] for i in dataset.unique_alphabet],
        y=likelihood,
        labels={'x': 'Alphabet', 'y': 'Likelihood [%]'},
    )
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
