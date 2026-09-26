# =====================================================
# anomaly_detector.py
# Part 1
# =====================================================

from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

import streamlit as st

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =====================================================
# 定数
# =====================================================

IMAGE_SIZE = 256

DEFAULT_BATCH_SIZE = 16

DEFAULT_LEARNING_RATE = 0.001

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# =====================================================
# Utility
# =====================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_image_files(
    folder: Path
):

    images = []

    if not folder.exists():

        return images

    for file in folder.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            images.append(file)

    images.sort()

    return images


def find_bottle_root(
    input_path: str
):

    root = (
        Path(input_path)
        .expanduser()
        .resolve()
    )

    candidates = [

        root,

        root / "bottle",

        root
        / "mvtec_anomaly_detection"
        / "bottle",

    ]

    for candidate in candidates:

        train_dir = (
            candidate
            / "train"
            / "good"
        )

        test_dir = (
            candidate
            / "test"
        )

        if (
            train_dir.exists()
            and test_dir.exists()
        ):
            return candidate

    raise FileNotFoundError(
        "MVTec AD bottle フォルダが見つかりません"
    )


# =====================================================
# Dataset
# =====================================================

class MVTecBottleDataset(
    Dataset
):

    def __init__(

        self,

        samples: List[
            Tuple[Path, int]
        ],

        image_size=IMAGE_SIZE,

    ):

        self.samples = samples

        self.transform = transforms.Compose(

            [

                transforms.Resize(
                    (
                        image_size,
                        image_size
                    )
                ),

                transforms.ToTensor(),

            ]

        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(
        self,
        index
    ):

        image_path, label = (
            self.samples[index]
        )

        image = (
            Image
            .open(image_path)
            .convert("RGB")
        )

        image = self.transform(
            image
        )

        return (

            image,

            label,

            str(image_path),

        )


# =====================================================
# Train Dataset
# =====================================================

def build_train_dataset(
    bottle_root: Path
):

    train_dir = (

        bottle_root
        / "train"
        / "good"

    )

    image_files = get_image_files(
        train_dir
    )

    samples = []

    for image_path in image_files:

        samples.append(
            (
                image_path,
                0
            )
        )

    dataset = (
        MVTecBottleDataset(
            samples=samples
        )
    )

    return dataset


# =====================================================
# Test Dataset
# =====================================================

def build_test_dataset(
    bottle_root: Path
):

    test_root = (
        bottle_root
        / "test"
    )

    samples = []

    categories = sorted(
        [
            folder
            for folder
            in test_root.iterdir()
            if folder.is_dir()
        ]
    )

    for category in categories:

        if category.name == "good":

            label = 0

        else:

            label = 1

        image_files = (
            get_image_files(
                category
            )
        )

        for image_path in image_files:

            samples.append(

                (
                    image_path,
                    label
                )

            )

    dataset = (
        MVTecBottleDataset(
            samples=samples
        )
    )

    return dataset


# =====================================================
# DataLoader
# =====================================================

def build_train_loader(
    bottle_root,
    batch_size=DEFAULT_BATCH_SIZE
):

    dataset = (
        build_train_dataset(
            bottle_root
        )
    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,

    )

    return loader


def build_test_loader(
    bottle_root,
    batch_size=DEFAULT_BATCH_SIZE
):

    dataset = (
        build_test_dataset(
            bottle_root
        )
    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,

    )

    return loader


# =====================================================
# AutoEncoder
# =====================================================

class AutoEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
# =====================================================
# AutoEncoder (continued)
# =====================================================

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            nn.Sigmoid(),

        )

    def forward(
        self,
        x
    ):

        latent = self.encoder(x)

        reconstructed = (
            self.decoder(
                latent
            )
        )

        return reconstructed


# =====================================================
# モデル保存
# =====================================================

def save_model(

    model,

    save_path,

    image_size=256,

):

    checkpoint = {

        "image_size":
        image_size,

        "model_state_dict":
        model.state_dict(),

    }

    torch.save(
        checkpoint,
        save_path
    )


# =====================================================
# モデル読込
# =====================================================

def load_model(

    model_path,

    device,

):

    checkpoint = torch.load(

        model_path,

        map_location=device,

    )

    model = (
        AutoEncoder()
        .to(device)
    )

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )

    model.eval()

    return model


# =====================================================
# 異常マップ
# =====================================================

def anomaly_map(

    original,

    reconstructed,

):

    """
    Pixel-wise MSE
    """

    error_map = torch.mean(

        (original - reconstructed) ** 2,

        dim=1,

    )

    return error_map


# =====================================================
# 異常スコア
# =====================================================

def anomaly_score(

    original,

    reconstructed,

):

    amap = anomaly_map(

        original,
        reconstructed

    )

    score = (

        amap

        .flatten(start_dim=1)

        .mean(dim=1)

    )

    return score


# =====================================================
# ヒートマップ作成
# =====================================================

def build_heatmap(

    original,

    reconstructed,

):

    heatmap = (

        anomaly_map(

            original,

            reconstructed,

        )

        .squeeze()

        .cpu()

        .numpy()

    )

    heatmap = (

        heatmap
        - heatmap.min()

    ) / (

        heatmap.max()
        - heatmap.min()
        + 1e-8

    )

    return heatmap


# =====================================================
# 学習
# =====================================================

def train_model(

    model,

    train_loader,

    epochs,

    learning_rate,

    device,

    logger=None,

    progress_bar=None,

    status_text=None,
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate

    )

    history = []

    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        total_images = 0

        for (

            images,
            _,
            _

        ) in train_loader:

            images = (
                images.to(device)
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(

                outputs,

                images,

            )

            loss.backward()

            optimizer.step()

            running_loss += (

                loss.item()

                * images.size(0)

            )

            total_images += (
                images.size(0)
            )

        epoch_loss = (

            running_loss

            / total_images

        )

        history.append(
            epoch_loss
        )

        if logger:

            logger(
                f"Epoch "
                f"{epoch+1}/{epochs} "
                f"Loss="
                f"{epoch_loss:.6f}"
            )
        if progress_bar:

            progress_bar.progress(
                (epoch + 1) / epochs
            )

        if status_text:

            status_text.text(
                f"Epoch {epoch+1}/{epochs} "
                f"Loss={epoch_loss:.6f}"
            )
    
    model.eval()

    return history


# =====================================================
# 推論
# =====================================================

@torch.no_grad()

def predict(

    model,

    image_tensor,

    threshold,

    device,

):

    image_tensor = (
        image_tensor
        .to(device)
    )

    model.eval()

    reconstructed = model(
        image_tensor
    )

    score = float(

        anomaly_score(

            image_tensor,

            reconstructed,

        )

        .item()

    )

    prediction = (

        1
        if score >= threshold
        else 0

    )

    return {

        "prediction":
        prediction,

        "score":
        score,

        "reconstructed":
        reconstructed,

    }


# =====================================================
# 評価
# =====================================================

@torch.no_grad()

def evaluate_model(

    model,

    test_loader,

    threshold,

    device,

):

    model.eval()

    labels = []

    scores = []

    predictions = []

    paths = []

    for (

        images,

        batch_labels,

        batch_paths,

    ) in test_loader:

        images = (
            images.to(device)
        )

        reconstructed = (
            model(images)
        )

        batch_scores = (

            anomaly_score(

                images,

                reconstructed

            )

            .cpu()

            .numpy()

        )

        for i in range(
            len(batch_scores)
        ):

            score = float(
                batch_scores[i]
            )

            label = int(
                batch_labels[i]
            )

            pred = (
                1
                if score >= threshold
                else 0
            )

            labels.append(
                label
            )

            scores.append(
                score
            )

            predictions.append(
                pred
            )

            paths.append(
                batch_paths[i]
            )

# Part 3へ続く
    # =================================
    # 評価指標計算
    # =================================

    metrics = {

        "Accuracy":

        accuracy_score(
            labels,
            predictions
        ),

        "Precision":

        precision_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "Recall":

        recall_score(
            labels,
            predictions,
            zero_division=0,
        ),

        "F1-score":

        f1_score(
            labels,
            predictions,
            zero_division=0,
        ),

    }

    confusion = (

        confusion_matrix(

            labels,

            predictions,

            labels=[0, 1]

        )

        .astype(int)

        .tolist()

    )

    result = {

        "metrics":
        metrics,

        "labels":
        labels,

        "scores":
        scores,

        "predictions":
        predictions,

        "paths":
        paths,

        "confusion":
        confusion,

    }

    return result


# =====================================================
# Streamlit
# =====================================================

st.set_page_config(

    page_title=
    "Bottle Anomaly Detector",

    page_icon="🔍",

    layout="wide",

)

st.title(
    "Bottle Anomaly Detector"
)

st.caption(
    "MVTec AD Bottle異常検知"
)

# =====================================================
# Session State
# =====================================================

if "logs" not in st.session_state:

    st.session_state.logs = []

if "model" not in st.session_state:

    st.session_state.model = None

if "history" not in st.session_state:

    st.session_state.history = []


def add_log(msg):

    st.session_state.logs.append(
        msg
    )


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.header("設定")

    dataset_path = st.text_input(

        "MVTec AD Path",

        value=
        "data/mvtec_anomaly_detection/bottle"

    )

    epochs = st.number_input(

        "Epoch",

        min_value=1,

        max_value=1000,

        value=30,

    )

    batch_size = st.number_input(

        "Batch Size",

        min_value=1,

        max_value=256,

        value=16,

    )

    learning_rate = st.number_input(

        "Learning Rate",

        value=0.001,

        format="%.6f",

    )

    threshold = st.number_input(

        "Threshold",

        value=0.01,

        format="%.6f",

    )

    st.info(
        f"Device : {get_device()}"
    )

# =====================================================
# Tab
# =====================================================

train_tab, eval_tab, infer_tab, model_tab = st.tabs(

    [

        "Train",

        "Evaluate",

        "Inference",

        "Model",

    ]

)

# =====================================================
# Train
# =====================================================

with train_tab:

    st.subheader(
        "Model Training"
    )

    if st.button(

        "Start Training",

        use_container_width=True,

    ):
    
        try:

            root = find_bottle_root(
                dataset_path
            )

            loader = build_train_loader(

                bottle_root=root,

                batch_size=
                batch_size,

            )

            model = (
                AutoEncoder()
                .to(get_device())
            )

            progress_bar = st.progress(0)

            status_text = st.empty()
            
            add_log(
                "Training Started"
            )

            history = train_model(

                model=model,

                train_loader=loader,

                epochs=epochs,

                learning_rate=
                learning_rate,

                device=get_device(),

                logger=add_log,
                
                progress_bar=
                progress_bar,
                
                status_text=
                status_text,
            )

            st.session_state.model = model

            st.session_state.history = (
                history
            )

            st.success(
                "Training Complete"
            )

        except Exception as e:

            st.exception(e)

    if len(
        st.session_state.history
    ) > 0:

        loss_df = pd.DataFrame({

            "loss":
            st.session_state.history

        })

        st.line_chart(
            loss_df
        )

# =====================================================
# Evaluation
# =====================================================

with eval_tab:

    st.subheader(
        "Model Evaluation"
    )

    if st.button(

        "Evaluate",

        use_container_width=True,

    ):

        if (
            st.session_state.model
            is None
        ):

            st.warning(
                "Train model first"
            )

        else:

            root = find_bottle_root(
                dataset_path
            )

            test_loader = (

                build_test_loader(

                    bottle_root=root,

                    batch_size=
                    batch_size,

                )

            )

            result = evaluate_model(

                model=
                st.session_state.model,

                test_loader=
                test_loader,

                threshold=
                threshold,

                device=
                get_device(),

            )

            cols = st.columns(4)

            metric_names = [

                "Accuracy",

                "Precision",

                "Recall",

                "F1-score",

            ]

            for col, name in zip(

                cols,
                metric_names

            ):

                col.metric(

                    name,

                    f"{result['metrics'][name]:.4f}"

                )

            st.write(
                "### Confusion Matrix"
            )

            cm = pd.DataFrame(

                result[
                    "confusion"
                ],

                index=[
                    "Actual Good",
                    "Actual Defect",
                ],

                columns=[
                    "Pred Good",
                    "Pred Defect",
                ],

            )

            st.dataframe(
                cm,
                use_container_width=True,
            )

            detail_df = pd.DataFrame({

                "Path":
                result["paths"],

                "Label":
                result["labels"],

                "Prediction":
                result["predictions"],

                "Score":
                result["scores"],

            })

            st.write(
                "### Detail"
            )

            st.dataframe(

                detail_df,

                use_container_width=True,

            )

# =====================================================
# Inference
# =====================================================

with infer_tab:

    st.subheader(
        "Single Image Inference"
    )

    uploaded_file = st.file_uploader(

        "Image",

        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
        ]

    )

    if (
        uploaded_file is not None
        and
        st.session_state.model
        is not None
    ):

        image = (
            Image.open(
                uploaded_file
            )
            .convert("RGB")
        )

        transform = transforms.Compose([

            transforms.Resize(

                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )

            ),

            transforms.ToTensor(),

        ])

        image_tensor = (

            transform(image)

            .unsqueeze(0)

            .to(get_device())

        )

        result = predict(

            model=
            st.session_state.model,

            image_tensor=
            image_tensor,

            threshold=
            threshold,

            device=
            get_device(),

        )

        reconstructed = (

            result[
                "reconstructed"
            ]

        )

        heatmap = build_heatmap(

            image_tensor,

            reconstructed,

        )

        col1, col2, col3 = st.columns(3)

        col1.image(

            image,

            caption=
            "Input Image",

            use_container_width=True,

        )

        recon_img = (

            reconstructed

            .squeeze(0)

            .permute(1, 2, 0)

            .cpu()

            .numpy()

        )

        col2.image(

            recon_img,

            caption=
            "Reconstruction",

            use_container_width=True,

        )

        fig, ax = plt.subplots(
            figsize=(5, 5)
        )

        ax.imshow(heatmap)

        ax.axis("off")

        col3.pyplot(fig)

        label = (

            "Defect"

            if result[
                "prediction"
            ] == 1

            else

            "Good"

        )

        st.metric(

            "Prediction",

            label

        )

        st.metric(

            "Anomaly Score",

            f"{result['score']:.6f}"

        )

# =====================================================
# Model Save / Load
# =====================================================

with model_tab:

    st.subheader(
        "Model Save / Load"
    )

    model_path = st.text_input(

        "Model Path",

        value=
        "bottle_autoencoder.pth"

    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Save Model"
        ):

            if (
                st.session_state.model
                is not None
            ):

                save_model(

                    st.session_state.model,

                    model_path,

                )

                st.success(
                    "Saved"
                )

    with col2:

        if st.button(
            "Load Model"
        ):

            model = load_model(

                model_path,

                get_device(),

            )

            st.session_state.model = model

            st.success(
                "Loaded"
            )

# =====================================================
# Log
# =====================================================

st.divider()

st.subheader(
    "Processing Log"
)

if len(
    st.session_state.logs
) > 0:

    st.code(

        "\n".join(
            st.session_state.logs
        )

    )

else:

    st.info(
        "No logs"
    )
