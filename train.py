import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


CSV_PATH = "clothing_catalog_clean.csv"
IMAGE_DIR = os.path.join("dataset", "images")

BATCH_SIZE = 13
EPOCHS = 6
LEARNING_RATE = 1e-3

TEXT_DIM = 78

maps = {}

extra_embeddings = {}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


df = pd.read_csv(CSV_PATH)

profile_columns = [
    "clothing_type_clean",
    "neckline_clean",
    "pattern_clean",
    "waistline_clean",
    "sleeves_clean",
    "fabric_clean",
    "silhouette_clean"
]

extra_columns = [
    "extra1",
    "extra2",
    "extra3"
]


for column in profile_columns + extra_columns:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )



for column in profile_columns:

    values = df[column].unique()

    maps[column] = {
        value: index
        for index, value in enumerate(values)
    }


class_dims = [
    len(maps[column])
    for column in profile_columns
]


joblib.dump(
    maps,
    "attribute_maps.pkl"
)



for column in extra_columns:

    vectorizer = TfidfVectorizer(
        max_features=TEXT_DIM,
        stop_words="english"
    )

    text = df[column].replace(
        "",
        "no additional detail"
    )

    embeddings = vectorizer.fit_transform(
        text
    ).toarray().astype(np.float32)

    extra_embeddings[column] = embeddings

    joblib.dump(
        vectorizer,
        f"{column}_vectorizer.pkl"
    )

    np.save(
        f"{column}_embeddings.npy",
        embeddings
    )



class ClothingDataset(Dataset):

    def __init__(
        self,
        dataframe,
        extra_embeddings,
        image_dir,
        transform=None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.extra_embeddings = extra_embeddings
        self.image_dir = image_dir
        self.transform = transform

        self.labels = torch.tensor(
            [
                [
                    maps[column][row[column]]
                    for column in profile_columns
                ]
                for _, row in self.df.iterrows()
            ],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_name = self.df.iloc[index]["image_name"]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        extras = []

        for column in extra_columns:

            extras.append(
                torch.from_numpy(
                    self.extra_embeddings[column][index]
                )
            )

        return (
            image,
            self.labels[index],
            *extras
        )


class FashionModel(nn.Module):

    def __init__(
        self,
        class_dims,
        text_dim
    ):
        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        features = backbone.fc.in_features

        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        
        self.classifier = nn.Linear(
            features,
            sum(class_dims)
        )

        self.extra1_head = nn.Sequential(
            nn.Linear(features, 128),
            nn.ReLU(),
            nn.Linear(128, text_dim)
        )

        self.extra2_head = nn.Sequential(
            nn.Linear(features, 128),
            nn.ReLU(),
            nn.Linear(128, text_dim)
        )

        self.extra3_head = nn.Sequential(
            nn.Linear(features, 128),
            nn.ReLU(),
            nn.Linear(128, text_dim)
        )

    def forward(self, image):

        features = torch.flatten(
            self.backbone(image),
            1
        )

        class_output = self.classifier(
            features
        )

        extra1 = self.extra1_head(
            features
        )

        extra2 = self.extra2_head(
            features
        )

        extra3 = self.extra3_head(
            features
        )

        return (
            class_output,
            extra1,
            extra2,
            extra3
        )


if __name__ == "__main__":

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    indices = np.arange(len(df))

    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    train_size = int(
        0.85 * len(indices)
    )

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]


    train_df = df.iloc[
        train_indices
    ].reset_index(drop=True)

    val_df = df.iloc[
        val_indices
    ].reset_index(drop=True)


    train_extras = {
        column: extra_embeddings[column][train_indices]
        for column in extra_columns
    }

    val_extras = {
        column: extra_embeddings[column][val_indices]
        for column in extra_columns
    }


    train_dataset = ClothingDataset(
        train_df,
        train_extras,
        IMAGE_DIR,
        transform=train_transforms
    )

    val_dataset = ClothingDataset(
        val_df,
        val_extras,
        IMAGE_DIR,
        transform=val_transforms
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    model = FashionModel(
        class_dims,
        TEXT_DIM
    ).to(DEVICE)


    classification_loss = nn.CrossEntropyLoss()
    text_loss = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )


    print("Starting training...")


    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0


        for (
            images,
            labels,
            extra1,
            extra2,
            extra3
        ) in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            extra1 = extra1.to(DEVICE)
            extra2 = extra2.to(DEVICE)
            extra3 = extra3.to(DEVICE)


            optimizer.zero_grad(
                set_to_none=True
            )


            (
                class_output,
                extra1_output,
                extra2_output,
                extra3_output
            ) = model(images)


            loss = 0.0

            start = 0


            for i, size in enumerate(class_dims):

                end = start + size

                loss += classification_loss(
                    class_output[:, start:end],
                    labels[:, i]
                )

                start = end


            loss += text_loss(
                extra1_output,
                extra1
            ) * 10.0

            loss += text_loss(
                extra2_output,
                extra2
            ) * 10.0

            loss += text_loss(
                extra3_output,
                extra3
            ) * 10.0


            loss.backward()
            optimizer.step()


            train_loss += loss.item()


        
        model.eval()

        validation_loss = 0.0


        with torch.no_grad():

            for (
                images,
                labels,
                extra1,
                extra2,
                extra3
            ) in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                extra1 = extra1.to(DEVICE)
                extra2 = extra2.to(DEVICE)
                extra3 = extra3.to(DEVICE)


                (
                    class_output,
                    extra1_output,
                    extra2_output,
                    extra3_output
                ) = model(images)


                loss = 0.0

                start = 0


                for i, size in enumerate(class_dims):

                    end = start + size

                    loss += classification_loss(
                        class_output[:, start:end],
                        labels[:, i]
                    )

                    start = end


                loss += text_loss(
                    extra1_output,
                    extra1
                ) * 10.0

                loss += text_loss(
                    extra2_output,
                    extra2
                ) * 10.0

                loss += text_loss(
                    extra3_output,
                    extra3
                ) * 10.0


                validation_loss += loss.item()


        train_loss /= len(train_loader)
        validation_loss /= len(val_loader)


        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| train: {train_loss:.4f} "
            f"| val: {validation_loss:.4f}"
        )


    torch.save(
        model.state_dict(),
        "fashion_tagger_model.pth"
    )


    print("\nModel saved.")
