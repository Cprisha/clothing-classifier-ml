import os

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image

from train import FashionModel


IMAGE_PATH = "testsubject.jpg"
CSV_PATH = "clothing_catalog_clean.csv"
MODEL_PATH = "fashion_tagger_model.pth"
MAPS_PATH = "attribute_maps.pkl"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


maps = joblib.load(
    MAPS_PATH
)

class_dims = [
    len(maps[column])
    for column in profile_columns
]


df = pd.read_csv(
    CSV_PATH
)

for column in extra_columns:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


extra_embeddings = {}

for column in extra_columns:

    path = f"{column}_embeddings.npy"

    extra_embeddings[column] = torch.from_numpy(
        np.load(path)
    ).to(DEVICE)


model = FashionModel(
    class_dims,
    extra_embeddings["extra1"].shape[1]
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(image_path):

    if not os.path.exists(image_path):
        print(f"Could not find: {image_path}")
        return None

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = test_transforms(image)

    return image.unsqueeze(0).to(DEVICE)


def get_predictions(class_output):

    predictions = {}

    start = 0

    for column, size in zip(
        profile_columns,
        class_dims
    ):

        end = start + size

        output = class_output[
            0,
            start:end
        ]

        predicted_id = torch.argmax(
            output
        ).item()

        reverse_map = {
            number: value
            for value, number in maps[column].items()
        }

        value = reverse_map[predicted_id]

        if value in ["", "unknown"]:
            value = "N/A"

        predictions[column] = value

        start = end

    return predictions


def find_extra(extra_output, column):

    catalog = extra_embeddings[column]

    extra_output = extra_output / (
        extra_output.norm(
            dim=1,
            keepdim=True
        ) + 1e-8
    )

    catalog = catalog / (
        catalog.norm(
            dim=1,
            keepdim=True
        ) + 1e-8
    )

    scores = torch.mm(
        extra_output,
        catalog.t()
    )[0]

    valid_rows = df[column].str.strip() != ""

    scores = scores.clone()

    scores[
        torch.tensor(
            (~valid_rows).values,
            device=DEVICE
        )
    ] = -1

    best_index = torch.argmax(
        scores
    ).item()

    value = df.iloc[
        best_index
    ][column]

    if not value or str(value).strip() == "":
        return "N/A"

    return str(value).strip()


def run_test(image_path):

    image = load_image(
        image_path
    )

    if image is None:
        return


    with torch.no_grad():

        (
            class_output,
            extra1_output,
            extra2_output,
            extra3_output
        ) = model(image)


    predictions = get_predictions(
        class_output
    )


    extra1 = find_extra(
        extra1_output,
        "extra1"
    )

    extra2 = find_extra(
        extra2_output,
        "extra2"
    )

    extra3 = find_extra(
        extra3_output,
        "extra3"
    )


    print()
    print("==========================================")
    print("Clothing description")
    print("==========================================")

    labels = {
        "clothing_type_clean": "Type",
        "neckline_clean": "Neckline",
        "pattern_clean": "Pattern",
        "waistline_clean": "Waistline",
        "sleeves_clean": "Sleeves",
        "fabric_clean": "Fabric",
        "silhouette_clean": "Silhouette"
    }


    for column in profile_columns:

        print(
            f"{labels[column]:11}: "
            f"{predictions[column]}"
        )


    print()
    print("Additional details")
    print("------------------------------------------")

    print(
        f"Extra 1     : {extra1}"
    )

    print(
        f"Extra 2     : {extra2}"
    )

    print(
        f"Extra 3     : {extra3}"
    )

    print("==========================================")
    print()


if __name__ == "__main__":
    run_test(IMAGE_PATH)
