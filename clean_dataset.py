import re
import pandas as pd


INPUT_FILE = "clothing_catalog_enhanced_final.csv"
OUTPUT_FILE = "clothing_catalog_clean.csv"


df = pd.read_csv(INPUT_FILE)


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    return value


def clean_neckline(value):
    value = clean_text(value)

    if not value:
        return "unknown"

    value = value.replace("v neckline", "v-neck")
    value = value.replace("v neck", "v-neck")

    value = value.replace("off shoulder", "off-shoulder")
    value = value.replace("one shoulder", "one-shoulder")

    if "sweetheart" in value:
        return "sweetheart"

    if "v-neck" in value:
        return "v-neck"

    if "square" in value:
        return "square neck"

    if "round" in value or "crew" in value:
        return "round neck"

    if "boat" in value:
        return "boat neck"

    if "halter" in value:
        return "halter neck"

    if "cowl" in value:
        return "cowl neck"

    if "turtleneck" in value or "turtle neck" in value:
        return "turtleneck"

    if "strapless" in value:
        return "strapless"

    if "off-shoulder" in value:
        return "off-shoulder"

    if "one-shoulder" in value:
        return "one-shoulder"

    if "collar" in value:
        return "collared"

    return value


def clean_sleeves(value):
    value = clean_text(value)

    if not value:
        return "unknown"

    value = value.replace("full sleeves", "long sleeve")
    value = value.replace("long sleeves", "long sleeve")
    value = value.replace("short sleeves", "short sleeve")

    if "sleeveless" in value:
        return "sleeveless"

    if "spaghetti" in value:
        return "spaghetti straps"

    if "puff" in value:
        return "puff sleeve"

    if "bell" in value:
        return "bell sleeve"

    if "flutter" in value:
        return "flutter sleeve"

    if "balloon" in value:
        return "balloon sleeve"

    if "cap" in value:
        return "cap sleeve"

    if "three-quarter" in value or "3/4" in value:
        return "three-quarter sleeve"

    if "short" in value:
        return "short sleeve"

    if "long" in value:
        return "long sleeve"

    return value


def clean_waistline(value):
    value = clean_text(value)

    if not value:
        return "unknown"

    if "empire" in value:
        return "empire waist"

    if "high" in value:
        return "high waist"

    if "low" in value:
        return "low rise"

    if "mid" in value:
        return "mid rise"

    if "natural" in value:
        return "natural waist"

    if "corset" in value or "fitted" in value:
        return "fitted/corset"

    if "elastic" in value or "smocked" in value:
        return "elastic/smocked"

    return value


def clean_fabric(value):
    value = clean_text(value)

    if not value:
        return "unknown"

    # Keep the material itself instead of deciding whether
    # something is a blend from the description.
    if "denim" in value:
        return "denim"

    if "cotton" in value:
        return "cotton"

    if "polyester" in value:
        return "polyester"

    if "silk" in value:
        return "silk"

    if "satin" in value:
        return "satin"

    if "chiffon" in value:
        return "chiffon"

    if "georgette" in value:
        return "georgette"

    if "linen" in value:
        return "linen"

    if "velvet" in value:
        return "velvet"

    if "lace" in value:
        return "lace"

    if "tulle" in value:
        return "tulle"

    if "organza" in value:
        return "organza"

    if "taffeta" in value:
        return "taffeta"

    if "crepe" in value:
        return "crepe"

    if "leather" in value:
        return "leather"

    if "wool" in value:
        return "wool"

    if "knit" in value:
        return "knit"

    return value


def clean_silhouette(value):
    value = clean_text(value)

    if not value:
        return "unknown"

    if "a-line" in value or "a line" in value:
        return "a-line"

    if "fit and flare" in value or "fit-and-flare" in value:
        return "fit-and-flare"

    if "mermaid" in value or "trumpet" in value:
        return "mermaid/trumpet"

    if "bodycon" in value:
        return "bodycon"

    if "fitted" in value or "slim" in value:
        return "fitted"

    if "oversized" in value:
        return "oversized"

    if "relaxed" in value or "loose" in value:
        return "relaxed"

    if "straight" in value:
        return "straight"

    if "wide leg" in value or "wide-leg" in value:
        return "wide leg"

    if "flared" in value or "flare" in value:
        return "flared"

    if "wrap" in value:
        return "wrap"

    if "pleated" in value:
        return "pleated"

    return value


df["clothing_type_clean"] = df["clothing_type"].apply(
    clean_text
)

df["neckline_clean"] = df["neckline"].apply(
    clean_neckline
)

df["pattern_clean"] = df["pattern"].apply(
    clean_text
)

df["waistline_clean"] = df["waistline"].apply(
    clean_waistline
)

df["sleeves_clean"] = df["sleeves"].apply(
    clean_sleeves
)

df["event_clean"] = df["event"].apply(
    clean_text
)

df["fabric_clean"] = df["fabric"].apply(
    clean_fabric
)

df["silhouette_clean"] = df["silhouette"].apply(
    clean_silhouette
)


df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved cleaned dataset to {OUTPUT_FILE}")
print(f"Rows: {len(df)}")
