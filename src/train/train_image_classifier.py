"""CNN image classifier for the slide images.

Decides the type of every image (diagram, chart, formula, table, photo, decoration),
so only the useful ones get a description later.

Run from the repo root:
    python -m src.train.train_image_classifier            # train, compare with CLIP, classify all images
    python -m src.train.train_image_classifier --no-clip  # same but skips CLIP (faster)
    python -m src.train.train_image_classifier --classify-only
"""
from __future__ import annotations  # lets us write types like "Path | None" in any Python version

import argparse  # reads the options we write after the command (--no-clip, --classify-only)
import csv  # reads the labels file and writes the classes file
import json  # writes the scores and the model info as readable text files
import os  # reads environment variables (used only by the tests)
import sys  # returns the exit code when the script finishes
from collections import Counter  # counts how many images there are of each class
from pathlib import Path  # handles file paths the same way on Windows and Mac

import numpy as np  # arrays of numbers, used for the features and the labels

from src.config import PATHS, load_config  # the shared folder paths and settings of the project

# the 6 types an image can be; the order matters because the model outputs one number per class in this order
CLASSES = ["diagram", "chart", "formula", "table", "photo", "decoration"]
# only these 4 carry information a student could ask about, so only these get a description later
USEFUL_CLASSES = {"diagram", "chart", "formula", "table"}
# for CLIP (the baseline) each class is described with a sentence, because CLIP compares images with text
CLIP_PROMPTS = {
    "diagram": "a slide diagram with boxes and arrows",
    "chart": "a chart or plot with axes, bars or lines",
    "formula": "a mathematical formula or equation",
    "table": "a table with rows and columns",
    "photo": "a photograph of people, places or objects",
    "decoration": "a decorative icon, logo or background",
}
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")  # file types we accept as images
SEED = 42  # fixed seed, so the folds and the training are the same every time we run it
FOLDS = 5  # 5-fold cross-validation: every image is used 4 times for training and once for testing
EPOCHS = 60  # passes over the training images when training the last layer (it is small, so this is fast)
BATCH = 32  # images processed together; 32 fits easily in the memory of a normal laptop
MODEL_DIR = PATHS["image_clf"]  # models/image_clf/, where the trained model is saved (not uploaded to GitHub)
REPORT = PATHS["reports"] / "image_clf_cv.json"  # the scores that go into the documentation
CONFUSION_PNG = PATHS["figures"] / "image_clf_confusion.png"  # the confusion matrix figure for the slides


#1 load the images we labelled by hand

def find_image(image_id: str) -> Path | None:
    """Return the file of an image from its ID, whatever its extension (.png, .jpg...)."""
    for ext in IMAGE_TYPES:  # try each possible extension
        path = PATHS["images"] / f"{image_id}{ext}"  # builds data/interim/images/<image_id>.<ext>
        if path.exists():  # if a file with that name is there
            return path  # we found the image
    return None  # the image is labelled but its file is not on this laptop


def load_labels() -> tuple[list[Path], np.ndarray, list[str]]:
    """Read the labels file and return the image files, their class numbers and their IDs."""
    labels_file = PATHS["image_labels"]  # data/labels/image_labels.csv, written by the labelling tool
    if not labels_file.exists():  # nobody has labelled anything yet
        raise SystemExit(f"No labels yet: {labels_file}. Label images with: streamlit run src/vision/label_tool.py")
    latest: dict[str, str] = {}  # image_id -> label, one entry per image
    with labels_file.open(encoding="utf-8", newline="") as f:  # opens the labels file
        for row in csv.DictReader(f):  # one row per click in the labelling tool
            if row["label"] in CLASSES:  # ignores anything that is not one of our 6 classes
                latest[row["image_id"]] = row["label"]  # if an image was labelled twice, the last label wins
    paths, ys, ids, missing = [], [], [], 0  # lists we fill below, and a counter of missing files
    for image_id, label in sorted(latest.items()):  # sorted so the order is always the same
        path = find_image(image_id)  # looks for the image file
        if path is None:  # labelled but the image file is missing
            missing += 1  # we count it to warn at the end
            continue  # and skip it, we can't train with an image we don't have
        paths.append(path)  # the image file
        ys.append(CLASSES.index(label))  # the model works with numbers, so "chart" becomes 1, "table" 3, etc.
        ids.append(image_id)  # the image ID, to know which image each row is
    if missing:  # warns if some labelled images were not found
        print(f"Warning: {missing} labelled images not found in {PATHS['images']}")
    counts = Counter(CLASSES[y] for y in ys)  # how many images we have of each class
    print("Labels per class:", {c: counts[c] for c in CLASSES})  # prints it, so we see if a class has too few
    # with 5 folds, each class needs at least 5 images so every fold can test it; we aim for 20 per class
    too_few = [c for c in CLASSES if counts[c] < FOLDS]  # classes with fewer than 5 images
    if too_few:  # if any, we stop and say which ones need more labels
        raise SystemExit(f"Need at least {FOLDS} labelled images per class (aim for 20). Too few: {too_few}")
    return paths, np.array(ys), ids  # labels as a numpy array, which is what the model and sklearn expect


def all_images() -> list[Path]:
    """Every extracted image, labelled or not (these are the ones we classify at the end)."""
    folder = PATHS["images"]  # data/interim/images/
    # every file in the folder with an image extension, sorted so the order is always the same
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES) if folder.exists() else []


#2 the CNN: a model already trained on millions of images, with a new last layer for our 6 classes

def load_backbone():
    """MobileNetV3-Small pretrained on ImageNet, without its last layer, so it outputs 576 numbers per image."""
    import torch  # imported here so the rest of the project can run without PyTorch installed
    from torchvision import models  # the library with ready-made image models

    # the tests set this variable to avoid downloading the weights; normally we always use the pretrained ones
    pretrained = os.environ.get("IMAGE_CLF_NO_PRETRAINED") != "1"
    # ImageNet weights: the model already knows edges, shapes and textures from 1.2 million photos.
    # This is transfer learning: with only ~300 labelled images we reuse what it already knows
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)  # small and fast, so it runs on a laptop without a GPU
    # the original last layer predicts the 1,000 ImageNet classes; we remove it so the model gives us
    # its 576 "features" (a summary of what it sees) and we put our own 6-class layer on top
    net.classifier = torch.nn.Identity()  # Identity = "pass the numbers through unchanged"
    net.eval()  # evaluation mode: no dropout, the same image always gives the same features
    for p in net.parameters():  # for every weight of the pretrained model
        p.requires_grad = False  # frozen: these layers don't change, only our new last layer learns
    return net  # the frozen CNN, ready to turn images into features


def preprocess():
    """The same transformation ImageNet images had, so the pretrained model sees what it expects."""
    from torchvision import transforms  # image transformation tools

    return transforms.Compose([  # Compose applies these steps one after the other
        transforms.Resize((224, 224)),  # the model was trained on 224 x 224 images
        transforms.ToTensor(),  # converts the image into numbers between 0 and 1
        # subtracts the average colour of ImageNet and divides by its spread, as in the original training
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def cnn_features(paths: list[Path], backbone) -> np.ndarray:
    """Pass every image once through the frozen CNN and keep its 576 features."""
    import torch  # tensors, the numbers format PyTorch works with
    from PIL import Image  # opens image files

    tf = preprocess()  # the transformation above
    feats = []  # we collect the features of each batch here
    with torch.no_grad():  # we are not training these layers, so we don't need gradients (faster, less memory)
        for i in range(0, len(paths), BATCH):  # 32 images at a time
            # opens each image in colour (some slides are grayscale or have transparency) and transforms it
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + BATCH]])
            feats.append(backbone(batch).numpy())  # 576 numbers per image, back to numpy
            print(f"  features {min(i + BATCH, len(paths))}/{len(paths)}", end="\r")  # progress on one line
    print()  # new line after the progress counter
    # since the CNN is frozen, the features never change: we compute them once and reuse them in every fold
    return np.concatenate(feats) if feats else np.zeros((0, 576), dtype=np.float32)


def train_head(x: np.ndarray, y: np.ndarray):
    """Train the new last layer: 576 features in, 6 class scores out."""
    import torch  # the training library

    torch.manual_seed(SEED)  # same starting weights every run, so results are reproducible
    head = torch.nn.Linear(x.shape[1], len(CLASSES))  # our new last layer: 576 -> 6
    # class weights: if there are 80 decorations and 20 formulas, a mistake on a formula counts 4 times more,
    # so the model doesn't learn to just say "decoration" (we also report macro-F1 for the same reason)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)  # images per class in this training set
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(CLASSES))  # rare class -> bigger weight
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)  # the standard loss for choosing one class out of several
    # Adam adjusts the weights; weight_decay keeps them small so the layer doesn't memorise the training images
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y)  # numpy arrays -> PyTorch tensors
    for _ in range(EPOCHS):  # 60 passes over the training images
        perm = torch.randperm(len(xt))  # shuffles the order every epoch, so batches are different each time
        for i in range(0, len(xt), BATCH):  # 32 images at a time
            idx = perm[i:i + BATCH]  # the positions of this batch's images
            opt.zero_grad()  # clears the corrections of the previous batch
            loss_fn(head(xt[idx]), yt[idx]).backward()  # measures the error and computes how to correct it
            opt.step()  # applies the correction to the weights
    head.eval()  # switches to prediction mode
    return head  # the trained last layer


def predict_head(head, x: np.ndarray) -> np.ndarray:
    """Return, for each image, the probability of each of the 6 classes (they add up to 1)."""
    import torch  # the prediction library

    with torch.no_grad():  # only predicting, no training, so no gradients needed
        # softmax turns the 6 raw scores into probabilities that add up to 1
        return torch.softmax(head(torch.tensor(x, dtype=torch.float32)), dim=1).numpy()


#3 the baseline: zero-shot CLIP (no training at all), to show that training our CNN was worth it

def clip_probs(paths: list[Path]) -> np.ndarray:
    """CLIP compares each image with the 6 class sentences and picks the closest one."""
    import open_clip  # the open-source version of CLIP
    import torch  # tensors
    from PIL import Image  # opens image files

    # CLIP was trained on 2 billion image-caption pairs, so it can match images with text it never saw
    model, _, tf = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")  # turns the sentences into the numbers CLIP reads
    model.eval()  # prediction mode
    with torch.no_grad():  # no training, so no gradients
        text = model.encode_text(tokenizer([CLIP_PROMPTS[c] for c in CLASSES]))  # one vector per class sentence
        text = text / text.norm(dim=-1, keepdim=True)  # normalises, so comparing is just a dot product
        out = []  # the probabilities of each batch
        for i in range(0, len(paths), BATCH):  # 32 images at a time
            imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + BATCH]])  # opens and transforms
            img = model.encode_image(imgs)  # one vector per image, in the same space as the sentences
            img = img / img.norm(dim=-1, keepdim=True)  # normalises the same way
            # similarity of each image with each sentence, turned into probabilities (100 is CLIP's usual scale)
            out.append((100.0 * img @ text.T).softmax(dim=-1).numpy())
    return np.concatenate(out)  # one row of 6 probabilities per image


#4 the scores we report

def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro-F1 (main metric) and accuracy for one fold."""
    from sklearn.metrics import accuracy_score, f1_score  # standard metric functions

    return {
        # macro-F1 averages the F1 of the 6 classes equally, so a rare class like "formula" counts as much as
        # "decoration"; accuracy alone would look good just by getting the most common class right
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(CLASSES))), zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),  # share of images classified correctly
    }


def summarise(per_fold: list[dict]) -> dict:
    """Mean and standard deviation over the 5 folds (the sd shows how stable the result is)."""
    # for each metric: the mean of the 5 folds and how much it changes from fold to fold
    return {k: {"mean": float(np.mean([f[k] for f in per_fold])), "sd": float(np.std([f[k] for f in per_fold]))}
            for k in per_fold[0]}


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Confusion matrix: which classes the CNN mixes up (e.g. charts taken for diagrams)."""
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # draws to a file without opening a window
    import matplotlib.pyplot as plt  # the plotting functions
    from sklearn.metrics import ConfusionMatrixDisplay  # draws the matrix for us

    CONFUSION_PNG.parent.mkdir(parents=True, exist_ok=True)  # creates docs/figures/ if it doesn't exist
    fig, ax = plt.subplots(figsize=(6, 5))  # a 6 x 5 inch figure
    # rows = real class, columns = predicted class; the diagonal are the correct ones
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=list(range(len(CLASSES))),
                                            display_labels=CLASSES, ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title("CNN image classifier, 5-fold cross-validation")  # the title shown in the documentation
    fig.tight_layout()  # so the labels are not cut off
    fig.savefig(CONFUSION_PNG, dpi=150)  # saves docs/figures/image_clf_confusion.png
    plt.close(fig)  # frees the memory


#5 train and evaluate: 5-fold cross-validation, CNN vs CLIP, then the final model on all labels

def train(cfg=None, use_clip: bool = True) -> dict:
    import torch  # to save the model
    from sklearn.model_selection import StratifiedKFold  # splits the images into the 5 folds

    paths, y, _ = load_labels()  # the labelled images and their classes (step 1)
    print(f"Computing CNN features for {len(paths)} labelled images...")
    backbone = load_backbone()  # the frozen pretrained CNN (step 2)
    x = cnn_features(paths, backbone)  # computed once, reused in all folds

    # stratified: every fold keeps the same share of each class as the whole set, so no fold is left without formulas
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    cnn_folds, clip_folds = [], []  # the scores of each fold, for the CNN and for CLIP
    oof = np.zeros(len(y), dtype=int)  # "out of fold": each image predicted by the model that did NOT train on it
    clip_all = None  # stays empty if we run with --no-clip
    if use_clip:  # the CLIP comparison (step 3)
        print("Computing zero-shot CLIP predictions...")
        clip_all = clip_probs(paths).argmax(axis=1)  # CLIP doesn't train, so we predict all images once
    for k, (tr, te) in enumerate(skf.split(x, y), 1):  # 5 rounds: tr = 4/5 of the images, te = the other 1/5
        head = train_head(x[tr], y[tr])  # trains only on this round's training images
        pred = predict_head(head, x[te]).argmax(axis=1)  # predicts images it has never seen (highest probability)
        oof[te] = pred  # keeps these predictions for the confusion matrix
        cnn_folds.append(scores(y[te], pred))  # the CNN's scores on this fold
        if clip_all is not None:  # if we also ran CLIP
            clip_folds.append(scores(y[te], clip_all[te]))  # CLIP scored on exactly the same images, fair comparison
        # prints each fold's result, so we see the progress and how much it varies
        print(f"  fold {k}: CNN macro-F1 {cnn_folds[-1]['macro_f1']:.3f}"
              + (f" | CLIP {clip_folds[-1]['macro_f1']:.3f}" if clip_folds else ""))

    # what goes into reports/image_clf_cv.json and then into the documentation
    report = {
        "n_images": int(len(y)),  # how many labelled images we used
        "per_class": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},  # how many of each class
        "folds": FOLDS,  # 5
        "cnn": {"model": "mobilenet_v3_small (ImageNet, frozen) + linear head", **summarise(cnn_folds)},  # CNN scores
    }
    if clip_folds:  # adds CLIP's scores if we ran it
        report["clip_zero_shot"] = {"model": "open_clip ViT-B-32 laion2b_s34b_b79k", **summarise(clip_folds)}
    save_confusion(y, oof)  # the confusion matrix uses only predictions on unseen images, so it is honest

    # cross-validation only measures; the model we actually use is trained on ALL the labels (more data = better)
    print("Training the final CNN on all labels...")
    final = train_head(x, y)  # the final last layer, trained on every labelled image
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # creates models/image_clf/ if it doesn't exist
    torch.save(final.state_dict(), MODEL_DIR / "head.pt")  # only our last layer; the frozen CNN is reloaded from torchvision
    # saves the class order and some info, so the model can be loaded correctly later
    (MODEL_DIR / "meta.json").write_text(json.dumps({"classes": CLASSES, "backbone": "mobilenet_v3_small",
                                                     "n_train": int(len(y))}, indent=2))
    REPORT.parent.mkdir(parents=True, exist_ok=True)  # creates reports/ if it doesn't exist
    REPORT.write_text(json.dumps(report, indent=2))  # saves reports/image_clf_cv.json
    print(json.dumps(report, indent=2))  # and prints it
    return report


#6 classify every extracted image with the final model

def classify_all(cfg=None) -> int:
    """Write the class of every image to data/interim/image_classes.csv (read by describe.py)."""
    import torch  # to load the model

    meta_file = MODEL_DIR / "meta.json"  # the info saved with the model
    if not meta_file.exists():  # the model hasn't been trained yet
        raise SystemExit("No trained model yet. Run: python -m src.train.train_image_classifier")
    classes = json.loads(meta_file.read_text())["classes"]  # the class order saved with the model
    paths = all_images()  # all images, not only the labelled ones
    print(f"Classifying {len(paths)} images...")
    backbone = load_backbone()  # the same frozen CNN as in training
    head = torch.nn.Linear(576, len(classes))  # same shape as when it was trained
    head.load_state_dict(torch.load(MODEL_DIR / "head.pt"))  # loads the trained weights
    head.eval()  # prediction mode
    probs = predict_head(head, cnn_features(paths, backbone))  # 6 probabilities per image
    out = PATHS["image_classes"]  # data/interim/image_classes.csv
    out.parent.mkdir(parents=True, exist_ok=True)  # creates the folder if it doesn't exist
    with out.open("w", encoding="utf-8", newline="") as f:  # opens the file to write it from scratch
        writer = csv.writer(f)
        writer.writerow(["image_id", "image_class", "class_prob"])  # the column names
        for path, p in zip(paths, probs):  # one row per image
            # the class with the highest probability, and that probability (how sure the model is)
            writer.writerow([path.stem, classes[int(p.argmax())], round(float(p.max()), 4)])
    counts = Counter(classes[int(p.argmax())] for p in probs)  # how many images of each class
    useful = sum(counts[c] for c in USEFUL_CLASSES)  # how many will be described in the next step
    print(f"Done: {dict(counts)} -> {useful} images will get a description")
    return len(paths)  # how many images were classified


def main(argv: list[str] | None = None) -> int:
    # reads the options written after the command
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-clip", action="store_true", help="skip the zero-shot CLIP comparison")
    parser.add_argument("--classify-only", action="store_true", help="only classify all images with the saved model")
    args = parser.parse_args(argv)
    cfg = load_config("configs/base.yaml")  # the project settings
    if not args.classify_only:  # normal run: train first
        train(cfg, use_clip=not args.no_clip)  # steps 1 to 5
    classify_all(cfg)  # step 6, always
    return 0  # 0 = finished without errors


if __name__ == "__main__":  # only runs when the file is executed, not when another file imports it
    sys.exit(main())