"""INFO 4670 Assignment 4: SMS spam classification.

Run this script in Colab or a Python environment with pandas, NumPy,
matplotlib, and scikit-learn installed. It follows the provided notebook's
six-step workflow and uses the UCI SMS Spam Collection.
"""

import io
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


RANDOM_STATE = 42
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


def load_data():
    """Download and return the 5,572-message UCI dataset."""
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open("SMSSpamCollection") as stream:
        data = pd.read_csv(stream, sep="\t", header=None, names=["label", "message"])
    data["spam"] = (data["label"] == "spam").astype(int)
    return data


def build_features(data):
    """Add the four starter binary features and two assignment features."""
    out = data.copy()
    msg = out["message"].str.lower()
    out["has_phone"] = msg.str.contains(
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|call.*\d{4,}|\b0\d{9,}\b",
        regex=True,
    ).astype(int)
    out["has_money"] = msg.str.contains(
        r"\b(?:free|win|winner|prize|cash|\$|claim|reward|won)\b", regex=True
    ).astype(int)
    out["has_urgency"] = msg.str.contains(
        r"\b(?:urgent|immediately|now|expires|limited|hurry|last chance|act)\b",
        regex=True,
    ).astype(int)
    out["is_long"] = (out["message"].str.len() > 130).astype(int)

    # New feature 1: a three-or-more-letter all-caps word (e.g., WIN, FREE).
    out["has_caps_word"] = out["message"].str.contains(
        r"\b[A-Z]{3,}\b", regex=True
    ).astype(int)
    # New feature 2: a text/reply instruction that asks the recipient to act.
    out["has_text_cta"] = msg.str.contains(
        r"\b(?:txt|text|reply)\b", regex=True
    ).astype(int)
    return out


def main():
    # Step 1 — Problem definition
    # False positives can block time-sensitive legitimate alerts; precision is
    # the primary metric so flagged messages are more likely to be actual spam.

    # Step 2 — Load and explore
    raw = load_data()
    print(f"Loaded {len(raw):,} messages.")
    counts = raw["spam"].value_counts().sort_index()
    print(f"Ham: {counts[0]:,} ({counts[0] / len(raw):.1%})")
    print(f"Spam: {counts[1]:,} ({counts[1] / len(raw):.1%})")
    dummy = np.zeros(len(raw), dtype=int)
    print(
        "Always-ham baseline: "
        f"accuracy={accuracy_score(raw['spam'], dummy):.3f}, "
        f"recall={recall_score(raw['spam'], dummy, zero_division=0):.3f}, "
        f"F1={f1_score(raw['spam'], dummy, zero_division=0):.3f}"
    )

    print("\nFirst five spam messages:")
    for i, message in enumerate(raw.loc[raw.spam == 1, "message"].head(5), start=1):
        print(f"{i}. {message[:130]}")
    print("\nFirst five ham messages:")
    for i, message in enumerate(raw.loc[raw.spam == 0, "message"].head(5), start=1):
        print(f"{i}. {message[:130]}")

    raw["msg_length"] = raw["message"].str.len()
    print("\nMean message length:")
    print(raw.groupby("label")["msg_length"].mean().round(1))
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [("ham", "#2196F3"), ("spam", "#F44336")]:
        ax.hist(
            raw.loc[raw.label == label, "msg_length"],
            bins=40,
            density=True,
            alpha=0.55,
            label=label.title(),
            color=color,
        )
    ax.set(xlabel="Message length (characters)", ylabel="Density", title="SMS length by class")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Step 3 — Engineer binary features
    data = build_features(raw)
    feature_cols = [
        c for c in data.columns if c not in ["label", "message", "spam", "msg_length"]
    ]
    print("\nFeature rates by class (and difference):")
    for feature in feature_cols:
        spam_rate = data.loc[data.spam == 1, feature].mean()
        ham_rate = data.loc[data.spam == 0, feature].mean()
        print(f"{feature:16s} spam={spam_rate:6.1%} ham={ham_rate:6.1%} gap={spam_rate-ham_rate:+6.1%}")

    fig, axes = plt.subplots(1, len(feature_cols), figsize=(3 * len(feature_cols), 3.8))
    for ax, feature in zip(axes, feature_cols):
        rates = [data.loc[data.spam == 1, feature].mean(), data.loc[data.spam == 0, feature].mean()]
        ax.bar(["Spam", "Ham"], rates, color=["#F44336", "#2196F3"])
        ax.set_title(feature)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share with feature")
        for j, rate in enumerate(rates):
            ax.text(j, rate + 0.025, f"{rate:.0%}", ha="center", fontsize=8)
    fig.suptitle("Binary feature rates by message class")
    plt.tight_layout()
    plt.show()

    # Step 4 — Stratified 75/25 split
    X = data[feature_cols].values
    y = data["spam"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"\nTrain: {len(y_train):,} rows, spam={y_train.mean():.1%}; "
        f"test: {len(y_test):,} rows, spam={y_test.mean():.1%}"
    )

    # Step 5A — Decision Tree
    tree = DecisionTreeClassifier(
        max_depth=4, random_state=RANDOM_STATE, class_weight="balanced"
    )
    tree.fit(X_train, y_train)
    print(f"\nDecision Tree leaves: {tree.get_n_leaves()}")
    print(export_text(tree, feature_names=feature_cols))
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        tree,
        feature_names=feature_cols,
        class_names=["Ham", "Spam"],
        filled=True,
        rounded=True,
        ax=ax,
    )
    plt.tight_layout()
    plt.show()

    # Step 5B — Feature importance
    importances = pd.Series(tree.feature_importances_, index=feature_cols).sort_values()
    colors = ["#F44336" if value == importances.max() and value > 0 else "#9E9E9E" for value in importances]
    ax = importances.plot(kind="barh", color=colors, figsize=(8, 4))
    ax.set(xlabel="Decision Tree feature importance", title="Feature importance")
    plt.tight_layout()
    plt.show()
    print("Most important feature:", importances.idxmax())

    # Step 5C — Tree depth comparison
    depths = [1, 2, 3, 4, 5, 6, None]
    train_f1, test_f1 = [], []
    for depth in depths:
        candidate = DecisionTreeClassifier(
            max_depth=depth, random_state=RANDOM_STATE, class_weight="balanced"
        )
        candidate.fit(X_train, y_train)
        train_f1.append(f1_score(y_train, candidate.predict(X_train), zero_division=0))
        test_f1.append(f1_score(y_test, candidate.predict(X_test), zero_division=0))
    depth_labels = [str(d) if d is not None else "None" for d in depths]
    plt.figure(figsize=(8, 4))
    plt.plot(depth_labels, train_f1, marker="o", label="Train F1")
    plt.plot(depth_labels, test_f1, marker="o", label="Test F1")
    plt.xlabel("Maximum tree depth")
    plt.ylabel("F1-score")
    plt.title("Decision Tree depth and generalization")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Depth, train F1, test F1:")
    for depth, tr, te in zip(depth_labels, train_f1, test_f1):
        print(f"{depth:>4}: {tr:.3f}, {te:.3f}")

    # Step 5D — Naive Bayes
    nb = BernoulliNB()
    nb.fit(X_train, y_train)

    # Step 6A/6B — Holdout confusion matrices and metric comparison
    models = {"Decision Tree": tree, "Naive Bayes": nb}
    predictions = {}
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, model) in zip(axes, models.items()):
        pred = model.predict(X_test)
        predictions[name] = pred
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
        print(f"\n{name} confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")
        print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1-Score": f1_score(y_test, pred, zero_division=0),
            }
        )
    plt.tight_layout()
    plt.show()

    metrics = pd.DataFrame(rows).set_index("Model")
    print("\nHoldout metrics:")
    print(metrics.round(3))
    metrics.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0)
    plt.title("Model performance on the test set")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

    # Step 6C — Five-fold stratified cross-validation (F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\nFive-fold cross-validation F1:")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        print(f"{name}: fold scores={np.round(scores, 3)}, mean={scores.mean():.3f}")


if __name__ == "__main__":
    main()
