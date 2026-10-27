
# ============================================================
# SMART MANAGEMENT FOR HEAVY DUTY VEHICLE
# Random Forest ML - Google Colab Full Code
# Fuel Leakage | Brake Fault | Engine Overheating
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ============================================================
# 1. SET GOOGLE COLAB PATHS
# ============================================================

DATA_FILE = Path(
    "/content/heavy_duty_vehicle_synthetic_dataset_10000.csv"
)

OUTDIR = Path("/content/vehicle_project_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("Dataset path:", DATA_FILE)
print("Output folder:", OUTDIR)

# ============================================================
# 2. LOAD EXISTING DATASET OR GENERATE A NEW ONE
# ============================================================

if DATA_FILE.exists():

    df = pd.read_csv(DATA_FILE)
    print("\nExisting dataset loaded successfully.")

else:

    print("\nDataset not found. Generating 10,000 synthetic records...")

    rng = np.random.default_rng(RANDOM_STATE)
    n = 10000

    df = pd.DataFrame({
        "record_id": np.arange(1, n + 1),

        # Fuel sensors
        "fuel_level_drop_pct": rng.uniform(0, 35, n),
        "fuel_pressure_drop_kpa": rng.uniform(0, 60, n),
        "hydrocarbon_vapor_ppm": rng.uniform(0, 600, n),
        "fuel_flow_variation_pct": rng.uniform(0, 40, n),

        # Brake sensors
        "brake_pressure_bar": rng.uniform(35, 150, n),
        "brake_fluid_loss_ml": rng.uniform(0, 35, n),
        "brake_temperature_c": rng.uniform(25, 220, n),
        "brake_pressure_variation_pct": rng.uniform(0, 45, n),

        # Engine sensors
        "engine_temperature_c": rng.uniform(60, 145, n),
        "engine_vibration_mm_s": rng.uniform(0, 18, n),
        "engine_oil_pressure_bar": rng.uniform(0.5, 6.5, n),
        "engine_rpm": rng.uniform(500, 2800, n)
    })

    # Demonstration fuel fault labels
    fuel_score = (
        0.9 * (df["fuel_level_drop_pct"] > 20).astype(int)
        + 1.0 * (df["fuel_pressure_drop_kpa"] > 35).astype(int)
        + 0.9 * (df["hydrocarbon_vapor_ppm"] > 350).astype(int)
        + 0.6 * (df["fuel_flow_variation_pct"] > 25).astype(int)
        + rng.normal(0, 0.75, n)
    )

    # Demonstration brake fault labels
    brake_score = (
        1.0 * (df["brake_pressure_bar"] < 65).astype(int)
        + 1.0 * (df["brake_fluid_loss_ml"] > 22).astype(int)
        + 0.7 * (df["brake_temperature_c"] > 160).astype(int)
        + 0.7 * (df["brake_pressure_variation_pct"] > 28).astype(int)
        + rng.normal(0, 0.75, n)
    )

    # Demonstration engine fault labels
    engine_score = (
        1.0 * (df["engine_temperature_c"] > 115).astype(int)
        + 0.8 * (df["engine_vibration_mm_s"] > 12).astype(int)
        + 0.8 * (df["engine_oil_pressure_bar"] < 1.5).astype(int)
        + 0.5 * (df["engine_rpm"] > 2400).astype(int)
        + rng.normal(0, 0.75, n)
    )

    df["fuel_fault"] = (fuel_score >= 1.65).astype(int)
    df["brake_fault"] = (brake_score >= 1.65).astype(int)
    df["engine_fault"] = (engine_score >= 1.65).astype(int)

    df.to_csv(DATA_FILE, index=False)
    print("Synthetic dataset created:", DATA_FILE)

# ============================================================
# 3. VALIDATE DATASET
# ============================================================

model_configs = {
    "Fuel": {
        "features": [
            "fuel_level_drop_pct",
            "fuel_pressure_drop_kpa",
            "hydrocarbon_vapor_ppm",
            "fuel_flow_variation_pct"
        ],
        "target": "fuel_fault"
    },

    "Brake": {
        "features": [
            "brake_pressure_bar",
            "brake_fluid_loss_ml",
            "brake_temperature_c",
            "brake_pressure_variation_pct"
        ],
        "target": "brake_fault"
    },

    "Engine": {
        "features": [
            "engine_temperature_c",
            "engine_vibration_mm_s",
            "engine_oil_pressure_bar",
            "engine_rpm"
        ],
        "target": "engine_fault"
    }
}

required_columns = ["record_id"]

for config in model_configs.values():
    required_columns.extend(config["features"])
    required_columns.append(config["target"])

missing_columns = sorted(set(required_columns) - set(df.columns))

if missing_columns:
    raise ValueError(
        "Dataset is missing required columns: "
        + ", ".join(missing_columns)
        + ". Generate the expected dataset or correct the CSV."
    )

if df.empty:
    raise ValueError("The dataset is empty.")

if df[required_columns].isnull().any().any():
    raise ValueError(
        "The dataset contains missing values. Clean them before training."
    )

for config in model_configs.values():
    target = config["target"]

    if df[target].nunique() < 2:
        raise ValueError(
            f"{target} must contain both Normal (0) and Fault (1) labels."
        )

    if not set(df[target].unique()).issubset({0, 1}):
        raise ValueError(f"{target} must contain only labels 0 and 1.")

print("\nDataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

# Save a copy of the dataset in the output folder
df.to_csv(OUTDIR / "heavy_duty_vehicle_dataset.csv", index=False)

# ============================================================
# 4. TRAIN THREE RANDOM FOREST MODELS
# ============================================================

results = []
trained_models = {}
test_results = {}

for model_name, config in model_configs.items():

    print("\n" + "=" * 55)
    print(f"TRAINING {model_name.upper()} MODEL")
    print("=" * 55)

    X = df[config["features"]]
    y = df[config["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    trained_models[model_name] = model
    test_results[model_name] = (y_test, y_pred)

    metrics = {
        "Model": model_name,
        "Algorithm": "Random Forest Classifier",
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ) * 100,
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ) * 100,
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ) * 100,
        "Training Records": len(X_train),
        "Testing Records": len(X_test)
    }

    results.append(metrics)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Fault"],
        zero_division=0
    ))

    # Save feature importance
    importance_df = pd.DataFrame({
        "Feature": config["features"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        OUTDIR / f"{model_name.lower()}_feature_importance.csv",
        index=False
    )

# ============================================================
# 5. SAVE MODEL PERFORMANCE METRICS
# ============================================================

results_df = pd.DataFrame(results)

results_path = OUTDIR / "ml_model_results.csv"
results_df.to_csv(results_path, index=False)

print("\nMODEL PERFORMANCE (%)")
print(
    results_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    ].round(2).to_string(index=False)
)

print("\nMetrics saved:", results_path)

# ============================================================
# 6. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, model_name in zip(axes, ["Fuel", "Brake", "Engine"]):

    y_test, y_pred = test_results[model_name]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fault"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(f"{model_name} Model")

fig.suptitle("Confusion Matrices - Random Forest Models")
fig.tight_layout()

cm_path = OUTDIR / "confusion_matrices.png"
fig.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Confusion matrices saved:", cm_path)

# ============================================================
# 7. LINE GRAPH: ACCURACY, PRECISION, RECALL, F1
# ============================================================

metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

fig, ax = plt.subplots(figsize=(9, 5.5))

for _, row in results_df.iterrows():

    values = [row[metric] for metric in metric_names]

    ax.plot(
        metric_names,
        values,
        marker="o",
        linewidth=2,
        label=f"{row['Model']} Model"
    )

ax.set_title("Random Forest Model Performance Comparison")
ax.set_xlabel("Evaluation Metrics")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 105)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
fig.tight_layout()

line_path = OUTDIR / "ml_performance_line_graph.png"
fig.savefig(line_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Performance graph saved:", line_path)

# ============================================================
# 8. FEATURE IMPORTANCE GRAPHS
# ============================================================

for model_name, model in trained_models.items():

    features = model_configs[model_name]["features"]

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    importance.plot(kind="barh", ax=ax)

    ax.set_title(f"{model_name} Model - Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Sensor Feature")
    fig.tight_layout()

    graph_path = OUTDIR / (
        f"{model_name.lower()}_feature_importance.png"
    )

    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ============================================================
# 9. SAVE TRAINED MODELS
# ============================================================

for model_name, model in trained_models.items():

    model_path = OUTDIR / (
        f"{model_name.lower()}_random_forest.joblib"
    )

    joblib.dump(model, model_path)
    print("Model saved:", model_path)

# ============================================================
# 10. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("PROJECT EXECUTION COMPLETED")
print("=" * 55)

print("Dataset:", DATA_FILE)
print("Output folder:", OUTDIR)

print("\nGenerated files:")

for file_path in sorted(OUTDIR.iterdir()):
    if file_path.is_file():
        print(f"- {file_path.name}")

print(
    "\nIMPORTANT: Generated fault labels and sensor values are synthetic. "
    "The measured scores demonstrate the workflow only and do not validate "
    "real-world vehicle safety or fault detection."
)
