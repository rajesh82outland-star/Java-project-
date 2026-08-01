import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    )
RANDOM_STATE = 42
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
def get_population_columns(df):
    pop_cols = [c for c in df.columns if c.startswith("Pop_")]
    pop_cols = sorted(pop_cols, key=lambda c: int(c.split("_")[1]))
    if len(pop_cols) < 5:
        raise ValueError("Not enough population columns found.")
    return pop_cols


def build_growth_target(df):
    """
    Required dependent variable: binary Growth (up/down).
    Use provided Growth column when available, else compute from Pop_1991 -> Pop_2001.
    """
    source_col = "Forecasted Growth (Up/Down)"
    growth_map = {"up": 1, "down": 0, "1": 1, "0": 0}

    if source_col in df.columns and df[source_col].notna().any():
        raw = df[source_col].astype(str).str.strip().str.lower()
        parsed = raw.map(growth_map)
        computed = (df["Pop_2001"] > df["Pop_1991"]).astype(int)
        df["Growth"] = parsed.fillna(computed).astype(int)
        print("\nTarget source: Forecasted Growth (with computed fallback).")
    else:
        df["Growth"] = (df["Pop_2001"] > df["Pop_1991"]).astype(int)
        print("\nTarget source: computed from Pop_1991 and Pop_2001.")
    return df

def add_geography_features(df):
    """
    Geographic relationship feature from map perspective.
    """
    geo_zone_map = {
        "City of London": "Central",
        "Barking and Dagenham": "East",
        "Barnet": "North",
        "Bexley": "South",
        "Bromley": "South",
        "Brent": "West",
        "Camden": "Central",
        "Croydon": "South",
        "Ealing": "West",
        "Enfield": "North",
        "Greenwich": "South",
        "Hackney": "East",
        "Hammersmith and Fulham": "West",
        "Haringey": "North",
        "Harrow": "North",
        "Havering": "East",
        "Hillingdon": "West",
        "Hounslow": "West",
        "Islington": "Central",
        "Kensington and Chelsea": "Central",
        "Kingston upon Thames": "South",
        "Lambeth": "Central",
        "Lewisham": "South",
        "Merton": "South",
        "Newham": "East",
        "Redbridge": "East",
        "Richmond upon Thames": "West",
        "Southwark": "Central",
        "Sutton": "South",
        "Tower Hamlets": "East",
        "Waltham Forest": "North",
        "Wandsworth": "South",
        "Westminster": "Central",
    }

    df["GeoZone"] = df["Area Name"].map(geo_zone_map)
    df["LondonType"] = np.where(df["Area Name"].eq("City of London"), "City", "Borough")

    missing_geo = sorted(df.loc[df["GeoZone"].isna(), "Area Name"].unique())
    if missing_geo:
        raise ValueError("Missing GeoZone mapping for: " + ", ".join(missing_geo))

    return df

def safe_rate(curr, prev):
    prev = prev.replace(0, np.nan)
    rate = (curr - prev) / prev
    return rate.replace([np.inf, -np.inf], np.nan)

def make_plots(df, pop_cols):
    years = [int(c.split("_")[1]) for c in pop_cols]
    mean_pop = df[pop_cols].mean()

    # EDA 1: trend across boroughs
    plt.figure(figsize=(10, 5))
    plt.plot(years, mean_pop.values, marker="o")
    plt.title("Average Population Trend Across London Boroughs")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_average_population_trend.png"), dpi=200)
    plt.show()

    # EDA 2: growth distribution
    plt.figure(figsize=(6, 4))
    df["Growth"].value_counts().sort_index().plot(kind="bar")
    plt.title("Growth Distribution (Dependent Variable)")
    plt.xlabel("Growth (0=Down, 1=Up)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_growth_distribution.png"), dpi=200)
    plt.show()

    # EDA 3: correlation heatmap
    corr_cols = ["Pop_1961", "Pop_1971", "Pop_1981", "Pop_1991", "Growth"]
    plt.figure(figsize=(7, 5))
    sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_correlation_heatmap.png"), dpi=200)
    plt.show()

    # EDA 4: map perspective (GeoZone)
    zone_growth = df.groupby("GeoZone")["Growth"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    zone_growth.plot(kind="bar")
    plt.title("Map Perspective: Growth Share by GeoZone")
    plt.xlabel("GeoZone")
    plt.ylabel("Share of Up Growth")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_growth_by_geozone.png"), dpi=200)
    plt.show()


def split_with_class_safety(X, y):
    class_counts = y.value_counts()
    min_class = int(class_counts.min())

    if min_class >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
        )
        print("\nSplit method: Stratified train/test split")
        return X_train, X_test, y_train, y_test

    # Fallback for extreme imbalance (minority class count == 1)
    print("\nSplit method: custom split (minority class forced into training)")
    minority_label = class_counts.idxmin()
    majority_idx = y[y != minority_label].index.to_numpy()

    rng = np.random.default_rng(RANDOM_STATE)
    test_n = max(1, int(round(0.30 * len(majority_idx))))
    test_majority_idx = rng.choice(majority_idx, size=test_n, replace=False)

    test_idx = pd.Index(test_majority_idx)
    train_idx = y.index.difference(test_idx)

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    return X_train, X_test, y_train, y_test
def make_cv(y, max_splits=5):
    min_class = int(y.value_counts().min())
    if min_class < 2:
        return None, None
    n_splits = min(max_splits, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return cv, n_splits
def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score (positive class): {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("Classification Report:\n", classification_report(y_true, y_pred, labels=[0, 1],
            target_names=["0 (Down)", "1 (Up)"], zero_division=0,),)
def cv_report(name, model, X, y, cv):
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    bacc = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
    f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"{name} Accuracy: {acc.mean():.4f} +/- {acc.std():.4f}")
    print(f"{name} Balanced Accuracy: {bacc.mean():.4f} +/- {bacc.std():.4f}")
    print(f"{name} F1: {f1.mean():.4f} +/- {f1.std():.4f}")

def main():
    df = pd.read_csv("census-historic-population-borough_London.csv")
    print("Columns:\n", df.columns)
    pop_cols = get_population_columns(df)
    # Keep borough-level geography rows only
    df = df.loc[~df["Area Name"].isin(["Inner London", "Outer London", "Greater London"])].copy()

    # Remove duplicate borough names if present
    before = len(df)
    df = df.drop_duplicates(subset=["Area Name"], keep="first").copy()
    removed = before - len(df)
    if removed > 0:
        print(f"\nRemoved duplicate borough rows: {removed}")

    # Basic cleaning
    df = df.dropna(subset=["Pop_1991", "Pop_2001"]).copy()
    print("\nDataset shape after cleaning:", df.shape)

    # Required target
    df = build_growth_target(df)
    print("\nGrowth distribution:\n", df["Growth"].value_counts())

    # Geographic relationship features
    df = add_geography_features(df)

    # Additional engineered predictors from historic trend
    df["GrowthRate_61_71"] = safe_rate(df["Pop_1971"], df["Pop_1961"])
    df["GrowthRate_71_81"] = safe_rate(df["Pop_1981"], df["Pop_1971"])
    df["GrowthRate_81_91"] = safe_rate(df["Pop_1991"], df["Pop_1981"])

    # EDA plots
    make_plots(df, pop_cols)

    # Feature matrix
    base_features = [
        "Pop_1961",
        "Pop_1971",
        "Pop_1981",
        "Pop_1991",
        "GrowthRate_61_71",
        "GrowthRate_71_81",
        "GrowthRate_81_91",
    ]
    X_num = df[base_features].fillna(0)
    X_cat = pd.get_dummies(df[["GeoZone", "LondonType"]], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["Growth"].astype(int)

    # Split
    X_train, X_test, y_train, y_test = split_with_class_safety(X, y)
    print("\nTrain class distribution:\n", y_train.value_counts())
    print("\nTest class distribution:\n", y_test.value_counts())

    # CV strategy
    cv_train, train_splits = make_cv(y_train)
    cv_all, all_splits = make_cv(y)

    # 1) Decision Tree
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    # 2) Random Forest
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    if cv_train is not None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 8, 12],
            "min_samples_leaf": [1, 2, 4],
        }
        grid = GridSearchCV(
            rf_base,
            param_grid=param_grid,
            cv=cv_train,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        rf_model = grid.best_estimator_
        print("\nBest Random Forest Parameters:", grid.best_params_)
    else:
        rf_model = rf_base.fit(X_train, y_train)
        print("\nRandom Forest grid search skipped (not enough minority samples for CV).")

    rf_pred = rf_model.predict(X_test)

    # 3) Logistic Regression
    log_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    log_pred = None
    try:
        log_model.fit(X_train, y_train)
        log_pred = log_model.predict(X_test)
    except ValueError as exc:
        print(f"\nLogistic Regression skipped: {exc}")

    # Evaluation
    evaluate("Decision Tree", y_test, dt_pred)
    evaluate("Random Forest", y_test, rf_pred)
    if log_pred is not None:
        evaluate("Logistic Regression", y_test, log_pred)

    # Cross Validation
    print("\n--- Cross Validation ---")
    if cv_all is None:
        print("Cross-validation skipped: minority class count < 2.")
    else:
        print(f"Using StratifiedKFold with n_splits={all_splits}")
        cv_report("Decision Tree", dt_model, X, y, cv_all)
        cv_report("Random Forest", rf_model, X, y, cv_all)
        cv_report("Logistic Regression", log_model, X, y, cv_all)

    # Feature importance
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(10, 6))
    importances.tail(15).plot(kind="barh")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_random_forest_feature_importance.png"), dpi=200)
    plt.show()
    print("\nAll figures were saved in the figures folder.")
if __name__ == "__main__":
    main()