import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_path):
    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Bagi data
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Aktifkan autolog MLflow
    mlflow.sklearn.autolog()

    # 4. Jalankan training dan logging
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="HDS_preprocessing.csv",  # ✅ Default agar tidak error kalau dijalankan manual
        help="Path ke file dataset CSV (default: HDS_preprocessing.csv)"
    )
    args = parser.parse_args()
    main(args.data_path)
