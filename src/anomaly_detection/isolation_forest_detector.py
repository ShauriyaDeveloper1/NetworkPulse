"""
NetworkPulse: Isolation Forest Anomaly Detection Engine
======================================================
Trains, evaluates, and deploys an unsupervised Isolation Forest model
to detect network latency degradation and operational anomalies in drone telemetry.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection.feature_engineering import extract_anomaly_features
import config


class IsolationForestDetector:
    """
    Unsupervised Isolation Forest detector for network latency and link performance anomalies.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.10,
        max_samples: Any = "auto",
        random_state: int = 42,
        features: Optional[List[str]] = None
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.features = features or config.IF_CONFIG["features"]

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, target_col: str = "round_trip_time") -> "IsolationForestDetector":
        """
        Fits StandardScaler and Isolation Forest on training data.
        """
        X = extract_anomaly_features(train_df, target_col=target_col, feature_list=self.features)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str = "round_trip_time"
    ) -> pd.DataFrame:
        """
        Generates predictions and anomaly scores for input observations.

        Returns DataFrame with:
            - timestamp (if available in index or column)
            - actual_rtt_ms
            - raw_decision_score: scikit-learn decision_function (negative = more anomalous)
            - anomaly_score: inverted & normalized [0, 1] severity indicator (higher = more anomalous)
            - is_anomaly: boolean (True = Anomaly)
            - anomaly_label: "Anomaly" or "Normal"
            - engineered features
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError("IsolationForestDetector must be fitted before predict() is called.")

        X = extract_anomaly_features(df, target_col=target_col, feature_list=self.features)
        X_scaled = self.scaler.transform(X)

        # Raw predictions: 1 for normal, -1 for anomaly
        raw_preds = self.model.predict(X_scaled)
        decision_scores = self.model.decision_function(X_scaled)

        # Higher score = more anomalous. Normalize between 0 and 1
        # decision_function typically ranges from ~ -0.3 to +0.2
        norm_scores = 1.0 / (1.0 + np.exp(decision_scores * 8.0))

        out_df = pd.DataFrame(index=df.index)
        if "timestamp" in df.columns:
            out_df["timestamp"] = df["timestamp"].astype(str)
        else:
            out_df["timestamp"] = df.index.astype(str)

        out_df["actual_rtt_ms"] = df[target_col].astype(float)
        out_df["raw_decision_score"] = np.round(decision_scores, 4)
        out_df["anomaly_score"] = np.round(norm_scores, 4)
        out_df["is_anomaly"] = raw_preds == -1
        out_df["anomaly_label"] = np.where(raw_preds == -1, "Anomaly", "Normal")

        # Attach engineered features for transparency
        for col in self.features:
            if col in X.columns:
                out_df[col] = np.round(X[col], 2)

        return out_df

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "round_trip_time"
    ) -> Dict[str, Any]:
        """
        Computes statistical anomaly evaluation metrics on the dataset.
        """
        preds_df = self.predict(df, target_col=target_col)
        total_obs = len(preds_df)
        anomaly_count = int(preds_df["is_anomaly"].sum())
        normal_count = total_obs - anomaly_count
        anomaly_pct = (anomaly_count / total_obs) * 100.0 if total_obs > 0 else 0.0

        anom_rtt = preds_df.loc[preds_df["is_anomaly"], "actual_rtt_ms"]
        norm_rtt = preds_df.loc[~preds_df["is_anomaly"], "actual_rtt_ms"]

        metrics = {
            "total_observations": total_obs,
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_percentage": round(anomaly_pct, 2),
            "anomaly_rtt_mean_ms": round(float(anom_rtt.mean()), 2) if len(anom_rtt) > 0 else 0.0,
            "anomaly_rtt_median_ms": round(float(anom_rtt.median()), 2) if len(anom_rtt) > 0 else 0.0,
            "normal_rtt_mean_ms": round(float(norm_rtt.mean()), 2) if len(norm_rtt) > 0 else 0.0,
            "normal_rtt_median_ms": round(float(norm_rtt.median()), 2) if len(norm_rtt) > 0 else 0.0,
            "mean_anomaly_score": round(float(preds_df["anomaly_score"].mean()), 4),
            "max_anomaly_score": round(float(preds_df["anomaly_score"].max()), 4),
            "min_anomaly_score": round(float(preds_df["anomaly_score"].min()), 4),
        }

        # 1. Latency Spike Benchmark (RTT > 50.0 ms)
        is_spike = (df[target_col].astype(float) > 50.0).values
        detected = preds_df["is_anomaly"].values
        tp_s = int(np.sum(detected & is_spike))
        fp_s = int(np.sum(detected & ~is_spike))
        fn_s = int(np.sum(~detected & is_spike))
        tn_s = int(np.sum(~detected & ~is_spike))
        
        prec_s = (tp_s / (tp_s + fp_s) * 100.0) if (tp_s + fp_s) > 0 else 0.0
        rec_s = (tp_s / (tp_s + fn_s) * 100.0) if (tp_s + fn_s) > 0 else 0.0
        acc_s = ((tp_s + tn_s) / total_obs * 100.0) if total_obs > 0 else 0.0
        f1_s = (2 * prec_s * rec_s / (prec_s + rec_s)) if (prec_s + rec_s) > 0 else 0.0

        metrics["latency_spike_benchmark"] = {
            "total_spike_records": int(np.sum(is_spike)),
            "true_positives": tp_s,
            "false_positives": fp_s,
            "false_negatives": fn_s,
            "true_negatives": tn_s,
            "precision_pct": round(prec_s, 2),
            "recall_pct": round(rec_s, 2),
            "accuracy_pct": round(acc_s, 2),
            "f1_score_pct": round(f1_s, 2),
            "target_precision_achieved": prec_s >= 90.0
        }

        # 2. Compare with ground-truth attack flags if present in df
        attack_cols = [c for c in df.columns if c.startswith("label_") and c != "label_normal"]
        if attack_cols:
            has_attack = (df[attack_cols].sum(axis=1) > 0).values
            # Cross-tabulation
            tp = int(np.sum(detected & has_attack))
            fp = int(np.sum(detected & ~has_attack))
            fn = int(np.sum(~detected & has_attack))
            tn = int(np.sum(~detected & ~has_attack))
            metrics["external_attack_benchmark"] = {
                "total_attack_records": int(np.sum(has_attack)),
                "overlap_with_detected_anomalies": tp,
                "precision_against_attack_flags": round(tp / (tp + fp) * 100.0, 2) if (tp + fp) > 0 else 0.0,
                "recall_against_attack_flags": round(tp / (tp + fn) * 100.0, 2) if (tp + fn) > 0 else 0.0,
                "note": "Isolation Forest is unsupervised. Attack flags serve strictly as an external operational reference."
            }

        return metrics

    def save_model(self, output_dir: str):
        """
        Saves isolation_forest.pkl, scaler.pkl, and feature_config.json.
        """
        os.makedirs(output_dir, exist_ok=True)

        model_file = os.path.join(output_dir, "isolation_forest.pkl")
        scaler_file = os.path.join(output_dir, "scaler.pkl")
        config_file = os.path.join(output_dir, "feature_config.json")

        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)

        with open(scaler_file, "wb") as f:
            pickle.dump(self.scaler, f)

        meta = {
            "model_type": "IsolationForest",
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "features": self.features,
            "feature_count": len(self.features)
        }
        with open(config_file, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_model(cls, model_dir: str) -> "IsolationForestDetector":
        """
        Loads isolation_forest.pkl, scaler.pkl, and feature_config.json.
        """
        model_file = os.path.join(model_dir, "isolation_forest.pkl")
        scaler_file = os.path.join(model_dir, "scaler.pkl")
        config_file = os.path.join(model_dir, "feature_config.json")

        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            raise FileNotFoundError(f"Model artifacts not found in {model_dir}")

        detector = cls()
        with open(model_file, "rb") as f:
            detector.model = pickle.load(f)

        with open(scaler_file, "rb") as f:
            detector.scaler = pickle.load(f)

        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                meta = json.load(f)
                detector.n_estimators = meta.get("n_estimators", 100)
                detector.contamination = meta.get("contamination", 0.10)
                detector.features = meta.get("features", detector.features)

        detector.is_fitted = True
        return detector
