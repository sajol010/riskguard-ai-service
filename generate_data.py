import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_common_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    currencies = ["USD", "EUR", "GBP", "CAD", "AUD"]
    countries = ["US", "GB", "CA", "DE", "FR", "AU", "IN", "BR", "NG", "CN"]

    billing = rng.choice(countries, n)
    shipping = rng.choice(countries, n)

    return pd.DataFrame({
        "order_amount": np.round(rng.lognormal(mean=4.0, sigma=1.0, size=n), 2),
        "currency": rng.choice(currencies, n),
        "billing_country": billing,
        "shipping_country": shipping,
        "country_mismatch": (billing != shipping).astype(int),
        "customer_total_orders": rng.poisson(lam=8, size=n),
        "customer_refund_rate": np.clip(np.round(rng.beta(2, 20, size=n), 4), 0, 1),
        "customer_dispute_rate": np.clip(np.round(rng.beta(1, 50, size=n), 4), 0, 1),
        "device_reuse_count": rng.poisson(lam=2, size=n),
        "ip_reuse_count": rng.poisson(lam=1, size=n),
        "order_velocity_24h": rng.poisson(lam=1.5, size=n),
    })


def _calibrate_probabilities(risk_scores: np.ndarray, target_rate: float) -> np.ndarray:
    min_r, max_r = risk_scores.min(), risk_scores.max()
    if max_r - min_r > 0:
        norm = (risk_scores - min_r) / (max_r - min_r)
    else:
        norm = np.full_like(risk_scores, 0.5)
    prob = norm * target_rate * 2
    return np.clip(prob, 0.01, 0.99)


def generate_fraud_labels(df: pd.DataFrame, fraud_rate: float, rng) -> np.ndarray:
    risk = (
        0.3 * df["country_mismatch"]
        + 0.2 * (df["order_velocity_24h"] > 3).astype(float)
        + 0.15 * (df["order_amount"] > 500).astype(float)
        + 0.15 * (df["device_reuse_count"] > 3).astype(float)
        + 0.1 * (df["ip_reuse_count"] > 2).astype(float)
        + 0.1 * df["customer_dispute_rate"] * 10
    )
    prob = _calibrate_probabilities(risk.values, fraud_rate)
    return rng.binomial(1, prob)


def generate_abuse_labels(df: pd.DataFrame, abuse_rate: float, rng) -> np.ndarray:
    risk = (
        0.25 * (df["customer_refund_rate"] > 0.15).astype(float)
        + 0.2 * (df["refund_reason"] == "not_as_described").astype(float)
        + 0.2 * (~df["delivery_confirmed"].astype(bool)).astype(float)
        + 0.15 * (df["days_since_purchase"] > 25).astype(float)
        + 0.1 * (df["order_amount"] > 300).astype(float)
        + 0.1 * (df["customer_dispute_rate"] > 0.05).astype(float)
    )
    prob = _calibrate_probabilities(risk.values, abuse_rate)
    return rng.binomial(1, prob)


def generate_return_labels(df: pd.DataFrame, return_rate: float, rng) -> np.ndarray:
    risk = (
        0.25 * (df["order_amount"] > 200).astype(float)
        + 0.2 * df["country_mismatch"]
        + 0.2 * (df["customer_refund_rate"] > 0.1).astype(float)
        + 0.15 * (df["order_velocity_24h"] > 2).astype(float)
        + 0.2 * (df["customer_total_orders"] < 3).astype(float)
    )
    prob = _calibrate_probabilities(risk.values, return_rate)
    return rng.binomial(1, prob)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--fraud-rate", type=float, default=0.05)
    parser.add_argument("--abuse-rate", type=float, default=0.08)
    parser.add_argument("--return-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Order Fraud Data
    df_order = generate_common_features(args.samples, rng)
    df_order["is_fraud"] = generate_fraud_labels(df_order, args.fraud_rate, rng)
    df_order.to_csv(data_dir / "order_fraud_data.csv", index=False)
    print(f"Order fraud data: {len(df_order)} rows, fraud rate: {df_order['is_fraud'].mean():.3f}")

    # Refund Abuse Data
    df_refund = generate_common_features(args.samples, rng)
    refund_reasons = ["not_as_described", "wrong_item", "damaged", "changed_mind", "other"]
    item_categories = ["electronics", "clothing", "home", "beauty", "sports", "food"]
    df_refund["refund_reason"] = rng.choice(refund_reasons, args.samples)
    df_refund["days_since_purchase"] = rng.poisson(lam=14, size=args.samples)
    df_refund["delivery_confirmed"] = rng.choice([True, False], args.samples, p=[0.85, 0.15])
    df_refund["item_category"] = rng.choice(item_categories, args.samples)
    df_refund["is_abuse"] = generate_abuse_labels(df_refund, args.abuse_rate, rng)
    df_refund.to_csv(data_dir / "refund_abuse_data.csv", index=False)
    print(f"Refund abuse data: {len(df_refund)} rows, abuse rate: {df_refund['is_abuse'].mean():.3f}")

    # Return Data
    df_return = generate_common_features(args.samples, rng)
    df_return["is_returned"] = generate_return_labels(df_return, args.return_rate, rng)
    df_return.to_csv(data_dir / "return_data.csv", index=False)
    print(f"Return data: {len(df_return)} rows, return rate: {df_return['is_returned'].mean():.3f}")


if __name__ == "__main__":
    main()
