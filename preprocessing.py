from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Common feature columns
NUMERIC_COMMON = [
    "order_amount",
    "customer_total_orders",
    "customer_refund_rate",
    "customer_dispute_rate",
    "device_reuse_count",
    "ip_reuse_count",
    "order_velocity_24h",
]

CATEGORICAL_COMMON = [
    "currency",
    "billing_country",
    "shipping_country",
]

BOOLEAN_COMMON = [
    "country_mismatch",
]

# Refund-specific extras
NUMERIC_REFUND_EXTRA = ["days_since_purchase"]
CATEGORICAL_REFUND_EXTRA = ["refund_reason", "item_category"]
BOOLEAN_REFUND_EXTRA = ["delivery_confirmed"]

# Feature sets by model
FEATURE_SETS = {
    "order_risk": {
        "numeric": NUMERIC_COMMON,
        "categorical": CATEGORICAL_COMMON,
        "boolean": BOOLEAN_COMMON,
    },
    "refund_risk": {
        "numeric": NUMERIC_COMMON + NUMERIC_REFUND_EXTRA,
        "categorical": CATEGORICAL_COMMON + CATEGORICAL_REFUND_EXTRA,
        "boolean": BOOLEAN_COMMON + BOOLEAN_REFUND_EXTRA,
    },
    "return_risk": {
        "numeric": NUMERIC_COMMON,
        "categorical": CATEGORICAL_COMMON,
        "boolean": BOOLEAN_COMMON,
    },
}


def build_preprocessor(feature_set: str) -> ColumnTransformer:
    fs = FEATURE_SETS[feature_set]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, fs["numeric"]),
            ("cat", categorical_transformer, fs["categorical"]),
            ("bool", "passthrough", fs["boolean"]),
        ]
    )


def build_pipeline(feature_set: str, xgb_params: dict = None) -> Pipeline:
    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if xgb_params:
        default_params.update(xgb_params)

    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(feature_set)),
        ("classifier", XGBClassifier(**default_params)),
    ])


def get_all_feature_columns(feature_set: str) -> list:
    fs = FEATURE_SETS[feature_set]
    return fs["numeric"] + fs["categorical"] + fs["boolean"]
