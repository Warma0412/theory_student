#!/usr/bin/env python3
"""Build the leakage-controlled Olist seller-month panel.

The pipeline aggregates each source at its natural grain before joining. This
prevents the item x payment x review Cartesian multiplication that otherwise
inflates GMV in a naive all-table merge.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIR = Path(__file__).resolve().parent
V1_DIR = CODE_DIR.parent
PROJECT_DIR = V1_DIR.parents[1]
MAIN_DIR = PROJECT_DIR / "原始数据" / "archive"
FUNNEL_DIR = PROJECT_DIR / "原始数据" / "archive (漏斗维度补充)"
OUT_DIR = V1_DIR / "data_processed"
RESULTS_DIR = V1_DIR / "results"
LOG_DIR = V1_DIR / "logs"

PRIMARY_START = pd.Timestamp("2017-01-01")
PRIMARY_FEATURE_END = pd.Timestamp("2018-07-01")


CANDIDATE_FEATURES = [
    "avg_price",
    "avg_freight",
    "avg_product_weight_g",
    "avg_product_length_cm",
    "avg_product_height_cm",
    "avg_product_width_cm",
    "avg_product_photos_qty",
    "avg_product_name_length",
    "avg_product_description_length",
    "avg_payment_installments",
    "avg_payment_sequential",
    "credit_card_ratio",
    "boleto_ratio",
    "voucher_ratio",
    "debit_card_ratio",
    "avg_review_score",
    "avg_delivery_days",
    "avg_approval_days",
    "avg_ship_days",
    "avg_estimate_gap_days",
    "item_count",
    "order_count",
    "category_count",
    "unique_customer_count",
    "buyer_state_diversity",
    "dominant_customer_state_share",
    "interstate_ratio",
    "avg_distance_km",
    "seller_state_frequency",
    "marketing_origin_frequency",
    "has_marketing_deal",
    "declared_revenue_log",
]

EXCLUDED_FEATURES = {
    "declared_revenue_log": "The feature is constant at zero in the analysis panel.",
}

FEATURES = [
    feature for feature in CANDIDATE_FEATURES if feature not in EXCLUDED_FEATURES
]

FEATURE_LABELS_ZH = {
    "avg_price": "平均商品价格",
    "avg_freight": "平均运费",
    "avg_product_weight_g": "平均商品重量",
    "avg_product_length_cm": "平均商品长度",
    "avg_product_height_cm": "平均商品高度",
    "avg_product_width_cm": "平均商品宽度",
    "avg_product_photos_qty": "平均商品图片数",
    "avg_product_name_length": "平均商品标题长度",
    "avg_product_description_length": "平均商品描述长度",
    "avg_payment_installments": "平均支付分期期数",
    "avg_payment_sequential": "平均支付序列数",
    "credit_card_ratio": "信用卡支付占比",
    "boleto_ratio": "Boleto支付占比",
    "voucher_ratio": "代金券支付占比",
    "debit_card_ratio": "借记卡支付占比",
    "avg_review_score": "平均评价得分",
    "avg_delivery_days": "平均下单至送达天数",
    "avg_approval_days": "平均下单至审核天数",
    "avg_ship_days": "平均审核至承运天数",
    "avg_estimate_gap_days": "平均预计提前送达天数",
    "item_count": "成交件数",
    "order_count": "成交订单数",
    "category_count": "活跃品类数",
    "unique_customer_count": "独立顾客数",
    "buyer_state_diversity": "买家州数量",
    "dominant_customer_state_share": "第一大买家州占比",
    "interstate_ratio": "跨州交易占比",
    "avg_distance_km": "平均卖家-买家距离",
    "seller_state_frequency": "卖家所在州频率",
    "marketing_origin_frequency": "营销来源频率",
    "has_marketing_deal": "是否匹配营销成交线索",
    "declared_revenue_log": "申报月收入对数",
}


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "build_panel.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def read_csv(directory: Path, name: str, **kwargs) -> pd.DataFrame:
    path = directory / name
    logging.info("Reading %s", path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def haversine_km(lat1, lon1, lat2, lon2):
    values = [np.radians(pd.to_numeric(x, errors="coerce")) for x in (lat1, lon1, lat2, lon2)]
    lat1r, lon1r, lat2r, lon2r = values
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def mode_or_nan(values: pd.Series):
    mode = values.dropna().mode()
    return mode.iloc[0] if len(mode) else np.nan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    payments = payments.copy()
    payments["payment_value"] = pd.to_numeric(payments["payment_value"], errors="coerce").fillna(0)
    payments["weighted_installments"] = (
        pd.to_numeric(payments["payment_installments"], errors="coerce").fillna(0)
        * payments["payment_value"]
    )
    base = payments.groupby("order_id", as_index=False).agg(
        payment_value_total=("payment_value", "sum"),
        weighted_installments=("weighted_installments", "sum"),
        avg_payment_sequential=("payment_sequential", "mean"),
    )
    base["avg_payment_installments"] = np.divide(
        base["weighted_installments"],
        base["payment_value_total"],
        out=np.zeros(len(base), dtype=float),
        where=base["payment_value_total"].to_numpy() > 0,
    )
    by_type = (
        payments.pivot_table(
            index="order_id", columns="payment_type", values="payment_value", aggfunc="sum", fill_value=0
        )
        .reset_index()
    )
    base = base.merge(by_type, on="order_id", how="left", validate="1:1")
    mapping = {
        "credit_card": "credit_card_ratio",
        "boleto": "boleto_ratio",
        "voucher": "voucher_ratio",
        "debit_card": "debit_card_ratio",
    }
    for raw, output in mapping.items():
        numerator = base[raw] if raw in base else 0.0
        base[output] = np.divide(
            numerator,
            base["payment_value_total"],
            out=np.zeros(len(base), dtype=float),
            where=base["payment_value_total"].to_numpy() > 0,
        )
    return base[[
        "order_id",
        "avg_payment_installments",
        "avg_payment_sequential",
        "credit_card_ratio",
        "boleto_ratio",
        "voucher_ratio",
        "debit_card_ratio",
    ]]


def build_panel() -> tuple[pd.DataFrame, dict]:
    orders = read_csv(MAIN_DIR, "olist_orders_dataset.csv")
    items = read_csv(MAIN_DIR, "olist_order_items_dataset.csv")
    products = read_csv(MAIN_DIR, "olist_products_dataset.csv")
    sellers = read_csv(MAIN_DIR, "olist_sellers_dataset.csv")
    customers = read_csv(MAIN_DIR, "olist_customers_dataset.csv")
    payments = read_csv(MAIN_DIR, "olist_order_payments_dataset.csv")
    reviews = read_csv(MAIN_DIR, "olist_order_reviews_dataset.csv")
    geolocation = read_csv(MAIN_DIR, "olist_geolocation_dataset.csv")
    closed_deals = read_csv(FUNNEL_DIR, "olist_closed_deals_dataset.csv")
    mql = read_csv(FUNNEL_DIR, "olist_marketing_qualified_leads_dataset.csv")

    source_rows = {
        "orders": len(orders),
        "items": len(items),
        "products": len(products),
        "sellers": len(sellers),
        "customers": len(customers),
        "payments": len(payments),
        "reviews": len(reviews),
        "geolocation": len(geolocation),
        "closed_deals": len(closed_deals),
        "mql": len(mql),
    }

    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in timestamp_cols:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")
    orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    orders["avg_delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    orders["avg_approval_days"] = (
        orders["order_approved_at"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    orders["avg_ship_days"] = (
        orders["order_delivered_carrier_date"] - orders["order_approved_at"]
    ).dt.total_seconds() / 86400
    orders["avg_estimate_gap_days"] = (
        orders["order_estimated_delivery_date"] - orders["order_delivered_customer_date"]
    ).dt.total_seconds() / 86400
    orders.loc[~orders["avg_delivery_days"].between(0, 210), "avg_delivery_days"] = np.nan
    orders.loc[~orders["avg_approval_days"].between(0, 30), "avg_approval_days"] = np.nan
    orders.loc[~orders["avg_ship_days"].between(0, 60), "avg_ship_days"] = np.nan
    orders.loc[~orders["avg_estimate_gap_days"].between(-180, 180), "avg_estimate_gap_days"] = np.nan

    # Keep successful transactions for a consistent realized-GMV estimand.
    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()
    delivered = delivered.merge(
        customers[["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_state"]],
        on="customer_id",
        how="left",
        validate="m:1",
    )

    reviews_order = reviews.groupby("order_id", as_index=False).agg(
        avg_review_score=("review_score", "mean"),
        review_record_count=("review_id", "size"),
    )
    payment_order = aggregate_payments(payments)

    geo = geolocation.copy()
    geo = geo.loc[
        geo["geolocation_lat"].between(-35.0, 6.0)
        & geo["geolocation_lng"].between(-75.0, -30.0)
    ]
    geo = geo.groupby("geolocation_zip_code_prefix", as_index=False).agg(
        latitude=("geolocation_lat", "median"), longitude=("geolocation_lng", "median")
    )
    seller_geo = geo.rename(
        columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "latitude": "seller_lat",
            "longitude": "seller_lng",
        }
    )
    customer_geo = geo.rename(
        columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "latitude": "customer_lat",
            "longitude": "customer_lng",
        }
    )
    seller_state_frequency = sellers["seller_state"].value_counts(normalize=True)
    sellers["seller_state_frequency"] = sellers["seller_state"].map(seller_state_frequency)
    sellers = sellers.merge(seller_geo, on="seller_zip_code_prefix", how="left", validate="m:1")
    delivered = delivered.merge(customer_geo, on="customer_zip_code_prefix", how="left", validate="m:1")

    product_cols = [
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    enriched = items.merge(delivered, on="order_id", how="inner", validate="m:1")
    enriched = enriched.merge(products[product_cols], on="product_id", how="left", validate="m:1")
    enriched = enriched.merge(
        sellers[[
            "seller_id",
            "seller_state",
            "seller_state_frequency",
            "seller_lat",
            "seller_lng",
        ]],
        on="seller_id",
        how="left",
        validate="m:1",
    )
    enriched["interstate"] = (
        enriched["seller_state"].notna()
        & enriched["customer_state"].notna()
        & enriched["seller_state"].ne(enriched["customer_state"])
    ).astype(float)
    enriched["distance_km"] = haversine_km(
        enriched["seller_lat"], enriched["seller_lng"], enriched["customer_lat"], enriched["customer_lng"]
    )
    enriched.loc[~enriched["distance_km"].between(0, 5000), "distance_km"] = np.nan

    renamed = {
        "product_weight_g": "avg_product_weight_g",
        "product_length_cm": "avg_product_length_cm",
        "product_height_cm": "avg_product_height_cm",
        "product_width_cm": "avg_product_width_cm",
        "product_photos_qty": "avg_product_photos_qty",
        "product_name_lenght": "avg_product_name_length",
        "product_description_lenght": "avg_product_description_length",
    }
    enriched = enriched.rename(columns=renamed)
    item_agg = enriched.groupby(["seller_id", "month"], as_index=False).agg(
        gmv=("price", "sum"),
        avg_price=("price", "mean"),
        avg_freight=("freight_value", "mean"),
        avg_product_weight_g=("avg_product_weight_g", "mean"),
        avg_product_length_cm=("avg_product_length_cm", "mean"),
        avg_product_height_cm=("avg_product_height_cm", "mean"),
        avg_product_width_cm=("avg_product_width_cm", "mean"),
        avg_product_photos_qty=("avg_product_photos_qty", "mean"),
        avg_product_name_length=("avg_product_name_length", "mean"),
        avg_product_description_length=("avg_product_description_length", "mean"),
        item_count=("order_item_id", "size"),
        order_count=("order_id", "nunique"),
        category_count=("product_category_name", "nunique"),
        unique_customer_count=("customer_unique_id", "nunique"),
        buyer_state_diversity=("customer_state", "nunique"),
        interstate_ratio=("interstate", "mean"),
        avg_distance_km=("distance_km", "mean"),
        seller_state_frequency=("seller_state_frequency", "first"),
    )
    state_counts = (
        enriched.groupby(["seller_id", "month", "customer_state"], as_index=False)
        .size()
        .groupby(["seller_id", "month"], as_index=False)["size"]
        .max()
        .rename(columns={"size": "dominant_customer_state_items"})
    )
    item_agg = item_agg.merge(state_counts, on=["seller_id", "month"], how="left", validate="1:1")
    item_agg["dominant_customer_state_share"] = (
        item_agg["dominant_customer_state_items"] / item_agg["item_count"]
    )

    seller_orders = enriched[["seller_id", "month", "order_id"]].drop_duplicates()
    order_features = delivered[[
        "order_id",
        "avg_delivery_days",
        "avg_approval_days",
        "avg_ship_days",
        "avg_estimate_gap_days",
    ]].merge(reviews_order, on="order_id", how="left", validate="1:1")
    order_features = order_features.merge(payment_order, on="order_id", how="left", validate="1:1")
    seller_orders = seller_orders.merge(order_features, on="order_id", how="left", validate="m:1")
    order_feature_cols = [
        "avg_payment_installments",
        "avg_payment_sequential",
        "credit_card_ratio",
        "boleto_ratio",
        "voucher_ratio",
        "debit_card_ratio",
        "avg_review_score",
        "avg_delivery_days",
        "avg_approval_days",
        "avg_ship_days",
        "avg_estimate_gap_days",
    ]
    order_agg = seller_orders.groupby(["seller_id", "month"], as_index=False)[order_feature_cols].mean()
    panel = item_agg.merge(order_agg, on=["seller_id", "month"], how="left", validate="1:1")

    deals = closed_deals.merge(mql, on="mql_id", how="left", validate="m:1")
    origin_frequency = deals["origin"].value_counts(normalize=True)
    seller_marketing = deals.groupby("seller_id", as_index=False).agg(
        marketing_origin=("origin", mode_or_nan),
        declared_monthly_revenue=("declared_monthly_revenue", "median"),
        marketing_deal_count=("mql_id", "size"),
    )
    seller_marketing["marketing_origin_frequency"] = seller_marketing["marketing_origin"].map(origin_frequency)
    seller_marketing["has_marketing_deal"] = 1.0
    seller_marketing["declared_revenue_log"] = np.log1p(
        pd.to_numeric(seller_marketing["declared_monthly_revenue"], errors="coerce").clip(lower=0)
    )
    panel = panel.merge(
        seller_marketing[[
            "seller_id",
            "marketing_origin_frequency",
            "has_marketing_deal",
            "declared_revenue_log",
        ]],
        on="seller_id",
        how="left",
        validate="m:1",
    )
    panel["has_marketing_deal"] = panel["has_marketing_deal"].fillna(0.0)
    panel["marketing_origin_frequency"] = panel["marketing_origin_frequency"].fillna(0.0)
    panel["declared_revenue_log"] = panel["declared_revenue_log"].fillna(0.0)

    future = panel[["seller_id", "month", "gmv", "order_count"]].copy()
    future["month"] = future["month"] - pd.offsets.MonthBegin(1)
    future = future.rename(columns={"gmv": "gmv_next_month", "order_count": "orders_next_month"})
    panel = panel.merge(future, on=["seller_id", "month"], how="left", validate="1:1")
    panel[["gmv_next_month", "orders_next_month"]] = panel[["gmv_next_month", "orders_next_month"]].fillna(0)
    panel["log_gmv"] = np.log1p(panel["gmv"])
    panel["log_gmv_next_month"] = np.log1p(panel["gmv_next_month"])
    panel["aov_next_month"] = np.divide(
        panel["gmv_next_month"],
        panel["orders_next_month"],
        out=np.zeros(len(panel), dtype=float),
        where=panel["orders_next_month"].to_numpy() > 0,
    )
    panel["log_aov_next_month"] = np.log1p(panel["aov_next_month"])
    panel["year_month"] = panel["month"].dt.strftime("%Y-%m")
    panel = panel.loc[
        panel["month"].between(PRIMARY_START, PRIMARY_FEATURE_END)
    ].sort_values(["month", "seller_id"]).reset_index(drop=True)

    duplicate_keys = int(panel.duplicated(["seller_id", "month"]).sum())
    if duplicate_keys:
        raise ValueError(f"Panel has {duplicate_keys} duplicated seller-month keys")
    if not np.isclose(panel["gmv"].sum(), enriched.loc[
        enriched["month"].between(PRIMARY_START, PRIMARY_FEATURE_END), "price"
    ].sum()):
        raise ValueError("GMV reconciliation failed")

    missing_rates = panel[CANDIDATE_FEATURES].isna().mean().sort_values(ascending=False)
    audit = {
        "source_rows": source_rows,
        "delivered_orders": int(len(delivered)),
        "delivered_items": int(len(enriched)),
        "panel_rows": int(len(panel)),
        "panel_sellers": int(panel["seller_id"].nunique()),
        "panel_months": int(panel["month"].nunique()),
        "panel_start": panel["year_month"].min(),
        "panel_end": panel["year_month"].max(),
        "gmv_total_brl": float(panel["gmv"].sum()),
        "next_month_zero_rate": float(panel["gmv_next_month"].eq(0).mean()),
        "duplicate_panel_keys": duplicate_keys,
        "missing_rate_max": float(missing_rates.max()),
        "missing_rate_by_feature": {k: float(v) for k, v in missing_rates.items()},
        "matched_marketing_sellers": int(seller_marketing["seller_id"].nunique()),
        "candidate_features_constructed": len(CANDIDATE_FEATURES),
        "analysis_features": len(FEATURES),
        "excluded_features": EXCLUDED_FEATURES,
    }
    return panel, audit


def save_outputs(panel: pd.DataFrame, audit: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUT_DIR / "seller_month_panel.csv"
    panel.to_csv(panel_path, index=False)
    audit["panel_sha256"] = sha256(panel_path)
    with (RESULTS_DIR / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    pd.DataFrame([
        {
            "feature": feature,
            "label_zh": FEATURE_LABELS_ZH[feature],
            "analysis_included": feature in FEATURES,
            "exclusion_reason": EXCLUDED_FEATURES.get(feature, ""),
        }
        for feature in CANDIDATE_FEATURES
    ]).to_csv(RESULTS_DIR / "feature_dictionary.csv", index=False)
    logging.info("Saved %s rows to %s", len(panel), panel_path)
    logging.info("Audit: %s", json.dumps(audit, ensure_ascii=False))


def main() -> None:
    configure_logging()
    panel, audit = build_panel()
    save_outputs(panel, audit)


if __name__ == "__main__":
    main()
