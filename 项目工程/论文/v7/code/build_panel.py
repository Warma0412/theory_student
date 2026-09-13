"""Reconstruct Olist predictors using information observed by each month end."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
RAW = PROJECT / "原始数据/论文实际使用数据汇总/01_实际读取原始表"
FEATURES = [
    "avg_price", "avg_freight", "avg_product_weight_g", "avg_product_length_cm",
    "avg_product_height_cm", "avg_product_width_cm", "avg_product_photos_qty",
    "avg_product_name_length", "avg_product_description_length",
    "avg_payment_installments", "avg_payment_sequential", "credit_card_ratio",
    "boleto_ratio", "voucher_ratio", "debit_card_ratio", "avg_review_score",
    "avg_delivery_days", "avg_approval_days", "avg_ship_days",
    "avg_estimate_gap_days", "item_count", "order_count", "category_count",
    "unique_customer_count", "buyer_state_diversity",
    "dominant_customer_state_share", "interstate_ratio", "avg_distance_km",
    "seller_state_frequency", "marketing_origin_frequency", "has_marketing_deal",
]
LABELS = [
    "平均商品价格", "平均运费", "平均商品重量", "平均商品长度",
    "平均商品高度", "平均商品宽度", "平均商品图片数", "平均商品标题长度",
    "平均商品描述长度", "平均支付分期期数", "平均支付序列数",
    "信用卡支付占比", "Boleto支付占比", "代金券支付占比", "借记卡支付占比",
    "月末可见平均评价", "月末已送达平均天数", "月末已审核平均天数",
    "月末已承运平均准备天数", "月末已送达预计提前天数", "购买件数",
    "购买订单数", "活跃品类数", "独立顾客数", "买家州数量",
    "第一大买家州占比", "跨州交易占比", "平均买卖家距离",
    "已观察卖家州频率", "已成交营销来源频率", "月末已匹配营销成交",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"olist_{name}_dataset.csv", low_memory=False)


def distance(a, b, c, d):
    a, b, c, d = (np.radians(x) for x in (a, b, c, d))
    h = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 12742.0176 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def build() -> None:
    sources = {}
    for path in sorted(RAW.glob("*.csv")):
        sources[path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    orders, items = read("orders"), read("order_items")
    products, customers, sellers = read("products"), read("customers"), read("sellers")
    payments, reviews, geo = read("order_payments"), read("order_reviews"), read("geolocation")
    deals, leads = read("closed_deals"), read("marketing_qualified_leads")
    for name in [
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ]:
        orders[name] = pd.to_datetime(orders[name], errors="coerce")
    orders["month"] = orders.order_purchase_timestamp.dt.to_period("M").dt.to_timestamp()
    orders["cutoff"] = orders.month + pd.offsets.MonthBegin(1)
    orders["target_month"] = orders["cutoff"]
    cutoff = orders.cutoff
    approval = orders.order_approved_at
    carrier = orders.order_delivered_carrier_date
    delivered = orders.order_delivered_customer_date
    purchase = orders.order_purchase_timestamp
    specs = {
        "avg_approval_days": (approval, purchase, approval < cutoff, 0, 30),
        "avg_ship_days": (carrier, approval, (carrier < cutoff) & (approval < cutoff), 0, 60),
        "avg_delivery_days": (delivered, purchase, delivered < cutoff, 0, 210),
        "avg_estimate_gap_days": (
            orders.order_estimated_delivery_date, delivered, delivered < cutoff, -180, 180
        ),
    }
    hidden_counts = {}
    for feature, (end, start, visible, low, high) in specs.items():
        value = (end - start).dt.total_seconds() / 86400
        hidden_counts[feature] = int((value.notna() & ~visible).sum())
        orders[feature] = value.where(visible & value.between(low, high))
    review_time = pd.to_datetime(reviews.review_answer_timestamp, errors="coerce")
    reviews["available_at"] = review_time
    reviews = reviews.merge(orders[["order_id", "cutoff"]], on="order_id", validate="m:1")
    hidden_counts["review_records_after_cutoff"] = int((reviews.available_at >= reviews.cutoff).sum())
    visible_reviews = reviews.loc[reviews.available_at < reviews.cutoff]
    review_agg = visible_reviews.groupby("order_id").review_score.mean().rename("avg_review_score")
    orders = orders.merge(review_agg, on="order_id", how="left", validate="1:1")
    payments["weighted_installments"] = payments.payment_value * payments.payment_installments
    pay = payments.groupby("order_id").agg(
        total=("payment_value", "sum"), weighted=("weighted_installments", "sum"),
        avg_payment_sequential=("payment_sequential", "mean"),
    )
    pay["avg_payment_installments"] = pay.weighted / pay.total.where(pay.total > 0)
    by_type = payments.pivot_table(index="order_id", columns="payment_type",
                                  values="payment_value", aggfunc="sum", fill_value=0)
    for name in ("credit_card", "boleto", "voucher", "debit_card"):
        pay[name + "_ratio"] = by_type.get(name, 0) / pay.total.where(pay.total > 0)
    orders = orders.merge(pay.drop(columns=["total", "weighted"]), on="order_id",
                          how="left", validate="1:1")
    orders = orders.merge(customers, on="customer_id", validate="m:1")
    geo = geo.loc[geo.geolocation_lat.between(-35, 6) & geo.geolocation_lng.between(-75, -30)]
    geo = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
    for side in ("seller", "customer"):
        frame = sellers if side == "seller" else orders
        frame = frame.merge(
            geo.rename(columns={"geolocation_lat": side + "_lat", "geolocation_lng": side + "_lng"}),
            left_on=side + "_zip_code_prefix", right_index=True, how="left", validate="m:1",
        )
        if side == "seller":
            sellers = frame
        else:
            orders = frame
    enriched = items.merge(orders, on="order_id", validate="m:1")
    enriched = enriched.merge(products, on="product_id", how="left", validate="m:1")
    enriched = enriched.merge(sellers, on="seller_id", how="left", validate="m:1")
    enriched["distance"] = distance(enriched.seller_lat, enriched.seller_lng,
                                     enriched.customer_lat, enriched.customer_lng)
    enriched["distance"] = enriched.distance.where(enriched.distance.between(0, 5000))
    enriched["interstate"] = (enriched.seller_state != enriched.customer_state).astype(float)
    products_map = {
        "product_weight_g": "avg_product_weight_g",
        "product_length_cm": "avg_product_length_cm",
        "product_height_cm": "avg_product_height_cm",
        "product_width_cm": "avg_product_width_cm",
        "product_photos_qty": "avg_product_photos_qty",
        "product_name_lenght": "avg_product_name_length",
        "product_description_lenght": "avg_product_description_length",
    }
    aggregations = {
        "gmv_booked": ("price", "sum"), "avg_price": ("price", "mean"),
        "avg_freight": ("freight_value", "mean"), "item_count": ("order_item_id", "size"),
        "order_count": ("order_id", "nunique"), "category_count": ("product_category_name", "nunique"),
        "unique_customer_count": ("customer_unique_id", "nunique"),
        "buyer_state_diversity": ("customer_state", "nunique"),
        "interstate_ratio": ("interstate", "mean"), "avg_distance_km": ("distance", "mean"),
        "seller_state": ("seller_state", "first"),
    }
    aggregations.update({new: (old, "mean") for old, new in products_map.items()})
    panel = enriched.groupby(["seller_id", "month"]).agg(**aggregations).reset_index()
    state_max = enriched.groupby(["seller_id", "month", "customer_state"]).size().groupby(level=[0, 1]).max()
    panel = panel.merge(state_max.rename("dominant_count"), on=["seller_id", "month"], validate="1:1")
    panel["dominant_customer_state_share"] = panel.dominant_count / panel.item_count
    order_cols = [c for c in FEATURES if c.startswith("avg_payment") or c.endswith("_ratio")
                  and c != "interstate_ratio"] + list(specs) + ["avg_review_score"]
    unique_orders = enriched.drop_duplicates(["seller_id", "month", "order_id"])
    order_panel = unique_orders.groupby(["seller_id", "month"])[order_cols].mean()
    panel = panel.merge(order_panel, on=["seller_id", "month"], how="left", validate="1:1")
    deals["won_date"] = pd.to_datetime(deals.won_date, errors="coerce")
    deals = deals.merge(leads[["mql_id", "origin"]], on="mql_id", how="left", validate="m:1")
    first_observed = enriched.groupby("seller_id").month.min()
    sellers = sellers.set_index("seller_id")
    for month in sorted(panel.month.unique()):
        mask = panel.month == month
        observed = first_observed.index[first_observed < month + pd.offsets.MonthBegin(1)]
        frequencies = sellers.loc[observed].seller_state.value_counts(normalize=True)
        panel.loc[mask, "seller_state_frequency"] = panel.loc[mask, "seller_state"].map(frequencies)
        visible = deals.loc[deals.won_date < month + pd.offsets.MonthBegin(1)]
        origins = visible.origin.fillna("unknown").value_counts(normalize=True)
        latest = visible.sort_values("won_date").drop_duplicates("seller_id", keep="last").set_index("seller_id")
        panel.loc[mask, "has_marketing_deal"] = panel.loc[mask, "seller_id"].isin(latest.index).astype(float)
        seller_origin = latest.origin.fillna("unknown").map(origins)
        panel.loc[mask, "marketing_origin_frequency"] = panel.loc[mask, "seller_id"].map(seller_origin).fillna(0)
    enriched["delivery_month"] = enriched.order_delivered_customer_date.dt.to_period("M").dt.to_timestamp()
    primary = enriched.loc[enriched.order_delivered_customer_date.notna()].groupby(
        ["seller_id", "delivery_month"]).price.sum()
    panel["target_month"] = panel.month + pd.offsets.MonthBegin(1)
    panel = panel.merge(primary.rename("gmv_next_month"), left_on=["seller_id", "target_month"],
                        right_index=True, how="left", validate="1:1")
    retrospective = enriched.loc[enriched.order_status.eq("delivered")].groupby(["seller_id", "month"]).price.sum()
    panel = panel.merge(retrospective.rename("gmv_next_purchase_retrospective"),
                        left_on=["seller_id", "target_month"], right_index=True, how="left", validate="1:1")
    for c in ("gmv_next_month", "gmv_next_purchase_retrospective"):
        panel[c] = panel[c].fillna(0)
    panel["log_gmv_next_month"] = np.log1p(panel.gmv_next_month)
    panel["log_gmv_next_purchase_retrospective"] = np.log1p(panel.gmv_next_purchase_retrospective)
    panel = panel.loc[panel.month.between("2017-01-01", "2018-06-01")].copy()
    panel["split"] = "test"
    panel.loc[panel.target_month <= "2017-12-01", "split"] = "train"
    panel.loc[panel.target_month.between("2018-01-01", "2018-02-01"), "split"] = "tune"
    panel.loc[panel.target_month.between("2018-03-01", "2018-04-01"), "split"] = "rank"
    panel = panel.sort_values(["month", "seller_id"]).reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel))
    assert not panel.duplicated(["seller_id", "month"]).any()
    assert len(FEATURES) == 31 and not panel[FEATURES].replace([np.inf, -np.inf], np.nan).isna().all().any()
    out = ROOT / "data_processed/seller_month_asof.parquet"
    panel.to_parquet(out, index=False)
    dictionary = pd.DataFrame({"feature": FEATURES, "label_zh": LABELS})
    dictionary["min"] = panel[FEATURES].min().to_numpy()
    dictionary["max"] = panel[FEATURES].max().to_numpy()
    dictionary["unique"] = panel[FEATURES].nunique().to_numpy()
    dictionary["missing_fraction"] = panel[FEATURES].isna().mean().to_numpy()
    dictionary.to_csv(ROOT / "results/feature_dictionary.csv", index=False)
    audit = {
        "sources": sources, "source_order_count": len(orders), "source_item_count": len(items),
        "panel_rows": len(panel), "panel_sellers": panel.seller_id.nunique(),
        "panel_months": panel.month.nunique(), "panel_sha256": sha(out),
        "hidden_late_information": hidden_counts,
        "split_rows": panel.groupby("split").size().to_dict(),
        "split_sellers": panel.groupby("split").seller_id.nunique().to_dict(),
        "zero_target_fraction": float(panel.gmv_next_month.eq(0).mean()),
        "gmv_reconciled_total_delivery": float(primary.sum()),
        "source_delivered_item_value": float(enriched.loc[
            enriched.order_delivered_customer_date.notna(), "price"].sum()),
        "source_last_delivery": str(enriched.order_delivered_customer_date.max()),
        "retrospective_target_correlation": float(panel[
            ["log_gmv_next_month", "log_gmv_next_purchase_retrospective"]].corr().iloc[0, 1]),
    }
    assert np.isclose(audit["gmv_reconciled_total_delivery"], audit["source_delivered_item_value"])
    (ROOT / "results/data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "sources"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
