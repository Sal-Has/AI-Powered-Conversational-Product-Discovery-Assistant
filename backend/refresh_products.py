import os
from datetime import datetime

from flask import Flask

from app import create_app
from models import db, ProductIndex
from chat_api import fetch_live_status


def refresh_top_products(limit: int = 100) -> None:
    """Refresh live status for a batch of products using ProductIndex.

    Strategy:
    - Order by last_checked_at ascending (oldest or never-checked first)
    - Update up to `limit` rows by calling fetch_live_status on their URL.
    """
    with create_app().app_context():
        # Select products with the oldest last_checked_at (or never checked)
        products = (
            db.session.query(ProductIndex)
            .order_by(ProductIndex.last_checked_at.asc().nullsfirst())
            .limit(limit)
            .all()
        )

        if not products:
            print("No ProductIndex rows found. Seed this table by calling the /api/products/<product_id>/check_live endpoint from the app first.")
            return

        print(f"Refreshing live status for {len(products)} products...")

        for idx in products:
            if not idx.url:
                continue

            price_num, price_text, status = fetch_live_status(idx.url)
            idx.last_checked_at = datetime.utcnow()

            if price_text:
                idx.last_seen_price_text = price_text
            if price_num is not None:
                idx.last_seen_price_numeric = price_num
            if status:
                idx.last_seen_status = status

        try:
            db.session.commit()
            print("✅ Live status refresh completed.")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Failed to commit live status updates: {e}")


if __name__ == "__main__":
    # You can adjust the limit via an environment variable if desired
    limit_env = os.getenv("REFRESH_LIMIT")
    limit = int(limit_env) if limit_env and limit_env.isdigit() else 100
    refresh_top_products(limit=limit)
