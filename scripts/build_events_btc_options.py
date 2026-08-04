"""Build top-10 LOB event JSONL for the btc4aug26 option chain (2026-08-01).

Uses the canonical event constructor from the simulation repo (imported, not
copied, per its CANONICAL SOURCE note). Instruments with no trades that day get
an empty trades frame — their books still produce LO/CO/IS events, just no MOs.
"""
import glob
import os
import re
import sys

import pandas as pd

SIM_REPO = os.path.expanduser("~/simulation")
sys.path.insert(0, SIM_REPO)
from volume_set_mtpp.process.event_construction_chunked import EventConstructor  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OB_DIR = os.path.join(ROOT, "data", "btc_options_4aug26")
TR_DIR = os.path.join(ROOT, "data", "btc_options_4aug26_trades")
OUT_DIR = os.path.join(ROOT, "data", "events", "btc_options_4aug26")
DATE = "2026-08-01"
EMPTY_TRADES = pd.DataFrame({"date": pd.Series(dtype="int64"),
                             "price": pd.Series(dtype="float64"),
                             "amount": pd.Series(dtype="float64"),
                             "sell": pd.Series(dtype="bool")})


def main() -> None:
    import gzip
    ob_files = sorted(glob.glob(os.path.join(OB_DIR, "*", "full_order_book_*.csv.gz")))
    print(f"{len(ob_files)} orderbook files")
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for ob in ob_files:
        inst = re.search(r"option_(btc\S+?)_2026", os.path.basename(ob)).group(1)
        subdir = os.path.basename(os.path.dirname(ob))
        tr = os.path.join(TR_DIR, subdir, f"trades_drbt_option_{inst}_{DATE}.csv.gz")
        if os.path.exists(tr):
            trades_df = pd.read_csv(tr, compression="gzip")
        else:
            trades_df = EMPTY_TRADES
        out = os.path.join(OUT_DIR, f"events_{inst}_{DATE}.jsonl.gz")
        # Deribit BTC options: price grid is 0.0001 BTC (0.0005 for higher
        # premiums, an exact multiple). Auto-detection from a day's sparse
        # option trades misfires (e.g. all prints at 0.075 -> tick 0.001,
        # which breaks the IS tick-alignment check against 0.0005-grid books).
        constructor = EventConstructor(k_levels=10, fast_mode=False, tick_size=0.0001)
        with gzip.open(out, "wt", compresslevel=6) as f:
            constructor.construct_events_chunked(
                orderbook_file=ob, trades_df=trades_df,
                chunksize=20000, stream_output=f, jsonl_mode=True)
        n_lines = sum(1 for _ in gzip.open(out, "rt"))
        summary.append((inst, len(trades_df), n_lines))
        print(f"== {inst}: trades={len(trades_df)}, event sets={n_lines}")
    print("\ninstrument, trades, event_sets")
    for inst, nt, ns in summary:
        print(f"{inst},{nt},{ns}")


if __name__ == "__main__":
    main()
