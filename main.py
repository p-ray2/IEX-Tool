"""IEX Market Microstructure Analyzer

Commands:
    ingest   --feed tops|dpls|both  [--limit N] [--overwrite]
    summary
    report   [--top N]
    notional [--top N]
    query    <SYMBOL>
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "cache", "market_data.duckdb")

TOPS_PCAP = os.path.join(BASE_DIR, "data_feeds_20260518_20260518_IEXTP1_TOPS1.6.pcap")
DPLS_PCAP = os.path.join(BASE_DIR, "data_feeds_20260518_20260518_IEXTP1_DPLS1.0.pcap")


def cmd_ingest(args) -> None:
    from analysis.ingest import ingest_tops, ingest_dpls

    feed = args.feed
    limit = args.limit

    if feed in ("tops", "both"):
        if not os.path.exists(TOPS_PCAP):
            sys.exit(f"TOPS pcap not found: {TOPS_PCAP}")
        ingest_tops(TOPS_PCAP, DB_PATH, limit=limit, overwrite=args.overwrite)

    if feed in ("dpls", "both"):
        if not os.path.exists(DPLS_PCAP):
            sys.exit(f"DPLS pcap not found: {DPLS_PCAP}")
        ingest_dpls(DPLS_PCAP, DB_PATH, limit=limit, overwrite=args.overwrite)


def cmd_summary(args) -> None:
    if not os.path.exists(DB_PATH):
        sys.exit("No database found. Run 'ingest' first.")
    from analysis.reports import print_summary_report
    print_summary_report(DB_PATH, date_filter=args.date)


def cmd_report(args) -> None:
    if not os.path.exists(DB_PATH):
        sys.exit("No database found. Run 'ingest' first.")
    from analysis.reports import print_market_report
    print_market_report(DB_PATH, top_n=args.top, date_filter=args.date)


def cmd_notional(args) -> None:
    if not os.path.exists(DB_PATH):
        sys.exit("No database found. Run 'ingest' first.")
    from analysis.reports import print_top_notional_report
    print_top_notional_report(DB_PATH, top_n=args.top, date_filter=args.date)


def cmd_query(args) -> None:
    if not os.path.exists(DB_PATH):
        sys.exit("No database found. Run 'ingest' first.")
    from analysis.reports import print_symbol_report
    print_symbol_report(DB_PATH, args.symbol, date_filter=args.date)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IEX Market Microstructure Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Parse PCAP files into the database")
    p_ingest.add_argument(
        "--feed", choices=["tops", "dpls", "both"], default="both",
        help="Which feed to parse (default: both)",
    )
    p_ingest.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Stop after N UDP packets (for testing)",
    )
    p_ingest.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing rows for this feed before ingesting",
    )

    # summary
    p_summary = sub.add_parser("summary", help="Print high-level data summary (symbols, trades, shares)")
    p_summary.add_argument("--date", metavar="YYYY-MM-DD", default=None,
                           help="Filter to a single trade date (default: all dates)")

    # report
    p_report = sub.add_parser("report", help="Print microstructure report for top symbols")
    p_report.add_argument("--top", type=int, default=25, metavar="N",
                          help="Number of top symbols to display (default: 25)")
    p_report.add_argument("--date", metavar="YYYY-MM-DD", default=None,
                          help="Filter to a single trade date (default: all dates)")

    # notional
    p_notional = sub.add_parser("notional", help="Print top N symbols by notional value")
    p_notional.add_argument("--top", type=int, default=25, metavar="N",
                            help="Number of top symbols to display (default: 25)")
    p_notional.add_argument("--date", metavar="YYYY-MM-DD", default=None,
                            help="Filter to a single trade date (default: all dates)")

    # query
    p_query = sub.add_parser("query", help="Print detailed stats for one symbol")
    p_query.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    p_query.add_argument("--date", metavar="YYYY-MM-DD", default=None,
                         help="Filter to a single trade date (default: all dates)")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "notional":
        cmd_notional(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
