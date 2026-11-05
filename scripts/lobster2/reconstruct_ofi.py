#!/usr/bin/env python3
"""Provisional Level-1 OFI from a message-only LOBSTER-style .7z archive.

Usage: python reconstruct_ofi.py /lobster/2025/20250731/EBAY.7z --output ebay_ofi_20250731.csv
No external Python packages required. Raw market data stays on the source server.
"""
import argparse
import csv
import heapq
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

OPEN = 9 * 3600 + 30 * 60
CLOSE = 16 * 3600
MINUTES = (CLOSE - OPEN) // 60

class Book:
    def __init__(self):
        self.orders = {}  # id -> (side, price, remaining shares)
        self.levels = {1: defaultdict(int), -1: defaultdict(int)}
        self.heaps = {1: [], -1: []}
        self.stats = defaultdict(int)

    def change_level(self, side, price, delta):
        levels = self.levels[side]
        old = levels[price]
        new = old + delta
        if new < 0:
            self.stats['negative_level_attempts'] += 1
            return False
        if old == 0 and new > 0:
            heapq.heappush(self.heaps[side], -price if side == 1 else price)
        if new:
            levels[price] = new
        else:
            levels.pop(price, None)
        return True

    def quote_side(self, side):
        heap = self.heaps[side]
        levels = self.levels[side]
        while heap and levels.get(-heap[0] if side == 1 else heap[0], 0) <= 0:
            heapq.heappop(heap)
        if not heap:
            return None
        price = -heap[0] if side == 1 else heap[0]
        return price, levels[price]

    def quote(self):
        bid, ask = self.quote_side(1), self.quote_side(-1)
        if bid is None or ask is None:
            return None
        if bid[0] >= ask[0]:
            self.stats['crossed_or_locked_states'] += 1
            return None
        return bid, ask

    def update(self, event, order_id, qty, price, side):
        if event == 1:  # insertion
            if side not in (1, -1) or qty <= 0 or order_id in self.orders:
                self.stats['invalid_additions'] += 1
                return False
            self.orders[order_id] = (side, price, qty)
            self.change_level(side, price, qty)
            return True
        if event not in (2, 3, 4):  # hidden trades, cross trades, halts: no displayed book change
            return False
        order = self.orders.get(order_id)
        if order is None:
            self.stats['unmatched_removals'] += 1
            return False
        old_side, old_price, remaining = order
        if side != old_side or price != old_price:
            self.stats['order_field_mismatches'] += 1
        if event == 3:
            if qty != remaining:
                self.stats['full_delete_size_mismatches'] += 1
            remove = remaining
        else:
            if qty <= 0 or qty > remaining:
                self.stats['invalid_partial_removals'] += 1
                return False
            remove = qty
        if not self.change_level(old_side, old_price, -remove):
            return False
        remaining -= remove
        if remaining:
            self.orders[order_id] = (old_side, old_price, remaining)
        else:
            del self.orders[order_id]
        return True


def ofi(previous, current):
    (pb0, qb0), (pa0, qa0) = previous
    (pb1, qb1), (pa1, qa1) = current
    return ((qb1 if pb1 >= pb0 else 0) - (qb0 if pb1 <= pb0 else 0)
            - (qa1 if pa1 <= pa0 else 0) + (qa0 if pa1 >= pa0 else 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('ofi_1min.csv'))
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f'Archive not found: {args.archive}')

    book = Book()
    counts = defaultdict(int)
    minute_ofi = [0] * MINUTES
    minute_changes = [0] * MINUTES
    minute_valid = [False] * MINUTES
    previous = None
    last_valid = None
    anchored = False
    first_anchor_time = None

    proc = subprocess.Popen(['7z', 'x', '-so', str(args.archive)], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        for row in csv.reader(proc.stdout):
            counts['rows'] += 1
            if len(row) < 6:
                counts['malformed_rows'] += 1
                continue
            try:
                t = float(row[0]); event = int(row[1]); oid = int(row[2])
                qty = int(row[3]); price = int(row[4]); side = int(row[5])
            except ValueError:
                counts['malformed_rows'] += 1
                continue
            counts[f'event_{event}'] += 1
            changed = book.update(event, oid, qty, price, side)
            if t < OPEN or t >= CLOSE:
                continue
            m = int((t - OPEN) // 60)
            if changed:
                current = book.quote()
                if current is None:
                    counts['invalid_quote_events'] += 1
                    previous = None
                    continue
                minute_valid[m] = True
                last_valid = current
                if not anchored:
                    anchored = True
                    first_anchor_time = t
                    previous = current  # no OFI across the opening boundary
                    continue
                if previous is not None:
                    minute_ofi[m] += ofi(previous, current)
                    minute_changes[m] += 1
                else:
                    counts['reanchors_after_invalid'] += 1
                previous = current
    finally:
        proc.stdout.close()
        result = proc.wait()
    if result:
        raise RuntimeError(f'7z returned exit status {result}')

    # A minute without updates after anchoring has zero observed quote-change OFI,
    # but no quote-level sampling/coverage guarantee is implied.
    date_text = args.archive.parent.name
    if len(date_text) == 8 and date_text.isdigit():
        date = datetime.strptime(date_text, '%Y%m%d')
    else:
        date = datetime(2000, 1, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['minute', 'ofi_shares', 'quote_transitions', 'had_valid_quote_update',
                    'after_first_valid_anchor'])
        first_valid_minute = int((first_anchor_time - OPEN) // 60) if first_anchor_time is not None else MINUTES
        for m in range(MINUTES):
            clock = date + timedelta(seconds=OPEN + m * 60)
            w.writerow([clock.strftime('%Y-%m-%d %H:%M'), minute_ofi[m], minute_changes[m],
                        int(minute_valid[m]), int(m >= first_valid_minute)])

    print('Output:', args.output.resolve())
    print('Rows read:', counts['rows'])
    print('Event counts:', {k: v for k, v in sorted(counts.items()) if k.startswith('event_')})
    print('First regular-hours valid anchor (seconds after midnight):', first_anchor_time)
    print('OFI transitions:', sum(minute_changes))
    print('Minutes with a valid quote update:', sum(minute_valid), 'of', MINUTES)
    print('Remaining active orders:', len(book.orders))
    print('Data warnings:', {k: v for k, v in sorted(book.stats.items())})
    print('Other warnings:', {k: v for k, v in sorted(counts.items()) if k not in ('rows',) and not k.startswith('event_')})
    print('CAUTION: provisional reconstruction; no independent initial-depth snapshot or ground-truth book validation.')

if __name__ == '__main__':
    main()