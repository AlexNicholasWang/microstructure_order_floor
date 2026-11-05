#!/usr/bin/env python3
"""Batch provisional L1 OFI reconstruction, preserving QC and missing data.

Run from ~/ofi-research:
 python build_ofi_matrix.py --limit 3
 python build_ofi_matrix.py
Raw input files stay on the server. Processes archives sequentially.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--universe', type=Path, default=Path('universe_candidates.csv'))
p.add_argument('--data-dir', type=Path, default=Path('/lobster/2025/20250731'))
p.add_argument('--script', type=Path, default=Path('reconstruct_ofi.py'))
p.add_argument('--output-dir', type=Path, default=Path('ofi_output'))
p.add_argument('--limit', type=int, default=None)
a = p.parse_args()
with a.universe.open(newline='') as f:
    tickers = [r['ticker'].strip() for r in csv.DictReader(f)]
if len(tickers) != len(set(tickers)):
    raise SystemExit('Duplicate tickers in universe')
if a.limit is not None:
    if a.limit < 1:
        raise SystemExit('--limit must be positive')
    tickers = tickers[:a.limit]
a.output_dir.mkdir(parents=True, exist_ok=True)
qc = []
series = {}
for i, ticker in enumerate(tickers, 1):
    archive = a.data_dir / (ticker + '.7z')
    output = a.output_dir / (ticker + '_ofi.csv')
    print(f'[{i}/{len(tickers)}] {ticker}', flush=True)
    record = {'ticker': ticker, 'status': 'failed', 'rows': 0, 'valid_minutes': 0,
              'transitions': 0, 'warnings': '', 'error': ''}
    if not archive.is_file():
        record['error'] = 'Missing archive'
        qc.append(record)
        continue
    try:
        proc = subprocess.run([sys.executable, str(a.script), str(archive), '--output', str(output)],
                              capture_output=True, text=True, check=False)
        log = a.output_dir / (ticker + '.log')
        log.write_text(proc.stdout + '\nSTDERR:\n' + proc.stderr)
        if proc.returncode:
            raise RuntimeError(f'Reconstruction exited {proc.returncode}; see {log}')
        with output.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 390:
            raise ValueError(f'Expected 390 rows; found {len(rows)}')
        minutes = [r['minute'] for r in rows]
        if len(set(minutes)) != 390:
            raise ValueError('Duplicate minute timestamps')
        values = []
        for r in rows:
            valid = r['had_valid_quote_update'] == '1' and r['after_first_valid_anchor'] == '1'
            values.append(float(r['ofi_shares']) if valid else None)
        record['rows'] = len(rows)
        record['valid_minutes'] = sum(x is not None for x in values)
        record['transitions'] = sum(int(r['quote_transitions']) for r in rows)
        match = re.search(r'^Data warnings: (.*)$', proc.stdout, re.MULTILINE)
        other = re.search(r'^Other warnings: (.*)$', proc.stdout, re.MULTILINE)
        record['warnings'] = f'data={match.group(1) if match else "unavailable"}; other={other.group(1) if other else "unavailable"}'
        record['status'] = 'processed_provisional'
        series[ticker] = dict(zip(minutes, values))
    except Exception as exc:
        record['error'] = str(exc)
    qc.append(record)

# Construct union with 390-minute reference from first successfully processed asset.
if not series:
    raise SystemExit('No successful reconstructions. Review ofi_output/*.log')
ref_minutes = list(next(iter(series.values())).keys())
for ticker, data in series.items():
    if set(data) != set(ref_minutes):
        raise SystemExit(f'Minute timestamps differ for {ticker}')
with (a.output_dir / 'ofi_matrix_provisional.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['minute'] + tickers)
    for minute in ref_minutes:
        writer.writerow([minute] + [series.get(t, {}).get(minute) if series.get(t, {}).get(minute) is not None else '' for t in tickers])
with (a.output_dir / 'quality_report.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(qc[0]))
    writer.writeheader()
    writer.writerows(qc)
print('Saved:', a.output_dir / 'ofi_matrix_provisional.csv')
print('QC:', a.output_dir / 'quality_report.csv')
print(f'Processed {len(series)}/{len(tickers)}; missing and unvalidated minutes remain blank.')
print('CAUTION: no snapshot comparison; do not interpret matrix as validated ground-truth OFI.')