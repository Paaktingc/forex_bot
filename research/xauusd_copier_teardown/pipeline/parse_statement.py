#!/usr/bin/env python3
"""
Phase 1-2: Parse the multi-sheet MT5 daily statement workbook into clean datasets.
Read-only. Does NOT repair data; flags anomalies instead.
"""
import re, sys, json
import openpyxl
import pandas as pd
import numpy as np

XLSX = '/sessions/affectionate-ecstatic-gauss/mnt/uploads/xauusd trading lot.xlsx'
OUT  = '/sessions/affectionate-ecstatic-gauss/mnt/outputs'

NBSP = '\xa0'

def clean(v):
    if v is None: return ''
    s = str(v).replace(NBSP, ' ').strip()
    return s

def num(v):
    s = clean(v).replace(' ', '').replace(',', '')
    if s in ('', '-'): return np.nan
    try: return float(s)
    except: return np.nan

SECTIONS = ['Orders:', 'Deals:', 'Positions:', 'Working Orders:', 'A/C Summary:']

def parse_sheet(ws, sheet_name):
    rows = [[clean(c) for c in r] for r in ws.iter_rows(values_only=True)]
    hdr = rows[0]
    meta = {'sheet': sheet_name}
    for cell in hdr:
        if cell.startswith('A/C No:'):   meta['account'] = cell.split(':',1)[1].strip()
        elif cell.startswith('Name:'):   meta['name'] = cell.split(':',1)[1].strip()
        elif cell.startswith('Currency:'): meta['currency'] = cell.split(':',1)[1].strip()
        elif re.match(r'^\d{4}\.\d{2}\.\d{2}', cell): meta['statement_dt'] = cell

    # locate sections
    idx = {}
    for i, r in enumerate(rows):
        if r and r[0] in SECTIONS:
            idx[r[0]] = i
    order_of = sorted(idx.items(), key=lambda kv: kv[1])

    blocks = {}
    for j,(name,start) in enumerate(order_of):
        end = order_of[j+1][1] if j+1 < len(order_of) else len(rows)
        blocks[name] = rows[start:end]

    out = {'meta': meta}

    def table(block, ncols):
        """block[0]=section title, block[1]=header, rest=data until blank/summary line"""
        if not block: return []
        header = block[1]
        recs = []
        for r in block[2:]:
            if not any(r): continue
            if r[0].startswith('No transactions'): break
            # stop at footer rows (e.g. totals / 'Floating P/L:')
            if not re.match(r'^\d{4}\.\d{2}\.\d{2}', r[0]): continue
            recs.append({header[k].replace(NBSP,'').strip(): r[k] for k in range(min(ncols, len(header)))})
        return recs

    out['orders']    = table(blocks.get('Orders:', []), 11)
    out['deals']     = table(blocks.get('Deals:', []), 13)
    out['positions'] = table(blocks.get('Positions:', []), 11)
    out['working']   = table(blocks.get('Working Orders:', []), 10)

    # summary key/value pairs (col0:col1  and col3:col4)
    summ = {}
    for r in blocks.get('A/C Summary:', [])[1:]:
        if len(r) > 1 and r[0]:
            summ[r[0].rstrip(':')] = r[1]
        if len(r) > 4 and r[3]:
            summ[r[3].rstrip(':')] = r[4]
    out['summary'] = summ
    return out


def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    parsed = [parse_sheet(wb[n], n) for n in wb.sheetnames]

    orders, deals, positions, summaries = [], [], [], []
    for p in parsed:
        sd = p['meta'].get('statement_dt','')
        for r in p['orders']:    r['_sheet']=p['meta']['sheet']; r['_stmt']=sd; orders.append(r)
        for r in p['deals']:     r['_sheet']=p['meta']['sheet']; r['_stmt']=sd; deals.append(r)
        for r in p['positions']: r['_sheet']=p['meta']['sheet']; r['_stmt']=sd; positions.append(r)
        s = dict(p['summary']); s['_sheet']=p['meta']['sheet']; s['_stmt']=sd
        s['_account']=p['meta'].get('account'); s['_currency']=p['meta'].get('currency')
        summaries.append(s)

    do = pd.DataFrame(orders)
    dd = pd.DataFrame(deals)
    dp = pd.DataFrame(positions)
    ds = pd.DataFrame(summaries)

    print("=== RAW SECTION COUNTS ===")
    print("orders rows   :", len(do))
    print("deals rows    :", len(dd))
    print("positions rows:", len(dp))
    print("summary rows  :", len(ds))
    print()
    print("ORDERS cols:", list(do.columns))
    print("DEALS  cols:", list(dd.columns))
    print("POS    cols:", list(dp.columns))
    print("SUMM   cols:", list(ds.columns))
    print()
    print("=== STATEMENT DATES / SHEETS ===")
    print(ds[['_sheet','_stmt','_account','_currency']].to_string(index=False))
    print()
    print("=== SUMMARY TABLE ===")
    with pd.option_context('display.width', 250, 'display.max_columns', 50):
        print(ds.to_string(index=False))

    do.to_csv(f'{OUT}/raw_orders.csv', index=False)
    dd.to_csv(f'{OUT}/raw_deals.csv', index=False)
    dp.to_csv(f'{OUT}/raw_positions.csv', index=False)
    ds.to_csv(f'{OUT}/raw_summary.csv', index=False)
    print("\nWrote raw_orders.csv raw_deals.csv raw_positions.csv raw_summary.csv")

if __name__ == '__main__':
    main()
