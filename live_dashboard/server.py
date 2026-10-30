import os
import sys
import time
import json
import asyncio
import threading
import collections
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import websocket
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ==============================================================================
# 🎯 6EZ6 (CME Euro FX Futures) LIVE DUAL-ENGINE MASTER TRADING SERVER
# 🛡️ Powered by Marcos López de Prado Microstructure AI + Dynamic Tape Router
# ==============================================================================

import base64

TICK = 0.00005
TICK_VALUE = 6.25  # $6.25 per tick per 1 standard micro contract
CONTRACT_ID = 3267437  # 6EZ6 CME Euro FX Futures
SYMBOL = "6EZ6"

TOKEN = "eyJraWQiOiI1OCIsImFsZyI6IkVkRFNBIn0.eyJzdWIiOiI5NjY0ODA5IiwiZXhwIjoxNzkzMzU4NTU1LCJqdGkiOiItNzM0NzgzOTY1MTg3NjEzNDMwMC0tNjk1OTQzMDAyNTI5MTY4NDA5MCIsInBocyI6LTc5NjcxMDY1LCJlbWFpbCI6ImdhdmFsaTM4YWdnMzhAZ21haWwuY29tIn0.BHlukot1dBmfOoP4cD_X5JpqpBEXAfVfTSslyL7Nt_SWnRmQHEjKGdtp5vg1_PNWgE56db1STDUWAmq3jo5SCQ"
WS_URL = "wss://md.tradovateapi.com/v1/websocket"
WS_DEMO_URL = "wss://md-demo.tradovateapi.com/v1/websocket"

TRADOVATE_REST_URL = "https://live.tradovateapi.com/v1/auth/accesstokenrequest"
TRADOVATE_LOGIN_PAYLOAD = {
  "appId": "tradovate_trader(web)",
  "appVersion": "3.260911.0",
  "chl": "208634754931",
  "cid": "1",
  "deviceId": "6478b472-1342-8826-af96-5b685a8df5e4",
  "enc": True,
  "environment": "live", 
  "name": "gavaliashutosh490@gmail.com",
  "password": "ViEqOXA3cm0mKkVQalNL",
  "sec": "f05206b8ac3ef4e527c9410e39fb2a4da1a801e19f51576d54ae1d398b578f8d"
}
TRADOVATE_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
    "Origin": "https://trader.tradovate.com",
    "Referer": "https://trader.tradovate.com/"
}

def fetch_fresh_token_automatic():
    global TOKEN
    try:
        res = requests.post(TRADOVATE_REST_URL, headers=TRADOVATE_HEADERS, json=TRADOVATE_LOGIN_PAYLOAD, timeout=10)
        if res.status_code == 200:
            data = res.json()
            tok = data.get("mdAccessToken") or data.get("accessToken")
            if tok:
                TOKEN = tok
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔑 [AUTO-AUTH] Fresh Tradovate Token Acquired Automatically! (Expires: {data.get('expirationTime')})")
                return tok
        print(f"⚠️ Tradovate auto-token failed with status {res.status_code}: {res.text[:100]}")
    except Exception as e:
        print(f"⚠️ Tradovate auto-token error: {e}")
    return None

def proactive_token_refresher():
    global TOKEN, current_ws_app
    while True:
        try:
            info = get_token_info(TOKEN)
            rem = info.get("remaining_seconds", 0)
            if rem <= 300: # 5 minutes before expiry
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Token expiring in {rem}s. Fetching fresh token...")
                new_tok = fetch_fresh_token_automatic()
                if new_tok:
                    TOKEN = new_tok
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 Fresh token ready. Refreshing WebSocket connection cleanly...")
                    if current_ws_app:
                        try:
                            current_ws_app.close()
                        except Exception:
                            pass
        except Exception as e:
            print(f"⚠️ proactive_token_refresher error: {e}")
        time.sleep(30)

def get_token_info(tok):
    try:
        payload_b64 = tok.split('.')[1]
        payload_b64 += '=' * ((4 - len(payload_b64) % 4) % 4)
        data = json.loads(base64.b64decode(payload_b64).decode('utf-8'))
        exp = data.get('exp', 0)
        rem = max(0, int(exp - time.time()))
        exp_dt = datetime.fromtimestamp(exp)
        return {
            "exp_ts": exp,
            "exp_str": exp_dt.strftime("%H:%M:%S"),
            "remaining_seconds": rem,
            "remaining_minutes": round(rem / 60, 1),
            "is_expired": rem <= 0
        }
    except Exception:
        return {"exp_ts": 0, "exp_str": "--:--:--", "remaining_seconds": 0, "remaining_minutes": 0, "is_expired": True}

MODEL_DIR = '/Users/admin/Downloads/nifty_level2/eurousd_models_10level'
call_model_path = os.path.join(MODEL_DIR, 'eurousd_consensus_call_model.txt')
put_model_path  = os.path.join(MODEL_DIR, 'eurousd_consensus_put_model.txt')

print("📦 Loading Production Dual-Engine LightGBM Models...")
model_call = lgb.Booster(model_file=call_model_path)
model_put  = lgb.Booster(model_file=put_model_path)
print("✅ Loaded Call & Put Dual-Engine Models successfully!")

DUAL_ENGINE_FEATURES = [
    'delta_l1', 'delta_l2', 'delta_l3', 'delta_l4', 'delta_l5',
    'imb_l1', 'imb_l2', 'imb_l3', 'imb_l4', 'imb_l5',
    'cum_delta_l2', 'cum_delta_l3', 'cum_delta_l4', 'cum_delta_l5',
    'cum_delta_l6', 'cum_delta_l7', 'cum_delta_l8', 'cum_delta_l9', 'cum_delta_l10',
    'cons_bull_l2', 'cons_bull_l3', 'cons_bull_l4', 'cons_bull_l5', 'cons_bull_l6', 'cons_bull_l7', 'cons_bull_l8', 'cons_bull_l9', 'cons_bull_l10',
    'cons_bear_l2', 'cons_bear_l3', 'cons_bear_l4', 'cons_bear_l5', 'cons_bear_l6', 'cons_bear_l7', 'cons_bear_l8', 'cons_bear_l9', 'cons_bear_l10',
    'consensus_depth_bull', 'consensus_depth_bear', 'consensus_depth_spread',
    'delta_slope_near_deep', 'delta_slope_deep_macro',
    'vol_ratio_l2', 'vol_ratio_l3', 'vol_ratio_l5', 'vol_ratio_l10',
    'diff_near', 'diff_deep', 'deep_near_ratio',
    'dist_super_res_ask', 'dist_super_sup_bid',
    'has_wall_at_l1_l2_ask', 'has_wall_at_l1_l2_bid',
    'vacuum_corridor_ask_2t', 'vacuum_corridor_bid_2t',
    'runway_ratio_call', 'runway_ratio_put',
    'ask_center_of_mass', 'bid_center_of_mass',
    'ask_vacuum_openness', 'bid_vacuum_openness',
    'cvd_180s', 'cvd_30s', 'cvd_10s', 'cvd_progression_ratio', 'range_180s_ticks',
    'ask_refill_speed', 'bid_refill_speed', 'ask_depletion_vel', 'bid_depletion_vel',
    'micro_lead_ticks', 'ofi_accel', 'fractal_er',
    'cubic_jerk', 'runup_45s', 'whirlpool_flips_6s'
]

# ==============================================================================
# 📊 SHARED REALTIME STATE
# ==============================================================================
state_lock = threading.Lock()

live_market = {
    "status": "CONNECTING",
    "timestamp": "",
    "mid_price": 0.0,
    "last_trade_price": 0.0,
    "spread": 0.0,
    "bids": [],  # [{"price": ..., "size": ...}] * 10
    "offers": [], # [{"price": ..., "size": ...}] * 10
}

# 1-second sampled rolling history (up to 900s for 15m macro volatility)
history_1s = collections.deque(maxlen=920) 
trade_events_recent = collections.deque(maxlen=300)

engine_state = {
    "active_trade": None,
    "last_trade_exit_time": 0.0,
    "last_call_exit_time": 0.0,
    "last_put_exit_time": 0.0,
    "cooldown_seconds": 20,
    "current_regime": "SCANNING",
    "session_trades": [],
    "total_wins": 0,
    "total_losses": 0,
    "win_rate": 0.0,
    "net_pnl": 0.0,
    "net_ticks": 0.0,
    "latest_signal": {
        "active": False,
        "direction": "SCANNING",
        "engine": "ROUTER",
        "grade": "Monitoring L2 Orderbook...",
        "runway": 0,
        "target_ticks": 2.0,
        "expected_hold": "30s - 65s",
        "predicted_candle": "WAITING FOR SETUP",
        "prob_call": 0.0,
        "prob_put": 0.0,
        "vol_ratio_5": 1.0,
        "vol_ratio_6_10": 1.0,
        "deep_vacuum_score": 0.0,
        "ask_trap_wall": 1.0,
        "bid_trap_wall": 1.0,
        "ask_runway": 10,
        "bid_runway": 10,
        "cvd_15s": 0.0,
        "regime": "DYNAMIC ROUTER"
    }
}
STATE_FILE = os.path.join(os.path.dirname(__file__), "saved_state.json")

def load_persisted_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                if saved.get("active_trade"):
                    engine_state["active_trade"] = saved["active_trade"]
                    print(f"🔄 [STATE RESTORED] Active trade restored: {saved['active_trade']['direction']} @ {saved['active_trade']['entry_price']}")
                if "session_trades" in saved and saved["session_trades"]:
                    engine_state["session_trades"] = saved["session_trades"]
                if "total_wins" in saved:
                    engine_state["total_wins"] = saved["total_wins"]
                if "total_losses" in saved:
                    engine_state["total_losses"] = saved["total_losses"]
                if "win_rate" in saved:
                    engine_state["win_rate"] = saved["win_rate"]
                if "net_pnl" in saved:
                    engine_state["net_pnl"] = saved["net_pnl"]
                if "net_ticks" in saved:
                    engine_state["net_ticks"] = saved["net_ticks"]
        except Exception as e:
            print(f"⚠️ Error loading persisted state: {e}")

def save_persisted_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({
                "active_trade": engine_state["active_trade"],
                "session_trades": engine_state["session_trades"],
                "total_wins": engine_state["total_wins"],
                "total_losses": engine_state["total_losses"],
                "win_rate": engine_state["win_rate"],
                "net_pnl": engine_state["net_pnl"],
                "net_ticks": engine_state["net_ticks"]
            }, f, indent=2)
    except Exception:
        pass

# Attempt immediate load
load_persisted_state()

connected_websockets = set()
current_ws_app = None

# ==============================================================================
# 📡 TRADOVATE WEBSOCKET CLIENT
# ==============================================================================
def tradovate_worker():
    global live_market, current_ws_app, TOKEN
    headers = [
        "Origin: https://trader.tradovate.com",
        "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ]

    while True:
        try:
            if not TOKEN or get_token_info(TOKEN)["is_expired"]:
                print("🔑 [STARTUP] Token expired or missing. Fetching fresh token...")
                fetch_fresh_token_automatic()

            ws = websocket.WebSocket()
            current_ws_app = ws
            ws.connect(WS_URL, header=headers, timeout=15)
            print("🚀 Connected to Tradovate Market Data WebSocket.")
            
            # Wait for 'o'
            m = ws.recv()
            if m == 'o':
                print("🟢 Tradovate handshake 'o' received. Sending auth...")
                ws.send(f"authorize\n2\n\n{TOKEN}")
                resp = ws.recv()
                if resp.startswith('a[{"s":200'):
                    print(f"✅ Tradovate Authorized! Subscribing to {SYMBOL} ({CONTRACT_ID})...")
                    ws.send(f'md/subscribeQuote\n2\n\n{{"symbol":"{SYMBOL}"}}')
                    ws.send(f'md/subscribeDOM\n3\n\n{{"symbol":"{SYMBOL}"}}')
                    ws.send(f'md/subscribeQuote\n4\n\n{{"symbol":{CONTRACT_ID}}}')
                    ws.send(f'md/subscribeDOM\n5\n\n{{"symbol":{CONTRACT_ID}}}')
                    with state_lock:
                        live_market["status"] = "LIVE STREAMING"
                else:
                    print(f"⚠️ Auth error: {resp}. Trying automatic token refresh...")
                    new_t = fetch_fresh_token_automatic()
                    if new_t:
                        ws.send(f"authorize\n2\n\n{new_t}")
                        resp = ws.recv()
                        if resp.startswith('a[{"s":200'):
                            print(f"✅ Tradovate Authorized via Auto-Refresh! Subscribing to {SYMBOL} ({CONTRACT_ID})...")
                            ws.send(f'md/subscribeQuote\n2\n\n{{"symbol":"{SYMBOL}"}}')
                            ws.send(f'md/subscribeDOM\n3\n\n{{"symbol":"{SYMBOL}"}}')
                            ws.send(f'md/subscribeQuote\n4\n\n{{"symbol":{CONTRACT_ID}}}')
                            ws.send(f'md/subscribeDOM\n5\n\n{{"symbol":{CONTRACT_ID}}}')
                            with state_lock:
                                live_market["status"] = "LIVE STREAMING"
                    else:
                        with state_lock:
                            live_market["status"] = "AUTH FAILED"
                        time.sleep(5)
                        continue

            # Stream loop
            while True:
                msg = ws.recv()
                if not msg:
                    break
                if msg == 'h':
                    continue
                if msg.startswith('a['):
                    try:
                        payload = json.loads(msg[1:])
                        for item in payload:
                            if item.get('e') == 'md':
                                process_market_data(item.get('d', {}))
                    except Exception:
                        pass
        except Exception as e:
            err_str = str(e)
            wait_time = 10 if "429" in err_str else 3
            print(f"🔴 Tradovate WS exception: {err_str}, waiting {wait_time}s...")
            with state_lock:
                live_market["status"] = "RATE LIMITED (429)" if "429" in err_str else "RECONNECTING"
            time.sleep(wait_time)

def process_market_data(md):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_ts = time.time()
    
    with state_lock:
        if 'quotes' in md:
            for q in md['quotes']:
                entries = q.get('entries', {})
                if 'Trade' in entries:
                    tp = float(entries['Trade']['price'])
                    live_market["last_trade_price"] = tp
                    trade_events_recent.append({
                        "time": now_ts,
                        "price": tp,
                        "size": entries['Trade'].get('size', 1)
                    })
                if 'Bid' in entries and 'Offer' in entries:
                    bp = float(entries['Bid']['price'])
                    bs = int(entries['Bid'].get('size', 20))
                    ap = float(entries['Offer']['price'])
                    asize = int(entries['Offer'].get('size', 20))
                    if not live_market["bids"]:
                        live_market["bids"] = [{"price": round(bp - k * TICK, 5), "size": bs} for k in range(10)]
                    else:
                        live_market["bids"][0] = {"price": bp, "size": bs}
                    if not live_market["offers"]:
                        live_market["offers"] = [{"price": round(ap + k * TICK, 5), "size": asize} for k in range(10)]
                    else:
                        live_market["offers"][0] = {"price": ap, "size": asize}
                    live_market["mid_price"] = (bp + ap) / 2.0
                    live_market["spread"] = ap - bp
                    live_market["timestamp"] = now_str
                    if live_market["last_trade_price"] == 0.0:
                        live_market["last_trade_price"] = live_market["mid_price"]

        if 'doms' in md:
            for d in md['doms']:
                if 'bids' in d and d['bids']:
                    live_market["bids"] = sorted(d['bids'], key=lambda x: x['price'], reverse=True)[:10]
                if 'offers' in d and d['offers']:
                    live_market["offers"] = sorted(d['offers'], key=lambda x: x['price'])[:10]

                if live_market["bids"] and live_market["offers"]:
                    p_bid = live_market["bids"][0]['price']
                    p_ask = live_market["offers"][0]['price']
                    live_market["mid_price"] = (p_bid + p_ask) / 2.0
                    live_market["spread"] = p_ask - p_bid
                    live_market["timestamp"] = now_str
                    if live_market["last_trade_price"] == 0.0:
                        live_market["last_trade_price"] = live_market["mid_price"]

# ==============================================================================
# 🧠 REAL-TIME FEATURE EXTRACTION & MASTER DUAL-ENGINE EVALUATION
# ==============================================================================
def evaluate_market_engine():
    global engine_state
    now_ts = time.time()
    now_dt = datetime.now()
    now_str = now_dt.strftime("%H:%M:%S")

    with state_lock:
        bids = list(live_market["bids"])
        offers = list(live_market["offers"])
        mid = float(live_market["mid_price"])
        last_trade = float(live_market["last_trade_price"])
        ts_str = live_market["timestamp"]

    if mid == 0.0 or not bids or not offers:
        return None

    # Pad bids to 10 levels if needed
    top_b = bids[0]['price']
    while len(bids) < 10:
        k = len(bids)
        bids.append({"price": round(top_b - k * TICK, 5), "size": 25})

    # Pad offers to 10 levels if needed
    top_a = offers[0]['price']
    while len(offers) < 10:
        k = len(offers)
        offers.append({"price": round(top_a + k * TICK, 5), "size": 25})

    # Trade Flow Aggressor
    aggressor = 1.0 if last_trade > mid else (-1.0 if last_trade < mid else 0.0)
    
    b_sizes = [b['size'] for b in bids]
    a_sizes = [a['size'] for a in offers]
    b5 = sum(b_sizes[:5])
    a5 = sum(a_sizes[:5])
    b10 = sum(b_sizes[:10])
    a10 = sum(a_sizes[:10])

    snap = {
        "time": now_ts,
        "mid": mid,
        "trade": last_trade,
        "aggressor": aggressor,
        "b_sizes": b_sizes,
        "a_sizes": a_sizes,
        "b5": b5, "a5": a5,
        "b10": b10, "a10": a10
    }
    history_1s.append(snap)
    hist_list = list(history_1s)
    H_len = len(hist_list)

    # Rolling window computations
    mids_15m = [h['mid'] for h in hist_list if now_ts - h['time'] <= 900]
    mids_180s = [h['mid'] for h in hist_list if now_ts - h['time'] <= 180]
    mids_300s = [h['mid'] for h in hist_list if now_ts - h['time'] <= 300]
    mids_60s  = [h['mid'] for h in hist_list if now_ts - h['time'] <= 60]

    range_15m = (max(mids_15m) - min(mids_15m)) / TICK if mids_15m else 5.0
    range_180s = (max(mids_180s) - min(mids_180s)) / TICK if mids_180s else 3.0
    range_300s = (max(mids_300s) - min(mids_300s)) / TICK if mids_300s else 3.0

    # Flips (price moves)
    flips_60s = sum(1 for k in range(1, len(mids_60s)) if abs(mids_60s[k] - mids_60s[k-1]) > 0)
    flips_300s = sum(1 for k in range(1, len(mids_300s)) if abs(mids_300s[k] - mids_300s[k-1]) > 0)

    # CVD windows
    cvd_180s = sum(h['aggressor'] for h in hist_list if now_ts - h['time'] <= 180)
    cvd_30s  = sum(h['aggressor'] for h in hist_list if now_ts - h['time'] <= 30)
    cvd_10s  = sum(h['aggressor'] for h in hist_list if now_ts - h['time'] <= 10)
    cvd_15s  = sum(h['aggressor'] for h in hist_list if now_ts - h['time'] <= 15)

    # Dynamic Regime Detector (Fast Trending vs Slow Chop)
    is_fast_regime = (range_15m >= 6.0) or (range_180s >= 3.5 and flips_60s >= 10)
    current_regime_name = "🚀 FAST TRENDING" if is_fast_regime else "🐢 SLOW THICK BOOK"
    engine_state["current_regime"] = current_regime_name

    # Feature Dict for LightGBM
    feat = {}
    for i in range(1, 6):
        bq = bids[i-1]['size'] if i-1 < len(bids) else 0
        aq = offers[i-1]['size'] if i-1 < len(offers) else 0
        feat[f'delta_l{i}'] = float(bq - aq)
        feat[f'imb_l{i}'] = float((bq - aq) / (bq + aq + 1e-5))

    for k in range(2, 11):
        cum_b = sum(bids[j]['size'] for j in range(min(k, len(bids))))
        cum_a = sum(offers[j]['size'] for j in range(min(k, len(offers))))
        feat[f'cum_delta_l{k}'] = float(cum_b - cum_a)

    for k in range(2, 11):
        feat[f'cons_bull_l{k}'] = 1.0 if all(feat.get(f'delta_l{j}', 0) > 0 for j in range(1, min(k, 5)+1)) else 0.0
        feat[f'cons_bear_l{k}'] = 1.0 if all(feat.get(f'delta_l{j}', 0) < 0 for j in range(1, min(k, 5)+1)) else 0.0

    feat['consensus_depth_bull'] = sum(1.0 for k in range(min(10, len(bids))) if bids[k]['size'] > offers[k]['size'])
    feat['consensus_depth_bear'] = sum(1.0 for k in range(min(10, len(offers))) if offers[k]['size'] > bids[k]['size'])
    feat['consensus_depth_spread'] = feat['consensus_depth_bull'] - feat['consensus_depth_bear']

    feat['delta_slope_near_deep'] = float((feat['cum_delta_l5'] - feat['cum_delta_l2']) / 3.0)
    feat['delta_slope_deep_macro'] = float((feat['cum_delta_l10'] - feat['cum_delta_l5']) / 5.0)

    feat['vol_ratio_l2'] = (sum(b_sizes[:2]) + 1.0) / (sum(a_sizes[:2]) + 1.0)
    feat['vol_ratio_l3'] = (sum(b_sizes[:3]) + 1.0) / (sum(a_sizes[:3]) + 1.0)
    feat['vol_ratio_l5'] = (b5 + 1.0) / (a5 + 1.0)
    feat['vol_ratio_l10'] = (b10 + 1.0) / (a10 + 1.0)

    feat['diff_near'] = feat['cum_delta_l3']
    feat['diff_deep'] = feat['cum_delta_l10'] - feat['cum_delta_l3']
    feat['deep_near_ratio'] = feat['diff_deep'] / (abs(feat['diff_near']) + 10.0)

    ask_matrix = np.array([offers[k]['size'] if k < len(offers) else 0 for k in range(10)])
    bid_matrix = np.array([bids[k]['size'] if k < len(bids) else 0 for k in range(10)])
    
    res_idx = np.where(ask_matrix >= 85.0)[0]
    sup_idx = np.where(bid_matrix >= 85.0)[0]
    feat['dist_super_res_ask'] = float(res_idx[0] + 1) if len(res_idx) > 0 else 11.0
    feat['dist_super_sup_bid'] = float(sup_idx[0] + 1) if len(sup_idx) > 0 else 11.0
    feat['has_wall_at_l1_l2_ask'] = 1.0 if feat['dist_super_res_ask'] <= 2.0 else 0.0
    feat['has_wall_at_l1_l2_bid'] = 1.0 if feat['dist_super_sup_bid'] <= 2.0 else 0.0

    # User Quant Architecture: 10-Level Ratio Wall Scanner (>= 1.45x depth imbalance)
    ask_cum = np.cumsum(ask_matrix)
    bid_cum = np.cumsum(bid_matrix)
    ratio_sup_levels = np.where((bid_cum + 1.0) / (ask_cum + 1.0) >= 1.45)[0]
    ratio_runway_put = int(ratio_sup_levels[0] + 1) if len(ratio_sup_levels) > 0 else 10
    
    ratio_res_levels = np.where((ask_cum + 1.0) / (bid_cum + 1.0) >= 1.45)[0]
    ratio_runway_call = int(ratio_res_levels[0] + 1) if len(ratio_res_levels) > 0 else 10

    # 🏛️ Dual-Depth Mathematical Ratio Architecture (L1-L3 Near vs L4-L10 Deep)
    b_near = float(np.sum(bid_matrix[:3])) + 1.0
    a_near = float(np.sum(ask_matrix[:3])) + 1.0
    b_deep = float(np.sum(bid_matrix[3:10])) + 1.0
    a_deep = float(np.sum(ask_matrix[3:10])) + 1.0

    r_near_bull = b_near / a_near
    r_deep_bull = b_deep / a_deep

    r_near_bear = a_near / b_near
    r_deep_bear = a_deep / b_deep

    feat['vacuum_corridor_ask_2t'] = float(sum(a_sizes[:2]))
    feat['vacuum_corridor_bid_2t'] = float(sum(b_sizes[:2]))
    feat['runway_ratio_call'] = (sum(b_sizes[:2]) + 1.0) / (sum(a_sizes[:2]) + 1.0)
    feat['runway_ratio_put']  = (sum(a_sizes[:2]) + 1.0) / (sum(b_sizes[:2]) + 1.0)

    weights = np.arange(1, 11)
    feat['ask_center_of_mass'] = float(np.sum(ask_matrix * weights) / (np.sum(ask_matrix) + 1.0))
    feat['bid_center_of_mass'] = float(np.sum(bid_matrix * weights) / (np.sum(bid_matrix) + 1.0))
    feat['ask_vacuum_openness'] = feat['ask_center_of_mass'] - 5.5
    feat['bid_vacuum_openness'] = feat['bid_center_of_mass'] - 5.5

    feat['cvd_180s'] = float(cvd_180s)
    feat['cvd_30s'] = float(cvd_30s)
    feat['cvd_10s'] = float(cvd_10s)
    feat['cvd_progression_ratio'] = float(cvd_10s / (abs(cvd_30s) + 1.0))
    feat['range_180s_ticks'] = float(range_180s)

    # Refill & Depletion speeds
    a10_4s = hist_list[-5]['a10'] if H_len >= 5 else a10
    b10_4s = hist_list[-5]['b10'] if H_len >= 5 else b10
    feat['ask_refill_speed'] = float(a10 - a10_4s)
    feat['bid_refill_speed'] = float(b10 - b10_4s)

    a1_3s = hist_list[-4]['a_sizes'][0] if H_len >= 4 and len(hist_list[-4]['a_sizes']) > 0 else a_sizes[0]
    b1_3s = hist_list[-4]['b_sizes'][0] if H_len >= 4 and len(hist_list[-4]['b_sizes']) > 0 else b_sizes[0]
    feat['ask_depletion_vel'] = -float(a_sizes[0] - a1_3s) / (a_sizes[0] + 10.0)
    feat['bid_depletion_vel'] = -float(b_sizes[0] - b1_3s) / (b_sizes[0] + 10.0)

    # Micro-lead & OFI Accel
    w_b = sum(bids[k]['price'] * bids[k]['size'] for k in range(min(10, len(bids))))
    w_a = sum(offers[k]['price'] * offers[k]['size'] for k in range(min(10, len(offers))))
    micro_p = (w_b + w_a) / (b10 + a10 + 1e-5)
    feat['micro_lead_ticks'] = float((micro_p - mid) / TICK)

    ofi_5s = cvd_10s  # Proxy
    feat['ofi_accel'] = float(cvd_10s - cvd_30s)

    # Fractal ER
    mid_30s_ago = hist_list[-31]['mid'] if H_len >= 31 else mid
    chg_30s = abs(mid - mid_30s_ago)
    path_30s = sum(abs(hist_list[j]['mid'] - hist_list[j-1]['mid']) for j in range(max(1, H_len-30), H_len))
    feat['fractal_er'] = float(chg_30s / (path_30s + 1e-5))

    mid_2s = hist_list[-3]['mid'] if H_len >= 3 else mid
    mid_4s = hist_list[-5]['mid'] if H_len >= 5 else mid
    feat['cubic_jerk'] = float((mid - 2*mid_2s + mid_4s) / TICK)

    mid_45s = hist_list[-46]['mid'] if H_len >= 46 else mid
    feat['runup_45s'] = float((mid - mid_45s) / TICK)

    feat['whirlpool_flips_6s'] = float(sum(1 for j in range(max(1, H_len-6), H_len) if np.sign(hist_list[j]['aggressor']) != np.sign(hist_list[j-1]['aggressor'])))

    # LightGBM Model Prediction
    feat_df = pd.DataFrame([feat])[DUAL_ENGINE_FEATURES]
    pred_c = float(model_call.predict(feat_df)[0])
    pred_p = float(model_put.predict(feat_df)[0])

    # 🏛️ 3-TIER VACUUM GAP ENGINE (Option 1: Original Magnitude Architecture)
    c_mag = abs(pred_c)
    p_mag = abs(pred_p)
    if c_mag >= p_mag:
        gap_points = int(round((c_mag - p_mag) * 100))   # Bullish Call (+) Vacuum
    else:
        gap_points = -int(round((p_mag - c_mag) * 100))  # Bearish Put (-) Vacuum
    abs_gap = abs(gap_points)

    if abs_gap >= 30:
        vacuum_tier = "🚀 BIG VACUUM"
        tier_target = 3.5
        tier_hold = 90
    elif abs_gap >= 20:
        vacuum_tier = "🎯 MEDIUM VACUUM"
        tier_target = 2.5
        tier_hold = 65
    elif abs_gap >= 10:
        vacuum_tier = "⚡ SMALL VACUUM (FILTERED)"
        tier_target = 2.0
        tier_hold = 45
    else:
        vacuum_tier = "⚖️ NEUTRAL CHOP"
        tier_target = 2.0
        tier_hold = 40

    # ─── ACTIVE TRADE ENGINE (Trailing Runner + Reverse Vacuum Inversion Lock) ───
    active = engine_state["active_trade"]
    trade_closed = False

    if active is not None:
        p_entry = active["entry_price"]
        dir_t = active["direction"]
        tgt_ticks = active["target_ticks"]
        sl_ticks = active["sl_ticks"]
        is_fast_t = active["is_fast"]

        diff = (mid - p_entry) / TICK if 'BUY' in dir_t else (p_entry - mid) / TICK
        active["current_move_ticks"] = round(diff, 2)
        active["current_pnl"] = round(diff * TICK_VALUE, 2)
        active["elapsed_seconds"] = int(now_ts - active["start_time"])

        if diff > active["peak_move_ticks"]:
            active["peak_move_ticks"] = round(diff, 2)

        peak_t = active["peak_move_ticks"]
        win = False
        exit_reason = ""

        # 🛡️ DYNAMIC RUNNER TRAILING LADDER (With Full 2.5t Breathing Room):
        # Only active for RUNNERS (Scalps take clean target at +2.0t / +2.5t before the wall)
        if not active.get("is_scalp"):
            # 1. Mega Runner (Peak >= 6.0t): Trail 2.0 ticks behind Peak to ride mega moves!
            if peak_t >= 6.0:
                new_floor = round(peak_t - 2.0, 2)
                active["trail_floor"] = max(active.get("trail_floor") or 0.0, new_floor)
            # 2. Medium Runner (Peak >= 4.0t): Trail 2.5 ticks behind Peak (locks +1.5t to +3.5t without choking!)
            elif peak_t >= 4.0:
                new_floor = round(peak_t - 2.5, 2)
                active["trail_floor"] = max(active.get("trail_floor") or 0.0, new_floor)
            # 3. Early Phase (Peak < 4.0t): Full breathing freedom! No premature choking at +0.5t!
            # The runner is protected by Orderbook Inversion Exit and Emergency Circuit Breaker!

        inv_target = active.get("reverse_thresh_val", -20 if 'BUY' in dir_t else 20)

        # 1. 🎯 CLEAN WALL SCALP EXIT (+2.0t / +2.5t Clean Profit):
        if active.get("is_scalp") and diff >= active.get("target_ticks", 2.0):
            win = True
            exit_reason = f"🎯 Wall Scalp Hit (+{active['target_ticks']:.1f}t Clean Profit)"
            trade_closed = True
        # 2. ⚡ RUNNER DEPTH RATIO INVERSION EXIT (Near Ratio inverts < 0.70x, locking peak profit):
        elif not active.get("is_scalp") and diff >= 1.0:
            if 'BUY' in dir_t and r_near_bull < 0.70:
                win = True
                exit_reason = f"⚡ Depth Inversion Exit ({r_near_bull:.2f}x Ratio, Peak: +{peak_t:.1f}t)"
                trade_closed = True
            elif 'SELL' in dir_t and r_near_bear < 0.70:
                win = True
                exit_reason = f"⚡ Depth Inversion Exit ({r_near_bear:.2f}x Ratio, Peak: +{peak_t:.1f}t)"
                trade_closed = True
        # 3. 🔄 PRIMARY EXIT: 5-POINT DYNAMIC INVERSION WITH ORDERBOOK CONFIRMATION:
        # Requires Orderbook to actually confirm that opposing pressure has taken over L1-L3!
        elif 'BUY' in dir_t and gap_points <= inv_target:
            book_confirms_reversal = (r_near_bull < 1.0)
            if book_confirms_reversal:
                active['inv_ticks_count'] = active.get('inv_ticks_count', 0) + 1
                if active['inv_ticks_count'] >= 2:
                    if not active.get("is_scalp") or diff <= 0:
                        win = (diff > 0)
                        exit_reason = f"🔄 Confirmed Inversion (Gap: {gap_points:+d}, Book: {r_near_bull:.2f}x)"
                        trade_closed = True
            else:
                active['inv_ticks_count'] = 0

        elif 'SELL' in dir_t and gap_points >= inv_target:
            book_confirms_reversal = (r_near_bear < 1.0)
            if book_confirms_reversal:
                active['inv_ticks_count'] = active.get('inv_ticks_count', 0) + 1
                if active['inv_ticks_count'] >= 2:
                    if not active.get("is_scalp") or diff <= 0:
                        win = (diff > 0)
                        exit_reason = f"🔄 Confirmed Inversion (Gap: {gap_points:+d}, Book: {r_near_bear:.2f}x)"
                        trade_closed = True
            else:
                active['inv_ticks_count'] = 0

        # 4. 🛡️ TRAILING RATCHET FLOOR (Protects accumulated profits for runners)
        elif active.get("trail_floor") is not None and diff <= active["trail_floor"]:
            win = (diff > 0)
            exit_reason = f"🛡️ Trailing Runner Floor (+{active['trail_floor']:.1f}t locked, Peak: +{peak_t:.1f}t)"
            trade_closed = True
        # 5. 🛑 EMERGENCY DISASTER CIRCUIT BREAKER (-4.0t extreme disaster shield only)
        elif diff <= -sl_ticks:
            win = False
            exit_reason = f"🛑 Emergency Circuit Breaker (-{sl_ticks:.1f}t)"
            trade_closed = True

        if trade_closed:
            final_ticks = round(diff, 2)
            pnl = round(final_ticks * TICK_VALUE, 2)
            tier_label = active.get("vacuum_tier", active.get("engine", "VACUUM"))
            entry_gap_val = active.get("entry_gap", 0)
            record = {
                "id": len(engine_state["session_trades"]) + 1,
                "enter_time": active["enter_time_str"],
                "exit_time": now_str,
                "duration": f"{active['elapsed_seconds']}s",
                "direction": dir_t,
                "candle_name": tier_label,
                "runway": f"Gap {entry_gap_val:+d}",
                "engine": tier_label,
                "target": f"Runner (Peak +{peak_t:.1f}t)",
                "final_ticks": f"{final_ticks:+.1f}t",
                "pnl": pnl,
                "status": "✅ WIN" if win else ("❌ LOSS" if final_ticks < 0 else "🟡 SCRATCH"),
                "reason": exit_reason
            }
            engine_state["session_trades"].append(record)
            if win: engine_state["total_wins"] += 1
            elif final_ticks < 0: engine_state["total_losses"] += 1

            engine_state["net_pnl"] = round(engine_state["net_pnl"] + pnl, 2)
            engine_state["net_ticks"] = round(engine_state["net_ticks"] + final_ticks, 2)
            tot_c = engine_state["total_wins"] + engine_state["total_losses"]
            engine_state["win_rate"] = round((engine_state["total_wins"] / tot_c * 100.0), 1) if tot_c > 0 else 0.0

            engine_state["last_trade_exit_time"] = now_ts
            if 'BUY' in dir_t: engine_state["last_call_exit_time"] = now_ts
            else: engine_state["last_put_exit_time"] = now_ts

            engine_state["active_trade"] = None
            save_persisted_state()

    # ─── 3-TIER VACUUM SIGNAL SCANNER (Only Medium ≥20 and Big ≥30 Vacuums) ───
    signal_dir = None
    target_ticks = tier_target
    sl_ticks = 4.0  # Emergency Circuit Breaker (-4.0t)
    engine_name = vacuum_tier
    max_hold_sec = tier_hold
    is_scalp_trade = False

    # Check Cooldown (Wait 15s after trade close to prevent instant re-entry)
    cd_required = 15
    in_cooldown = (now_ts - engine_state.get("last_trade_exit_time", 0.0) < cd_required)

    is_sideways_regime = (range_15m < 6.0)

    if not in_cooldown and engine_state["active_trade"] is None:
        # 🏛️ INSTITUTIONAL ORDERBOOK QUANT ENGINE:
        # 1. 🚀 BIG VACUUM (abs_gap >= 30):
        # Massive momentum that breaks resting walls (Liquidity Absorption)! Direct execution!
        # 2. 🎯 MEDIUM VACUUM (abs_gap >= 22):
        # Sensible corridor check (runway >= 2 or near ratio >= 1.05) to avoid sideways traps.

        if gap_points >= 22:
            # 🛡️ MANDATORY WALL SHIELD: Runway >= 2 (Level 1 par deewar nahi honi chahiye!)
            if ratio_runway_call >= 2 and r_near_bull >= 1.05:
                is_big_vacuum = (gap_points >= 30)
                is_runner = (r_near_bull >= 1.30 and r_deep_bull >= 1.05 and ratio_runway_call >= 3)
                is_scalp_trade = not is_runner
                target_ticks = 2.5 if is_big_vacuum else 2.0
                sl_ticks = 4.0
                max_hold_sec = tier_hold
                engine_name = f"{vacuum_tier} (RUNNER)" if is_runner else f"{vacuum_tier} (SCALP {target_ticks:.1f}t)"
                signal_dir = "🟢 BUY (CALL)"

        elif gap_points <= -22:
            # 🛡️ MANDATORY WALL SHIELD: Runway >= 2 (Level 1 par deewar nahi honi chahiye!)
            if ratio_runway_put >= 2 and r_near_bear >= 1.05:
                is_big_vacuum = (abs_gap >= 30)
                is_runner = (r_near_bear >= 1.30 and r_deep_bear >= 1.05 and ratio_runway_put >= 3)
                is_scalp_trade = not is_runner
                target_ticks = 2.5 if is_big_vacuum else 2.0
                sl_ticks = 4.0
                max_hold_sec = tier_hold
                engine_name = f"{vacuum_tier} (RUNNER)" if is_runner else f"{vacuum_tier} (SCALP {target_ticks:.1f}t)"
                signal_dir = "🔴 SELL (PUT)"

    # Trigger Trade
    if signal_dir and engine_state["active_trade"] is None:
        p_entry = bids[0]['price'] if 'BUY' in signal_dir else offers[0]['price']
        # 5-POINT DYNAMIC INVERSION RULE (User Quant Logic):
        # Ulta vacuum threshold = Entry gap se 5 points kam (minimum 20 gap floor)
        if 'BUY' in signal_dir:
            inv_thresh_val = -max(20, gap_points - 5)
            inv_thresh = f"{inv_thresh_val:+d} Gap"
        else:
            inv_thresh_val = max(20, abs(gap_points) - 5)
            inv_thresh = f"+{inv_thresh_val} Gap"

        engine_state["active_trade"] = {
            "direction": signal_dir,
            "engine": engine_name,
            "vacuum_tier": vacuum_tier,
            "is_scalp": is_scalp_trade,
            "entry_gap": gap_points,
            "reverse_inversion_thresh": inv_thresh,
            "reverse_thresh_val": inv_thresh_val,
            "is_fast": (engine_name == "FAST TREND"),
            "entry_price": p_entry,
            "target_ticks": target_ticks,
            "sl_ticks": sl_ticks,
            "max_hold": max_hold_sec,
            "reached_1_5t": False,
            "trail_floor": None,
            "peak_move_ticks": 0.0,
            "current_move_ticks": 0.0,
            "current_pnl": 0.0,
            "start_time": now_ts,
            "enter_time_str": now_str,
            "elapsed_seconds": 0
        }
        mode_label = f"🎯 SCALP (+{target_ticks:.1f}t Target)" if is_scalp_trade else "🏃 RUNNER MODE"
        print(f"🔥 [SIGNAL EXECUTED] {signal_dir} via {engine_name} (Gap: {gap_points:+d}) @ {p_entry:.5f} | Mode: {mode_label} | Exit Thresh: {inv_thresh} | (SL: -{sl_ticks}t)")

    # Update state for UI
    delta_status = "⚡ BALANCED ORDER FLOW"
    if cvd_15s >= 8: delta_status = "🟢 HEAVY BUY INFLOW"
    elif cvd_15s <= -8: delta_status = "🔴 HEAVY SELL OUTFLOW"
    elif feat['dist_super_res_ask'] <= 2.0: delta_status = "🪤 TOP TRAP RESISTANCE"
    elif feat['dist_super_sup_bid'] <= 2.0: delta_status = "🪤 BOTTOM TRAP SUPPORT"

    active_now = engine_state["active_trade"]
    wall_runway = ratio_runway_call if gap_points >= 0 else ratio_runway_put
    is_upcoming_runner = (r_near_bull >= 1.35 and r_deep_bull >= 1.10 and wall_runway >= 4) if gap_points >= 0 else (r_near_bear >= 1.35 and r_deep_bear >= 1.10 and wall_runway >= 4)
    upcoming_mode = "🏃 RUNNER MODE" if is_upcoming_runner else f"🎯 SCALP (+{(2.5 if abs_gap >= 30 else 2.0):.1f}t)"

    if active_now is not None:
        display_dir = active_now["direction"]
        mode_text = f"🎯 SCALP (+{active_now.get('target_ticks', 2.0):.1f}t)" if active_now.get("is_scalp") else "🏃 RUNNER"
        floor_disp = f" | Floor: +{active_now['trail_floor']:.1f}t" if active_now.get('trail_floor') else " | Dynamic Inversion"
        display_grade = f"⚡ ENTRY: {active_now.get('entry_gap', 0):+d} GAP | {mode_text}{floor_disp}"
        display_candle = f"🔥 Active: {display_dir} @ {active_now['entry_price']:.5f} (Entry Gap: {active_now.get('entry_gap', 0):+d} | Live Gap: {gap_points:+d})"
    elif signal_dir:
        display_dir = signal_dir
        display_grade = f"⚡ LIVE GAP: {gap_points:+d} | {vacuum_tier} | {upcoming_mode}"
        display_candle = f"🎯 {signal_dir} via {vacuum_tier} ({upcoming_mode} | Runway: {wall_runway}t)"
    else:
        display_dir = "⚡ SCANNING FOR VACUUM"
        display_grade = f"⚡ LIVE GAP: {gap_points:+d} | {vacuum_tier} | Next: {upcoming_mode}"
        display_candle = f"Live Gap: {gap_points:+d} (Need >= ±25) | Mode: {upcoming_mode} (Runway: {wall_runway}t | L1-L3: {(r_near_bull if gap_points >= 0 else r_near_bear):.2f}x)"

    engine_state["latest_signal"] = {
        "active": (signal_dir is not None or active_now is not None),
        "direction": display_dir,
        "engine": vacuum_tier,
        "grade": display_grade,
        "predicted_candle": display_candle,
        "gap_points": gap_points,
        "upcoming_mode": upcoming_mode,
        "vacuum_tier": vacuum_tier,
        "reverse_thresh": "±20 Gap Inversion",
        "delta_status": delta_status,
        "new_alert": (signal_dir is not None and active_now is None),
        "target_ticks": target_ticks,
        "expected_hold": f"{max_hold_sec}s",
        "prob_call": round(pred_c, 3),
        "prob_put": round(pred_p, 3),
        "vol_ratio_5": round(feat['vol_ratio_l5'], 2),
        "vol_ratio_6_10": round(feat['vol_ratio_l10'], 2),
        "near_ratio": round(r_near_bull if gap_points >= 0 else r_near_bear, 2),
        "deep_ratio": round(r_deep_bull if gap_points >= 0 else r_deep_bear, 2),
        "deep_vacuum_score": round(feat['consensus_depth_spread'], 1),
        "ask_trap_wall": round(feat['has_wall_at_l1_l2_ask'], 1),
        "bid_trap_wall": round(feat['has_wall_at_l1_l2_bid'], 1),
        "ask_runway": ratio_runway_call,
        "bid_runway": ratio_runway_put,
        "runway": wall_runway,
        "cvd_15s": cvd_15s,
        "regime": current_regime_name
    }

    all_sizes = [b['size'] for b in bids] + [o['size'] for o in offers]
    max_size = max(all_sizes) if all_sizes else 200

    # Broadcast packet
    payload = {
        "server_time": datetime.now().strftime("%H:%M:%S"),
        "status": live_market["status"],
        "timestamp": live_market["timestamp"],
        "mid_price": round(mid, 5),
        "last_trade_price": round(last_trade, 5),
        "spread": round(live_market["spread"] / TICK, 1),
        "bids": bids,
        "offers": offers,
        "regime": current_regime_name,
        "range_15m": round(range_15m, 1),
        "range_180s": round(range_180s, 1),
        "cvd_180s": cvd_180s,
        "market": {
            "symbol": SYMBOL,
            "contractId": CONTRACT_ID,
            "status": live_market["status"],
            "mid_price": f"{mid:.5f}",
            "last_trade": f"{last_trade:.5f}",
            "spread": f"{live_market['spread'] / TICK:.1f}t ({live_market['spread']:.5f})",
            "spread_val": live_market["spread"],
            "bids": bids,
            "offers": offers,
            "max_size": max_size
        },
        "signal": engine_state["latest_signal"],
        "active_trade": engine_state["active_trade"],
        "performance": {
            "total_trades": len(engine_state["session_trades"]),
            "wins": engine_state["total_wins"],
            "losses": engine_state["total_losses"],
            "win_rate": f"{engine_state['win_rate']:.1f}%",
            "net_pnl": f"${engine_state['net_pnl']:+.2f}",
            "net_ticks": f"{engine_state['net_ticks']:+.1f}t"
        },
        "trades": engine_state["session_trades"][-25:],
        "mt5_bridge": {
            "connected": True,
            "account": "6EZ6_LIVE",
            "last_order": None
        },
        "token_info": get_token_info(TOKEN)
    }
    return payload

# ==============================================================================
# 🌐 FASTAPI APPLICATION & ENDPOINTS
# ==============================================================================
app = FastAPI(title="6EZ6 Master Dual-Engine AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def get_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/status")
def get_status():
    with state_lock:
        bids = list(live_market["bids"])
        offers = list(live_market["offers"])
        all_sizes = [b.get('size', 0) for b in bids] + [o.get('size', 0) for o in offers]
        max_size = max(all_sizes) if all_sizes else 100
        return {
            "status": live_market["status"],
            "symbol": SYMBOL,
            "mid_price": live_market["mid_price"],
            "spread": round(live_market["spread"] / TICK, 1) if live_market.get("spread") else 1.0,
            "bids": bids,
            "offers": offers,
            "max_size": max_size,
            "regime": engine_state["current_regime"],
            "wins": engine_state["total_wins"],
            "losses": engine_state["total_losses"],
            "net_pnl": engine_state["net_pnl"],
            "active_trade": engine_state["active_trade"],
            "signal": engine_state.get("latest_signal", {})
        }

@app.post("/api/update_token")
async def update_token_endpoint(request: Request):
    global TOKEN, current_ws_app
    try:
        body = await request.json()
        new_token = body.get("token")
        if new_token and len(new_token) > 20:
            if new_token != TOKEN:
                TOKEN = new_token
                print(f"🔑 [TOKEN UPDATED] Fresh token injected: {new_token[:30]}...")
                if current_ws_app:
                    current_ws_app.close()
            return {"status": "success", "msg": "Token updated successfully"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}
    return {"status": "ignored"}

@app.get("/api/mt5_signal")
def get_mt5_signal():
    """Ultra-fast JSON endpoint for MetaTrader 5 Expert Advisor Bridge"""
    with state_lock:
        act = engine_state.get("active_trade")
        if act:
            dir_str = "BUY" if "BUY" in act["direction"] else "SELL"
            return {
                "active": True,
                "action": dir_str,
                "symbol": "EURUSD",
                "volume": 1.0,
                "engine": act.get("engine", "MASTER"),
                "trade_id": int(act.get("start_time", 0)),
                "target_ticks": act.get("target_ticks", 2.0),
                "sl_ticks": act.get("sl_ticks", 1.5),
                "entry_price": act.get("entry_price", 0.0),
            }
        else:
            last_trade = engine_state["session_trades"][-1] if engine_state["session_trades"] else None
            return {
                "active": False,
                "action": "CLOSE",
                "symbol": "EURUSD",
                "last_exit_reason": last_trade.get("reason", "") if last_trade else "",
                "total_trades": len(engine_state["session_trades"])
            }

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_websockets.add(ws)
    print("💻 Browser Dashboard Client Connected to Live Stream!")
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(ws)
        print("💻 Browser Dashboard Client Disconnected.")
    except Exception:
        if ws in connected_websockets:
            connected_websockets.remove(ws)

async def broadcast_loop():
    while True:
        try:
            payload = evaluate_market_engine()
            if payload and connected_websockets:
                data_str = json.dumps(payload)
                dead = set()
                for ws in list(connected_websockets):
                    try:
                        await ws.send_text(data_str)
                    except Exception:
                        dead.add(ws)
                for d in dead:
                    connected_websockets.discard(d)
        except Exception:
            pass
        await asyncio.sleep(0.15)

background_tasks = set()

import socket

def start_port_forwarder(source_port=8088, target_port=8000):
    def proxy():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', source_port))
            s.listen(15)
            while True:
                client, _ = s.accept()
                def handle(c):
                    try:
                        remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        remote.connect(('127.0.0.1', target_port))
                        def p2c():
                            try:
                                while True:
                                    d = remote.recv(8192)
                                    if not d: break
                                    c.sendall(d)
                            except: pass
                            finally:
                                try: c.close()
                                except: pass
                        def c2p():
                            try:
                                while True:
                                    d = c.recv(8192)
                                    if not d: break
                                    remote.sendall(d)
                            except: pass
                            finally:
                                try: remote.close()
                                except: pass
                        threading.Thread(target=p2c, daemon=True).start()
                        threading.Thread(target=c2p, daemon=True).start()
                    except:
                        c.close()
                threading.Thread(target=handle, args=(client,), daemon=True).start()
        except Exception:
            pass
    threading.Thread(target=proxy, daemon=True).start()

@app.on_event("startup")
async def startup_event():
    start_port_forwarder(source_port=8088, target_port=8000)
    threading.Thread(target=proactive_token_refresher, daemon=True).start()
    t = threading.Thread(target=tradovate_worker, daemon=True)
    t.start()
    b_task = asyncio.create_task(broadcast_loop())
    background_tasks.add(b_task)
    b_task.add_done_callback(background_tasks.discard)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
