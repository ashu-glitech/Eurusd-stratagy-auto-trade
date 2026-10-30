import os
import sys
import time
import json
import pickle
import asyncio
import threading
import collections
from datetime import datetime
import numpy as np
import pandas as pd
import websocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ==============================================================================
# 🎯 6EZ6 (CME Euro FX Futures) LIVE VACUUM SNIPER SERVER
# ==============================================================================

TICK = 0.00005
TICK_VALUE = 6.25  # $6.25 per tick per 1 standard contract
CONTRACT_ID = 3267437  # 6EZ6
SYMBOL = "6EZ6"

TOKEN = "eyJraWQiOiI1NyIsImFsZyI6IkVkRFNBIn0.eyJzdWIiOiI5NjY0ODA5IiwiZXhwIjoxNzkyNzQzNjA1LCJqdGkiOiItMTY5NDQ3OTUxNDQ2MTMxMDQxOC0xNDkwMzM2NTUzNDIyOTEzNDQxIiwicGhzIjotMTY0NTI4OTYwMSwiZW1haWwiOiJnYXZhbGkzOGFnZzM4QGdtYWlsLmNvbSJ9.FZhLY10P5xYkg9AlOS1WIgilybK4LCUTvL0F5Z3a-WP02_E_0hvrNh2B5fsG1vGAliGrUNnrK1SYadunft6PAw"
WS_URL = "wss://md.tradovateapi.com/v1/websocket"

MODEL_DIR = '/Users/admin/Downloads/nifty_level2/eurousd_models_10level'

# Load Models
print("📦 Loading AI Models & Feature Registry...")
with open(f'{MODEL_DIR}/eurousd_features.pkl', 'rb') as f:
    FEATURES = pickle.load(f)
with open(f'{MODEL_DIR}/eurousd_barrier_call_model.pkl', 'rb') as f:
    call_model = pickle.load(f)
with open(f'{MODEL_DIR}/eurousd_barrier_put_model.pkl', 'rb') as f:
    put_model = pickle.load(f)
print(f"✅ Loaded Call & Put Models with {len(FEATURES)} features.")

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

# 1-second sampled rolling history
history_1s = collections.deque(maxlen=300) # [ {time, mid, trade, b5, a5, b10, a10, aggressor} ]
trade_events_recent = collections.deque(maxlen=200) # raw trades

# Trade Engine State
engine_state = {
    "active_trade": None,       # Dict when trade is active
    "last_trade_exit_time": 0.0,
    "last_call_exit_time": 0.0,
    "last_put_exit_time": 0.0,
    "cooldown_seconds": 35,
    "session_trades": [],       # Completed trades
    "total_wins": 0,
    "total_losses": 0,
    "win_rate": 0.0,
    "net_pnl": 0.0,
    "net_ticks": 0.0,
    "latest_signal": {
        "active": False,
        "direction": "SCANNING",
        "grade": "Monitoring L2 Orderbook...",
        "runway": 0,
        "target_ticks": 2.0,
        "expected_hold": "30s - 60s",
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
        "cvd_15s": 0.0
    }
}

connected_websockets = set()

# ==============================================================================
# 📡 TRADOVATE WEBSOCKET CLIENT
# ==============================================================================
def tradovate_worker():
    global live_market, current_ws_app
    headers = [
        "Origin: https://trader.tradovate.com",
        "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ]

    has_sub = False

    def on_open(ws):
        print("🚀 Connecting to Tradovate Market Data WebSocket...")

    def on_message(ws, message):
        nonlocal has_sub
        if message == 'h':
            ws.send('[]')
            return
        if message == 'o':
            print("🟢 Tradovate handshake 'o' received. Sending auth...")
            ws.send(f"authorize\n2\n\n{TOKEN}")
            return
        if not has_sub and message.startswith('a[{"s":200'):
            print("✅ Tradovate Authorized! Subscribing to 6EZ6 (3267437)...")
            ws.send(f'md/subscribeDom\n3\n\n{{"symbol":{CONTRACT_ID}}}')
            ws.send(f'md/subscribeQuote\n4\n\n{{"symbol":{CONTRACT_ID}}}')
            has_sub = True
            with state_lock:
                live_market["status"] = "LIVE STREAMING"
            return
        if message.startswith('a['):
            try:
                payload = json.loads(message[1:])
                for item in payload:
                    if item.get('e') == 'md':
                        process_market_data(item.get('d', {}))
            except Exception:
                pass

    def on_error(ws, error):
        print(f"⚠️ Tradovate WS error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("🔴 Tradovate WS connection closed, reconnecting in 3s...")
        with state_lock:
            live_market["status"] = "RECONNECTING"

    while True:
        has_sub = False
        try:
            ws_app = websocket.WebSocketApp(
                WS_URL,
                header=headers,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            current_ws_app = ws_app
            ws_app.run_forever()
        except Exception as e:
            print(f"WS loop exception: {e}")
        time.sleep(10)

def process_market_data(md):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    now_ts = time.time()
    
    with state_lock:
        # Handle Quotes
        if 'quotes' in md:
            for q in md['quotes']:
                entries = q.get('entries', {})
                if 'Trade' in entries:
                    tp = entries['Trade']['price']
                    live_market["last_trade_price"] = tp
                    trade_events_recent.append({
                        "time": now_ts,
                        "price": tp,
                        "size": entries['Trade'].get('size', 1)
                    })
        
        # Handle 10-level DOM
        if 'doms' in md:
            for d in md['doms']:
                bids = d.get('bids', [])
                offers = d.get('offers', [])
                if bids and offers:
                    live_market["bids"] = bids[:10]
                    live_market["offers"] = offers[:10]
                    p_bid = bids[0]['price']
                    p_ask = offers[0]['price']
                    live_market["mid_price"] = (p_bid + p_ask) / 2.0
                    live_market["spread"] = p_ask - p_bid
                    live_market["timestamp"] = now_str
                    if live_market["last_trade_price"] == 0.0:
                        live_market["last_trade_price"] = live_market["mid_price"]

# ==============================================================================
# 🧠 1Hz FEATURE GENERATION & VACUUM SNIPER EVALUATION
# ==============================================================================
def evaluate_market_engine():
    global engine_state
    now_ts = time.time()
    
    with state_lock:
        bids = live_market["bids"]
        offers = live_market["offers"]
        mid = live_market["mid_price"]
        last_trade = live_market["last_trade_price"]
        ts_str = live_market["timestamp"]

    if len(bids) < 5 or len(offers) < 5 or mid == 0.0:
        return None

    # Calculate Aggressor Trade Flow (Lee-Ready)
    aggressor = 0.0
    if last_trade > mid:
        aggressor = 1.0
    elif last_trade < mid:
        aggressor = -1.0

    b5 = sum(bids[k]['size'] for k in range(min(5, len(bids))))
    a5 = sum(offers[k]['size'] for k in range(min(5, len(offers))))
    b10 = sum(bids[k]['size'] for k in range(min(10, len(bids))))
    a10 = sum(offers[k]['size'] for k in range(min(10, len(offers))))

    snap = {
        "time": now_ts,
        "mid": mid,
        "aggressor": aggressor,
        "b5": b5,
        "a5": a5,
        "b10": b10,
        "a10": a10
    }
    history_1s.append(snap)

    # Need at least 15s of history for full features, but fill sensibly
    hist_list = list(history_1s)
    
    # CVD 15s & 60s
    cvd_15s = sum(h['aggressor'] for h in hist_list if now_ts - h['time'] <= 15)
    cvd_60s = sum(h['aggressor'] for h in hist_list if now_ts - h['time'] <= 60)

    # Footprint Delta Divergence & Range Position (60s Window)
    recent_mids = [h['mid'] for h in hist_list if now_ts - h['time'] <= 60]
    if not recent_mids:
        recent_mids = [mid]
    min_mid_60s = min(recent_mids)
    max_mid_60s = max(recent_mids)
    rng_60s = max_mid_60s - min_mid_60s
    price_pos = (mid - min_mid_60s) / (rng_60s + 1e-6) if rng_60s > 0 else 0.5

    # Bearish Delta Divergence (Top Trap): Price near highs (>=0.70) but aggressive selling (cvd_15s <= -8)
    bearish_delta_div = (price_pos >= 0.70 and cvd_15s <= -8)
    # Bullish Delta Divergence (Bottom Trap): Price near lows (<=0.30) but aggressive buying (cvd_15s >= +8)
    bullish_delta_div = (price_pos <= 0.30 and cvd_15s >= +8)

    if bearish_delta_div:
        delta_status = "⚠️ TOP TRAP (Smart Sellers Absorption)"
    elif bullish_delta_div:
        delta_status = "⚠️ BOTTOM TRAP (Smart Buyers Accumulation)"
    elif cvd_15s >= 8:
        delta_status = "🟢 AGGRESSIVE BUY INFLOW (+CVD)"
    elif cvd_15s <= -8:
        delta_status = "🔴 AGGRESSIVE SELL OUTFLOW (-CVD)"
    else:
        delta_status = "⚖️ BALANCED ORDER FLOW"

    # 10-level imbalances
    imbs = {}
    for i in range(1, 11):
        bq = bids[i-1]['size'] if i-1 < len(bids) else 0
        aq = offers[i-1]['size'] if i-1 < len(offers) else 0
        imbs[f'imb_l{i}'] = (bq - aq) / (bq + aq + 1e-6)

    vr_5 = b5 / (a5 + 1.0)
    vr_10 = b10 / (a10 + 1.0)
    vr_6_10 = (b10 - b5) / (a10 - a5 + 1.0)

    # Trap walls
    ask_q1 = offers[0]['size'] if len(offers) > 0 else 1
    ask_q2 = offers[1]['size'] if len(offers) > 1 else 1
    bid_q1 = bids[0]['size'] if len(bids) > 0 else 1
    bid_q2 = bids[1]['size'] if len(bids) > 1 else 1
    ask_trap_wall = ask_q2 / (ask_q1 + 1.0)
    bid_trap_wall = bid_q2 / (bid_q1 + 1.0)

    # Dynamic Brick Wall threshold based on actual Euro market liquidity
    # Average depth is ~150-180 lots. True institutional walls are 240+ lots.
    avg_ask_size = sum(offers[lvl]['size'] for lvl in range(min(10, len(offers)))) / max(1, min(10, len(offers)))
    avg_bid_size = sum(bids[lvl]['size'] for lvl in range(min(10, len(bids)))) / max(1, min(10, len(bids)))
    brick_wall_threshold = max(240, int(max(avg_ask_size, avg_bid_size) * 1.6))

    # Runway Depths (nearest wall >= brick_wall_threshold)
    ask_runway = 10
    for lvl in range(min(10, len(offers))):
        if offers[lvl]['size'] >= brick_wall_threshold:
            ask_runway = lvl + 1
            break

    bid_runway = 10
    for lvl in range(min(10, len(bids))):
        if bids[lvl]['size'] >= brick_wall_threshold:
            bid_runway = lvl + 1
            break

    # Price gaps
    ask_p1 = offers[0]['price']
    ask_p5 = offers[min(4, len(offers)-1)]['price']
    ask_p10 = offers[min(9, len(offers)-1)]['price']
    bid_p1 = bids[0]['price']
    bid_p5 = bids[min(4, len(bids)-1)]['price']
    bid_p10 = bids[min(9, len(bids)-1)]['price']

    ask_gap_l5 = ask_p5 - ask_p1
    bid_gap_l5 = bid_p1 - bid_p5
    gap_asymmetry_5 = ask_gap_l5 - bid_gap_l5

    ask_gap_l10 = ask_p10 - ask_p1
    bid_gap_l10 = bid_p1 - bid_p10
    gap_asymmetry_10 = ask_gap_l10 - bid_gap_l10

    # Refill speed (diff 4s)
    a5_4s_ago = hist_list[-5]['a5'] if len(hist_list) >= 5 else a5
    b5_4s_ago = hist_list[-5]['b5'] if len(hist_list) >= 5 else b5
    ask_refill_speed = a5 - a5_4s_ago
    bid_refill_speed = b5 - b5_4s_ago

    # Micro price
    w_b = sum(bids[k]['price'] * bids[k]['size'] for k in range(min(10, len(bids))))
    w_a = sum(offers[k]['price'] * offers[k]['size'] for k in range(min(10, len(offers))))
    micro_price = (w_b + w_a) / (b10 + a10 + 1.0)
    micro_lead = micro_price - mid

    # Momentum
    mid_15s_ago = hist_list[-16]['mid'] if len(hist_list) >= 16 else mid
    mid_60s_ago = hist_list[-61]['mid'] if len(hist_list) >= 61 else mid
    mom_15s = mid - mid_15s_ago
    mom_60s = mid - mid_60s_ago

    deep_vacuum_score = (np.sign(imbs['imb_l1']) + np.sign(imbs['imb_l5']) + np.sign(imbs['imb_l10'])) / 3.0
    spread = ask_p1 - bid_p1

    # Feature vector matching exact trained XGBoost model
    feat_dict = {
        'imb_l1': imbs['imb_l1'],
        'imb_l5': imbs['imb_l5'],
        'imb_l10': imbs['imb_l10'],
        'vol_ratio_5': vr_5,
        'vol_ratio_10': vr_10,
        'vol_ratio_6_10': vr_6_10,
        'ask_gap_l5': ask_gap_l5,
        'bid_gap_l5': bid_gap_l5,
        'ask_gap_l10': ask_gap_l10,
        'bid_gap_l10': bid_gap_l10,
        'gap_asymmetry_5': gap_asymmetry_5,
        'gap_asymmetry_10': gap_asymmetry_10,
        'ask_refill_speed': ask_refill_speed,
        'bid_refill_speed': bid_refill_speed,
        'micro_lead': micro_lead,
        'mom_15s': mom_15s,
        'mom_60s': mom_60s,
        'deep_vacuum_score': deep_vacuum_score,
        'spread': spread
    }

    # Model Inference
    X = np.array([[feat_dict[k] for k in FEATURES]])
    pred_c = float(call_model.predict(X)[0])
    pred_p = float(put_model.predict(X)[0])

    # ─── ACTIVE TRADE MONITORING ───
    active = engine_state["active_trade"]
    trade_closed = False
    new_alert_fired = False

    if active is not None:
        p_entry = active["entry_price"]
        dir_t = active["direction"]
        tgt_ticks = active["target_ticks"]
        target_pts = tgt_ticks * TICK
        sl_pts = 2.0 * TICK if tgt_ticks == 1.0 else 3.0 * TICK
        
        move = (mid - p_entry) if 'BUY' in dir_t else (p_entry - mid)
        active["current_move_ticks"] = round(move / TICK, 2)
        active["current_pnl"] = round(active["current_move_ticks"] * TICK_VALUE, 2)
        active["elapsed_seconds"] = int(now_ts - active["start_time"])

        if move > active["peak_move_pts"]:
            active["peak_move_pts"] = move
            active["peak_move_ticks"] = round(move / TICK, 2)

        peak_m = active["peak_move_pts"]
        peak_ticks = active["peak_move_ticks"]
        win = False
        exit_reason = ""
        exit_p = mid

        # 1. Target Hit or Near-Target Harvest (Real +2.0t to +4.0t Moves)
        if move >= target_pts:
            win = True; exit_reason = f"Target Hit (+{tgt_ticks:.1f}t)"
            trade_closed = True
        # 2. Near-Target Harvest: If within 0.5t of big target after 20s
        elif tgt_ticks >= 2.5 and (move >= target_pts - 0.5 * TICK) and (active["elapsed_seconds"] >= 20):
            win = True; exit_reason = f"Near-Target Harvest (+{move/TICK:.1f}t)"
            trade_closed = True
        # 3. Dynamic Trailing Profit Lock: Lock in real profits when market runs
        elif peak_ticks >= 3.5 and move <= (peak_m - 0.5 * TICK):
            win = True; exit_reason = f"Profit Lock Runner (+{move/TICK:.1f}t)"
            trade_closed = True
        elif peak_ticks >= 2.5 and move <= (peak_m - 0.5 * TICK):
            win = True; exit_reason = f"Profit Lock Scalp (+{move/TICK:.1f}t)"
            trade_closed = True
        # 4. Breakeven Floor: If trade touched +1.5t ($9.38), never let it turn red! Scratch at 0.0t if it reverses
        elif peak_ticks >= 1.5 and move <= 0.0:
            win = True; exit_reason = f"Breakeven Floor (0.0t)"
            trade_closed = True
        # 5. Immediate Structural Flip Scratch: Only after 15s and at 0.0t
        elif active["elapsed_seconds"] >= 15 and move <= 0.0 and (('BUY' in dir_t and deep_vacuum_score <= -0.33) or ('SELL' in dir_t and deep_vacuum_score >= 0.33)):
            win = True; exit_reason = "Vacuum Scratch (0.0t)"
            trade_closed = True
        # 6. Hard Stop Loss
        elif move <= -sl_pts:
            win = False; exit_reason = f"Stop Loss (-{sl_pts/TICK:.0f}t)"
            trade_closed = True
        # 7. Stagnation Scratch: 60s for 1.5t scalp, 120s for runner
        elif active["elapsed_seconds"] >= (60 if tgt_ticks <= 1.5 else 120) and move <= 0.0:
            win = (move >= 0); exit_reason = f"Stagnation Scratch ({move/TICK:+.1f}t)"
            trade_closed = True
        # 8. Lifespan End (180s)
        elif active["elapsed_seconds"] >= 180:
            win = (move >= 0); exit_reason = f"Lifespan End ({move/TICK:+.1f}t)"
            trade_closed = True

        if trade_closed:
            net_ticks = round((exit_p - p_entry) / TICK if 'BUY' in dir_t else (p_entry - exit_p) / TICK, 2)
            pnl = round(net_ticks * TICK_VALUE, 2)

            # Candle Classification
            trig = active.get("trigger_type", "")
            if trig == "TOP_TRAP":
                candle_name = "🪤 TOP TRAP REVERSAL"
            elif trig == "BOTTOM_TRAP":
                candle_name = "🪤 BOTTOM TRAP REVERSAL"
            elif trig == "ROCKET_VACUUM":
                candle_name = "🚀 ROCKET VACUUM SCALP"
            elif 'Target Hit' in exit_reason or net_ticks >= 2.0:
                candle_name = "🟥 BEARISH MARUBOZU" if 'SELL' in dir_t else "🟩 BULLISH MARUBOZU"
            elif 'Vacuum Scratch' in exit_reason and net_ticks == 0.0:
                candle_name = "➕ DOJI / EQUILIBRIUM"
            elif 'Stagnation Scratch' in exit_reason and net_ticks == 0.0:
                candle_name = "⏳ FLAT CHOP"
            elif net_ticks < 0:
                candle_name = "🔨 REJECTION PINBAR"
            else:
                candle_name = "⚡ QUICK RUNWAY SCALP"

            record = {
                "id": len(engine_state["session_trades"]) + 1,
                "enter_time": active["enter_time_str"],
                "exit_time": datetime.now().strftime("%H:%M:%S"),
                "duration": f"{active['elapsed_seconds']}s",
                "direction": dir_t,
                "candle_name": candle_name,
                "runway": f"{active['runway']}t Runway",
                "target": f"+{tgt_ticks:.0f}t",
                "final_ticks": f"{net_ticks:+.1f}t",
                "pnl": pnl,
                "status": "✅ WIN" if win else "❌ LOSS",
                "reason": exit_reason
            }
            engine_state["session_trades"].append(record)
            if win:
                engine_state["total_wins"] += 1
            else:
                engine_state["total_losses"] += 1

            engine_state["net_pnl"] = round(engine_state["net_pnl"] + pnl, 2)
            engine_state["net_ticks"] = round(engine_state["net_ticks"] + net_ticks, 2)
            tot = engine_state["total_wins"] + engine_state["total_losses"]
            engine_state["win_rate"] = round((engine_state["total_wins"] / tot * 100.0), 1) if tot > 0 else 0.0

            engine_state["last_trade_exit_time"] = now_ts
            if 'BUY' in dir_t:
                engine_state["last_call_exit_time"] = now_ts
            else:
                engine_state["last_put_exit_time"] = now_ts

            engine_state["active_trade"] = None
            active = None

    # ─── MICRO-VACUUM SCALP SCANNER (L1-L2 SENSITIVITY + ADAPTIVE TARGET + DELTA GATING) ───
    signal_dir = None
    target_ticks = 2.0
    runway_depth = 0
    grade = ""
    exp_hold = "25s - 50s"
    pred_candle = "DOJI / EQUILIBRIUM"
    trigger_type = "STANDARD_VACUUM"

    # Global Cooldown Check (Wait at least 35s after any trade to avoid whipsaw)
    in_cooldown = (now_ts - engine_state.get("last_trade_exit_time", 0.0) < engine_state["cooldown_seconds"])

    # Top-of-Book (L1-L2) Micro-Vacuum Ratios
    l1_ratio = (bid_q1 + 1.0) / (ask_q1 + 1.0)
    b2 = (bids[0]['size'] if len(bids) > 0 else 0) + (bids[1]['size'] if len(bids) > 1 else 0)
    a2 = (offers[0]['size'] if len(offers) > 0 else 0) + (offers[1]['size'] if len(offers) > 1 else 0)
    l2_ratio = (b2 + 1.0) / (a2 + 1.0)

    # ─── 1. 🟢 BULLISH VACUUM RADAR (PRIORITY 1: RIDE THE GREEN TREND!) ───
    # If buyers dominate and ask runway is open, GO WITH THE TREND (BUY CALL)!
    if not in_cooldown and (vr_5 >= 1.12 or l1_ratio >= 1.4 or l2_ratio >= 1.35) and ask_trap_wall <= 8.0 and cvd_15s >= -10:
        if pred_c >= -0.18:
            signal_dir = "🟢 BUY (CALL)"
            runway_depth = ask_runway
            # Tier 3: BIG VACUUM + Delta Inflow Rocket
            if (runway_depth >= 5 or vr_10 >= 1.40) and cvd_15s >= 5:
                target_ticks = 4.0; exp_hold = "60s - 120s"
                pred_candle = "🚀 ROCKET VACUUM (Delta Inflow)"
                grade = "🏆 Level 10 Corridor + CVD Inflow (+4.0t)"
                trigger_type = "ROCKET_VACUUM"
            elif runway_depth >= 5 or vr_10 >= 1.40:
                target_ticks = 3.5; exp_hold = "50s - 90s"
                pred_candle = "🚀 BIG VACUUM (Level 10 Corridor)"
                grade = "🏆 Level 10 Big Corridor (+3.5t Runner)"
            # Tier 2: MEDIUM VACUUM
            elif runway_depth >= 3 or vr_5 >= 1.25:
                target_ticks = 2.5; exp_hold = "30s - 60s"
                pred_candle = "🎯 MEDIUM VACUUM (Level 5 Void)"
                grade = "🎯 Level 5 Medium Void (+2.5t Sniper)"
            # Tier 1: CLEAN VACUUM (Min 2.0t = 1.0 pip = $10.00 to guarantee clean profit after spread)
            else:
                target_ticks = 2.0; exp_hold = "25s - 50s"
                pred_candle = "⚡ CLEAN VACUUM (Level 3 Scalp)"
                grade = "⚡ Level 3 Clean Vacuum (+2.0t Target / +$10.00)"

    # ─── 2. 🔴 BEARISH VACUUM RADAR (PRIORITY 1: RIDE THE RED TREND!) ───
    # If sellers dominate and bid runway is open, GO WITH THE TREND (SELL PUT)!
    elif not in_cooldown and (vr_5 <= 0.88 or l1_ratio <= 0.70 or l2_ratio <= 0.74) and bid_trap_wall <= 8.0 and cvd_15s <= +10:
        if pred_p >= -0.18:
            signal_dir = "🔴 SELL (PUT)"
            runway_depth = bid_runway
            # Tier 3: BIG VACUUM + Delta Outflow Rocket
            if (runway_depth >= 5 or vr_10 <= 0.70) and cvd_15s <= -5:
                target_ticks = 4.0; exp_hold = "60s - 120s"
                pred_candle = "🚀 ROCKET VACUUM (Delta Outflow)"
                grade = "🏆 Level 10 Corridor + CVD Outflow (+4.0t)"
                trigger_type = "ROCKET_VACUUM"
            elif runway_depth >= 5 or vr_10 <= 0.70:
                target_ticks = 3.5; exp_hold = "50s - 90s"
                pred_candle = "🚀 BIG VACUUM (Level 10 Corridor)"
                grade = "🏆 Level 10 Big Corridor (+3.5t Runner)"
            # Tier 2: MEDIUM VACUUM
            elif runway_depth >= 3 or vr_5 <= 0.80:
                target_ticks = 2.5; exp_hold = "30s - 60s"
                pred_candle = "🎯 MEDIUM VACUUM (Level 5 Void)"
                grade = "🎯 Level 5 Medium Void (+2.5t Sniper)"
            # Tier 1: CLEAN VACUUM (Min 2.0t = 1.0 pip = $10.00)
            else:
                target_ticks = 2.0; exp_hold = "25s - 50s"
                pred_candle = "⚡ CLEAN VACUUM (Level 3 Scalp)"
                grade = "⚡ Level 3 Clean Vacuum (+2.0t Target / +$10.00)"

    # ─── 3. 🪤 TOP TRAP REVERSAL (ONLY IF NOT IN A BULL TREND! NEVER SHORT RAGING GREEN CANDLES!) ───
    elif not in_cooldown and bearish_delta_div and mom_60s <= 0.0 and pred_p >= -0.15:
        signal_dir = "🔴 SELL (PUT)"
        target_ticks = 2.5
        exp_hold = "30s - 60s"
        runway_depth = max(bid_runway, 3)
        pred_candle = "🪤 TOP TRAP REVERSAL (+2.5t)"
        grade = "🏆 Sideways Top Trap (Resistance Absorption)"
        trigger_type = "TOP_TRAP"

    # ─── 4. 🪤 BOTTOM TRAP REVERSAL (ONLY IF NOT IN A BEAR TREND! NEVER BUY PLUNGING RED CANDLES!) ───
    elif not in_cooldown and bullish_delta_div and mom_60s >= 0.0 and pred_c >= -0.15:
        signal_dir = "🟢 BUY (CALL)"
        target_ticks = 2.5
        exp_hold = "30s - 60s"
        runway_depth = max(ask_runway, 3)
        pred_candle = "🪤 BOTTOM TRAP REVERSAL (+2.5t)"
        grade = "🏆 Sideways Bottom Trap (Support Accumulation)"
        trigger_type = "BOTTOM_TRAP"

    # Trigger New Trade with Exact Limit Order Pricing
    if signal_dir and engine_state["active_trade"] is None:
        new_alert_fired = True
        limit_entry = bid_p1 if 'BUY' in signal_dir else ask_p1
        limit_target = round(limit_entry + target_ticks * TICK if 'BUY' in signal_dir else limit_entry - target_ticks * TICK, 5)
        limit_sl = round(limit_entry - 2.5 * TICK if 'BUY' in signal_dir else limit_entry + 2.5 * TICK, 5)

        engine_state["active_trade"] = {
            "direction": signal_dir,
            "entry_price": limit_entry,
            "target_price": limit_target,
            "sl_price": limit_sl,
            "target_ticks": target_ticks,
            "runway": runway_depth,
            "grade": grade,
            "trigger_type": trigger_type,
            "enter_time_str": datetime.now().strftime("%H:%M:%S"),
            "start_time": now_ts,
            "peak_move_pts": 0.0,
            "peak_move_ticks": 0.0,
            "current_move_ticks": 0.0,
            "current_pnl": 0.0,
            "elapsed_seconds": 0
        }

    # Update latest signal display
    active_runway_display = runway_depth if signal_dir else min(ask_runway, bid_runway)
    limit_entry_disp = (bid_p1 if 'BUY' in (signal_dir or '') else ask_p1) if signal_dir else mid
    engine_state["latest_signal"] = {
        "active": (signal_dir is not None),
        "new_alert": new_alert_fired,
        "direction": signal_dir if signal_dir else "SCANNING MICRO-VACUUM",
        "grade": grade if signal_dir else ("L1-L2 Dynamic Scan" if spread <= 0.0001 else "Spread Wide"),
        "runway": active_runway_display,
        "target_ticks": target_ticks,
        "expected_hold": exp_hold,
        "predicted_candle": pred_candle if signal_dir else "WAITING FOR MICRO-VACUUM",
        "limit_entry": f"{limit_entry_disp:.5f}",
        "prob_call": round(float(pred_c), 3),
        "prob_put": round(float(pred_p), 3),
        "vol_ratio_5": round(float(vr_5), 2),
        "vol_ratio_6_10": round(float(vr_6_10), 2),
        "deep_vacuum_score": round(float(deep_vacuum_score), 2),
        "ask_trap_wall": round(float(ask_trap_wall), 1),
        "bid_trap_wall": round(float(bid_trap_wall), 1),
        "ask_runway": ask_runway,
        "cvd_15s": float(cvd_15s),
        "delta_status": delta_status,
        "price_pos": round(float(price_pos), 2)
    }

    # Construct complete payload for Web Dashboard
    with state_lock:
        book_bids = list(live_market["bids"])
        book_offers = list(live_market["offers"])

    # Normalization for DOM UI progress bars (max quantity out of all levels)
    all_sizes = [b['size'] for b in book_bids] + [o['size'] for o in book_offers]
    max_size = max(all_sizes) if all_sizes else 200

    payload = {
        "server_time": datetime.now().strftime("%H:%M:%S"),
        "market": {
            "symbol": SYMBOL,
            "contractId": CONTRACT_ID,
            "status": live_market["status"],
            "mid_price": f"{mid:.5f}",
            "last_trade": f"{last_trade:.5f}",
            "spread": f"{spread / TICK:.1f}t ({spread:.5f})",
            "spread_val": spread,
            "bids": book_bids,
            "offers": book_offers,
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
        "trades": engine_state["session_trades"], # all session trades
        "mt5_bridge": {
            "connected": (time.time() - mt5_state["last_ping"] < 4.0),
            "account": mt5_state["account"],
            "last_order": mt5_state["last_order"],
            "last_ping_ago": round(time.time() - mt5_state["last_ping"], 1) if mt5_state["last_ping"] > 0 else -1
        }
    }

    return payload

# ==============================================================================
# 🌐 FASTAPI APPLICATION & WEBSOCKET BROADCAST
# ==============================================================================
app = FastAPI(title="6EZ6 Live Vacuum Sniper Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_ws_app = None

@app.post("/api/update_token")
async def update_token_api(request: Request):
    global TOKEN, current_ws_app
    try:
        body = await request.json()
        new_token = body.get("token")
        if new_token and len(new_token) > 20:
            if new_token != TOKEN:
                TOKEN = new_token
                print(f"🔑 [AUTO-SYNC] Received fresh token from Tradovate Web: {new_token[:30]}...")
                if current_ws_app:
                    current_ws_app.close()
            return {"status": "success", "msg": "Token updated successfully"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}
    return {"status": "ignored"}

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

mt5_state = {
    "status": "DISCONNECTED",
    "account": "480459863",
    "last_ping": 0,
    "last_order": None
}

@app.get("/")
def get_dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/mt5_signal")
def get_mt5_signal():
    """Ultra-fast JSON endpoint for MetaTrader 5 Expert Advisor Bridge"""
    with state_lock:
        mt5_state["last_ping"] = time.time()
        mt5_state["status"] = "CONNECTED"
        
        act = engine_state.get("active_trade")
        if act:
            dir_str = "BUY" if "BUY" in act["direction"] else "SELL"
            return {
                "active": True,
                "action": dir_str,
                "symbol": "EURUSD",
                "volume": 1.0,
                "trade_id": int(act.get("start_time", 0)),
                "target_ticks": act.get("target_ticks", 2.0),
                "sl_ticks": 2.5,
                "entry_price": act.get("entry_price", 0.0),
                "reason": act.get("trigger_type", "")
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

@app.post("/api/mt5_execution")
async def mt5_execution_report(request: Request):
    """Called by MT5 EA when an order is successfully punched on Exness"""
    try:
        body = await request.json()
        with state_lock:
            mt5_state["last_order"] = {
                "ticket": body.get("ticket", 0),
                "action": body.get("action", ""),
                "price": body.get("price", 0.0),
                "lots": body.get("lots", 1.0),
                "time": datetime.now().strftime("%H:%M:%S")
            }
            print(f"⚡ [EXNESS MT5 EXECUTED] {body.get('action')} Ticket #{body.get('ticket')} @ {body.get('price')}")
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_websockets.add(ws)
    print("💻 Browser Client Connected to Live Feed!")
    try:
        while True:
            # Keep alive and receive any client pings
            await ws.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(ws)
        print("💻 Browser Client Disconnected.")
    except Exception:
        if ws in connected_websockets:
            connected_websockets.remove(ws)

async def broadcast_loop():
    """Pumps live evaluation to all connected browser clients every 1 second"""
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
        except Exception as e:
            pass
        await asyncio.sleep(1.0)

@app.on_event("startup")
async def startup_event():
    # Start Tradovate WS in background thread
    t = threading.Thread(target=tradovate_worker, daemon=True)
    t.start()
    # Start FastAPI broadcast loop
    asyncio.create_task(broadcast_loop())

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="warning")
