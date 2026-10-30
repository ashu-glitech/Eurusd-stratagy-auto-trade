# =========================================================================================================================
# 🏛️ CME EURO FX FUTURES (`6EZ6`): MASTER PRODUCTION DUAL-ENGINE HFT ROUTER
# 🛡️ Quantitative Architecture (Marcos López de Prado Microstructure Framework):
#    1. Dynamic Tape Regime Switching: Fast Trending Momentum vs Slow/Thick Matrix
#    2. Fast Trend Engine: 20s Wave Cooldown + +3.0t Runner Target + CVD Divergence Protection + 1.0t Micro-Ratchet
#    3. Slow Market Engine: 5m Macro Activity Sensor (flips >= 22) + Stacked L1-L3 Matrix + 80s Patient Hold
#    4. Strict Target & Stops: +2.0t Target ($12.50) / +3.0t Runner ($18.75) | -1.5t Stop Loss ($9.375)
#    5. Round-Trip Commission: Active broker commission strictly deducted at $2.00/trade on 1 Micro Lot ($6.25/tick)
# 🧪 Out-of-Sample Holdout Days: Oct 19, 20, 21, 22, 26, and Oct 27
# =========================================================================================================================
import sys, os, warnings
import pandas as pd, numpy as np, lightgbm as lgb
warnings.filterwarnings('ignore')

sys.path.append('/Users/admin/Downloads/nifty_level2/eurousd_models_10level')
from kaggle_training_master_script import extract_dual_engine_features, DUAL_ENGINE_FEATURES, TICK, TICK_VALUE, TARGET_TICKS, RUNNER_TICKS, STOP_TICKS

call_path = '/Users/admin/Downloads/nifty_level2/eurousd_models_10level/eurousd_consensus_call_model.txt'
put_path  = '/Users/admin/Downloads/nifty_level2/eurousd_models_10level/eurousd_consensus_put_model.txt'
model_call = lgb.Booster(model_file=call_path)
model_put  = lgb.Booster(model_file=put_path)

DATA_DIR = '/Users/admin/Downloads/euro_usd_10level_clean_parquet'
files = [
    f'{DATA_DIR}/tradovate_6EZ6_live_data_2026-10-19.parquet',
    f'{DATA_DIR}/tradovate_6EZ6_live_data_2026-10-20.parquet',
    f'{DATA_DIR}/tradovate_6EZ6_live_data_2026-10-21.parquet',
    f'{DATA_DIR}/tradovate_6EZ6_live_data_2026-10-22.parquet',
    f'{DATA_DIR}/tradovate_6EZ6_live_data_2026-10-26_clean.parquet',
    f'{DATA_DIR}/tradovate_6EZ6_live_data_2026-10-27_clean.parquet',
]

print("=" * 145)
print("🏛️ CME EURO FX FUTURES (`6EZ6`): MASTER PRODUCTION DUAL-ENGINE BENCHMARK (6 BLIND OUT-OF-SAMPLE DAYS)")
print("   Strict 1 Micro Lot ($6.25/t) | +2.0t Target / +3.0t Runner | Full Broker Fee Deducted ($2.00/trade)")
print("=" * 145)

results = []

for tf in files:
    lbl = os.path.basename(tf).replace('tradovate_6EZ6_live_data_', '').replace('.parquet', '').replace('_clean', '')
    df = extract_dual_engine_features(tf)
    p = df['mid_price'].values
    ps = pd.Series(p)
    N = len(df)
    
    dp = ps.diff().abs()
    is_move = (dp > 0).astype(int)
    flips_60s = is_move.rolling(60).sum().fillna(0).values
    flips_300s = is_move.rolling(300).sum().fillna(0).values
    range_180s = df['range_180s_ticks'].values
    range_300s = (ps.rolling(300).max() - ps.rolling(300).min()).fillna(0).values / TICK
    range_15m = (ps.rolling(900).max() - ps.rolling(900).min()).fillna(0).values / TICK
    
    df['pred_c'] = model_call.predict(df[DUAL_ENGINE_FEATURES])
    df['pred_p'] = model_put.predict(df[DUAL_ENGINE_FEATURES])
    df['consensus_ratio'] = (df['consensus_depth_bull'] - df['consensus_depth_bear']) / 10.0
    
    b_mat = df['bid_q1'] + df['bid_q2'] + df['bid_q3']
    a_mat = df['ask_q1'] + df['ask_q2'] + df['ask_q3']
    mat_ratio = (b_mat / np.maximum(1.0, a_mat)).values
    df['cum_ratio_l3'] = mat_ratio
    
    active_exit = -1
    lock_call = -1; lock_put = -1
    w = 0; l = 0; s = 0; gross = 0.0
    fast_t = 0; slow_t = 0
    
    for i in range(300, N - 95):
        if i <= active_exit: continue
        r = df.iloc[i]; curr_p = p[i]
        hour = pd.to_datetime(r['datetime']).hour
        
        # Session filter: active liquid trading window (02:00 to 10:00 UTC)
        if hour < 2 or hour > 10: continue
        if r['range_180s_ticks'] > 5.5 or r['range_180s_ticks'] < 2.0: continue
        if r['fractal_er'] < 0.35 or r['whirlpool_flips_6s'] > 4: continue
        
        is_fast_regime = (range_15m[i] >= 6.0) or (range_180s[i] >= 3.5 and flips_60s[i] >= 10)
        trade = None
        engine_used = None
        hold_sec = 65 if is_fast_regime else 80
        cd_now = 20 if is_fast_regime else 50
        
        if is_fast_regime:
            cvd10 = r['cvd_10s']
            has_w_c = (cvd10 <= -3) or (r['runup_45s'] >= 2.0) or (r['cvd_180s'] < -20)
            has_w_p = (cvd10 >= 3) or (r['runup_45s'] <= -2.0) or (r['cvd_180s'] > 20)
            c_ratio = r['consensus_ratio']
            cond_c = (c_ratio >= 0.10) and (r['cum_ratio_l3'] >= 1.25) and (r['ask_refill_speed'] <= 0)
            cond_p = (c_ratio <= -0.10) and (r['cum_ratio_l3'] <= 0.80) and (r['bid_refill_speed'] <= 0)
            if not has_w_c and cond_c and r['micro_lead_ticks'] >= -0.05 and r['pred_c'] >= 0.0 and i > lock_call:
                trade = 'CALL'; engine_used = 'FAST'
            elif not has_w_p and cond_p and r['micro_lead_ticks'] <= 0.05 and r['pred_p'] >= 0.0 and i > lock_put:
                trade = 'PUT'; engine_used = 'FAST'
        else:
            if flips_300s[i] < 22 or range_300s[i] < 2.5: continue
            is_c = (r['ask_q1'] <= 30) and (r['bid_q1'] >= 55) and (r['bid_q2'] >= 100) and (mat_ratio[i] >= 1.35)
            is_p = (r['bid_q1'] <= 30) and (r['ask_q1'] >= 55) and (r['ask_q2'] >= 100) and (mat_ratio[i] <= 0.74)
            if is_c and i > lock_call: trade = 'CALL'; engine_used = 'SLOW'
            elif is_p and i > lock_put: trade = 'PUT'; engine_used = 'SLOW'
            
        if not trade: continue
        if engine_used == 'FAST': fast_t += 1
        else: slow_t += 1
        
        entry_p = curr_p; outcome = None; final_tick = 0.0; reached_1_5t = False
        exit_idx = -1
        
        for sec in range(1, hold_sec + 1):
            curr_idx = i + sec
            diff = (p[curr_idx] - entry_p)/TICK if trade == 'CALL' else (entry_p - p[curr_idx])/TICK
            
            if is_fast_regime:
                # Fast trend: capture up to +3.0 ticks runner ($18.75)
                if diff >= 3.0: outcome = 'RUNNER'; final_tick = 3.0; exit_idx = curr_idx; break
                if diff >= 1.5 and not reached_1_5t: reached_1_5t = True
                if reached_1_5t and diff <= 1.0: outcome = 'WIN_LOCK'; final_tick = 1.0; exit_idx = curr_idx; break
                if diff <= -1.5: outcome = 'LOSS'; final_tick = -1.5; exit_idx = curr_idx; break
                if sec == hold_sec:
                    if diff > 0.0: outcome = 'EXP_WIN'; final_tick = diff
                    elif diff < 0.0: outcome = 'EXP_LOSS'; final_tick = diff
                    else: outcome = 'SCRT'; final_tick = 0.0
                    exit_idx = curr_idx; break
            else:
                # Slow regime: strict +2.0 ticks target ($12.50)
                if diff >= 2.0: outcome = 'WIN_2T'; final_tick = 2.0; exit_idx = curr_idx; break
                elif diff <= -1.5: outcome = 'LOSS'; final_tick = -1.5; exit_idx = curr_idx; break
                elif sec == hold_sec:
                    if diff > 0.0: outcome = 'EXP_WIN'; final_tick = diff
                    elif diff < 0.0: outcome = 'EXP_LOSS'; final_tick = diff
                    else: outcome = 'SCRT'; final_tick = 0.0
                    exit_idx = curr_idx; break
                    
        active_exit = exit_idx
        c = final_tick * TICK_VALUE; gross += c
        if 'WIN' in outcome or 'RUNNER' in outcome: w += 1
        elif 'LOSS' in outcome: l += 1
        else: s += 1
        
        if trade == 'CALL': lock_call = active_exit + cd_now
        else: lock_put = active_exit + cd_now
        
    tot = w + l + s
    comm = tot * 2.00
    net = gross - comm
    cwr = (w / (w + l) * 100) if (w + l) > 0 else 0
    results.append({'lbl': lbl, 'tot': tot, 'fast': fast_t, 'slow': slow_t, 'w': w, 'l': l, 's': s, 'gross': gross, 'comm': comm, 'net': net, 'cwr': cwr})
    print(f"| {lbl:12s} | Trades={tot:2d} (F={fast_t:2d}, S={slow_t:2d}) | W={w:2d} 💎 | L={l:2d} ❌ | S={s:2d} | Gross=${gross:+8.2f} 💰 | Net Cash=${net:+8.2f} 🟢 | Clean WR={cwr:5.1f}% |")

tot_trades = sum(r['tot'] for r in results)
tot_fast = sum(r['fast'] for r in results)
tot_slow = sum(r['slow'] for r in results)
tot_w = sum(r['w'] for r in results)
tot_l = sum(r['l'] for r in results)
tot_s = sum(r['s'] for r in results)
tot_gross = sum(r['gross'] for r in results)
tot_comm = sum(r['comm'] for r in results)
tot_net = sum(r['net'] for r in results)
tot_cwr = (tot_w / (tot_w + tot_l) * 100) if (tot_w + tot_l) > 0 else 0

print("=" * 145)
print(f"| TOTAL 6 DAYS | Trades={tot_trades:2d} (F={tot_fast:2d}, S={tot_slow:2d}) | W={tot_w:2d} 💎 | L={tot_l:2d} ❌ | S={tot_s:2d} | Gross=${tot_gross:+8.2f} 💰 | PURE NET CASH=${tot_net:+8.2f} 🏆 | Clean WR={tot_cwr:5.1f}% |")
print("=" * 145)
