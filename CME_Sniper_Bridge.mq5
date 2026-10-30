//+------------------------------------------------------------------+
//|                                           CME_Sniper_Bridge.mq5  |
//|               Copyright 2026, Institutional CME ML Sniper Bridge |
//|                   Zero Commission Exness MT5 Auto Execution EA   |
//+------------------------------------------------------------------+
#property copyright "CME ML Sniper Bridge"
#property link      "http://127.0.0.1:8000"
#property version   "1.00"
#property description "Auto-Executes CME Euro FX 10-Level Vacuum Scalps directly on Exness MT5"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//--- Input Parameters
input group "=== CME BRIDGE SETTINGS ==="
input string   InpServerUrl     = "http://127.0.0.1:8000/api/mt5_signal"; // Local Dashboard API URL
input double   InpDefaultLots   = 1.0;                                     // Trade Volume (Lots)
input ulong    InpMagicNumber   = 987654;                                  // Magic Number
input int      InpCheckTimerMs     = 100;                                     // Scan Interval (Milliseconds)
input int      InpSlippage         = 20;                                      // Max Slippage (Points)
input int      InpTakeProfitPoints = 25;                                      // Broker Server TP (25 Points = 2.5 Pips = +$25.00 Target)
input int      InpStopLossPoints   = 15;                                      // Broker Server SL (15 Points = 1.5 Pips = -$15.00 Risk)
input int      InpLockTriggerPoints = 10;                                     // Lock Trigger at +1.0 Pip (+10 Points = Covers spread & locks profit)
input int      InpLockProfitPoints  = 6;                                      // Min Lock Floor at +0.6 Pip (+6 Points = Guaranteed +$6.00 Cash)
input int      InpTrailingStep      = 3;                                      // Trailing Distance behind peak (3 Points = 0.3 Pips)

//--- Global Objects
CTrade         m_trade;
CPositionInfo  m_position;
long           g_last_trade_id  = 0;
bool           g_in_position    = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   m_trade.SetExpertMagicNumber(InpMagicNumber);
   m_trade.SetDeviationInPoints(InpSlippage);
   m_trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Start High-Speed 100ms Micro-Scalping Timer
   EventSetMillisecondTimer(InpCheckTimerMs);

   Print("==================================================================");
   Print("🚀 CME LIVE VACUUM SNIPER BRIDGE INITIALIZED!");
   Print("   Target Broker: Exness MT5");
   Print("   Signal URL: ", InpServerUrl);
   Print("   Default Lot Size: ", DoubleToString(InpDefaultLots, 2));
   Print("   Magic Number: ", InpMagicNumber);
   Print("==================================================================");

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("🛑 CME Sniper Bridge Stopped.");
}

//+------------------------------------------------------------------+
//| Expert tick function (Instant millisecond price updates)          |
//+------------------------------------------------------------------+
void OnTick()
{
   ManageTrailingProfitLock(_Symbol);
}

//+------------------------------------------------------------------+
//| Expert timer function (Runs every 100 milliseconds)              |
//+------------------------------------------------------------------+
void OnTimer()
{
   ManageTrailingProfitLock(_Symbol);

   string headers = "User-Agent: CME-Sniper-MT5\r\n";
   char post[], result[];
   string result_headers;
   int timeout = 80; // 80ms timeout to maintain hyper-speed

   ResetLastError();
   int res = WebRequest("GET", InpServerUrl, headers, timeout, post, result, result_headers);

   if(res == 200)
   {
      string json = CharArrayToString(result);
      ProcessBridgeSignal(json);
   }
   else if(res == -1)
   {
      int err = GetLastError();
      if(err == 4014) // ERR_FUNCTION_NOT_ALLOWED
      {
         PrintOnce("⚠️ WebRequest blocked! In MT5: Go to Tools -> Options -> Expert Advisors -> Allow WebRequest for: http://127.0.0.1:8000");
      }
   }
}

//+------------------------------------------------------------------+
//| Process JSON Signal from Python Server                           |
//+------------------------------------------------------------------+
void ProcessBridgeSignal(string json)
{
   // Parse simple json fields
   string action = ExtractJsonValue(json, "action");
   long trade_id = (long)StringToInteger(ExtractJsonValue(json, "trade_id"));
   string sym = ExtractJsonValue(json, "symbol");
   if(sym == "" || !SymbolInfoInteger(sym, SYMBOL_VISIBLE))
   {
      sym = _Symbol; // Fallback to current chart symbol (EURUSDm)
   }

   // ─── 1. CLOSE POSITION (EXIT SIGNAL FROM PYTHON SNIPER) ───
   if(action == "CLOSE")
   {
      CloseOurPositions(sym);
      g_last_trade_id = 0;
      return;
   }

   // ─── 2. NEW BUY ENTRY ───
   if(action == "BUY")
   {
      if(trade_id != g_last_trade_id && !HasOpenPosition(sym))
      {
         double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
         double point = SymbolInfoDouble(sym, SYMBOL_POINT);
         int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

         double tp = (InpTakeProfitPoints > 0) ? NormalizeDouble(ask + InpTakeProfitPoints * point, digits) : 0;
         double sl = (InpStopLossPoints > 0) ? NormalizeDouble(ask - InpStopLossPoints * point, digits) : 0;

         if(m_trade.Buy(InpDefaultLots, sym, ask, sl, tp, "CME Vacuum Sniper"))
         {
            g_last_trade_id = trade_id;
            ulong ticket = m_trade.ResultOrder();
            Print("✅ [EXNESS] 🟢 BUY 1.0 LOT with HARD TP @ ", DoubleToString(tp, digits), " SL @ ", DoubleToString(sl, digits), " Ticket #", ticket);
            ReportExecutionToDashboard("BUY", ticket, ask, InpDefaultLots);
         }
         else
         {
            Print("❌ BUY Execution Failed: ", m_trade.ResultRetcodeDescription());
         }
      }
      return;
   }

   // ─── 3. NEW SELL ENTRY ───
   if(action == "SELL")
   {
      if(trade_id != g_last_trade_id && !HasOpenPosition(sym))
      {
         double bid = SymbolInfoDouble(sym, SYMBOL_BID);
         double point = SymbolInfoDouble(sym, SYMBOL_POINT);
         int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

         double tp = (InpTakeProfitPoints > 0) ? NormalizeDouble(bid - InpTakeProfitPoints * point, digits) : 0;
         double sl = (InpStopLossPoints > 0) ? NormalizeDouble(bid + InpStopLossPoints * point, digits) : 0;

         if(m_trade.Sell(InpDefaultLots, sym, bid, sl, tp, "CME Vacuum Sniper"))
         {
            g_last_trade_id = trade_id;
            ulong ticket = m_trade.ResultOrder();
            Print("✅ [EXNESS] 🔴 SELL 1.0 LOT with HARD TP @ ", DoubleToString(tp, digits), " SL @ ", DoubleToString(sl, digits), " Ticket #", ticket);
            ReportExecutionToDashboard("SELL", ticket, bid, InpDefaultLots);
         }
         else
         {
            Print("❌ SELL Execution Failed: ", m_trade.ResultRetcodeDescription());
         }
      }
      return;
   }
}

//+------------------------------------------------------------------+
//| Report order to UI Dashboard in Real Time                        |
//+------------------------------------------------------------------+
void ReportExecutionToDashboard(string act, ulong ticket, double price, double lots)
{
   string url = "http://127.0.0.1:8000/api/mt5_execution";
   string headers = "Content-Type: application/json\r\n";
   string body = StringFormat("{\"ticket\":%I64u,\"action\":\"%s\",\"price\":%.5f,\"lots\":%.2f}", ticket, act, price, lots);
   char post[], result[];
   string result_headers;
   StringToCharArray(body, post, 0, StringLen(body));
   WebRequest("POST", url, headers, 100, post, result, result_headers);
}

//+------------------------------------------------------------------+
//| Helper: Check if we have an open position on symbol              |
//+------------------------------------------------------------------+
bool HasOpenPosition(string sym)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Magic() == InpMagicNumber && m_position.Symbol() == sym)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Helper: Close all positions with our Magic Number                |
//+------------------------------------------------------------------+
void CloseOurPositions(string sym)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Magic() == InpMagicNumber && m_position.Symbol() == sym)
         {
            m_trade.PositionClose(m_position.Ticket());
            Print("🔒 [EXNESS] POSITION CLOSED! Profit locked into account.");
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage Trailing Profit Lock Directly on Exness Broker Server     |
//+------------------------------------------------------------------+
void ManageTrailingProfitLock(string sym)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Magic() == InpMagicNumber && m_position.Symbol() == sym)
         {
            double open_price = m_position.PriceOpen();
            double sl = m_position.StopLoss();
            double tp = m_position.TakeProfit();
            double point = SymbolInfoDouble(sym, SYMBOL_POINT);
            int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

            // BUY POSITION TRAILING LOCK
            if(m_position.PositionType() == POSITION_TYPE_BUY)
            {
               double bid = SymbolInfoDouble(sym, SYMBOL_BID);
               double gain_points = (bid - open_price) / point;

               // Once price gains InpLockTriggerPoints (e.g. +8 points = +$8.00):
               if(gain_points >= InpLockTriggerPoints)
               {
                  double desired_sl = NormalizeDouble(open_price + (gain_points - InpTrailingStep) * point, digits);
                  double min_lock_sl = NormalizeDouble(open_price + InpLockProfitPoints * point, digits);
                  if(desired_sl < min_lock_sl) desired_sl = min_lock_sl;

                  // Move SL into profit if desired_sl is higher than current SL
                  if(sl < desired_sl)
                  {
                     if(m_trade.PositionModify(m_position.Ticket(), desired_sl, tp))
                     {
                        Print("🔒 [EXNESS TRAILING LOCK] 🟢 BUY SL Moved UP to ", DoubleToString(desired_sl, digits), 
                              " | Profit Locked: +$", DoubleToString((desired_sl - open_price)/point, 1));
                     }
                  }
               }
            }
            // SELL POSITION TRAILING LOCK
            else if(m_position.PositionType() == POSITION_TYPE_SELL)
            {
               double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
               double gain_points = (open_price - ask) / point;

               // Once price gains InpLockTriggerPoints (e.g. +8 points = +$8.00):
               if(gain_points >= InpLockTriggerPoints)
               {
                  double desired_sl = NormalizeDouble(open_price - (gain_points - InpTrailingStep) * point, digits);
                  double min_lock_sl = NormalizeDouble(open_price - InpLockProfitPoints * point, digits);
                  if(desired_sl > min_lock_sl) desired_sl = min_lock_sl;

                  // Move SL down into profit if desired_sl is lower than current SL or current SL == 0
                  if(sl > desired_sl || sl == 0.0)
                  {
                     if(m_trade.PositionModify(m_position.Ticket(), desired_sl, tp))
                     {
                        Print("🔒 [EXNESS TRAILING LOCK] 🔴 SELL SL Moved DOWN to ", DoubleToString(desired_sl, digits), 
                              " | Profit Locked: +$", DoubleToString((open_price - desired_sl)/point, 1));
                     }
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Simple JSON parser helper                                        |
//+------------------------------------------------------------------+
string ExtractJsonValue(string json, string key)
{
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0)
   {
      pattern = "\"" + key + "\" :";
      pos = StringFind(json, pattern);
   }
   if(pos < 0) return "";

   int start = pos + StringLen(pattern);
   while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' || StringGetCharacter(json, start) == '\"'))
      start++;

   int end = start;
   while(end < StringLen(json))
   {
      ushort c = StringGetCharacter(json, end);
      if(c == '\"' || c == ',' || c == '}' || c == ']' || c == '\r' || c == '\n')
         break;
      end++;
   }

   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Print message only once                                          |
//+------------------------------------------------------------------+
void PrintOnce(string msg)
{
   static string last_msg = "";
   if(msg != last_msg)
   {
      Print(msg);
      last_msg = msg;
   }
}
