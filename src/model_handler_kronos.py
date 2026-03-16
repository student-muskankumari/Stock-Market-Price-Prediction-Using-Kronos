# src/model_handler_kronos.py
"""
Robust Kronos handler with graceful fallback.

Behavior:
 - Try to load a Kronos tokenizer (installed package or HF tokenizer).
 - Try to load a Kronos HF model via transformers (trust_remote_code=True).
 - If tokenizer or model are missing / incompatible:
    - Do NOT raise a fatal error.
    - Use a deterministic CPU fallback predictor that extrapolates the recent Close
      prices using a linear trend (numpy.polyfit degree 1), then synthesizes
      OHLCV candlesticks around those predicted closes.
 - This ensures Streamlit UI remains usable and never throws a blocking error.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class KronosError(RuntimeError):
    pass

class KronosModelHandler:
    def __init__(self, model_id: str = "NeoQuasar/Kronos-mini", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None
        self.tokenizer_available = False
        self.model_available = False
        self._load_attempted = False
        self._load()

    def _load(self):
        """
        Attempt to load tokenizer and model. If any step fails, log warnings and
        set flags so we can use a safe fallback predictor.
        """
        self._load_attempted = True
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
            self.AutoModelForCausalLM = AutoModelForCausalLM
        except Exception as e:
            # transformers missing -> fallback only
            logger.warning("transformers not available or failed to import: %s", e)
            self.tokenizer_available = False
            self.model_available = False
            return

       
        try:
           
            try:
                from kronos.tokenizer import KronosTokenizer  # type: ignore
                
                if hasattr(KronosTokenizer, "from_pretrained"):
                    try:
                        self.tokenizer = KronosTokenizer.from_pretrained(self.model_id.replace("Kronos-mini","Kronos-Tokenizer-base"))
                    except Exception:
                       
                        self.tokenizer = KronosTokenizer
                else:
                    self.tokenizer = KronosTokenizer
                self.tokenizer_available = True
                logger.info("Loaded KronosTokenizer from local kronos package.")
            except Exception as e_local:
                
                try:
                    hf_tok_repo = self.model_id.replace("Kronos-mini","Kronos-Tokenizer-base")
                    self.tokenizer = self.AutoTokenizer.from_pretrained(hf_tok_repo, use_fast=False)
                    self.tokenizer_available = True
                    logger.info("Loaded Kronos tokenizer from HF repo: %s", hf_tok_repo)
                except Exception as e_hf:
                    
                    try:
                        self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
                        self.tokenizer_available = True
                        logger.info("Loaded AutoTokenizer fallback from model repo: %s", self.model_id)
                    except Exception as e_auto:
                        self.tokenizer = None
                        self.tokenizer_available = False
                        logger.warning("No Kronos tokenizer found - will use text/quant fallback. HF error: %s ; Auto fallback error: %s", e_hf, e_auto)
        except Exception as e:
            logger.warning("Tokenizer loading encountered unexpected error: %s", e)
            self.tokenizer = None
            self.tokenizer_available = False

        
        try:
            self.model = self.AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
            
            try:
                import torch
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model.to("cuda")
                    logger.info("Model moved to CUDA.")
            except Exception:
                logger.info("Torch not available or CUDA not used; model remains on CPU.")
            self.model_available = True
            logger.info("Loaded Kronos model: %s", self.model_id)
        except Exception as e:

            logger.warning("Failed to load Kronos model %s: %s", self.model_id, e)
            self.model = None
            self.model_available = False

  
    def encode_klines(self, klines: List[Dict[str, float]]) -> List[int]:
        """
        Use tokenizer.encode_klines if available; else fallback to string encoding or quantized ints.
        """
        if self.tokenizer_available and hasattr(self.tokenizer, "encode_klines"):
            try:
                return self.tokenizer.encode_klines(klines)
            except Exception as e:
                logger.warning("tokenizer.encode_klines failed: %s (falling back)", e)

        if self.tokenizer_available and hasattr(self.tokenizer, "encode"):
            text = "|".join([f"{k['open']},{k['high']},{k['low']},{k['close']},{k.get('volume',0.0)}" for k in klines])
            return self.tokenizer.encode(text, add_special_tokens=False)

       
        flat = []
        for k in klines:
            flat.extend([k["open"], k["high"], k["low"], k["close"], k.get("volume", 0.0)])
        arr = np.array(flat, dtype=np.float32)
        scaled = (arr * 100).astype(np.int32).tolist()
        return [int(x & 0x7fffffff) for x in scaled]

    def decode_klines(self, token_ids: List[int]) -> List[Dict[str, float]]:
        """
        Use tokenizer.decode_klines or tokenizer.decode if available else remap quantized ints to floats.
        """
        if self.tokenizer_available and hasattr(self.tokenizer, "decode_klines"):
            try:
                return self.tokenizer.decode_klines(token_ids)
            except Exception as e:
                logger.warning("tokenizer.decode_klines failed: %s (falling back)", e)

        if self.tokenizer_available and hasattr(self.tokenizer, "decode"):
            txt = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            parts = [p.strip() for p in txt.replace("|", ",").split(",") if p.strip() != ""]
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except:
                    pass
            klines = []
            for i in range(0, len(floats), 5):
                chunk = floats[i:i+5]
                if len(chunk) < 4:
                    break
                k = {"open": chunk[0], "high": chunk[1], "low": chunk[2], "close": chunk[3], "volume": chunk[4] if len(chunk) > 4 else 0.0}
                klines.append(k)
            return klines

       
        floats = [float(t) / 100.0 for t in token_ids]
        klines = []
        for i in range(0, len(floats), 5):
            chunk = floats[i:i+5]
            if len(chunk) < 4:
                break
            k = {"open": chunk[0], "high": chunk[1], "low": chunk[2], "close": chunk[3], "volume": chunk[4] if len(chunk) > 4 else 0.0}
            klines.append(k)
        return klines

   
    def _fallback_predict_closes(self, history_closes: List[float], horizon: int) -> List[float]:
        """
        Simple deterministic extrapolation:
        - Fit a linear trend to the last N closes (default N=30 or fewer)
        - Predict future closes by extending the line.
        - If linear fit is poor, revert to a simple last-value repeat.
        """
        arr = np.asarray(history_closes, dtype=np.float64)
        n = min(len(arr), 60)
        if n < 2:
          
            return [float(arr[-1]) for _ in range(horizon)]
        y = arr[-n:]
        x = np.arange(len(y), dtype=np.float64)
        try:
           
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs[0], coeffs[1]
            
            preds = []
            for i in range(1, horizon + 1):
                xi = len(y) - 1 + i
                val = slope * xi + intercept
                preds.append(float(val))
           
            if not np.isfinite(slope) or abs(slope) > 1e6:
                return [float(y[-1]) for _ in range(horizon)]
            return preds
        except Exception:
            return [float(y[-1]) for _ in range(horizon)]

    def _synthesize_klines_from_closes(self, last_close: float, predicted_closes: List[float], history_volumes: Optional[List[float]] = None) -> List[Dict[str, float]]:
        """
        Given predicted close prices, synthesize reasonable open/high/low/volume for each kline:
         - open = previous close
         - close = predicted close
         - high = max(open, close) * (1 + spread)
         - low = min(open, close) * (1 - spread)
         - spread is a small percent based on recent volatility (use sliding std)
         - volume = median of history volumes (or 0)
        """
        out = []
       
        vol = 0.01  # default 1%
        if history_volumes is None:
            history_volumes = []
        
        if history_volumes is None:
            history_volumes = []
       
        spread_base = 0.008  # 0.8%
        prev = float(last_close)
        med_vol = float(np.median(history_volumes)) if len(history_volumes) > 0 else 0.0

        for c in predicted_closes:
           
            rel = abs((c - prev) / (prev + 1e-9))
            spread = spread_base + min(0.02, rel * 0.5)
            open_p = prev
            close_p = float(c)
            high_p = max(open_p, close_p) * (1.0 + spread)
            low_p = min(open_p, close_p) * (1.0 - spread)
            volume_p = med_vol if med_vol > 0 else (history_volumes[-1] if len(history_volumes) > 0 else 0.0)
            out.append({"open": open_p, "high": float(high_p), "low": float(low_p), "close": float(close_p), "volume": float(volume_p)})
            prev = close_p
        return out

    
    def predict(self, history_klines: List[Dict[str, float]], horizon_klines: int = 10, max_tokens_per_kline: int = 8, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Generate future K-lines:
         - If model_available and tokenizer present: run the Kronos LM generation pipeline
         - Else: use deterministic CPU fallback predictor
        Returns: dict { "klines": [...], "tokens": [...] (may be empty for fallback) }
        """
        
        if not isinstance(history_klines, list) or len(history_klines) == 0:
            raise ValueError("history_klines must be a non-empty list of kline dicts")

       
        if self.model_available and (self.tokenizer_available or True):
           
            try:
                import torch
                input_ids = self.encode_klines(history_klines)
                if len(input_ids) == 0:
                    raise RuntimeError("Input encoding returned empty token list for LM inference")
                input_tensor = torch.tensor([input_ids], dtype=torch.long)
                if self.device == "cuda" and torch.cuda.is_available():
                    input_tensor = input_tensor.to("cuda")
                tokens_per_kline = int(max_tokens_per_kline)
                max_new_tokens = horizon_klines * tokens_per_kline + 16
                try:
                    outputs = self.model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        do_sample=False if temperature == 0.0 else True,
                        temperature=float(temperature),
                        pad_token_id=(self.tokenizer.pad_token_id if getattr(self.tokenizer, "pad_token_id", None) is not None else None),
                        eos_token_id=(self.tokenizer.eos_token_id if getattr(self.tokenizer, "eos_token_id", None) is not None else None),
                    )
                except TypeError:
                    outputs = self.model.generate(
                        input_tensor,
                        max_length=input_tensor.shape[1] + max_new_tokens,
                        do_sample=False if temperature == 0.0 else True,
                        temperature=float(temperature),
                    )
                gen = outputs[0].cpu().tolist()
                gen_only = gen[len(input_ids):]
                pred_klines = self.decode_klines(gen_only)
                
                if len(pred_klines) > horizon_klines:
                    pred_klines = pred_klines[:horizon_klines]
                return {"klines": pred_klines, "tokens": gen_only}
            except Exception as e:
                logger.warning("HF Kronos inference failed, switching to CPU fallback predictor: %s", e)
              
        
        closes = [float(k["close"]) for k in history_klines if "close" in k]
        volumes = [float(k.get("volume", 0.0)) for k in history_klines]
        last_close = closes[-1] if len(closes) > 0 else 0.0
        predicted_closes = self._fallback_predict_closes(closes, horizon_klines)
        pred_klines = self._synthesize_klines_from_closes(last_close, predicted_closes, history_volumes=volumes)
        # tokens empty because we didn't use LM
        return {"klines": pred_klines, "tokens": []}
