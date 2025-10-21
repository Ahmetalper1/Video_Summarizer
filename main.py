# -*- coding: utf-8 -*-
"""
Transcriber + (Opsiyonel) Diarization + Yerel mT5 Özetleme (TR + EN)
- faster-whisper ile ASR
- pyannote (opsiyonel) konuşmacı ayrımı (TorchCodec VARSA: doğrudan dosya, YOKSA: RAM'den dalga)
- mT5 (LoRA-merge) yerel modelle içerik-odaklı özet (madde/paragraf, kısa/orta/detaylı)

Bu sürüm: Transcribe ve Summarize ayrı butonlar.
Ayarlar değişince script rerun olur ama transcribe çıktısı session_state içinde korunur.
"""

import os, io, sys, json, csv, time, re, base64, mimetypes, html
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any, Tuple
from datetime import datetime
import warnings
import torch
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import ffmpeg

from faster_whisper import WhisperModel
import imageio_ffmpeg

# === YENİ: soundfile + numpy (dalga RAM'den pyannote'a verme) ===
import soundfile as sf
import numpy as np

# -------------------- FFmpeg yolunu sabitle --------------------
os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\Users\akiziltunc\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

_ffmpeg_dir = str(Path(FFMPEG_BIN).parent)
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# -------------------- (Opsiyonel) pyannote --------------------
warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec is not installed correctly.*",
    category=UserWarning,
)
HAS_PYANNOTE = False  # import'u enable_diar True olduğunda yapacağız

# === TorchCodec entegrasyonu (VAR/YOK algıla) ===
TORCHCODEC_OK = False
try:
    import torchcodec  # yalnızca varlığını kontrol etmek yeterli
    TORCHCODEC_OK = True
except Exception:
    TORCHCODEC_OK = False

# -------------------- mT5 / Transformers --------------------
import sentencepiece  # tokenizer için
from transformers import AutoTokenizer, MT5ForConditionalGeneration

# ================== Streamlit ==================
st.set_page_config(page_title="VIDEO SUMMARIZER", layout="wide")

# ================== Session State Init ==================
def _init_state():
    ss = st.session_state
    ss.setdefault("in_path", "")
    ss.setdefault("src_name", "")
    ss.setdefault("out_root", str(Path.cwd() / "outputs"))
    ss.setdefault("out_dir", "")
    ss.setdefault("base_name", "")
    ss.setdefault("seg_list", [])           # ASR segmentleri
    ss.setdefault("words", [])              # kelime zamanları
    ss.setdefault("diar_segments", [])      # diarization aralıkları
    ss.setdefault("transcript_text", "")
    ss.setdefault("asr_done", False)
    ss.setdefault("summary_text", "")
    ss.setdefault("paths", {})              # üretilen dosya yolları
    ss.setdefault("model_loaded_keys", {})  # cache info
    ss.setdefault("uploaded_temp", "")      # yüklenen dosyanın kopyalandığı geçici path
_init_state()

# ================== Yardımcılar (metin/format) ==================
def ensure_str(x) -> str:
    return x if isinstance(x, str) else str(x or "")

def clean_text(t: Any) -> str:
    t = ensure_str(t).replace("\r", " ").replace("\n", " ")
    return " ".join(t.split())

def collapse_repeats(text: str, max_repeat=1) -> str:
    text = ensure_str(text)
    toks = re.split(r'(\s+)', text)
    out, last, rep = [], None, 0
    for t in toks:
        if t.isspace():
            out.append(t); continue
        w = t.lower()
        if re.fullmatch(r"[A-Za-zğüşöçıİĞÜŞÖÇ']+", t):
            if w == last:
                rep += 1
                if rep <= max_repeat: out.append(t)
            else:
                last = w; rep = 1; out.append(t)
        else:
            last, rep = None, 0; out.append(t)
    return clean_text("".join(out))

def human_ts(seconds: float, srt=False):
    if seconds < 0: seconds = 0.0
    h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = seconds % 60
    if srt:
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    return f"{h:02d}:{m:02d}:{s:06.3f}"

# ================== Parçalama ==================
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

def chunk_text_mode(text: str, mode: str) -> List[str]:
    base_max = {"short": 1400, "medium": 2000, "long": 2600}[mode]  # karakter
    base_ovl = {"short": 120,  "medium": 150,  "long": 180}[mode]
    text = ensure_str(text)
    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sents:
        s = ensure_str(s)
        if len(cur) + len(s) + 1 <= base_max:
            cur = (cur + " " + s).strip()
        else:
            if cur: chunks.append(cur)
            cur = (cur[-base_ovl:] + " " + s).strip() if base_ovl and len(cur) > base_ovl else s
    if cur: chunks.append(cur)
    return chunks

# ================== Prompt & Temizleyici ==================
def build_task_prompt(style: str, length_mode: str, lang_ui: str, txt: str) -> str:
    if lang_ui == "turkish":
        task = (
            "<task>\n"
            "YouTube konuşmasını içerik odaklı özetle. Metni kopyalama, kendi cümlelerinle yaz.\n"
            "Şema:\n"
            "1) Merkez tez / ana mesaj\n"
            "2) Destekleyici kanıtlar ve anekdotlar\n"
            "3) Neden önemli (çıkarım/etki)\n"
            "4) Pratik çıkarımlar / öneriler\n"
            "5) Tek cümlelik öz mesaj\n"
            "Meta cümleler veya doğrudan alıntı yok.\n"
            "</task>\n"
        )
    else:
        task = (
            "<task>\n"
            "Summarize the talk with a content-first lens, in your own words.\n"
            "Outline:\n"
            "1) Central thesis\n2) Key supports & anecdotes\n3) Why it matters\n4) Practical takeaways\n5) One-sentence takeaway\n"
            "No meta phrases or verbatim quotes.\n"
            "</task>\n"
        )
    style_hint = "<format>concise bullet points</format>\n" if style == "bullet" \
                 else "<format>coherent short paragraphs</format>\n"
    lm = {"short": "short", "medium": "medium-length", "long": "detailed"}[length_mode]
    len_hint = f"<length>{lm}</length>\n"
    return f"{task}{style_hint}{len_hint}<text>\n{txt}\n</text>\n<output>\n"

def clean_summary(s: str) -> str:
    bad_patterns = [
        r"(?i)\bplease write\b", r"(?i)\bproduce a .* summary\b",
        r"(?i)\bthank you\b", r"(?i)\bthanks\b", r"(?i)\bapplause\b",
        r"(?i)\btranscriber\b", r"(?i)\breviewer\b",
        r"^•\s*produce\b.*$", r"^-\s*produce\b.*$",
    ]
    for pat in bad_patterns:
        s = re.sub(pat, "", s)
    s = re.sub(r"</?output>|</?task>|</?text>|</?format>|</?length>", "", s, flags=re.I)
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    return "\n".join(lines).strip()

# ================== mT5 Model (Yerel) ==================
LENGTH_PRESETS = {
    "short":  {"chunk_min_new": 25,  "chunk_max_new": 90,  "final_min_new": 60,  "final_max_new": 150},
    "medium": {"chunk_min_new": 50,  "chunk_max_new": 150, "final_min_new": 90,  "final_max_new": 220},
    "long":   {"chunk_min_new": 90,  "chunk_max_new": 260, "final_min_new": 140, "final_max_new": 380},
}

@st.cache_resource(show_spinner=False)
def load_sum_model(model_path: str):
    model_path = ensure_str(model_path)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    mdl = MT5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
    pad_id = tok.pad_token_id or tok.convert_tokens_to_ids("</s>")
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("</s>")
    mdl.config.pad_token_id = pad_id
    mdl.config.eos_token_id = eos_id
    mdl.config.decoder_start_token_id = pad_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device); mdl.eval()
    return tok, mdl, device

def summarize_local(chunks: List[str], model_path: str, style: str, length_mode: str, lang_ui: str) -> str:
    tok, mdl, device = load_sum_model(model_path)

    bad_words_ids: List[List[int]] = []
    try:
        for i in range(100):
            tid = tok.convert_tokens_to_ids(f"<extra_id_{i}>")
            if isinstance(tid, int) and tid >= 0:
                bad_words_ids.append([tid])
    except Exception:
        pass
    try:
        if isinstance(tok.unk_token_id, int) and tok.unk_token_id >= 0:
            bad_words_ids.append([tok.unk_token_id])
    except Exception:
        pass

    pad_id = mdl.config.pad_token_id
    eos_id = mdl.config.eos_token_id
    dec_start_id = mdl.config.decoder_start_token_id or pad_id

    p = LENGTH_PRESETS[length_mode]
    cmin_new, cmax_new = p["chunk_min_new"], p["chunk_max_new"]
    fmin_new, fmax_new = p["final_min_new"], p["final_max_new"]

    def _gen_core(prompt_text: str, min_new: int, max_new: int):
        inputs = tok(prompt_text, return_tensors="pt", truncation=True, max_length=896).to(device)
        gen_kwargs = dict(
            pad_token_id=pad_id, eos_token_id=eos_id, decoder_start_token_id=dec_start_id,
            do_sample=False, num_beams=5, no_repeat_ngram_size=4, repetition_penalty=1.1,
            min_new_tokens=min_new, max_new_tokens=max_new,
            length_penalty=1.05 if length_mode != "short" else 1.0,
            early_stopping=True,
        )
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids
        with torch.inference_mode():
            out = mdl.generate(**inputs, **gen_kwargs)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    def gen(txt: str, min_new: int, max_new: int):
        prompt = build_task_prompt(style, length_mode, lang_ui, txt)
        raw = _gen_core(prompt, min_new, max_new)
        return clean_summary(raw)

    parts = [gen(ensure_str(c), cmin_new, cmax_new) for c in chunks if ensure_str(c).strip()]
    if not parts:
        return ""

    merged = "\n\n".join(parts)
    final = gen(merged, fmin_new, fmax_new)

    if style == "bullet":
        lines = [re.sub(r'^[•\-\*]\s*', '', l).strip() for l in final.splitlines() if l.strip()]
        final = "• " + "\n• ".join(lines[:80])
    return final

# ================== Ses/ASR yardımcıları ==================
def ensure_audio(input_path: str) -> str:
    """Kaynağı tek kanallı 16kHz WAV'a dönüştür (ASR için)."""
    in_path = Path(input_path)
    out_wav = in_path.with_suffix(".16k.wav")
    if out_wav.exists():
        return str(out_wav)
    if not Path(FFMPEG_BIN).exists():
        raise FileNotFoundError(f"FFmpeg bulunamadı: {FFMPEG_BIN}")
    (
        ffmpeg
        .input(str(in_path))
        .output(str(out_wav), ac=1, ar=16000, vn=None, loglevel="error", threads=0)
        .overwrite_output()
        .run(cmd=FFMPEG_BIN)
    )
    return str(out_wav)

def load_waveform_dict(wav_path: str) -> Dict[str, Any]:
    """pyannote'a TorchCodec'siz giriş: {'waveform': (ch, time) float32 Tensor, 'sample_rate': int}"""
    data, sr = sf.read(wav_path, dtype="float32")  # (time,) veya (time, ch)
    if data.ndim == 1:
        data = np.expand_dims(data, 0)             # (1, time)
    else:
        data = data.T                               # (ch, time)
    return {"waveform": torch.from_numpy(data), "sample_rate": int(sr)}

def write_srt(segments: List[Dict], srt_path: str):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            spk = seg.get("speaker")
            text_line = seg['text'].strip() if seg['text'] else ""
            if spk: text_line = f"{spk}: {text_line}"
            f.write(f"{i}\n")
            f.write(f"{human_ts(seg['start'], srt=True)} --> {human_ts(seg['end'], srt=True)}\n")
            f.write(text_line + "\n\n")

def write_vtt(segments: List[Dict], vtt_path: str):
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            spk = seg.get("speaker")
            text_line = seg['text'].strip() if seg['text'] else ""
            if spk: text_line = f"{spk}: {text_line}"
            f.write(f"{human_ts(seg['start'])}.000 --> {human_ts(seg['end'])}.000\n")
            f.write(text_line + "\n\n")

# ================== Diarization yardımcıları ==================
def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    left = max(a_start, b_start); right = min(a_end, b_end)
    return max(0.0, right - left)

def assign_speakers_to_segments(asr_segments: List[Dict], diar_segments: List[Dict]) -> List[Dict]:
    if not asr_segments or not diar_segments:
        return asr_segments
    ds = [
        {"start": float(d.get("start", 0.0)),
         "end": float(d.get("end", 0.0)),
         "speaker": str(d.get("speaker", "") or "")}
        for d in diar_segments
        if float(d.get("end", 0.0)) > float(d.get("start", 0.0))
    ]
    ds.sort(key=lambda x: (x["speaker"], x["start"], x["end"]))
    merged = []
    for d in ds:
        if not merged or d["speaker"] != merged[-1]["speaker"] or d["start"] > merged[-1]["end"] + 0.05:
            merged.append(d.copy())
        else:
            merged[-1]["end"] = max(merged[-1]["end"], d["end"])
    ds = merged

    out = []
    for seg in asr_segments:
        s0 = float(seg.get("start", 0.0)); s1 = float(seg.get("end", 0.0))
        if s1 <= s0:
            out.append(seg); continue
        best_spk, best_ov = None, 0.0
        for d in ds:
            if d["end"] < s0: continue
            if d["start"] > s1: break
            ov = _interval_overlap(s0, s1, d["start"], d["end"])
            if ov > best_ov: best_ov, best_spk = ov, d["speaker"]
        new_rec = dict(seg)
        new_rec["speaker"] = best_spk if best_ov >= 0.15 else None
        out.append(new_rec)
    return out

def write_rttm(diar_segments: List[Dict], rttm_path: str, file_id: Optional[str] = None):
    if not diar_segments: return
    if file_id is None: file_id = "audio"
    lines = []
    for d in diar_segments:
        stt = float(d.get("start", 0.0)); end = float(d.get("end", 0.0))
        if end <= stt: continue
        dur = end - stt
        spk = str(d.get("speaker", "SPK")).replace(" ", "_")
        lines.append(f"SPEAKER {file_id} 1 {stt:.3f} {dur:.3f} <NA> <NA> <NA> {spk} <NA>")
    with open(rttm_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def render_video_with_clickable_segments(video_path: str, segments: List[Dict], max_rows: int = 300, height_px: int = 560):
    try:
        fsize = os.path.getsize(video_path)
        if fsize > 120 * 1024 * 1024:
            st.warning("Video 120 MB’dan büyük. Bu boyutta videoyu sayfaya gömmek ağır olabilir.")
            return
    except Exception:
        pass
    mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
    try:
        with open(video_path, "rb") as vf:
            b64 = base64.b64encode(vf.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Video gömme sırasında hata: {e}")
        return

    rows = []
    for i, s in enumerate(segments[:max_rows], 1):
        stt = float(s.get("start", 0.0)); end = float(s.get("end", 0.0))
        txt = html.escape((s.get("text") or "").strip())
        spk = s.get("speaker")
        label = f"{human_ts(stt)} → {human_ts(end)}"
        if spk:
            txt = f"<strong>{html.escape(spk)}:</strong> " + txt
        rows.append(
            f"""
            <tr>
              <td class="idx">{i}</td>
              <td class="timecell">
                <button class="tsbtn" onclick="seekTo({stt})" title="Bu zamana git">▶</button>
                <a class="tslink" href="javascript:void(0)" onclick="seekTo({stt})">{label}</a>
              </td>
              <td class="textcell">{txt}</td>
            </tr>
            """
        )

    html_doc = f"""
    <style>
      .tswrap {{ color: #fff; }}
      .tswrap table, .tswrap th, .tswrap td {{ border-color: rgba(255,255,255,0.12); }}
      .tswrap thead {{ background: rgba(255,255,255,0.06); color: #fff; }}
      .tswrap tbody tr:nth-child(odd) {{ background: rgba(255,255,255,0.03); }}
      .tswrap tbody tr:nth-child(even) {{ background: rgba(255,255,255,0.015); }}
      .tswrap .idx {{ white-space: nowrap; padding: 8px; }}
      .tswrap .timecell {{ white-space: nowrap; padding: 8px; }}
      .tswrap .textcell {{ word-break: break-word; padding: 8px; }}
      .tslink {{ color: #fff; text-decoration: none; font-weight: 600; margin-left: 6px; }}
      .tslink:hover {{ text-decoration: underline; }}
      .tsbtn {{
        color: #fff; background: transparent; border: 1px solid rgba(255,255,255,0.6);
        border-radius: 6px; padding: 2px 6px; cursor: pointer;
      }}
      .tsbtn:hover {{ border-color: #fff; }}
    </style>

    <div class="tswrap" style="display:grid; grid-template-columns: 1fr; gap: 12px;">
      <video id="trans_vid" controls preload="metadata" style="width:100%; border-radius:10px; outline:none;"
             src="data:{mime};base64,{b64}">
        Tarayıcınız video oynatmayı desteklemiyor.
      </video>

      <div style="max-height:{height_px}px; overflow:auto; border:1px solid rgba(255,255,255,0.12); border-radius:8px;">
        <table style="width:100%; border-collapse:collapse; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial;">
          <thead style="position:sticky; top:0;">
            <tr>
              <th style="text-align:left; padding:8px;">#</th>
              <th style="text-align:left; padding:8px;">Zaman</th>
              <th style="text-align:left; padding:8px;">Metin</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </div>

    <script>
      function seekTo(sec) {{
        const v = document.getElementById("trans_vid");
        if (v) {{
          if (v.readyState < 1) {{
            v.addEventListener('loadedmetadata', () => {{ v.currentTime = sec; v.play(); }}, {{once:true}});
            v.load();
          }} else {{
            v.currentTime = sec;
            v.play();
          }}
        }}
      }}
    </script>
    """
    components.html(html_doc, height=height_px + 420, scrolling=False)

# ================== Model/ASR yardımcıları ==================
def pick_device_and_compute():
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type

@st.cache_resource(show_spinner=False)
def load_model_cached(model_size: str, device_hint: str, compute_type: str):
    cpu_threads = os.cpu_count() or 4
    num_workers = 1 if device_hint == "cuda" else 2
    return WhisperModel(
        model_size,
        device=device_hint,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )

# ================== UI: Sidebar ==================
st.sidebar.title("⚙️ Ayarlar")

# ASR
model_size = st.sidebar.selectbox(
    "ASR Model",
    ["large-v3", "medium", "small", "turbo"],
    index=0,
    help="Kalite/hız dengesi: large-v3 en kaliteli. CPU’da medium/small daha hızlı."
)
language_asr = st.sidebar.selectbox(
    "ASR Dil",
    ["auto", "tr", "en"],
    index=0,
    help="auto: otomatik algıla (TR+EN karışıkta önerilir)."
)
beam_size = st.sidebar.slider("Beam size", 1, 10, 5, help="5–10 kaliteyi artırır, biraz yavaşlatır.")
vad_min_silence = st.sidebar.slider("VAD min. sessizlik (ms)", 100, 1000, 300, 50)
word_timestamps = st.sidebar.checkbox("Kelime zaman bilgisi (CSV üret)", value=True)

st.sidebar.divider()

# Diarization
enable_diar = st.sidebar.checkbox("Konuşmacı ayrımı (pyannote)", value=False,
                                  help="pyannote.audio kurulu olmalı ve HF token gerekli.")
hf_token = st.sidebar.text_input("Hugging Face Token", value=os.environ.get("HF_TOKEN", ""),
                                 type="password",
                                 help="https://huggingface.co/settings/tokens")

st.sidebar.caption(f"TorchCodec durumu: {'✅ Etkin' if TORCHCODEC_OK else '⚠️ Yok — RAM fallback kullanılacak'}")

st.sidebar.divider()

# Özetleme
st.sidebar.subheader("📝 Özet Ayarları (Yerel mT5)")
DEFAULT_MODEL_PATH = r"C:\Users\akiziltunc\Desktop\video_ts\yt-sum-mt5-merged4"
sum_model_path = st.sidebar.text_input("Model klasörü (mT5 LoRA merged)", DEFAULT_MODEL_PATH)
sum_lang = st.sidebar.selectbox("Özet dili", ["english", "turkish"], index=0)
sum_style = st.sidebar.selectbox("Biçim", ["paragraphs", "bullet"], index=0)
length_mode_label = st.sidebar.selectbox("Özet türü", ["Kısa", "Orta", "Detaylı"], index=1)
length_mode = {"Kısa": "short", "Orta": "medium", "Detaylı": "long"}[length_mode_label]
show_preface = st.sidebar.checkbox("Ön açıklama göster", value=True)

# Video + clickable timestamps
preview_video_clickseek = st.sidebar.checkbox(
    "🎬 Video önizleme + tıklanabilir timestamp",
    value=True,
    help="Segmentlere tıklayınca video ilgili zamana atlar."
)

out_root = st.sidebar.text_input("Çıktı klasörü", value=st.session_state.out_root)
st.sidebar.caption("Çıktılar TXT, SRT, VTT, JSONL, kelime zamanları CSV, (varsa) RTTM ve ÖZET olarak kaydedilir.")
st.session_state.out_root = out_root  # state'e güncel kökü yaz

# ================== UI: Ana alan ==================
st.title("🎙️ Transcriber + 🗣️ Speaker Diarization + 📝 Summarizer (TR + EN)")
st.write("Aşağıdan **dosya yükleyin** veya **dosya yolu** girin. Ardından **Transcribe** tuşuna basın. Transcribe tamamlanınca **Summarize** tuşuyla özet alın.")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Dosya yükle (mp4, mkv, mov, mp3, wav...)", type=["mp4","mkv","mov","mp3","wav","m4a","aac","flac"], key="uploader")
with col2:
    file_path = st.text_input("Veya yerel/network path girin", value=st.session_state.in_path or "", placeholder=r"C:\klasor\video.mp4 / /home/user/audio.wav", key="file_path_text")

# Dosya seçiminde hemen transcribe yapmıyoruz; sadece temp'e kopyalayıp state'e yazarız
if uploaded is not None:
    tmp_dir = Path(st.session_state.out_root) / "_uploaded_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state.uploaded_temp = str(tmp_path)
    st.session_state.src_name = uploaded.name
elif file_path.strip():
    st.session_state.uploaded_temp = ""   # file_path kullanacağız
    st.session_state.src_name = Path(file_path).name

# Butonlar
run_transcribe = st.button("🚀 Transcribe", type="primary", key="btn_transcribe")
run_summarize  = st.button("📝 Summarize", key="btn_summarize")

# ================== İş mantığı (BUTON TETİKLİ) ==================
def do_transcribe():
    ss = st.session_state
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(ss.out_root) / f"job_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Girdi seçimi
    if ss.uploaded_temp:
        in_path = ss.uploaded_temp
    elif ss.file_path_text.strip():
        if not Path(ss.file_path_text).exists():
            st.error(f"Dosya bulunamadı: {ss.file_path_text}")
            return
        in_path = ss.file_path_text
    else:
        st.error("Lütfen bir dosya yükleyin veya geçerli bir path girin.")
        return

    ss.in_path = str(in_path)
    ss.out_dir = str(out_dir)
    ss.base_name = Path(in_path).stem

    st.info(f"Kaynak: {ss.in_path}")

    # Cihaz/compute
    device_hint, compute_type = pick_device_and_compute()
    st.write(f"🧠 ASR Model: `{model_size}` | 🎛️ Cihaz: `{device_hint}` | ⌗ compute_type: `{compute_type}`")

    # ASR modeli
    with st.spinner("ASR modeli yükleniyor... (ilk seferde indirilebilir)"):
        model = load_model_cached(model_size, device_hint, compute_type)

    # Ses dosyası (16kHz mono) — ASR için
    with st.spinner("Ses hazırlanıyor (16kHz mono WAV)..."):
        wav_path = ensure_audio(ss.in_path)

    # (Opsiyonel) Diarization
    diar_spk_segments: List[Dict] = []
    diar_warn = None
    if enable_diar:
        global HAS_PYANNOTE
        if not HAS_PYANNOTE:
            try:
                from pyannote.audio import Pipeline
                HAS_PYANNOTE = True
            except Exception:
                HAS_PYANNOTE = False

        if not HAS_PYANNOTE:
            diar_warn = "pyannote.audio import edilemedi. `pip install pyannote.audio` ile kurun."
        elif not hf_token:
            diar_warn = "Hugging Face token gerekli. Sidebar'a token girin."
        else:
            try:
                with st.spinner("Konuşmacı ayrımı modeli yükleniyor..."):
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                    pipeline.to(device_hint)

                with st.spinner("Konuşmacı ayrımı çalışıyor..."):
                    if TORCHCODEC_OK:
                        diarization = pipeline(audio=ss.in_path)
                    else:
                        audio_dict = load_waveform_dict(wav_path)
                        diarization = pipeline(audio=audio_dict)

                    for seg, _, label in diarization.itertracks(yield_label=True):
                        diar_spk_segments.append({
                            "speaker": label,
                            "start": float(seg.start),
                            "end": float(seg.end),
                        })
            except Exception as e:
                diar_warn = f"Konuşmacı ayrımı sırasında hata: {e}"

    if diar_warn:
        st.warning(diar_warn)

    # ASR (faster-whisper)
    with st.spinner("Transcribe çalışıyor..."):
        segments_gen, info = model.transcribe(
            wav_path,
            language=None if language_asr == "auto" else language_asr,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=int(vad_min_silence)),
            beam_size=int(beam_size),
            temperature=0.0,
            word_timestamps=bool(word_timestamps),
            condition_on_previous_text=False,
            chunk_length=15,  # saniye
        )

        seg_list = []
        words: List[Dict] = []
        for seg in segments_gen:
            rec = {"start": float(seg.start), "end": float(seg.end), "text": (seg.text or "").strip()}
            seg_list.append(rec)
            if word_timestamps and getattr(seg, "words", None):
                for w in seg.words:
                    words.append({"start": round(w.start,3), "end": round(w.end,3), "word": (w.word or "").strip()})

    # Diarization sonucu varsa konuşmacı ata
    if enable_diar and diar_spk_segments:
        seg_list = assign_speakers_to_segments(seg_list, diar_spk_segments)

    # Çıktı yolları
    base_name = ss.base_name
    out_dir = Path(ss.out_dir)
    txt_path   = str(out_dir / f"{base_name}.transcript.txt")
    srt_path   = str(out_dir / f"{base_name}.srt")
    vtt_path   = str(out_dir / f"{base_name}.vtt")
    jsonl_path = str(out_dir / f"{base_name}.segments.jsonl")
    words_csv  = str(out_dir / f"{base_name}.words.csv")
    rttm_path  = str(out_dir / f"{base_name}.rttm")
    summary_path = str(out_dir / f"{base_name}.summary.txt")

    # Kayıtlar: TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        lines = []
        for s in seg_list:
            spk = s.get("speaker")
            if spk:
                lines.append(f"[{spk}] {s['text']}")
            else:
                lines.append(s["text"])
        f.write("\n".join(lines) + "\n")
    transcript_text = "\n".join([ensure_str(s["text"]) for s in seg_list])

    # JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in seg_list:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # SRT / VTT
    write_srt(seg_list, srt_path)
    write_vtt(seg_list, vtt_path)

    # Kelime zamanları CSV
    if word_timestamps and len(words) > 0:
        pd.DataFrame(words).to_csv(words_csv, index=False, encoding="utf-8")

    # RTTM
    if enable_diar and diar_spk_segments:
        write_rttm(diar_spk_segments, rttm_path, file_id=base_name)

    # ---- State'e yaz ----
    ss.seg_list = seg_list
    ss.words = words
    ss.diar_segments = diar_spk_segments
    ss.transcript_text = transcript_text
    ss.asr_done = True
    ss.summary_text = ""  # yeni transcribe sonrası eski özet silinsin
    ss.paths = {
        "txt": txt_path,
        "srt": srt_path,
        "vtt": vtt_path,
        "jsonl": jsonl_path,
        "words_csv": words_csv if (word_timestamps and len(words) > 0) else "",
        "rttm": rttm_path if (enable_diar and diar_spk_segments) else "",
        "summary": summary_path,
    }

def do_summarize():
    ss = st.session_state
    if not ss.asr_done or not ss.transcript_text.strip():
        st.warning("Önce Transcribe çalıştırın.")
        return
    tx = collapse_repeats(ss.transcript_text)
    chunks = chunk_text_mode(tx, length_mode)
    if show_preface:
        if sum_lang == "turkish":
            st.info(f"Seçtiğiniz metnin **{length_mode_label.lower()}** özetini üretiyorum…")
        else:
            st.info(f"Producing a **{length_mode}** summary for this transcript…")
    with st.spinner("Yerel mT5 modeli ile özetleniyor…"):
        try:
            summary = summarize_local(
                chunks=chunks,
                model_path=sum_model_path,
                style=sum_style,
                length_mode=length_mode,
                lang_ui=sum_lang
            )
        except Exception as e:
            st.error(f"Özetleme sırasında hata: {e}")
            return
    summary = summary.strip()
    ss.summary_text = summary
    # Dosyaya yaz
    if ss.paths.get("summary"):
        with open(ss.paths["summary"], "w", encoding="utf-8") as f:
            f.write(summary)

# ---- Butonlara göre çalıştır ----
if run_transcribe:
    do_transcribe()

if run_summarize:
    do_summarize()

# ================== Görselleştirme: state'i göster ==================
ss = st.session_state

# Kaynak bilgisi
if ss.in_path:
    st.info(f"Kaynak: {ss.in_path}")

# Transcribe tamamlandıysa çıktıları göster
if ss.asr_done:
    st.success("✅ Transcribe tamam!")

    # Video + tıklanabilir timestamp
    if preview_video_clickseek:
        st.subheader("🎬 Video + Tıklanabilir Timestamper")
        st.caption("Listeden bir zamana tıkladığınızda video o noktaya atlar.")
        try:
            render_video_with_clickable_segments(ss.in_path, ss.seg_list, max_rows=300, height_px=520)
        except Exception as e:
            st.warning(f"Video önizlemede sorun oluştu: {e}")

    # Tüm segmentler (tam liste)
    with st.expander("🧾 Tüm segmentler (tam liste)"):
        df_all = pd.DataFrame(ss.seg_list)
        st.dataframe(df_all, use_container_width=True)

    with st.expander("Tam metin (TXT)"):
        try:
            with open(ss.paths.get("txt",""), "r", encoding="utf-8") as f:
                st.text(f.read()[:15000])
        except Exception:
            if ss.transcript_text:
                st.text(ss.transcript_text[:15000])

    # Özet alanı
    st.subheader("📝 Özet (Yerel mT5)")
    if ss.summary_text:
        st.success(ss.summary_text)
    else:
        st.info("Henüz özet üretilmedi. **Summarize** butonuna basın.")

    # İndir butonları
    st.subheader("📥 Çıktıları indir")
    cols = st.columns(6)
    with cols[0]:
        p = ss.paths.get("txt","")
        if p and Path(p).exists():
            with open(p, "rb") as f:
                st.download_button("TXT indir", f.read(), file_name=Path(p).name, mime="text/plain")
    with cols[1]:
        p = ss.paths.get("srt","")
        if p and Path(p).exists():
            with open(p, "rb") as f:
                st.download_button("SRT indir", f.read(), file_name=Path(p).name, mime="application/x-subrip")
    with cols[2]:
        p = ss.paths.get("vtt","")
        if p and Path(p).exists():
            with open(p, "rb") as f:
                st.download_button("VTT indir", f.read(), file_name=Path(p).name, mime="text/vtt")
    with cols[3]:
        p = ss.paths.get("jsonl","")
        if p and Path(p).exists():
            with open(p, "rb") as f:
                st.download_button("SEG (JSONL) indir", f.read(), file_name=Path(p).name, mime="application/jsonl")
    with cols[4]:
        p = ss.paths.get("words_csv","")
        if p and p and Path(p).exists():
            with open(p, "rb") as f:
                st.download_button("Kelime CSV indir", f.read(), file_name=Path(p).name, mime="text/csv")
    with cols[5]:
        p = ss.paths.get("summary","")
        if p and Path(p).exists():
            with open(p, "rb") as f:
                st.download_button("Özet indir", f.read(), file_name=Path(p).name, mime="text/plain")

    if ss.paths.get("rttm") and Path(ss.paths["rttm"]).exists():
        st.download_button("RTTM indir", Path(ss.paths["rttm"]).read_bytes(),
                           file_name=Path(ss.paths["rttm"]).name, mime="text/plain")

    st.info(f"📂 Çıktılar klasörü: {ss.out_dir}")

else:
    st.caption("Transcribe henüz çalışmadı veya sonuç yok. Dosya seçip **Transcribe** tuşuna basın.")
