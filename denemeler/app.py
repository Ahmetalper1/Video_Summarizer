# -*- coding: utf-8 -*-
"""
Transcriber + (Opsiyonel) Diarization + Yerel mT5 Özetleme (TR + EN)
- faster-whisper ile ASR
- pyannote (opsiyonel) konuşmacı ayrımı
- mT5 (LoRA-merge) yerel modelle içerik-odaklı özet (madde/paragraf, kısa/orta/detaylı)
"""

import os, io, sys, json, csv, time, re
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Any, Tuple
from datetime import datetime
import warnings
import torch
import streamlit as st
import pandas as pd
import ffmpeg

from faster_whisper import WhisperModel
import imageio_ffmpeg

# -------------------- FFmpeg yolunu sabitle --------------------
os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\Users\akiziltunc\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

_ffmpeg_dir = str(Path(FFMPEG_BIN).parent)
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# -------------------- (Opsiyonel) pyannote --------------------
HAS_PYANNOTE = False
try:
    from pyannote.audio import Pipeline
    HAS_PYANNOTE = True
except Exception:
    HAS_PYANNOTE = False

# -------------------- mT5 / Transformers --------------------
import sentencepiece  # tokenizer için
from transformers import AutoTokenizer, MT5ForConditionalGeneration

# ================== Streamlit ==================
st.set_page_config(page_title="Transcriber + Diarization + Özet", layout="wide")

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
# NLTK cümle bölücü
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

    # <extra_id_*> ve <unk> ban (boşsa HF bug'ını tetiklememek için hiç göndermiyoruz)
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
        if bad_words_ids:  # sadece doluysa ekle
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
    """
    Kaynağı tek kanallı 16kHz WAV'a dönüştür.
    """
    in_path = Path(input_path)
    out_wav = in_path.with_suffix(".16k.wav")
    if out_wav.exists():
        return str(out_wav)
    if not Path(FFMPEG_BIN).exists():
        raise FileNotFoundError(f"FFmpeg bulunamadı: {FFMPEG_BIN}")
    (
        ffmpeg
        .input(str(in_path))
        .output(str(out_wav), ac=1, ar=16000, vn=None, loglevel="error")
        .overwrite_output()
        .run(cmd=FFMPEG_BIN)
    )
    return str(out_wav)

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

def pick_device_and_compute():
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type

@st.cache_resource(show_spinner=False)
def load_model_cached(model_size: str, device_hint: str, compute_type: str):
    return WhisperModel(model_size, device=device_hint, compute_type=compute_type)

def overlap(a_start, a_end, b_start, b_end) -> float:
    start = max(a_start, b_start); end = min(a_end, b_end)
    return max(0.0, end - start)

def assign_speakers_to_segments(asr_segments: List[Dict], spk_segments: List[Dict]) -> List[Dict]:
    if not spk_segments:
        return asr_segments
    for seg in asr_segments:
        scores = {}
        for spk in spk_segments:
            o = overlap(seg["start"], seg["end"], spk["start"], spk["end"])
            if o > 0:
                scores[spk["speaker"]] = scores.get(spk["speaker"], 0.0) + o
        if scores:
            seg["speaker"] = max(scores.items(), key=lambda x: x[1])[0]
        else:
            seg["speaker"] = "SPEAKER_??"
    return asr_segments

def write_rttm(spk_segments: List[Dict], rttm_path: str, file_id: str = "audio"):
    with open(rttm_path, "w", encoding="utf-8") as f:
        for s in spk_segments:
            dur = max(0.0, s["end"] - s["start"])
            f.write(f"SPEAKER {file_id} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n")

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

st.sidebar.divider()

# Özetleme
st.sidebar.subheader("📝 Özet Ayarları (Yerel mT5)")
DEFAULT_MODEL_PATH = r"C:\Users\akiziltunc\Desktop\video_ts\yt-sum-mt5-merged5"
sum_model_path = st.sidebar.text_input("Model klasörü (mT5 LoRA merged)", DEFAULT_MODEL_PATH)
sum_lang = st.sidebar.selectbox("Özet dili", ["english", "turkish"], index=0)
sum_style = st.sidebar.selectbox("Biçim", ["bullet", "paragraphs"], index=0)
length_mode_label = st.sidebar.selectbox("Özet türü", ["Kısa", "Orta", "Detaylı"], index=1)
length_mode = {"Kısa": "short", "Orta": "medium", "Detaylı": "long"}[length_mode_label]
show_preface = st.sidebar.checkbox("Ön açıklama göster", value=True)

out_root = st.sidebar.text_input("Çıktı klasörü", value=str(Path.cwd() / "outputs"))
st.sidebar.caption("Çıktılar TXT, SRT, VTT, JSONL, kelime zamanları CSV, (varsa) RTTM ve ÖZET olarak kaydedilir.")

# ================== UI: Ana alan ==================
st.title("🎙️ Transcriber + 🗣️ Speaker Diarization + 📝 Summarizer (TR + EN)")
st.write("Aşağıdan **dosya yükleyin** veya **dosya yolu** girin. Ardından **Transcribe** tuşuna basın. Transcribe tamamlanınca özet’i oluşturabilirsiniz.")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Dosya yükle (mp4, mkv, mov, mp3, wav...)", type=["mp4","mkv","mov","mp3","wav","m4a","aac","flac"])
with col2:
    file_path = st.text_input("Veya yerel/network path girin", value="", placeholder=r"C:\klasor\video.mp4 / /home/user/audio.wav")

run_btn = st.button("🚀 Transcribe")

# Bu değişkenleri özet aşamasında kullanacağız
seg_list: List[Dict] = []
transcript_text: str = ""
out_dir: Path = None  # type: ignore
in_path: str = ""

# ================== İş mantığı ==================
if run_btn:
    if uploaded is None and not file_path.strip():
        st.error("Lütfen bir dosya yükleyin veya geçerli bir path girin.")
        st.stop()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) / f"job_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Girdi
    if uploaded is not None:
        src_name = uploaded.name
        src_path = out_dir / src_name
        with open(src_path, "wb") as f:
            f.write(uploaded.getbuffer())
        in_path = str(src_path)
    else:
        if not Path(file_path).exists():
            st.error(f"Dosya bulunamadı: {file_path}")
            st.stop()
        in_path = file_path

    st.info(f"Kaynak: {in_path}")

    # Cihaz/compute
    device_hint, compute_type = pick_device_and_compute()
    st.write(f"🧠 ASR Model: `{model_size}` | 🎛️ Cihaz: `{device_hint}` | ⌗ compute_type: `{compute_type}`")

    # ASR modeli
    with st.spinner("ASR modeli yükleniyor... (ilk seferde indirilebilir)"):
        model = load_model_cached(model_size, device_hint, compute_type)

    # Ses dosyası (16kHz mono)
    with st.spinner("Ses hazırlanıyor (16kHz mono WAV)..."):
        wav_path = ensure_audio(in_path)

    # (Opsiyonel) Diarization
    diar_spk_segments: List[Dict] = []
    diar_warn = None
    if enable_diar:
        if not HAS_PYANNOTE:
            diar_warn = "pyannote.audio kurulu değil. `pip install pyannote.audio` ile kurun."
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
                    diarization = pipeline(wav_path)
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
            word_timestamps=True
        )

        seg_list = []
        words: List[Dict] = []
        for seg in segments_gen:
            rec = {"start": float(seg.start), "end": float(seg.end), "text": (seg.text or "").strip()}
            seg_list.append(rec)
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append({"start": round(w.start,3), "end": round(w.end,3), "word": (w.word or "").strip()})

    # Diarization sonucu varsa konuşmacı ata
    if enable_diar and diar_spk_segments:
        seg_list = assign_speakers_to_segments(seg_list, diar_spk_segments)

    # Çıktı yolları
    base_name = Path(in_path).stem
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
    # Ayrıca düz transcript_text oluştur (özet için)
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
        write_rttm(diar_spk_segments, rttm_path, file_id=Path(in_path).stem)

    st.success("✅ Transcribe tamam!")

    # Önizleme
    with st.expander("Önizleme (ilk 10 segment)"):
        df_prev = pd.DataFrame(seg_list[:10])
        st.dataframe(df_prev, use_container_width=True)

    with st.expander("Tam metin (TXT)"):
        with open(txt_path, "r", encoding="utf-8") as f:
            st.text(f.read()[:15000])

    # ============ ÖZETLEME (Yerel mT5) ============
    st.subheader("📝 Özet (Yerel mT5)")
    if show_preface:
        if sum_lang == "turkish":
            st.info(f"Tamamdır, seçtiğiniz metnin **{length_mode_label.lower()}** özetini çıkarıyorum…")
        else:
            st.info(f"Great! I'll produce a **{length_mode}** summary for this transcript…")

    if not transcript_text.strip():
        st.warning("Transcript boş görünüyor. Önce transcribe çalıştırın.")
    else:
        with st.spinner("Yerel mT5 modeli ile özetleniyor…"):
            tx = collapse_repeats(transcript_text)
            chunks = chunk_text_mode(tx, length_mode)
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
                summary = ""

        if summary.strip():
            st.success(summary)
            # dosyaya yaz
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
        else:
            st.warning("Özet üretilemedi.")

    # İndir butonları
    st.subheader("📥 Çıktıları indir")
    cols = st.columns(6)
    with cols[0]:
        with open(txt_path, "rb") as f:
            st.download_button("TXT indir", f.read(), file_name=Path(txt_path).name, mime="text/plain")
    with cols[1]:
        with open(srt_path, "rb") as f:
            st.download_button("SRT indir", f.read(), file_name=Path(srt_path).name, mime="application/x-subrip")
    with cols[2]:
        with open(vtt_path, "rb") as f:
            st.download_button("VTT indir", f.read(), file_name=Path(vtt_path).name, mime="text/vtt")
    with cols[3]:
        with open(jsonl_path, "rb") as f:
            st.download_button("SEG (JSONL) indir", f.read(), file_name=Path(jsonl_path).name, mime="application/jsonl")
    with cols[4]:
        if Path(words_csv).exists():
            with open(words_csv, "rb") as f:
                st.download_button("Kelime CSV indir", f.read(), file_name=Path(words_csv).name, mime="text/csv")
    with cols[5]:
        if Path(summary_path).exists():
            with open(summary_path, "rb") as f:
                st.download_button("Özet indir", f.read(), file_name=Path(summary_path).name, mime="text/plain")

    if enable_diar and Path(rttm_path).exists():
        st.download_button("RTTM indir", Path(rttm_path).read_bytes(), file_name=Path(rttm_path).name, mime="text/plain")

    st.info(f"📂 Çıktılar klasörü: {out_dir}")
