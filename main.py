import streamlit as st
import io
import os
import re
import tempfile
from typing import Optional, Tuple

from PIL import Image
import speech_recognition as sr
from gtts import gTTS
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import torch


STORY_WORD_RANGE = (50, 60)
BLIP_MODEL = "Salesforce/blip-image-captioning-large"
FLAN_MODEL = "google/flan-t5-base"


# ── Model loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading caption model (first run only)...")
def _load_blip():
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="Loading story model (first run only)...")
def _load_flan():
    # Load T5 tokenizer and model directly — avoids pipeline task-name issues
    # across different transformers versions
    tokenizer = T5Tokenizer.from_pretrained(FLAN_MODEL, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(FLAN_MODEL)
    model.eval()
    return tokenizer, model


# ── Caption generation ────────────────────────────────────────────────────────

def _generate_caption(image_path: str) -> str:
    """Generate an accurate image caption using BLIP-large."""
    processor, model = _load_blip()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, "a detailed photo of", return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=5,
            early_stopping=True,
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    caption = re.sub(r"[.!?]+$", "", caption.strip())
    return caption


# ── Story generation ──────────────────────────────────────────────────────────

def _generate_story(
    caption: str,
    keywords: str,
    word_range: Tuple[int, int] = STORY_WORD_RANGE,
) -> str:
    """Generate a vivid short story using Flan-T5-base."""
    flan = _load_flan()   # returns (tokenizer, model) tuple
    min_w, max_w = word_range

    kw_line = f" Include these themes: {keywords}." if keywords.strip() else ""

    prompt = (
        f"Write a creative short story in exactly {min_w} to {max_w} words "
        f"based on this image description: {caption}.{kw_line} "
        "Use third person present tense. Be vivid and specific. "
        "Output only the story, no title or extra text."
    )

    try:
        tokenizer, model = flan
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True,
        ).input_ids
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=120,
                min_new_tokens=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        story = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    except Exception:
        story = ""

    if not story:
        story = _fallback_story(caption, keywords)

    return _enforce_word_range(story, min_w, max_w)


def _fallback_story(caption: str, keywords: str) -> str:
    """Rule-based fallback if model output is empty."""
    base = f"{caption}." if caption else "The scene unfolds quietly."
    extras = [
        "Every detail tells its own story.",
        "The moment is alive with energy and presence.",
        "Time seems to slow as the scene captures something real.",
        "There is beauty in the simplicity of the moment.",
        "It is a snapshot of life, vivid and true.",
    ]
    story = base
    for e in extras:
        story += " " + e
        if len(story.split()) >= 50:
            break
    if keywords.strip():
        story += f" The {keywords} adds depth to the scene."
    return story


# ── Word-count enforcement ────────────────────────────────────────────────────

def _enforce_word_range(text: str, min_words: int, max_words: int) -> str:
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]

    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        if not truncated.endswith((".", "!", "?")):
            truncated = (
                (truncated.rsplit(".", 1)[0] + ".") if "." in truncated else truncated + "."
            )
        return truncated.strip()

    if len(words) < min_words:
        padding = [
            " The moment is captured perfectly in time.",
            " Every detail reflects something real and vivid.",
            " The scene is alive with quiet energy.",
        ]
        out = text.strip()
        idx = 0
        while len([w for w in re.split(r"\s+", out) if w]) < min_words:
            out += padding[idx % len(padding)]
            idx += 1
        words = [w for w in re.split(r"\s+", out) if w]
        return " ".join(words[:max_words])

    return " ".join(words)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _normalize_keywords(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip()


def _combine_keywords(text_keywords: str, spoken_keywords: str) -> str:
    parts = [p.strip() for p in [text_keywords, spoken_keywords] if p.strip()]
    return " | ".join(parts)


def _transcribe_audio_to_text(audio_file) -> Tuple[str, Optional[str]]:
    if audio_file is None:
        return "", None

    # Support both UploadedFile and BytesIO objects
    try:
        audio_bytes = audio_file.getvalue()
    except AttributeError:
        try:
            audio_bytes = audio_file.read()
        except Exception:
            return "", "Could not read the recorded audio."
    except Exception:
        return "", "Could not read the recorded audio."

    if not audio_bytes:
        return "", "Recorded audio was empty."

    recognizer = sr.Recognizer()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            tmp_path = f.name
            f.write(audio_bytes)
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return _normalize_keywords(text), None
    except sr.UnknownValueError:
        return "", "Could not understand the audio — please speak clearly."
    except sr.RequestError as e:
        return "", f"Speech-to-text request failed: {e}"
    except Exception as e:
        return "", f"Speech-to-text failed: {e}"
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _story_to_speech_bytes(story: str) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        tts = gTTS(text=story, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)          # rewind so st.audio can read from the start
        return buf.getvalue(), None
    except Exception as e:
        return None, f"Text-to-speech failed: {e}"


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Image Caption & Story Generator",
        page_icon="🖼️",
        layout="centered",
    )

    st.title("🖼️ Image Caption & Story Generator")
    st.write(
        "Upload an image, optionally add keywords (text or microphone), "
        "then generate a caption, a short story, and listen to it."
    )

    # ── Image upload ──────────────────────────────────────────────────────────
    uploaded_image = st.file_uploader(
        "Choose an image…", type=["jpg", "jpeg", "png", "webp"]
    )

    # ── Keyword inputs ────────────────────────────────────────────────────────
    text_keywords = st.text_input(
        "Optional keywords (text)",
        value="",
        placeholder="e.g. playful, sunny, garden",
        help="These themes will be woven into the story.",
    )

    spoken_keywords = ""
    stt_error: Optional[str] = None

    # st.audio_input was introduced in Streamlit 1.38; graceful fallback
    try:
        audio = st.audio_input("Optional keywords (speak into microphone)")
        if audio is not None:
            spoken_keywords, stt_error = _transcribe_audio_to_text(audio)
    except AttributeError:
        st.info("Microphone input not available in this Streamlit version. Upload a .wav file instead.")
        wav_file = st.file_uploader(
            "Optional keywords (upload audio .wav)", type=["wav"], key="wav_fallback"
        )
        if wav_file is not None:
            spoken_keywords, stt_error = _transcribe_audio_to_text(wav_file)

    if stt_error:
        st.warning(f"⚠️ {stt_error}")
    if spoken_keywords:
        st.caption(f"🎙️ Detected speech keywords: **{spoken_keywords}**")

    # ── Main generation block ─────────────────────────────────────────────────
    if uploaded_image is not None:
        # Save upload to a temp file that persists across the button click
        suffix = os.path.splitext(uploaded_image.name)[1].lower() or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            image_path = f.name
            f.write(uploaded_image.getbuffer())

        # Show a preview immediately
        st.image(image_path, caption="Uploaded image", use_container_width=True)

        if st.button("✨ Generate caption + story", type="primary"):
            # ── Caption ───────────────────────────────────────────────────────
            with st.spinner("Generating caption…"):
                try:
                    caption = _generate_caption(image_path)
                except Exception as e:
                    st.error(f"Caption error: {e}")
                    caption = ""

            st.subheader("📝 Caption")
            st.write(caption if caption else "*(No caption generated.)*")

            combined_keywords = _combine_keywords(text_keywords, spoken_keywords)

            # ── Story ─────────────────────────────────────────────────────────
            st.subheader("📖 Story (50–60 words)")
            with st.spinner("Generating story…"):
                try:
                    story = _generate_story(
                        caption=caption,
                        keywords=combined_keywords,
                        word_range=STORY_WORD_RANGE,
                    )
                except Exception as e:
                    st.error(f"Story error: {e}")
                    story = ""

            if story:
                st.write(story)
                st.caption(f"Word count: {len(story.split())}")

                # ── Text-to-speech ────────────────────────────────────────────
                with st.spinner("Generating audio…"):
                    audio_bytes, tts_error = _story_to_speech_bytes(story)

                if tts_error:
                    st.warning(f"⚠️ {tts_error}")
                elif audio_bytes:
                    st.subheader("🔊 Listen")
                    st.audio(audio_bytes, format="audio/mp3")
            else:
                st.info("No story was generated.")

        # Clean up the temp image file after the page run
        try:
            os.remove(image_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()