import os, io, re, json, time
import numpy as np
import librosa, soundfile as sf
from mutagen import File as MutagenFile
import streamlit as st
from openai import OpenAI

# ---------------------------
# 공통: OpenAI 클라이언트
# ---------------------------
def make_client(api_key: str) -> OpenAI:
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key)

# ---------------------------
# 오디오 유틸
# ---------------------------
def get_duration_seconds(path: str) -> float:
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        return float(len(y)/sr)
    except Exception:
        pass
    try:
        mf = MutagenFile(path)
        if mf and getattr(mf, "info", None) and getattr(mf.info, "length", None):
            return float(mf.info.length)
    except Exception:
        pass
    try:
        with sf.SoundFile(path) as f:
            return float(len(f)/f.samplerate)
    except Exception:
        pass
    return 0.0

def basic_audio_metrics(path: str, transcript: str) -> dict:
    dur = max(get_duration_seconds(path), 1e-6)
    words = len(re.findall(r"\b[\w가-힣]+\b", transcript))
    wpm = words / (dur/60.0)
    filler_pats = [
        r"\b음+\b", r"\b어+\b", r"\b그\b", r"\bum+\b", r"\buh+\b", r"\ber+\b", r"\blike\b", r"\byou know\b"
    ]
    filler_count = sum(len(re.findall(p, transcript, flags=re.IGNORECASE)) for p in filler_pats)

    # 간단 무성구간 비율
    silence_ratio = None
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        thresh = np.percentile(rms, 20)
        silence_ratio = float(np.mean(rms < thresh))
    except Exception:
        pass

    sents = [s for s in re.split(r"[\.!\?…]+", transcript.strip()) if s.strip()]
    avg_wps = (sum(len(s.split()) for s in sents)/len(sents)) if sents else 0.0
    return {
        "duration_sec": round(dur, 2),
        "words": words,
        "wpm": round(wpm, 1),
        "filler_count": filler_count,
        "silence_ratio": round(silence_ratio, 3) if silence_ratio is not None else None,
        "avg_words_per_sentence": round(avg_wps, 1),
        "sentence_count": len(sents),
    }

def transcribe_whisper(client: OpenAI, file_bytes: bytes, filename: str="audio.wav") -> str:
    bio = io.BytesIO(file_bytes); bio.name = filename
    return client.audio.transcriptions.create(
        model="whisper-1",
        file=bio,
        response_format="text"
    )

# ---------------------------
# LLM 프롬프트
# ---------------------------
def speech_coach_prompt(transcript: str, metrics: dict, lang="ko"):
    if lang == "en":
        guide = (
            "You are a speech coach. Based on the transcript and metrics, give 5–8 actionable tips. "
            "Be friendly yet firm; add brief evidence (metric/snippet). "
            "Finish with a 1-minute rehearsal checklist (numbered)."
        )
    else:
        guide = (
            "너는 스피치 코치다. 전사와 계량지표를 바탕으로 5~8개의 실행가능한 조언을 제시하라. "
            "친절하되 단호하게, 각 항목에 근거(지표/전사 일부)를 짧게 붙여라. "
            "마지막에 1분 리허설 체크리스트를 번호 목록으로 제시하라."
        )
    return f"""{guide}

[Metrics]
{json.dumps(metrics, ensure_ascii=False, indent=2)}

[Transcript]
{transcript}
"""

def text_edit_prompt(text: str, lang="ko"):
    if lang == "en":
        return (
            "You are an expert writing tutor. Given the user's script, point out ungrammatical or awkward parts, "
            "verbosity, filler, logical leaps, or unclear expressions. Then provide a corrected version. "
            "Output sections:\n"
            "1) Issues (bullet list)\n2) Revised Script (polished, natural, concise)\n"
            "Keep original meaning. Maintain tone appropriate for a speech."
            f"\n\n[Script]\n{text}\n"
        )
    else:
        return (
            "너는 전문 글쓰기 튜터다. 사용자의 대본에서 문법 오류, 어색한 표현, 군더더기, 논리적 비약, 불명확 표현을 지적하고, "
            "개선된 수정본을 제시하라. 출력 형식:\n"
            "1) 문제점(불릿)\n2) 수정본(자연스럽고 간결하게, 의미 유지)\n"
            "연설에 적합한 어조를 유지하라."
            f"\n\n[대본]\n{text}\n"
        )

def chatgpt_coach(client: OpenAI, transcript: str, metrics: dict, lang="ko") -> str:
    model = "gpt-5-mini"  # 가용 모델에 맞게 조정 (안되면 gpt-4.1-mini)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are an expert speech coach."},
            {"role":"user","content":speech_coach_prompt(transcript, metrics, lang=lang)}
        ],
        temperature=1,
    )

    return r.choices[0].message.content.strip()

def chatgpt_text_edit(client: OpenAI, text: str, lang="ko") -> str:
    model = "gpt-5-mini"
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are an expert writing tutor and speech editor."},
            {"role":"user","content":text_edit_prompt(text, lang=lang)}
        ],
        temperature=1,
    )
    return r.choices[0].message.content.strip()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Speech Analyzer & Script Editor", page_icon="🎤", layout="centered")
st.title("🎤 Speech Analyzer & ✍️ Script Editor")

with st.sidebar:
    api_key = st.text_input("OPENAI_API_KEY", type="password")
    lang = st.selectbox("언어(Language)", ["ko","en"], index=0)
    st.caption("Codespaces에서 실행 시 이 입력창에 API 키를 넣으세요.")

tab1, tab2 = st.tabs(["🔊 음성 업로드 분석", "✍️ 대본 교정(텍스트/파일)"])

# ---------------------------
# Tab 1: 음성 업로드 → 전사 → 지표 → 코칭
# ---------------------------
with tab1:
    st.subheader("1) 음성 파일 업로드")
    audio_file = st.file_uploader("파일 선택 (wav/mp3/m4a/ogg)", type=["wav","mp3","m4a","ogg"], key="audio_up")
    if audio_file is not None:
        tmp_path = f"./_tmp_{int(time.time())}_{audio_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(audio_file.getvalue())
        st.success(f"업로드 완료: {audio_file.name}")

        colA, colB = st.columns(2)
        with colA:
            if st.button("전사 실행", use_container_width=True):
                if not api_key:
                    st.error("API 키를 입력하세요.")
                else:
                    try:
                        client = make_client(api_key)
                        with st.spinner("Whisper 전사 중..."):
                            transcript = transcribe_whisper(client, audio_file.getvalue(), filename=audio_file.name)
                        st.session_state["transcript"] = transcript
                        st.text_area("전사 결과", transcript, height=220)
                        st.success("전사 완료")
                    except Exception as e:
                        st.error(f"전사 실패: {e}")

        with colB:
            if st.button("분석 + 코칭", use_container_width=True):
                if not api_key:
                    st.error("API 키를 입력하세요.")
                else:
                    transcript = st.session_state.get("transcript", "")
                    if not transcript:
                        st.warning("전사 결과가 없습니다. 먼저 전사를 실행하세요.")
                    else:
                        try:
                            metrics = basic_audio_metrics(tmp_path, transcript)
                            st.markdown("**기초 지표**")
                            st.json(metrics)
                            client = make_client(api_key)
                            with st.spinner("코칭 생성 중..."):
                                advice = chatgpt_coach(client, transcript, metrics, lang=lang)
                            st.markdown("**코칭 결과**")
                            st.write(advice)
                        except Exception as e:
                            st.error(f"분석/코칭 실패: {e}")

# ---------------------------
# Tab 2: 대본 업로드/입력 → 어색한 표현 지적 + 수정본
# ---------------------------
with tab2:
    st.subheader("1) 대본 입력 또는 파일 업로드")
    text_input = st.text_area("대본을 직접 붙여넣기", height=200, key="script_text")
    script_file = st.file_uploader("또는 텍스트 파일 업로드(.txt, .md)", type=["txt","md"], key="script_file")

    if st.button("대본 교정 실행"):
        if not api_key:
            st.error("API 키를 입력하세요.")
        else:
            script = text_input.strip()
            if not script and script_file is not None:
                script = script_file.getvalue().decode("utf-8", errors="ignore").strip()
            if not script:
                st.warning("대본이 비었습니다. 입력하거나 파일을 올리세요.")
            else:
                try:
                    client = make_client(api_key)
                    with st.spinner("교정/수정본 생성 중..."):
                        edited = chatgpt_text_edit(client, script, lang=lang)
                    st.markdown("**교정 결과**")
                    st.write(edited)
                except Exception as e:
                    st.error(f"교정 실패: {e}")
