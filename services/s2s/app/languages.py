"""Languages the call can use, from what each model service supports.

"speak" languages are the ones with an IndicConformer model in the ASR NIM (Malayalam,
Hindi, Kannada, Telugu, Tamil; services/asr-nim/build adds more). "hear" languages are
what indic-translate can target and indic-speak can voice: the 22 scheduled Indian
languages plus English. TTS quality tiers come from the indic-speak model card.

On other model servers (docker-compose.shared.yml: Parakeet, Riva Translate, Magpie),
S2S_SPEAK_LANGUAGES and S2S_HEAR_LANGUAGES narrow both lists to what those cover.
"""
import os

LANGUAGES = {
    # code: (English name, native name, TTS tier)
    "as": ("Assamese", "অসমীয়া", "production"),
    "bn": ("Bengali", "বাংলা", "production"),
    "brx": ("Bodo", "बड़ो", "preview"),
    "doi": ("Dogri", "डोगरी", "preview"),
    "gu": ("Gujarati", "ગુજરાતી", "production"),
    "hi": ("Hindi", "हिन्दी", "production"),
    "kn": ("Kannada", "ಕನ್ನಡ", "production"),
    "kok": ("Konkani", "कोंकणी", "preview"),
    "ks": ("Kashmiri", "کٲشُر", "preview"),
    "mai": ("Maithili", "मैथिली", "preview"),
    "ml": ("Malayalam", "മലയാളം", "production"),
    "mni": ("Manipuri", "মৈতৈলোন্", "preview"),
    "mr": ("Marathi", "मराठी", "production"),
    "ne": ("Nepali", "नेपाली", "preview"),
    "or": ("Odia", "ଓଡ଼ିଆ", "production"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ", "production"),
    "sa": ("Sanskrit", "संस्कृतम्", "preview"),
    "sat": ("Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "preview"),
    "sd": ("Sindhi", "सिन्धी", "preview"),
    "ta": ("Tamil", "தமிழ்", "production"),
    "te": ("Telugu", "తెలుగు", "production"),
    "ur": ("Urdu", "اردو", "production"),
    "en": ("English", "English", "production"),
}

def _env_list(name: str, default: set[str]) -> set[str]:
    return {c.strip() for c in os.environ.get(name, "").split(",") if c.strip() in LANGUAGES} or default


# Languages with a speech recognition model in the ASR NIM. Riva serves each under its own
# language code (ml-IN, hi-IN, ...; English: en-US, see ASR_CODE).
ASR_LANGUAGES = _env_list("S2S_SPEAK_LANGUAGES", {"hi", "kn", "ml", "ta", "te"})
ASR_CODE = {code: f"{code}-IN" for code in LANGUAGES} | {"en": "en-US"}
SPEAK = [code for code in LANGUAGES if code in ASR_LANGUAGES]
HEAR = [code for code in LANGUAGES if code in _env_list("S2S_HEAR_LANGUAGES", set(LANGUAGES))]

# The target-language names indic-translate's prompt expects (LANGUAGE_NAMES in the model's
# scripts/prompt_contract.py): the English names above, with the script spelled out for
# languages the model writes in two scripts. The qualifier selects the output script.
PROMPT_NAME = {code: v[0] for code, v in LANGUAGES.items()} | {
    "ks": "Kashmiri (Perso-Arabic script)",
    "mni": "Manipuri (Meitei script)",
    "sd": "Sindhi (Devanagari script)",
}

# The script each language is written in (the first word of its letters' Unicode names), as
# PROMPT_NAME selects it. Used to spot translations that copied source-script words over.
SCRIPT = {code: "DEVANAGARI" for code in ("brx", "doi", "hi", "kok", "mai", "mr", "ne", "sa", "sd")} | {
    "as": "BENGALI", "bn": "BENGALI", "gu": "GUJARATI", "kn": "KANNADA", "ks": "ARABIC",
    "ml": "MALAYALAM", "mni": "MEETEI", "or": "ORIYA", "pa": "GURMUKHI", "sat": "OL",
    "ta": "TAMIL", "te": "TELUGU", "ur": "ARABIC", "en": "LATIN",
}


# ---------------------------------------------------------------- voices
# Each caller picks the voice the others hear them in. indic-speak (TTS_API=indic-speak) has a
# recommended female and male voice per language. Magpie (TTS_API=magpie) has named speakers
# per locale: these are the ones that exist in every locale the call can be heard in (en-US and
# hi-IN), so a caller's voice works whatever language a listener picks. NVIDIA doesn't publish
# their genders; these come from their measured pitch (YIN, English and Hindi, 2026-10-28):
# female 194-240 Hz average, male 139-177 Hz.
TTS_API = os.environ.get("TTS_API", "indic-speak")
MAGPIE_LOCALE = {"en": "en-US", "hi": "hi-IN"}            # the call languages Magpie voices
_MAGPIE_VOICES = {"Louise": "female", "Aria": "female", "Sofia": "female", "Isabela": "female", "Leo": "female",
                  "Jason": "male", "Pascal": "male", "Diego": "male"}
VOICES = _MAGPIE_VOICES if TTS_API == "magpie" else {"female": "female", "male": "male"}
DEFAULT_VOICE = ({"female": os.environ.get("MAGPIE_FEMALE_VOICE", "Louise"), "male": os.environ.get("MAGPIE_MALE_VOICE", "Jason")}
                 if TTS_API == "magpie" else {"female": "female", "male": "male"})


def voice_id(v: str | None) -> str:
    """A caller's voice choice: one of VOICES, "female"/"male" for that gender's default."""
    return v if v in VOICES else DEFAULT_VOICE.get(v or "", DEFAULT_VOICE["female"])


def describe() -> dict:
    return {
        "speak": [{"code": c, "name": LANGUAGES[c][0], "native": LANGUAGES[c][1]} for c in SPEAK],
        "hear": [{"code": c, "name": LANGUAGES[c][0], "native": LANGUAGES[c][1],
                  "preview": LANGUAGES[c][2] == "preview"} for c in HEAR],
        "voices": [{"id": v, "name": v.capitalize(), "gender": g} for v, g in VOICES.items()],
        "default_voice": DEFAULT_VOICE,
        "voice_preview": TTS_API == "magpie",
        # background noise removal (the agent's bnr.py), which each caller can turn off for their mic
        "noise_removal": bool(os.environ.get("BNR_GRPC_URL")),
    }
