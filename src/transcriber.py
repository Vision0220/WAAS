from typing import Any

# import whisper
from faster_whisper import WhisperModel
from sentry_sdk import set_user
import time

def map_language(language: str):
    # map from afganistan to af
    # all list: af, am, ar, as, az, ba, be, bg, bn, bo, br, bs, ca, cs, cy, da, de, el, en, es, et, eu, fa, fi, fo, fr, gl, gu, ha, haw, he, hi, hr, ht, hu, hy, id, is, it, ja, jw, ka, kk, km, kn, ko, la, lb, ln, lo, lt, lv, mg, mi, mk, ml, mn, mr, ms, mt, my, ne, nl, nn, no, oc, pa, pl, ps, pt, ro, ru, sa, sd, si, sk, sl, sn, so, sq, sr, su, sv, sw, ta, te, tg, th, tk, tl, tr, tt, uk, ur, uz, vi, yi, yo, zh, yue
    mapping = {
        "english": "en",
        "chinese": "zh",
        "afrikaans": "af",
        "amharic": "am",
        "arabic": "ar",
        "assamese": "as",
        "azerbaijani": "az",
        "bashkir": "ba",
        "belarusian": "be",
        "bulgarian": "bg",
        "bengali": "bn",
        "tibetan": "bo",
        "breton": "br",
        "bosnian": "bs",
        "catalan": "ca",
        "czech": "cs",
        "welsh": "cy",
        "danish": "da",
        "german": "de",
        "greek": "el",
        "spanish": "es",
        "estonian": "et",
        "basque": "eu",
        "persian": "fa",
        "finnish": "fi",
        "faroese": "fo",
        "french": "fr",
        "galician": "gl",
        "gujarati": "gu",
        "hausa": "ha",
        "hawaiian": "haw",
        "hebrew": "he",
        "hindi": "hi",
        "croatian": "hr",
        "haitian": "ht",
        "hungarian": "hu",
        "armenian": "hy",
        "indonesian": "id",
        "icelandic": "is",
        "italian": "it",
        "japanese": "ja",
        "javanese": "jw",
        "georgian": "ka",
        "kazakh": "kk",
        "khmer": "km",
        "kannada": "kn",
        "korean": "ko",
        "latin": "la",
        "luxembourgish": "lb",
        "lingala": "ln",
        "lao": "lo",
        "lithuanian": "lt",
        "latvian": "lv",
        "malagasy": "mg",
        "maori": "mi",
        "macedonian": "mk",
        "malayalam": "ml",
        "mongolian": "mn",
        "marathi": "mr",
        "malay": "ms",
        "maltese": "mt",
        "burmese": "my",
        "nepali": "ne",
        "dutch": "nl",
        "norwegian nynorsk": "nn",
        "norwegian": "no",
        "occitan": "oc",
        "punjabi": "pa",
        "polish": "pl",
        "pashto": "ps",
        "portuguese": "pt",
        "romanian": "ro",
        "russian": "ru",
        "sanskrit": "sa",
        "sindhi": "sd",
        "sinhala": "si",
        "slovak": "sk",
        "slovenian": "sl",
        "shona": "sn",
        "somali": "so",
        "albanian": "sq",
        "serbian": "sr",
        "sundanese": "su",
        "swedish": "sv",
        "swahili": "sw",
        "tamil": "ta",
        "telugu": "te",
        "tajik": "tg",
        "thai": "th",
        "turkmen": "tk",
        "tagalog": "tl",
        "turkish": "tr",
        "tatar": "tt",
        "ukrainian": "uk",
        "urdu": "ur",
        "uzbek": "uz",
        "vietnamese": "vi",
        "yiddish": "yi",
        "yoruba": "yo",
        "cantonese": "yue"
    }

    if language == None:
        return None
    else:
        return mapping[language.lower()]
        

def transcribe(filename: str, requestedModel: str, task: str, language: str, email: str, webhook_id: str) -> dict[str, Any]:
    # Mail is not used here, but it makes it easier for the worker to log mail
    print("Executing transcribing of " + filename + " for "+(email or webhook_id) + " using " + requestedModel + " model ")
    set_user({"email": email})
    # model = whisper.load_model(requestedModel)
    
    # model = WhisperModel(requestedModel, device="cuda", compute_type="float32")
    print("start loading model")
    t1 = time.time()
    model = WhisperModel(requestedModel, device="cuda", compute_type="int8")
    print("end loading model, time: ", time.time() - t1)
    t2 = time.time()
    segments, info = model.transcribe(filename, language=map_language(language), task=task, beam_size=5, vad_filter=True)
    print('end transcribe, time: ', time.time() - t2)
    print('real time factor: ', info.duration / (time.time() - t2))
    
    
    segments_list = [i for i in segments]
    result = {
            'text': " ".join([i.text for i in segments_list]),
            'segments':[{
                "id":i.id,
                "seek":i.seek,
                "start":i.start,
                "end":i.end,
                "text":i.text,
                "tokens":i.tokens,
                "temperature":i.temperature,
                "avg_logprob":i.avg_logprob,
                "compression_ratio":i.compression_ratio,
                "no_speech_prob":i.no_speech_prob,                 
                } for i in segments_list],
            'language': info.language,
        }
    return result
