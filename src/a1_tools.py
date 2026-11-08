"""Pieces carried over unchanged from Assignment 1 (BBC Urdu pipeline).

reading the numbered article blocks, the token constants, and the rule based
UrduLemmatizer / UrduStemmer that the Part 2 POS tagger is built on.
"""
import re
from collections import Counter

NUM_TOKEN = "<NUM>"
PUNCT_CHARS = "۔؟!،"
PUNCT_TOKENS = set(PUNCT_CHARS)

POSTPOSITIONS = {
    "کا", "کی", "کے", "کو", "میں", "سے", "پر", "نے", "تک", "تلک", "ساتھ", "لیے", "لئے", "بعد", "پہلے",
    "اندر", "باہر", "بارے", "طرف", "خلاف", "دوران", "مطابق", "بغیر", "علاوہ", "جانب", "ذریعے", "سبب", "وجہ",
    "درمیان", "برابر", "بجائے", "مقابلے", "پاس", "نزدیک", "اوپر", "نیچے",
}
FUNCTION_WORDS = {
    "میں", "ہم", "تم", "آپ", "وہ", "یہ", "اس", "ان", "اسی", "انہی", "جو", "جس", "جن", "کس", "کن", "کون", "کیا",
    "کچھ", "سب", "کوئی", "کسی", "خود", "مجھے", "ہمیں", "تمہیں", "اسے", "انھیں", "انہیں", "انھوں", "انہوں", "انھی",
    "ہے", "ہیں", "ہو", "ہوں", "ہوا", "ہوئی", "ہوئے", "ہوتا", "ہوتی", "ہوتے", "تھا", "تھی", "تھے", "تھیں", "گا", "گی",
    "گے", "رہا", "رہی", "رہے", "رہیں", "سکتا", "سکتی", "سکتے", "سکتیں", "چاہیے", "چاہئے", "گیا", "گئی", "گئے", "گئیں",
    "کیے", "دیا", "دیے", "لیا", "لی", "دی", "لے", "دے", "کرے", "اور", "یا", "کہ", "لیکن", "مگر", "تاہم",
    "بلکہ", "اگر", "تو", "بھی", "ہی", "نہ", "نہیں", "جب", "تب", "جبکہ", "کیونکہ", "چنانچہ", "اگرچہ", "بہت", "زیادہ",
    "کم", "ہر", "ایک", "دو", "کبھی", "اب", "پھر", "یہاں", "وہاں", "کہاں", "کب", "کیوں", "کیسے", "کتنا", "بھر", "ابھی",
    "آج", "کل", "صرف", "ہمیشہ", "یوں", "یہیں", "وہیں", "کہیں", "جہاں", "جیسے", "جیسا", "تھوڑا", "ورنہ",
}
PROTECTED_WORDS = POSTPOSITIONS | FUNCTION_WORDS

LEMMA_DICT = {
    "اچھی": "اچھا", "اچھے": "اچھا", "بڑی": "بڑا", "بڑے": "بڑا", "چھوٹی": "چھوٹا", "چھوٹے": "چھوٹا",
    "نئی": "نیا", "نئے": "نیا", "پرانی": "پرانا", "پرانے": "پرانا", "لمبی": "لمبا", "لمبے": "لمبا",
    "پہلی": "پہلا", "دوسری": "دوسرا", "دوسرے": "دوسرا", "تیسری": "تیسرا", "تیسرے": "تیسرا", "ایسی": "ایسا",
    "ایسے": "ایسا", "کیسی": "کیسا", "اتنی": "اتنا", "اتنے": "اتنا", "کتنی": "کتنا", "کتنے": "کتنا",
    "اپنی": "اپنا", "اپنے": "اپنا", "میری": "میرا", "میرے": "میرا", "ہماری": "ہمارا", "ہمارے": "ہمارا",
    "مہنگی": "مہنگا", "مہنگے": "مہنگا", "ساری": "سارا", "سارے": "سارا", "پوری": "پورا", "پورے": "پورا",
    "پچھلی": "پچھلا", "پچھلے": "پچھلا", "اگلی": "اگلا", "اگلے": "اگلا", "اونچی": "اونچا", "اونچے": "اونچا",
    "لڑکی": "لڑکا", "لڑکے": "لڑکا", "لڑکوں": "لڑکا", "لڑکیاں": "لڑکا", "لڑکیوں": "لڑکا",
    "بچی": "بچہ", "بچے": "بچہ", "بچوں": "بچہ", "بچیاں": "بچہ", "بچیوں": "بچہ",
    "بیٹی": "بیٹا", "بیٹے": "بیٹا", "بیٹوں": "بیٹا", "بیٹیاں": "بیٹا", "بیٹیوں": "بیٹا",
    "بھائیوں": "بھائی", "بہنوں": "بہن", "بہنیں": "بہن", "مردوں": "مرد", "عورتیں": "عورت", "عورتوں": "عورت",
    "خواتین": "خاتون", "علاقے": "علاقہ", "علاقوں": "علاقہ", "فیصلے": "فیصلہ", "فیصلوں": "فیصلہ",
    "مسائل": "مسئلہ", "مسئلے": "مسئلہ", "واقعات": "واقعہ", "واقعے": "واقعہ", "نتائج": "نتیجہ",
    "حالات": "حالت", "اقدامات": "اقدام", "انتخابات": "انتخاب", "افراد": "فرد", "ارکان": "رکن",
    "اراکین": "رکن", "وزرا": "وزیر", "وزراء": "وزیر", "ممالک": "ملک", "دھماکے": "دھماکہ", "دھماکوں": "دھماکہ",
    "حملے": "حملہ", "حملوں": "حملہ", "مقدمے": "مقدمہ", "مقدمات": "مقدمہ", "ادارے": "ادارہ", "اداروں": "ادارہ",
    "مہینے": "مہینہ", "مہینوں": "مہینہ", "ہفتے": "ہفتہ", "ہفتوں": "ہفتہ", "معاملے": "معاملہ", "معاملات": "معاملہ",
    "الزامات": "الزام", "تفصیلات": "تفصیل", "زخمیوں": "زخمی", "کتابیں": "کتاب", "کتابوں": "کتاب",
    "حکومتیں": "حکومت", "حکومتوں": "حکومت", "ریاستیں": "ریاست", "ریاستوں": "ریاست", "قیمتیں": "قیمت", "قیمتوں": "قیمت",
}
KEEP_AS_IS = {"معلومات", "مذاکرات", "ملاقات", "بات", "رات", "ذات", "نجات", "حالت", "وقت", "طاقت", "حکومت", "عوام", "سعودی"}
PLURAL_RULES = [
    ("یوں", ["ی"]),
    ("وں", ["", "ا", "ہ"]),
    ("یاں", ["ی"]),
    ("یں", [""]),
    ("ات", ["ت", "", "ہ"]),
]
STEM_SUFFIXES = sorted(["انہ", "یوں", "وں", "یاں", "یں", "ات", "ی"], key=len, reverse=True)
LEMMA_TARGETS = set(LEMMA_DICT.values())
VERB_ROOTS = {"کر", "جا", "ہو", "آ", "دے", "لے", "رہ", "کہہ", "سن", "لگ", "مل", "بن", "رکھ", "پڑ", "چل"}


class UrduLemmatizer:
    def __init__(self, word_freq, min_count=2):
        self.word_freq = word_freq
        self.min_count = min_count

    def lemmatize(self, word):
        if word in PROTECTED_WORDS or word in KEEP_AS_IS:
            return word
        if word in LEMMA_DICT:
            return LEMMA_DICT[word]
        for suffix, replacements in PLURAL_RULES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 2:
                for replacement in replacements:
                    candidate = word[:-len(suffix)] + replacement
                    if (candidate != word and candidate not in VERB_ROOTS
                            and self.word_freq.get(candidate, 0) >= self.min_count):
                        return candidate
        return word


class UrduStemmer:
    def __init__(self, word_freq, min_count=2, min_stem=3):
        self.word_freq = word_freq
        self.min_count = min_count
        self.min_stem = min_stem

    def stem(self, word):
        if word in PROTECTED_WORDS or word in KEEP_AS_IS or word in LEMMA_TARGETS:
            return word
        for suffix in STEM_SUFFIXES:
            if word.endswith(suffix):
                candidate = word[:-len(suffix)]
                if (len(candidate) >= self.min_stem and candidate not in PROTECTED_WORDS
                        and candidate not in VERB_ROOTS and self.word_freq.get(candidate, 0) >= self.min_count):
                    return candidate
        return word


def read_blocks(path):
    """{article_number: body text} from a file of '[Article N]' blocks (raw.txt / cleaned.txt)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    pieces = re.split(r"^\[Article (\d+)\]\s*$", text, flags=re.M)
    return {int(pieces[i]): pieces[i + 1].strip() for i in range(1, len(pieces), 2)}


def read_cleaned(path):
    """cleaned.txt -> {article_number: [sentence token lists]} (one sentence per line)."""
    return {num: [line.split() for line in body.split("\n") if line.strip()] for num, body in read_blocks(path).items()}


RAW_TOKEN_RE = re.compile(r"[^\s۔؟!،.,?:;\"'()‘’“”!]+|[۔؟!،.,?:;\"'()‘’“”!]")


def read_raw(path):
    """raw.txt -> {article_number: [sentence token lists]} with no normalisation at all (the A1 raw baseline)."""
    docs = {}
    for num, body in read_blocks(path).items():
        sentences = []
        for paragraph in body.split("\n"):
            for piece in re.split(r"(?<=[۔؟!.?])\s+", paragraph):
                tokens = RAW_TOKEN_RE.findall(piece)
                if tokens:
                    sentences.append(tokens)
        docs[num] = sentences
    return docs


def corpus_word_freq(docs):
    return Counter(t for sents in docs.values() for s in sents for t in s if t not in PUNCT_TOKENS and t != NUM_TOKEN)
