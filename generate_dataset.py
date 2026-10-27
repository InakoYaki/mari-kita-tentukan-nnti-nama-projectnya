"""
Starter dataset generator v2 for the sensitive-data masking thesis.
Standard library only. Python 3.9+.

Run:  python generate_dataset.py
Out:  data/train.jsonl, data/dev.jsonl, data/test.jsonl, data/labels.json
Each line: {"id","split","lang","style","text","spans":[{"label","start","end","attr"?}]}

Design rules baked in:
  * Templates are split FIRST (train/dev/test never share a template), and
    name pools are split too.
  * Every value is randomly generated. No real people, no real credentials,
    no real addresses (do not paste your own into the pools).
  * Hard negatives (8/13/16-digit order, invoice, tracking numbers) have NO
    label, to measure over-masking.
  * Identifiers with legacy/new forms are separate labels (NPWP_15 vs NPWP_NIK,
    SIM_LEGACY vs SIM_NIK). The NIK-shaped ones (NIK, KK, NPWP_NIK, SIM_NIK)
    are indistinguishable by digits alone; ONLY the cue word tells them apart.
  * "informal" templates imitate how people really type to an AI: lowercase,
    slang, code-mixing, messy number formats, landmark-style addresses.

Annotation rules (write these in your guideline):
  * PERSON: any reference to a person (full name, first name only, nickname).
    Honorifics (Pak/Bu/Mas/Mbak) stay OUTSIDE the span. attr = FULL | PARTIAL.
  * ADDRESS_ID: the whole contiguous location description that helps find a
    specific home, INCLUDING landmarks ("rumah hook, depan taman"). Cue words
    like "alamat" / "di" stay outside. attr = FORMAL | INFORMAL.
  * IP_ADDRESS: both public and private ranges are labeled (decide and document).

Format status (see labels.json): several formats are USER-REPORTED or from
third-party docs and must be checked against official sources before you trust
any headline number: SIM_LEGACY, PASSPORT, KK layout, AKTA_LAHIR (new model only).
"""
import argparse
import json
import os
import random
import re
import string
from collections import Counter

LABELS = [
    "PERSON", "EMAIL", "PHONE", "PHONE_ID", "ADDRESS", "ADDRESS_ID", "IP_ADDRESS", "USERNAME",
    "CREDIT_CARD", "BANK_ACCOUNT", "PASSWORD", "API_KEY_TOKEN",
    "NIK", "KK", "NPWP_15", "NPWP_NIK", "BPJS", "SIM_LEGACY", "SIM_NIK",
    "STNK_SERIAL", "AKTA_LAHIR", "PASSPORT",
]

STATUS = {
    "NIK": "verified: 16 digits",
    "KK": "verified: 16 digits (internal layout not verified)",
    "NPWP_15": "verified: 15-digit format still issued",
    "NPWP_NIK": "verified: individuals' 16-digit NPWP equals NIK",
    "BPJS": "verified: BPJS Kesehatan 13 digits",
    "STNK_SERIAL": "verified: 8-digit serial on the STNKB (not the plate number)",
    "AKTA_LAHIR": "verified for the new model only: ####-LU/LT-ddmmyyyy-####",
    "SIM_LEGACY": "UNVERIFIED: user-reported 4-4-6 (14 digits); third-party docs say 12 digits",
    "SIM_NIK": "policy news: new SIM number follows the NIK",
    "PASSPORT": "UNVERIFIED: user-reported letter + 7 digits",
}
TIER2 = ["SIM_LEGACY", "SIM_NIK", "STNK_SERIAL", "AKTA_LAHIR", "PASSPORT", "USERNAME"]

# ---- pools (name pools disjoint per split) ----------------------------------
FIRST = {
    "train": ["Budi", "Siti", "Agus", "Dewi", "Rizky", "Putri", "Andi", "Wulan"],
    "dev": ["Joko", "Ayu", "Fajar"],
    "test": ["Bambang", "Intan", "Hendra", "Maya"],
}
LAST = {
    "train": ["Santoso", "Rahmawati", "Wijaya", "Lestari", "Pratama", "Hidayat", "Kusuma", "Saputra"],
    "dev": ["Nugroho", "Permata", "Setiawan"],
    "test": ["Susanto", "Anggraini", "Firmansyah", "Wibowo"],
}
EMAIL_DOMAINS = ["contoh.co.id", "mail.test", "perusahaan.id", "example.com"]
STREETS_ID = ["Melati", "Mawar", "Kenanga", "Cendana", "Pahlawan", "Merdeka"]
KELURAHAN = ["Sukamaju", "Cipete", "Kebon Baru", "Tegal Alur", "Menteng Dalam"]
KECAMATAN = ["Kebayoran Lama", "Cengkareng", "Setiabudi", "Coblong", "Tebet"]
CITIES_ID = ["Jakarta Selatan", "Bandung", "Surabaya", "Medan", "Semarang"]
STREETS_EN = ["Maple", "Oak", "Cedar", "Pine"]
CITIES_EN = ["Springfield", "Rivertown", "Lakeside"]
AREAS = ["Kelapa Gading", "Pondok Indah", "Bintaro", "Cibubur", "Tebet", "Kemang",
         "BSD City", "Depok", "Dago Bandung", "Sunter"]
LANDMARKS = ["Rumah Hook", "depan taman", "sebelah masjid", "dekat Indomaret", "pagar hitam",
             "pojok belokan", "seberang sekolah", "samping warung", "rumah cat putih",
             "belakang pos satpam"]
STREET_NAMES = ["Melati", "Kenanga", "Cendrawasih", "Pahlawan", "Mawar", "Sudirman",
                "Anggrek", "Flamboyan"]


def digits(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def grouped(s, sizes, sep):
    out, i = [], 0
    for n in sizes:
        out.append(s[i:i + n])
        i += n
    return sep.join(out)


def luhn_card(rng):
    body = "4" + digits(rng, 14)
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return body + str((10 - total % 10) % 10)


# ---- value generators: fn(rng, split) -> str --------------------------------
def person(rng, s):
    return f"{rng.choice(FIRST[s])} {rng.choice(LAST[s])}"


def person_first(rng, s):
    return rng.choice(FIRST[s])


def email(rng, s):
    f, l = rng.choice(FIRST[s]).lower(), rng.choice(LAST[s]).lower()
    return f"{f}.{l}{rng.randint(1, 99)}@{rng.choice(EMAIL_DOMAINS)}"


def username(rng, s):
    f, l = rng.choice(FIRST[s]).lower(), rng.choice(LAST[s]).lower()
    return rng.choice([f"{f}_{l}{rng.randint(1, 99)}", f"{f}.{l}", f"{f}{rng.randint(100, 999)}"])


def phone(rng, s):  # generic / US-style
    return f"({digits(rng, 3)}) {digits(rng, 3)}-{digits(rng, 4)}"


def phone_id(rng, s):
    op = "8" + digits(rng, 2)
    rest = digits(rng, rng.choice([7, 8]))
    return rng.choice([
        f"0{op}{rest}",
        f"+62{op}{rest}",
        f"+62 {op}-{rest[:4]}-{rest[4:]}",
        f"0{op} {rest[:4]} {rest[4:]}",
        f"62{op}{rest}",
    ])


def address(rng, s):
    return f"{rng.randint(10, 999)} {rng.choice(STREETS_EN)} Street, {rng.choice(CITIES_EN)}"


def address_id(rng, s):  # formal
    return (f"Jl. {rng.choice(STREETS_ID)} No. {rng.randint(1, 120)}, "
            f"RT {rng.randint(1, 15):02d}/RW {rng.randint(1, 12):02d}, "
            f"Kel. {rng.choice(KELURAHAN)}, Kec. {rng.choice(KECAMATAN)}, {rng.choice(CITIES_ID)}")


def _street_phrase(rng):
    name = rng.choice(STREET_NAMES)
    kind = rng.choice(["jalan", "gang", "komplek"])
    if kind == "jalan":
        return (f"{rng.choice(['Jalan', 'Jl.', 'jln'])} {name} "
                f"Blok {rng.choice('ABCDEFGHJKLMNPQRST')} No {rng.randint(1, 200)}")
    if kind == "gang":
        return f"Gang {name} {rng.randint(1, 9)} no.{rng.randint(1, 60)}"
    return f"Komplek {name} Raya {rng.choice('ABCDEFG')}{rng.randint(1, 30)}"


def address_id_inf(rng, s):  # informal, landmark-style
    area = rng.choice(AREAS)
    lms = rng.sample(LANDMARKS, rng.choice([1, 2]))
    st = _street_phrase(rng)
    form = rng.randrange(4)
    if form == 0:
        return ", ".join([area] + lms + [st])
    if form == 1:
        return ", ".join([st] + lms + [area])
    if form == 2:
        return f"{' '.join(lms)}, {st} {area}"
    return st


def ip_address(rng, s):
    kind = rng.randrange(3)
    if kind == 0:
        return f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    if kind == 1:
        return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return ".".join(str(rng.randint(1, 223)) for _ in range(4))


def credit_card(rng, s):
    n = luhn_card(rng)
    return grouped(n, [4, 4, 4, 4], " ") if rng.random() < 0.8 else n


def bank_account(rng, s):
    return digits(rng, rng.choice([10, 11, 12]))


def password(rng, s):
    pool = string.ascii_letters + string.digits + "!@#$%"
    return "".join(rng.choice(pool) for _ in range(rng.randint(10, 16)))


def api_key(rng, s):
    alnum = string.ascii_letters + string.digits
    kind = rng.choice(["sk", "ghp", "AKIA"])
    if kind == "sk":
        return "sk-" + "".join(rng.choice(alnum) for _ in range(32))
    if kind == "ghp":
        return "ghp_" + "".join(rng.choice(alnum) for _ in range(36))
    return "AKIA" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16))


def nik(rng, s):  # 16 digits: region(6) + ddmmyy(6, day+40 for women) + serial(4)
    day = rng.randint(1, 28) + (40 if rng.random() < 0.5 else 0)
    n = f"{digits(rng, 6)}{day:02d}{rng.randint(1, 12):02d}{rng.randint(60, 99):02d}{digits(rng, 4)}"
    return grouped(n, [4, 4, 4, 4], " ") if rng.random() < 0.15 else n


def kk(rng, s):  # 16 digits (internal layout not verified)
    return digits(rng, 16)


def npwp_15(rng, s):
    if rng.random() < 0.7:
        return f"{digits(rng, 2)}.{digits(rng, 3)}.{digits(rng, 3)}.{digits(rng, 1)}-{digits(rng, 3)}.{digits(rng, 3)}"
    return digits(rng, 15)


def bpjs(rng, s):
    return digits(rng, 13)


def sim_legacy(rng, s):
    if rng.random() < 0.6:  # user-reported 4-4-6 (14 digits)
        return rng.choice(["-", " ", ""]).join([digits(rng, 4), digits(rng, 4), digits(rng, 6)])
    d = digits(rng, 12)  # 12-digit layout from third-party validator docs
    return d if rng.random() < 0.5 else grouped(d, [4, 2, 2, 4], " ")


def stnk_serial(rng, s):
    return digits(rng, 8)


def akta_lahir(rng, s):  # new-model pattern only
    return (f"{digits(rng, 4)}-{rng.choice(['LU', 'LT'])}-"
            f"{rng.randint(1, 28):02d}{rng.randint(1, 12):02d}{rng.randint(2005, 2024)}-{digits(rng, 4)}")


def passport(rng, s):  # user-reported: 1 letter + 7 digits
    return rng.choice(string.ascii_uppercase) + digits(rng, 7)


GEN = {
    "PERSON": person, "PERSON_FIRST": person_first, "EMAIL": email, "USERNAME": username,
    "PHONE": phone, "PHONE_ID": phone_id, "ADDRESS": address, "ADDRESS_ID": address_id,
    "ADDRESS_ID_INF": address_id_inf, "IP_ADDRESS": ip_address,
    "CREDIT_CARD": credit_card, "BANK_ACCOUNT": bank_account, "PASSWORD": password,
    "API_KEY_TOKEN": api_key, "NIK": nik, "KK": kk, "NPWP_15": npwp_15,
    "NPWP_NIK": nik, "BPJS": bpjs, "SIM_LEGACY": sim_legacy, "SIM_NIK": nik,
    "STNK_SERIAL": stnk_serial, "AKTA_LAHIR": akta_lahir, "PASSPORT": passport,
    "NEG8": lambda r, s: digits(r, 8),
    "NEG13": lambda r, s: digits(r, 13),
    "NEG16": lambda r, s: digits(r, 16),
}

SLOT_LABEL = {
    "PERSON": ("PERSON", "FULL"),
    "PERSON_FIRST": ("PERSON", "PARTIAL"),
    "ADDRESS_ID": ("ADDRESS_ID", "FORMAL"),
    "ADDRESS_ID_INF": ("ADDRESS_ID", "INFORMAL"),
}
LOWERABLE = {"PERSON", "ADDRESS_ID", "ADDRESS"}

# ---- templates, split BEFORE generation: (lang, style, text) ------------------
TEMPLATES = {
    "train": [
        ("id", "informal", "tolong bikinin pesan buat kurir ya, alamat gue di {ADDRESS_ID_INF}, hp {PHONE_ID}, atas nama {PERSON}"),
        ("id", "informal", "bantuin isi formulir bpjs dong, namanya {PERSON}, no bpjs {BPJS}, nik {NIK}, kk {KK}"),
        ("id", "informal", "sim gue hilang, nomornya {SIM_LEGACY}. gimana cara ngurusnya? nama gue {PERSON_FIRST}"),
        ("id", "informal", "kemarin gue bikin sim baru, nomor sim {SIM_NIK} ternyata sama kayak ktp, aman ga?"),
        ("id", "informal", "npwp lama gue {NPWP_15}, npwp baru {NPWP_NIK}, yang dipake buat lapor spt yang mana?"),
        ("id", "informal", "mau cek pajak motor, nomor seri stnk {STNK_SERIAL}, pemiliknya bu {PERSON_FIRST}"),
        ("id", "formal", "Akta kelahiran anak saya nomor {AKTA_LAHIR}, atas nama {PERSON}. Tolong buatkan surat permohonan."),
        ("mix", "informal", "paspor gue {PASSPORT} expired bulan depan, how to renew? nama di paspor {PERSON}"),
        ("en", "formal", "Draft an email to {PERSON} ({EMAIL}) confirming the meeting; call {PHONE} if urgent."),
        ("en", "formal", "Customer {PERSON} lives at {ADDRESS}. Please verify card {CREDIT_CARD}."),
        ("id", "formal", "Transfer ke rekening {BANK_ACCOUNT} a.n. {PERSON}, jangan lupa konfirmasi ke {EMAIL}."),
        ("en", "formal", "Debug this config: API_KEY = '{API_KEY_TOKEN}' and DB_PASSWORD = '{PASSWORD}' on host {IP_ADDRESS}."),
        ("mix", "informal", "login wifi kantor: user {USERNAME} pass {PASSWORD}, router ip {IP_ADDRESS}. bantu bikin pengumuman ke tim"),
        ("id", "informal", "nomor resi {NEG13} udah dikirim ke {PERSON_FIRST}, invoice {NEG8}, order {NEG16}"),
        ("id", "informal", "rumah gue susah dicari, {ADDRESS_ID_INF}. kasih tau abang ojolnya dong"),
        ("mix", "formal", "Analisis tabel ini: | nama | NIK |\n| {PERSON} | {NIK} |"),
    ],
    "dev": [
        ("id", "informal", "cek apakah npwp {NPWP_NIK} punya {PERSON} udah terdaftar, hp {PHONE_ID}"),
        ("en", "formal", "Summarize: {PERSON_FIRST} emailed {EMAIL} about passport {PASSPORT}."),
        ("id", "informal", "tolong buatin surat, alamat {ADDRESS_ID_INF}, sim {SIM_LEGACY}, stnk {STNK_SERIAL}"),
        ("id", "formal", "Akta {AKTA_LAHIR} dan kk {KK} atas nama {PERSON}, bpjs {BPJS}, npwp {NPWP_15}; rekening {BANK_ACCOUNT}"),
        ("en", "formal", "password {PASSWORD} username {USERNAME} token {API_KEY_TOKEN} ip {IP_ADDRESS}"),
        ("en", "formal", "my nik {NIK}, sim {SIM_NIK}, card {CREDIT_CARD}, call {PHONE}, ship to {ADDRESS}"),
    ],
    "test": [
        ("id", "informal", "gue lagi di {ADDRESS_ID_INF}, tolong cariin kurir. hubungi {PERSON_FIRST} di {PHONE_ID}"),
        ("en", "formal", "Here is my token {API_KEY_TOKEN}, why does the login fail for {EMAIL}?"),
        ("mix", "formal", "Tolong translate: Pak {PERSON}, NIK {NIK}, kartu {CREDIT_CARD}, rekening {BANK_ACCOUNT}, kk {KK}."),
        ("id", "informal", "npwp lama {NPWP_15}, npwp baru {NPWP_NIK}? bingung gue. bpjs gue {BPJS}"),
        ("id", "informal", "stnk {STNK_SERIAL} sama sim {SIM_LEGACY} ketinggalan di mobil {PERSON_FIRST}, sim baru {SIM_NIK}"),
        ("en", "formal", "Set password to {PASSWORD} for user {USERNAME} ({PERSON}) at {ADDRESS}; router {IP_ADDRESS}."),
        ("id", "formal", "Akta kelahiran {AKTA_LAHIR} dan paspor {PASSPORT}, a.n. {PERSON}, hp {PHONE}."),
        ("id", "informal", "ref {NEG16} resi {NEG13} invoice {NEG8} buat {PERSON}"),
    ],
}

SLOT = re.compile(r"\{([A-Z_0-9]+)\}")


def fill(template, rng, split, style):
    text, spans, cursor = "", [], 0
    for m in SLOT.finditer(template):
        text += template[cursor:m.start()]
        slot = m.group(1)
        val = GEN[slot](rng, split)
        label, attr = SLOT_LABEL.get(slot, (slot, None))
        if label in LOWERABLE and style == "informal" and rng.random() < 0.5:
            val = val.lower()
        if label in LABELS:
            span = {"label": label, "start": len(text), "end": len(text) + len(val)}
            if attr:
                span["attr"] = attr
            spans.append(span)
        text += val
        cursor = m.end()
    text += template[cursor:]
    return text, spans


def build(split, n, rng):
    rows = []
    for i in range(n):
        lang, style, tpl = rng.choice(TEMPLATES[split])
        text, spans = fill(tpl, rng, split, style)
        for sp in spans:
            assert 0 <= sp["start"] < sp["end"] <= len(text)
        rows.append({"id": f"{split}-{i:05d}", "split": split, "lang": lang,
                     "style": style, "text": text, "spans": spans})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-dev", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=600)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    for split, n in [("train", args.n_train), ("dev", args.n_dev), ("test", args.n_test)]:
        rows = build(split, n, rng)
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = Counter(sp["label"] for r in rows for sp in r["spans"])
        missing = [l for l in LABELS if counts[l] == 0]
        print(f"{split}: {len(rows)} rows, {sum(counts.values())} spans"
              + (f" | MISSING labels: {missing}" if missing else " | all labels covered"))
    bio = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "bio_tags": bio, "status": STATUS, "tier2": TIER2},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
