"""Text normalisation (problem.md decision D1).

Two layers:
  * basic_name / basic_address (stage 01): Indic->Latin transliteration, HTML entities, accent
    stripping, lower-case, '&'->'and', punctuation, placeholders, record flags. Needs nothing learned.
  * canon_name / canon_address (stage 03): applies the learned rewrite tables from stage 02 (training
    data only), then legal-form, street/unit/direction, state and house-number canonicalisation.

The same code path runs for every country. Country only selects a state-alias table when one is
known; unknown countries (e.g. France in test) simply get no state extraction. Nothing is filtered or
branched on beyond that lookup.
"""
import html
import re
import unicodedata

# --------------------------------------------------------------------------- record flags
F_INDIC_NAME = 1 << 0
F_INDIC_ADDR = 1 << 1
F_DOMAIN = 1 << 2
F_ALLCAPS = 1 << 3
F_ACCENTED = 1 << 4
F_LEGAL_FRONT = 1 << 5
F_ADDR_EMPTY = 1 << 6
F_ADDR_PLACEHOLDER = 1 << 7
F_LANDMARK = 1 << 8

# --------------------------------------------------------------------------- Indic -> Latin
# The nine Brahmic blocks used in the data (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
# Telugu, Kannada, Malayalam) share the ISCII-derived layout, so one table keyed by the offset inside
# the 0x80-wide block covers all of them. Output is deliberately simple and consistent; the learned
# rewrite table (stage 02) maps its systematic deviations onto the S1 spellings.
_INDIC_LO, _INDIC_HI = 0x0900, 0x0D7F
_INDIC_RE = re.compile("[ऀ-ൿ]")

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_SCRIPT_CONS = {  # (block base, offset) overrides
    (0x0980, 0x70): "r", (0x0980, 0x71): "w",     # Assamese ra / wa
    (0x0B00, 0x71): "w",                          # Oriya wa
}
_DEAD = {  # consonants that carry no inherent vowel
    0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k",  # Malayalam chillu
}
_SCRIPT_DEAD = {(0x0980, 0x4E): "t"}             # Bengali khanda ta
_INDEP = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x60: "ri", 0x61: "li",
}
_SIGN = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
    0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li",
    0x63: "li",
}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}   # candrabindu, anusvara, visarga, tippi
_SCRIPT_NASAL = {(0x0B80, 0x03): "h"}
_NUKTA = {"k": "q", "j": "z", "ph": "f", "d": "r", "dh": "rh", "g": "g", "kh": "kh"}
_VIRAMA, _NUKTA_CP = 0x4D, 0x3C
_DANDA = (0x64, 0x65)


def has_indic(text):
    return _INDIC_RE.search(text) is not None


def transliterate_indic(text):
    """Rule-based Brahmic -> Latin. Inherent 'a' is dropped at word end (schwa deletion)."""
    out = []
    pending_a = False        # last consonant still carries its inherent vowel
    last_cons = -1
    for ch in text:
        cp = ord(ch)
        if _INDIC_LO <= cp <= _INDIC_HI:
            base, off = cp & ~0x7F, cp & 0x7F
            key = (base, off)
            if base == 0x0A00 and off == 0x71:          # Gurmukhi addak: gemination, no sound
                continue
            if key in _SCRIPT_DEAD or (off in _DEAD and key not in _SCRIPT_CONS):
                if pending_a:
                    out.append("a")
                out.append(_SCRIPT_DEAD.get(key) or _DEAD[off])
                pending_a = False
            elif key in _SCRIPT_CONS or (off in _CONS and key not in _SCRIPT_NASAL):
                if pending_a:
                    out.append("a")
                out.append(_SCRIPT_CONS.get(key) or _CONS[off])
                last_cons = len(out) - 1
                pending_a = True
            elif off in _SIGN:
                out.append(_SIGN[off])
                pending_a = False
            elif off == _VIRAMA:
                pending_a = False
            elif off == _NUKTA_CP:
                if last_cons >= 0:
                    out[last_cons] = _NUKTA.get(out[last_cons], out[last_cons])
            elif off in _INDEP:
                if pending_a:
                    out.append("a")
                out.append(_INDEP[off])
                pending_a = False
            elif key in _SCRIPT_NASAL or off in _NASAL:
                if pending_a:
                    out.append("a")
                pending_a = False
                out.append(_SCRIPT_NASAL.get(key) or _NASAL[off])
            elif 0x66 <= off <= 0x6F:
                pending_a = False
                out.append(chr(ord("0") + off - 0x66))
            elif off in _DANDA:
                pending_a = False
                out.append(" ")
            # everything else (avagraha, length marks, rare signs) is dropped
        elif cp in (0x200C, 0x200D):
            continue
        else:
            pending_a = False
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- basic layer
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_ADDR_PUNCT = re.compile(r"[^a-z0-9 ]+")
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*)\.([a-z]{2,6})/?$")
PLACEHOLDERS = {"", "<null>", "null", "none", "nan", "n/a", "na", "-", "--", "unknown",
                "not available", "<none>", "nil", "n.a.", "<na>"}
PLACEHOLDER_PARTS = {"null", "none", "nan", "na", "n a", "unknown", "not available", "nil"}


def to_ascii_lower(text):
    if "&" in text:
        text = html.unescape(text)
    if text.isascii():
        return text.lower()
    if has_indic(text):
        text = transliterate_indic(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().encode("ascii", "ignore").decode("ascii")


def _squash(text):
    return " ".join(text.split())


def basic_name(raw):
    """-> (name_basic, flags)"""
    s = raw.strip()
    if not s:
        return "", 0
    flags = 0
    indic = has_indic(s)
    if indic:
        flags |= F_INDIC_NAME
    elif not s.isascii():
        flags |= F_ACCENTED
    if s.isupper() and len(s) >= 4:
        flags |= F_ALLCAPS
    t = to_ascii_lower(s).strip()
    m = _DOMAIN_RE.match(t) if " " not in t else None
    if m:
        flags |= F_DOMAIN
        t = m.group(1).split(".")[0].replace("-", " ")
    t = t.replace("&", " and ")
    t = t.replace(".", "").replace("'", "").replace("`", "")   # l.l.c -> llc, pvt. -> pvt
    return _squash(_NON_ALNUM.sub(" ", t)), flags


def basic_address(raw):
    """-> (addr_basic with parts joined by '|', indic_parts joined by '|', flags)"""
    s = raw.strip()
    if s.lower() in PLACEHOLDERS:
        return "", "", F_ADDR_EMPTY | (F_ADDR_PLACEHOLDER if s else 0)
    flags = 0
    parts, indic_parts = [], []
    for raw_part in s.split(","):
        indic = has_indic(raw_part)
        if indic:
            flags |= F_INDIC_ADDR
        p = to_ascii_lower(raw_part).replace("&", " and ")
        p = _squash(_ADDR_PUNCT.sub(" ", p))
        if not p or p in PLACEHOLDER_PARTS:
            if p:
                flags |= F_ADDR_PLACEHOLDER
            continue
        parts.append(p)
        if indic:
            indic_parts.append(p)
    if not parts:
        flags |= F_ADDR_EMPTY
    return "|".join(parts), "|".join(indic_parts), flags


def country_key(country):
    return " ".join(country.strip().lower().split())


# --------------------------------------------------------------------------- canonical layer
LEGAL_CANON = {
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pvte": "pvt",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pc": "pc",
    "pllc": "pllc", "plc": "plc", "opc": "opc",
}
NAME_STOP = {"and", "the", "of"}

STREET_CANON = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "avn": "ave", "road": "rd",
    "drive": "dr", "drv": "dr", "boulevard": "blvd", "blv": "blvd", "lane": "ln", "court": "ct",
    "place": "pl", "trail": "trl", "circle": "cir", "highway": "hwy", "parkway": "pkwy",
    "pkway": "pkwy", "terrace": "ter", "square": "sq", "crossing": "xing", "expressway": "expy",
    "freeway": "fwy", "point": "pt", "mount": "mt", "heights": "hts", "junction": "jct",
    "center": "ctr", "centre": "ctr", "plaza": "plz", "cove": "cv", "creek": "crk",
    "ridge": "rdg", "valley": "vly", "hollow": "holw", "alley": "aly", "bypass": "byp",
    "causeway": "cswy", "estates": "ests", "extension": "ext", "extn": "ext", "gardens": "gdns",
    "garden": "gdn", "grove": "grv", "harbor": "hbr", "landing": "lndg", "manor": "mnr",
    "meadows": "mdws", "route": "rte", "station": "sta", "turnpike": "tpke", "village": "vlg",
    "sector": "sec", "nagar": "ngr", "colony": "col", "marg": "marg", "chowk": "chowk",
    "cross": "crs", "main": "main", "layout": "lyt", "phase": "ph", "block": "blk",
}
DIRECTION_CANON = {
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
}
UNIT_DESIGNATORS = {
    "apartment", "apt", "unit", "suite", "ste", "fl", "floor", "rm", "room", "no", "nos",
    "number", "num", "door", "hno", "h", "flat", "plot", "shop", "house", "bldg", "building",
    "office", "dno", "khasra", "kh",
}
LANDMARK_WORDS = {"near", "nr", "opp", "opposite", "behind", "beside", "adjacent", "next"}
GENERIC_ADDR = (set(STREET_CANON) | set(STREET_CANON.values()) | set(DIRECTION_CANON)
                | set(DIRECTION_CANON.values()) | UNIT_DESIGNATORS | LANDMARK_WORDS
                | {"and", "the", "of", "to", "at", "po", "box", "dist", "district", "city", "town",
                   "post", "taluk", "tq", "via"})

_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr", "guam": "gu", "virgin islands": "vi", "american samoa": "as",
}
_IN_STATES = {
    "andaman and nicobar islands": "an", "andaman and nicobar": "an", "andhra pradesh": "ap",
    "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chandigarh": "ch",
    "chhattisgarh": "cg", "chattisgarh": "cg", "ct": "cg", "dadra and nagar haveli": "dn",
    "daman and diu": "dd", "dadra and nagar haveli and daman and diu": "dn", "delhi": "dl",
    "new delhi": "dl", "nct of delhi": "dl", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jammu and kashmir": "jk", "jharkhand": "jh", "karnataka": "ka",
    "kerala": "kl", "ladakh": "la", "lakshadweep": "ld", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "or": "od", "puducherry": "py", "pondicherry": "py",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "ts": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "ut": "uk", "west bengal": "wb",
}


def _with_codes(table):
    out = dict(table)
    for code in set(table.values()):
        out.setdefault(code, code)
    return out


# Known alias tables (normalisation only). Keys are country_key() values.
STATE_ALIASES = {"us": _with_codes(_US_STATES), "india": _with_codes(_IN_STATES)}

_HNUM_RE = re.compile(r"^0*(\d+)([a-z]{0,2})$")
_ORDINAL = {"st", "nd", "rd", "th"}


def house_number(token):
    """'0123' -> '123', '12a' -> '12'; ordinals ('9th') and non-numbers -> None."""
    m = _HNUM_RE.match(token)
    if not m or m.group(2) in _ORDINAL:
        return None
    return m.group(1) or "0"


class Canonicalizer:
    """Applies the training-learned rewrite tables plus the fixed canonicalisation rules."""

    def __init__(self, rewrite_map=None):
        rewrite_map = rewrite_map or {}
        self.name_rewrite = rewrite_map.get("name_tokens", {})
        self.state_parts = rewrite_map.get("state_parts", {})   # {country_key: {part: state}}

    # ---- names
    def canon_name(self, name_basic, flags):
        toks = name_basic.split()
        if flags & F_INDIC_NAME and self.name_rewrite:
            toks = [self.name_rewrite.get(t, t) for t in toks]
            toks = " ".join(toks).split()   # a rule may map to a multi-word value
        canon, legal, is_legal = [], [], []
        for i, t in enumerate(toks):
            c = LEGAL_CANON.get(t)
            if c is None and t == "public" and i + 1 < len(toks) and LEGAL_CANON.get(toks[i + 1]) == "ltd":
                c = "public"
            canon.append(c or t)
            is_legal.append(c is not None)
            if c:
                legal.append(c)
        legal_front = bool(toks) and is_legal[0] and not all(is_legal)
        core = [t for t, lg in zip(canon, is_legal) if not lg and t not in NAME_STOP]
        if not core:
            core = [t for t, lg in zip(canon, is_legal) if not lg] or canon
        return canon, core, " ".join(sorted(set(legal))), legal_front

    # ---- addresses
    def canon_address(self, addr_basic, country):
        """-> dict(tokens, hnums, skeys, state, idwords, landmark, sorted_str)"""
        parts = addr_basic.split("|") if addr_basic else []
        ck = country_key(country)
        hard = STATE_ALIASES.get(ck, {})
        learned = self.state_parts.get(ck, {})
        state = ""
        for j in range(len(parts) - 1, -1, -1):
            v = hard.get(parts[j]) or learned.get(parts[j])
            if v:
                state = v
                del parts[j]
                break
        tokens, hnums, skeys, idwords = [], [], [], []
        seen = set()
        landmark = False
        for part in parts:
            ptoks = []
            for t in part.split():
                if t in LANDMARK_WORDS:
                    landmark = True
                    continue
                if t in UNIT_DESIGNATORS:
                    continue
                ptoks.append(STREET_CANON.get(t) or DIRECTION_CANON.get(t) or t)
            nums = [house_number(t) for t in ptoks]
            for k, t in enumerate(ptoks):
                n = nums[k]
                if n is not None:
                    t = n
                    if n not in hnums:
                        hnums.append(n)
                    if k + 1 < len(ptoks) and nums[k + 1] is None and len(skeys) < 3:
                        sk = n + " " + ptoks[k + 1]
                        if sk not in skeys:
                            skeys.append(sk)
                elif t.isalpha() and len(t) >= 3 and t not in GENERIC_ADDR and t not in idwords:
                    idwords.append(t)
                if t not in seen:
                    seen.add(t)
                    tokens.append(t)
        return {"tokens": tokens, "hnums": hnums, "skeys": skeys, "state": state,
                "idwords": idwords, "landmark": landmark, "sorted_str": " ".join(sorted(tokens))}

    def state_of_parts(self, addr_basic, country):
        """State of an S1 address (used by stage 02 to learn Indic state-part aliases)."""
        parts = addr_basic.split("|") if addr_basic else []
        hard = STATE_ALIASES.get(country_key(country), {})
        for p in reversed(parts):
            if p in hard:
                return hard[p]
        return parts[-1] if parts else ""
