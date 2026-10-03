"""Tiny synthetic dataset with the competition schema, for smoke tests only (never mixed with real data)."""
from __future__ import annotations

import random
from pathlib import Path

WORDS = ["meridian", "cedar", "summit", "lotus", "surya", "orion", "harbor", "maple", "apex", "zenith",
         "crystal", "golden", "river", "bright", "sunrise", "silver", "royal", "prime", "nova", "aditya"]
KINDS = ["consulting", "construction", "traders", "marketing", "dental", "realty", "foods", "logistics"]
LEGAL = {"US": ["LLC", "Inc", "Corp"], "India": ["Private Limited", "Ltd", "LLP"]}
STREETS = ["Main Street", "Oak Avenue", "MG Road", "Park Lane", "Station Road", "Elm Drive"]
CITIES = {"US": [("Phoenix", "AZ"), ("Duluth", "MN"), ("Austin", "TX")],
          "India": [("Kolkata", "West Bengal"), ("Pune", "Maharashtra"), ("Delhi", "Delhi")]}
DEVANAGARI = {"lotus": "लोटस", "surya": "सूर्या", "traders": "ट्रेडर्स", "Ltd": "लि.", "marketing": "मार्केटिंग"}


def _entity(rng, country, name_pool):
    core = rng.choice(name_pool)
    name = f"{core.title()} {rng.choice(KINDS).title()} {rng.choice(LEGAL[country])}"
    city, state = rng.choice(CITIES[country])
    addr = f"{rng.randint(1, 9999)} {rng.choice(STREETS)}, {city}, {state}"
    return name, addr


def _noisy(rng, name, addr, country):
    n = name
    r = rng.random()
    if r < 0.2:
        n = n.upper()
    elif r < 0.35 and country == "India":
        n = " ".join(DEVANAGARI.get(w.lower(), DEVANAGARI.get(w, w)) for w in n.split())
    elif r < 0.5:
        w = list(n)
        i = rng.randrange(len(w))
        w[i] = rng.choice("aeiou")
        n = "".join(w)
    a = addr
    r = rng.random()
    if r < 0.1:
        a = ""
    elif r < 0.2:
        parts = a.split(", ")
        parts.insert(1, rng.choice(["NULL", "null", "N/A"]))
        a = ", ".join(parts)
    elif r < 0.4:
        a = a.upper().replace("STREET", "ST").replace("AVENUE", "AVE")
    return n, a


def make_split(out: Path, split: str, n_s1: int, seed: int):
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    pool = WORDS[:8]  # small pool -> many same-name twins at different addresses
    eid = lambda p: f"{p}-{rng.randint(10 ** 6, 10 ** 9 - 1)}"
    for _ in range(n_s1):
        country = rng.choice(["US", "India"])
        name, addr = _entity(rng, country, pool)
        sid = eid("S1")
        s1.append((sid, name, addr, country))
        links = []
        for _ in range(rng.choice([0, 1, 2, 3, 3, 4])):
            src = rng.choice([2, 3])
            tid = eid(f"S{src}")
            n, a = _noisy(rng, name, addr, country)
            (s2 if src == 2 else s3).append((tid, n, a, country))
            links.append(tid)
        gt.append((sid, ",".join(links)))
    for _ in range(n_s1):  # distractors: businesses absent from S1
        country = rng.choice(["US", "India"])
        name, addr = _entity(rng, country, WORDS)
        src = rng.choice([2, 3])
        (s2 if src == 2 else s3).append((eid(f"S{src}"), *_noisy(rng, name, addr, country), country))
    rng.shuffle(s2)
    rng.shuffle(s3)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    for k, rows in ((1, s1), (2, s2), (3, s3)):
        with open(out / f"{split}_source{k}.tsv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(header)
            fh.writelines("\t".join(r) + "\n" for r in rows)
    if split == "train":
        with open(out / "train_ground_truth.tsv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("source1_entity_id\tmatched_entity_ids\n")
            fh.writelines(f"{a}\t{b}\n" for a, b in gt)


def make_dataset(root: Path, n_train=600, n_test=300):
    make_split(root / "train", "train", n_train, 1)
    make_split(root / "test", "test", n_test, 2)


def make_tiny_model(out: Path, dataset: Path):
    """Random-init 2-layer BERT with a character WordPiece vocab built from the synthetic text (offline)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    chars = set()
    for p in dataset.rglob("*.tsv"):
        chars |= set(p.read_text(encoding="utf-8"))
    chars |= set("nameaddress:")
    chars -= {"\t", "\n", " "}
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(chars) + sorted("##" + c for c in chars)
    vocab = {t: i for i, t in enumerate(dict.fromkeys(vocab))}
    out.mkdir(parents=True, exist_ok=True)
    t = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                     special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")
    tok.save_pretrained(out)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=256)
    BertModel(cfg).save_pretrained(out)
