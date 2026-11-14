"""
IndFin: retrieve WHO Global Health Observatory (GHO) indicators from plain-language queries.

Pipeline (run top to bottom; each "# %%" block is a notebook-style cell, so the file also
opens as a notebook in VS Code, Jupyter (jupytext) and Colab):

  1. Download GHO indicator names
  2. Split by indicator into train / held-out test
  3. Generate training queries with an open LLM (Qwen) -- needs a GPU
  4. Baseline evaluation of all-MiniLM-L6-v2
  5. Fine-tune with mined hard negatives
  6. Blend base and fine-tuned weights, choose the blend on dev data, evaluate on held-out data
  7. Demo, save locally, and (optionally) push to the Hugging Face Hub

Environment variables
  INDFIN_WORKDIR   output directory for intermediate files and saved models (default: ./indfin_outputs)
  HF_USER          Hugging Face username to push to (default: SallySims)
  HF_TOKEN         Hugging Face write token (only needed for the push step)
  INDFIN_PUSH=1    actually push to the Hub (default: skip)

Requires a CUDA GPU. See requirements.txt and README.md.
"""

# %%
import os

# Where intermediate files (paraphrases_local.json, saved models) are written.
WORKDIR = os.environ.get("INDFIN_WORKDIR", "indfin_outputs")
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

def get_secret(name):
    """Read a secret from Colab Secrets if available, otherwise from the environment."""
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Set the {name} environment variable.")
        return value

# %% [markdown]
# ## 1. Pull indicators from the WHO GHO API

# %%
import re, json, random, os
import requests, pandas as pd, torch
random.seed(42)

r = requests.get("https://ghoapi.azureedge.net/api/Indicator", timeout=60)
r.raise_for_status()
df = pd.DataFrame(r.json()["value"])[["IndicatorCode", "IndicatorName"]].dropna()
df["name"] = df["IndicatorName"].str.strip()
df = df.drop_duplicates("name").reset_index(drop=True)
names = df["name"].tolist()
print(len(df), "indicators")
print(df.sample(5, random_state=0))

# %% [markdown]
# ## 2. Split by indicator (no indicator appears in both train and test)

# %%
ids = list(range(len(df)))
random.shuffle(ids)
n_test = int(0.15 * len(ids))
test_ids, train_ids = ids[:n_test], ids[n_test:]
print(len(train_ids), "train indicators /", len(test_ids), "held-out indicators")

# %% [markdown]
# ## 3. Build training queries
# Training queries are paraphrases written by an open model (Qwen), in three styles. Progress is saved to
# `paraphrases_local.json` in the working directory, so if the run stops you can start it again and it resumes.
# Templates are not used: they taught an earlier version to match words instead of meaning.

# %%
from collections import defaultdict

def clean(n):
    n = re.sub(r"\s*\([^)]*\)", "", n)
    return re.sub(r"\s+", " ", n).strip(" ,;:")

def norm(q):
    return re.sub(r"\s+", " ", q.lower()).strip(" ?.!")

GEN_NAME = "Qwen/Qwen2.5-3B-Instruct"   # "Qwen/Qwen2.5-1.5B-Instruct" is ~2x faster but lower quality
PARA_FILE = "paraphrases_local.json"
N_PER_STYLE = 12
STYLES = {
    "researcher": "an academic researcher writing a precise data search",
    "everyday":   "a non-expert describing what they want in everyday words",
    "keywords":   "someone typing short search-engine keywords, sometimes with abbreviations",
}
codes = df["IndicatorCode"].tolist()

def make_prompt(i, style):
    return (
        "You help researchers search a catalogue of WHO health indicators.\n"
        f'Indicator: "{names[i]}"\n'
        f"Write {N_PER_STYLE} different search queries someone might type to find exactly this indicator, "
        f"written as {STYLES[style]}.\n"
        "Use different words from the indicator name where possible (synonyms, plain language) but keep the same meaning. "
        "Do not mention other indicators.\n\n"
        'Example indicator: "Under-five mortality rate (per 1000 live births)"\n'
        "Example queries:\n"
        "child death rates before age five\n"
        "how many children die young\n"
        "U5MR by country\n\n"
        'Example indicator: "Prevalence of anaemia in women of reproductive age (%)"\n'
        "Example queries:\n"
        "share of women with low haemoglobin\n"
        "anemia among women 15-49\n"
        "iron deficiency in women statistics\n\n"
        f"Now output exactly {N_PER_STYLE} queries, one per line, with no numbering and no other text.")

def parse(text, i):
    bad = {norm(names[i]), norm(clean(names[i]))}
    out = []
    for l in text.splitlines():
        l = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip().strip('"“”')
        if (5 <= len(l) <= 120 and not l.endswith(":") and norm(l) not in bad
                and not l.lower().startswith(("example", "indicator", "queries"))):
            out.append(l)
    return out[:N_PER_STYLE]

store = json.load(open(PARA_FILE)) if os.path.exists(PARA_FILE) else {}
jobs = [(i, s) for i in range(len(names)) for s in STYLES if f"{codes[i]}|{s}" not in store]
print(len(jobs), "generations to run,", len(store), "already saved")

if jobs:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(GEN_NAME, padding_side="left")
    try:
        lm = AutoModelForCausalLM.from_pretrained(GEN_NAME, dtype=torch.float16, device_map="cuda")
    except TypeError:
        lm = AutoModelForCausalLM.from_pretrained(GEN_NAME, torch_dtype=torch.float16, device_map="cuda")
    BS = 32
    for b in range(0, len(jobs), BS):
        batch = jobs[b:b + BS]
        prompts = [tok.apply_chat_template([{"role": "user", "content": make_prompt(i, s)}],
                                           tokenize=False, add_generation_prompt=True) for i, s in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = lm.generate(**enc, max_new_tokens=260, do_sample=True, temperature=0.8,
                              top_p=0.95, pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for (i, s), t in zip(batch, texts):
            store[f"{codes[i]}|{s}"] = parse(t, i)
        if (b // BS) % 20 == 0:
            json.dump(store, open(PARA_FILE, "w"))
            print(f"  {b}/{len(jobs)}")
    json.dump(store, open(PARA_FILE, "w"))
    del lm
    torch.cuda.empty_cache()

para = {i: [q for s in STYLES for q in store.get(f"{codes[i]}|{s}", [])] for i in range(len(names))}
print(sum(1 for v in para.values() if v), "of", len(names), "indicators have paraphrases")
for i in [j for j in train_ids if para[j]][:3]:
    print("\n" + names[i]); print("  ", para[i][:6])

# Training pairs: paraphrases + one cleaned-name pair per indicator. Drop duplicates and ambiguous queries.
def candidates(idx, with_name):
    out = []
    for i in idx:
        out += [(q, i) for q in para[i]]
        if with_name:
            out.append((clean(names[i]), i))
    return out

train_c, test_c = candidates(train_ids, True), candidates(test_ids, False)
owners = defaultdict(set)
for q, i in train_c + test_c:
    owners[norm(q)].add(i)

seen, train_pairs = set(), []
for q, i in train_c:
    k = norm(q)
    if len(owners[k]) == 1 and k not in seen:
        seen.add(k)
        train_pairs.append((q, names[i]))

# Synthetic eval: held-out indicators only, up to 5 paraphrases each, none seen in training
eval_pairs = []
for i in test_ids:
    pool = [q for q in para[i] if len(owners[norm(q)]) == 1 and norm(q) not in seen]
    eval_pairs += [(q, i) for q in random.sample(pool, min(5, len(pool)))]

print(f"\n{len(train_pairs)} unique train pairs / {len(eval_pairs)} synthetic eval queries")
print(random.sample(train_pairs, 12))

# %% [markdown]
# ## 4. Baseline evaluation (Recall@k over the full indicator list)

# %%
from sentence_transformers import SentenceTransformer

def recall(model, ks=(1, 5, 10)):
    c = model.encode(names, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
    q = model.encode([x for x, _ in eval_pairs], convert_to_tensor=True,
                     normalize_embeddings=True, batch_size=128)
    top = (q @ c.T).topk(max(ks), dim=1).indices.cpu()
    gold = torch.tensor([i for _, i in eval_pairs]).unsqueeze(1)
    hits = top == gold
    return {f"R@{k}": round(hits[:, :k].any(1).float().mean().item(), 4) for k in ks}

BASE = "sentence-transformers/all-MiniLM-L6-v2"
base_res = recall(SentenceTransformer(BASE))
print("base:", base_res)

# %% [markdown]
# ## 5. Fine-tune with hard negatives
# `EPOCHS`/`LR` are 2 epochs and 2e-5 because blending in section 6 limits any forgetting. Hard negatives are mined with the *base* model: for each query, the nearest wrong indicators that are not near-duplicates of the right one (`HN_SIM_MAX`), so valid alternatives are not treated as wrong.

# %%
from datasets import Dataset
from sentence_transformers import (SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses)
from sentence_transformers.training_args import BatchSamplers
import numpy as np

EPOCHS, LR = 2, 2e-5
USE_HARD_NEG = True
HN_SIM_MAX = 0.85     # a negative must be less similar than this to the correct indicator

base_model = SentenceTransformer(BASE)
name_emb = base_model.encode(names, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
name_sim = (name_emb @ name_emb.T).cpu().numpy()
name_idx = {n: i for i, n in enumerate(names)}

anchors, positives, negatives = [], [], []
if USE_HARD_NEG:
    q_emb = base_model.encode([q for q, _ in train_pairs], convert_to_tensor=True,
                              normalize_embeddings=True, batch_size=256)
    for s in range(0, len(train_pairs), 2048):
        top = (q_emb[s:s + 2048] @ name_emb.T).topk(30, dim=1).indices.cpu().tolist()
        for k, row in enumerate(top):
            q, p = train_pairs[s + k]
            pi = name_idx[p]
            cands = [j for j in row if j != pi and name_sim[pi, j] < HN_SIM_MAX]
            if cands:
                anchors.append(q); positives.append(p); negatives.append(names[random.choice(cands[:5])])
    train_ds = Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives}).shuffle(seed=42)
    print(len(train_ds), "triplets with hard negatives. Examples:")
    for k in random.sample(range(len(anchors)), 5):
        print(f"  Q: {anchors[k]}\n    +  {positives[k]}\n    -  {negatives[k]}")
else:
    train_ds = Dataset.from_dict({"anchor": [q for q, _ in train_pairs],
                                  "positive": [p for _, p in train_pairs]}).shuffle(seed=42)

model = SentenceTransformer(BASE)
loss = losses.MultipleNegativesRankingLoss(model)
args = SentenceTransformerTrainingArguments(
    output_dir="indicatorfinder-out",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=64,
    learning_rate=LR,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
)
SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss).train()

# %% [markdown]
# ## 6. Blend, choose the blend strength, and evaluate
# 1. The hand-written queries (plus any in `MY_QUERIES`) are split into **dev** (even positions) and **test** (odd positions).
# 2. For each blend strength alpha (0 = base model, 1 = fully fine-tuned) the notebook scores the synthetic held-out set and the dev half, and picks the best alpha.
# 3. The chosen model is judged on the **test half**, which was not used for the choice.
#
# A hand-written result counts as correct if the indicator name matches the query's regex. Matching is lenient, so check the printed notes, and review or replace the regexes.

# %%
import copy

HW = [
 ("child deaths from diarrhoea", r"(death|mortality|causes of death).*diarrh|diarrh.*(death|mortality)"),
 ("how many women get skilled birth care", r"skilled health personnel"),
 ("adult smoking rates", r"prevalence.*(smok|tobacco)|(smok|tobacco).*prevalence"),
 ("HIV infection rate among adults", r"HIV.*prevalence|prevalence.*HIV"),
 ("how many people get malaria each year", r"malaria.*incidence|incidence.*malaria"),
 ("tuberculosis new cases per 100000", r"tuberculosis.*incidence|incidence.*tuberculosis"),
 ("measles vaccination coverage in children", r"MCV|measles.*(immuni|vaccin|coverage)"),
 ("maternal death rate", r"maternal mortality"),
 ("average life expectancy", r"life expectancy"),
 ("how much do households pay out of pocket for health", r"out-of-pocket"),
 ("high blood pressure prevalence", r"hypertension|blood pressure"),
 ("obesity in adults", r"obes|body mass index|overweight"),
 ("alcohol consumption per person", r"alcohol.*(consumption|per capita)"),
 ("number of doctors per 10000 people", r"physicians|medical doctors"),
 ("hospital beds availability", r"hospital beds"),
 ("road accident deaths", r"road traffic"),
 ("suicide rate", r"suicide"),
 ("deaths caused by air pollution", r"air pollution"),
 ("access to safe drinking water", r"drinking.water"),
 ("access to toilets and sanitation", r"sanitation"),
 ("children who are too short for their age", r"stunt"),
 ("babies breastfed only for the first six months", r"exclusive.*breast|breastfe"),
 ("death of newborns in the first month", r"neonatal mortality"),
 ("teenage pregnancy birth rate", r"adolescent birth rate"),
 ("women using modern contraception", r"contracepti|family planning"),
 ("people receiving HIV treatment", r"antiretroviral"),
 ("nurses and midwives workforce", r"nurs"),
 ("child vaccination DTP3 coverage", r"DTP3|diphtheria"),
 ("wasting in children", r"wast"),
 ("cholera cases reported", r"cholera"),
 ("early deaths from chronic diseases", r"premature|dying between age 30"),
 ("cervical cancer screening", r"cervical cancer"),
 ("how common is diabetes", r"diabetes|blood glucose"),
 ("taxes on cigarettes", r"tax.*(cigarette|tobacco)|(cigarette|tobacco).*tax"),
 ("salt intake per person", r"salt|sodium"),
 ("people who do not exercise enough", r"physical(ly)? inactiv|physical activity"),
 ("psychiatrists and mental health staff", r"psychiatrist|psychologist|mental health (workers|professionals)"),
 ("treatment coverage for drug use disorders", r"substance use disorders?.*(treatment|coverage)|coverage.*substance"),
 ("caesarean section rate", r"caesarean|cesarean"),
 ("antenatal care visits during pregnancy", r"antenatal"),
 ("babies born underweight", r"low birth.?weight|birth weight"),
 ("HPV vaccine uptake", r"HPV|human papilloma"),
 ("polio cases", r"polio"),
 ("dengue fever cases", r"dengue"),
 ("insecticide treated bed nets", r"insecticide|bed net|ITN"),
 ("hepatitis B infection", r"hepatitis B"),
 ("COVID-19 deaths", r"COVID|coronavirus"),
 ("how many healthy years can people expect to live", r"healthy life expectancy|HALE"),
 ("infant mortality rate", r"infant mortality|infant deaths"),
 ("universal health coverage service index", r"universal health coverage|UHC"),
 ("government health spending per person", r"expenditure on health.*per capita|per capita.*expenditure on health"),
 ("number of pharmacists", r"pharmacist"),
 ("dentists per population", r"dentist"),
 ("clean cooking fuels in homes", r"clean.*(fuel|cooking)|household air pollution"),
 ("handwashing facilities at home", r"hand.?washing"),
 ("drowning deaths", r"drowning"),
 ("homicide rate", r"homicide"),
 ("violence by a partner against women", r"intimate partner|partner violence"),
 ("female genital cutting", r"genital mutilation|FGM"),
 ("girls married before 18", r"married|child marriage"),
 ("tobacco use among teenagers", r"(adolescent|youth|student).*(tobacco|smok)|(tobacco|smok).*(adolescent|youth|student)"),
 ("tuberculosis treatment success rate", r"treatment success"),
 ("malaria deaths", r"malaria.*(death|mortality)|(death|mortality).*malaria"),
 ("HIV testing coverage", r"HIV.*test|know.*HIV status"),
 ("premature births", r"preterm"),
 ("stillbirths", r"stillbirth"),
 ("tooth decay in children", r"caries|dental|tooth|oral health"),
 ("neglected tropical diseases", r"neglected tropical|NTD"),
 ("rabies", r"rabies"),
 ("leprosy cases", r"leprosy"),
]

# Add your own (query, regex of acceptable indicator names) here, e.g. ("my query", r"pattern")
MY_QUERIES = []

HW_ALL = HW + MY_QUERIES
valid = {q: {i for i, n in enumerate(names) if re.search(rx, n, re.I)} for q, rx in HW_ALL}
for q, v in valid.items():
    if not v:
        print("skipped (no indicator matches the regex):", q)
    elif len(v) > 25:
        print(f"note: {len(v)} indicators match the regex for: {q} (lenient)")
HW_USED = [q for q, _ in HW_ALL if valid[q]]
HW_DEV, HW_TEST = HW_USED[::2], HW_USED[1::2]
print(f"\n{len(HW_USED)} usable hand-written queries: {len(HW_DEV)} dev / {len(HW_TEST)} test")

def hw_score(m, qs, ks=(1, 3, 5)):
    c = m.encode(names, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
    qv = m.encode(qs, convert_to_tensor=True, normalize_embeddings=True)
    top = (qv @ c.T).topk(max(ks), dim=1).indices.cpu().tolist()
    return {f"hit@{k}": round(sum(any(i in valid[q] for i in t[:k]) for q, t in zip(qs, top)) / len(qs), 3)
            for k in ks}

def blend(base_m, ft_m, alpha):
    m = copy.deepcopy(ft_m)
    sd_b, sd_f = base_m.state_dict(), ft_m.state_dict()
    new = {k: ((sd_b[k].to(sd_f[k].device).float() * (1 - alpha) + sd_f[k].float() * alpha).to(sd_f[k].dtype)
               if sd_f[k].is_floating_point() else sd_f[k]) for k in sd_f}
    m.load_state_dict(new)
    return m

# Choose blend strength using synthetic eval + the DEV half only
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
rows = {}
for a in ALPHAS:
    m = base_model if a == 0 else blend(base_model, model, a)
    rows[a] = {"syn": recall(m), "dev": hw_score(m, HW_DEV)}
    print(f"alpha={a}: synthetic={rows[a]['syn']}  dev={rows[a]['dev']}")
best_alpha, best_score = 0.0, -1
for a in ALPHAS:   # ties go to the smaller alpha (closer to the base model)
    sc = rows[a]["dev"]["hit@1"] + rows[a]["dev"]["hit@5"] + rows[a]["syn"]["R@1"] + rows[a]["syn"]["R@5"]
    if sc > best_score + 1e-9:
        best_alpha, best_score = a, sc
print("\nchosen alpha:", best_alpha)
final_model = base_model if best_alpha == 0 else blend(base_model, model, best_alpha)

# Final comparison, judged on the TEST half
models = {"base (all-MiniLM-L6-v2)": base_model,
          "fine-tuned (alpha=1)": model,
          f"IndicatorFinder (alpha={best_alpha})": final_model}
syn_df = pd.DataFrame({k: recall(m) for k, m in models.items()}).T
test_df = pd.DataFrame({k: hw_score(m, HW_TEST) for k, m in models.items()}).T
all_df = pd.DataFrame({k: hw_score(m, HW_USED) for k, m in models.items()}).T
print("\nSynthetic held-out test:"); display(syn_df)
print(f"Hand-written TEST half ({len(HW_TEST)} queries):"); display(test_df)
print(f"Hand-written, all queries ({len(HW_USED)}), for reference (dev half was used to pick alpha):"); display(all_df)

# Which queries changed rank, base vs chosen model (rank of first correct result; None = not in top 5)
def ranks(m):
    c = m.encode(names, convert_to_tensor=True, normalize_embeddings=True)
    qv = m.encode(HW_USED, convert_to_tensor=True, normalize_embeddings=True)
    top = (qv @ c.T).topk(5, dim=1).indices.cpu().tolist()
    return {q: next((r + 1 for r, i in enumerate(t) if i in valid[q]), None) for q, t in zip(HW_USED, top)}
rb, rf = ranks(base_model), ranks(final_model)
print("\nQueries whose rank changed:")
for q in HW_USED:
    if rb[q] != rf[q]:
        print(f"  {q!r}: base {rb[q]} -> final {rf[q]}")

# Push gate: better on the (large) synthetic set, and no worse than base on the hand-written TEST half
# (allowing a one-query difference, since each query is worth 1/len(HW_TEST))
base_name, final_name = "base (all-MiniLM-L6-v2)", f"IndicatorFinder (alpha={best_alpha})"
tol = 1 / len(HW_TEST) + 1e-9
READY = (syn_df.loc[final_name, "R@1"] > syn_df.loc[base_name, "R@1"]
         and test_df.loc[final_name, "hit@1"] >= test_df.loc[base_name, "hit@1"] - tol
         and test_df.loc[final_name, "hit@5"] >= test_df.loc[base_name, "hit@5"] - tol)
print("\nREADY to push:", READY)

# %% [markdown]
# ### Bootstrap confidence intervals (run after section 6, in the same process)

# %%
def top1_hits(m):
    c = m.encode(names, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
    q = m.encode([x for x, _ in eval_pairs], convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
    gold = np.array([i for _, i in eval_pairs])
    return ((q @ c.T).argmax(1).cpu().numpy() == gold).astype(float), gold

hb, gold = top1_hits(base_model)
hf, _ = top1_hits(final_model)
groups = {}
for k, g in enumerate(gold):
    groups.setdefault(g, []).append(k)
keys, rng = list(groups), np.random.default_rng(0)
d = []
for _ in range(2000):
    idx = np.concatenate([groups[keys[s]] for s in rng.choice(len(keys), len(keys))])
    d.append(hf[idx].mean() - hb[idx].mean())
print("Synthetic R@1 gain: %.3f, 95%% CI [%.3f, %.3f]" % (hf.mean() - hb.mean(), *np.percentile(d, [2.5, 97.5])))

def hw_hit1(m, qs):
    c = m.encode(names, convert_to_tensor=True, normalize_embeddings=True)
    t = (m.encode(qs, convert_to_tensor=True, normalize_embeddings=True) @ c.T).argmax(1).cpu().tolist()
    return np.array([float(i in valid[q]) for q, i in zip(qs, t)])

a, b = hw_hit1(base_model, HW_TEST), hw_hit1(final_model, HW_TEST)
d2 = [(b[i] - a[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(5000))]
print("Hand-written test hit@1 gain: %.3f, 95%% CI [%.3f, %.3f]" % (b.mean() - a.mean(), *np.percentile(d2, [2.5, 97.5])))

CI_SYN = "R@1 gain %+.3f, 95%% CI [%+.3f, %+.3f] (paired bootstrap over indicators)" % (
    hf.mean() - hb.mean(), *np.percentile(d, [2.5, 97.5]))
CI_HW = "hit@1 difference vs base %+.3f, 95%% CI [%+.3f, %+.3f] (bootstrap over queries)" % (
    b.mean() - a.mean(), *np.percentile(d2, [2.5, 97.5]))
print("\nFor the model card:\n ", CI_SYN, "\n ", CI_HW)

# %% [markdown]
# ## 7. Demo: base vs final model, side by side (local, nothing is pushed yet)
# Read these results yourself. Judge whether the final model returns the indicators you would pick.

# %%
demo_qs = ["child deaths from diarrhoea", "how many women get skilled birth care", "adult smoking rates",
           "people living with HIV", "children who are too thin", "spending on health as share of GDP",
           "teenage tobacco use", "how many doctors are there", "malaria cases per year"]

def top3(m, qs):
    c = m.encode(names, convert_to_tensor=True, normalize_embeddings=True)
    res = {}
    for q in qs:
        s = (m.encode(q, convert_to_tensor=True, normalize_embeddings=True) @ c.T).topk(3)
        res[q] = [(float(v), names[int(i)]) for v, i in zip(s.values, s.indices)]
    return res

b, f = top3(base_model, demo_qs), top3(final_model, demo_qs)
for q in demo_qs:
    print("\n" + q)
    print("  BASE")
    for v, n in b[q]: print(f"    {v:.2f}  {n}")
    print(f"  FINAL (alpha={best_alpha})")
    for v, n in f[q]: print(f"    {v:.2f}  {n}")

# %% [markdown]
# ## 8. Save locally (nothing goes to Hugging Face)

# %%
final_model.save("indicatorfinder-minilm-v3")
model.save("indicatorfinder-minilm-v3-unblended")
json.dump({"train_pairs": train_pairs, "eval_pairs": [(q, names[i]) for q, i in eval_pairs],
           "best_alpha": best_alpha}, open("pairs_v3.json", "w"))
print("saved to", os.getcwd())

# %% [markdown]
# ## 9. Push to the Hugging Face Hub (optional)
# Skipped unless INDFIN_PUSH=1. Stops unless the final model passes the checks against the base model
# (READY is True); set FORCE_PUSH = True only after reading the demo output and deciding the model is good
# despite the numbers. Needs HF_TOKEN with write access and HF_USER set to your username.

# %%
FORCE_PUSH = False
DO_PUSH = os.environ.get("INDFIN_PUSH") == "1"

USER = os.environ.get("HF_USER", "SallySims")
MODEL_NAME = "indfin-gho-minilm"
DATA_NAME = "indfin-gho-pairs"
MODEL_ID = f"{USER}/{MODEL_NAME}"
DATA_ID = f"{USER}/{DATA_NAME}"
# Edit this if you reviewed or rewrote the hand-written queries yourself
HW_NOTE = "drafted with AI assistance (Claude) and not independently validated"
COMMIT_MSG = "Update: paraphrase training data, hard negatives, weight blending"

def push_to_hub():
    assert READY or FORCE_PUSH, "Final model does not pass the checks against the base model. Not pushing."
    assert os.path.exists("indicatorfinder-minilm-v3"), "Run the save step first."
    assert "CI_SYN" in globals(), "Run the bootstrap block first so the model card can report the intervals."

    from huggingface_hub import login, HfApi
    from datasets import Dataset, DatasetDict
    login(get_secret("HF_TOKEN"))
    api = HfApi()

    final_model.push_to_hub(MODEL_ID, exist_ok=True, commit_message=COMMIT_MSG)

    # Dataset (train and test must have the same columns)
    code_of = dict(zip(names, df["IndicatorCode"]))
    DatasetDict({
        "train": Dataset.from_dict({"query": [q for q, _ in train_pairs],
                                    "indicator_name": [p for _, p in train_pairs],
                                    "indicator_code": [code_of[p] for _, p in train_pairs]}),
        "test": Dataset.from_dict({"query": [q for q, _ in eval_pairs],
                                   "indicator_name": [names[i] for _, i in eval_pairs],
                                   "indicator_code": [df["IndicatorCode"][i] for _, i in eval_pairs]}),
    }).push_to_hub(DATA_ID, commit_message=COMMIT_MSG)

    # Model card
    lo_hw, hi_hw = np.percentile(d2, [2.5, 97.5])
    zero_note = (" The interval includes zero, so a top-1 difference from the base model cannot be established on this set."
                 if lo_hw <= 0 <= hi_hw else "")
    hn_note = (f"hard negatives mined with the base model (similarity cap {HN_SIM_MAX})"
               if USE_HARD_NEG else "in-batch negatives only")
    card = f"""---
library_name: sentence-transformers
base_model: {BASE}
pipeline_tag: sentence-similarity
tags: [sentence-transformers, retrieval, public-health, who-gho]
license: cc-by-nc-sa-3.0
datasets: [{DATA_ID}]
---
# IndFin (IndicatorFinder for WHO GHO indicators)

IndFin is a sentence-embedding model fine-tuned to match plain-language queries to WHO Global Health Observatory (GHO) indicator names.

## Intended use
Searching indicator *metadata* (names) to help researchers find the right GHO indicator. It does not contain or predict health data and must not be used for clinical decisions.

## Training
Base: `{BASE}`. Loss: MultipleNegativesRankingLoss ({EPOCHS} epochs, learning rate {LR}), {hn_note}. Pairs: {len(train_pairs)} unique (query, indicator name) pairs for {len(train_ids)} indicators; queries are paraphrases generated by `{GEN_NAME}` in three styles, plus the cleaned indicator name. {len(test_ids)} indicators were held out entirely. The released weights are a linear blend of the base and fine-tuned weights (alpha = {best_alpha}), with alpha chosen on a synthetic validation set and the dev half of the hand-written queries.

## Evaluation
Retrieval over all {len(names)} indicators. Higher is better; the base model is `all-MiniLM-L6-v2`.

**Synthetic held-out test** (generated paraphrases of held-out indicators; likely easier than real use, and generated by the same model family that wrote the training queries, so part of the gain may reflect that model's phrasing):

{syn_df.to_markdown()}

{CI_SYN}.

**Hand-written test half** ({len(HW_TEST)} queries, {HW_NOTE}; not used to choose alpha; a result counts as correct if its indicator name matches a pattern, so near-duplicate indicators all count):

{test_df.to_markdown()}

{CI_HW}.{zero_note}

## Limitations
- English only; indicator names only (no definitions); single snapshot of the GHO list.
- Training queries are machine-generated by a small model and may not match how researchers search.
- The hand-written test is small ({len(HW_TEST)} queries), so differences of one or two queries are not meaningful.
- The model can still return related but non-equivalent indicators. Example: for "adult smoking rates", a survey-metadata indicator about smokeless tobacco appears in the top three. Results vary somewhat between training runs.

## Data
Indicator names from the WHO GHO OData API. Check WHO's current terms of use before reuse.
"""
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=MODEL_ID, repo_type="model", commit_message=COMMIT_MSG)

    # Check: reload from the Hub and compare with the local model
    pushed = SentenceTransformer(MODEL_ID)
    print("Hub model  :", hw_score(pushed, HW_TEST), recall(pushed))
    print("Local model:", hw_score(final_model, HW_TEST), recall(final_model))
    print("https://huggingface.co/" + MODEL_ID)

if DO_PUSH:
    push_to_hub()
else:
    print("Skipping the Hugging Face push (set INDFIN_PUSH=1 to enable).")
