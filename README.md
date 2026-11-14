# IndFin

IndFin retrieves WHO Global Health Observatory (GHO) indicators from plain-language queries.
It is a small sentence-embedding model (all-MiniLM-L6-v2, 22M parameters) fine-tuned on
machine-generated queries, with hard negatives and base/fine-tuned weight blending.

- Model: https://huggingface.co/SallySims/indfin-gho-minilm
- Dataset: https://huggingface.co/SallySims/indfin-gho-pairs

## Results (preliminary)

| Test | Base | IndFin | Difference |
| --- | --- | --- | --- |
| Synthetic held-out queries, Recall@1 | 0.302 | 0.361 | +5.8 points (95% CI 4.0 to 7.6) |
| Hand-written test half (34 queries), hit@1 | 0.853 | 0.882 | +2.9 points (95% CI -5.9 to +11.8) |

The synthetic queries come from the same model family that wrote the training queries, so that gain
may partly reflect its phrasing. The hand-written set is small, and its interval includes zero, so no
top-1 difference can be established there. The hand-written queries and their relevance patterns were
drafted with AI assistance and not independently validated. Results vary slightly between training runs.

## Run it

Needs a CUDA GPU (developed and run as a notebook on a Google Colab T4).

```bash
pip install -r requirements.txt
python indfin.py
```

`indfin.py` is written in percent format (`# %%` blocks), so it also opens as a notebook in VS Code,
Jupyter (via jupytext) and Colab. Run it top to bottom.

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `INDFIN_WORKDIR` | Where paraphrases, saved models and pairs are written | `./indfin_outputs` |
| `HF_USER` | Hugging Face username to push to | `SallySims` |
| `HF_TOKEN` | Hugging Face write token (push step only) | none |
| `INDFIN_PUSH` | Set to `1` to push the model, dataset and card to the Hub | skipped |

Generating the training queries is the slow step. Progress is saved to `paraphrases_local.json`, so an
interrupted run resumes. The push step stops unless the final model passes the checks against the base
model, and it needs `HF_TOKEN` and `HF_USER` for your own account. Never commit a token.

## Data

Indicator names come from the WHO GHO OData API. Check WHO's current terms of use before redistributing
the names or the derived dataset.

## Limitations

English only; indicator names only (no definitions); one snapshot of the GHO list; machine-generated
training queries; small hand-written evaluation set; related but non-equivalent indicators can still rank
near the top (for example "adult smoking rates").

## AI-use disclosure

Qwen2.5-3B-Instruct generates the training and synthetic test queries. The hand-written queries and
patterns, and parts of this code, were drafted with AI assistance (Claude, Anthropic). [Edit to match
what you did.]

## Licence and citation

[Add a licence for the code, for example MIT.] [Add a citation once the preprint is on arXiv.]
