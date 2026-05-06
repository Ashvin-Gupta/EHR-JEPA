"""
Offline script: encode MEDS vocabulary codes into text embeddings.

This script is run ONCE before training when using embedding_type='text_based'.
It:
  1. Loads the vocabulary (built from the training split).
  2. Translates each code string to a natural language description using
     MEDSCodeTranslator.
  3. Batch-encodes descriptions with a clinical language model
     (ClinicalBERT, BioBERT, or any HuggingFace encoder).
  4. Saves the resulting embedding matrix to disk as a .pt file.
     Shape: (vocab_size, hidden_dim)  — includes the UNK row.

The saved file is then referenced via config.code_embeddings_path.

Usage
-----
    python scripts/encode_text_embeddings.py \
        --vocab_path       /path/to/vocab.json \
        --output_path      /path/to/code_embeddings.pt \
        --model_name       emilyalsentzer/Bio_ClinBERT \
        --batch_size       64 \
        --labitems_file    /path/to/d_labitems.csv.gz \
        --diagnoses_file   /path/to/d_icd_diagnoses.csv.gz \
        --procedures_file  /path/to/d_icd_procedures.csv.gz
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data.code_translator import MEDSCodeTranslator
from data.vocab import UNK_TOKEN, Vocab


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using the attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


def encode_descriptions(
    descriptions: list[str],
    model_name: str,
    batch_size: int = 64,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Encode a list of text descriptions using a HuggingFace encoder.

    Returns a FloatTensor of shape (len(descriptions), hidden_dim).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    for start in tqdm(range(0, len(descriptions), batch_size), desc="Encoding"):
        batch_texts = descriptions[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        embeddings = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Encode vocabulary codes into text embeddings.")
    parser.add_argument("--vocab_path", required=True, help="Path to vocab.json")
    parser.add_argument("--output_path", required=True, help="Output .pt file path")
    parser.add_argument(
        "--model_name",
        default="emilyalsentzer/Bio_ClinBERT",
        help="HuggingFace model name for encoding",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda', 'cuda:0', etc.")
    parser.add_argument("--labitems_file", default=None)
    parser.add_argument("--diagnoses_file", default=None)
    parser.add_argument("--procedures_file", default=None)
    args = parser.parse_args()

    print(f"Loading vocabulary from {args.vocab_path}")
    vocab = Vocab.load(args.vocab_path)
    print(f"  Vocabulary size: {vocab.vocab_size} (including UNK)")

    print("Loading code translator...")
    translator = MEDSCodeTranslator.from_csv_files(
        labitems_file=args.labitems_file,
        diagnoses_file=args.diagnoses_file,
        procedures_file=args.procedures_file,
    )

    # Build description list in index order (UNK gets an empty string → zero vec)
    idx_to_code = vocab.idx_to_code
    descriptions = []
    for idx in range(vocab.vocab_size):
        code = idx_to_code.get(idx, UNK_TOKEN)
        if code == UNK_TOKEN:
            descriptions.append("")
        else:
            descriptions.append(translator.translate(code) or code)

    print(f"Encoding {vocab.vocab_size} descriptions with {args.model_name} ...")
    embeddings = encode_descriptions(
        descriptions,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Zero out the UNK row (EventEmbedding will replace it with the mean at load time)
    unk_idx = vocab.unk_idx
    embeddings[unk_idx] = 0.0

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(embeddings, args.output_path)
    print(f"Saved embeddings tensor {tuple(embeddings.shape)} to {args.output_path}")


if __name__ == "__main__":
    main()
