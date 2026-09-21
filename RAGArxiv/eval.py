"""Baslangic olcumu: bugunku sistemi golden set'in 50 sorusunda calistirir. Cevap URETMEZ,
sadece retrieval'i olcer (50 soru embedding'i, maliyet ~0).

    python eval.py                 # tablo + results/baseline.json

Faz 1'deki her deney bu dosyanin cikardigi sayilari gecmeye calisacak.
"""

import json
import statistics as st
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from index import CHROMA_DIR, COLLECTION, EMBEDDING_MODEL
from query import GOLDEN_SET

KS = [1, 5, 10, 20]  # hepsi tek aramadan: en buyuk k ile bir kez getirip kesiyoruz
OUT = Path(__file__).parent / "results" / "baseline.json"


def recall_at_k(gold: set[str], retrieved_refs: list[str], k: int) -> float:
    """Altin pasajlarin kaci ilk k chunk'ta? Multihop'ta 2 altindan 1'i bulunduysa 0.5."""
    return len(gold & set(retrieved_refs[:k])) / len(gold)


def main() -> None:
    load_dotenv()
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )
    questions = [json.loads(l) for l in GOLDEN_SET.open(encoding="utf-8")]

    rows = []
    for q in questions:
        results = vectorstore.similarity_search_with_score(q["question"], k=max(KS))
        refs = [d.metadata["ref"] for d, _ in results]
        gold = set(q["gold_passages"])
        row = {
            "id": q["id"], "kind": q["kind"],
            "top1_distance": round(results[0][1], 4),
            # sonraki deneylerde "neden kacti" diye bakmak icin getirilenin kendisi
            "retrieved": [[d.metadata["ref"], round(score, 4)] for d, score in results],
            # her altin pasajin ilk goruldugu sira (1'den baslar), top-20'de yoksa None
            "gold_rank": {g: (refs.index(g) + 1 if g in refs else None) for g in sorted(gold)},
        }
        if gold:
            row.update({f"recall@{k}": recall_at_k(gold, refs, k) for k in KS})
        rows.append(row)

    # --- Recall tablosu: factual ve multihop ayri, cunku zorluklari farkli
    print(f"\n{'':10}{'n':>4}" + "".join(f"{'R@' + str(k):>8}" for k in KS))
    for kind in ["factual", "multihop"]:
        rs = [r for r in rows if r["kind"] == kind]
        print(f"{kind:10}{len(rs):>4}" + "".join(f"{st.mean(r[f'recall@{k}'] for r in rs):>8.2f}" for k in KS))
    pos = [r for r in rows if r["kind"] != "negative"]
    print(f"{'toplam':10}{len(pos):>4}" + "".join(f"{st.mean(r[f'recall@{k}'] for r in pos):>8.2f}" for k in KS))

    # --- Top-20'de bile bulunamayan altin pasajlar: k buyutmek bunlari kurtarmaz
    missed = [(r["id"], g) for r in pos for g, rank in r["gold_rank"].items() if rank is None]
    print(f"\ntop-20'de hic yok: {len(missed)} altin pasaj -> {', '.join(i for i, _ in missed)}")

    # --- Mesafe: negatif sorularin en yakin chunk'i pozitiflerden belirgin uzak mi?
    def dist(rs):
        d = sorted(r["top1_distance"] for r in rs)
        return f"min {d[0]:.3f} · medyan {st.median(d):.3f} · max {d[-1]:.3f}"
    neg = [r for r in rows if r["kind"] == "negative"]
    print("\ntop-1 mesafe (kucuk = daha benzer)")
    print(f"  pozitif (40): {dist(pos)}")
    print(f"  negatif (10): {dist(neg)}")
    neg_min = min(r["top1_distance"] for r in neg)
    overlap = sum(r["top1_distance"] >= neg_min for r in pos)
    print(f"  en yakin negatiften ({neg_min:.3f}) daha uzak pozitif: {overlap}/40")

    OUT.write_text(json.dumps({
        "config": {"chunking": "paragraf + recursive 1000/200 karakter", "embedding": EMBEDDING_MODEL,
                   "store": "chroma", "search": "vektor"},
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
