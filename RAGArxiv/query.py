"""index.py'nin diske yazdigi Chroma'yi acar, soru sorar, cevap uretir.

    python query.py "How does SORT link detections across frames?"
    python query.py --qid q001     # golden set'ten soru: top-k'da altin pasaj var mi, isaretler
    python query.py --qid q041 --k 10
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from index import CHROMA_DIR, COLLECTION, EMBEDDING_MODEL  # indeksle AYNI ayarlar

GOLDEN_SET = Path.home() / "Projects" / "rag-lab" / "eval" / "golden_set.jsonl"
LLM_MODEL = "gpt-3.5-turbo-0125"  # kursun modeli

# Kursun prompt'u aynen ("rlm/rag-prompt").
PROMPT = ChatPromptTemplate.from_messages([(
    "human",
    "You are an assistant for question-answering tasks. Use the following "
    "pieces of retrieved context to answer the question. If you don't know "
    "the answer, just say that you don't know. Use three sentences maximum "
    "and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context} \n"
    "Answer:",
)])


def load_question(qid: str) -> dict:
    with GOLDEN_SET.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["id"] == qid:
                return q
    raise SystemExit(f"golden set'te yok: {qid}")


def format_docs(docs) -> str:
    # Kurstan tek fark: her chunk'in basina adresi. Model hangi pasajdan okudugunu gorur,
    # Faz 1'in atif dogrulugu deneyi bunun ustune kurulacak.
    return "\n\n".join(f"[{d.metadata['ref']}]\n{d.page_content}" for d in docs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="serbest soru")
    parser.add_argument("--qid", help="golden set soru id'si, orn. q001")
    parser.add_argument("--k", type=int, default=5, help="kac chunk getirilsin")
    args = parser.parse_args()
    if not (args.question or args.qid):
        parser.error("bir soru ya da --qid ver")

    load_dotenv()

    gold = None
    if args.qid:
        q = load_question(args.qid)
        question, gold = q["question"], set(q["gold_passages"])
        print(f"[{q['id']} · {q['kind']}] {question}\n")
    else:
        question = args.question

    # 1) INDEKSI AC — embed yok, diskten okunur. Soru ise indeksle ayni modelle embed edilir.
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )

    # 2) GETIR — kursta retriever chain'in icindeydi, ne getirdigi gorunmuyordu.
    #    Burada ayri cagiriyoruz ki cevaptan once neyi getirdigini gorelim.
    #    Skor = mesafe: KUCUK = daha benzer.
    results = vectorstore.similarity_search_with_score(question, k=args.k)
    docs = [d for d, _ in results]

    print(f"top-{args.k}:")
    for rank, (d, score) in enumerate(results, 1):
        ref = d.metadata["ref"]
        mark = "✅" if gold and ref in gold else "  "
        print(f" {mark} {rank}. {score:.3f}  {ref:<24} {d.metadata['section'][:40]}")

    if gold is not None:
        found = gold & {d.metadata["ref"] for d in docs}
        if gold:
            print(f"\naltin pasaj top-{args.k} icinde: {len(found)}/{len(gold)}")
        else:
            print("\nnegatif soru: corpus'ta cevabi yok, dogru cevap 'bilmiyorum'")

    # 3) URET — kursun chain'inin retriever'siz hali: context'i zaten biz verdik.
    chain = PROMPT | ChatOpenAI(model=LLM_MODEL) | StrOutputParser()
    answer = chain.invoke({"question": question, "context": format_docs(docs)})
    print(f"\ncevap:\n{answer}")

    if args.qid and q["expected_answer"]:
        print(f"\nbeklenen:\n{q['expected_answer']}")


if __name__ == "__main__":
    main()
