"""arXiv corpus'unu indeksler: corpus.jsonl -> Document -> chunk -> embedding -> Chroma (diske).

Bir kez calisir. Soru sormak query.py'nin isi; o, burada diske yazilan Chroma'yi acar.

    python index.py --dry-run      # API'ye gitmez: sayilar + tahmini maliyet
    python index.py --limit 5      # ilk 5 makale, boru hattini denemek icin
    python index.py                # tum corpus
"""

import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

CORPUS = Path.home() / "Projects" / "rag-lab" / "data" / "docs" / "corpus.jsonl"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "arxiv"
EMBEDDING_MODEL = "text-embedding-3-small"  # query.py AYNI modeli kullanmak zorunda
PRICE_PER_1M_TOKENS = 0.02  # text-embedding-3-small, USD (tahmin icin)
BATCH = 500  # Chroma'ya tek seferde eklenen chunk sayisi


# 1) YUKLE — kursta WebBaseLoader'in yaptigi is
def load_documents(limit: int | None = None) -> list[Document]:
    """Her paragraf bir Document.

    metadata["ref"], golden set'teki altin pasaj adresiyle ayni formatta
    ("2609.08265#S1.p2"). Donen chunk'in dogru pasaj olup olmadigini buna bakarak anlariz.
    """
    docs = []
    with CORPUS.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            paper = json.loads(line)
            aid, title = paper["arxiv_id"], paper["title"]

            docs.append(Document(
                page_content=paper["abstract"],
                metadata={"ref": f"{aid}#abstract", "title": title,
                          "section": "Abstract", "kind": "abstract"},
            ))
            for sec in paper["sections"]:
                for par in sec["paragraphs"]:
                    docs.append(Document(
                        page_content=par["text"],
                        metadata={"ref": f"{aid}#{par['id']}", "title": title,
                                  "section": sec["title"], "kind": par["kind"]},
                    ))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="ilk N makale")
    parser.add_argument("--dry-run", action="store_true", help="API'ye gitme, sadece say")
    args = parser.parse_args()

    load_dotenv()

    docs = load_documents(args.limit)

    # 2) BOL — kursun ayari aynen. Paragraflarin cogu 1000 karakterden kisa, bolunmez.
    #    add_start_index: uzun paragraf bolunurse her parcanin paragraf icindeki konumu.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                              add_start_index=True)
    chunks = splitter.split_documents(docs)

    n_chars = sum(len(c.page_content) for c in chunks)
    est_tokens = n_chars / 4  # Ingilizce metinde kaba kural: ~4 karakter = 1 token
    print(f"makale: {args.limit or 'hepsi'} | paragraf: {len(docs)} | chunk: {len(chunks)}")
    print(f"~{est_tokens / 1e6:.2f}M token | tahmini maliyet: ${est_tokens / 1e6 * PRICE_PER_1M_TOKENS:.3f}")
    if args.dry_run:
        return

    # 3) GOM + YAZ — kursta Chroma.from_documents tek satirdi, bellekte kaliyordu.
    #    Burada persist_directory ile diske yaziyoruz ki query.py tekrar embed etmesin.
    #    Indeks her calistirmada sifirdan kurulur: yarim kalmis eski indeksle karismaz.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )

    # Parti parti ekle: ilerleme gorunur, bir hata olursa hangi partide oldugu belli olur.
    # id = ref + paragraf icindeki konum -> ayni chunk iki kez eklenmez.
    for start in tqdm(range(0, len(chunks), BATCH), desc="embedding"):
        batch = chunks[start:start + BATCH]
        ids = [f"{c.metadata['ref']}@{c.metadata['start_index']}" for c in batch]
        vectorstore.add_documents(batch, ids=ids)

    print(f"bitti: {len(chunks)} chunk -> {CHROMA_DIR}")


if __name__ == "__main__":
    main()
