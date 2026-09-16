# LLM Projeleri

LangChain, RAG ve embedding konularını öğrenirken yazdığım deneme kodları ve küçük projeler.

## Klasörler

| Klasör | İçerik |
|---|---|
| `VectorStore/` | Chroma vector store ile temel benzerlik araması örneği. |
| `RAGIntro/` | Tek bir blog yazısı üzerinde basit RAG zinciri (yükle → böl → indeksle → cevapla). |
| [`CorrectiveRAGProject/`](CorrectiveRAGProject/) | LangGraph ile Corrective RAG (CRAG): soru yönlendirme, doküman değerlendirme, Tavily web araması ve hallucination/cevap kontrolü. |

## Kurulum

Her proje kendi sanal ortamını kullanıyor:

```bash
cd VectorStore
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ortam değişkenleri

Her klasördeki `.env.example` dosyasını `.env` olarak kopyalayıp kendi anahtarlarınızı girin:

```bash
cp .env.example .env
```

`.env` dosyaları `.gitignore` içinde — API anahtarları repoya girmez.
