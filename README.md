# LLM Projeleri

LangChain, RAG, embedding ve Transformer mimarileri üzerine çalışırken yazdığım deneme kodları ve küçük projeler.

## Klasörler

| Klasör | İçerik |
|---|---|
| `VectorStore/` | Chroma vector store ile temel benzerlik araması örneği. |
| `RAGIntro/` | Tek bir blog yazısı üzerinde basit RAG zinciri (yükle → böl → indeksle → cevapla). |
| [`CorrectiveRAGProject/`](CorrectiveRAGProject/) | LangGraph ile Corrective RAG (CRAG): soru yönlendirme, doküman değerlendirme, Tavily web araması ve hallucination/cevap kontrolü. |
| `RAGArxiv/` | arXiv makaleleri üzerinde RAG: `index.py` corpus'u Chroma'ya indeksler, `query.py` soru sorar, `eval.py` 50 soruluk golden set ile retrieval'ı ölçer (`results/baseline.json`). |
| `VisionTransformer/` | PyTorch ile sıfırdan Vision Transformer (ViT), Jupyter notebook üzerinde. Veri seti Caltech101. |

## Kurulum

Her proje kendi sanal ortamını kullanıyor:

```bash
cd VectorStore
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`CorrectiveRAGProject/` [uv](https://docs.astral.sh/uv/) ile de kurulabilir: `uv sync`.

### VisionTransformer

Notebook'u çalıştırmak için ortamı Jupyter kernel'i olarak kaydedin:

```bash
cd VisionTransformer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name vision-transformer --display-name "Python (VisionTransformer)"
jupyter lab
```

Caltech101 veri seti ilk çalıştırmada `torchvision` tarafından `data/` klasörüne indirilir (~150 MB, repoya girmez). CUDA destekli bir GPU varsa otomatik kullanılır.


`
