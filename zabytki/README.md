# Projekt Segmentacji Fasad Architektonicznych

Celem niniejszego projektu jest wytworzenie modelu, potrafiącego poprawnie wysegmentować elementy fasad budynków w sposób podobny do SegmentAnything, z promptem będącym polskimi nazwami konkretnych architektonicznych elementów. Problem polega tutaj na tym, że chcemy, żeby model 'rozumiał' 1) nazwy po polsku 2) specyficzną nomenklaturę właściwą historii sztuki i architekturze.

Podejście opiera się o wyżej wspomniany model SegmentAnythig. Będziemy finetunowali / w inny sposób adaptowali jedną jego część, odpowiedzialną za przetwarzanie języka naturalnego na maski segmentacyjne.

Finalnie chcielibyśmy wytworzyć model zdolny do wytwarzania samodzielnie, bez ludzkieog inputu, ze zdjęcia fasady budynku jej pełen, formalny opis.

Obecnie jesteśmy na etapie zbierania i tworzenia datasetu, jak i również majstrowania przy SegmentAnything - trzeba tu dokładnie wybadać co i jak zfinetunować, żeby zrobić to minimalnym nakładem środków.

Później (najpewniej duuuużo później) zajmiemy się generowaniem na podstawie poprawnie wysegmentowanego zdjęcia opisu - są też inne pomysły na wykorzystanie tak jak wyżej opisane sfinetunowanego SegmentAnything - mamy całkiem dobrą relację z instytutem historii sztuki UŁ.

**Główny ogarniacz:** Maciej Cichoń

## Zalecane pozycje do przejrzenia

### Kluczowe Publikacje
0. **The annotated Transformer**
   -[Jak ktoś nie wie co to transformer czy jak działa to *koniecznie* do ogarnięcia](https://nlp.seas.harvard.edu/annotated-transformer/)

1. **Segment Anything Model (SAM)**
   - [Segment Anything](https://arxiv.org/abs/2304.02643)

2. **CLIP**
   - [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)


### Dodatkowe Materiały
- [Transformers in Vision: A Survey](https://arxiv.org/abs/2101.01169) - przegląd zastosowań transformerów w zadaniach wizyjnych
