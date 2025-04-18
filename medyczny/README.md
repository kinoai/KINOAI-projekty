# Projekt Segmentacji Medycznej

## Opis projektu
Projekt dedykowany segmentacji medycznej.

## Początkowe cele
Na poczatku projekt skupiał się na reprodukcji bardzo prostej sprawy - porównanie działąnia sieci Unetowych na segmentacje obrazów CT raka wątroby (LITS benchmark) z działaniem sieci Vnetowych.

## Stan obecny
- Skrypt do konwersji z dicoma na nifti z całym chrzanieniem się które to potencjalnie włącza
- Skrypciki od trenowania Vnetów vs Unetów
- Skrypt od robienia porównań

## Kierunki rozwoju
Projekt preferencyjnie powinien w tym punkcie zrobić branching - off, tj. obrać jakiś bardziej światły cel.

Obecnie głównym kandydatem do tej roli jest **interpretowalność medycznych modeli**.

Co by to włączało:
- Trenowanie SAE (sparse autoencoders), na modłę interpretowalnośći w LLMach
- Zaadaptowanie metod do segmentacji obrazu
- Korzystanie z innych metod z pola zaadaptowanych na obrazki
- Testowanie, czy modele do segmentacji medycznej z przodujących papierów faktycznie generalizują, czy też nie


Natomiast z małych rzeczy to dobrze by było porobić więcej eksperymentów (tj. po pierwsze poprzestawiać hiperparametry na każdą stronę i porobić z tego wykresiki, po drugie na czymś innym niż wątroba/rak wątroby) i jakoś to co jest ładnie opisać, czy to w formie posterowej, czy to w formie paperu, czy jakiegoś referatu.
## Ogarniacze
- **Główny ogarniacz #1**: Maciej Cichoń (.grandad na Discordzie)
- **Główny ogarniacz #2**: Natalia Błaszczyk (blatalia na Discordzie)
