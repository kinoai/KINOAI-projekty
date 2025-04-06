Projekt dedykowany segmentacji medycznej.
Na poczatku projekt skupiał się na reprodukcji bardzo prostej sprawy - porównanie działąnia sieci Unetowych na segmentacje obrazów CT raka trzustki i wątroby (LITS benchmark) z działaniem sieci Vnetowych. 
Implementacja Uneta była naturalnie basicowa, nie była ona pełnym nnUnetem, dopieszczonym do standardów SOTA.
Trochę było szarpania z formatami CT (.nii i ), ze zrobieniem jakiegoś pipelinu do ogarniania tych (i potencjalnie kiedyś większej ilości) danych.
Głównym powodem tego projektu było zapozanie się z działąniem segmentacji specyficznie w domenie medycznej.
Projekt preferencyjnie powinien w tym punkcie zrobić branching - off, tj. obrać jakiś bardziej światły cel.
Obecnie głównym kandydatem do tej roli jest interpretowalność medycznych modeli.
Co by to włączało: trenowanie SAE (sparse autoencoders), na modłe interpretowalnośći w LLMach,zaadaptowanych do segmentacji obrazu, jak i korzystanie z innych metod z pola zaadaptowanych na obrazki, w celu stestowania tego, czy modele do segmentacji medycznej z przodującyhc papierów faktycznie generalizują, czy też nie.

Główny ogarniacz: Maciej Cichoń (.grandad na Discordzie)
Główniejszy ogarniacz: Natalia Błaszczyk (blatalia na Discordzie)
