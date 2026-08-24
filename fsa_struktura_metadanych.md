# FSA/OWI — struktura metadanych i strategia parowania

## Notatka techniczna o tym, jak rozróżnić "wybrane" od "killed" i jak budować pary

---

## 1. Jak LoC oznacza zaakceptowane vs odrzucone

Z oficjalnej dokumentacji Library of Congress wynikają trzy sygnały rozróżniające klatki wydrukowane (wybrane przez Strykera) od "killed" (odrzucone):

### Sygnał główny — tytuł "Untitled"

Odrzucone obrazy sklasyfikowano jako "killed". Ponieważ wielu z nich brakowało karty z podpisem, większość ma w katalogu po prostu tytuł "Untitled". Oficjalna instrukcja LoC: wyszukaj "Untitled", by zobaczyć obrazy, które nie miały podpisu i przypuszczalnie nie zostały wydrukowane.

Konkretny wzorzec dla untitled z roboczym tytułem:
```
Untitled photo, possibly related to: [tytuł wydrukowanej klatki-rodzica]
```

Ten wzorzec jest złotem — bezpośrednio linkuje killed do jego printed odpowiednika.

### Sygnał uzupełniający — hole punch

We wczesnych fazach projektu (1935–1939) "killed" negatywy oznaczano fizyczną dziurką przebitą w negatywie. Później tej praktyki zaniechano. W metadanych LoC można je znaleźć, wyszukując "hole punch". Uwaga: to oznacza tylko podzbiór killed (wczesne lata), więc jest sygnałem częściowym. Informacja zwykle w polu opisu/notes itemu, nie jako osobna flaga w metadanych z listy.

### Mechanizm łączenia — sąsiednie call numbers

Najważniejsze dla parowania. LoC wprost opisuje metodę identyfikacji untitled: szukaj obrazów, które mają **sąsiednie call numbers**, są **podobne wizualnie**, i **mają tytuły**. Funkcja "Browse neighboring items by call number" naśladuje contact sheet — pokazuje sąsiednie klatki tej samej sesji.

---

## 2. Struktura metadanych w dumpie z list

Dump przez endpoint listy (100 rek./żądanie) daje 41 kolumn. Kluczowe pola do parowania:

| Pole | Zawartość | Przydatność |
|---|---|---|
| `shelf_id` | `FSA/OWI COLL - C 6257 [item] [P&P]` | **Klucz parowania** — seria + numer |
| `title` | tytuł lub "Untitled photo, possibly related to: X" | **Sygnał killed + link do rodzica** |
| `item` | zagnieżdżony JSON z `call_number`, `created_published`, `notes` | Dodatkowe metadane |
| `image_url` | link do obrazu | Do pobrania przez IIIF |
| `id` | URL itemu | Identyfikator unikalny |
| `description` | format fizyczny ("1 photographic print...") | Mało przydatne do parowania |
| `subject` | tematy | Pomocnicze do walidacji |
| `location_*` | miasto, stan, hrabstwo, kraj | Pomocnicze do grupowania |

### Parsowanie shelf_id

Wzorzec regex:
```python
import re
m = re.search(r'FSA/OWI COLL\s*-\s*([A-Z]+)\s*(\d+)', shelf_id)
series, num = m.group(1), int(m.group(2))  # np. ('C', 6257)
```

Serie literowe (C, D, E, F, G, H, J...) odpowiadają różnym typom filmu/agencji. W obrębie jednej serii numer rośnie wzdłuż katalogu — sąsiednie numery to zazwyczaj klatki tej samej sesji.

---

## 3. Dlaczego pierwsze 1000 rekordów nie dało par

Diagnoza problemu z początkową próbką:

- Tylko **85/1000** to faktyczne rekordy "FSA/OWI COLL". Reszta: "BIOG FILE" (portrety samych fotografów), inne serie OWI, materiały nabyte z zewnątrz.
- W tych 85 — **zero untitled**. To początek katalogu, gdzie trafiły wyłącznie wydrukowane, otytułowane zdjęcia.
- Killed/untitled (ok. 100 000 z 270 000 zrobionych zdjęć) są rozłożone w **pełnym** katalogu 175k, nie w pierwszym tysiącu.

Wniosek: parowanie wymaga **pełnego dumpu**. Na próbce początkowej nie ma materiału do parowania.

### Pułapka — false positive "killed"

W tytułach pojawia się słowo "killed", ale to często "**s-killed** worker" (wykwalifikowany robotnik) w wojennych zdjęciach przemysłowych — NIE "killed negative". Detekcja killed musi opierać się na "Untitled" / "possibly related" / "hole punch", nie na samym słowie "killed" w tytule.

---

## 4. Strategia parowania (do implementacji na pełnym CSV)

Trzy komplementarne sygnały:

### Sygnał A — "possibly related to" (najsilniejszy)

Untitled klatka mówi wprost: "Untitled photo, possibly related to: [X]". Szukamy printed klatki, której tytuł = X. Bezpośredni link killed→printed podany przez LoC.

```python
# pseudokod
if 'possibly related to:' in untitled_title:
    parent_desc = extract_after('possibly related to:', untitled_title)
    # znajdź printed o tytule pasującym do parent_desc
```

### Sygnał B — sąsiedztwo shelf_id

Dla klatek bez "possibly related": grupuj po serii, szukaj sąsiednich numerów (±1-3) w tej samej serii. Printed (z tytułem) + sąsiednia untitled = para.

```python
# pseudokod
for series in all_series:
    sort by num
    for printed_frame:
        find untitled neighbors within ±3 num in same series
```

### Sygnał C — filtr wizualny CLIP (następny krok)

Eliminuje fałszywe pary (różne sceny z tej samej rolki). Cosine similarity embeddingu CLIP między printed a untitled. Zostaw pary o wysokim podobieństwie (ta sama scena, inny moment).

### Ground truth

- **Pozytyw** (decydujący moment): klatka titled = wybrana przez Strykera do druku
- **Negatyw** (prawie): sąsiednia untitled/killed z tej samej sceny

---

## 5. Niuans naukowy — co Stryker faktycznie wybierał

Ważne dla uczciwej interpretacji wyników (w duchu metodologicznej szczerości z projektu AI/Real).

Stryker nie wybierał czysto za "decydujący moment" w sensie Cartier-Bressona. Obserwacja z badań nad kolekcją: miał tendencję do "zabijania" zdjęć z uśmiechniętymi ludźmi — w zestawach tych samych tematów wybierał chwile poważne zamiast wesołych. Jego kryteria obejmowały:

- **Redakcyjną narrację** — patos, powagę, "amerykański duch", rugged individualism
- **Przydatność dokumentacyjną / propagandową** — pasowanie do celów FSA/OWI
- **Jakość techniczną** — ostrość, ekspozycja
- **Wartość kompozycyjną i emocjonalną**

Konsekwencja: sygnał "wybrane vs killed" częściowo koduje **redakcyjną/narracyjną preferencję Strykera**, nie czysty decydujący moment. To trzeba nazwać w raporcie — operacyjna definicja brzmi: "klatka wybrana przez redaktora FSA vs odrzucona z tej samej sceny", co jest proxy dla decydującego momentu obciążonym redakcyjną wizją.

Mimo to: jest to wciąż najlepszy dostępny sygnał, znacznie czystszy niż ikoniczne-vs-random (eliminuje konfaundy epoki, filmu, fotografa, sceny — bo para pochodzi z tej samej sesji).

---

## 6. Następne kroki

1. **Dokończyć pełny dump 175k** (notebook `fsa_dump_metadanych.ipynb`) — bez tego brak materiału do parowania.
2. **Parownik offline** na kompletnym CSV — sygnały A + B, z miejscem na C (CLIP).
3. **Walidacja wizualna** — przejrzeć próbkę par, ocenić jakość (ta sama scena? widoczna różnica momentu?).
4. **Filtr CLIP** — automatyczne odsianie fałszywych par.
5. **Faza 1 probing** — d', decisive-direction na czystych parach.

---

*Notatka techniczna do projektu "Decydujący moment". Struktura danych FSA/OWI i metodologia parowania.*
