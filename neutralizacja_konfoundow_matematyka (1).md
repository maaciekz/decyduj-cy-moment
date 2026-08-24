# Neutralizacja konfoundów na embeddingach — matematyka

## Jak wykryć i usunąć artefakt (hole punch) z reprezentacji obrazu, nie psując sygnału

> Kontekst: zdjęcia FSA "killed" mają fizyczną dziurkę w negatywie (czarna kropka). Dziurka jest **skorelowana z etykietą** (rejected). Bez interwencji model nauczyłby się wykrywać dziurkę zamiast decydującego momentu. Ten dokument wyjaśnia matematykę neutralizacji.

---

## Spis treści

1. [Dlaczego konfound skorelowany z etykietą jest groźny](#1-dlaczego-konfound-skorelowany-z-etykietą-jest-groźny)
2. [Detekcja dziurki — klasyczne przetwarzanie obrazu](#2-detekcja-dziurki--klasyczne-przetwarzanie-obrazu)
3. [Embedding jako punkt w przestrzeni](#3-embedding-jako-punkt-w-przestrzeni)
4. [Kierunek konfoundu — arytmetyka wektorów](#4-kierunek-konfoundu--arytmetyka-wektorów)
5. [Struktura zmiany — lokalna czy rozproszona](#5-struktura-zmiany--lokalna-czy-rozproszona)
6. [Projekcja ortogonalna — usuwanie kierunku](#6-projekcja-ortogonalna--usuwanie-kierunku)
7. [Weryfikacja — czy nie psujemy sygnału](#7-weryfikacja--czy-nie-psujemy-sygnału)
8. [Ograniczenia i warstwy obrony](#8-ograniczenia-i-warstwy-obrony)
9. [Od dziurki do ogólnego pipeline'u](#9-od-dziurki-do-ogólnego-pipelineu)

---

## 1. Dlaczego konfound skorelowany z etykietą jest groźny

Mamy pary chosen (wybrane) vs rejected (killed). Killed negatywy mają hole punch:

```
hole punch  ⟹  rejected   (prawie zawsze)
brak dziurki ⟹  chosen    (prawie zawsze)
```

Model uczący się chosen/rejected wybiera **najłatwiejszą** cechę dyskryminacyjną. Czarne koło w kadrze to pojedyncza, wysokokontrastowa, lokalna plama — najłatwiejsza możliwa cecha. Model nauczy się jej w pierwszej epoce i nigdy nie spojrzy na kompozycję czy moment.

Efekt: "oś decydującego momentu", która w rzeczywistości jest "osią obecności dziurki". Wysokie d', świetna separacja — i całkowicie fałszywy wynik. To ta sama pułapka co konfound spektralny w projekcie AI/Real, ale groźniejsza, bo cecha jest dosłownie namalowana na obrazie.

**Dlatego konfound trzeba zneutralizować zanim zaczniemy szukać prawdziwego sygnału.**

---

## 2. Detekcja dziurki — klasyczne przetwarzanie obrazu

Dziurka ma charakterystyczne cechy fizyczne: bardzo ciemny (prawie czarny), wypełniony, w miarę okrągły obszar, otoczony jaśniejszą treścią, wewnątrz kadru (nie na czarnej ramce filmu). To pozwala wykryć ją **bez ML**, klasycznym CV.

### Algorytm

1. **Progowanie**: piksele o jasności < 30 (prawie czarne) → maska binarna
2. **Usunięcie ramki**: zerujemy margines ~12% z każdej strany (tam jest perforacja filmu)
3. **Domknięcie morfologiczne**: wypełnia drobne dziury, łączy fragmenty
4. **Analiza konturów**: dla każdego ciemnego obszaru liczymy:
   - **circularity** = 4π·area / perimeter² (1.0 = idealne koło)
   - **fill** = area / (π·r²) — ile koła otaczającego jest wypełnione
5. **Decyzja**: dziurka = circularity > 0.6 AND fill > 0.55 AND rozsądny promień

### Dlaczego to działa niezawodnie

Hole punch jest geometrycznie wyjątkowy — żaden naturalny element zdjęcia (cień, obiekt) nie jest jednocześnie tak ciemny, tak okrągły i tak wypełniony. Na zdjęciach testowych: circularity ~0.88, fill ~0.92 — bardzo wysoka pewność.

Wynik detekcji: pozycja (cx, cy) i promień dziurki → maska do inpaintingu lub do projekcji.

### Inpainting — jak działa rekonstrukcja dziurki

Mając maskę, usuwamy dziurkę przez **inpainting** — rekonstrukcję brakującego fragmentu na podstawie otoczenia. Użyliśmy metody Telea (FMM, Fast Marching Method) z OpenCV.

Mechanika: algorytm wypełnia dziurę **od brzegu do środka**, warstwa po warstwie (jak zamarzająca woda od krawędzi). Dla każdego pikselna do wypełnienia liczy **ważoną średnią** już znanych pikseli w małym sąsiedztwie, gdzie wagi zależą od trzech rzeczy:
- **bliskości** — bliższe piksele ważniejsze
- **kierunku gradientu** — zachowanie i przedłużanie krawędzi
- **odległości od brzegu** dziury

Kluczowe: algorytm propaguje nie tylko kolor, ale i **gradient** — dlatego potrafi przedłużyć krawędź lub teksturę wchodzącą w dziurę, zamiast zostawić płaską plamę. Na zdjęciach świń łata wtopiła się w tło (ciało schodziło z góry, ziemia wchodziła z dołu).

Parametr `inpaintRadius` (u nas 5) to promień sąsiedztwa do czerpania informacji: mały = ostrzejsza lokalna rekonstrukcja, duży = gładsza ale może rozmazać.

**Ograniczenie**: inpainting świetnie radzi sobie z gładkim tłem i prostą teksturą, ale gdyby dziurka leżała na czymś złożonym (twarz, tekst), rekonstrukcja byłaby zmyślona — algorytm nie wie, co tam "naprawdę" było. Dla naszych dziurek (zwykle na tle/ciele/ziemi) działa dobrze, ale zostawia **subtelny ślad**. To właśnie powód, dla którego projekcja embeddingu jest potrzebna jako druga warstwa — na wypadek tego resztkowego śladu.

---

## 3. Embedding jako punkt w przestrzeni

Model wizyjny (CLIP, VAE encoder) zamienia obraz w **wektor** — punkt w przestrzeni d-wymiarowej (CLIP ViT-B/32: d=512). Podobne obrazy → bliskie punkty.

Kluczowa własność: **kierunki w tej przestrzeni odpowiadają konceptom**. To słynne `king − man + woman ≈ queen` z word2vec, ale działa też dla obrazów (InterFaceGAN: kierunki "uśmiech", "wiek", "okulary" w latent space twarzy).

Nasza hipoteza: skoro koncepty mają kierunki, to **"obecność dziurki" też ma swój kierunek** — oś, wzdłuż której obrazy z dziurką są przesunięte względem tych bez. Jeśli znajdziemy ten kierunek, możemy go odjąć.

### Materiał: ten sam obraz, dwie wersje

Klucz do izolacji konfoundu: weź **ten sam obraz** z dziurką i bez (przez inpainting). Wtedy jedyna różnica między embeddingami to dziurka — wszystko inne (treść, kompozycja, jasność) jest identyczne.

```
e_orig  = embedding(obraz z dziurką)
e_clean = embedding(obraz po inpaintingu)
różnica = e_orig − e_clean   ≈ "co dziurka dodaje do reprezentacji"
```

---

## 4. Kierunek konfoundu — arytmetyka wektorów

Mając wiele par (orig, clean), liczymy **uśredniony kierunek dziurki**:

```
v_hole = mean(e_orig) − mean(e_clean)
```

To jeden wektor wskazujący "stronę dziurki" w przestrzeni embeddingów. Uśrednianie po wielu parach uśrednia przypadkowe różnice, zostawiając systematyczny efekt dziurki.

Normalizujemy do długości 1:
```
v̂_hole = v_hole / ||v_hole||
```

### Obserwacja, która buduje intuicję

Na prostym embedderze siatkowym (gdzie każdy wymiar = jasność komórki obrazu) najsilniejsze wymiary `v_hole` odpowiadają **komórkom w centrum obrazu** — dokładnie tam, gdzie fizycznie jest dziurka. Matematyka sama wskazuje miejsce konfoundu, choć nigdzie nie powiedzieliśmy jej, gdzie szukać. Kierunek "wie", gdzie jest dziurka.

To dowód, że `v_hole` faktycznie koduje dziurkę, a nie przypadkowy szum.

---

## 5. Struktura zmiany — lokalna czy rozproszona?

Naturalne pytanie: czy dziurka zmienia **każdy** element wektora, czy tylko **konkretne**? Odpowiedź ma praktyczne znaczenie dla wyboru metody neutralizacji.

### Wynik empiryczny (embedder siatkowy)

Na embedderze siatkowym (każdy wymiar = jasność komórki obrazu) dziurka zmienia **tylko kilka wymiarów**:
- 3-4 z 64 wymiarów istotnie zmienione (~5%)
- 55% całej zmiany skupione w 5 najsilniejszych wymiarach
- te wymiary odpowiadają komórkom w **centrum obrazu** — dokładnie tam, gdzie fizycznie jest dziurka

Zmiana jest więc **skupiona (lokalna)**, nie rozproszona po całym wektorze.

### Spójność kierunku — warunek konieczny neutralizacji

Najważniejsza miara:
```
cos(diff_zdjęcie1, diff_zdjęcie2) = 0.93
```

Wektor zmiany z **różnych zdjęć** wskazuje niemal ten sam kierunek. To znaczy, że dziurka zmienia te same wymiary **niezależnie od treści zdjęcia**. To jest warunek konieczny, by neutralizacja działała: gdyby dziurka zmieniała losowe wymiary za każdym razem, nie istniałby jeden `v_hole` do odjęcia.

Interpretacja progu:
- **cos > 0.7** → spójny kierunek → projekcja zadziała
- **cos < 0.3** → brak wspólnego kierunku → projekcja nie pomoże (sygnał, że konfound nie jest liniowo separowalny)

### Ważne zastrzeżenie — embedder siatkowy vs CLIP

Embedder siatkowy jest **z definicji lokalny** — jego wymiary to komórki obrazu, więc lokalna dziurka daje lokalną zmianę niemal tautologicznie. CLIP zachowa się inaczej:

- jego wymiary nie mają interpretacji przestrzennej (abstrakcyjne cechy z setek milionów obrazów)
- dziurka prawdopodobnie zmieni **więcej wymiarów** (czarne koło aktywuje wiele detektorów: kształt, kontrast, "obiekt na środku")
- ale wciąż z **dominującym, spójnym kierunkiem** (powtarzalny bodziec)

Czyli na CLIP zmiana będzie **bardziej rozproszona**, ale nadal skupiona w pewnej podprzestrzeni o wysokim cosinusie między zdjęciami.

### Konsekwencja dla metody — dlaczego projekcja, nie zerowanie wymiarów

To uzasadnia wybór narzędzia:
- **gdyby zmiana była skupiona w kilku wymiarach** → można by je po prostu wyzerować (prościej, ale ryzykowniej — te wymiary mogą nieść też sygnał)
- **gdyby rozproszona** (jak w CLIP) → zerowanie pojedynczych wymiarów nie zadziała; **trzeba projekcji kierunkowej**, bo konfound jest "rozmazany" jako jeden kierunek po wielu wymiarach

Dlatego projekcja ortogonalna (następna sekcja) jest właściwym, **ogólnym** narzędziem — działa niezależnie od tego, czy konfound jest lokalny czy rozproszony, byle miał spójny kierunek.

---

## 6. Projekcja ortogonalna — usuwanie kierunku

Mając kierunek `v̂_hole`, usuwamy go z dowolnego embeddingu `e`:

```
e_czyste = e − (e · v̂_hole) · v̂_hole
```

### Rozłożenie na czynniki

- `e · v̂_hole` — iloczyn skalarny = **rzut e na kierunek dziurki** = "ile dziurki jest w e"
- `(e · v̂_hole) · v̂_hole` — ten komponent jako wektor (wzdłuż osi dziurki)
- `e − (...)` — odejmujemy go, zostaje część **prostopadła** do dziurki

### Dlaczego to działa zawsze (geometria)

Gdy odejmiesz od wektora jego rzut na pewien kierunek, pozostałość jest **z definicji ortogonalna** do tego kierunku. Sprawdzenie: rzut `e_czyste` na `v̂_hole` wynosi dokładnie 0.

```
e_czyste · v̂_hole = (e − (e·v̂_hole)·v̂_hole) · v̂_hole
                   = e·v̂_hole − (e·v̂_hole)·(v̂_hole·v̂_hole)
                   = e·v̂_hole − (e·v̂_hole)·1
                   = 0
```

Po projekcji embedding nie zawiera **żadnej** informacji o pozycji wzdłuż osi dziurki. Konfound liniowo usunięty.

### Geometryczna intuicja

Wyobraź sobie przestrzeń jako pokój. Konfound (dziurka) to jeden kierunek — powiedzmy "góra-dół". Projekcja **spłaszcza pokój** wzdłuż tej osi: wszystkie obiekty rzutujemy na podłogę. Ruch góra-dół (dziurka) przestaje istnieć, ale pozycja na podłodze (cały prawdziwy sygnał: kompozycja, moment) zostaje nietknięta.

---

## 7. Weryfikacja — czy nie psujemy sygnału

Projekcja działa na **wszystkie** embeddingi, też te bez dziurki. Trzeba sprawdzić, że im nie szkodzi.

### Test 1 — orig zbliża się do clean

Po projekcji embedding z dziurką powinien być **bliższy** swojej wersji bez dziurki:
```
cos(e_orig, e_clean)       PRZED projekcją
cos(proj(e_orig), e_clean) PO projekcji   → powinno wzrosnąć
```

### Test 2 — clean prawie się nie zmienia

Embedding bez dziurki po projekcji powinien zmienić się minimalnie:
```
||proj(e_clean) − e_clean||  → mała wartość
```

Jeśli ta zmiana jest mała, projekcja jest "chirurgiczna" — wycina tylko oś dziurki, nie rusza obrazów, które jej nie mają.

### Test 3 — komponent dziurki wyzerowany

```
proj(e) · v̂_hole ≈ 0   dla każdego e
```

Bezpośredni dowód, że konfound zniknął.

---

## 8. Ograniczenia i warstwy obrony

### Projekcja usuwa tylko liniowy komponent

Jeśli dziurka wpływa na embedding **nieliniowo** (a w głębokich sieciach często tak jest), część sygnału zostaje. Dlatego łączymy trzy warstwy:

| Warstwa | Co robi | Kiedy |
|---|---|---|
| **Pomiar** | mierzy siłę konfoundu | zawsze najpierw |
| **Inpainting** | usuwa dziurkę z pikseli (u źródła) | gdy dziurka duża/częsta |
| **Projekcja** | usuwa resztkowy kierunek z embeddingów | jako pas bezpieczeństwa |

### Ryzyko nadmiernej korekty

Jeśli kierunek dziurki przypadkiem koreluje z czymś sensownym (np. dziurka zwykrywana w centrum, a centrum to też ważny obszar kompozycyjny), odejmując dziurkę odejmiemy trochę sygnału. Dlatego warto:
- liczyć `v_hole` z par tego-samego-obrazu (orig/clean), gdzie różnica to CZYSTO dziurka
- weryfikować, że projekcja nie obniża prawdziwej separacji (Test 2)

### Liczba par

Kierunek z 2 par jest niestabilny. Dla CLIP licz `v_hole` z 100-200 par orig/clean — im więcej, tym czystszy kierunek.

---

## 9. Od dziurki do ogólnego pipeline'u

Dziurka jest **widzialnym** przypadkiem testowym. Ten sam mechanizm działa na każdy konfound, którego kierunek umiemy oszacować:

- **Jasność** — killed mogą być średnio ciemniejsze (gorsza ekspozycja)
- **Kontrast / sépia** — inne starzenie filmu
- **Ramka filmu** — różne formaty kadru
- dowolny inny systematyczny artefakt

Ogólny pipeline ma cztery moduły:

1. **Pomiar** — zmierz, jak silny jest każdy konfound (czy warto ruszać)
2. **Detektor** — zidentyfikuj cechę (CV dla dziurki, statystyki dla jasności)
3. **Estymator kierunku** — policz `v_konfound` z par "z cechą / bez cechy"
4. **Neutralizator** — projekcja ortogonalna usuwająca kierunek(ki), z weryfikacją

Można usunąć **wiele** kierunków naraz (projekcja na ortogonalne dopełnienie podprzestrzeni rozpiętej przez kilka `v_konfound`). To pełna kontrola konfoundów.

### Kluczowa zasada

> Najpierw **zmierz**, potem **usuń**, zawsze **zweryfikuj**, że nie zniszczyłeś sygnału.

---

## Podsumowanie — trzy operacje do zapamiętania

```
1. KIERUNEK:   v = mean(e_z_cechą) − mean(e_bez_cechy)
2. PROJEKCJA:  e_czyste = e − (e · v̂) · v̂
3. WERYFIKACJA: proj(e) · v̂ ≈ 0   oraz   clean prawie bez zmian
```

Te trzy linijki to cała istota neutralizacji konfoundu na embeddingach. Reszta (detekcja, inpainting, wiele kierunków) to rozbudowa wokół tego rdzenia.

Dziurka nauczyła nas budować narzędzie, które ochroni projekt przed konfoundami — także tymi niewidocznymi.

---

*Dokument matematyczny. Projekt "Decydujący moment". Neutralizacja konfoundów na embeddingach.*
