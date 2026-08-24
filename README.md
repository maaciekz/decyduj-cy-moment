# Czy sztuczna inteligencja rozumie „decydujący moment" w fotografii?

> Próba matematycznego uchwycenia pojęcia, którego przez sto lat nie udało się zamknąć w słowach.

---

## Pytanie, od którego zaczyna się projekt

W 1952 roku Henri Cartier-Bresson opisał to, co odróżnia zwykłe zdjęcie od takiego, które zostaje w pamięci:

> *„Jednoczesne rozpoznanie, w ułamku sekundy, znaczenia wydarzenia oraz precyzyjnej organizacji form, które nadają temu wydarzeniu właściwy wyraz."*

To jedno z najsłynniejszych zdań w historii fotografii i zarazem jedno z najbardziej nieuchwytnych. Fotograf zwykle „czuje", kiedy moment jest właściwy, ale rzadko potrafi zapisać to w formie przepisu. To wiedza w palcach, nie w słowach.

Projekt stawia pytanie, które brzmi prowokacyjnie: czy współczesne modele sztucznej inteligencji, które „widziały" miliardy zdjęć, nauczyły się tego pojęcia samodzielnie? A jeśli tak, czy da się to wykazać matematycznie?

To nie jest pytanie o to, czy AI zrobi ładne zdjęcie. To pytanie o to, czy w liczbowej reprezentacji obrazu, jaką posługuje się maszyna, istnieje wymierny ślad czegoś tak ulotnego jak decydujący moment.

---

## Kilka pojęć na początek

Zanim pójdziemy dalej, warto wyjaśnić cztery terminy, które wracają w całym opisie. Wszystkie mają prostą intuicję.

**Embedding (wektor reprezentacji).** Komputer nie widzi obrazu tak jak człowiek. Żeby cokolwiek z nim zrobić, zamienia zdjęcie na listę liczb, na przykład 512 liczb. Tę listę nazywa się embeddingiem. Można myśleć o niej jak o współrzędnych punktu w przestrzeni. Dwa zdjęcia podobne treściowo dostają punkty leżące blisko siebie, dwa różne, punkty odległe. Cała „wiedza" modelu o obrazie zawiera się w tym, gdzie ten punkt się znajduje.

**CLIP.** To model, który zamienia obrazy (i teksty) właśnie na takie embeddingi. Jego wyjątkowość polega na sposobie nauki: „oglądał" setki milionów par obraz i podpis z internetu, ucząc się układać obrazy o podobnym znaczeniu blisko siebie. Efekt jest taki, że przestrzeń embeddingów CLIP koduje pojęcia, nie tylko piksele. Co ważne, w tym projekcie CLIP nie jest dodatkowo trenowany na wstępnym etapie. Pytanie brzmi: czy „decydujący moment" już istnieje w jego gotowej przestrzeni.

**d' (miara separacji).** Kiedy mamy dwie grupy zdjęć i chcemy sprawdzić, jak wyraźnie są od siebie oddzielone, potrzebna jest liczba. Tą liczbą jest d'. Wartość 0 oznacza, że grupy są nierozróżnialne (całkowicie się nakładają). Im wyższe d', tym wyraźniejsza granica między grupami. Dla porównania: d' równe 0.2 to efekt subtelny, ale realny, a 0.8 to efekt duży. Ta sama miara bywa nazywana wielkością efektu i jest standardowym narzędziem w statystyce.

**Ranking.** Klasyfikator odpowiada na pytanie „czy to zdjęcie należy do grupy A czy B". Ranking odpowiada na inne pytanie: „które z tych zdjęć jest lepsze". Różnica jest praktyczna. Przy selekcji z sesji nie interesuje nas próg „dobre albo złe", tylko kolejność od najlepszego do najgorszego, bo chcemy wybrać na przykład dziesięć najlepszych klatek z czterystu.

---

## Pierwsza próba: generatory obrazów

Punktem wyjścia był prosty test. Najnowsze generatory obrazów (FLUX i inne) dostały polecenie stworzenia „decydującego momentu".

Obraz wygenerowany przez Nano Banana (prompty w katalogu PROPMTY/turysci.txt)
<img width="1408" height="768" alt="Gemini_Generated_Image_okvfodokvfodokvf" src="https://github.com/user-attachments/assets/f1f3194a-6acf-4dd0-8e26-d269b7f6f749" />


Obraz wygenerowany przez Flux (prompty w katalogu PROPMTY/Paryz.txt)
<img width="768" height="768" alt="generation-d23a9666-4026-4cdc-94f0-424a708a91b1" src="https://github.com/user-attachments/assets/4e6526bc-c1cf-4fca-89c8-ed85f5b77f5f" />


Wynik był pouczający. Modele odtwarzają powierzchnię: epokę, paletę barw, ziarno filmu, a nawet przyzwoitą kompozycję. Brakuje jednak elementu, który definiuje decydujący moment. 


---

## Problem zmiennych zakłócających

Wróćmy do poszukiwania sposobu, aby matematycznie udowodnic, ze decydujacy moment jest rozumialy przez AI. Naiwne podejście wygląda tak: wziąć zdjęcia ikoniczne (Cartier-Bresson, Magnum, World Press Photo) jako grupę „dobre", a losowe zdjęcia jako „zwykłe", i kazać modelowi znaleźć różnicę.

Ta metoda prowadzi do wyniku mylącego. Zobaczmy, czym różnią się obie grupy:

| Zdjęcia ikoniczne często mają | Zdjęcia zwykłe często mają |
|---|---|
| Czerń i biel | Kolor |
| Ziarno filmu | Cyfrową ostrość |
| Epokę 1930 do 1970 | Współczesność |
| **Decydujący moment** | **Przypadkowy moment** |

Model wybiera najłatwiejszą do wykrycia różnicę. Najłatwiejsza to „czarno białe plus ziarno plus vintage", a nie „decydujący moment". Wynikiem byłby detektor estetyki połowy XX wieku, mylnie uznany za detektor jakości momentu. Żadna metryka nie ostrzegłaby przed tym błędem, bo z punktu widzenia liczb zadanie zostałoby rozwiązane. Ten problem nosi nazwę problemu zmiennych zakłócających (konfoundów).

Ta obserwacja zdeterminowała cały projekt. Zamiast walczyć z konfoundami, kierunek badania przesunął się ku danym, w których konfoundy nie występują.

---

## Rozwiązanie ukryte w archiwum sprzed 90 lat

Odpowiedzią okazał się amerykański projekt dokumentalny FSA/OWI (1935 do 1944), rządowa misja fotografowania Wielkiego Kryzysu, dla której pracowali Dorothea Lange, Walker Evans, Gordon Parks i inni. Stąd pochodzi słynna „Migrant Mother".

Archiwum ma cechę, która czyni je dobrym laboratorium. Redaktor Roy Stryker przeglądał klatki i dziurkował negatywy, które odrzucał, dosłownie przebijając w nich otwór. Zostały więc dwa rodzaje zdjęć z tej samej sceny, często z tej samej minuty: te, które wybrał (chosen), i te, które odrzucił (rejected).

To rozwiązuje problem konfoundów. Skoro chosen i rejected pochodzą z tej samej sceny, tego samego fotografa, tej samej epoki i tego samego filmu, wszystkie zmienne zakłócające się znoszą. Zostaje jedyna różnica: redakcyjna decyzja. Oko Strykera.

> **Uwaga o prawach autorskich:** Projekt świadomie nie korzysta ze zbiorów Magnum czy World Press Photo, które są chronione prawem autorskim. Opiera się wyłącznie na FSA, należącym do domeny publicznej. Wiarygodność badania nie powinna stać na naruszeniu cudzych praw.

---

## Budowa zbioru: od 152 tysięcy do 14 tysięcy par

Z publicznego API Biblioteki Kongresu pobrano 152 404 rekordy metadanych. Przez parowanie klatek z tych samych scen i rygorystyczne filtrowanie (odrzucając pary zbyt różne lub będące duplikatami) powstał ostateczny, czysty zbiór:

14 266 par „wybrane vs odrzucone", każda z tej samej sceny.

Dodatkowo pary zostały podzielone według tego, jak bardzo klatki w nich się różnią: od „mikroruchu" (klatki niemal identyczne) po „wyraźne" (realnie inne ujęcie tej samej sceny). Ten podział okazał się później kluczowym narzędziem weryfikacji.

Roznice miedzy zdjeciami 0.95
<img width="817" height="1118" alt="image" src="https://github.com/user-attachments/assets/0bfc21b2-d6ee-4496-a579-b64c7462b8be" />

Roznice miedzy zdjeciami 0.8
<img width="757" height="1161" alt="image" src="https://github.com/user-attachments/assets/efe7614e-76c3-4bdf-b107-7a95c32c54d0" />


Roznice miedzy zdjeciami 0.6
<img width="768" height="1152" alt="image" src="https://github.com/user-attachments/assets/3e952ea8-0843-498b-ad0d-1c2cc6e64abf" />

---

## Odkrycie: oś istnieje

Każde zdjęcie zostało zamienione na embedding za pomocą CLIP. Następnie padło pytanie: czy istnieje w tej przestrzeni kierunek, wzdłuż którego zdjęcia wybrane różnią się od odrzuconych.

Odpowiedź brzmi: tak. Surowy wynik był jednak podejrzanie dobry.

Okazało się, że część „sygnału" pochodziła z fizycznej dziurki wybitej w odrzuconych negatywach. Model, zgodnie z przewidywaniami, znalazł najłatwiejszą cechę: nauczył się rozpoznawać otwór, a nie jakość zdjęcia. Pomiar to potwierdził wprost. Sama dziurka dawała siłę sygnału d'=0.204, podczas gdy całość surowego sygnału wynosiła d'=0.240. Innymi słowy, bez oczyszczenia powstałby detektor perforacji zamiast modelu oka Strykera.

Artefakty techniczne (dziurka, ostrość, jasność, kontrast) zostały usunięte metodą matematyczną, z pomiarem każdego z nich przed usunięciem, aby wiadomo było, co dokładnie znika. Po oczyszczeniu został prawdziwy sygnał:

**d'=0.188**, słaby, ale solidny, blisko ośmiokrotnie silniejszy niż przypadkowy szum.

Najmocniejszy dowód, że jest to sygnał treściowy, a nie kolejny artefakt, płynie z podziału par. Im bardziej klatki w parze się różniły, tym wyraźniejszy był sygnał. Dla par niemal identycznych oś praktycznie milczała (d' około 0.04), dla par wyraźnie różnych mówiła głośno (d' około 0.28). Gdyby był to artefakt techniczny, byłby wszędzie taki sam. To, że sygnał rośnie razem z realnością wyboru redaktorskiego, jest sygnaturą prawdziwej treści.

---

## Czym jest ta oś

Znalezienie kierunku to jedno, zrozumienie, co koduje, to drugie. Tekstowa część modelu CLIP pozwala „zapytać" oś, z jakimi pojęciami się pokrywa (CLIP koduje teksty i obrazy w tej samej przestrzeni, więc można mierzyć ich bliskość).

Zwycięskie skojarzenie było jednoznaczne: „intymna ludzka więź" (*an intimate human connection*), najsilniejsze w całym badaniu. Zaraz za nim „przekonująca ludzka historia".

Skrajności osi dopełniają obraz. Na jednym biegunie znajdują się ludzie, blisko, w interakcji: praca zespołowa, rozmowy, twarze. Na drugim puste krajobrazy, niebo, sceny bez człowieka. W parach z tej samej sceny wybór Strykera niemal zawsze padał na wersję bliższą ludziom, zgodnie ze słynną zasadą Roberta Capy: „Jeśli twoje zdjęcia nie są dość dobre, to znaczy, że nie podszedłeś dość blisko".

Znaleziona oś nie koduje „momentu czasowego". Koduje ludzką historię w kadrze: obecność, bliskość, interakcję, przeciwstawione pustce i przypadkowi.

Pojęcie czasu, ten Cartier-Bressonowski „ułamek sekundy", okazało się w danych nieobecne (najsłabsze ze wszystkich testowanych kategorii). Jest to wynik uczciwy i przewidziany. Pojedyncza klatka nie niesie informacji o momencie w czasie. Do uchwycenia „ułamka sekundy" potrzeba sekwencji, a FSA to pojedyncze zdjęcia. Uchwycone zostało znaczenie, nie czas, czyli dwie z trzech współrzędnych definicji Cartier-Bressona.

---

## Test transferu: czy działa poza archiwum

Sygnał znaleziony w jednym zbiorze może być jego lokalną osobliwością. Właściwy test brzmi: czy oś wytrenowana na czarno białych zdjęciach z lat 30. rozpozna coś we współczesnych zdjęciach, kolorowych, cyfrowych, z innego kontynentu.

Przez oś przeszło kilkaset takich fotografii. Wynik rozdzielił się na dwie części.

Semantyka przeniosła się w całości. Oś ułożyła zdjęcia dokładnie wzdłuż swojego pojęcia: na górze rankingu znalazły się tłumy, kolarze, procesje, ludzie w akcji, na dole ciemne, nastrojowe, samotne kadry. Pojęcie „ludzkiej historii" przetrwało zmianę epoki, techniki, palety i kontynentu. Kontrola wykluczyła odczytywanie koloru: po konwersji do czerni i bieli ranking pozostał niemal ten sam (korelacja 0.86 do 0.88). Jest to najmocniejszy dowód, że oś koduje treść, nie powierzchnię.

Preferencja się nie przeniosła, i jest to wynik pozytywny. Oś nie przewidziała wyborów konkretnego autora, ponieważ różni fotografowie cenią różne rzeczy (nastrój, samotność, światło), podczas gdy oś Strykera ceni obecność ludzi i dokumentalną czytelność. Gdyby oś trafiała w gust każdego fotografa, byłaby podejrzanie ogólnikowa. To, że rozpoznaje pojęcie, ale nie narzuca gustu, czyni z niej model konkretnego historycznego spojrzenia, oka Roya Strykera, a nie uniwersalnego sędziego.

---

## Od analizy do narzędzi

Skoro oś działa, można na niej zbudować narzędzia praktyczne.

### Kadrowanie: przeszukiwanie kadrów metodą przesuwnego okna

Jeśli oś rozpoznaje „ludzką historię w kadrze", może wskazać, gdzie w zdjęciu ta historia się znajduje, i zaproponować kadrowanie, które ją wydobywa. Działa to metodą przesuwnego okna (sliding window). Intuicja jest prosta: po zdjęciu przesuwa się prostokątną ramkę o różnych rozmiarach i proporcjach, jak przez lupę. Każdy taki wycinek dostaje swój embedding i swój wynik na osi. Narzędzie wybiera ten wycinek, który daje najwyższy „score Strykera". Dla źle skadrowanej klatki z ludźmi w tle potrafi wyciąć właśnie tę grupę, robiąc to, co redaktor robił ołówkiem na stykówce.

Ważne zastrzeżenie: kadrowanie nie tworzy treści. Na pustym pejzażu nie wygeneruje ludzi i poprawnie sygnalizuje brak możliwości poprawy. Aby uniknąć trywialnego rozwiązania (maksymalne przybliżenie na dowolną twarz), narzędzie ma wbudowane ograniczenia: dolny limit rozmiaru kadru i kara za skrajne proporcje. Dzięki temu wybiera kompozycję z ludźmi, a nie samo zbliżenie.

### Przełamanie granicy: krytyk uczony

Oś liniowa zatrzymała się na d'=0.188. Powstało pytanie, czy jest to granica metody (prostej linii), czy granica informacji (tyle, ile w danych w ogóle się znajduje).

Odpowiedź przyniósł trening. Zamiast prostej linii zastosowano uczoną funkcję, która potrafi „wyginać przestrzeń" i rozdzielać wzorce, których prosta nie łapie. Kluczowy był sposób uczenia: model porównywał zdjęcia parami z tej samej sceny, a jego zadaniem było przyznać wybranemu wyższą ocenę niż odrzuconemu. Takie porównanie wewnątrz pary automatycznie neutralizuje zakłócenia sceniczne. Skuteczność rozpoznawania wyboru Strykera wzrosła z 59% (oś liniowa) do około 64.5% na nowych, niewidzianych parach.

Oznacza to, że 0.188 było granicą metody, nie informacji. W reprezentacji CLIP znajduje się więcej „oka Strykera", niż widać z prostej projekcji. Weryfikacja potwierdziła, że sygnał zachował swoją sygnaturę (rósł razem z realnością wyboru), co świadczy o tym, że model łapie decyzję redakcyjną, a nie skrót. Wątek ten wymaga jeszcze jednej kontroli, usunięcia wpływu dziurki z danych treningowych, aby liczba była w pełni czysta. Kontrola jest opisana w dokumentacji.

### Ranking do selekcji

Wytrenowany krytyk przypisuje każdemu zdjęciu jedną liczbę, opisującą „ile w nim oka Strykera". Powstaje z tego praktyczne narzędzie. Po wgraniu kilkuset zdjęć z sesji zwraca je posortowane od najbardziej do najmniej „Strykerowskich", z podglądem miniatur. Nie zastępuje ludzkiego wyboru, lecz daje drugie spojrzenie, wyćwiczone na dziesiątkach tysięcy redakcyjnych decyzji sprzed dziewięćdziesięciu lat.

---

## Wnioski

Najważniejszy wynik nie jest liczbą. Jest nim to, że udało się empirycznie dotknąć estetycznej definicji sprzed siedemdziesięciu lat.

Cartier-Bresson mówił o jednoczesnym rozpoznaniu znaczenia i formy. Badanie pokazało, że w reprezentacji maszynowej te dwie rzeczy nie są osobne. Oś nie koduje formy obok znaczenia, lecz obniża ocenę formy pozbawionej znaczenia: harmonijne, ale puste pejzaże lądują po stronie odrzuconych. Jest to dokładnie Cartier-Bressonowskie „małżeństwo formy i znaczenia", wykryte liczbowo. „Ułamek sekundy" pozostał poza zasięgiem, co również jest uczciwą częścią mapy tego, co da się zmierzyć.

Projekt zebrał po drodze kilka lekcji wykraczających poza fotografię:

- Model zawsze wybiera najłatwiejszą drogę. Dziurka w negatywie niosła silniejszy sygnał niż cała reszta. Bez świadomego jej usunięcia powstałby wynik fałszywy. Zasada brzmi: najpierw zmierz, potem wierz.
- „Czyste" pojęcia techniczne prawie nie istnieją. Ostrość okazała się spleciona z treścią (teksturowe pola kontra gładkie niebo). W prawdziwych danych jakość i treść są nierozłączne.
- Wynik negatywny bywa cenny. Nieobecność „momentu czasowego" nie jest brakiem badania, lecz precyzyjną granicą tego, co pojedyncza klatka może przekazać.

---

## Dla zainteresowanych szczegółami

Projekt jest udokumentowany warstwowo, od intuicji po matematykę:

- **Metodologia i matematyka**: jak mierzy się separację (d'), jak neutralizuje się konfoundy (projekcja ortogonalna), jak czyta się wyniki.
- **Wyniki pełne**: wszystkie etapy badania, liczby, tabele, audyt tezy Cartier-Bressona rozłożony na pięć komponentów.
- **Narzędzia**: notebooki do kadrowania, treningu krytyka i rankingu własnych zdjęć, z dokumentacją kodu.
- **Źródła**: prace naukowe i materiały dydaktyczne stojące za każdym użytym pojęciem.

---

## Jak rozumieć ten projekt

Nie jest to dowód na to, że „AI rozumie sztukę". Jest to teza skromniejsza i zarazem ciekawsza: pojęcie uważane za czysto intuicyjne zostawia mierzalny ślad w sposobie, w jaki maszyna reprezentuje obrazy. Przy odpowiedniej dyscyplinie (parowanie zdjęć, neutralizacja konfoundów, uczciwe testy transferu) ten ślad da się wydobyć, nazwać i odróżnić od powierzchownych artefaktów.

Decydujący moment okazał się nie „ułamkiem sekundy", lecz ludzką historią w kadrze, przynajmniej w oczach jednego redaktora, którego wybory z lat trzydziestych wciąż da się odczytać z liczbowego cienia, jaki rzucają jego zdjęcia.

---

*Projekt badawczy na styku uczenia maszynowego, wizji komputerowej i teorii fotografii. Zbudowany wyłącznie na danych z domeny publicznej.*
