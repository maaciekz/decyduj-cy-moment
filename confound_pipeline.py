"""
confound_pipeline.py — modułowy pipeline kontroli konfoundów na embeddingach.

Cztery moduły:
  1. POMIAR        — mierzy siłę i spójność konfoundu (czy warto usuwać)
  2. DEFINICJA     — trzy sposoby: detektor CV / etykiety / prompt CLIP
  3. KIERUNEK      — estymuje v_konfound; dla wielu: ortogonalizacja (QR)
  4. NEUTRALIZACJA — projekcja na dopełnienie ortogonalne podprzestrzeni

Matematyka (zweryfikowana):
  - jeden konfound:  e_czyste = e - (e·v̂) v̂
  - wiele naraz:     e_czyste = e - Q (Qᵀ e),  gdzie Q = baza ortonormalna (QR)
    Usuwanie pojedynczo jest BŁĘDNE dla nieortogonalnych kierunków.

Projekt "Decydujący moment". Niezależne od konkretnego embeddera.
"""
import numpy as np


# ======================================================================
# Reprezentacja pojedynczego kierunku konfoundu
# ======================================================================
class ConfoundDirection:
    """Jeden konfound: nazwa, znormalizowany wektor kierunku, metoda, metadane."""

    def __init__(self, name, vector, method, meta=None):
        self.name = name
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            raise ValueError(f"Kierunek '{name}' ma zerową długość — "
                             f"konfound nieoznaczalny (brak różnicy między grupami).")
        self.vector = np.asarray(vector, dtype=np.float64) / norm
        self.method = method
        self.meta = meta or {}

    def __repr__(self):
        return (f"ConfoundDirection('{self.name}', method='{self.method}', "
                f"dim={len(self.vector)})")


# ======================================================================
# Główny kontroler — spina cztery moduły
# ======================================================================
class ConfoundController:
    """Pipeline kontroli konfoundów.

    Użycie:
        ctrl = ConfoundController(image_embedder=clip_img, text_embedder=clip_txt)
        ctrl.from_detector('hole', images, detect_hole, repair_inpaint)
        ctrl.from_labels('brightness', emb_dark, emb_bright)
        ctrl.from_clip_prompt('sharpness', 'a sharp photo', 'a blurry photo')
        report = ctrl.measure_all(embeddings, labels)
        clean = ctrl.neutralize(embeddings)            # usuwa WSZYSTKIE naraz
    """

    def __init__(self, image_embedder=None, text_embedder=None):
        # image_embedder: callable(img_bgr) -> wektor (znormalizowany)
        # text_embedder:  callable(str)     -> wektor (znormalizowany), opcjonalny
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder
        self.directions = []   # lista ConfoundDirection

    # ------------------------------------------------------------------
    # MODUŁ 2 — DEFINICJA KONFOUNDU (trzy sposoby)
    # ------------------------------------------------------------------
    def from_detector(self, name, images, detector_fn, repair_fn):
        """Konfound z detektora CV. Dla obrazów, gdzie detektor coś znajduje,
        tworzy parę (orig, naprawiony) i liczy kierunek = mean(orig) - mean(clean).

        detector_fn(img) -> detekcja (cokolwiek prawdziwego) lub None
        repair_fn(img, detekcja) -> obraz bez cechy (np. inpainting)
        """
        if self.image_embedder is None:
            raise ValueError("from_detector wymaga image_embedder")
        E_orig, E_clean, diffs = [], [], []
        n_detected = 0
        for img in images:
            det = detector_fn(img)
            if not det:
                continue
            n_detected += 1
            clean = repair_fn(img, det)
            eo = self.image_embedder(img)
            ec = self.image_embedder(clean)
            E_orig.append(eo)
            E_clean.append(ec)
            diffs.append(eo - ec)
        if not diffs:
            raise ValueError(f"Detektor '{name}' nie znalazł cechy w żadnym obrazie.")
        diffs = np.array(diffs)
        v = np.mean(E_orig, 0) - np.mean(E_clean, 0)
        cd = ConfoundDirection(name, v, 'detector', {
            'n_pairs': len(diffs),
            'n_detected': n_detected,
            'diffs': diffs,                      # do pomiaru spójności
        })
        self.directions.append(cd)
        return cd

    def from_labels(self, name, emb_group_a, emb_group_b):
        """Konfound z etykiet. Kierunek = mean(grupa_A) - mean(grupa_B).
        Np. A=ciemne, B=jasne -> kierunek 'jasność'.
        """
        a = np.asarray(emb_group_a, dtype=np.float64)
        b = np.asarray(emb_group_b, dtype=np.float64)
        v = a.mean(0) - b.mean(0)
        cd = ConfoundDirection(name, v, 'labels', {
            'n_a': len(a), 'n_b': len(b),
        })
        self.directions.append(cd)
        return cd

    def from_clip_prompt(self, name, positive, negative):
        """Konfound z promptów CLIP. Kierunek = text(pozytyw) - text(negatyw).
        Np. positive='a sharp photo', negative='a blurry photo'.
        positive/negative: str lub lista str (uśrednianych).
        """
        if self.text_embedder is None:
            raise ValueError("from_clip_prompt wymaga text_embedder (CLIP text encoder)")
        vp = self._embed_prompts(positive)
        vn = self._embed_prompts(negative)
        v = vp - vn
        cd = ConfoundDirection(name, v, 'clip_prompt', {
            'positive': positive, 'negative': negative,
        })
        self.directions.append(cd)
        return cd

    def _embed_prompts(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        vecs = [self.text_embedder(p) for p in prompts]
        return np.mean(vecs, 0)

    # ------------------------------------------------------------------
    # MODUŁ 1 — POMIAR (diagnostyka: czy warto usuwać)
    # ------------------------------------------------------------------
    def measure(self, direction, embeddings, labels):
        """Mierzy siłę konfoundu względem etykiet zadania.

        Zwraca:
          dprime       — separacja klas wzdłuż kierunku konfoundu (|d'|)
                         duże = konfound silnie skorelowany z etykietą = GROŹNY
          consistency  — spójność kierunku (śr. cosinus par diffs), tylko detektor
                         duże = kierunek powtarzalny = neutralizacja zadziała
          corr         — korelacja rzutu z etykietą
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels)
        proj = embeddings @ direction.vector

        # d' między grupami (labels: 1 vs 0)
        a = proj[labels == 1]
        b = proj[labels == 0]
        pooled_sd = np.sqrt(0.5 * (a.var() + b.var()) + 1e-12)
        dprime = abs(a.mean() - b.mean()) / pooled_sd

        # korelacja rzutu z etykietą
        corr = np.corrcoef(proj, labels.astype(float))[0, 1]

        # spójność kierunku (jeśli mamy pary diffs z detektora)
        consistency = None
        if 'diffs' in direction.meta:
            diffs = direction.meta['diffs']
            if len(diffs) >= 2:
                norm = diffs / (np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-12)
                sim = norm @ norm.T
                iu = np.triu_indices(len(diffs), k=1)
                consistency = float(sim[iu].mean())

        return {
            'name': direction.name,
            'dprime': float(dprime),
            'corr': float(corr),
            'consistency': consistency,
        }

    def measure_all(self, embeddings, labels):
        """Pomiar dla wszystkich zdefiniowanych konfoundów. Zwraca listę dictów."""
        return [self.measure(d, embeddings, labels) for d in self.directions]

    # ------------------------------------------------------------------
    # MODUŁ 3 — KIERUNEK / PODPRZESTRZEŃ (ortogonalizacja wielu)
    # ------------------------------------------------------------------
    def build_subspace(self, names=None):
        """Buduje ortonormalną bazę podprzestrzeni konfoundów (QR).

        Zwraca (Q, dirs):
          Q    — macierz (dim x k), kolumny = baza ortonormalna
          dirs — użyte ConfoundDirection
        """
        dirs = [d for d in self.directions
                if names is None or d.name in names]
        if not dirs:
            raise ValueError("Brak konfoundów do zbudowania podprzestrzeni.")
        V = np.stack([d.vector for d in dirs], axis=1)   # dim x k
        Q, _ = np.linalg.qr(V)
        # Zabezpieczenie: gdy kierunki są (prawie) liniowo zależne, QR może
        # dać kolumny o znikomej normie — odfiltruj je.
        keep = np.linalg.norm(Q, axis=0) > 1e-8
        Q = Q[:, keep]
        return Q, dirs

    # ------------------------------------------------------------------
    # MODUŁ 4 — NEUTRALIZACJA (projekcja na dopełnienie ortogonalne)
    # ------------------------------------------------------------------
    def neutralize(self, embeddings, names=None):
        """Usuwa WSZYSTKIE wskazane konfoundy naraz: e_czyste = e - Q (Qᵀ e).

        names=None -> wszystkie zdefiniowane konfoundy.
        Zwraca oczyszczone embeddingi (ta sama forma co wejście).
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings[None, :]
        Q, _ = self.build_subspace(names)
        cleaned = embeddings - (embeddings @ Q) @ Q.T
        return cleaned[0] if single else cleaned

    def verify(self, embeddings, names=None, labels=None):
        """Weryfikuje neutralizację. Zwraca dict z metrykami:
          residual_proj — maks. rzut oczyszczonych na kierunki konfoundu (cel ~0)
          mean_change   — śr. ||e_after - e_before|| (jak bardzo zmieniliśmy)
          dprime_before/after — separacja wzdłuż konfoundu przed/po (jeśli labels)
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        Q, dirs = self.build_subspace(names)
        cleaned = embeddings - (embeddings @ Q) @ Q.T

        # 1. Rzut oczyszczonych na każdy kierunek konfoundu (powinien ~0)
        residuals = {}
        for d in dirs:
            residuals[d.name] = float(np.abs(cleaned @ d.vector).max())

        # 2. Jak bardzo zmieniliśmy embeddingi
        mean_change = float(np.linalg.norm(cleaned - embeddings, axis=1).mean())

        out = {
            'residual_proj': residuals,
            'max_residual': max(residuals.values()),
            'mean_change': mean_change,
            'subspace_dim': Q.shape[1],
        }

        # 3. Separacja wzdłuż konfoundu przed/po (dowód, że konfound zniknął)
        if labels is not None:
            labels = np.asarray(labels)
            for d in dirs:
                pb = embeddings @ d.vector
                pa = cleaned @ d.vector
                def dprime(p):
                    a, b = p[labels == 1], p[labels == 0]
                    return abs(a.mean()-b.mean()) / (np.sqrt(0.5*(a.var()+b.var()))+1e-12)
                out[f'dprime_{d.name}_before'] = float(dprime(pb))
                out[f'dprime_{d.name}_after'] = float(dprime(pa))

        return out

    # ------------------------------------------------------------------
    # Raport zbiorczy
    # ------------------------------------------------------------------
    def report(self, embeddings, labels):
        """Drukuje czytelny raport: pomiar + neutralizacja + weryfikacja."""
        print("=" * 64)
        print("RAPORT KONTROLI KONFOUNDÓW")
        print("=" * 64)
        print(f"Zdefiniowane konfoundy: {[d.name for d in self.directions]}\n")

        print("MODUŁ 1 — POMIAR (siła konfoundu względem etykiety):")
        for m in self.measure_all(embeddings, labels):
            cons = f"{m['consistency']:.3f}" if m['consistency'] is not None else "—"
            flag = "⚠ GROŹNY" if m['dprime'] > 1.0 else "ok"
            print(f"  {m['name']:18} d'={m['dprime']:.3f}  corr={m['corr']:+.3f}  "
                  f"spójność={cons}  [{flag}]")

        print("\nMODUŁ 3 — PODPRZESTRZEŃ:")
        Q, dirs = self.build_subspace()
        print(f"  {len(dirs)} kierunków -> baza ortonormalna wymiaru {Q.shape[1]}")

        print("\nMODUŁ 4 — NEUTRALIZACJA + WERYFIKACJA:")
        v = self.verify(embeddings, labels=labels)
        print(f"  maks. rzut po neutralizacji: {v['max_residual']:.2e}  (cel ~0)")
        print(f"  śr. zmiana embeddingu:        {v['mean_change']:.4f}")
        for d in dirs:
            kb = v.get(f'dprime_{d.name}_before')
            ka = v.get(f'dprime_{d.name}_after')
            if kb is not None:
                print(f"  d'[{d.name}]: {kb:.3f} -> {ka:.3f}  "
                      f"(spadek separacji = konfound usunięty)")
        print("=" * 64)
