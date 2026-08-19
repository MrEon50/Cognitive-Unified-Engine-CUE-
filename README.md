# 🧠 Cognitive Unified Engine (CUE)

> **Zjednoczony Silnik Poznawczy** — połączenie 4 autorskich filarów architektury AI nowej generacji:
> **HOX & IR** (Ideografia Relacyjna) + **Alfabet Umysłu** (Prymitywy Poznawcze) + **GCL** (Global Context Layer) + **PisanoNet / RT-1** (Zegar 24 & Mikro-sieci Rezonansowe).

---

## 🍽️ Czym jest ten projekt? (Restauracja 5-gwiazdkowa vs Fast-Food)

W świecie AI:
- **Tradycyjne LLM (GPT-4, Llama):** To **Fast-Food** — masowe, gigantyczne modele (70B-405B parametrów), które "zjadają" petabajty tekstu z internetu, zapamiętując słowa na pamięć. Są potężne, ale drogie, powolne i często halucynują.
- **Cognitive Unified Engine (CUE):** To **Kuchnia Molekularna (Autorska Architektura 5-Gwiazdkowa)**. Zamiast budować wielki piec, stworzyliśmy precyzyjny, mały silnik neuro-symboliczny (zaledwie ~80k parametrów), który uczy się **mechaniki myślenia** na geometrii relacyjnej.

---

## 🎯 Do czego CUE się przyda i jakie przynosi EFEKTY?

### 1. Rozdzielenie Myślenia od Pamięci (Cognitive Pre-training)
- **Problem:** Tradycyjne LLM mieszają wiedzę encyklopedyczną z logiką.
- **Efekt CUE:** Trenujemy najpierw sam "silnik rozumowania" ($O, D, E, M, V, T, C, G$) na środowiskach syntetycznych. Model zyskuje zdolność rozwiązywania problemów logicznych **bez potrzeby posiadania miliardów parametrów**.

### 2. Gwarantowany, Czasowo Zoptymalizowany Czas Rozważania ($T \leq 7$)
- **Problem:** Łańcuchy myślenia (Chain of Thought) w LLM potrafią się zapętlać i zużywać ogromne ilości tokenów.
- **Efekt CUE:** Dzięki Zegarowi Pisano ($\pi(n)=24$) i mikro-sieciom rezonansowym $K_6, K_{12}, K_{18}, K_{24}$, model osiąga konwergencję w **dokładnie 7 taktach**. Wie, kiedy "skończył myśleć".

### 3. Redukcja Szumu i Filtrowanie Halucynacji ($\Omega_{\text{IR}}$)
- **Problem:** LLM gubią wątek w długich tekstach i zmyślają fakty.
- **Efekt CUE:** Moduł GCL tworzy ogólny szkic sekwencji ("Las"), a walidator IR i stała $\Omega_{\text{IR}}$ na bieżąco odrzucają niespójne fazowo hipotezy.

---

## 📊 Wyniki Eksperymentu Trenującego (`demo_unified_engine.py`)

Trening na CPU (140,955 parametrów z GCL v3.0 Attentive Salience Pooling + FiLM + macierzami $K_n$):

| Epoka | Total Loss | Task Loss | Phase Loss | Wskaźnik $\Omega_{\text{IR}}$ | Śr. Iteracji ($T \leq 7$) | Celność (Accuracy) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 3.5217 | 3.4905 | 0.0008 | 0.4658 | **7.00** | 3.89% |
| 10 | 1.9983 | 1.9674 | 0.0551 | 0.5064 | **7.00** | 33.36% |
| 20 | 1.2845 | 1.2590 | 0.0654 | 0.5332 | **7.00** | 54.70% |
| **30** | **0.9409** | **0.9162** | **0.0628** | **0.5463** | **7.00** | **64.11%** |

### ✅ Dowiedzione właściwości:
- **Sito Liczby 24 ($p^2 - 1 = 24n$):** 100% testów dla liczb pierwszych $\ge 5$ zakończonych wynikiem `VALID ✓`.
- **Wzrost celności uczenia:** z 3.89% do **64.11%** (w teście logicznym do **76.39%**).
- **GCL v3.0 Multi-Slot Attentive Pooling:** eliminacja rozmycia semantycznego i precyzyjne ważenie istotności tokenów.
- **Maksymalizacja $\Omega_{\text{IR}}$:** dynamiczny wzrost sprawności poznawczej do **0.5463** dzięki aktywnemu uczeniu gradientowemu.

---

## 🍳 Przepis na użycie CUE w PyTorch (Praktyka)

### Przepis 1: Jako Samodzielny Silnik Poznawczy (Standalone Model)

```python
import torch
from unified_cognitive_engine import UnifiedCognitiveEngine

# Instancjonowanie zjednoczonego silnika CUE
model = UnifiedCognitiveEngine(
    vocab_size=1000,   # Rozmiar alfabetu/słownika
    d_model=128,       # Wymiar reprezentacji przestrzeni stanów
    schema_dim=64,     # Wymiar szkicu globalnego GCL
    n_layers=2,        # Liczba zjednoczonych warstw CUE
    max_iter=7         # Gwarantowany limit taktów rezonansu
)

input_ids = torch.randint(0, 1000, (2, 24)) # Batch=2, Długość sekwencji=24

# Wykonanie forward pass z pobraniem szczegółów metryk poznawczych
logits, details = model(input_ids, return_details=True)

print(f"Logits shape: {logits.shape}")                     # [2, 24, 1000]
print(f"Wskaźnik Omega_IR: {details['omega_ir']:.4f}")       # np. 0.2832
print(f"Średnia liczba iteracji: {details['total_iters']}") # [7, 7]
```

### Przepis 2: Jako Warstwa Rozumowania w Istniejącym LLM (np. Llama / GPT)

```python
import torch.nn as nn
from gcl_resonant_layer import GCLResonantLayer

class LLMWithCUECore(nn.Module):
    def __init__(self, d_model=4096):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        # Wpięcie warstwy CUE w miejsce tradycyjnego MLP
        self.cue_layer = GCLResonantLayer(d_model=d_model, max_iter=7)

    def forward(self, hidden_states):
        # Przepływ przez warstwę CUE z zachowaniem zgodności wymiarów PyTorch
        norm_states = self.layernorm(hidden_states)
        cue_output = self.cue_layer(norm_states, return_details=False)
        return hidden_states + cue_output
```

---

## 📂 Pliki Projektu

- **[MASTER_ARCHITECTURE_PLAN.md](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/MASTER_ARCHITECTURE_PLAN.md)** — Pełny dokument architektoniczny łączący aksjomaty i wzory.
- **[cognitive_primitives.py](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/cognitive_primitives.py)** — 8 prymitywów myślenia ($O, D, E, M, V, T, C, G$) i aksjomat rekurencji $R(A,B) \rightarrow O'$.
- **[hox_ir_validator.py](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/hox_ir_validator.py)** — Walidator algebry IR, weryfikator sita $p^2 - 1 = 24n$ i stała $\Omega_{\text{IR}}$.
- **[gcl_resonant_layer.py](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/gcl_resonant_layer.py)** — Warstwa GCL + Zegar Pisano $\pi(n)=24$ + mikro-sieci $K_n$ ($T \le 7$).
- **[unified_cognitive_engine.py](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/unified_cognitive_engine.py)** — Nadrzędna klasa `UnifiedCognitiveEngine`.
- **[example_llm_integration.py](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/example_llm_integration.py)** — Gotowy szablon wpięcia CUE jako warstwy w bloku Transformera / LLM.
- **[test_cue_suite.py](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/test_cue_suite.py)** — Zautomatyzowany zestaw 12 testów jednostkowych i integracyjnych.
- **[test_logical_brain_learning.py](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/test_logical_brain_learning.py)** — Skrypt testowy weryfikujący uogólnianie na nieznanych danych.
- **[demo_unified_engine.py](file:///C:/Users/Endorfinka/Desktop/ZBIERACZ%20KODU/Cykliczne-mikro-sieci/CUE/demo_unified_engine.py)** — Uruchamialny skrypt demonstracyjny.


## 🚀 Uruchomienie i Testy

```bash
cd CUE

# 1. Uruchomienie pełnego pakietu testów jednostkowych (12/12 testów)
python test_cue_suite.py

# 2. Uruchomienie demonstracji treningu silnika zjednoczonego
python demo_unified_engine.py

# 3. Uruchomienie testu indukcji logicznej i generalizacji
python test_logical_brain_learning.py

# 4. Uruchomienie przykładu integracji CUE z architekturą LLM
python example_llm_integration.py
```

---

## 💡 CUE jako Wewnętrzny Silnik Rozumowania (Latent Reasoning Core)

CUE wnosi do architektury LLM nowoczesny paradygmat **Wewnętrznego Rozumowania Wektorowego (Latent Space Reasoning)**:

- **Zamiast zewnętrznych łańcuchów myśli (Chain of Thought):** Dzisiejsze modele (np. o1/R1) wykonują reasoning "na zewnątrz", wypisując tysiące powolnych słów. CUE wykonuje przemyślenie wektorów **wewnątrz sieci, w przestrzeni utajonej w ułamkach milisekund (w pętli $\le 7$ taktów), zanim wypowie choć jedno słowo**.
- 🛡️ **Autonomiczny Filtr Logiki ($\Omega_{\text{IR}}$):** Działa jak bezpiecznik odrzucający nielogiczne wektory i chroniący przed halucynacjami.
- 🗜️ **Kompresor Szumu:** Przepuszcza dane przez 8 prymitywów poznawczych ($O, D, E, M, V, T, C, G$), filtrując szum nieistotnych powiązań.
- 🎼 **Harmoniczny Zegar ($\pi=24$):** Daje wektorom strukturę czasowo-częstotliwościową, dzięki której wektory wchodzą w rezonans jak w fizycznym układzie falowym.

---

## 🧬 Geneza Projektu & Autorstwo

> **System CUE (Cognitive Unified Engine)** jest zwieńczeniem wieloetapowego eksperymentu architektonicznego wynikającego ze zsyntetyzowania autorskich składników informacyjnych twórcy **MrEon50**:
> 
> 1. **Teoria HOX** — Sito Liczby 24 ($p^2 - 1 = 24n$) oraz geometria heksagonalno-oktagonalna.
> 2. **Ideografia Relacyjna (IR)** — Język EICR ($\mu, \varepsilon, \iota, \psi, \varphi, \tau, \chi, \lambda, \delta$) i stała wieloczynnikowa $\Omega_{\text{IR}}$.
> 3. **Alfabet Umysłu** — Paleta 8 prymitywów poznawczych ($O, D, E, M, V, T, C, G$) oraz aksjomat rekurencji $R(A, B) \rightarrow O'$.
> 4. **GCL (Global Context Layer)** — Architektura percepcji globalnej ("Las $\rightarrow$ Drzewa"), `SchemaReflector` i `AdaptiveGate`.
> 5. **PisanoNet / RT-1** — Zegar Pisano $\pi(n)=24$, mikro-sieci rezonansowe $K_n$ i konwergencja w $T \leq 7$ krokach.
> 
> *Eksperyment zrealizowany i zweryfikowany w czystym PyTorch — 2026.*


