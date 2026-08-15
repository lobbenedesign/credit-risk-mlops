# credit-risk-mlops

Pipeline di credit scoring con **compliance-as-code per l'AI Act**: modello
interpretabile con reason code nativi (Art. 13), documentazione tecnica
generata da codice (Art. 11), log automatico di ogni inferenza (Art. 12),
override umano con motivazione obbligatoria (Art. 14), fairness report su
un bias di proxy realistico — **misurato, non assunto**.

[![CI](https://img.shields.io/badge/CI-configured-blue)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![Coverage](https://img.shields.io/badge/coverage-99%25-green)]()

## Problema

Il credit scoring è un sistema *high-risk* secondo l'Annex III dell'AI Act,
con obblighi rinviati al 2 dicembre 2027 dal Digital Omnibus — le banche
sono nella fase di progettazione della compliance, non ancora di esecuzione.
Questo repository non implementa "un classificatore che predice il
default": implementa il classificatore *insieme* ai quattro artefatti che
l'AI Act richiede per un sistema del genere, generati dal codice stesso
invece che scritti a mano dopo il fatto — la differenza fra "abbiamo un
modello" e "abbiamo un sistema che si documenta da solo ogni volta che
qualcosa cambia".

**Cosa NON fa questo progetto** (dichiarato, non omesso): nessun dato
creditizio reale è stato usato (dataset interamente sintetico, vedi
`docs/adr/0003`); la soglia di decisione (0.5) non è ottimizzata per un
costo asimmetrico di falsi positivi/negativi specifico di un prodotto reale;
il registro modelli non è un vero MLflow (`docs/adr/0004`); non c'è
persistenza, stesso limite dichiarato negli altri due repository di questo
portfolio.

## Architettura

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Synthetic data   │───▶│  Out-of-time      │───▶│  FeaturePreproc.  │
│  (bias di proxy   │    │  split            │    │  (fit solo su     │
│  by design)       │    └──────────────────┘    │  train)          │
└──────────────────┘                              └──────────────────┘
                                                            │
                                    ┌───────────────────────┴───────────────────────┐
                                    ▼                                               ▼
                         ┌──────────────────┐                            ┌──────────────────┐
                         │  ScorecardModel   │                            │  BlackBoxModel     │
                         │  (LogReg, decide) │                            │  (RandomForest,     │
                         │  reason code nat. │                            │  solo confronto AUC)│
                         └──────────────────┘                            └──────────────────┘
                                    │
                    ┌───────────────┼───────────────┬──────────────────┐
                    ▼               ▼               ▼                  ▼
          ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
          │ InferenceLog  │ │ Oversight     │ │ FairnessReport│ │ TechnicalDossier│
          │ (Art. 12)     │ │ Log (Art. 14) │ │              │ │ (Art. 11, fallisce│
          │              │ │               │ │              │ │ se manca un dato)│
          └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

`protected_group` è tracciato per il fairness monitoring ma **non è mai una
feature del modello** (non compare in `FEATURE_COLUMNS`) — un vincolo di
design verificato, non solo dichiarato: usarlo direttamente come input
sarebbe disparità di trattamento diretta, non il problema di proxy
discrimination che questo repository dimostra di saper misurare.

## Numeri misurati

Eseguendo `python scripts/demo.py` (3000 domande sintetiche, 24 mesi, split
out-of-time 75/25):

| Metrica | Valore |
|---|---|
| Domande generate | 3000 (tasso di default reale: 24,8%) |
| AUC scorecard (glass-box, usato per le decisioni) | **0,9232** |
| AUC black-box (random forest, solo confronto) | 0,9185 |
| Gap (black-box − scorecard) | **-0,0046** — il modello interpretabile è leggermente *migliore*, non peggiore |
| Demographic parity difference (scorecard) | **9,99%** (gruppo A approvato all'80,3%, gruppo B al 70,4%) |
| Equal opportunity difference (scorecard) | **4,35%** (fra i non-default reali, TPR 94,0% vs 89,7%) |
| Copertura test | **99%** |
| Test totali | 65, tutti verdi |

**Il gap di fairness è un risultato di un esperimento controllato, non un
numero scelto a caso.** Il generatore sintetico (`docs/adr/0003`) rende il
vero rischio di default *identico* in distribuzione fra i due gruppi
protetti — l'appartenenza al gruppo non ha alcun effetto causale sulla
probabilità di default — ma introduce un'offset sistematico di -70 punti
sul punteggio bureau (`credit_history_score`) per il gruppo minoritario,
riproducendo il problema reale del "thin file": chi ha una storia creditizia
più corta (spesso più giovane, spesso nuovo nel sistema) ottiene punteggi
bureau più bassi per ragioni non legate al vero rischio. Lo scorecard, pur
mai vedendo `protected_group`, riproduce comunque il gap — perché lo assorbe
attraverso `credit_history_score`, la feature che quel bias porta con sé.
Questo è esattamente il fenomeno di *proxy discrimination* che un fairness
audit deve saper catturare, e questo repository lo cattura con un numero,
non con un'affermazione.

## Come si esegue

```bash
git clone <repo> && cd credit-risk-mlops
make install
make demo     # genera dati, addestra entrambi i modelli, fairness report, override, dossier tecnico
make test     # 65 test, coverage report
make serve    # FastAPI su http://localhost:8002
              #   POST /score (scoring + reason code)
              #   POST /override/{decision_id} (Art. 14, motivazione obbligatoria)
              #   GET  /fairness · /dossier · /healthz
```

Oppure via Docker: `docker compose up --build`.

## Cosa ho imparato / limiti noti

- **Un dataset sintetico "onesto" per la fairness non è banale da
  costruire.** La prima tentazione è iniettare il bias direttamente nella
  label (il gruppo B "davvero" fa più default nei dati) — ma questo confonde
  un modello che riflette un vero rischio differenziale con uno che
  amplifica un artefatto di misura. Il generatore qui separa esplicitamente
  le due cose: rischio vero identico, una feature-proxy sistematicamente
  distorta — è quello che rende il fairness report un test, non una
  tautologia.
- **Il tasso di default della prima versione del generatore era ~50%** —
  ricalibrato empiricamente (non a tavolino) provando diverse intercette
  finché non si è avvicinato a un tasso plausibile per un portafoglio di
  credito al consumo reale. Documentato in `docs/adr/0003` col processo, non
  solo col risultato finale — stesso principio del debug di retrieval in
  `document-intelligence-rag/docs/adr/0003`.
- **Un endpoint di validazione (Art. 14: motivazione obbligatoria) non è un
  errore server.** Il primo tentativo lasciava che il `ValueError` del
  livello di dominio propagasse fino all'handler globale, restituendo 500.
  Un 500 dice "abbiamo un bug"; un reviewer che dimentica la motivazione non
  è un bug del sistema, è un input non valido — corretto a un 400 esplicito
  in `api/main.py`, con test che verificano il messaggio d'errore, non solo
  lo status code.
- **Scegliere l'interpretabilità non è sempre un compromesso.** Il confronto
  misurato (AUC scorecard vs black-box, gap -0,0046) è il tipo di numero che
  rende "abbiamo scelto la trasparenza" una decisione difendibile invece di
  un'affermazione — vedi `docs/adr/0002`.

## Decisioni architetturali

- [`ADR-0001`](docs/adr/0001-out-of-time-split-and-shared-preprocessor.md) — split out-of-time e preprocessor condiviso fra i due modelli
- [`ADR-0002`](docs/adr/0002-native-reason-codes-not-shap-on-blackbox.md) — reason code nativi dallo scorecard, non SHAP sul black-box
- [`ADR-0003`](docs/adr/0003-synthetic-proxy-bias-by-design.md) — il dataset sintetico inietta un bias di proxy reale, non un caso di giocattolo
- [`ADR-0004`](docs/adr/0004-lightweight-staged-registry-not-mlflow.md) — registro modelli a stadi leggero, non un server MLflow

## Nel contesto del portfolio

Quarto repository di [`banca-sandbox`](../BANKING-PORTFOLIO/ROADMAP.md): il
progetto AI Act della roadmap originale, ora costruito con lo stesso
standard maturato negli altri tre — numeri misurati non aggettivi, ADR che
documentano il processo oltre al risultato, `make demo` come unico comando
richiesto, un bias di fairness reale trovato (per costruzione, qui) e
misurato invece che dichiarato in astratto.
