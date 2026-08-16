# ADR 0005 — Nessun Redis, nessuna blockchain

## Stato
Accettata

## Contesto
Due domande distinte, sollevate esplicitamente: serve un sistema stile
Redis? Serve una blockchain?

## Decisione: Redis — non nello scope attuale
Non aggiunto. Il modello, il training result e il fairness report sono
calcolati **una sola volta all'avvio** del processo FastAPI e tenuti in
memoria per servire `/score`, `/fairness`, `/dossier` — non c'è cache da
invalidare perché non c'è un secondo calcolo costoso da evitare: scorare
una singola domanda con un modello di regressione logistica già addestrato
è già rapido quanto lo sarebbe leggerlo da una cache.

L'unico dato che *cresce* durante la vita del processo è `InferenceLog`
(ogni chiamata a `/score` vi si aggiunge, per l'Art. 12). A questa scala —
una demo, un processo singolo — tenerlo in memoria è corretto. Un vero
deployment con più istanze del servizio avrebbe bisogno che l'inference
log sia condiviso e durevole: lì la risposta naturale è un database
(Postgres, append-only), non Redis — l'Art. 12 richiede un registro di
audit permanente e interrogabile, non un dato con TTL. Redis diventerebbe
rilevante per un problema diverso e più specifico: se il servizio dovesse
tenere un **rate limit o una cache di risposta per applicant_id** per
evitare rivalutazioni ripetute nello stesso brevissimo intervallo — non
implementato qui perché non è un problema che questo demo ha (ogni
richiesta al simulatore è intenzionale, non un evento ad alta frequenza da
proteggere).

## Decisione: blockchain — non pertinente
Non aggiunta. Il credit scoring è una decisione che **la banca prende
unilateralmente e deve poter giustificare a un'autorità di vigilanza** —
non un problema di consenso fra parti che non si fidano l'una dell'altra.
L'audit trail (`InferenceLog`, `HumanOversightLog`) esiste per rendere
quella decisione tracciabile e contestabile dall'interessato, non per
farla concordare fra più attori indipendenti — un log Python in memoria
(o un database in un deployment reale) risolve esattamente questo, senza
bisogno di un ledger distribuito. Stessa conclusione, per ragioni
analoghe, negli ADR equivalenti di `psd3-open-banking-gw` e
`aml-fraud-graph`.

## Conseguenze
- Nessun cambiamento di codice da questo ADR.
- Se questo servizio evolvesse verso un deployment multi-istanza,
  `InferenceLog` e `HumanOversightLog` diventerebbero un database
  relazionale condiviso (non Redis) — un audit trail regolamentare deve
  sopravvivere a un riavvio e restare interrogabile per anni, non solo
  per la durata di una cache.
