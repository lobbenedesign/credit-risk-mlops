# ADR 0004 — Registro modelli a stadi leggero, non un server MLflow

## Stato
Accettata

## Contesto
La roadmap originale del progetto citava un "registro modelli MLflow con
promozione a stadi (dev → shadow → prod) e rollback" come componente del
sistema. MLflow Model Registry offre esattamente questa semantica, ma
richiede un tracking server (locale o remoto) in esecuzione — infrastruttura
aggiuntiva che `make demo` dovrebbe evitare, per lo stesso principio già
applicato al resto di questo portfolio (TransactionStore in-memory in
`transaction-intelligence-agent`, InMemoryVectorStore in
`document-intelligence-rag`).

## Decisione
`registry/model_registry.py` implementa la *disciplina* di promozione a
stadi di MLflow Model Registry — `dev → shadow → prod`, transizioni non
saltabili, un solo modello in `prod` alla volta (la promozione di un nuovo
modello a `prod` archivia automaticamente quello precedente), rollback
sempre possibile verso `archived` da qualunque stadio — senza dipendere da
un tracking server in esecuzione. `RegisteredModel.stage_history` registra
ogni transizione con timestamp, lo stesso audit trail che un vero registro
esporrebbe.

## Alternative scartate
- **MLflow reale, con tracking server locale (SQLite backend).** Valutato:
  aggiunge una dipendenza pesante e un processo da avviare per
  `make demo`, per una funzionalità (promozione a stadi) che qui è
  interamente logica applicativa, non richiede tracking di esperimenti,
  artifact store o UI — tutte cose che MLflow offre ma che questo progetto
  non usa. Resterebbe la scelta corretta in un contesto con più modelli,
  più esperimenti da confrontare, e un team che già usa MLflow altrove.
- **Nessuna disciplina di stadi, un solo "modello corrente".** Più semplice,
  ma perde esattamente il punto: un modello non deve poter saltare da "appena
  addestrato" a "in produzione" senza passare da una fase di shadow — la
  stessa ragione per cui un vero deployment bancario userebbe un registro
  del genere.

## Conseguenze
- `tests/test_model_registry.py::test_only_one_model_can_be_in_production_at_a_time`
  verifica che la promozione di un modello a `prod` archivi automaticamente
  il precedente — l'invariante che rende "qual è il modello in produzione"
  una domanda con una sola risposta possibile in ogni momento.
- Limite dichiarato: nessuna persistenza, nessun confronto multi-esperimento,
  nessun artifact store per i pesi del modello stesso — solo la disciplina
  di promozione. Migrare a MLflow reale richiederebbe di implementare la
  stessa interfaccia (`register`, `promote`, `current_production`) sopra
  l'SDK MLflow, senza toccare i chiamanti.
