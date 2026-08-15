# ADR 0001 — Split out-of-time e preprocessor condiviso fra i due modelli

## Stato
Accettata

## Contesto
Un backtest di credit scoring che mescola casualmente le domande fra
training e test permette al modello di essere valutato su esempi
temporalmente precedenti a esempi su cui è stato addestrato — non lo scenario
che affronta in produzione, dove ogni domanda scorata è per definizione
successiva a tutti i dati di training. Un AUC misurato con uno split
casuale è quindi ottimista rispetto alle performance reali. Separatamente,
se scorecard e black-box venissero preprocessati con due `StandardScaler`
diversi (o uno fittato su train+test), una differenza di AUC fra i due
modelli confonderebbe l'effetto del modello con quello del preprocessing.

## Decisione
1. `model/split.py`: `split_out_of_time` ordina per `application_date` e
   usa la frazione più recente come test set — niente shuffle, il cutoff è
   derivato dai dati, non hardcoded.
2. `model/features.py`: un solo `FeaturePreprocessor` (uno `StandardScaler`)
   viene fittato *solo* su `train_df` e riusato per trasformare sia i dati
   di training sia quelli di test, sia per lo scorecard sia per il
   black-box — nessun leakage di statistiche del test set nel training,
   nessuna differenza di preprocessing che confonda il confronto AUC.

## Alternative scartate
- **K-fold cross-validation.** Standard per molti problemi, ma non cattura
  il vincolo specifico del credit scoring: un modello deve generalizzare
  nel tempo (nuove condizioni macroeconomiche, nuovi pattern di frode), non
  solo su dati non visti ma temporalmente mescolati. Un backtest
  out-of-time è la pratica di settore per questo motivo.
- **Un preprocessor per modello.** Isolerebbe eventuali bug di
  preprocessing specifici di un modello, ma introduce esattamente il
  confondimento che questa ADR vuole evitare — non giustificato quando
  entrambi i modelli condividono lo stesso set di feature numeriche.

## Conseguenze
- Il numero di AUC riportato in README è quello che ci si aspetterebbe di
  vedere in un vero monitoraggio di produzione, non un limite superiore
  ottimistico.
- `tests/test_split.py` verifica esplicitamente che ogni riga di train
  preceda ogni riga di test — non un dettaglio implementativo, è
  l'invariante che rende il numero di AUC significativo.
