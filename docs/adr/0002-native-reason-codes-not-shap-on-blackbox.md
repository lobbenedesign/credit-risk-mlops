# ADR 0002 — Reason code nativi dallo scorecard, non SHAP sul black-box

## Stato
Accettata

## Contesto
L'AI Act (Art. 13, trasparenza) e le prassi di adverse action notice
richiedono che un rifiuto di credito sia accompagnato da ragioni specifiche
e verificabili. Un approccio comune è addestrare il modello più performante
possibile (spesso un ensemble o gradient boosting) e poi spiegarlo con SHAP
o LIME — un'approssimazione post-hoc di cosa il modello "probabilmente" ha
fatto, non un resoconto esatto.

## Decisione
Il modello che prende le decisioni è `ScorecardModel`, una logistic
regression su feature standardizzate. I reason code sono calcolati come
`coefficiente × valore_standardizzato` per la specifica domanda — una
scomposizione esatta dello score lineare del modello, non un'approssimazione.
`BlackBoxModel` (random forest) esiste solo per rispondere a "quanto AUC
guadagneremmo con un modello più flessibile?" — non ha un metodo
`reason_codes` e non è mai usato per una decisione reale.

Numero misurato (non assunto): su questo dataset, la risposta a quella
domanda è **-0,0046** di AUC — il modello black-box è leggermente *peggiore*
del glass-box, non migliore. Scegliere l'interpretabilità qui non costa
nulla in performance misurata.

## Alternative scartate
- **SHAP su random forest, usato come reason code ufficiali.** Tecnicamente
  praticabile, ma un'approssimazione (i valori di Shapley di un ensemble
  dipendono dal background dataset scelto, dal metodo di campionamento) è
  un fondamento più debole per una decisione contestabile legalmente di un
  coefficiente esatto di un modello lineare. Anche il costo di dipendenza
  (SHAP compila estensioni native, aggiunge tempo di installazione e
  superficie di build) è stato un fattore, ma non quello decisivo — la
  scelta sarebbe stata la stessa anche a costo zero.
- **Un solo modello, addestrato per essere sia performante sia
  interpretabile (es. un albero poco profondo).** Scartato perché il
  confronto esplicito fra due modelli separati è ciò che rende la scelta
  dell'interpretabilità una decisione *misurata*, non solo dichiarata — un
  singolo modello "abbastanza interpretabile" non risponde alla domanda
  "quanto stiamo pagando per questo?".

## Conseguenze
- `tests/test_models_training.py::test_the_interpretable_model_is_not_meaningfully_worse`
  è un regression test su questo specifico trade-off: se un giorno il
  black-box sorpassasse lo scorecard di un margine significativo, il test
  fallirebbe e la decisione andrebbe rivista esplicitamente, non lasciata
  scadere silenziosamente.
- Limite dichiarato: il black-box non è mai stato tunato con la stessa cura
  dello scorecard (iperparametri ragionevoli, non ottimizzati via grid
  search) — il confronto è onesto ma non è "il miglior black-box possibile
  contro il miglior glass-box possibile", è "un confronto di buon senso fra
  i due approcci a parità di sforzo".
