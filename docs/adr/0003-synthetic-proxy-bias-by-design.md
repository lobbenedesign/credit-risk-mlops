# ADR 0003 — Il dataset sintetico inietta un bias di proxy reale, non un caso di giocattolo

## Stato
Accettata

## Contesto
Un dataset sintetico per testare fairness può fallire in due modi opposti:
essere così ovviamente truccato (il gruppo protetto determina direttamente
l'esito) che il fairness check "trova" un problema che nessun modello
onesto produrrebbe mai per caso, oppure essere così pulito (protected_group
davvero indipendente da tutto) che il fairness report è sempre a zero e non
dimostra che il codice funzioni su un caso reale.

## Decisione
Il generatore sintetico (`data/synthetic.py`) è costruito per riprodurre lo
scenario di **discriminazione da proxy** reale nel credit scoring: il
cosiddetto problema del "thin file" (file creditizio sottile), dove nuovi
entranti nel sistema creditizio — spesso più giovani, spesso immigrati —
ottengono punteggi bureau sistematicamente più bassi non perché più
rischiosi, ma perché il punteggio bureau tradizionale è calibrato su una
storia creditizia lunga che non hanno ancora avuto modo di costruire.

Nel generatore:
- Il vero rischio di default dipende da `existing_debt_ratio`,
  `income_to_loan_ratio`, `employment_years` e un fattore latente di
  affidabilità — **identico in distribuzione** fra `protected_group` "A" e
  "B". L'appartenenza al gruppo non ha alcun effetto causale sul vero
  rischio, per costruzione.
- `credit_history_score` (la feature che un vero scorecard userebbe) è in
  parte determinato dallo stesso fattore latente (è realmente predittivo,
  non rumore puro) ma porta anche un'offset sistematico di -70 punti per il
  gruppo "B" — lo stesso "thin file" effect.

## Il processo di calibrazione
La prima versione del generatore produceva un tasso di default del 50%
circa — irrealistico per un portafoglio di credito al consumo reale, dove
tassi del 5-25% sono tipici. Ricalibrato empiricamente (non a tavolino):
provati diversi valori di intercetta nella funzione logistica finché il
tasso di default non si è stabilizzato intorno al 20-25%, un valore
plausibile per un dataset dimostrativo con segnale forte.

## Alternative scartate
- **Iniettare il bias direttamente nella label `defaulted`.** Avrebbe reso
  il modello "giustamente" discriminatorio (il gruppo B davvero fa più
  default nei dati), mascherando la differenza fra un modello che riflette
  un vero rischio differenziale e uno che amplifica un artefatto di misura
  — la distinzione che il problema del "thin file" esiste apposta per
  segnalare.
- **Nessuna differenza fra gruppi in nessuna feature.** Avrebbe reso il
  fairness report un test che passa sempre — non dimostra che
  `compute_fairness_report` rilevi davvero un problema quando c'è.

## Conseguenze
- Il fairness report (README §Numeri misurati) mostra un gap reale di
  parità demografica del 10,0% e di equal opportunity del 4,4% sullo
  scorecard — un risultato di un esperimento controllato, non un numero
  scelto per sembrare interessante.
- `tests/test_synthetic_data.py` verifica sia che il vero tasso di default
  resti simile fra i gruppi (bias assente nella verità di base) sia che
  `credit_history_score` mostri il gap iniettato — entrambe le metà
  dell'esperimento sono regression-tested, non solo il risultato finale.
