import os
import sys
# Sostituisci con il tokenizer corretto del tuo modello se non usa Hugging Face standard
# Ad esempio AutoTokenizer se dentro la cartella c'è un file tokenizer.json o tokenizer_config.json
from transformers import AutoTokenizer

# 1. Configura il percorso locale del modello (la stessa DIR passata a --model)
MODEL_DIR = "/home/albus/.lmstudio/hub/models/lmstudio-community/gpt-oss-120b/"  # Modifica se necessario
print("Caricamento del tokenizer in corso...", file=sys.stderr)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
except Exception as e:
    print(f"Errore nel caricamento del tokenizer da {MODEL_DIR}: {e}", file=sys.stderr)
    print("Assicurati che i file del tokenizer siano presenti nella directory del modello.", file=sys.stderr)
    sys.exit(1)

# 2. Lista completa dei tuoi prompt (Primi 75 inseriti)
prompts = [
    "Spiega la differenza tra RAM e memoria SSD in modo semplice, con un esempio pratico.",
    "Riassumi questo testo in massimo cinque punti, mantenendo tutte le informazioni importanti: [inserisci qui un testo di esempio].",
    "Scrivi una funzione Python che verifichi se una stringa è palindroma e includi alcuni test.",
    "Analizza questa funzione Python, individua eventuali bug e proponi una versione corretta: `def average(nums): return sum(nums) / len(nums)`.",
    "Un negozio vende un prodotto a 80 euro con uno sconto del 15%. Calcola il prezzo finale e spiega i passaggi.",
    "Confronta i vantaggi e gli svantaggi di utilizzare un modello linguistico locale rispetto a un modello tramite API cloud.",
    "Scrivi una query SQL per trovare i cinque clienti con la spesa totale più alta, ipotizzando le tabelle customers e orders.",
    "Spiega passo dopo passo come risolvere un sistema di equazioni lineari con due incognite e mostra un esempio.",
    "Genera un file JSON valido che descriva tre prodotti, ciascuno con id, nome, prezzo e disponibilità.",
    "Individua i possibili problemi di sicurezza in questo frammento di codice e suggerisci come correggerli: `query = \"SELECT * FROM users WHERE name = '\" + name + \"'\"`.",
    "Scrivi un'email professionale per comunicare a un cliente che una consegna subirà un ritardo di due giorni, proponendo una soluzione.",
    "Analizza questi dati di vendita mensili: gennaio 120, febbraio 135, marzo 110, aprile 160, maggio 175. Individua l'andamento e calcola la media.",
    "Progetta un algoritmo che deduplichi una lista di elementi mantenendo l'ordine originale. Spiega complessità temporale e spaziale.",
    "Leggi il seguente scenario e proponi tre soluzioni alternative, confrontandole per costo, complessità e rischi: un'applicazione web diventa lenta quando aumenta il numero di utenti.",
    "Analizza un testo lungo suddividendolo in sezioni tematiche, individua le contraddizioni interne e formula una conclusione che distingua fatti, ipotesi e informazioni mancanti.",
    "Spiega la differenza tra HTTP e HTTPS in modo semplice, con un esempio pratico.",
    "Spiega il concetto di debito tecnico a un manager non tecnico usando una metafora efficace.",
    "Spiega come funziona la crittografia end-to-end a un bambino di 10 anni.",
    "Spiega la differenza tra intelligenza artificiale, machine learning e deep learning.",
    "Spiega il concetto di blockchain senza usare termini tecnici o gergali.",
    "Spiega come funziona l'algoritmo di PageRank di Google in modo intuitivo.",
    "Spiega la differenza tra Docker e una macchina virtuale (VM) usando un'analogia con il mondo reale.",
    "Spiega cosa sono le API (Application Programming Interface) facendo l'esempio di un ristorante.",
    "Spiega la differenza tra un database SQL e uno NoSQL in base ai casi d'uso.",
    "Spiega il concetto di compilatore rispetto a un interprete nella programmazione.",
    "Riassumi questo testo in massimo tre punti chiave, focalizzandoti sulle conclusioni: [inserisci testo].",
    "Estrai tutte le date e gli eventi principali da questo testo e digitali in ordine cronologico: [inserisci testo].",
    "Riscrivi questo paragrafo per renderlo più professionale e persuasivo: [inserisci testo].",
    "Correggi la grammatica e la sintassi di questo testo mantenendo il tono originale: [inserisci testo].",
    "Trasforma questo articolo di blog in uno script per un video TikTok di 60 secondi: [inserisci testo].",
    "Analizza il tono di voce di questo testo e suggerisci come renderlo più empatico: [inserisci testo].",
    "Estrai le parole chiave principali e i concetti correlati da questo documento: [inserisci testo].",
    "Riscrivi questa email aziendale per renderla estremamente breve e diretta: [inserisci testo].",
    "Riassumi questo lungo verbale di riunione evidenziando solo le azioni da compiere e i responsabili: [inserisci testo].",
    "Trasforma questa lista di dati tecnici in un paragrafo discorsivo facile da leggere: [inserisci testo].",
    "Scrivi una funzione Python che calcoli il fattoriale di un numero in modo ricorsivo.",
    "Scrivi una funzione Python per invertire l'ordine delle parole in una stringa, escludendo la punteggiatura.",
    "Scrivi uno script Python che legga un file CSV e ne stampi la media di una colonna numerica.",
    "Scrivi una funzione Python per verificare se due stringhe sono anagrammi l'una dell'altra.",
    "Scrivi una funzione Python che trovi il numero più grande in una lista senza usare la funzione integrata max().",
    "Scrivi uno script Python per scaricare il contenuto HTML di una pagina web data la sua URL.",
    "Scrivi una funzione Python che converta una temperatura da Celsius a Fahrenheit e viceversa.",
    "Scrivi una funzione Python che generi una password sicura casuale di lunghezza configurabile.",
    "Scrivi una funzione Python per unire due dizionari sommando i valori delle chiavi comuni.",
    "Scrivi uno script Python che calcoli i primi N numeri della successione di Fibonacci.",
    "Analizza questa funzione Python, individua perché va in loop infinito e correggila: [inserisci codice].",
    "Analizza questo codice Python e ottimizzalo per ridurre il consumo di memoria: [inserisci codice].",
    "Trova l'errore di sintassi o di logica che fa restituire None a questa funzione: [inserisci codice].",
    "Analizza questo frammento di codice e riscrivilo seguendo le linee guida ufficiali di PEP 8: [inserisci codice].",
    "Individua perché questa funzione solleva un errore IndexError e implementa una soluzione sicura: [inserisci codice].",
    "Riscrivi questa funzione Python sostituendo il ciclo for con una list comprehension più efficiente: [inserisci codice].",
    "Analizza questo codice e spiega perché la variabile globale non viene aggiornata correttamente: [inserisci codice].",
    "Trova il bug in questa gestione delle eccezioni che nasconde gli errori reali del programma: [inserisci codice].",
    "Analizza questo script e proponi un modo per renderlo asincrono usando asyncio: [inserisci codice].",
    "Riscrivi questa funzione nidificata per ridurne la complessità ciclomatica e migliorarne la leggibilità.",
    "Un negozio offre un secondo prodotto a metà prezzo. Calcola il prezzo totale per due articoli da 60 e 40 euro.",
    "Calcola la probabilità di ottenere due volte testa lanciando una moneta equa tre volte di seguito.",
    "Spiega passo dopo passo come calcolare l'area di un triangolo scaleno conoscendo solo i tre lati.",
    "Risolvi questa proporzione matematica spiegando logica e passaggi intermedi: x : 8 = 15 : 6.",
    "Calcola il ritorno sull'investimento (ROI) se spendi 500 euro in pubblicità e ottieni 1200 euro di ricavi.",
    "Spiega come calcolare la media ponderata dei voti universitari e mostra un esempio pratico.",
    "Un treno viaggia a 90 km/h. Calcola quanti minuti impiega a percorrere una distanza di 45 chilometri.",
    "Trova il valore dell'incognita X nella seguente equazione di primo grado: 3x + 5 = 20.",
    "Spiega la logica dietro il teorema di Pitagora e mostra come applicarlo per trovare l'ipotenusa.",
    "Calcola lo sconto effettivo se un prodotto già scontato del 20% riceve un ulteriore sconto del 10%.",
    "Confronta i vantaggi e gli svantaggi di SQL Server rispetto a PostgreSQL in ambito enterprise.",
    "Confronta lo sviluppo di app native (iOS/Android) rispetto sullo sviluppo cross-platform (Flutter/React Native).",
    "Confronta l'architettura a monolito rispetto a quella a microservizi per una startup in crescita.",
    "Confronta i vantaggi di utilizzare GraphQL rispetto a una classica architettura REST API.",
    "Confronta l'uso di Git Merge rispetto a Git Rebase, indicando quando preferire l'uno o l'altro.",
    "Confronta i linguaggi Python e Go per lo sviluppo di backend scalabili.",
    "Confronta l'architettura Serverless (AWS Lambda) rispetto a server dedicati (AWS EC2) per costi e gestione.",
    "Confronta i dischi SSD SATA con i dischi SSD NVMe in termini di velocità e costi.",
    "Confronta i vantaggi di SaaS (Software as a Service) rispetto a software installato On-Premise.",
    "Confronta la programmazione orientata agli oggetti (OOP) con la programmazione funzionale.",
    "Scrivi una query SQL per contare il numero totale di ordini effettuati da ogni singolo cliente.",
    "Scrivi una query SQL per trovare tutti i dipendenti che hanno uno stipendio superiore alla media aziendale.",
    "Scrivi una query SQL per eliminare i record duplicati da una tabella mantenendo solo quello con l'ID più basso.",
    "Scrivi una query SQL che utilizzi una LEFT JOIN per trovare i clienti che"
]

# ... (lascia invariata la parte iniziale del caricamento del tokenizer e la lista dei prompt)

# 3. Elaborazione e Output pulito
# Mandiamo le scritte informative su sys.stderr, così non finiscono nel file .txt
print("\n--- COMANDI FORMATTATI PER L'ENGINE (MAX 64 NUOVI TOKEN) ---", file=sys.stderr)

for idx, p in enumerate(prompts, 1):
    # Converte il testo in una lista di interi (Token ID)
    token_ids = tokenizer.encode(p, add_special_tokens=False)
    
    # Trasforma la lista di interi in una stringa separata da virgole senza spazi
    token_str = ",".join(map(str, token_ids))
    
    # Genera la riga di comando pulita che gptoss_run si aspetta
    command_line = f"GEN 64 temperature=0.7 {token_str}"
    
    # Questo finirà solo sullo schermo per tua comodità visiva
    print(f"Elaborato Prompt {idx}/{len(prompts)}: {p[:40]}...", file=sys.stderr)
    
    # Questo è l'UNICO output reale che verrà scritto nel file .txt
    print(command_line)

print("Conversione completata con successo!", file=sys.stderr)
