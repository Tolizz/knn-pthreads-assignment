# k-NN Parallelization using Pthreads

Αυτό το αποθετήριο περιέχει την υλοποίηση του αλγορίθμου k-Nearest Neighbors (k-NN) σε C, παραλληλοποιημένου με χρήση POSIX Threads (Pthreads) και της βιβλιοθήκης OpenBLAS για βελτιστοποιημένους υπολογισμούς μητρώων.

## Προαπαιτούμενα (Dependencies)
Για τη σωστή μεταγλώττιση και εκτέλεση του κώδικα σε περιβάλλον Linux (ή WSL), απαιτείται η βιβλιοθήκη OpenBLAS. Μπορείτε να την εγκαταστήσετε τρέχοντας:
```bash
sudo apt-get update
sudo apt-get install libopenblas-dev

```

## Αναπαραγωγή Πειραμάτων (Benchmarking)

Για την αυτοματοποίηση των μετρήσεων έχει αναπτυχθεί το bash script `run_knn_benchmarks.sh`. Το script αναλαμβάνει να μεταγλωττίσει τον κώδικα για διαφορετικούς αριθμούς νημάτων (1, 2, 4, 6, 8, 12) και να εξάγει τους χρόνους σε ένα αρχείο `.csv`.

### Βήματα Εκτέλεσης:

**1. Κλωνοποίηση του αποθετηρίου:**

```bash
git clone https://github.com/Tolizz/knn-pthreads-assignment.git
cd knn-pthreads-assignment

```

**2. Κλείδωμα OpenBLAS (Σημαντικό!):**
Για να αποφευχθεί το φαινόμενο του Nested Parallelism (Thread Contention) όπου η OpenBLAS δημιουργεί δικά της εσωτερικά νήματα που συγκρούονται με τα Pthreads μας, πρέπει να κλειδώσουμε την OpenBLAS σε 1 νήμα:

```bash
export OPENBLAS_NUM_THREADS=1

```

**3. Εκτέλεση του Script:**
Δώστε δικαιώματα εκτέλεσης και τρέξτε το script:

```bash
chmod +x run_knn_benchmarks.sh
./run_knn_benchmarks.sh

```

Μόλις ολοκληρωθεί η διαδικασία (θα διαρκέσει μερικά λεπτά), τα αποτελέσματα των μετρήσεων θα είναι διαθέσιμα στο αρχείο `knn_results.csv`.

Η σχετική αναφορά βρίσκεται στο Project0.pdf
