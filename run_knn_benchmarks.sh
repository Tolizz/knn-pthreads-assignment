#!/bin/bash

CSV_FILE="knn_results.csv"

# Φτιάχνουμε την επικεφαλίδα του CSV
echo "Threads,Run 1,Run 2,Run 3" > $CSV_FILE

# Οι αριθμοί των νημάτων που θέλουμε να ελέγξουμε (1 για σειριακό/baseline, και μετά τα pthreads)
THREADS_LIST="1 2 4 6 8 12"

echo "Ξεκινάει η δοκιμή αντοχής (Stress Test) για το k-NN..."
echo "Αυτό μπορεί να πάρει μερικά λεπτά, παρακαλώ περιμένετε."
echo "--------------------------------------------------------"

for t in $THREADS_LIST
do
    echo "-> Μεταγλώττιση και 3 εκτελέσεις για $t νήματα..."
    
    # Μεταγλώττιση: Περνάμε τον αριθμό των νημάτων με το -DNUM_THREADS
    gcc -O3 -Wall -DNUM_THREADS=$t -o knn knn.c -lopenblas -lm -lpthread
    
    # Γράφουμε τον αριθμό των νημάτων στο CSV
    echo -n "$t" >> $CSV_FILE
    
    # Τρέχουμε το πρόγραμμα 3 φορές (ο πίνακας 100.000 σημείων είναι βαρύς, οι 3 φορές αρκούν)
    for i in {1..3}
    do
        # Το awk '{print $7}' παίρνει ακριβώς τον αριθμό των δευτερολέπτων από το print της C
        EXEC_TIME=$(./knn | grep "Total Execution Time" | awk '{print $7}')
        echo -n ",$EXEC_TIME" >> $CSV_FILE
    done
    
    echo "" >> $CSV_FILE
done

echo "--------------------------------------------------------"
echo "Τέλεια! Τα πειράματα ολοκληρώθηκαν."
echo "Τα αποτελέσματα αποθηκεύτηκαν στο αρχείο: $CSV_FILE"