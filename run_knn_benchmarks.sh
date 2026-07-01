#!/bin/bash

CSV_FILE="knn_results.csv"

# Create the CSV header for 5 runs
echo "Threads,Run 1,Run 2,Run 3,Run 4,Run 5" > $CSV_FILE

# The thread counts to benchmark (1 is the baseline)
THREADS_LIST="1 2 4 6 8 12"

echo "Starting k-NN Stress Test..."
echo "This will take several minutes. Please wait."
echo "--------------------------------------------------------"

for t in $THREADS_LIST
do
    echo "-> Compiling and executing 5 runs for $t threads..."
    
    # Compile passing the thread count
    gcc -O3 -Wall -DNUM_THREADS=$t -o knn knn.c -lopenblas -lm -lpthread
    
    # Write the thread count to the CSV
    echo -n "$t" >> $CSV_FILE
    
    # Execute 5 times
    for i in {1..5}
    do
        # Extract the execution time
        EXEC_TIME=$(./knn | grep "Total Execution Time" | awk '{print $7}')
        echo -n ",$EXEC_TIME" >> $CSV_FILE
    done
    
    echo "" >> $CSV_FILE
done

echo "--------------------------------------------------------"
echo "Done! All experiments completed successfully."
echo "Results saved to: $CSV_FILE"