#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>

// Συνάρτηση για τη δημιουργία τυχαίων πινάκων 1D (αφού η C αποθηκεύει τους 2D ως 1D)
void generate_random_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX; // Τυχαίοι αριθμοί από 0.0 έως 1.0
    }
}

int main() {
    srand(time(NULL));

    // Ορισμός διαστάσεων (μικρά νούμερα για αρχή ώστε να ελέγχουμε εύκολα)
    int N = 1000; // Αριθμός σημείων στο Corpus set (C)
    int M = 50;   // Αριθμός σημείων στο Query set (Q)
    int d = 128;  // Αριθμός διαστάσεων

    printf("Δημιουργία δεδομένων: C(%d x %d), Q(%d x %d)...\n", N, d, M, d);

    // Δέσμευση μνήμης για τους πίνακες
    float *C = (float *)malloc(N * d * sizeof(float));
    float *Q = (float *)malloc(M * d * sizeof(float));
    float *D = (float *)malloc(N * M * sizeof(float)); // Πίνακας Αποστάσεων D(M x N)

    // Γέμισμα με τυχαία δεδομένα
    generate_random_matrix(C, N, d);
    generate_random_matrix(Q, M, d);

    printf("Τα δεδομένα δημιουργήθηκαν επιτυχώς!\n");

    // --- ΕΔΩ ΘΑ ΜΠΕΙ Ο ΚΩΔΙΚΑΣ ΤΗΣ OPENBLAS ---
    
    // Απελευθέρωση μνήμης στο τέλος
    free(C);
    free(Q);
    free(D);

    return 0;
}