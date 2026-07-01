#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>

// Συνάρτηση για τη δημιουργία τυχαίων πινάκων
void generate_random_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    srand(time(NULL));

    int N = 1000; // Corpus (C)
    int M = 50;   // Queries (Q)
    int d = 128;  // Διαστάσεις

    // 1. Δέσμευση Μνήμης
    float *C = (float *)malloc(N * d * sizeof(float));
    float *Q = (float *)malloc(M * d * sizeof(float));
    float *D = (float *)malloc(M * N * sizeof(float)); // Ο D θα έχει μέγεθος M x N

    generate_random_matrix(C, N, d);
    generate_random_matrix(Q, M, d);

    // 2. Υπολογισμός Αθροισμάτων Τετραγώνων (C^2 και Q^2)
    float *C_sq = (float *)calloc(N, sizeof(float));
    float *Q_sq = (float *)calloc(M, sizeof(float));

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < d; j++) C_sq[i] += C[i * d + j] * C[i * d + j];
    }
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < d; j++) Q_sq[i] += Q[i * d + j] * Q[i * d + j];
    }

    // 3. Πολλαπλασιασμός Μητρώων με OpenBLAS: Υπολογισμός D = -2 * Q * C^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, d,
                -2.0f, Q, d,
                C, d,
                0.0f, D, N);

    // 4. Ολοκλήρωση της Εξίσωσης (D = C^2 - 2QC^T + Q^2) και Τετραγωνική Ρίζα
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            D[i * N + j] += Q_sq[i] + C_sq[j]; // Πρόσθεση των τετραγώνων
            
            // Αποφυγή αρνητικών αριθμών λόγω σφαλμάτων κινητής υποδιαστολής (floating point precision)
            if (D[i * N + j] < 0) D[i * N + j] = 0; 
            
            D[i * N + j] = sqrt(D[i * N + j]); // Τετραγωνική ρίζα element-wise
        }
    }

    printf("Ο ακριβής πίνακας αποστάσεων %d x %d υπολογίστηκε επιτυχώς!\n", M, N);

    // 5. Απελευθέρωση Μνήμης
    free(C); free(Q); free(D);
    free(C_sq); free(Q_sq);

    return 0;
}