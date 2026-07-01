#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>
#include <pthread.h>

#define NUM_THREADS 4 // Μπορείς να το αλλάξεις ανάλογα με τους πυρήνες του υπολογιστή σου

// Δομή για να περνάμε τα ορίσματα στο κάθε νήμα
typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    int M;           // Αριθμός Queries
    int d;           // Διαστάσεις
    int total_N;     // Το συνολικό N (απαραίτητο για το offset του D)
    float *Q;
    float *C_block;  // Δείκτης στο σημείο εκκίνησης του C για αυτό το thread
    float *D_block;  // Δείκτης στο σημείο εκκίνησης του D για αυτό το thread
    float *D_full;   // Ολόκληρος ο πίνακας D
    float *Q_sq;
    float *C_sq;
} ThreadData;

void generate_random_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Η συνάρτηση που εκτελεί το κάθε νήμα
void* compute_knn_block(void* arg) {
    ThreadData *data = (ThreadData*)arg;
    int local_N = data->end_row - data->start_row; // Πόσα σημεία αναλαμβάνει το νήμα

    printf("Νήμα %d: Υπολογισμός σημείων από %d έως %d...\n", 
           data->thread_id, data->start_row, data->end_row - 1);

    // 1. Πολλαπλασιασμός Μητρώων: D_block = -2 * Q * C_block^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                data->M, local_N, data->d,
                -2.0f, data->Q, data->d,
                data->C_block, data->d,
                0.0f, data->D_block, data->total_N); // Προσοχή: Το ldc είναι το total_N

    // 2. Ολοκλήρωση της Εξίσωσης: Τετράγωνα και Ρίζα
    for (int i = 0; i < data->M; i++) {
        for (int j = 0; j < local_N; j++) {
            // Ο πραγματικός δείκτης j σε σχέση με ολόκληρο τον πίνακα
            int global_j = data->start_row + j; 
            
            // Η ακριβής θέση στον μονοδιάστατο πίνακα D
            int index = i * data->total_N + global_j; 
            
            data->D_full[index] += data->Q_sq[i] + data->C_sq[global_j];
            
            if (data->D_full[index] < 0) data->D_full[index] = 0;
            data->D_full[index] = sqrt(data->D_full[index]);
        }
    }

    return NULL;
}

int main() {
    srand(time(NULL));

    int N = 4000; // Corpus (C) - Το αυξήσαμε λίγο για να φανεί η δουλειά των νημάτων
    int M = 50;   // Queries (Q)
    int d = 128;  // Διαστάσεις

    float *C = (float *)malloc(N * d * sizeof(float));
    float *Q = (float *)malloc(M * d * sizeof(float));
    float *D = (float *)malloc(M * N * sizeof(float));

    generate_random_matrix(C, N, d);
    generate_random_matrix(Q, M, d);

    float *C_sq = (float *)calloc(N, sizeof(float));
    float *Q_sq = (float *)calloc(M, sizeof(float));

    // Υπολογισμός τετραγώνων (Γίνεται σειριακά γιατί είναι πολύ γρήγορο, 
    // αλλά θα μπορούσε να μπει και στα νήματα)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < d; j++) C_sq[i] += C[i * d + j] * C[i * d + j];
    }
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < d; j++) Q_sq[i] += Q[i * d + j] * Q[i * d + j];
    }

    // --- ΕΝΑΡΞΗ PTHREADS ---
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    int block_size = N / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        thread_data[t].thread_id = t;
        thread_data[t].start_row = t * block_size;
        
        // Το τελευταίο νήμα παίρνει και τυχόν υπόλοιπα αν το N δεν διαιρείται ακριβώς
        thread_data[t].end_row = (t == NUM_THREADS - 1) ? N : (t + 1) * block_size;
        
        thread_data[t].M = M;
        thread_data[t].d = d;
        thread_data[t].total_N = N;
        thread_data[t].Q = Q;
        thread_data[t].C_block = C + (thread_data[t].start_row * d); // Μετατόπιση δείκτη στο C
        thread_data[t].D_block = D + thread_data[t].start_row;       // Μετατόπιση δείκτη στο D
        thread_data[t].D_full = D;
        thread_data[t].Q_sq = Q_sq;
        thread_data[t].C_sq = C_sq;

        // Δημιουργία και εκτέλεση νήματος
        pthread_create(&threads[t], NULL, compute_knn_block, (void*)&thread_data[t]);
    }

    // Αναμονή όλων των νημάτων να τελειώσουν
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    // --- ΛΗΞΗ PTHREADS ---

    printf("\nΟ παράλληλος υπολογισμός των αποστάσεων ολοκληρώθηκε επιτυχώς με %d νήματα!\n", NUM_THREADS);

    free(C); free(Q); free(D);
    free(C_sq); free(Q_sq);

    return 0;
}