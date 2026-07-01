#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <math.h>
#include <pthread.h>

#define NUM_THREADS 4

// --- Βοηθητικές συναρτήσεις για τον Quick-Select ---
void swap_float(float *a, float *b) { float t = *a; *a = *b; *b = t; }
void swap_int(int *a, int *b) { int t = *a; *a = *b; *b = t; }

int partition(float *arr, int *idx, int left, int right) {
    float pivot = arr[right];
    int i = left - 1;
    for (int j = left; j < right; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap_float(&arr[i], &arr[j]);
            swap_int(&idx[i], &idx[j]);
        }
    }
    swap_float(&arr[i + 1], &arr[right]);
    swap_int(&idx[i + 1], &idx[right]);
    return i + 1;
}

// Βρίσκει τα k μικρότερα στοιχεία και τα τοποθετεί στις θέσεις 0 έως k-1
void quick_select(float *arr, int *idx, int left, int right, int k) {
    if (left >= right) return;
    int pivot_index = partition(arr, idx, left, right);
    int count = pivot_index - left + 1;
    if (count == k) return;
    if (count > k) quick_select(arr, idx, left, pivot_index - 1, k);
    else quick_select(arr, idx, pivot_index + 1, right, k - count);
}
// --------------------------------------------------

typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    int M;
    int d;
    int total_N;
    int k;              // Πόσους γείτονες ψάχνουμε
    float *Q;
    float *C_block;
    float *D_block;
    float *D_full;
    float *Q_sq;
    float *C_sq;
    float *local_k_dists;  // Πίνακας για τις k τοπικές αποστάσεις αυτού του νήματος
    int *local_k_indices;  // Πίνακας για τους k τοπικούς δείκτες αυτού του νήματος
} ThreadData;

void generate_random_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

void* compute_knn_block(void* arg) {
    ThreadData *data = (ThreadData*)arg;
    int local_N = data->end_row - data->start_row;

    // 1. Υπολογισμός Αποστάσεων (OpenBLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                data->M, local_N, data->d,
                -2.0f, data->Q, data->d,
                data->C_block, data->d,
                0.0f, data->D_block, data->total_N);

    // Βοηθητικός πίνακας δεικτών για τον quick-select (τον φτιάχνουμε μία φορά ανά νήμα)
    int *indices = (int *)malloc(local_N * sizeof(int));

    // 2. Ολοκλήρωση εξίσωσης και εύρεση τοπικών k γειτόνων για κάθε Query
    for (int i = 0; i < data->M; i++) {
        for (int j = 0; j < local_N; j++) {
            int global_j = data->start_row + j;
            int index = i * data->total_N + global_j;
            
            data->D_full[index] += data->Q_sq[i] + data->C_sq[global_j];
            if (data->D_full[index] < 0) data->D_full[index] = 0;
            data->D_full[index] = sqrt(data->D_full[index]);
            
            indices[j] = global_j; // Αποθήκευση του αρχικού δείκτη
        }

        // Τρέχουμε quick-select ΜΟΝΟ στο κομμάτι που υπολόγισε αυτό το νήμα
        float *current_row_dists = data->D_full + (i * data->total_N) + data->start_row;
        
        // Ψάχνουμε τα k μικρότερα (ή local_N αν το k είναι μεγαλύτερο από το μπλοκ)
        int elements_to_find = (data->k < local_N) ? data->k : local_N;
        quick_select(current_row_dists, indices, 0, local_N - 1, elements_to_find);

        // Αποθήκευση των τοπικών k γειτόνων στους κοινούς πίνακες επιστροφής
        for (int x = 0; x < elements_to_find; x++) {
            int out_idx = (i * NUM_THREADS * data->k) + (data->thread_id * data->k) + x;
            data->local_k_dists[out_idx] = current_row_dists[x];
            data->local_k_indices[out_idx] = indices[x];
        }
    }

    free(indices);
    return NULL;
}

int main() {
    srand(time(NULL));

    int N = 4000; 
    int M = 50;   
    int d = 128;  
    int k = 5;    // Θέλουμε τους 5 πιο κοντινούς γείτονες

    float *C = (float *)malloc(N * d * sizeof(float));
    float *Q = (float *)malloc(M * d * sizeof(float));
    float *D = (float *)malloc(M * N * sizeof(float));

    generate_random_matrix(C, N, d);
    generate_random_matrix(Q, M, d);

    float *C_sq = (float *)calloc(N, sizeof(float));
    float *Q_sq = (float *)calloc(M, sizeof(float));

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < d; j++) C_sq[i] += C[i * d + j] * C[i * d + j];
    }
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < d; j++) Q_sq[i] += Q[i * d + j] * Q[i * d + j];
    }

    // Πίνακες για να μαζέψουμε τα τοπικά αποτελέσματα από τα νήματα
    // Μέγεθος: M queries * NUM_THREADS * k γείτονες
    float *all_local_dists = (float *)malloc(M * NUM_THREADS * k * sizeof(float));
    int *all_local_indices = (int *)malloc(M * NUM_THREADS * k * sizeof(int));

    // --- ΕΝΑΡΞΗ PTHREADS ---
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    int block_size = N / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        thread_data[t].thread_id = t;
        thread_data[t].start_row = t * block_size;
        thread_data[t].end_row = (t == NUM_THREADS - 1) ? N : (t + 1) * block_size;
        thread_data[t].M = M;
        thread_data[t].d = d;
        thread_data[t].total_N = N;
        thread_data[t].k = k;
        thread_data[t].Q = Q;
        thread_data[t].C_block = C + (thread_data[t].start_row * d);
        thread_data[t].D_block = D + thread_data[t].start_row;
        thread_data[t].D_full = D;
        thread_data[t].Q_sq = Q_sq;
        thread_data[t].C_sq = C_sq;
        thread_data[t].local_k_dists = all_local_dists;
        thread_data[t].local_k_indices = all_local_indices;

        pthread_create(&threads[t], NULL, compute_knn_block, (void*)&thread_data[t]);
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    // --- ΛΗΞΗ PTHREADS ---

    // --- ΒΗΜΑ ΣΥΓΧΩΝΕΥΣΗΣ (MERGE) ---
    // Τώρα έχουμε NUM_THREADS * k υποψήφιους γείτονες για κάθε query.
    // Πρέπει να βρούμε τους τελικούς k.
    printf("\nΈνωση αποτελεσμάτων και εύρεση των %d τελικών γειτόνων...\n", k);
    for (int i = 0; i < M; i++) {
        int candidates_count = NUM_THREADS * k;
        float *query_candidates_dists = all_local_dists + (i * candidates_count);
        int *query_candidates_indices = all_local_indices + (i * candidates_count);

        quick_select(query_candidates_dists, query_candidates_indices, 0, candidates_count - 1, k);
        
        // Εκτύπωση για επαλήθευση (για το πρώτο query μόνο, για να μη γεμίσει η οθόνη)
        if (i == 0) {
            printf("Για το Query 0, οι %d πιο κοντινοί γείτονες είναι:\n", k);
            for (int x = 0; x < k; x++) {
                printf(" - Σημείο %d (Απόσταση: %f)\n", query_candidates_indices[x], query_candidates_dists[x]);
            }
        }
    }

    free(C); free(Q); free(D); free(C_sq); free(Q_sq);
    free(all_local_dists); free(all_local_indices);

    return 0;
}