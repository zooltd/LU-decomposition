#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdexcept>

#define ZERO 1e-5

/* matrix size */
int msz = 8000;
/* block size */
int bsz = 64;
/* number of blocks */
int nb;

#define IDX(row, col) ((row) *msz + (col))

void randomize(double *A, double min, double max);
void blocked_lu(double *A, int *P, double ***blocks, double *buf);
void validate(double *A, const int *P, const double *A_cp);

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4)
        throw std::invalid_argument("./lup [matrix size] [number of threads] [block size](optional)(200 by default)");

    msz = (int) std::stol(argv[1]);
    int num_threads = (int) std::stol(argv[2]);
    if (argc == 4) bsz = (int) std::stol(argv[3]);

    assert(bsz >= 1 && msz >= 1);
    /* assume matrix size is multiple of block size for simplicity :) */
    assert(msz % bsz == 0);

    nb = msz / bsz;

    omp_set_num_threads(num_threads);

    double *A = (double *) aligned_alloc(64, msz * msz * sizeof(double));
    randomize(A, 0, 1e5);

    double *A_cp = (double *) aligned_alloc(64, msz * msz * sizeof(double));
    memcpy((void *) A_cp, (void *) A, sizeof(double) * msz * msz);

    int *P = new int[msz];
    for (int i = 0; i < msz; ++i) P[i] = i;

    double *buf = (double *) aligned_alloc(64, msz * sizeof(double));
    double ***blocks = (double ***) malloc(nb * sizeof(double **));
    /* map matrix A to tiled matrix */
    for (int i = 0; i < nb; i++) {
        blocks[i] = (double **) malloc(nb * sizeof(double *));
        for (int j = 0; j < nb; j++)
            blocks[i][j] = A + (i * msz + j) * bsz;
    }

    double t1 = omp_get_wtime();
    blocked_lu(A, P, blocks, buf);
    double t2 = omp_get_wtime();
    std::cout << "Duration: " << t2 - t1 << std::endl;

    free(buf);
    for (int i = 0; i < nb; i++) free(blocks[i]);
    free(blocks);

    validate(A, P, A_cp);

    free(A);
    free(A_cp);
    free(P);
    return 0;
}

void randomize(double *A, double min, double max) {
    class rng {
    public:
        rng() = default;
        rng(double min, double max) : gen{std::random_device()()}, dist{min, max} {}
        double operator()() { return dist(gen); }

    private:
        std::mt19937 gen;
        std::uniform_real_distribution<double> dist;
    };

    int max_threads = omp_get_max_threads();
    /* each thread use a rng with a different seed */
    rng *rngs = new rng[max_threads];
    for (int i = 0; i < max_threads; ++i) rngs[i] = rng(min, max);
#pragma omp parallel for schedule(static, 1) num_threads(max_threads)
    for (int i = 0; i < msz; ++i) {
        for (int j = 0; j < msz; ++j) {
            int thread_id = omp_get_thread_num();
            rng &rng = rngs[thread_id];
            A[IDX(i, j)] = rng();
        }
    }
    delete[] rngs;
}


class rng {
public:
    rng() = default;
    rng(double min, double max) : gen{std::random_device()()}, dist{min, max} {}
    double operator()() { return dist(gen); }

private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist;
};


void block_pivot(int k, double *A, int *P, double *buf);
void cal_sub_matrix(double *C, const double *A, const double *B);
void cal_lu_decomp(double *A);
void cal_row(double *B, const double *A);
void cal_col(double *B, const double *A);
void panel_decomp(int bi, double *A, int *P, double *buf);


void blocked_lu(double *A, int *P, double ***blocks, double *buf) {
#pragma omp parallel
#pragma omp single
    for (int k = 0; k < nb; k++) {

        /* decomposition blocks[k:,k]*/
        panel_decomp(k, A, P, buf);

        /* cal blocks[k,k:]*/
        /* blocks[k][j] = L(blocks[k][k]) * blocks[k][j] */
        for (int j = k + 1; j < nb; j++)
#pragma omp task depend(inout : blocks[k][j])
            cal_row(blocks[k][j], blocks[k][k]);

        /* update blocks[k+bsz:,k+bsz:]*/
        /* blocks[ii][jj] -= blocks[ii][k] * blocks[k][jj] */
        for (int ii = k + 1; ii < nb; ii++)
            for (int jj = k + 1; jj < nb; jj++)
#pragma omp task depend(inout : blocks[ii][jj]) depend(in : blocks[k][jj])
                cal_sub_matrix(blocks[ii][jj], blocks[ii][k], blocks[k][jj]);
#pragma omp taskwait
    }
}

/* matrix elem */
struct elem {
    int idx;
    double val;
};

/* single thread in-place LU decomposition with partial pivoting */
void panel_decomp(int bi, double *A, int *P, double *buf) {
    elem pivot{};
    double recAii;
    for (int i = bi * bsz; i < bi * bsz + bsz; ++i) {
        pivot = {0, 0};

        for (int j = i; j < msz; ++j) {
            if (pivot.val < std::abs(A[IDX(j, i)])) {
                pivot.val = std::abs(A[IDX(j, i)]);
                pivot.idx = j;
            }
        }

        /* check for numeral stability */
        if (pivot.val < ZERO)
            throw std::invalid_argument("singular matrix");

        if (pivot.idx != i) {
            std::swap(P[pivot.idx], P[i]);
            /* a(k,:) <-> a(k',:) */
            std::memcpy(buf, &A[IDX(i, 0)], msz * sizeof(double));
            std::memcpy(&A[IDX(i, 0)], &A[IDX(pivot.idx, 0)], msz * sizeof(double));
            std::memcpy(&A[IDX(pivot.idx, 0)], buf, msz * sizeof(double));
        }

        /* strength reduction: division -> multiplication */
        recAii = 1.0f / A[IDX(i, i)];
        for (int j = i + 1; j < msz; j++)
            A[IDX(j, i)] *= recAii;
        for (int j = i + 1; j < msz; j++)
            for (int k = i + 1; k < bi * bsz + bsz; k++)
                A[IDX(j, k)] -= A[IDX(j, i)] * A[IDX(i, k)];
    }
}


/* forward substitution */
/* B = L(A) * B', solve B' */
void cal_row(double *B, const double *A) {
    for (int i = 0; i < bsz; ++i)
        for (int j = 0; j < bsz; ++j)
            for (int k = 0; k < j; ++k)
                B[IDX(j, i)] -= A[IDX(j, k)] * B[IDX(k, i)];
}

/* backward substitution */
/* B = B' * U(A), solve B' */
void cal_col(double *B, const double *A) {
    for (int i = 0; i < bsz; ++i)
        for (int j = 0; j < bsz; ++j) {
            for (int k = 0; k < j; ++k)
                B[IDX(i, j)] -= A[IDX(k, j)] * B[IDX(i, k)];
            B[IDX(i, j)] /= A[IDX(j, j)];
        }
}

/* C -= A*B */
void cal_sub_matrix(double *C, const double *A, const double *B) {
    for (int i = 0; i < bsz; ++i)
        for (int j = 0; j < bsz; ++j)
            for (int k = 0; k < bsz; ++k)
                C[IDX(i, j)] -= A[IDX(i, k)] * B[IDX(k, j)];
}

/* in-place LU decomposition without pivoting */
void cal_lu_decomp(double *A) {
    for (int i = 0; i < bsz; i++) {
        const double div = 1 / A[IDX(i, i)];
        for (int j = i + 1; j < bsz; j++) {
            A[IDX(j, i)] *= div;
            for (int k = i + 1; k < bsz; k++)
                A[IDX(j, k)] -= A[IDX(j, i)] * A[IDX(i, k)];
        }
    }
}

void cal_L(const double *A, double *L) {
    for (int i = 0; i < msz; ++i) {
        L[IDX(i, i)] = 1;
        for (int j = 0; j < i; ++j)
            L[IDX(i, j)] = A[IDX(i, j)];
    }
}

void cal_U(const double *A, double *U) {
    for (int i = 0; i < msz; ++i)
        for (int j = i; j < msz; ++j)
            U[IDX(i, j)] = A[IDX(i, j)];
}

void cal_PA(double *PA, const int *P, const double *A) {
    for (int i = 0; i < msz; ++i)
        for (int j = 0; j < msz; ++j)
            PA[IDX(i, j)] = A[IDX(P[i], j)];
}

void cal_LU(double *LU, const double *L, const double *U) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < msz; ++i)
        for (int j = 0; j < msz; ++j)
            for (int k = 0; k < msz; ++k)
                LU[IDX(i, j)] += L[IDX(i, k)] * U[IDX(k, j)];
}

double l21_norm(double *A, const double *LU) {
    for (int i = 0; i < msz; i++)
        for (int j = 0; j < msz; j++)
            A[IDX(i, j)] -= LU[IDX(i, j)];

    double norm = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : norm)
    for (int col = 0; col < msz; ++col) {
        double ans = 0.0;
        for (int row = 0; row < msz; ++row)
            ans += std::pow(A[IDX(row, col)], 2);
        norm += std::sqrt(ans);
    }
    return norm;
}

void validate(double *A, const int *P, const double *A_cp) {
    double *L = new double[msz * msz]{0};
    double *U = new double[msz * msz]{0};
    double *PA = new double[msz * msz]{0};
    double *LU = new double[msz * msz]{0};

    cal_L(A, L);
    cal_U(A, U);
    cal_PA(PA, P, A_cp);
    cal_LU(LU, L, U);

    double norm = l21_norm(PA, LU);
    std::cout << "L2,1 norm: " << norm << std::endl;

    delete[] L;
    delete[] U;
    delete[] LU;
    delete[] PA;
}
