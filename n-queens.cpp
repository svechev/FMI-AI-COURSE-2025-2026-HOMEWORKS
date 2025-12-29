#define _CRT_SECURE_NO_WARNINGS
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdlib>
using namespace std;

const bool PRINT_BOARD = false;



void print_board(const int* board, int n) {
    for (int r = 0; r < n; r++) {
        string row = "";
        for (int c = 0; c < n; c++) {
            if (board[c] == r)
                row.push_back('*');
            else {
                row.push_back('_');
            }
        }
        cout << row << "\n";
    }
}

void print_arr(const int* board, int n) {
    std::cout << "[";
    for (size_t i = 0; i < n - 1; i++) {
        std::cout << board[i] << ", ";
    }
    std::cout << board[n - 1] << "]";
}


int* solve_n_queens(int* board, int n) {
    int* row_count = new int[n];
    int* diag1_count = new int[2 * n - 1];
    int* diag2_count = new int[2 * n - 1];

    vector<int> max_cols;
    vector<int> best_rows;

    int rand_ind, max_conf;

    while (true) {

        fill(row_count, row_count + n, 0);
        fill(diag1_count, diag1_count + 2 * n - 1, 0);
        fill(diag2_count, diag2_count + 2 * n - 1, 0);

        //fill(board, board+n, -1);

        // Optimized MIN-CONFLICTS INITIALIZATION
        for (int c = 0; c < n; c++) {
            int min_conf = n + 420;

            for (int r = 0; r < n; r++) {
                int cval = row_count[r]
                    + diag1_count[r - c + n - 1]
                    + diag2_count[r + c];

                if (cval < min_conf) {
                    min_conf = cval;
                    best_rows.clear();
                    best_rows.push_back(r);
                }
                else if (cval == min_conf) {
                    best_rows.push_back(r);
                }
            }


            int r;
            if (best_rows.size() == 1) {
                r = best_rows[0];
            }
            else {
                rand_ind = rand() % best_rows.size();
                r = best_rows[rand_ind];
            }

            board[c] = r;
            row_count[r]++;
            diag1_count[r - c + n - 1]++;
            diag2_count[r + c]++;

            best_rows.clear();
        }


        // ---------------------------------------------------------
        // Main loop: O(n) iterations
        // ---------------------------------------------------------
        for (int _i = 0; _i < n * 5; _i++) {

            max_conf = -1;

            for (int c = 0; c < n; c++) {
                int r = board[c];
                int cval =
                    row_count[r] +
                    diag1_count[r - c + n - 1] +
                    diag2_count[r + c] -
                    3;

                if (cval > max_conf) {
                    max_conf = cval;
                    max_cols.clear();
                    max_cols.push_back(c);
                }
                else if (cval == max_conf) {
                    max_cols.push_back(c);
                }
            }

            if (max_conf == 0)
                return board;

            int col;
            if (max_cols.size() == 1) {
                col = max_cols[0];
            }
            else {
                rand_ind = rand() % max_cols.size();
                //cout << "randind: " << rand_ind << " and size: " << max_cols.size() << "\n";
                col = max_cols[rand_ind];
            }

            int old_row = board[col];
            //board[col] = -1;

            row_count[old_row]--;
            diag1_count[old_row - col + n - 1]--;
            diag2_count[old_row + col]--;

            int min_conf = n + 420;

            for (int r = 0; r < n; r++) {
                if (r == old_row) continue;
                int cval =
                    row_count[r] +
                    diag1_count[r - col + n - 1] +
                    diag2_count[r + col];

                if (cval < min_conf) {
                    min_conf = cval;
                    best_rows.clear();
                    best_rows.push_back(r);
                }
                else if (cval == min_conf) {
                    best_rows.push_back(r);
                }
            }

            int new_row;
            if (best_rows.size() == 1) {
                new_row = best_rows[0];
            }
            else {
                rand_ind = rand() % best_rows.size();
                new_row = best_rows[rand_ind];
            }

            board[col] = new_row;

            row_count[new_row]++;
            diag1_count[new_row - col + n - 1]++;
            diag2_count[new_row + col]++;

            max_cols.clear();
            best_rows.clear();
        }

        // if we reach here, restart
    }
}

int main() {
    int n;
    cin >> n;
    auto start = std::chrono::high_resolution_clock::now();
    //# TIMES_MS: alg=

    int* board = new int[n];
    if (n == 1) {
        board[0] = 0;
    }
    else if (n == 2 || n == 3) {
        cout << -1 << endl;
        return 0;
    }
    else {
        solve_n_queens(board, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);


    const char* fmi_env = std::getenv("FMI_TIME_ONLY");
    if (fmi_env && std::string(fmi_env) == "1") {
        std::cout << "# TIMES_MS: alg=" << time.count() << endl;
        return 0;
    }

    if (PRINT_BOARD) {
        print_board(board, n);
    }
    else {
        print_arr(board, n);
    }
    delete[] board;
    return 0;
}