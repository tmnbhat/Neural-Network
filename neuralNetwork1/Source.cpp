#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iomanip>
#include <math.h>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;

int inputLayerSize = 400;
int hiddenLayerSize = 25;
int numLabels = 10;
int lambda = 0;

class Matrix {
public:
	//long double arr[100][100];	//FOR SOME REASON UNABLE TO SET SIZE GREATER THAN 110x110
	long double *arr;
	int row, col;
	int size;
	long double J;
	Matrix() {
		arr = new long double[2100000];	//Matrix stored in a 1D array of RxC size in Heap form
	}
	Matrix(int row, int col) {
		this->row = row;
		this->col = col;
		this->arr = new long double[(row + 1) * (col + 1)];
	}
	~Matrix() {}
	void load(string filename, int row, int col);
	void multiply(Matrix* m1, Matrix* m2);			//OUTPUT IS ALWAYS THE MATRIX THAT IS BEING OPERATED ON.
	void multiply1(Matrix* m1, Matrix* m2);
	//	void recurMulti(Matrix* m1, Matrix* m2, Matrix*m, int size);
	void display();
	void appendOne(Matrix* m);
	void trans(Matrix* m);
	void sigmoid(Matrix* m);
	void vectorForm(Matrix* m, int k);
	void add(Matrix* m1, Matrix* m2);
	void subtract(Matrix* m1, Matrix* m2);
	void dotProduct(Matrix* m1, Matrix* m2);
	long double sum();
	void minus(Matrix* m);
	void ln(Matrix* m);
	void oneMatrix(int Row, int Col);
	void removeCol(Matrix* m);
	void dotSquare(Matrix* m);
	void generateRandWeights(int row, int col);
	void sigmoidGradient(Matrix* m);
	void divide(Matrix* M, long double m);
	void editThetaGrad(long double lambda, long double m, Matrix* Theta1);
	void combine(Matrix* m1, Matrix* m2);
	void decombine(Matrix* m, int sel, int row, int col);
	void copy(Matrix* input);
	void costFunction(Matrix* Theta, Matrix* X, Matrix* y, int input_layer_size, int hidden_layer_size, int num_labels, long double m, long double lambda, long double alpha);
	void feedForward(Matrix* Theta1, Matrix* Theta2, Matrix* X);
	void predict(Matrix* a3);
	double predAccuracy(Matrix* y);
};

void Matrix::load(string filename, int r, int c) {
	row = r;
	col = c;
	size_t count{};
	std::cout.precision(20);

	std::ifstream inFile{ filename }; // Create input stream object
	if (!inFile)					// Make sure the file stream is good
	{
		std::cout << "Failed to open file " << filename << std::endl;
		exit(0);
	}
	for (int i = 0; i < row*col; i++) {
		inFile >> arr[i];		//Read value from input
		count++;
		if (inFile.eof()) break;	//Break if EOF is reached
		if (row * col > 10)
			if (!(count % ((row * col) / 10)))
				std::cout << ". ";
	}
	//std::cout << '\n' << arr[1649];

	std::cout << "\n" << count << " numbers read from " << filename << std::endl;
}

void Matrix::multiply(Matrix* m1, Matrix* m2)
{
	time_t t1 = time(0);
	if ((*m2).row != (*m1).col) {
		std::cout << "Error : Cannot multiply matrices of sizes " << row << 'x' << col << " and " << (*m2).row << 'x' << (*m2).col << " respectively" << endl;
		exit(0);
	}
	col = (*m2).col;
	row = (*m1).row;
	long double temp = 0.0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			temp = 0.0;
			for (int k = 0; k < (*m1).col; k++)
				temp += ((*m1).arr[i * ((*m1).col) + k]) * ((*m2).arr[((*m2).col) * k + j]);
			arr[i * col + j] = temp;
		}
	}
	//cout << time(0) - t1 << endl;
	//cout << "Matrix sucessfully multiplied." << endl;
}

void copyMatrixtoVec(int R, int C, int random, Matrix* copyfrom, vector <long double> &copyto) {

	int N = R * C;
	int r = (*copyfrom).row;
	int c = (*copyfrom).col;
	long double a;
	/*for (int i = 0; i < n; i++)
	for (int j = 0; j < n; j++)
	v[i * n + j] = 0;*/
	// allocates each row of the matrix
	int k = 0;
	for (int i = 0; i < R; i++)
		for (int j = 0; j < C; j++)
		{
			if ((i >= r) || (j >= c)) {
				copyto[i * C + j] = 0.0;
				continue;
			}
			// initializes the matrix with random values between 0 and 10 (not used so far)
			if (random == 1) {
				a = rand();
				copyto[i * C + j] = (a - (int)a) * 10;
				continue;
			}
			if (random == 0) {
				copyto[i * C + j] = (*copyfrom).arr[k++];
			}
		}
}

void mmult(int nX[2], int nY[2], int Xpitch, vector <long double> X, int Ypitch, vector <long double> Y,
	int Zpitch, vector <long double> &Z) {
	long double temp = 0.0;
	for (int i = 0; i < nX[0]; i++) {
		for (int j = 0; j < nY[1]; j++) {
			temp = 0.0;
			for (int k = 0; k < nX[1]; k++)
				temp += (X[i * (Xpitch)+k]) * (Y[(Ypitch)* k + j]);
			Z[i * Zpitch + j] = temp;
		}
	}
}

//
// S = X + Y
//
void madd(int Rmode, int Cmode, int r, int c, vector <long double> &X, vector <long double> &Y, int Spitch, vector <long double> &S) {
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			S[(i + r*Rmode)*Spitch + (j + c*Cmode)] = X[i*c + j] + Y[i*c + j];
}

//
// S = X - Y
//
void msub(int Rmode, int Cmode, int r, int c, vector <long double> &X, vector <long double> &Y, int Spitch, vector <long double> &S) {
	for (int i = 0; i < r; i++)
		for (int j = 0; j < c; j++)
			S[(i + r*Rmode)*Spitch + (j + c*Cmode)] = X[i*c + j] - Y[i*c + j];
}

unsigned int nextPowerOfTwo(int n) {
	return pow(2, int(ceil(log2(n))));
}

void mmult_fast(int NX[2], int NY[2],
	int Xpitch, vector <long double> &X,
	int Ypitch, vector <long double> &Y,
	int Zpitch, vector <long double> &Z) {
	//
	// Recursive base case.
	// If matrices have ANY ONE DIMENSION 'limitSize' or smaller we just use
	// the conventional algorithm.
	// At what size we should switch will vary based
	// on hardware platform.
	//
	int limitSize = 16;
	//cout << " NX = " << NX[0] << ' ' << NX[1] << endl;
	//cout << " NY = " << NY[0] << ' ' << NY[1] << '\n' << endl;
	if ((NX[0] <= limitSize) || (NX[1] <= limitSize) || (NY[0] <= limitSize) || (NY[1] <= limitSize)) {
		mmult(NX, NY, Xpitch, X, Ypitch, Y, Zpitch, Z);
		return;
	}

	int nX[] = { NX[0] / 2, NX[1] / 2 };    // sizes of sub-matrices of X, Y (row, col) 
	int nY[] = { NY[0] / 2, NY[1] / 2 };

	int sz1 = nX[0] * nX[1];
	int sz2 = nY[0] * nY[1];
	int sz3 = nX[0] * nY[1];

	vector <long double> A(sz1, 0);
	vector <long double> B(sz1, 0);
	vector <long double> C(sz1, 0);
	vector <long double> D(sz1, 0);

	vector <long double> E(sz2, 0);
	vector <long double> F(sz2, 0);
	vector <long double> G(sz2, 0);
	vector <long double> H(sz2, 0);
	/*long double *A = (long double *)malloc(sz1);
	long double *B = (long double *)malloc(sz1);
	long double *C = (long double *)malloc(sz1);
	long double *D = (long double *)malloc(sz1);

	A = X;    // A-D matrices embedded in X
	B = X + nX[1];
	C = X + nX[0] * Xpitch;
	D = C + nX[1];

	long double *E = (long double *)malloc(sz2);
	long double *F = (long double *)malloc(sz2);
	long double *G = (long double *)malloc(sz2);
	long double *H = (long double *)malloc(sz2);

	E = Y;    // E-H matrices embeded in Y
	F = Y + nY[1];
	G = Y + nY[0] * Ypitch;
	H = G + nY[1];

	long double *P[7];    // allocate temp matrices off heap
	for (int i = 0; i < 7; i++)
	P[i] = (long double *)malloc(sz3);
	long double *T1 = (long double *)malloc(sz1);
	long double *T2 = (long double *)malloc(sz2);
	long double *U1 = (long double *)malloc(sz3);
	long double *U2 = (long double *)malloc(sz3);*/

	for (int i = 0; i < nX[0]; i++) {
		for (int j = 0; j < nX[1]; j++) {
			A[i * nX[1] + j] = X[i * NX[1] + j];
			B[i * nX[1] + j] = X[i * NX[1] + (j + nX[1])];
			C[i * nX[1] + j] = X[(i + nX[1]) * NX[0] + j];
			D[i * nX[1] + j] = X[(i + nX[1]) * NX[0] + (j + nX[1])];
		}
	}

	for (int i = 0; i < nY[0]; i++) {
		for (int j = 0; j < nY[1]; j++) {
			E[i * nY[1] + j] = Y[i * NY[1] + j];
			F[i * nY[1] + j] = Y[i * NY[1] + (j + nY[1])];
			G[i * nY[1] + j] = Y[(i + nY[1]) * NY[0] + j];
			H[i * nY[1] + j] = Y[(i + nY[1]) * NY[0] + (j + nY[1])];
		}
	}

	vector<vector<long double>> P(7, vector<long double>(sz3));
	vector <long double> T1(sz1, 0);
	vector <long double> T2(sz2, 0);
	vector <long double> U1(sz3, 0);
	vector <long double> U2(sz3, 0);

	vector <long double> c11(sz3, 0);
	vector <long double> c12(sz3, 0);
	vector <long double> c21(sz3, 0);
	vector <long double> c22(sz3, 0);

	// P0 = A*(F - H);
	msub(0, 0, nY[0], nY[1], F, H, nY[1], T2);
	mmult_fast(nX, nY, nX[1], A, nY[1], T2, nY[1], P[0]);

	// P1 = (A + B)*H
	madd(0, 0, nX[0], nX[1], A, B, nX[1], T1);
	mmult_fast(nX, nY, nX[1], T1, nY[1], H, nY[1], P[1]);

	// P2 = (C + D)*E
	madd(0, 0, nX[0], nX[1], C, D, nX[1], T1);
	mmult_fast(nX, nY, nX[1], T1, nY[1], E, nY[1], P[2]);

	// P3 = D*(G - E);
	msub(0, 0, nY[0], nY[1], G, E, nY[1], T2);
	mmult_fast(nX, nY, nX[1], D, nY[1], T2, nY[1], P[3]);

	// P4 = (A + D)*(E + H)
	madd(0, 0, nX[0], nX[1], A, D, nX[1], T1);
	madd(0, 0, nY[0], nY[1], E, H, nY[1], T2);
	mmult_fast(nX, nY, nX[1], T1, nY[1], T2, nY[1], P[4]);

	// P5 = (B - D)*(G + H)
	msub(0, 0, nX[0], nX[1], B, D, nX[1], T1);
	madd(0, 0, nY[0], nY[1], G, H, nY[1], T2);
	mmult_fast(nX, nY, nX[1], T1, nY[1], T2, nY[1], P[5]);

	// P6 = (A - C)*(E + F)
	msub(0, 0, nX[0], nX[1], A, C, nX[1], T1);
	madd(0, 0, nY[0], nY[1], E, F, nY[1], T2);
	mmult_fast(nX, nY, nX[1], T1, nY[1], T2, nY[1], P[6]);

	// Z upper left = (P3 + P4) + (P5 - P1)
	madd(0, 0, nX[0], nY[1], P[4], P[3], nY[1], U1);
	msub(0, 0, nX[0], nY[1], P[5], P[1], nY[1], U2);
	madd(0, 0, nX[0], nY[1], U1, U2, Zpitch, Z);
	//madd(0, 0, nX[0], nY[1], U1, U2, nY[1], c11);

	// Z lower left = P2 + P3
	//madd(nX[0], nY[1], nY[1], P[2], nY[1], P[3], Zpitch, Z + nX[0] * Zpitch);
	madd(1, 0, nX[0], nY[1], P[2], P[3], Zpitch, Z);
	//madd(0, 0, nX[0], nY[1], P[2], P[3], nY[1], c21);

	// Z upper right = P0 + P1
	//madd(nX[0], nY[1], nY[1], P[0], nY[1], P[1], Zpitch, Z + nY[1]);
	madd(0, 1, nX[0], nY[1], P[0], P[1], Zpitch, Z);
	//madd(0, 0, nX[0], nY[1], P[0], P[1], nY[1], c12);

	// Z lower right = (P0 + P4) - (P2 + P6)
	madd(0, 0, nX[0], nY[1], P[0], P[4], nY[1], U1);
	madd(0, 0, nX[0], nY[1], P[2], P[6], nY[1], U2);
	//msub(nX[0], nY[1], nY[1], U1, nY[1], U2, Zpitch, Z + nX[0] * Zpitch + nY[1]);
	msub(1, 1, nX[0], nY[1], U1, U2, Zpitch, Z);
	//msub(0, 0, nX[0], nY[1], U1, U2, nY[1], c22);

	/*for (int i = 0; i < nX[0]; i++)
	for (int j = 0; j < nY[1]; j++) {
	Z[i * Zpitch + j] = c11[i * nY[1] + j];
	Z[i * Zpitch + (j + nY[1])] = c12[i * nY[1] + j];
	Z[(i + nX[0]) * Zpitch + j] = c21[i * nY[1] + j];
	Z[(i + nX[0]) * Zpitch + (j + nY[1])] = c22[i * nY[1] + j];
	}*/

	/*free(U1);  // deallocate temp matrices
	free(T1);
	free(U2);
	free(T2);
	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);
	free(H);
	for (int i = 6; i >= 0; i--)
	free(P[i]);*/
}

void Matrix::multiply1(Matrix* m1, Matrix* m2)
{
	time_t t1 = time(0);
	if ((*m2).row != (*m1).col) {
		std::cout << "Error : Cannot multiply matrices of sizes " << row << 'x' << col << " and " << (*m2).row << 'x' << (*m2).col << " respectively" << endl;
		exit(0);
	}
	col = (*m2).col;
	row = (*m1).row;

	/*long double* A;
	long double* B;
	long double* C;*/
	int M = nextPowerOfTwo((*m1).row);
	int N = nextPowerOfTwo((*m1).col);
	int P = nextPowerOfTwo((*m2).col);
	vector <long double> X(M * N, 0);
	vector <long double> Y(N * P, 0);
	vector <long double> Z(M * P, 0);
	copyMatrixtoVec(M, N, 0, m1, X);
	copyMatrixtoVec(N, P, 0, m2, Y);
	//C = allocate_real_matrix(M, P, row, col, 0, (*m2).arr);
	int NX[] = { M, N };
	int NY[] = { N, P };

	mmult_fast(NX, NY, N, X, P, Y, P, Z);		//TO DEBUG!!	
	int k = 0;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < P; j++) {
			if (i >= row || j >= col)
				continue;
			arr[k++] = Z[i * P + j];
		}
	/*free(A);
	free(B);
	free(C);*/
	cout << time(0) - t1 << endl;
	//cout << "Matrix sucessfully multiplied." << endl;
}

void Matrix::display() {
	std::cout.precision(3);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			std::cout << setw(6) << arr[i * col + j];
		std::cout << endl;
	}
	std::cout << endl;
}

void Matrix::trans(Matrix* m) {
	row = (*m).col;
	col = (*m).row;
	for (int i = 0; i < row * col; i++) {
		arr[i] = (*m).arr[(i % (*m).row) * (*m).col + i / (*m).row];
	}
	//cout << "Transpose taken." << endl;
}

void Matrix::appendOne(Matrix* m) {
	row = (*m).row;
	col = (*m).col + 1;
	int k = 0;
	for (int i = 0; i < row * (col + 1); i++) {
		if (!(i % col)) {
			arr[i] = 1.0;
			k++;
		}
		else
			arr[i] = (*m).arr[i - k];
	}
	//delete[] m.arr;
	//cout << row << " ones appended." << endl;
}

void Matrix::sigmoid(Matrix* m) {
	row = (*m).row;
	col = (*m).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = 1.0 / (1.0 + exp(-(*m).arr[i]));
	//cout << "sigmoid values taken." << endl;
}

void Matrix::vectorForm(Matrix* m, int k) {
	if ((*m).col != 1) {
		std::cout << "Error : Cannot convert into vector form since matrix of size " << row << 'x' << col << endl;
		exit(0);
	}
	row = (*m).row;
	col = k;
	for (int i = 0; i < (*m).row * (*m).col; i++)
		for (int j = 0; j < k; j++)
			arr[k * i + j] = (((*m).arr[i] - 1) == j);
	//cout << "Converted to vector form." << endl;
}

void Matrix::add(Matrix* m1, Matrix* m2) {
	if ((row != (*m2).row) || (col != (*m2).col)) {
		std::cout << "Error : Matrix addition not possible since matrices are of size " << row << 'x' << col << " and " << (*m2).row << 'x' << (*m2).col << " respectively" << endl;
		exit(0);
	}
	row = (*m1).row;
	col = (*m1).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = (*m1).arr[i] + (*m2).arr[i];
}

void Matrix::subtract(Matrix *m1, Matrix *m2) {
	if (((*m1).row != (*m2).row) || ((*m1).col != (*m2).col)) {
		std::cout << "Error : Matrix subtraction not possible since matrices are of size " << row << 'x' << col << " and " << (*m2).row << 'x' << (*m2).col << " respectively" << endl;
		exit(0);
	}
	row = (*m1).row;
	col = (*m1).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = (*m1).arr[i] - (*m2).arr[i];
}

void Matrix::dotProduct(Matrix* m1, Matrix* m2) {
	if (((*m1).row != (*m2).row) || ((*m1).col != (*m2).col)) {
		std::cout << "Error : Matrix dot product not possible." << endl;
		exit(0);
	}
	row = (*m1).row;
	col = (*m1).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = (*m1).arr[i] * (*m2).arr[i];
}

long double Matrix::sum() {
	long double s = 0;
	for (int i = 0; i < row * col; i++)
		s += arr[i];
	return s;
}

void Matrix::minus(Matrix* m) {
	row = (*m).row;
	col = (*m).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = -(*m).arr[i];
}

void Matrix::ln(Matrix* m) {
	row = (*m).row;
	col = (*m).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = logl((*m).arr[i]);
}

void Matrix::oneMatrix(int Row, int Col) {
	row = Row;
	col = Col;
	for (int i = 0; i < row * col; i++)
		arr[i] = 1.0;
}

void Matrix::removeCol(Matrix* m) {
	row = (*m).row;
	col = (*m).col - 1;
	int k = 0;
	for (int i = 0; i < row * col; i++) {
		if (!(i % col))
			k++;
		arr[i] = (*m).arr[i + k];
	}
}

void Matrix::dotSquare(Matrix* m) {
	row = (*m).row;
	col = (*m).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = pow((*m).arr[i], 2);
}

void Matrix::generateRandWeights(int r, int c) {
	row = r;
	col = c;
	time_t seconds;
	time(&seconds);
	double epsilon_init = .12;
	srand((unsigned int)seconds);
	for (int i = 0; i < r * c; i++)
		arr[i] = ((rand() % 10000) / 10000.0) * 2 * epsilon_init - epsilon_init;
	std::cout << "Generated random weights for Theta." << endl;
}

void Matrix::sigmoidGradient(Matrix* m) {
	long double sig;
	row = (*m).row;
	col = (*m).col;
	for (int i = 0; i < row * col; i++) {
		sig = 1.0 / (1.0 + exp(-(*m).arr[i]));
		arr[i] = (sig)* (1 - sig);
	}
}

void Matrix::divide(Matrix* temp, long double m) {
	row = (*temp).row;
	col = (*temp).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = (*temp).arr[i] / m;
}

void Matrix::editThetaGrad(long double lambda, long double m, Matrix* Theta1) {
	for (int i = 0; i < row * col; i++) {
		if (!(i % col))
			continue;
		else
			arr[i] += (*Theta1).arr[i] * (lambda / m);
	}
}

void Matrix::combine(Matrix* m1, Matrix* m2) {
	row = (*m1).row;										//kindof unecessary
	col = ((*m1).row * (*m1).col + (*m2).row * (*m2).col) / row;	//^
	size = (*m1).row * (*m1).col + (*m2).row * (*m2).col;				//the only place where "size" matters :P
																		//cout << com.row << "x" << com.col << endl;
	int i = 0;
	for (; i < (*m1).row * (*m1).col; i++)
		arr[i] = (*m1).arr[i];
	for (int j = 0; i < size; i++, j++)
		arr[i] = (*m2).arr[j];
}

void Matrix::decombine(Matrix* m, int sel, int r, int c) {
	//cout << size << endl;
	size = r * c;
	if (!sel) {
		row = r;
		col = c;
		for (int i = 0; i < r * c; i++)
			arr[i] = (*m).arr[i];
	}
	else {
		row = r;
		col = c;
		int j = 0;
		for (int i = (*m).size - r * c; i < (*m).size; i++, j++)
			arr[j] = (*m).arr[i];
	}
}

void Matrix::copy(Matrix *input) {
	row = (*input).row;
	col = (*input).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = (*input).arr[i];
	J = (*input).J;
}

void Matrix::costFunction(Matrix* Theta, Matrix* X, Matrix* y, int input_layer_size, int hidden_layer_size, int num_labels, long double m, long double lambda, long double alpha) {

	cout.precision(5);
	Matrix Theta1(hidden_layer_size, input_layer_size + 1), Theta2(num_labels, hidden_layer_size + 1);
	Matrix a1(m, input_layer_size + 1), z2(m, hidden_layer_size), a2(m, hidden_layer_size + 1), z3(m, num_labels), a3(m, num_labels);
	Matrix y_vec;
	Matrix d3(m, num_labels), d2(m, hidden_layer_size), D1(hidden_layer_size, m + 1), D2(num_labels, hidden_layer_size + 1);
	Matrix Theta1_grad(hidden_layer_size, input_layer_size + 1), Theta2_grad(num_labels, hidden_layer_size + 1);

	string yLocation = "C:\\Users\\Madhav\\Documents\\Visual Studio 2013\\Projects\\neuralNetworkPractice\\y.txt";
	string XLocation = "C:\\Users\\Madhav\\Documents\\Visual Studio 2013\\Projects\\neuralNetworkPractice\\X.txt";

	Theta1.decombine(Theta, 0, hidden_layer_size, input_layer_size + 1);
	Theta2.decombine(Theta, 1, num_labels, hidden_layer_size + 1);
	//cout << '\n' << "Theta1 and Theta2 extraction completed." << '\n' << endl;

	a1.appendOne(X);
	Matrix Theta1T(input_layer_size + 1, hidden_layer_size);
	Theta1T.trans(&Theta1);
	z2.multiply(&a1, &Theta1T);
	Matrix z2sig(m, hidden_layer_size);
	z2sig.sigmoid(&z2);
	a2.appendOne(&z2sig);
	Matrix Theta2T(hidden_layer_size + 1, num_labels);
	Theta2T.trans(&Theta2);
	z3.multiply(&a2, &Theta2T);
	a3.sigmoid(&z3);
	y_vec.vectorForm(y, num_labels);
	Matrix one(m, num_labels);
	one.oneMatrix(m, num_labels);

	delete[] Theta1T.arr;
	delete[] z2sig.arr;
	delete[] Theta2T.arr;

	Matrix oneMinusyvec(m, num_labels), lna3(m, num_labels), minusyvec(m, num_labels), oneMinusa3(m, num_labels);
	oneMinusyvec.subtract(&one, &y_vec);
	lna3.ln(&a3);
	minusyvec.minus(&y_vec);
	oneMinusa3.subtract(&one, &a3);
	Matrix lnoneMinusa3(m, num_labels), J1(m, num_labels), J2(m, num_labels);
	lnoneMinusa3.ln(&oneMinusa3);
	J1.dotProduct(&minusyvec, &lna3);
	J2.dotProduct(&oneMinusyvec, &lnoneMinusa3);
	Matrix Jm(m, num_labels);
	Jm.subtract(&J1, &J2);
	J = (Jm.sum()) / m;
	Matrix Theta1RemCol(hidden_layer_size, input_layer_size), Theta2RemCol(num_labels, hidden_layer_size);
	Theta1RemCol.removeCol(&Theta1);
	Theta2RemCol.removeCol(&Theta2);
	Matrix Theta1RemColDotSqrd(hidden_layer_size, input_layer_size), Theta2RemColDotSqrd(num_labels, hidden_layer_size);
	Theta1RemColDotSqrd.dotSquare(&Theta1RemCol);
	Theta2RemColDotSqrd.dotSquare(&Theta2RemCol);
	J += lambda / (2 * m) * (Theta1RemColDotSqrd.sum() + Theta2RemColDotSqrd.sum());

	//cout << "a3 values " << a3.arr[0] << endl << a3.arr[1] << endl << a3.arr[2] << endl << a3.arr[3] << endl << a3.arr[4] << endl;
	//cout << '\n' << "Feedforward step completed." << '\n' << endl;
	//clean up
	delete[] oneMinusa3.arr;
	delete[] oneMinusyvec.arr;
	delete[] lna3.arr;
	delete[] minusyvec.arr;
	delete[] lnoneMinusa3.arr;
	delete[] J1.arr;
	delete[] J2.arr;
	delete[] Jm.arr;
	delete[] Theta1RemCol.arr;
	delete[] Theta1RemColDotSqrd.arr;
	delete[] Theta2RemCol.arr;
	delete[] Theta2RemColDotSqrd.arr;
	delete[] one.arr;

	d3.subtract(&a3, &y_vec);
	Matrix d3xTheta2(m, hidden_layer_size + 1);
	Matrix z2Appended(m, hidden_layer_size + 1);
	Matrix z2AppendedSiggrad(m, hidden_layer_size + 1);
	z2Appended.appendOne(&z2);
	d3xTheta2.multiply(&d3, &Theta2);
	z2AppendedSiggrad.sigmoidGradient(&z2Appended);
	Matrix d2PlusCol(m, hidden_layer_size + 1);
	d2PlusCol.dotProduct(&d3xTheta2, &z2AppendedSiggrad);
	d2.removeCol(&d2PlusCol);
	Matrix d2T(hidden_layer_size, m), d3T(num_labels, m);
	d2T.trans(&d2);
	d3T.trans(&d3);
	D1.multiply(&d2T, &a1);
	D2.multiply(&d3T, &a2);
	Theta1_grad.divide(&D1, m);
	Theta2_grad.divide(&D2, m);

	//clean up
	delete[] d2PlusCol.arr;
	delete[] d2T.arr;
	delete[] d3T.arr;
	delete[] z2Appended.arr;
	delete[] d3xTheta2.arr;
	delete[] z2AppendedSiggrad.arr;
	delete[] a1.arr;
	delete[] a2.arr;
	delete[] z2.arr;
	delete[] z3.arr;
	delete[] a3.arr;
	delete[] y_vec.arr;
	delete[] d3.arr;
	delete[] d2.arr;
	delete[] D1.arr;
	delete[] D2.arr;

	Theta1_grad.editThetaGrad(lambda, m, &Theta1);
	Theta2_grad.editThetaGrad(lambda, m, &Theta2);
	Matrix alphaTheta1_grad(hidden_layer_size, input_layer_size + 1);
	Matrix alphaTheta2_grad(num_labels, hidden_layer_size + 1);
	alphaTheta1_grad.divide(&Theta1_grad, 1 / alpha);
	alphaTheta2_grad.divide(&Theta2_grad, 1 / alpha);
	Theta1.subtract(&Theta1, &alphaTheta1_grad);
	Theta2.subtract(&Theta2, &alphaTheta2_grad);
	//cout << "Theta2_grad values " << Theta2_grad.arr[0] << endl << Theta2_grad.arr[1] << endl << Theta2_grad.arr[2] << endl << Theta2_grad.arr[3] << endl << Theta2_grad.arr[4] << endl;

	Matrix newTheta;
	newTheta.combine(&Theta1, &Theta2);
	row = (newTheta).row;
	col = (newTheta).col;
	for (int i = 0; i < row * col; i++)
		arr[i] = (newTheta).arr[i];
	//cout << "Theta1_grad values " << Theta1_grad.arr[0] << endl << Theta1_grad.arr[1] << endl << Theta1_grad.arr[2] << endl << Theta1_grad.arr[3] << endl << Theta1_grad.arr[4] << endl;


	//clean up
	delete[] alphaTheta1_grad.arr;
	delete[] alphaTheta2_grad.arr;
	delete[] newTheta.arr;
	delete[] Theta1_grad.arr;
	delete[] Theta2_grad.arr;
	delete[] Theta1.arr;
	delete[] Theta2.arr;
	//cout << '\n' << "Backpropagation step completed" << '\n' << endl;*/
}

void Matrix::feedForward(Matrix* Theta1, Matrix* Theta2, Matrix* X) {
	int hidden_layer_size = (*Theta1).row;
	int input_layer_size = (*Theta1).col - 1;
	int num_labels = (*Theta2).row;
	long double m = (*X).row;
	Matrix a1(m, input_layer_size + 1), z2(m, hidden_layer_size), a2(m, hidden_layer_size + 1), z3(m, num_labels), a3(m, num_labels);

	a1.appendOne(X);
	Matrix Theta1T(input_layer_size + 1, hidden_layer_size);
	Theta1T.trans(Theta1);
	z2.multiply(&a1, &Theta1T);
	Matrix z2sig(m, hidden_layer_size);
	z2sig.sigmoid(&z2);
	a2.appendOne(&z2sig);
	Matrix Theta2T(hidden_layer_size + 1, num_labels);
	Theta2T.trans(Theta2);
	z3.multiply(&a2, &Theta2T);
	a3.sigmoid(&z3);

	row = a3.row;
	col = a3.col;
	for (int i = 0; i < row*col; i++)
		arr[i] = a3.arr[i];

	delete (z2sig.arr, Theta1T.arr, Theta2T.arr, a1.arr, z2.arr, a2.arr, z3.arr, a3.arr);
	//cout << '\n' << "Feedforward step completed." << '\n' << endl;
}

void Matrix::predict(Matrix* a3) {
	row = (*a3).row;
	col = 1;
	int pred;
	for (int r = 0; r < (*a3).row; r++) {
		double k = -1000;
		for (int c = 0; c < (*a3).col; c++) {
			if ((*a3).arr[r * (*a3).col + c] > k) {
				k = (*a3).arr[r * (*a3).col + c];
				pred = c;
			}
		}
		arr[r] = pred + 1;
	}
}

double Matrix::predAccuracy(Matrix* y) {
	double error = 0;
	for (int i = 0; i < row*col; i++)
		if (arr[i] != (*y).arr[i])
			error++;
	error /= (row / 100.0);
	return error;
}

long double X[5000][400];
long double Theta1[25][401];
long double Theta2[10][26];
long double y[1][5000];

int main() {
	const int N = 2, M = 8, P = 2;
	int input_layer_size = 400;
	int hidden_layer_size = 25;
	int num_labels = 10;
	long double m = 5000;
	long double lambda = .1;
	long double alpha = 1;

	Matrix X(m, input_layer_size), Theta1(hidden_layer_size, input_layer_size + 1), Theta2(num_labels, hidden_layer_size + 1), y(1, m), Theta;
	Matrix test1(N, N), test2(N, N), testRes1(N, N), testRes2(N, N);
	string Theta1primeLocation = "C:\\Users\\Madhav\\Documents\\Visual Studio 2015\\Projects\\neuralNetwork\\Theta1prime.txt";
	string Theta2primeLocation = "C:\\Users\\Madhav\\Documents\\Visual Studio 2015\\Projects\\neuralNetwork\\Theta2prime.txt";
	string yLocation = "C:\\Users\\Madhav\\Documents\\Visual Studio 2015\\Projects\\neuralNetwork\\y.txt";
	string XLocation = "C:\\Users\\Madhav\\Documents\\Visual Studio 2015\\Projects\\neuralNetwork\\X.txt";
	string testSample1 = "C:\\Users\\Madhav\\Documents\\Visual Studio 2015\\Projects\\neuralNetwork\\testSample1.txt";
	string testSample2 = "C:\\Users\\Madhav\\Documents\\Visual Studio 2015\\Projects\\neuralNetwork\\testSample2.txt";

	/*test1.load(testSample1,5,3);
	test2.load(testSample2,5,3);
	test1.row = M;
	test1.col = N;
	test2.row = N;
	test2.col = P;
	for (int i = 0; i < M; i++)
	for (int j = 0; j < N; j++)
	test1.arr[i * N + j] = rand() % 10;
	for (int i = 0; i < N; i++)
	for (int j = 0; j < P; j++)
	test2.arr[i * P + j] = rand() % 10;
	//test1.display();
	//test2.display();
	testRes1.multiply1(&test1, &test2);
	cout << "regular multiplication done." << endl;
	testRes2.multiply(&test1, &test2);
	cout << "strassen multiplication done." << endl;
	testRes1.display();
	testRes2.display();*/
	y.load(yLocation, m, 1);
	X.load(XLocation, m, input_layer_size);
	Theta1.generateRandWeights(hidden_layer_size, input_layer_size + 1);
	Theta2.generateRandWeights(num_labels, hidden_layer_size + 1);
	//Theta1.load(Theta1primeLocation, 25, 401);
	//Theta2.load(Theta2primeLocation, 10, 26);
	Theta.combine(&Theta1, &Theta2);
	Theta.J = 1000;
	Matrix newTheta;
	for (int iter = 1; iter <= 300; iter++) {
		long double Jprev = Theta.J;
		newTheta.costFunction(&Theta, &X, &y, input_layer_size, hidden_layer_size, num_labels, m, lambda, alpha);
		Theta.copy(&newTheta);
		std::cout << "\n" << "Iteration " << iter << " completed." << " -> " << "cost = " << Theta.J << endl;
		//		if ((Jprev - Theta.J < 0) && (alpha > .05))
		//			alpha -= .05;
		//		if (!iter%100)
		//			cout << "-> alpha = " << alpha << endl;
	}
	Matrix a3(m, num_labels), prediction(m, 1);
	Theta1.decombine(&Theta, 0, hidden_layer_size, input_layer_size + 1);
	Theta2.decombine(&Theta, 1, num_labels, hidden_layer_size + 1);
	a3.feedForward(&Theta1, &Theta2, &X);

	prediction.predict(&a3);
	cout << prediction.arr[0] << endl;
	double error = 0;
	error = prediction.predAccuracy(&y);
	std::cout << '\n' << "Training set accuracy is : " << 100 - error << '%' << endl;

	//clean up
	delete[](X.arr, Theta1.arr, Theta2.arr, y.arr, Theta.arr, a3.arr, prediction.arr, newTheta.arr);
	return 0;
}
