//TODO: create a matrix class to do operations
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <fstream>
#include <string>
#include <time.h>
#include <queue>
#include <math.h>
#include <pthread.h>

#define MAX_WINDOW_SIZE 5
#define VEC_DIM 100
#define MAX_VOCAB_SIZE 100000
#define NUM_ITERATIONS 25
using namespace std;


map<string, int> vocab;
map<string, int> hash;
deque<int> window;
map<pair<int, int>, int> co;
int VOCAB_SIZE = 0;
//The word vector of size (2*Vocabsize)*Dim because one is for word as main(center) 
//and other for the word as context. 
double wordVec[2 * MAX_VOCAB_SIZE][VEC_DIM];

//Bias terms for every word vector
double bias[2 * MAX_VOCAB_SIZE];
 
//To store sum of squares of all previous gradients for training
double grad_sq[2 * MAX_VOCAB_SIZE][VEC_DIM];

//Sum of squared gradients of bias vectors
double grad_sq_bias[2 * MAX_VOCAB_SIZE];

double grad_main[VEC_DIM], grad_context[VEC_DIM];

//double gradsq_root_main[VEC_DIM], gradsq_root_context[VEC_DIM];


class matrix{

};

double dot_prod(int main, int cont)
{
	double prod = 0.0;
	for (int i = 0 ; i < VEC_DIM ; i++ ) {
		prod += wordVec[main][i] * wordVec[cont][i];
	}
	return prod;
}

// scalar multiplication of a vector 
void mult_scalar(int main, int cont, double scalar)
{
	for (int i = 0 ; i < VEC_DIM ; i++ ) {
		grad_main[i] = scalar * wordVec[main][i];
		grad_context[i] = scalar * wordVec[cont][i];
	}
}

double cosineDist(int i, int j)
{
	double dist = 0.0;
	double sq_sum1 = 0.0, sq_sum2 = 0.0;
	
	for (int k = 0 ; k < VEC_DIM ; k++) {
		dist += (wordVec[i][k] * wordVec[j][k]);
		sq_sum1 += pow(wordVec[i][k] , 2);
		sq_sum2 += pow(wordVec[j][k] , 2);
	}
	//cout << "in dist" << ( dist / (sqrt(sq_sum1) * sqrt(sq_sum2))) << endl;
	return (1.0 - dist / (sqrt(sq_sum1) * sqrt(sq_sum2)));
}
double euclidDist(int i, int j)
{
	double dist = 0.0;
	
	for (int k = 0 ; k < VEC_DIM ; k++) {
		dist += pow(wordVec[i][k] - wordVec[j][k], 2);
	}
	return dist;
}

void iteration(int main, int cont, double occur, double learning_rate = 0.05, double x_max = 100, double alpha = 0.75)
{
	double weight = 1.0, cost_dot, cost;
	double global_cost = 0.0;	
	double grad_bias_main, grad_bias_context;

	if (occur < x_max) {
		weight = pow(occur / x_max, alpha);
	}

	cost_dot = dot_prod(main, cont) + bias[main] + bias[cont] - log( 1 + occur) ;
	cost = weight * pow(cost_dot, 2);
	global_cost += 0.5 * cost;
	//cout << "cost is " << dot_prod(main, cont) << endl;
	//computing gradients of vector terms
	mult_scalar(cont, main, weight * cost_dot);

	//computing vector of bias terms
	grad_bias_main = grad_bias_context = weight * cost_dot;

	//perform adaptive updates
	for (int i = 0 ; i < VEC_DIM ; i++) {
		wordVec[main][i] -= (learning_rate * grad_main[i]) / sqrt(grad_sq[main][i]);
		wordVec[cont][i] -= (learning_rate * grad_context[i]) / sqrt(grad_sq[cont][i]);
	}
	bias[main] -= (learning_rate * grad_bias_main) / sqrt(grad_sq_bias[main]);
	bias[cont] -= (learning_rate * grad_bias_context) / sqrt(grad_sq_bias[cont]);

	//update squared gradient sums
	for (int i = 0 ; i < VEC_DIM ; i++ ) {
		grad_sq[main][i] += pow(grad_main[i], 2);
		grad_sq[cont][i] += pow(grad_context[i], 2);
	}
	grad_sq_bias[main] += pow(grad_bias_main, 2);
	grad_sq_bias[cont] += pow(grad_bias_context, 2);
}


double r2()
{
    return (double)srand() / (double)RAND_MAX ;
}

int main(){
	clock_t t1,t2;
    t1=clock();
	ifstream myfile;
	myfile.open("text8");
	string word;
	long long tot = 0, ind = 0;

	// making the vocabulary and hashing for every string to a number using maps
	while (myfile >> word && ind < 100000){
		if (vocab.find(word) != vocab.end()){
			vocab[word]++;
			VOCAB_SIZE++;
		}
		else {
			hash[word] = vocab.size();
			vocab[word]++;
		}
		ind++;
	}
	cout << "read" << endl;
	myfile.close();
	myfile.open("text8");
	ind = 0;
	//creating the cooccurrence matrix	
	while (myfile >> word && ind < 100000) {
		ind++;
		for (int i = 0 ; i < window.size() ; i++) {
			double distance = 1.0 / (i + 1);
			co[make_pair(hash[word], window[i])] += distance;
			co[make_pair(window[i], hash[word])] += distance;	
		}
		window.push_front(hash[word]);
		if (window.size() > MAX_WINDOW_SIZE)
			window.pop_back();
	}
	cout << "made the matrix" << endl;
	//initialise the vectors and biases
	for (int i = 0 ; i < 2 * VOCAB_SIZE ; i++) {
		bias[i] = r2() - 0.5;
		grad_sq_bias[i] = 1.0;
		for (int j = 0 ; j < VEC_DIM ; j++) {
			wordVec[i][j] = r2() - 0.5;
			grad_sq[i][j] = 1.0;
		}
	}

	cout<< "starting the training" << endl;
	//The algorithm used for training is adaptive gradient descent
	for (int k = 0 ; k < NUM_ITERATIONS ; k++ ) {
		cout << "iteration no" << k + 1 << endl;
		for (map<pair<int, int> , int >::iterator it = co.begin() ; it != co.end() ; it++) {
			iteration(it->first.first, it->first.second + VOCAB_SIZE, it->second);
		}
	}
	while(1){
		map<double, string> synonym;
		cin >> word;
		cout << word << " " << hash[word] << endl;
		for (map<string , int >::iterator it = hash.begin(); it != hash.end() ; it++) {
			if(hash[word] != it->second){
				double dist = cosineDist(hash[word] , it->second); 
				//cout << "here" << dist << endl;
				synonym[dist] = it->first;
			}
		}

		int count = 0;
		for (map<double , string>::iterator it = synonym.begin() ; it != synonym.end() ; it++) {
			count++;
			cout << it->first << "\t" << it->second << endl;
			if (count > 10)
				break;
		}
	}
    t2=clock();
    double diff ((double)t2 - (double)t1);
    cout << diff/CLOCKS_PER_SEC << endl;
	cout << tot << " " << vocab.size() << " " << co.size() << " " << ind << endl;
	return 0;
}