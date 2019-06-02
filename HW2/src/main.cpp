#include "../include/Scan.h"
#include <fstream>
#include <iostream>


using namespace std;

int main() {
    ifstream input;
    input.open("input.txt");

    int size;
    input >> size;
    vector<double> array;
    for (int i = 0; i < size; ++i) {
        double value;
        input >> value;
        array.push_back(value);
    }

    Scan sum;
    vector<double> ans(array.size());
    sum.hillis_steele(array, ans);

    ofstream output;
    output.open("output.txt");
    for (double value : ans) {
        output << value << ' ';
    }
    output.close();
}

