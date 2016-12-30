#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
public:
	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization

	int labelSize; // negative and positive
	int wordDim;
public:
	HyperParams() {
		bAssigned = true;
	}

public:
	void setRequared(Options& opt){
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		labelSize = 2;
		bAssigned = true;
	}

public:
	bool bValid(){
		return bAssigned;
	}

public:
	void print(){

	}

private:
	bool bAssigned;
};

#endif 