#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

class ModelParams{
public:
	Alphabet wordAlpha;
	LookupTable words;
	UniParams olayer_linear;
public:
	SoftMaxLoss loss;

	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){
		// some model parameters should be initialized outside
		if (words.nVSize <= 0){
			cout << "words size is error" << endl;
			return false;
		}
		olayer_linear.initial(opts.labelSize, 2 * opts.wordDim, false, mem);
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}
	

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}


	void saveModel(std::ofstream &os) {
		words.saveEmb(os);
	}

	void loadModel(const string& inFile){

	}
};

#endif 