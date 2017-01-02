#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:

public:
	// node instances
	LookupNode _target_word;
	LookupNode _context_word;
	ConcatNode _concat;
	LinearNode _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 

	inline void clear(){
		Graph::clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		_target_word.setParam(&model.words);
		_context_word.setParam(&model.words);
		_target_word.init(opts.wordDim, -1, mem);
		_context_word.init(opts.wordDim, -1, mem);
		_concat.init(opts.wordDim + opts.wordDim, -1, mem);
		_output.setParam(&model.olayer_linear);
		_output.init(opts.labelSize, -1, mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		// second step: build graph
		//forward
		_target_word.forward(this, feature.target_word);
		_context_word.forward(this, feature.context_word);
		_concat.forward(this, &_target_word, &_context_word);
		_output.forward(this, &_concat);
	}
};

#endif /* SRC_ComputionGraph_H_ */