#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_exam_size = 1000;
public:
	// node instances
	vector<LookupNode> _target_words;
	vector<LookupNode> _context_words;
	vector<ConcatNode> _concats;
	vector<LinearNode> _outputs;
public:
	ComputionGraph() : Graph(){

	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int exam_size){
		_target_words.resize(exam_size);
		_context_words.resize(exam_size);
		_concats.resize(exam_size);
		_outputs.resize(exam_size);
	}

	inline void clear(){
		Graph::clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		int max_size = _target_words.size();
		for (int idx = 0; idx < max_size; idx++) {
			_target_words[idx].setParam(&model.words);
			_context_words[idx].setParam(&model.words);
			_target_words[idx].init(opts.wordDim, -1, mem);
			_context_words[idx].init(opts.wordDim, -1, mem);
			_concats[idx].init(opts.wordDim + opts.wordDim, -1, mem);
			_outputs[idx].setParam(&model.olayer_linear);
			_outputs[idx].init(opts.labelSize, -1, mem);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		// second step: build graph
		//forward
		int exam_size = feature.target_words.size();
		if (exam_size > max_exam_size)
			exam_size = max_exam_size;
		for (int idx = 0; idx < exam_size; idx++) {
			_target_words[idx].forward(this, feature.target_words[idx]);
			_context_words[idx].forward(this, feature.context_words[idx]);
			_concats[idx].forward(this, &_target_words[idx], &_context_words[idx]);
			_outputs[idx].forward(this, &_concats[idx]);
		}
	}
};

#endif /* SRC_ComputionGraph_H_ */