#include "NNWordEmbTrainer.h"

#include "Argument_helper.h"

Trainer::Trainer(int memsize):m_driver(memsize){
	instances_count = 0;
	buffer_size = 1;
	START = "<s>";
	END = "</s>";
}

Trainer::~Trainer(){}

void Trainer::createWordTable(const string& file_name) {
	m_pipe.initInputFile(file_name.c_str());
	Instance *pInstance = m_pipe.nextInstance();
	vector<Instance> insts;
	insts.clear();
	while (pInstance) {
		Instance trainInstance;
		trainInstance.copyValuesFrom(*pInstance);
		insts.push_back(trainInstance);
		if (insts.size() == buffer_size) {
			addWord2Table(insts);
			insts.clear();
		}
		pInstance = m_pipe.nextInstance();
	}
	if (insts.size() != 0)
		addWord2Table(insts);
	m_pipe.uninitInputFile();
	cout << endl << "word size: "<< m_word_stats.size() << endl;
}

void Trainer::addWord2Table(const vector<Instance>& vecInsts){
	int numInstance = vecInsts.size();
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_words;
		int words_num = words.size();
		for (int i = 0; i < words_num; i++)
		{
			string curword = normalize_to_lowerwithdigit(words[i]);
			m_word_stats[curword]++;
		}

		if (m_options.maxInstance > 0 && instances_count == m_options.maxInstance)
			break;
	}

	instances_count += numInstance;
	cout << instances_count << " ";
}

void Trainer:: convert2Example(const Instance* pInstance, vector<Example>& vecExams){
	int context_size = 2;
	vecExams.clear();
	vector<string> words = pInstance->m_words;
	int word_size = words.size();
	for (int idx = 0; idx < words.size(); idx++) {
		for (int offset = 1; offset <= context_size; offset++) {
			Example exam;
			exam.m_feature.target_word = words[idx];
			if (idx - offset < 0)
				exam.m_feature.context_word = START;
			else
				exam.m_feature.context_word = words[idx - offset];
			exam.is_positive();
			vecExams.push_back(exam);
		}
		for (int offset = 1; offset <= context_size; offset++) {
			Example exam;
			exam.m_feature.target_word = words[idx];
			if (idx + offset > word_size)
				exam.m_feature.context_word = END;
			else
				exam.m_feature.context_word = words[idx + offset];
			exam.is_positive();
			vecExams.push_back(exam);
		}
	}
}

void Trainer::train(const string& trainFile, const string& modelFile, const string& optionFile) {
	cout << "Create Alphabet....." << endl;
	createWordTable(trainFile);
	m_word_stats[START] = 1;
	m_word_stats[END] = 1;
}

int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	int memsize = 0;
	dsr::Argument_helper ah;

	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_int("memsize", "memorySize", "named_int", "This argument decides the size of static memory allocation", memsize);

	ah.process(argc, argv);

	if (memsize < 0)
		memsize = 0;
	Trainer the_trainer(memsize);
	the_trainer.train(trainFile, modelFile, optionFile);
	/*
	if (bTrain) {
		the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		the_classifier.test(testFile, outputFile, modelFile);
	}
	*/
	getchar();
	//test(argv);
	//ah.write_values(std::cout);
}