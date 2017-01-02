#include "NNWordEmbTrainer.h"

#include "Argument_helper.h"

Trainer::Trainer(int memsize):m_driver(memsize){
	instances_count = 0;
	buffer_size = 100;
	context_size = 2;
	error_size = 5;
	table_size = 1e8;
	neg_word_size = 5;
	table = new int[table_size];
	START = "<s>";
	END = "</s>";
}

Trainer::~Trainer(){
	delete[] table;
}

void Trainer::createWordStates(const string& file_name) {
	m_pipe.initInputFile(file_name.c_str());
	Instance *pInstance = m_pipe.nextInstance();
	vector<Instance> insts;
	insts.clear();
	while (pInstance) {
		Instance trainInstance;
		trainInstance.copyValuesFrom(*pInstance);
		insts.push_back(trainInstance);
		if (insts.size() == buffer_size) {
			addWord2States(insts);
			insts.clear();
		}
		pInstance = m_pipe.nextInstance();
	}
	if (insts.size() != 0)
		addWord2States(insts);
	m_pipe.uninitInputFile();
	m_word_stats[START] = m_options.wordCutOff + 1;
	m_word_stats[END] = m_options.wordCutOff + 1;
	cout << endl << "word size: "<< m_word_stats.size() << endl;
}


void Trainer::createRandomTable(){
	int vocab_size = m_word_stats.size();
	if (vocab_size == 0) {
		cout << "error word count" << endl;
		return;
	}
	if (m_driver._modelparams.wordAlpha.size() == 0) {
		cout << "error wordAlpha in modelparam " << endl;
		return;
	}
	// copy m_word_stats;
	for (unordered_map<string, int>::iterator it = m_word_stats.begin();
		it != m_word_stats.end(); it++) {
		m_word_vect.push_back(make_pair(it->first, it->second));
	}

	double d1, power = 0.75, train_words_pow = 0;
	for (int idx = 0; idx < vocab_size; idx++) {
		train_words_pow += pow(m_word_vect[idx].second , power);
	}
	int i = 0;
	d1 = pow(m_word_vect[i].second, power)/ train_words_pow;
	for (int a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(m_word_vect[i].second, power) / train_words_pow;
		}
		if (i >= vocab_size)
			i = vocab_size - 1;
	}
}

void Trainer::createNegWord(const string& context_word, vector<string>& neg_words){
	neg_words.clear();
	int random_index, word_index;
	for (int i = 0; i < neg_word_size; i++) {
		random_index = rand() % table_size;
		word_index = table[random_index];
		if (word_index >= m_word_stats.size() || word_index < 0) {
			cout << "random error" << endl;
			break;
		}
		if (m_word_vect[word_index].first == context_word)
			i--;
		else
			neg_words.push_back(m_word_vect[word_index].first);
	}
}

void Trainer::createNegExamples(const string& target_word, const vector<string>& neg_words, vector<Example>& neg_exams){
	neg_exams.clear();
	int neg_exam_size = neg_words.size();
	for (int i = 0; i < neg_exam_size; i++) {
		Example exam;
		exam.m_feature.target_word = target_word;
		exam.m_feature.context_word = neg_words[i];
		exam.is_negative();
		neg_exams.push_back(exam);
	}
}


void Trainer::addWord2States(const vector<Instance>& vecInsts){
	int numInstance = vecInsts.size();
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_words;
		int words_num = words.size();
		for (int i = 0; i < words_num; i++)
		{
			//string curword = normalize_to_lowerwithdigit(words[i]);
			m_word_stats[words[i]]++;
		}
		if (m_options.maxInstance > 0 && instances_count == m_options.maxInstance)
			break;
	}
	instances_count += numInstance;
	cout << instances_count << " ";
}

void Trainer:: convert2Example(const Instance* pInstance, vector<Example>& vecExams){
	vecExams.clear();
	vector<string> words = pInstance->m_words;
	vector<string> neg_words;
	vector<Example> neg_exams;
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
			neg_words.clear();
			createNegWord(exam.m_feature.context_word, neg_words);
			neg_exams.clear();
			createNegExamples(exam.m_feature.target_word, neg_words, neg_exams);
			vecExams.insert(vecExams.end(), neg_exams.begin(), neg_exams.end());
		}
		for (int offset = 1; offset <= context_size; offset++) {
			Example exam;
			exam.m_feature.target_word = words[idx];
			if (idx + offset >= word_size)
				exam.m_feature.context_word = END;
			else
				exam.m_feature.context_word = words[idx + offset];
			exam.is_positive();
			vecExams.push_back(exam);
			neg_words.clear();
			createNegWord(exam.m_feature.context_word, neg_words);
			neg_exams.clear();
			createNegExamples(exam.m_feature.target_word, neg_words, neg_exams);
			vecExams.insert(vecExams.end(), neg_exams.begin(), neg_exams.end());
		}
	}
}

void Trainer::train(const string& trainFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_driver._hyperparams.setRequared(m_options);
	cout << "Create Alphabet....." << endl;
	createWordStates(trainFile);
	m_driver._modelparams.wordAlpha.initial(m_word_stats, m_options.wordCutOff);
	m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordEmbSize, true);
	m_driver.initial();
	createRandomTable();
	trainEmb(trainFile);
	cout << "Saving model..." << endl;
	writeModelFile(modelFile);
	cout << "Save complete!" << endl;
}

void Trainer::writeModelFile(const string& outputModelFile) {
	ofstream os(outputModelFile);
	if (os.is_open())
	{
		m_driver._modelparams.saveModel(os);
	}
	else
	{
		cout << "write model error." << endl;
	}
}

dtype Trainer::trainInstances(const vector<Instance>& vecInst){
	int vecSize = vecInst.size();
	int examSize;
	dtype cost = 0;
	vector<Example> exams;
	static vector<Example> subExamples;
	for (int idx = 0; idx < vecSize; idx++) {
		exams.clear();
		convert2Example(&vecInst[idx], exams);
		examSize = exams.size();
		for (int idy = 0; idy < examSize; idy++) {
			subExamples.clear();
			subExamples.push_back(exams[idy]);
			cost += m_driver.train(subExamples, 1);
			m_driver.updateModel();
		}
	}
	return cost;
}

void Trainer::trainEmb(const string& trainFile){
	m_pipe.initInputFile(trainFile.c_str());
	Instance *pInstance = m_pipe.nextInstance();
	vector<Instance> insts;
	insts.clear();
	dtype cost;
	int count = 0;
	while (pInstance) {
		Instance trainInstance;
		trainInstance.copyValuesFrom(*pInstance);
		insts.push_back(trainInstance);
		if (insts.size() == buffer_size) {
			cost = trainInstances(insts);
			cout << "cost: " << cost << endl;
			cout << "count: " << count << endl;
			insts.clear();
			count+=buffer_size;
		}
		pInstance = m_pipe.nextInstance();
	}
	if (insts.size() != 0)
	{
		cost = trainInstances(insts);
		cout << "cost: " << cost << endl;
		cout << "count: " << count << endl;
	}
	m_pipe.uninitInputFile();
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