#include "NNWordEmbTrainer.h"

#include "Argument_helper.h"

Trainer::Trainer(int memsize, int graphnum):m_driver(memsize, graphnum){
	instances_count = 0;
	buffer_size = graphnum;
	context_size = 2;
	table_size = 1e8;
	neg_word_size = 5;
	table.resize(table_size);
	START = "<s>";
	END = "</s>";
}

Trainer::~Trainer(){
}

void Trainer::createWordStates(const string& file_name) {
	m_pipe.initInputFile(file_name.c_str());
	Instance *pInstance = m_pipe.nextInstance();
	vector<Instance> insts;
	insts.clear();
	while (pInstance) {
		Instance trainInstance;
		trainInstance.copyValuesFrom(*pInstance);
		insts.emplace_back(trainInstance);
		if (insts.size() == buffer_size) {
			addWord2Stats(insts);
			insts.clear();
		}
		pInstance = m_pipe.nextInstance();
	}
	if (insts.size() != 0)
		addWord2Stats(insts);
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
		vocab.emplace_back(make_pair(it->first, it->second));
	}

	double d1, power = 0.75, train_words_pow = 0;
	for (int idx = 0; idx < vocab_size; idx++) {
		train_words_pow += pow(vocab[idx].second , power);
	}
	int i = 0;
	d1 = pow(vocab[i].second, power)/ train_words_pow;
	for (int a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(vocab[i].second, power) / train_words_pow;
		}
		if (i >= vocab_size)
			i = vocab_size - 1;
	}
}

void Trainer::createNegWords(const string& context_word, vector<string>& neg_words){
	neg_words.clear();
	int random_index, word_index;
	for (int i = 0; i < neg_word_size; i++) {
		random_index = rand() % table_size;
		word_index = table[random_index];
		if (word_index >= m_word_stats.size() || word_index < 0) {
			cout << "random error" << endl;
			break;
		}
		if (vocab[word_index].first != context_word)
			neg_words.emplace_back(vocab[word_index].first);
	}
}

void Trainer::createNegExample(const string& target_word, const vector<string>& neg_words, Example& exam){
	int neg_exam_size = neg_words.size();
	for (int i = 0; i < neg_exam_size; i++) {
		exam.m_feature.target_words.emplace_back(target_word);
		exam.m_feature.context_words.emplace_back(neg_words[i]);
		exam.is_negative();
	}
}

void Trainer::createPosExample(const string& target_word, const string& context_word, Example& exam){
		exam.m_feature.target_words.emplace_back(target_word);
		exam.m_feature.context_words.emplace_back(context_word);
		exam.is_positive();
}

void Trainer::addWord2Stats(const vector<Instance>& vecInsts){
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

void Trainer:: convert2Example(const Instance* pInstance, Example& exam){
	vector<string> words = pInstance->m_words;
	vector<string> neg_words;
	int word_size = words.size();
	string curr_context_word;
	for (int idx = 0; idx < words.size(); idx++) {
		for (int offset = 1; offset <= context_size; offset++) {
			neg_words.clear();
			if (idx - offset < 0)
				curr_context_word = START;
			else
				curr_context_word = words[idx - offset];

			createPosExample(words[idx], curr_context_word, exam);
			createNegWords(curr_context_word, neg_words);
			createNegExample(words[idx], neg_words, exam);
		}

		for (int offset = 1; offset <= context_size; offset++) {
			neg_words.clear();
			if (idx + offset >= word_size)
				curr_context_word = END;
			else
				curr_context_word = words[idx + offset];

			createPosExample(words[idx], curr_context_word, exam);
			createNegWords(curr_context_word, neg_words);
			createNegExample(words[idx], neg_words, exam);
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
	clock_t start_time = clock();
	trainEmb(trainFile);
	cout << endl << "Training cost time :" << (clock() - start_time) / CLOCKS_PER_SEC  << endl;
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
	Example exam;
	vector<Example> vecExams;
	clock_t start_time = clock();
	for (int idx = 0; idx < vecSize; idx++) {
		exam.clear();
		convert2Example(&vecInst[idx], exam);
		vecExams.push_back(exam);
		//examSize = exams.size();
		//for (int idy = 0; idy < examSize; idy++) {
			//subExamples.clear();
			//subExamples.emplace_back(exams[idy]);
		//cost += m_driver.train(exams, 1);
		//}
		//m_driver.updateModel();
	}
	cost += m_driver.train(vecExams, 1);
	m_driver.updateModel();
	cout << "one buffer cost time :" << (clock() - start_time) / CLOCKS_PER_SEC  << endl;
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
		insts.emplace_back(trainInstance);
		if (insts.size() == buffer_size) {
			cost = trainInstances(insts);
			cout << "cost: " << cost << endl;
			insts.clear();
			count+=buffer_size;
			cout << "count: " << count << endl;
		}
		pInstance = m_pipe.nextInstance();
	}
	if (insts.size() > 0)
	{
		cost = trainInstances(insts);
		cout << "cost: " << cost << endl;
		count += buffer_size;
		cout << "count: " << count << endl;
	}
	m_pipe.uninitInputFile();
}

int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	int memsize = 0;
	int thread = 1;
	dsr::Argument_helper ah;

	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_int("memsize", "memorySize", "named_int", "This argument decides the size of static memory allocation", memsize);
	ah.new_named_int("thread", "thread num", "named_int", "The thread size", thread);

	ah.process(argc, argv);

	if (memsize < 0)
		memsize = 0;
	//omp_set_num_threads(thread);
	cout << "Thread num: "<<  thread << endl;
	omp_set_num_threads(thread);

	int graph_num = 100;
	Trainer the_trainer(memsize, graph_num);
	the_trainer.train(trainFile, modelFile, optionFile);
	/*
	if (bTrain) {
		the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		the_classifier.test(testFile, outputFile, modelFile);
	}
	*/
	//test(argv);
	//ah.write_values(std::cout);
}