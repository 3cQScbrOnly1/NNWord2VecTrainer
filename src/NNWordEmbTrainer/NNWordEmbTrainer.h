#include "N3L.h"

#include "Options.h"
#include "Instance.h"
#include "Example.h"
#include "Driver.h"
#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Trainer{
public:
	unordered_map<string, int> m_word_stats;
	vector<pair<string, int> > m_word_vect;
	int instances_count;
	int neg_word_size;
	int buffer_size;
	int context_size;
	int error_size;
	vector<int> table;
	int table_size;
	string START;
	string END;
public:
	Options m_options;
	
	Driver m_driver;

	Pipe m_pipe;

public:
	Trainer(int mem);
	virtual ~Trainer();

public:
	void createWordStates(const string& file_name);
	void addWord2States(const vector<Instance>& insts);
	void train(const string& trainFile, const string& modelFile, const string& optionFile);
	void convert2Example(const Instance* pInstance, vector<Example>& exam);
	void trainEmb(const string& trainFile);
	dtype trainInstances(const vector<Instance>& vecInst);
	void createRandomTable();
	void createNegWord(const string& context_word, vector<string>& neg_words);
	void createNegExamples(const string& target_word, const vector<string>& neg_words, vector<Example>& neg_exams);
	void writeModelFile(const string& outputModelFile);
};