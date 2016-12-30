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
	int instances_count;
	int buffer_size;
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
	void createWordTable(const string& file_name);
	void addWord2Table(const vector<Instance>& insts);
	void train(const string& trainFile, const string& modelFile, const string& optionFile);
	void convert2Example(const Instance* pInstance, vector<Example>& exam);
};