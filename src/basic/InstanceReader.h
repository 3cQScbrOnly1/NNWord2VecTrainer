#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3L.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader : public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext() {
		m_instance.clear();
		string strLine;
		if (!my_getline(m_inf, strLine))
			return NULL;
		if (strLine.empty())
			return NULL;


		split_bychars(strLine, m_instance.m_words, " ");

		return &m_instance;
	}
};

#endif

