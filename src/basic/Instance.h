#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
using namespace std;

class Instance
{
public:
	void copyValuesFrom(const Instance& anInstance){
		m_words = anInstance.m_words;
	}

	void clear()
	{
		m_words.clear();
	}

	int size() const {
		return m_words.size();
	}

	void allocate(int length)
	{
		clear();
		m_words.resize(length);
	}
public:
	vector<string> m_words;
};

#endif /*_INSTANCE_H_*/
