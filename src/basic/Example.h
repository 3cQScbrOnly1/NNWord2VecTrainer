#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>

using namespace std;

class Feature
{
public:
	string target_word;
	string context_word;
public:
	void clear()
	{
		target_word = "";
		context_word = "";
	}
};

class Example
{
public:
	Feature m_feature;
	vector<double> m_label;

public:
	void is_positive(){
		m_label.clear();
		m_label.push_back(1);
		m_label.push_back(0);
	}
	void is_negative(){
		m_label.clear();
		m_label.push_back(0);
		m_label.push_back(1);
	}
	void clear()
	{
		m_feature.clear();
		m_label.clear();
	}
};

#endif /*_EXAMPLE_H_*/