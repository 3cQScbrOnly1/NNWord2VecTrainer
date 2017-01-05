#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>

using namespace std;

class Feature
{
public:
	vector<string> target_words;
	vector<string> context_words;
public:
	void clear()
	{
		target_words.clear();
		context_words.clear();
	}
};

class Example
{
private:
	vector<double> neg_label;
	vector<double> pos_label;
public:
	Feature m_feature;
	vector<vector<double> > m_labels;
public:
	Example(){
		neg_label.push_back(1);
		neg_label.push_back(0);

		pos_label.push_back(0);
		pos_label.push_back(1);
	}

	void is_positive(){
		m_labels.push_back(pos_label);
	}
	void is_negative(){
		m_labels.push_back(neg_label);
	}
	void clear()
	{
		m_feature.clear();
		m_labels.clear();
	}
};

#endif /*_EXAMPLE_H_*/