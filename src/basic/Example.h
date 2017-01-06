#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>

using namespace std;

class Feature
{
public:
	int target_word_index;
	int context_word_index;
public:
	Feature(){}
	Feature(int t_index, int c_index) :target_word_index(t_index), context_word_index(c_index) {}
	void clear()
	{
		target_word_index = -1;
		context_word_index = -1;
	}
};

class Example
{
private:
public:
	Feature m_feature;
	vector<double> m_label;
public:
	Example():m_label(2){
	}

	Example(int t_index, int c_index) :m_feature(t_index, c_index), m_label(2){}


	void is_positive(){
		m_label[0] = 1;
		m_label[1] = 0;
	}
	void is_negative(){
		m_label[0] = 0;
		m_label[1] = 1;
	}
	void clear()
	{
		m_feature.clear();
		m_label.clear();
	}
};

#endif /*_EXAMPLE_H_*/